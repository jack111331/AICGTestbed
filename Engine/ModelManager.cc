#include "pch.h"
#include "ModelManager.hpp"

#include "OnnxModel.hpp"
#include "OrtModel.hpp"

#include "imgui.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <type_traits>

using namespace DirectX;

namespace NeuralModelIntegrateTestbed {

const char* ToString(ModelBackend backend) {
    switch (backend) {
        case ModelBackend::DirectMLGraph: return "DirectML graph";
        case ModelBackend::OnnxRuntime:   return "ONNX Runtime (DirectML EP)";
    }
    return "unknown";
}

namespace {

template <typename T>
T RoundUpToMultiple(T value, T multiple) {
    static_assert(std::is_integral_v<T>);

    T remainder = value % multiple;
    if (remainder != 0) {
        value += multiple - remainder;
    }
    return value;
}

void SetError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

HRESULT CreateDefaultBuffer(ID3D12Device* device,
                            uint64_t bytes,
                            D3D12_RESOURCE_STATES state,
                            Microsoft::WRL::ComPtr<ID3D12Resource>& out) {
    const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const auto desc =
        CD3DX12_RESOURCE_DESC::Buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
                                           nullptr, IID_PPV_ARGS(out.ReleaseAndGetAddressOf()));
}

// ---------------------------------------------------------------------------
// The DirectMLGraph backend: a compiled IDMLCompiledOperator whose dispatch can
// be recorded into any command list, including the renderer's own.
// ---------------------------------------------------------------------------
class DmlGraphRunner : public IModelRunner {
public:
    DmlGraphRunner(ID3D12Device* device,
                   ID3D12CommandQueue* queue,
                   IDMLDevice* dmlDevice,
                   IDMLCommandRecorder* recorder)
        : m_device(device), m_queue(queue), m_dmlDevice(dmlDevice), m_recorder(recorder) {}

    ModelBackend Backend() const override { return ModelBackend::DirectMLGraph; }
    const ModelInfo& Info() const override { return m_info; }

    ID3D12Resource* InputResource() override { return m_input.Get(); }
    ID3D12Resource* OutputResource() override { return m_output.Get(); }

    // Builds, compiles, uploads weights and initializes. On failure nothing is
    // left usable, and ModelManager discards the runner rather than registering
    // a half-built one.
    bool Build(std::unique_ptr<NNModel> source, std::string* error);

    bool RecordInto(ID3D12GraphicsCommandList* commandList) override {
        if (commandList == nullptr || !m_graph || !m_bindingTable) {
            return false;
        }
        // TODO the input and output buffers must already be in
        // D3D12_RESOURCE_STATE_UNORDERED_ACCESS when a caller feeds them from
        // its own passes.
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"DML ops");
        ID3D12DescriptorHeap* heaps[] = {m_descriptorHeap->Heap()};
        commandList->SetDescriptorHeaps(_countof(heaps), heaps);
        // RecordDispatch also resets the compute root signature and the
        // pipeline state, so a caller with its own compute state must rebind.
        m_recorder->RecordDispatch(commandList, m_graph.Get(), m_bindingTable.Get());
        PIXEndEvent(commandList);
        return true;
    }

    bool Enqueue(std::string* error) override {
        // Recording onto this runner's own list and submitting it, so the work
        // is ordered against the renderer's submissions on the shared queue
        // exactly as the ORT backend's is.
        if (!BeginList(error)) {
            return false;
        }
        if (!RecordInto(m_commandList.Get())) {
            SetError(error, "nothing to record");
            return false;
        }
        if (FAILED(m_commandList->Close())) {
            SetError(error, "command list close failed");
            return false;
        }
        ID3D12CommandList* lists[] = {m_commandList.Get()};
        m_queue->ExecuteCommandLists(1, lists);
        return true;
    }

    bool RunInference(const void* input,
                      std::size_t inputBytes,
                      std::vector<float>& output,
                      std::string* error) override;

private:
    bool BeginList(std::string* error) {
        // The allocator may still back an in-flight submission from a previous
        // Enqueue, so wait on THIS runner's fence rather than the whole GPU.
        if (m_pendingFence != 0 && m_fence->GetCompletedValue() < m_pendingFence) {
            if (!WaitForFence(m_pendingFence, error)) {
                return false;
            }
        }
        if (FAILED(m_allocator->Reset()) ||
            FAILED(m_commandList->Reset(m_allocator.Get(), nullptr))) {
            SetError(error, "command list reset failed");
            return false;
        }
        return true;
    }

    bool WaitForFence(uint64_t value, std::string* error) {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event == nullptr) {
            SetError(error, "could not create a wait event");
            return false;
        }
        m_fence->SetEventOnCompletion(value, event);
        const DWORD waited = WaitForSingleObject(event, 60000);
        CloseHandle(event);
        if (waited != WAIT_OBJECT_0) {
            SetError(error, "timed out waiting for the GPU");
            return false;
        }
        return true;
    }

    bool SubmitAndWait(std::string* error) {
        if (FAILED(m_commandList->Close())) {
            SetError(error, "command list close failed");
            return false;
        }
        ID3D12CommandList* lists[] = {m_commandList.Get()};
        m_queue->ExecuteCommandLists(1, lists);
        ++m_pendingFence;
        if (FAILED(m_queue->Signal(m_fence.Get(), m_pendingFence))) {
            SetError(error, "fence signal failed");
            return false;
        }
        return WaitForFence(m_pendingFence, error);
    }

    bool UploadWeights(const std::vector<ModelWeight>& weights, std::string* error);
    bool InitializeOperator(std::string* error);

    ID3D12Device* m_device = nullptr;
    ID3D12CommandQueue* m_queue = nullptr;
    IDMLDevice* m_dmlDevice = nullptr;
    IDMLCommandRecorder* m_recorder = nullptr;

    std::unique_ptr<NNModel> m_source;

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_graph;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
    std::unique_ptr<DirectX::DescriptorHeap> m_descriptorHeap;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_input;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_output;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_readback;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_weights;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_temporary;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistent;

    dml::TensorDesc m_inputDesc;
    dml::TensorDesc m_outputDesc;
    std::vector<DML_BUFFER_BINDING> m_weightBindings;

    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    uint64_t m_pendingFence = 0;

    ModelInfo m_info;
};

bool DmlGraphRunner::UploadWeights(const std::vector<ModelWeight>& weights,
                                   std::string* error) {
    m_weightBindings.clear();
    m_info.weightCount = weights.size();
    m_info.weightBytes = 0;
    if (weights.empty()) {
        return true;
    }

    // The initializer binds weights as one array in graph input order, so the
    // declarations must cover 1..N with no gaps and no repeats. A gap would
    // shift every later weight onto the wrong tensor, which trains nothing and
    // crashes nowhere.
    std::vector<const ModelWeight*> ordered(weights.size(), nullptr);
    for (const ModelWeight& weight : weights) {
        const uint32_t index = weight.graphInputIndex;
        if (index < 1 || index > weights.size()) {
            SetError(error, "weight '" + weight.name + "' has graph input index " +
                                std::to_string(index) + ", outside 1.." +
                                std::to_string(weights.size()));
            return false;
        }
        if (ordered[index - 1] != nullptr) {
            SetError(error,
                     "two weights both claim graph input index " + std::to_string(index));
            return false;
        }
        ordered[index - 1] = &weight;
    }

    std::vector<uint64_t> offsets(ordered.size());
    uint64_t total = 0;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        const ModelWeight& weight = *ordered[i];
        const uint64_t declared = weight.desc.totalTensorSizeInBytes;
        if (weight.data.size() != declared) {
            SetError(error, "weight '" + weight.name + "' supplies " +
                                std::to_string(weight.data.size()) +
                                " bytes but its tensor is " + std::to_string(declared));
            return false;
        }
        offsets[i] = total;
        total += RoundUpToMultiple<uint64_t>(declared, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);
    }

    if (FAILED(CreateDefaultBuffer(m_device, total, D3D12_RESOURCE_STATE_COMMON,
                                   m_weights))) {
        SetError(error, "weight buffer creation failed");
        return false;
    }

    // Staged through an UPLOAD buffer rather than written directly: the weights
    // live in a DEFAULT heap, which is not CPU-visible. The pre-refactor code
    // created the buffer and never wrote it at all, so inference read whatever
    // the heap happened to contain.
    Microsoft::WRL::ComPtr<ID3D12Resource> staging;
    {
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(total);
        if (FAILED(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nullptr, IID_PPV_ARGS(&staging)))) {
            SetError(error, "weight staging buffer creation failed");
            return false;
        }
    }

    uint8_t* mapped = nullptr;
    if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) {
        SetError(error, "weight staging buffer map failed");
        return false;
    }
    std::memset(mapped, 0, static_cast<std::size_t>(total));
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        std::memcpy(mapped + offsets[i], ordered[i]->data.data(), ordered[i]->data.size());
    }
    staging->Unmap(0, nullptr);

    if (!BeginList(error)) {
        return false;
    }
    m_commandList->CopyBufferRegion(m_weights.Get(), 0, staging.Get(), 0, total);
    const auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_weights.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_commandList->ResourceBarrier(1, &barrier);
    if (!SubmitAndWait(error)) {
        return false;
    }
    // staging stays alive until here because SubmitAndWait blocks.

    m_weightBindings.resize(ordered.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        m_weightBindings[i] = DML_BUFFER_BINDING{
            m_weights.Get(), offsets[i],
            RoundUpToMultiple<uint64_t>(ordered[i]->desc.totalTensorSizeInBytes,
                                        DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)};
    }
    m_info.weightBytes = total;
    return true;
}

bool DmlGraphRunner::InitializeOperator(std::string* error) {
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator* graphs[] = {m_graph.Get()};
    if (FAILED(m_dmlDevice->CreateOperatorInitializer(1, graphs,
                                                      IID_PPV_ARGS(&initializer)))) {
        SetError(error, "CreateOperatorInitializer failed");
        return false;
    }

    const DML_BINDING_PROPERTIES initProps = initializer->GetBindingProperties();
    const DML_BINDING_PROPERTIES execProps = m_graph->GetBindingProperties();

    // One heap per model, holding the initializer table then the execute table,
    // so registering a model cannot disturb one already registered.
    m_descriptorHeap = std::make_unique<DirectX::DescriptorHeap>(
        m_device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        initProps.RequiredDescriptorCount + execProps.RequiredDescriptorCount);

    m_info.temporaryBytes = execProps.TemporaryResourceSize;
    m_info.persistentBytes = execProps.PersistentResourceSize;

    if (execProps.PersistentResourceSize > 0 &&
        FAILED(CreateDefaultBuffer(m_device, execProps.PersistentResourceSize,
                                   D3D12_RESOURCE_STATE_COMMON, m_persistent))) {
        SetError(error, "persistent resource creation failed");
        return false;
    }
    if (execProps.TemporaryResourceSize > 0 &&
        FAILED(CreateDefaultBuffer(m_device, execProps.TemporaryResourceSize,
                                   D3D12_RESOURCE_STATE_COMMON, m_temporary))) {
        SetError(error, "temporary resource creation failed");
        return false;
    }

    // Initialization may need more scratch than execution does; reuse the
    // execute buffer when it is already big enough.
    Microsoft::WRL::ComPtr<ID3D12Resource> initTemporary;
    if (initProps.TemporaryResourceSize > execProps.TemporaryResourceSize) {
        if (FAILED(CreateDefaultBuffer(m_device, initProps.TemporaryResourceSize,
                                       D3D12_RESOURCE_STATE_COMMON, initTemporary))) {
            SetError(error, "initializer temporary resource creation failed");
            return false;
        }
    } else if (initProps.TemporaryResourceSize > 0) {
        initTemporary = m_temporary;
    }

    Microsoft::WRL::ComPtr<IDMLBindingTable> initTable;
    DML_BINDING_TABLE_DESC tableDesc = {initializer.Get(), m_descriptorHeap->GetCpuHandle(0),
                                        m_descriptorHeap->GetGpuHandle(0),
                                        initProps.RequiredDescriptorCount};
    if (FAILED(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initTable)))) {
        SetError(error, "initializer binding table creation failed");
        return false;
    }

    tableDesc = {m_graph.Get(),
                 m_descriptorHeap->GetCpuHandle(initProps.RequiredDescriptorCount),
                 m_descriptorHeap->GetGpuHandle(initProps.RequiredDescriptorCount),
                 execProps.RequiredDescriptorCount};
    if (FAILED(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&m_bindingTable)))) {
        SetError(error, "execute binding table creation failed");
        return false;
    }

    // Weights are declared DML_TENSOR_FLAG_OWNED_BY_DML, which means DirectML
    // takes a copy during initialization into its own layout. They are bound
    // HERE and deliberately NOT at execute time.
    {
        std::vector<DML_BUFFER_BINDING> bindings;
        bindings.reserve(1 + m_weightBindings.size());
        bindings.push_back(DML_BUFFER_BINDING{});  // input 0, supplied at execute
        bindings.insert(bindings.end(), m_weightBindings.begin(), m_weightBindings.end());

        DML_BUFFER_ARRAY_BINDING arrayBinding = {static_cast<UINT>(bindings.size()),
                                                 bindings.data()};
        DML_BINDING_DESC desc = {DML_BINDING_TYPE_BUFFER_ARRAY, &arrayBinding};
        initTable->BindInputs(1, &desc);
    }

    if (initTemporary) {
        DML_BUFFER_BINDING binding = {initTemporary.Get(), 0, initTemporary->GetDesc().Width};
        DML_BINDING_DESC desc = {DML_BINDING_TYPE_BUFFER, &binding};
        initTable->BindTemporaryResource(&desc);
    }

    // A persistent resource is an OUTPUT of the initializer and an input to
    // every later execution.
    if (m_persistent) {
        DML_BUFFER_BINDING binding = {m_persistent.Get(), 0, m_persistent->GetDesc().Width};
        DML_BINDING_DESC desc = {DML_BINDING_TYPE_BUFFER, &binding};
        initTable->BindOutputs(1, &desc);
        m_bindingTable->BindPersistentResource(&desc);
    }

    if (m_temporary) {
        DML_BUFFER_BINDING binding = {m_temporary.Get(), 0, m_temporary->GetDesc().Width};
        DML_BINDING_DESC desc = {DML_BINDING_TYPE_BUFFER, &binding};
        m_bindingTable->BindTemporaryResource(&desc);
    }

    // Input 0 is the only real input; the weight slots are bound as NONE
    // because DirectML already owns those tensors after initialization.
    DML_BUFFER_BINDING inputBinding = {m_input.Get(), 0, m_input->GetDesc().Width};
    std::vector<DML_BINDING_DESC> inputDescs(1 + m_weightBindings.size());
    inputDescs[0] = {DML_BINDING_TYPE_BUFFER, &inputBinding};
    for (std::size_t i = 1; i < inputDescs.size(); ++i) {
        inputDescs[i] = {DML_BINDING_TYPE_NONE, nullptr};
    }
    m_bindingTable->BindInputs(static_cast<UINT>(inputDescs.size()), inputDescs.data());

    DML_BUFFER_BINDING outputBinding = {m_output.Get(), 0, m_output->GetDesc().Width};
    DML_BINDING_DESC outputDesc = {DML_BINDING_TYPE_BUFFER, &outputBinding};
    m_bindingTable->BindOutputs(1, &outputDesc);

    if (!BeginList(error)) {
        return false;
    }
    ID3D12DescriptorHeap* heaps[] = {m_descriptorHeap->Heap()};
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_recorder->RecordDispatch(m_commandList.Get(), initializer.Get(), initTable.Get());
    return SubmitAndWait(error);
}

bool DmlGraphRunner::Build(std::unique_ptr<NNModel> source, std::string* error) {
    m_info.name = source->Name();
    m_info.backend = ModelBackend::DirectMLGraph;
    m_inputDesc = source->InputDesc();

    if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&m_allocator))) ||
        FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           m_allocator.Get(), nullptr,
                                           IID_PPV_ARGS(&m_commandList))) ||
        FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        SetError(error, "could not create the runner's command list");
        return false;
    }
    m_commandList->Close();

    dml::Graph graph(m_dmlDevice);
    auto input = dml::InputTensor(graph, 0, m_inputDesc);

    ModelGraph built{input, {}};
    try {
        built = source->Build(graph, input);
    } catch (const std::exception& e) {
        // OnnxModel reports an untranslated operator this way, and DirectMLX
        // itself throws on a shape it cannot express.
        SetError(error, "building '" + m_info.name + "' failed: " + e.what());
        return false;
    }
    m_outputDesc = built.output.GetOutputDesc();

    // TODO consider DML_EXECUTION_FLAG_DISABLE_META_COMMANDS to keep vendor
    // metacommands out of the way while shader-debugging.
    const DML_EXECUTION_FLAGS executionFlags =
        DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    try {
        m_graph = graph.Compile(executionFlags, {built.output});
    } catch (const std::exception& e) {
        SetError(error, "compiling '" + m_info.name + "' failed: " + e.what());
        return false;
    }
    if (!m_graph) {
        SetError(error, "graph compilation failed for '" + m_info.name + "'");
        return false;
    }

    const uint64_t inputBytes = m_inputDesc.totalTensorSizeInBytes;
    const uint64_t outputBytes = m_outputDesc.totalTensorSizeInBytes;

    if (FAILED(CreateDefaultBuffer(m_device, inputBytes, D3D12_RESOURCE_STATE_COMMON,
                                   m_input))) {
        SetError(error, "input buffer creation failed");
        return false;
    }
    if (FAILED(CreateDefaultBuffer(m_device, outputBytes, D3D12_RESOURCE_STATE_COMMON,
                                   m_output))) {
        SetError(error, "output buffer creation failed");
        return false;
    }
    {
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(outputBytes);
        if (FAILED(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&m_readback)))) {
            SetError(error, "readback buffer creation failed");
            return false;
        }
    }

    m_info.inputSizes.assign(m_inputDesc.sizes.begin(), m_inputDesc.sizes.end());
    m_info.outputSizes.assign(m_outputDesc.sizes.begin(), m_outputDesc.sizes.end());
    m_info.inputBytes = inputBytes;
    m_info.outputBytes = outputBytes;

    if (!UploadWeights(built.weights, error)) {
        return false;
    }
    if (!InitializeOperator(error)) {
        return false;
    }

    m_source = std::move(source);
    return true;
}

bool DmlGraphRunner::RunInference(const void* input,
                                  std::size_t inputBytes,
                                  std::vector<float>& output,
                                  std::string* error) {
    if (inputBytes != m_info.inputBytes) {
        SetError(error, "input is " + std::to_string(inputBytes) + " bytes but '" +
                            m_info.name + "' expects " + std::to_string(m_info.inputBytes));
        return false;
    }
    if (m_outputDesc.dataType != DML_TENSOR_DATA_TYPE_FLOAT32) {
        SetError(error, "output of '" + m_info.name +
                            "' is not FLOAT32; readback not supported");
        return false;
    }

    Microsoft::WRL::ComPtr<ID3D12Resource> staging;
    {
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(inputBytes);
        if (FAILED(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nullptr, IID_PPV_ARGS(&staging)))) {
            SetError(error, "input staging buffer creation failed");
            return false;
        }
    }
    void* mapped = nullptr;
    if (FAILED(staging->Map(0, nullptr, &mapped))) {
        SetError(error, "input staging buffer map failed");
        return false;
    }
    std::memcpy(mapped, input, inputBytes);
    staging->Unmap(0, nullptr);

    if (!BeginList(error)) {
        return false;
    }

    auto toCopyDest = CD3DX12_RESOURCE_BARRIER::Transition(
        m_input.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST);
    m_commandList->ResourceBarrier(1, &toCopyDest);
    m_commandList->CopyBufferRegion(m_input.Get(), 0, staging.Get(), 0, inputBytes);
    auto toUav = CD3DX12_RESOURCE_BARRIER::Transition(m_input.Get(),
                                                      D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_commandList->ResourceBarrier(1, &toUav);

    ID3D12DescriptorHeap* heaps[] = {m_descriptorHeap->Heap()};
    m_commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    m_recorder->RecordDispatch(m_commandList.Get(), m_graph.Get(), m_bindingTable.Get());

    auto outToCopySrc = CD3DX12_RESOURCE_BARRIER::Transition(
        m_output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    m_commandList->ResourceBarrier(1, &outToCopySrc);
    m_commandList->CopyBufferRegion(m_readback.Get(), 0, m_output.Get(), 0,
                                    m_info.outputBytes);
    auto outBack = CD3DX12_RESOURCE_BARRIER::Transition(m_output.Get(),
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_commandList->ResourceBarrier(1, &outBack);

    if (!SubmitAndWait(error)) {
        return false;
    }

    void* readMapped = nullptr;
    if (FAILED(m_readback->Map(0, nullptr, &readMapped))) {
        SetError(error, "readback map failed");
        return false;
    }
    output.resize(static_cast<std::size_t>(m_info.outputBytes / sizeof(float)));
    std::memcpy(output.data(), readMapped, output.size() * sizeof(float));
    m_readback->Unmap(0, nullptr);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelManager
// ---------------------------------------------------------------------------

void ModelManager::Initialize(std::shared_ptr<DX::DeviceResources> deviceResources) {
    m_deviceResources = std::move(deviceResources);
    auto device = m_deviceResources->GetD3DDevice();

#if _DEBUG
    DX::ThrowIfFailed(
        DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice)));
#else
    DX::ThrowIfFailed(
        DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
#endif

    // DirectML uses a CommandRecorder to turn operator dispatches into ordinary
    // command list commands, which is what lets DirectMLGraph models be
    // recorded into the same list as rendering.
    DX::ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder)));
}

bool ModelManager::AddModel(std::unique_ptr<NNModel> source, std::string* error) {
    if (!m_dmlDevice) {
        SetError(error, "ModelManager::Initialize has not run");
        return false;
    }
    if (source == nullptr) {
        SetError(error, "null model");
        return false;
    }
    const std::string name = source->Name();
    if (name.empty()) {
        SetError(error, "model name is empty");
        return false;
    }
    if (m_models.find(name) != m_models.end()) {
        SetError(error, "a model named '" + name + "' is already registered");
        return false;
    }

    // Built into a local and only moved into the registry on success, so a
    // failure part way through cannot leave a half-built model findable.
    auto runner = std::make_unique<DmlGraphRunner>(m_deviceResources->GetD3DDevice(),
                                                   m_deviceResources->GetCommandQueue(),
                                                   m_dmlDevice.Get(),
                                                   m_dmlCommandRecorder.Get());
    if (!runner->Build(std::move(source), error)) {
        return false;
    }
    m_models.emplace(name, std::move(runner));
    return true;
}

bool ModelManager::AddOnnxModel(const std::string& name,
                                const std::filesystem::path& path,
                                ModelBackend backend,
                                std::string* error) {
    if (!m_dmlDevice) {
        SetError(error, "ModelManager::Initialize has not run");
        return false;
    }
    if (m_models.find(name) != m_models.end()) {
        SetError(error, "a model named '" + name + "' is already registered");
        return false;
    }

    if (backend == ModelBackend::DirectMLGraph) {
        auto translated = OnnxModel::Load(name, path, error);
        if (translated == nullptr) {
            return false;
        }
        return AddModel(std::move(translated), error);
    }

    auto runner = OrtModelRunner::Create(name, path, m_deviceResources->GetD3DDevice(),
                                         m_deviceResources->GetCommandQueue(),
                                         m_dmlDevice.Get(), error);
    if (runner == nullptr) {
        return false;
    }
    m_models.emplace(name, std::move(runner));
    return true;
}

bool ModelManager::RecordModelDispatch(ID3D12GraphicsCommandList* commandList,
                                       const std::string& name) {
    const auto it = m_models.find(name);
    if (it == m_models.end()) {
        return false;
    }
    // An OnnxRuntime model returns false here: ORT builds its own command
    // lists, so there is nothing to append. Use EnqueueModel for those.
    return it->second->RecordInto(commandList);
}

bool ModelManager::EnqueueModel(const std::string& name, std::string* error) {
    const auto it = m_models.find(name);
    if (it == m_models.end()) {
        SetError(error, "no model named '" + name + "'");
        return false;
    }
    return it->second->Enqueue(error);
}

bool ModelManager::RunInference(const std::string& name,
                                const void* input,
                                std::size_t inputBytes,
                                std::vector<float>& output,
                                std::string* error) {
    const auto it = m_models.find(name);
    if (it == m_models.end()) {
        SetError(error, "no model named '" + name + "'");
        return false;
    }
    return it->second->RunInference(input, inputBytes, output, error);
}

bool ModelManager::Has(const std::string& name) const {
    return m_models.find(name) != m_models.end();
}

std::vector<std::string> ModelManager::ModelNames() const {
    std::vector<std::string> names;
    names.reserve(m_models.size());
    for (const auto& entry : m_models) {
        names.push_back(entry.first);
    }
    return names;
}

const ModelInfo* ModelManager::Info(const std::string& name) const {
    const auto it = m_models.find(name);
    return it == m_models.end() ? nullptr : &it->second->Info();
}

ID3D12Resource* ModelManager::InputResource(const std::string& name) {
    const auto it = m_models.find(name);
    return it == m_models.end() ? nullptr : it->second->InputResource();
}

ID3D12Resource* ModelManager::OutputResource(const std::string& name) {
    const auto it = m_models.find(name);
    return it == m_models.end() ? nullptr : it->second->OutputResource();
}

void ModelManager::ShowImgui() {
    if (!ImGui::CollapsingHeader("Neural models")) {
        return;
    }
    if (m_models.empty()) {
        ImGui::TextUnformatted("No models registered.");
        return;
    }

    const auto shape = [](const std::vector<uint32_t>& sizes) {
        std::string out;
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            out += (i == 0 ? "" : "x") + std::to_string(sizes[i]);
        }
        return out;
    };

    ImGui::Text("%zu model(s)", m_models.size());
    for (const auto& entry : m_models) {
        const ModelInfo& info = entry.second->Info();
        if (!ImGui::TreeNode(info.name.c_str())) {
            continue;
        }
        ImGui::Text("Backend: %s", ToString(info.backend));
        // Submits on the shared queue. Safe to press at any time: the work is
        // ordered after whatever the renderer has already submitted.
        if (ImGui::Button("Enqueue now")) {
            std::string enqueueError;
            if (!entry.second->Enqueue(&enqueueError)) {
                std::printf("enqueueing '%s' failed: %s\n", info.name.c_str(),
                            enqueueError.c_str());
            }
        }
        ImGui::Text("Input:  %s  (%llu bytes)", shape(info.inputSizes).c_str(),
                    static_cast<unsigned long long>(info.inputBytes));
        ImGui::Text("Output: %s  (%llu bytes)", shape(info.outputSizes).c_str(),
                    static_cast<unsigned long long>(info.outputBytes));
        if (info.backend == ModelBackend::DirectMLGraph) {
            ImGui::Text("Weights: %zu tensors, %llu bytes", info.weightCount,
                        static_cast<unsigned long long>(info.weightBytes));
            ImGui::Text("Scratch: %llu temporary, %llu persistent",
                        static_cast<unsigned long long>(info.temporaryBytes),
                        static_cast<unsigned long long>(info.persistentBytes));
            ImGui::TextUnformatted("Recordable into the render command list");
        } else {
            // ORT does not report the session's weight footprint, and guessing
            // would be worse than saying nothing.
            ImGui::TextUnformatted("Weights are held inside the ORT session");
            ImGui::TextUnformatted("Submits its own command lists on the shared queue");
        }
        ImGui::TreePop();
    }
}

// ---------------------------------------------------------------------------
// FloodDiffusionNNModel
// ---------------------------------------------------------------------------

dml::TensorDesc FloodDiffusionNNModel::InputDesc() const {
    // TODO the diffusion model proper wants (Batch, Length, Features) =
    // { 1, 128, 512 }; this is the convolution test shape.
    return dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32,
                           dml::TensorDesc::Dimensions{1, 3, 256, 256});
}

ModelGraph FloodDiffusionNNModel::Build(dml::Graph& graph, dml::Expression input) {
    ModelGraph result{input, {}};

    // OWNED_BY_DML lets DirectML keep its own reordered copy of the weights,
    // which is why they are bound to the initializer and not at execute time.
    const DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_OWNED_BY_DML;

    uint32_t nextInput = 1;

    const dml::TensorDesc::Dimensions filterShape = {32, 3, 3, 3};
    const dml::TensorDesc filterDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags, filterShape);
    auto filter = dml::InputTensor(graph, nextInput, filterDesc);
    // Zeros keep this honest: the point of this model is to exercise the
    // manager, and the smoke test asserts on values it can predict.
    result.weights.push_back(ModelWeight{
        "conv.filter", nextInput, filterDesc,
        std::vector<uint8_t>(static_cast<std::size_t>(filterDesc.totalTensorSizeInBytes), 0)});
    ++nextInput;

    const dml::TensorDesc::Dimensions biasShape = {1, filterShape[0], 1, 1};
    const dml::TensorDesc biasDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags, biasShape);
    auto bias = dml::InputTensor(graph, nextInput, biasDesc);
    result.weights.push_back(ModelWeight{
        "conv.bias", nextInput, biasDesc,
        std::vector<uint8_t>(static_cast<std::size_t>(biasDesc.totalTensorSizeInBytes), 0)});
    ++nextInput;

    const uint32_t filterHeight = filterShape[2];
    const uint32_t filterWidth = filterShape[3];
    const std::array<uint32_t, 2> padding = {filterHeight / 2, filterWidth / 2};
    const std::array<uint32_t, 2> strides = {1, 1};

    result.output = dml::ConvolutionBuilder(input, filter, bias)
                        .StartPadding(padding)
                        .EndPadding(padding)
                        .Strides(strides)
                        .FusedActivation(dml::FusedActivation::LeakyRelu(0.1f))
                        .Build();
    return result;
}

}  // namespace NeuralModelIntegrateTestbed
