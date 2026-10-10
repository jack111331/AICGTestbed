#include "pch.h"
#include "OrtModel.hpp"
#include "OrtEnv.hpp"

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <stdexcept>

namespace NeuralModelIntegrateTestbed {

namespace {

// The environment, the DirectML function table and the error convention are
// shared with Engine/FloodDiffusion.cc through OrtEnv.hpp: one Ort::Env has to
// cover every session in the process.
void SetError(std::string* error, std::string message) {
    SetOrtError(error, std::move(message));
}

Ort::Env& SharedEnv() {
    return SharedOrtEnv();
}

const OrtDmlApi* DmlApi(std::string* error) {
    return SharedDmlApi(error);
}

uint64_t ElementCount(const std::vector<int64_t>& shape) {
    uint64_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<uint64_t>(dim <= 0 ? 1 : dim);
    }
    return count;
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device,
                                                    uint64_t bytes,
                                                    D3D12_RESOURCE_STATES state,
                                                    D3D12_HEAP_TYPE heapType) {
    Microsoft::WRL::ComPtr<ID3D12Resource> out;
    const auto heap = CD3DX12_HEAP_PROPERTIES(heapType);
    const auto flags = heapType == D3D12_HEAP_TYPE_DEFAULT
                           ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                           : D3D12_RESOURCE_FLAG_NONE;
    const auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
                                               nullptr, IID_PPV_ARGS(&out)))) {
        return nullptr;
    }
    return out;
}

}  // namespace

struct OrtModelRunner::Impl {
    ModelInfo info;

    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    const OrtDmlApi* dmlApi = nullptr;

    std::unique_ptr<Ort::SessionOptions> options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;

    std::string inputName;
    std::string outputName;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;

    // The buffers ORT reads and writes, owned here so a caller can feed the
    // input from its own passes and sample the output afterwards.
    Microsoft::WRL::ComPtr<ID3D12Resource> input;
    Microsoft::WRL::ComPtr<ID3D12Resource> output;

    // Opaque DML EP allocations wrapping the two resources above. They must
    // outlive the Ort::Values built on them, and be released with
    // FreeGPUAllocation rather than delete.
    void* inputAllocation = nullptr;
    void* outputAllocation = nullptr;

    // Rebuilt lazily: Ort::Value is move-only and holds no ownership of the
    // allocation, so keeping them is purely to avoid re-wrapping per run.
    std::unique_ptr<Ort::Value> inputValue;
    std::unique_ptr<Ort::Value> outputValue;

    // For RunInference's upload and readback. Its own allocator and list, so
    // nothing is borrowed from the renderer.
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    uint64_t fenceValue = 0;

    ~Impl() {
        // Ort::Values first: they reference the allocations.
        inputValue.reset();
        outputValue.reset();
        session.reset();
        if (dmlApi != nullptr) {
            if (inputAllocation != nullptr) dmlApi->FreeGPUAllocation(inputAllocation);
            if (outputAllocation != nullptr) dmlApi->FreeGPUAllocation(outputAllocation);
        }
    }

    bool WaitForGpu(std::string* error) {
        ++fenceValue;
        if (FAILED(queue->Signal(fence.Get(), fenceValue))) {
            SetError(error, "fence signal failed");
            return false;
        }
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event == nullptr) {
            SetError(error, "could not create a wait event");
            return false;
        }
        fence->SetEventOnCompletion(fenceValue, event);
        const DWORD waited = WaitForSingleObject(event, 60000);
        CloseHandle(event);
        if (waited != WAIT_OBJECT_0) {
            SetError(error, "timed out waiting for the GPU");
            return false;
        }
        return true;
    }
};

OrtModelRunner::OrtModelRunner() : m_impl(std::make_unique<Impl>()) {}
OrtModelRunner::~OrtModelRunner() = default;

const ModelInfo& OrtModelRunner::Info() const { return m_impl->info; }

bool OrtModelRunner::RecordInto(ID3D12GraphicsCommandList*) {
    // ONNX Runtime builds and submits its own command lists. Use Enqueue; the
    // work still lands on the shared queue and is ordered against the
    // renderer's submissions.
    return false;
}

ID3D12Resource* OrtModelRunner::InputResource() { return m_impl->input.Get(); }
ID3D12Resource* OrtModelRunner::OutputResource() { return m_impl->output.Get(); }

std::unique_ptr<OrtModelRunner> OrtModelRunner::Create(const std::string& name,
                                                       const std::filesystem::path& path,
                                                       ID3D12Device* device,
                                                       ID3D12CommandQueue* queue,
                                                       IDMLDevice* dmlDevice,
                                                       std::string* error) {
    const auto fail = [&](const std::string& message) -> std::unique_ptr<OrtModelRunner> {
        SetError(error, message);
        return nullptr;
    };

    if (!std::filesystem::exists(path)) {
        return fail("cannot open '" + path.string() + "'");
    }
    const OrtDmlApi* dmlApi = DmlApi(error);
    if (dmlApi == nullptr) {
        return nullptr;
    }

    auto runner = std::unique_ptr<OrtModelRunner>(new OrtModelRunner());
    Impl& impl = *runner->m_impl;
    impl.device = device;
    impl.queue = queue;
    impl.dmlApi = dmlApi;
    impl.info.name = name;
    impl.info.backend = ModelBackend::OnnxRuntime;

    try {
        impl.options = std::make_unique<Ort::SessionOptions>();
        // Both are required by the DirectML EP: it does not implement ORT's
        // memory-pattern optimisation, and parallel execution is unsupported.
        // Advisable per DirectML's documentation, though measured on ORT
        // 1.24.4 a session runs correctly without them; see
        // skills/onnx-directml-ep/SKILL.md.
        impl.options->DisableMemPattern();
        impl.options->SetExecutionMode(ORT_SEQUENTIAL);

        const OrtStatus* status = dmlApi->SessionOptionsAppendExecutionProvider_DML1(
            *impl.options, dmlDevice, queue);
        if (status != nullptr) {
            const std::string message = Ort::GetApi().GetErrorMessage(status);
            Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
            return fail("attaching the DirectML EP failed: " + message);
        }

        impl.session = std::make_unique<Ort::Session>(SharedEnv(), path.c_str(),
                                                      *impl.options);
    } catch (const Ort::Exception& e) {
        return fail(std::string("ONNX Runtime rejected '") + path.string() + "': " + e.what());
    }

    // This wrapper drives one input and one output, matching the rest of
    // ModelManager. A multi-input model is reported rather than half-bound.
    if (impl.session->GetInputCount() != 1 || impl.session->GetOutputCount() != 1) {
        return fail("'" + path.string() + "' has " +
                    std::to_string(impl.session->GetInputCount()) + " inputs and " +
                    std::to_string(impl.session->GetOutputCount()) +
                    " outputs; exactly one of each is supported");
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        impl.inputName = impl.session->GetInputNameAllocated(0, allocator).get();
        impl.outputName = impl.session->GetOutputNameAllocated(0, allocator).get();

        // The Ort::TypeInfo must be held in a named local. It OWNS the
        // OrtTypeInfo, while GetTensorTypeAndShapeInfo returns a
        // ConstTensorTypeAndShapeInfo that merely points into it -- so calling
        // both in one expression leaves the view dangling the moment the
        // temporary TypeInfo is destroyed. That read-after-free returned the
        // right element type for a while and then started reporting a
        // non-FLOAT32 tensor, failing every ONNX Runtime model here.
        const Ort::TypeInfo inputTypeInfo = impl.session->GetInputTypeInfo(0);
        const Ort::TypeInfo outputTypeInfo = impl.session->GetOutputTypeInfo(0);
        const auto inputInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        const auto outputInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

        if (inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return fail("only FLOAT32 input and output are supported");
        }

        impl.inputShape = inputInfo.GetShape();
        impl.outputShape = outputInfo.GetShape();
    } catch (const Ort::Exception& e) {
        return fail(std::string("reading the model's signature failed: ") + e.what());
    }

    // A symbolic dimension comes back as -1. DirectML needs a concrete size, so
    // it is pinned to 1 -- the same choice the hand translator makes.
    for (int64_t& dim : impl.inputShape) {
        if (dim <= 0) dim = 1;
    }
    for (int64_t& dim : impl.outputShape) {
        if (dim <= 0) dim = 1;
    }

    const uint64_t inputBytes = ElementCount(impl.inputShape) * sizeof(float);
    const uint64_t outputBytes = ElementCount(impl.outputShape) * sizeof(float);

    impl.input = CreateBuffer(device, inputBytes, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_HEAP_TYPE_DEFAULT);
    impl.output = CreateBuffer(device, outputBytes, D3D12_RESOURCE_STATE_COMMON,
                               D3D12_HEAP_TYPE_DEFAULT);
    if (!impl.input || !impl.output) {
        return fail("could not create the input/output buffers");
    }

    if (dmlApi->CreateGPUAllocationFromD3DResource(impl.input.Get(),
                                                   &impl.inputAllocation) != nullptr ||
        dmlApi->CreateGPUAllocationFromD3DResource(impl.output.Get(),
                                                   &impl.outputAllocation) != nullptr) {
        return fail("wrapping the D3D12 resources as DML allocations failed");
    }

    try {
        // "DML" memory means the tensor data lives in the allocation above
        // rather than in CPU memory, which is what keeps this copy-free.
        impl.memoryInfo = std::make_unique<Ort::MemoryInfo>("DML", OrtDeviceAllocator, 0,
                                                            OrtMemTypeDefault);
        impl.inputValue = std::make_unique<Ort::Value>(Ort::Value::CreateTensor(
            *impl.memoryInfo, impl.inputAllocation, inputBytes, impl.inputShape.data(),
            impl.inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        impl.outputValue = std::make_unique<Ort::Value>(Ort::Value::CreateTensor(
            *impl.memoryInfo, impl.outputAllocation, outputBytes, impl.outputShape.data(),
            impl.outputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    } catch (const Ort::Exception& e) {
        return fail(std::string("binding the D3D12 resources to tensors failed: ") +
                    e.what());
    }

    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              IID_PPV_ARGS(&impl.allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         impl.allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&impl.commandList))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&impl.fence)))) {
        return fail("could not create the runner's command list");
    }
    impl.commandList->Close();

    impl.info.inputBytes = inputBytes;
    impl.info.outputBytes = outputBytes;
    for (const int64_t dim : impl.inputShape) {
        impl.info.inputSizes.push_back(static_cast<uint32_t>(dim));
    }
    for (const int64_t dim : impl.outputShape) {
        impl.info.outputSizes.push_back(static_cast<uint32_t>(dim));
    }
    // Weights are inside the session; ORT does not report their size, and
    // guessing would be worse than leaving it zero.

    return runner;
}

bool OrtModelRunner::Enqueue(std::string* error) {
    Impl& impl = *m_impl;
    const char* inputNames[] = {impl.inputName.c_str()};
    const char* outputNames[] = {impl.outputName.c_str()};
    try {
        // Records and returns. The work is submitted on the queue the EP was
        // given, so it is ordered against the renderer's submissions without a
        // fence -- but a pass that reads the output must be submitted after.
        impl.session->Run(Ort::RunOptions{nullptr}, inputNames, impl.inputValue.get(), 1,
                          outputNames, impl.outputValue.get(), 1);
    } catch (const Ort::Exception& e) {
        SetError(error, std::string("Run failed: ") + e.what());
        return false;
    }
    return true;
}

bool OrtModelRunner::RunInference(const void* input,
                                  std::size_t inputBytes,
                                  std::vector<float>& output,
                                  std::string* error) {
    Impl& impl = *m_impl;

    if (inputBytes != impl.info.inputBytes) {
        SetError(error, "input is " + std::to_string(inputBytes) + " bytes but '" +
                            impl.info.name + "' expects " +
                            std::to_string(impl.info.inputBytes));
        return false;
    }

    auto staging = CreateBuffer(impl.device, inputBytes, D3D12_RESOURCE_STATE_GENERIC_READ,
                                D3D12_HEAP_TYPE_UPLOAD);
    auto readback = CreateBuffer(impl.device, impl.info.outputBytes,
                                 D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);
    if (!staging || !readback) {
        SetError(error, "could not create the staging buffers");
        return false;
    }

    void* mapped = nullptr;
    if (FAILED(staging->Map(0, nullptr, &mapped))) {
        SetError(error, "input staging map failed");
        return false;
    }
    std::memcpy(mapped, input, inputBytes);
    staging->Unmap(0, nullptr);

    // Upload on our own list and submit, then let ORT's work queue behind it.
    //
    // The drain is not optional. An allocator may only be reset once every
    // submission recorded from it has COMPLETED, and a previous Enqueue or
    // RunInference may still be in flight -- Enqueue explicitly does not wait.
    // Reset does not report this: it returns S_OK and corrupts the allocator,
    // and only the debug layer notices (EXECUTION ERROR #552,
    // COMMAND_ALLOCATOR_SYNC), which with break-on-error makes it fatal.
    if (!impl.WaitForGpu(error)) {
        return false;
    }
    if (FAILED(impl.allocator->Reset()) ||
        FAILED(impl.commandList->Reset(impl.allocator.Get(), nullptr))) {
        SetError(error, "command list reset failed");
        return false;
    }
    impl.commandList->CopyBufferRegion(impl.input.Get(), 0, staging.Get(), 0, inputBytes);
    if (FAILED(impl.commandList->Close())) {
        SetError(error, "command list close failed");
        return false;
    }
    ID3D12CommandList* lists[] = {impl.commandList.Get()};
    impl.queue->ExecuteCommandLists(1, lists);

    if (!Enqueue(error)) {
        return false;
    }

    // Queue order guarantees the upload ran before ORT's work and that ORT's
    // work ran before this copy, so no barriers between them are needed.
    //
    // Resetting the allocator is a different matter: ordering is not enough,
    // the earlier submissions have to have finished. So drain here too, before
    // reusing the allocator for the readback copy.
    if (!impl.WaitForGpu(error)) {
        return false;
    }
    if (FAILED(impl.allocator->Reset())) {
        SetError(error, "command allocator reset failed");
        return false;
    }
    if (FAILED(impl.commandList->Reset(impl.allocator.Get(), nullptr))) {
        SetError(error, "command list reset failed");
        return false;
    }
    impl.commandList->CopyBufferRegion(readback.Get(), 0, impl.output.Get(), 0,
                                       impl.info.outputBytes);
    impl.commandList->Close();
    impl.queue->ExecuteCommandLists(1, lists);

    if (!impl.WaitForGpu(error)) {
        return false;
    }

    void* readMapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &readMapped))) {
        SetError(error, "readback map failed");
        return false;
    }
    output.resize(static_cast<std::size_t>(impl.info.outputBytes / sizeof(float)));
    std::memcpy(output.data(), readMapped, output.size() * sizeof(float));
    readback->Unmap(0, nullptr);
    return true;
}

}  // namespace NeuralModelIntegrateTestbed
