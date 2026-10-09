#include "pch.h"
#include "ModelManager.hpp"
#include <DirectMLX.h>

using namespace DirectX;

namespace NeuralModelIntegrateTestbed {

void ModelManager::Initialize(std::shared_ptr<DX::DeviceResources> deviceResources) {
    auto device = deviceResources->GetD3DDevice();
    #if _DEBUG
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice)));
    #else
        DX::ThrowIfFailed(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
    #endif
    // DirectML use CommandRecorder to abstract the operator computing commands into CommandList
    DX::ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder)));

    // TODO test
    // https://learn.microsoft.com/en-us/windows/win32/api/directml/nf-directml-idmlcommandrecorder-recorddispatch
    // Describe, create, and compile upsample operator
    // To make NN model runs, we need to first RecordDispatch to Initialize compiled operators then manually run this operators, and we need to ensure synchronization using resource barrier
    // NVM, I saw DMLX encapsulation defintion https://github.com/microsoft/DirectML/blob/8700779fe7a09ea7a007cf3d7ab4293c78e41017/Samples/DirectMLSuperResolution/DirectMLXResourceBuilder.cpp#L255
    // https://github.com/microsoft/DirectML/blob/8700779fe7a09ea7a007cf3d7ab4293c78e41017/Samples/DirectMLSuperResolution/DirectMLResourceBuilder.cpp#L280
    // m_dmlCommandRecorder->RecordDispatch()
    // TODO I should use ONNX instead of create and initialize each operators required by models for efficiency
}

// TODO Create new ModelDescriptor class to describe Input, output?
void ModelManager::AddNewModel(std::shared_ptr<DX::DeviceResources> deviceResources) {
    auto commandList = deviceResources->GetCommandList();
    commandList->Reset(deviceResources->GetCommandAllocator(), nullptr);

    dml::Graph graph(m_dmlDevice.Get());
    // (Batch, Length, Features)
    uint32_t featureSize = 512;
    // TODO this is for FloodDiffusion
    // dml::TensorDesc::Dimensions inputSizes = { 1, 128, featureSize };
    // This is for test convolution
    dml::TensorDesc::Dimensions inputSizes = { 1, 3, 256, 256 };
    auto input = dml::InputTensor(graph, 0, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, inputSizes));

    uint64_t modelInputBufferSize = input.GetOutputDesc().totalTensorSizeInBytes;
    //  TODO construct model and retrieve output DML::Expression
    // Compile the model into a DML graph
    // TODO maybe consider adding DML_EXECUTION_FLAG_DISABLE_META_COMMANDS to prevent shader from using vendor-specific implementation that hinder shader debugging?
    
    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    NNModel *nnModel = new FloodDiffusionNNModel();
    // TODO should register weight of the model and test load it
    auto modelOutput = nnModel->ConstructModel(deviceResources, graph, input);
    m_dmlGraph = graph.Compile(executionFlags, { modelOutput });

   // Resource for input tensor
   auto device = deviceResources->GetD3DDevice();

    DX::ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(1, m_dmlGraph.GetAddressOf(), IID_PPV_ARGS(&m_dmlOpInitializer)));
    DML_BINDING_PROPERTIES initBindingProps = m_dmlOpInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProps = m_dmlGraph->GetBindingProperties();

    m_dmlDescriptorHeap = std::make_unique<DirectX::DescriptorHeap>(
        deviceResources->GetD3DDevice(),
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        initBindingProps.RequiredDescriptorCount + executeBindingProps.RequiredDescriptorCount + 1); // +1 is for model input uav descriptor

   DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(modelInputBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelInput)));

    // Describe and create a UAV for the original input tensor.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = static_cast<UINT>(modelInputBufferSize / sizeof(float));
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    device->CreateUnorderedAccessView(m_modelInput.Get(), nullptr, &uavDesc, m_dmlDescriptorHeap->GetCpuHandle(initBindingProps.RequiredDescriptorCount + executeBindingProps.RequiredDescriptorCount));

    // Create resources to hold the model outputs and to read them back from the GPU
    m_modelOutput.desc = modelOutput.GetOutputDesc();
    uint64_t sbboxResourceSize = m_modelOutput.desc.totalTensorSizeInBytes;
    DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(sbboxResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelOutput.output)));
    DX::ThrowIfFailed(device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(sbboxResourceSize),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_modelOutput.readback)));


    // TODO need to directly use command list to initialize operators
    ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
    commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

    // Create any persistent resources required for the operators.
    if (executeBindingProps.PersistentResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            executeBindingProps.PersistentResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelPersistentResource)));
    }

    // Temporary resource for execution
    if (executeBindingProps.TemporaryResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            executeBindingProps.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelTemporaryResource)));
    }

    // If the execute temporary resource isn't big enough for initialization, create a bigger buffer
    Microsoft::WRL::ComPtr<ID3D12Resource> initTemporaryResource;
    if (initBindingProps.TemporaryResourceSize > executeBindingProps.TemporaryResourceSize)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            initBindingProps.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&initTemporaryResource)));
    }
    else if (initBindingProps.TemporaryResourceSize > 0)
    {
        initTemporaryResource = m_modelTemporaryResource;
    }

    Microsoft::WRL::ComPtr<IDMLBindingTable> initBindingTable;

    // Binding table is like Descriptor table in D3D12, except it's for single model's list of UAV/SRV/CBV resources and don't require root parameter
    // TODO For now, we create another descriptor heap, but for future, we will integrate it into DynamicDescriptorHeap
    DML_BINDING_TABLE_DESC tableDesc =
    {
        m_dmlOpInitializer.Get(),
        m_dmlDescriptorHeap->GetCpuHandle(0),
        m_dmlDescriptorHeap->GetGpuHandle(0),
        initBindingProps.RequiredDescriptorCount
    };
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable)));

    // Create the binding table for execution
    tableDesc =
    {
        m_dmlGraph.Get(),
        m_dmlDescriptorHeap->GetCpuHandle(initBindingProps.RequiredDescriptorCount),
        m_dmlDescriptorHeap->GetGpuHandle(initBindingProps.RequiredDescriptorCount),
        executeBindingProps.RequiredDescriptorCount
    };
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&m_dmlBindingTable)));

    DML_BUFFER_BINDING inputBufferBinding{ m_modelInput.Get(), 0, m_modelInput->GetDesc().Width };
    dml::Span<const DML_BUFFER_BINDING> weightBufferBindings = nnModel->GetBindings();

    {
        std::vector<DML_BUFFER_BINDING> initBufferBindings;
        initBufferBindings.push_back(DML_BUFFER_BINDING{}); // Model input
        initBufferBindings.insert(initBufferBindings.end(), weightBufferBindings.begin(), weightBufferBindings.end()); // Weights

        DML_BUFFER_ARRAY_BINDING initInputBinding = { (UINT)initBufferBindings.size(), initBufferBindings.data() };
        DML_BINDING_DESC initInputArrayBindingDesc = { DML_BINDING_TYPE_BUFFER_ARRAY, &initInputBinding };
        initBindingTable->BindInputs(1, &initInputArrayBindingDesc); // when this called, it's like CopyDescriptors (CPU-operation) that copy the binding desc into CPU descriptor heap
    }

    if (initTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { initTemporaryResource.Get(), 0, initTemporaryResource->GetDesc().Width };
        DML_BINDING_DESC initTempResourceBindingDesc = { DML_BINDING_TYPE_BUFFER, &binding };
        initBindingTable->BindTemporaryResource(&initTempResourceBindingDesc);
    }

    // If the operator requires a persistent resource, it must be bound as output for the initializer.
    if (m_modelPersistentResource)
    {
        DML_BUFFER_BINDING binding = { m_modelPersistentResource.Get(), 0, m_modelPersistentResource->GetDesc().Width };
        DML_BINDING_DESC outputBindingDesc = { DML_BINDING_TYPE_BUFFER, &binding };
        initBindingTable->BindOutputs(1, &outputBindingDesc);

        DML_BINDING_DESC persistenceResourceBindingDesc = { DML_BINDING_TYPE_BUFFER, &binding };
        m_dmlBindingTable->BindPersistentResource(&persistenceResourceBindingDesc);
    }

    if (m_modelTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { m_modelTemporaryResource.Get(), 0, m_modelTemporaryResource->GetDesc().Width };
        DML_BINDING_DESC modelTempResourceBindingDesc = { DML_BINDING_TYPE_BUFFER, &binding };
        m_dmlBindingTable->BindTemporaryResource(&modelTempResourceBindingDesc);
    }

    // Bind model inputs and outputs
    std::vector<DML_BINDING_DESC> inputBindings(1 + weightBufferBindings.size());

    // Bind only the model input
    inputBindings[0] = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    m_dmlBindingTable->BindInputs((UINT)inputBindings.size(), inputBindings.data());

    DML_BUFFER_BINDING outputBufferBindings[] =
    {
        { m_modelOutput.output.Get(), 0, m_modelOutput.output->GetDesc().Width },
    };

    DML_BINDING_DESC outputBindings[] =
    {
        { DML_BINDING_TYPE_BUFFER, &outputBufferBindings[0] },
    };

    m_dmlBindingTable->BindOutputs(ARRAYSIZE(outputBindings), outputBindings);

    // Record the initialization
    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializer.Get(), initBindingTable.Get());

    DX::ThrowIfFailed(commandList->Close());
    deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    // Wait until initialization has been finished on the GPU.
    deviceResources->WaitForGpu();
}

void ModelManager::RecordModelDispatch(ID3D12GraphicsCommandList *commandList) {
    // TODO we need to ensure that all the resources should be in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"DML ops");

    ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
    commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);
    // Note that RecordDispatch reset:
    // 1. Compute root signature
    // 2. Pipeline state
    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlGraph.Get(), m_dmlBindingTable.Get());
    PIXEndEvent(commandList);
}

template <typename T>
T RoundUpToMultiple(T value, T multiple)
{
    static_assert(std::is_integral_v<T>);

    T remainder = value % multiple;
    if (remainder != 0)
    {
        value += multiple - remainder;
    }

    return value;
}

dml::Expression FloodDiffusionNNModel::ConstructModel(std::shared_ptr<DX::DeviceResources> deviceResources, dml::Graph &graph, dml::Expression inputTensor) {
    const uint32_t joinAxis = 1; // Concatenate along channels

    size_t modelInputCount = 1; // 0 is input
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
    flags |= DML_TENSOR_FLAG_OWNED_BY_DML; // DML_MANAGED_WEIGHTS
    dml::TensorDesc::Dimensions filterShape = { 32, 3, 3, 3 };
    dml::TensorDesc filterDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags, filterShape);
    auto filter = dml::InputTensor(graph, modelInputCount, filterDesc);

    uint32_t filterCount = filter.GetOutputDesc().sizes[0]; // N dimension is the filter count
    const size_t requiredAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
    size_t offsetInBytes = 0;
    uint32_t filterSize =
            filter.GetOutputDesc().sizes[1] *
            filter.GetOutputDesc().sizes[2] *
            filter.GetOutputDesc().sizes[3]; // Size of each individual filter
        size_t filterSizeInBytes = RoundUpToMultiple((filterCount * filterSize) * sizeof(float), requiredAlignment);
    m_bindings.push_back(DML_BUFFER_BINDING{ nullptr, offsetInBytes, filterSizeInBytes });
    offsetInBytes += filterSizeInBytes;

    ++modelInputCount;

    uint32_t filterHeight = filter.GetOutputDesc().sizes[2];
    uint32_t filterWidth = filter.GetOutputDesc().sizes[3];
    std::array<uint32_t, 2> padding = { filterHeight / 2, filterWidth / 2 };

    std::array<uint32_t, 2> strides = {};
    strides = { 1, 1 };

    dml::FusedActivation fusedActivation = dml::FusedActivation::None();
    fusedActivation = dml::FusedActivation::LeakyRelu(0.1f);

    dml::TensorDesc::Dimensions biasShape = { 1, filterShape[0], 1, 1 };
    dml::TensorDesc biasDesc(DML_TENSOR_DATA_TYPE_FLOAT32, flags, biasShape);
    auto bias = dml::InputTensor(graph, modelInputCount, biasDesc);
    size_t biasSizeInBytes = RoundUpToMultiple((filterCount) * sizeof(float), requiredAlignment);
    m_bindings.push_back(DML_BUFFER_BINDING{ nullptr, offsetInBytes, biasSizeInBytes });
    offsetInBytes += biasSizeInBytes;
    ++modelInputCount;

    // Since our goal is to test run model, we don't care about weight uploading

    uint64_t resourceSizeInBytes = offsetInBytes;
    DX::ThrowIfFailed(deviceResources->GetD3DDevice()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(resourceSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&m_weightBuffer)));

    for (auto &binding: m_bindings) {
        binding.Buffer = m_weightBuffer.Get();
    }

    auto conv = dml::ConvolutionBuilder(inputTensor, filter, bias)
        .StartPadding(padding)
        .EndPadding(padding)
        .Strides(strides)
        .FusedActivation(fusedActivation)
        .Build();
    return conv;
}
}