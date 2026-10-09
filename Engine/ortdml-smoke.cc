// Proves the ONNX Runtime DirectML execution provider can be driven the way
// ModelManager needs it, before any of ModelManager is built on top:
//
//   1. the EP accepts OUR IDMLDevice and OUR ID3D12CommandQueue
//      (SessionOptionsAppendExecutionProvider_DML1),
//   2. OUR ID3D12Resource binds into an Ort::Value with no copy
//      (OrtDmlApi::CreateGPUAllocationFromD3DResource),
//   3. Run() returns without waiting for the GPU -- measured, not assumed,
//   4. the numbers come out right once the queue is drained.
//
// The model is built in memory with the onnx protobuf schema, so nothing is
// checked in. y = Relu(x * {2,3,4,5}), which is wrong in an obvious way if the
// weights or the resource binding do not arrive.
#include "pch.h"

#include <onnx/onnx_pb.h>

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Expect(const char* what, bool condition) {
    std::printf("  %-50s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

void ExpectFloat(const char* what, float got, float expected) {
    const bool ok = std::fabs(got - expected) < 1e-4f;
    std::printf("  %-50s %8.3f  expected %8.3f  %s\n", what, got, expected,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

std::filesystem::path WriteMulReluModel() {
    onnx::ModelProto model;
    model.set_ir_version(onnx::IR_VERSION);
    model.set_producer_name("ptflio ortdml-smoke");
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("MulRelu");

    const auto setType = [](onnx::ValueInfoProto* value, const char* name) {
        value->set_name(name);
        auto* t = value->mutable_type()->mutable_tensor_type();
        t->set_elem_type(onnx::TensorProto_DataType_FLOAT);
        auto* shape = t->mutable_shape();
        shape->add_dim()->set_dim_value(1);
        shape->add_dim()->set_dim_value(4);
    };
    setType(graph->add_input(), "x");
    setType(graph->add_output(), "y");

    onnx::TensorProto* scale = graph->add_initializer();
    scale->set_name("scale");
    scale->set_data_type(onnx::TensorProto_DataType_FLOAT);
    scale->add_dims(1);
    scale->add_dims(4);
    const std::vector<float> values = {2.0f, 3.0f, 4.0f, 5.0f};
    scale->set_raw_data(values.data(), values.size() * sizeof(float));

    onnx::NodeProto* mul = graph->add_node();
    mul->set_name("mul");
    mul->set_op_type("Mul");
    mul->add_input("x");
    mul->add_input("scale");
    mul->add_output("scaled");

    onnx::NodeProto* relu = graph->add_node();
    relu->set_name("relu");
    relu->set_op_type("Relu");
    relu->add_input("scaled");
    relu->add_output("y");

    const auto path = std::filesystem::temp_directory_path() / "ptflio_ort_mulrelu.onnx";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    model.SerializeToOstream(&out);
    return path;
}

// A model heavy enough that its GPU time is measurable. The Mul+Relu above is
// four floats, so it finishes during Run() no matter how Run() behaves, which
// makes it useless for telling "records and returns" from "waits". This is 12
// convolutions over a 64x256x256 tensor -- about 29 G MACs with only ~150 KB of
// weights per layer, so GPU time dominates and CPU setup does not.
std::filesystem::path WriteHeavyModel(int64_t channels, int64_t spatial, int layers) {
    onnx::ModelProto model;
    model.set_ir_version(onnx::IR_VERSION);
    model.set_producer_name("ptflio ortdml-smoke heavy");
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("HeavyConv");

    const auto setType = [&](onnx::ValueInfoProto* value, const char* name) {
        value->set_name(name);
        auto* t = value->mutable_type()->mutable_tensor_type();
        t->set_elem_type(onnx::TensorProto_DataType_FLOAT);
        auto* shape = t->mutable_shape();
        shape->add_dim()->set_dim_value(1);
        shape->add_dim()->set_dim_value(channels);
        shape->add_dim()->set_dim_value(spatial);
        shape->add_dim()->set_dim_value(spatial);
    };
    setType(graph->add_input(), "x");
    setType(graph->add_output(), "y");

    // One small 3x3 filter reused by every layer: the point is arithmetic
    // volume, not parameter count.
    const std::size_t filterElements =
        static_cast<std::size_t>(channels) * channels * 3 * 3;
    std::vector<float> filter(filterElements, 0.0f);
    // A near-identity kernel keeps activations bounded across 12 layers; a
    // random filter would overflow to inf and the output check would be
    // meaningless.
    for (int64_t c = 0; c < channels; ++c) {
        // centre tap of the diagonal channel
        const std::size_t index =
            static_cast<std::size_t>(c) * channels * 9 + static_cast<std::size_t>(c) * 9 + 4;
        filter[index] = 1.0f;
    }

    std::string previous = "x";
    for (int layer = 0; layer < layers; ++layer) {
        const std::string weightName = "W" + std::to_string(layer);
        onnx::TensorProto* w = graph->add_initializer();
        w->set_name(weightName);
        w->set_data_type(onnx::TensorProto_DataType_FLOAT);
        w->add_dims(channels);
        w->add_dims(channels);
        w->add_dims(3);
        w->add_dims(3);
        w->set_raw_data(filter.data(), filter.size() * sizeof(float));

        const bool last = layer == layers - 1;
        const std::string outName = last ? "y" : ("h" + std::to_string(layer));

        onnx::NodeProto* conv = graph->add_node();
        conv->set_name("conv" + std::to_string(layer));
        conv->set_op_type("Conv");
        conv->add_input(previous);
        conv->add_input(weightName);
        conv->add_output(outName);

        auto* kernel = conv->add_attribute();
        kernel->set_name("kernel_shape");
        kernel->set_type(onnx::AttributeProto_AttributeType_INTS);
        kernel->add_ints(3);
        kernel->add_ints(3);

        auto* pads = conv->add_attribute();
        pads->set_name("pads");
        pads->set_type(onnx::AttributeProto_AttributeType_INTS);
        for (int i = 0; i < 4; ++i) {
            pads->add_ints(1);
        }

        previous = outName;
    }

    const auto path = std::filesystem::temp_directory_path() / "ptflio_ort_heavy.onnx";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    model.SerializeToOstream(&out);
    return path;
}

// A DEFAULT-heap buffer, as a renderer would hand over.
Microsoft::WRL::ComPtr<ID3D12Resource> MakeBuffer(ID3D12Device* device, UINT64 bytes,
                                                  D3D12_RESOURCE_STATES state) {
    Microsoft::WRL::ComPtr<ID3D12Resource> out;
    const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const auto desc =
        CD3DX12_RESOURCE_DESC::Buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
                                               nullptr, IID_PPV_ARGS(&out)))) {
        return nullptr;
    }
    return out;
}

}  // namespace

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // --- our own device, queue and DML device -------------------------------
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::printf("no D3D12 device; skipping\n");
        return 0;
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)))) {
        std::printf("no command queue; skipping\n");
        return 0;
    }
    Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice;
    if (FAILED(DMLCreateDevice(device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                               IID_PPV_ARGS(&dmlDevice)))) {
        std::printf("no DirectML device; skipping\n");
        return 0;
    }
    std::printf("D3D12 device, command queue and DirectML device created\n");

    const auto modelPath = WriteMulReluModel();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ptflio");

    // --- the EP must take our device and queue ------------------------------
    std::printf("DirectML execution provider\n");
    const OrtApi& ortApi = Ort::GetApi();
    const OrtDmlApi* dmlApi = nullptr;
    const OrtStatus* status = ortApi.GetExecutionProviderApi(
        "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dmlApi));
    Expect("OrtDmlApi is available", status == nullptr && dmlApi != nullptr);
    if (dmlApi == nullptr) {
        std::printf("cannot continue without OrtDmlApi\n");
        return 1;
    }

    Ort::SessionOptions options;
    // Must be DISABLE_ALL: the DML EP does not support ORT's memory-pattern
    // optimisation, and leaving it on makes session creation fail.
    options.DisableMemPattern();
    options.SetExecutionMode(ORT_SEQUENTIAL);

    const OrtStatus* epStatus = dmlApi->SessionOptionsAppendExecutionProvider_DML1(
        options, dmlDevice.Get(), queue.Get());
    const bool epOk = epStatus == nullptr;
    if (!epOk) {
        std::printf("    EP error: %s\n", ortApi.GetErrorMessage(epStatus));
        ortApi.ReleaseStatus(const_cast<OrtStatus*>(epStatus));
    }
    Expect("EP accepts our IDMLDevice + ID3D12CommandQueue", epOk);
    if (!epOk) {
        return 1;
    }

    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
    } catch (const Ort::Exception& e) {
        std::printf("    session error: %s\n", e.what());
        Expect("session is created", false);
        return 1;
    }
    Expect("session is created", true);
    Expect("one input, one output",
           session->GetInputCount() == 1 && session->GetOutputCount() == 1);

    // --- bind OUR D3D12 resources into Ort::Value ---------------------------
    std::printf("D3D12 resource binding\n");
    constexpr std::size_t kCount = 4;
    constexpr UINT64 kBytes = kCount * sizeof(float);

    auto inputBuffer = MakeBuffer(device.Get(), kBytes, D3D12_RESOURCE_STATE_COMMON);
    auto outputBuffer = MakeBuffer(device.Get(), kBytes, D3D12_RESOURCE_STATE_COMMON);
    Expect("input and output buffers created",
           inputBuffer != nullptr && outputBuffer != nullptr);

    void* inputAllocation = nullptr;
    void* outputAllocation = nullptr;
    Expect("input resource wraps as a DML allocation",
           dmlApi->CreateGPUAllocationFromD3DResource(inputBuffer.Get(), &inputAllocation) ==
               nullptr);
    Expect("output resource wraps as a DML allocation",
           dmlApi->CreateGPUAllocationFromD3DResource(outputBuffer.Get(), &outputAllocation) ==
               nullptr);

    // Upload the input through a staging buffer on our own command list, so the
    // data path is the same one a renderer would use.
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> staging;
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(kBytes);
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                        IID_PPV_ARGS(&staging));
        const float input[kCount] = {1.0f, -1.0f, 2.0f, -2.0f};
        void* mapped = nullptr;
        staging->Map(0, nullptr, &mapped);
        std::memcpy(mapped, input, kBytes);
        staging->Unmap(0, nullptr);

        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&allocator));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                  nullptr, IID_PPV_ARGS(&list));
        list->CopyBufferRegion(inputBuffer.Get(), 0, staging.Get(), 0, kBytes);
        list->Close();
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);

        // Drain, because the staging buffer dies at the end of this scope.
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        queue->Signal(fence.Get(), 1);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(1, event);
        WaitForSingleObject(event, 10000);
        CloseHandle(event);
    }

    const Ort::MemoryInfo memoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const std::vector<int64_t> shape = {1, static_cast<int64_t>(kCount)};

    Ort::Value inputValue = Ort::Value::CreateTensor(
        memoryInfo, inputAllocation, kBytes, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Ort::Value outputValue = Ort::Value::CreateTensor(
        memoryInfo, outputAllocation, kBytes, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Expect("tensors wrap the D3D12 allocations", inputValue.IsTensor() &&
                                                     outputValue.IsTensor());

    // --- Run() must not wait for the GPU ------------------------------------
    std::printf("Run() latency\n");
    const char* inputNames[] = {"x"};
    const char* outputNames[] = {"y"};

    const auto before = std::chrono::steady_clock::now();
    try {
        session->Run(Ort::RunOptions{nullptr}, inputNames, &inputValue, 1, outputNames,
                     &outputValue, 1);
    } catch (const Ort::Exception& e) {
        std::printf("    run error: %s\n", e.what());
        Expect("Run succeeds", false);
        return 1;
    }
    const auto afterRun = std::chrono::steady_clock::now();

    // Now drain the queue ourselves. If Run() had blocked until completion,
    // this wait would be ~free and the two numbers would not differ.
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    queue->Signal(fence.Get(), 1);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(1, event);
    WaitForSingleObject(event, 10000);
    CloseHandle(event);
    const auto afterDrain = std::chrono::steady_clock::now();

    const double runMs =
        std::chrono::duration<double, std::milli>(afterRun - before).count();
    const double drainMs =
        std::chrono::duration<double, std::milli>(afterDrain - afterRun).count();
    std::printf("  Run() returned in %.3f ms; draining the queue took %.3f ms\n", runMs,
                drainMs);
    Expect("Run() returns promptly (under 250 ms)", runMs < 250.0);

    // --- the numbers --------------------------------------------------------
    std::printf("output values\n");
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> readback;
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(kBytes);
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                        IID_PPV_ARGS(&readback));

        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&allocator));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                  nullptr, IID_PPV_ARGS(&list));
        list->CopyBufferRegion(readback.Get(), 0, outputBuffer.Get(), 0, kBytes);
        list->Close();
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);

        Microsoft::WRL::ComPtr<ID3D12Fence> f2;
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2));
        queue->Signal(f2.Get(), 1);
        HANDLE e2 = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        f2->SetEventOnCompletion(1, e2);
        WaitForSingleObject(e2, 10000);
        CloseHandle(e2);

        void* mapped = nullptr;
        readback->Map(0, nullptr, &mapped);
        const float* values = static_cast<const float*>(mapped);
        ExpectFloat("1 * 2 = 2", values[0], 2.0f);
        ExpectFloat("-1 * 3 -> Relu -> 0", values[1], 0.0f);
        ExpectFloat("2 * 4 = 8", values[2], 8.0f);
        ExpectFloat("-2 * 5 -> Relu -> 0", values[3], 0.0f);
        readback->Unmap(0, nullptr);
    }

    // --- is Run() really non-blocking? ------------------------------------
    // The check above cannot answer this: four floats finish before Run()
    // returns whatever Run() does. With a model whose GPU time is tens of
    // milliseconds, a Run() that merely records comes back far sooner than the
    // queue drains -- and a Run() that waited would leave nothing to drain.
    std::printf("Run() against a heavy model\n");
    {
        constexpr int64_t kChannels = 64;
        constexpr int64_t kSpatial = 256;
        constexpr int kLayers = 12;
        const auto heavyPath = WriteHeavyModel(kChannels, kSpatial, kLayers);

        Ort::SessionOptions heavyOptions;
        heavyOptions.DisableMemPattern();
        heavyOptions.SetExecutionMode(ORT_SEQUENTIAL);
        const OrtStatus* heavyEp = dmlApi->SessionOptionsAppendExecutionProvider_DML1(
            heavyOptions, dmlDevice.Get(), queue.Get());
        if (heavyEp != nullptr) {
            std::printf("    EP error: %s\n", ortApi.GetErrorMessage(heavyEp));
            ortApi.ReleaseStatus(const_cast<OrtStatus*>(heavyEp));
            Expect("heavy session EP attaches", false);
        } else {
            std::unique_ptr<Ort::Session> heavy;
            try {
                heavy = std::make_unique<Ort::Session>(env, heavyPath.c_str(), heavyOptions);
            } catch (const Ort::Exception& e) {
                std::printf("    heavy session error: %s\n", e.what());
            }
            Expect("heavy session is created", heavy != nullptr);

            if (heavy != nullptr) {
                const UINT64 elementCount =
                    static_cast<UINT64>(kChannels) * kSpatial * kSpatial;
                const UINT64 heavyBytes = elementCount * sizeof(float);

                auto heavyIn = MakeBuffer(device.Get(), heavyBytes,
                                          D3D12_RESOURCE_STATE_COMMON);
                auto heavyOut = MakeBuffer(device.Get(), heavyBytes,
                                           D3D12_RESOURCE_STATE_COMMON);
                void* heavyInAlloc = nullptr;
                void* heavyOutAlloc = nullptr;
                dmlApi->CreateGPUAllocationFromD3DResource(heavyIn.Get(), &heavyInAlloc);
                dmlApi->CreateGPUAllocationFromD3DResource(heavyOut.Get(), &heavyOutAlloc);

                const std::vector<int64_t> heavyShape = {1, kChannels, kSpatial, kSpatial};
                Ort::Value heavyInValue = Ort::Value::CreateTensor(
                    memoryInfo, heavyInAlloc, heavyBytes, heavyShape.data(),
                    heavyShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
                Ort::Value heavyOutValue = Ort::Value::CreateTensor(
                    memoryInfo, heavyOutAlloc, heavyBytes, heavyShape.data(),
                    heavyShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

                // One warm-up run, so kernel compilation and resource creation
                // are not what gets timed.
                try {
                    heavy->Run(Ort::RunOptions{nullptr}, inputNames, &heavyInValue, 1,
                               outputNames, &heavyOutValue, 1);
                } catch (const Ort::Exception& e) {
                    std::printf("    warm-up error: %s\n", e.what());
                }
                {
                    Microsoft::WRL::ComPtr<ID3D12Fence> warmFence;
                    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&warmFence));
                    queue->Signal(warmFence.Get(), 1);
                    HANDLE warmEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                    warmFence->SetEventOnCompletion(1, warmEvent);
                    WaitForSingleObject(warmEvent, 60000);
                    CloseHandle(warmEvent);
                }

                const auto t0 = std::chrono::steady_clock::now();
                bool ranHeavy = true;
                try {
                    heavy->Run(Ort::RunOptions{nullptr}, inputNames, &heavyInValue, 1,
                               outputNames, &heavyOutValue, 1);
                } catch (const Ort::Exception& e) {
                    std::printf("    heavy run error: %s\n", e.what());
                    ranHeavy = false;
                }
                const auto t1 = std::chrono::steady_clock::now();

                Microsoft::WRL::ComPtr<ID3D12Fence> heavyFence;
                device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&heavyFence));
                queue->Signal(heavyFence.Get(), 1);
                HANDLE heavyEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                heavyFence->SetEventOnCompletion(1, heavyEvent);
                WaitForSingleObject(heavyEvent, 60000);
                CloseHandle(heavyEvent);
                const auto t2 = std::chrono::steady_clock::now();

                Expect("heavy run succeeds", ranHeavy);
                const double heavyRunMs =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                const double heavyDrainMs =
                    std::chrono::duration<double, std::milli>(t2 - t1).count();
                std::printf("  %d convolutions, %lld channels at %lldx%lld\n", kLayers,
                            static_cast<long long>(kChannels),
                            static_cast<long long>(kSpatial),
                            static_cast<long long>(kSpatial));
                std::printf("  Run() returned in %.3f ms; the GPU still needed %.3f ms\n",
                            heavyRunMs, heavyDrainMs);
                // The evidence: there was real GPU work outstanding when Run()
                // came back. A blocking Run() would leave nothing to wait for.
                Expect("GPU work was still outstanding after Run() returned",
                       heavyDrainMs > 1.0);
                Expect("Run() returned faster than the GPU took",
                       heavyRunMs < heavyDrainMs);

                dmlApi->FreeGPUAllocation(heavyInAlloc);
                dmlApi->FreeGPUAllocation(heavyOutAlloc);
            }
        }
        std::filesystem::remove(heavyPath);
    }

    dmlApi->FreeGPUAllocation(inputAllocation);
    dmlApi->FreeGPUAllocation(outputAllocation);
    session.reset();
    std::filesystem::remove(modelPath);
    google::protobuf::ShutdownProtobufLibrary();

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
