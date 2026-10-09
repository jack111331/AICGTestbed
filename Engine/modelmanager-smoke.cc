// Smoke test for ModelManager's named-model registry and the ONNX -> DirectML
// translator.
//
// The interesting assertions are numeric. Weight upload is the part that was
// missing before this refactor -- the old code created a weight buffer and
// never wrote it -- and a model whose weights are zero still compiles, still
// dispatches, and still produces a plausible-looking buffer of zeros. So each
// ONNX model here is built so the correct output is arithmetic: if the weights
// did not arrive, the numbers are wrong rather than the run failing.
//
// ONNX files are constructed in memory through the same protobuf schema the
// loader reads, so the test needs no checked-in .onnx asset.
#include "pch.h"

#include "ModelManager.hpp"
#include "OnnxModel.hpp"

#include <onnx/onnx_pb.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using NeuralModelIntegrateTestbed::FloodDiffusionNNModel;
using NeuralModelIntegrateTestbed::ModelBackend;
using NeuralModelIntegrateTestbed::ModelInfo;
using NeuralModelIntegrateTestbed::ModelManager;
using NeuralModelIntegrateTestbed::OnnxModel;

namespace {

int g_failures = 0;

// The Mul+Relu model, kept on disk so both backends can load the same file.
std::filesystem::path g_mulReluPath;

void Expect(const char* what, bool condition) {
    std::printf("  %-46s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

void ExpectFloat(const char* what, float got, float expected) {
    constexpr float kEps = 1e-4f;
    const bool ok = std::fabs(got - expected) < kEps;
    std::printf("  %-46s %8.3f  expected %8.3f  %s\n", what, got, expected,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

// --- ONNX construction helpers ---------------------------------------------

void SetTensorType(onnx::ValueInfoProto* value,
                   onnx::TensorProto_DataType type,
                   const std::vector<int64_t>& dims) {
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    auto* shape = tensor->mutable_shape();
    for (const int64_t dim : dims) {
        shape->add_dim()->set_dim_value(dim);
    }
}

onnx::TensorProto* AddFloatInitializer(onnx::GraphProto* graph,
                                       const std::string& name,
                                       const std::vector<int64_t>& dims,
                                       const std::vector<float>& values) {
    onnx::TensorProto* tensor = graph->add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(onnx::TensorProto_DataType_FLOAT);
    for (const int64_t dim : dims) {
        tensor->add_dims(dim);
    }
    // raw_data is what real exporters emit, so it is the path worth covering.
    tensor->set_raw_data(values.data(), values.size() * sizeof(float));
    return tensor;
}

onnx::ModelProto NewModel(const char* graphName) {
    onnx::ModelProto model;
    model.set_ir_version(onnx::IR_VERSION);
    model.set_producer_name("ptflio modelmanager-smoke");
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);
    model.mutable_graph()->set_name(graphName);
    return model;
}

// Writes the model beside the test's other temporaries and returns the path.
std::filesystem::path WriteModel(const onnx::ModelProto& model, const char* fileName) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / fileName;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    model.SerializeToOstream(&out);
    out.close();
    return path;
}

// y = Relu(x * scale), with scale an initializer. Output is predictable, and
// zero weights would give all zeros.
std::filesystem::path MakeMulReluModel() {
    onnx::ModelProto model = NewModel("MulRelu");
    onnx::GraphProto* graph = model.mutable_graph();

    SetTensorType(graph->add_input(), onnx::TensorProto_DataType_FLOAT, {1, 1, 1, 4});
    graph->mutable_input(0)->set_name("x");
    SetTensorType(graph->add_output(), onnx::TensorProto_DataType_FLOAT, {1, 1, 1, 4});
    graph->mutable_output(0)->set_name("y");

    AddFloatInitializer(graph, "scale", {1, 1, 1, 4}, {2.0f, 3.0f, 4.0f, 5.0f});

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

    return WriteModel(model, "ptflio_mulrelu.onnx");
}

// A 1x1 convolution with one channel in and out: y = 3x + 1. Exercises the
// Conv attribute handling and a bias, both as OWNED_BY_DML weights.
std::filesystem::path MakeConvModel() {
    onnx::ModelProto model = NewModel("Conv1x1");
    onnx::GraphProto* graph = model.mutable_graph();

    SetTensorType(graph->add_input(), onnx::TensorProto_DataType_FLOAT, {1, 1, 2, 2});
    graph->mutable_input(0)->set_name("x");
    SetTensorType(graph->add_output(), onnx::TensorProto_DataType_FLOAT, {1, 1, 2, 2});
    graph->mutable_output(0)->set_name("y");

    AddFloatInitializer(graph, "W", {1, 1, 1, 1}, {3.0f});
    AddFloatInitializer(graph, "B", {1}, {1.0f});

    onnx::NodeProto* conv = graph->add_node();
    conv->set_name("conv");
    conv->set_op_type("Conv");
    conv->add_input("x");
    conv->add_input("W");
    conv->add_input("B");
    conv->add_output("y");

    auto* kernel = conv->add_attribute();
    kernel->set_name("kernel_shape");
    kernel->set_type(onnx::AttributeProto_AttributeType_INTS);
    kernel->add_ints(1);
    kernel->add_ints(1);

    auto* strides = conv->add_attribute();
    strides->set_name("strides");
    strides->set_type(onnx::AttributeProto_AttributeType_INTS);
    strides->add_ints(1);
    strides->add_ints(1);

    auto* pads = conv->add_attribute();
    pads->set_name("pads");
    pads->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (int i = 0; i < 4; ++i) {
        pads->add_ints(0);
    }

    return WriteModel(model, "ptflio_conv.onnx");
}

// Uses an operator the translator does not handle, to check the failure names
// it rather than silently producing a graph that is missing a step.
std::filesystem::path MakeUnsupportedModel() {
    onnx::ModelProto model = NewModel("Unsupported");
    onnx::GraphProto* graph = model.mutable_graph();

    SetTensorType(graph->add_input(), onnx::TensorProto_DataType_FLOAT, {1, 1, 1, 4});
    graph->mutable_input(0)->set_name("x");
    SetTensorType(graph->add_output(), onnx::TensorProto_DataType_FLOAT, {1, 1, 1, 4});
    graph->mutable_output(0)->set_name("y");

    onnx::NodeProto* node = graph->add_node();
    node->set_name("softmax");
    node->set_op_type("Softmax");
    node->add_input("x");
    node->add_output("y");

    return WriteModel(model, "ptflio_unsupported.onnx");
}

}  // namespace

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto deviceResources = std::make_shared<DX::DeviceResources>();
    try {
        // No window: the swapchain comes from CreateWindowSizeDependentResources,
        // which this test never needs.
        deviceResources->CreateDeviceResources();
    } catch (const std::exception& e) {
        std::printf("no D3D12 device (%s); skipping\n", e.what());
        return 0;
    }

    ModelManager manager;
    try {
        manager.Initialize(deviceResources);
    } catch (const std::exception& e) {
        std::printf("DirectML unavailable (%s); skipping\n", e.what());
        return 0;
    }
    std::printf("DirectML device created\n");

    // ---------------------------------------------------------------------
    // The registry itself.
    // ---------------------------------------------------------------------
    std::cout << "named registry" << std::endl;
    std::string error;
    Expect("hand-written model registers",
           manager.AddModel(std::make_unique<FloodDiffusionNNModel>("conv"), &error));
    if (!error.empty()) std::printf("    error: %s\n", error.c_str());

    Expect("it is findable by name", manager.Has("conv"));
    Expect("an unregistered name is not", !manager.Has("nope"));
    Expect("one model listed", manager.ModelNames().size() == 1);

    error.clear();
    Expect("a duplicate name is rejected",
           !manager.AddModel(std::make_unique<FloodDiffusionNNModel>("conv"), &error));
    Expect("and says why", error.find("already registered") != std::string::npos);

    const ModelInfo* info = manager.Info("conv");
    Expect("Info returns the registered model", info != nullptr);
    if (info != nullptr) {
        // 1x3x256x256 floats in, 32 output channels out.
        Expect("input shape is 1x3x256x256",
               info->inputSizes.size() == 4 && info->inputSizes[0] == 1 &&
                   info->inputSizes[1] == 3 && info->inputSizes[2] == 256 &&
                   info->inputSizes[3] == 256);
        Expect("output has 32 channels",
               info->outputSizes.size() == 4 && info->outputSizes[1] == 32);
        Expect("input bytes match the shape",
               info->inputBytes == 1ull * 3 * 256 * 256 * sizeof(float));
        Expect("two weight tensors declared", info->weightCount == 2);
        Expect("weights occupy a buffer", info->weightBytes > 0);
    }
    Expect("Info on an unknown name is null", manager.Info("nope") == nullptr);

    // ---------------------------------------------------------------------
    // Dispatch by name.
    // ---------------------------------------------------------------------
    std::cout << "dispatch by name" << std::endl;
    {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        auto device = deviceResources->GetD3DDevice();
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&allocator));
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                  nullptr, IID_PPV_ARGS(&list));

        Expect("a known name records", manager.RecordModelDispatch(list.Get(), "conv"));
        // The point of the name-keyed API: a typo must be inert, not fatal.
        Expect("an unknown name records nothing",
               !manager.RecordModelDispatch(list.Get(), "misspelled"));
        list->Close();
    }

    // ---------------------------------------------------------------------
    // ONNX: y = Relu(x * scale). Checks the weights actually arrived.
    // ---------------------------------------------------------------------
    std::cout << "ONNX Mul + Relu" << std::endl;
    {
        const auto path = MakeMulReluModel();
        error.clear();
        const bool added = manager.AddOnnxModel("mulrelu", path,
                                                ModelBackend::DirectMLGraph, &error);
        Expect("ONNX model loads and registers", added);
        if (!added) {
            std::printf("    error: %s\n", error.c_str());
        } else {
            const ModelInfo* onnxInfo = manager.Info("mulrelu");
            Expect("one initializer became one weight",
                   onnxInfo != nullptr && onnxInfo->weightCount == 1);

            // scale = {2,3,4,5}; Relu clamps the negatives.
            const std::vector<float> input = {1.0f, -1.0f, 2.0f, -2.0f};
            std::vector<float> output;
            error.clear();
            const bool ran = manager.RunInference("mulrelu", input.data(),
                                                  input.size() * sizeof(float), output,
                                                  &error);
            Expect("inference runs", ran);
            if (!ran) {
                std::printf("    error: %s\n", error.c_str());
            } else {
                Expect("four outputs", output.size() == 4);
                if (output.size() == 4) {
                    // All four would be 0 if the weights had not been uploaded,
                    // which is exactly the bug this refactor fixes.
                    ExpectFloat("1 * 2 = 2", output[0], 2.0f);
                    ExpectFloat("-1 * 3 -> Relu -> 0", output[1], 0.0f);
                    ExpectFloat("2 * 4 = 8", output[2], 8.0f);
                    ExpectFloat("-2 * 5 -> Relu -> 0", output[3], 0.0f);
                }
            }

            // Wrong-sized input must be refused rather than reading past it.
            error.clear();
            std::vector<float> ignored;
            Expect("a short input is rejected",
                   !manager.RunInference("mulrelu", input.data(), sizeof(float), ignored,
                                         &error));
            Expect("and says the expected size",
                   error.find("expects") != std::string::npos);
        }
        g_mulReluPath = path;  // kept for the cross-backend comparison below
    }

    // ---------------------------------------------------------------------
    // ONNX: a 1x1 convolution, y = 3x + 1. Covers Conv attributes and bias.
    // ---------------------------------------------------------------------
    std::cout << "ONNX Conv 1x1 with bias" << std::endl;
    {
        const auto path = MakeConvModel();
        error.clear();
        const bool added = manager.AddOnnxModel("conv1x1", path,
                                                ModelBackend::DirectMLGraph, &error);
        Expect("Conv model loads", added);
        if (!added) {
            std::printf("    error: %s\n", error.c_str());
        } else {
            Expect("filter and bias became two weights",
                   manager.Info("conv1x1") != nullptr &&
                       manager.Info("conv1x1")->weightCount == 2);

            const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> output;
            error.clear();
            const bool ran = manager.RunInference("conv1x1", input.data(),
                                                  input.size() * sizeof(float), output,
                                                  &error);
            Expect("inference runs", ran);
            if (!ran) {
                std::printf("    error: %s\n", error.c_str());
            } else if (output.size() == 4) {
                ExpectFloat("3*1 + 1", output[0], 4.0f);
                ExpectFloat("3*2 + 1", output[1], 7.0f);
                ExpectFloat("3*3 + 1", output[2], 10.0f);
                ExpectFloat("3*4 + 1", output[3], 13.0f);
            }
        }
        std::filesystem::remove(path);
    }

    // ---------------------------------------------------------------------
    // An untranslated operator must fail by name, and leave nothing behind.
    // ---------------------------------------------------------------------
    std::cout << "unsupported operator" << std::endl;
    {
        const auto path = MakeUnsupportedModel();
        error.clear();
        Expect("an unsupported op fails to register",
               !manager.AddOnnxModel("softmax", path, ModelBackend::DirectMLGraph,
                                     &error));
        Expect("the error names the operator",
               error.find("Softmax") != std::string::npos);
        Expect("the error lists what is supported",
               error.find("Supported") != std::string::npos);
        // A failed build must not leave a half-registered model findable.
        Expect("nothing was registered", !manager.Has("softmax"));
        std::filesystem::remove(path);
    }

    std::cout << "missing file" << std::endl;
    error.clear();
    Expect("a missing ONNX file fails cleanly",
           !manager.AddOnnxModel("absent", "does/not/exist.onnx",
                                 ModelBackend::DirectMLGraph, &error));
    Expect("and says it cannot open it", error.find("cannot open") != std::string::npos);

    // ---------------------------------------------------------------------
    // The same ONNX file through the OnnxRuntime backend. Both backends must
    // agree on the numbers -- that is the strongest check available here,
    // because it cross-validates the hand-written translator against a real
    // ONNX implementation rather than against my own expectations.
    // ---------------------------------------------------------------------
    std::cout << "ONNX Runtime backend" << std::endl;
    {
        error.clear();
        const bool added = manager.AddOnnxModel("mulrelu.ort", g_mulReluPath,
                                                ModelBackend::OnnxRuntime, &error);
        Expect("the same file loads through ONNX Runtime", added);
        if (!added) {
            std::printf("    error: %s\n", error.c_str());
        } else {
            const ModelInfo* ortInfo = manager.Info("mulrelu.ort");
            Expect("it reports the OnnxRuntime backend",
                   ortInfo != nullptr && ortInfo->backend == ModelBackend::OnnxRuntime);
            Expect("the DirectML-graph copy still reports its own backend",
                   manager.Info("mulrelu") != nullptr &&
                       manager.Info("mulrelu")->backend == ModelBackend::DirectMLGraph);
            Expect("input size agrees between backends",
                   ortInfo != nullptr && manager.Info("mulrelu") != nullptr &&
                       ortInfo->inputBytes == manager.Info("mulrelu")->inputBytes);

            // ORT builds its own command lists, so there is nothing to append.
            Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
            auto device = deviceResources->GetD3DDevice();
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           IID_PPV_ARGS(&allocator));
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                      nullptr, IID_PPV_ARGS(&list));
            Expect("RecordModelDispatch refuses an ORT model",
                   !manager.RecordModelDispatch(list.Get(), "mulrelu.ort"));
            Expect("RecordModelDispatch still accepts a DirectML-graph model",
                   manager.RecordModelDispatch(list.Get(), "mulrelu"));
            list->Close();

            // EnqueueModel works for BOTH backends: that is the API a renderer
            // uses, since the work lands on the shared queue either way.
            error.clear();
            Expect("EnqueueModel works for the ORT model",
                   manager.EnqueueModel("mulrelu.ort", &error));
            if (!error.empty()) std::printf("    error: %s\n", error.c_str());
            error.clear();
            Expect("EnqueueModel works for the DirectML-graph model",
                   manager.EnqueueModel("mulrelu", &error));
            if (!error.empty()) std::printf("    error: %s\n", error.c_str());
            Expect("EnqueueModel rejects an unknown name",
                   !manager.EnqueueModel("nope", &error));

            // Both resources are ours, so a renderer could read them directly.
            Expect("the ORT model exposes its D3D12 buffers",
                   manager.InputResource("mulrelu.ort") != nullptr &&
                       manager.OutputResource("mulrelu.ort") != nullptr);

            const std::vector<float> input = {1.0f, -1.0f, 2.0f, -2.0f};
            std::vector<float> ortOutput;
            error.clear();
            const bool ran = manager.RunInference("mulrelu.ort", input.data(),
                                                  input.size() * sizeof(float), ortOutput,
                                                  &error);
            Expect("ORT inference runs", ran);
            if (!ran) {
                std::printf("    error: %s\n", error.c_str());
            } else if (ortOutput.size() == 4) {
                ExpectFloat("ORT: 1 * 2 = 2", ortOutput[0], 2.0f);
                ExpectFloat("ORT: -1 * 3 -> Relu -> 0", ortOutput[1], 0.0f);
                ExpectFloat("ORT: 2 * 4 = 8", ortOutput[2], 8.0f);
                ExpectFloat("ORT: -2 * 5 -> Relu -> 0", ortOutput[3], 0.0f);

                // And the two backends agree element for element.
                std::vector<float> dmlOutput;
                if (manager.RunInference("mulrelu", input.data(),
                                         input.size() * sizeof(float), dmlOutput, &error) &&
                    dmlOutput.size() == ortOutput.size()) {
                    bool same = true;
                    for (std::size_t i = 0; i < dmlOutput.size(); ++i) {
                        if (std::fabs(dmlOutput[i] - ortOutput[i]) > 1e-4f) same = false;
                    }
                    Expect("both backends agree on every element", same);
                } else {
                    Expect("both backends agree on every element", false);
                }
            }

            // Softmax has no DirectMLX translation here but ORT has a kernel --
            // the whole reason for adding this backend.
            const auto softmaxPath = MakeUnsupportedModel();
            error.clear();
            const bool softmaxOk = manager.AddOnnxModel("softmax.ort", softmaxPath,
                                                        ModelBackend::OnnxRuntime, &error);
            Expect("ORT loads a model the translator cannot handle", softmaxOk);
            if (!softmaxOk) std::printf("    error: %s\n", error.c_str());
            std::filesystem::remove(softmaxPath);
        }
        std::filesystem::remove(g_mulReluPath);
    }

    std::printf("registered models: %zu\n", manager.ModelNames().size());
    google::protobuf::ShutdownProtobufLibrary();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all checks passed" << std::endl;
    return 0;
}
