#pragma once

#include "DeviceResources.hpp"
#include <DirectMLX.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Neural models evaluated on the same D3D12 device as the renderer, so their
// work is ordered against rendering by queue order alone and their buffers can
// be shared rather than copied across an API boundary.
//
// TODO future plan may need to integrate source from Ollama or HuggingFace
namespace NeuralModelIntegrateTestbed {

// How a registered model is executed. Both run on this project's own
// ID3D12Device, IDMLDevice and ID3D12CommandQueue; they differ in who owns the
// command list and in operator coverage.
enum class ModelBackend {
    // The graph is translated into DirectMLX expressions here (Engine/
    // OnnxModel.cc for an .onnx file, or a hand-written NNModel) and compiled
    // to an IDMLCompiledOperator. Can be recorded straight into the caller's
    // command list, which is what RecordModelDispatch needs. Limited to the
    // operators OnnxModel::SupportedOps() lists.
    DirectMLGraph,

    // ONNX Runtime's DirectML execution provider, given our IDMLDevice and
    // ID3D12CommandQueue. Full operator coverage without writing translations,
    // and Run() records and returns rather than waiting -- measured at 0.08 ms
    // against 1.36 ms of outstanding GPU work in //Engine:ortdml-smoke. It
    // records onto its OWN command lists and submits them to the shared queue,
    // so it cannot be recorded into a list we hand it.
    OnnxRuntime,
};

const char* ToString(ModelBackend backend);

// What a registered model turned out to need, for the UI and for tests.
struct ModelInfo {
    std::string name;
    ModelBackend backend = ModelBackend::DirectMLGraph;

    std::vector<uint32_t> inputSizes;
    std::vector<uint32_t> outputSizes;
    uint64_t inputBytes = 0;
    uint64_t outputBytes = 0;

    // DirectMLGraph only: weights are uploaded by this project. The ORT backend
    // loads them itself as part of the session, so these stay zero.
    uint64_t weightBytes = 0;
    std::size_t weightCount = 0;
    uint64_t temporaryBytes = 0;
    uint64_t persistentBytes = 0;
};

// One tensor a DirectMLGraph model needs supplied as a constant: a convolution
// filter, a bias, a learned embedding. The model DECLARES these; ModelManager
// owns the GPU buffer and performs the upload, so every model shares one path.
struct ModelWeight {
    // For diagnostics. From ONNX this is the initializer name.
    std::string name;

    // Which dml::InputTensor index this fills. Index 0 is always the model
    // input, so weights start at 1 and must be contiguous -- the initializer
    // binds them as one array in index order, and a gap would silently shift
    // every weight after it onto the wrong tensor.
    uint32_t graphInputIndex = 0;

    dml::TensorDesc desc;

    // Raw element bytes in the layout `desc` describes. Size must equal
    // desc.totalTensorSizeInBytes; AddModel rejects a mismatch rather than
    // uploading a short buffer and reading whatever follows it.
    std::vector<uint8_t> data;
};

// What a DirectMLGraph model hands back from Build.
struct ModelGraph {
    dml::Expression output;
    std::vector<ModelWeight> weights;
};

// A source of a DirectML graph: hand-written in C++, or translated from ONNX.
//
// Build() only DESCRIBES. It creates dml::Expression nodes and declares the
// constants it needs, and must not create D3D12 resources -- ModelManager owns
// those, which is what gives every model one lifetime and one upload path.
class NNModel {
public:
    virtual ~NNModel() = default;

    // The key RecordModelDispatch and RunInference look this model up by.
    virtual std::string Name() const = 0;

    // Shape and type of graph input 0, needed before Build so the manager can
    // size the input buffer.
    virtual dml::TensorDesc InputDesc() const = 0;

    virtual ModelGraph Build(dml::Graph& graph, dml::Expression input) = 0;
};

// What ModelManager needs from a backend. Declared here so the manager can hold
// one without seeing ONNX Runtime's headers, which are large and would
// otherwise reach Application.hpp through this file.
class IModelRunner {
public:
    virtual ~IModelRunner() = default;

    virtual ModelBackend Backend() const = 0;
    virtual const ModelInfo& Info() const = 0;

    // Records into the caller's list. Only DirectMLGraph can; the ORT backend
    // returns false, because ORT builds and submits its own lists.
    virtual bool RecordInto(ID3D12GraphicsCommandList* commandList) = 0;

    // Submits this model's work on the shared queue. Ordered against anything
    // else submitted there without a fence, and does not wait for completion.
    virtual bool Enqueue(std::string* error) = 0;

    // Uploads `input`, submits, waits and reads the result back. Synchronous by
    // design: this is the "does the model compute the right thing" path.
    virtual bool RunInference(const void* input,
                              std::size_t inputBytes,
                              std::vector<float>& output,
                              std::string* error) = 0;

    // The buffers the model reads and writes, for a caller that wants to feed
    // the input from its own passes or sample the output in a shader.
    virtual ID3D12Resource* InputResource() = 0;
    virtual ID3D12Resource* OutputResource() = 0;
};

class ModelManager {
public:
    void Initialize(std::shared_ptr<DX::DeviceResources> deviceResources);

    // --- registration -------------------------------------------------------

    // A hand-written DirectML graph. Registered under model->Name().
    bool AddModel(std::unique_ptr<NNModel> model, std::string* error = nullptr);

    // An .onnx file. `backend` picks who runs it:
    //   OnnxRuntime    full operator coverage, submits its own command lists
    //   DirectMLGraph  translated here, recordable into your command list
    // OnnxRuntime is the default because it does not need an operator to be
    // implemented before a model will load.
    bool AddOnnxModel(const std::string& name,
                      const std::filesystem::path& path,
                      ModelBackend backend = ModelBackend::OnnxRuntime,
                      std::string* error = nullptr);

    // --- execution ----------------------------------------------------------

    // Records `name`'s graph into `commandList`. DirectMLGraph models only;
    // returns false for an unknown name OR an OnnxRuntime model, having
    // recorded nothing.
    //
    // NOTE this sets its own descriptor heaps, and DirectML's RecordDispatch
    // also resets the compute root signature and pipeline state. A caller that
    // had its own heaps or compute state bound must rebind afterwards.
    bool RecordModelDispatch(ID3D12GraphicsCommandList* commandList,
                             const std::string& name);

    // Submits `name`'s work on the shared command queue. Works for either
    // backend. Because it goes on the same queue the renderer uses, the work is
    // ordered against the renderer's submissions with no fence -- but that also
    // means a pass that reads the output must be submitted AFTER this.
    bool EnqueueModel(const std::string& name, std::string* error = nullptr);

    bool RunInference(const std::string& name,
                      const void* input,
                      std::size_t inputBytes,
                      std::vector<float>& output,
                      std::string* error = nullptr);

    // --- inspection ---------------------------------------------------------

    bool Has(const std::string& name) const;
    std::vector<std::string> ModelNames() const;
    const ModelInfo* Info(const std::string& name) const;
    ID3D12Resource* InputResource(const std::string& name);
    ID3D12Resource* OutputResource(const std::string& name);

    void ShowImgui();

private:
    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder;

    // Keyed by name, which is what every execution call takes. std::map rather
    // than unordered_map so ModelNames() and the UI list in a stable order.
    std::map<std::string, std::unique_ptr<IModelRunner>> m_models;
};

// The hand-written convolution that predated ONNX support, kept as a worked
// example of the NNModel interface and as something to test the manager
// against without a file on disk.
class FloodDiffusionNNModel : public NNModel {
public:
    explicit FloodDiffusionNNModel(std::string name = "FloodDiffusion")
        : m_name(std::move(name)) {}

    std::string Name() const override { return m_name; }
    dml::TensorDesc InputDesc() const override;
    ModelGraph Build(dml::Graph& graph, dml::Expression input) override;

private:
    std::string m_name;
};

}  // namespace NeuralModelIntegrateTestbed
