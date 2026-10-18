#pragma once

#include "DeviceResources.hpp"
#include "Ui.hpp"
#include <DirectMLX.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Neural models evaluated on the same D3D12 device as the renderer -- so their
// buffers can be shared rather than copied across an API boundary -- but on
// this manager's OWN compute queue rather than the renderer's graphics queue.
//
// Sharing the graphics queue bought ordering for free: anything submitted
// after a model's work was guaranteed to see its output, with no fence
// anywhere. It cost exactly what that guarantee is worth, which is that model
// work and frame work could never overlap. A dispatch sat between two frames
// instead of beside one.
//
// An independent queue gives the overlap back and takes the free ordering
// away. A consumer of a model's output now has to say so, which is what
// OrderAfterModelWork is for. Nothing in this project reads a model's output
// on the GPU yet; the day something does, that call is the whole difference
// between a correct frame and a race that only shows up under load.
//
// TODO future plan may need to integrate source from Ollama or HuggingFace
namespace NeuralModelIntegrateTestbed {

// How a registered model is executed. Both run on this project's own
// ID3D12Device and IDMLDevice and submit to the manager's compute queue; they
// differ in who owns the command list and in operator coverage.
enum class ModelBackend {
    // The graph is translated into DirectMLX expressions here (Engine/
    // OnnxModel.cc for an .onnx file, or a hand-written NNModel) and compiled
    // to an IDMLCompiledOperator. Can be recorded straight into the caller's
    // command list, which is what RecordModelDispatch needs. Limited to the
    // operators OnnxModel::SupportedOps() lists.
    DirectMLGraph,

    // ONNX Runtime's DirectML execution provider, given our IDMLDevice and the
    // manager's compute queue. Full operator coverage without writing
    // translations, and Run() records and returns rather than waiting --
    // measured at 0.08 ms against 1.36 ms of outstanding GPU work in
    // //Engine:ortdml-smoke. It records onto its OWN command lists and submits
    // them to that queue, so it cannot be recorded into a list we hand it.
    //
    // The DirectML EP accepts a DIRECT or a COMPUTE queue, which is what makes
    // this backend movable at all.
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

    // Submits this model's work on the manager's compute queue. Ordered against
    // other model work on that queue without a fence, NOT against the
    // renderer's; does not wait for completion.
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
    // Drains the compute queue before anything it owns is released. Required
    // rather than tidy: the manager owns the queue now, so nothing else is
    // going to wait for it, and releasing a resource a submitted command list
    // still references is a use-after-free the runtime reports as a device
    // removal several frames later.
    ~ModelManager();

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

    // Submits `name`'s work on the manager's compute queue. Works for either
    // backend, and is the API a renderer should use: it is the only one that
    // puts the work somewhere it can overlap the frame.
    //
    // Nothing orders this against the renderer. A pass that reads the output
    // must be preceded by OrderAfterModelWork on the graphics queue.
    bool EnqueueModel(const std::string& name, std::string* error = nullptr);

    // Makes `queue` wait until every model submission made so far has
    // finished. This is the fence that replaces what sharing the graphics
    // queue used to provide implicitly, and the call a pass reading a model's
    // output cannot do without.
    //
    // A GPU-side wait: it returns immediately and costs the calling queue
    // nothing until it reaches the wait.
    bool OrderAfterModelWork(ID3D12CommandQueue* queue, std::string* error = nullptr);

    // Blocks the CPU until every model submission made so far has finished.
    // For teardown and for tests; a frame should use OrderAfterModelWork.
    bool WaitForModelWork(std::string* error = nullptr);

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

    void ShowImgui(UiSection section);

    // The DirectML device this manager created on the renderer's D3D12 device,
    // or null if Initialize has not run or DirectML was unavailable. Exposed so
    // another ONNX Runtime consumer -- FloodDiffusionPipeline -- can attach its
    // DirectML execution provider to the SAME device rather than creating a
    // second one on the same adapter.
    IDMLDevice* DmlDevice() const { return m_dmlDevice.Get(); }

    // The compute queue every model submits to. Exposed so another consumer of
    // the same IDMLDevice can be put on it too rather than back on the
    // renderer's queue, and so a test can assert what type it is.
    ID3D12CommandQueue* ComputeQueue() const { return m_computeQueue.Get(); }

private:
    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder;

    // The queue, and the one fence every cross-queue wait is expressed
    // through. One fence for the whole manager rather than one per model: a
    // wait is always "after everything submitted so far", which is what a
    // consumer of any model's output actually needs, and a monotonic counter
    // on one queue gives that for free.
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_computeQueue;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_computeFence;
    std::uint64_t m_computeFenceValue = 0;

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
