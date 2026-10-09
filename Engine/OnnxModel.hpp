#pragma once

#include "ModelManager.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace NeuralModelIntegrateTestbed {

// An ONNX graph translated into DirectML expressions.
//
// Deliberately not ONNX Runtime: a translated graph compiles to an
// IDMLCompiledOperator, which ModelManager can record into the renderer's own
// command list. ONNX Runtime owns execution behind session.Run() and cannot.
// The price is that only the operators handled in OnnxModel.cc work --
// SupportedOps() lists them, and an unsupported one is reported by name rather
// than silently skipped.
//
// The ONNX protobuf headers are expensive and leak protobuf into everything
// that includes them, so the parsed model hides behind a pimpl; this header
// stays cheap enough for Application.hpp to pull in through ModelManager.hpp.
class OnnxModel : public NNModel {
public:
    // Parses `path`. Returns nullptr and fills `error` when the file cannot be
    // read, is not a valid ModelProto, or has a graph shape this translator
    // cannot describe -- so a bad file never reaches ModelManager::AddModel.
    static std::unique_ptr<OnnxModel> Load(const std::string& name,
                                           const std::filesystem::path& path,
                                           std::string* error);

    OnnxModel();
    ~OnnxModel() override;

    std::string Name() const override { return m_name; }
    dml::TensorDesc InputDesc() const override { return m_inputDesc; }

    // Throws std::runtime_error naming the operator when the graph uses one
    // that is not translated. ModelManager::AddModel catches it, so a caller
    // sees a failed AddOnnxModel with the reason rather than an exception.
    ModelGraph Build(dml::Graph& graph, dml::Expression input) override;

    // Operator types this translator understands, for diagnostics and the UI.
    static std::vector<std::string> SupportedOps();

    // The graph input the model is driven through, and what the file called it.
    const std::string& InputTensorName() const { return m_inputName; }
    const std::string& OutputTensorName() const { return m_outputName; }
    std::size_t NodeCount() const;
    std::size_t InitializerCount() const;

private:
    struct Asset;
    std::unique_ptr<Asset> m_asset;

    std::string m_name;
    std::string m_inputName;
    std::string m_outputName;
    dml::TensorDesc m_inputDesc;
};

}  // namespace NeuralModelIntegrateTestbed
