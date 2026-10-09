#pragma once

#include "ModelManager.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace NeuralModelIntegrateTestbed {

// An ONNX model run by ONNX Runtime's DirectML execution provider, on this
// project's own device, DirectML device and command queue.
//
// The three things that make this usable from a renderer, all verified in
// //Engine:ortdml-smoke:
//
//   * OrtDmlApi::SessionOptionsAppendExecutionProvider_DML1 takes OUR
//     IDMLDevice and ID3D12CommandQueue, so ORT's work lands on the same queue
//     as the renderer's and is ordered against it without a fence.
//   * OrtDmlApi::CreateGPUAllocationFromD3DResource binds OUR ID3D12Resource
//     into an Ort::Value, so input and output stay in GPU memory we own and
//     nothing is copied through the CPU.
//   * Session::Run records and returns. Measured at 0.080 ms while the GPU
//     still had 1.363 ms of queued work outstanding.
//
// What it cannot do is record into a command list handed to it: ORT builds its
// own. Hence RecordInto returns false and callers use Enqueue.
//
// ONNX Runtime's headers are large, so the session hides behind a pimpl; this
// header has to stay cheap because ModelManager.hpp reaches Application.hpp.
class OrtModelRunner : public IModelRunner {
public:
    // Creates a session for `path`. Returns nullptr and fills `error` on a
    // missing file, a model ORT rejects, or a shape this wrapper cannot drive
    // (more than one input or output, or a non-FLOAT32 tensor).
    static std::unique_ptr<OrtModelRunner> Create(const std::string& name,
                                                  const std::filesystem::path& path,
                                                  ID3D12Device* device,
                                                  ID3D12CommandQueue* queue,
                                                  IDMLDevice* dmlDevice,
                                                  std::string* error);

    ~OrtModelRunner() override;

    ModelBackend Backend() const override { return ModelBackend::OnnxRuntime; }
    const ModelInfo& Info() const override;

    // Always false: ONNX Runtime submits its own command lists.
    bool RecordInto(ID3D12GraphicsCommandList* commandList) override;

    bool Enqueue(std::string* error) override;
    bool RunInference(const void* input,
                      std::size_t inputBytes,
                      std::vector<float>& output,
                      std::string* error) override;

    ID3D12Resource* InputResource() override;
    ID3D12Resource* OutputResource() override;

private:
    OrtModelRunner();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace NeuralModelIntegrateTestbed
