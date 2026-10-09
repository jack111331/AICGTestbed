#pragma once

#include "DeviceResources.hpp"
#include <DirectMLX.h>


// This should be integrated into Render graph, as some resource may be require during rendering
// TODO future plan may need to integrate source from Ollama or HuggingFace
namespace NeuralModelIntegrateTestbed {
class ModelManager {
public:
void Initialize(std::shared_ptr<DX::DeviceResources> deviceResources);
void AddNewModel(std::shared_ptr<DX::DeviceResources> deviceResources);
void RecordModelDispatch(ID3D12GraphicsCommandList *commandList);
private:
    struct ModelOutput
    {
        // DEFAULT buffer containing the output contents
        Microsoft::WRL::ComPtr<ID3D12Resource>      output;

        // READBACK buffer for retrieving the output contents from the GPU
        Microsoft::WRL::ComPtr<ID3D12Resource>      readback;

        // Size, format, etc. of the output data
        dml::TensorDesc                             desc;
    };

    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder;
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlGraph;
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_dmlOpInitializer;
    std::unique_ptr<DirectX::DescriptorHeap> m_dmlDescriptorHeap;
    Microsoft::WRL::ComPtr<IDMLBindingTable> m_dmlBindingTable;

    Microsoft::WRL::ComPtr<ID3D12Resource> m_modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_modelTemporaryResource;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_modelPersistentResource;

    ModelOutput                                     m_modelOutput;

};

// NOTE: We implement MultiHeadAttention Expression ourselves

class NNModel {
public:
    virtual dml::Expression ConstructModel(std::shared_ptr<DX::DeviceResources> deviceResources, dml::Graph &graph, dml::Expression inputTensor) = 0;
    dml::Span<const DML_BUFFER_BINDING> GetBindings() const
    {
        return m_bindings;
    }

protected:
    std::vector<DML_BUFFER_BINDING> m_bindings;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_weightBuffer;
};

class FloodDiffusionNNModel: public NNModel {
public:
    dml::Expression ConstructModel(std::shared_ptr<DX::DeviceResources> deviceResources, dml::Graph &graph, dml::Expression inputTensor) override;
};

}
