#pragma once

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <memory>
#include <vector>
#include "DeviceResources.hpp"
#include "Camera.hpp"

static constexpr std::size_t kBackBufferSize = 2;
static constexpr std::size_t kWorkerThreadSize = 4;

namespace NeuralModelIntegrateTestbed {
    struct ViewFrustum {

    };

    struct PBREffectConstants
    {    
        DirectX::XMVECTOR eyePosition;
        DirectX::XMMATRIX world;
        DirectX::XMVECTOR worldInverseTranspose[3];
        DirectX::XMMATRIX worldViewProj;
        DirectX::XMMATRIX prevWorldViewProj; // for velocity generation

        DirectX::XMVECTOR lightDirection[4];           
        DirectX::XMVECTOR lightDiffuseColor[4];
        
        // PBR Parameters
        DirectX::XMVECTOR Albedo;
        float    Metallic;
        float    Roughness;
        int      numRadianceMipLevels;

        // Size of render target 
        float   targetWidth;
        float   targetHeight;
    };

    static_assert( ( sizeof(PBREffectConstants) % 16 ) == 0, "CB size not padded correctly" );

    // The node describe what things in a scene will input into AI model or what things need AI model to evaluate
    class GLTFAdapter {
    public:
        fastgltf::Expected<fastgltf::Asset> Initialize(const std::string &gltfFilepath);
        void PrepareBuffer(std::shared_ptr<DX::DeviceResources> deviceResources);
        void PrepareImage(std::shared_ptr<DX::DeviceResources> deviceResources);
        void PreparePSO(std::shared_ptr<DX::DeviceResources> deviceResources);
        void AssignCamera(std::shared_ptr<Camera> camera) { m_camera = camera; }
        void Render(std::shared_ptr<DX::DeviceResources> deviceResources, const DirectX::SimpleMath::Matrix &worldMatrix, const DirectX::SimpleMath::Matrix &viewMatrix, const DirectX::SimpleMath::Matrix &projectionMatrix);
        void ShowImgui();
        void SetDescriptorHeap(std::shared_ptr<DirectX::DescriptorHeap> descriptorHeap, std::shared_ptr<DirectX::CommonStates> commonStates) {
            m_descriptorHeap = descriptorHeap;
            m_commonStates = commonStates;
        }
        void QueryViewFrustum(const ViewFrustum &viewFrustum, std::vector<fastgltf::Node> &visibleNodes);
    private:
        fastgltf::Asset m_gltf;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pso;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> images;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocs[kBackBufferSize][kWorkerThreadSize];
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
        Microsoft::WRL::ComPtr<ID3D12Resource> cbuffer;
        ID3D12RootSignature*                       pRootSignature;
        std::size_t m_currentSceneIdx;
        std::size_t descriptorHeapStartIdx = 3; // Since Application.hpp's descriptor heap already occupied the first 0-2, we use 3
        std::shared_ptr<Camera> m_camera;
        std::shared_ptr<DirectX::DescriptorHeap> m_descriptorHeap;
        std::shared_ptr<DirectX::CommonStates> m_commonStates;
        float m_position[3] = {3.0f, -2.0f, -4.0f};
    };
}