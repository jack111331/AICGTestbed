#pragma once

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <memory>
#include <vector>
#include "DeviceResources.hpp"
#include "Camera.hpp"
#include "Descriptors/DescriptorAllocation.hpp"
#include "Descriptors/DescriptorAllocator.hpp"
#include "Descriptors/DescriptorHeapBinder.hpp"
#include "Descriptors/DynamicDescriptorHeap.hpp"
#include "Descriptors/RootSignature.hpp"

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
        // Only the sampler heap comes from the caller now. SRV descriptors for
        // glTF images are allocated from this class's own DescriptorAllocator
        // instead of borrowing slots out of the application's fixed heap.
        void SetCommonStates(std::shared_ptr<DirectX::CommonStates> commonStates) {
            m_commonStates = commonStates;
        }

        // Recycles descriptors retired by earlier frames. Call once per frame.
        void ReleaseStaleDescriptors(uint64_t frameNumber);
        void QueryViewFrustum(const ViewFrustum &viewFrustum, std::vector<fastgltf::Node> &visibleNodes);
    private:
        fastgltf::Asset m_gltf;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pso;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> images;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocs[kBackBufferSize][kWorkerThreadSize];
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
        Microsoft::WRL::ComPtr<ID3D12Resource> cbuffer;
        std::size_t m_currentSceneIdx;
        std::shared_ptr<Camera> m_camera;
        std::shared_ptr<DirectX::CommonStates> m_commonStates;

        // --- Page-based descriptor management -------------------------------
        // The root signature, wrapped so DynamicDescriptorHeap can read its
        // descriptor-table layout.
        Descriptors::RootSignature m_rootSignature;
        // CPU-visible SRV descriptors handed out in pages: one allocation per
        // glTF image, held as long as the image lives. The allocation's
        // destructor returns it to its page.
        std::unique_ptr<Descriptors::DescriptorAllocator> m_srvAllocator;
        std::vector<Descriptors::DescriptorAllocation> m_imageDescriptors;
        // Copies staged CPU descriptors into a GPU-visible heap at draw time and
        // binds the resulting table.
        std::unique_ptr<Descriptors::DynamicDescriptorHeap> m_srvDynamicHeap;
        // Keeps the dynamic SRV heap and the DirectXTK12 sampler heap bound
        // together across dynamic-heap switches.
        Descriptors::HeapBinder m_heapBinder;
        float m_position[3] = {3.0f, -2.0f, -4.0f};
    };
}