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
#include "SceneGraph.hpp"

static constexpr std::size_t kBackBufferSize = 2;
static constexpr std::size_t kWorkerThreadSize = 4;

namespace NeuralModelIntegrateTestbed {
    struct ViewFrustum {

    };

    // Mirrors `cbuffer PBR_Constants : register(b0)` in shaders/no_texture.fx.
    // The HLSL side spells out a packoffset for every field, so the two layouts
    // have to be changed together; the comments below give each field's
    // register so that stays checkable by eye.
    struct PBREffectConstants
    {
        DirectX::XMVECTOR eyePosition;                 // c0
        DirectX::XMMATRIX world;                       // c1..c4
        DirectX::XMVECTOR worldInverseTranspose[3];    // c5..c7  (float3x3)
        DirectX::XMMATRIX worldViewProj;               // c8..c11
        DirectX::XMMATRIX prevWorldViewProj;           // c12..c15, for velocity generation

        // Lights from the glTF asset's KHR_lights_punctual, resolved against the
        // node instancing each one. Replaces the earlier pair of loose
        // lightDirection[]/lightDiffuseColor[] arrays, which could not express
        // point or spot lights -- and which nothing ever filled in.
        //
        // Only the first `lightCount` entries hold a light; the rest are zeroed.
        ShaderLight lights[kMaxShaderLights];          // c16..c31 (4 registers each)

        // PBR Parameters
        DirectX::XMVECTOR Albedo;                      // c32
        float    Metallic;                             // c33.x
        float    Roughness;                            // c33.y
        int      numRadianceMipLevels;                 // c33.z
        int      lightCount;                           // c33.w

        // Size of render target
        float   targetWidth;                           // c34.x
        float   targetHeight;                          // c34.y
        float   padding[2];                            // c34.zw
    };

    static_assert( ( sizeof(PBREffectConstants) % 16 ) == 0, "CB size not padded correctly" );

    // Field offsets, verified against the layout dxc reports for
    // cbuffer PBR_Constants in shaders/no_texture.fx. A mismatch here would
    // otherwise show up only as the shader reading the wrong bytes -- which
    // renders something plausible rather than failing.
    static_assert(offsetof(PBREffectConstants, eyePosition)            ==   0, "c0");
    static_assert(offsetof(PBREffectConstants, world)                  ==  16, "c1");
    static_assert(offsetof(PBREffectConstants, worldInverseTranspose)  ==  80, "c5");
    static_assert(offsetof(PBREffectConstants, worldViewProj)          == 128, "c8");
    static_assert(offsetof(PBREffectConstants, prevWorldViewProj)      == 192, "c12");
    static_assert(offsetof(PBREffectConstants, lights)                 == 256, "c16");
    static_assert(offsetof(PBREffectConstants, Albedo)                 == 512, "c32");
    static_assert(offsetof(PBREffectConstants, Metallic)               == 528, "c33.x");
    static_assert(offsetof(PBREffectConstants, Roughness)              == 532, "c33.y");
    static_assert(offsetof(PBREffectConstants, numRadianceMipLevels)   == 536, "c33.z");
    static_assert(offsetof(PBREffectConstants, lightCount)             == 540, "c33.w");
    static_assert(offsetof(PBREffectConstants, targetWidth)            == 544, "c34.x");
    static_assert(offsetof(PBREffectConstants, targetHeight)           == 548, "c34.y");
    static_assert( sizeof(PBREffectConstants) == 560,
                   "PBR_Constants packoffsets in no_texture.fx assume this layout" );

    // The node describe what things in a scene will input into AI model or what things need AI model to evaluate
    class GLTFAdapter {
    public:
        fastgltf::Expected<fastgltf::Asset> Initialize(const std::string &gltfFilepath);
        void PrepareBuffer(std::shared_ptr<DX::DeviceResources> deviceResources);
        void PrepareImage(std::shared_ptr<DX::DeviceResources> deviceResources);
        void PreparePSO(std::shared_ptr<DX::DeviceResources> deviceResources);
        // Per-image colour space, decided from the material slots that
        // reference each image. PrepareImage computes this before creating any
        // texture, so each one gets the matching DXGI format.
        const ImageColorSpaceClassification &ImageColorSpaces() const {
            return m_imageColorSpaces;
        }

        // Resolves the scene hierarchy and per-primitive buffer views.
        // Must run after PrepareBuffer and PrepareImage, whose results it
        // references.
        void BuildSceneGraph();
        void AssignCamera(std::shared_ptr<Camera> camera) { m_camera = camera; }

        // Advances the selected animation and poses the scene's nodes. Call
        // once per frame BEFORE Render, which is what turns those local
        // transforms into world transforms and joint matrices.
        //
        // Nothing about this touches the shader: the animated node transforms
        // reach it through the joint matrices already bound at b2, and through
        // PBR_World for an animated node that is not skinned.
        void UpdateAnimation(float deltaSeconds);
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

        // The loaded scene as a hierarchy of SceneNode, each carrying its own
        // transform and the buffer views needed to draw it.
        const SceneGraph &Scene() const { return m_sceneGraph; }

        // Where the whole scene sits. Retargeting needs this to aim at a WORLD
        // position: the hips translation it writes is in the hips' parent
        // space, and this placement is the outermost part of that chain, so
        // ignoring it would offset the character by exactly this much.
        DirectX::SimpleMath::Matrix ScenePlacement() const {
            return DirectX::SimpleMath::Matrix::CreateTranslation(
                m_position[0], m_position[1], m_position[2]);
        }

        // Retargeting poses the graph from outside, so it needs write access.
        // Like ApplyAnimation, anything it writes must happen before Render
        // composes the world transforms.
        SceneGraph &Scene() { return m_sceneGraph; }
    private:
        fastgltf::Asset m_gltf;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pso;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> images;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocs[kBackBufferSize][kWorkerThreadSize];
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
        Microsoft::WRL::ComPtr<ID3D12Resource> cbuffer;
        std::size_t m_currentSceneIdx = 0;
        // Built once in PrepareImage, walked top-down every frame in Render.
        SceneGraph m_sceneGraph;
        std::shared_ptr<Camera> m_camera;
        std::shared_ptr<DirectX::CommonStates> m_commonStates;

        // --- Page-based descriptor management -------------------------------
        // The Core idea of DescriptorAllocator and DynamicDescriptorHeap:
        // DescriptorAllocator is for descriptor generation/destruction for resources access, which can be very messy and generally not GPU-favorable (GPU favor continuous descriptor access)
        // DynamicDescriptorHeap maintains a structured descriptor layout for GPU to use it (Through binding DescriptorTable and assign descriptor range on continuous descriptor handles)
        // First, we generate resources corresponding srv, uav, rtv, …, etc into CPU-side descriptor handle in DescriptorAllocator
        // Then, when they need to be render, they will first arrange(StageDescriptors) what the CPU-side descriptor handle (in DescriptorAllocator) will map to the other CPU-side descriptor handle (in DynamicDescriptorHeap), and then CommitStageDescriptors to actually copy descriptors to CPU-side DynamicDescriptorHeap via specified handle mapping all at once. In the executecommandlist, the GPU-side descriptor handle in DynamicDescriptorHeap will automatically reference the CPU-side descriptor handle

        // The root signature, wrapped so DynamicDescriptorHeap can read its
        // descriptor-table layout.
        Descriptors::RootSignature m_rootSignature;
        // CPU-visible SRV descriptors handed out in pages: one allocation per
        // glTF image, held as long as the image lives. The allocation's
        // destructor returns it to its page.
        std::unique_ptr<Descriptors::DescriptorAllocator> m_srvAllocator;
        std::vector<Descriptors::DescriptorAllocation> m_imageDescriptors;
        // Filled at the top of PrepareImage, before any texture is created.
        ImageColorSpaceClassification m_imageColorSpaces;
        // A 1x1 opaque white texture, staged into any material texture slot the
        // material leaves empty. Keeps every descriptor in the five-wide SRV
        // table valid, so the shader can sample unconditionally and branch on
        // the Mat_Has*Texture flags instead.
        Microsoft::WRL::ComPtr<ID3D12Resource> m_fallbackTexture;
        Descriptors::DescriptorAllocation m_fallbackTextureDescriptor;
        // Copies staged CPU descriptors into a GPU-visible heap at draw time and
        // binds the resulting table.
        std::unique_ptr<Descriptors::DynamicDescriptorHeap> m_srvDynamicHeap;
        // Keeps the dynamic SRV heap and the DirectXTK12 sampler heap bound
        // together across dynamic-heap switches.
        Descriptors::HeapBinder m_heapBinder;
        float m_position[3] = {3.0f, -2.0f, -4.0f};

        // --- Animation playback ---------------------------------------------
        // Empty when the asset has no animations, or when the user selects
        // "none" to see the authored bind pose.
        std::optional<std::size_t> m_activeAnimation;
        // Distinguishes "no clip chosen yet" from "the user deliberately chose
        // the authored pose". Without it an empty selection is indistinguishable
        // from a fresh start and clip 0 gets re-selected every frame.
        bool m_animationAutoSelected = false;
        float m_animationTime = 0.0f;
        float m_animationSpeed = 1.0f;
        bool m_animationPlaying = true;
        bool m_animationLoop = true;
    };
}