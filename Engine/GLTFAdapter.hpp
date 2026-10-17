#pragma once

#include <filesystem>
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
#include "Lightmap.hpp"
#include "SceneGraph.hpp"
#include "Ui.hpp"

static constexpr std::size_t kBackBufferSize = 2;
static constexpr std::size_t kWorkerThreadSize = 4;

namespace NeuralModelIntegrateTestbed {
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

        // Longest edge any texture is allowed to keep, or 0 for "whatever the
        // asset says". Must be set before PrepareImage.
        //
        // Every image is decoded to R8G8B8A8 and given a full mip chain -- no
        // BC compression anywhere -- so a 4096x4096 texture costs 85 MB of
        // video memory. A scene with a few hundred of those does not fit, and
        // the failure is an out-of-memory on CreateCommittedResource part way
        // through loading rather than anything diagnosable. Halving the longest
        // edge quarters the cost.
        void SetMaxTextureDimension(std::uint32_t maxDimension) {
            m_maxTextureDimension = maxDimension;
        }

        // Mirrors `cbuffer LightmapBakeConstants : register(b4)` in
        // shaders/no_texture.fx. Offsets are the ones dxc reports for it, not
        // ones computed by hand.
        struct LightmapBakeConstants {
            DirectX::XMMATRIX sunViewProjection;   // 0
            DirectX::XMFLOAT4 chartScaleOffset;    // 64
            float depthBias = 0.0f;                // 80
            float normalBias = 0.0f;               // 84
            float sunTexelSize = 0.0f;             // 88
            float sunUvTexel = 0.0f;               // 92
        };
        static_assert(sizeof(LightmapBakeConstants) == 96,
                      "dxc reports LightmapBakeConstants as 96 bytes");
        static_assert(offsetof(LightmapBakeConstants, chartScaleOffset) == 64, "c4");
        static_assert(offsetof(LightmapBakeConstants, depthBias) == 80, "c5.x");
        static_assert(offsetof(LightmapBakeConstants, sunUvTexel) == 92, "c5.w");

        // Per-frame counters, filled by RecordGeometry. The point of loading a
        // large scene is to watch these.
        struct GeometryStats {
            std::size_t nodes = 0;
            std::size_t draws = 0;
            std::size_t triangles = 0;
            // Draws whose material differed from the previous draw's, and so
            // had to rebind the SRV table. No descriptors are copied for these
            // any more -- the table is built once at load -- so this is now
            // just a root-table set, not a staging cost.
            std::size_t materialTableBinds = 0;
            std::size_t pipelineSwitches = 0;
            std::size_t skinnedNodes = 0;
            // Primitives the frustum rejected, and those it was not allowed to
            // judge: a primitive with no authored POSITION bounds, and any
            // skinned one, whose vertices are moved by its joint matrices and
            // so are not bounded by the authored box.
            std::size_t culledPrimitives = 0;
            std::size_t unculledPrimitives = 0;
            // Summed over every cascade, so this is larger than the primitive
            // count when more than one cascade sees the same thing.
            std::size_t shadowDraws = 0;
            std::size_t shadowCulled = 0;
        };
        const GeometryStats &Stats() const { return m_stats; }
        // Whether any drawable node can move. False means every shadow map in
        // the scene can be rendered once and reused.
        bool HasMovingGeometry() const {
            return m_sceneGraph.MovingGeometryNodeCount() != 0;
        }

        // The baked-lighting chart allocation, and what it covers. Empty until
        // BuildLightmapCharts has run.
        const LightmapAllocation &Lightmap() const { return m_lightmap; }
        const LightmapSetup &LightmapConfig() const { return m_lightmapSetup; }
        // Static instances that got a chart, and static instances that could
        // not have one because their mesh carries no second UV set. The second
        // number is what an unwrapper would fix.
        std::size_t LightmapChartCount() const { return m_lightmapCharts; }
        std::size_t LightmapMissingUvCount() const { return m_lightmapMissingUv; }
        float LightmapAreaCovered() const { return m_lightmapArea; }
        std::size_t LightmapBakedPrimitives() const {
            return m_lightmapBakedPrimitives;
        }
        void PreparePSO(std::shared_ptr<DX::DeviceResources> deviceResources);
        // The GPU handle of a primitive's five-descriptor material block.
        D3D12_GPU_DESCRIPTOR_HANDLE MaterialTable(std::uint32_t index) const {
            D3D12_GPU_DESCRIPTOR_HANDLE handle = m_materialTableStart;
            handle.ptr += static_cast<UINT64>(index) * kMaterialTextureSlotCount *
                          m_materialDescriptorIncrement;
            return handle;
        }
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
        // Builds m_materialTableHeap from the scene graph and assigns every
        // primitive its block index. Called at the end of PreparePSO.
        void BuildMaterialDescriptorTables(
            std::shared_ptr<DX::DeviceResources> deviceResources);
        // Sizes and places one lightmap chart per static instance that has
        // lightmap UVs, writing the placement into each PrimitiveResource.
        // Updates the world transforms first, so it does not care whether the
        // caller has framed the scene yet.
        void BuildLightmapCharts();
        // Points the material heap's last slot at the lightmap atlas. Called
        // from the G-buffer pass, which is where the atlas's view is available.
        void SetLightmapAtlasSrv(D3D12_CPU_DESCRIPTOR_HANDLE srv);
        // Draws every charted static primitive into its own lightmap chart,
        // testing each surface point against `sunDepth`. One call bakes the
        // whole atlas; the caller has already bound it as the render target.
        void RecordLightmapBake(ID3D12GraphicsCommandList *commandList,
                                const DirectX::SimpleMath::Matrix &sunViewProjection,
                                float sunTexelWorldSize,
                                D3D12_CPU_DESCRIPTOR_HANDLE sunDepth);
        void AssignCamera(std::shared_ptr<Camera> camera) { m_camera = camera; }

        // Advances the selected animation and poses the scene's nodes. Call
        // once per frame BEFORE Render, which is what turns those local
        // transforms into world transforms and joint matrices.
        //
        // Nothing about this touches the shader: the animated node transforms
        // reach it through the joint matrices already bound at b2, and through
        // PBR_World for an animated node that is not skinned.
        void UpdateAnimation(float deltaSeconds);

        // Composes every node's world transform and places the scene's lights.
        // Split out of the old Render because the deferred frame needs the
        // results before it starts recording: the lighting pass wants the lights
        // and retargeting wants the transforms, and both run outside the
        // geometry pass.
        void UpdateSceneTransforms();

        // Records the geometry draws for the G-buffer pass into the caller's
        // command list.
        //
        // Deliberately does not touch render targets, viewports or barriers --
        // the render graph owns those, and a pass that re-binds them behind the
        // graph's back is how a frame ends up with the right draws going to the
        // wrong place. It also no longer owns a command list: one list per frame
        // means the graph can put a barrier between this and the lighting pass.
        //
        // UpdateSceneTransforms must have run first.
        void RecordGeometry(ID3D12GraphicsCommandList *commandList,
                            const DirectX::SimpleMath::Matrix &viewMatrix,
                            const DirectX::SimpleMath::Matrix &projectionMatrix);

        // Records the same geometry depth-only, from a light's point of view,
        // for one shadow cascade. `lightViewProjection` goes in where the
        // camera's view-projection normally does, which is all VSShadow needs:
        // it is VSGBuffer, and VSGBuffer takes its clip position from
        // PBR_WorldViewProj.
        //
        // Culls against the cascade's own frustum, which is why it takes the
        // combined matrix rather than the camera's two.
        //
        // Does not set a viewport or scissor: the caller places each cascade in
        // its tile of the atlas, and the render graph has already bound the
        // depth target.
        // `movingOnly` selects which half of the scene to draw: the static
        // half goes into the cached atlas when a cascade is refitted, the
        // moving half into the live atlas every frame. See
        // SceneNode::movesAtRuntime for what counts as moving.
        void RecordShadowGeometry(ID3D12GraphicsCommandList *commandList,
                                  const DirectX::SimpleMath::Matrix &lightViewProjection,
                                  bool movingOnly);
        void ShowImgui(UiSection section);
        // The three tabs this class draws into. Declared here rather than
        // private only because ShowImgui dispatches to them and nothing
        // else does; the access section is left as it is on purpose.
        void ShowRenderUi();
        void ShowSceneUi();
        void ShowPerformanceUi();
        // Only the sampler heap comes from the caller now. SRV descriptors for
        // glTF images are allocated from this class's own DescriptorAllocator
        // instead of borrowing slots out of the application's fixed heap.
        void SetCommonStates(std::shared_ptr<DirectX::CommonStates> commonStates) {
            m_commonStates = commonStates;
        }

        // Recycles descriptors retired by earlier frames. Call once per frame.
        void ReleaseStaleDescriptors(uint64_t frameNumber);

        // Frustum culling, on by default. The toggle is here because the only
        // way to be sure culling is not eating something it should have kept is
        // to turn it off and look.
        void SetFrustumCulling(bool enabled) { m_frustumCulling = enabled; }
        bool FrustumCulling() const { return m_frustumCulling; }

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

        // World-space bounds of the loaded scene, for framing a camera on it.
        // Composes the transforms first, so it is usable straight after
        // BuildSceneGraph and before the first frame.
        bool SceneBounds(DirectX::XMFLOAT3& outMin, DirectX::XMFLOAT3& outMax) {
            m_sceneGraph.UpdateTransforms(ScenePlacement());
            return m_sceneGraph.WorldBounds(outMin, outMax);
        }

        // Retargeting poses the graph from outside, so it needs write access.
        // Like ApplyAnimation, anything it writes must happen before Render
        // composes the world transforms.
        SceneGraph &Scene() { return m_sceneGraph; }
    private:
        fastgltf::Asset m_gltf;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_pso;
        // The same pipeline with CullMode NONE, for materials glTF marks
        // doubleSided. Selected per primitive; a scene with none of them never
        // binds it.
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoDoubleSided;
        // Depth-only, no render targets, culling off: a shadow caster is a
        // caster from either side, and the alpha-clipped foliage that most
        // needs shadowing is single-sided geometry seen from the sun's side as
        // often as not.
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_shadowPso;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> images;
        Microsoft::WRL::ComPtr<ID3D12Resource> cbuffer;
        std::size_t m_currentSceneIdx = 0;
        // Built once in PrepareImage, walked top-down every frame in Render.
        SceneGraph m_sceneGraph;
        std::shared_ptr<Camera> m_camera;
        std::shared_ptr<DirectX::CommonStates> m_commonStates;

        // --- Page-based descriptor management -------------------------------
        // DescriptorAllocator hands out CPU-visible descriptors in pages and
        // keeps them alive for as long as the thing they describe: one per glTF
        // image, allocated by PrepareImage. Nothing samples from those -- a
        // shader reads descriptors out of a shader-visible heap, which an
        // allocator page is not.
        //
        // BuildMaterialDescriptorTables copies them into one shader-visible
        // heap at load, five at a time, one block per distinct material texture
        // tuple. A draw then binds its block by offset. The descriptors never
        // change after load, so there is nothing for a frame to copy.
        //
        // The root signature, wrapped because Descriptors::RootSignature is
        // what serialises a D3D12_ROOT_SIGNATURE_DESC1 here. Its parsed
        // table layout is no longer read by anything in this class.
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
        // No DynamicDescriptorHeap here any more: staging descriptors per
        // draw is exactly what the material heap below replaces. The class is
        // still used by DeferredRenderer, whose fullscreen passes bind a
        // handful of views that genuinely do change per frame.
        //
        // Every material's SRV table, in one GPU-visible heap built once at
        // load: a five-descriptor block per distinct texture tuple. A draw
        // binds a block by offset, so no descriptors are copied per frame.
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_materialTableHeap;
        D3D12_GPU_DESCRIPTOR_HANDLE m_materialTableStart = {};
        UINT m_materialDescriptorIncrement = 0;
        std::size_t m_materialTableCount = 0;
        // The slot in that heap holding the lightmap atlas, one past the
        // material blocks.
        std::size_t m_lightmapAtlasSlot = 0;
        bool m_lightmapAtlasBound = false;

        // --- baked lighting ---------------------------------------------
        LightmapSetup m_lightmapSetup;
        LightmapAllocation m_lightmap;
        std::size_t m_lightmapCharts = 0;
        std::size_t m_lightmapMissingUv = 0;
        float m_lightmapArea = 0.0f;
        // Rasterises a primitive into its chart rather than into the camera.
        // No depth buffer, no culling: the chart has to be covered whichever
        // way the triangle faces.
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_lightmapBakePso;
        Descriptors::RootSignature m_lightmapBakeRootSignature;
        // One shader-visible descriptor, for the sun depth map the bake reads.
        // A bake binds this instead of the material heap: it samples no
        // material textures at all.
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_lightmapBakeHeap;
        UINT m_lightmapBakeDescriptorSize = 0;
        // In the sun projection's depth units. Small, like the cascades':
        // the normal offset below is what actually clears acne.
        float m_lightmapDepthBias = 0.0005f;
        // In sun-depth texels, along the surface normal.
        float m_lightmapNormalBias = 1.5f;
        std::size_t m_lightmapBakedPrimitives = 0;
        // Keeps the material SRV heap and the DirectXTK12 sampler heap bound
        // together.
        Descriptors::HeapBinder m_heapBinder;
        float m_position[3] = {3.0f, -2.0f, -4.0f};
        bool m_frustumCulling = true;

        // Where the .gltf lives. Image URIs are resolved against it, because
        // images are no longer pre-loaded by fastgltf -- PrepareImage opens them
        // itself so a missing file is a warning rather than a failed parse.
        std::filesystem::path m_assetDirectory;

        // 0 means unlimited; see SetMaxTextureDimension.
        std::uint32_t m_maxTextureDimension = 0;
        // Totals from PrepareImage, for the UI. Video memory is the sum of
        // every mip of every texture, which is the number that decides whether
        // a scene loads at all.
        std::uint64_t m_textureBytes = 0;
        std::size_t m_texturesDownscaled = 0;
        // Block-compressed textures whose largest mip is already over the cap,
        // so the cap could not be applied.
        std::size_t m_texturesOverBudget = 0;
        // Images whose file could not be opened, and images whose bytes could
        // not be decoded. Both get a magenta placeholder so the index space
        // stays aligned with what the materials resolved against.
        std::size_t m_imagesMissing = 0;
        std::size_t m_imagesFailed = 0;
        // Images no texture points at, skipped without being read. An
        // MSFT_texture_dds asset lists one per texture.
        std::size_t m_imagesUnused = 0;
        GeometryStats m_stats;
        // The shadow passes record BEFORE the G-buffer pass, so the frame's
        // shadow counters are already filled when RecordGeometry resets the
        // rest and must survive that reset. This says whether a shadow pass has
        // run since the last G-buffer pass: false means the counters belong to
        // this frame, true means no shadow pass contributed to them and they
        // should start from zero -- which is also what happens, every frame,
        // when shadows are switched off.
        bool m_shadowStatsStale = true;

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