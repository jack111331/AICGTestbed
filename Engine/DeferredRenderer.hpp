#pragma once

// Deferred shading, built on Engine/RenderGraph.hpp.
//
// Forward shading runs the light loop once per pixel per primitive that covers
// it, so the cost is lights x overdraw. Deferred runs the geometry once to
// record each pixel's material inputs, then the light loop once per pixel --
// the cost becomes lights + overdraw. That is the whole reason to do it, and it
// is what makes a scene with more than a handful of lights affordable.
//
// What it costs in exchange, all of which is visible in this file:
//   * Bandwidth. Four render targets written and read back every frame.
//   * No per-material shading. Every pixel is shaded by one pixel shader, so a
//     material model has to be expressible in the G-buffer's channels.
//   * Transparency does not fit. A blended surface needs the lit colour of what
//     is behind it, which the G-buffer cannot hold -- one pixel, one surface.
//     Nothing in the scene is blended yet; when something is, it belongs in a
//     forward pass after the lighting resolve, sharing the same depth buffer.
//
// This class owns the G-buffer targets, the lighting root signature and PSO,
// and the depth shader resource view. It does not own the geometry: the caller
// hands it a callback, so it has no dependency on GLTFAdapter.

#include <cstddef>
#include <functional>
#include <memory>
#include <cstdint>
#include <string>

#include <directx/d3d12.h>
#include <wrl/client.h>

#include <DirectXMath.h>
#include <SimpleMath.h>

#include "DeviceResources.hpp"
#include "Descriptors/DescriptorAllocation.hpp"
#include "Descriptors/DescriptorAllocator.hpp"
#include "Descriptors/DescriptorHeapBinder.hpp"
#include "Descriptors/DynamicDescriptorHeap.hpp"
#include "Descriptors/RootSignature.hpp"
#include "GBuffer.hpp"
#include "RenderGraph.hpp"
#include "Lightmap.hpp"
#include "SceneGraph.hpp"
#include "ShadowCascades.hpp"
#include "Ui.hpp"

namespace NeuralModelIntegrateTestbed {

// How many lights one lighting pass can apply. Must equal DEFERRED_MAX_LIGHTS
// in shaders/no_texture.fx.
//
// 128 rather than the forward path's 4 because this buffer is uploaded once per
// frame instead of once per node, so the room is nearly free: 128 lights is
// 8 KB against a 64 KB constant buffer limit. It was 64 until the Lumberyard
// Bistro turned up with 97 (96 point plus one directional), which is exactly
// the kind of silent truncation that reads as "the scene is too dark". Beyond a few dozen the loop
// itself becomes the cost and the answer is to stop testing every light against
// every pixel -- tiled or clustered culling -- which is a pass of its own and
// is why the number is not simply raised to the limit.
constexpr std::size_t kMaxDeferredLights = 128;

// Mirrors `struct ShadowCascadeData` in shaders/no_texture.fx. Six constant
// registers: the matrix, the atlas rectangle, and two scalars with padding.
struct ShadowCascadeConstants {
    DirectX::XMMATRIX viewProjection;        // +0
    DirectX::XMFLOAT4 atlasScaleOffset;      // +64
    float splitDistance;                     // +80
    float texelWorldSize;                    // +84
    float padding[2];                        // +88
};
static_assert(sizeof(ShadowCascadeConstants) == 96,
              "ShadowCascadeData in no_texture.fx assumes six registers per cascade");

// Mirrors `cbuffer DeferredLightingConstants : register(b3)` in
// shaders/no_texture.fx. The HLSL side spells out every packoffset, so the two
// have to be changed together; the static_asserts below are what notice.
struct DeferredLightingConstants {
    DirectX::XMMATRIX invViewProj;               // c0..c3
    DirectX::XMFLOAT3 eyePosition;               // c4.xyz
    int lightCount;                              // c4.w
    float targetWidth;                           // c5.x
    float targetHeight;                          // c5.y
    float padding[2];                            // c5.zw
    ShaderLight lights[kMaxDeferredLights];      // c6..

    // Appended after the lights, which is why the offsets jump: 128 lights at
    // 64 bytes each run from 96 to 8288. Verified against the layout dxc
    // reports for the cbuffer, not just computed -- see the asserts below.
    ShadowCascadeConstants cascades[kMaxShadowCascades];  // 8288
    int shadowCascadeCount;                               // 8672
    float shadowDepthBias;                                // 8676
    float shadowNormalBias;                               // 8680
    float shadowAtlasResolution;                          // 8684
};

static_assert(offsetof(DeferredLightingConstants, invViewProj) == 0, "c0");
static_assert(offsetof(DeferredLightingConstants, eyePosition) == 64, "c4.xyz");
static_assert(offsetof(DeferredLightingConstants, lightCount) == 76, "c4.w");
static_assert(offsetof(DeferredLightingConstants, targetWidth) == 80, "c5.x");
static_assert(offsetof(DeferredLightingConstants, targetHeight) == 84, "c5.y");
static_assert(offsetof(DeferredLightingConstants, lights) == 96, "c6");
static_assert(offsetof(DeferredLightingConstants, cascades) == 8288, "c518");
static_assert(offsetof(DeferredLightingConstants, shadowCascadeCount) == 8672, "c542.x");
static_assert(offsetof(DeferredLightingConstants, shadowDepthBias) == 8676, "c542.y");
static_assert(offsetof(DeferredLightingConstants, shadowNormalBias) == 8680, "c542.z");
static_assert(offsetof(DeferredLightingConstants, shadowAtlasResolution) == 8684,
              "c542.w");
static_assert(sizeof(DeferredLightingConstants) == 8688,
              "dxc reports this cbuffer as 8688 bytes");
static_assert((sizeof(DeferredLightingConstants) % 16) == 0, "CB size not padded correctly");
static_assert(sizeof(DeferredLightingConstants) <= 65536,
              "a constant buffer may not exceed 64 KB; lower kMaxDeferredLights");

// Where every pass before the tonemap writes.
//
// Half float, not the back buffer's 8-bit UNORM, because the whole point is to
// keep radiance before the display transform: a sun at 10 W/m^2 lights a white
// surface to 3.18, which an 8-bit target cannot hold and which has to survive
// until PSTonemap decides what to do with it.
//
// R11G11B10_FLOAT would be half the bandwidth and is the usual choice, but it
// has no alpha and no sign; this keeps RGBA16F until something needs the
// bandwidth back.
constexpr DXGI_FORMAT kSceneColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Mirrors `cbuffer SkyboxConstants : register(b4)` in shaders/no_texture.fx.
struct SkyboxConstants {
    DirectX::XMMATRIX invViewProj;   // c0..c3
    DirectX::XMFLOAT3 eyePosition;   // c4.xyz
    float exposure;                  // c4.w
    float targetWidth;               // c5.x
    float targetHeight;              // c5.y
    float yaw;                       // c5.z
    float padding;                   // c5.w
};
static_assert(offsetof(SkyboxConstants, invViewProj) == 0, "c0");
static_assert(offsetof(SkyboxConstants, eyePosition) == 64, "c4.xyz");
static_assert(offsetof(SkyboxConstants, exposure) == 76, "c4.w");
static_assert(offsetof(SkyboxConstants, targetWidth) == 80, "c5.x");
static_assert(offsetof(SkyboxConstants, targetHeight) == 84, "c5.y");
static_assert(offsetof(SkyboxConstants, yaw) == 88, "c5.z");
static_assert((sizeof(SkyboxConstants) % 16) == 0, "CB size not padded correctly");

// What the frame resolves to the back buffer.
//
// Every one of these but Scene replaces the display transform rather than
// drawing over it. A thumbnail in the corner would be resampled twice and
// would hide exactly the thing one of these views is usually opened to find:
// one wrong texel, or a channel that is zero where it should not be.
//
// Must stay in step with kDebugViews in DeferredRenderer.cc, which gives each
// one its resource and its channel mapping.
enum class DebugView {
    Scene = 0,          // the real frame: scene colour through the tonemap
    SceneColor,         // the same target with no display transform
    BaseColor,
    Normal,
    Metallic,
    Roughness,
    Occlusion,
    BakedSunVisibility, // the lightmap's term as the G-buffer carries it
    Emissive,
    Depth,
    ShadowAtlas,
    LightmapAtlas,
    LightmapSunDepth,
    Count,
};

// How PSDebugView turns the sampled texel into something to look at. Must
// match the DEBUG_CHANNELS_* defines in shaders/no_texture.fx.
enum class DebugChannels {
    Tonemapped = 0,
    Color = 1,
    Normal = 2,
    Red = 3,
    Green = 4,
    Blue = 5,
    Alpha = 6,
    Depth = 7,
};

// Mirrors `cbuffer DebugViewConstants : register(b5)` in no_texture.fx.
struct DebugViewConstants {
    int channels;        // c0.x
    float sourceWidth;   // c0.y
    float sourceHeight;  // c0.z
    float padding;       // c0.w
    float targetWidth;   // c1.x
    float targetHeight;  // c1.y
    float rangeMin;      // c1.z
    float rangeMax;      // c1.w
    float nearPlane;     // c2.x
    float farPlane;      // c2.y
    float padding2[2];   // c2.zw
};
static_assert(offsetof(DebugViewConstants, channels) == 0, "c0.x");
static_assert(offsetof(DebugViewConstants, targetWidth) == 16, "c1.x");
static_assert(offsetof(DebugViewConstants, rangeMin) == 24, "c1.z");
static_assert(offsetof(DebugViewConstants, nearPlane) == 32, "c2.x");
static_assert((sizeof(DebugViewConstants) % 16) == 0, "CB size not padded correctly");

const char* DebugViewName(DebugView view);

// A view-time override of the sun's direction, in Euler angles.
//
// TWO angles, not three. A direction has two degrees of freedom; a third Euler
// angle would be a roll about the direction itself and would change nothing.
//
// The convention is the camera panel's, applied to the direction the light
// TRAVELS -- which is what glTF stores and what the shader consumes, the node's
// -Z axis in world space:
//
//   pitch  elevation of that direction. 0 is horizontal, -90 is straight down,
//          so a sun in the sky has a NEGATIVE pitch.
//   yaw    measured from -Z, so (0, 0) is the glTF default direction (0,0,-1).
//
// A view-time override rather than an edit to the SceneLight, because
// SceneGraph::UpdateTransforms recomputes that direction from the light's node
// every frame and would undo it.
struct SunDirectionOverride {
    bool enabled = false;
    float pitchDegrees = -45.0f;
    float yawDegrees = 0.0f;
};

// The two conversions these angles need are in ShadowCascades.hpp:
// SunDirectionFromEuler and SunEulerFromDirection. They live there because
// they are arithmetic with no device in them, which is what lets
// //Engine:scenegraph-smoke assert on the round trip.

class DeferredRenderer {
public:
    // What one frame's lighting pass needs. Gathered by the caller before the
    // graph is built, because the lights come from the scene graph and the
    // matrices from the camera, and neither belongs to this class.
    struct FrameInputs {
        DirectX::SimpleMath::Matrix view;
        DirectX::SimpleMath::Matrix projection;
        DirectX::XMFLOAT3 eyePosition = {0.0f, 0.0f, 0.0f};
        const ShaderLight* lights = nullptr;
        std::size_t lightCount = 0;

        // The scene's bounds, so the cascade fit can reach every caster: an
        // object behind a slice still has to be in the light's view or its
        // shadow does not exist.
        DirectX::SimpleMath::Vector3 sceneMin;
        DirectX::SimpleMath::Vector3 sceneMax;
        bool hasSceneBounds = false;

        // Whether anything in the scene can move. With nothing moving there is
        // no compositing to do at all: the cached static atlas IS the frame's
        // shadow map, and the lighting pass samples it directly.
        bool hasMovingGeometry = false;
    };

    // Records the geometry that writes the G-buffer. Everything the graph owns
    // -- targets, viewport, barriers -- is already set when this is called.
    // Takes the lightmap atlas's view as well as the command list: the
    // G-buffer pass is where the baked term is sampled, and the atlas is
    // the graph's resource rather than the recorder's.
    using GeometryRecorder =
        std::function<void(ID3D12GraphicsCommandList*,
                          D3D12_CPU_DESCRIPTOR_HANDLE)>;

    // Which half of the scene a shadow recording covers. The split is the
    // whole basis of the cache: static depth is the same in every frame, so it
    // is rendered only when a cascade is refitted, while moving geometry has to
    // be redrawn on top of it every frame.
    enum class ShadowGeometry {
        Static,
        Moving,
    };

    // Records geometry depth-only for one cascade, given that cascade's light
    // view-projection. Separate from GeometryRecorder because the caller binds
    // a different pipeline for it.
    using ShadowRecorder = std::function<void(ID3D12GraphicsCommandList*,
                                              const DirectX::SimpleMath::Matrix&,
                                              ShadowGeometry)>;

    // Draws every charted static primitive into its own lightmap chart. Called
    // once, on the frame the bake happens, with the whole-scene sun matrix, one
    // sun-depth texel in world units, and the depth map to test against.
    using LightmapBakeRecorder =
        std::function<void(ID3D12GraphicsCommandList*,
                           const DirectX::SimpleMath::Matrix&, float,
                           D3D12_CPU_DESCRIPTOR_HANDLE)>;

    // The two resources the frame is resolved into. Imported rather than
    // declared, because DeviceResources creates and recreates both.
    struct FrameTargets {
        Render::ResourceHandle color;
        Render::ResourceHandle depth;
    };

    // Declares the G-buffer in the graph and builds the lighting pipeline.
    // The graph keeps the targets across frames, so this runs once.
    void CreateDeviceDependentResources(ID3D12Device* device, Render::RenderGraph& graph);

    // Rebuilds the depth shader resource view against the current depth buffer.
    // Must run after every DeviceResources::CreateWindowSizeDependentResources,
    // which recreates that resource.
    void CreateWindowSizeDependentResources(DX::DeviceResources& deviceResources);

    void OnDeviceLost();

    // Loads an equirectangular environment map to use as the sky.
    //
    // Two containers, both of which the renderer can already read:
    //   .hdr  Radiance RGBE, through stb_image's float loader
    //   .dds  anything DirectXTK12 understands, including BC6H and
    //         R16G16B16A16_FLOAT
    //
    // NOT .exr. OpenEXR is its own decoder -- the common compression for a
    // released HDRI is PIZ, a wavelet plus Huffman scheme that is not something
    // to reimplement -- and nothing in this project links one. Convert it; see
    // the README.
    //
    // Returns false and fills `error` rather than throwing: a missing or
    // unreadable sky should cost the sky, not the scene.
    bool LoadEnvironment(DX::DeviceResources& deviceResources, const std::string& path,
                         std::string* error);

    bool HasEnvironment() const { return m_environment != nullptr; }


    // Imports this frame's back buffer and depth buffer. Separate from
    // AddPasses so the caller can add its own passes against the same handles.
    //
    // `clearColor` is what the lighting pass clears the back buffer to, and so
    // what shows through wherever no geometry was drawn.
    FrameTargets ImportFrameTargets(Render::RenderGraph& graph,
                                    DX::DeviceResources& deviceResources,
                                    const float clearColor[4]);

    // Adds the G-buffer pass, the sky and the lighting pass. All of them
    // resolve into the HDR scene colour target, NOT the back buffer -- see
    // AddTonemapPass.
    void AddPasses(Render::RenderGraph& graph, const FrameTargets& targets,
                   const FrameInputs& inputs, GeometryRecorder recordGeometry,
                   ShadowRecorder recordShadows = nullptr,
                   LightmapBakeRecorder recordLightmapBake = nullptr);

    // Asks for the lightmap to be baked on the next frame that has a sun and
    // something to bake. Called once at load, and again whenever the bake is
    // invalidated -- which, because a bake fixes the light into the data, means
    // whenever the sun moves.
    void RequestLightmapBake() { m_lightmapBakePending = true; }
    bool LightmapBaked() const { return m_lightmapBaked; }

    // Sizes the shadow distance to the scene, so a two-metre fixture and a
    // hundred-and-sixty-metre street both get cascades worth having. Called
    // from wherever the camera is framed.
    void SetSceneScale(float sceneRadius);

    // Reads the scene colour target and writes the back buffer through the
    // display transform. Added last of the world-space passes, so anything
    // that wants to be tonemapped -- the sky, the lighting resolve, the forward
    // overlays -- has to go in before it, and anything authored in display
    // space, like the UI, after.
    void AddTonemapPass(Render::RenderGraph& graph, Render::ResourceHandle backBuffer);

    // The HDR target the world-space passes share. Exposed so a caller can add
    // its own forward pass alongside them.
    Render::ResourceHandle SceneColor() const { return m_sceneColor; }

    Render::ResourceHandle GBufferHandle(GBufferTarget target) const {
        return m_gbuffer[static_cast<std::size_t>(target)];
    }

    // What the last Compile() produced, for the UI. Set by AddPasses' caller
    // rather than read from the graph, so a frame that failed to compile still
    // shows why.
    void SetPlanDescription(std::string description) {
        m_planDescription = std::move(description);
    }

    void ShowImgui(UiSection section);

private:
    void CreateLightingPipeline(ID3D12Device* device, DXGI_FORMAT backBufferFormat);
    void CreateSkyboxPipeline(ID3D12Device* device, DXGI_FORMAT backBufferFormat);
    void CreateTonemapPipeline(ID3D12Device* device, DXGI_FORMAT backBufferFormat);
    void CreateDebugViewPipeline(ID3D12Device* device,
                                 DXGI_FORMAT backBufferFormat);
    // The one pass of the two that AddTonemapPass actually adds.
    void AddDebugViewPass(Render::RenderGraph& graph,
                          Render::ResourceHandle backBuffer);
    void ShowRenderUi();
    void ShowPerformanceUi();
    // Finds this frame's sun and settles which direction it is actually
    // shining in, before anything reads it. Hoisted out of
    // UpdateShadowCascades because the lighting pass needs the same answer
    // even with shadows switched off, and two places deciding it separately is
    // how the shadows end up disagreeing with the shading.
    void ResolveSun(const FrameInputs& inputs);

    // Recomputes the cascades for this frame, and reports whether there is a
    // directional light to compute them for at all.
    bool UpdateShadowCascades(const FrameInputs& inputs);

    ID3D12Device* m_device = nullptr;

    Render::ResourceHandle m_gbuffer[kGBufferTargetCount];
    Render::ResourceHandle m_sceneColor;
    Render::ResourceHandle m_shadowAtlas;
    // The cached half: static geometry's depth, per cascade, under the fit that
    // cascade is currently holding. Only the tiles of refitted cascades are
    // rewritten, so this survives from frame to frame.
    Render::ResourceHandle m_shadowStaticAtlas;
    // Whichever of the two the lighting pass should sample this frame.
    Render::ResourceHandle m_shadowSampledAtlas;

    // The baked result, indexed by surface rather than by light: one chart per
    // static instance. Written once by the bake, read by the G-buffer pass.
    Render::ResourceHandle m_lightmapAtlas;
    // The whole scene from the sun, in one projection, for the bake to test
    // against. Not a cascade -- a lightmap texel anywhere can need any caster.
    Render::ResourceHandle m_lightmapSunDepth;
    bool m_lightmapBakePending = true;
    bool m_lightmapBaked = false;
    // Kept so a bake can be requested again when the sun moves, and so the UI
    // can say what the bake was made for.
    DirectX::SimpleMath::Vector3 m_bakedSunDirection;

    // The lighting pass: one fullscreen triangle, no vertex buffer, no depth.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_lightingPso;
    Descriptors::RootSignature m_lightingRootSignature;
    std::unique_ptr<Descriptors::DynamicDescriptorHeap> m_lightingSrvHeap;
    Descriptors::HeapBinder m_lightingHeapBinder;

    // DeviceResources owns the depth buffer but has no shader-visible heap to
    // put a view in, so the view lives here and is rebuilt on resize.
    std::unique_ptr<Descriptors::DescriptorAllocator> m_srvAllocator;
    Descriptors::DescriptorAllocation m_depthSrv;
    ID3D12Resource* m_depthSrvResource = nullptr;

    // --- cascaded shadow maps --------------------------------------------
    ShadowCascadeSetup m_shadowSetup;
    ShadowCascades m_shadowCascades;
    bool m_shadowsEnabled = true;
    // Subtracted from the comparison depth. Kept small on purpose: the normal
    // offset below is what actually clears acne, and a depth bias large enough
    // to do it alone detaches shadows from their casters.
    float m_shadowDepthBias = 0.0005f;
    // In cascade texels. Around one texel is the usual sweet spot.
    float m_shadowNormalBias = 1.5f;
    // UI only.
    bool m_hasDirectionalLight = false;

    // --- the sun ---------------------------------------------------------
    // Which entry of this frame's light array is the sun, and the direction it
    // is shining in once the override below has had its say. Both are settled
    // once per frame by ResolveSun; everything else -- the cascade fit, the
    // bake, the direction uploaded to the lighting pass -- reads these rather
    // than the light array, so there is one answer per frame by construction.
    static constexpr std::size_t kNoSunLight = static_cast<std::size_t>(-1);
    std::size_t m_sunLightIndex = kNoSunLight;
    DirectX::SimpleMath::Vector3 m_sunDirection = {0.0f, -1.0f, 0.0f};
    SunDirectionOverride m_sunOverride;
    // True when the override is standing in for a sun the scene does not have,
    // rather than redirecting one it does. The distinction is the lighting
    // pass's: a synthesised sun has no slot in the scene's light array and has
    // to be appended to the constants, where an overridden one is already
    // there and only its direction is rewritten.
    bool m_sunIsSynthesized = false;

    // Set false to refit and re-render every cascade every frame, which is
    // what this did before the cache existed. Kept because "turn it off and
    // look" is the only way to be sure a cache is not serving stale depth.
    bool m_cacheStaticShadows = true;
    // Counters for the UI, per frame and cumulative.
    int m_shadowCascadesRefitted = 0;
    std::uint64_t m_shadowStaticRenders = 0;
    std::uint64_t m_shadowFrames = 0;
    bool m_lastFrameHadMovingGeometry = false;

    // --- the display transform -------------------------------------------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_tonemapPso;
    Descriptors::RootSignature m_tonemapRootSignature;
    std::unique_ptr<Descriptors::DynamicDescriptorHeap> m_tonemapSrvHeap;
    Descriptors::HeapBinder m_tonemapHeapBinder;

    // --- the debug view --------------------------------------------------
    // Shares the tonemap's slot in the frame: whichever of the two is wanted
    // is the pass that writes the back buffer, so an intermediate is shown at
    // the back buffer's own resolution with nothing drawn over it.
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_debugViewPso;
    Descriptors::RootSignature m_debugViewRootSignature;
    std::unique_ptr<Descriptors::DynamicDescriptorHeap> m_debugViewSrvHeap;
    Descriptors::HeapBinder m_debugViewHeapBinder;
    DebugView m_debugView = DebugView::Scene;
    // The window onto a scalar channel. 0..1 shows the stored value as it is,
    // which is the right default and useless for an orthographic depth map.
    float m_debugRangeMin = 0.0f;
    float m_debugRangeMax = 1.0f;
    // Recovered from this frame's projection matrix, because linearising the
    // depth buffer needs them and the renderer is not told the camera's
    // planes -- see AddDebugViewPass.
    float m_debugNearPlane = 0.1f;
    float m_debugFarPlane = 1000.0f;
    // This frame's depth, kept so the debug pass can read it: the handle is an
    // import and so is only valid for the frame that produced it.
    Render::ResourceHandle m_frameDepth;

    // --- the sky ---------------------------------------------------------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_skyboxPso;
    Descriptors::RootSignature m_skyboxRootSignature;
    std::unique_ptr<Descriptors::DynamicDescriptorHeap> m_skyboxSrvHeap;
    Descriptors::HeapBinder m_skyboxHeapBinder;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_environment;
    Descriptors::DescriptorAllocation m_environmentSrv;
    std::string m_environmentPath;
    std::uint32_t m_environmentWidth = 0;
    std::uint32_t m_environmentHeight = 0;
    bool m_skyboxEnabled = true;
    float m_skyboxExposure = 1.0f;
    float m_skyboxYawDegrees = 0.0f;

    // UI only.
    std::string m_planDescription;
    std::size_t m_lastLightCount = 0;
    std::size_t m_droppedLights = 0;
    bool m_showPlan = false;
};

}  // namespace NeuralModelIntegrateTestbed
