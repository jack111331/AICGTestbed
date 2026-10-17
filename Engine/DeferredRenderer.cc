#include "pch.h"

#include "DeferredRenderer.hpp"

#include <cstring>

#include <algorithm>
#include <filesystem>

#include <stb_image.h>

#include "imgui.h"
#include <no_texture.h>

using namespace DirectX;

namespace NeuralModelIntegrateTestbed {

namespace {

// Everything that differs between one debug view and the next, apart from
// which resource it reads -- which has to be a switch, because two of them are
// chosen per frame and the handles are members.
struct DebugViewInfo {
    const char* name;
    DebugChannels channels;
};

constexpr DebugViewInfo kDebugViews[static_cast<std::size_t>(DebugView::Count)] = {
    {"Scene (tonemapped)", DebugChannels::Tonemapped},
    // The same target without the transform, which is what makes the two worth
    // having side by side: everything above 1.0 clips to white here, so this is
    // where an over-exposed scene reads as over-exposed rather than as a flat
    // region the tonemap has rolled off.
    {"Scene colour (no tonemap)", DebugChannels::Color},
    {"G-buffer: base colour", DebugChannels::Color},
    {"G-buffer: normal", DebugChannels::Normal},
    {"G-buffer: metallic", DebugChannels::Red},
    {"G-buffer: roughness", DebugChannels::Green},
    {"G-buffer: occlusion", DebugChannels::Blue},
    // The baked term rides in the material target's alpha, so this view is
    // also the check that the bake reached the G-buffer at all.
    {"G-buffer: baked sun visibility", DebugChannels::Alpha},
    {"G-buffer: emissive", DebugChannels::Color},
    {"Depth (linear, / far plane)", DebugChannels::Depth},
    {"Shadow atlas", DebugChannels::Red},
    {"Lightmap atlas", DebugChannels::Red},
    {"Lightmap sun depth", DebugChannels::Red},
};

}  // namespace

const char* DebugViewName(DebugView view) {
    const std::size_t index = static_cast<std::size_t>(view);
    return index < std::size(kDebugViews) ? kDebugViews[index].name : "unknown";
}

void DeferredRenderer::CreateDeviceDependentResources(ID3D12Device* device,
                                                      Render::RenderGraph& graph) {
    m_device = device;

    // The G-buffer is declared, not imported: the graph owns these, keeps them
    // across frames, and recreates them on resize. Declaring them as starting
    // in RENDER_TARGET means the first frame's geometry pass needs no barrier.
    for (std::size_t i = 0; i < kGBufferTargetCount; ++i) {
        const GBufferTarget target = static_cast<GBufferTarget>(i);
        Render::TextureDesc desc;
        desc.name = GBufferTargetName(target);
        desc.format = GBufferFormat(target);
        desc.allowRenderTarget = true;
        desc.allowShaderResource = true;
        desc.initialState = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // Cleared to zero every frame. Pixels the geometry pass does not cover
        // are never read -- the lighting pass discards them on depth -- so the
        // value only matters when looking at the targets in a capture.
        m_gbuffer[i] = graph.DeclareTexture(desc);
    }

    // The HDR target everything before the tonemap resolves into. Declared
    // alongside the G-buffer so the graph recreates it on resize too.
    {
        Render::TextureDesc desc;
        desc.name = "SceneColor";
        desc.format = kSceneColorFormat;
        desc.allowRenderTarget = true;
        desc.allowShaderResource = true;
        desc.initialState = D3D12_RESOURCE_STATE_RENDER_TARGET;
        m_sceneColor = graph.DeclareTexture(desc);
    }

    // The shadow atlas. Absolute size, not swapchain-relative: a cascade's
    // resolution has nothing to do with the window's, and resizing the window
    // should not quietly change shadow quality.
    //
    // One texture holding every cascade in a grid, rather than a texture array,
    // because the render graph creates single-slice views -- an atlas needs no
    // graph changes, one DSV and one SRV, and each cascade is placed with a
    // viewport instead of a slice index.
    {
        std::uint32_t x = 0, y = 0, size = 0, atlas = 0;
        ShadowCascadeAtlasRect(0, m_shadowSetup.cascadeCount, m_shadowSetup.resolution,
                               x, y, size, atlas);
        Render::TextureDesc desc;
        desc.name = "ShadowAtlas";
        desc.format = kShadowResourceFormat;
        desc.depthStencilFormat = kShadowDepthFormat;
        desc.shaderResourceFormat = kShadowShaderResourceFormat;
        desc.sizeMode = Render::SizeMode::Absolute;
        desc.width = atlas;
        desc.height = atlas;
        desc.allowDepthStencil = true;
        desc.allowShaderResource = true;
        desc.clearDepth = 1.0f;
        desc.initialState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        // Both atlases carry content between frames, which is the point of the
        // static one and harmless for the live one. It also keeps Compile()
        // from objecting when a frame samples an atlas no pass wrote, which is
        // exactly what happens in a scene where nothing moves.
        desc.persistentContent = true;
        m_shadowAtlas = graph.DeclareTexture(desc);

        desc.name = "ShadowStaticAtlas";
        m_shadowStaticAtlas = graph.DeclareTexture(desc);
    }

    // The lightmap, and the sun depth map the bake tests against. Both are
    // declared at a fixed size rather than relative to the swapchain: what they
    // cover is the scene, which has nothing to do with the window.
    {
        Render::TextureDesc desc;
        desc.name = "LightmapAtlas";
        desc.format = kLightmapFormat;
        desc.sizeMode = Render::SizeMode::Absolute;
        desc.width = kLightmapAtlasResolution;
        desc.height = kLightmapAtlasResolution;
        desc.allowRenderTarget = true;
        desc.allowShaderResource = true;
        desc.persistentContent = true;
        // Cleared to LIT. A texel no chart covers -- the gaps between charts,
        // and every chart of a scene that was never baked -- then reads as
        // "the sun reaches here", which is the harmless answer. Clearing to 0
        // would drop an unbaked scene into total darkness.
        desc.clearColor[0] = 1.0f;
        desc.clearColor[1] = 1.0f;
        desc.clearColor[2] = 1.0f;
        desc.clearColor[3] = 1.0f;
        desc.initialState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        m_lightmapAtlas = graph.DeclareTexture(desc);

        Render::TextureDesc depth;
        depth.name = "LightmapSunDepth";
        depth.format = kShadowResourceFormat;
        depth.depthStencilFormat = kShadowDepthFormat;
        depth.shaderResourceFormat = kShadowShaderResourceFormat;
        depth.sizeMode = Render::SizeMode::Absolute;
        depth.width = kLightmapSunDepthResolution;
        depth.height = kLightmapSunDepthResolution;
        depth.allowDepthStencil = true;
        depth.allowShaderResource = true;
        depth.clearDepth = 1.0f;
        depth.persistentContent = true;
        depth.initialState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        m_lightmapSunDepth = graph.DeclareTexture(depth);
    }

    m_srvAllocator = std::make_unique<Descriptors::DescriptorAllocator>(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16);

    // Five descriptors per draw and one draw per frame, so the page can be
    // small; the dynamic heap allocates another if it ever runs dry.
    m_lightingSrvHeap = std::make_unique<Descriptors::DynamicDescriptorHeap>(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64);
}

void DeferredRenderer::CreateLightingPipeline(ID3D12Device* device,
                                              DXGI_FORMAT backBufferFormat) {
    // Must stay in lockstep with DeferredLightingRootSignature in
    // shaders/no_texture.fx, which the shader carries its own copy of.
    //
    //   b3  DeferredLightingConstants (the inverse view-projection and lights)
    //   t5..t9  the four G-buffer targets plus depth
    //
    // No sampler: the lighting pass runs at the G-buffer's own resolution and
    // reads with Load(), so there is nothing to filter.
    CD3DX12_ROOT_PARAMETER1 rootParameters[2] = {};
    rootParameters[0].InitAsConstantBufferView(3, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                                               D3D12_SHADER_VISIBILITY_ALL);
    // Two ranges in one table: the G-buffer at t5..t9 and the shadow atlas at
    // t12. Not contiguous because t10 and t11 belong to the skybox and the
    // tonemap, and every entry point shares this file's register space.
    // RootSignature sums the ranges, so the table's descriptors are still
    // staged at contiguous offsets -- the atlas is offset kGBufferSrvCount.
    const CD3DX12_DESCRIPTOR_RANGE1 lightingRanges[2] = {
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                                  static_cast<UINT>(kGBufferSrvCount), 5, 0,
                                  D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE),
        CD3DX12_DESCRIPTOR_RANGE1(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12, 0,
                                  D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE),
    };
    rootParameters[1].InitAsDescriptorTable(2, lightingRanges,
                                            D3D12_SHADER_VISIBILITY_PIXEL);

    // Comparison sampler for the shadow atlas, matching the StaticSampler in
    // DeferredLightingRootSignature. Static because there is exactly one and it
    // never changes.
    const CD3DX12_STATIC_SAMPLER_DESC shadowSampler(
        2, D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f, 16,
        D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE, 0.0f,
        D3D12_FLOAT32_MAX, D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
    rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
    rsigDesc.pParameters = rootParameters;
    rsigDesc.NumStaticSamplers = 1;
    rsigDesc.pStaticSamplers = &shadowSampler;
    // No input assembler: the fullscreen triangle comes from SV_VertexID, so
    // ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT is deliberately absent here where the
    // geometry pass's signature needs it.
    rsigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
    m_lightingRootSignature.SetRootSignatureDesc(rsigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_lightingRootSignature.GetRootSignature().Get();
    psoDesc.VS = {g_deferred_lighting_vertex_shader_bytecode,
                  sizeof(g_deferred_lighting_vertex_shader_bytecode)};
    psoDesc.PS = {g_deferred_lighting_pixel_shader_bytecode,
                  sizeof(g_deferred_lighting_pixel_shader_bytecode)};
    // No input layout at all: three vertices generated from SV_VertexID.
    psoDesc.InputLayout = {nullptr, 0};
    psoDesc.BlendState = CommonStates::Opaque;
    // No depth: the pass covers the whole target and decides coverage itself,
    // from the depth it reads as a texture. Binding depth here would also be a
    // read-write hazard on the same resource.
    psoDesc.DepthStencilState = CommonStates::DepthNone;
    psoDesc.RasterizerState = CommonStates::CullNone;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.NumRenderTargets = 1;
    // The scene colour target, not the back buffer: the lighting resolve is
    // linear radiance and only PSTonemap turns it into display values.
    psoDesc.RTVFormats[0] = kSceneColorFormat;
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    psoDesc.SampleDesc.Count = 1;

    DX::ThrowIfFailed(
        device->CreateGraphicsPipelineState(&psoDesc, IID_GRAPHICS_PPV_ARGS(&m_lightingPso)));
    m_lightingPso->SetName(L"Deferred lighting");
}

void DeferredRenderer::CreateSkyboxPipeline(ID3D12Device* device,
                                            DXGI_FORMAT backBufferFormat) {
    // Must stay in lockstep with SkyboxRootSignature in no_texture.fx.
    //
    //   b4  SkyboxConstants
    //   t10 the environment map
    //   s1  a static sampler, wrapping in u
    //
    // The sampler is static rather than a table because there is exactly one of
    // it and it never changes; wrapping in u matters because longitude is
    // periodic and a clamp would smear the seam across the sky.
    CD3DX12_ROOT_PARAMETER1 rootParameters[2] = {};
    rootParameters[0].InitAsConstantBufferView(4, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                                               D3D12_SHADER_VISIBILITY_ALL);
    const CD3DX12_DESCRIPTOR_RANGE1 environmentRange(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 10, 0,
        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    rootParameters[1].InitAsDescriptorTable(1, &environmentRange,
                                            D3D12_SHADER_VISIBILITY_PIXEL);

    const CD3DX12_STATIC_SAMPLER_DESC sampler(
        1, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f, 16,
        D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE, 0.0f,
        D3D12_FLOAT32_MAX, D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
    rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
    rsigDesc.pParameters = rootParameters;
    rsigDesc.NumStaticSamplers = 1;
    rsigDesc.pStaticSamplers = &sampler;
    rsigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
    m_skyboxRootSignature.SetRootSignatureDesc(rsigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_skyboxRootSignature.GetRootSignature().Get();
    psoDesc.VS = {g_skybox_vertex_shader_bytecode, sizeof(g_skybox_vertex_shader_bytecode)};
    psoDesc.PS = {g_skybox_pixel_shader_bytecode, sizeof(g_skybox_pixel_shader_bytecode)};
    psoDesc.InputLayout = {nullptr, 0};
    psoDesc.BlendState = CommonStates::Opaque;
    // No depth at all: this pass runs before anything has been resolved into
    // the back buffer and covers every pixel, so there is nothing to test
    // against and nothing to preserve.
    psoDesc.DepthStencilState = CommonStates::DepthNone;
    psoDesc.RasterizerState = CommonStates::CullNone;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = kSceneColorFormat;
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    psoDesc.SampleDesc.Count = 1;

    DX::ThrowIfFailed(
        device->CreateGraphicsPipelineState(&psoDesc, IID_GRAPHICS_PPV_ARGS(&m_skyboxPso)));
    m_skyboxPso->SetName(L"Skybox");

    m_skyboxSrvHeap = std::make_unique<Descriptors::DynamicDescriptorHeap>(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16);
}

void DeferredRenderer::SetSceneScale(float sceneRadius) {
    if (sceneRadius <= 0.0f) {
        return;
    }
    // Twice the radius covers a camera anywhere in the scene looking across
    // it, without spending cascades on the empty space beyond.
    m_shadowSetup.shadowDistance = sceneRadius * 2.0f;
}

void DeferredRenderer::ResolveSun(const FrameInputs& inputs) {
    m_sunLightIndex = kNoSunLight;
    m_hasDirectionalLight = false;
    if (inputs.lights == nullptr) {
        return;
    }

    // The first directional light wins. Cascades are for a light with no
    // position, and nothing in these scenes has two suns.
    for (std::size_t i = 0; i < inputs.lightCount; ++i) {
        if (inputs.lights[i].type == static_cast<int>(LightType::Directional)) {
            m_sunLightIndex = i;
            break;
        }
    }
    if (m_sunLightIndex == kNoSunLight) {
        return;
    }
    m_hasDirectionalLight = true;

    const DirectX::XMFLOAT3& authored = inputs.lights[m_sunLightIndex].direction;
    m_sunDirection =
        m_sunOverride.enabled
            ? SunDirectionFromEuler(m_sunOverride.pitchDegrees, m_sunOverride.yawDegrees)
            : DirectX::SimpleMath::Vector3(authored.x, authored.y, authored.z);

    // A bake fixes the light into the data, so a sun that moves invalidates it
    // -- which is now a thing the UI can do, not only a thing an animation can.
    if ((m_sunDirection - m_bakedSunDirection).LengthSquared() > 1e-10f) {
        m_bakedSunDirection = m_sunDirection;
        m_lightmapBakePending = true;
        m_lightmapBaked = false;
    }
}

bool DeferredRenderer::UpdateShadowCascades(const FrameInputs& inputs) {
    m_shadowCascadesRefitted = 0;
    // Kept rather than cleared: it is what the new fit is held against, and
    // dropping it would refit every cascade every frame.
    const ShadowCascades previous = m_shadowCascades;
    m_shadowCascades = ShadowCascades{};
    // ResolveSun has already run and found the sun, or not.
    if (!m_shadowsEnabled || !m_hasDirectionalLight || !inputs.hasSceneBounds) {
        return false;
    }

    m_shadowCascades = ComputeShadowCascades(
        inputs.view, inputs.projection, m_sunDirection,
        inputs.sceneMin, inputs.sceneMax, m_shadowSetup,
        m_cacheStaticShadows ? &previous : nullptr);

    for (int i = 0; i < m_shadowCascades.count; ++i) {
        if (m_shadowCascades.cascades[i].refitted) {
            ++m_shadowCascadesRefitted;
        }
    }
    return m_shadowCascades.count > 0;
}

void DeferredRenderer::CreateTonemapPipeline(ID3D12Device* device,
                                             DXGI_FORMAT backBufferFormat) {
    // Must stay in lockstep with TonemapRootSignature in no_texture.fx. One
    // SRV and nothing else: no constants, because the transform has no
    // parameters yet. An exposure control would belong here.
    CD3DX12_ROOT_PARAMETER1 rootParameters[1] = {};
    const CD3DX12_DESCRIPTOR_RANGE1 sceneRange(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 11, 0,
        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    rootParameters[0].InitAsDescriptorTable(1, &sceneRange,
                                            D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
    rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
    rsigDesc.pParameters = rootParameters;
    rsigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
    m_tonemapRootSignature.SetRootSignatureDesc(rsigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_tonemapRootSignature.GetRootSignature().Get();
    psoDesc.VS = {g_tonemap_vertex_shader_bytecode,
                  sizeof(g_tonemap_vertex_shader_bytecode)};
    psoDesc.PS = {g_tonemap_pixel_shader_bytecode,
                  sizeof(g_tonemap_pixel_shader_bytecode)};
    psoDesc.InputLayout = {nullptr, 0};
    psoDesc.BlendState = CommonStates::Opaque;
    psoDesc.DepthStencilState = CommonStates::DepthNone;
    psoDesc.RasterizerState = CommonStates::CullNone;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.NumRenderTargets = 1;
    // This one really does write the back buffer -- it is the last thing that
    // touches the frame before the UI.
    psoDesc.RTVFormats[0] = backBufferFormat;
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    psoDesc.SampleDesc.Count = 1;

    DX::ThrowIfFailed(
        device->CreateGraphicsPipelineState(&psoDesc, IID_GRAPHICS_PPV_ARGS(&m_tonemapPso)));
    m_tonemapPso->SetName(L"Tonemap");

    m_tonemapSrvHeap = std::make_unique<Descriptors::DynamicDescriptorHeap>(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16);
}

void DeferredRenderer::CreateDebugViewPipeline(ID3D12Device* device,
                                              DXGI_FORMAT backBufferFormat) {
    // Must stay in lockstep with DebugViewRootSignature in no_texture.fx.
    //
    //   b5  DebugViewConstants (which channels, the two sizes, the window)
    //   t14 whichever resource is being looked at
    //   s5  a static linear sampler, clamped
    //
    // Linear rather than point because a 4096-wide atlas has to be fitted into
    // the window; a screen-sized source lands on texel centres and so is
    // unaffected by the filter.
    CD3DX12_ROOT_PARAMETER1 rootParameters[2] = {};
    rootParameters[0].InitAsConstantBufferView(5, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                                               D3D12_SHADER_VISIBILITY_ALL);
    const CD3DX12_DESCRIPTOR_RANGE1 sourceRange(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 14, 0,
        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    rootParameters[1].InitAsDescriptorTable(1, &sourceRange,
                                            D3D12_SHADER_VISIBILITY_PIXEL);

    const CD3DX12_STATIC_SAMPLER_DESC sampler(
        5, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f, 16,
        D3D12_COMPARISON_FUNC_LESS_EQUAL, D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE, 0.0f,
        D3D12_FLOAT32_MAX, D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
    rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
    rsigDesc.pParameters = rootParameters;
    rsigDesc.NumStaticSamplers = 1;
    rsigDesc.pStaticSamplers = &sampler;
    rsigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                     D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
    m_debugViewRootSignature.SetRootSignatureDesc(rsigDesc,
                                                  D3D_ROOT_SIGNATURE_VERSION_1_1);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_debugViewRootSignature.GetRootSignature().Get();
    psoDesc.VS = {g_debug_view_vertex_shader_bytecode,
                  sizeof(g_debug_view_vertex_shader_bytecode)};
    psoDesc.PS = {g_debug_view_pixel_shader_bytecode,
                  sizeof(g_debug_view_pixel_shader_bytecode)};
    psoDesc.InputLayout = {nullptr, 0};
    psoDesc.BlendState = CommonStates::Opaque;
    psoDesc.DepthStencilState = CommonStates::DepthNone;
    psoDesc.RasterizerState = CommonStates::CullNone;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.NumRenderTargets = 1;
    // The back buffer: this stands in for the tonemap, not alongside it.
    psoDesc.RTVFormats[0] = backBufferFormat;
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    psoDesc.SampleDesc.Count = 1;

    DX::ThrowIfFailed(device->CreateGraphicsPipelineState(
        &psoDesc, IID_GRAPHICS_PPV_ARGS(&m_debugViewPso)));
    m_debugViewPso->SetName(L"Debug view");

    m_debugViewSrvHeap = std::make_unique<Descriptors::DynamicDescriptorHeap>(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16);
}

void DeferredRenderer::AddDebugViewPass(Render::RenderGraph& graph,
                                        Render::ResourceHandle backBuffer) {
    const std::size_t index = static_cast<std::size_t>(m_debugView);
    DebugChannels channels = kDebugViews[index].channels;

    // Which resource, and how big it is. The size cannot come from the graph --
    // PassResources reports the pass's target size, not the source's -- so each
    // case says it, which is also what keeps an atlas from being stretched to
    // the window's aspect.
    Render::ResourceHandle source;
    std::uint32_t sourceWidth = 0;   // 0 means "the same size as the target"
    std::uint32_t sourceHeight = 0;
    switch (m_debugView) {
        case DebugView::SceneColor:
            source = m_sceneColor;
            break;
        case DebugView::BaseColor:
            source = m_gbuffer[static_cast<std::size_t>(GBufferTarget::BaseColor)];
            break;
        case DebugView::Normal:
            source = m_gbuffer[static_cast<std::size_t>(GBufferTarget::Normal)];
            break;
        case DebugView::Metallic:
        case DebugView::Roughness:
        case DebugView::Occlusion:
        case DebugView::BakedSunVisibility:
            source = m_gbuffer[static_cast<std::size_t>(GBufferTarget::Material)];
            break;
        case DebugView::Emissive:
            source = m_gbuffer[static_cast<std::size_t>(GBufferTarget::Emissive)];
            break;
        case DebugView::Depth:
            source = m_frameDepth;
            break;
        case DebugView::ShadowAtlas: {
            // Whichever half the lighting pass sampled, so this shows the map
            // the frame was actually lit by rather than one of its inputs.
            source = m_shadowSampledAtlas.IsValid() ? m_shadowSampledAtlas
                                                    : m_shadowStaticAtlas;
            std::uint32_t x = 0, y = 0, size = 0, atlas = 0;
            ShadowCascadeAtlasRect(0, m_shadowSetup.cascadeCount,
                                   m_shadowSetup.resolution, x, y, size, atlas);
            sourceWidth = atlas;
            sourceHeight = atlas;
            break;
        }
        case DebugView::LightmapAtlas:
            source = m_lightmapAtlas;
            sourceWidth = kLightmapAtlasResolution;
            sourceHeight = kLightmapAtlasResolution;
            break;
        case DebugView::LightmapSunDepth:
            source = m_lightmapSunDepth;
            sourceWidth = kLightmapSunDepthResolution;
            sourceHeight = kLightmapSunDepthResolution;
            break;
        case DebugView::Scene:
        case DebugView::Count:
            break;
    }

    // Nothing selects Scene or Count here -- AddTonemapPass filtered the first
    // and the combo never offers the second -- but a pass declaring an invalid
    // read fails Compile() and takes the whole frame with it, so the frame is
    // worth more than the view.
    if (!source.IsValid()) {
        source = m_sceneColor;
        channels = DebugChannels::Tonemapped;
    }

    Render::RenderPassDesc pass;
    pass.name = "DebugView";
    pass.reads.push_back({source, Render::Access::PixelShaderResource});
    pass.writes.push_back({backBuffer, Render::Access::RenderTarget});
    pass.execute = [this, source, channels, sourceWidth, sourceHeight](
                       ID3D12GraphicsCommandList* commandList,
                       const Render::PassResources& resources) {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        DX::ThrowIfFailed(commandList->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));

        DebugViewConstants constants = {};
        constants.channels = static_cast<int>(channels);
        constants.targetWidth = static_cast<float>(resources.Width());
        constants.targetHeight = static_cast<float>(resources.Height());
        constants.sourceWidth = sourceWidth != 0 ? static_cast<float>(sourceWidth)
                                                 : constants.targetWidth;
        constants.sourceHeight = sourceHeight != 0 ? static_cast<float>(sourceHeight)
                                                   : constants.targetHeight;
        constants.rangeMin = m_debugRangeMin;
        constants.rangeMax = m_debugRangeMax;
        constants.nearPlane = m_debugNearPlane;
        constants.farPlane = m_debugFarPlane;

        SharedGraphicsResource constantBuffer =
            GraphicsMemory::Get(device.Get()).AllocateConstant(constants);

        m_debugViewHeapBinder.Reset(commandList);
        m_debugViewSrvHeap->Reset();
        commandList->SetGraphicsRootSignature(
            m_debugViewRootSignature.GetRootSignature().Get());
        m_debugViewSrvHeap->ParseRootSignature(m_debugViewRootSignature);
        commandList->SetPipelineState(m_debugViewPso.Get());
        commandList->SetGraphicsRootConstantBufferView(0, constantBuffer.GpuAddress());
        m_debugViewSrvHeap->StageDescriptors(1, 0, 1, resources.Srv(source));
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 0, nullptr);
        commandList->IASetIndexBuffer(nullptr);
        m_debugViewSrvHeap->CommitStagedDescriptorsForDraw(commandList,
                                                           m_debugViewHeapBinder);
        commandList->DrawInstanced(3, 1, 0, 0);
    };
    graph.AddPass(std::move(pass));
}

void DeferredRenderer::AddTonemapPass(Render::RenderGraph& graph,
                                      Render::ResourceHandle backBuffer) {
    // One slot in the frame, two things that can fill it. Dispatched here rather
    // than at the call site so the application does not have to know a debug
    // view exists -- and so whatever it adds afterwards, the screen overlays
    // and the UI, lands on top of either one.
    if (m_debugView != DebugView::Scene && m_debugViewPso != nullptr) {
        AddDebugViewPass(graph, backBuffer);
        return;
    }

    Render::RenderPassDesc pass;
    pass.name = "Tonemap";
    pass.reads.push_back({m_sceneColor, Render::Access::PixelShaderResource});
    pass.writes.push_back({backBuffer, Render::Access::RenderTarget});
    // No clear: this covers every pixel of the back buffer by construction.
    pass.execute = [this](ID3D12GraphicsCommandList* commandList,
                          const Render::PassResources& resources) {
        m_tonemapHeapBinder.Reset(commandList);
        m_tonemapSrvHeap->Reset();
        commandList->SetGraphicsRootSignature(
            m_tonemapRootSignature.GetRootSignature().Get());
        m_tonemapSrvHeap->ParseRootSignature(m_tonemapRootSignature);
        commandList->SetPipelineState(m_tonemapPso.Get());
        m_tonemapSrvHeap->StageDescriptors(0, 0, 1, resources.Srv(m_sceneColor));
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetVertexBuffers(0, 0, nullptr);
        commandList->IASetIndexBuffer(nullptr);
        m_tonemapSrvHeap->CommitStagedDescriptorsForDraw(commandList,
                                                         m_tonemapHeapBinder);
        commandList->DrawInstanced(3, 1, 0, 0);
    };
    graph.AddPass(std::move(pass));
}

bool DeferredRenderer::LoadEnvironment(DX::DeviceResources& deviceResources,
                                       const std::string& path, std::string* error) {
    ID3D12Device* device = deviceResources.GetD3DDevice();
    if (device == nullptr) {
        if (error != nullptr) *error = "no device";
        return false;
    }

    const std::filesystem::path file(path);
    std::error_code ignored;
    if (!std::filesystem::exists(file, ignored)) {
        if (error != nullptr) *error = "no such file: " + path;
        return false;
    }

    std::string extension = file.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (extension == ".exr") {
        // Said plainly, because the failure is otherwise a black sky with no
        // explanation. See LoadEnvironment's declaration.
        if (error != nullptr) {
            *error = "OpenEXR is not supported; convert " + file.filename().string() +
                     " to .hdr or .dds first";
        }
        return false;
    }

    ResourceUploadBatch upload(device);
    upload.Begin();

    Microsoft::WRL::ComPtr<ID3D12Resource> texture;
    if (extension == ".dds") {
        // sRGB is wrong for an environment map: the values are radiance, and a
        // released HDRI is stored linear.
        const HRESULT hr = CreateDDSTextureFromFileEx(
            device, upload, file.wstring().c_str(), 0, D3D12_RESOURCE_FLAG_NONE,
            DDS_LOADER_DEFAULT, texture.GetAddressOf());
        if (FAILED(hr)) {
            if (error != nullptr) {
                char buffer[64] = {};
                std::snprintf(buffer, sizeof(buffer), "0x%08X", static_cast<unsigned>(hr));
                *error = "CreateDDSTextureFromFileEx failed " + std::string(buffer);
            }
            return false;
        }
        // No auto-exposure on this path: the pixels go straight from the file
        // to the GPU, often block compressed, so there is nothing on the CPU to
        // take a median of. The slider is the answer.
        std::fprintf(stderr, "skybox: .dds loaded; exposure left at %.3f\n",
                     m_skyboxExposure);
    } else {
        // stb_image's float path reads Radiance .hdr (RGBE) and will also take
        // an ordinary 8-bit image, which is useful for testing even though it
        // is not high dynamic range.
        int width = 0, height = 0, channels = 0;
        float* pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
        if (pixels == nullptr) {
            if (error != nullptr) {
                const char* reason = stbi_failure_reason();
                *error = std::string("stb_image could not read it: ") +
                         (reason != nullptr ? reason : "unknown");
            }
            return false;
        }

        // Half rather than full float: an environment map is read, never
        // accumulated into, and 16 bits of mantissa is well beyond what a sky
        // needs. Halves the 32 MB a 2048x1024 RGBA32F image would cost.
        std::vector<std::uint16_t> halves(static_cast<std::size_t>(width) * height * 4);
        std::vector<float> luminance;
        luminance.reserve(static_cast<std::size_t>(width) * height);
        for (std::size_t i = 0; i < halves.size(); ++i) {
            halves[i] = DirectX::PackedVector::XMConvertFloatToHalf(pixels[i]);
        }
        for (std::size_t i = 0; i + 2 < halves.size(); i += 4) {
            luminance.push_back(0.2126f * pixels[i] + 0.7152f * pixels[i + 1] +
                                0.0722f * pixels[i + 2]);
        }
        stbi_image_free(pixels);

        // Exposure from the image rather than a guess. An HDRI's absolute scale
        // is a free choice of whoever produced it: this one's median luminance
        // is 0.187, which needs no correction, while one authored in absolute
        // physical units would be thousands and render as a white rectangle.
        //
        // The median is mapped to 0.18, linear mid-grey. The median rather than
        // the mean because the sun is in these images -- this one peaks at 132
        // against a median of 0.187, so a mean-based key would be dragged
        // several stops off by a handful of pixels.
        //
        // This does NOT fix the dynamic range: 22% of this sky is still above
        // 1.0 and clips, because nothing tonemaps yet. It only ensures the sky
        // lands in a usable place to start from.
        if (!luminance.empty()) {
            const std::size_t middle = luminance.size() / 2;
            std::nth_element(luminance.begin(), luminance.begin() + middle,
                             luminance.end());
            const float median = luminance[middle];
            if (median > 1e-6f) {
                m_skyboxExposure = std::min(1000.0f, std::max(0.001f, 0.18f / median));
            }
            std::fprintf(stderr, "skybox: median luminance %.4f -> exposure %.3f\n",
                         median, m_skyboxExposure);
        }

        const auto desc = CD3DX12_RESOURCE_DESC::Tex2D(
            DXGI_FORMAT_R16G16B16A16_FLOAT, static_cast<UINT64>(width),
            static_cast<UINT>(height), 1, 0);
        const CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
        DX::ThrowIfFailed(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_GRAPHICS_PPV_ARGS(texture.GetAddressOf())));

        D3D12_SUBRESOURCE_DATA initial = {};
        initial.pData = halves.data();
        initial.RowPitch = static_cast<LONG_PTR>(width) * 4 * sizeof(std::uint16_t);
        initial.SlicePitch = initial.RowPitch * height;
        upload.Upload(texture.Get(), 0, &initial, 1);
        upload.Transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        // Mips matter here: a wide field of view minifies the sky heavily
        // towards the horizon, and without them it shimmers as the camera turns.
        if (upload.IsSupportedForGenerateMips(DXGI_FORMAT_R16G16B16A16_FLOAT)) {
            upload.GenerateMips(texture.Get());
        }
    }

    auto finish = upload.End(deviceResources.GetCommandQueue());
    finish.wait();

    const D3D12_RESOURCE_DESC created = texture->GetDesc();
    if (m_environmentSrv.IsNull()) {
        m_environmentSrv = m_srvAllocator->Allocate(1);
    }
    CreateShaderResourceView(device, texture.Get(), m_environmentSrv.GetDescriptorHandle());

    m_environment = texture;
    m_environmentPath = path;
    m_environmentWidth = static_cast<std::uint32_t>(created.Width);
    m_environmentHeight = created.Height;
    SetDebugObjectName(m_environment.Get(), L"Environment");

    // 2:1 is what an equirectangular map is; anything else will still render,
    // just stretched, and saying so beats a silently squashed sky.
    if (m_environmentWidth != m_environmentHeight * 2) {
        std::fprintf(stderr,
                     "skybox: %s is %ux%u, which is not the 2:1 an equirectangular "
                     "map should be\n",
                     file.filename().string().c_str(), m_environmentWidth,
                     m_environmentHeight);
    }
    std::fprintf(stderr, "skybox: %s, %ux%u\n", file.filename().string().c_str(),
                 m_environmentWidth, m_environmentHeight);
    std::fflush(stderr);
    return true;
}

void DeferredRenderer::CreateWindowSizeDependentResources(
    DX::DeviceResources& deviceResources) {
    ID3D12Device* device = deviceResources.GetD3DDevice();
    if (device == nullptr) {
        return;
    }
    if (m_lightingPso == nullptr) {
        CreateLightingPipeline(device, deviceResources.GetBackBufferFormat());
    }
    if (m_skyboxPso == nullptr) {
        CreateSkyboxPipeline(device, deviceResources.GetBackBufferFormat());
    }
    if (m_tonemapPso == nullptr) {
        CreateTonemapPipeline(device, deviceResources.GetBackBufferFormat());
    }
    if (m_debugViewPso == nullptr) {
        CreateDebugViewPipeline(device, deviceResources.GetBackBufferFormat());
    }

    // The depth buffer is a new resource after every resize, so the view has to
    // be rewritten -- a descriptor is a snapshot of the resource, not a handle
    // to it. Reusing the allocation rather than freeing it keeps the handle
    // stable, which matters because nothing else is told it changed.
    ID3D12Resource* depth = deviceResources.GetDepthStencil();
    if (depth == nullptr) {
        return;
    }
    if (m_depthSrv.IsNull()) {
        m_depthSrv = m_srvAllocator->Allocate(1);
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    // The typed single-channel read of the typeless depth resource. This pairing
    // is why DeviceResources creates the depth buffer as R32_TYPELESS rather
    // than D32_FLOAT -- see DeviceResources::TypelessDepthFormat.
    srvDesc.Format = deviceResources.GetDepthShaderResourceFormat();
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(depth, &srvDesc, m_depthSrv.GetDescriptorHandle());
    m_depthSrvResource = depth;
}

void DeferredRenderer::OnDeviceLost() {
    m_lightingPso.Reset();
    m_lightingSrvHeap.reset();
    m_skyboxPso.Reset();
    m_skyboxSrvHeap.reset();
    m_tonemapPso.Reset();
    m_tonemapSrvHeap.reset();
    m_debugViewPso.Reset();
    m_debugViewSrvHeap.reset();
    m_shadowCascades = ShadowCascades{};
    m_environment.Reset();
    m_environmentSrv = Descriptors::DescriptorAllocation();
    m_environmentPath.clear();
    m_depthSrv = Descriptors::DescriptorAllocation();
    m_depthSrvResource = nullptr;
    m_srvAllocator.reset();
    m_device = nullptr;
}

DeferredRenderer::FrameTargets DeferredRenderer::ImportFrameTargets(
    Render::RenderGraph& graph, DX::DeviceResources& deviceResources,
    const float clearColor[4]) {
    const RECT size = deviceResources.GetOutputSize();

    Render::ImportedTextureDesc color;
    color.name = "BackBuffer";
    color.resource = deviceResources.GetRenderTarget();
    // DeviceResources::Prepare has already transitioned the back buffer out of
    // PRESENT by the time the graph runs, so the graph is told RENDER_TARGET and
    // asked to hand it back the same way; Present() does the final transition.
    color.currentState = D3D12_RESOURCE_STATE_RENDER_TARGET;
    color.finalState = D3D12_RESOURCE_STATE_RENDER_TARGET;
    color.rtv = deviceResources.GetRenderTargetView();
    color.width = static_cast<std::uint32_t>(size.right);
    color.height = static_cast<std::uint32_t>(size.bottom);
    std::memcpy(color.clearColor, clearColor, sizeof(color.clearColor));

    Render::ImportedTextureDesc depth;
    depth.name = "SceneDepth";
    depth.resource = deviceResources.GetDepthStencil();
    depth.currentState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    // Left writable, because the next frame's geometry pass expects to find it
    // that way and because nothing outside the graph touches it.
    depth.finalState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    depth.dsv = deviceResources.GetDepthStencilView();
    depth.srv = m_depthSrv.IsNull() ? D3D12_CPU_DESCRIPTOR_HANDLE{}
                                    : m_depthSrv.GetDescriptorHandle();
    depth.width = static_cast<std::uint32_t>(size.right);
    depth.height = static_cast<std::uint32_t>(size.bottom);
    depth.clearDepth = 1.0f;

    FrameTargets targets;
    targets.color = graph.ImportTexture(color);
    targets.depth = graph.ImportTexture(depth);
    return targets;
}

void DeferredRenderer::AddPasses(Render::RenderGraph& graph, const FrameTargets& targets,
                                 const FrameInputs& inputs,
                                 GeometryRecorder recordGeometry,
                                 ShadowRecorder recordShadows,
                                 LightmapBakeRecorder recordLightmapBake) {
    m_lastLightCount = inputs.lightCount;
    // Kept for the debug view, which is added later in the frame than this and
    // cannot be handed the targets: an imported handle is only valid for the
    // frame that produced it.
    m_frameDepth = targets.depth;
    // The camera's planes, recovered from the projection, because linearising
    // the depth buffer needs them and nothing tells the renderer what they are
    // -- the camera belongs to the application. The right-handed projection
    // SimpleMath builds carries them in two entries:
    //
    //   m22 = f/(n-f)   m32 = n*f/(n-f)   =>   n = m32/m22, f = m32/(m22+1)
    //
    // Both are negative for n < f, which is what the guard checks: an
    // orthographic or otherwise unexpected matrix leaves the last good pair
    // alone rather than producing a depth view full of NaN.
    {
        const float a = inputs.projection.m[2][2];
        const float b = inputs.projection.m[3][2];
        if (a < -1.0f && b < 0.0f && (a + 1.0f) < -1e-6f) {
            m_debugNearPlane = b / a;
            m_debugFarPlane = b / (a + 1.0f);
        }
    }

    // Before everything, including the cascades: the sun's direction decides
    // the cascade fit, the bake and what the lighting pass is told, so it is
    // settled once here.
    ResolveSun(inputs);

    // Before the G-buffer: the cascades are fitted to this frame's camera, and
    // the lighting pass reads the result out of the same constants it reads the
    // lights from.
    const bool castShadows = UpdateShadowCascades(inputs) && recordShadows != nullptr;
    // The bake, once. It is deliberately before everything else in the frame:
    // the G-buffer pass reads the atlas, so it has to exist by then, and the
    // sun depth map it needs is used by nothing else.
    if (m_lightmapBakePending && m_hasDirectionalLight && recordShadows != nullptr &&
        recordLightmapBake != nullptr && inputs.hasSceneBounds) {
        DirectX::SimpleMath::Matrix sunViewProjection;
        float sunTexelWorldSize = 0.0f;
        if (SceneSunViewProjection(m_bakedSunDirection, inputs.sceneMin,
                                   inputs.sceneMax, kLightmapSunDepthResolution,
                                   sunViewProjection, sunTexelWorldSize)) {
            {
                Render::RenderPassDesc pass;
                pass.name = "LightmapSunDepth";
                pass.writes.push_back(
                    {m_lightmapSunDepth, Render::Access::DepthWrite});
                pass.clears.push_back(m_lightmapSunDepth);
                pass.execute = [this, recorder = recordShadows, sunViewProjection](
                                   ID3D12GraphicsCommandList* commandList,
                                   const Render::PassResources&) {
                    // The whole map, not a tile: there are no cascades here.
                    D3D12_VIEWPORT viewport = {};
                    viewport.Width = static_cast<float>(kLightmapSunDepthResolution);
                    viewport.Height = static_cast<float>(kLightmapSunDepthResolution);
                    viewport.MinDepth = D3D12_MIN_DEPTH;
                    viewport.MaxDepth = D3D12_MAX_DEPTH;
                    const D3D12_RECT scissor = {
                        0, 0, static_cast<LONG>(kLightmapSunDepthResolution),
                        static_cast<LONG>(kLightmapSunDepthResolution)};
                    commandList->RSSetViewports(1, &viewport);
                    commandList->RSSetScissorRects(1, &scissor);
                    // Static casters only. A mover's depth baked in would
                    // leave its shadow painted on the floor after it left.
                    recorder(commandList, sunViewProjection, ShadowGeometry::Static);
                };
                graph.AddPass(std::move(pass));
            }
            {
                Render::RenderPassDesc pass;
                pass.name = "LightmapBake";
                pass.reads.push_back(
                    {m_lightmapSunDepth, Render::Access::PixelShaderResource});
                pass.writes.push_back({m_lightmapAtlas, Render::Access::RenderTarget});
                pass.clears.push_back(m_lightmapAtlas);
                pass.execute = [this, recorder = std::move(recordLightmapBake),
                                sunViewProjection, sunTexelWorldSize](
                                   ID3D12GraphicsCommandList* commandList,
                                   const Render::PassResources& resources) {
                    D3D12_VIEWPORT viewport = {};
                    viewport.Width = static_cast<float>(kLightmapAtlasResolution);
                    viewport.Height = static_cast<float>(kLightmapAtlasResolution);
                    viewport.MinDepth = D3D12_MIN_DEPTH;
                    viewport.MaxDepth = D3D12_MAX_DEPTH;
                    const D3D12_RECT scissor = {
                        0, 0, static_cast<LONG>(kLightmapAtlasResolution),
                        static_cast<LONG>(kLightmapAtlasResolution)};
                    commandList->RSSetViewports(1, &viewport);
                    commandList->RSSetScissorRects(1, &scissor);
                    recorder(commandList, sunViewProjection, sunTexelWorldSize,
                             resources.Srv(m_lightmapSunDepth));
                };
                graph.AddPass(std::move(pass));
            }
            m_lightmapBakePending = false;
            m_lightmapBaked = true;
        }
    }

    if (castShadows) {
        ++m_shadowFrames;

        // Three passes rather than one, because the frame's shadow map is now
        // assembled from a cached half and a live half:
        //
        //   ShadowStatic     static depth, only for cascades refitted this
        //                    frame -- usually none
        //   ShadowComposite  the cached tiles copied into the live atlas
        //   ShadowDynamic    the handful of moving primitives, on top
        //
        // The composite is what keeps the result bit-identical to drawing
        // everything in one pass: depth with a LESS test is a per-texel
        // minimum, and a minimum does not care which order its terms arrive
        // in. Sampling two maps and taking the lesser of two PCF results would
        // NOT be identical, because filtering the minimum is not the minimum
        // of the filtered.
        const bool moving = inputs.hasMovingGeometry;

        // Tiles to redraw, decided before any pass runs so each lambda agrees.
        bool refitted[kMaxShadowCascades] = {};
        for (int i = 0; i < m_shadowCascades.count; ++i) {
            refitted[i] = m_shadowCascades.cascades[i].refitted;
        }

        const auto tileViewport = [this](int cascade, D3D12_VIEWPORT& viewport,
                                         D3D12_RECT& scissor) {
            std::uint32_t x = 0, y = 0, size = 0, atlas = 0;
            ShadowCascadeAtlasRect(cascade, m_shadowCascades.count,
                                   m_shadowSetup.resolution, x, y, size, atlas);
            viewport = {};
            viewport.TopLeftX = static_cast<float>(x);
            viewport.TopLeftY = static_cast<float>(y);
            viewport.Width = static_cast<float>(size);
            viewport.Height = static_cast<float>(size);
            viewport.MinDepth = D3D12_MIN_DEPTH;
            viewport.MaxDepth = D3D12_MAX_DEPTH;
            // The scissor matters as much as the viewport: without it a
            // triangle crossing a tile edge writes into the neighbouring
            // cascade's depths.
            scissor = {static_cast<LONG>(x), static_cast<LONG>(y),
                       static_cast<LONG>(x + size), static_cast<LONG>(y + size)};
        };

        if (m_shadowCascades.AnyRefitted()) {
            Render::RenderPassDesc pass;
            pass.name = "ShadowStatic";
            pass.writes.push_back({m_shadowStaticAtlas, Render::Access::DepthWrite});
            // Deliberately NOT pass.clears: that clears the whole resource,
            // which would wipe the tiles of cascades that are still holding
            // their fit. Each dirty tile is cleared with a rect instead.
            pass.execute = [this, recorder = recordShadows, refitted, tileViewport](
                               ID3D12GraphicsCommandList* commandList,
                               const Render::PassResources& resources) {
                const D3D12_CPU_DESCRIPTOR_HANDLE dsv =
                    resources.Dsv(m_shadowStaticAtlas);
                for (int cascade = 0; cascade < m_shadowCascades.count; ++cascade) {
                    if (!refitted[cascade]) {
                        continue;
                    }
                    D3D12_VIEWPORT viewport = {};
                    D3D12_RECT scissor = {};
                    tileViewport(cascade, viewport, scissor);
                    commandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH,
                                                       1.0f, 0, 1, &scissor);
                    commandList->RSSetViewports(1, &viewport);
                    commandList->RSSetScissorRects(1, &scissor);
                    recorder(commandList,
                             m_shadowCascades.cascades[cascade].viewProjection,
                             ShadowGeometry::Static);
                    ++m_shadowStaticRenders;
                }
            };
            graph.AddPass(std::move(pass));
        }

        if (moving) {
            Render::RenderPassDesc copy;
            copy.name = "ShadowComposite";
            copy.reads.push_back({m_shadowStaticAtlas, Render::Access::CopySource});
            copy.writes.push_back({m_shadowAtlas, Render::Access::CopyDest});
            copy.bindRenderTargets = false;
            copy.execute = [this](ID3D12GraphicsCommandList* commandList,
                                  const Render::PassResources& resources) {
                ID3D12Resource* source = resources.Resource(m_shadowStaticAtlas);
                ID3D12Resource* destination = resources.Resource(m_shadowAtlas);
                if (source == nullptr || destination == nullptr) {
                    return;
                }
                // Whole atlas in one call: the cascades tile it completely, so
                // copying per tile would be the same bytes in more calls.
                const D3D12_TEXTURE_COPY_LOCATION from{
                    source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
                const D3D12_TEXTURE_COPY_LOCATION to{
                    destination, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
                commandList->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            };
            graph.AddPass(std::move(copy));

            Render::RenderPassDesc pass;
            pass.name = "ShadowDynamic";
            pass.writes.push_back({m_shadowAtlas, Render::Access::DepthWrite});
            // No clear: the copy above just put the cached static depth here,
            // and these draws belong on top of it.
            pass.execute = [this, recorder = recordShadows, tileViewport](
                               ID3D12GraphicsCommandList* commandList,
                               const Render::PassResources&) {
                for (int cascade = 0; cascade < m_shadowCascades.count; ++cascade) {
                    D3D12_VIEWPORT viewport = {};
                    D3D12_RECT scissor = {};
                    tileViewport(cascade, viewport, scissor);
                    commandList->RSSetViewports(1, &viewport);
                    commandList->RSSetScissorRects(1, &scissor);
                    recorder(commandList,
                             m_shadowCascades.cascades[cascade].viewProjection,
                             ShadowGeometry::Moving);
                }
            };
            graph.AddPass(std::move(pass));
        }

        // With nothing moving, the cached atlas IS this frame's shadow map and
        // there is no per-frame shadow work at all.
        m_shadowSampledAtlas = moving ? m_shadowAtlas : m_shadowStaticAtlas;
        m_lastFrameHadMovingGeometry = moving;
    }

    {
        Render::RenderPassDesc pass;
        pass.name = "GBuffer";
        for (std::size_t i = 0; i < kGBufferTargetCount; ++i) {
            pass.writes.push_back({m_gbuffer[i], Render::Access::RenderTarget});
            pass.clears.push_back(m_gbuffer[i]);
        }
        pass.writes.push_back({targets.depth, Render::Access::DepthWrite});
        pass.clears.push_back(targets.depth);
        // The baked lightmap is sampled here, so this pass reads it. Declared
        // as a read rather than bound behind the graph's back, which is what
        // puts the barrier between the bake and this in the right place on the
        // frame the bake happens.
        pass.reads.push_back({m_lightmapAtlas, Render::Access::PixelShaderResource});
        pass.execute = [this, recorder = std::move(recordGeometry)](
                           ID3D12GraphicsCommandList* commandList,
                           const Render::PassResources& resources) {
            if (recorder) {
                recorder(commandList, resources.Srv(m_lightmapAtlas));
            }
        };
        graph.AddPass(std::move(pass));
    }

    // The sky, before the lighting resolve and owning the clear. Added only
    // when there is something to draw -- which is the render graph earning its
    // keep: with the pass absent the lighting pass clears instead, and no
    // barrier anywhere else has to know the difference.
    const bool drawSky = m_skyboxEnabled && m_environment != nullptr;
    if (drawSky) {
        Render::RenderPassDesc pass;
        pass.name = "Skybox";
        pass.writes.push_back({m_sceneColor, Render::Access::RenderTarget});
        pass.clears.push_back(m_sceneColor);
        pass.execute = [this, inputs](ID3D12GraphicsCommandList* commandList,
                                      const Render::PassResources& resources) {
            Microsoft::WRL::ComPtr<ID3D12Device> device;
            DX::ThrowIfFailed(commandList->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));

            SkyboxConstants constants = {};
            const SimpleMath::Matrix viewProjection = inputs.view * inputs.projection;
            constants.invViewProj = XMMatrixTranspose(viewProjection.Invert());
            constants.eyePosition = inputs.eyePosition;
            constants.exposure = m_skyboxExposure;
            constants.targetWidth = static_cast<float>(resources.Width());
            constants.targetHeight = static_cast<float>(resources.Height());
            constants.yaw = XMConvertToRadians(m_skyboxYawDegrees);

            SharedGraphicsResource constantBuffer =
                GraphicsMemory::Get(device.Get()).AllocateConstant(constants);

            m_skyboxHeapBinder.Reset(commandList);
            m_skyboxSrvHeap->Reset();
            commandList->SetGraphicsRootSignature(
                m_skyboxRootSignature.GetRootSignature().Get());
            m_skyboxSrvHeap->ParseRootSignature(m_skyboxRootSignature);
            commandList->SetPipelineState(m_skyboxPso.Get());
            commandList->SetGraphicsRootConstantBufferView(0, constantBuffer.GpuAddress());
            m_skyboxSrvHeap->StageDescriptors(1, 0, 1,
                                              m_environmentSrv.GetDescriptorHandle());
            commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandList->IASetVertexBuffers(0, 0, nullptr);
            commandList->IASetIndexBuffer(nullptr);
            m_skyboxSrvHeap->CommitStagedDescriptorsForDraw(commandList, m_skyboxHeapBinder);
            commandList->DrawInstanced(3, 1, 0, 0);
        };
        graph.AddPass(std::move(pass));
    }

    {
        Render::RenderPassDesc pass;
        pass.name = "DeferredLighting";
        for (std::size_t i = 0; i < kGBufferTargetCount; ++i) {
            pass.reads.push_back({m_gbuffer[i], Render::Access::PixelShaderResource});
        }
        // Reading the same depth buffer the geometry pass just wrote is the
        // transition the graph exists to get right.
        pass.reads.push_back({targets.depth, Render::Access::PixelShaderResource});
        if (castShadows) {
            pass.reads.push_back(
                {m_shadowSampledAtlas, Render::Access::PixelShaderResource});
        }
        pass.writes.push_back({m_sceneColor, Render::Access::RenderTarget});
        // Cleared here only when no skybox pass ran: whichever of the two is
        // first has to clear, and what shows through where this shader discards
        // is then either the sky or the clear colour.
        if (!drawSky) {
            pass.clears.push_back(m_sceneColor);
        }

        pass.execute = [this, targets, inputs, castShadows](
                           ID3D12GraphicsCommandList* commandList,
                           const Render::PassResources& resources) {
            Microsoft::WRL::ComPtr<ID3D12Device> device;
            DX::ThrowIfFailed(commandList->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));

            DeferredLightingConstants constants = {};
            // Rebuilds a world position from a pixel's depth. Transposed on
            // upload for the same reason every other matrix in this project is:
            // HLSL reads cbuffer matrices column-major, so a row-major matrix
            // written straight in arrives transposed.
            const SimpleMath::Matrix viewProjection = inputs.view * inputs.projection;
            constants.invViewProj = XMMatrixTranspose(viewProjection.Invert());
            constants.eyePosition = inputs.eyePosition;
            constants.lightCount =
                static_cast<int>(std::min(inputs.lightCount, kMaxDeferredLights));
            constants.targetWidth = static_cast<float>(resources.Width());
            constants.targetHeight = static_cast<float>(resources.Height());
            if (inputs.lights != nullptr && inputs.lightCount > 0) {
                std::memcpy(constants.lights, inputs.lights,
                            sizeof(ShaderLight) *
                                std::min(inputs.lightCount, kMaxDeferredLights));
            }
            // The sun's direction comes from ResolveSun rather than from the
            // array, so the shading and the shadows cannot disagree about it.
            // Written unconditionally: with no override it is the same vector
            // that was just copied, and one code path cannot drift from itself.
            if (m_sunLightIndex != kNoSunLight &&
                m_sunLightIndex < std::min(inputs.lightCount, kMaxDeferredLights)) {
                constants.lights[m_sunLightIndex].direction = DirectX::XMFLOAT3(
                    m_sunDirection.x, m_sunDirection.y, m_sunDirection.z);
            }

            // The cascades, or zeroed when there are none: a cascade count of
            // zero makes SelectShadowCascade return -1 for every pixel, which
            // SampleShadow answers with full light.
            if (castShadows) {
                constants.shadowCascadeCount = m_shadowCascades.count;
                constants.shadowDepthBias = m_shadowDepthBias;
                constants.shadowNormalBias = m_shadowNormalBias;
                constants.shadowAtlasResolution =
                    static_cast<float>(m_shadowCascades.atlasResolution);
                for (int cascade = 0; cascade < m_shadowCascades.count; ++cascade) {
                    const ShadowCascade& source = m_shadowCascades.cascades[cascade];
                    ShadowCascadeConstants& out = constants.cascades[cascade];
                    // Transposed on upload like every other matrix here,
                    // because HLSL reads cbuffer matrices column-major.
                    out.viewProjection = XMMatrixTranspose(source.viewProjection);
                    out.atlasScaleOffset =
                        DirectX::XMFLOAT4(source.atlasScaleOffset.x,
                                          source.atlasScaleOffset.y,
                                          source.atlasScaleOffset.z,
                                          source.atlasScaleOffset.w);
                    out.splitDistance = source.splitDistance;
                    out.texelWorldSize = source.texelWorldSize;
                }
            }

            SharedGraphicsResource constantBuffer =
                GraphicsMemory::Get(device.Get()).AllocateConstant(constants);

            // Same dance as the geometry pass: the dynamic heap has to be reset
            // in step with the command list, and ParseRootSignature re-called
            // after the reset because Reset() clears the parsed layout.
            m_lightingHeapBinder.Reset(commandList);
            m_lightingSrvHeap->Reset();

            commandList->SetGraphicsRootSignature(
                m_lightingRootSignature.GetRootSignature().Get());
            m_lightingSrvHeap->ParseRootSignature(m_lightingRootSignature);
            commandList->SetPipelineState(m_lightingPso.Get());
            commandList->SetGraphicsRootConstantBufferView(0, constantBuffer.GpuAddress());

            // Staged one at a time because StageDescriptors copies a contiguous
            // source range and these five views come from separate allocations.
            for (std::size_t i = 0; i < kGBufferTargetCount; ++i) {
                m_lightingSrvHeap->StageDescriptors(1, static_cast<uint32_t>(i), 1,
                                                    resources.Srv(m_gbuffer[i]));
            }
            m_lightingSrvHeap->StageDescriptors(
                1, static_cast<uint32_t>(kGBufferDepthSrvSlot), 1,
                resources.Srv(targets.depth));
            // The atlas sits just past the G-buffer in the table. Staged even
            // when no shadows were rendered: a bound table may not contain an
            // uninitialised descriptor, whether the shader reads it or not.
            m_lightingSrvHeap->StageDescriptors(
                1, static_cast<uint32_t>(kGBufferSrvCount), 1,
                resources.Srv(m_shadowSampledAtlas));

            // Three vertices, no vertex or index buffer: VSLighting builds a
            // triangle that covers the viewport from SV_VertexID.
            commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandList->IASetVertexBuffers(0, 0, nullptr);
            commandList->IASetIndexBuffer(nullptr);
            m_lightingSrvHeap->CommitStagedDescriptorsForDraw(commandList,
                                                              m_lightingHeapBinder);
            commandList->DrawInstanced(3, 1, 0, 0);
        };
        graph.AddPass(std::move(pass));
    }
}

void DeferredRenderer::ShowImgui(UiSection section) {
    switch (section) {
        case UiSection::Render:      ShowRenderUi(); break;
        case UiSection::Performance: ShowPerformanceUi(); break;
        case UiSection::Scene:
        case UiSection::Neural:
            break;
    }
}

void DeferredRenderer::ShowRenderUi() {
    ImGui::SeparatorText("View");
    // What the back buffer shows. First because it is the control most likely
    // to be wanted and the only one that changes what is on screen wholesale,
    // so finding the frame replaced by a grey atlas should not mean hunting
    // for the reason.
    int view = static_cast<int>(m_debugView);
    if (ImGui::BeginCombo("Shown", DebugViewName(m_debugView))) {
        for (int i = 0; i < static_cast<int>(DebugView::Count); ++i) {
            const DebugView candidate = static_cast<DebugView>(i);
            if (ImGui::Selectable(DebugViewName(candidate), i == view)) {
                m_debugView = candidate;
                // The window means nothing across two different buffers, and
                // leaving it where the last view wanted it is how a correct
                // buffer gets mistaken for a blank one.
                m_debugRangeMin = 0.0f;
                m_debugRangeMax = 1.0f;
            }
        }
        ImGui::EndCombo();
    }
    if (m_debugView != DebugView::Scene) {
        // Every scalar view shares these, so they mean the same thing in all
        // of them: what reads as black and what reads as white. An
        // orthographic depth map is the case that needs them -- its occupied
        // range is often a few per cent wide and raw it is a white rectangle.
        const int channels =
            static_cast<int>(kDebugViews[static_cast<std::size_t>(m_debugView)].channels);
        if (channels >= static_cast<int>(DebugChannels::Red)) {
            ImGui::SliderFloat("Black at", &m_debugRangeMin, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("White at", &m_debugRangeMax, 0.0f, 1.0f, "%.3f");
            if (ImGui::SmallButton("Full range")) {
                m_debugRangeMin = 0.0f;
                m_debugRangeMax = 1.0f;
            }
        }
        // Said plainly: this replaces the display transform, so the frame
        // looking wrong is the point rather than a symptom.
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                           "the tonemap is replaced while this is not Scene");
        if (m_debugView == DebugView::Depth) {
            ImGui::Text("near %.3f, far %.0f (from the projection)",
                        m_debugNearPlane, m_debugFarPlane);
        }
    }

    ImGui::SeparatorText("Sun");
    if (!m_hasDirectionalLight) {
        // Worth saying: a scene lit only by point lights has no sun to steer,
        // and an absent sun looks exactly like broken shadows.
        ImGui::TextUnformatted("no directional light in this scene");
    } else {
        // Two angles rather than a vector, because a vector typed in by hand is
        // three numbers of which only two matter and which have to be
        // renormalised afterwards. The components are shown below, derived.
        bool enabled = m_sunOverride.enabled;
        if (ImGui::Checkbox("Override the asset's sun direction", &enabled)) {
            if (enabled && !m_sunOverride.enabled) {
                // Seeded from wherever the sun is now, so switching this on
                // moves nothing. Without it the light jumps to the struct's
                // default and the scene appears to break on a checkbox.
                SunEulerFromDirection(m_sunDirection, m_sunOverride.pitchDegrees,
                                      m_sunOverride.yawDegrees);
            }
            m_sunOverride.enabled = enabled;
        }

        // The direction the light TRAVELS, which is what glTF stores: a sun in
        // the sky shines downwards and so has a negative pitch. Said on the
        // labels because the sign is the one thing that is genuinely
        // ambiguous here.
        if (!m_sunOverride.enabled) {
            ImGui::BeginDisabled();
        }
        float pitch = m_sunOverride.pitchDegrees;
        float yaw = m_sunOverride.yawDegrees;
        if (!m_sunOverride.enabled) {
            // Showing the asset's own angles while disabled, so the sliders are
            // where the sun is rather than where the override last was.
            SunEulerFromDirection(m_sunDirection, pitch, yaw);
        }
        if (ImGui::SliderFloat("Pitch (deg, -90 = straight down)", &pitch, -90.0f,
                               90.0f, "%.1f")) {
            m_sunOverride.pitchDegrees = pitch;
        }
        // Wraps, so dragging past the end comes round rather than sticking: a
        // yaw of 181 and one of -179 are the same sun.
        if (ImGui::DragFloat("Yaw (deg, 0 = -Z)", &yaw, 0.5f, 0.0f, 0.0f, "%.1f")) {
            if (yaw > 180.0f) yaw -= 360.0f;
            if (yaw < -180.0f) yaw += 360.0f;
            m_sunOverride.yawDegrees = yaw;
        }
        if (!m_sunOverride.enabled) {
            ImGui::EndDisabled();
        }

        ImGui::Text("direction (%.3f, %.3f, %.3f)%s", m_sunDirection.x,
                    m_sunDirection.y, m_sunDirection.z,
                    m_sunOverride.enabled ? "" : "  (from the asset)");
        if (m_sunOverride.enabled) {
            // Not a scene edit: the Scene tab's hierarchy reads the light's own
            // node and still shows what the asset authored.
            ImGui::TextUnformatted("the Scene tab still shows the authored "
                                   "direction");
            // A bake fixes the sun into the lightmap, so every change here
            // throws it away and asks for another. Worth saying, because on a
            // large scene the rebake is a visible hitch on the next frame.
            ImGui::TextUnformatted("moving it invalidates the baked lightmap");
        }
    }

    ImGui::SeparatorText("Shadows");
    ImGui::Checkbox("Cascaded shadows", &m_shadowsEnabled);
    if (m_hasDirectionalLight && m_shadowsEnabled) {
        ImGui::SliderFloat("Shadow distance", &m_shadowSetup.shadowDistance, 5.0f,
                           1000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Split lambda", &m_shadowSetup.splitLambda, 0.0f, 1.0f,
                           "%.2f");
        ImGui::SliderFloat("Depth bias", &m_shadowDepthBias, 0.0f, 0.01f, "%.5f");
        ImGui::SliderFloat("Normal bias (texels)", &m_shadowNormalBias, 0.0f, 8.0f,
                           "%.2f");

        // Turning this off refits and re-renders every cascade every frame,
        // which is what this did before the cache existed. It is the only way
        // to be sure the cache is not serving depth that has gone stale, and
        // the image must not change when it is toggled.
        ImGui::Checkbox("Cache static shadow depth", &m_cacheStaticShadows);
        if (m_cacheStaticShadows) {
            // The margin is what the camera travels through before a cascade
            // has to be refitted, so it is the whole reason the cache hits --
            // paid for in resolution, since the same texels cover a sphere
            // this much larger. At 0 nothing is ever reusable.
            ImGui::SliderFloat("Fit margin", &m_shadowSetup.fitMargin, 0.0f, 0.5f,
                               "%.2f");
        }

        // A bake fixes the sun into the data, so moving the sun invalidates it
        // and the renderer requests another automatically. The button is for
        // everything else -- a changed setting, or simply wanting to watch it
        // happen.
        if (ImGui::Button("Rebake the lightmap")) {
            RequestLightmapBake();
        }
        ImGui::SameLine();
        ImGui::Text("baked: %s", m_lightmapBaked ? "yes" : "no");
    }

    ImGui::SeparatorText("Sky");
    if (m_environment != nullptr) {
        ImGui::Checkbox("Skybox", &m_skyboxEnabled);
        ImGui::Text("%s, %ux%u", m_environmentPath.c_str(), m_environmentWidth,
                    m_environmentHeight);
        // An HDRI's absolute values are arbitrary until something tonemaps
        // them, and nothing here does yet, so exposure is the only control that
        // makes the sky usable.
        ImGui::SliderFloat("Sky exposure", &m_skyboxExposure, 0.01f, 10.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        // Captured facing some arbitrary direction, almost never the one the
        // scene wants.
        ImGui::SliderFloat("Sky yaw", &m_skyboxYawDegrees, -180.0f, 180.0f, "%.0f deg");
    } else {
        ImGui::TextUnformatted("No environment map loaded (PTFLIO_SKYBOX)");
    }
}

void DeferredRenderer::ShowPerformanceUi() {
    if (!ImGui::CollapsingHeader("Deferred shading", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    ImGui::Text("G-buffer: %zu targets + depth", kGBufferTargetCount);
    for (std::size_t i = 0; i < kGBufferTargetCount; ++i) {
        const GBufferTarget target = static_cast<GBufferTarget>(i);
        ImGui::BulletText("%zu %s (format %d)", i, GBufferTargetName(target),
                          static_cast<int>(GBufferFormat(target)));
    }

    // Worth showing even when it is not near the cap: a scene that silently
    // drops lights looks like a scene that is simply too dark.
    ImGui::Text("Lights this frame: %zu of %zu", m_lastLightCount, kMaxDeferredLights);
    if (m_lastLightCount >= kMaxDeferredLights) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                           "at the cap -- further lights are dropped");
    }

    ImGui::Separator();
    if (!m_shadowsEnabled) {
        ImGui::TextUnformatted("Shadows: off");
    } else if (!m_hasDirectionalLight) {
        ImGui::TextUnformatted("Shadows: no directional light in this scene");
    } else {
        ImGui::Text("%d cascades in a %ux%u atlas", m_shadowCascades.count,
                    m_shadowCascades.atlasResolution, m_shadowCascades.atlasResolution);
        if (m_cacheStaticShadows) {
            ImGui::Text("%d of %d cascade(s) refitted this frame",
                        m_shadowCascadesRefitted, m_shadowCascades.count);
            if (m_shadowFrames != 0) {
                ImGui::Text("%llu static cascade render(s) over %llu frame(s)",
                            static_cast<unsigned long long>(m_shadowStaticRenders),
                            static_cast<unsigned long long>(m_shadowFrames));
                // The number that says whether the cache is earning anything:
                // without it this would be count * frames.
                const double naive = static_cast<double>(m_shadowFrames) *
                                     static_cast<double>(m_shadowCascades.count);
                if (naive > 0.0) {
                    ImGui::Text("   %.1f%% of what refitting every frame would cost",
                                100.0 * static_cast<double>(m_shadowStaticRenders) /
                                    naive);
                }
            }
            if (!m_lastFrameHadMovingGeometry) {
                // Worth saying: this is the best case, and it is easy to
                // mistake "no shadow passes in the plan" for a broken frame.
                ImGui::TextUnformatted("nothing in this scene moves, so the cached "
                                       "atlas is used directly");
            }
        } else {
            ImGui::TextUnformatted("cache off: every cascade is refitted and "
                                   "redrawn every frame");
        }
        for (int cascade = 0; cascade < m_shadowCascades.count; ++cascade) {
            ImGui::BulletText("cascade %d: to %.1f m, %.3f m per texel", cascade,
                              m_shadowCascades.cascades[cascade].splitDistance,
                              m_shadowCascades.cascades[cascade].texelWorldSize);
        }
    }

    ImGui::Separator();
    ImGui::Checkbox("Show the frame's pass plan", &m_showPlan);
    if (m_showPlan) {
        // The barriers the graph placed, as it placed them. Cheaper to read than
        // a capture when the question is only "did the transitions happen".
        ImGui::TextUnformatted(m_planDescription.empty() ? "(no plan recorded)"
                                                         : m_planDescription.c_str());
    }
}

}  // namespace NeuralModelIntegrateTestbed
