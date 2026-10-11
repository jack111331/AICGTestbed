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
#include "SceneGraph.hpp"

namespace NeuralModelIntegrateTestbed {

// How many lights one lighting pass can apply. Must equal DEFERRED_MAX_LIGHTS
// in shaders/no_texture.fx.
//
// 64 rather than the forward path's 4 because this buffer is uploaded once per
// frame instead of once per node, so the room is nearly free: 64 lights is
// 4 KB against a 64 KB constant buffer limit. Beyond a few dozen the loop
// itself becomes the cost and the answer is to stop testing every light against
// every pixel -- tiled or clustered culling -- which is a pass of its own and
// is why the number is not simply raised to the limit.
constexpr std::size_t kMaxDeferredLights = 64;

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
};

static_assert(offsetof(DeferredLightingConstants, invViewProj) == 0, "c0");
static_assert(offsetof(DeferredLightingConstants, eyePosition) == 64, "c4.xyz");
static_assert(offsetof(DeferredLightingConstants, lightCount) == 76, "c4.w");
static_assert(offsetof(DeferredLightingConstants, targetWidth) == 80, "c5.x");
static_assert(offsetof(DeferredLightingConstants, targetHeight) == 84, "c5.y");
static_assert(offsetof(DeferredLightingConstants, lights) == 96, "c6");
static_assert((sizeof(DeferredLightingConstants) % 16) == 0, "CB size not padded correctly");
static_assert(sizeof(DeferredLightingConstants) <= 65536,
              "a constant buffer may not exceed 64 KB; lower kMaxDeferredLights");

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
    };

    // Records the geometry that writes the G-buffer. Everything the graph owns
    // -- targets, viewport, barriers -- is already set when this is called.
    using GeometryRecorder = std::function<void(ID3D12GraphicsCommandList*)>;

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

    // Imports this frame's back buffer and depth buffer. Separate from
    // AddPasses so the caller can add its own passes against the same handles.
    //
    // `clearColor` is what the lighting pass clears the back buffer to, and so
    // what shows through wherever no geometry was drawn.
    FrameTargets ImportFrameTargets(Render::RenderGraph& graph,
                                    DX::DeviceResources& deviceResources,
                                    const float clearColor[4]);

    // Adds the G-buffer pass and the lighting pass.
    void AddPasses(Render::RenderGraph& graph, const FrameTargets& targets,
                   const FrameInputs& inputs, GeometryRecorder recordGeometry);

    Render::ResourceHandle GBufferHandle(GBufferTarget target) const {
        return m_gbuffer[static_cast<std::size_t>(target)];
    }

    // What the last Compile() produced, for the UI. Set by AddPasses' caller
    // rather than read from the graph, so a frame that failed to compile still
    // shows why.
    void SetPlanDescription(std::string description) {
        m_planDescription = std::move(description);
    }

    void ShowImgui();

private:
    void CreateLightingPipeline(ID3D12Device* device, DXGI_FORMAT backBufferFormat);

    ID3D12Device* m_device = nullptr;

    Render::ResourceHandle m_gbuffer[kGBufferTargetCount];

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

    // UI only.
    std::string m_planDescription;
    std::size_t m_lastLightCount = 0;
    std::size_t m_droppedLights = 0;
    bool m_showPlan = false;
};

}  // namespace NeuralModelIntegrateTestbed
