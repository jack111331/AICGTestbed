#include "pch.h"

#include "DeferredRenderer.hpp"

#include <cstring>

#include "imgui.h"
#include <no_texture.h>

using namespace DirectX;

namespace NeuralModelIntegrateTestbed {

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
    const CD3DX12_DESCRIPTOR_RANGE1 gbufferRange(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, static_cast<UINT>(kGBufferSrvCount), 5, 0,
        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    rootParameters[1].InitAsDescriptorTable(1, &gbufferRange, D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
    rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
    rsigDesc.pParameters = rootParameters;
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
    psoDesc.RTVFormats[0] = backBufferFormat;
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    psoDesc.SampleDesc.Count = 1;

    DX::ThrowIfFailed(
        device->CreateGraphicsPipelineState(&psoDesc, IID_GRAPHICS_PPV_ARGS(&m_lightingPso)));
    m_lightingPso->SetName(L"Deferred lighting");
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
                                 GeometryRecorder recordGeometry) {
    m_lastLightCount = inputs.lightCount;

    {
        Render::RenderPassDesc pass;
        pass.name = "GBuffer";
        for (std::size_t i = 0; i < kGBufferTargetCount; ++i) {
            pass.writes.push_back({m_gbuffer[i], Render::Access::RenderTarget});
            pass.clears.push_back(m_gbuffer[i]);
        }
        pass.writes.push_back({targets.depth, Render::Access::DepthWrite});
        pass.clears.push_back(targets.depth);
        pass.execute = [recorder = std::move(recordGeometry)](
                           ID3D12GraphicsCommandList* commandList,
                           const Render::PassResources&) {
            if (recorder) {
                recorder(commandList);
            }
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
        pass.writes.push_back({targets.color, Render::Access::RenderTarget});
        // Cleared here rather than before the graph runs, so the clear colour is
        // what shows wherever the pixel shader discards.
        pass.clears.push_back(targets.color);

        pass.execute = [this, targets, inputs](ID3D12GraphicsCommandList* commandList,
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

void DeferredRenderer::ShowImgui() {
    if (!ImGui::CollapsingHeader("Deferred shading")) {
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

    ImGui::Checkbox("Show the frame's pass plan", &m_showPlan);
    if (m_showPlan) {
        // The barriers the graph placed, as it placed them. Cheaper to read than
        // a capture when the question is only "did the transitions happen".
        ImGui::TextUnformatted(m_planDescription.empty() ? "(no plan recorded)"
                                                         : m_planDescription.c_str());
    }
}

}  // namespace NeuralModelIntegrateTestbed
