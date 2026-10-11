// Smoke test for Engine/RenderGraph.cc.
//
// The whole point of compiling a plan separately from recording it is that the
// interesting part -- where the resource barriers land -- needs no GPU to be
// wrong. So this runs with no device at all: RenderGraph::SetDevice is never
// called, declared textures never get a D3D12 resource, and Compile() still
// produces the full barrier list.
//
// Imported resources stand in for the swapchain back buffer and the depth
// buffer. An import's views are what the graph checks a declared access
// against, so the fakes below carry non-zero handles; ptr == 0 is the graph's
// "no view of that kind" sentinel and a real heap never hands it out.
//
// Two kinds of assertion:
//   * The deferred frame the engine actually builds, barrier by barrier. That
//     catches a reordering or a dropped transition.
//   * The errors Compile() is supposed to refuse. A render graph that silently
//     accepts a mis-ordered pass list is worse than no render graph, because
//     the barriers it does emit look authoritative.
#include "pch.h"

#include "RenderGraph.hpp"

#include <cstdio>
#include <string>
#include <vector>

using NeuralModelIntegrateTestbed::Render::Access;
using NeuralModelIntegrateTestbed::Render::BarrierPlan;
using NeuralModelIntegrateTestbed::Render::GraphPlan;
using NeuralModelIntegrateTestbed::Render::ImportedTextureDesc;
using NeuralModelIntegrateTestbed::Render::PassAccess;
using NeuralModelIntegrateTestbed::Render::RenderGraph;
using NeuralModelIntegrateTestbed::Render::RenderPassDesc;
using NeuralModelIntegrateTestbed::Render::ResourceHandle;
using NeuralModelIntegrateTestbed::Render::SizeMode;
using NeuralModelIntegrateTestbed::Render::StateName;
using NeuralModelIntegrateTestbed::Render::TextureDesc;

namespace {

int g_failures = 0;

void Expect(const char* what, bool condition) {
    std::printf("  %-56s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

// Distinct non-zero handles, so "the graph bound the view it was given" is
// checkable and so an import reads as having that kind of view.
D3D12_CPU_DESCRIPTOR_HANDLE FakeHandle(std::size_t id) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = {};
    handle.ptr = static_cast<SIZE_T>(0x1000 + id * 0x40);
    return handle;
}

struct DeferredFrame {
    ResourceHandle baseColor;
    ResourceHandle normal;
    ResourceHandle material;
    ResourceHandle emissive;
    ResourceHandle depth;
    ResourceHandle backBuffer;
};

// Declares the same G-buffer DeferredRenderer does. Kept here rather than
// calling into DeferredRenderer so the test has no device dependency, at the
// cost of the two having to be changed together -- which the G-buffer target
// count assertion below is there to notice.
DeferredFrame DeclareGBuffer(RenderGraph& graph) {
    DeferredFrame frame;

    TextureDesc colorTarget;
    colorTarget.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    colorTarget.allowRenderTarget = true;
    colorTarget.allowShaderResource = true;
    colorTarget.initialState = D3D12_RESOURCE_STATE_RENDER_TARGET;

    colorTarget.name = "GBufferBaseColor";
    frame.baseColor = graph.DeclareTexture(colorTarget);

    colorTarget.name = "GBufferNormal";
    colorTarget.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    frame.normal = graph.DeclareTexture(colorTarget);

    colorTarget.name = "GBufferMaterial";
    colorTarget.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    frame.material = graph.DeclareTexture(colorTarget);

    colorTarget.name = "GBufferEmissive";
    colorTarget.format = DXGI_FORMAT_R11G11B10_FLOAT;
    frame.emissive = graph.DeclareTexture(colorTarget);

    return frame;
}

// The frame Application builds: geometry writes the G-buffer, lighting reads it
// and resolves into the back buffer, overlays draw forward on top, ImGui last.
void BuildDeferredFrame(RenderGraph& graph, DeferredFrame& frame) {
    graph.BeginFrame();

    ImportedTextureDesc backBuffer;
    backBuffer.name = "BackBuffer";
    backBuffer.currentState = D3D12_RESOURCE_STATE_PRESENT;
    backBuffer.finalState = D3D12_RESOURCE_STATE_PRESENT;
    backBuffer.rtv = FakeHandle(0);
    frame.backBuffer = graph.ImportTexture(backBuffer);

    ImportedTextureDesc depth;
    depth.name = "SceneDepth";
    depth.currentState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    depth.finalState = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    depth.dsv = FakeHandle(1);
    depth.srv = FakeHandle(2);
    frame.depth = graph.ImportTexture(depth);

    RenderPassDesc geometry;
    geometry.name = "GBuffer";
    geometry.writes = {
        {frame.baseColor, Access::RenderTarget}, {frame.normal, Access::RenderTarget},
        {frame.material, Access::RenderTarget},  {frame.emissive, Access::RenderTarget},
        {frame.depth, Access::DepthWrite},
    };
    geometry.clears = {frame.baseColor, frame.normal, frame.material, frame.emissive,
                       frame.depth};
    graph.AddPass(geometry);

    RenderPassDesc lighting;
    lighting.name = "DeferredLighting";
    lighting.reads = {
        {frame.baseColor, Access::PixelShaderResource},
        {frame.normal, Access::PixelShaderResource},
        {frame.material, Access::PixelShaderResource},
        {frame.emissive, Access::PixelShaderResource},
        {frame.depth, Access::PixelShaderResource},
    };
    lighting.writes = {{frame.backBuffer, Access::RenderTarget}};
    lighting.clears = {frame.backBuffer};
    graph.AddPass(lighting);

    RenderPassDesc overlay;
    overlay.name = "ForwardOverlay";
    overlay.writes = {{frame.backBuffer, Access::RenderTarget},
                      {frame.depth, Access::DepthWrite}};
    graph.AddPass(overlay);

    RenderPassDesc imgui;
    imgui.name = "ImGui";
    imgui.writes = {{frame.backBuffer, Access::RenderTarget}};
    imgui.bindRenderTargets = false;
    graph.AddPass(imgui);
}

// Finds the barrier a pass emits for one resource, if any.
const BarrierPlan* FindBarrier(const GraphPlan& plan, std::size_t passIndex,
                               ResourceHandle resource) {
    for (const auto& pass : plan.passes) {
        if (pass.passIndex != passIndex) continue;
        for (const BarrierPlan& barrier : pass.barriers) {
            if (barrier.resource == resource) return &barrier;
        }
    }
    return nullptr;
}

std::size_t TotalBarriers(const GraphPlan& plan) {
    std::size_t total = 0;
    for (const auto& pass : plan.passes) total += pass.barriers.size();
    return total;
}

void TestDeferredFrame() {
    std::printf("the deferred frame\n");

    RenderGraph graph;
    graph.Resize(1280, 720);
    DeferredFrame frame = DeclareGBuffer(graph);
    BuildDeferredFrame(graph, frame);

    std::string error;
    const bool compiled = graph.Compile(&error);
    Expect("the frame compiles", compiled);
    if (!compiled) {
        std::printf("  error: %s\n", error.c_str());
        return;
    }
    const GraphPlan& plan = graph.Plan();
    std::fputs(graph.DescribePlan().c_str(), stdout);

    Expect("four passes", plan.passes.size() == 4);
    Expect("the G-buffer pass binds four render targets",
           plan.passes[0].renderTargets.size() == 4);
    Expect("the G-buffer pass binds depth",
           plan.passes[0].depthStencil.has_value() &&
               *plan.passes[0].depthStencil == frame.depth);

    // The G-buffer targets are declared starting in RENDER_TARGET and the depth
    // buffer is imported in DEPTH_WRITE, so the first pass should need nothing.
    Expect("the first frame's G-buffer pass needs no barriers",
           plan.passes[0].barriers.empty());

    // The transition that matters: every G-buffer target, and the depth buffer,
    // becomes readable for lighting.
    const ResourceHandle gbuffer[] = {frame.baseColor, frame.normal, frame.material,
                                      frame.emissive};
    bool allReadable = true;
    for (const ResourceHandle target : gbuffer) {
        const BarrierPlan* barrier = FindBarrier(plan, 1, target);
        allReadable = allReadable && barrier != nullptr &&
                      barrier->before == D3D12_RESOURCE_STATE_RENDER_TARGET &&
                      barrier->after == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }
    Expect("lighting makes every G-buffer target readable", allReadable);

    const BarrierPlan* depthRead = FindBarrier(plan, 1, frame.depth);
    Expect("lighting transitions depth DEPTH_WRITE -> PIXEL_SHADER_RESOURCE",
           depthRead != nullptr &&
               depthRead->before == D3D12_RESOURCE_STATE_DEPTH_WRITE &&
               depthRead->after == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    const BarrierPlan* backBufferWrite = FindBarrier(plan, 1, frame.backBuffer);
    Expect("lighting transitions the back buffer PRESENT -> RENDER_TARGET",
           backBufferWrite != nullptr &&
               backBufferWrite->after == D3D12_RESOURCE_STATE_RENDER_TARGET);

    // The forward overlays depth-test against the deferred geometry, so depth
    // has to go back to DEPTH_WRITE. This is the transition that is easy to
    // forget by hand, because the symptom is overlays that z-fight rather than
    // a validation error.
    const BarrierPlan* depthRestored = FindBarrier(plan, 2, frame.depth);
    Expect("the overlay pass transitions depth back to DEPTH_WRITE",
           depthRestored != nullptr &&
               depthRestored->before == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE &&
               depthRestored->after == D3D12_RESOURCE_STATE_DEPTH_WRITE);

    // The back buffer is already a render target by then, and ImGui writes it
    // too, so neither pass should re-transition it.
    Expect("the overlay pass does not re-transition the back buffer",
           FindBarrier(plan, 2, frame.backBuffer) == nullptr);
    Expect("the ImGui pass emits no barriers", plan.passes[3].barriers.empty());
    Expect("a pass that binds its own targets still gets its writes tracked",
           plan.passes[3].renderTargets.size() == 1);

    // 4 G-buffer + depth + back buffer for lighting, depth back for overlays.
    Expect("seven barriers in total, and no more", TotalBarriers(plan) == 7);

    // Only the back buffer needs anything after the last pass: the overlay pass
    // already left depth in DEPTH_WRITE, which is its declared final state, and
    // the G-buffer targets declare no final state at all.
    Expect("exactly one epilogue barrier", plan.epilogue.size() == 1);
    Expect("and it hands the back buffer to Present",
           plan.epilogue.size() == 1 && plan.epilogue[0].resource == frame.backBuffer &&
               plan.epilogue[0].before == D3D12_RESOURCE_STATE_RENDER_TARGET &&
               plan.epilogue[0].after == D3D12_RESOURCE_STATE_PRESENT);
}

void TestStateCarriesAcrossFrames() {
    std::printf("\nstate carried between frames\n");

    RenderGraph graph;
    graph.Resize(1280, 720);
    DeferredFrame frame = DeclareGBuffer(graph);

    std::string error;
    BuildDeferredFrame(graph, frame);
    Expect("frame 1 compiles", graph.Compile(&error));

    // Second frame, same structure. The G-buffer targets ended frame 1 as
    // PIXEL_SHADER_RESOURCE, so this time the geometry pass has to transition
    // them back -- the thing a caller writing barriers by hand has to remember
    // and the graph should not need told.
    BuildDeferredFrame(graph, frame);
    const bool compiled = graph.Compile(&error);
    Expect("frame 2 compiles", compiled);
    if (!compiled) {
        std::printf("  error: %s\n", error.c_str());
        return;
    }
    const GraphPlan& plan = graph.Plan();

    const BarrierPlan* reacquired = FindBarrier(plan, 0, frame.baseColor);
    Expect("frame 2's G-buffer pass reacquires RENDER_TARGET",
           reacquired != nullptr &&
               reacquired->before == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE &&
               reacquired->after == D3D12_RESOURCE_STATE_RENDER_TARGET);
    // Four barriers, one per G-buffer target. Depth is imported fresh every
    // frame already in DEPTH_WRITE, so it is not among them -- the graph only
    // carries state for the resources it owns.
    Expect("all four G-buffer targets are reacquired",
           plan.passes[0].barriers.size() == 4);
    Expect("imported depth needs no reacquire",
           FindBarrier(plan, 0, frame.depth) == nullptr);
}

// Each of these is a mistake the graph exists to refuse. The assertion is that
// Compile() fails AND says something specific enough to act on.
void TestRefusals() {
    std::printf("\nmistakes Compile() refuses\n");

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        TextureDesc desc;
        desc.name = "Target";
        desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.allowRenderTarget = true;
        const ResourceHandle target = graph.DeclareTexture(desc);

        graph.BeginFrame();
        RenderPassDesc reader;
        reader.name = "Reader";
        reader.reads = {{target, Access::PixelShaderResource}};
        graph.AddPass(reader);

        std::string error;
        const bool compiled = graph.Compile(&error);
        Expect("reading a target nothing wrote is refused", !compiled);
        Expect("and the message names the pass and the resource",
               error.find("Reader") != std::string::npos &&
                   error.find("Target") != std::string::npos);
        std::printf("    %s\n", error.c_str());
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        TextureDesc desc;
        desc.name = "History";
        desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.allowRenderTarget = true;
        desc.persistentContent = true;
        const ResourceHandle history = graph.DeclareTexture(desc);

        graph.BeginFrame();
        RenderPassDesc reader;
        reader.name = "Reader";
        reader.reads = {{history, Access::PixelShaderResource}};
        graph.AddPass(reader);

        std::string error;
        Expect("but a persistentContent resource may be read unwritten",
               graph.Compile(&error));
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        TextureDesc desc;
        desc.name = "ColorOnly";
        desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.allowRenderTarget = true;
        const ResourceHandle color = graph.DeclareTexture(desc);

        graph.BeginFrame();
        RenderPassDesc pass;
        pass.name = "Depth";
        pass.writes = {{color, Access::DepthWrite}};
        graph.AddPass(pass);

        std::string error;
        Expect("using a colour target as depth is refused", !graph.Compile(&error));
        std::printf("    %s\n", error.c_str());
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        TextureDesc desc;
        desc.name = "Both";
        desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.allowRenderTarget = true;
        const ResourceHandle both = graph.DeclareTexture(desc);

        graph.BeginFrame();
        RenderPassDesc writer;
        writer.name = "Writer";
        writer.writes = {{both, Access::RenderTarget}};
        graph.AddPass(writer);
        // A pass that reads what it writes cannot be expressed as one resource
        // state, so it has to be split into two passes or a ping-pong pair.
        RenderPassDesc readWrite;
        readWrite.name = "ReadWrite";
        readWrite.reads = {{both, Access::PixelShaderResource}};
        readWrite.writes = {{both, Access::RenderTarget}};
        graph.AddPass(readWrite);

        std::string error;
        Expect("reading and writing one resource in one pass is refused",
               !graph.Compile(&error));
        std::printf("    %s\n", error.c_str());
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        TextureDesc desc;
        desc.name = "Target";
        desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.allowRenderTarget = true;
        const ResourceHandle target = graph.DeclareTexture(desc);

        graph.BeginFrame();
        RenderPassDesc pass;
        pass.name = "ClearOnly";
        pass.clears = {target};
        graph.AddPass(pass);

        std::string error;
        Expect("clearing a resource the pass did not declare is refused",
               !graph.Compile(&error));
        std::printf("    %s\n", error.c_str());
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        graph.BeginFrame();
        // An import with no RTV: the graph has no flags to check, so the
        // missing view is what tells it the access is wrong.
        ImportedTextureDesc imported;
        imported.name = "DepthOnlyImport";
        imported.dsv = FakeHandle(7);
        const ResourceHandle handle = graph.ImportTexture(imported);

        RenderPassDesc pass;
        pass.name = "Colour";
        pass.writes = {{handle, Access::RenderTarget}};
        graph.AddPass(pass);

        std::string error;
        Expect("an import with no RTV cannot be a render target",
               !graph.Compile(&error));
        std::printf("    %s\n", error.c_str());
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        graph.BeginFrame();
        RenderPassDesc pass;
        pass.name = "Stray";
        pass.writes = {{ResourceHandle{}, Access::RenderTarget}};
        graph.AddPass(pass);

        std::string error;
        Expect("an invalid handle is refused", !graph.Compile(&error));
        std::printf("    %s\n", error.c_str());
    }

    {
        RenderGraph graph;
        graph.Resize(64, 64);
        TextureDesc desc;
        desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.allowRenderTarget = true;
        std::vector<ResourceHandle> targets;
        for (int i = 0; i < 9; ++i) {
            desc.name = "Target" + std::to_string(i);
            targets.push_back(graph.DeclareTexture(desc));
        }
        graph.BeginFrame();
        RenderPassDesc pass;
        pass.name = "TooWide";
        for (const ResourceHandle target : targets) {
            pass.writes.push_back({target, Access::RenderTarget});
        }
        graph.AddPass(pass);

        std::string error;
        Expect("more than eight render targets is refused", !graph.Compile(&error));
        std::printf("    %s\n", error.c_str());
    }
}

void TestExecuteWithoutDevice() {
    std::printf("\nexecuting without a device\n");

    RenderGraph graph;
    graph.Resize(64, 64);
    TextureDesc desc;
    desc.name = "Target";
    desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.allowRenderTarget = true;
    desc.initialState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    const ResourceHandle target = graph.DeclareTexture(desc);

    graph.BeginFrame();
    RenderPassDesc pass;
    pass.name = "Writer";
    pass.writes = {{target, Access::RenderTarget}};
    graph.AddPass(pass);

    std::string error;
    Expect("a plan still compiles with no device", graph.Compile(&error));
    // Which is the point of the split: the barrier is known even though there
    // is nothing to record it into.
    Expect("and the barrier is planned", graph.Plan().passes[0].barriers.size() == 1);

    Expect("Execute without a command list fails rather than crashing",
           !graph.Execute(nullptr, &error));
    std::printf("    %s\n", error.c_str());

    RenderGraph fresh;
    Expect("Execute before Compile fails", !fresh.Execute(nullptr, &error));
    std::printf("    %s\n", error.c_str());
}

void TestStateNames() {
    std::printf("\nstate naming\n");
    // Used in every error message above, so worth one assertion that it is not
    // quietly returning hex for the states the graph itself produces.
    Expect("RENDER_TARGET names itself",
           std::string(StateName(D3D12_RESOURCE_STATE_RENDER_TARGET)) == "RENDER_TARGET");
    Expect("PIXEL_SHADER_RESOURCE names itself",
           std::string(StateName(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)) ==
               "PIXEL_SHADER_RESOURCE");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    TestDeferredFrame();
    TestStateCarriesAcrossFrames();
    TestRefusals();
    TestExecuteWithoutDevice();
    TestStateNames();

    std::printf("\n%s\n", g_failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
