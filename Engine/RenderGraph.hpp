#pragma once

// A small render graph: passes declare which resources they read and write,
// the graph works out the resource barriers between them, and then records the
// whole frame into one command list.
//
// Why bother, rather than keeping the barriers where the draws are? Three
// reasons, all of which the deferred path immediately needs:
//
//   * The G-buffer depth buffer is written by the geometry pass, READ by the
//     lighting pass, and written again by the forward overlays. That is two
//     transitions per frame on one resource, in an order that depends on which
//     passes are enabled. Hand-written barriers get this wrong silently -- the
//     debug layer complains, or worse, a driver tolerates it.
//   * Adding a pass (shadow maps, SSAO, a tonemap) should not require auditing
//     every other pass's barriers.
//   * The plan is inspectable. Compile() produces a GraphPlan of barriers and
//     bindings WITHOUT a D3D12 device, so //Engine:rendergraph-smoke can assert
//     on the transitions a given frame structure produces.
//
// What this deliberately does NOT do yet:
//   * No transient memory aliasing. Declared textures are committed resources
//     that live until the next Resize. A real aliasing allocator needs resource
//     lifetime analysis plus aliasing barriers; the pass list is the hard part
//     and it is here, so that can be added behind the same API.
//   * No automatic pass culling. Every added pass runs. Culling needs a notion
//     of which outputs the frame actually consumes, which only matters once
//     passes are added conditionally.
//   * No split barriers, no async compute queue, no multi-threaded recording.
//
// Usage per frame:
//
//     graph.BeginFrame();
//     const ResourceHandle backBuffer = graph.ImportTexture(...);
//     graph.AddPass(gbufferPass);
//     graph.AddPass(lightingPass);
//     std::string error;
//     if (!graph.Compile(&error)) { ... }
//     graph.Execute(commandList, &error);
//
// Declared (graph-owned) textures are registered once with DeclareTexture and
// survive across frames; their current resource state is remembered, so a
// G-buffer left in PIXEL_SHADER_RESOURCE at the end of one frame transitions
// back to RENDER_TARGET at the start of the next without anyone asking.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <directx/d3d12.h>
#include <directx/d3dx12.h>
#include <wrl/client.h>

#include "Descriptors/DescriptorAllocation.hpp"
#include "Descriptors/DescriptorAllocator.hpp"

namespace NeuralModelIntegrateTestbed::Render {

// Identifies a resource inside one graph. Not a pointer: a declared texture's
// underlying ID3D12Resource is recreated on every Resize, and an imported one
// changes every frame with the back buffer index, so the handle has to outlive
// both.
struct ResourceHandle {
    static constexpr std::uint32_t kInvalid = 0xFFFFFFFFu;

    std::uint32_t index = kInvalid;

    bool IsValid() const { return index != kInvalid; }

    friend bool operator==(ResourceHandle a, ResourceHandle b) {
        return a.index == b.index;
    }
    friend bool operator!=(ResourceHandle a, ResourceHandle b) { return !(a == b); }
};

// How a pass touches a resource. The graph maps each of these to one
// D3D12_RESOURCE_STATES; keeping the enum at this level rather than taking raw
// states means a pass says what it is doing, and Compile() can check that
// against the flags the resource was created with.
enum class Access {
    RenderTarget,            // bound as an RTV and written
    DepthWrite,              // bound as a DSV, depth writes enabled
    DepthRead,               // bound as a DSV, depth test only
    PixelShaderResource,     // sampled or Load()ed from a pixel shader
    NonPixelShaderResource,  // read from a vertex/compute shader
    UnorderedAccess,         // read/write through a UAV
    CopySource,
    CopyDest,
    Present,                 // handed to IDXGISwapChain::Present
};

// Maps an Access to the resource state it requires. Exposed because the smoke
// test asserts on states, and because an imported resource's initial and final
// states are given in the same terms.
D3D12_RESOURCE_STATES StateForAccess(Access access);

// Human-readable, for error messages and the smoke test's output.
const char* AccessName(Access access);
const char* StateName(D3D12_RESOURCE_STATES state);

// Whether a declared texture tracks the swapchain's size or has its own.
enum class SizeMode {
    SwapchainRelative,  // width = round(swapchainWidth * widthScale)
    Absolute,           // width/height as given
};

// A graph-owned texture. Formats are split out because a depth target needs
// three different ones for the same resource: a TYPELESS resource format so an
// SRV is legal at all, a typed depthStencilFormat for the DSV, and the matching
// single-channel shaderResourceFormat for the SRV. Getting this wrong is the
// classic "CreateShaderResourceView failed on a D32_FLOAT resource" error.
struct TextureDesc {
    std::string name;

    // The resource's own format. For depth, the TYPELESS form (R32_TYPELESS for
    // a D32_FLOAT view).
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    // View formats. UNKNOWN means "same as format", which is right for every
    // colour target and wrong for every depth one.
    DXGI_FORMAT renderTargetFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT depthStencilFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT shaderResourceFormat = DXGI_FORMAT_UNKNOWN;

    SizeMode sizeMode = SizeMode::SwapchainRelative;
    float widthScale = 1.0f;
    float heightScale = 1.0f;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    bool allowRenderTarget = false;
    bool allowDepthStencil = false;
    bool allowUnorderedAccess = false;
    bool allowShaderResource = true;

    // Used by a pass that lists this resource in RenderPassDesc::clears, and as
    // the committed resource's optimized clear value.
    float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float clearDepth = 1.0f;
    std::uint8_t clearStencil = 0;

    // The state a freshly created resource starts in, and the one the graph
    // assumes at the top of the first frame after a Resize.
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;

    // Set for a resource whose content is meant to survive into the next frame
    // -- a reprojection history, an accumulation buffer. Without it, reading a
    // declared resource that no pass in THIS frame wrote is a Compile() error,
    // which is the check that catches a mis-ordered pass list.
    bool persistentContent = false;
};

// A resource the graph does not own: the swapchain back buffer, DeviceResources'
// depth buffer. The caller supplies the views, because whoever created the
// resource already has them.
struct ImportedTextureDesc {
    std::string name;

    ID3D12Resource* resource = nullptr;

    // What state the resource is in when Execute() is called, and what state it
    // must be left in. finalState drives the graph's epilogue barriers -- for
    // the back buffer that is PRESENT, which is what DeviceResources::Present
    // expects to find.
    D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON;
    std::optional<D3D12_RESOURCE_STATES> finalState;

    // Zeroed handles are treated as absent; a pass that needs a view the
    // import did not supply is a Compile() error rather than a null binding.
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = {};
    D3D12_CPU_DESCRIPTOR_HANDLE dsv = {};
    D3D12_CPU_DESCRIPTOR_HANDLE srv = {};

    std::uint32_t width = 0;
    std::uint32_t height = 0;

    float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float clearDepth = 1.0f;
    std::uint8_t clearStencil = 0;
};

// One resource a pass touches.
struct PassAccess {
    ResourceHandle resource;
    Access access = Access::PixelShaderResource;
};

// What a pass's execute callback can ask the graph for. Handles rather than
// resources, because the callback is written once and the views behind a handle
// change with the back buffer index and with every Resize.
class PassResources {
public:
    PassResources(const class RenderGraph& graph, std::uint32_t width, std::uint32_t height)
        : m_graph(graph), m_width(width), m_height(height) {}

    ID3D12Resource* Resource(ResourceHandle handle) const;
    // Zeroed if the resource has no view of that kind. Compile() has already
    // rejected a pass that declared an access needing a view it does not have,
    // so a pass reading its own declared resources never sees a zero here.
    D3D12_CPU_DESCRIPTOR_HANDLE Rtv(ResourceHandle handle) const;
    D3D12_CPU_DESCRIPTOR_HANDLE Dsv(ResourceHandle handle) const;
    D3D12_CPU_DESCRIPTOR_HANDLE Srv(ResourceHandle handle) const;

    std::uint32_t Width() const { return m_width; }
    std::uint32_t Height() const { return m_height; }
    D3D12_VIEWPORT Viewport() const;
    D3D12_RECT ScissorRect() const;

private:
    const class RenderGraph& m_graph;
    std::uint32_t m_width;
    std::uint32_t m_height;
};

struct RenderPassDesc {
    std::string name;

    // Resources this pass reads. Compile() requires each to have been written by
    // an earlier pass in the same frame, or to be an import -- an import is
    // assumed to carry content from outside the graph.
    std::vector<PassAccess> reads;

    // Resources this pass writes. Those with Access::RenderTarget are bound as
    // render targets in this order; one DepthWrite or DepthRead entry is bound
    // as the depth target.
    std::vector<PassAccess> writes;

    // Cleared at the start of the pass, before the callback runs. Each must also
    // appear in `writes`.
    std::vector<ResourceHandle> clears;

    // Set the render targets, viewport and scissor from `writes` before calling
    // the callback. Off for a pass that binds its own (ImGui's backend sets its
    // own render target) or that issues no draws at all.
    bool bindRenderTargets = true;

    // Recorded into the frame's command list between this pass's barriers and
    // the next pass's.
    std::function<void(ID3D12GraphicsCommandList*, const PassResources&)> execute;
};

// --- the compiled plan -------------------------------------------------------
// Separated from Execute() so it can be produced, and asserted on, with no
// device: the barrier placement is the part most worth testing and the part
// that needs no GPU to be wrong.

struct BarrierPlan {
    ResourceHandle resource;
    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_COMMON;
};

struct PassPlan {
    std::size_t passIndex = 0;
    std::string name;
    // Issued as one ResourceBarrier call before the pass runs.
    std::vector<BarrierPlan> barriers;
    std::vector<ResourceHandle> renderTargets;
    std::optional<ResourceHandle> depthStencil;
};

struct GraphPlan {
    std::vector<PassPlan> passes;
    // Returns imported resources to their declared finalState after the last
    // pass. The back buffer's PRESENT transition lands here.
    std::vector<BarrierPlan> epilogue;
};

class RenderGraph {
public:
    RenderGraph() = default;
    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;

    // Without a device the graph still declares resources and compiles plans;
    // it just cannot create textures or views. That is the mode the smoke test
    // runs in.
    void SetDevice(ID3D12Device* device);
    ID3D12Device* Device() const { return m_device; }

    // Registers a graph-owned texture. Safe before SetDevice: the resource is
    // created by the next Resize.
    ResourceHandle DeclareTexture(const TextureDesc& desc);

    // (Re)creates every declared texture at this size. A no-op if the size is
    // unchanged and the resources already exist.
    void Resize(std::uint32_t width, std::uint32_t height);

    std::uint32_t Width() const { return m_width; }
    std::uint32_t Height() const { return m_height; }

    // Drops the previous frame's passes and imports. Declared textures and the
    // states they were left in survive.
    void BeginFrame();

    ResourceHandle ImportTexture(const ImportedTextureDesc& desc);

    std::size_t AddPass(RenderPassDesc pass);
    std::size_t PassCount() const { return m_passes.size(); }

    // Walks the passes in order, validating accesses and placing barriers.
    // Returns false and fills `error` on the first problem; the plan is then
    // unusable. Does not touch the GPU.
    bool Compile(std::string* error);

    const GraphPlan& Plan() const { return m_plan; }

    // Records the compiled plan. Fails rather than recording a partial frame if
    // a pass needs a view that is missing, which can only happen when the graph
    // has no device.
    bool Execute(ID3D12GraphicsCommandList* commandList, std::string* error);

    // Formats the plan as one line per barrier and binding. Used by the smoke
    // test and handy to drop into the UI when a frame looks wrong.
    std::string DescribePlan() const;

    const std::string& ResourceName(ResourceHandle handle) const;

    // Releases every declared texture and its views. Resize() rebuilds them.
    void Reset();

private:
    friend class PassResources;

    struct Resource {
        TextureDesc desc;
        bool imported = false;

        Microsoft::WRL::ComPtr<ID3D12Resource> owned;
        ID3D12Resource* imported_resource = nullptr;

        // Declared resources allocate their own views; imported ones carry the
        // creator's.
        Descriptors::DescriptorAllocation rtvAllocation;
        Descriptors::DescriptorAllocation dsvAllocation;
        Descriptors::DescriptorAllocation srvAllocation;
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = {};
        D3D12_CPU_DESCRIPTOR_HANDLE dsv = {};
        D3D12_CPU_DESCRIPTOR_HANDLE srv = {};

        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        std::optional<D3D12_RESOURCE_STATES> finalState;

        std::uint32_t width = 0;
        std::uint32_t height = 0;

        // Reset each frame by Compile(): whether any pass has written this
        // resource yet, which is what makes "read before anything wrote it" an
        // error for declared resources but not for imports.
        bool writtenThisFrame = false;
    };

    ID3D12Resource* ResourcePointer(const Resource& resource) const;
    void CreateTexture(Resource& resource);
    void CreateViews(Resource& resource);
    bool ValidateAccess(const Resource& resource, Access access, const std::string& passName,
                        std::string* error) const;

    ID3D12Device* m_device = nullptr;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;

    std::vector<Resource> m_resources;
    // Declared resources occupy [0, m_declaredCount); imports are appended past
    // that and truncated by BeginFrame.
    std::size_t m_declaredCount = 0;

    std::vector<RenderPassDesc> m_passes;
    GraphPlan m_plan;
    bool m_compiled = false;

    // RTV/DSV heaps are never shader-visible, so the existing CPU descriptor
    // allocator serves all three. Pages are small: a frame needs a handful.
    std::unique_ptr<Descriptors::DescriptorAllocator> m_rtvAllocator;
    std::unique_ptr<Descriptors::DescriptorAllocator> m_dsvAllocator;
    std::unique_ptr<Descriptors::DescriptorAllocator> m_srvAllocator;
};

}  // namespace NeuralModelIntegrateTestbed::Render
