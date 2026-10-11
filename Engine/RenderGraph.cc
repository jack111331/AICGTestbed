#include "pch.h"

#include "RenderGraph.hpp"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <utility>

#include "Descriptors/DescriptorContext.hpp"

namespace NeuralModelIntegrateTestbed::Render {

namespace {

// A zeroed CPU handle means "this resource has no view of that kind". Real
// descriptor heaps never hand out ptr == 0, so the sentinel is unambiguous.
bool HasView(D3D12_CPU_DESCRIPTOR_HANDLE handle) { return handle.ptr != 0; }

DXGI_FORMAT Coalesce(DXGI_FORMAT specific, DXGI_FORMAT fallback) {
    return specific == DXGI_FORMAT_UNKNOWN ? fallback : specific;
}

bool IsDepthAccess(Access access) {
    return access == Access::DepthWrite || access == Access::DepthRead;
}

}  // namespace

D3D12_RESOURCE_STATES StateForAccess(Access access) {
    switch (access) {
        case Access::RenderTarget:           return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case Access::DepthWrite:             return D3D12_RESOURCE_STATE_DEPTH_WRITE;
        case Access::DepthRead:              return D3D12_RESOURCE_STATE_DEPTH_READ;
        case Access::PixelShaderResource:    return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case Access::NonPixelShaderResource: return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case Access::UnorderedAccess:        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case Access::CopySource:             return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case Access::CopyDest:               return D3D12_RESOURCE_STATE_COPY_DEST;
        case Access::Present:                return D3D12_RESOURCE_STATE_PRESENT;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

const char* AccessName(Access access) {
    switch (access) {
        case Access::RenderTarget:           return "render target";
        case Access::DepthWrite:             return "depth write";
        case Access::DepthRead:              return "depth read";
        case Access::PixelShaderResource:    return "pixel shader resource";
        case Access::NonPixelShaderResource: return "non-pixel shader resource";
        case Access::UnorderedAccess:        return "unordered access";
        case Access::CopySource:             return "copy source";
        case Access::CopyDest:               return "copy dest";
        case Access::Present:                return "present";
    }
    return "unknown";
}

const char* StateName(D3D12_RESOURCE_STATES state) {
    // D3D12_RESOURCE_STATES is a bitfield, but the graph only ever produces the
    // single-bit states StateForAccess returns, plus COMMON. Anything else is a
    // resource imported in a state the caller chose, so fall through to hex.
    switch (state) {
        case D3D12_RESOURCE_STATE_COMMON:                   return "COMMON/PRESENT";
        case D3D12_RESOURCE_STATE_RENDER_TARGET:            return "RENDER_TARGET";
        case D3D12_RESOURCE_STATE_DEPTH_WRITE:              return "DEPTH_WRITE";
        case D3D12_RESOURCE_STATE_DEPTH_READ:               return "DEPTH_READ";
        case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:    return "PIXEL_SHADER_RESOURCE";
        case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:return "NON_PIXEL_SHADER_RESOURCE";
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:         return "UNORDERED_ACCESS";
        case D3D12_RESOURCE_STATE_COPY_SOURCE:              return "COPY_SOURCE";
        case D3D12_RESOURCE_STATE_COPY_DEST:                return "COPY_DEST";
        case D3D12_RESOURCE_STATE_GENERIC_READ:             return "GENERIC_READ";
        default: break;
    }
    static char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%X", static_cast<unsigned>(state));
    return buffer;
}

// --- PassResources -----------------------------------------------------------

ID3D12Resource* PassResources::Resource(ResourceHandle handle) const {
    if (handle.index >= m_graph.m_resources.size()) {
        return nullptr;
    }
    return m_graph.ResourcePointer(m_graph.m_resources[handle.index]);
}

D3D12_CPU_DESCRIPTOR_HANDLE PassResources::Rtv(ResourceHandle handle) const {
    if (handle.index >= m_graph.m_resources.size()) {
        return {};
    }
    return m_graph.m_resources[handle.index].rtv;
}

D3D12_CPU_DESCRIPTOR_HANDLE PassResources::Dsv(ResourceHandle handle) const {
    if (handle.index >= m_graph.m_resources.size()) {
        return {};
    }
    return m_graph.m_resources[handle.index].dsv;
}

D3D12_CPU_DESCRIPTOR_HANDLE PassResources::Srv(ResourceHandle handle) const {
    if (handle.index >= m_graph.m_resources.size()) {
        return {};
    }
    return m_graph.m_resources[handle.index].srv;
}

D3D12_VIEWPORT PassResources::Viewport() const {
    D3D12_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(m_width);
    viewport.Height = static_cast<float>(m_height);
    viewport.MinDepth = D3D12_MIN_DEPTH;
    viewport.MaxDepth = D3D12_MAX_DEPTH;
    return viewport;
}

D3D12_RECT PassResources::ScissorRect() const {
    return D3D12_RECT{0, 0, static_cast<LONG>(m_width), static_cast<LONG>(m_height)};
}

// --- RenderGraph -------------------------------------------------------------

void RenderGraph::SetDevice(ID3D12Device* device) {
    if (m_device == device) {
        return;
    }
    // Every declared texture and view belongs to the old device.
    Reset();
    m_device = device;
    if (m_device == nullptr) {
        return;
    }

    // The ported descriptor classes reach for the device through this context
    // rather than holding one. Initialize is idempotent, so calling it here as
    // well as in GLTFAdapter::PrepareBuffer costs nothing and means the graph
    // works whichever runs first.
    Descriptors::Context().Initialize(m_device);

    // 64 per page: a frame's render targets and depth views number in the
    // handful, and an unused page is only wasted descriptor heap space.
    m_rtvAllocator = std::make_unique<Descriptors::DescriptorAllocator>(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 64);
    m_dsvAllocator = std::make_unique<Descriptors::DescriptorAllocator>(
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 64);
    m_srvAllocator = std::make_unique<Descriptors::DescriptorAllocator>(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64);
}

ResourceHandle RenderGraph::DeclareTexture(const TextureDesc& desc) {
    Resource resource;
    resource.desc = desc;
    resource.state = desc.initialState;
    m_resources.push_back(std::move(resource));
    m_declaredCount = m_resources.size();
    return ResourceHandle{static_cast<std::uint32_t>(m_resources.size() - 1)};
}

void RenderGraph::Resize(std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0) {
        return;
    }
    const bool sameSize = (width == m_width && height == m_height);
    m_width = width;
    m_height = height;
    if (m_device == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < m_declaredCount; ++i) {
        Resource& resource = m_resources[i];
        if (sameSize && resource.owned) {
            continue;
        }
        CreateTexture(resource);
        CreateViews(resource);
    }
}

void RenderGraph::BeginFrame() {
    // Imports are appended past the declared resources, so dropping them is a
    // truncation. Declared resources keep the state the last frame left them in,
    // which is what lets the graph transition a G-buffer back to RENDER_TARGET
    // without the caller remembering that it ended up readable.
    m_resources.resize(m_declaredCount);
    m_passes.clear();
    m_plan = {};
    m_compiled = false;
}

ResourceHandle RenderGraph::ImportTexture(const ImportedTextureDesc& desc) {
    Resource resource;
    resource.imported = true;
    resource.imported_resource = desc.resource;
    resource.desc.name = desc.name;
    // An import has no creation flags to check an access against, so the views
    // it supplied stand in for them: declaring Access::RenderTarget on an import
    // that passed no RTV is then a Compile() error rather than a null binding.
    resource.desc.allowRenderTarget = HasView(desc.rtv);
    resource.desc.allowDepthStencil = HasView(desc.dsv);
    resource.desc.allowShaderResource = HasView(desc.srv);
    resource.desc.clearColor[0] = desc.clearColor[0];
    resource.desc.clearColor[1] = desc.clearColor[1];
    resource.desc.clearColor[2] = desc.clearColor[2];
    resource.desc.clearColor[3] = desc.clearColor[3];
    resource.desc.clearDepth = desc.clearDepth;
    resource.desc.clearStencil = desc.clearStencil;
    resource.rtv = desc.rtv;
    resource.dsv = desc.dsv;
    resource.srv = desc.srv;
    resource.state = desc.currentState;
    resource.finalState = desc.finalState;
    resource.width = desc.width != 0 ? desc.width : m_width;
    resource.height = desc.height != 0 ? desc.height : m_height;
    m_resources.push_back(std::move(resource));
    return ResourceHandle{static_cast<std::uint32_t>(m_resources.size() - 1)};
}

std::size_t RenderGraph::AddPass(RenderPassDesc pass) {
    m_passes.push_back(std::move(pass));
    return m_passes.size() - 1;
}

bool RenderGraph::ValidateAccess(const Resource& resource, Access access,
                                 const std::string& passName, std::string* error) const {
    const char* missing = nullptr;
    switch (access) {
        case Access::RenderTarget:
            if (!resource.desc.allowRenderTarget) missing = "render target view";
            break;
        case Access::DepthWrite:
        case Access::DepthRead:
            if (!resource.desc.allowDepthStencil) missing = "depth stencil view";
            break;
        case Access::PixelShaderResource:
        case Access::NonPixelShaderResource:
            if (!resource.desc.allowShaderResource) missing = "shader resource view";
            break;
        case Access::UnorderedAccess:
            if (!resource.desc.allowUnorderedAccess) missing = "unordered access view";
            break;
        default:
            break;
    }
    if (missing != nullptr) {
        if (error != nullptr) {
            *error = "pass '" + passName + "' uses '" + resource.desc.name + "' as " +
                     AccessName(access) + ", but it has no " + missing;
        }
        return false;
    }
    return true;
}

bool RenderGraph::Compile(std::string* error) {
    m_plan = {};
    m_compiled = false;

    // An import carries content produced outside the graph, and a resource
    // declared persistent carries last frame's, so neither counts as unwritten.
    for (Resource& resource : m_resources) {
        resource.writtenThisFrame = resource.imported || resource.desc.persistentContent;
    }

    // Worked on a copy so a Compile() that fails half way does not leave the
    // tracked states describing a frame that was never recorded.
    std::vector<D3D12_RESOURCE_STATES> states(m_resources.size());
    for (std::size_t i = 0; i < m_resources.size(); ++i) {
        states[i] = m_resources[i].state;
    }

    for (std::size_t passIndex = 0; passIndex < m_passes.size(); ++passIndex) {
        const RenderPassDesc& pass = m_passes[passIndex];
        PassPlan plan;
        plan.passIndex = passIndex;
        plan.name = pass.name;

        // One required state per resource for this pass, in first-mention order.
        std::vector<std::pair<std::uint32_t, D3D12_RESOURCE_STATES>> required;

        const auto require = [&](const PassAccess& accessed, bool isWrite) -> bool {
            if (!accessed.resource.IsValid() ||
                accessed.resource.index >= m_resources.size()) {
                if (error != nullptr) {
                    *error = "pass '" + pass.name + "' names a resource handle that is not "
                             "part of this graph";
                }
                return false;
            }
            Resource& resource = m_resources[accessed.resource.index];
            if (!ValidateAccess(resource, accessed.access, pass.name, error)) {
                return false;
            }
            if (!isWrite && !resource.writtenThisFrame) {
                if (error != nullptr) {
                    *error = "pass '" + pass.name + "' reads '" + resource.desc.name +
                             "', which no earlier pass in this frame wrote";
                }
                return false;
            }
            const D3D12_RESOURCE_STATES wanted = StateForAccess(accessed.access);
            for (const auto& entry : required) {
                if (entry.first != accessed.resource.index) {
                    continue;
                }
                if (entry.second != wanted) {
                    if (error != nullptr) {
                        *error = "pass '" + pass.name + "' needs '" + resource.desc.name +
                                 "' in two states at once: " +
                                 std::string(StateName(entry.second)) + " and " +
                                 StateName(wanted);
                    }
                    return false;
                }
                return true;  // same state asked for twice -- harmless.
            }
            required.emplace_back(accessed.resource.index, wanted);
            return true;
        };

        for (const PassAccess& read : pass.reads) {
            if (!require(read, false)) {
                return false;
            }
        }
        for (const PassAccess& write : pass.writes) {
            if (!require(write, true)) {
                return false;
            }
        }

        for (const auto& entry : required) {
            if (states[entry.first] == entry.second) {
                continue;
            }
            BarrierPlan barrier;
            barrier.resource = ResourceHandle{entry.first};
            barrier.before = states[entry.first];
            barrier.after = entry.second;
            plan.barriers.push_back(barrier);
            states[entry.first] = entry.second;
        }

        for (const PassAccess& write : pass.writes) {
            if (write.access == Access::RenderTarget) {
                plan.renderTargets.push_back(write.resource);
            } else if (IsDepthAccess(write.access)) {
                if (plan.depthStencil.has_value()) {
                    if (error != nullptr) {
                        *error = "pass '" + pass.name + "' writes two depth targets";
                    }
                    return false;
                }
                plan.depthStencil = write.resource;
            }
        }
        if (plan.renderTargets.size() > D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT) {
            if (error != nullptr) {
                *error = "pass '" + pass.name + "' writes " +
                         std::to_string(plan.renderTargets.size()) +
                         " render targets; the limit is " +
                         std::to_string(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT);
            }
            return false;
        }

        // A clear is a write: clearing a resource the pass did not declare would
        // escape the barrier tracking entirely.
        for (const ResourceHandle cleared : pass.clears) {
            bool declared = false;
            for (const PassAccess& write : pass.writes) {
                if (write.resource == cleared) {
                    declared = true;
                    break;
                }
            }
            if (!declared) {
                if (error != nullptr) {
                    const std::string name =
                        cleared.index < m_resources.size()
                            ? m_resources[cleared.index].desc.name
                            : std::string("<unknown>");
                    *error = "pass '" + pass.name + "' clears '" + name +
                             "' without listing it as a write";
                }
                return false;
            }
        }

        for (const PassAccess& write : pass.writes) {
            m_resources[write.resource.index].writtenThisFrame = true;
        }

        m_plan.passes.push_back(std::move(plan));
    }

    for (std::size_t i = 0; i < m_resources.size(); ++i) {
        const Resource& resource = m_resources[i];
        if (!resource.finalState.has_value() || states[i] == *resource.finalState) {
            continue;
        }
        BarrierPlan barrier;
        barrier.resource = ResourceHandle{static_cast<std::uint32_t>(i)};
        barrier.before = states[i];
        barrier.after = *resource.finalState;
        m_plan.epilogue.push_back(barrier);
        states[i] = *resource.finalState;
    }

    for (std::size_t i = 0; i < m_resources.size(); ++i) {
        m_resources[i].state = states[i];
    }
    m_compiled = true;
    return true;
}

bool RenderGraph::Execute(ID3D12GraphicsCommandList* commandList, std::string* error) {
    if (!m_compiled) {
        if (error != nullptr) {
            *error = "Execute() called before a successful Compile()";
        }
        return false;
    }
    if (commandList == nullptr) {
        if (error != nullptr) {
            *error = "Execute() called with no command list";
        }
        return false;
    }

    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    const auto issue = [&](const std::vector<BarrierPlan>& planned) -> bool {
        barriers.clear();
        barriers.reserve(planned.size());
        for (const BarrierPlan& barrier : planned) {
            ID3D12Resource* resource = ResourcePointer(m_resources[barrier.resource.index]);
            if (resource == nullptr) {
                if (error != nullptr) {
                    *error = "'" + m_resources[barrier.resource.index].desc.name +
                             "' has no D3D12 resource; was Resize() called after SetDevice()?";
                }
                return false;
            }
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(resource, barrier.before,
                                                                    barrier.after));
        }
        if (!barriers.empty()) {
            commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
        }
        return true;
    };

    for (const PassPlan& plan : m_plan.passes) {
        const RenderPassDesc& pass = m_passes[plan.passIndex];
        if (!issue(plan.barriers)) {
            return false;
        }

        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, "%s", pass.name.c_str());

        // The viewport follows the pass's own targets rather than the swapchain,
        // so a half-resolution pass does not silently render into a quarter of
        // its target.
        std::uint32_t passWidth = m_width;
        std::uint32_t passHeight = m_height;
        const ResourceHandle sizeSource =
            !plan.renderTargets.empty()
                ? plan.renderTargets.front()
                : (plan.depthStencil.has_value() ? *plan.depthStencil : ResourceHandle{});
        if (sizeSource.IsValid()) {
            const Resource& resource = m_resources[sizeSource.index];
            if (resource.width != 0 && resource.height != 0) {
                passWidth = resource.width;
                passHeight = resource.height;
            }
        }
        const PassResources resources(*this, passWidth, passHeight);

        if (pass.bindRenderTargets) {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandles[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
            for (std::size_t i = 0; i < plan.renderTargets.size(); ++i) {
                rtvHandles[i] = m_resources[plan.renderTargets[i].index].rtv;
                if (!HasView(rtvHandles[i])) {
                    if (error != nullptr) {
                        *error = "pass '" + pass.name + "' binds '" +
                                 m_resources[plan.renderTargets[i].index].desc.name +
                                 "' as a render target, but it has no RTV";
                    }
                    return false;
                }
            }
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = {};
            if (plan.depthStencil.has_value()) {
                dsvHandle = m_resources[plan.depthStencil->index].dsv;
                if (!HasView(dsvHandle)) {
                    if (error != nullptr) {
                        *error = "pass '" + pass.name + "' binds '" +
                                 m_resources[plan.depthStencil->index].desc.name +
                                 "' as a depth target, but it has no DSV";
                    }
                    return false;
                }
            }
            commandList->OMSetRenderTargets(static_cast<UINT>(plan.renderTargets.size()),
                                            plan.renderTargets.empty() ? nullptr : rtvHandles,
                                            FALSE,
                                            plan.depthStencil.has_value() ? &dsvHandle : nullptr);

            const D3D12_VIEWPORT viewport = resources.Viewport();
            const D3D12_RECT scissor = resources.ScissorRect();
            commandList->RSSetViewports(1, &viewport);
            commandList->RSSetScissorRects(1, &scissor);
        }

        // After binding, so a cleared target is the one just bound.
        for (const ResourceHandle cleared : pass.clears) {
            const Resource& resource = m_resources[cleared.index];
            Access access = Access::RenderTarget;
            for (const PassAccess& write : pass.writes) {
                if (write.resource == cleared) {
                    access = write.access;
                    break;
                }
            }
            if (IsDepthAccess(access)) {
                commandList->ClearDepthStencilView(resource.dsv, D3D12_CLEAR_FLAG_DEPTH,
                                                   resource.desc.clearDepth,
                                                   resource.desc.clearStencil, 0, nullptr);
            } else {
                commandList->ClearRenderTargetView(resource.rtv, resource.desc.clearColor, 0,
                                                   nullptr);
            }
        }

        if (pass.execute) {
            pass.execute(commandList, resources);
        }

        PIXEndEvent(commandList);
    }

    return issue(m_plan.epilogue);
}

std::string RenderGraph::DescribePlan() const {
    std::ostringstream out;
    if (!m_compiled) {
        out << "(not compiled)\n";
        return out.str();
    }
    for (const PassPlan& plan : m_plan.passes) {
        out << "pass " << plan.passIndex << " " << plan.name << "\n";
        for (const BarrierPlan& barrier : plan.barriers) {
            out << "    barrier " << ResourceName(barrier.resource) << " "
                << StateName(barrier.before) << " -> " << StateName(barrier.after) << "\n";
        }
        for (const ResourceHandle target : plan.renderTargets) {
            out << "    rtv     " << ResourceName(target) << "\n";
        }
        if (plan.depthStencil.has_value()) {
            out << "    dsv     " << ResourceName(*plan.depthStencil) << "\n";
        }
    }
    for (const BarrierPlan& barrier : m_plan.epilogue) {
        out << "epilogue barrier " << ResourceName(barrier.resource) << " "
            << StateName(barrier.before) << " -> " << StateName(barrier.after) << "\n";
    }
    return out.str();
}

const std::string& RenderGraph::ResourceName(ResourceHandle handle) const {
    static const std::string unknown = "<unknown>";
    if (handle.index >= m_resources.size()) {
        return unknown;
    }
    return m_resources[handle.index].desc.name;
}

void RenderGraph::Reset() {
    // m_device is cleared along with the allocators so the invariant
    // "allocators exist iff m_device != nullptr" holds. Without it, a device
    // recreated at the same address after a device-lost would make SetDevice
    // below take its early return and leave the allocators null.
    m_device = nullptr;
    m_passes.clear();
    m_plan = {};
    m_compiled = false;
    m_resources.clear();
    m_declaredCount = 0;
    m_rtvAllocator.reset();
    m_dsvAllocator.reset();
    m_srvAllocator.reset();
}

ID3D12Resource* RenderGraph::ResourcePointer(const Resource& resource) const {
    return resource.imported ? resource.imported_resource : resource.owned.Get();
}

void RenderGraph::CreateTexture(Resource& resource) {
    const TextureDesc& desc = resource.desc;
    std::uint32_t width = m_width;
    std::uint32_t height = m_height;
    if (desc.sizeMode == SizeMode::Absolute) {
        width = desc.width;
        height = desc.height;
    } else {
        width = static_cast<std::uint32_t>(
            std::max(1.0f, static_cast<float>(m_width) * desc.widthScale + 0.5f));
        height = static_cast<std::uint32_t>(
            std::max(1.0f, static_cast<float>(m_height) * desc.heightScale + 0.5f));
    }
    resource.width = width;
    resource.height = height;

    D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        desc.format, width, height, 1, 1);
    if (desc.allowRenderTarget) {
        resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if (desc.allowDepthStencil) {
        resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    }
    if (desc.allowUnorderedAccess) {
        resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if (!desc.allowShaderResource) {
        resourceDesc.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    }

    // A committed resource may only be given an optimized clear value for a
    // format it can actually be cleared to, and that format must be TYPED even
    // when the resource itself is typeless.
    D3D12_CLEAR_VALUE clearValue = {};
    const D3D12_CLEAR_VALUE* clearValuePointer = nullptr;
    if (desc.allowDepthStencil) {
        clearValue.Format = Coalesce(desc.depthStencilFormat, desc.format);
        clearValue.DepthStencil.Depth = desc.clearDepth;
        clearValue.DepthStencil.Stencil = desc.clearStencil;
        clearValuePointer = &clearValue;
    } else if (desc.allowRenderTarget) {
        clearValue.Format = Coalesce(desc.renderTargetFormat, desc.format);
        std::memcpy(clearValue.Color, desc.clearColor, sizeof(clearValue.Color));
        clearValuePointer = &clearValue;
    }

    const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);
    resource.owned.Reset();
    DX::ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc, desc.initialState,
        clearValuePointer, IID_PPV_ARGS(resource.owned.ReleaseAndGetAddressOf())));
    resource.state = desc.initialState;

    const std::wstring wideName(desc.name.begin(), desc.name.end());
    resource.owned->SetName(wideName.c_str());
}

void RenderGraph::CreateViews(Resource& resource) {
    const TextureDesc& desc = resource.desc;
    ID3D12Resource* raw = resource.owned.Get();

    // Each allocation is made once and the view rewritten in place on every
    // resize. A descriptor is a rewritable slot, and freeing one here would
    // only mark it stale -- nothing calls ReleaseStaleDescriptors for these
    // allocators, so a window dragged to a new size would walk through pages.
    if (desc.allowRenderTarget) {
        if (resource.rtvAllocation.IsNull()) {
            resource.rtvAllocation = m_rtvAllocator->Allocate(1);
        }
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
        rtvDesc.Format = Coalesce(desc.renderTargetFormat, desc.format);
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        resource.rtv = resource.rtvAllocation.GetDescriptorHandle();
        m_device->CreateRenderTargetView(raw, &rtvDesc, resource.rtv);
    }
    if (desc.allowDepthStencil) {
        if (resource.dsvAllocation.IsNull()) {
            resource.dsvAllocation = m_dsvAllocator->Allocate(1);
        }
        D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
        dsvDesc.Format = Coalesce(desc.depthStencilFormat, desc.format);
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        resource.dsv = resource.dsvAllocation.GetDescriptorHandle();
        m_device->CreateDepthStencilView(raw, &dsvDesc, resource.dsv);
    }
    if (desc.allowShaderResource) {
        if (resource.srvAllocation.IsNull()) {
            resource.srvAllocation = m_srvAllocator->Allocate(1);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        // For a depth target this is where the typeless resource format pays
        // off: the SRV asks for the single-channel typed form of the same bits.
        srvDesc.Format = Coalesce(desc.shaderResourceFormat, desc.format);
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;
        resource.srv = resource.srvAllocation.GetDescriptorHandle();
        m_device->CreateShaderResourceView(raw, &srvDesc, resource.srv);
    }
}

}  // namespace NeuralModelIntegrateTestbed::Render
