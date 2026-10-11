#pragma once

// The G-buffer layout, in one place because three things have to agree on it:
//
//   * struct GBufferOutput in Engine/shaders/no_texture.fx, which writes it
//   * the geometry PSO in GLTFAdapter::PreparePSO, whose RTVFormats must match
//     the targets the render graph will bind
//   * DeferredRenderer, which declares the targets and the lighting pass's
//     SRV table
//
// Mismatched RTV formats between a PSO and the bound targets is a validation
// error rather than a silent one, so this is mostly here to keep the three
// edits together -- and to put the reasoning for each format somewhere that is
// not a shader comment.

#include <cstddef>

#include <directx/d3d12.h>

namespace NeuralModelIntegrateTestbed {

// Colour targets only. Depth is DeviceResources' depth buffer, read back
// through an SRV rather than duplicated here.
enum class GBufferTarget : std::size_t {
    BaseColor = 0,  // rgb base colour (linear), a = alpha
    Normal,         // world-space shading normal, after normal mapping
    Material,       // r metallic, g roughness, b occlusion
    Emissive,       // rgb emissive
    Count,
};

constexpr std::size_t kGBufferTargetCount = static_cast<std::size_t>(GBufferTarget::Count);

// The lighting pass's SRV table is this wide: the colour targets plus depth,
// which is why it is not simply kGBufferTargetCount.
constexpr std::size_t kGBufferSrvCount = kGBufferTargetCount + 1;

// Slot of the depth SRV within that table. Matches register t9 in
// no_texture.fx, whose table starts at t5.
constexpr std::size_t kGBufferDepthSrvSlot = kGBufferTargetCount;

// Per-target formats.
//
//   BaseColor  8-bit UNORM is enough for an albedo and the cheapest thing that
//              is. Deliberately NOT _SRGB -- see the note in no_texture.fx.
//   Normal     16-bit float. An 8-bit UNORM normal bands visibly across a
//              smooth specular highlight, which is the one place this renderer
//              looks at normals closely. Octahedral packing into R16G16 would
//              halve this if bandwidth ever matters.
//   Material   three independent [0,1] scalars; 8 bits each is standard.
//   Emissive   R11G11B10_FLOAT, because emissive is the one channel that is
//              legitimately allowed above 1.0 and UNORM would clamp it.
inline DXGI_FORMAT GBufferFormat(GBufferTarget target) {
    switch (target) {
        case GBufferTarget::BaseColor: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case GBufferTarget::Normal:    return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case GBufferTarget::Material:  return DXGI_FORMAT_R8G8B8A8_UNORM;
        case GBufferTarget::Emissive:  return DXGI_FORMAT_R11G11B10_FLOAT;
        case GBufferTarget::Count:     break;
    }
    return DXGI_FORMAT_UNKNOWN;
}

inline const char* GBufferTargetName(GBufferTarget target) {
    switch (target) {
        case GBufferTarget::BaseColor: return "GBufferBaseColor";
        case GBufferTarget::Normal:    return "GBufferNormal";
        case GBufferTarget::Material:  return "GBufferMaterial";
        case GBufferTarget::Emissive:  return "GBufferEmissive";
        case GBufferTarget::Count:     break;
    }
    return "GBufferUnknown";
}

}  // namespace NeuralModelIntegrateTestbed
