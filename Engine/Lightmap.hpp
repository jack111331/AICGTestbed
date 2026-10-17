#pragma once

// Packing baked per-surface results into one atlas.
//
// A shadow map is indexed by the LIGHT's view; a lightmap is indexed by the
// RECEIVER's surface. That is the whole difference, and it is what makes a
// lightmap view-independent: once baked, a surface's shadow is a property of
// the surface rather than of where the camera happens to be.
//
// Addressing a surface needs a parameterisation of it -- a UV set that covers
// the surface without overlapping. glTF's TEXCOORD_0 will not do: it is a
// texturing UV, and on a converted architectural scene it tiles, so it
// addresses the same texel from many different places on the mesh. That is why
// this consumes TEXCOORD_1 (or an unwrapper's output) and not TEXCOORD_0.
//
// Pure arithmetic, no D3D12, so //Engine:scenegraph-smoke can assert on it.
// Packing is the part most worth testing: two charts overlapping by one texel
// produces a scene where one object wears another's shadow, which looks like a
// baking bug rather than an allocation bug.
//
// Charts are per INSTANCE, not per mesh. The same mesh in two places receives
// two different shadows -- the Bistro instances 2909 nodes from 551 meshes,
// one of them 84 times -- so a chart layout can be shared but a chart cannot.

#include <cstdint>
#include <vector>

#include <directx/d3d12.h>

#include <DirectXMath.h>

namespace NeuralModelIntegrateTestbed {

// Shared so the renderer can declare the atlas before a scene is loaded and
// the packer can size charts against the same number. Changing it changes a
// resource declaration, which is why it is a constant rather than a setting.
constexpr std::uint32_t kLightmapAtlasResolution = 4096;

// The depth map the bake is tested against: the whole scene from the sun, once.
// 2048 over the Bistro's 166-unit diagonal is 8 cm a texel, finer than the
// 12.5 cm lightmap texels it feeds, so the bake is limited by the atlas rather
// than by this.
constexpr std::uint32_t kLightmapSunDepthResolution = 2048;

// One channel: a shadow term is a single number. 8 bits gives 256 levels of
// penumbra, which is more than a 5x5 kernel over the sun map can distinguish.
constexpr DXGI_FORMAT kLightmapFormat = DXGI_FORMAT_R8_UNORM;

struct LightmapSetup {
    // Texels per world metre of surface. The quality dial: 8 gives 12.5 cm
    // texels, which is between the cascades' 1.6 cm near and 24 cm far.
    float texelsPerMetre = 8.0f;

    // Side of the square atlas, in texels.
    std::uint32_t atlasResolution = kLightmapAtlasResolution;

    // Chart edges are clamped to this range and rounded to a power of two.
    // Powers of two are not an aesthetic choice: they let charts of equal size
    // tile a row exactly, so the packer below wastes nothing inside a row.
    std::uint32_t minChartEdge = 4;
    std::uint32_t maxChartEdge = 256;

    // Texels of guard band around each chart's content. A bilinear tap at a
    // chart's edge reaches one texel past it, so without a border it reads the
    // neighbouring chart -- the classic lightmap seam.
    std::uint32_t padding = 2;
};

struct LightmapChart {
    // Placement in the atlas, in texels, including the padding border.
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t size = 0;

    // Maps a [0,1] lightmap UV into this chart's interior: uv * xy + zw.
    // Already inset by the padding, so a UV of exactly 0 or 1 still lands
    // inside the chart's own texels.
    DirectX::XMFLOAT4 scaleOffset = {0.0f, 0.0f, 0.0f, 0.0f};

    bool placed = false;
};

struct LightmapAllocation {
    // Parallel to the areas handed to PackLightmap.
    std::vector<LightmapChart> charts;

    std::uint32_t atlasResolution = 0;
    // Texels inside placed charts, padding included. The atlas is
    // atlasResolution^2, so the difference is what packing wasted.
    std::uint64_t texelsPlaced = 0;
    std::size_t placedCount = 0;
    // False when something did not fit. The caller then has the choice of a
    // bigger atlas or fewer texels per metre; what it must NOT do is carry on,
    // because an unplaced chart has no valid UVs.
    bool complete = false;
};

// `instanceAreas` is each instance's world-space surface area in square
// metres. Order is preserved: charts[i] belongs to instanceAreas[i].
//
// An area of zero or less gets no chart -- a degenerate primitive has no
// surface to bake -- and reports placed == false without failing the pack.
LightmapAllocation PackLightmap(const std::vector<float>& instanceAreas,
                                const LightmapSetup& setup);

// The chart edge, in texels, that an area of this size earns: the clamped,
// power-of-two-rounded result of sqrt(area) * texelsPerMetre, plus padding on
// both sides. Exposed because sizing and packing are worth testing apart.
std::uint32_t LightmapChartEdge(float area, const LightmapSetup& setup);

}  // namespace NeuralModelIntegrateTestbed
