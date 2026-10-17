#include "pch.h"

#include "Lightmap.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace NeuralModelIntegrateTestbed {

namespace {

std::uint32_t RoundUpToPowerOfTwo(std::uint32_t value) {
    std::uint32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

std::uint32_t RoundDownToPowerOfTwo(std::uint32_t value) {
    std::uint32_t result = 1;
    while (result * 2 <= value) {
        result *= 2;
    }
    return result;
}

}  // namespace

std::uint32_t LightmapChartEdge(float area, const LightmapSetup& setup) {
    if (!(area > 0.0f) || setup.texelsPerMetre <= 0.0f) {
        return 0;
    }

    // sqrt(area) is the edge of the square with this area, so scaling it by
    // texels per metre gives the texels that edge deserves. A chart is kept
    // square: the real surface is not, but its UV chart has already been
    // normalised to [0,1] in both axes by whatever unwrapped it.
    const float wanted = std::sqrt(area) * setup.texelsPerMetre;

    // The guard band sits OUTSIDE the content, so the chart is wider than what
    // it holds -- and the clamp has to be applied to the chart, not to the
    // content, or a capped chart comes out larger than the cap and can
    // overflow an atlas that was sized against it.
    const std::uint32_t border = 2 * setup.padding;
    const std::uint32_t smallest =
        RoundUpToPowerOfTwo(std::max(setup.minChartEdge, border + 1));
    const std::uint32_t largest =
        std::max(smallest, RoundDownToPowerOfTwo(setup.maxChartEdge));

    const std::uint32_t content = static_cast<std::uint32_t>(
        std::lround(std::max(wanted, 1.0f)));
    // Powers of two throughout, because that is what lets a row of equal
    // charts tile the atlas exactly.
    return std::clamp(RoundUpToPowerOfTwo(content + border), smallest, largest);
}

LightmapAllocation PackLightmap(const std::vector<float>& instanceAreas,
                                const LightmapSetup& setup) {
    LightmapAllocation out;
    out.charts.resize(instanceAreas.size());
    out.atlasResolution = setup.atlasResolution;
    if (setup.atlasResolution == 0) {
        return out;
    }

    // Sized first, then placed largest-first. Grouping by size is what makes
    // this tight: every chart in a row is the same power of two, so a row
    // divides exactly and the only waste is the tail of the last row of each
    // size and whatever is left at the bottom.
    struct Entry {
        std::size_t index;
        std::uint32_t edge;
    };
    std::vector<Entry> entries;
    entries.reserve(instanceAreas.size());
    for (std::size_t i = 0; i < instanceAreas.size(); ++i) {
        const std::uint32_t edge = LightmapChartEdge(instanceAreas[i], setup);
        if (edge == 0 || edge > setup.atlasResolution) {
            // No surface to bake, or a chart that cannot fit the atlas at all
            // even alone. Both leave placed == false; only the second is a
            // failure, and it is reported by `complete` below.
            continue;
        }
        entries.push_back({i, edge});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.edge != b.edge) {
            return a.edge > b.edge;
        }
        return a.index < b.index;  // stable, so a pack is reproducible
    });

    std::uint32_t shelfY = 0;       // top of the row being filled
    std::uint32_t shelfHeight = 0;  // its height, which is its charts' edge
    std::uint32_t cursorX = 0;

    bool overflowed = false;
    for (const Entry& entry : entries) {
        if (shelfHeight != entry.edge || cursorX + entry.edge > setup.atlasResolution) {
            // Either the size changed or this row is full: start a new one.
            shelfY += shelfHeight;
            shelfHeight = entry.edge;
            cursorX = 0;
        }
        if (shelfY + entry.edge > setup.atlasResolution) {
            overflowed = true;
            continue;
        }

        LightmapChart& chart = out.charts[entry.index];
        chart.x = cursorX;
        chart.y = shelfY;
        chart.size = entry.edge;
        chart.placed = true;

        // The content sits inside the guard band, so a UV of 0 maps to the
        // first content texel and 1 to the last.
        const float inverseAtlas = 1.0f / static_cast<float>(setup.atlasResolution);
        const float content =
            static_cast<float>(entry.edge - 2 * setup.padding);
        chart.scaleOffset = DirectX::XMFLOAT4(
            content * inverseAtlas, content * inverseAtlas,
            static_cast<float>(cursorX + setup.padding) * inverseAtlas,
            static_cast<float>(shelfY + setup.padding) * inverseAtlas);

        cursorX += entry.edge;
        out.texelsPlaced += static_cast<std::uint64_t>(entry.edge) * entry.edge;
        ++out.placedCount;
    }

    // Every instance with surface either got a chart or did not fit. An area of
    // zero is not a failure: there is nothing to bake for it.
    std::size_t wanted = 0;
    for (const float area : instanceAreas) {
        if (LightmapChartEdge(area, setup) != 0) {
            ++wanted;
        }
    }
    out.complete = !overflowed && out.placedCount == wanted;
    return out;
}

}  // namespace NeuralModelIntegrateTestbed
