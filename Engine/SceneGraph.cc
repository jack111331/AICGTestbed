#include "pch.h"

#include "SceneGraph.hpp"

#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <variant>

namespace NeuralModelIntegrateTestbed {

namespace {

// fastgltf follows glTF/GLM: fmat4x4 stores columns (m[c][r]) and composes as
// parent * child. DirectXMath stores rows (m[r][c]) and composes as
// child * parent. Writing column c of the source into row c of the destination
// is the transpose, which converts between both conventions at once -- so a
// chain built with fastgltf's operator* comes out correctly ordered for
// row-vector multiplication in the shader.
DirectX::SimpleMath::Matrix ToSimpleMath(const fastgltf::math::fmat4x4& m) {
    DirectX::SimpleMath::Matrix out;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out.m[c][r] = m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)];
        }
    }
    return out;
}

// fastgltf orders its enum Directional/Spot/Point; the shader numbers them
// Directional/Point/Spot. Spelled out so neither can drift into the other.
LightType FromFastgltf(fastgltf::LightType type) {
    switch (type) {
        case fastgltf::LightType::Directional: return LightType::Directional;
        case fastgltf::LightType::Point:       return LightType::Point;
        case fastgltf::LightType::Spot:        return LightType::Spot;
    }
    return LightType::Directional;
}

// True when an accessor's bytes are actually reachable on the CPU.
// fastgltf's DefaultBufferDataAdapter only handles the decoded sources; for a
// sources::URI that was never loaded it asserts in debug and hands back an
// empty span in release, which the subsequent subspan then walks off the end
// of. Checking first turns a crash into substituted identities.
bool AccessorBytesAvailable(const fastgltf::Asset& asset, std::size_t accessorIndex) {
    if (accessorIndex >= asset.accessors.size()) {
        return false;
    }
    const fastgltf::Accessor& accessor = asset.accessors[accessorIndex];
    if (!accessor.bufferViewIndex.has_value() ||
        accessor.bufferViewIndex.value() >= asset.bufferViews.size()) {
        return false;
    }
    const fastgltf::BufferView& view = asset.bufferViews[accessor.bufferViewIndex.value()];
    if (view.bufferIndex >= asset.buffers.size()) {
        return false;
    }
    const auto& data = asset.buffers[view.bufferIndex].data;
    return std::holds_alternative<fastgltf::sources::Array>(data) ||
           std::holds_alternative<fastgltf::sources::Vector>(data) ||
           std::holds_alternative<fastgltf::sources::ByteView>(data);
}

// Whether a mesh declares the paired attributes a skinned primitive needs.
// JOINTS_n without its WEIGHTS_n is malformed glTF: every weight then reads as
// zero and the mesh collapses towards the origin, which looks like a transform
// bug rather than a missing attribute.
// Records the format a skinning attribute arrived in. The first one wins,
// because the input layout is baked into a single PSO; a later primitive in a
// different format is reported rather than silently reinterpreted, which would
// read joint indices as the wrong width and scatter vertices across the rig.
void NoteSkinningFormat(DXGI_FORMAT found, DXGI_FORMAT& slot, bool& seen,
                        std::vector<SkinDiagnostic>& diagnostics) {
    if (found == DXGI_FORMAT_UNKNOWN) {
        diagnostics.push_back({SkinIssue::MixedJointFormats, 0, std::nullopt});
        return;
    }
    if (!seen) {
        slot = found;
        seen = true;
        return;
    }
    if (slot != found) {
        diagnostics.push_back({SkinIssue::MixedJointFormats, 0, std::nullopt});
    }
}

void CheckSkinningAttributes(const fastgltf::Asset& asset,
                             std::size_t meshIndex,
                             std::size_t skinIndex,
                             std::size_t nodeIndex,
                             std::vector<SkinDiagnostic>& out) {
    if (meshIndex >= asset.meshes.size()) {
        return;
    }
    bool joints = false;
    bool weights = false;
    for (const fastgltf::Primitive& primitive : asset.meshes[meshIndex].primitives) {
        if (primitive.findAttribute("JOINTS_0") != primitive.attributes.end()) {
            joints = true;
        }
        if (primitive.findAttribute("WEIGHTS_0") != primitive.attributes.end()) {
            weights = true;
        }
    }
    if (!joints) {
        out.push_back({SkinIssue::PrimitiveMissingJoints, skinIndex, nodeIndex});
    }
    if (!weights) {
        out.push_back({SkinIssue::PrimitiveMissingWeights, skinIndex, nodeIndex});
    }
}

}  // namespace

const char* ToString(AnimationPath path) {
    switch (path) {
        case AnimationPath::Translation: return "translation";
        case AnimationPath::Rotation:    return "rotation";
        case AnimationPath::Scale:       return "scale";
        case AnimationPath::Weights:     return "weights";
    }
    return "unknown";
}

const char* ToString(AnimationInterpolation interpolation) {
    switch (interpolation) {
        case AnimationInterpolation::Linear:      return "LINEAR";
        case AnimationInterpolation::Step:        return "STEP";
        case AnimationInterpolation::CubicSpline: return "CUBICSPLINE";
    }
    return "unknown";
}

DirectX::SimpleMath::Matrix NodeTransform::ToMatrix() const {
    // Row-vector order. glTF defines the local transform as T * R * S for
    // column vectors; reversed for row vectors that is S * R * T, which is what
    // DirectXMath's multiply order below produces. Built directly rather than
    // going through fastgltf, so there is NO transpose here -- unlike
    // ToSimpleMath, which converts fastgltf's column-major output.
    return DirectX::SimpleMath::Matrix::CreateScale(scale) *
           DirectX::SimpleMath::Matrix::CreateFromQuaternion(rotation) *
           DirectX::SimpleMath::Matrix::CreateTranslation(translation);
}

bool SampleAnimation(const AnimationSampler& sampler, float time,
                     DirectX::XMFLOAT4& out) {
    const std::size_t keyCount = sampler.KeyCount();
    if (keyCount == 0 || sampler.values.empty()) {
        return false;
    }

    // CubicSpline stores in-tangent, value, out-tangent per key, so the value
    // for key i lives at 3i+1.
    const bool cubic = sampler.interpolation == AnimationInterpolation::CubicSpline;
    const auto valueAt = [&](std::size_t key) -> DirectX::XMFLOAT4 {
        const std::size_t index = cubic ? key * 3 + 1 : key;
        return index < sampler.values.size() ? sampler.values[index]
                                             : sampler.values.back();
    };

    // glTF clamps outside the keyframe range rather than extrapolating.
    if (keyCount == 1 || time <= sampler.times.front()) {
        out = valueAt(0);
        return true;
    }
    if (time >= sampler.times.back()) {
        out = valueAt(keyCount - 1);
        return true;
    }

    // First key strictly after `time`; the interval is [upper-1, upper].
    const auto upper = std::upper_bound(sampler.times.begin(), sampler.times.end(), time);
    const std::size_t next = static_cast<std::size_t>(upper - sampler.times.begin());
    const std::size_t prev = next - 1;

    const float t0 = sampler.times[prev];
    const float t1 = sampler.times[next];
    const float span = t1 - t0;
    // Equal timestamps are legal input and would divide by zero.
    const float u = span > 0.0f ? (time - t0) / span : 0.0f;

    if (sampler.interpolation == AnimationInterpolation::Step) {
        out = valueAt(prev);
        return true;
    }

    const DirectX::XMFLOAT4 a = valueAt(prev);
    const DirectX::XMFLOAT4 b = valueAt(next);

    if (cubic) {
        // Cubic Hermite, as the glTF specification writes it. m0 is the
        // out-tangent of the previous key, m1 the in-tangent of the next, and
        // both are scaled by the interval -- omitting that scale is the usual
        // mistake and gives tangents that are wrong by a factor of the span.
        const DirectX::XMFLOAT4 m0 = sampler.values[prev * 3 + 2];
        const DirectX::XMFLOAT4 m1 = sampler.values[next * 3 + 0];
        const float u2 = u * u;
        const float u3 = u2 * u;
        const float h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
        const float h10 = u3 - 2.0f * u2 + u;
        const float h01 = -2.0f * u3 + 3.0f * u2;
        const float h11 = u3 - u2;
        const auto mix = [&](float pa, float ta, float pb, float tb) {
            return h00 * pa + h10 * span * ta + h01 * pb + h11 * span * tb;
        };
        out.x = mix(a.x, m0.x, b.x, m1.x);
        out.y = mix(a.y, m0.y, b.y, m1.y);
        out.z = mix(a.z, m0.z, b.z, m1.z);
        out.w = mix(a.w, m0.w, b.w, m1.w);
        if (sampler.componentCount == 4) {
            // A splined quaternion is not unit length; the spec says normalise.
            DirectX::XMStoreFloat4(
                &out, DirectX::XMQuaternionNormalize(DirectX::XMLoadFloat4(&out)));
        }
        return true;
    }

    // Linear.
    if (sampler.componentCount == 4) {
        // Rotations must be slerped. A component-wise lerp of two quaternions
        // shortens the arc and changes the rotation speed, which reads as a
        // joint "snapping" mid-interval rather than as an obvious error.
        // XMQuaternionSlerp already takes the shortest path.
        const DirectX::XMVECTOR qa = DirectX::XMLoadFloat4(&a);
        const DirectX::XMVECTOR qb = DirectX::XMLoadFloat4(&b);
        DirectX::XMStoreFloat4(&out, DirectX::XMQuaternionSlerp(qa, qb, u));
        return true;
    }

    out.x = a.x + (b.x - a.x) * u;
    out.y = a.y + (b.y - a.y) * u;
    out.z = a.z + (b.z - a.z) * u;
    out.w = a.w + (b.w - a.w) * u;
    return true;
}

const char* ToString(SkinIssue issue) {
    switch (issue) {
        case SkinIssue::NoInverseBindMatrices:
            return "skin declares no inverseBindMatrices; using identities";
        case SkinIssue::InverseBindMatricesUnreadable:
            return "inverseBindMatrices accessor is not readable; using identities";
        case SkinIssue::InverseBindMatrixCountMismatch:
            return "inverseBindMatrices count does not match joint count; using identities";
        case SkinIssue::JointIndexOutOfRange:
            return "joint index out of range; joint dropped";
        case SkinIssue::SkinnedNodeWithoutMesh:
            return "node has a skin but no mesh";
        case SkinIssue::PrimitiveMissingJoints:
            return "skinned mesh has no JOINTS_0 attribute";
        case SkinIssue::PrimitiveMissingWeights:
            return "skinned mesh has no WEIGHTS_0 attribute; skinning disabled, "
                   "mesh stays in its bind pose";
        case SkinIssue::TooManyJoints:
            return "skin has more joints than the shader array holds; extra joints "
                   "clamped";
        case SkinIssue::MixedJointFormats:
            return "skinning attribute format is unsupported or inconsistent across "
                   "primitives";
    }
    return "unknown skin issue";
}

ShaderLight ToShaderLight(const SceneLight& light) {
    ShaderLight out;
    out.position = DirectX::XMFLOAT3(light.position.x, light.position.y, light.position.z);
    // glTF treats a missing range as infinite. 0 is the sentinel for that
    // because a real range is always positive, so the shader can test `> 0`
    // without needing a second flag.
    out.range = light.range.value_or(0.0f);
    out.direction = DirectX::XMFLOAT3(light.direction.x, light.direction.y, light.direction.z);
    out.intensity = light.intensity;
    out.color = DirectX::XMFLOAT3(light.color.x, light.color.y, light.color.z);
    out.type = static_cast<int>(light.type);
    out.innerConeCos = std::cos(light.innerConeAngle);
    out.outerConeCos = std::cos(light.outerConeAngle);
    // cos is decreasing, so a wider outer cone gives the *smaller* cosine:
    // inner >= outer always holds for a well-formed spot light.
    return out;
}

void PackWorldMatrices(const DirectX::SimpleMath::Matrix& world,
                       DirectX::XMMATRIX& outWorld,
                       DirectX::XMVECTOR outWorldInverseTranspose[3]) {
    // Transposing cancels HLSL's column-major read, leaving the shader with the
    // row-vector matrix this engine uses everywhere else.
    outWorld = DirectX::XMMatrixTranspose(world);

    // Wanted in the shader: M == transpose(inverse(world)), so that
    // mul(normal, M) keeps the normal perpendicular. To make HLSL read back
    // some matrix X, the rows uploaded must be X's columns, i.e. upload
    // transpose(X). Here transpose(transpose(inverse(world))) is just
    // inverse(world), so its first three rows go straight across -- no explicit
    // transpose appears, which is exactly why this is worth a test.
    const DirectX::XMMATRIX inverseWorld = DirectX::XMMatrixInverse(nullptr, world);
    for (std::size_t row = 0; row < 3; ++row) {
        outWorldInverseTranspose[row] = inverseWorld.r[row];
    }
}

ImageColorSpaceClassification ClassifyImageColorSpaces(const fastgltf::Asset& asset) {
    ImageColorSpaceClassification result;
    result.perImage.assign(asset.images.size(), ImageColorSpace::Linear);

    // Tracks which images have been claimed, so a second claim in the other
    // space is reported rather than silently overwriting the first.
    std::vector<bool> claimed(asset.images.size(), false);

    const auto claim = [&](std::size_t textureIndex, MaterialTextureSlot slot) {
        if (textureIndex >= asset.textures.size()) {
            return;
        }
        const fastgltf::Texture& texture = asset.textures[textureIndex];
        if (!texture.imageIndex.has_value()) {
            return;
        }
        const std::size_t imageIdx = texture.imageIndex.value();
        if (imageIdx >= result.perImage.size()) {
            return;
        }

        const ImageColorSpace wanted = ColorSpaceForSlot(slot);
        if (!claimed[imageIdx]) {
            claimed[imageIdx] = true;
            result.perImage[imageIdx] = wanted;
            return;
        }
        if (result.perImage[imageIdx] == wanted) {
            return;
        }

        // Shared between an sRGB slot and a linear slot. Resolved towards sRGB:
        // an image reaching a base-colour or emissive slot is colour data, and
        // showing colour un-decoded is a larger visible error than treating a
        // packed map as sRGB. Fixing it properly needs two resources for the one
        // image, or a TYPELESS resource with one SRV per space.
        result.perImage[imageIdx] = ImageColorSpace::Srgb;
        if (std::find(result.conflicts.begin(), result.conflicts.end(), imageIdx) ==
            result.conflicts.end()) {
            result.conflicts.push_back(imageIdx);
        }
    };

    for (const fastgltf::Material& material : asset.materials) {
        if (material.pbrData.baseColorTexture.has_value()) {
            claim(material.pbrData.baseColorTexture.value().textureIndex,
                  MaterialTextureSlot::BaseColor);
        }
        if (material.pbrData.metallicRoughnessTexture.has_value()) {
            claim(material.pbrData.metallicRoughnessTexture.value().textureIndex,
                  MaterialTextureSlot::MetallicRoughness);
        }
        if (material.normalTexture.has_value()) {
            claim(material.normalTexture.value().textureIndex,
                  MaterialTextureSlot::Normal);
        }
        if (material.occlusionTexture.has_value()) {
            claim(material.occlusionTexture.value().textureIndex,
                  MaterialTextureSlot::Occlusion);
        }
        if (material.emissiveTexture.has_value()) {
            claim(material.emissiveTexture.value().textureIndex,
                  MaterialTextureSlot::Emissive);
        }
    }

    return result;
}

void SceneGraph::Build(const fastgltf::Asset& asset,
                       std::size_t sceneIndex,
                       const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
                       std::size_t imageDescriptorCount) {
    m_nodes.clear();
    m_lights.clear();
    m_skins.clear();
    m_animations.clear();
    m_skinDiagnostics.clear();
    m_jointIndexFormatSeen = false;
    m_jointWeightFormatSeen = false;
    m_jointIndexFormat = DXGI_FORMAT_R8G8B8A8_UINT;
    m_jointWeightFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
    m_roots.clear();
    m_drawOrder.clear();
    m_selected.reset();
    m_sceneName.clear();

    // glTF allows an asset with no scenes at all. The previous renderer walked
    // Asset::nodes flat, so falling back to "every parentless node is a root"
    // keeps such an asset visible instead of silently drawing nothing.
    const bool hasScene = !asset.scenes.empty() && sceneIndex < asset.scenes.size();
    if (hasScene) {
        const fastgltf::Scene& scene = asset.scenes[sceneIndex];
        m_sceneName = scene.name.empty() ? std::string("(unnamed scene)")
                                         : std::string(scene.name);
    } else {
        m_sceneName = "(no scene; all parentless nodes treated as roots)";
    }

    // Kept parallel to asset.nodes so child indices need no remapping. glTF
    // requires nodes to form a disjoint set of strict trees, so a node has at
    // most one parent and this stays unambiguous.
    m_nodes.resize(asset.nodes.size());
    for (std::size_t i = 0; i < asset.nodes.size(); ++i) {
        const fastgltf::Node& src = asset.nodes[i];
        SceneNode& dst = m_nodes[i];

        dst.gltfNodeIndex = i;
        dst.name = src.name.empty() ? ("node " + std::to_string(i))
                                    : std::string(src.name);
        dst.localTransform = ToSimpleMath(fastgltf::getLocalTransformMatrix(src));
        dst.worldTransform = dst.localTransform;

        // Keep the authored TRS so an animation can override one component and
        // leave the rest alone. A node that stored a matrix instead cannot be
        // animated (glTF forbids it), so it is marked and left as built.
        if (const auto* trs = std::get_if<fastgltf::TRS>(&src.transform)) {
            dst.hasTrs = true;
            dst.baseTransform.translation = DirectX::SimpleMath::Vector3(
                trs->translation[0], trs->translation[1], trs->translation[2]);
            dst.baseTransform.rotation = DirectX::SimpleMath::Quaternion(
                trs->rotation[0], trs->rotation[1], trs->rotation[2], trs->rotation[3]);
            dst.baseTransform.scale = DirectX::SimpleMath::Vector3(
                trs->scale[0], trs->scale[1], trs->scale[2]);
        }
        dst.children.assign(src.children.begin(), src.children.end());

        if (src.meshIndex.has_value() && src.meshIndex.value() < asset.meshes.size()) {
            dst.meshIndex = src.meshIndex.value();
            BuildPrimitives(asset, asset.meshes[dst.meshIndex.value()], buffers,
                            imageDescriptorCount, dst.primitives);
        }

        // Asset::lights is only populated when KHR_lights_punctual was enabled
        // on the Parser *and* the asset uses it, so a node carrying a
        // lightIndex with an empty lights array means the extension was not
        // requested. Range-checking covers both that and a malformed index.
        if (src.lightIndex.has_value() && src.lightIndex.value() < asset.lights.size()) {
            const fastgltf::Light& srcLight = asset.lights[src.lightIndex.value()];

            SceneLight light;
            light.gltfLightIndex = src.lightIndex.value();
            light.gltfNodeIndex = i;
            light.name = srcLight.name.empty()
                             ? ("light " + std::to_string(src.lightIndex.value()))
                             : std::string(srcLight.name);
            light.type = FromFastgltf(srcLight.type);
            light.color = DirectX::SimpleMath::Vector3(
                srcLight.color[0], srcLight.color[1], srcLight.color[2]);
            light.intensity = static_cast<float>(srcLight.intensity);
            if (srcLight.range.has_value()) {
                light.range = static_cast<float>(srcLight.range.value());
            }
            // fastgltf already applies the glTF defaults (inner 0, outer pi/4)
            // for spot lights; these stay at SceneLight's own defaults for the
            // other two types, where cone angles have no meaning.
            if (srcLight.innerConeAngle.has_value()) {
                light.innerConeAngle = static_cast<float>(srcLight.innerConeAngle.value());
            }
            if (srcLight.outerConeAngle.has_value()) {
                light.outerConeAngle = static_cast<float>(srcLight.outerConeAngle.value());
            }

            // Placed by the first UpdateTransforms; until then only the node's
            // own local transform is known, so leave the defaults in place.
            dst.lightIndex = m_lights.size();
            m_lights.push_back(std::move(light));
        }
    }

    // Both need the nodes to exist: skins range-check joint indices against
    // them, animations range-check channel targets.
    BuildSkins(asset);
    BuildAnimations(asset);

    // Parent links, so the UI and any future picking can walk upwards.
    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        for (std::size_t child : m_nodes[i].children) {
            if (child < m_nodes.size()) {
                m_nodes[child].parent = i;
            }
        }
    }

    if (hasScene) {
        for (std::size_t root : asset.scenes[sceneIndex].nodeIndices) {
            if (root < m_nodes.size()) {
                m_roots.push_back(root);
            }
        }
    } else {
        for (std::size_t i = 0; i < m_nodes.size(); ++i) {
            if (!m_nodes[i].parent.has_value()) {
                m_roots.push_back(i);
            }
        }
    }
}


namespace {

// Resolves a glTF texture reference down to an index into the renderer's
// per-image SRV descriptor array, or nothing when the reference is missing,
// out of range, or points at a texture with no image source.
std::optional<std::size_t> ResolveTextureImage(const fastgltf::Asset& asset,
                                               std::size_t textureIndex,
                                               std::size_t imageDescriptorCount) {
    if (textureIndex >= asset.textures.size()) {
        return std::nullopt;
    }
    const fastgltf::Texture& texture = asset.textures[textureIndex];
    if (!texture.imageIndex.has_value() ||
        texture.imageIndex.value() >= imageDescriptorCount) {
        return std::nullopt;
    }
    return texture.imageIndex.value();
}

// Fills one vertex buffer view from a named glTF attribute. Returns false, and
// leaves the view zeroed, when the attribute is missing or unusable -- a zeroed
// view binds nothing for that slot, which the shader reads as zero.
//
// TODO (carried over): assumes `componentCount` 32-bit floats per vertex rather
// than consulting accessor.componentType / accessor.type.
bool ResolveVertexStream(const fastgltf::Asset& asset,
                         const fastgltf::Primitive& primitive,
                         const char* attributeName,
                         std::size_t componentCount,
                         const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
                         D3D12_VERTEX_BUFFER_VIEW& view,
                         const fastgltf::Accessor** outAccessor = nullptr) {
    const auto it = primitive.findAttribute(attributeName);
    if (it == primitive.attributes.end()) {
        return false;
    }
    if (it->accessorIndex >= asset.accessors.size()) {
        return false;
    }
    const fastgltf::Accessor& accessor = asset.accessors[it->accessorIndex];
    if (!accessor.bufferViewIndex.has_value()) {
        return false;
    }
    const fastgltf::BufferView& bufferView =
        asset.bufferViews[accessor.bufferViewIndex.value()];
    if (bufferView.bufferIndex >= buffers.size() || !buffers[bufferView.bufferIndex]) {
        return false;
    }

    // componentCount is now only a sanity check on the accessor's type: a
    // VEC3 where VEC4 was expected would otherwise be bound at the wrong rate.
    if (fastgltf::getNumComponents(accessor.type) != componentCount) {
        return false;
    }

    // The element size has to come from the accessor, not from assuming floats.
    // JOINTS_0 is typically a VEC4 of unsigned bytes -- 4 bytes, not 16 -- and
    // using the float size would stride through the buffer four times too fast
    // while still binding successfully.
    //
    // bufferView.byteStride wins when present, which is how glTF expresses
    // interleaved attributes; the element size is the tightly-packed case.
    const UINT elementSize = static_cast<UINT>(
        fastgltf::getElementByteSize(accessor.type, accessor.componentType));
    const UINT stride = bufferView.byteStride.has_value()
                            ? static_cast<UINT>(bufferView.byteStride.value())
                            : elementSize;

    view.BufferLocation = buffers[bufferView.bufferIndex]->GetGPUVirtualAddress() +
                          bufferView.byteOffset + accessor.byteOffset;
    view.StrideInBytes = stride;
    // The last element only needs its own size, not a full stride, but a whole
    // number of strides keeps this simple and stays within the buffer view.
    view.SizeInBytes = static_cast<UINT>(stride * accessor.count);
    if (outAccessor != nullptr) {
        *outAccessor = &accessor;
    }
    return true;
}

// The DXGI format matching a glTF accessor's component type, for the skinning
// attributes. Returns UNKNOWN for anything glTF does not allow there.
DXGI_FORMAT JointIndexFormatFor(const fastgltf::Accessor& accessor) {
    switch (accessor.componentType) {
        case fastgltf::ComponentType::UnsignedByte:  return DXGI_FORMAT_R8G8B8A8_UINT;
        case fastgltf::ComponentType::UnsignedShort: return DXGI_FORMAT_R16G16B16A16_UINT;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

DXGI_FORMAT JointWeightFormatFor(const fastgltf::Accessor& accessor) {
    switch (accessor.componentType) {
        case fastgltf::ComponentType::Float: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        // glTF allows these only as normalised, which is what _UNORM gives the
        // shader: the integers are scaled back into 0..1 by the input assembler.
        case fastgltf::ComponentType::UnsignedByte:
            return accessor.normalized ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_UNKNOWN;
        case fastgltf::ComponentType::UnsignedShort:
            return accessor.normalized ? DXGI_FORMAT_R16G16B16A16_UNORM
                                       : DXGI_FORMAT_UNKNOWN;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

int AlphaModeToInt(fastgltf::AlphaMode mode) {
    switch (mode) {
        case fastgltf::AlphaMode::Opaque: return 0;
        case fastgltf::AlphaMode::Mask:   return 1;
        case fastgltf::AlphaMode::Blend:  return 2;
    }
    return 0;
}

}  // namespace

void SceneGraph::ResolveMaterial(const fastgltf::Asset& asset,
                                 const fastgltf::Material& material,
                                 std::size_t imageDescriptorCount,
                                 PrimitiveResource& res) const {
    MaterialConstants& c = res.material;

    c.baseColorFactor = {
        static_cast<float>(material.pbrData.baseColorFactor[0]),
        static_cast<float>(material.pbrData.baseColorFactor[1]),
        static_cast<float>(material.pbrData.baseColorFactor[2]),
        static_cast<float>(material.pbrData.baseColorFactor[3]),
    };
    c.metallicFactor = static_cast<float>(material.pbrData.metallicFactor);
    c.roughnessFactor = static_cast<float>(material.pbrData.roughnessFactor);

    c.emissiveFactor = {
        static_cast<float>(material.emissiveFactor[0]),
        static_cast<float>(material.emissiveFactor[1]),
        static_cast<float>(material.emissiveFactor[2]),
    };
    c.emissiveStrength = static_cast<float>(material.emissiveStrength);

    c.alphaCutoff = static_cast<float>(material.alphaCutoff);
    c.alphaMode = AlphaModeToInt(material.alphaMode);
    c.doubleSided = material.doubleSided ? 1 : 0;
    c.ior = static_cast<float>(material.ior);

    const auto assign = [&](MaterialTextureSlot slot,
                            std::size_t textureIndex,
                            std::size_t texCoordIndex,
                            int& hasFlag,
                            int& texCoordOut) {
        const auto image = ResolveTextureImage(asset, textureIndex, imageDescriptorCount);
        res.textureImageIndex[static_cast<std::size_t>(slot)] = image;
        hasFlag = image.has_value() ? 1 : 0;
        texCoordOut = static_cast<int>(texCoordIndex);
    };

    if (material.pbrData.baseColorTexture.has_value()) {
        const auto& info = material.pbrData.baseColorTexture.value();
        assign(MaterialTextureSlot::BaseColor, info.textureIndex, info.texCoordIndex,
               c.hasBaseColorTexture, c.baseColorTexCoord);
    }
    if (material.pbrData.metallicRoughnessTexture.has_value()) {
        const auto& info = material.pbrData.metallicRoughnessTexture.value();
        assign(MaterialTextureSlot::MetallicRoughness, info.textureIndex, info.texCoordIndex,
               c.hasMetallicRoughnessTexture, c.metallicRoughnessTexCoord);
    }
    if (material.normalTexture.has_value()) {
        const auto& info = material.normalTexture.value();
        assign(MaterialTextureSlot::Normal, info.textureIndex, info.texCoordIndex,
               c.hasNormalTexture, c.normalTexCoord);
        c.normalScale = static_cast<float>(info.scale);
    }
    if (material.occlusionTexture.has_value()) {
        const auto& info = material.occlusionTexture.value();
        assign(MaterialTextureSlot::Occlusion, info.textureIndex, info.texCoordIndex,
               c.hasOcclusionTexture, c.occlusionTexCoord);
        c.occlusionStrength = static_cast<float>(info.strength);
    }
    if (material.emissiveTexture.has_value()) {
        const auto& info = material.emissiveTexture.value();
        assign(MaterialTextureSlot::Emissive, info.textureIndex, info.texCoordIndex,
               c.hasEmissiveTexture, c.emissiveTexCoord);
    }
}

void SceneGraph::BuildPrimitives(
    const fastgltf::Asset& asset,
    const fastgltf::Mesh& mesh,
    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
    std::size_t imageDescriptorCount,
    std::vector<PrimitiveResource>& out) {
    out.clear();
    out.reserve(mesh.primitives.size());

    for (const fastgltf::Primitive& primitive : mesh.primitives) {
        // Indexed draws only, as before.
        if (!primitive.indicesAccessor.has_value()) {
            continue;
        }

        PrimitiveResource res;

        const fastgltf::Accessor& indexAccessor =
            asset.accessors[primitive.indicesAccessor.value()];
        if (!indexAccessor.bufferViewIndex.has_value()) {
            continue;
        }
        const fastgltf::BufferView& indexView =
            asset.bufferViews[indexAccessor.bufferViewIndex.value()];
        if (indexView.bufferIndex >= buffers.size() || !buffers[indexView.bufferIndex]) {
            continue;
        }

        const bool wideIndices =
            indexAccessor.componentType == fastgltf::ComponentType::UnsignedInt;
        res.indexBufferView.BufferLocation =
            buffers[indexView.bufferIndex]->GetGPUVirtualAddress() +
            indexView.byteOffset + indexAccessor.byteOffset;
        res.indexBufferView.Format = wideIndices ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
        res.indexBufferView.SizeInBytes = static_cast<UINT>(
            indexAccessor.count * (wideIndices ? sizeof(uint32_t) : sizeof(uint16_t)));
        res.indexCount = static_cast<uint32_t>(indexAccessor.count);

        // POSITION is required; the rest are optional and leave their slot
        // zeroed when absent. Looked up by name rather than by the old
        // attributes[0] assumption that position comes first.
        if (!ResolveVertexStream(asset, primitive, "POSITION", 3, buffers,
                                 res.vertexBufferViews[static_cast<std::size_t>(
                                     VertexStream::Position)])) {
            continue;
        }
        ResolveVertexStream(asset, primitive, "TEXCOORD_0", 2, buffers,
                            res.vertexBufferViews[static_cast<std::size_t>(
                                VertexStream::TexCoord0)]);
        ResolveVertexStream(asset, primitive, "NORMAL", 3, buffers,
                            res.vertexBufferViews[static_cast<std::size_t>(
                                VertexStream::Normal)]);
        // TANGENT is VEC4 in glTF, not VEC3: xyz is the tangent and w is the
        // handedness sign used to build the bitangent. Passing 4 here is what
        // makes the stride 16 rather than 12 -- getting it wrong would walk the
        // buffer at the wrong rate and skew every tangent after the first.
        ResolveVertexStream(asset, primitive, "TANGENT", 4, buffers,
                            res.vertexBufferViews[static_cast<std::size_t>(
                                VertexStream::Tangent)]);

        // JOINTS_0 and WEIGHTS_0 are both VEC4 but in different component
        // types, and the accessors are kept so the input layout can be built
        // from the formats this asset actually used.
        const fastgltf::Accessor* jointAccessor = nullptr;
        const fastgltf::Accessor* weightAccessor = nullptr;
        ResolveVertexStream(asset, primitive, "JOINTS_0", 4, buffers,
                            res.vertexBufferViews[static_cast<std::size_t>(
                                VertexStream::Joints0)],
                            &jointAccessor);
        ResolveVertexStream(asset, primitive, "WEIGHTS_0", 4, buffers,
                            res.vertexBufferViews[static_cast<std::size_t>(
                                VertexStream::Weights0)],
                            &weightAccessor);
        if (jointAccessor != nullptr) {
            NoteSkinningFormat(JointIndexFormatFor(*jointAccessor), m_jointIndexFormat,
                               m_jointIndexFormatSeen, m_skinDiagnostics);
        }
        if (weightAccessor != nullptr) {
            NoteSkinningFormat(JointWeightFormatFor(*weightAccessor),
                               m_jointWeightFormat, m_jointWeightFormatSeen,
                               m_skinDiagnostics);
        }

        // Material factors and the five texture slots.
        if (primitive.materialIndex.has_value() &&
            primitive.materialIndex.value() < asset.materials.size()) {
            ResolveMaterial(asset, asset.materials[primitive.materialIndex.value()],
                            imageDescriptorCount, res);
        }
        // Primitives with no material keep MaterialConstants' defaults, which
        // are glTF's own defaults for a missing material.

        // After ResolveMaterial, which owns every other field of this buffer.
        res.material.hasTangents = res.HasTangents() ? 1 : 0;

        out.push_back(res);
    }
}

namespace {

// Maps fastgltf's channel path onto ours. Weights is carried through so the
// channel can be counted as skipped rather than silently vanishing.
bool ToAnimationPath(fastgltf::AnimationPath path, AnimationPath& out) {
    switch (path) {
        case fastgltf::AnimationPath::Translation: out = AnimationPath::Translation; return true;
        case fastgltf::AnimationPath::Rotation:    out = AnimationPath::Rotation;    return true;
        case fastgltf::AnimationPath::Scale:       out = AnimationPath::Scale;       return true;
        case fastgltf::AnimationPath::Weights:     out = AnimationPath::Weights;     return true;
    }
    return false;
}

AnimationInterpolation ToInterpolation(fastgltf::AnimationInterpolation interpolation) {
    switch (interpolation) {
        case fastgltf::AnimationInterpolation::Linear:      return AnimationInterpolation::Linear;
        case fastgltf::AnimationInterpolation::Step:        return AnimationInterpolation::Step;
        case fastgltf::AnimationInterpolation::CubicSpline: return AnimationInterpolation::CubicSpline;
    }
    return AnimationInterpolation::Linear;
}

}  // namespace

void SceneGraph::BuildAnimations(const fastgltf::Asset& asset) {
    m_animations.resize(asset.animations.size());

    for (std::size_t a = 0; a < asset.animations.size(); ++a) {
        const fastgltf::Animation& src = asset.animations[a];
        SceneAnimation& dst = m_animations[a];

        dst.name = src.name.empty() ? ("animation " + std::to_string(a))
                                    : std::string(src.name);

        // Samplers first: the channels below index into them.
        dst.samplers.resize(src.samplers.size());
        for (std::size_t s = 0; s < src.samplers.size(); ++s) {
            const fastgltf::AnimationSampler& srcSampler = src.samplers[s];
            AnimationSampler& sampler = dst.samplers[s];
            sampler.interpolation = ToInterpolation(srcSampler.interpolation);

            if (!AccessorBytesAvailable(asset, srcSampler.inputAccessor) ||
                !AccessorBytesAvailable(asset, srcSampler.outputAccessor)) {
                // Leaves an empty sampler, which SampleAnimation reports as
                // unusable rather than reading from nothing.
                continue;
            }

            const fastgltf::Accessor& in = asset.accessors[srcSampler.inputAccessor];
            const fastgltf::Accessor& outAccessor = asset.accessors[srcSampler.outputAccessor];
            if (in.type != fastgltf::AccessorType::Scalar) {
                continue;
            }

            sampler.times.reserve(in.count);
            fastgltf::iterateAccessor<float>(asset, in, [&](float t) {
                sampler.times.push_back(t);
            });

            // Rotations are VEC4 quaternions, translation and scale VEC3.
            if (outAccessor.type == fastgltf::AccessorType::Vec4) {
                sampler.componentCount = 4;
                sampler.values.reserve(outAccessor.count);
                fastgltf::iterateAccessor<fastgltf::math::fvec4>(
                    asset, outAccessor, [&](const fastgltf::math::fvec4& v) {
                        sampler.values.push_back(
                            DirectX::XMFLOAT4(v[0], v[1], v[2], v[3]));
                    });
            } else if (outAccessor.type == fastgltf::AccessorType::Vec3) {
                sampler.componentCount = 3;
                sampler.values.reserve(outAccessor.count);
                fastgltf::iterateAccessor<fastgltf::math::fvec3>(
                    asset, outAccessor, [&](const fastgltf::math::fvec3& v) {
                        sampler.values.push_back(
                            DirectX::XMFLOAT4(v[0], v[1], v[2], 0.0f));
                    });
            } else {
                // Scalar output is only used by morph weights, which this
                // renderer cannot apply; leave the sampler empty.
                sampler.times.clear();
                continue;
            }

            if (!sampler.times.empty()) {
                dst.duration = std::max(dst.duration, sampler.times.back());
            }
        }

        for (const fastgltf::AnimationChannel& srcChannel : src.channels) {
            AnimationPath path;
            if (!srcChannel.nodeIndex.has_value() ||
                srcChannel.nodeIndex.value() >= m_nodes.size() ||
                srcChannel.samplerIndex >= dst.samplers.size() ||
                !ToAnimationPath(srcChannel.path, path) ||
                path == AnimationPath::Weights) {
                // Morph weights land here: counted, so the UI can say the
                // animation is only partly represented instead of looking
                // complete while a face never moves.
                ++dst.skippedChannels;
                continue;
            }
            AnimationChannel channel;
            channel.nodeIndex = srcChannel.nodeIndex.value();
            channel.path = path;
            channel.samplerIndex = srcChannel.samplerIndex;
            dst.channels.push_back(channel);
        }
    }
}

bool SceneGraph::ApplyPoseOverrides(const std::vector<PoseOverride>& overrides) {
    for (const PoseOverride& entry : overrides) {
        if (entry.nodeIndex >= m_nodes.size()) {
            return false;
        }
    }

    using DirectX::SimpleMath::Matrix;
    for (const PoseOverride& entry : overrides) {
        SceneNode& node = m_nodes[entry.nodeIndex];

        // A node that stored a matrix rather than TRS has no authored scale or
        // translation to preserve, so the override supplies the whole
        // transform. glTF forbids animating such a node for the same reason.
        DirectX::SimpleMath::Vector3 scale =
            node.hasTrs ? node.baseTransform.scale
                        : DirectX::SimpleMath::Vector3(1.0f, 1.0f, 1.0f);
        if (entry.hasScale) {
            scale = entry.scale;
        }
        const DirectX::SimpleMath::Vector3 translation =
            entry.hasTranslation ? entry.translation
                                 : (node.hasTrs ? node.baseTransform.translation
                                                : DirectX::SimpleMath::Vector3::Zero);

        // Same composition order as NodeTransform::ToMatrix: scale, then
        // rotate, then translate, in the row-vector convention.
        node.localTransform = Matrix::CreateScale(scale) *
                              Matrix::CreateFromQuaternion(entry.rotation) *
                              Matrix::CreateTranslation(translation);
    }
    return true;
}

void SceneGraph::ResetToBasePose() {
    for (SceneNode& node : m_nodes) {
        if (node.hasTrs) {
            node.localTransform = node.baseTransform.ToMatrix();
        }
    }
}

bool SceneGraph::ApplyAnimation(std::size_t animationIndex, float time) {
    if (animationIndex >= m_animations.size()) {
        return false;
    }
    const SceneAnimation& animation = m_animations[animationIndex];

    // Start from the authored pose every time. A channel overrides one
    // component, so without this reset the untouched components would keep
    // drifting from whatever the previous frame left behind, and a node the
    // animation stops targeting would freeze instead of returning.
    // A member rather than a local: this runs every frame, and 100-odd nodes
    // of TRS is a heap allocation per frame for no reason.
    m_poseScratch.resize(m_nodes.size());
    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        m_poseScratch[i] = m_nodes[i].baseTransform;
    }
    std::vector<NodeTransform>& pose = m_poseScratch;

    for (const AnimationChannel& channel : animation.channels) {
        const AnimationSampler& sampler = animation.samplers[channel.samplerIndex];
        DirectX::XMFLOAT4 value;
        if (!SampleAnimation(sampler, time, value)) {
            continue;
        }
        NodeTransform& target = pose[channel.nodeIndex];
        switch (channel.path) {
            case AnimationPath::Translation:
                target.translation = DirectX::SimpleMath::Vector3(value.x, value.y, value.z);
                break;
            case AnimationPath::Rotation:
                target.rotation =
                    DirectX::SimpleMath::Quaternion(value.x, value.y, value.z, value.w);
                break;
            case AnimationPath::Scale:
                target.scale = DirectX::SimpleMath::Vector3(value.x, value.y, value.z);
                break;
            case AnimationPath::Weights:
                break;  // filtered out at load
        }
    }

    for (std::size_t i = 0; i < m_nodes.size(); ++i) {
        // A node storing a matrix cannot be animated per the spec, so its
        // localTransform is left exactly as built.
        if (m_nodes[i].hasTrs) {
            m_nodes[i].localTransform = pose[i].ToMatrix();
        }
    }
    return true;
}

void SceneGraph::BuildSkins(const fastgltf::Asset& asset) {
    // Parallel to Asset::skins, so Node::skinIndex carries over unchanged.
    m_skins.resize(asset.skins.size());

    for (std::size_t s = 0; s < asset.skins.size(); ++s) {
        const fastgltf::Skin& src = asset.skins[s];
        SceneSkin& dst = m_skins[s];

        dst.gltfSkinIndex = s;
        dst.name = src.name.empty() ? ("skin " + std::to_string(s))
                                    : std::string(src.name);

        if (src.skeleton.has_value() && src.skeleton.value() < m_nodes.size()) {
            dst.skeletonRoot = src.skeleton.value();
        }

        // Joint order is what JOINTS_n indexes, so an out-of-range entry cannot
        // simply be skipped without shifting every joint after it. Keep the slot
        // and point it at node 0, whose transform is at least valid; the
        // diagnostic says which skin is affected.
        dst.joints.reserve(src.joints.size());
        for (const std::size_t joint : src.joints) {
            if (joint < m_nodes.size()) {
                dst.joints.push_back(joint);
            } else {
                m_skinDiagnostics.push_back({SkinIssue::JointIndexOutOfRange, s, joint});
                dst.joints.push_back(0);
            }
        }

        for (const std::size_t joint : dst.joints) {
            m_nodes[joint].isJoint = true;
        }

        if (dst.joints.size() > kMaxJointMatrices) {
            m_skinDiagnostics.push_back({SkinIssue::TooManyJoints, s, std::nullopt});
        }

        dst.inverseBindMatrices.assign(dst.joints.size(),
                                       DirectX::SimpleMath::Matrix::Identity);

        if (!src.inverseBindMatrices.has_value()) {
            // Legal: glTF says to treat them as identity, which means the mesh
            // is already in each joint's space.
            m_skinDiagnostics.push_back({SkinIssue::NoInverseBindMatrices, s, std::nullopt});
            continue;
        }

        const std::size_t accessorIndex = src.inverseBindMatrices.value();
        if (!AccessorBytesAvailable(asset, accessorIndex)) {
            m_skinDiagnostics.push_back(
                {SkinIssue::InverseBindMatricesUnreadable, s, std::nullopt});
            continue;
        }

        const fastgltf::Accessor& accessor = asset.accessors[accessorIndex];
        if (accessor.type != fastgltf::AccessorType::Mat4 ||
            accessor.count != dst.joints.size()) {
            m_skinDiagnostics.push_back(
                {SkinIssue::InverseBindMatrixCountMismatch, s, std::nullopt});
            continue;
        }

        // Same column-major to row-vector transpose the node transforms get, so
        // the matrices compose in this engine's order. Reading them as-is would
        // leave every joint transposed, which bends the mesh plausibly enough to
        // look like a weighting problem.
        std::size_t joint = 0;
        fastgltf::iterateAccessor<fastgltf::math::fmat4x4>(
            asset, accessor, [&](const fastgltf::math::fmat4x4& m) {
                if (joint < dst.inverseBindMatrices.size()) {
                    dst.inverseBindMatrices[joint] = ToSimpleMath(m);
                }
                ++joint;
            });
        dst.inverseBindMatricesLoaded = true;
    }

    // Link the nodes, and check each skinned mesh has the attributes to use it.
    for (std::size_t i = 0; i < asset.nodes.size() && i < m_nodes.size(); ++i) {
        const fastgltf::Node& src = asset.nodes[i];
        if (!src.skinIndex.has_value()) {
            continue;
        }
        const std::size_t skinIndex = src.skinIndex.value();
        if (skinIndex >= m_skins.size()) {
            continue;
        }
        m_nodes[i].skinIndex = skinIndex;

        if (!src.meshIndex.has_value()) {
            m_skinDiagnostics.push_back({SkinIssue::SkinnedNodeWithoutMesh, skinIndex, i});
            continue;
        }
        CheckSkinningAttributes(asset, src.meshIndex.value(), skinIndex, i,
                                m_skinDiagnostics);

        // Both halves are known only here: the node supplies the skin, the
        // primitive supplies the attributes. A primitive missing either one is
        // left unskinned so it renders in its bind pose -- with zero weights it
        // would otherwise collapse onto the origin, which reads as a transform
        // bug rather than a missing attribute.
        for (PrimitiveResource& primitive : m_nodes[i].primitives) {
            primitive.material.isSkinned = primitive.CanSkin() ? 1 : 0;
        }
    }
}

bool SceneGraph::ComputeJointMatrices(
    // GLTF typically separate mesh node and skeleton node
    // To apply skinning, the vertex need to transform to neutral bone coordinate (as defined in inverseBindMatrices) first
    // Since the hierarchy of skeleton node imply each bone's world transformation, the previous transformed vertex can later apply skeleton
    // node's world transformation to move according to skeleton bone's transformation, canceling out the inverseBindMatrices
    // neutral bone coordinate
    // If there is no animation, then the skeleton bone's transformation should be BindMatrices and make the character at rest pose
    // Since we will still move character, we also need to cancel out character mesh node's world transformation
    std::size_t nodeIndex, std::vector<DirectX::SimpleMath::Matrix>& out) const {
    if (nodeIndex >= m_nodes.size() || !m_nodes[nodeIndex].skinIndex.has_value()) {
        return false;
    }
    const SceneSkin& skin = m_skins[m_nodes[nodeIndex].skinIndex.value()];

    // The skinned node's own transform is divided out here because the renderer
    // applies it again afterwards, exactly as it does for an unskinned mesh.
    // Leaving it in would apply it twice.
    const DirectX::SimpleMath::Matrix inverseNodeWorld =
        m_nodes[nodeIndex].worldTransform.Invert();

    out.resize(skin.joints.size());
    for (std::size_t j = 0; j < skin.joints.size(); ++j) {
        const DirectX::SimpleMath::Matrix& jointWorld =
            m_nodes[skin.joints[j]].worldTransform;
        // Row-vector order, so glTF's
        //   inverse(nodeWorld) * jointWorld * inverseBindMatrix
        // is written back to front.
        out[j] = skin.inverseBindMatrices[j] * jointWorld * inverseNodeWorld;
    }
    return true;
}

void SceneGraph::UpdateTransforms(const DirectX::SimpleMath::Matrix& rootTransform) {
    m_drawOrder.clear();
    for (SceneNode& node : m_nodes) {
        node.inScene = false;
    }
    // Lights the walk below does not reach stay inactive, so a light parented
    // under another scene -- or under a subtree hidden in the UI -- drops out
    // of the upload instead of lingering at a stale position.
    for (SceneLight& light : m_lights) {
        light.active = false;
    }
    for (std::size_t root : m_roots) {
        VisitForTransform(root, rootTransform, true);
    }
}

void SceneGraph::VisitForTransform(std::size_t nodeIndex,
                                   const DirectX::SimpleMath::Matrix& parentWorld,
                                   bool parentVisible) {
    if (nodeIndex >= m_nodes.size()) {
        return;
    }
    SceneNode& node = m_nodes[nodeIndex];

    // Already visited: a malformed asset with a cycle or a shared child would
    // otherwise recurse forever.
    if (node.inScene) {
        return;
    }
    node.inScene = true;

    // Row-vector convention: the node's own transform applies first, then its
    // parent's, so the child goes on the left.
    node.worldTransform = node.localTransform * parentWorld;

    const bool visible = parentVisible && node.visible;
    if (visible && node.HasGeometry()) {
        m_drawOrder.push_back(nodeIndex);
    }

    // A light node usually carries no mesh, so it never reaches the draw order;
    // it still has to be placed from the world transform just computed.
    if (node.lightIndex.has_value() && node.lightIndex.value() < m_lights.size()) {
        PlaceLight(m_lights[node.lightIndex.value()], node.worldTransform, visible);
    }

    for (std::size_t child : node.children) {
        VisitForTransform(child, node.worldTransform, visible);
    }
}

void SceneGraph::PlaceLight(SceneLight& light,
                           const DirectX::SimpleMath::Matrix& nodeWorld,
                           bool visible) const {
    light.active = visible;

    // KHR_lights_punctual: the light sits at the node's origin and shines along
    // the node's local -Z. Both come straight out of the world matrix -- the
    // translation row, and the negated third basis row.
    light.position = nodeWorld.Translation();

    // TransformNormal applies the 3x3 part only, which is what a direction
    // wants. This is deliberately *not* the inverse-transpose used for surface
    // normals: -Z here is an axis of the node's own frame, so it follows the
    // frame directly. The two agree under rotation and differ under non-uniform
    // scale, where the axis is the correct interpretation.
    const DirectX::SimpleMath::Vector3 forward =
        DirectX::SimpleMath::Vector3::TransformNormal(
            DirectX::SimpleMath::Vector3(0.0f, 0.0f, -1.0f), nodeWorld);

    // A degenerate node transform (zero scale on an axis) would otherwise
    // produce a NaN direction and poison every lit pixel.
    const float lengthSq = forward.LengthSquared();
    if (lengthSq > 1e-12f) {
        light.direction = forward / std::sqrt(lengthSq);
    } else {
        light.direction = DirectX::SimpleMath::Vector3(0.0f, 0.0f, -1.0f);
    }
}

std::size_t SceneGraph::GatherShaderLights(ShaderLight* out, std::size_t maxLights) const {
    if (out == nullptr || maxLights == 0) {
        return 0;
    }
    std::size_t written = 0;
    for (const SceneLight& light : m_lights) {
        if (!light.active) {
            continue;
        }
        if (written == maxLights) {
            break;
        }
        out[written++] = ToShaderLight(light);
    }
    // Slots past the count are zeroed rather than left as whatever the previous
    // frame wrote, so a shader that ignores the count reads black lights rather
    // than stale ones.
    for (std::size_t i = written; i < maxLights; ++i) {
        out[i] = ShaderLight{};
    }
    return written;
}

std::size_t SceneGraph::DroppedLightCount(std::size_t maxLights) const {
    std::size_t active = 0;
    for (const SceneLight& light : m_lights) {
        if (light.active) {
            ++active;
        }
    }
    return active > maxLights ? active - maxLights : 0;
}

void SceneGraph::DrawHierarchyUI() {
    if (m_nodes.empty()) {
        ImGui::TextUnformatted("No scene loaded.");
        return;
    }

    ImGui::Text("Scene: %s", m_sceneName.c_str());
    ImGui::Text("%zu nodes, %zu drawn", m_nodes.size(), m_drawOrder.size());

    // Lights are easy to get silently wrong -- an asset exported without the
    // extension, or a light parented outside the scene, both just read as
    // darkness -- so the counts are shown whether or not any were found.
    std::size_t activeLights = 0;
    for (const SceneLight& light : m_lights) {
        if (light.active) {
            ++activeLights;
        }
    }
    if (m_lights.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "No punctual lights (KHR_lights_punctual)");
    } else {
        ImGui::Text("%zu lights, %zu active", m_lights.size(), activeLights);
        const std::size_t dropped = DroppedLightCount(kMaxShaderLights);
        if (dropped != 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                               "%zu light(s) beyond the %zu PBR_Constants holds",
                               dropped, kMaxShaderLights);
        }
        for (const SceneLight& light : m_lights) {
            if (!ImGui::TreeNode(light.name.c_str())) {
                continue;
            }
            ImGui::Text("Type: %s%s", ToString(light.type),
                        light.active ? "" : "  (inactive)");
            ImGui::Text("Colour: %.3f %.3f %.3f", light.color.x, light.color.y,
                        light.color.z);
            ImGui::Text("Intensity: %.3f %s", light.intensity,
                        light.type == LightType::Directional ? "lux" : "candela");
            if (light.type != LightType::Directional) {
                ImGui::Text("Position: %.3f %.3f %.3f", light.position.x,
                            light.position.y, light.position.z);
                if (light.range.has_value()) {
                    ImGui::Text("Range: %.3f", light.range.value());
                } else {
                    ImGui::TextUnformatted("Range: unlimited");
                }
            }
            if (light.type != LightType::Point) {
                ImGui::Text("Direction: %.3f %.3f %.3f", light.direction.x,
                            light.direction.y, light.direction.z);
            }
            if (light.type == LightType::Spot) {
                ImGui::Text("Cone: inner %.1f deg, outer %.1f deg",
                            DirectX::XMConvertToDegrees(light.innerConeAngle),
                            DirectX::XMConvertToDegrees(light.outerConeAngle));
            }
            ImGui::Text("Node: %s", m_nodes[light.gltfNodeIndex].name.c_str());
            ImGui::TreePop();
        }
    }
    // Skins, and anything suspect about them. A rig that silently fell back to
    // identity bind matrices renders as a plausibly-wrong pose, so the
    // diagnostics are worth having in front of you rather than in a log.
    if (!m_skins.empty()) {
        ImGui::Text("%zu skin(s)", m_skins.size());
        for (const SceneSkin& skin : m_skins) {
            char label[256];
            std::snprintf(label, sizeof(label), "%s  [%zu joints]", skin.name.c_str(),
                          skin.JointCount());
            if (!ImGui::TreeNode(label)) {
                continue;
            }
            ImGui::Text("Inverse bind matrices: %s",
                        skin.inverseBindMatricesLoaded ? "loaded"
                                                       : "substituted identities");
            if (skin.skeletonRoot.has_value()) {
                ImGui::Text("Skeleton root: %s",
                            m_nodes[skin.skeletonRoot.value()].name.c_str());
            } else {
                ImGui::TextUnformatted("Skeleton root: (not declared)");
            }
            // Joint order is what JOINTS_n indexes, so it is shown in order.
            if (ImGui::TreeNode("Joints")) {
                for (std::size_t j = 0; j < skin.joints.size(); ++j) {
                    ImGui::Text("%3zu  %s", j, m_nodes[skin.joints[j]].name.c_str());
                }
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }
    }
    if (!m_skinDiagnostics.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "%zu skin issue(s)",
                           m_skinDiagnostics.size());
        for (const SkinDiagnostic& d : m_skinDiagnostics) {
            if (d.nodeIndex.has_value() && d.nodeIndex.value() < m_nodes.size()) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "  skin %zu (%s): %s",
                                   d.skinIndex,
                                   m_nodes[d.nodeIndex.value()].name.c_str(),
                                   ToString(d.issue));
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "  skin %zu: %s",
                                   d.skinIndex, ToString(d.issue));
            }
        }
    }

    ImGui::Separator();

    for (std::size_t root : m_roots) {
        DrawNodeUI(root);
    }

    if (m_selected.has_value() && m_selected.value() < m_nodes.size()) {
        const SceneNode& node = m_nodes[m_selected.value()];
        ImGui::Separator();
        ImGui::Text("Selected: %s", node.name.c_str());
        if (node.parent.has_value()) {
            ImGui::Text("Parent: %s", m_nodes[node.parent.value()].name.c_str());
        } else {
            ImGui::TextUnformatted("Parent: (scene root)");
        }
        ImGui::Text("Primitives: %zu", node.primitives.size());
        if (node.IsSkinned() && node.skinIndex.value() < m_skins.size()) {
            const SceneSkin& skin = m_skins[node.skinIndex.value()];
            ImGui::Text("Skin: %s (%zu joints)", skin.name.c_str(), skin.JointCount());
        }
        if (node.isJoint) {
            ImGui::TextUnformatted("Is a joint of a skin");
        }

        // World transform, row per row. Shows the accumulated result, which is
        // the point of the top-down pass.
        const DirectX::SimpleMath::Vector3 translation = node.worldTransform.Translation();
        ImGui::Text("World position: %.3f %.3f %.3f",
                    translation.x, translation.y, translation.z);
        if (ImGui::TreeNode("World matrix")) {
            for (int r = 0; r < 4; ++r) {
                ImGui::Text("%8.3f %8.3f %8.3f %8.3f",
                            node.worldTransform.m[r][0], node.worldTransform.m[r][1],
                            node.worldTransform.m[r][2], node.worldTransform.m[r][3]);
            }
            ImGui::TreePop();
        }
    }
}

void SceneGraph::DrawNodeUI(std::size_t nodeIndex) {
    if (nodeIndex >= m_nodes.size()) {
        return;
    }
    SceneNode& node = m_nodes[nodeIndex];

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                               ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_DefaultOpen;
    if (node.children.empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }
    if (m_selected.has_value() && m_selected.value() == nodeIndex) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    ImGui::PushID(static_cast<int>(nodeIndex));

    // The checkbox sits before the tree node so clicking it never toggles the
    // tree open state.
    ImGui::Checkbox("##visible", &node.visible);
    ImGui::SameLine();

    // A light node carries no geometry, so without this it is
    // indistinguishable from an empty transform node in the tree.
    const char* lightTag = "";
    if (node.lightIndex.has_value() && node.lightIndex.value() < m_lights.size()) {
        lightTag = ToString(m_lights[node.lightIndex.value()].type);
    }

    // "joint" marks a node some skin drives; "skinned" marks the node whose
    // mesh is deformed by one. They are different roles and a node can be both.
    const char* jointTag = node.isJoint ? "  [joint]" : "";

    char label[256];
    if (node.HasGeometry() && node.IsSkinned()) {
        std::snprintf(label, sizeof(label), "%s  [mesh %zu, %zu prim, skin %zu]%s",
                      node.name.c_str(), node.meshIndex.value_or(0),
                      node.primitives.size(), node.skinIndex.value(), jointTag);
    } else if (node.HasGeometry()) {
        std::snprintf(label, sizeof(label), "%s  [mesh %zu, %zu prim]%s",
                      node.name.c_str(), node.meshIndex.value_or(0),
                      node.primitives.size(), jointTag);
    } else if (node.isJoint) {
        std::snprintf(label, sizeof(label), "%s  [joint]", node.name.c_str());
    } else if (*lightTag != '\0') {
        std::snprintf(label, sizeof(label), "%s  [%s light]", node.name.c_str(),
                      lightTag);
    } else {
        std::snprintf(label, sizeof(label), "%s", node.name.c_str());
    }

    const bool open = ImGui::TreeNodeEx(label, flags);
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        m_selected = nodeIndex;
    }
    if (open) {
        for (std::size_t child : node.children) {
            DrawNodeUI(child);
        }
        ImGui::TreePop();
    }

    ImGui::PopID();
}

}  // namespace NeuralModelIntegrateTestbed
