#pragma once

// A render-ready view of one glTF scene.
//
// The renderer used to walk Asset::nodes flat, ignoring the hierarchy and each
// node's own transform, and rebuilt every vertex/index buffer view from
// accessors on every frame. SceneGraph instead resolves all of that once at load
// time into SceneNode objects, and recomputes only the world transforms per
// frame by walking parents before children.

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <directx/d3d12.h>
#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "SimpleMath.h"

namespace NeuralModelIntegrateTestbed {

// The texture slots of a glTF metallic-roughness material, in the order they
// occupy the shader's SRV descriptor table (t0..t4).
enum class MaterialTextureSlot : std::size_t {
    BaseColor = 0,
    MetallicRoughness,
    Normal,
    Occlusion,
    Emissive,
};
constexpr std::size_t kMaterialTextureSlotCount = 5;

// How an image's bytes should be interpreted when it is sampled.
enum class ImageColorSpace {
    Linear,  // DXGI_FORMAT_R8G8B8A8_UNORM
    Srgb,    // DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
};

// glTF 2.0 fixes this per material slot: base colour and emissive hold
// sRGB-encoded colour, while normal, metallic-roughness and occlusion hold
// linear measurements. Decoding the latter as sRGB skews the values the shading
// maths depends on, so the format has to follow the slot rather than being one
// setting for every image.
constexpr ImageColorSpace ColorSpaceForSlot(MaterialTextureSlot slot) {
    switch (slot) {
        case MaterialTextureSlot::BaseColor:
        case MaterialTextureSlot::Emissive:
            return ImageColorSpace::Srgb;
        case MaterialTextureSlot::MetallicRoughness:
        case MaterialTextureSlot::Normal:
        case MaterialTextureSlot::Occlusion:
            return ImageColorSpace::Linear;
    }
    return ImageColorSpace::Linear;
}

constexpr const char* ToString(ImageColorSpace space) {
    return space == ImageColorSpace::Srgb ? "sRGB" : "linear";
}

struct ImageColorSpaceClassification {
    // Parallel to Asset::images. Images no material references default to
    // Linear: this renderer never samples them, and linear is the neutral
    // choice if something starts to.
    std::vector<ImageColorSpace> perImage;

    // Images reached from both an sRGB slot and a linear slot. One resource
    // cannot satisfy both, so these resolve to sRGB and are listed here.
    std::vector<std::size_t> conflicts;
};

// Walks every material and decides, per image, whether it holds sRGB-encoded
// colour or linear data. Pure asset reasoning -- no device required -- so the
// result can be computed (and tested) before any texture is created.
ImageColorSpaceClassification ClassifyImageColorSpaces(const fastgltf::Asset& asset);

// ---------------------------------------------------------------------------
// Lights (KHR_lights_punctual)
// ---------------------------------------------------------------------------

// Light kinds, numbered as the shader's PunctualLight::Type expects. This is
// deliberately not fastgltf::LightType, whose order is Directional/Spot/Point;
// the mapping between them is explicit in SceneGraph.cc so neither side can be
// renumbered by accident.
enum class LightType : int {
    Directional = 0,
    Point = 1,
    Spot = 2,
};

constexpr const char* ToString(LightType type) {
    switch (type) {
        case LightType::Directional: return "directional";
        case LightType::Point:       return "point";
        case LightType::Spot:        return "spot";
    }
    return "unknown";
}

// One light as instanced by one node. A glTF light is a reusable definition in
// the asset's KHR_lights_punctual array; a node references it, and the node's
// transform is what places and aims it. Two nodes pointing at the same light
// definition are therefore two SceneLights, which is why these are keyed by
// node rather than by light index.
struct SceneLight {
    std::string name;
    // Index into Asset::lights -- the shared definition.
    std::size_t gltfLightIndex = 0;
    // Index into Asset::nodes (and into SceneGraph::Nodes()) of the node that
    // instances it, which supplies the transform below.
    std::size_t gltfNodeIndex = 0;

    LightType type = LightType::Directional;

    // Linear RGB, straight from the asset; glTF states these are already linear
    // so there is no decode to undo here.
    DirectX::SimpleMath::Vector3 color = {1.0f, 1.0f, 1.0f};

    // lux (lm/m^2) for directional lights, candela (lm/sr) for point and spot.
    // The two are not interchangeable, so a shader has to branch on `type`
    // before using it.
    float intensity = 1.0f;

    // Point and spot only. Absent means the light reaches infinitely far;
    // ShaderLight encodes that as 0.
    std::optional<float> range;

    // Spot only, radians. glTF defaults: inner 0, outer pi/4.
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7853981634f;

    // --- Recomputed every frame by UpdateTransforms --------------------------
    // World-space node origin. Meaningless for directional lights, which glTF
    // defines as infinitely distant.
    DirectX::SimpleMath::Vector3 position = {0.0f, 0.0f, 0.0f};

    // The direction the light *travels*, i.e. the node's -Z axis in world
    // space, normalised. glTF fixes this: a punctual light with an identity
    // transform shines down -Z. Shading usually wants the opposite vector (from
    // the surface towards the light), so this needs negating at use.
    DirectX::SimpleMath::Vector3 direction = {0.0f, 0.0f, -1.0f};

    // False when the instancing node is outside the chosen scene, or sits in a
    // subtree hidden from the hierarchy UI. Only active lights are uploaded.
    bool active = false;
};

// Mirrors `struct PunctualLight` in shaders/no_texture.fx, which relies on HLSL
// packing each float3+float pair into one register. The static_assert below
// pins the size; the field order has to be kept in step by hand.
struct ShaderLight {
    // c0
    DirectX::XMFLOAT3 position = {0.0f, 0.0f, 0.0f};
    float range = 0.0f;              // 0 == unlimited
    // c1
    DirectX::XMFLOAT3 direction = {0.0f, 0.0f, -1.0f};
    float intensity = 0.0f;
    // c2
    DirectX::XMFLOAT3 color = {0.0f, 0.0f, 0.0f};
    int type = 0;                    // LightType
    // c3 -- cosines rather than the angles themselves: the spot falloff test is
    // a comparison against dot(), so converting once per frame on the CPU saves
    // the shader two transcendentals per light per pixel. SceneLight keeps the
    // raw angles for the UI.
    float innerConeCos = 1.0f;
    float outerConeCos = 0.7071067812f;
    float padding[2] = {0.0f, 0.0f};
};
static_assert(sizeof(ShaderLight) == 64,
              "PunctualLight in no_texture.fx assumes 4 constant registers per light");
// Offsets as dxc reports them for struct PunctualLight. The float3/float pairing
// is the fragile part: HLSL only packs them into one register while nothing
// straddles a 16-byte boundary.
static_assert(offsetof(ShaderLight, position) == 0, "PunctualLight::Position");
static_assert(offsetof(ShaderLight, range) == 12, "PunctualLight::Range");
static_assert(offsetof(ShaderLight, direction) == 16, "PunctualLight::Direction");
static_assert(offsetof(ShaderLight, intensity) == 28, "PunctualLight::Intensity");
static_assert(offsetof(ShaderLight, color) == 32, "PunctualLight::Color");
static_assert(offsetof(ShaderLight, type) == 44, "PunctualLight::Type");
static_assert(offsetof(ShaderLight, innerConeCos) == 48, "PunctualLight::InnerConeCos");
static_assert(offsetof(ShaderLight, outerConeCos) == 52, "PunctualLight::OuterConeCos");

// How many lights PBR_Constants carries. Must equal the array size of
// PBR_Lights in shaders/no_texture.fx; raising it on one side only will
// silently misalign everything after the array in that constant buffer.
constexpr std::size_t kMaxShaderLights = 4;

// Packs one resolved light into its shader form.
ShaderLight ToShaderLight(const SceneLight& light);

// Packs a node's world matrix into the two forms PBR_Constants carries.
//
// HLSL stores cbuffer matrices column-major by default, so register k of an
// uploaded matrix becomes *column* k of the shader's matrix. A row-major
// XMMATRIX written straight in therefore arrives transposed. Both outputs below
// pre-compensate for that, so in the shader:
//
//   mul(float4(position, 1), PBR_World)         == position * world
//   mul(normal, PBR_WorldInverseTranspose)      == normal * transpose(inverse(world))
//
// The second is what keeps normals perpendicular to the surface under
// non-uniform scale. Both are easy to get subtly wrong -- a transpose slip still
// renders a recognisable image -- so see scenegraph-smoke.cc, which checks them
// by replicating HLSL's own reading of the uploaded bytes.
void PackWorldMatrices(const DirectX::SimpleMath::Matrix& world,
                       DirectX::XMMATRIX& outWorld,
                       DirectX::XMVECTOR outWorldInverseTranspose[3]);

// ---------------------------------------------------------------------------
// Animations
// ---------------------------------------------------------------------------
//
// Nothing here reaches the shader directly. An animation drives node TRS, which
// SceneGraph::UpdateTransforms turns into world transforms, which
// ComputeJointMatrices turns into the joint matrices already uploaded to b2. So
// the skinning in no_texture.fx animates without a line of new shader code; an
// animated node with no skin simply moves, via PBR_World as before.

enum class AnimationPath {
    Translation,
    Rotation,  // quaternion; must be slerped, not lerped
    Scale,
    Weights,   // morph targets, which this renderer has no support for
};

enum class AnimationInterpolation {
    Linear,
    Step,
    CubicSpline,
};

const char* ToString(AnimationPath path);
const char* ToString(AnimationInterpolation interpolation);

// One sampler's keyframes, copied to the CPU at load time. Values are kept as
// float4 whatever the path needs -- a VEC3 leaves w unused -- so one array type
// serves translation, rotation and scale.
struct AnimationSampler {
    // Seconds, non-decreasing, as glTF requires.
    std::vector<float> times;

    // One entry per time for Linear and Step. CubicSpline stores THREE per
    // time -- in-tangent, value, out-tangent -- so the value for key i is at
    // 3i+1. Indexing this without accounting for that reads tangents as values.
    std::vector<DirectX::XMFLOAT4> values;

    AnimationInterpolation interpolation = AnimationInterpolation::Linear;

    // 3 for translation and scale, 4 for rotation.
    std::size_t componentCount = 3;

    // How many keyframes, i.e. times.size(). Values may be 3x this.
    std::size_t KeyCount() const { return times.size(); }
};

struct AnimationChannel {
    std::size_t nodeIndex = 0;
    AnimationPath path = AnimationPath::Translation;
    std::size_t samplerIndex = 0;
};

struct SceneAnimation {
    std::string name;
    std::vector<AnimationChannel> channels;
    std::vector<AnimationSampler> samplers;

    // The largest keyframe time across every sampler, in seconds. Zero for an
    // animation whose samplers are all empty.
    float duration = 0.0f;

    // Channels dropped at load: an unknown target, a missing node, or a path
    // this renderer cannot apply (morph weights).
    std::size_t skippedChannels = 0;
};

// A node's own TRS, kept so an animation can override one component and leave
// the others at the asset's values.
struct NodeTransform {
    DirectX::SimpleMath::Vector3 translation = {0.0f, 0.0f, 0.0f};
    DirectX::SimpleMath::Quaternion rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    DirectX::SimpleMath::Vector3 scale = {1.0f, 1.0f, 1.0f};

    // Row-vector order: scale, then rotate, then translate, so that
    // p' = p * S * R * T. glTF writes the same transform as T * R * S for
    // column vectors -- the reversal is the convention change, not an error.
    DirectX::SimpleMath::Matrix ToMatrix() const;
};

// Samples one sampler at `time` seconds. Clamps outside the keyframe range
// rather than extrapolating, which is what glTF specifies. Returns false for an
// empty sampler, leaving `out` untouched.
bool SampleAnimation(const AnimationSampler& sampler, float time,
                     DirectX::XMFLOAT4& out);

// ---------------------------------------------------------------------------
// Skins
// ---------------------------------------------------------------------------

// Things worth reporting about a skin. Every one still yields a usable scene --
// identities substituted, bad joints dropped -- rather than a load failure, so
// they have to be surfaced or they pass unnoticed as "the mesh looks wrong".
enum class SkinIssue {
    NoInverseBindMatrices,           // none declared; identities substituted
    InverseBindMatricesUnreadable,   // declared but not loadable; identities
    InverseBindMatrixCountMismatch,  // count != joint count; identities
    JointIndexOutOfRange,            // dropped from the joint list
    SkinnedNodeWithoutMesh,          // glTF requires a mesh alongside a skin
    PrimitiveMissingJoints,          // no JOINTS_0 on a skinned primitive
    PrimitiveMissingWeights,         // no WEIGHTS_0 -- all weights read as zero
    TooManyJoints,                   // more joints than the shader's array holds
    MixedJointFormats,               // primitives disagree on the attribute format
};

const char* ToString(SkinIssue issue);

struct SkinDiagnostic {
    SkinIssue issue;
    std::size_t skinIndex = 0;
    // The node or joint the issue concerns, where one applies.
    std::optional<std::size_t> nodeIndex;
};

// One glTF skin: the joint nodes it drives, and the matrix that takes each joint
// from its bind pose back to mesh space.
//
// Unlike lights, a skin is NOT instanced per node. The joint list and the
// inverse bind matrices are properties of the skin alone, so this array is kept
// parallel to Asset::skins and SceneNode::skinIndex indexes straight into it.
// What *does* depend on the referencing node is the joint matrices, because the
// formula divides out that node's own world transform -- hence
// ComputeJointMatrices takes the node rather than living on the skin.
struct SceneSkin {
    std::string name;
    std::size_t gltfSkinIndex = 0;

    // Indices into Asset::nodes, which is also into SceneGraph::Nodes(). The
    // ORDER is significant: this is what a vertex's JOINTS_n components index,
    // so it must not be sorted or deduplicated.
    std::vector<std::size_t> joints;

    // Parallel to `joints`. Converted to the row-vector convention at load, the
    // same transpose node transforms get. Identity for every joint when the
    // skin declares none, or when what it declares cannot be read.
    std::vector<DirectX::SimpleMath::Matrix> inverseBindMatrices;

    // glTF's optional hint at the common root of the joint hierarchy. Purely
    // informational: joint transforms come from the nodes either way, and glTF
    // does not require the joints to share a single root.
    std::optional<std::size_t> skeletonRoot;

    // False when inverseBindMatrices holds substituted identities rather than
    // the asset's own values.
    bool inverseBindMatricesLoaded = false;

    std::size_t JointCount() const { return joints.size(); }
};

// Mirrors `cbuffer MaterialConstants : register(b1)` in shaders/no_texture.fx.
// Every glTF 2.0 metallic-roughness factor is uploaded whether or not the
// shader reads it yet, so the PBR implementation has everything to hand.
//
// Layout must stay in lockstep with the HLSL packoffsets; the static_assert
// below only catches size drift, not field reordering.
struct MaterialConstants {
    // c0
    DirectX::XMFLOAT4 baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    // c1
    DirectX::XMFLOAT3 emissiveFactor = {0.0f, 0.0f, 0.0f};
    float emissiveStrength = 1.0f;          // KHR_materials_emissive_strength
    // c2
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;               // normalTexture.scale
    float occlusionStrength = 1.0f;         // occlusionTexture.strength
    // c3
    float alphaCutoff = 0.5f;
    int alphaMode = 0;                      // 0 opaque, 1 mask, 2 blend
    int doubleSided = 0;
    float ior = 1.5f;
    // c4 -- which slots actually have a texture. Absent slots still get a valid
    // descriptor (a 1x1 white fallback), so the shader can branch on these
    // rather than needing separate permutations.
    int hasBaseColorTexture = 0;
    int hasMetallicRoughnessTexture = 0;
    int hasNormalTexture = 0;
    int hasOcclusionTexture = 0;
    // c5
    int hasEmissiveTexture = 0;
    // Which TEXCOORD set each texture samples. Only TEXCOORD_0 is uploaded to
    // the GPU today, so these are informational until more sets are wired up.
    int baseColorTexCoord = 0;
    int metallicRoughnessTexCoord = 0;
    int normalTexCoord = 0;
    // c6
    int occlusionTexCoord = 0;
    int emissiveTexCoord = 0;
    // Geometry, not material -- but this buffer is uploaded per primitive, so it
    // is the available per-primitive channel, and normal mapping cannot be done
    // without knowing whether TANGENT was supplied. glTF says to derive tangents
    // from the UVs when it is absent. Fills what used to be padding, so the rest
    // of the layout is unchanged.
    int hasTangents = 0;
    // Whether this primitive is actually skinned: its node references a skin
    // AND it supplies both JOINTS_0 and WEIGHTS_0. Set by BuildSkins, which is
    // the only place that knows both halves. Fills the last padding slot, so
    // the layout is unchanged again.
    int isSkinned = 0;
};
static_assert(sizeof(MaterialConstants) % 16 == 0,
              "MaterialConstants must be a multiple of 16 bytes for a CBV");
// Field offsets, verified against the layout dxc reports for
// cbuffer MaterialConstants in shaders/no_texture.fx. Size alone does not catch
// a reordering, and reordering is the mistake that produces a plausible render
// off the wrong bytes rather than a failure.
static_assert(offsetof(MaterialConstants, baseColorFactor)    ==   0, "c0");
static_assert(offsetof(MaterialConstants, emissiveFactor)     ==  16, "c1");
static_assert(offsetof(MaterialConstants, emissiveStrength)   ==  28, "c1.w");
static_assert(offsetof(MaterialConstants, metallicFactor)     ==  32, "c2.x");
static_assert(offsetof(MaterialConstants, roughnessFactor)    ==  36, "c2.y");
static_assert(offsetof(MaterialConstants, normalScale)        ==  40, "c2.z");
static_assert(offsetof(MaterialConstants, occlusionStrength)  ==  44, "c2.w");
static_assert(offsetof(MaterialConstants, alphaCutoff)        ==  48, "c3.x");
static_assert(offsetof(MaterialConstants, alphaMode)          ==  52, "c3.y");
static_assert(offsetof(MaterialConstants, doubleSided)        ==  56, "c3.z");
static_assert(offsetof(MaterialConstants, ior)                ==  60, "c3.w");
static_assert(offsetof(MaterialConstants, hasBaseColorTexture) == 64, "c4.x");
static_assert(offsetof(MaterialConstants, hasMetallicRoughnessTexture) == 68, "c4.y");
static_assert(offsetof(MaterialConstants, hasNormalTexture)   ==  72, "c4.z");
static_assert(offsetof(MaterialConstants, hasOcclusionTexture) == 76, "c4.w");
static_assert(offsetof(MaterialConstants, hasEmissiveTexture) ==  80, "c5.x");
static_assert(offsetof(MaterialConstants, baseColorTexCoord)  ==  84, "c5.y");
static_assert(offsetof(MaterialConstants, metallicRoughnessTexCoord) == 88, "c5.z");
static_assert(offsetof(MaterialConstants, normalTexCoord)     ==  92, "c5.w");
static_assert(offsetof(MaterialConstants, occlusionTexCoord)  ==  96, "c6.x");
static_assert(offsetof(MaterialConstants, emissiveTexCoord)   == 100, "c6.y");
static_assert(offsetof(MaterialConstants, hasTangents)        == 104, "c6.z");
static_assert(offsetof(MaterialConstants, isSkinned)         == 108, "c6.w");
static_assert(sizeof(MaterialConstants) == 112,
              "MaterialConstants packoffsets in no_texture.fx assume this layout");

// Mirrors `cbuffer JointMatrices : register(b2)` in shaders/no_texture.fx.
//
// A fixed array rather than a structured buffer so it rides the same
// root-CBV path as the other two constant buffers; 128 covers the 65-joint
// mixamo rigs with headroom, at 8 KB per allocation. A skin with more joints
// than this is reported and clamped rather than reading past the array.
constexpr std::size_t kMaxJointMatrices = 128;

struct JointMatrixConstants {
    // Transposed on upload, like PBREffectConstants::world, because HLSL reads
    // cbuffer matrices column-major. Unused entries are identity.
    DirectX::XMMATRIX joints[kMaxJointMatrices];
};
static_assert(sizeof(JointMatrixConstants) == kMaxJointMatrices * 64,
              "JointMatrices in no_texture.fx assumes 4 registers per joint");

// Input assembler slots, matching the input layout in
// GLTFAdapter::PreparePSO and the VSInput struct in shaders/no_texture.fx.
// NORMAL is appended rather than inserted so TEXCOORD_0 keeps its slot.
enum class VertexStream : std::size_t {
    Position = 0,
    TexCoord0,
    Normal,
    Tangent,
    Joints0,
    Weights0,
};
constexpr std::size_t kVertexStreamCount = 6;

// Everything needed to issue one DrawIndexedInstanced, resolved at load time.
struct PrimitiveResource {
    // All slots are always handed to IASetVertexBuffers because the input
    // layout declares them all; a primitive missing an optional attribute
    // leaves that slot zeroed, which binds nothing and reads as zero.
    D3D12_VERTEX_BUFFER_VIEW vertexBufferViews[kVertexStreamCount] = {};
    D3D12_INDEX_BUFFER_VIEW indexBufferView = {};
    uint32_t indexCount = 0;

    // Per-slot index into the owner's per-image SRV descriptor array. Empty
    // where the material has no texture for that slot.
    std::optional<std::size_t> textureImageIndex[kMaterialTextureSlotCount];

    // Uploaded to b1 before this primitive is drawn.
    MaterialConstants material;

    bool HasStream(VertexStream stream) const {
        return vertexBufferViews[static_cast<std::size_t>(stream)].SizeInBytes != 0;
    }
    bool HasTexcoords() const { return HasStream(VertexStream::TexCoord0); }
    bool HasNormals() const { return HasStream(VertexStream::Normal); }
    bool HasTangents() const { return HasStream(VertexStream::Tangent); }
    bool HasJoints() const { return HasStream(VertexStream::Joints0); }
    bool HasWeights() const { return HasStream(VertexStream::Weights0); }

    // glTF requires JOINTS_n and WEIGHTS_n to come as a pair. With only one of
    // them a primitive cannot be skinned: no weights means every weight reads
    // as zero, which collapses the mesh onto the origin rather than leaving it
    // in its bind pose. Both present is the only usable case.
    bool CanSkin() const { return HasJoints() && HasWeights(); }

    std::optional<std::size_t> ImageIndex(MaterialTextureSlot slot) const {
        return textureImageIndex[static_cast<std::size_t>(slot)];
    }

    // Kept for readability at the call site; base colour is the only slot the
    // shader samples so far.
    std::optional<std::size_t> BaseColorImageIndex() const {
        return ImageIndex(MaterialTextureSlot::BaseColor);
    }
};

// One glTF node: where it sits in the hierarchy, its transform, and the
// resources required to draw whatever mesh it references.
class SceneNode {
public:
    std::string name;

    // Index into Asset::nodes. SceneGraph keeps its node array parallel to the
    // asset's, so this doubles as the node's own index.
    std::size_t gltfNodeIndex = 0;

    // The node's own transform, from its TRS or matrix, converted once at build
    // time. Row-vector convention, as DirectXMath expects. Overwritten by
    // SceneGraph::ApplyAnimation for animated nodes.
    DirectX::SimpleMath::Matrix localTransform;

    // The TRS the asset authored, before any animation. Kept so a channel can
    // override one component -- rotation, say -- and leave translation and
    // scale at their authored values, and so clearing the animation restores
    // the original pose exactly.
    NodeTransform baseTransform;

    // False when the node stored a matrix rather than TRS. glTF forbids
    // animating such a node, so baseTransform is not meaningful for it and
    // localTransform is left as built.
    bool hasTrs = false;

    // localTransform composed with every ancestor and the scene root transform.
    // Recomputed by SceneGraph::UpdateTransforms.
    DirectX::SimpleMath::Matrix worldTransform;

    std::optional<std::size_t> parent;
    std::vector<std::size_t> children;

    std::optional<std::size_t> meshIndex;
    std::vector<PrimitiveResource> primitives;

    // Index into SceneGraph::Lights() when this node instances a punctual
    // light. Not an index into Asset::lights -- see SceneLight.
    std::optional<std::size_t> lightIndex;

    // Index into SceneGraph::Skins() when this node's mesh is skinned. This one
    // IS the asset's own skin index, because skins are shared rather than
    // instanced; see SceneSkin.
    std::optional<std::size_t> skinIndex;

    // Cleared for nodes the chosen scene does not reach. Those keep stale
    // transforms and are never drawn.
    bool inScene = false;

    // Hides this node and its whole subtree. Nothing sets this except the
    // hierarchy UI.
    bool visible = true;

    // Set when some skin drives this node. Not mutually exclusive with the
    // above: a joint may itself carry a skinned mesh.
    bool isJoint = false;

    bool HasGeometry() const { return !primitives.empty(); }
    bool IsSkinned() const { return skinIndex.has_value(); }
};

class SceneGraph {
public:
    // Resolves the scene's nodes and their primitives' buffer views.
    // `buffers` are the uploaded glTF buffers, indexed as Asset::buffers is;
    // `imageDescriptorCount` bounds the base-colour image indices so a malformed
    // asset cannot produce an out-of-range descriptor lookup at draw time.
    void Build(const fastgltf::Asset& asset,
               std::size_t sceneIndex,
               const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
               std::size_t imageDescriptorCount);

    // Walks roots to leaves, composing each node's local transform onto its
    // parent's world transform. `rootTransform` is applied outermost, so it
    // behaves as the whole scene's placement.
    void UpdateTransforms(const DirectX::SimpleMath::Matrix& rootTransform);

    bool Empty() const { return m_nodes.empty(); }
    std::size_t NodeCount() const { return m_nodes.size(); }
    const std::vector<SceneNode>& Nodes() const { return m_nodes; }
    const std::vector<std::size_t>& RootNodes() const { return m_roots; }
    const std::string& SceneName() const { return m_sceneName; }

    // Nodes in the order a parent-before-child walk visits them, skipping
    // subtrees hidden in the UI. The render loop iterates this instead of
    // recursing itself.
    const std::vector<std::size_t>& DrawOrder() const { return m_drawOrder; }

    // Every punctual light the asset instances, in node order. Positions and
    // directions are only meaningful after UpdateTransforms has run, and the
    // `active` flag reflects the most recent walk.
    const std::vector<SceneLight>& Lights() const { return m_lights; }

    // Copies the active lights into `out`, newest-first-wins up to `maxLights`,
    // and returns how many were written. Lights beyond the limit are dropped
    // rather than blended, so the count is also how many the shader may read.
    std::size_t GatherShaderLights(ShaderLight* out, std::size_t maxLights) const;

    // Lights the asset instances but PBR_Constants has no room for. Non-zero
    // means the render is missing light, so it is worth surfacing.
    std::size_t DroppedLightCount(std::size_t maxLights) const;

    // The asset's animations, parallel to Asset::animations.
    const std::vector<SceneAnimation>& Animations() const { return m_animations; }

    // Poses every animated node by sampling `animationIndex` at `time` seconds.
    // Nodes the animation does not target keep their authored TRS. Must run
    // BEFORE UpdateTransforms, which is what turns these local transforms into
    // the world transforms the joint matrices are built from.
    //
    // Returns false for an out-of-range index, having changed nothing.
    bool ApplyAnimation(std::size_t animationIndex, float time);

    // Restores every node to the TRS the asset authored.
    void ResetToBasePose();

    // The asset's skins, parallel to Asset::skins.
    const std::vector<SceneSkin>& Skins() const { return m_skins; }

    // The DXGI formats the skinning attributes were found in. glTF allows
    // JOINTS_0 as unsigned byte or unsigned short and WEIGHTS_0 as float or
    // normalised byte/short, and the vertex buffers are the asset's own bytes
    // bound directly -- so the input layout baked into the PSO has to match
    // whatever this asset used. Detected from the first skinned primitive;
    // a later primitive that disagrees raises MixedJointFormats, since one PSO
    // cannot satisfy both.
    DXGI_FORMAT JointIndexFormat() const { return m_jointIndexFormat; }
    DXGI_FORMAT JointWeightFormat() const { return m_jointWeightFormat; }

    // Anything suspect found while resolving the skins. Empty is the good case.
    const std::vector<SkinDiagnostic>& SkinDiagnostics() const {
        return m_skinDiagnostics;
    }

    // The joint matrices for the skinned mesh at `nodeIndex`, in joint order, as
    // the JOINTS_n attributes index them. Each is
    //
    //     inverseBindMatrix[j] * jointWorldTransform[j] * inverse(nodeWorld)
    //
    // which is glTF's formula written for row vectors, so the glTF order is
    // reversed. The trailing inverse(nodeWorld) cancels the skinned node's own
    // transform: the spec has the renderer apply that transform afterwards, so
    // including it here would apply it twice. A vertex is then skinned in the
    // node's local space exactly as an unskinned one is.
    //
    // Returns false and leaves `out` alone when the node is not skinned. Not
    // called from UpdateTransforms -- nothing consumes joint matrices yet, so
    // this stays on demand rather than costing every frame.
    bool ComputeJointMatrices(std::size_t nodeIndex,
                              std::vector<DirectX::SimpleMath::Matrix>& out) const;

    // Renders the hierarchy as an ImGui tree. Selection is kept here so the
    // caller can show details for whichever node is highlighted.
    void DrawHierarchyUI();
    std::optional<std::size_t> SelectedNode() const { return m_selected; }

private:
    // Not const: it records which formats the skinning attributes came in, so
    // the input layout can be built to match the asset.
    void BuildPrimitives(const fastgltf::Asset& asset,
                         const fastgltf::Mesh& mesh,
                         const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
                         std::size_t imageDescriptorCount,
                         std::vector<PrimitiveResource>& out);

    void ResolveMaterial(const fastgltf::Asset& asset,
                         const fastgltf::Material& material,
                         std::size_t imageDescriptorCount,
                         PrimitiveResource& res) const;

    void VisitForTransform(std::size_t nodeIndex,
                           const DirectX::SimpleMath::Matrix& parentWorld,
                           bool parentVisible);

    void DrawNodeUI(std::size_t nodeIndex);

    // Fills in a light's world-space position and direction from the node
    // that instances it.
    void PlaceLight(SceneLight& light,
                    const DirectX::SimpleMath::Matrix& nodeWorld,
                    bool visible) const;

    // Resolves Asset::skins into m_skins and links the nodes that reference
    // them. Runs after the nodes exist, since it range-checks joint indices
    // against them.
    void BuildSkins(const fastgltf::Asset& asset);

    // Copies every animation's keyframes to the CPU. Runs after the nodes
    // exist, since channel targets are range-checked against them.
    void BuildAnimations(const fastgltf::Asset& asset);

    std::vector<SceneNode> m_nodes;
    std::vector<SceneLight> m_lights;
    std::vector<SceneSkin> m_skins;
    std::vector<SceneAnimation> m_animations;
    // Scratch for ApplyAnimation, kept to avoid a per-frame allocation.
    std::vector<NodeTransform> m_poseScratch;
    std::vector<SkinDiagnostic> m_skinDiagnostics;
    // Defaults matter: an asset with no skinned primitive still needs an input
    // layout, and these are the glTF-typical choices.
    DXGI_FORMAT m_jointIndexFormat = DXGI_FORMAT_R8G8B8A8_UINT;
    DXGI_FORMAT m_jointWeightFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
    bool m_jointIndexFormatSeen = false;
    bool m_jointWeightFormatSeen = false;
    std::vector<std::size_t> m_roots;
    std::vector<std::size_t> m_drawOrder;
    std::string m_sceneName;
    std::optional<std::size_t> m_selected;
};

}  // namespace NeuralModelIntegrateTestbed
