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
    float padding[2] = {0.0f, 0.0f};
};
static_assert(sizeof(MaterialConstants) % 16 == 0,
              "MaterialConstants must be a multiple of 16 bytes for a CBV");

// Everything needed to issue one DrawIndexedInstanced, resolved at load time.
struct PrimitiveResource {
    // Slot 0 is POSITION, slot 1 is TEXCOORD_0. Both slots are always bound
    // because the input layout declares two streams; a primitive without
    // texcoords leaves slot 1 zeroed, which is what the previous per-frame code
    // did as well.
    D3D12_VERTEX_BUFFER_VIEW vertexBufferViews[2] = {};
    D3D12_INDEX_BUFFER_VIEW indexBufferView = {};
    uint32_t indexCount = 0;

    // Per-slot index into the owner's per-image SRV descriptor array. Empty
    // where the material has no texture for that slot.
    std::optional<std::size_t> textureImageIndex[kMaterialTextureSlotCount];

    // Uploaded to b1 before this primitive is drawn.
    MaterialConstants material;

    bool HasTexcoords() const { return vertexBufferViews[1].SizeInBytes != 0; }

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
    // time. Row-vector convention, as DirectXMath expects.
    DirectX::SimpleMath::Matrix localTransform;

    // localTransform composed with every ancestor and the scene root transform.
    // Recomputed by SceneGraph::UpdateTransforms.
    DirectX::SimpleMath::Matrix worldTransform;

    std::optional<std::size_t> parent;
    std::vector<std::size_t> children;

    std::optional<std::size_t> meshIndex;
    std::vector<PrimitiveResource> primitives;

    // Cleared for nodes the chosen scene does not reach. Those keep stale
    // transforms and are never drawn.
    bool inScene = false;

    // Hides this node and its whole subtree. Nothing sets this except the
    // hierarchy UI.
    bool visible = true;

    bool HasGeometry() const { return !primitives.empty(); }
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

    // Renders the hierarchy as an ImGui tree. Selection is kept here so the
    // caller can show details for whichever node is highlighted.
    void DrawHierarchyUI();
    std::optional<std::size_t> SelectedNode() const { return m_selected; }

private:
    void BuildPrimitives(const fastgltf::Asset& asset,
                         const fastgltf::Mesh& mesh,
                         const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
                         std::size_t imageDescriptorCount,
                         std::vector<PrimitiveResource>& out) const;

    void ResolveMaterial(const fastgltf::Asset& asset,
                         const fastgltf::Material& material,
                         std::size_t imageDescriptorCount,
                         PrimitiveResource& res) const;

    void VisitForTransform(std::size_t nodeIndex,
                           const DirectX::SimpleMath::Matrix& parentWorld,
                           bool parentVisible);

    void DrawNodeUI(std::size_t nodeIndex);

    std::vector<SceneNode> m_nodes;
    std::vector<std::size_t> m_roots;
    std::vector<std::size_t> m_drawOrder;
    std::string m_sceneName;
    std::optional<std::size_t> m_selected;
};

}  // namespace NeuralModelIntegrateTestbed
