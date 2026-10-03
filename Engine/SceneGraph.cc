#include "pch.h"

#include "SceneGraph.hpp"

#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>

#include "imgui.h"

#include <cstdio>

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

}  // namespace

void SceneGraph::Build(const fastgltf::Asset& asset,
                       std::size_t sceneIndex,
                       const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& buffers,
                       std::size_t imageDescriptorCount) {
    m_nodes.clear();
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
        dst.children.assign(src.children.begin(), src.children.end());

        if (src.meshIndex.has_value() && src.meshIndex.value() < asset.meshes.size()) {
            dst.meshIndex = src.meshIndex.value();
            BuildPrimitives(asset, asset.meshes[dst.meshIndex.value()], buffers,
                            imageDescriptorCount, dst.primitives);
        }
    }

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
    std::vector<PrimitiveResource>& out) const {
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

        // POSITION. The previous code took attributes[0] on the assumption that
        // position comes first; look it up by name instead, which is what the
        // TODO there asked for and costs nothing at load time.
        const auto positionIt = primitive.findAttribute("POSITION");
        if (positionIt == primitive.attributes.end()) {
            continue;
        }
        const fastgltf::Accessor& positionAccessor =
            asset.accessors[positionIt->accessorIndex];
        if (!positionAccessor.bufferViewIndex.has_value()) {
            continue;
        }
        const fastgltf::BufferView& positionView =
            asset.bufferViews[positionAccessor.bufferViewIndex.value()];
        if (positionView.bufferIndex >= buffers.size() || !buffers[positionView.bufferIndex]) {
            continue;
        }
        res.vertexBufferViews[0].BufferLocation =
            buffers[positionView.bufferIndex]->GetGPUVirtualAddress() +
            positionView.byteOffset + positionAccessor.byteOffset;
        // TODO as before: assumes float3 positions rather than consulting
        // positionAccessor.componentType / .type.
        res.vertexBufferViews[0].StrideInBytes = sizeof(float) * 3;
        res.vertexBufferViews[0].SizeInBytes =
            static_cast<UINT>(sizeof(float) * 3 * positionAccessor.count);

        // TEXCOORD_0, when present.
        const auto texcoordIt = primitive.findAttribute("TEXCOORD_0");
        if (texcoordIt != primitive.attributes.end()) {
            const fastgltf::Accessor& texcoordAccessor =
                asset.accessors[texcoordIt->accessorIndex];
            if (texcoordAccessor.bufferViewIndex.has_value()) {
                const fastgltf::BufferView& texcoordView =
                    asset.bufferViews[texcoordAccessor.bufferViewIndex.value()];
                if (texcoordView.bufferIndex < buffers.size() &&
                    buffers[texcoordView.bufferIndex]) {
                    res.vertexBufferViews[1].BufferLocation =
                        buffers[texcoordView.bufferIndex]->GetGPUVirtualAddress() +
                        texcoordView.byteOffset + texcoordAccessor.byteOffset;
                    // TODO as before: assumes float2 texcoords.
                    res.vertexBufferViews[1].StrideInBytes = sizeof(float) * 2;
                    res.vertexBufferViews[1].SizeInBytes =
                        static_cast<UINT>(sizeof(float) * 2 * texcoordAccessor.count);
                }
            }
        }

        // Material factors and the five texture slots.
        if (primitive.materialIndex.has_value() &&
            primitive.materialIndex.value() < asset.materials.size()) {
            ResolveMaterial(asset, asset.materials[primitive.materialIndex.value()],
                            imageDescriptorCount, res);
        }
        // Primitives with no material keep MaterialConstants' defaults, which
        // are glTF's own defaults for a missing material.

        out.push_back(res);
    }
}

void SceneGraph::UpdateTransforms(const DirectX::SimpleMath::Matrix& rootTransform) {
    m_drawOrder.clear();
    for (SceneNode& node : m_nodes) {
        node.inScene = false;
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

    for (std::size_t child : node.children) {
        VisitForTransform(child, node.worldTransform, visible);
    }
}

void SceneGraph::DrawHierarchyUI() {
    if (m_nodes.empty()) {
        ImGui::TextUnformatted("No scene loaded.");
        return;
    }

    ImGui::Text("Scene: %s", m_sceneName.c_str());
    ImGui::Text("%zu nodes, %zu drawn", m_nodes.size(), m_drawOrder.size());
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

    char label[256];
    if (node.HasGeometry()) {
        std::snprintf(label, sizeof(label), "%s  [mesh %zu, %zu prim]",
                      node.name.c_str(), node.meshIndex.value_or(0),
                      node.primitives.size());
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
