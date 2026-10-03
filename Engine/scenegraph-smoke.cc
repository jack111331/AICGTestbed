// Smoke test for SceneGraph's hierarchy walk and transform composition.
//
// The error-prone part of this is the convention mismatch: fastgltf/glTF store
// matrices column-major and compose parent * child, DirectXMath stores
// row-major and composes child * parent. A transpose or an ordering slip still
// renders *something*, so it is worth asserting numerically rather than looking
// at the screen. No D3D12 device is involved -- SceneGraph only needs buffers
// for primitives, and this asset has no meshes.
#include "pch.h"

#include "SceneGraph.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <vector>

using NeuralModelIntegrateTestbed::kMaterialTextureSlotCount;
using NeuralModelIntegrateTestbed::SceneGraph;
using NeuralModelIntegrateTestbed::SceneNode;

namespace {

// Parent rotates +90 degrees about Z (mapping +X onto +Y); Child sits one unit
// along the parent's local +X; GrandChild adds another unit along its own +X.
// Order and handedness both matter here: if the composition were reversed the
// child would land at (1,0,0) instead of (0,1,0).
// A node carrying real geometry, so the PrimitiveResource path can be checked
// too: one triangle with POSITION, TEXCOORD_0 and ushort indices packed into a
// single base64 buffer (positions at 0, texcoords at 36, indices at 60).
constexpr std::string_view kMeshGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"name": "MeshScene", "nodes": [0]}],
  "nodes": [{"name": "Holder", "translation": [0.0, 3.0, 0.0], "children": [1]},
            {"name": "Tri", "mesh": 0}],
  "meshes": [{"name": "TriMesh", "primitives": [{
    "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
    "indices": 2,
    "material": 0
  }]}],
  "materials": [{
    "name": "FullMaterial",
    "pbrMetallicRoughness": {
      "baseColorFactor": [0.2, 0.4, 0.6, 0.8],
      "metallicFactor": 0.25,
      "roughnessFactor": 0.75,
      "baseColorTexture": {"index": 0},
      "metallicRoughnessTexture": {"index": 1}
    },
    "normalTexture": {"index": 2, "scale": 1.5},
    "occlusionTexture": {"index": 3, "strength": 0.6},
    "emissiveTexture": {"index": 4},
    "emissiveFactor": [0.1, 0.2, 0.3],
    "alphaMode": "MASK",
    "alphaCutoff": 0.33,
    "doubleSided": true
  }],
  "textures": [{"source": 0}, {"source": 1}, {"source": 2}, {"source": 3}, {"source": 4}],
  "images": [{"uri": "a.png"}, {"uri": "b.png"}, {"uri": "c.png"},
             {"uri": "d.png"}, {"uri": "e.png"}],
  "buffers": [{
    "byteLength": 66,
    "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAABAAIA"
  }],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0,  "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 24, "target": 34962},
    {"buffer": 0, "byteOffset": 60, "byteLength": 6,  "target": 34963}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
     "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ]
})GLTF";

constexpr std::string_view kGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"name": "TestScene", "nodes": [0]}],
  "nodes": [
    {"name": "Parent", "rotation": [0.0, 0.0, 0.7071068, 0.7071068], "children": [1]},
    {"name": "Child", "translation": [1.0, 0.0, 0.0], "children": [2]},
    {"name": "GrandChild", "translation": [1.0, 0.0, 0.0]},
    {"name": "Orphan", "translation": [99.0, 99.0, 99.0]}
  ]
})GLTF";

int g_failures = 0;

void ExpectVec(const char* what, const DirectX::SimpleMath::Vector3& got,
               float x, float y, float z) {
    constexpr float kEps = 1e-4f;
    const bool ok = std::fabs(got.x - x) < kEps &&
                    std::fabs(got.y - y) < kEps &&
                    std::fabs(got.z - z) < kEps;
    std::printf("  %-34s (%7.3f %7.3f %7.3f)  expected (%7.3f %7.3f %7.3f)  %s\n",
                what, got.x, got.y, got.z, x, y, z, ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

void ExpectFloat(const char* what, float got, float expected) {
    constexpr float kEps = 1e-4f;
    const bool ok = std::fabs(got - expected) < kEps;
    std::printf("  %-34s %8.3f  expected %8.3f  %s\n", what, got, expected,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

void Expect(const char* what, bool condition) {
    std::printf("  %-34s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

// Parses an in-memory glTF, exiting the test on failure.
fastgltf::Expected<fastgltf::Asset> Parse(std::string_view json) {
    auto data = fastgltf::GltfDataBuffer::FromBytes(
        reinterpret_cast<const std::byte*>(json.data()), json.size());
    if (data.error() != fastgltf::Error::None) {
        return data.error();
    }
    fastgltf::Parser parser;
    return parser.loadGltfJson(data.get(), ".", fastgltf::Options::None);
}

}  // namespace

int main() {
    auto asset = Parse(kGltf);
    if (asset.error() != fastgltf::Error::None) {
        std::cerr << "hierarchy asset parse failed: "
                  << fastgltf::getErrorMessage(asset.error()) << std::endl;
        return 1;
    }

    SceneGraph graph;
    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> noBuffers;
    graph.Build(asset.get(), asset->defaultScene.value_or(0), noBuffers, 0);

    std::cout << "SceneGraph built" << std::endl;
    std::printf("  scene name: %s\n", graph.SceneName().c_str());

    std::cout << "structure" << std::endl;
    Expect("node count == 4", graph.NodeCount() == 4);
    Expect("one root", graph.RootNodes().size() == 1);
    Expect("root is node 0", !graph.RootNodes().empty() && graph.RootNodes()[0] == 0);
    Expect("Child's parent is Parent", graph.Nodes()[1].parent.has_value() &&
                                           graph.Nodes()[1].parent.value() == 0);
    Expect("GrandChild's parent is Child", graph.Nodes()[2].parent.has_value() &&
                                               graph.Nodes()[2].parent.value() == 1);
    Expect("Parent has no parent", !graph.Nodes()[0].parent.has_value());

    // Identity root: world transforms come purely from the hierarchy.
    graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);

    std::cout << "transforms (identity scene root)" << std::endl;
    ExpectVec("Parent world position", graph.Nodes()[0].worldTransform.Translation(),
              0.0f, 0.0f, 0.0f);
    // Parent's +90 deg Z rotation carries the child's local +X onto world +Y.
    ExpectVec("Child world position", graph.Nodes()[1].worldTransform.Translation(),
              0.0f, 1.0f, 0.0f);
    // The grandchild's own +X is rotated the same way, so it stacks along +Y.
    ExpectVec("GrandChild world position", graph.Nodes()[2].worldTransform.Translation(),
              0.0f, 2.0f, 0.0f);

    std::cout << "reachability" << std::endl;
    Expect("Parent in scene", graph.Nodes()[0].inScene);
    Expect("GrandChild in scene", graph.Nodes()[2].inScene);
    Expect("Orphan not in scene", !graph.Nodes()[3].inScene);
    Expect("no geometry, so nothing drawn", graph.DrawOrder().empty());

    // A translated scene root must shift the whole hierarchy.
    graph.UpdateTransforms(DirectX::SimpleMath::Matrix::CreateTranslation(0.0f, 0.0f, 5.0f));
    std::cout << "transforms (scene root translated +5 Z)" << std::endl;
    ExpectVec("Child world position", graph.Nodes()[1].worldTransform.Translation(),
              0.0f, 1.0f, 5.0f);
    ExpectVec("GrandChild world position", graph.Nodes()[2].worldTransform.Translation(),
              0.0f, 2.0f, 5.0f);

    // Hiding a subtree must drop its descendants from the walk as well.
    graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
    const std::size_t reachableBefore = graph.Nodes()[2].inScene ? 1u : 0u;
    Expect("GrandChild reachable before hiding", reachableBefore == 1);

    // ---------------------------------------------------------------------
    // Primitive resources. Needs a real ID3D12Resource so the buffer views get
    // genuine GPU virtual addresses; a WARP or hardware device is fine, nothing
    // is ever submitted.
    // ---------------------------------------------------------------------
    auto meshAsset = Parse(kMeshGltf);
    if (meshAsset.error() != fastgltf::Error::None) {
        std::cerr << "mesh asset parse failed: "
                  << fastgltf::getErrorMessage(meshAsset.error()) << std::endl;
        return 1;
    }

    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::cout << "primitives: skipped, no D3D12 device available" << std::endl;
    } else {
        Microsoft::WRL::ComPtr<ID3D12Resource> gltfBuffer;
        const CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
        const auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(256);
        if (FAILED(device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&gltfBuffer)))) {
            std::cerr << "CreateCommittedResource failed" << std::endl;
            return 1;
        }

        const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers{gltfBuffer};
        SceneGraph meshGraph;
        // Five image descriptors available, one per texture in the asset.
        meshGraph.Build(meshAsset.get(), 0, buffers, 5);
        meshGraph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);

        std::cout << "primitive resources" << std::endl;
        Expect("2 nodes", meshGraph.NodeCount() == 2);
        Expect("one node drawn", meshGraph.DrawOrder().size() == 1);
        Expect("drawn node is the mesh node",
               meshGraph.DrawOrder().size() == 1 && meshGraph.DrawOrder()[0] == 1);
        // The holder's translation must reach the mesh node.
        ExpectVec("mesh node world position",
                  meshGraph.Nodes()[1].worldTransform.Translation(), 0.0f, 3.0f, 0.0f);

        if (!meshGraph.Nodes()[1].primitives.empty()) {
            const auto& prim = meshGraph.Nodes()[1].primitives[0];
            const D3D12_GPU_VIRTUAL_ADDRESS base = gltfBuffer->GetGPUVirtualAddress();
            Expect("index count == 3", prim.indexCount == 3);
            Expect("16-bit indices", prim.indexBufferView.Format == DXGI_FORMAT_R16_UINT);
            Expect("index view size == 6", prim.indexBufferView.SizeInBytes == 6);
            Expect("index view at buffer+60",
                   prim.indexBufferView.BufferLocation == base + 60);
            Expect("position view at buffer+0",
                   prim.vertexBufferViews[0].BufferLocation == base + 0);
            Expect("position stride == 12", prim.vertexBufferViews[0].StrideInBytes == 12);
            Expect("position size == 36", prim.vertexBufferViews[0].SizeInBytes == 36);
            Expect("texcoord view at buffer+36",
                   prim.vertexBufferViews[1].BufferLocation == base + 36);
            Expect("texcoord stride == 8", prim.vertexBufferViews[1].StrideInBytes == 8);
            Expect("has texcoords", prim.HasTexcoords());

            // ------------------------------------------------------------- //
            // Material texture slots: each of the five must resolve to its own
            // image, in the order the SRV table stages them.
            // ------------------------------------------------------------- //
            std::cout << "material texture slots" << std::endl;
            const char* slotNames[kMaterialTextureSlotCount] = {
                "base colour -> image 0", "metallic-roughness -> image 1",
                "normal -> image 2", "occlusion -> image 3", "emissive -> image 4"};
            for (std::size_t slot = 0; slot < kMaterialTextureSlotCount; ++slot) {
                Expect(slotNames[slot],
                       prim.textureImageIndex[slot].has_value() &&
                           prim.textureImageIndex[slot].value() == slot);
            }
            Expect("BaseColorImageIndex() helper agrees",
                   prim.BaseColorImageIndex().has_value() &&
                       prim.BaseColorImageIndex().value() == 0);

            // ------------------------------------------------------------- //
            // Material constants, as uploaded to b1.
            // ------------------------------------------------------------- //
            std::cout << "material constants" << std::endl;
            const NeuralModelIntegrateTestbed::MaterialConstants& m = prim.material;
            ExpectVec("baseColorFactor rgb",
                      DirectX::SimpleMath::Vector3(m.baseColorFactor.x, m.baseColorFactor.y,
                                                   m.baseColorFactor.z),
                      0.2f, 0.4f, 0.6f);
            ExpectFloat("baseColorFactor a", m.baseColorFactor.w, 0.8f);
            ExpectFloat("metallicFactor", m.metallicFactor, 0.25f);
            ExpectFloat("roughnessFactor", m.roughnessFactor, 0.75f);
            ExpectVec("emissiveFactor",
                      DirectX::SimpleMath::Vector3(m.emissiveFactor.x, m.emissiveFactor.y,
                                                   m.emissiveFactor.z),
                      0.1f, 0.2f, 0.3f);
            ExpectFloat("normalScale", m.normalScale, 1.5f);
            ExpectFloat("occlusionStrength", m.occlusionStrength, 0.6f);
            ExpectFloat("alphaCutoff", m.alphaCutoff, 0.33f);
            Expect("alphaMode == MASK (1)", m.alphaMode == 1);
            Expect("doubleSided", m.doubleSided == 1);
            Expect("all five hasTexture flags set",
                   m.hasBaseColorTexture == 1 && m.hasMetallicRoughnessTexture == 1 &&
                       m.hasNormalTexture == 1 && m.hasOcclusionTexture == 1 &&
                       m.hasEmissiveTexture == 1);
        } else {
            Expect("mesh node has primitives", false);
        }

        // With no image descriptors available the texture must not be bound,
        // rather than indexing past the end of the descriptor array.
        SceneGraph noImages;
        noImages.Build(meshAsset.get(), 0, buffers, 0);
        noImages.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
        bool allUnbound = !noImages.Nodes()[1].primitives.empty();
        if (allUnbound) {
            const auto& prim = noImages.Nodes()[1].primitives[0];
            for (std::size_t slot = 0; slot < kMaterialTextureSlotCount; ++slot) {
                if (prim.textureImageIndex[slot].has_value()) allUnbound = false;
            }
            // Factors must still load even with no textures available.
            if (std::fabs(prim.material.metallicFactor - 0.25f) > 1e-4f) allUnbound = false;
            if (prim.material.hasBaseColorTexture != 0) allUnbound = false;
        }
        Expect("no descriptors -> slots unbound, factors kept", allUnbound);
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all checks passed" << std::endl;
    return 0;
}
