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
#include <string>
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
// Exercises animation sampling and the TRS -> local transform rebuild.
//
// Keyframes are chosen so every expected value is arithmetic: translation goes
// (0,0,0) -> (10,0,0) -> (10,20,0) over two seconds, rotation turns 0 -> 90 ->
// 180 degrees about Z, and scale doubles then doubles again. The STEP and
// CUBICSPLINE samplers are separate channels on their own nodes.
//
// The rotation channel is the one that matters most: a component-wise lerp of
// two quaternions instead of a slerp still produces a rotation, just the wrong
// one, so the half-way sample is asserted numerically.
constexpr std::string_view kAnimGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"name": "Root", "children": [1, 2, 3]},
    {"name": "Moved", "translation": [100.0, 100.0, 100.0], "scale": [9.0, 9.0, 9.0]},
    {"name": "Stepped"},
    {"name": "Splined"}
  ],
  "animations": [
    {
      "name": "Clip",
      "channels": [
        {"sampler": 0, "target": {"node": 1, "path": "translation"}},
        {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
        {"sampler": 2, "target": {"node": 1, "path": "scale"}},
        {"sampler": 3, "target": {"node": 2, "path": "translation"}},
        {"sampler": 4, "target": {"node": 3, "path": "translation"}}
      ],
      "samplers": [
        {"input": 0, "output": 1, "interpolation": "LINEAR"},
        {"input": 0, "output": 2, "interpolation": "LINEAR"},
        {"input": 0, "output": 3, "interpolation": "LINEAR"},
        {"input": 4, "output": 5, "interpolation": "STEP"},
        {"input": 6, "output": 7, "interpolation": "CUBICSPLINE"}
      ]
    },
    {
      "name": "MorphOnly",
      "channels": [{"sampler": 0, "target": {"node": 1, "path": "weights"}}],
      "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}]
    }
  ],
  "buffers": [{"byteLength": 244,
    "uri": "data:application/octet-stream;base64,AAAAAAAAgD8AAABAAAAAAAAAAAAAAAAAAAAgQQAAAAAAAAAAAAAgQQAAoEEAAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAA8wQ1P/MENT8AAAAAAAAAAAAAgD8yMY0kAACAPwAAgD8AAIA/AAAAQAAAAEAAAABAAACAQAAAgEAAAIBAAAAAAAAAgD8AAKBAAAAAAAAAAAAAAOBAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEEAAAAAAAAAAAAAAAAAAAAAAAAAAA=="}],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 12},
    {"buffer": 0, "byteOffset": 12, "byteLength": 36},
    {"buffer": 0, "byteOffset": 48,   "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 36},
    {"buffer": 0, "byteOffset": 132, "byteLength": 8},
    {"buffer": 0, "byteOffset": 140, "byteLength": 24},
    {"buffer": 0, "byteOffset": 164,  "byteLength": 8},
    {"buffer": 0, "byteOffset": 172,  "byteLength": 72}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR",
     "min": [0.0], "max": [2.0]},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR",
     "min": [0.0], "max": [1.0]},
    {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"},
    {"bufferView": 6, "componentType": 5126, "count": 2, "type": "SCALAR",
     "min": [0.0], "max": [1.0]},
    {"bufferView": 7, "componentType": 5126, "count": 6, "type": "VEC3"}
  ]
})GLTF";

// Exercises skin parsing and the joint-matrix formula.
//
// Two joints with known bind poses: joint 0 binds at x=2, joint 1 at y=3, so
// their inverse bind matrices are translations of -2 and -3. At run time the
// joints sit elsewhere, which is what makes the formula observable: a vertex at
// a joint's BIND position must land on that joint's CURRENT position.
//
// The matrices below are column-major, as glTF stores them, so a loader missing
// the transpose to row-vector convention fails here rather than in a render.
constexpr std::string_view kSkinGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"name": "Root", "children": [1, 2, 3]},
    {"name": "Joint0", "translation": [5.0, 0.0, 0.0]},
    {"name": "Joint1", "translation": [0.0, 7.0, 0.0]},
    {"name": "SkinnedMesh", "mesh": 0, "skin": 0}
  ],
  "skins": [{
    "name": "TestRig",
    "joints": [1, 2],
    "skeleton": 1,
    "inverseBindMatrices": 3
  }],
  "meshes": [{"name": "SkinnedTri", "primitives": [{
    "attributes": {"POSITION": 0, "JOINTS_0": 4, "WEIGHTS_0": 5},
    "indices": 1
  }]}],
  "buffers": [
    {"byteLength": 128, "uri": "data:application/octet-stream;base64,AACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAwAAAAAAAAAAAAACAPwAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAEDAAAAAAAAAgD8="},
    {"byteLength": 78,  "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAABAAIAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0,  "byteLength": 128},
    {"buffer": 1, "byteOffset": 0,  "byteLength": 36, "target": 34962},
    {"buffer": 1, "byteOffset": 36, "byteLength": 6,  "target": 34963},
    {"buffer": 1, "byteOffset": 42, "byteLength": 36, "target": 34962}
  ],
  "accessors": [
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 0, "componentType": 5126, "count": 2, "type": "MAT4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"}
  ]
})GLTF";

// Exercises KHR_lights_punctual. Nodes place and aim the lights, so the numbers
// asserted against this are the composition of a node transform with the
// glTF-mandated forward axis (0,0,-1) -- a transpose or sign slip in either
// still produces a plausible-looking render, so they are checked numerically.
//
// Every node carrying a light is rotated -90 degrees about X, which maps that
// forward axis onto (0,-1,0): straight down.
//
// Includes a light definition instanced by two different nodes (each instance
// must be placed separately), a light on a node outside the scene (inactive),
// and one more active light than PBR_Constants has room for.
constexpr std::string_view kLightGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_lights_punctual"],
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"name": "Root", "children": [1, 2, 3, 4, 6]},
    {"name": "DirLight",
     "rotation": [-0.7071068, 0.0, 0.0, 0.7071068],
     "extensions": {"KHR_lights_punctual": {"light": 0}}},
    {"name": "PointLight", "translation": [5.0, 0.0, 0.0],
     "extensions": {"KHR_lights_punctual": {"light": 1}}},
    {"name": "SpotLight", "translation": [0.0, 10.0, 0.0],
     "rotation": [-0.7071068, 0.0, 0.0, 0.7071068],
     "extensions": {"KHR_lights_punctual": {"light": 2}}},
    {"name": "DirLightAgain", "translation": [1.0, 2.0, 3.0],
     "rotation": [-0.7071068, 0.0, 0.0, 0.7071068],
     "extensions": {"KHR_lights_punctual": {"light": 0}}},
    {"name": "OrphanLight", "translation": [99.0, 99.0, 99.0],
     "extensions": {"KHR_lights_punctual": {"light": 1}}},
    {"name": "ExtraLight", "extensions": {"KHR_lights_punctual": {"light": 0}}}
  ],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {"name": "Sun", "type": "directional", "color": [1.0, 1.0, 1.0], "intensity": 2.0},
        {"name": "Bulb", "type": "point", "color": [1.0, 0.5, 0.25],
         "intensity": 100.0, "range": 20.0},
        {"name": "Lamp", "type": "spot", "color": [0.25, 0.5, 1.0], "intensity": 50.0,
         "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.5}}
      ]
    }
  }
})GLTF";

// Exercises the colour-space classifier: every material slot, the common case
// of occlusion and metallic-roughness sharing one packed image, an image no
// material references, and an image used as BOTH base colour and normal.
constexpr std::string_view kColorSpaceGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": []}],
  "nodes": [],
  "materials": [
    {
      "name": "Standard",
      "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 1}
      },
      "normalTexture": {"index": 2},
      "occlusionTexture": {"index": 1},
      "emissiveTexture": {"index": 3}
    },
    {
      "name": "ReusesColourAsNormal",
      "pbrMetallicRoughness": {"baseColorTexture": {"index": 5}},
      "normalTexture": {"index": 5}
    }
  ],
  "textures": [{"source": 0}, {"source": 1}, {"source": 2},
               {"source": 3}, {"source": 4}, {"source": 5}],
  "images": [{"uri": "basecolor.png"}, {"uri": "orm.png"}, {"uri": "normal.png"},
             {"uri": "emissive.png"}, {"uri": "unused.png"}, {"uri": "shared.png"}]
})GLTF";

// A node carrying real geometry, so the PrimitiveResource path can be checked
// too: one triangle with POSITION, TEXCOORD_0, NORMAL and ushort indices packed
// into a single base64 buffer (positions at 0, texcoords at 36, indices at 60,
// normals at 66).
constexpr std::string_view kMeshGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"name": "MeshScene", "nodes": [0]}],
  "nodes": [{"name": "Holder", "translation": [0.0, 3.0, 0.0], "children": [1]},
            {"name": "Tri", "mesh": 0, "skin": 0}],
  "skins": [{"name": "MeshSkin", "joints": [0],
             "inverseBindMatrices": 7}],
  "meshes": [{"name": "TriMesh", "primitives": [{
    "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 3, "TANGENT": 4,
                   "JOINTS_0": 5, "WEIGHTS_0": 6},
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
    "byteLength": 274,
    "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAABAAIAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AACAPwAAAAAAAAAAAACAPwAAAAAAAIA/AAAAAAAAgL8AAAAAAAAAAAAAgD8AAIA/AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAgL8AAAAAAACAPw=="
  }],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0,  "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 24, "target": 34962},
    {"buffer": 0, "byteOffset": 60, "byteLength": 6,  "target": 34963},
    {"buffer": 0, "byteOffset": 66, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 102, "byteLength": 48, "target": 34962},
    {"buffer": 0, "byteOffset": 150, "byteLength": 12, "target": 34962},
    {"buffer": 0, "byteOffset": 162, "byteLength": 48, "target": 34962},
    {"buffer": 0, "byteOffset": 210, "byteLength": 64}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
     "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 5, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 6, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 7, "componentType": 5126, "count": 1, "type": "MAT4"}
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

void ExpectSpace(const char* what, NeuralModelIntegrateTestbed::ImageColorSpace got,
                 NeuralModelIntegrateTestbed::ImageColorSpace expected) {
    const bool ok = got == expected;
    std::printf("  %-34s %-7s expected %-7s  %s\n", what,
                NeuralModelIntegrateTestbed::ToString(got),
                NeuralModelIntegrateTestbed::ToString(expected),
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

void Expect(const char* what, bool condition) {
    std::printf("  %-34s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

// --- Replicating HLSL's side of the constant buffer --------------------------
// A cbuffer matrix is column_major by default, so register k of what we upload
// becomes column k of the shader's matrix: M[r][c] == uploaded.r[c][r]. The two
// helpers below evaluate HLSL's mul(rowVector, M) directly from the uploaded
// registers, which is the only way to check the packing without a GPU.

DirectX::XMFLOAT4 HlslMulRowVector(const DirectX::XMFLOAT4& v,
                                   const DirectX::XMMATRIX& uploaded) {
    DirectX::XMFLOAT4 reg[4];
    for (int i = 0; i < 4; ++i) {
        DirectX::XMStoreFloat4(&reg[i], uploaded.r[i]);
    }
    const float vv[4] = {v.x, v.y, v.z, v.w};
    DirectX::XMFLOAT4 out{};
    float* o = &out.x;
    for (int c = 0; c < 4; ++c) {
        const float* column = &reg[c].x;  // register c is HLSL's column c
        float sum = 0.0f;
        for (int r = 0; r < 4; ++r) {
            sum += vv[r] * column[r];     // M[r][c]
        }
        o[c] = sum;
    }
    return out;
}

DirectX::XMFLOAT3 HlslMulRowVector3(const DirectX::XMFLOAT3& v,
                                    const DirectX::XMVECTOR uploaded[3]) {
    DirectX::XMFLOAT4 reg[3];
    for (int i = 0; i < 3; ++i) {
        DirectX::XMStoreFloat4(&reg[i], uploaded[i]);
    }
    const float vv[3] = {v.x, v.y, v.z};
    DirectX::XMFLOAT3 out{};
    float* o = &out.x;
    for (int c = 0; c < 3; ++c) {
        const float* column = &reg[c].x;
        float sum = 0.0f;
        for (int r = 0; r < 3; ++r) {
            sum += vv[r] * column[r];
        }
        o[c] = sum;
    }
    return out;
}

// Parses an in-memory glTF, exiting the test on failure. Extensions have to be
// requested up front -- fastgltf silently drops an unrequested one, so the light
// asset below would parse clean and yield no lights at all.
fastgltf::Expected<fastgltf::Asset> Parse(
    std::string_view json,
    fastgltf::Extensions extensions = fastgltf::Extensions::None) {
    auto data = fastgltf::GltfDataBuffer::FromBytes(
        reinterpret_cast<const std::byte*>(json.data()), json.size());
    if (data.error() != fastgltf::Error::None) {
        return data.error();
    }
    fastgltf::Parser parser(extensions);
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

            // NORMAL occupies slot 2, appended so TEXCOORD_0 keeps slot 1.
            const std::size_t normalSlot =
                static_cast<std::size_t>(NeuralModelIntegrateTestbed::VertexStream::Normal);
            Expect("has normals", prim.HasNormals());
            Expect("normal view at buffer+66",
                   prim.vertexBufferViews[normalSlot].BufferLocation == base + 66);
            Expect("normal stride == 12",
                   prim.vertexBufferViews[normalSlot].StrideInBytes == 12);
            Expect("normal size == 36",
                   prim.vertexBufferViews[normalSlot].SizeInBytes == 36);
            // Position stayed in slot 0 and texcoords in slot 1.
            Expect("position still slot 0",
                   prim.vertexBufferViews[static_cast<std::size_t>(
                       NeuralModelIntegrateTestbed::VertexStream::Position)].BufferLocation == base + 0);
            Expect("texcoord still slot 1",
                   prim.vertexBufferViews[static_cast<std::size_t>(
                       NeuralModelIntegrateTestbed::VertexStream::TexCoord0)].BufferLocation == base + 36);

            // TANGENT occupies slot 3, appended again so nothing above moves.
            // It is the only VEC4 attribute, so its stride is 16 -- treating it
            // as VEC3 would read at 12 and skew every tangent after the first.
            const std::size_t tangentSlot =
                static_cast<std::size_t>(NeuralModelIntegrateTestbed::VertexStream::Tangent);
            Expect("has tangents", prim.HasTangents());
            Expect("tangent view at buffer+102",
                   prim.vertexBufferViews[tangentSlot].BufferLocation == base + 102);
            Expect("tangent stride == 16 (VEC4, not VEC3)",
                   prim.vertexBufferViews[tangentSlot].StrideInBytes == 16);
            Expect("tangent size == 48",
                   prim.vertexBufferViews[tangentSlot].SizeInBytes == 48);
            // The shader branches on this rather than on tangent length.
            Expect("material reports tangents present", prim.material.hasTangents == 1);
            // Earlier slots are untouched by the append.
            Expect("normal still slot 2",
                   prim.vertexBufferViews[normalSlot].BufferLocation == base + 66);

            // Skinning streams. The joint indices are a VEC4 of unsigned
            // BYTES, so the stride is 4 -- assuming floats, as the stream
            // resolver used to, would stride at 16 and read four times too fast
            // through the buffer while still binding successfully.
            const std::size_t jointSlot =
                static_cast<std::size_t>(NeuralModelIntegrateTestbed::VertexStream::Joints0);
            const std::size_t weightSlot =
                static_cast<std::size_t>(NeuralModelIntegrateTestbed::VertexStream::Weights0);
            Expect("has joint indices", prim.HasJoints());
            Expect("has joint weights", prim.HasWeights());
            Expect("joints view at buffer+150",
                   prim.vertexBufferViews[jointSlot].BufferLocation == base + 150);
            Expect("joint stride == 4 (ubyte4, not float4)",
                   prim.vertexBufferViews[jointSlot].StrideInBytes == 4);
            Expect("joints size == 12",
                   prim.vertexBufferViews[jointSlot].SizeInBytes == 12);
            Expect("weights view at buffer+162",
                   prim.vertexBufferViews[weightSlot].BufferLocation == base + 162);
            Expect("weight stride == 16 (float4)",
                   prim.vertexBufferViews[weightSlot].StrideInBytes == 16);
            // Both attributes present and the node has a skin, so this really
            // is skinned -- the flag the vertex shader branches on.
            Expect("primitive can skin", prim.CanSkin());
            Expect("material reports skinned", prim.material.isSkinned == 1);
            // Earlier slots unmoved by the two appended streams.
            Expect("tangent still slot 3",
                   prim.vertexBufferViews[tangentSlot].BufferLocation == base + 102);

            // The input layout is built from these, since the vertex buffers are
            // the asset's own bytes.
            Expect("joint index format is ubyte4",
                   meshGraph.JointIndexFormat() == DXGI_FORMAT_R8G8B8A8_UINT);
            Expect("joint weight format is float4",
                   meshGraph.JointWeightFormat() == DXGI_FORMAT_R32G32B32A32_FLOAT);

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

        // A primitive lacking optional attributes: NORMAL absent must leave
        // slot 2 zeroed (an unbound slot) while POSITION still resolves.
        {
            std::string positionOnly(kMeshGltf);
            const std::string withNormal = R"("NORMAL": 3)";
            const std::size_t at = positionOnly.find(withNormal);
            if (at != std::string::npos) {
                positionOnly.erase(at - 2, withNormal.size() + 2);  // drop ", "NORMAL": 3"
            }
            auto strippedAsset = Parse(positionOnly);
            Expect("stripped asset parses", strippedAsset.error() == fastgltf::Error::None);
            if (strippedAsset.error() == fastgltf::Error::None) {
                SceneGraph stripped;
                stripped.Build(strippedAsset.get(), 0, buffers, 5);
                stripped.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
                const bool drawable = stripped.DrawOrder().size() == 1 &&
                                      !stripped.Nodes()[1].primitives.empty();
                Expect("no NORMAL -> still drawable", drawable);
                if (drawable) {
                    const auto& p = stripped.Nodes()[1].primitives[0];
                    Expect("no NORMAL -> slot 2 unbound", !p.HasNormals());
                    Expect("no NORMAL -> position still bound",
                           p.HasStream(NeuralModelIntegrateTestbed::VertexStream::Position));
                    // Dropping NORMAL must not disturb the slot after it.
                    Expect("no NORMAL -> tangents still bound", p.HasTangents());
                }
            }

            // And the same for TANGENT, which is the common case: the repo's own
            // WithTexture.gltf has no tangents, so this is the default path
            // rather than an edge case.
            std::string noTangent(kMeshGltf);
            const std::string withTangent = R"(, "TANGENT": 4)";
            const std::size_t tangentAt = noTangent.find(withTangent);
            Expect("test asset declares TANGENT", tangentAt != std::string::npos);
            if (tangentAt != std::string::npos) {
                noTangent.erase(tangentAt, withTangent.size());
                auto noTangentAsset = Parse(noTangent);
                Expect("asset without TANGENT parses",
                       noTangentAsset.error() == fastgltf::Error::None);
                if (noTangentAsset.error() == fastgltf::Error::None) {
                    SceneGraph g;
                    g.Build(noTangentAsset.get(), 0, buffers, 5);
                    g.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
                    const bool drawable2 = g.DrawOrder().size() == 1 &&
                                           !g.Nodes()[1].primitives.empty();
                    Expect("no TANGENT -> still drawable", drawable2);
                    if (drawable2) {
                        const auto& p = g.Nodes()[1].primitives[0];
                        Expect("no TANGENT -> slot 3 unbound", !p.HasTangents());
                        // The flag the shader branches on has to agree with the
                        // unbound slot, or normal mapping reads zeroed tangents
                        // believing they are real.
                        Expect("no TANGENT -> flag is 0", p.material.hasTangents == 0);
                        Expect("no TANGENT -> normals still bound", p.HasNormals());
                        Expect("no TANGENT -> position still bound",
                               p.HasStream(NeuralModelIntegrateTestbed::VertexStream::Position));
                    }
                }
            }
        }

        // The skin on this asset: node 0 is its only joint, sitting at y=3
        // while its inverse bind matrix says the bind pose was y=1. A mesh-space
        // vertex at the joint's bind origin must therefore end up at the joint's
        // current world position.
        {
            using DirectX::SimpleMath::Vector3;
            std::vector<DirectX::SimpleMath::Matrix> jm;
            Expect("mesh node is skinned", meshGraph.Nodes()[1].IsSkinned());
            Expect("the joint node is marked", meshGraph.Nodes()[0].isJoint);
            Expect("joint matrices for the mesh node",
                   meshGraph.ComputeJointMatrices(1, jm) && jm.size() == 1);
            if (jm.size() == 1) {
                // Skin in node-local space, then apply the node transform, as
                // the vertex shader does.
                const Vector3 local = Vector3::Transform(Vector3(0.0f, 1.0f, 0.0f), jm[0]);
                ExpectVec("bind-pose vertex reaches the joint",
                          Vector3::Transform(local, meshGraph.Nodes()[1].worldTransform),
                          0.0f, 3.0f, 0.0f);
            }
        }

        // A primitive with joints but no weights must be left UNSKINNED rather
        // than skinned with zeros: zero weights collapse every vertex onto the
        // origin, which reads as a transform bug instead of a missing attribute.
        {
            std::string noWeights(kMeshGltf);
            const std::string ref = R"(, "WEIGHTS_0": 6)";
            const std::size_t at = noWeights.find(ref);
            Expect("mesh asset declares WEIGHTS_0", at != std::string::npos);
            if (at != std::string::npos) {
                noWeights.erase(at, ref.size());
                auto a = Parse(noWeights);
                if (a.error() == fastgltf::Error::None) {
                    SceneGraph g;
                    g.Build(a.get(), 0, buffers, 5);
                    g.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
                    if (!g.Nodes()[1].primitives.empty()) {
                        const auto& p = g.Nodes()[1].primitives[0];
                        Expect("joints without weights -> cannot skin", !p.CanSkin());
                        Expect("joints without weights -> not skinned",
                               p.material.isSkinned == 0);
                        Expect("joints without weights -> joints still bound",
                               p.HasJoints());
                    }
                }
            }
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

    // ---------------------------------------------------------------------
    // Image colour spaces. glTF fixes this per material slot: base colour and
    // emissive are sRGB-encoded, normal / metallic-roughness / occlusion are
    // linear. Decoding the latter as sRGB would quietly skew the shading
    // inputs, so assert the classification rather than eyeball the render.
    // ---------------------------------------------------------------------
    auto csAsset = Parse(kColorSpaceGltf);
    if (csAsset.error() != fastgltf::Error::None) {
        std::cerr << "colour-space asset parse failed: "
                  << fastgltf::getErrorMessage(csAsset.error()) << std::endl;
        return 1;
    }
    {
        using NeuralModelIntegrateTestbed::ClassifyImageColorSpaces;
        using NeuralModelIntegrateTestbed::ImageColorSpace;

        const auto cs = ClassifyImageColorSpaces(csAsset.get());
        std::cout << "image colour spaces" << std::endl;
        Expect("classified all 6 images", cs.perImage.size() == 6);
        if (cs.perImage.size() == 6) {
            ExpectSpace("image 0 (base colour)", cs.perImage[0], ImageColorSpace::Srgb);
            // Shared by metallicRoughness and occlusion -- both linear, so this
            // must NOT be reported as a conflict.
            ExpectSpace("image 1 (packed ORM)", cs.perImage[1], ImageColorSpace::Linear);
            ExpectSpace("image 2 (normal)", cs.perImage[2], ImageColorSpace::Linear);
            ExpectSpace("image 3 (emissive)", cs.perImage[3], ImageColorSpace::Srgb);
            // Unreferenced: never sampled, defaults to linear.
            ExpectSpace("image 4 (unreferenced)", cs.perImage[4], ImageColorSpace::Linear);
            // Used as base colour AND normal: resolves to sRGB and is flagged.
            ExpectSpace("image 5 (colour + normal)", cs.perImage[5], ImageColorSpace::Srgb);
        }
        Expect("exactly one conflict reported", cs.conflicts.size() == 1);
        Expect("the conflict is image 5",
               cs.conflicts.size() == 1 && cs.conflicts[0] == 5);

        // The slot -> space table itself, since everything above depends on it.
        using NeuralModelIntegrateTestbed::ColorSpaceForSlot;
        using NeuralModelIntegrateTestbed::MaterialTextureSlot;
        Expect("BaseColor slot is sRGB",
               ColorSpaceForSlot(MaterialTextureSlot::BaseColor) == ImageColorSpace::Srgb);
        Expect("Emissive slot is sRGB",
               ColorSpaceForSlot(MaterialTextureSlot::Emissive) == ImageColorSpace::Srgb);
        Expect("MetallicRoughness slot is linear",
               ColorSpaceForSlot(MaterialTextureSlot::MetallicRoughness) == ImageColorSpace::Linear);
        Expect("Normal slot is linear",
               ColorSpaceForSlot(MaterialTextureSlot::Normal) == ImageColorSpace::Linear);
        Expect("Occlusion slot is linear",
               ColorSpaceForSlot(MaterialTextureSlot::Occlusion) == ImageColorSpace::Linear);
    }

    // ---------------------------------------------------------------------
    // Punctual lights. A glTF light is a shared definition; the node that
    // references it supplies the position and the aim, so what is asserted here
    // is the composition of the node's world transform with the forward axis
    // the spec fixes at (0,0,-1).
    // ---------------------------------------------------------------------
    auto lightAsset = Parse(kLightGltf, fastgltf::Extensions::KHR_lights_punctual);
    if (lightAsset.error() != fastgltf::Error::None) {
        std::cerr << "light asset parse failed: "
                  << fastgltf::getErrorMessage(lightAsset.error()) << std::endl;
        return 1;
    }
    {
        using NeuralModelIntegrateTestbed::kMaxShaderLights;
        using NeuralModelIntegrateTestbed::LightType;
        using NeuralModelIntegrateTestbed::SceneLight;
        using NeuralModelIntegrateTestbed::ShaderLight;

        SceneGraph lightGraph;
        lightGraph.Build(lightAsset.get(), lightAsset->defaultScene.value_or(0),
                         noBuffers, 0);

        std::cout << "punctual lights" << std::endl;
        Expect("3 light definitions parsed", lightAsset->lights.size() == 3);
        // Six nodes carry a light reference, so six instances -- including the
        // one outside the scene and the two sharing definition 0.
        Expect("6 light instances built", lightGraph.Lights().size() == 6);

        // Identity root first, so these are the asset's own numbers.
        lightGraph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);

        if (lightGraph.Lights().size() == 6) {
            const std::vector<SceneLight>& lights = lightGraph.Lights();

            // --- directional: aim only, no position -------------------------
            Expect("light 0 is directional", lights[0].type == LightType::Directional);
            Expect("light 0 named Sun", lights[0].name == "Sun");
            ExpectFloat("dir light aims -Y (x)", lights[0].direction.x, 0.0f);
            ExpectFloat("dir light aims -Y (y)", lights[0].direction.y, -1.0f);
            ExpectFloat("dir light aims -Y (z)", lights[0].direction.z, 0.0f);
            ExpectFloat("dir light intensity (lux)", lights[0].intensity, 2.0f);

            // --- point: position and range ---------------------------------
            Expect("light 1 is point", lights[1].type == LightType::Point);
            ExpectFloat("point light at x=5", lights[1].position.x, 5.0f);
            ExpectFloat("point light colour g", lights[1].color.y, 0.5f);
            Expect("point light range is 20",
                   lights[1].range.has_value() && lights[1].range.value() == 20.0f);

            // --- spot: position, aim, cone, and an absent range -------------
            Expect("light 2 is spot", lights[2].type == LightType::Spot);
            ExpectFloat("spot light at y=10", lights[2].position.y, 10.0f);
            ExpectFloat("spot light aims -Y (y)", lights[2].direction.y, -1.0f);
            ExpectFloat("spot inner cone angle", lights[2].innerConeAngle, 0.2f);
            ExpectFloat("spot outer cone angle", lights[2].outerConeAngle, 0.5f);
            Expect("spot light range unlimited", !lights[2].range.has_value());

            // --- the second instance of definition 0 ------------------------
            // Same shared definition as light 0, placed by its own node: proof
            // that instances are keyed by node and not deduplicated.
            Expect("light 3 reuses definition 0", lights[3].gltfLightIndex == 0);
            Expect("light 0 and 3 are separate instances",
                   lights[0].gltfNodeIndex != lights[3].gltfNodeIndex);
            ExpectFloat("second instance at x=1", lights[3].position.x, 1.0f);
            ExpectFloat("second instance at y=2", lights[3].position.y, 2.0f);
            ExpectFloat("second instance at z=3", lights[3].position.z, 3.0f);

            // --- reachability -----------------------------------------------
            Expect("in-scene lights are active",
                   lights[0].active && lights[1].active && lights[2].active &&
                       lights[3].active && lights[5].active);
            Expect("light outside the scene is inactive", !lights[4].active);
        }

        // --- the scene root transform carries the lights with it ------------
        // Render() composes a placement matrix onto every node; lights have to
        // move with it, or they stay put while the model turns underneath them.
        // +90 degrees about Z sends (0,-1,0) to (1,0,0) and (1,2,3) to (-2,1,3).
        lightGraph.UpdateTransforms(
            DirectX::SimpleMath::Matrix::CreateRotationZ(DirectX::XM_PIDIV2));
        if (lightGraph.Lights().size() == 6) {
            const std::vector<SceneLight>& lights = lightGraph.Lights();
            ExpectFloat("rotated dir light aims +X (x)", lights[0].direction.x, 1.0f);
            ExpectFloat("rotated dir light aims +X (y)", lights[0].direction.y, 0.0f);
            ExpectFloat("rotated instance pos x", lights[3].position.x, -2.0f);
            ExpectFloat("rotated instance pos y", lights[3].position.y, 1.0f);
            ExpectFloat("rotated instance pos z", lights[3].position.z, 3.0f);
            ExpectFloat("direction stays unit length", lights[0].direction.Length(), 1.0f);
        }

        // --- packing into the constant buffer -------------------------------
        // Pre-filled with a sentinel so "slots past the count are zeroed" is
        // actually tested rather than passing on a fresh array.
        ShaderLight packed[kMaxShaderLights];
        for (ShaderLight& slot : packed) {
            slot.intensity = -1.0f;
            slot.type = 99;
        }
        const std::size_t written =
            lightGraph.GatherShaderLights(packed, kMaxShaderLights);

        std::cout << "shader light packing" << std::endl;
        Expect("4 registers per light", sizeof(ShaderLight) == 64);
        // Five lights are active but PBR_Constants holds four.
        Expect("fills the array to capacity", written == kMaxShaderLights);
        Expect("reports the one it could not fit",
               lightGraph.DroppedLightCount(kMaxShaderLights) == 1);
        Expect("directional packs as type 0", packed[0].type == 0);
        Expect("point packs as type 1", packed[1].type == 1);
        Expect("spot packs as type 2", packed[2].type == 2);
        // An absent range becomes 0, the shader's "unlimited" sentinel.
        ExpectFloat("absent range packs as 0", packed[2].range, 0.0f);
        ExpectFloat("present range packs through", packed[1].range, 20.0f);
        // Cosines, not angles: the shader compares them against a dot product.
        ExpectFloat("inner cone cosine", packed[2].innerConeCos, std::cos(0.2f));
        ExpectFloat("outer cone cosine", packed[2].outerConeCos, std::cos(0.5f));
        Expect("cone cosines are ordered inner >= outer",
               packed[2].innerConeCos >= packed[2].outerConeCos);

        // A smaller capacity writes only the slots it was given; the four
        // dropped lights are reported rather than blended away.
        Expect("honours a smaller capacity",
               lightGraph.GatherShaderLights(packed, 1) == 1);
        Expect("drops 4 when only 1 fits", lightGraph.DroppedLightCount(1) == 4);

        // An asset with no lights must not look like an error -- and the slots
        // the count leaves unused must come back zeroed rather than holding
        // whatever the previous call left there, so a shader that ignores
        // PBR_LightCount reads black lights instead of stale ones.
        SceneGraph unlitGraph;
        unlitGraph.Build(csAsset.get(), 0, noBuffers, 0);
        unlitGraph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
        Expect("an unlit asset yields no lights", unlitGraph.Lights().empty());
        Expect("and packs a count of 0",
               unlitGraph.GatherShaderLights(packed, kMaxShaderLights) == 0);
        bool allZeroed = true;
        for (const ShaderLight& slot : packed) {
            allZeroed = allZeroed && slot.intensity == 0.0f && slot.type == 0 &&
                        slot.range == 0.0f && slot.color.x == 0.0f;
        }
        Expect("unused slots are zeroed", allZeroed);
    }

    // ---------------------------------------------------------------------
    // PBR_Constants matrix packing. HLSL reads cbuffer matrices column-major,
    // so what the shader ends up with is not what a row-major XMMATRIX looks
    // like in memory. A transpose slip here still renders a recognisable image,
    // so the check replicates HLSL's own read of the uploaded registers and
    // compares against the transform it is supposed to perform.
    // ---------------------------------------------------------------------
    {
        using DirectX::SimpleMath::Matrix;
        using DirectX::SimpleMath::Vector3;
        using NeuralModelIntegrateTestbed::PackWorldMatrices;

        // Non-uniform scale is the case that separates a correct normal matrix
        // from the world matrix: under uniform scale and rotation alone the two
        // agree, so a wrong one would pass unnoticed.
        const Matrix world = Matrix::CreateScale(2.0f, 3.0f, 0.5f) *
                             Matrix::CreateRotationY(0.7f) *
                             Matrix::CreateRotationZ(-0.3f) *
                             Matrix::CreateTranslation(1.0f, 2.0f, 3.0f);

        DirectX::XMMATRIX packedWorld;
        DirectX::XMVECTOR packedNormalMatrix[3];
        PackWorldMatrices(world, packedWorld, packedNormalMatrix);

        std::cout << "PBR_Constants matrix packing" << std::endl;

        // --- positions ---------------------------------------------------
        // mul(float4(pos, 1), PBR_World) must equal pos * world.
        const Vector3 localPos(0.3f, -0.7f, 1.4f);
        const DirectX::XMFLOAT4 asShaderSees = HlslMulRowVector(
            DirectX::XMFLOAT4(localPos.x, localPos.y, localPos.z, 1.0f), packedWorld);
        const Vector3 expectedPos = Vector3::Transform(localPos, world);
        ExpectFloat("shader world position x", asShaderSees.x, expectedPos.x);
        ExpectFloat("shader world position y", asShaderSees.y, expectedPos.y);
        ExpectFloat("shader world position z", asShaderSees.z, expectedPos.z);
        ExpectFloat("shader world position w", asShaderSees.w, 1.0f);

        // --- normals -----------------------------------------------------
        // The property that actually matters: a normal stays perpendicular to a
        // tangent after both are transformed. Asserting this rather than the
        // matrix entries means the test does not just restate the derivation.
        // Oblique to the scale axes on purpose: a diagonal scale maps
        // orthogonal *axis* vectors to orthogonal vectors, so axis-aligned
        // choices here would stay perpendicular however the matrix was built
        // and would prove nothing.
        const Vector3 normal = Vector3(1.0f, 1.0f, 0.0f) / std::sqrt(2.0f);
        const Vector3 tangent = Vector3(1.0f, -1.0f, 0.0f) / std::sqrt(2.0f);
        const Vector3 tangent2(0.0f, 0.0f, 1.0f);  // also perpendicular to normal

        const DirectX::XMFLOAT3 shaderNormal = HlslMulRowVector3(
            DirectX::XMFLOAT3(normal.x, normal.y, normal.z), packedNormalMatrix);
        const Vector3 n(shaderNormal.x, shaderNormal.y, shaderNormal.z);

        // Tangents are directions on the surface, so they follow the world
        // matrix's 3x3 part, not the inverse transpose.
        const Vector3 t = Vector3::TransformNormal(tangent, world);
        const Vector3 t2 = Vector3::TransformNormal(tangent2, world);

        ExpectFloat("normal stays perpendicular to tangent", n.Dot(t), 0.0f);
        ExpectFloat("normal stays perpendicular to tangent 2", n.Dot(t2), 0.0f);
        Expect("transformed normal is non-degenerate", n.Length() > 1e-3f);

        // And confirm the inverse transpose is actually needed here: simply
        // transforming the normal by the world matrix would NOT stay
        // perpendicular under this non-uniform scale. Without this, the test
        // above would still pass if PackWorldMatrices wrote the world matrix
        // into both outputs.
        const Vector3 naive = Vector3::TransformNormal(normal, world);
        Expect("naive transform would have been wrong",
               std::fabs(naive.Dot(t)) > 1e-3f);
    }

    // ---------------------------------------------------------------------
    // Skins. The joint-matrix formula is the part worth asserting: glTF writes
    // it for column vectors and this engine composes row vectors, so the order
    // reverses, and the inverse bind matrices themselves need the same
    // column-major transpose node transforms get. Either slip bends the mesh
    // into something that looks like a weighting bug.
    // ---------------------------------------------------------------------
    auto skinAsset = Parse(kSkinGltf);
    if (skinAsset.error() != fastgltf::Error::None) {
        std::cerr << "skin asset parse failed: "
                  << fastgltf::getErrorMessage(skinAsset.error()) << std::endl;
        return 1;
    }
    {
        using DirectX::SimpleMath::Matrix;
        using DirectX::SimpleMath::Vector3;
        using NeuralModelIntegrateTestbed::SceneSkin;
        using NeuralModelIntegrateTestbed::SkinIssue;

        SceneGraph skinGraph;
        skinGraph.Build(skinAsset.get(), 0, noBuffers, 0);
        skinGraph.UpdateTransforms(Matrix::Identity);

        std::cout << "skins" << std::endl;
        Expect("one skin parsed", skinGraph.Skins().size() == 1);
        Expect("no skin issues reported", skinGraph.SkinDiagnostics().empty());
        if (!skinGraph.SkinDiagnostics().empty()) {
            for (const auto& d : skinGraph.SkinDiagnostics()) {
                std::printf("    unexpected: skin %zu: %s\n", d.skinIndex,
                            NeuralModelIntegrateTestbed::ToString(d.issue));
            }
        }

        if (skinGraph.Skins().size() == 1) {
            const SceneSkin& skin = skinGraph.Skins()[0];
            Expect("skin named TestRig", skin.name == "TestRig");
            Expect("two joints", skin.JointCount() == 2);
            // Joint ORDER is what a vertex's JOINTS_n components index.
            Expect("joint 0 is node 1",
                   skin.joints.size() == 2 && skin.joints[0] == 1);
            Expect("joint 1 is node 2",
                   skin.joints.size() == 2 && skin.joints[1] == 2);
            Expect("skeleton root recorded",
                   skin.skeletonRoot.has_value() && skin.skeletonRoot.value() == 1);
            Expect("inverse bind matrices came from the asset",
                   skin.inverseBindMatricesLoaded);

            // The transpose: a column-major translation must arrive in the row
            // where SimpleMath keeps translation. Read untransposed, these would
            // be zero and the translation would sit in the last column instead.
            if (skin.inverseBindMatrices.size() == 2) {
                ExpectVec("IBM 0 is translate(-2,0,0)",
                          skin.inverseBindMatrices[0].Translation(), -2.0f, 0.0f, 0.0f);
                ExpectVec("IBM 1 is translate(0,-3,0)",
                          skin.inverseBindMatrices[1].Translation(), 0.0f, -3.0f, 0.0f);
            }
        }

        // --- node <-> skin correspondence -----------------------------------
        Expect("node 3 is the skinned node",
               skinGraph.Nodes()[3].IsSkinned() &&
                   skinGraph.Nodes()[3].skinIndex.value() == 0);
        Expect("the mesh node is not itself a joint", !skinGraph.Nodes()[3].isJoint);
        Expect("node 1 is marked a joint", skinGraph.Nodes()[1].isJoint);
        Expect("node 2 is marked a joint", skinGraph.Nodes()[2].isJoint);
        Expect("the root is neither", !skinGraph.Nodes()[0].isJoint &&
                                         !skinGraph.Nodes()[0].IsSkinned());

        // --- joint matrices --------------------------------------------------
        // Joint 0 binds at x=2 and now sits at x=5, so a vertex sitting at the
        // joint's bind position must be carried to its current position.
        std::vector<Matrix> jointMatrices;
        Expect("joint matrices computed for the skinned node",
               skinGraph.ComputeJointMatrices(3, jointMatrices));
        Expect("one matrix per joint", jointMatrices.size() == 2);
        if (jointMatrices.size() == 2) {
            const Vector3 atJoint0Bind(2.0f, 0.0f, 0.0f);
            ExpectVec("vertex at joint 0's bind pose follows joint 0",
                      Vector3::Transform(atJoint0Bind, jointMatrices[0]),
                      5.0f, 0.0f, 0.0f);
            const Vector3 atJoint1Bind(0.0f, 3.0f, 0.0f);
            ExpectVec("vertex at joint 1's bind pose follows joint 1",
                      Vector3::Transform(atJoint1Bind, jointMatrices[1]),
                      0.0f, 7.0f, 0.0f);
        }

        // An unskinned node must say so rather than returning stale matrices.
        std::vector<Matrix> unused;
        Expect("unskinned node yields no joint matrices",
               !skinGraph.ComputeJointMatrices(0, unused));

        // --- the skinned node's own transform must cancel out ----------------
        // glTF has the renderer apply the skinned node's transform after
        // skinning, so the joint matrices divide it out. Moving that node must
        // therefore leave the final world position unchanged -- if the division
        // were missing, the transform would be applied twice.
        {
            auto movedAsset = Parse(kSkinGltf);
            SceneGraph moved;
            moved.Build(movedAsset.get(), 0, noBuffers, 0);
            // Place the whole scene somewhere arbitrary, which moves the
            // skinned node and every joint together.
            const Matrix placement = Matrix::CreateRotationZ(0.4f) *
                                     Matrix::CreateTranslation(10.0f, -4.0f, 2.0f);
            moved.UpdateTransforms(placement);

            std::vector<Matrix> movedJoints;
            moved.ComputeJointMatrices(3, movedJoints);
            if (movedJoints.size() == 2) {
                // Skin in the node's local space, then apply the node transform,
                // exactly as the renderer does for an unskinned mesh.
                const Vector3 skinned = Vector3::Transform(
                    Vector3(2.0f, 0.0f, 0.0f), movedJoints[0]);
                const Vector3 world = Vector3::Transform(
                    skinned, moved.Nodes()[3].worldTransform);
                // Joint 0's world position under the same placement.
                const Vector3 expected = moved.Nodes()[1].worldTransform.Translation();
                ExpectVec("node transform cancels, not applied twice", world,
                          expected.x, expected.y, expected.z);
            }
        }

        // --- degradation ------------------------------------------------------
        // No inverseBindMatrices is legal glTF: identities mean the mesh is
        // already in joint space. It must be reported, not silently assumed.
        {
            std::string noIbm(kSkinGltf);
            const std::string ibmRef = R"(,
    "inverseBindMatrices": 3)";
            const std::size_t at = noIbm.find(ibmRef);
            Expect("test asset declares inverseBindMatrices", at != std::string::npos);
            if (at != std::string::npos) {
                noIbm.erase(at, ibmRef.size());
                auto a = Parse(noIbm);
                Expect("asset without inverseBindMatrices parses",
                       a.error() == fastgltf::Error::None);
                if (a.error() == fastgltf::Error::None) {
                    SceneGraph g;
                    g.Build(a.get(), 0, noBuffers, 0);
                    g.UpdateTransforms(Matrix::Identity);
                    bool reported = false;
                    for (const auto& d : g.SkinDiagnostics()) {
                        if (d.issue == SkinIssue::NoInverseBindMatrices) reported = true;
                    }
                    Expect("missing IBMs reported", reported);
                    Expect("missing IBMs flagged as not loaded",
                           !g.Skins().empty() && !g.Skins()[0].inverseBindMatricesLoaded);
                    Expect("missing IBMs substituted with identity",
                           !g.Skins().empty() &&
                               g.Skins()[0].inverseBindMatrices.size() == 2 &&
                               g.Skins()[0].inverseBindMatrices[0] == Matrix::Identity);
                    // With identity IBMs a vertex at the origin follows the joint.
                    std::vector<Matrix> m;
                    g.ComputeJointMatrices(3, m);
                    if (m.size() == 2) {
                        ExpectVec("identity IBM -> origin follows the joint",
                                  Vector3::Transform(Vector3(0.0f, 0.0f, 0.0f), m[0]),
                                  5.0f, 0.0f, 0.0f);
                    }
                }
            }
        }

        // A primitive with no WEIGHTS_0 is malformed glTF -- every weight reads
        // as zero and the mesh collapses, which looks like a transform bug. The
        // repo's own animated character has exactly this defect.
        {
            std::string noWeights(kSkinGltf);
            const std::string weightsRef = R"(, "WEIGHTS_0": 5)";
            const std::size_t at = noWeights.find(weightsRef);
            Expect("test asset declares WEIGHTS_0", at != std::string::npos);
            if (at != std::string::npos) {
                noWeights.erase(at, weightsRef.size());
                auto a = Parse(noWeights);
                if (a.error() == fastgltf::Error::None) {
                    SceneGraph g;
                    g.Build(a.get(), 0, noBuffers, 0);
                    bool missingWeights = false;
                    bool missingJoints = false;
                    for (const auto& d : g.SkinDiagnostics()) {
                        if (d.issue == SkinIssue::PrimitiveMissingWeights) missingWeights = true;
                        if (d.issue == SkinIssue::PrimitiveMissingJoints) missingJoints = true;
                    }
                    Expect("missing WEIGHTS_0 reported", missingWeights);
                    Expect("JOINTS_0 still seen as present", !missingJoints);
                    // The skin itself must still resolve; only the mesh is at fault.
                    Expect("skin still usable", !g.Skins().empty() &&
                                                    g.Skins()[0].JointCount() == 2);
                }
            }
        }

        // An out-of-range joint must keep its slot, or every joint after it
        // shifts and JOINTS_n indexes the wrong bone.
        {
            std::string badJoint(kSkinGltf);
            const std::string joints = R"("joints": [1, 2])";
            const std::size_t at = badJoint.find(joints);
            if (at != std::string::npos) {
                badJoint.replace(at, joints.size(), R"("joints": [1, 99])");
                auto a = Parse(badJoint);
                if (a.error() == fastgltf::Error::None) {
                    SceneGraph g;
                    g.Build(a.get(), 0, noBuffers, 0);
                    bool reported = false;
                    for (const auto& d : g.SkinDiagnostics()) {
                        if (d.issue == SkinIssue::JointIndexOutOfRange) reported = true;
                    }
                    Expect("out-of-range joint reported", reported);
                    Expect("out-of-range joint keeps its slot",
                           !g.Skins().empty() && g.Skins()[0].JointCount() == 2);
                }
            }
        }

        // An asset with no skins must not look like an error.
        SceneGraph unskinned;
        unskinned.Build(csAsset.get(), 0, noBuffers, 0);
        Expect("an unskinned asset yields no skins", unskinned.Skins().empty());
        Expect("and no skin issues", unskinned.SkinDiagnostics().empty());
    }

    // ---------------------------------------------------------------------
    // Animations. Sampling is where this goes wrong quietly: a lerped
    // quaternion is still a rotation, a mis-indexed CUBICSPLINE triplet still
    // produces motion, and a TRS rebuilt in the wrong order still renders.
    // ---------------------------------------------------------------------
    auto animAsset = Parse(kAnimGltf);
    if (animAsset.error() != fastgltf::Error::None) {
        std::cerr << "animation asset parse failed: "
                  << fastgltf::getErrorMessage(animAsset.error()) << std::endl;
        return 1;
    }
    {
        using DirectX::SimpleMath::Matrix;
        using DirectX::SimpleMath::Quaternion;
        using DirectX::SimpleMath::Vector3;
        using NeuralModelIntegrateTestbed::AnimationInterpolation;
        using NeuralModelIntegrateTestbed::AnimationPath;
        using NeuralModelIntegrateTestbed::SceneAnimation;

        SceneGraph animGraph;
        animGraph.Build(animAsset.get(), 0, noBuffers, 0);

        std::cout << "animations" << std::endl;
        Expect("two animations parsed", animGraph.Animations().size() == 2);
        if (animGraph.Animations().size() != 2) {
            std::cerr << "cannot continue animation checks" << std::endl;
        } else {
            const SceneAnimation& clip = animGraph.Animations()[0];
            Expect("clip named", clip.name == "Clip");
            Expect("five channels kept", clip.channels.size() == 5);
            Expect("no channels skipped in the TRS clip", clip.skippedChannels == 0);
            ExpectFloat("duration is the last keyframe", clip.duration, 2.0f);
            Expect("interpolation modes preserved",
                   clip.samplers.size() == 5 &&
                       clip.samplers[0].interpolation == AnimationInterpolation::Linear &&
                       clip.samplers[3].interpolation == AnimationInterpolation::Step &&
                       clip.samplers[4].interpolation == AnimationInterpolation::CubicSpline);
            // A rotation sampler is VEC4; translation and scale are VEC3.
            Expect("rotation sampler has 4 components",
                   clip.samplers[1].componentCount == 4);
            Expect("translation sampler has 3 components",
                   clip.samplers[0].componentCount == 3);
            // CUBICSPLINE stores three values per key.
            Expect("cubic sampler has 3 values per key",
                   clip.samplers[4].KeyCount() == 2 &&
                       clip.samplers[4].values.size() == 6);

            // A clip of nothing but morph weights must report its channels as
            // skipped rather than appearing to work.
            const SceneAnimation& morph = animGraph.Animations()[1];
            Expect("morph-only clip keeps no channels", morph.channels.empty());
            Expect("morph-only clip counts the skip", morph.skippedChannels == 1);

            // --- the authored pose is preserved --------------------------
            // Node 1 has its own translation and scale, which the animation
            // overrides; getting back to them must be exact.
            animGraph.ResetToBasePose();
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("authored pose before animating",
                      animGraph.Nodes()[1].worldTransform.Translation(),
                      100.0f, 100.0f, 100.0f);

            // The TRS rebuild must agree with what fastgltf produced at load.
            // It is built with DirectXMath directly, so unlike ToSimpleMath
            // there is no transpose -- an easy place to add one by reflex.
            {
                const Matrix rebuilt = animGraph.Nodes()[1].baseTransform.ToMatrix();
                const Matrix fromLoad = animGraph.Nodes()[1].localTransform;
                bool same = true;
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        if (std::fabs(rebuilt.m[r][c] - fromLoad.m[r][c]) > 1e-4f)
                            same = false;
                Expect("TRS rebuild matches fastgltf's local transform", same);
            }

            // --- sampling at keyframes -----------------------------------
            animGraph.ApplyAnimation(0, 0.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("t=0 translation", animGraph.Nodes()[1].worldTransform.Translation(),
                      0.0f, 0.0f, 0.0f);

            animGraph.ApplyAnimation(0, 1.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("t=1 translation", animGraph.Nodes()[1].worldTransform.Translation(),
                      10.0f, 0.0f, 0.0f);

            animGraph.ApplyAnimation(0, 2.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("t=2 translation", animGraph.Nodes()[1].worldTransform.Translation(),
                      10.0f, 20.0f, 0.0f);

            // --- between keyframes ---------------------------------------
            animGraph.ApplyAnimation(0, 0.5f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("t=0.5 translation is halfway",
                      animGraph.Nodes()[1].worldTransform.Translation(),
                      5.0f, 0.0f, 0.0f);

            // --- clamping outside the range ------------------------------
            // glTF clamps rather than extrapolating.
            animGraph.ApplyAnimation(0, -5.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("before the first key clamps",
                      animGraph.Nodes()[1].worldTransform.Translation(),
                      0.0f, 0.0f, 0.0f);
            animGraph.ApplyAnimation(0, 99.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("after the last key clamps",
                      animGraph.Nodes()[1].worldTransform.Translation(),
                      10.0f, 20.0f, 0.0f);

            // --- rotation must be slerped, not lerped --------------------
            // Half way from 0 to 90 degrees about Z is 45 degrees, so +X maps
            // onto (cos45, sin45, 0). A component-wise lerp of the two
            // quaternions, renormalised, gives a different angle.
            animGraph.ApplyAnimation(0, 0.5f);
            animGraph.UpdateTransforms(Matrix::Identity);
            {
                // baseTransform holds the AUTHORED rotation, so the posed one has
                // to come out of the local transform. Decompose is non-const, so
                // this works on a copy.
                Matrix posed = animGraph.Nodes()[1].localTransform;
                Vector3 scaleOut;
                Quaternion rotOut;
                Vector3 transOut;
                Expect("posed transform decomposes",
                       posed.Decompose(scaleOut, rotOut, transOut));
                const Vector3 rotated = Vector3::Transform(Vector3(1.0f, 0.0f, 0.0f), rotOut);
                const float expected = std::sqrt(2.0f) / 2.0f;
                ExpectFloat("slerp halfway: x == cos45", rotated.x, expected);
                ExpectFloat("slerp halfway: y == sin45", rotated.y, expected);
            }

            // --- scale ----------------------------------------------------
            animGraph.ApplyAnimation(0, 1.0f);
            {
                Matrix posed = animGraph.Nodes()[1].localTransform;
                Vector3 scaleOut;
                Quaternion rotOut;
                Vector3 transOut;
                posed.Decompose(scaleOut, rotOut, transOut);
                ExpectFloat("scale at t=1", scaleOut.x, 2.0f);
            }

            // --- STEP holds the previous key ------------------------------
            animGraph.ApplyAnimation(0, 0.9f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("STEP holds the lower key at t=0.9",
                      animGraph.Nodes()[2].worldTransform.Translation(),
                      5.0f, 0.0f, 0.0f);
            animGraph.ApplyAnimation(0, 1.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("STEP takes the next key at t=1",
                      animGraph.Nodes()[2].worldTransform.Translation(),
                      7.0f, 0.0f, 0.0f);

            // --- CUBICSPLINE ---------------------------------------------
            // Zero tangents reduce the Hermite basis to smoothstep, so the
            // midpoint is exactly halfway between 0 and 8. Reading the triplets
            // at the wrong offset would sample a tangent (0) instead.
            animGraph.ApplyAnimation(0, 0.5f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("CUBICSPLINE midpoint with zero tangents",
                      animGraph.Nodes()[3].worldTransform.Translation(),
                      4.0f, 0.0f, 0.0f);
            animGraph.ApplyAnimation(0, 1.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("CUBICSPLINE endpoint reads the value, not a tangent",
                      animGraph.Nodes()[3].worldTransform.Translation(),
                      8.0f, 0.0f, 0.0f);

            // --- untargeted nodes keep their authored pose ----------------
            animGraph.ApplyAnimation(0, 1.0f);
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("the root is untouched by the animation",
                      animGraph.Nodes()[0].worldTransform.Translation(),
                      0.0f, 0.0f, 0.0f);

            // --- reset restores the authored pose exactly -----------------
            animGraph.ResetToBasePose();
            animGraph.UpdateTransforms(Matrix::Identity);
            ExpectVec("reset restores the authored translation",
                      animGraph.Nodes()[1].worldTransform.Translation(),
                      100.0f, 100.0f, 100.0f);

            Expect("out-of-range animation index is rejected",
                   !animGraph.ApplyAnimation(99, 0.0f));
        }

        // An asset with no animations must not look like an error.
        SceneGraph still;
        still.Build(csAsset.get(), 0, noBuffers, 0);
        Expect("an unanimated asset yields no animations", still.Animations().empty());
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all checks passed" << std::endl;
    return 0;
}
