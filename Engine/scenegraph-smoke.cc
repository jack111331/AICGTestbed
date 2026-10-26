// Smoke test for SceneGraph's hierarchy walk and transform composition.
//
// The error-prone part of this is the convention mismatch: fastgltf/glTF store
// matrices column-major and compose parent * child, DirectXMath stores
// row-major and composes child * parent. A transpose or an ordering slip still
// renders *something*, so it is worth asserting numerically rather than looking
// at the screen. No D3D12 device is involved -- SceneGraph only needs buffers
// for primitives, and this asset has no meshes.
#include "pch.h"

#include "MotionRetarget.hpp"
#include "SceneGraph.hpp"
#include "Lightmap.hpp"
#include "ShadowCascades.hpp"

#include <filesystem>

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
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

void ExpectEqualU(const char* what, std::uint32_t got, std::uint32_t expected) {
    const bool ok = got == expected;
    std::printf("  %-52s %6u  expected %6u  %s\n", what, got, expected,
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

// ---------------------------------------------------------------------------
// Retargeting: the rotation port, the configuration, and binding to the real
// Mixamo character.
// ---------------------------------------------------------------------------

// utils/paramUtil.py, momask-codes: the rest direction of each HumanML3D bone.
const DirectX::SimpleMath::Vector3 kRawOffsets[22] = {
    {0, 0, 0},  {1, 0, 0},  {-1, 0, 0}, {0, 1, 0},  {0, -1, 0}, {0, -1, 0},
    {0, 1, 0},  {0, -1, 0}, {0, -1, 0}, {0, 1, 0},  {0, 0, 1},  {0, 0, 1},
    {0, 1, 0},  {1, 0, 0},  {-1, 0, 0}, {0, 0, 1},  {0, -1, 0}, {0, -1, 0},
    {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
};

// A deterministic pose: every joint gets a distinct local rotation, written as
// the cont6d the feature vector would carry. Built by rotating the identity
// frame, so the ground truth is known independently of the decoder.
void BuildSyntheticFeature(std::vector<float>& feature, int joints,
                           std::vector<DirectX::SimpleMath::Quaternion>& locals,
                           float rootAngle) {
    using DirectX::SimpleMath::Matrix;
    using DirectX::SimpleMath::Quaternion;
    using DirectX::SimpleMath::Vector3;

    const int motionDim = 4 + 9 * (joints - 1) + 3 * joints + 4;
    feature.assign(static_cast<std::size_t>(motionDim), 0.0f);
    locals.assign(static_cast<std::size_t>(joints), Quaternion::Identity);

    const int rotationOffset = 4 + (joints - 1) * 3;
    for (int joint = 1; joint < joints; ++joint) {
        // Oblique axis and a distinct angle per joint, so a transposed or
        // mis-ordered composition cannot pass by symmetry.
        Vector3 axis(0.3f + 0.01f * joint, 0.5f, -0.8f + 0.02f * joint);
        axis.Normalize();
        const float angle = 0.15f + 0.05f * static_cast<float>(joint);
        const Quaternion local = Quaternion::CreateFromAxisAngle(axis, angle);
        locals[static_cast<std::size_t>(joint)] = local;

        // cont6d is the first two COLUMNS of the column-vector rotation, which
        // in DirectXMath's row-vector matrix are the first two ROWS.
        const Matrix m = Matrix::CreateFromQuaternion(local);
        float* c = feature.data() + rotationOffset + (joint - 1) * 6;
        c[0] = m._11; c[1] = m._12; c[2] = m._13;
        c[3] = m._21; c[4] = m._22; c[5] = m._23;
    }
    (void)rootAngle;
}

// Cascaded shadow map fitting.
//
// The assertions worth having here are not "does it produce a matrix" but the
// properties that distinguish a working cascade set from one that produces
// shadows in the wrong place:
//
//   * every point inside a slice must land inside that cascade's clip box,
//     or the shadow for it simply does not exist;
//   * a caster BEHIND the slice must still be inside the depth range, or
//     objects stop casting onto things in front of them;
//   * rotating the camera in place must not change the fitted radius, which is
//     what the bounding-sphere fit buys and what stops the edges crawling;
//   * the fitted centre must move in whole texels, which is what stops them
//     shimmering.
void TestShadowCascades() {
    using DirectX::SimpleMath::Matrix;
    using DirectX::SimpleMath::Vector3;
    using DirectX::SimpleMath::Vector4;
    using NeuralModelIntegrateTestbed::ComputeShadowCascades;
    using NeuralModelIntegrateTestbed::ShadowCascades;
    using NeuralModelIntegrateTestbed::ShadowCascadeSetup;

    std::cout << "cascaded shadow maps" << std::endl;

    const Matrix view = Matrix::CreateLookAt(Vector3(0.0f, 2.0f, 20.0f),
                                             Vector3(0.0f, 2.0f, 0.0f), Vector3::UnitY);
    const Matrix projection = Matrix::CreatePerspectiveFieldOfView(
        DirectX::XMConvertToRadians(70.0f), 16.0f / 9.0f, 0.1f, 200.0f);
    // Afternoon sun: travelling down and sideways.
    Vector3 lightDirection(-0.4f, -0.8f, -0.45f);
    lightDirection.Normalize();
    const Vector3 sceneMin(-60.0f, -2.0f, -60.0f);
    const Vector3 sceneMax(60.0f, 30.0f, 60.0f);

    ShadowCascadeSetup setup;
    setup.cascadeCount = 4;
    setup.shadowDistance = 100.0f;
    setup.resolution = 1024;

    const ShadowCascades cascades =
        ComputeShadowCascades(view, projection, lightDirection, sceneMin, sceneMax, setup);
    Expect("four cascades are produced", cascades.count == 4);
    Expect("the atlas is two cascades on a side",
           cascades.atlasResolution == 2048);

    // Splits must be increasing and end at the shadow distance.
    bool increasing = true;
    for (int i = 1; i < cascades.count; ++i) {
        increasing = increasing && cascades.cascades[i].splitDistance >
                                       cascades.cascades[i - 1].splitDistance;
    }
    Expect("split distances increase", increasing);
    ExpectFloat("the last split reaches the shadow distance",
                cascades.cascades[cascades.count - 1].splitDistance, 100.1f);

    // The near cascade must be tighter than the far one, or the split scheme
    // is doing nothing useful.
    Expect("the near cascade has smaller texels than the far one",
           cascades.cascades[0].texelWorldSize <
               cascades.cascades[cascades.count - 1].texelWorldSize);

    // --- containment ---------------------------------------------------
    // Points along the view axis, each inside a known slice, must land in that
    // cascade's clip box.
    const auto insideClip = [](const Matrix& viewProjection, const Vector3& world,
                               float slack) {
        const Vector4 clip = Vector4::Transform(Vector4(world.x, world.y, world.z, 1.0f),
                                                viewProjection);
        if (std::fabs(clip.w) < 1e-9f) return false;
        const Vector3 ndc(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        return ndc.x >= -1.0f - slack && ndc.x <= 1.0f + slack &&
               ndc.y >= -1.0f - slack && ndc.y <= 1.0f + slack &&
               ndc.z >= -slack && ndc.z <= 1.0f + slack;
    };

    bool allContained = true;
    for (int i = 0; i < cascades.count; ++i) {
        const float nearEdge =
            (i == 0) ? 0.1f : cascades.cascades[i - 1].splitDistance;
        const float midway = 0.5f * (nearEdge + cascades.cascades[i].splitDistance);
        // The camera looks down -Z from z = 20.
        const Vector3 point(0.0f, 2.0f, 20.0f - midway);
        const bool contained = insideClip(cascades.cascades[i].viewProjection, point, 0.0f);
        if (!contained) {
            std::printf("    cascade %d does not contain its own midpoint z=%.2f\n", i,
                        point.z);
        }
        allContained = allContained && contained;
    }
    Expect("each cascade contains the midpoint of its own slice", allContained);

    // A caster well above the slice, inside the scene bounds, must be within
    // the depth range: this is what lets a roof shadow the street.
    Expect("a caster above the slice is still inside the light's depth range",
           insideClip(cascades.cascades[1].viewProjection,
                      Vector3(0.0f, 28.0f, 20.0f - cascades.cascades[0].splitDistance -
                                               1.0f),
                      0.0f));

    // --- rotation stability --------------------------------------------
    // Yawing the camera in place must not change the fitted radius. A bounding
    // box fitted to the slice corners would change it by up to 40%, and the
    // shadow texels would change size as you turned.
    const Matrix rotated = Matrix::CreateLookAt(
        Vector3(0.0f, 2.0f, 20.0f),
        Vector3(0.0f, 2.0f, 20.0f) + Vector3(0.7071f, 0.0f, -0.7071f), Vector3::UnitY);
    const ShadowCascades turned = ComputeShadowCascades(
        rotated, projection, lightDirection, sceneMin, sceneMax, setup);
    bool stable = true;
    for (int i = 0; i < cascades.count; ++i) {
        const float before = cascades.cascades[i].texelWorldSize;
        const float after = turned.cascades[i].texelWorldSize;
        stable = stable && std::fabs(before - after) < before * 1e-3f;
    }
    Expect("yawing the camera does not change the texel size", stable);

    // --- texel snapping ------------------------------------------------
    // Nudge the camera by a fraction of a texel and the fitted projection must
    // either not move at all or move by a whole texel. Measured through where
    // a fixed world point lands in shadow-map texels: without snapping this
    // drifts smoothly and every edge shimmers.
    const float texel = cascades.cascades[0].texelWorldSize;
    const Vector3 probe(1.0f, 2.0f, 15.0f);
    const auto texelCoordinate = [&](const Matrix& viewProjection) {
        const Vector4 clip = Vector4::Transform(Vector4(probe.x, probe.y, probe.z, 1.0f),
                                                viewProjection);
        return (clip.x / clip.w * 0.5f + 0.5f) * static_cast<float>(setup.resolution);
    };
    const float base = texelCoordinate(cascades.cascades[0].viewProjection);
    float worstFraction = 0.0f;
    for (int step = 1; step <= 8; ++step) {
        // Sub-texel camera motion along the light's right-ish axis.
        const float nudge = texel * 0.1f * static_cast<float>(step);
        const Matrix moved = Matrix::CreateLookAt(Vector3(nudge, 2.0f, 20.0f),
                                                  Vector3(nudge, 2.0f, 0.0f),
                                                  Vector3::UnitY);
        const ShadowCascades shifted = ComputeShadowCascades(
            moved, projection, lightDirection, sceneMin, sceneMax, setup);
        const float moved_coordinate = texelCoordinate(shifted.cascades[0].viewProjection);
        const float delta = moved_coordinate - base;
        const float fraction = std::fabs(delta - std::round(delta));
        worstFraction = std::max(worstFraction, fraction);
    }
    std::printf("  %-52s %.4f\n", "worst sub-texel drift under camera nudges",
                worstFraction);
    Expect("the projection moves in whole texels", worstFraction < 0.02f);

    // --- holding a fit across camera motion ----------------------------
    // Snapping stabilises the texel grid, NOT the matrix: the centre's
    // component along the light, the fitted radius and the light's eye all
    // slide continuously with the camera. Measured on a Bistro-sized setup,
    // every cascade's matrix changed after 1 mm of motion and 0.01 degrees of
    // yaw, with texels of 1.6 to 24 cm. So nothing can be cached against the
    // matrix alone, and ComputeShadowCascades holds a padded fit instead.
    {
        ShadowCascadeSetup held = setup;
        held.fitMargin = 0.12f;
        const ShadowCascades first = ComputeShadowCascades(
            view, projection, lightDirection, sceneMin, sceneMax, held);
        Expect("every cascade is refitted on the first fit", first.AnyRefitted());
        bool allRefitted = true;
        for (int i = 0; i < first.count; ++i) {
            allRefitted = allRefitted && first.cascades[i].refitted;
        }
        Expect("...and that means all of them", allRefitted);

        // The same camera again must reuse everything.
        const ShadowCascades again = ComputeShadowCascades(
            view, projection, lightDirection, sceneMin, sceneMax, held, &first);
        Expect("a stationary camera refits nothing", !again.AnyRefitted());
        bool identical = true;
        for (int i = 0; i < again.count; ++i) {
            identical = identical && std::memcmp(&again.cascades[i].viewProjection,
                                                 &first.cascades[i].viewProjection,
                                                 sizeof(Matrix)) == 0;
        }
        Expect("a held cascade's matrix is bit-identical", identical);

        // A nudge far smaller than the margin must still reuse cascade 0 --
        // this is the whole point, and what the old code could not do.
        const Matrix nudged = Matrix::CreateLookAt(
            Vector3(0.001f, 2.0f, 20.0f),
            Vector3(0.001f, 2.0f, 0.0f), Vector3::UnitY);
        const ShadowCascades afterNudge = ComputeShadowCascades(
            nudged, projection, lightDirection, sceneMin, sceneMax, held, &first);
        Expect("a 1 mm nudge refits nothing", !afterNudge.AnyRefitted());

        // Yaw in place: a bounding sphere is rotation-invariant in SIZE, but
        // its centre swings along the new view axis, which is what used to
        // force a refit after a hundredth of a degree.
        const Matrix turned = Matrix::CreateLookAt(
            Vector3(0.0f, 2.0f, 20.0f),
            Vector3(0.0f, 2.0f, 20.0f) + Vector3(0.0872f, 0.0f, -0.9962f),
            Vector3::UnitY);
        const ShadowCascades afterYaw = ComputeShadowCascades(
            turned, projection, lightDirection, sceneMin, sceneMax, held, &first);
        Expect("five degrees of yaw still reuses the far cascade",
               !afterYaw.cascades[afterYaw.count - 1].refitted);

        // Far enough and it must refit, or the slice would be sampling a map
        // that does not cover it.
        const Matrix moved = Matrix::CreateLookAt(Vector3(500.0f, 2.0f, 20.0f),
                                                  Vector3(500.0f, 2.0f, 0.0f),
                                                  Vector3::UnitY);
        const ShadowCascades afterMove = ComputeShadowCascades(
            moved, projection, lightDirection, sceneMin, sceneMax, held, &first);
        bool everyCascadeRefitted = true;
        for (int i = 0; i < afterMove.count; ++i) {
            everyCascadeRefitted =
                everyCascadeRefitted && afterMove.cascades[i].refitted;
        }
        Expect("moving 500 m refits every cascade", everyCascadeRefitted);

        // A held cascade must actually CONTAIN its slice, which is the
        // correctness condition the whole scheme rests on. Walk the camera out
        // in small steps and check containment on every frame, held or not.
        bool containedThroughout = true;
        bool sawAHold = false;
        bool sawARefit = false;
        int holds = 0;
        int refits = 0;
        ShadowCascades rolling = ComputeShadowCascades(
            view, projection, lightDirection, sceneMin, sceneMax, held);
        for (int step = 1; step <= 400; ++step) {
            const float x = 0.05f * static_cast<float>(step);
            const Matrix walk = Matrix::CreateLookAt(Vector3(x, 2.0f, 20.0f),
                                                     Vector3(x, 2.0f, 0.0f),
                                                     Vector3::UnitY);
            const ShadowCascades next = ComputeShadowCascades(
                walk, projection, lightDirection, sceneMin, sceneMax, held, &rolling);
            for (int i = 0; i < next.count; ++i) {
                if (next.cascades[i].refitted) {
                    sawARefit = true;
                    ++refits;
                } else {
                    sawAHold = true;
                    ++holds;
                }
            }
            // The near cascade's own slice midpoint must land inside the
            // cascade it is held for.
            const float nearEdge = 0.1f;
            const float midway = 0.5f * (nearEdge + next.cascades[0].splitDistance);
            const Vector3 point(x, 2.0f, 20.0f - midway);
            containedThroughout = containedThroughout &&
                                  insideClip(next.cascades[0].viewProjection, point, 0.0f);
            rolling = next;
        }
        std::printf("  %-52s %d of %d (%.0f%%)\n",
                    "cascade fits held while walking 20 m", holds, holds + refits,
                    100.0 * holds / static_cast<double>(holds + refits));
        Expect("a held cascade still contains its slice while walking",
               containedThroughout);
        Expect("walking 20 m both holds and refits", sawAHold && sawARefit);

        // Margin 0 must behave exactly as the old code did: refit every frame.
        ShadowCascadeSetup unheld = setup;
        unheld.fitMargin = 0.0f;
        const ShadowCascades a = ComputeShadowCascades(
            view, projection, lightDirection, sceneMin, sceneMax, unheld);
        const ShadowCascades b = ComputeShadowCascades(
            nudged, projection, lightDirection, sceneMin, sceneMax, unheld, &a);
        Expect("a zero margin refits on the smallest motion", b.AnyRefitted());

        // Changing the light must invalidate every held fit: the held spheres
        // and the fitted depth range were both chosen against the old one.
        const ShadowCascades relit = ComputeShadowCascades(
            view, projection, Vector3(-0.4f, -0.8f, 0.2f), sceneMin, sceneMax,
            held, &first);
        bool allRelit = true;
        for (int i = 0; i < relit.count; ++i) {
            allRelit = allRelit && relit.cascades[i].refitted;
        }
        Expect("moving the light refits every cascade", allRelit);
    }

    // --- refusals ------------------------------------------------------
    ShadowCascadeSetup degenerate = setup;
    degenerate.shadowDistance = 0.0f;
    Expect("a zero shadow distance produces no cascades",
           ComputeShadowCascades(view, projection, lightDirection, sceneMin, sceneMax,
                                 degenerate)
                   .count == 0);
    Expect("a zero light direction produces no cascades",
           ComputeShadowCascades(view, projection, Vector3::Zero, sceneMin, sceneMax,
                                 setup)
                   .count == 0);

    // A sun straight overhead is the case that breaks an up vector of +Y.
    const ShadowCascades overhead = ComputeShadowCascades(
        view, projection, Vector3(0.0f, -1.0f, 0.0f), sceneMin, sceneMax, setup);
    Expect("a light straight overhead still fits", overhead.count == 4);
    bool overheadFinite = true;
    for (int i = 0; i < overhead.count; ++i) {
        const float* m = &overhead.cascades[i].viewProjection._11;
        for (int k = 0; k < 16; ++k) {
            overheadFinite = overheadFinite && std::isfinite(m[k]);
        }
    }
    Expect("and its matrices are finite", overheadFinite);
}

// Frustum culling, against the same CreateLookAt and
// CreatePerspectiveFieldOfView the application builds its matrices with.
//
// The plane extraction is derived from the clip-space inequalities rather than
// taken from a library, so the convention is exactly what these assertions say
// it is: a box in front of the camera is kept, one behind it is not. Getting a
// sign wrong here culls the entire scene or nothing at all, and both look like
// a different bug.
void TestViewFrustum() {
    using DirectX::SimpleMath::Matrix;
    using DirectX::SimpleMath::Vector3;

    std::cout << "view frustum culling" << std::endl;

    // Looking down -Z from the origin, the way a right-handed view matrix does.
    const Matrix view = Matrix::CreateLookAt(Vector3(0.0f, 0.0f, 10.0f),
                                             Vector3::Zero, Vector3::UnitY);
    const Matrix projection = Matrix::CreatePerspectiveFieldOfView(
        DirectX::XMConvertToRadians(70.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    const NeuralModelIntegrateTestbed::ViewFrustum frustum =
        NeuralModelIntegrateTestbed::ViewFrustum::FromViewProjection(view * projection);

    const Vector3 unit(0.5f, 0.5f, 0.5f);

    Expect("a box at the look-at point is visible",
           frustum.Intersects(Vector3::Zero, unit));
    Expect("a box just inside the near plane is visible",
           frustum.Intersects(Vector3(0.0f, 0.0f, 9.0f), Vector3(0.1f, 0.1f, 0.1f)));

    // Behind the camera. This is the one that matters: the near plane and the
    // far plane are the pair most easily swapped, and swapping them keeps
    // everything in front visible while also keeping everything behind.
    Expect("a box behind the camera is culled",
           !frustum.Intersects(Vector3(0.0f, 0.0f, 20.0f), unit));
    Expect("a box beyond the far plane is culled",
           !frustum.Intersects(Vector3(0.0f, 0.0f, -95.0f), unit));

    // Sideways. At 10 units of depth the horizontal half-extent is
    // 10 * tan(35) * 16/9 == 12.45, so 40 is well outside and 2 is well inside.
    Expect("a box far to the left is culled",
           !frustum.Intersects(Vector3(-40.0f, 0.0f, 0.0f), unit));
    Expect("a box far to the right is culled",
           !frustum.Intersects(Vector3(40.0f, 0.0f, 0.0f), unit));
    Expect("a box slightly off-axis is visible",
           frustum.Intersects(Vector3(2.0f, 0.0f, 0.0f), unit));
    Expect("a box far above is culled",
           !frustum.Intersects(Vector3(0.0f, 40.0f, 0.0f), unit));
    Expect("a box far below is culled",
           !frustum.Intersects(Vector3(0.0f, -40.0f, 0.0f), unit));

    // Conservative, in the direction that keeps things: a box whose centre is
    // outside but which reaches in must not be culled.
    Expect("a box straddling the left plane is kept",
           frustum.Intersects(Vector3(-13.0f, 0.0f, 0.0f), Vector3(2.0f, 2.0f, 2.0f)));

    // A box enclosing the whole frustum is trivially visible, and is the case a
    // naive "are all corners inside" test gets wrong.
    Expect("a box containing the camera is visible",
           frustum.Intersects(Vector3::Zero, Vector3(1000.0f, 1000.0f, 1000.0f)));

    // --- the AABB transform --------------------------------------------
    NeuralModelIntegrateTestbed::PrimitiveResource primitive;
    primitive.hasBounds = true;
    primitive.localMin = {-1.0f, -1.0f, -1.0f};
    primitive.localMax = {1.0f, 1.0f, 1.0f};

    Vector3 centre;
    Vector3 extent;
    Expect("bounds resolve under an identity transform",
           NeuralModelIntegrateTestbed::PrimitiveWorldBounds(primitive, Matrix::Identity,
                                                             centre, extent));
    ExpectFloat("identity keeps the centre at the origin", centre.Length(), 0.0f);
    ExpectFloat("identity keeps the extent at 1", extent.x, 1.0f);

    const Matrix translated = Matrix::CreateTranslation(5.0f, 0.0f, 0.0f);
    NeuralModelIntegrateTestbed::PrimitiveWorldBounds(primitive, translated, centre,
                                                      extent);
    ExpectFloat("translation moves the centre", centre.x, 5.0f);
    ExpectFloat("translation leaves the extent", extent.x, 1.0f);

    const Matrix scaled = Matrix::CreateScale(2.0f, 3.0f, 4.0f);
    NeuralModelIntegrateTestbed::PrimitiveWorldBounds(primitive, scaled, centre, extent);
    ExpectFloat("scale scales the extent in x", extent.x, 2.0f);
    ExpectFloat("scale scales the extent in y", extent.y, 3.0f);
    ExpectFloat("scale scales the extent in z", extent.z, 4.0f);

    // 45 degrees about Z grows a unit cube's x and y extents to sqrt(2): the
    // case a transform that forgets to take absolute values gets wrong, because
    // the contributions cancel instead of adding.
    const Matrix rotated = Matrix::CreateRotationZ(DirectX::XM_PIDIV4);
    NeuralModelIntegrateTestbed::PrimitiveWorldBounds(primitive, rotated, centre, extent);
    ExpectFloat("a 45 degree rotation grows the x extent to sqrt(2)", extent.x,
               std::sqrt(2.0f));
    ExpectFloat("and the y extent to sqrt(2)", extent.y, std::sqrt(2.0f));
    ExpectFloat("and leaves z alone", extent.z, 1.0f);

    // A primitive with no authored bounds must report failure, so the caller
    // knows not to cull it.
    NeuralModelIntegrateTestbed::PrimitiveResource unbounded;
    Expect("a primitive without bounds reports failure",
           !NeuralModelIntegrateTestbed::PrimitiveWorldBounds(unbounded, Matrix::Identity,
                                                              centre, extent));

    // --- end to end ----------------------------------------------------
    // A primitive placed outside the frustum by its node transform alone.
    NeuralModelIntegrateTestbed::PrimitiveWorldBounds(
        primitive, Matrix::CreateTranslation(0.0f, 0.0f, 30.0f), centre, extent);
    Expect("a node transform can move a primitive out of view",
           !frustum.Intersects(centre, extent));
    NeuralModelIntegrateTestbed::PrimitiveWorldBounds(
        primitive, Matrix::CreateTranslation(0.0f, 0.0f, 2.0f), centre, extent);
    Expect("and back into it", frustum.Intersects(centre, extent));
}

// Which order does SimpleMath's Quaternion::Concatenate compose in?
//
// Every rotation chain in the retargeter depends on this, and the two readings
// differ by exactly the kind of error that leaves a character folded rather
// than obviously broken. SimpleMath follows XNA here rather than DirectXMath's
// own XMQuaternionMultiply, so it is worth pinning down in a test rather than
// inferring from a doc comment.
void TestConcatenateOrder() {
    using DirectX::SimpleMath::Quaternion;
    using DirectX::SimpleMath::Vector3;

    std::cout << "quaternion composition order" << std::endl;

    // Rotate +X by 90 degrees about Z -> +Y. Then 90 degrees about X -> +Z.
    const Quaternion aboutZ =
        Quaternion::CreateFromAxisAngle(Vector3::UnitZ, DirectX::XM_PIDIV2);
    const Quaternion aboutX =
        Quaternion::CreateFromAxisAngle(Vector3::UnitX, DirectX::XM_PIDIV2);

    const Vector3 start = Vector3::UnitX;
    const Vector3 stepwise =
        Vector3::Transform(Vector3::Transform(start, aboutZ), aboutX);

    const Vector3 zThenX = Vector3::Transform(start, Quaternion::Concatenate(aboutZ, aboutX));
    const Vector3 xThenZ = Vector3::Transform(start, Quaternion::Concatenate(aboutX, aboutZ));

    std::printf("  stepwise (Z then X)      (%.3f, %.3f, %.3f)\n", stepwise.x,
                stepwise.y, stepwise.z);
    std::printf("  Concatenate(Z, X)        (%.3f, %.3f, %.3f)\n", zThenX.x,
                zThenX.y, zThenX.z);
    std::printf("  Concatenate(X, Z)        (%.3f, %.3f, %.3f)\n", xThenZ.x,
                xThenZ.y, xThenZ.z);

    const bool firstArgFirst = (zThenX - stepwise).Length() < 1e-5f;
    const bool secondArgFirst = (xThenZ - stepwise).Length() < 1e-5f;
    Expect("exactly one ordering matches stepwise application",
           firstArgFirst != secondArgFirst);
    if (firstArgFirst) {
        std::printf("  Concatenate(a, b) applies A FIRST, then b\n");
    } else {
        std::printf("  Concatenate(a, b) applies B FIRST, then a\n");
    }
    // Recorded so a change in SimpleMath's convention breaks here, loudly,
    // rather than silently folding every retargeted limb. Measured: the SECOND
    // argument is applied first.
    Expect("Concatenate(a, b) applies b first", secondArgFirst);
}

void TestRetargetRotations() {
    using DirectX::SimpleMath::Quaternion;
    using DirectX::SimpleMath::Vector3;

    std::cout << "retarget: source world rotations" << std::endl;

    NeuralModelIntegrateTestbed::MotionRetargeter retargeter;
    std::string error;
    const std::filesystem::path config = "resources/retarget/mixamo.txt";
    if (!std::filesystem::exists(config)) {
        std::printf("  %-52s skipped\n", "resources/retarget/mixamo.txt absent");
        return;
    }
    if (!retargeter.LoadConfig(config, &error)) {
        std::printf("  loading failed: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    const int joints = retargeter.SourceJointCount();
    Expect("22 source joints", joints == 22);
    Expect("18 mapped bones", retargeter.Bones().size() == 18);
    Expect("five kinematic chains", retargeter.Chains().size() == 5);

    // The hips correction is a 150-degree turn about X, which is what
    // momask's mapping specifies and what the file's own quaternion field
    // independently agrees with.
    const NeuralModelIntegrateTestbed::RetargetBone& hips = retargeter.Bones()[0];
    Expect("the first bone is the hips", hips.sourceJoint == 0 &&
                                             hips.destinationName == "mixamorig:Hips");
    Expect("only the hips carry position", hips.setPosition);
    const float hipsAngle = 2.0f * std::acos(std::min(1.0f, std::fabs(hips.correction.w)));
    Expect("hips correction is 150 degrees",
           std::fabs(DirectX::XMConvertToDegrees(hipsAngle) - 150.0f) < 0.5f);

    // --- the port itself ---------------------------------------------------
    // Forward kinematics from the recovered world rotations has to reproduce
    // the positions the same pose implies. This is the check that catches a
    // wrong cont6d column order, a transposed rotation, or a reversed
    // quaternion composition -- all of which still produce a moving skeleton.
    std::vector<float> feature;
    std::vector<Quaternion> locals;
    const float rootAngle = 0.7f;
    BuildSyntheticFeature(feature, joints, locals, rootAngle);

    std::vector<Quaternion> world;
    Expect("source rotations recovered",
           retargeter.SourceWorldRotations(feature.data(),
                                           static_cast<int>(feature.size()), rootAngle,
                                           world));
    Expect("one rotation per joint", world.size() == static_cast<std::size_t>(joints));

    // Independent reference: accumulate the same locals down each chain.
    std::vector<Quaternion> expected(static_cast<std::size_t>(joints),
                                     Quaternion::Identity);
    // TWICE the angle, because HumanML3D stores the root as (cos a, 0, sin a, 0)
    // and a unit quaternion whose w is cos(a) turns by 2a. This reference used
    // to seed cos(a/2) and so pinned the half-rotation the code used to apply;
    // //Engine:retarget-probe measured the difference (67.0 -> 51.7 degrees of
    // implied rest-offset scatter), which is what identified it as a bug rather
    // than a convention. Seeding it right here is what keeps this test an
    // independent check of the ACCUMULATION rather than of the seed.
    expected[0] = Quaternion::CreateFromAxisAngle(Vector3::UnitY, 2.0f * rootAngle);
    float worstRotation = 0.0f;
    for (const auto& chain : retargeter.Chains()) {
        Quaternion accumulated = expected[static_cast<std::size_t>(chain.front())];
        for (std::size_t k = 1; k < chain.size(); ++k) {
            const int joint = chain[k];
            accumulated = Quaternion::Concatenate(
                accumulated, locals[static_cast<std::size_t>(joint)]);
            expected[static_cast<std::size_t>(joint)] = accumulated;
        }
    }
    for (int joint = 0; joint < joints; ++joint) {
        // Compare by what the rotation DOES to an oblique vector, so a sign
        // flip on the quaternion is not counted as a difference.
        const Vector3 probe(0.37f, -0.62f, 0.69f);
        const Vector3 a = Vector3::Transform(probe, world[static_cast<std::size_t>(joint)]);
        const Vector3 b =
            Vector3::Transform(probe, expected[static_cast<std::size_t>(joint)]);
        worstRotation = std::max(worstRotation, (a - b).Length());
    }
    std::printf("  %-52s %.6f\n", "worst world-rotation disagreement", worstRotation);
    Expect("world rotations match a chain-accumulated reference",
           worstRotation < 1e-4f);

    // Every recovered rotation must be a rotation: unit quaternion, so the
    // Gram-Schmidt in the port produced an orthonormal frame.
    float worstNorm = 0.0f;
    for (const Quaternion& q : world) {
        const float length =
            std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        worstNorm = std::max(worstNorm, std::fabs(length - 1.0f));
    }
    Expect("every rotation is unit length", worstNorm < 1e-4f);

    // A degenerate cont6d (two parallel vectors) describes no frame. It must
    // come back as identity rather than NaN, which would otherwise propagate
    // along the whole chain and poison every descendant.
    std::vector<float> degenerate = feature;
    const int rotationOffset = 4 + (joints - 1) * 3;
    for (int i = 0; i < 6; ++i) {
        degenerate[static_cast<std::size_t>(rotationOffset + i)] = (i % 3 == 0) ? 1.0f : 0.0f;
    }
    std::vector<Quaternion> fromDegenerate;
    retargeter.SourceWorldRotations(degenerate.data(),
                                    static_cast<int>(degenerate.size()), rootAngle,
                                    fromDegenerate);
    bool finite = true;
    for (const Quaternion& q : fromDegenerate) {
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) ||
            !std::isfinite(q.w)) {
            finite = false;
        }
    }
    Expect("a degenerate rotation does not produce NaN", finite);

    // A short feature vector has to be refused, not read past.
    std::vector<Quaternion> ignored;
    Expect("a too-short feature frame is refused",
           !retargeter.SourceWorldRotations(feature.data(), 10, rootAngle, ignored));
    Expect("a null feature frame is refused",
           !retargeter.SourceWorldRotations(nullptr, static_cast<int>(feature.size()),
                                            rootAngle, ignored));
}

void TestRetargetBinding() {
    std::cout << "retarget: binding to the character" << std::endl;

    const std::filesystem::path character =
        "resources/microphone_and_animated_char/scene_withlight_animated_char.gltf";
    const std::filesystem::path config = "resources/retarget/mixamo.txt";
    if (!std::filesystem::exists(character) || !std::filesystem::exists(config)) {
        std::printf("  %-52s skipped\n", "character or config absent");
        return;
    }

    NeuralModelIntegrateTestbed::MotionRetargeter retargeter;
    std::string error;
    if (!retargeter.LoadConfig(config, &error)) {
        std::printf("  loading failed: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    // Everything down to the position-driven section below is about the
    // ROTATION path, which these synthetic features are built to exercise:
    // they carry a known 6D block and no positions at all. The shipped default
    // is RetargetSource::Positions, so it has to be said out loud here.
    retargeter.SetSource(NeuralModelIntegrateTestbed::RetargetSource::Rotations);

    fastgltf::Parser parser(fastgltf::Extensions::KHR_lights_punctual);
    auto data = fastgltf::GltfDataBuffer::FromPath(character);
    if (data.error() != fastgltf::Error::None) {
        std::printf("  cannot read the character: %s\n",
                    fastgltf::getErrorMessage(data.error()).data());
        ++g_failures;
        return;
    }
    auto asset = parser.loadGltf(data.get(), character.parent_path(),
                                 fastgltf::Options::LoadExternalBuffers);
    if (asset.error() != fastgltf::Error::None) {
        std::printf("  cannot parse the character: %s\n",
                    fastgltf::getErrorMessage(asset.error()).data());
        ++g_failures;
        return;
    }

    SceneGraph graph;
    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> noBuffers;
    graph.Build(asset.get(), asset->defaultScene.value_or(0), noBuffers, 0);
    std::printf("  %-52s %zu\n", "character nodes", graph.NodeCount());
    std::printf("  %-52s %zu\n", "skins", graph.Skins().size());

    // --- discovery ---------------------------------------------------------
    const std::vector<NeuralModelIntegrateTestbed::RetargetTarget> targets =
        retargeter.DiscoverTargets(graph);
    std::printf("  %-52s %zu\n", "retarget targets discovered", targets.size());
    Expect("one target per skin", targets.size() == graph.Skins().size());
    if (!targets.empty()) {
        const NeuralModelIntegrateTestbed::RetargetTarget& first = targets[0];
        std::printf("    %s: %d joints, %d/%zu bones matched\n", first.name.c_str(),
                    first.jointCount, first.matchedBones,
                    retargeter.Bones().size());
        // Named after the node instancing the skin, which is what Mixamo names
        // after the character.
        Expect("the target is named after the character node", first.name == "Ch43");
        Expect("the target reports the skin's joint count", first.jointCount == 65);
        Expect("the mapping covers the whole rig",
               first.matchedBones == static_cast<int>(retargeter.Bones().size()));
        Expect("the target knows its mesh node", first.meshNode.has_value());
    }

    // --- scoped binding ----------------------------------------------------
    // Every Mixamo rig names its bones "mixamorig:*", so an unscoped search
    // would bind to whichever character came first. BindTo must only ever
    // resolve to joints of the skin it was given.
    {
        std::vector<std::string> scopedMissing;
        const int scoped = retargeter.BindTo(graph, 0, &scopedMissing);
        std::printf("  %-52s %d\n", "bones bound within skin 0", scoped);
        Expect("BindTo bound the whole rig",
               scoped == static_cast<int>(retargeter.Bones().size()));
        Expect("BoundSkin reports the skin",
               retargeter.BoundSkin().has_value() && *retargeter.BoundSkin() == 0);

        const std::vector<std::size_t>& joints = graph.Skins()[0].joints;
        bool allWithinSkin = true;
        for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
             retargeter.Bones()) {
            if (!bone.destinationNode.has_value()) continue;
            if (std::find(joints.begin(), joints.end(), *bone.destinationNode) ==
                joints.end()) {
                allWithinSkin = false;
            }
        }
        Expect("every bound node is a joint of that skin", allWithinSkin);

        std::vector<std::string> badMissing;
        Expect("binding to a skin that does not exist binds nothing",
               retargeter.BindTo(graph, 99, &badMissing) == 0);
        Expect("  and clears the bound skin", !retargeter.BoundSkin().has_value());
        Expect("  and leaves no bone bound", retargeter.BoundCount() == 0);
    }

    std::vector<std::string> missing;
    const int bound = retargeter.Bind(graph, &missing);
    std::printf("  %-52s %d of %zu\n", "bones bound by name", bound,
                retargeter.Bones().size());
    for (const std::string& name : missing) {
        std::printf("    unmatched: %s\n", name.c_str());
    }
    Expect("every mapped bone found its node",
           bound == static_cast<int>(retargeter.Bones().size()));
    Expect("BoundCount agrees", retargeter.BoundCount() == bound);

    // Posing has to run end to end on the real hierarchy.
    std::vector<float> feature;
    std::vector<DirectX::SimpleMath::Quaternion> locals;
    BuildSyntheticFeature(feature, retargeter.SourceJointCount(), locals, 0.3f);
    // --- bind-pose measurement, which is what sizes the character ----------
    const NeuralModelIntegrateTestbed::BindPoseMetrics metrics =
        retargeter.MeasureBindPose(graph, 0);
    Expect("the bind pose measured", metrics.valid);
    std::printf("  %-52s %.4f\n", "hips height above the lowest joint",
                metrics.hipsHeight);
    std::printf("  %-52s %.4f .. %.4f\n", "joint Y range", metrics.lowestY,
                metrics.highestY);
    // This character stands on its own ground plane already, so the lowest
    // joint is at zero and the hips are a leg's length above it. Measured from
    // the asset, so a change here means the asset changed.
    Expect("the character stands at about zero", std::fabs(metrics.lowestY) < 0.02f);
    Expect("the hips are about 0.77 m up",
           std::fabs(metrics.hipsHeight - 0.774f) < 0.02f);
    Expect("the measurement excludes the scene placement",
           std::fabs(metrics.hipsY - 0.776f) < 0.02f);

    // Measuring must not depend on the current pose: a retarget already
    // applied to the graph must not change what the rest pose measures.
    std::vector<float> posed;
    std::vector<DirectX::SimpleMath::Quaternion> posedLocals;
    BuildSyntheticFeature(posed, retargeter.SourceJointCount(), posedLocals, 1.1f);
    retargeter.ApplyFrame(graph, posed.data(), static_cast<int>(posed.size()),
                          nullptr, 1.1f,
                          DirectX::SimpleMath::Vector3(0.0f, 0.9f, 0.0f), 1.0f,
                          DirectX::SimpleMath::Matrix::Identity, &error);
    const NeuralModelIntegrateTestbed::BindPoseMetrics again =
        retargeter.MeasureBindPose(graph, 0);
    Expect("the rest pose measures the same after posing",
           std::fabs(again.hipsHeight - metrics.hipsHeight) < 1e-4f);

    // --- the motion follows the character's own placement ------------------
    // The target is in the character's FILE space, so the hips must land where
    // that file's placement puts it -- not at the raw coordinates. This is the
    // property that makes the motion behave like everything else in its glTF:
    // move the file and the character goes with it.
    //
    // It used to assert the opposite, that the hips landed at the requested
    // WORLD position whatever the placement was. That made the motion the one
    // thing in the scene that did not follow its own file, and it disagreed
    // with the bone rotations, which always composed onto the placement.
    const auto hipsWorld = [&](const char* what,
                               const DirectX::SimpleMath::Vector3& expected) {
        for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
             retargeter.Bones()) {
            if (bone.sourceJoint != 0 || !bone.destinationNode.has_value()) continue;
            const DirectX::SimpleMath::Matrix& m =
                graph.Nodes()[*bone.destinationNode].worldTransform;
            const DirectX::SimpleMath::Vector3 got(m._41, m._42, m._43);
            std::printf("  %-52s (%.3f, %.3f, %.3f)\n", what, got.x, got.y, got.z);
            ExpectVec(what, got, expected.x, expected.y, expected.z);
            return;
        }
    };

    const DirectX::SimpleMath::Matrix placement =
        DirectX::SimpleMath::Matrix::CreateTranslation(3.0f, -2.0f, -4.0f);
    const DirectX::SimpleMath::Vector3 wanted(0.5f, 0.9f, -0.25f);
    Expect("a frame applies with a scene placement",
           retargeter.ApplyFrame(graph, feature.data(),
                                 static_cast<int>(feature.size()), nullptr, 0.0f,
                                 wanted, 1.0f, placement, &error));
    graph.UpdateTransforms(placement);
    hipsWorld("hips carried by the scene placement",
              DirectX::SimpleMath::Vector3(3.5f, -1.1f, -4.25f));

    // --- whose size wins ----------------------------------------------------
    //
    // The motion is brought down to the character, not the character up to the
    // motion. This used to be the other way round: the ratio was written onto
    // the hips node as a scale, which resized the rig and the skinned mesh and
    // silently discarded whatever size the character's file and its placement
    // had given it -- so a character deliberately placed at 0.8 changed size
    // the moment generated motion drove it, while glTF animation (which writes
    // no scale) left it alone.
    std::size_t hipsNode = graph.Nodes().size();
    for (const NeuralModelIntegrateTestbed::RetargetBone& bone : retargeter.Bones()) {
        if (bone.sourceJoint == 0 && bone.destinationNode.has_value()) {
            hipsNode = *bone.destinationNode;
            break;
        }
    }
    Expect("the hips bone names a node", hipsNode < graph.Nodes().size());
    if (hipsNode < graph.Nodes().size()) {
        const DirectX::SimpleMath::Vector3 authoredScale =
            graph.Nodes()[hipsNode].hasTrs
                ? graph.Nodes()[hipsNode].baseTransform.scale
                : DirectX::SimpleMath::Vector3(1.0f, 1.0f, 1.0f);

        // A motion twice the character's size: the target halves, and nothing
        // about the rig changes.
        Expect("a frame applies with the motion twice the character's size",
               retargeter.ApplyFrame(graph, feature.data(),
                                     static_cast<int>(feature.size()), nullptr, 0.0f,
                                     wanted, 2.0f, placement, &error));
        graph.UpdateTransforms(placement);
        hipsWorld("hips at half the requested height",
                  DirectX::SimpleMath::Vector3(3.25f, -1.55f, -4.125f));

        DirectX::SimpleMath::Vector3 posedScale;
        DirectX::SimpleMath::Quaternion posedRotation;
        DirectX::SimpleMath::Vector3 posedTranslation;
        // Decompose is not const, so the transform is copied out first.
        DirectX::SimpleMath::Matrix posedLocal = graph.Nodes()[hipsNode].localTransform;
        Expect("the posed hips transform decomposes",
               posedLocal.Decompose(posedScale, posedRotation, posedTranslation));
        std::printf("  %-52s (%.3f, %.3f, %.3f)\n", "hips scale after retargeting",
                    posedScale.x, posedScale.y, posedScale.z);
        ExpectVec("the retarget left the character's own size alone", posedScale,
                  authoredScale.x, authoredScale.y, authoredScale.z);

        // And when an override DOES carry a scale, it composes with the
        // authored one rather than replacing it -- deliberately unlike a glTF
        // animation scale channel, because an override is a pose laid on a
        // character something else has already sized.
        SceneGraph::PoseOverride scaled;
        scaled.nodeIndex = hipsNode;
        scaled.rotation = DirectX::SimpleMath::Quaternion::Identity;
        scaled.hasScale = true;
        scaled.scale = DirectX::SimpleMath::Vector3(2.0f, 2.0f, 2.0f);
        Expect("an override with a scale applies",
               graph.ApplyPoseOverrides({scaled}));
        posedLocal = graph.Nodes()[hipsNode].localTransform;
        Expect("the composed hips transform decomposes",
               posedLocal.Decompose(posedScale, posedRotation, posedTranslation));
        ExpectVec("an override scale composes with the authored one", posedScale,
                  authoredScale.x * 2.0f, authoredScale.y * 2.0f,
                  authoredScale.z * 2.0f);
    }

    // Now the file's OWN placement, which is what a manifest or the Scene tab
    // sets per glTF. Translation, a quarter turn and a scale at once, because
    // each one fails differently: a missed translation leaves the character
    // behind, a missed rotation walks it the wrong way while its body faces the
    // right one, and a missed scale gives a giant with a tiny stride.
    //
    // +X at scale 2 is (1, 0, 0) -> (2, 0, 0), a 90-degree yaw takes that onto
    // (0, 0, -2), and the move adds (10, 0, 0) -- so (10, 1.8, -2). The 1.8 is
    // the target's own 0.9 doubled.
    {
        NeuralModelIntegrateTestbed::SceneTransform assetPlacement;
        assetPlacement.translation[0] = 10.0f;
        assetPlacement.rotation[1] = 90.0f;
        assetPlacement.scale = 2.0f;
        graph.SetAssetPlacement(0, NeuralModelIntegrateTestbed::PlacementMatrix(
                                       assetPlacement));

        const DirectX::SimpleMath::Vector3 inFile(1.0f, 0.9f, 0.0f);
        Expect("a frame applies with the file placed, turned and scaled",
               retargeter.ApplyFrame(graph, feature.data(),
                                     static_cast<int>(feature.size()), nullptr, 0.0f,
                                     inFile, 1.0f,
                                     DirectX::SimpleMath::Matrix::Identity,
                                     &error));
        graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
        hipsWorld("hips carried by the file's own placement",
                  DirectX::SimpleMath::Vector3(10.0f, 1.8f, -2.0f));

        // Both placements at once: the file's own, then the whole scene's
        // outside it. The order is what ScenePlacement and SetAssetPlacement
        // promise, and getting it backwards would scale the scene offset.
        Expect("a frame applies with both placements",
               retargeter.ApplyFrame(graph, feature.data(),
                                     static_cast<int>(feature.size()), nullptr, 0.0f,
                                     inFile, 1.0f, placement, &error));
        graph.UpdateTransforms(placement);
        hipsWorld("hips carried by both placements",
                  DirectX::SimpleMath::Vector3(13.0f, -0.2f, -6.0f));

        // And the ORIENTATION, which is the other half of "follows the
        // file's transform". It already did -- a driven bone's local rotation
        // is derived against the authored parent chain, and UpdateTransforms
        // composes the placement on top of that -- but the translation did
        // not, and the two disagreeing was the bug. This pins them together:
        // the hips' world axes under a quarter-turned file must be the
        // identity-placement axes, quarter-turned.
        const auto hipsAxis = [&]() {
            for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
                 retargeter.Bones()) {
                if (bone.sourceJoint != 0 || !bone.destinationNode.has_value()) {
                    continue;
                }
                DirectX::SimpleMath::Matrix world =
                    graph.Nodes()[*bone.destinationNode].worldTransform;
                world._41 = world._42 = world._43 = 0.0f;
                DirectX::SimpleMath::Vector3 axis =
                    DirectX::SimpleMath::Vector3::TransformNormal(
                        DirectX::SimpleMath::Vector3::UnitX, world);
                axis.Normalize();
                return axis;
            }
            return DirectX::SimpleMath::Vector3::Zero;
        };

        graph.SetAssetPlacement(0, DirectX::SimpleMath::Matrix::Identity);
        retargeter.ApplyFrame(graph, feature.data(),
                              static_cast<int>(feature.size()), nullptr, 0.0f, inFile,
                              1.0f, DirectX::SimpleMath::Matrix::Identity, &error);
        graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
        const DirectX::SimpleMath::Vector3 unplacedAxis = hipsAxis();

        graph.SetAssetPlacement(0, NeuralModelIntegrateTestbed::PlacementMatrix(
                                       assetPlacement));
        retargeter.ApplyFrame(graph, feature.data(),
                              static_cast<int>(feature.size()), nullptr, 0.0f, inFile,
                              1.0f, DirectX::SimpleMath::Matrix::Identity, &error);
        graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
        const DirectX::SimpleMath::Vector3 placedAxis = hipsAxis();

        // A 90-degree yaw takes (x, y, z) onto (z, y, -x), which is the same
        // mapping TestPlacementMatrix pins by sending +X onto -Z.
        ExpectVec("the hips' orientation follows the file's rotation", placedAxis,
                  unplacedAxis.z, unplacedAxis.y, -unplacedAxis.x);

        // Put it back, so what follows measures the character where the rest of
        // this test expects it.
        graph.SetAssetPlacement(0, DirectX::SimpleMath::Matrix::Identity);
    }

    const bool applied = retargeter.ApplyFrame(
        graph, feature.data(), static_cast<int>(feature.size()), nullptr, 0.3f,
        DirectX::SimpleMath::Vector3(0.0f, 0.9f, 0.0f), 1.0f,
        DirectX::SimpleMath::Matrix::Identity, &error);
    Expect("a frame applies to the real character", applied);
    if (!applied) {
        std::printf("    %s\n", error.c_str());
    }

    // The driven nodes' transforms must be finite and must actually have
    // changed -- a retarget that silently leaves the bind pose alone would
    // otherwise look like success.
    graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);
    bool finite = true;
    int moved = 0;
    for (const NeuralModelIntegrateTestbed::RetargetBone& bone : retargeter.Bones()) {
        if (!bone.destinationNode.has_value()) continue;
        const SceneNode& node = graph.Nodes()[*bone.destinationNode];
        const DirectX::SimpleMath::Matrix& m = node.localTransform;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                if (!std::isfinite(m.m[row][col])) finite = false;
            }
        }
        DirectX::SimpleMath::Matrix copy = node.localTransform;
        DirectX::SimpleMath::Vector3 scale;
        DirectX::SimpleMath::Quaternion rotation;
        DirectX::SimpleMath::Vector3 translation;
        if (copy.Decompose(scale, rotation, translation)) {
            DirectX::SimpleMath::Quaternion delta = node.baseTransform.rotation;
            delta.Conjugate();
            delta = DirectX::SimpleMath::Quaternion::Concatenate(rotation, delta);
            if (std::fabs(delta.w) < 0.9999f) ++moved;
        }
    }
    Expect("every driven transform is finite", finite);

    // --- the property that decides whether the pose is right ---------------
    // Each driven bone must end up pointing where the motion points it. This
    // is what momask's correction constants failed at (about 90 degrees out,
    // because they were tuned for a Blender BVH armature's bone axes) and what
    // the derived rest alignment fixes. A character whose bones point the wrong
    // way still animates, and still looks like a character -- it just folds its
    // limbs and floats above the ground, which is exactly how this surfaced.
    {
        std::vector<DirectX::SimpleMath::Quaternion> sourceWorld;
        retargeter.SourceWorldRotations(feature.data(),
                                        static_cast<int>(feature.size()), 0.3f,
                                        sourceWorld);
        float worstAngle = 0.0f;
        int compared = 0;
        for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
             retargeter.Bones()) {
            if (!bone.restResolved || !bone.destinationNode.has_value() ||
                !bone.childDestinationNode.has_value() || bone.childJoint < 0) {
                continue;
            }
            // Where the motion puts this bone: the child joint's world rotation
            // applied to the child's rest offset.
            DirectX::SimpleMath::Vector3 want = DirectX::SimpleMath::Vector3::Transform(
                kRawOffsets[bone.childJoint],
                sourceWorld[static_cast<std::size_t>(bone.childJoint)]);
            if (want.Length() < 1e-6f) continue;
            want.Normalize();

            const DirectX::SimpleMath::Matrix& a =
                graph.Nodes()[*bone.destinationNode].worldTransform;
            const DirectX::SimpleMath::Matrix& b =
                graph.Nodes()[*bone.childDestinationNode].worldTransform;
            DirectX::SimpleMath::Vector3 got(b._41 - a._41, b._42 - a._42,
                                             b._43 - a._43);
            if (got.Length() < 1e-6f) continue;
            got.Normalize();

            const float dot = std::max(-1.0f, std::min(1.0f, got.Dot(want)));
            worstAngle = std::max(worstAngle,
                                  DirectX::XMConvertToDegrees(std::acos(dot)));
            ++compared;
        }
        // Bisection: does the FORMULA predict the right direction, and does the
        // applied pose match the formula? Separating the two says whether a
        // disagreement lives in the composition or in how the local rotation is
        // derived from it.
        float worstFormula = 0.0f;
        float worstApplied = 0.0f;
        for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
             retargeter.Bones()) {
            if (!bone.restResolved || !bone.destinationNode.has_value() ||
                !bone.childDestinationNode.has_value()) {
                continue;
            }
            DirectX::SimpleMath::Vector3 want =
                DirectX::SimpleMath::Vector3::Transform(
                    kRawOffsets[bone.childJoint],
                    sourceWorld[static_cast<std::size_t>(bone.childJoint)]);
            want.Normalize();

            const DirectX::SimpleMath::Quaternion W =
                DirectX::SimpleMath::Quaternion::Concatenate(
                    sourceWorld[static_cast<std::size_t>(bone.childJoint)],
                    DirectX::SimpleMath::Quaternion::Concatenate(
                        bone.restAlignment, bone.restRotation));

            DirectX::SimpleMath::Quaternion inverseRest = bone.restRotation;
            inverseRest.Conjugate();
            const DirectX::SimpleMath::Vector3 localDir =
                DirectX::SimpleMath::Vector3::Transform(bone.restDirection,
                                                        inverseRest);
            DirectX::SimpleMath::Vector3 predicted =
                DirectX::SimpleMath::Vector3::Transform(localDir, W);
            predicted.Normalize();
            worstFormula = std::max(
                worstFormula, DirectX::XMConvertToDegrees(std::acos(std::max(
                                  -1.0f, std::min(1.0f, predicted.Dot(want))))));

            const DirectX::SimpleMath::Matrix& fromNode =
                graph.Nodes()[*bone.destinationNode].worldTransform;
            const DirectX::SimpleMath::Matrix& toNode =
                graph.Nodes()[*bone.childDestinationNode].worldTransform;
            DirectX::SimpleMath::Vector3 actual(toNode._41 - fromNode._41,
                                                toNode._42 - fromNode._42,
                                                toNode._43 - fromNode._43);
            if (actual.Length() > 1e-6f) {
                actual.Normalize();
                worstApplied = std::max(
                    worstApplied,
                    DirectX::XMConvertToDegrees(std::acos(std::max(
                        -1.0f, std::min(1.0f, actual.Dot(predicted))))));
            }
        }
        std::printf("  %-52s %.4f deg\n", "formula vs motion", worstFormula);
        std::printf("  %-52s %.4f deg\n", "applied pose vs formula", worstApplied);

        std::printf("  %-52s %d\n", "bones compared against the motion", compared);
        std::printf("  %-52s %.4f deg\n", "worst bone-direction error", worstAngle);
        Expect("enough bones to be meaningful", compared >= 12);
        Expect("every bone points where the motion points it", worstAngle < 0.5f);
    }
    std::printf("  %-52s %d\n", "driven bones that left the bind pose", moved);
    Expect("the pose actually changed the rig", moved >= 12);

    // --- the position-driven path ------------------------------------------
    // The same property as above, from the other encoding: each driven bone
    // must end up pointing along normalise(position[child] - position[parent]).
    // That is this path's whole claim, and it holds BY CONSTRUCTION -- no rest
    // alignment, no root convention, nothing of what //Engine:retarget-probe
    // has to sweep over for the rotation block.
    //
    // The positions are deliberately not a plausible pose: a spiral, so every
    // joint gets a different, non-cardinal direction and a bone wired to the
    // wrong joint cannot pass by accident the way it could in a pose where a
    // whole limb points one way.
    {
        retargeter.SetSource(NeuralModelIntegrateTestbed::RetargetSource::Positions);

        const int sourceJoints = retargeter.SourceJointCount();
        std::vector<DirectX::XMFLOAT3> positions(
            static_cast<std::size_t>(sourceJoints));
        for (int j = 0; j < sourceJoints; ++j) {
            const float t = static_cast<float>(j);
            positions[static_cast<std::size_t>(j)] = {
                0.31f * std::cos(0.7f * t), 0.19f * t - 0.4f,
                0.23f * std::sin(0.7f * t) + 0.05f * t};
        }

        std::vector<DirectX::SimpleMath::Vector3> directions;
        Expect("bone directions come out of the positions",
               retargeter.SourceBoneDirections(positions.data(), directions));
        Expect("the root has no direction of its own",
               directions[0].LengthSquared() < 1e-12f);

        Expect("a frame applies from positions alone",
               retargeter.ApplyFrame(graph, nullptr, 0, positions.data(), 0.0f,
                                     DirectX::SimpleMath::Vector3(0.0f, 0.9f, 0.0f),
                                     1.0f, DirectX::SimpleMath::Matrix::Identity,
                                     &error));
        graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);

        float worstDirection = 0.0f;
        float worstTwist = 0.0f;
        int comparedFromPositions = 0;
        for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
             retargeter.Bones()) {
            if (!bone.restResolved || !bone.destinationNode.has_value() ||
                !bone.childDestinationNode.has_value() || bone.childJoint < 0) {
                continue;
            }
            const DirectX::SimpleMath::Vector3& want =
                directions[static_cast<std::size_t>(bone.childJoint)];
            if (want.LengthSquared() < 1e-12f) continue;

            const DirectX::SimpleMath::Matrix& a =
                graph.Nodes()[*bone.destinationNode].worldTransform;
            const DirectX::SimpleMath::Matrix& b =
                graph.Nodes()[*bone.childDestinationNode].worldTransform;
            DirectX::SimpleMath::Vector3 got(b._41 - a._41, b._42 - a._42,
                                             b._43 - a._43);
            if (got.Length() < 1e-6f) continue;
            got.Normalize();
            worstDirection = std::max(
                worstDirection, DirectX::XMConvertToDegrees(std::acos(std::max(
                                    -1.0f, std::min(1.0f, got.Dot(want))))));
            ++comparedFromPositions;

            // And the roll, which a direction cannot carry and this path
            // therefore must not invent. A minimal arc turns about an axis
            // perpendicular to both directions, so it has no component along
            // the bone: whatever twist the character was authored with is what
            // it keeps. Checking the axis rather than the result, because a
            // wrong-but-pointing-the-right-way rotation passes the test above.
            const DirectX::SimpleMath::Quaternion arc =
                NeuralModelIntegrateTestbed::MotionRetargeter::AlignDirections(
                    bone.restDirection, want);
            DirectX::SimpleMath::Vector3 axis(arc.x, arc.y, arc.z);
            if (axis.Length() > 1e-6f) {
                axis.Normalize();
                worstTwist = std::max(worstTwist, std::fabs(axis.Dot(want)));
            }
        }
        std::printf("  %-52s %d\n", "bones compared against the positions",
                    comparedFromPositions);
        std::printf("  %-52s %.4f deg\n",
                    "worst direction error against the positions", worstDirection);
        std::printf("  %-52s %.6f\n", "worst twist about the bone", worstTwist);
        Expect("enough bones to be meaningful", comparedFromPositions >= 12);
        Expect("every bone points along the measured direction",
               worstDirection < 0.5f);
        Expect("the arc adds no twist of its own", worstTwist < 1e-4f);

        // --- the degree of freedom the test above cannot see ---------------
        // Every bone pointing exactly where it was asked to does NOT mean the
        // character is posed right. The hips bone points straight up, so a
        // rotation about it is pure yaw -- which way the body FACES -- and
        // aiming it with a minimal arc leaves that at the rest pose's while
        // the legs follow the motion. //Engine:retarget-probe measured 91.5
        // degrees of mean facing error that way, with a direction test like
        // the one above reading 0.00. So the facing needs its own assertion.
        //
        // NOT the spiral, which is not a skeleton: its hip-to-hip runs nearly
        // along its spine, so there is no sideways direction to read a twist
        // from and the construction correctly declines to invent one. This
        // fixture is the source skeleton's own rest shape, walked out from
        // t2m_raw_offsets, and then turned by a known angle -- so the answer
        // is known rather than merely self-consistent.
        {
            const float yaw = DirectX::XMConvertToRadians(50.0f);
            const DirectX::SimpleMath::Matrix turn =
                DirectX::SimpleMath::Matrix::CreateRotationY(yaw);
            std::vector<DirectX::XMFLOAT3> rest(
                static_cast<std::size_t>(sourceJoints));
            std::vector<DirectX::SimpleMath::Vector3> world(
                static_cast<std::size_t>(sourceJoints),
                DirectX::SimpleMath::Vector3::Zero);
            for (const auto& chain : retargeter.Chains()) {
                for (std::size_t k = 1; k < chain.size(); ++k) {
                    const int joint = chain[k];
                    const int parent = chain[k - 1];
                    if (joint >= sourceJoints || parent >= sourceJoints) continue;
                    world[static_cast<std::size_t>(joint)] =
                        world[static_cast<std::size_t>(parent)] +
                        kRawOffsets[joint] * 0.4f;
                }
            }
            for (int j = 0; j < sourceJoints; ++j) {
                const DirectX::SimpleMath::Vector3 turned =
                    DirectX::SimpleMath::Vector3::Transform(
                        world[static_cast<std::size_t>(j)], turn);
                rest[static_cast<std::size_t>(j)] = {turned.x, turned.y, turned.z};
            }

            Expect("the rest-shaped frame applies",
                   retargeter.ApplyFrame(graph, nullptr, 0, rest.data(), 0.0f,
                                         DirectX::SimpleMath::Vector3(0.0f, 0.9f,
                                                                      0.0f),
                                         1.0f,
                                         DirectX::SimpleMath::Matrix::Identity,
                                         &error));
            graph.UpdateTransforms(DirectX::SimpleMath::Matrix::Identity);

            std::optional<std::size_t> leftLeg;
            std::optional<std::size_t> rightLeg;
            for (const NeuralModelIntegrateTestbed::RetargetBone& bone :
                 retargeter.Bones()) {
                if (bone.destinationName == "mixamorig:LeftUpLeg") {
                    leftLeg = bone.destinationNode;
                }
                if (bone.destinationName == "mixamorig:RightUpLeg") {
                    rightLeg = bone.destinationNode;
                }
            }
            Expect("the legs resolved", leftLeg.has_value() && rightLeg.has_value());
            if (leftLeg.has_value() && rightLeg.has_value()) {
                const DirectX::SimpleMath::Matrix& l =
                    graph.Nodes()[*leftLeg].worldTransform;
                const DirectX::SimpleMath::Matrix& r =
                    graph.Nodes()[*rightLeg].worldTransform;
                DirectX::SimpleMath::Vector3 character(
                    r._41 - l._41, r._42 - l._42, r._43 - l._43);
                DirectX::SimpleMath::Vector3 motion(
                    rest[2].x - rest[1].x, rest[2].y - rest[1].y,
                    rest[2].z - rest[1].z);
                character.Normalize();
                motion.Normalize();
                const float facing = DirectX::XMConvertToDegrees(std::acos(
                    std::max(-1.0f, std::min(1.0f, character.Dot(motion)))));
                std::printf("  %-52s %.4f deg\n",
                            "hip-to-hip facing error under a 50 deg turn", facing);
                // The failure this catches is tens of degrees: without the
                // twist reference the character keeps its REST facing, so a 50
                // degree turn of the source reads back as 50 degrees of error.
                Expect("the character faces where the motion faces", facing < 5.0f);
            }
        }

        // Positions are not optional in this mode. Silently falling back to a
        // rotation block that does not describe the same pose is exactly the
        // failure this path exists to avoid, so it has to be an error rather
        // than a fallback.
        Expect("the position path refuses a frame with no positions",
               !retargeter.ApplyFrame(graph, feature.data(),
                                      static_cast<int>(feature.size()), nullptr, 0.0f,
                                      DirectX::SimpleMath::Vector3(0.0f, 0.9f, 0.0f),
                                      1.0f, DirectX::SimpleMath::Matrix::Identity,
                                      &error));
        Expect("  and says why", error.find("positions") != std::string::npos);
    }
}



// The sun's direction as two Euler angles. Worth asserting rather than eyeing
// on screen: every sign slip here still produces a lit scene, just one lit from
// somewhere else, and the yaw zero has to agree with the camera panel's or an
// angle read off one means something different in the other.
void TestSunEuler() {
    using NeuralModelIntegrateTestbed::SunDirectionFromEuler;
    using NeuralModelIntegrateTestbed::SunEulerFromDirection;
    using DirectX::SimpleMath::Vector3;

    std::printf("Sun direction from Euler angles\n");

    // glTF's default: an untransformed punctual light shines down -Z, so that
    // is what (0, 0) has to mean.
    ExpectVec("(0, 0) is glTF's -Z", SunDirectionFromEuler(0.0f, 0.0f), 0.0f, 0.0f,
              -1.0f);
    // Negative pitch is downwards, which is where a sun is.
    ExpectVec("pitch -90 shines straight down", SunDirectionFromEuler(-90.0f, 0.0f),
              0.0f, -1.0f, 0.0f);
    ExpectVec("pitch +90 shines straight up", SunDirectionFromEuler(90.0f, 0.0f),
              0.0f, 1.0f, 0.0f);
    // Yaw turns towards +X first, matching atan2(x, -z).
    ExpectVec("yaw 90 shines towards +X", SunDirectionFromEuler(0.0f, 90.0f), 1.0f,
              0.0f, 0.0f);
    ExpectVec("yaw 180 shines towards +Z", SunDirectionFromEuler(0.0f, 180.0f), 0.0f,
              0.0f, 1.0f);
    ExpectVec("yaw -90 shines towards -X", SunDirectionFromEuler(0.0f, -90.0f), -1.0f,
              0.0f, 0.0f);

    // Unit by construction, so nothing downstream has to normalise. Checked at
    // an angle where neither component is 0 or 1.
    const Vector3 oblique = SunDirectionFromEuler(-33.0f, 57.0f);
    ExpectFloat("an oblique direction is unit", oblique.Length(), 1.0f);

    // The round trip is what the UI depends on: enabling the override reads the
    // angles back out of whatever direction the asset authored, and a sun that
    // jumps on a checkbox is the failure this prevents.
    float worstPitch = 0.0f;
    float worstYaw = 0.0f;
    for (int pitchStep = -8; pitchStep <= 8; ++pitchStep) {
        for (int yawStep = -12; yawStep < 12; ++yawStep) {
            // Deliberately off the poles: at pitch +/-90 the direction is
            // vertical and its yaw is genuinely undefined, so a round trip
            // cannot recover it and nothing should claim otherwise.
            const float pitch = pitchStep * 10.0f;
            const float yaw = yawStep * 15.0f;
            float gotPitch = 0.0f;
            float gotYaw = 0.0f;
            SunEulerFromDirection(SunDirectionFromEuler(pitch, yaw), gotPitch, gotYaw);
            worstPitch = std::max(worstPitch, std::fabs(gotPitch - pitch));
            // 180 and -180 are the same yaw, so the error is the circular one.
            float yawError = std::fabs(gotYaw - yaw);
            if (yawError > 180.0f) yawError = 360.0f - yawError;
            worstYaw = std::max(worstYaw, yawError);
        }
    }
    std::printf("  round trip over 17 x 24 angles: worst pitch %.4f deg, "
                "worst yaw %.4f deg\n", worstPitch, worstYaw);
    ExpectFloat("round trip recovers the pitch", worstPitch, 0.0f);
    ExpectFloat("round trip recovers the yaw", worstYaw, 0.0f);

    // A vertical direction has no yaw. The pitch still has to come back, and
    // nothing may come back NaN.
    float polePitch = 0.0f;
    float poleYaw = 0.0f;
    SunEulerFromDirection(Vector3(0.0f, -1.0f, 0.0f), polePitch, poleYaw);
    ExpectFloat("straight down reads as pitch -90", polePitch, -90.0f);
    ExpectFloat("straight down has a finite yaw", poleYaw - poleYaw, 0.0f);

    // A degenerate direction is what an absent sun leaves behind; it must not
    // produce NaN angles that then poison a cascade matrix.
    float zeroPitch = 0.0f;
    float zeroYaw = 0.0f;
    SunEulerFromDirection(Vector3::Zero, zeroPitch, zeroYaw);
    ExpectFloat("a zero direction falls back to straight down", zeroPitch, -90.0f);

    // Not normalised on the way in. The UI reads back from ShaderLight, which
    // is whatever the asset's node produced, and nothing promises unit length.
    float longPitch = 0.0f;
    float longYaw = 0.0f;
    SunEulerFromDirection(Vector3(0.0f, -7.0f, 0.0f), longPitch, longYaw);
    ExpectFloat("a long direction is normalised first", longPitch, -90.0f);

    // And the cascades have to accept what this produces: a light direction is
    // the one input ComputeShadowCascades cannot work without.
    const NeuralModelIntegrateTestbed::ShadowCascadeSetup setup;
    const DirectX::SimpleMath::Matrix view = DirectX::SimpleMath::Matrix::CreateLookAt(
        Vector3(0.0f, 2.0f, 10.0f), Vector3::Zero, Vector3::UnitY);
    const DirectX::SimpleMath::Matrix projection =
        DirectX::SimpleMath::Matrix::CreatePerspectiveFieldOfView(1.2f, 1.777f, 0.1f,
                                                                 100.0f);
    const NeuralModelIntegrateTestbed::ShadowCascades fromEuler =
        NeuralModelIntegrateTestbed::ComputeShadowCascades(
            view, projection, SunDirectionFromEuler(-45.0f, 30.0f),
            Vector3(-10.0f, -1.0f, -10.0f), Vector3(10.0f, 5.0f, 10.0f), setup);
    ExpectFloat("a Euler sun still fits cascades",
                static_cast<float>(fromEuler.count),
                static_cast<float>(setup.cascadeCount));
    // Two different suns must give two different fits, or the angles are not
    // reaching the matrix at all.
    const NeuralModelIntegrateTestbed::ShadowCascades other =
        NeuralModelIntegrateTestbed::ComputeShadowCascades(
            view, projection, SunDirectionFromEuler(-45.0f, 120.0f),
            Vector3(-10.0f, -1.0f, -10.0f), Vector3(10.0f, 5.0f, 10.0f), setup);
    const float matrixDelta =
        (fromEuler.cascades[0].viewProjection.Translation() -
         other.cascades[0].viewProjection.Translation())
            .Length();
    std::printf("  a 90 deg yaw change moves cascade 0 by %.4f\n", matrixDelta);
    Expect("turning the sun moves the cascade matrix", matrixDelta > 1e-3f);
}

void TestSceneSunView() {
    using DirectX::SimpleMath::Matrix;
    using DirectX::SimpleMath::Vector3;
    using DirectX::SimpleMath::Vector4;
    using NeuralModelIntegrateTestbed::SceneSunViewProjection;

    std::cout << "whole-scene sun view (for baking)" << std::endl;

    const Vector3 sceneMin(-52.4f, -4.7f, -46.9f);
    const Vector3 sceneMax(58.7f, 27.2f, 72.4f);
    Matrix viewProjection;
    float texel = 0.0f;
    Expect("a scene-wide sun view is produced",
           SceneSunViewProjection(Vector3(0.3f, -0.9f, 0.3f), sceneMin, sceneMax,
                                  4096, viewProjection, texel));

    // The Bistro is 166 units across, so a 4096-texel map over its bounding
    // sphere resolves about 4 cm -- finer than the lightmap's own 12.5 cm
    // texels at 8 per metre, which is what keeps the bake from being limited
    // by the depth map rather than by the atlas.
    std::printf("  %-52s %.4f m\n", "sun depth texel at 4096", texel);
    Expect("the sun depth map outresolves an 8-per-metre lightmap",
           texel < 1.0f / 8.0f);

    // EVERY corner of the scene must be inside the fit. A cascade may
    // legitimately exclude geometry outside its slice; this one may not
    // exclude anything, because a lightmap texel anywhere can need it.
    bool allInside = true;
    for (int corner = 0; corner < 8; ++corner) {
        const Vector3 point((corner & 1) ? sceneMax.x : sceneMin.x,
                            (corner & 2) ? sceneMax.y : sceneMin.y,
                            (corner & 4) ? sceneMax.z : sceneMin.z);
        const Vector4 clip =
            Vector4::Transform(Vector4(point.x, point.y, point.z, 1.0f), viewProjection);
        const Vector3 ndc(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        allInside = allInside && ndc.x >= -1.0f && ndc.x <= 1.0f && ndc.y >= -1.0f &&
                    ndc.y <= 1.0f && ndc.z >= 0.0f && ndc.z <= 1.0f;
    }
    Expect("every corner of the scene is inside the fit", allInside);

    // Rotating the light must not change the resolution: the fit is a sphere.
    Matrix other;
    float otherTexel = 0.0f;
    SceneSunViewProjection(Vector3(-0.7f, -0.2f, 0.6f), sceneMin, sceneMax, 4096,
                           other, otherTexel);
    ExpectFloat("the texel size does not depend on the light direction",
                otherTexel, texel);

    Expect("a degenerate light is refused",
           !SceneSunViewProjection(Vector3::Zero, sceneMin, sceneMax, 4096,
                                   viewProjection, texel));
    Expect("empty bounds are refused",
           !SceneSunViewProjection(Vector3::UnitY, sceneMin, sceneMin, 4096,
                                   viewProjection, texel));
    Expect("a zero resolution is refused",
           !SceneSunViewProjection(Vector3::UnitY, sceneMin, sceneMax, 0,
                                   viewProjection, texel));
}

void TestLightmapPacking() {
    using NeuralModelIntegrateTestbed::LightmapAllocation;
    using NeuralModelIntegrateTestbed::LightmapChartEdge;
    using NeuralModelIntegrateTestbed::LightmapSetup;
    using NeuralModelIntegrateTestbed::PackLightmap;

    std::cout << "lightmap packing" << std::endl;

    LightmapSetup setup;
    setup.texelsPerMetre = 8.0f;
    setup.atlasResolution = 1024;
    setup.minChartEdge = 4;
    setup.maxChartEdge = 256;
    setup.padding = 2;

    // --- sizing ---------------------------------------------------------
    // A 1 m^2 surface earns 8 texels of content, which with a 2-texel border
    // on each side is 12, rounded up to 16.
    ExpectEqualU("one square metre earns a 16-texel chart",
                 LightmapChartEdge(1.0f, setup), 16);
    // 16 m^2 is a 4 m edge, so 32 texels of content, 36 with the border -> 64.
    ExpectEqualU("sixteen square metres earn 64", LightmapChartEdge(16.0f, setup), 64);
    ExpectEqualU("no area earns no chart", LightmapChartEdge(0.0f, setup), 0);
    ExpectEqualU("a negative area earns no chart",
                 LightmapChartEdge(-5.0f, setup), 0);
    // Bigger never means smaller, which is the property a resolution dial has
    // to have for the result to be predictable.
    bool monotonic = true;
    std::uint32_t previous = 0;
    for (int i = 1; i <= 400; ++i) {
        const std::uint32_t edge =
            LightmapChartEdge(static_cast<float>(i) * 0.5f, setup);
        monotonic = monotonic && edge >= previous;
        previous = edge;
    }
    Expect("chart size never shrinks as area grows", monotonic);
    // And it is capped, or one enormous floor would eat the atlas.
    ExpectEqualU("a huge surface is capped", LightmapChartEdge(1.0e6f, setup),
                 setup.maxChartEdge);

    // --- packing --------------------------------------------------------
    // A spread of sizes, deliberately not sorted, so the packer's own
    // ordering is what decides the layout.
    std::vector<float> areas;
    for (int i = 0; i < 300; ++i) {
        areas.push_back(0.05f + static_cast<float>((i * 37) % 97) * 0.9f);
    }
    // 1024 is genuinely too small for this mix -- a third of these areas earn a
    // 128-texel chart, which is 1.6 M texels on its own -- so the atlas that
    // has to hold them is bigger. That the 1024 case fills to exactly 100%
    // before overflowing is the packer being tight, not the test being lucky.
    LightmapSetup roomy = setup;
    roomy.atlasResolution = 2048;
    const LightmapAllocation packed = PackLightmap(areas, roomy);
    Expect("every chart was placed", packed.complete);
    ExpectEqualU("charts match instances",
                 static_cast<std::uint32_t>(packed.charts.size()),
                 static_cast<std::uint32_t>(areas.size()));

    // Nothing may leave the atlas.
    bool inside = true;
    for (const auto& chart : packed.charts) {
        if (!chart.placed) continue;
        inside = inside && chart.x + chart.size <= roomy.atlasResolution &&
                 chart.y + chart.size <= roomy.atlasResolution;
    }
    Expect("every chart is inside the atlas", inside);

    // And no two may share a texel. This is the assertion the whole file
    // exists for: a one-texel overlap makes one object wear another object's
    // shadow, which reads as a baking bug and is an allocation bug.
    bool disjoint = true;
    std::size_t firstA = 0, firstB = 0;
    for (std::size_t a = 0; a < packed.charts.size() && disjoint; ++a) {
        if (!packed.charts[a].placed) continue;
        for (std::size_t b = a + 1; b < packed.charts.size(); ++b) {
            if (!packed.charts[b].placed) continue;
            const auto& p = packed.charts[a];
            const auto& q = packed.charts[b];
            const bool apart = p.x + p.size <= q.x || q.x + q.size <= p.x ||
                               p.y + p.size <= q.y || q.y + q.size <= p.y;
            if (!apart) {
                disjoint = false;
                firstA = a;
                firstB = b;
                break;
            }
        }
    }
    if (!disjoint) {
        std::printf("    charts %zu and %zu overlap\n", firstA, firstB);
    }
    Expect("no two charts overlap", disjoint);

    // The UV transform must land inside the chart's own content, border
    // excluded -- that border is what stops a bilinear tap at the edge of one
    // chart from reading the next one.
    bool uvInside = true;
    for (const auto& chart : packed.charts) {
        if (!chart.placed) continue;
        const float lowU = chart.scaleOffset.z;
        const float lowV = chart.scaleOffset.w;
        const float highU = chart.scaleOffset.x + chart.scaleOffset.z;
        const float highV = chart.scaleOffset.y + chart.scaleOffset.w;
        const float atlas = static_cast<float>(roomy.atlasResolution);
        uvInside = uvInside &&
                   lowU * atlas >= static_cast<float>(chart.x + setup.padding) - 1e-3f &&
                   lowV * atlas >= static_cast<float>(chart.y + setup.padding) - 1e-3f &&
                   highU * atlas <= static_cast<float>(chart.x + chart.size -
                                                       setup.padding) + 1e-3f &&
                   highV * atlas <= static_cast<float>(chart.y + chart.size -
                                                       setup.padding) + 1e-3f;
    }
    Expect("each UV transform stays inside its chart's content", uvInside);

    // How much of the atlas the pack actually used. Printed rather than
    // asserted on: it is a property of the size mix, not a correctness claim.
    const double occupancy = 100.0 * static_cast<double>(packed.texelsPlaced) /
                             (static_cast<double>(roomy.atlasResolution) *
                              roomy.atlasResolution);
    std::printf("  %-52s %.1f%% of %ux%u\n", "atlas occupancy", occupancy,
                roomy.atlasResolution, roomy.atlasResolution);
    // Not a tight bound, just a guard: a packer that started leaving most of
    // the atlas empty would still pass every correctness check above.
    Expect("the pack uses a reasonable share of the atlas", occupancy > 40.0);

    // --- overflow is reported, not papered over -------------------------
    LightmapSetup tiny = setup;
    tiny.atlasResolution = 64;
    const LightmapAllocation overflowed = PackLightmap(areas, tiny);
    Expect("an atlas that is too small reports it", !overflowed.complete);
    bool stillDisjoint = true;
    for (std::size_t a = 0; a < overflowed.charts.size() && stillDisjoint; ++a) {
        if (!overflowed.charts[a].placed) continue;
        for (std::size_t b = a + 1; b < overflowed.charts.size(); ++b) {
            if (!overflowed.charts[b].placed) continue;
            const auto& p = overflowed.charts[a];
            const auto& q = overflowed.charts[b];
            stillDisjoint = stillDisjoint &&
                            (p.x + p.size <= q.x || q.x + q.size <= p.x ||
                             p.y + p.size <= q.y || q.y + q.size <= p.y);
        }
    }
    // The important half of an overflow: what DID fit is still valid, so the
    // caller can fall back per instance rather than losing the whole bake.
    Expect("an overflowed pack is still internally consistent", stillDisjoint);

    // --- reproducibility ------------------------------------------------
    const LightmapAllocation again = PackLightmap(areas, roomy);
    bool same = again.charts.size() == packed.charts.size();
    for (std::size_t i = 0; same && i < again.charts.size(); ++i) {
        same = again.charts[i].x == packed.charts[i].x &&
               again.charts[i].y == packed.charts[i].y &&
               again.charts[i].size == packed.charts[i].size;
    }
    Expect("packing the same scene twice gives the same atlas", same);

    // A degenerate primitive asks for nothing and must not break the pack.
    std::vector<float> withZeroes = {0.0f, 4.0f, 0.0f, 9.0f};
    const LightmapAllocation mixed = PackLightmap(withZeroes, setup);
    Expect("zero-area instances are skipped, not failed", mixed.complete);
    Expect("...and get no chart", !mixed.charts[0].placed && !mixed.charts[2].placed);
    Expect("...while the others do", mixed.charts[1].placed && mixed.charts[3].placed);
}

// Several glTF files merged into one graph, which is how a skinned character in
// its own file ends up standing in a set authored in another.
//
// Everything here is about the index offsets. An asset's node, skin, animation
// and image indices are all asset-local, and a missed offset does not fail --
// it silently names something in the OTHER file, which renders as the wrong
// mesh in the wrong place with the wrong texture. So each one is asserted
// numerically.
void TestMultipleAssets() {
    using DirectX::SimpleMath::Matrix;
    std::cout << "multiple assets merged into one graph" << std::endl;

    auto skinAsset = Parse(kSkinGltf);
    auto animAsset = Parse(kAnimGltf);
    if (skinAsset.error() != fastgltf::Error::None ||
        animAsset.error() != fastgltf::Error::None) {
        std::cerr << "merge assets failed to parse" << std::endl;
        ++g_failures;
        return;
    }

    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> noBuffers;
    SceneGraph graph;
    graph.Clear();

    // Asset 0: four nodes, one skin whose joints are its nodes 1 and 2.
    const std::size_t firstBase =
        graph.Append(skinAsset.get(), 0, noBuffers, 0, 0, 0);
    Expect("the first asset starts at node 0", firstBase == 0);
    Expect("the first asset's nodes are all there", graph.NodeCount() == 4);

    // Asset 1: four more nodes and two animations.
    const std::size_t secondBase =
        graph.Append(animAsset.get(), animAsset->defaultScene.value_or(0),
                     noBuffers, 0, 0, 1);
    ExpectEqualU("the second asset starts after the first",
                 static_cast<std::uint32_t>(secondBase), 4);
    ExpectEqualU("both assets' nodes are in the graph",
                 static_cast<std::uint32_t>(graph.NodeCount()), 8);

    // Asset 2: the first asset again, which is the case most likely to expose a
    // missed offset -- every index it carries is one the graph already has.
    const std::size_t thirdBase =
        graph.Append(skinAsset.get(), 0, noBuffers, 0, 0, 2);
    ExpectEqualU("a repeated asset starts after the second",
                 static_cast<std::uint32_t>(thirdBase), 8);
    ExpectEqualU("every node is in the graph",
                 static_cast<std::uint32_t>(graph.NodeCount()), 12);
    ExpectEqualU("three assets are registered",
                 static_cast<std::uint32_t>(graph.AssetCount()), 3);

    const std::vector<SceneNode>& nodes = graph.Nodes();

    std::cout << "  node identity" << std::endl;
    Expect("each node knows which asset it came from",
           nodes[0].assetIndex == 0 && nodes[4].assetIndex == 1 &&
               nodes[8].assetIndex == 2);
    Expect("each node keeps its asset-local index",
           nodes[4].assetNodeIndex == 0 && nodes[9].assetNodeIndex == 1);
    Expect("gltfNodeIndex is the merged index", nodes[9].gltfNodeIndex == 9);
    Expect("the repeated asset's nodes are named the same",
           nodes[8].name == nodes[0].name);

    std::cout << "  hierarchy" << std::endl;
    // Three roots, one per asset, in append order.
    Expect("one root per asset", graph.RootNodes().size() == 3);
    Expect("the roots are each asset's own",
           graph.RootNodes()[0] == 0 && graph.RootNodes()[1] == 4 &&
               graph.RootNodes()[2] == 8);
    // The repeated asset's "Root" has children 1,2,3 in its own numbering,
    // which must come out as 9,10,11 and not as 1,2,3.
    Expect("children are offset into the appended asset",
           nodes[8].children.size() == 3 && nodes[8].children[0] == 9 &&
               nodes[8].children[2] == 11);
    Expect("parents are offset too", nodes[9].parent.has_value() &&
                                         nodes[9].parent.value() == 8);
    Expect("the first asset's links are untouched",
           nodes[1].parent.has_value() && nodes[1].parent.value() == 0);

    std::cout << "  skins" << std::endl;
    Expect("each asset contributed its skin", graph.Skins().size() == 2);
    Expect("the first skin keeps its own joints",
           graph.Skins()[0].joints.size() == 2 && graph.Skins()[0].joints[0] == 1 &&
               graph.Skins()[0].joints[1] == 2);
    Expect("the repeated skin's joints are offset",
           graph.Skins()[1].joints.size() == 2 && graph.Skins()[1].joints[0] == 9 &&
               graph.Skins()[1].joints[1] == 10);
    Expect("the repeated skin's skeleton root is offset",
           graph.Skins()[1].skeletonRoot.has_value() &&
               graph.Skins()[1].skeletonRoot.value() == 9);
    // Node 3 of the repeated asset is its skinned mesh, so node 11 must point
    // at skin 1 -- pointing at skin 0 would deform it with the other copy's
    // joints, which sit in the same place here and so would look right.
    Expect("the repeated mesh uses the repeated skin",
           nodes[11].skinIndex.has_value() && nodes[11].skinIndex.value() == 1);
    Expect("the first mesh still uses the first skin",
           nodes[3].skinIndex.has_value() && nodes[3].skinIndex.value() == 0);
    Expect("joints are flagged in both copies", nodes[1].isJoint && nodes[9].isJoint);

    std::cout << "  animations" << std::endl;
    // Only the middle asset has any, and its channels target its own nodes.
    Expect("the animated asset contributed its clips", graph.Animations().size() == 2);
    bool channelsOffset = !graph.Animations()[0].channels.empty();
    for (const auto& channel : graph.Animations()[0].channels) {
        if (channel.nodeIndex < 4 || channel.nodeIndex > 7) {
            channelsOffset = false;
        }
    }
    Expect("every channel targets the asset it came from", channelsOffset);
    Expect("an out-of-range clip is rejected",
           !graph.ApplyAnimations({SceneGraph::ActiveAnimation{99, 0.0f}}));

    std::cout << "  per-asset placement" << std::endl;
    // 100 units along +X, so the two copies of the same asset are
    // distinguishable: without the placement they would sit on top of each
    // other, which is exactly what makes a missed offset invisible.
    graph.SetAssetPlacement(2, Matrix::CreateTranslation(100.0f, 0.0f, 0.0f));
    graph.UpdateTransforms(Matrix::Identity);
    // Joint0 is authored at x=5 in both copies.
    ExpectVec("the first copy stays where it was authored",
              nodes[1].worldTransform.Translation(), 5.0f, 0.0f, 0.0f);
    ExpectVec("the placed copy moves with its asset",
              nodes[9].worldTransform.Translation(), 105.0f, 0.0f, 0.0f);
    ExpectVec("an unplaced asset is unaffected",
              nodes[4].worldTransform.Translation(), 0.0f, 0.0f, 0.0f);

    // The scene placement still applies outside every asset's own.
    graph.UpdateTransforms(Matrix::CreateTranslation(0.0f, 1.0f, 0.0f));
    ExpectVec("the scene placement composes outside the asset's",
              nodes[9].worldTransform.Translation(), 105.0f, 1.0f, 0.0f);

    // Scale inside the translation, which is the order the UI's two controls
    // mean: the position is in world units and must not be multiplied by the
    // scale. Joint0 is authored at x=5, so x2 puts it at 10 and the move then
    // adds 100 -- the other order would give (5+100)*2.
    graph.SetAssetPlacement(2, Matrix::CreateScale(2.0f) *
                                   Matrix::CreateTranslation(100.0f, 0.0f, 0.0f));
    graph.UpdateTransforms(Matrix::Identity);
    ExpectVec("a scaled asset scales about its own origin",
              nodes[9].worldTransform.Translation(), 110.0f, 0.0f, 0.0f);
    graph.SetAssetPlacement(2, Matrix::CreateTranslation(100.0f, 0.0f, 0.0f));
    graph.UpdateTransforms(Matrix::Identity);

    std::cout << "  one clip per asset" << std::endl;
    // The animated asset's node 1 goes to (10,20,0) at t=2 by its own
    // keyframes; the two static assets must not budge.
    Expect("a clip from one asset applies",
           graph.ApplyAnimations({SceneGraph::ActiveAnimation{0, 2.0f}}));
    graph.UpdateTransforms(Matrix::Identity);
    ExpectVec("the animated asset's node is posed",
              nodes[5].worldTransform.Translation(), 10.0f, 20.0f, 0.0f);
    ExpectVec("the other assets are untouched by it",
              nodes[1].worldTransform.Translation(), 5.0f, 0.0f, 0.0f);
    Expect("no clips at all is not an error", graph.ApplyAnimations({}));

    std::cout << "  material image indices" << std::endl;
    auto meshAsset = Parse(kMeshGltf);
    if (meshAsset.error() != fastgltf::Error::None) {
        std::cerr << "mesh asset failed to parse" << std::endl;
        ++g_failures;
        return;
    }
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&device)))) {
        std::cout << "  material image indices: skipped, no D3D12 device"
                  << std::endl;
        return;
    }
    Microsoft::WRL::ComPtr<ID3D12Resource> gltfBuffer;
    const CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
    const auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(256);
    if (FAILED(device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&gltfBuffer)))) {
        std::cerr << "CreateCommittedResource failed" << std::endl;
        ++g_failures;
        return;
    }
    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> buffers{gltfBuffer};

    SceneGraph textured;
    textured.Clear();
    // The same five-texture asset twice. The second copy's images live at 5..9
    // in the renderer's descriptor array, so its material must resolve there --
    // resolving to 0..4 would sample the first copy's textures, which are the
    // same file and so would look correct.
    textured.Append(meshAsset.get(), 0, buffers, 0, 5, 0);
    textured.Append(meshAsset.get(), 0, buffers, 5, 5, 1);
    textured.UpdateTransforms(Matrix::Identity);
    ExpectEqualU("both copies are drawn",
                 static_cast<std::uint32_t>(textured.DrawOrder().size()), 2);
    if (textured.DrawOrder().size() == 2) {
        const auto& first =
            textured.Nodes()[textured.DrawOrder()[0]].primitives.at(0);
        const auto& second =
            textured.Nodes()[textured.DrawOrder()[1]].primitives.at(0);
        using NeuralModelIntegrateTestbed::MaterialTextureSlot;
        Expect("the first copy's base colour is image 0",
               first.ImageIndex(MaterialTextureSlot::BaseColor).value_or(999) == 0);
        Expect("the second copy's base colour is offset to image 5",
               second.ImageIndex(MaterialTextureSlot::BaseColor).value_or(999) == 5);
        Expect("the first copy's emissive is image 4",
               first.ImageIndex(MaterialTextureSlot::Emissive).value_or(999) == 4);
        Expect("the second copy's emissive is offset to image 9",
               second.ImageIndex(MaterialTextureSlot::Emissive).value_or(999) == 9);
    }

    std::cout << "  per-asset visibility" << std::endl;
    // Switching one file off has to take it out of the draw order itself, not
    // leave it to be drawn somewhere it cannot be seen -- the shadow passes and
    // the light upload read the same walk.
    textured.SetAssetVisible(1, false);
    textured.UpdateTransforms(Matrix::Identity);
    ExpectEqualU("hiding one file leaves the other drawn",
                 static_cast<std::uint32_t>(textured.DrawOrder().size()), 1);
    Expect("the file still showing is the one left",
           textured.DrawOrder().size() == 1 &&
               textured.Nodes()[textured.DrawOrder()[0]].assetIndex == 0);
    textured.SetAssetVisible(0, false);
    textured.UpdateTransforms(Matrix::Identity);
    Expect("hiding every file draws nothing", textured.DrawOrder().empty());
    textured.SetAssetVisible(0, true);
    textured.SetAssetVisible(1, true);
    textured.UpdateTransforms(Matrix::Identity);
    ExpectEqualU("switching them back on restores both",
                 static_cast<std::uint32_t>(textured.DrawOrder().size()), 2);
    Expect("an asset nothing has spoken about is shown", textured.AssetVisible(7));

    // An asset whose images were not loaded resolves nothing, and must not
    // reach the base offset either: a descriptor index it never earned would
    // point into another asset's textures.
    SceneGraph noImages;
    noImages.Clear();
    noImages.Append(meshAsset.get(), 0, buffers, 0, 5, 0);
    noImages.Append(meshAsset.get(), 0, buffers, 5, 0, 1);
    noImages.UpdateTransforms(Matrix::Identity);
    if (noImages.DrawOrder().size() == 2) {
        const auto& second =
            noImages.Nodes()[noImages.DrawOrder()[1]].primitives.at(0);
        using NeuralModelIntegrateTestbed::MaterialTextureSlot;
        Expect("an asset with no images resolves no texture",
               !second.ImageIndex(MaterialTextureSlot::BaseColor).has_value());
    }
}

// The placement a scene manifest describes, composed into a matrix. Every way
// of getting this wrong still renders something: the wrong compose order puts
// a scaled file at a scaled-up position, and a degrees/radians slip turns it by
// a plausible-looking amount. So the three are asserted against hand-computed
// results.
void TestPlacementMatrix() {
    using DirectX::SimpleMath::Matrix;
    using DirectX::SimpleMath::Vector3;
    using NeuralModelIntegrateTestbed::PlacementMatrix;
    using NeuralModelIntegrateTestbed::SceneTransform;
    std::cout << "manifest placement -> matrix" << std::endl;

    {
        SceneTransform identity;
        const Matrix m = PlacementMatrix(identity);
        ExpectVec("the default placement moves nothing",
                  Vector3::Transform(Vector3(1.0f, 2.0f, 3.0f), m), 1.0f, 2.0f,
                  3.0f);
    }
    {
        // Scale INSIDE the translation: the point at x=1 scales to 2 and the
        // move then adds 10. The other order would give (1+10)*2 = 22, which is
        // the mistake that makes a position slider behave differently depending
        // on the scale.
        SceneTransform t;
        t.scale = 2.0f;
        t.translation[0] = 10.0f;
        ExpectVec("scale applies before the translation",
                  Vector3::Transform(Vector3(1.0f, 0.0f, 0.0f),
                                     PlacementMatrix(t)),
                  12.0f, 0.0f, 0.0f);
    }
    {
        // 90 degrees about Y takes +X onto -Z in this project's left-handed
        // convention. Degrees, not radians: a transform given 90 and read as
        // radians would turn by about 5157 degrees and land somewhere arbitrary.
        SceneTransform t;
        t.rotation[1] = 90.0f;
        ExpectVec("a yaw of 90 degrees turns +X onto -Z",
                  Vector3::Transform(Vector3(1.0f, 0.0f, 0.0f),
                                     PlacementMatrix(t)),
                  0.0f, 0.0f, -1.0f);
    }
    {
        // 180 degrees is what a character facing away from the camera needs,
        // which is the case the example manifest covers.
        SceneTransform t;
        t.rotation[1] = 180.0f;
        ExpectVec("a yaw of 180 degrees reverses +Z",
                  Vector3::Transform(Vector3(0.0f, 0.0f, 1.0f),
                                     PlacementMatrix(t)),
                  0.0f, 0.0f, -1.0f);
    }
    {
        // +Y onto +Z, not -Z: XMMatrixRotationX turns the second axis towards
        // the third for a positive angle, and a yaw does the opposite (+X onto
        // -Z, above) because the axis it turns about is the second one. The two
        // signs looking inconsistent is the convention, not a slip.
        SceneTransform t;
        t.rotation[0] = 90.0f;
        ExpectVec("a pitch of 90 degrees turns +Y onto +Z",
                  Vector3::Transform(Vector3(0.0f, 1.0f, 0.0f),
                                     PlacementMatrix(t)),
                  0.0f, 0.0f, 1.0f);
    }
    {
        // All three at once, in the order the comment claims: the point is
        // scaled, then turned, then moved. x=1 -> x=2 -> z=-2 -> (5, 0, -2).
        SceneTransform t;
        t.scale = 2.0f;
        t.rotation[1] = 90.0f;
        t.translation[0] = 5.0f;
        ExpectVec("scale, then rotation, then translation",
                  Vector3::Transform(Vector3(1.0f, 0.0f, 0.0f),
                                     PlacementMatrix(t)),
                  5.0f, 0.0f, -2.0f);
    }
    {
        // What the renderer actually does with it: hand the matrix to
        // SetAssetPlacement and the file's nodes move by it.
        auto asset = Parse(kSkinGltf);
        if (asset.error() != fastgltf::Error::None) {
            std::cerr << "placement asset failed to parse" << std::endl;
            ++g_failures;
            return;
        }
        const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> noBuffers;
        SceneGraph graph;
        graph.Build(asset.get(), 0, noBuffers, 0);
        SceneTransform t;
        t.rotation[1] = 90.0f;
        t.translation[1] = 1.0f;
        graph.SetAssetPlacement(0, PlacementMatrix(t));
        graph.UpdateTransforms(Matrix::Identity);
        // Joint0 is authored at x=5, so a yaw of 90 puts it at z=-5 and the
        // move lifts it to y=1.
        ExpectVec("a manifest placement reaches the nodes",
                  graph.Nodes()[1].worldTransform.Translation(), 0.0f, 1.0f,
                  -5.0f);
    }
}

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

    TestViewFrustum();
    TestShadowCascades();
    TestConcatenateOrder();
    TestRetargetRotations();
    TestRetargetBinding();

    TestSunEuler();
    TestSceneSunView();
    TestLightmapPacking();
    TestMultipleAssets();
    TestPlacementMatrix();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all checks passed" << std::endl;
    return 0;
}
