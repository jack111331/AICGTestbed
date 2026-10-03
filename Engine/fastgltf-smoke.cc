// Smoke test for the local fastgltf checkout (@fastgltf, built from source).
//
// Beyond checking that headers resolve and the library links, this parses an
// asset using the project's own MYLAB_generative vendor extension, which exists
// only in the modified checkout -- so a regression back to the stock vcpkg
// fastgltf fails here rather than silently dropping the extension data.
#include <fastgltf/core.hpp>
#include <fastgltf/math.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <cstddef>
#include <iostream>
#include <string_view>

namespace {

// A minimal glTF 2.0 asset: one triangle, positions inlined as a base64 buffer.
constexpr std::string_view kGltf = R"GLTF({
  "asset": {"version": "2.0", "generator": "ptflio bazel smoke test"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0, "name": "Triangle"}],
  "meshes": [{"name": "TriMesh", "primitives": [{"attributes": {"POSITION": 0}}]}],
  "buffers": [{
    "byteLength": 36,
    "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"
  }],
  "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962}],
  "accessors": [{
    "bufferView": 0, "byteOffset": 0, "componentType": 5126,
    "count": 3, "type": "VEC3",
    "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]
  }]
})GLTF";

// Trimmed from tests/gltf/mylab_generative_valid.gltf in the fastgltf checkout.
constexpr std::string_view kGenerativeGltf = R"GLTF({
  "asset": {"version": "2.0"},
  "extensionsUsed": ["MYLAB_generative"],
  "extensions": {
    "MYLAB_generative": {
      "generators": [{
        "name": "cam_gen",
        "task": "text_to_camera_trajectory",
        "model": {
          "uri": "models/camtraj.onnx",
          "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
        "output": {"fps": 30, "cameraConvention": "opencv", "poseType": "cam_to_world"}
      }],
      "timeline": {"duration": 8.0}
    }
  },
  "cameras": [{"type": "perspective", "perspective": {"yfov": 0.7, "znear": 0.01}}],
  "nodes": [
    {"name": "Actor"},
    {
      "name": "MainCamera",
      "camera": 0,
      "extensions": {
        "MYLAB_generative": {
          "generator": 0,
          "prompt": "slow orbit around the actor, ending on a close-up",
          "seed": 7,
          "start": 0.5,
          "duration": 6.0,
          "conditionOn": {"lookAtNode": 0}
        }
      }
    }
  ],
  "scene": 0,
  "scenes": [{"nodes": [0, 1]}]
})GLTF";

fastgltf::Expected<fastgltf::Asset> Parse(std::string_view json,
                                          fastgltf::Extensions extensions) {
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
  auto asset = Parse(kGltf, fastgltf::Extensions::None);
  if (asset.error() != fastgltf::Error::None) {
    std::cerr << "base parse failed: "
              << fastgltf::getErrorMessage(asset.error()) << std::endl;
    return 1;
  }

  std::cout << "fastgltf linked OK" << std::endl
            << "  generator: " << asset->assetInfo->generator << std::endl
            << "  meshes:    " << asset->meshes.size() << std::endl;

  const auto& primitive = asset->meshes[0].primitives[0];
  const auto& accessor =
      asset->accessors[primitive.findAttribute("POSITION")->accessorIndex];
  fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
      asset.get(), accessor, [](fastgltf::math::fvec3 v, std::size_t i) {
        std::cout << "    [" << i << "] (" << v.x() << ", " << v.y() << ", "
                  << v.z() << ")" << std::endl;
      });

  // The local modification: a vendor extension the stock fastgltf has no idea about.
  auto generative =
      Parse(kGenerativeGltf, fastgltf::Extensions::MYLAB_generative);
  if (generative.error() != fastgltf::Error::None) {
    std::cerr << "MYLAB_generative parse failed: "
              << fastgltf::getErrorMessage(generative.error()) << std::endl;
    return 1;
  }
  if (generative->generators.empty()) {
    std::cerr << "MYLAB_generative parsed but the generator registry is empty"
              << std::endl;
    return 1;
  }

  const auto& node = generative->nodes[1];
  if (!node.generative) {
    std::cerr << "node '" << node.name << "' has no generative data"
              << std::endl;
    return 1;
  }

  std::cout << "MYLAB_generative OK" << std::endl
            << "  generators:  " << generative->generators.size() << std::endl
            << "  model uri:   "
            << generative->generators[node.generative->generator].model.uri
            << std::endl
            << "  driven node: " << node.name << std::endl
            << "  prompt:      " << node.generative->prompt << std::endl
            << "  seed:        " << node.generative->seed.value_or(0)
            << std::endl;
  return 0;
}
