#pragma once

// A JSON manifest naming the glTF files that make up one scene, and where each
// one goes.
//
// The renderer can merge several glTF files into one scene (see
// SceneGraph::Append), and each needs a placement: a character authored at the
// origin in metres has to be moved, turned to face the camera and often scaled
// by a round factor to match the set it stands in. Those four numbers used to
// live nowhere -- the files came from a ';'-separated environment variable and
// the placement was sliders that reset every run, so reproducing a shot meant
// dragging them back by eye.
//
// So the manifest holds both: which files, and the transform for each.
//
//   {
//     "name": "Bistro with the dancer",
//     "skybox": "sky.hdr",
//     "maxTextureDimension": 2048,
//     "camera": {
//       "position": [4.0, 1.6, -6.0],
//       "target": [0.0, 1.0, 0.0],
//       "up": [0.0, 1.0, 0.0],
//       "fovDegrees": 50.0,
//       "near": 0.05,
//       "far": 120.0
//     },
//     "scenes": [
//       "bistro/bistro.gltf",
//       {
//         "path": "../characters/dancer.gltf",
//         "translation": [2.5, 0.0, -1.0],
//         "rotation": [0.0, 180.0, 0.0],
//         "scale": 0.01,
//         "visible": true
//       }
//     ]
//   }
//
// `scenes` is the only required field. An entry may be a bare string, which is
// a path with the default transform, or an object with any of the fields above.
//
// Paths are resolved against the MANIFEST's own directory, not the working
// directory, so a manifest sits beside its assets and the pair can be moved or
// checked out anywhere without being rewritten.
//
// Rotation is XYZ Euler angles in DEGREES, not a quaternion, even though glTF
// stores quaternions. This file is written and read by a person, and it is the
// same number the UI's sliders show -- one representation from the JSON through
// the sliders to the matrix means there is no conversion to be lossy or to
// disagree about a convention.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace NeuralModelIntegrateTestbed {

// Where the application looks for a manifest when nothing names one, and where
// the Scene tab's save button writes one for a scene that was not loaded from a
// manifest. The two are the same string on purpose: that is what makes "drag
// the sliders, save, restart" work with no environment variable at all, which
// is the point of having the format.
inline constexpr const char* kDefaultSceneManifest = "scenes.json";

// Where one glTF file sits within the merged scene. Composed as scale, then
// rotation, then translation, in the row-vector order this project uses
// throughout -- the same order NodeTransform::ToMatrix composes a glTF node's
// own TRS in, so a placement behaves exactly like a parent node would.
//
// Deliberately plain floats rather than SimpleMath types: ImGui's sliders write
// straight into these, and this struct is also what the manifest parser fills,
// which needs no DirectXMath at all.
struct SceneTransform {
    float translation[3] = {0.0f, 0.0f, 0.0f};
    // XYZ Euler angles in degrees; see the header comment.
    float rotation[3] = {0.0f, 0.0f, 0.0f};
    // Uniform. glTF is metres by convention and exporters disagree by round
    // factors, which is one number; per-axis scale would also skew a skinned
    // character, which is the hardest kind of wrong to diagnose.
    float scale = 1.0f;
    bool visible = true;

    // Whether this is the transform a file gets when the manifest says nothing
    // about it. Used when writing a manifest back out, so an untouched entry
    // stays the one-line string form it was read as.
    bool IsIdentity() const {
        return translation[0] == 0.0f && translation[1] == 0.0f &&
               translation[2] == 0.0f && rotation[0] == 0.0f &&
               rotation[1] == 0.0f && rotation[2] == 0.0f && scale == 1.0f &&
               visible;
    }
};

// A camera the manifest pins, instead of the one the application computes by
// framing the scene's bounding sphere.
//
// The computed one is a guess -- a raised three-quarter view at a distance that
// fits the whole scene -- which is the right thing when a scene is opened for
// the first time and the wrong thing as soon as the shot matters. A scene with
// a character in it is looked at from somewhere specific, and that somewhere
// has to survive a restart.
//
// Only `position` and `target` are required. `up` defaults to +Y, and the
// three projection fields are each left at whatever the application already
// uses when absent, so a manifest can pin where the camera stands without
// having an opinion about its lens.
struct SceneCamera {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float target[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};

    // Each is "the manifest said so", because every value these could take is
    // also a legitimate one to leave alone.
    bool hasFov = false;
    float fovDegrees = 70.0f;
    bool hasNear = false;
    float nearPlane = 0.01f;
    bool hasFar = false;
    float farPlane = 100.0f;
};

struct SceneManifestEntry {
    // As written in the manifest, so writing it back out does not turn every
    // relative path into an absolute one.
    std::string path;
    // `path` resolved against the manifest's directory. This is what to load.
    std::string resolvedPath;
    SceneTransform transform;
};

struct SceneManifest {
    // For the title bar and the UI; optional in the file.
    std::string name;
    std::vector<SceneManifestEntry> scenes;
    // Optional. Empty means "look for an .hdr beside the first scene", which is
    // what the application did before manifests existed.
    std::string skybox;
    std::string resolvedSkybox;
    // Optional; 0 is a valid value meaning "no cap", so absence is a separate
    // flag rather than a sentinel.
    bool hasMaxTextureDimension = false;
    std::uint32_t maxTextureDimension = 0;

    // Optional. Absent means the application frames the camera on the scene's
    // bounds, as it did before manifests existed.
    bool hasCamera = false;
    SceneCamera camera;

    // Where it was loaded from, for the UI and for writing it back.
    std::string sourcePath;
};

// Parses `path`. Returns false with `error` filled on anything wrong -- a
// missing file, malformed JSON, a `scenes` array that is absent or empty, or a
// field of the wrong type. Deliberately strict: a manifest with a typo that
// loaded "most of" the scene would be read as the renderer losing the file.
bool LoadSceneManifest(const std::filesystem::path& path, SceneManifest& out,
                       std::string* error);

// The same, from a string already in memory. `directory` is what relative paths
// resolve against. Exists so the smoke test can cover the parser without
// writing files.
bool ParseSceneManifest(std::string_view json,
                        const std::filesystem::path& directory,
                        SceneManifest& out, std::string* error);

// Serialises back to the format above, two-space indented. An entry whose
// transform is the default comes out as a bare string, so a manifest written
// after nothing was touched reads as one a person would have typed.
std::string WriteSceneManifest(const SceneManifest& manifest);

// Writes it to `path`, creating the parent directory if need be. Used by the
// Scene tab's save button, which is what makes the sliders worth touching: the
// placement found by dragging is the placement the next run starts from.
bool SaveSceneManifest(const std::filesystem::path& path,
                       const SceneManifest& manifest, std::string* error);

}  // namespace NeuralModelIntegrateTestbed
