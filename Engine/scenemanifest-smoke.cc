// Smoke test for the JSON scene manifest.
//
// The parser is deliberately strict -- an unknown field is an error, a
// two-element translation is an error -- because a manifest is hand-written and
// the alternative is a typo that loads "most of" the scene, which reads as the
// renderer having lost a file. So the rejections are asserted as carefully as
// the successes.
//
// No D3D12 and no glTF here: the manifest is text in, a path and four numbers
// out, which is exactly what makes it testable without a device.

#include "SceneManifest.hpp"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

using NeuralModelIntegrateTestbed::ParseSceneManifest;
using NeuralModelIntegrateTestbed::SceneCamera;
using NeuralModelIntegrateTestbed::SceneManifest;
using NeuralModelIntegrateTestbed::SceneTransform;
using NeuralModelIntegrateTestbed::WriteSceneManifest;

namespace {

int g_failures = 0;

void Expect(const char* what, bool condition) {
    std::printf("  %-56s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

void ExpectEqualF(const char* what, float got, float expected) {
    const bool ok = std::fabs(got - expected) < 1e-4f;
    std::printf("  %-44s %8.3f  expected %8.3f  %s\n", what, got, expected,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

void ExpectEqualS(const char* what, const std::string& got,
                  const std::string& expected) {
    const bool ok = got == expected;
    std::printf("  %-32s %-28s %s\n", what, got.c_str(), ok ? "ok" : "MISMATCH");
    if (!ok) {
        std::printf("      expected %s\n", expected.c_str());
        ++g_failures;
    }
}

// Every one of these must be refused, with a message that names what is wrong.
// A silent acceptance is the failure mode worth testing for.
void ExpectRejected(const char* what, std::string_view json) {
    SceneManifest manifest;
    std::string error;
    const bool parsed = ParseSceneManifest(json, "", manifest, &error);
    const bool ok = !parsed && !error.empty();
    std::printf("  %-56s %s\n", what, ok ? "ok" : "MISMATCH");
    if (!ok) {
        std::printf("      accepted, or refused with no message\n");
        ++g_failures;
    } else {
        std::printf("      %s\n", error.c_str());
    }
}

}  // namespace

int main() {
    std::cout << "manifest: the shorthand form" << std::endl;
    {
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(
            R"({"scenes": ["a.gltf", "sub/b.glb"]})", "", manifest, &error);
        Expect("a list of bare paths parses", parsed);
        if (parsed) {
            Expect("both entries are there", manifest.scenes.size() == 2);
            ExpectEqualS("the first path", manifest.scenes[0].path, "a.gltf");
            ExpectEqualS("the second path", manifest.scenes[1].path, "sub/b.glb");
            Expect("a bare path gets the default transform",
                   manifest.scenes[0].transform.IsIdentity());
            Expect("and is visible", manifest.scenes[0].transform.visible);
            Expect("no skybox is named", manifest.skybox.empty());
            Expect("no texture cap is named", !manifest.hasMaxTextureDimension);
        } else {
            std::printf("      %s\n", error.c_str());
            ++g_failures;
        }
    }

    std::cout << "manifest: the full form" << std::endl;
    {
        constexpr std::string_view kJson = R"({
          "name": "Bistro with the dancer",
          "skybox": "sky.hdr",
          "maxTextureDimension": 1024,
          "scenes": [
            "bistro/bistro.gltf",
            {
              "path": "characters/dancer.gltf",
              "translation": [2.5, 0, -1],
              "rotation": [0, 180, 0],
              "scale": 0.01,
              "visible": false
            }
          ]
        })";
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(kJson, "", manifest, &error);
        Expect("the full form parses", parsed);
        if (!parsed) {
            std::printf("      %s\n", error.c_str());
            ++g_failures;
        } else {
            ExpectEqualS("name", manifest.name, "Bistro with the dancer");
            ExpectEqualS("skybox", manifest.skybox, "sky.hdr");
            Expect("the texture cap is read",
                   manifest.hasMaxTextureDimension &&
                       manifest.maxTextureDimension == 1024);
            Expect("both scenes are there", manifest.scenes.size() == 2);
            if (manifest.scenes.size() == 2) {
                const SceneTransform& t = manifest.scenes[1].transform;
                ExpectEqualF("translation x", t.translation[0], 2.5f);
                // Written as 0 and -1 rather than 0.0 and -1.0: JSON integers
                // must read as numbers, not as a type error.
                ExpectEqualF("translation y (written as an integer)",
                             t.translation[1], 0.0f);
                ExpectEqualF("translation z (written as an integer)",
                             t.translation[2], -1.0f);
                ExpectEqualF("rotation y in degrees", t.rotation[1], 180.0f);
                ExpectEqualF("scale", t.scale, 0.01f);
                Expect("visible false is honoured", !t.visible);
                Expect("the first entry keeps the default transform",
                       manifest.scenes[0].transform.IsIdentity());
            }
        }
    }

    std::cout << "manifest: paths resolve against the manifest, not the cwd"
              << std::endl;
    {
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(
            R"({"skybox": "sky.hdr", "scenes": ["set/room.gltf"]})",
            std::filesystem::path("C:/assets/shot"), manifest, &error);
        Expect("it parses", parsed);
        if (parsed) {
            ExpectEqualS("the resolved scene path",
                         manifest.scenes[0].resolvedPath,
                         "C:/assets/shot/set/room.gltf");
            ExpectEqualS("the resolved skybox", manifest.resolvedSkybox,
                         "C:/assets/shot/sky.hdr");
            ExpectEqualS("the written path is left relative",
                         manifest.scenes[0].path, "set/room.gltf");
        }
    }
    {
        // An absolute path is how a scene staged outside the repository is
        // named, so it must survive untouched.
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(
            R"({"scenes": ["C:/Code/niagara_bistro/bistro.gltf"]})",
            std::filesystem::path("C:/assets/shot"), manifest, &error);
        Expect("an absolute path parses", parsed);
        if (parsed) {
            ExpectEqualS("an absolute path is not rebased",
                         manifest.scenes[0].resolvedPath,
                         "C:/Code/niagara_bistro/bistro.gltf");
        }
    }

    std::cout << "manifest: a pinned camera" << std::endl;
    {
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(R"({
              "camera": {
                "position": [4, 1.6, -6],
                "target": [0, 1, 0],
                "up": [0, 1, 0],
                "fovDegrees": 50,
                "near": 0.05,
                "far": 120
              },
              "scenes": ["a.gltf"]
            })",
                                               "", manifest, &error);
        Expect("a full camera parses", parsed);
        if (!parsed) {
            std::printf("      %s\n", error.c_str());
            ++g_failures;
        } else {
            Expect("the camera is reported as present", manifest.hasCamera);
            ExpectEqualF("eye x", manifest.camera.position[0], 4.0f);
            ExpectEqualF("eye y", manifest.camera.position[1], 1.6f);
            ExpectEqualF("target y", manifest.camera.target[1], 1.0f);
            ExpectEqualF("up y", manifest.camera.up[1], 1.0f);
            Expect("the field of view is flagged as given",
                   manifest.camera.hasFov);
            ExpectEqualF("field of view", manifest.camera.fovDegrees, 50.0f);
            ExpectEqualF("near plane", manifest.camera.nearPlane, 0.05f);
            ExpectEqualF("far plane", manifest.camera.farPlane, 120.0f);
        }
    }
    {
        // Position and target alone: the lens is left to the application, and
        // each of the three has to say it was not given.
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(
            R"({"camera": {"position": [0,0,5], "target": [0,0,0]},
                "scenes": ["a.gltf"]})",
            "", manifest, &error);
        Expect("a camera with no lens parses", parsed);
        if (parsed) {
            Expect("the camera is present", manifest.hasCamera);
            Expect("up defaults to +Y", manifest.camera.up[1] == 1.0f &&
                                            manifest.camera.up[0] == 0.0f);
            Expect("no field of view was given", !manifest.camera.hasFov);
            Expect("no near plane was given", !manifest.camera.hasNear);
            Expect("no far plane was given", !manifest.camera.hasFar);
        }
    }
    {
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(R"({"scenes": ["a.gltf"]})", "",
                                               manifest, &error);
        Expect("a manifest with no camera says so",
               parsed && !manifest.hasCamera);
    }

    std::cout << "manifest: what must be refused" << std::endl;
    ExpectRejected("a camera that is an array",
                   R"({"camera": [0,0,0], "scenes": ["a.gltf"]})");
    ExpectRejected("a camera with no target",
                   R"({"camera": {"position": [0,0,5]}, "scenes": ["a.gltf"]})");
    ExpectRejected("a camera with no position",
                   R"({"camera": {"target": [0,0,0]}, "scenes": ["a.gltf"]})");
    ExpectRejected("a misspelled camera field",
                   R"({"camera": {"position": [0,0,5], "target": [0,0,0],
                       "fov": 50}, "scenes": ["a.gltf"]})");
    // CreateLookAt produces NaNs for these two, and one NaN in the view matrix
    // takes the frame -- and the UI that would fix it -- with it.
    ExpectRejected("a camera looking at its own position",
                   R"({"camera": {"position": [1,2,3], "target": [1,2,3]},
                       "scenes": ["a.gltf"]})");
    ExpectRejected("a camera with no up vector",
                   R"({"camera": {"position": [0,0,5], "target": [0,0,0],
                       "up": [0,0,0]}, "scenes": ["a.gltf"]})");
    ExpectRejected("a far plane behind the near plane",
                   R"({"camera": {"position": [0,0,5], "target": [0,0,0],
                       "near": 10, "far": 1}, "scenes": ["a.gltf"]})");
    ExpectRejected("a field of view of 180 degrees",
                   R"({"camera": {"position": [0,0,5], "target": [0,0,0],
                       "fovDegrees": 180}, "scenes": ["a.gltf"]})");
    ExpectRejected("malformed JSON", R"({"scenes": [)");
    ExpectRejected("a manifest that is an array", R"(["a.gltf"])");
    ExpectRejected("no scenes array", R"({"name": "nothing"})");
    ExpectRejected("an empty scenes array", R"({"scenes": []})");
    ExpectRejected("scenes that is not an array", R"({"scenes": "a.gltf"})");
    ExpectRejected("an entry that is a number", R"({"scenes": [7]})");
    ExpectRejected("an entry with no path", R"({"scenes": [{"scale": 2}]})");
    ExpectRejected("an empty path", R"({"scenes": [""]})");
    // The one that matters most: a plausible misspelling does nothing at all if
    // it is ignored, and the person is left wondering why the slider in the UI
    // disagrees with the file.
    ExpectRejected("a misspelled entry field",
                   R"({"scenes": [{"path": "a.gltf", "translate": [1,0,0]}]})");
    ExpectRejected("a misspelled top-level field",
                   R"({"scene": ["a.gltf"]})");
    ExpectRejected("a two-element translation",
                   R"({"scenes": [{"path": "a.gltf", "translation": [1, 0]}]})");
    ExpectRejected("a translation of strings",
                   R"({"scenes": [{"path": "a.gltf", "translation": ["1","0","0"]}]})");
    ExpectRejected("a quaternion where Euler degrees go",
                   R"({"scenes": [{"path": "a.gltf", "rotation": [0,0,0,1]}]})");
    // Zero scale makes every world matrix singular, and ComputeJointMatrices
    // inverts one of them.
    ExpectRejected("a scale of zero",
                   R"({"scenes": [{"path": "a.gltf", "scale": 0}]})");
    ExpectRejected("a negative scale",
                   R"({"scenes": [{"path": "a.gltf", "scale": -1}]})");
    ExpectRejected("visible as a string",
                   R"({"scenes": [{"path": "a.gltf", "visible": "yes"}]})");

    std::cout << "manifest: written out and read back" << std::endl;
    {
        SceneManifest manifest;
        std::string error;
        const bool parsed = ParseSceneManifest(R"({
              "name": "Shot",
              "skybox": "sky.hdr",
              "maxTextureDimension": 2048,
              "scenes": [
                "set/room.gltf",
                {"path": "char.gltf", "translation": [1, 2, 3],
                 "rotation": [0, 90, 0], "scale": 0.01, "visible": false}
              ]
            })",
                                               "", manifest, &error);
        Expect("the source parses", parsed);
        const std::string written = WriteSceneManifest(manifest);
        std::printf("--- written ---\n%s---------------\n", written.c_str());

        SceneManifest again;
        const bool reparsed = ParseSceneManifest(written, "", again, &error);
        Expect("what was written parses again", reparsed);
        if (!reparsed) {
            std::printf("      %s\n", error.c_str());
            ++g_failures;
        } else {
            // A round trip is the property that matters: the Scene tab's save
            // button writes this file, and the next run has to start from the
            // placement the sliders were left at.
            ExpectEqualS("name survives", again.name, manifest.name);
            Expect("a manifest with no camera writes none", !again.hasCamera);
            ExpectEqualS("skybox survives", again.skybox, manifest.skybox);
            Expect("the texture cap survives",
                   again.hasMaxTextureDimension &&
                       again.maxTextureDimension == manifest.maxTextureDimension);
            Expect("both entries survive", again.scenes.size() == 2);
            if (again.scenes.size() == 2) {
                ExpectEqualS("the untouched path survives", again.scenes[0].path,
                             "set/room.gltf");
                Expect("an untouched entry stays the shorthand form",
                       written.find("\"set/room.gltf\"") != std::string::npos);
                const SceneTransform& t = again.scenes[1].transform;
                ExpectEqualF("translation x survives", t.translation[0], 1.0f);
                ExpectEqualF("translation z survives", t.translation[2], 3.0f);
                ExpectEqualF("rotation survives", t.rotation[1], 90.0f);
                // 0.01 is the value that matters: an exporter in centimetres.
                ExpectEqualF("a fractional scale survives", t.scale, 0.01f);
                Expect("visible false survives", !t.visible);
            }
        }
    }
    {
        // The camera, which is what the Scene tab's save button writes after
        // the view has been flown to by hand.
        SceneManifest manifest;
        NeuralModelIntegrateTestbed::SceneManifestEntry entry;
        entry.path = "a.gltf";
        manifest.scenes.push_back(entry);
        manifest.hasCamera = true;
        SceneCamera& camera = manifest.camera;
        camera.position[0] = 1.5f;
        camera.position[1] = 2.25f;
        camera.position[2] = -3.0f;
        camera.target[1] = 1.0f;
        camera.up[1] = 1.0f;
        camera.hasFov = true;
        camera.fovDegrees = 42.5f;
        camera.hasNear = true;
        camera.nearPlane = 0.025f;
        camera.hasFar = true;
        camera.farPlane = 250.0f;

        const std::string written = WriteSceneManifest(manifest);
        std::printf("--- written (camera) ---\n%s------------------------\n",
                    written.c_str());
        SceneManifest again;
        std::string error;
        const bool reparsed = ParseSceneManifest(written, "", again, &error);
        Expect("a written camera reads back", reparsed);
        if (!reparsed) {
            std::printf("      %s\n", error.c_str());
            ++g_failures;
        } else {
            Expect("it is still a camera", again.hasCamera);
            ExpectEqualF("eye x survives", again.camera.position[0], 1.5f);
            ExpectEqualF("eye y survives", again.camera.position[1], 2.25f);
            ExpectEqualF("eye z survives", again.camera.position[2], -3.0f);
            ExpectEqualF("target survives", again.camera.target[1], 1.0f);
            ExpectEqualF("field of view survives", again.camera.fovDegrees, 42.5f);
            ExpectEqualF("near plane survives", again.camera.nearPlane, 0.025f);
            ExpectEqualF("far plane survives", again.camera.farPlane, 250.0f);
        }
    }
    {
        // Windows paths are what a person pastes in, and a backslash is an
        // escape in JSON: writing one unescaped produces a file that will not
        // read back.
        SceneManifest manifest;
        NeuralModelIntegrateTestbed::SceneManifestEntry entry;
        entry.path = "C:\\Code\\niagara_bistro\\bistro.gltf";
        manifest.scenes.push_back(entry);
        const std::string written = WriteSceneManifest(manifest);
        SceneManifest again;
        std::string error;
        const bool reparsed = ParseSceneManifest(written, "", again, &error);
        Expect("a backslash path is escaped and reads back", reparsed);
        if (reparsed) {
            ExpectEqualS("the backslashes survive", again.scenes[0].path,
                         entry.path);
        } else {
            std::printf("      %s\n", error.c_str());
            ++g_failures;
        }
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all checks passed" << std::endl;
    return 0;
}
