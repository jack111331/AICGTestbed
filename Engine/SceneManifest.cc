#include "SceneManifest.hpp"

#include <simdjson.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>

namespace NeuralModelIntegrateTestbed {

namespace {

// simdjson rather than a hand-rolled parser or a new vcpkg port: fastgltf
// already vendors it and //third exposes it as its own target, so this costs
// one Bazel dep and no step the person building has to run.
//
// The error-code API throughout, never the exception-throwing accessors: a
// manifest is user-authored text and every way it can be wrong has to come back
// as a message naming the field.

const char* TypeName(simdjson::dom::element_type type) {
    switch (type) {
        case simdjson::dom::element_type::ARRAY:      return "an array";
        case simdjson::dom::element_type::OBJECT:     return "an object";
        case simdjson::dom::element_type::INT64:
        case simdjson::dom::element_type::UINT64:
        case simdjson::dom::element_type::DOUBLE:     return "a number";
        case simdjson::dom::element_type::STRING:     return "a string";
        case simdjson::dom::element_type::BOOL:       return "a boolean";
        case simdjson::dom::element_type::NULL_VALUE: return "null";
    }
    return "something unexpected";
}

void Fail(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

// A number, whether the JSON wrote it as 1, 1.0 or -1e2. simdjson types those
// as three different element types, and a manifest that says "scale": 1 must
// not be a type error.
bool ReadNumber(simdjson::dom::element element, const std::string& what,
                double& out, std::string* error) {
    double value = 0.0;
    if (element.get_double().get(value) == simdjson::SUCCESS) {
        out = value;
        return true;
    }
    std::int64_t signedValue = 0;
    if (element.get_int64().get(signedValue) == simdjson::SUCCESS) {
        out = static_cast<double>(signedValue);
        return true;
    }
    std::uint64_t unsignedValue = 0;
    if (element.get_uint64().get(unsignedValue) == simdjson::SUCCESS) {
        out = static_cast<double>(unsignedValue);
        return true;
    }
    Fail(error, what + " must be a number, not " + TypeName(element.type()));
    return false;
}

// Three numbers into `out`. Rejects any other length rather than padding: a
// two-element translation is a typo, and silently reading it as (x, y, 0) puts
// the character somewhere the file does not say.
bool ReadVector3(simdjson::dom::element element, const std::string& what,
                 float (&out)[3], std::string* error) {
    simdjson::dom::array array;
    if (element.get_array().get(array) != simdjson::SUCCESS) {
        Fail(error, what + " must be an array of 3 numbers, not " +
                        TypeName(element.type()));
        return false;
    }
    if (array.size() != 3) {
        Fail(error, what + " must have 3 numbers, not " +
                        std::to_string(array.size()));
        return false;
    }
    std::size_t index = 0;
    for (simdjson::dom::element component : array) {
        double value = 0.0;
        if (!ReadNumber(component, what + "[" + std::to_string(index) + "]", value,
                        error)) {
            return false;
        }
        if (!std::isfinite(value)) {
            Fail(error, what + "[" + std::to_string(index) + "] is not finite");
            return false;
        }
        out[index] = static_cast<float>(value);
        ++index;
    }
    return true;
}

// Resolves one manifest path against the manifest's own directory. An absolute
// path is left alone, so a manifest can point at a scene staged outside the
// repository -- which is how the Bistro is used here.
std::string ResolveAgainst(const std::filesystem::path& directory,
                           const std::string& path) {
    const std::filesystem::path given(path);
    if (given.is_absolute() || directory.empty()) {
        return given.generic_string();
    }
    return (directory / given).lexically_normal().generic_string();
}

// One optional float field, left alone when the manifest does not mention it.
bool ReadOptionalFloat(simdjson::dom::element element, const std::string& what,
                       float& out, bool& present, float low, float high,
                       std::string* error) {
    double value = 0.0;
    if (!ReadNumber(element, what, value, error)) {
        return false;
    }
    if (!std::isfinite(value) || value < low || value > high) {
        char range[128];
        std::snprintf(range, sizeof(range), " must be between %g and %g", low, high);
        Fail(error, what + range);
        return false;
    }
    out = static_cast<float>(value);
    present = true;
    return true;
}

bool ReadCamera(simdjson::dom::element element, SceneCamera& out,
                std::string* error) {
    simdjson::dom::object object;
    if (element.get_object().get(object) != simdjson::SUCCESS) {
        Fail(error, std::string("\"camera\" must be an object, not ") +
                        TypeName(element.type()));
        return false;
    }

    bool havePosition = false;
    bool haveTarget = false;
    for (auto field : object) {
        const std::string_view key = field.key;
        if (key == "position") {
            if (!ReadVector3(field.value, "camera.position", out.position, error)) {
                return false;
            }
            havePosition = true;
        } else if (key == "target") {
            if (!ReadVector3(field.value, "camera.target", out.target, error)) {
                return false;
            }
            haveTarget = true;
        } else if (key == "up") {
            if (!ReadVector3(field.value, "camera.up", out.up, error)) {
                return false;
            }
        } else if (key == "fovDegrees") {
            // The same bounds the UI's slider offers. Zero or 180 degrees are
            // both a projection matrix of infinities.
            if (!ReadOptionalFloat(field.value, "camera.fovDegrees", out.fovDegrees,
                                   out.hasFov, 1.0f, 179.0f, error)) {
                return false;
            }
        } else if (key == "near") {
            if (!ReadOptionalFloat(field.value, "camera.near", out.nearPlane,
                                   out.hasNear, 1e-6f, 1e9f, error)) {
                return false;
            }
        } else if (key == "far") {
            if (!ReadOptionalFloat(field.value, "camera.far", out.farPlane,
                                   out.hasFar, 1e-6f, 1e9f, error)) {
                return false;
            }
        } else {
            Fail(error, std::string("camera has an unknown field \"") +
                            std::string(key) + "\"");
            return false;
        }
    }

    if (!havePosition || !haveTarget) {
        // Either alone is not a camera: a position with no target would have to
        // invent a direction, which is the guess the manifest exists to avoid.
        Fail(error, "camera needs both \"position\" and \"target\"");
        return false;
    }
    // CreateLookAt produces NaNs when the eye and the target coincide, and one
    // NaN in the view matrix takes the whole frame with it -- including the UI
    // that would let you fix it.
    const float dx = out.target[0] - out.position[0];
    const float dy = out.target[1] - out.position[1];
    const float dz = out.target[2] - out.position[2];
    if (dx * dx + dy * dy + dz * dz < 1e-12f) {
        Fail(error, "camera.target is the same point as camera.position");
        return false;
    }
    if (out.up[0] * out.up[0] + out.up[1] * out.up[1] + out.up[2] * out.up[2] <
        1e-12f) {
        Fail(error, "camera.up has no length");
        return false;
    }
    if (out.hasNear && out.hasFar && out.farPlane <= out.nearPlane) {
        Fail(error, "camera.far must be greater than camera.near");
        return false;
    }
    return true;
}

bool ReadEntry(simdjson::dom::element element,
               const std::filesystem::path& directory, std::size_t index,
               SceneManifestEntry& out, std::string* error) {
    const std::string what = "scenes[" + std::to_string(index) + "]";

    // The shorthand: a bare string is a path with the default transform, which
    // is what most entries are.
    std::string_view asString;
    if (element.get_string().get(asString) == simdjson::SUCCESS) {
        out.path.assign(asString);
        out.resolvedPath = ResolveAgainst(directory, out.path);
        return !out.path.empty()
                   ? true
                   : (Fail(error, what + " is an empty path"), false);
    }

    simdjson::dom::object object;
    if (element.get_object().get(object) != simdjson::SUCCESS) {
        Fail(error, what + " must be a path string or an object, not " +
                        TypeName(element.type()));
        return false;
    }

    bool havePath = false;
    for (auto field : object) {
        const std::string_view key = field.key;
        if (key == "path") {
            std::string_view value;
            if (field.value.get_string().get(value) != simdjson::SUCCESS) {
                Fail(error, what + ".path must be a string, not " +
                                TypeName(field.value.type()));
                return false;
            }
            out.path.assign(value);
            havePath = true;
        } else if (key == "translation") {
            if (!ReadVector3(field.value, what + ".translation",
                             out.transform.translation, error)) {
                return false;
            }
        } else if (key == "rotation") {
            if (!ReadVector3(field.value, what + ".rotation",
                             out.transform.rotation, error)) {
                return false;
            }
        } else if (key == "scale") {
            double value = 0.0;
            if (!ReadNumber(field.value, what + ".scale", value, error)) {
                return false;
            }
            // Zero collapses the file to a point and makes its world matrices
            // singular, which ComputeJointMatrices inverts -- a skinned
            // character would come out as NaNs rather than as something small.
            if (!std::isfinite(value) || value <= 0.0) {
                Fail(error, what + ".scale must be greater than zero");
                return false;
            }
            out.transform.scale = static_cast<float>(value);
        } else if (key == "visible") {
            bool value = true;
            if (field.value.get_bool().get(value) != simdjson::SUCCESS) {
                Fail(error, what + ".visible must be true or false, not " +
                                TypeName(field.value.type()));
                return false;
            }
            out.transform.visible = value;
        } else {
            // Named rather than ignored. A manifest is hand-written, and
            // "translate" or "rotationDegrees" would otherwise be accepted in
            // silence and simply not do anything.
            Fail(error, what + " has an unknown field \"" + std::string(key) + "\"");
            return false;
        }
    }

    if (!havePath || out.path.empty()) {
        Fail(error, what + " has no \"path\"");
        return false;
    }
    out.resolvedPath = ResolveAgainst(directory, out.path);
    return true;
}

}  // namespace

bool ParseSceneManifest(std::string_view json,
                        const std::filesystem::path& directory,
                        SceneManifest& out, std::string* error) {
    out = SceneManifest{};

    // simdjson reads past the end of the buffer by design, so it needs the
    // padded copy rather than the caller's view.
    simdjson::padded_string padded(json);
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    const simdjson::error_code parsed = parser.parse(padded).get(document);
    if (parsed != simdjson::SUCCESS) {
        Fail(error, std::string("not valid JSON: ") + simdjson::error_message(parsed));
        return false;
    }

    simdjson::dom::object root;
    if (document.get_object().get(root) != simdjson::SUCCESS) {
        Fail(error, std::string("the manifest must be an object, not ") +
                        TypeName(document.type()));
        return false;
    }

    bool haveScenes = false;
    for (auto field : root) {
        const std::string_view key = field.key;
        if (key == "scenes") {
            simdjson::dom::array array;
            if (field.value.get_array().get(array) != simdjson::SUCCESS) {
                Fail(error, std::string("\"scenes\" must be an array, not ") +
                                TypeName(field.value.type()));
                return false;
            }
            std::size_t index = 0;
            for (simdjson::dom::element element : array) {
                SceneManifestEntry entry;
                if (!ReadEntry(element, directory, index, entry, error)) {
                    return false;
                }
                out.scenes.push_back(std::move(entry));
                ++index;
            }
            haveScenes = true;
        } else if (key == "name") {
            std::string_view value;
            if (field.value.get_string().get(value) != simdjson::SUCCESS) {
                Fail(error, std::string("\"name\" must be a string, not ") +
                                TypeName(field.value.type()));
                return false;
            }
            out.name.assign(value);
        } else if (key == "skybox") {
            std::string_view value;
            if (field.value.get_string().get(value) != simdjson::SUCCESS) {
                Fail(error, std::string("\"skybox\" must be a string, not ") +
                                TypeName(field.value.type()));
                return false;
            }
            out.skybox.assign(value);
            out.resolvedSkybox = ResolveAgainst(directory, out.skybox);
        } else if (key == "camera") {
            if (!ReadCamera(field.value, out.camera, error)) {
                return false;
            }
            out.hasCamera = true;
        } else if (key == "maxTextureDimension") {
            double value = 0.0;
            if (!ReadNumber(field.value, "\"maxTextureDimension\"", value, error)) {
                return false;
            }
            if (!std::isfinite(value) || value < 0.0 || value > 65536.0) {
                Fail(error, "\"maxTextureDimension\" must be between 0 and 65536");
                return false;
            }
            out.hasMaxTextureDimension = true;
            out.maxTextureDimension = static_cast<std::uint32_t>(value);
        } else {
            Fail(error, std::string("unknown field \"") + std::string(key) + "\"");
            return false;
        }
    }

    if (!haveScenes) {
        Fail(error, "the manifest has no \"scenes\" array");
        return false;
    }
    if (out.scenes.empty()) {
        // Not an empty scene to render: far more likely a manifest half
        // written, and an empty window is a poor way to report it.
        Fail(error, "\"scenes\" is empty");
        return false;
    }
    return true;
}

bool LoadSceneManifest(const std::filesystem::path& path, SceneManifest& out,
                       std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        Fail(error, "could not open " + path.generic_string());
        return false;
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    const std::string json = contents.str();

    // The manifest's own directory, which is what its relative paths mean.
    std::error_code ignored;
    const std::filesystem::path absolute = std::filesystem::absolute(path, ignored);
    const std::filesystem::path directory =
        (ignored ? path : absolute).parent_path();

    if (!ParseSceneManifest(json, directory, out, error)) {
        if (error != nullptr) {
            *error = path.generic_string() + ": " + *error;
        }
        return false;
    }
    out.sourcePath = path.generic_string();
    return true;
}

namespace {

// %g would write 0.01 as 0.01 but 100000 as 1e+05, and JSON has no exponent
// problem -- it is just hard to read. %.6g with the exponent cases spelled out
// is not worth it; these are small numbers a person typed.
std::string Number(float value) {
    char buffer[64];
    // A whole number comes out without a decimal point, so a translation of 2
    // does not become 2.000000.
    if (value == std::floor(value) && std::fabs(value) < 1e7f) {
        std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(value));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.6g", static_cast<double>(value));
    }
    return buffer;
}

std::string Vector3(const float (&v)[3]) {
    return "[" + Number(v[0]) + ", " + Number(v[1]) + ", " + Number(v[2]) + "]";
}

// JSON string escaping, for the paths. Only the two characters that can
// actually appear in one: a Windows path written with backslashes, and a quote
// in a file name.
std::string Quote(const std::string& value) {
    std::string out = "\"";
    for (const char c : value) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

}  // namespace

std::string WriteSceneManifest(const SceneManifest& manifest) {
    std::string out = "{\n";
    if (!manifest.name.empty()) {
        out += "  \"name\": " + Quote(manifest.name) + ",\n";
    }
    if (!manifest.skybox.empty()) {
        out += "  \"skybox\": " + Quote(manifest.skybox) + ",\n";
    }
    if (manifest.hasMaxTextureDimension) {
        out += "  \"maxTextureDimension\": " +
               std::to_string(manifest.maxTextureDimension) + ",\n";
    }
    if (manifest.hasCamera) {
        const SceneCamera& camera = manifest.camera;
        out += "  \"camera\": {\n";
        out += "    \"position\": " + Vector3(camera.position) + ",\n";
        out += "    \"target\": " + Vector3(camera.target) + ",\n";
        out += "    \"up\": " + Vector3(camera.up);
        // Each written only when it was given, so a manifest that pins where
        // the camera stands but not its lens round-trips as that.
        if (camera.hasFov) {
            out += ",\n    \"fovDegrees\": " + Number(camera.fovDegrees);
        }
        if (camera.hasNear) {
            out += ",\n    \"near\": " + Number(camera.nearPlane);
        }
        if (camera.hasFar) {
            out += ",\n    \"far\": " + Number(camera.farPlane);
        }
        out += "\n  },\n";
    }
    out += "  \"scenes\": [\n";
    for (std::size_t i = 0; i < manifest.scenes.size(); ++i) {
        const SceneManifestEntry& entry = manifest.scenes[i];
        const char* separator = (i + 1 == manifest.scenes.size()) ? "\n" : ",\n";
        if (entry.transform.IsIdentity()) {
            // Back to the shorthand, so a file nobody has moved stays one line.
            out += "    " + Quote(entry.path) + separator;
            continue;
        }
        out += "    {\n";
        out += "      \"path\": " + Quote(entry.path) + ",\n";
        out += "      \"translation\": " + Vector3(entry.transform.translation) + ",\n";
        out += "      \"rotation\": " + Vector3(entry.transform.rotation) + ",\n";
        out += "      \"scale\": " + Number(entry.transform.scale);
        // Written only when it is false: `"visible": true` on every entry is
        // noise, and its absence already means shown.
        if (!entry.transform.visible) {
            out += ",\n      \"visible\": false";
        }
        out += "\n    }";
        out += separator;
    }
    out += "  ]\n}\n";
    return out;
}

bool SaveSceneManifest(const std::filesystem::path& path,
                       const SceneManifest& manifest, std::string* error) {
    std::error_code code;
    const std::filesystem::path directory = path.parent_path();
    if (!directory.empty()) {
        std::filesystem::create_directories(directory, code);
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        Fail(error, "could not write " + path.generic_string());
        return false;
    }
    const std::string text = WriteSceneManifest(manifest);
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!file) {
        Fail(error, "failed while writing " + path.generic_string());
        return false;
    }
    return true;
}

}  // namespace NeuralModelIntegrateTestbed
