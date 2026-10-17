# Builds a specular-glossiness fixture and predicts the metallic, roughness and
# base colour that PSGBuffer must write into the G-buffer for it.
#
# Usage:
#
#     python tools/make_specgloss_fixture.py resources/specgloss-test/specgloss-test.gltf
#     PTFLIO_SCENE=resources/specgloss-test/specgloss-test.gltf bazel-bin/Engine/hello-world.exe
#
# Deliberately FACTOR-ONLY, no textures: the renderer's 1x1 white fallback makes
# the sampled diffuse exactly (1,1,1), so Mat_BaseColorFactor and
# Mat_SpecularFactor are the only inputs and the conversion has known values to
# hit. That also means this fixture is what catches a packoffset mismatch
# between MaterialConstants and the HLSL cbuffer, which dxc cannot see.
#
# The conversion itself is the archived extension's own, from
# extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/examples/
# convert-between-workflows/js/ -- three.pbrUtilities.js for the solve and
# three.colorExtensions.js for getPerceivedBrightness.
import base64
import json
import math
import os
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "specgloss-test.gltf"

DIELECTRIC_SPECULAR = 0.04
EPSILON = 1e-6
PLACEMENT = (3.0, -2.0, -4.0)

# name, centre x, diffuseFactor rgb, specularFactor rgb, glossinessFactor
QUADS = [
    ("dielectric", -0.45, (0.50, 0.50, 0.50), (0.04, 0.04, 0.04), 0.50),
    ("metal",      -0.15, (0.00, 0.00, 0.00), (1.00, 1.00, 1.00), 0.80),
    ("red dielec",  0.15, (0.80, 0.20, 0.20), (0.10, 0.10, 0.10), 0.30),
    ("ambiguous",   0.45, (0.30, 0.30, 0.30), (0.30, 0.30, 0.30), 0.60),
]
HALF_W, HALF_H = 0.13, 0.6


def perceived_brightness(c):
    return math.sqrt(0.299 * c[0] * c[0] + 0.587 * c[1] * c[1] + 0.114 * c[2] * c[2])


def solve_metallic(diffuse, specular, one_minus_specular_strength):
    if specular < DIELECTRIC_SPECULAR:
        return 0.0
    a = DIELECTRIC_SPECULAR
    b = (diffuse * one_minus_specular_strength / (1.0 - DIELECTRIC_SPECULAR)
         + specular - 2.0 * DIELECTRIC_SPECULAR)
    c = DIELECTRIC_SPECULAR - specular
    d = max(0.0, b * b - 4.0 * a * c)
    return min(1.0, max(0.0, (-b + math.sqrt(d)) / (2.0 * a)))


def convert(diffuse, specular, glossiness):
    oms = 1.0 - max(specular)
    metallic = solve_metallic(perceived_brightness(diffuse),
                              perceived_brightness(specular), oms)
    from_diffuse = [c * (oms / (1.0 - DIELECTRIC_SPECULAR) /
                         max(1.0 - metallic, EPSILON)) for c in diffuse]
    from_specular = [(c - DIELECTRIC_SPECULAR * (1.0 - metallic)) *
                     (1.0 / max(metallic, EPSILON)) for c in specular]
    t = metallic * metallic
    base = [min(1.0, max(0.0, from_diffuse[i] * (1.0 - t) + from_specular[i] * t))
            for i in range(3)]
    return metallic, 1.0 - glossiness, base


# ----------------------------------------------------------------- glTF ------
blob = bytearray()
views, accessors, meshes, materials, nodes = [], [], [], [], []


def add_view(data):
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(data)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
    return len(views) - 1


def add_accessor(view, ctype, count, tname, mins=None, maxs=None):
    a = {"bufferView": view, "componentType": ctype, "count": count, "type": tname}
    if mins is not None:
        a["min"], a["max"] = mins, maxs
    accessors.append(a)
    return len(accessors) - 1


for name, cx, diffuse, specular, glossiness in QUADS:
    x0, x1 = cx - HALF_W, cx + HALF_W
    y0, y1 = -HALF_H, HALF_H
    positions = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    p_acc = add_accessor(add_view(b"".join(struct.pack("<3f", *p) for p in positions)),
                         5126, 4, "VEC3", [x0, y0, 0.0], [x1, y1, 0.0])
    n_acc = add_accessor(add_view(b"".join(struct.pack("<3f", 0.0, 0.0, 1.0)
                                           for _ in positions)), 5126, 4, "VEC3")
    t_acc = add_accessor(add_view(b"".join(struct.pack("<2f", u, v) for u, v in
                                           [(0, 1), (1, 1), (1, 0), (0, 0)])),
                         5126, 4, "VEC2")
    i_acc = add_accessor(add_view(struct.pack("<6H", 0, 1, 2, 0, 2, 3)),
                         5123, 6, "SCALAR")

    materials.append({
        "name": name,
        # A spec-gloss material still carries a pbrMetallicRoughness block;
        # glTF says a renderer that does not understand the extension falls back
        # to it. Values here are deliberately absurd so that if the renderer
        # ever silently took this path instead, the test would fail loudly.
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 0.0, 1.0, 1.0],
            "metallicFactor": 1.0,
            "roughnessFactor": 0.0,
        },
        "extensions": {"KHR_materials_pbrSpecularGlossiness": {
            "diffuseFactor": list(diffuse) + [1.0],
            "specularFactor": list(specular),
            "glossinessFactor": glossiness,
        }},
        "alphaMode": "OPAQUE",
        "doubleSided": False,
    })
    meshes.append({"name": name, "primitives": [{
        "attributes": {"POSITION": p_acc, "NORMAL": n_acc, "TEXCOORD_0": t_acc},
        "indices": i_acc, "material": len(materials) - 1}]})
    nodes.append({"name": name, "mesh": len(meshes) - 1})

root = len(nodes)
nodes.append({"name": "cancel-placement",
              "translation": [-PLACEMENT[0], -PLACEMENT[1], -PLACEMENT[2]],
              "children": list(range(len(nodes)))})

gltf = {
    "asset": {"version": "2.0", "generator": "ptflio spec-gloss fixture"},
    "extensionsUsed": ["KHR_materials_pbrSpecularGlossiness"],
    "scene": 0,
    "scenes": [{"nodes": [root]}],
    "nodes": nodes,
    "meshes": meshes,
    "materials": materials,
    "accessors": accessors,
    "bufferViews": views,
    "buffers": [{"byteLength": len(blob),
                 "uri": "data:application/octet-stream;base64," +
                        base64.b64encode(bytes(blob)).decode("ascii")}],
}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(gltf, handle, indent=1)
print("wrote", OUT)

# ----------------------------------------------------------- prediction ------
WIDTH, HEIGHT = 1280, 720
m11 = 1.0 / (math.tan(math.radians(70.0) / 2.0) * (WIDTH / HEIGHT))
CAMERA_Z = 3.0

print()
print("Put the camera at (0, 0, 3) looking at the origin, then read the G-buffer.")
print("Values are the bytes an R8G8B8A8_UNORM target holds.")
print()
print("%-12s %8s %8s %9s %-16s" % ("quad", "screen u", "metallic", "roughness",
                                   "base colour"))
print("-" * 60)
for name, cx, diffuse, specular, glossiness in QUADS:
    metallic, roughness, base = convert(diffuse, specular, glossiness)
    u = (cx * m11 / CAMERA_Z) * 0.5 + 0.5
    print("%-12s %8.4f %8d %9d  %3d,%3d,%3d" % (
        name, u, round(metallic * 255), round(roughness * 255),
        round(base[0] * 255), round(base[1] * 255), round(base[2] * 255)))
print()
print("If any base colour reads 255,0,255 the extension was ignored and the")
print("fallback pbrMetallicRoughness block was used instead.")
