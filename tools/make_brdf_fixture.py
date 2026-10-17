# Builds a deterministic BRDF fixture and predicts what PSLighting must put on
# screen for it, so the shader edit can be checked against the Khronos formulas
# end to end rather than only on paper.
#
# Four quads in the XY plane at z=0, normals +Z, one point light on the axis,
# camera on the axis. No textures: the renderer's 1x1 white fallback makes
# baseColor exactly (1,1,1), and the metallic/roughness factors are the only
# variables. Every dot product at a quad's centre is then closed-form.
import base64
import json
import math
import os
import struct
import sys

# Usage:
#
#     python tools/make_brdf_fixture.py resources/brdf-test/brdf-test.gltf
#     PTFLIO_SCENE=resources/brdf-test/brdf-test.gltf bazel-bin/Engine/hello-world.exe
#
# Then put the camera at (0, 0, 3) looking at the origin -- the Camera panel
# takes those directly -- and the four quads must read, left to right, the
# "byte" column this script prints. The "before" column is what the shader gave
# when the specular term was G*D with no 1/(4 NdotV NdotL): note the metals move
# a long way and the dielectrics barely do, because a dielectric's 0.04 Fresnel
# damps the specular almost entirely.
OUT = sys.argv[1] if len(sys.argv) > 1 else "brdf-test.gltf"

# --- the scene ---------------------------------------------------------------
# Everything lifted together, so the quad centres stay head-on to both the
# camera and the light while the app's ground grid -- which draws into the same
# target before the tonemap -- stays well below the sample row.
CAMERA = (0.0, 3.0, 3.0)
LIGHT = (0.0, 3.0, 2.0)
# Authored in CANDELA, as KHR_lights_punctual requires. ToShaderLight divides
# by 683 lm/W on load, so this is 0.4 W/sr by the time the shader sees it --
# chosen to put the quads in a readable range rather than for realism.
INTENSITY_CD = 0.4 * 683.0
WATTS_TO_LUMENS = 683.0
INTENSITY = INTENSITY_CD / WATTS_TO_LUMENS
# Cancels GLTFAdapter::m_position, so the quads land where this script assumes.
PLACEMENT = (3.0, -2.0, -4.0)

# name, centre x, metallic, roughness
QUADS = [
    ("metal  r=0.4", -0.45, 1.0, 0.4),
    ("dielec r=0.5", -0.15, 0.0, 0.5),
    ("metal  r=0.2", 0.15, 1.0, 0.2),
    ("dielec r=0.9", 0.45, 0.0, 0.9),
]
# Lifted clear of y = 0. The app's ground grid draws into the same target
# before the tonemap, and a camera at y = 0 sees it edge-on as a line straight
# across the middle of the frame -- which is where these would otherwise be.
CENTRE_Y = 3.0
HALF_W, HALF_H = 0.13, 0.4

# --- the glTF ----------------------------------------------------------------
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


for name, cx, metallic, roughness in QUADS:
    x0, x1 = cx - HALF_W, cx + HALF_W
    y0, y1 = CENTRE_Y - HALF_H, CENTRE_Y + HALF_H
    positions = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", 0.0, 0.0, 1.0) for _ in positions)
    uv_bytes = b"".join(struct.pack("<2f", u, v)
                        for u, v in [(0, 1), (1, 1), (1, 0), (0, 0)])
    idx_bytes = struct.pack("<6H", 0, 1, 2, 0, 2, 3)

    p_acc = add_accessor(add_view(pos_bytes), 5126, 4, "VEC3",
                         [x0, y0, 0.0], [x1, y1, 0.0])
    n_acc = add_accessor(add_view(nrm_bytes), 5126, 4, "VEC3")
    t_acc = add_accessor(add_view(uv_bytes), 5126, 4, "VEC2")
    i_acc = add_accessor(add_view(idx_bytes), 5123, 6, "SCALAR")

    materials.append({
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": metallic,
            "roughnessFactor": roughness,
        },
        "alphaMode": "OPAQUE",
        "doubleSided": False,
    })
    meshes.append({"name": name, "primitives": [{
        "attributes": {"POSITION": p_acc, "NORMAL": n_acc, "TEXCOORD_0": t_acc},
        "indices": i_acc, "material": len(materials) - 1}]})
    nodes.append({"name": name, "mesh": len(meshes) - 1})

# One white point light with unlimited range, so attenuation is exactly 1/d^2.
light_node = len(nodes)
nodes.append({"name": "key", "translation": list(LIGHT),
              "extensions": {"KHR_lights_punctual": {"light": 0}}})

root = len(nodes)
nodes.append({"name": "cancel-placement",
              "translation": [-PLACEMENT[0], -PLACEMENT[1], -PLACEMENT[2]],
              "children": list(range(len(nodes)))})

gltf = {
    "asset": {"version": "2.0", "generator": "ptflio brdf fixture"},
    "extensionsUsed": ["KHR_lights_punctual"],
    "extensions": {"KHR_lights_punctual": {"lights": [
        {"type": "point", "color": [1.0, 1.0, 1.0], "intensity": INTENSITY_CD}]}},
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
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(gltf, handle, indent=1)
print("wrote", OUT)

# --- the prediction ----------------------------------------------------------
PI = math.pi
GAMMA = 2.2
WIDTH, HEIGHT = 1280, 720
FOV_Y = 70.0


def norm(v):
    length = math.sqrt(sum(c * c for c in v))
    return tuple(c / length for c in v), length


def d_ggx(ndoth, rough):
    a2 = (rough * rough) ** 2
    f = ndoth * ndoth * (a2 - 1.0) + 1.0
    return a2 / (PI * f * f)


def g_schlick(ndotv, ndotl, rough):
    """The superseded term: G, with no 1/(4 NdotV NdotL) applied."""
    k = (rough + 1.0) ** 2 / 8.0
    return (ndotv / (ndotv * (1.0 - k) + k)) * (ndotl / (ndotl * (1.0 - k) + k))


def v_ggx(ndotl, ndotv, rough):
    a2 = (rough * rough) ** 2
    gv = ndotl * math.sqrt(ndotv * ndotv * (1.0 - a2) + a2)
    gl = ndotv * math.sqrt(ndotl * ndotl * (1.0 - a2) + a2)
    return 0.5 / (gv + gl) if (gv + gl) > 0.0 else 0.0


aspect = WIDTH / HEIGHT
m11 = 1.0 / (math.tan(math.radians(FOV_Y) / 2.0) * aspect)

print()
print("camera %s, light %s" % (CAMERA, LIGHT))
print("light authored at %.1f cd -> %.3f W/sr after the 683 conversion"
      % (INTENSITY_CD, INTENSITY))
print("aim the camera at (0, %.1f, 0) from %s; sample at the printed u,v"
      % (CENTRE_Y, CAMERA))
print()
print("%-14s %15s %9s %6s %6s" % ("quad", "screen u,v", "linear", "byte", "before"))
print("-" * 48)
for name, cx, metallic, roughness in QUADS:
    # The G-buffer's material target is R8G8B8A8_UNORM, so this is what the
    # lighting pass actually reads -- not the authored factor.
    metallic = round(metallic * 255.0) / 255.0
    roughness = round(roughness * 255.0) / 255.0
    P = (cx, CENTRE_Y, 0.0)
    V, _ = norm(tuple(CAMERA[i] - P[i] for i in range(3)))
    L, distance = norm(tuple(LIGHT[i] - P[i] for i in range(3)))
    H, _ = norm(tuple(L[i] + V[i] for i in range(3)))
    N = (0.0, 0.0, 1.0)
    ndotl = max(0.0, sum(N[i] * L[i] for i in range(3)))
    ndotv = max(0.0, sum(N[i] * V[i] for i in range(3)))
    ndoth = max(0.0, sum(N[i] * H[i] for i in range(3)))
    vdoth = max(0.0, sum(V[i] * H[i] for i in range(3)))

    radiance = INTENSITY / (distance * distance)
    spec = radiance * ndotl * v_ggx(ndotl, ndotv, roughness) * d_ggx(ndoth, roughness)
    diff = radiance * ndotl / PI
    f_diel = 0.04 + 0.96 * (1.0 - vdoth) ** 5
    f_metal = 1.0  # F0 == baseColor == 1, so Schlick returns 1 at any angle
    metal = f_metal * spec
    dielectric = diff * (1.0 - f_diel) + spec * f_diel
    colour = dielectric * (1.0 - metallic) + metal * metallic
    colour += 0.03  # the flat ambient term, baseColor 1 and AO 1
    tonemapped = colour / (1.0 + colour)
    displayed = tonemapped ** (1.0 / GAMMA)

    # What the shader produced before the fix: G instead of V, no denominator.
    spec_before = radiance * ndotl * g_schlick(ndotv, ndotl, roughness) *         d_ggx(ndoth, roughness)
    metal_before = f_metal * spec_before
    diel_before = diff * (1.0 - f_diel) + spec_before * f_diel
    before = diel_before * (1.0 - metallic) + metal_before * metallic + 0.03
    before = (before / (1.0 + before)) ** (1.0 / GAMMA)

    # World x -> NDC -> screen, for the readback.
    u = (cx * m11 / (CAMERA[2] - P[2])) * 0.5 + 0.5
    # The camera sits at the quads' own height and looks horizontally, so
    # their centres land on the middle row.
    v = 0.5
    print("%-14s %7.4f,%-7.4f %9.4f %6d %6d"
          % (name, u, v, colour, round(displayed * 255.0), round(before * 255.0)))
