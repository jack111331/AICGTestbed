# A floor, an occluder above it, and one directional light at 45 degrees, so a
# readback can tell a shadowed floor pixel from a lit one.
#
# Usage:
#
#     python tools/make_shadow_fixture.py resources/shadow-test/shadow-test.gltf
#     PTFLIO_SCENE=resources/shadow-test/shadow-test.gltf bazel-bin/Engine/AICGTestbed.exe
#
# Then put the camera at (0, 10, 0) looking at the origin with up (0, 0, -1) --
# straight down at the floor. The occluder sits in the middle of the frame and
# its shadow falls to the right of it; the printed u coordinates say where to
# look.
#
# The light travels along (1, -1, 0), so a point at height h casts its shadow h
# units to the +x side. The occluder spans x in [-1, 1] at y = 2, so its shadow
# lands on x in [1, 3] while the occluder itself only hides x in [-1, 1] from a
# camera looking straight down -- which is what leaves the shadow visible.
import base64
import json
import math
import os
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "shadow-test.gltf"

PLACEMENT = (3.0, -2.0, -4.0)
WATTS_TO_LUMENS = 683.0
# 3 W/m^2 once ToShaderLight has divided by 683.
SUN_LUX = 3.0 * WATTS_TO_LUMENS
CAMERA = (0.0, 10.0, 0.0)
FOV_Y = 70.0
WIDTH, HEIGHT = 1280, 720

FLOOR_HALF = 5.0
OCCLUDER_HALF = 1.0
OCCLUDER_Y = 2.0

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


def add_quad(name, half, y):
    """An axis-aligned quad in the XZ plane at height y, normal +Y."""
    positions = [(-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half)]
    p = add_accessor(add_view(b"".join(struct.pack("<3f", *q) for q in positions)),
                     5126, 4, "VEC3", [-half, y, -half], [half, y, half])
    n = add_accessor(add_view(b"".join(struct.pack("<3f", 0.0, 1.0, 0.0)
                                       for _ in positions)), 5126, 4, "VEC3")
    t = add_accessor(add_view(b"".join(struct.pack("<2f", u, v) for u, v in
                                       [(0, 0), (1, 0), (1, 1), (0, 1)])),
                     5126, 4, "VEC2")
    # TEXCOORD_1 is the LIGHTMAP UV set: a non-overlapping parameterisation of
    # the surface. For a single quad the texturing UVs already are one, so the
    # same values serve -- which is exactly why a quad fixture can exercise the
    # bake without an unwrapper in the build. A real asset's TEXCOORD_0 tiles
    # and cannot be reused this way.
    t1 = add_accessor(add_view(b"".join(struct.pack("<2f", u, v) for u, v in
                                        [(0, 0), (1, 0), (1, 1), (0, 1)])),
                      5126, 4, "VEC2")
    # Counter-clockwise seen from +Y, which is glTF's front face for a surface
    # whose normal is +Y.
    i = add_accessor(add_view(struct.pack("<6H", 0, 2, 1, 0, 3, 2)), 5123, 6, "SCALAR")
    materials.append({
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.8,
        },
        "alphaMode": "OPAQUE",
        "doubleSided": True,
    })
    meshes.append({"name": name, "primitives": [{
        "attributes": {"POSITION": p, "NORMAL": n, "TEXCOORD_0": t,
                       "TEXCOORD_1": t1},
        "indices": i, "material": len(materials) - 1}]})
    nodes.append({"name": name, "mesh": len(meshes) - 1})


add_quad("floor", FLOOR_HALF, 0.0)
add_quad("occluder", OCCLUDER_HALF, OCCLUDER_Y)

# One directional light travelling down and to +x. KHR_lights_punctual points a
# light down -Z by default and rotates it by the node, so the rotation below is
# the one taking -Z to the normalised (1, -1, 0).
#
# Axis-angle from -Z to d: axis = normalize(cross(-Z, d)), angle = acos(dot).
dx, dy, dz = 1.0, -1.0, 0.0
length = math.sqrt(dx * dx + dy * dy + dz * dz)
dx, dy, dz = dx / length, dy / length, dz / length
fx, fy, fz = 0.0, 0.0, -1.0
ax, ay, az = fy * dz - fz * dy, fz * dx - fx * dz, fx * dy - fy * dx
axis_len = math.sqrt(ax * ax + ay * ay + az * az)
dot = fx * dx + fy * dy + fz * dz
if axis_len < 1e-8:
    rotation = [0.0, 0.0, 0.0, 1.0] if dot > 0 else [1.0, 0.0, 0.0, 0.0]
else:
    ax, ay, az = ax / axis_len, ay / axis_len, az / axis_len
    angle = math.acos(max(-1.0, min(1.0, dot)))
    sin_half = math.sin(angle * 0.5)
    rotation = [ax * sin_half, ay * sin_half, az * sin_half, math.cos(angle * 0.5)]

nodes.append({"name": "sun", "rotation": rotation,
              "extensions": {"KHR_lights_punctual": {"light": 0}}})

root = len(nodes)
nodes.append({"name": "cancel-placement",
              "translation": [-PLACEMENT[0], -PLACEMENT[1], -PLACEMENT[2]],
              "children": list(range(len(nodes)))})

gltf = {
    "asset": {"version": "2.0", "generator": "ptflio shadow fixture"},
    "extensionsUsed": ["KHR_lights_punctual"],
    "extensions": {"KHR_lights_punctual": {"lights": [
        {"type": "directional", "color": [1.0, 1.0, 1.0], "intensity": SUN_LUX}]}},
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

# --- where to sample ---------------------------------------------------------
# Looking straight down, everything on a plane is at a constant distance, so the
# mapping from world x to screen u is linear.
aspect = WIDTH / HEIGHT
half_extent_at = lambda distance: distance * math.tan(math.radians(FOV_Y) / 2.0) * aspect
floor_half_width = half_extent_at(CAMERA[1] - 0.0)
occluder_half_width = half_extent_at(CAMERA[1] - OCCLUDER_Y)


def floor_u(x):
    return 0.5 + x / (2.0 * floor_half_width)


def occluder_u(x):
    return 0.5 + x / (2.0 * occluder_half_width)


shadow_near = max(floor_u(OCCLUDER_HALF), occluder_u(OCCLUDER_HALF))
shadow_far = floor_u(OCCLUDER_HALF + OCCLUDER_Y)
print()
print("camera (0, 10, 0) -> origin, up (0, 0, -1); sample at v = 0.5")
print()
print("  occluder hides u      %.3f .. %.3f" % (occluder_u(-OCCLUDER_HALF),
                                                occluder_u(OCCLUDER_HALF)))
print("  its shadow covers u   %.3f .. %.3f" % (floor_u(OCCLUDER_HALF - OCCLUDER_Y + 1.0),
                                                shadow_far))
print("  VISIBLE shadow at u   %.3f .. %.3f  -> sample %.3f"
      % (shadow_near, shadow_far, 0.5 * (shadow_near + shadow_far)))
print("  lit floor at u        %.3f and %.3f" % (floor_u(-3.0), floor_u(4.0)))
print()
print("A shadowed floor pixel must be much darker than a lit one, and with")
print("shadows disabled the two must read the same.")
print()
# The lightmap is addressed by the surface, not the screen, so its expectations
# are independent of where the camera is -- which is the whole point of baking.
print("--- in the floor's LIGHTMAP space (TEXCOORD_1, 0..1 over the quad) ---")
print()
floor_uv = lambda x: (x + FLOOR_HALF) / (2.0 * FLOOR_HALF)
print("  the floor spans x %.1f .. %.1f  ->  u 0.000 .. 1.000"
      % (-FLOOR_HALF, FLOOR_HALF))
print("  its shadow spans x %.1f .. %.1f  ->  u %.3f .. %.3f"
      % (OCCLUDER_HALF, OCCLUDER_HALF + OCCLUDER_Y,
         floor_uv(OCCLUDER_HALF), floor_uv(OCCLUDER_HALF + OCCLUDER_Y)))
print("  sample shadowed at u %.3f, lit at u %.3f and %.3f"
      % (floor_uv(OCCLUDER_HALF + OCCLUDER_Y * 0.5), floor_uv(-3.0),
         floor_uv(4.5)))
print()
print("A baked floor texel inside that u band must be dark and one outside lit,")
print("at every v, and neither must change when the camera moves.")
