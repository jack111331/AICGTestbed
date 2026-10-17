"""How many primitives can each point light actually reach?

Composes node world transforms and transforms each primitive's POSITION
min/max (glTF requires those on POSITION, so no binary data is read). Then
intersects every light's range sphere against every primitive's world AABB.

Read-only JSON parse of untrusted asset data.
"""
import json
import math
import statistics
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    g = json.load(handle)

nodes = g.get("nodes", [])
meshes = g.get("meshes", [])
accessors = g.get("accessors", [])
lights = g.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])


def mat_identity():
    return [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def mat_mul(a, b):
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[r * 4 + c] = sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4))
    return out


def node_local(n):
    if "matrix" in n:
        m = n["matrix"]  # glTF matrices are column-major
        return [m[0], m[4], m[8], m[12],
                m[1], m[5], m[9], m[13],
                m[2], m[6], m[10], m[14],
                m[3], m[7], m[11], m[15]]
    t = n.get("translation", [0.0, 0.0, 0.0])
    r = n.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = n.get("scale", [1.0, 1.0, 1.0])
    x, y, z, w = r
    rot = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0,
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0,
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0,
        0, 0, 0, 1,
    ]
    for c in range(3):
        for r_ in range(3):
            rot[r_ * 4 + c] *= s[c]
    rot[3] = t[0]
    rot[7] = t[1]
    rot[11] = t[2]
    return rot


# Walk the default scene, composing world transforms.
scene = g.get("scenes", [{}])[g.get("scene", 0)]
world = {}
stack = [(i, mat_identity()) for i in scene.get("nodes", [])]
while stack:
    i, parentWorld = stack.pop()
    w = mat_mul(parentWorld, node_local(nodes[i]))
    world[i] = w
    for c in nodes[i].get("children", []):
        stack.append((c, w))


def transform_aabb(w, lo, hi):
    """Centre/extent form, taking absolute values of the linear part."""
    centre = [(lo[k] + hi[k]) * 0.5 for k in range(3)]
    extent = [(hi[k] - lo[k]) * 0.5 for k in range(3)]
    oc = [w[r * 4 + 0] * centre[0] + w[r * 4 + 1] * centre[1] +
          w[r * 4 + 2] * centre[2] + w[r * 4 + 3] for r in range(3)]
    oe = [abs(w[r * 4 + 0]) * extent[0] + abs(w[r * 4 + 1]) * extent[1] +
          abs(w[r * 4 + 2]) * extent[2] for r in range(3)]
    return oc, oe


# Every primitive's world AABB.
prims = []
for i, n in enumerate(nodes):
    if "mesh" not in n or i not in world:
        continue
    for prim in meshes[n["mesh"]].get("primitives", []):
        pos = prim.get("attributes", {}).get("POSITION")
        if pos is None:
            continue
        a = accessors[pos]
        if "min" not in a or "max" not in a:
            continue
        prims.append(transform_aabb(world[i], a["min"], a["max"]))

print("primitives with bounds:", len(prims))

# Light world positions.
point_lights = []
for i, n in enumerate(nodes):
    ext = n.get("extensions", {}).get("KHR_lights_punctual")
    if ext is None or i not in world:
        continue
    light = lights[ext.get("light", 0)]
    if light.get("type") != "point":
        continue
    w = world[i]
    point_lights.append(((w[3], w[7], w[11]), light.get("range")))

print("point lights placed:", len(point_lights))

# Scene bounds, for scale.
lo = [min(c[k] - e[k] for c, e in prims) for k in range(3)]
hi = [max(c[k] + e[k] for c, e in prims) for k in range(3)]
print("scene bounds %.1f..%.1f  %.1f..%.1f  %.1f..%.1f"
      % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
print("scene diagonal %.1f"
      % math.sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(3))))


def sphere_hits_aabb(p, r, centre, extent):
    d2 = 0.0
    for k in range(3):
        delta = abs(p[k] - centre[k]) - extent[k]
        if delta > 0:
            d2 += delta * delta
    return d2 <= r * r


counts = []
for p, r in point_lights:
    if r is None:
        counts.append(len(prims))
        continue
    counts.append(sum(1 for c, e in prims if sphere_hits_aabb(p, r, c, e)))

counts.sort()
print()
print("primitives within a point light's range:")
print("  min %d  median %d  mean %.1f  max %d"
      % (counts[0], counts[len(counts) // 2],
         statistics.fmean(counts), counts[-1]))
print("  total over all lights: %d" % sum(counts))
print("  as a share of %d x %d = %d: %.2f%%"
      % (len(point_lights), len(prims), len(point_lights) * len(prims),
         100.0 * sum(counts) / (len(point_lights) * len(prims))))
print("  lights reaching nothing: %d" % sum(1 for c in counts if c == 0))
