"""Can this asset carry a baked per-object shadow?

A per-object bake needs somewhere to store the result on the surface:
  * a second UV set (TEXCOORD_1) that is a non-overlapping atlas of the
    surface, which is what a lightmap is sampled with; or
  * failing that, enough vertices to store a usable per-vertex term.

So this reports, per primitive: which TEXCOORD sets exist, whether TEXCOORD_0's
range escapes [0,1] (which means it tiles, and tiled UVs cannot address a
lightmap), the vertex and triangle counts, and the world-space surface area
implied by the bounds -- which is what sets the lightmap memory bill.

Read-only JSON parse of untrusted asset data; no binary buffers are read.
"""
import collections
import json
import math
import statistics
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    g = json.load(handle)

accessors = g.get("accessors", [])
meshes = g.get("meshes", [])
nodes = g.get("nodes", [])


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
        m = n["matrix"]
        return [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13],
                m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]]
    t = n.get("translation", [0.0, 0.0, 0.0])
    r = n.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = n.get("scale", [1.0, 1.0, 1.0])
    x, y, z, w = r
    rot = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0,
           2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0,
           2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0,
           0, 0, 0, 1]
    for c in range(3):
        for r_ in range(3):
            rot[r_ * 4 + c] *= s[c]
    rot[3], rot[7], rot[11] = t[0], t[1], t[2]
    return rot


scene = g.get("scenes", [{}])[g.get("scene", 0)]
world = {}
stack = [(i, mat_identity()) for i in scene.get("nodes", [])]
while stack:
    i, pw = stack.pop()
    w = mat_mul(pw, node_local(nodes[i]))
    world[i] = w
    for c in nodes[i].get("children", []):
        stack.append((c, w))

uv_sets = collections.Counter()
tiling = 0
inside_unit = 0
no_uv_range = 0
verts = []
tris = []
areas = []
prim_count = 0

for i, n in enumerate(nodes):
    if "mesh" not in n or i not in world:
        continue
    w = world[i]
    # Rough world scale, for turning local bounds into world extents.
    scale = [math.sqrt(w[0] ** 2 + w[4] ** 2 + w[8] ** 2),
             math.sqrt(w[1] ** 2 + w[5] ** 2 + w[9] ** 2),
             math.sqrt(w[2] ** 2 + w[6] ** 2 + w[10] ** 2)]
    for prim in meshes[n["mesh"]].get("primitives", []):
        attrs = prim.get("attributes", {})
        if "POSITION" not in attrs:
            continue
        prim_count += 1
        sets = tuple(sorted(k for k in attrs if k.startswith("TEXCOORD")))
        uv_sets[sets] += 1

        uv0 = attrs.get("TEXCOORD_0")
        if uv0 is not None:
            a = accessors[uv0]
            if "min" in a and "max" in a:
                lo, hi = a["min"], a["max"]
                if min(lo) < -0.001 or max(hi) > 1.001:
                    tiling += 1
                else:
                    inside_unit += 1
            else:
                no_uv_range += 1

        pos = accessors[attrs["POSITION"]]
        verts.append(pos.get("count", 0))
        if "indices" in prim:
            tris.append(accessors[prim["indices"]].get("count", 0) // 3)
        if "min" in pos and "max" in pos:
            e = [(pos["max"][k] - pos["min"][k]) * scale[k] for k in range(3)]
            e.sort()
            # Surface area of the box, which over-counts a flat panel by 2x and
            # under-counts a crumpled mesh. Good enough to size a lightmap.
            areas.append(2.0 * (e[0] * e[1] + e[0] * e[2] + e[1] * e[2]))

print("primitives:", prim_count)
print()
print("UV sets present:")
for sets, n in uv_sets.most_common():
    print("  %-28s %d" % (",".join(s.replace("TEXCOORD_", "uv") for s in sets)
                          or "(none)", n))
print()
print("TEXCOORD_0 range:")
print("  stays inside [0,1]:      %d" % inside_unit)
print("  escapes it (tiles):      %d" % tiling)
print("  no min/max authored:     %d" % no_uv_range)

verts.sort()
tris.sort()
print()
print("vertices per primitive: min %d median %d mean %.0f max %d  (total %d)"
      % (verts[0], verts[len(verts) // 2], statistics.fmean(verts), verts[-1],
         sum(verts)))
print("triangles per primitive: median %d  (total %d)"
      % (tris[len(tris) // 2], sum(tris)))

total_area = sum(areas)
print()
print("approx world surface area: %.0f m^2" % total_area)
for texels_per_m in (4, 8, 16, 32):
    # One R8 channel per texel, plus the padding a real packer needs.
    bytes_needed = total_area * texels_per_m * texels_per_m
    print("  lightmap at %2d texels/m: %7.1f MB (R8), %.0f texels"
          % (texels_per_m, bytes_needed / 1048576.0,
             total_area * texels_per_m * texels_per_m))
print()
print("per-vertex instead: %d vertices x 1 byte = %.2f MB"
      % (sum(verts), sum(verts) / 1048576.0))
print("vertex density: %.2f vertices per m^2 of surface"
      % (sum(verts) / total_area if total_area else 0.0))
