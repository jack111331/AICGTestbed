"""What atlas does the Bistro need, under Engine/Lightmap.cc's exact sizing rule?

Mirrors LightmapChartEdge: content = round(sqrt(area) * texelsPerMetre), then
the chart is the next power of two at or above content + 2*padding, clamped to
[minChartEdge, maxChartEdge] as powers of two.

Read-only JSON parse of untrusted asset data.
"""
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    g = json.load(handle)

accessors, meshes, nodes = g["accessors"], g["meshes"], g["nodes"]


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

areas = []
for i, n in enumerate(nodes):
    if "mesh" not in n or i not in world:
        continue
    w = world[i]
    scale = [math.sqrt(w[0] ** 2 + w[4] ** 2 + w[8] ** 2),
             math.sqrt(w[1] ** 2 + w[5] ** 2 + w[9] ** 2),
             math.sqrt(w[2] ** 2 + w[6] ** 2 + w[10] ** 2)]
    for prim in meshes[n["mesh"]].get("primitives", []):
        pos_i = prim.get("attributes", {}).get("POSITION")
        if pos_i is None:
            continue
        pos = accessors[pos_i]
        if "min" not in pos:
            continue
        e = sorted((pos["max"][k] - pos["min"][k]) * scale[k] for k in range(3))
        areas.append(max(e[1] * e[2], 0.0))


def pow2_up(v):
    r = 1
    while r < v:
        r *= 2
    return r


def pow2_down(v):
    r = 1
    while r * 2 <= v:
        r *= 2
    return r


def chart_edge(area, tpm, padding=2, min_edge=4, max_edge=256):
    if area <= 0.0:
        return 0
    border = 2 * padding
    smallest = pow2_up(max(min_edge, border + 1))
    largest = max(smallest, pow2_down(max_edge))
    content = round(max(math.sqrt(area) * tpm, 1.0))
    return min(max(pow2_up(content + border), smallest), largest)


print("instances: %d, total face area %.0f m^2" % (len(areas), sum(areas)))
print()
print("%-6s %-12s %-12s %-10s %s" % ("tpm", "texels", "atlas needed",
                                     "R8 size", "capped charts"))
for tpm in (2, 4, 8, 16):
    for cap in (256, 512):
        edges = [chart_edge(a, tpm, max_edge=cap) for a in areas]
        texels = sum(e * e for e in edges)
        capped = sum(1 for a, e in zip(areas, edges)
                     if e == pow2_down(cap) and a > 0)
        # The atlas has to be at least this, and packing is not perfect; the
        # measured occupancy on a comparable mix was 55%.
        side = pow2_up(int(math.sqrt(texels / 0.55)) + 1)
        print("%-6d %-12d %-12s %-10s %d"
              % (tpm, texels, "%dx%d" % (side, side),
                 "%.1f MB" % (side * side / 1048576.0), capped))
    print()
