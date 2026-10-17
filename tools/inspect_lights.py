"""Inventory a glTF asset's KHR_lights_punctual lights and whether anything is
animated. Read-only; the asset is untrusted data, so this only parses JSON."""
import collections
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    g = json.load(handle)

lights = g.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
print("lights declared:", len(lights))

kinds = collections.Counter(l.get("type", "?") for l in lights)
for kind, n in sorted(kinds.items()):
    print("  %-12s %d" % (kind, n))

# Ranges: glTF says an absent range means unlimited.
print()
for kind in sorted(kinds):
    ranged = [l.get("range") for l in lights if l.get("type") == kind]
    have = [r for r in ranged if r is not None]
    print("%-12s range given on %d of %d" % (kind, len(have), len(ranged)))
    if have:
        have.sort()
        print("             min %.2f  median %.2f  max %.2f"
              % (have[0], have[len(have) // 2], have[-1]))
    intens = [l.get("intensity", 1.0) for l in lights if l.get("type") == kind]
    intens.sort()
    if intens:
        print("             intensity min %.1f median %.1f max %.1f"
              % (intens[0], intens[len(intens) // 2], intens[-1]))
    if kind == "spot":
        outer = [l.get("spot", {}).get("outerConeAngle", math.pi / 4.0)
                 for l in lights if l.get("type") == "spot"]
        outer.sort()
        print("             outer cone deg min %.1f median %.1f max %.1f"
              % tuple(math.degrees(o) for o in
                      (outer[0], outer[len(outer) // 2], outer[-1])))

# Which nodes reference a light, and do any of them have a parent chain that
# an animation touches?
nodes = g.get("nodes", [])
light_nodes = [i for i, n in enumerate(nodes)
               if "KHR_lights_punctual" in n.get("extensions", {})]
print()
print("nodes referencing a light:", len(light_nodes))

animations = g.get("animations", [])
print("animations:", len(animations))
animated = set()
for a in animations:
    for ch in a.get("channels", []):
        t = ch.get("target", {}).get("node")
        if t is not None:
            animated.add(t)
print("directly animated nodes:", len(animated))

# Propagate: a node moves if it or any ancestor is animated.
parent = {}
for i, n in enumerate(nodes):
    for c in n.get("children", []):
        parent[c] = i


def moves(i):
    seen = set()
    while i is not None and i not in seen:
        if i in animated:
            return True
        seen.add(i)
        i = parent.get(i)
    return False


moving_lights = [i for i in light_nodes if moves(i)]
mesh_nodes = [i for i, n in enumerate(nodes) if "mesh" in n]
moving_meshes = [i for i in mesh_nodes if moves(i)]
skinned = [i for i, n in enumerate(nodes) if "skin" in n]
print("light nodes that move:", len(moving_lights))
print("mesh nodes:", len(mesh_nodes), "of which move:", len(moving_meshes),
      "skinned:", len(skinned))
