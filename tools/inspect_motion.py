"""Which nodes actually move, and what are they? Read-only JSON parse."""
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    g = json.load(handle)

nodes = g.get("nodes", [])
lights = g.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])

parent = {}
for i, n in enumerate(nodes):
    for c in n.get("children", []):
        parent[c] = i

animated = {}
for ai, a in enumerate(g.get("animations", [])):
    for ch in a.get("channels", []):
        t = ch.get("target", {}).get("node")
        if t is not None:
            animated.setdefault(t, []).append(
                (a.get("name", "anim%d" % ai), ch.get("target", {}).get("path")))

print("directly animated nodes:")
for i, chans in sorted(animated.items()):
    n = nodes[i]
    kind = []
    if "mesh" in n:
        kind.append("mesh")
    ext = n.get("extensions", {}).get("KHR_lights_punctual")
    if ext is not None:
        light = lights[ext.get("light", 0)]
        kind.append("light:" + light.get("type", "?"))
    if "children" in n:
        kind.append("%d children" % len(n["children"]))
    print("  node %-5d %-28s %-22s %s"
          % (i, n.get("name", "(unnamed)")[:28], ",".join(kind) or "transform-only",
             sorted(set(chans))))


def descendants(i):
    out = []
    stack = list(nodes[i].get("children", []))
    while stack:
        j = stack.pop()
        out.append(j)
        stack.extend(nodes[j].get("children", []))
    return out


print()
print("what each animated node drags with it:")
for i in sorted(animated):
    kids = descendants(i)
    meshes = [k for k in kids if "mesh" in nodes[k]]
    lit = [k for k in kids
           if "KHR_lights_punctual" in nodes[k].get("extensions", {})]
    own_mesh = 1 if "mesh" in nodes[i] else 0
    own_light = 1 if "KHR_lights_punctual" in nodes[i].get("extensions", {}) else 0
    print("  node %-5d -> %d mesh node(s), %d light(s)"
          % (i, len(meshes) + own_mesh, len(lit) + own_light))
