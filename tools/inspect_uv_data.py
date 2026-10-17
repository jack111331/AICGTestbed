"""Do this asset's TEXCOORD_0 values tile, and do primitives overlap in UV?

A lightmap is addressed by a UV set that is a non-overlapping atlas of the
surface. Tiled UVs (the usual choice for architectural texturing) address the
same texel from many places, so they cannot store a per-point result. This reads
the real TEXCOORD_0 data for a sample of primitives and reports their ranges.

Read-only; the asset is untrusted data, so this only reads bytes at the offsets
the JSON names and never executes anything from the asset's directory.
"""
import json
import os
import struct
import sys

path = sys.argv[1]
sample = int(sys.argv[2]) if len(sys.argv) > 2 else 60

with open(path, "r", encoding="utf-8") as handle:
    g = json.load(handle)

base = os.path.dirname(os.path.abspath(path))
accessors = g["accessors"]
views = g["bufferViews"]
buffers = g["buffers"]

# Map buffer index -> open file handle, resolving only plain relative URIs.
handles = {}


def buffer_bytes(index, offset, length):
    if index not in handles:
        uri = buffers[index].get("uri")
        if uri is None or uri.startswith("data:"):
            raise SystemExit("buffer %d is not an external file" % index)
        resolved = os.path.normpath(os.path.join(base, uri))
        if not resolved.startswith(base + os.sep):
            raise SystemExit("buffer URI escapes the asset directory")
        handles[index] = open(resolved, "rb")
    f = handles[index]
    f.seek(offset)
    return f.read(length)


COMPONENT = {5126: ("f", 4)}

meshes = g["meshes"]
prims = []
for mesh in meshes:
    for prim in mesh.get("primitives", []):
        uv = prim.get("attributes", {}).get("TEXCOORD_0")
        if uv is not None:
            prims.append(uv)

print("primitives with TEXCOORD_0:", len(prims))
step = max(1, len(prims) // sample)
checked = 0
tiling = 0
unit = 0
degenerate = 0
ranges = []

for uv_index in prims[::step][:sample]:
    a = accessors[uv_index]
    fmt, size = COMPONENT.get(a["componentType"], (None, None))
    if fmt is None or a["type"] != "VEC2":
        continue
    view = views[a["bufferView"]]
    stride = view.get("byteStride", 2 * size)
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    count = a["count"]
    raw = buffer_bytes(view["buffer"], start, stride * (count - 1) + 2 * size)
    us, vs = [], []
    for i in range(count):
        o = i * stride
        u, v = struct.unpack_from("<2f", raw, o)
        us.append(u)
        vs.append(v)
    lo = (min(us), min(vs))
    hi = (max(us), max(vs))
    checked += 1
    ranges.append((lo, hi))
    if hi[0] - lo[0] <= 1e-6 and hi[1] - lo[1] <= 1e-6:
        degenerate += 1
    elif lo[0] < -0.001 or lo[1] < -0.001 or hi[0] > 1.001 or hi[1] > 1.001:
        tiling += 1
    else:
        unit += 1

for f in handles.values():
    f.close()

print("sampled:", checked)
print("  UVs inside [0,1]:   %d" % unit)
print("  UVs outside [0,1]:  %d   <- tiled, unusable as lightmap UVs" % tiling)
print("  degenerate (all one point): %d" % degenerate)
print()
print("first 12 ranges:")
for lo, hi in ranges[:12]:
    print("  u %9.3f .. %9.3f   v %9.3f .. %9.3f" % (lo[0], hi[0], lo[1], hi[1]))
