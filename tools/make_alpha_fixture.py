# Builds a tiny glTF that exercises alpha coverage and face culling, with no
# textures at all -- every material relies on the renderer's 1x1 opaque white
# fallback, whose alpha is exactly 1.0. That makes the alpha cutoff the only
# variable: a MASK material with cutoff 0.9 must survive and one with cutoff
# 1.1 must be clipped away entirely.
#
# Each quad is identified in the readback by its metallicFactor, which PSGBuffer
# writes straight into the G-buffer material target's red channel (the
# metallic-roughness branch is skipped when the material has no such texture).
# That makes the check independent of where the quads land on screen.
import base64
import json
import struct

# Written where the app's PTFLIO_SCENE override can pick it up:
#
#     python tools/make_alpha_fixture.py
#     PTFLIO_SCENE=resources/alpha-test/alpha-test.gltf bazel-bin/Engine/hello-world.exe
#
# Six quads appear in a row. Reading left to right they must look like:
# visible, visible, MISSING, visible, visible, MISSING. The two missing ones
# are the point -- a MASK material whose cutoff the fallback alpha cannot
# beat, and a back-facing single-sided quad.
OUT = "resources/alpha-test/alpha-test.gltf"

# name, metallic (the readback tag), alphaMode, alphaCutoff, doubleSided,
# front-facing, what we expect
QUADS = [
    ("opaque",            0.10, "OPAQUE", 0.5, False, True,  "drawn"),
    ("mask-passes",       0.20, "MASK",   0.9, False, True,  "drawn"),
    ("mask-clipped",      0.30, "MASK",   1.1, False, True,  "clipped"),
    ("doublesided-back",  0.40, "OPAQUE", 0.5, True,  False, "drawn"),
    ("blend",             0.50, "BLEND",  0.5, False, True,  "drawn"),
    ("singlesided-back",  0.60, "OPAQUE", 0.5, False, False, "culled"),
]

# One quad's worth of geometry, reused by every mesh through its own accessors.
# Positions are per quad so they do not overlap on screen.
blob = bytearray()
accessors = []
buffer_views = []


def add_view(data, target=None):
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(data)
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
    if target is not None:
        view["target"] = target
    buffer_views.append(view)
    return len(buffer_views) - 1


def add_accessor(view, component_type, count, type_name, mins=None, maxs=None):
    accessor = {
        "bufferView": view,
        "componentType": component_type,
        "count": count,
        "type": type_name,
    }
    if mins is not None:
        accessor["min"] = mins
        accessor["max"] = maxs
    accessors.append(accessor)
    return len(accessors) - 1


FLOAT = 5126
USHORT = 5123

meshes = []
materials = []
nodes = []

half_width = 0.12
half_height = 0.2
spacing = 0.3
first_x = -spacing * (len(QUADS) - 1) / 2.0

for index, (name, metallic, mode, cutoff, double_sided, front, _) in enumerate(QUADS):
    centre_x = first_x + spacing * index
    x0, x1 = centre_x - half_width, centre_x + half_width
    y0, y1 = -half_height, half_height

    positions = [
        (x0, y0, 0.0),
        (x1, y0, 0.0),
        (x1, y1, 0.0),
        (x0, y1, 0.0),
    ]
    position_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    # The authored normal always faces the camera. A back-facing quad is made
    # back-facing by its winding, so PSGBuffer's SV_IsFrontFace flip is what
    # has to put its normal right -- which is the behaviour being exercised.
    normal_bytes = b"".join(struct.pack("<3f", 0.0, 0.0, 1.0) for _ in positions)
    uv_bytes = b"".join(struct.pack("<2f", u, v)
                        for u, v in [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])

    # Counter-clockwise seen from +Z is glTF's front face. Reversed for the
    # quads that must be culled by the single-sided pipeline.
    indices = [0, 1, 2, 0, 2, 3]
    if not front:
        indices = list(reversed(indices))
    index_bytes = struct.pack("<6H", *indices)

    position_accessor = add_accessor(
        add_view(position_bytes), FLOAT, 4, "VEC3",
        [x0, y0, 0.0], [x1, y1, 0.0])
    normal_accessor = add_accessor(add_view(normal_bytes), FLOAT, 4, "VEC3")
    uv_accessor = add_accessor(add_view(uv_bytes), FLOAT, 4, "VEC2")
    index_accessor = add_accessor(add_view(index_bytes), USHORT, 6, "SCALAR")

    material = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": metallic,
            "roughnessFactor": 0.5,
        },
        "alphaMode": mode,
        "doubleSided": bool(double_sided),
    }
    if mode == "MASK":
        material["alphaCutoff"] = cutoff
    materials.append(material)

    meshes.append({
        "name": name,
        "primitives": [{
            "attributes": {
                "POSITION": position_accessor,
                "NORMAL": normal_accessor,
                "TEXCOORD_0": uv_accessor,
            },
            "indices": index_accessor,
            "material": len(materials) - 1,
        }],
    })
    nodes.append({"name": name, "mesh": len(meshes) - 1})

# The app offsets the whole scene by GLTFAdapter::m_position, which defaults to
# (3, -2, -4). The root cancels it so the quads land in front of the default
# camera at (0, 0.7, 1.5) without the app needing to be touched.
root_children = list(range(len(nodes)))
nodes.append({
    "name": "cancel-scene-placement",
    "translation": [-3.0, 2.0, 4.0],
    "children": root_children,
})

gltf = {
    "asset": {"version": "2.0", "generator": "ptflio alpha/culling fixture"},
    "scene": 0,
    "scenes": [{"nodes": [len(nodes) - 1]}],
    "nodes": nodes,
    "meshes": meshes,
    "materials": materials,
    "accessors": accessors,
    "bufferViews": buffer_views,
    "buffers": [{
        "byteLength": len(blob),
        "uri": "data:application/octet-stream;base64," +
               base64.b64encode(bytes(blob)).decode("ascii"),
    }],
}

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(gltf, handle, indent=1)

print("wrote", OUT)
for name, metallic, mode, cutoff, double_sided, front, expect in QUADS:
    print("  metallic %.2f -> %-8s  %-18s %s%s" % (
        metallic, expect, name,
        "" if front else "back-facing ",
        "doubleSided" if double_sided else ""))
