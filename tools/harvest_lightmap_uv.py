"""Merge a Blender UV export's lightmap UVs back into the original glTF.

Blender unwraps well but a full glTF round trip through it does not just ADD
UVs -- it rewrites materials (the Bistro's KHR_materials_pbrSpecularGlossiness
becomes metallic-roughness) and drops extension references like
MSFT_texture_dds. So tools/blender_unwrap.py exports geometry only, and this
puts the one thing worth keeping -- the second UV set -- back into the original,
leaving everything else exactly as the asset had it.

The output is a complete glTF the renderer can load directly. Nothing in the
engine needs to know Blender was involved: it reads TEXCOORD_1, which is now
simply there.

    python tools/harvest_lightmap_uv.py <original.gltf> <unwrapped.gltf> <out.gltf>

HOW MESHES ARE MATCHED, because this is the part that is easy to get wrong and
whose failure looks like a mangled export rather than a bad matcher:

  * not by name -- glTF does not require unique mesh names and this asset does
    not make them so; the Bistro has many meshes called "subset_0";
  * not by list position -- an import/export may reorder;
  * not by bounding box -- an architectural scene repeats props, so many meshes
    share a shape and the first candidate is often the wrong one.

Instead: bucket by face count, then confirm a candidate by walking its corners.
Verified by tools/compare_uv_export.py, which found 20 of 60 sampled meshes
needed that search rather than the first candidate.

HOW FACES ARE MATCHED. Blender discards degenerate faces on import -- measured
644 of the Bistro's 1,753,630, concentrated in 6 of 551 primitives. So the
export's face order is NOT always the original's, and faces are therefore
matched on their CORNER POSITIONS rather than their index. That recovers those
6 instead of letting them fall back to the shadow map.

The vertex correspondence falls out of the face match: for a matched pair of
faces, export corner c belongs to the original vertex at the original face's
corner c. Export vertices are already split at seams -- that is what unwrapping
does -- so each one maps to exactly one original vertex, and the other
attributes are gathered through that mapping.
"""
import collections
import json
import os
import struct
import sys

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
COMPONENT_FORMAT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}

# The attributes carried over to the rebuilt vertex buffer. Everything else a
# primitive might have -- JOINTS_0, WEIGHTS_0 -- belongs to skinned geometry,
# which is never baked, so those primitives are left untouched entirely.
CARRIED = [
    ("POSITION", "VEC3", 3),
    ("NORMAL", "VEC3", 3),
    ("TANGENT", "VEC4", 4),
    ("TEXCOORD_0", "VEC2", 2),
]


class Asset:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            self.g = json.load(handle)
        self.path = os.path.abspath(path)
        self.base = os.path.dirname(self.path)
        self._buffers = {}
        self._cache = {}

    def buffer(self, index):
        if index not in self._buffers:
            uri = self.g["buffers"][index].get("uri", "")
            if uri.startswith("data:"):
                import base64
                raw = base64.b64decode(uri.split(",", 1)[1])
            else:
                resolved = os.path.normpath(os.path.join(self.base, uri))
                if not (resolved == self.base or
                        resolved.startswith(self.base + os.sep)):
                    raise SystemExit("buffer URI escapes its directory: %s" % uri)
                with open(resolved, "rb") as handle:
                    raw = handle.read()
            self._buffers[index] = raw
        return self._buffers[index]

    def read(self, index):
        if index in self._cache:
            return self._cache[index]
        a = self.g["accessors"][index]
        components = COMPONENTS[a["type"]]
        fmt = COMPONENT_FORMAT[a["componentType"]]
        size = COMPONENT_SIZE[a["componentType"]]
        count = a["count"]
        view = self.g["bufferViews"][a["bufferView"]]
        raw = self.buffer(view["buffer"])
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", components * size)
        if stride == components * size:
            out = struct.unpack_from("<%d%s" % (count * components, fmt), raw, start)
        else:
            values = []
            for i in range(count):
                values.extend(struct.unpack_from("<%d%s" % (components, fmt), raw,
                                                 start + i * stride))
            out = tuple(values)
        self._cache[index] = out
        return out

    def forget(self):
        # The gathered streams are what matter from here; the raw accessor
        # tuples for a finished mesh are not, and on a scene this size they add
        # up to more memory than the asset itself.
        self._cache.clear()


def describe(asset):
    out = []
    for mi, mesh in enumerate(asset.g.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs or "indices" not in prim:
                continue
            pos = asset.g["accessors"][attrs["POSITION"]]
            idx = asset.g["accessors"][prim["indices"]]
            out.append({
                "mesh": mi, "prim": pi, "name": mesh.get("name", ""),
                "attrs": attrs, "indices": prim["indices"],
                "vertices": pos["count"], "faces": idx["count"] // 3,
                "min": tuple(pos.get("min", (0.0,) * 3)),
                "max": tuple(pos.get("max", (0.0,) * 3)),
            })
    return out


def tolerance_for(p):
    extent = max(hi - lo for hi, lo in zip(p["max"], p["min"]))
    return max(extent * 1e-4, 1e-6)


def corners_line_up(original, a, unwrapped, b, tolerance):
    """Index-keyed corner walk, used only to CONFIRM a mesh pairing. Cheap and
    exact where Blender kept the face order, which is the common case."""
    if a["faces"] != b["faces"]:
        return False
    pa = original.read(a["attrs"]["POSITION"])
    ia = original.read(a["indices"])
    pb = unwrapped.read(b["attrs"]["POSITION"])
    ib = unwrapped.read(b["indices"])
    if len(ia) != len(ib):
        return False
    for c in range(len(ia)):
        va, vb = ia[c] * 3, ib[c] * 3
        if va + 2 >= len(pa) or vb + 2 >= len(pb):
            return False
        if max(abs(pa[va + k] - pb[vb + k]) for k in range(3)) > tolerance:
            return False
    return True


def quantise(x, y, z, scale):
    return (int(round(x * scale)), int(round(y * scale)), int(round(z * scale)))


def match_mesh(original, a, unwrapped, candidates):
    """The export primitive that is this original primitive, or None.

    Tries the index-keyed confirmation first; where the face counts differ --
    Blender having dropped degenerate faces -- falls back to comparing the
    sets of face corner positions, which does not depend on face order.
    """
    tolerance = tolerance_for(a)
    for b in candidates.get(a["faces"], []):
        if corners_line_up(original, a, unwrapped, b, tolerance):
            return b, "index"

    # Degenerate faces dropped: look among primitives with slightly fewer
    # faces, confirming on a sample of corner positions.
    pa = original.read(a["attrs"]["POSITION"])
    extent = max(hi - lo for hi, lo in zip(a["max"], a["min"]))
    scale = 1.0 / max(extent * 1e-4, 1e-6)
    want = set()
    for i in range(0, len(pa), 3):
        want.add(quantise(pa[i], pa[i + 1], pa[i + 2], scale))

    best = None
    for faces, group in candidates.items():
        if faces > a["faces"] or faces < a["faces"] * 0.9:
            continue
        for b in group:
            pb = unwrapped.read(b["attrs"]["POSITION"])
            hits = 0
            tried = 0
            # A sample is enough to identify a mesh; the full check is the face
            # matching that follows, which would fail loudly if this were wrong.
            for i in range(0, len(pb), max(3, (len(pb) // 300) * 3 or 3)):
                tried += 1
                if quantise(pb[i], pb[i + 1], pb[i + 2], scale) in want:
                    hits += 1
            if tried and hits == tried:
                if best is None or faces > best[0]["faces"]:
                    best = (b, "positions")
    return best if best else (None, None)


def build_face_map(positions, indices, scale):
    """Keyed by the face's three corner positions, order-independent."""
    out = {}
    for f in range(len(indices) // 3):
        key = tuple(sorted(
            quantise(positions[indices[f * 3 + c] * 3],
                     positions[indices[f * 3 + c] * 3 + 1],
                     positions[indices[f * 3 + c] * 3 + 2], scale)
            for c in range(3)))
        out.setdefault(key, []).append(f)
    return out


def harvest(original, a, unwrapped, b):
    """New streams for this primitive, or None if the faces cannot be paired.

    Returns (vertexCount, {attribute: flat list}, indices, uv1).
    """
    pa = original.read(a["attrs"]["POSITION"])
    ia = original.read(a["indices"])
    pb = unwrapped.read(b["attrs"]["POSITION"])
    ib = unwrapped.read(b["indices"])
    uv1_accessor = b["attrs"].get("TEXCOORD_1")
    if uv1_accessor is None:
        return None
    uvb = unwrapped.read(uv1_accessor)

    extent = max(hi - lo for hi, lo in zip(a["max"], a["min"]))
    scale = 1.0 / max(extent * 1e-4, 1e-6)

    # Export vertex -> original vertex. Filled from matched faces; a vertex
    # only reachable through a face Blender dropped stays unset.
    origin = [None] * (len(pb) // 3)
    faceCount = len(ib) // 3

    if a["faces"] == b["faces"] and corners_line_up(original, a, unwrapped, b,
                                                    tolerance_for(a)):
        # The common case: face order preserved, so the mapping is direct.
        for c in range(len(ib)):
            origin[ib[c]] = ia[c]
    else:
        # Faces were dropped, so pair them on their corner positions instead.
        originalFaces = build_face_map(pa, ia, scale)
        for f in range(faceCount):
            key = tuple(sorted(
                quantise(pb[ib[f * 3 + c] * 3], pb[ib[f * 3 + c] * 3 + 1],
                         pb[ib[f * 3 + c] * 3 + 2], scale)
                for c in range(3)))
            group = originalFaces.get(key)
            if not group:
                continue
            of = group[0]
            if len(group) > 1:
                group.pop(0)
            # Corner order within a face may differ, so pair each export corner
            # with the original corner at the same position.
            for c in range(3):
                ev = ib[f * 3 + c]
                target = quantise(pb[ev * 3], pb[ev * 3 + 1], pb[ev * 3 + 2], scale)
                for k in range(3):
                    ov = ia[of * 3 + k]
                    if quantise(pa[ov * 3], pa[ov * 3 + 1], pa[ov * 3 + 2],
                                scale) == target:
                        origin[ev] = ov
                        break

    unresolved = sum(1 for o in origin if o is None)
    if unresolved == len(origin):
        return None
    # An unresolved vertex is only referenced by faces that no longer exist;
    # pointing it at vertex 0 keeps the buffers well-formed and nothing samples
    # it. Counted by the caller.
    fallback = 0
    origin = [fallback if o is None else o for o in origin]

    streams = {}
    for name, _, n in CARRIED:
        accessor = a["attrs"].get(name)
        if accessor is None:
            continue
        source = original.read(accessor)
        gathered = []
        for o in origin:
            base = o * n
            gathered.extend(source[base:base + n])
        streams[name] = gathered

    uv1 = []
    for j in range(len(origin)):
        uv1.extend(uvb[j * 2:j * 2 + 2])

    return len(origin), streams, list(ib), uv1, unresolved


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    original = Asset(sys.argv[1])
    unwrapped = Asset(sys.argv[2])
    out_path = os.path.abspath(sys.argv[3])
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    pa = describe(original)
    pb = describe(unwrapped)
    candidates = collections.defaultdict(list)
    for p in pb:
        candidates[p["faces"]].append(p)
    print("original:  %d primitive(s)" % len(pa))
    print("unwrapped: %d primitive(s)" % len(pb))

    g = json.loads(json.dumps(original.g))  # a copy to rewrite

    blob = bytearray()
    new_views = []
    new_accessors = []

    def add(values, fmt, size, components, gltf_type, component_type,
            minmax=False):
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(struct.pack("<%d%s" % (len(values), fmt), *values))
        new_views.append({"buffer": None, "byteOffset": offset,
                          "byteLength": len(values) * size})
        accessor = {"bufferView": None, "componentType": component_type,
                    "count": len(values) // components, "type": gltf_type}
        if minmax:
            lo = [min(values[i::components]) for i in range(components)]
            hi = [max(values[i::components]) for i in range(components)]
            accessor["min"], accessor["max"] = lo, hi
        new_accessors.append((len(new_views) - 1, accessor))
        return len(new_accessors) - 1

    matched_index = 0
    matched_positions = 0
    unmatched = 0
    rewritten = 0
    unresolved_total = 0
    faces_before = 0
    faces_after = 0

    for a in pa:
        b, how = match_mesh(original, a, unwrapped, candidates)
        if b is None:
            unmatched += 1
            continue
        result = harvest(original, a, unwrapped, b)
        if result is None:
            unmatched += 1
            continue
        count, streams, indices, uv1, unresolved = result
        if how == "index":
            matched_index += 1
        else:
            matched_positions += 1
        unresolved_total += unresolved
        faces_before += a["faces"]
        faces_after += len(indices) // 3

        target = g["meshes"][a["mesh"]]["primitives"][a["prim"]]
        for name, gltf_type, n in CARRIED:
            if name not in streams:
                continue
            target["attributes"][name] = add(
                streams[name], "f", 4, n, gltf_type, 5126,
                minmax=(name == "POSITION"))
        target["attributes"]["TEXCOORD_1"] = add(uv1, "f", 4, 2, "VEC2", 5126)
        target["indices"] = add(indices, "I", 4, 1, "SCALAR", 5125)
        rewritten += 1
        original.forget()
        unwrapped.forget()
        if rewritten % 50 == 0:
            print("  rewritten %d/%d primitive(s), %.0f MB of new geometry"
                  % (rewritten, len(pa), len(blob) / 1048576.0))

    # The new geometry goes in its own buffer; the asset's own buffers are left
    # alone, because animations and skins still reference them.
    bin_name = os.path.splitext(os.path.basename(out_path))[0] + ".bin"
    buffer_index = len(g["buffers"])
    g["buffers"].append({"uri": bin_name, "byteLength": len(blob)})
    view_base = len(g["bufferViews"])
    for view in new_views:
        view["buffer"] = buffer_index
        g["bufferViews"].append(view)
    accessor_base = len(g["accessors"])
    for view_index, accessor in new_accessors:
        accessor["bufferView"] = view_base + view_index
        g["accessors"].append(accessor)

    # Accessor indices handed out above were local; shift them to the real ones.
    def shift(value):
        return accessor_base + value

    for a in pa:
        target = g["meshes"][a["mesh"]]["primitives"][a["prim"]]
        if "TEXCOORD_1" not in target["attributes"]:
            continue
        for name in [n for n, _, _ in CARRIED] + ["TEXCOORD_1"]:
            if name in target["attributes"] and \
                    target["attributes"][name] < len(new_accessors):
                target["attributes"][name] = shift(target["attributes"][name])
        if target["indices"] < len(new_accessors):
            target["indices"] = shift(target["indices"])

    # The output sits somewhere else, so every URI the asset already had has to
    # be re-pointed at where it still lives.
    def repoint(uri):
        if not uri or uri.startswith("data:") or uri == bin_name:
            return uri
        absolute = os.path.normpath(os.path.join(original.base, uri))
        return os.path.relpath(absolute, out_dir).replace(os.sep, "/")

    for i, buf in enumerate(g["buffers"]):
        if i != buffer_index and "uri" in buf:
            buf["uri"] = repoint(buf["uri"])
    for image in g.get("images", []):
        if "uri" in image:
            image["uri"] = repoint(image["uri"])

    with open(os.path.join(out_dir, bin_name), "wb") as handle:
        handle.write(blob)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(g, handle)

    print()
    print("matched by face order:      %d" % matched_index)
    print("matched by face positions:  %d" % matched_positions)
    print("not matched (left as-is):   %d" % unmatched)
    print("primitives rewritten:       %d of %d" % (rewritten, len(pa)))
    print("faces %d -> %d (%d dropped as degenerate by Blender)"
          % (faces_before, faces_after, faces_before - faces_after))
    if unresolved_total:
        print("vertices only reachable through dropped faces: %d" % unresolved_total)
    print()
    print("wrote %s (%.1f MB of new geometry in %s)"
          % (out_path, len(blob) / 1048576.0, bin_name))


if __name__ == "__main__":
    main()
