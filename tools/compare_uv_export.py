"""Does a Blender UV export still correspond to the asset it came from?

The renderer keeps the ORIGINAL asset authoritative and takes only the extra UV
set from a Blender export (tools/blender_unwrap.py), so the two have to line up
well enough to move UVs from one to the other. Unwrapping splits vertices at
seams, so the export legitimately has MORE vertices -- what has to survive is
the FACE order, because that is what a harvest can key on:

    original corner (face f, corner c) -> original index buffer[f * 3 + c]
    export   corner (face f, corner c) -> export   index buffer[f * 3 + c]

If those two corners are the same point in space for every corner of every
face, the export's per-corner UV can be attached to the original's vertex, and
corners of one original vertex that disagree are a seam -- exactly the split a
lightmap needs.

Matching meshes is the part that is easy to get wrong, and getting it wrong
looks exactly like the exporter having mangled the asset:

  * not by NAME -- glTF does not require mesh names to be unique and this asset
    does not make them so; the Bistro has many meshes called "subset_0";
  * not by LIST POSITION -- an import/export is free to reorder;
  * not by BOUNDING BOX either -- an architectural scene repeats props, so many
    meshes share a shape and picking the first of them pairs the wrong one.

So this buckets by face count, which is an integer and cannot drift, and then
PROVES each pairing by walking the corners. Distances are compared against a
tolerance scaled to the mesh, because a round trip trades float32 for double
and rotates Y-up to Z-up and back.

Usage:

    python tools/compare_uv_export.py <original.gltf> <unwrapped.gltf> [samples]

Read-only.
"""
import json
import os
import struct
import sys

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
COMPONENT_FORMAT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}


class Asset:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            self.g = json.load(handle)
        self.base = os.path.dirname(os.path.abspath(path))
        self._buffers = {}
        self._cache = {}

    def buffer(self, index):
        if index in self._buffers:
            return self._buffers[index]
        uri = self.g["buffers"][index].get("uri", "")
        if uri.startswith("data:"):
            import base64
            raw = base64.b64decode(uri.split(",", 1)[1])
        else:
            resolved = os.path.normpath(os.path.join(self.base, uri))
            if not (resolved == self.base or resolved.startswith(self.base + os.sep)):
                raise SystemExit("buffer URI escapes the asset directory: %s" % uri)
            with open(resolved, "rb") as handle:
                raw = handle.read()
        self._buffers[index] = raw
        return raw

    def read(self, index):
        """An accessor as a flat tuple, read in bulk where it is tightly packed."""
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
            # The common case, and worth special-casing: one unpack of the whole
            # block instead of one per element.
            out = struct.unpack_from("<%d%s" % (count * components, fmt), raw, start)
        else:
            values = []
            for i in range(count):
                values.extend(struct.unpack_from("<%d%s" % (components, fmt), raw,
                                                 start + i * stride))
            out = tuple(values)
        self._cache[index] = out
        return out


def primitives(asset):
    out = []
    for mi, mesh in enumerate(asset.g.get("meshes", [])):
        for pi, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs or "indices" not in prim:
                continue
            pos = asset.g["accessors"][attrs["POSITION"]]
            idx = asset.g["accessors"][prim["indices"]]
            out.append({
                "mesh": mi,
                "prim": pi,
                "name": mesh.get("name", ""),
                "position": attrs["POSITION"],
                "indices": prim["indices"],
                "uv1": attrs.get("TEXCOORD_1"),
                "vertices": pos["count"],
                "faces": idx["count"] // 3,
                "min": tuple(pos.get("min", (0.0, 0.0, 0.0))),
                "max": tuple(pos.get("max", (0.0, 0.0, 0.0))),
            })
    return out


def corner_distance(original, a, unwrapped, b, limit=None):
    """Worst distance between corresponding face corners, or None if the shapes
    make the comparison meaningless. `limit` stops the walk once exceeded,
    which is what makes searching a bucket of candidates affordable."""
    if a["faces"] != b["faces"]:
        return None
    pos_a = original.read(a["position"])
    idx_a = original.read(a["indices"])
    pos_b = unwrapped.read(b["position"])
    idx_b = unwrapped.read(b["indices"])
    if len(idx_a) != len(idx_b):
        return None
    worst = 0.0
    for corner in range(len(idx_a)):
        va = idx_a[corner] * 3
        vb = idx_b[corner] * 3
        if va + 2 >= len(pos_a) or vb + 2 >= len(pos_b):
            return None
        d = max(abs(pos_a[va + k] - pos_b[vb + k]) for k in range(3))
        if d > worst:
            worst = d
            if limit is not None and worst > limit:
                return worst
    return worst


def tolerance_for(p):
    extent = max(hi - lo for hi, lo in zip(p["max"], p["min"]))
    return max(extent * 1e-4, 1e-6)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    original = Asset(sys.argv[1])
    unwrapped = Asset(sys.argv[2])
    sample_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    pa = primitives(original)
    pb = primitives(unwrapped)
    print("original:  %d primitive(s) with positions and indices" % len(pa))
    print("unwrapped: %d primitive(s) with positions and indices" % len(pb))
    with_uv1 = sum(1 for p in pb if p["uv1"] is not None)
    print("export primitives carrying TEXCOORD_1: %d of %d" % (with_uv1, len(pb)))
    print()

    buckets = {}
    for p in pb:
        buckets.setdefault(p["faces"], []).append(p)

    step = max(1, len(pa) // sample_limit)
    sampled = pa[::step][:sample_limit]
    print("searching for a corresponding export mesh for %d sampled primitive(s),"
          % len(sampled))
    print("bucketed by face count, then verified corner by corner")
    print()

    matched_first = 0
    matched_after_search = 0
    no_candidate = 0
    none_matched = 0
    split_meshes = 0
    identical_meshes = 0
    worst_matched = 0.0
    failures = []

    for a in sampled:
        candidates = buckets.get(a["faces"], [])
        if not candidates:
            no_candidate += 1
            failures.append((a, "no export mesh has %d faces" % a["faces"]))
            continue
        tolerance = tolerance_for(a)
        found = None
        attempts = 0
        best = None
        for b in candidates:
            attempts += 1
            d = corner_distance(original, a, unwrapped, b, limit=tolerance)
            if d is None:
                continue
            if best is None or d < best:
                best = d
            if d <= tolerance:
                found = b
                break
        if found is None:
            none_matched += 1
            failures.append((a, "%d candidate(s), best corner distance %s"
                             % (len(candidates),
                                "n/a" if best is None else "%.3e" % best)))
            continue
        if attempts == 1:
            matched_first += 1
        else:
            matched_after_search += 1
        worst_matched = max(worst_matched, best if best is not None else 0.0)
        if found["vertices"] > a["vertices"]:
            split_meshes += 1
        elif found["vertices"] == a["vertices"]:
            identical_meshes += 1

    total = len(sampled)
    print("  matched on the first candidate: %d" % matched_first)
    print("  matched after searching:        %d" % matched_after_search)
    print("  candidates existed, none fit:   %d" % none_matched)
    print("  no candidate at all:            %d" % no_candidate)
    print("  worst distance among matches:   %.3e" % worst_matched)
    print()
    print("  of the matches, the export split vertices: %d" % split_meshes)
    print("  ...and kept the vertex count exactly:      %d" % identical_meshes)
    if failures:
        print()
        print("  first few that did not match:")
        for a, why in failures[:6]:
            print("    mesh %-4d %-22s %6d faces, %6d verts -- %s"
                  % (a["mesh"], a["name"][:22], a["faces"], a["vertices"], why))
    print()

    if none_matched == 0 and no_candidate == 0:
        print("VERDICT: viable. Every sampled mesh has an export counterpart whose")
        print("         faces and corners line up, so a harvest can read the")
        print("         export's per-corner UV and attach it to the original's")
        print("         vertex, splitting where two corners of one vertex")
        print("         disagree. The harvest has to do this same search rather")
        print("         than trust list order.")
    else:
        print("VERDICT: %d of %d sampled meshes have no counterpart that lines up."
              % (none_matched + no_candidate, total))
        print("         Harvesting those would put the wrong UVs on them, so they")
        print("         would have to fall back to the shadow map.")


if __name__ == "__main__":
    main()
