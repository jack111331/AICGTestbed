"""Stages a FloodDiffusion ONNX export into resources/FloodDiffusion/.

    python tools/prepare_flooddiffusion.py C:/Code/FloodDiffusion/onnx_models/tiny

Takes the directory written by FloodDiffusion's own export_onnx.py and produces
the layout Engine/FloodDiffusion.cc expects:

    denoiser.onnx             copied (or hard-linked) as-is
    vae_decoder_first.onnx
    vae_decoder_step.onnx
    text_encoder.onnx         optional; skipped with --skip-text-encoder
    tokenizer.bin             tokenizer.json baked flat, see pack_tokenizer
    pipeline.txt              config.json plus shapes read from the models

Two things are converted rather than copied:

* tokenizer.json is 16 MB of JSON holding a 256k-entry Unigram vocabulary.
  Parsing that at startup would mean adding a JSON library to the engine and
  spending a second on it every launch, so it is baked into a flat table the
  loader reads in one go.
* config.json is rewritten as `key value` lines for the same reason -- it is
  the only other JSON in the pipeline, and a dozen scalars do not justify a
  dependency.

The pipeline shapes that config.json does NOT record (the text feature width,
the VAE cache count and each cache shape) are read out of the .onnx files
themselves with tools/onnx_signature.py, so they cannot drift from the models.
"""
import argparse
import json
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_signature

TOKENIZER_MAGIC = b"PTFDTOK1"
TOKENIZER_VERSION = 1

# Pre-tokenizer behaviour the engine implements. Baked into the header so the
# loader refuses a tokenizer whose configuration it would silently mis-apply.
FLAG_WHITESPACE_SPLIT = 1 << 0
FLAG_METASPACE_PREPEND = 1 << 1
FLAG_BYTE_FALLBACK = 1 << 2

METASPACE = "\u2581"

MODELS = ("denoiser.onnx", "vae_decoder_first.onnx", "vae_decoder_step.onnx")


def pack_tokenizer(tokenizer_json, out_path):
    """tokenizer.json -> flat table. Returns a summary dict.

    Layout, all little-endian:
        magic "PTFDTOK1", u32 version, u32 vocab_count, u32 unk_id, u32 eos_id,
        u32 max_token_bytes, f32 min_score, u32 string_bytes, u32 flags
        vocab_count * (u32 offset, u32 length, f32 score)
        string_bytes of UTF-8 token text
    Offsets are relative to the start of the string block.
    """
    with open(tokenizer_json, encoding="utf-8") as handle:
        spec = json.load(handle)

    model = spec["model"]
    if model.get("type") != "Unigram":
        raise SystemExit("tokenizer model is %r; only Unigram is implemented"
                         % model.get("type"))
    if spec.get("normalizer") is not None:
        raise SystemExit("tokenizer declares a normalizer, which the engine "
                         "does not apply: %r" % spec["normalizer"].get("type"))

    flags = 0
    if model.get("byte_fallback"):
        flags |= FLAG_BYTE_FALLBACK
    pre = spec.get("pre_tokenizer") or {}
    stages = pre.get("pretokenizers", [pre]) if pre else []
    for entry in stages:
        kind = entry.get("type")
        if kind == "WhitespaceSplit":
            flags |= FLAG_WHITESPACE_SPLIT
        elif kind == "Metaspace":
            if entry.get("replacement") != METASPACE:
                raise SystemExit("Metaspace replacement is %r, expected U+2581"
                                 % entry.get("replacement"))
            if entry.get("prepend_scheme") != "always":
                raise SystemExit("Metaspace prepend_scheme is %r, expected "
                                 "always" % entry.get("prepend_scheme"))
            flags |= FLAG_METASPACE_PREPEND
        else:
            raise SystemExit("unhandled pre-tokenizer stage %r" % kind)

    # The end-of-sequence token the TemplateProcessing post-processor appends.
    post = spec.get("post_processor") or {}
    eos_id = None
    for entry in post.get("single", []):
        token = entry.get("SpecialToken")
        if token is not None:
            eos_id = post["special_tokens"][token["id"]]["ids"][0]
    if eos_id is None:
        raise SystemExit("post-processor appends no special token; the engine "
                         "expects an end-of-sequence id")

    vocab = model["vocab"]
    blob = bytearray()
    entries = bytearray()
    max_token_bytes = 0
    min_score = 0.0
    for token, score in vocab:
        encoded = token.encode("utf-8")
        entries += struct.pack("<IIf", len(blob), len(encoded), score)
        blob += encoded
        max_token_bytes = max(max_token_bytes, len(encoded))
        min_score = min(min_score, score)

    header = TOKENIZER_MAGIC + struct.pack(
        "<IIIIIfII", TOKENIZER_VERSION, len(vocab), model["unk_id"], eos_id,
        max_token_bytes, min_score, len(blob), flags)
    with open(out_path, "wb") as handle:
        handle.write(header)
        handle.write(entries)
        handle.write(blob)

    return {"vocab": len(vocab), "unk_id": model["unk_id"], "eos_id": eos_id,
            "max_token_bytes": max_token_bytes, "min_score": min_score,
            "flags": flags, "bytes": len(header) + len(entries) + len(blob)}


def dim(entry, index, what):
    """A fixed dimension, or a failure naming what was symbolic instead."""
    value = entry["shape"][index]
    if not isinstance(value, int):
        raise SystemExit("%s: dimension %d of %r is dynamic (%r); the engine "
                         "needs it fixed" % (what, index, entry["name"], value))
    return value


def stage(src, dst, link):
    """Hard-link or copy one file, replacing whatever is at dst."""
    if os.path.exists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass  # different volume, or no permission; fall back to a copy
    shutil.copyfile(src, dst)
    return "copied"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="an onnx_models/<name> directory from "
                                       "FloodDiffusion's export_onnx.py")
    parser.add_argument("--out", default="resources/FloodDiffusion",
                        help="destination, relative to the repository root")
    parser.add_argument("--link", action="store_true",
                        help="hard-link the .onnx files instead of copying "
                             "them; saves 1.2 GB when both are on one volume")
    parser.add_argument("--skip-text-encoder", action="store_true",
                        help="leave out text_encoder.onnx; the pipeline then "
                             "loads but cannot encode a prompt")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    config_path = os.path.join(source, "config.json")
    if not os.path.exists(config_path):
        raise SystemExit("%s has no config.json; point this at the directory "
                         "export_onnx.py wrote" % source)
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    # --- models -------------------------------------------------------------
    wanted = list(MODELS)
    if not args.skip_text_encoder:
        wanted.append("text_encoder.onnx")
    for name in wanted:
        src = os.path.join(source, name)
        if not os.path.exists(src):
            raise SystemExit("%s is missing from %s" % (name, source))
        how = stage(src, os.path.join(out, name), args.link)
        size = os.path.getsize(src) / (1024 * 1024)
        print("  %-24s %8.1f MB  %s" % (name, size, how))
        # A model over the 2 GB protobuf limit keeps its weights alongside.
        external = src + ".data"
        if os.path.exists(external):
            how = stage(external, os.path.join(out, name + ".data"), args.link)
            size = os.path.getsize(external) / (1024 * 1024)
            print("  %-24s %8.1f MB  %s" % (name + ".data", size, how))

    # --- tokenizer ----------------------------------------------------------
    tokenizer_json = os.path.join(source, "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        raise SystemExit("%s is missing" % tokenizer_json)
    summary = pack_tokenizer(tokenizer_json, os.path.join(out, "tokenizer.bin"))
    print("  %-24s %8.1f MB  baked (%d tokens, unk %d, eos %d)"
          % ("tokenizer.bin", summary["bytes"] / (1024 * 1024), summary["vocab"],
             summary["unk_id"], summary["eos_id"]))

    # --- shapes read back out of the models ---------------------------------
    denoiser = onnx_signature.signature(os.path.join(source, "denoiser.onnx"))
    by_name = {entry["name"]: entry for entry in denoiser["inputs"]}
    for required in ("x", "t", "context", "context_lens", "text_idx"):
        if required not in by_name:
            raise SystemExit("denoiser.onnx has no input %r; this export does "
                             "not match what the engine drives" % required)
    text_dim = dim(by_name["context"], 3, "denoiser.onnx")
    latent_dim = dim(by_name["x"], 2, "denoiser.onnx")

    step = onnx_signature.signature(
        os.path.join(source, "vae_decoder_step.onnx"))
    caches = [e for e in step["inputs"] if e["name"].startswith("cache_in_")]
    caches.sort(key=lambda e: int(e["name"].rsplit("_", 1)[1]))
    frames_per_step = dim(step["outputs"][0], 1, "vae_decoder_step.onnx")
    motion_dim = dim(step["outputs"][0], 2, "vae_decoder_step.onnx")

    first = onnx_signature.signature(
        os.path.join(source, "vae_decoder_first.onnx"))
    frames_first = dim(first["outputs"][0], 1, "vae_decoder_first.onnx")

    if latent_dim != config["latent_dim"]:
        raise SystemExit("config.json says latent_dim %d but denoiser.onnx "
                         "says %d" % (config["latent_dim"], latent_dim))
    if motion_dim != config["motion_dim"]:
        raise SystemExit("config.json says motion_dim %d but "
                         "vae_decoder_step.onnx says %d"
                         % (config["motion_dim"], motion_dim))
    if len(caches) != config["vae_num_caches"]:
        raise SystemExit("config.json says %d VAE caches but "
                         "vae_decoder_step.onnx takes %d"
                         % (config["vae_num_caches"], len(caches)))

    # HumanML3D's 263-dim feature vector, which is what the VAE decodes to:
    #   1 root yaw velocity + 2 root XZ velocity + 1 root height
    #   + (J-1)*3 joint positions relative to the root
    #   + (J-1)*6 joint rotations + J*3 local velocities + 4 foot contacts
    # Solving 263 = 4 + 9*(J-1) + 3*J + 4 gives J = 22.
    joints = (motion_dim - 8 + 9) // 12
    if 4 + 9 * (joints - 1) + 3 * joints + 4 != motion_dim:
        raise SystemExit("motion_dim %d is not a HumanML3D feature width; the "
                         "engine joint recovery would read past the ric "
                         "block" % motion_dim)

    lines = [
        ("chunk_size", config["chunk_size"]),
        ("noise_steps", config["noise_steps"]),
        ("cfg_scale", repr(float(config["cfg_scale"]))),
        ("prediction_type", config["prediction_type"]),
        ("text_len", config["text_len"]),
        ("text_clean", config["text_clean"] or "none"),
        ("latent_dim", latent_dim),
        ("motion_dim", motion_dim),
        ("text_dim", text_dim),
        ("joints", joints),
        ("vae_num_caches", len(caches)),
        ("vae_upsample_factor", config["vae_upsample_factor"]),
        ("vae_frames_first", frames_first),
        ("vae_frames_step", frames_per_step),
        ("has_text_encoder", 0 if args.skip_text_encoder else 1),
        ("source_config", config.get("source_config", "unknown")),
    ]
    for entry in caches:
        lines.append(("vae_cache_shape",
                      "%s %d %d" % (entry["name"],
                                    dim(entry, 1, "vae_decoder_step.onnx"),
                                    dim(entry, 2, "vae_decoder_step.onnx"))))

    pipeline = os.path.join(out, "pipeline.txt")
    with open(pipeline, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Written by tools/prepare_flooddiffusion.py from\n")
        handle.write("# %s\n" % source.replace("\\", "/"))
        handle.write("# Shapes are read out of the .onnx files, not copied "
                     "from config.json.\n")
        for key, value in lines:
            handle.write("%s %s\n" % (key, value))

    print("  %-24s %8s     %d keys" % ("pipeline.txt", "", len(lines)))
    print("\n%s is ready: %d joints, %d-dim latents, %d-dim motion, "
          "%d VAE caches" % (args.out, joints, latent_dim, motion_dim,
                             len(caches)))
    print("chunk %d, %d denoise steps, cfg %.1f, prediction %s"
          % (config["chunk_size"], config["noise_steps"],
             float(config["cfg_scale"]), config["prediction_type"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
