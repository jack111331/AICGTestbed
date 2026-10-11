load("@rules_cc//cc:defs.bzl", "cc_library")

# Debug information for RenderDoc.
#
# -Zi             emit debug info: line tables, original variable names, and the
#                 HLSL source text itself. Without it RenderDoc has only DXIL
#                 disassembly -- no source view and no shader debugging.
# -Qembed_debug   keep that info inside the shader container rather than in a
#                 side PDB, so RenderDoc finds it with no PDB search path
#                 configured anywhere. -Zi alone embeds it too but warns;
#                 this is the explicit, warning-free spelling.
# -Od             only in -c dbg. Optimised DXIL reorders instructions and folds
#                 locals away, so stepping through it does not line up with the
#                 source and many variables read as "optimised out". It costs
#                 shader performance, which is why it is not on everywhere.
#
# Reflection data is deliberately left in (no -Qstrip_reflect) -- RenderDoc uses
# it to name constant buffer fields and resource bindings.
#
# The cost is container size: about 8 KB -> 39 KB per shader for no_texture.fx,
# which lands in the generated C header. To keep non-debug builds lean instead,
# move "-Zi -Qembed_debug" into the dbg branch only.
_DXC_DEBUG_FLAGS = select({
    "//:dbg_build": "-Zi -Qembed_debug -Od",
    "//conditions:default": "-Zi -Qembed_debug",
})

_DXC_PATH = "\"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.26100.0\\x64\\dxc.exe\""

def hlsl_shader(name, src, entries, profile = "6_0", out = None):
    """Compiles one or more vertex/pixel shader pairs out of a single HLSL file.

    Deferred shading needs two pairs from no_texture.fx -- the G-buffer write
    and the lighting resolve -- so the rule takes a dict rather than the single
    `entry` it used to. One genrule still produces one header, because the
    byte arrays are only useful together and splitting them would mean a
    cc_library per pair.

    Args:
      name:    target name; also the genrule that runs dxc.
      src:     the .fx file, relative to Engine/.
      entries: {entry suffix: C symbol prefix}. "GBuffer": "g_gbuffer" compiles
               VSGBuffer and PSGBuffer into g_gbuffer_vertex_shader_bytecode and
               g_gbuffer_pixel_shader_bytecode. Starlark dicts keep insertion
               order, so the generated header is stable across builds.
      profile: shader model, without the vs_/ps_ prefix.
      out:     generated header name; defaults to <name>.h.
    """
    if not out:
        out = name + ".h"
    if not entries:
        fail("hlsl_shader(%s): `entries` must name at least one VS/PS pair" % name)

    src_path = "Engine/" + src

    # Every dxc invocation writes its own header next to the output and they are
    # concatenated in order. The first one creates the file; the rest append.
    # Each invocation needs the debug flags, so the command is assembled by
    # concatenation rather than .format(): _DXC_DEBUG_FLAGS is a select() and a
    # select cannot be interpolated into a format string.
    # : > truncates, so a rebuild into a stale output directory does not append
    # to the previous run's header.
    #
    # Assembled by repeated `+` rather than " && ".join(): _DXC_DEBUG_FLAGS is a
    # select(), and a select can be concatenated with strings but cannot be an
    # element of a list that join() walks.
    cmd = ": > $(OUTS)"
    for entry, symbol in entries.items():
        for stage, stage_entry, stage_symbol in (
            ("vs_", "VS" + entry, symbol + "_vertex_shader_bytecode"),
            ("ps_", "PS" + entry, symbol + "_pixel_shader_bytecode"),
        ):
            part = "$(OUTS)." + stage_entry
            cmd = cmd + (
                " && " + _DXC_PATH + " -T " + stage + profile + " -E " + stage_entry + " "
            ) + _DXC_DEBUG_FLAGS + (
                " -Fh " + part + " -Vn " + stage_symbol + " " + src_path +
                " && cat " + part + " >> $(OUTS)"
            )

    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = cmd,
        visibility = ["//visibility:private"],
    )

    cc_library(
        # Named for the file rather than the target, because every consumer
        # says #include <no_texture.h> and expects one library to depend on.
        name = "shader_bytecode",
        hdrs = [":" + name],
        # Expose the Bazel output directory to the preprocessor include path
        includes = ["."],
        visibility = ["//visibility:public"],
    )
