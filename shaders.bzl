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

def hlsl_shader(name, src, entry, profile = "6_0", out = None):
    if not out:
        out = name + ".h"
        
    vs_profile = "vs_" + profile
    ps_profile = "ps_" + profile

    vs_entry = "VS" + entry
    ps_entry = "PS" + entry

    dxc_path = "\"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.26100.0\\x64\\dxc.exe\""

    src_path = "Engine/" + src

    # Each dxc invocation needs the debug flags, so the command is built by
    # concatenation rather than .format(): _DXC_DEBUG_FLAGS is a select() and a
    # select cannot be interpolated into a format string.
    cmd = (
        dxc_path + " -T " + vs_profile + " -E " + vs_entry + " "
    ) + _DXC_DEBUG_FLAGS + (
        " -Fh $(OUTS) -Vn g_vertex_shader_bytecode " + src_path + " && " +
        dxc_path + " -T " + ps_profile + " -E " + ps_entry + " "
    ) + _DXC_DEBUG_FLAGS + (
        " -Fh $(OUTS)temp -Vn g_pixel_shader_bytecode " + src_path +
        " && cat $(OUTS)temp >> $(OUTS)"
    )

    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = cmd,
        visibility = ["//visibility:private"],
    )

    cc_library(
        # name = name + "_bytecode",
        name = "shader_bytecode",
        hdrs = [":" + name],
        # Expose the Bazel output directory to the preprocessor include path
        includes = ["."], 
        visibility = ["//visibility:public"],
    )