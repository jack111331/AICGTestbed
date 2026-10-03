load("@rules_cc//cc:defs.bzl", "cc_library")
def hlsl_shader(name, src, entry, profile = "6_0", out = None):
    if not out:
        out = name + ".h"
        
    vs_profile = "vs_" + profile
    ps_profile = "ps_" + profile

    vs_entry = "VS" + entry
    ps_entry = "PS" + entry

    dxc_path = "\"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.26100.0\\x64\\dxc.exe\""

    # TODO Temporary append "Engine/" to source, need to figure out how to switch context in cmd 
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = "{} -T {} -E {} -Fh $(OUTS) -Vn g_vertex_shader_bytecode {} && {} -T {} -E {} -Fh $(OUTS)temp -Vn g_pixel_shader_bytecode {} && cat $(OUTS)temp >> $(OUTS)".format(
            dxc_path, vs_profile, vs_entry, "Engine/" + src,
            dxc_path, ps_profile, ps_entry, "Engine/" + src,
            
        ),
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