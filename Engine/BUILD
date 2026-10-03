load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# The upstream .vcxproj sets these; Bazel's MSVC toolchain does not. Without
# UNICODE the MAKEINTRESOURCE macros expand to their ANSI form and calls like
# LoadCursorW(nullptr, IDC_ARROW) fail to compile (C2664).
SAMPLE_DEFINES = [
    "UNICODE",
    "_UNICODE",
]

# Runtime assets live in //:media (workspace-root Media/), because
# DX::FindMediaFile resolves them by walking up from the executable.

# pch.h pulls in <directx/d3d12.h>, <directx/d3dx12.h> and the whole DirectXTK12
# header set, so every target that includes it needs those include paths and the
# USING_DIRECTX_HEADERS define. Owning that here means a new target only has to
# depend on :pch, instead of rediscovering pch.h's transitive needs.
load("//:shaders.bzl", "hlsl_shader")

hlsl_shader(
    name = "no_texture",
    src = "shaders/no_texture.fx",
    entry = "Straight",
    profile = "6_0", # Shader Model 6.0 for DX12
    out = "no_texture.h"
)


# genrule(
#     name = "compile_shader",
#     srcs = ["shaders/no_texture.fx"],
#     outs = ["no_texture.h"],
#     # /T: Shader model target (e.g., ps_6_0 for modern D3D12 DXC, or ps_5_0 for FXC)
#     # /E: Entry point function name
#     # /Fh: Output path for the generated C/C++ header array
#     # /Vn: Variable name used for the byte array in the generated header
#     cmd = "\"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\x64\\dxc.exe\" " +
#     "-T vs_6_0 -E VSStraight -Fh $(OUTS) -Vn g_vertex_shader_bytecode Engine/shaders/no_texture.fx &&" +
#     "\"C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\x64\\dxc.exe\" " +
#     "-T ps_6_0 -E PSStraight -Fh $(OUTS)temp -Vn g_pixel_shader_bytecode Engine/shaders/no_texture.fx && cat $(OUTS)temp >> $(OUTS)",
#     visibility = ["//visibility:private"],
# )

# # 2. Wrap the generated header into a cc_library so other C++ code can include it
# cc_library(
#     name = "shader_bytecode",
#     hdrs = [":compile_shader"],
#     # Expose the Bazel output directory to the preprocessor include path
#     includes = ["."], 
#     visibility = ["//visibility:public"],
# )

cc_library(
    name = "pch",
    hdrs = ["pch.h"],
    defines = SAMPLE_DEFINES,
    deps = [
        "@vcpkg_directxtk12//:directx_headers",
        "@vcpkg_directxtk12//:directxtk12",
    ],
)

cc_library(
    name = "DeviceResources",
    srcs = ["DeviceResources.cc"],
    hdrs = [
        "DeviceResources.hpp",
        "StepTimer.hpp",
    ],
    defines = SAMPLE_DEFINES,
    deps = [
        ":pch",
        "@vcpkg_directxtk12//:directxtk12",
    ],
)

cc_library(
    name = "GLTFAdapter",
    srcs = ["GLTFAdapter.cc"],
    hdrs = [
        "GLTFAdapter.hpp",
    ],
    defines = SAMPLE_DEFINES,
    # GLTFAdapter.hpp includes DeviceResources.hpp and holds
    # ComPtr<ID3D12Resource> members; GLTFAdapter.cc includes pch.h.
    deps = [
        ":DeviceResources",
        ":pch",
        "@fastgltf//:fastgltf",
        "//third:stb",
        ":shader_bytecode"
    ],
)

cc_library(
    name = "Application",
    srcs = ["Application.cc", "Camera.cc"],
    hdrs = [
        "Application.hpp",
        "FindMedia.h",
        "Camera.hpp"
    ],
    defines = SAMPLE_DEFINES,
    deps = [
        ":DeviceResources",
        ":GLTFAdapter",
        ":pch",
        "@vcpkg_directxtk12//:directxtk12",
        "@vcpkg_directxtk12//:imgui"
    ],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    data = ["//:media"],
    # A Win32 GUI app: WinMain, not main. runtimeobject.lib is needed for the
    # RoInitializeWrapper in Main.cpp.
    linkopts = [
        "-SUBSYSTEM:WINDOWS",
        "-ENTRY:wWinMainCRTStartup",
        "-DEFAULTLIB:runtimeobject.lib",
        "-DEFAULTLIB:user32.lib",
        "-DEFAULTLIB:gdi32.lib",
        "-DEFAULTLIB:shell32.lib",

    ],
    defines = SAMPLE_DEFINES,
    deps = [
        ":Application",
        ":pch",
        "@vcpkg_directxtk12//:directxtk12",
    ],
)

cc_binary(
    name = "directxtk12-smoke",
    srcs = ["directxtk12-smoke.cc"],
    deps = ["@vcpkg_directxtk12//:directxtk12"],
)

cc_binary(
    name = "fastgltf-smoke",
    srcs = ["fastgltf-smoke.cc"],
    deps = ["@fastgltf//:fastgltf"],
)

cc_binary(
    name = "imgui-smoke",
    srcs = ["imgui-smoke.cc"],
    deps = ["@vcpkg_directxtk12//:imgui"],
)

cc_binary(
    name = "directml-smoke",
    srcs = ["directml-smoke.cc"],
    deps = ["//third:directml"],
)

cc_binary(
    name = "stb-smoke",
    srcs = ["stb-smoke.cc"],
    deps = ["//third:stb"],
)
