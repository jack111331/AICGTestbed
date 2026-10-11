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

# Two entry pairs: the G-buffer write and the deferred lighting resolve. Both
# live in one .fx file so they share the PBR helpers, the light struct and the
# material constants -- see the register note in no_texture.fx for why their
# cbuffers do not overlap.
hlsl_shader(
    name = "no_texture",
    src = "shaders/no_texture.fx",
    entries = {
        "GBuffer": "g_gbuffer",
        "Lighting": "g_deferred_lighting",
    },
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
        "//third:directml",
    ],
)

# Page-based descriptor management, ported from LearningDirectX12 v0.0.3 (MIT).
# See Descriptors/LICENSE-LearningDirectX12.txt for the upstream licence and the
# header comment in each file for what was adapted.
#
# includes = ["."] puts Engine/ on the search path so these files can say
# #include "pch.h" and #include "Descriptors/...".
cc_library(
    name = "Descriptors",
    srcs = glob(["Descriptors/*.cc"]),
    hdrs = glob(["Descriptors/*.hpp"]),
    defines = SAMPLE_DEFINES,
    includes = ["."],
    deps = [":pch"],
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
    name = "Camera",
    srcs = ["Camera.cc"],
    hdrs = ["Camera.hpp"],
    defines = SAMPLE_DEFINES,
    # Camera.hpp has no includes of its own -- it relies on pch.h having been
    # read first for the DirectXMath/SimpleMath types.
    deps = [":pch"],
)

# Retargeting HumanML3D motion onto a rigged glTF character: a port of the
# KeeMap Blender addon that momask-codes' README uses, driven by the bone
# mapping tools/prepare_retarget.py converts from momask's assets/mapping.json.
cc_library(
    name = "GLTFAdapter",
    srcs = [
        "GLTFAdapter.cc",
        # Retargeting poses the SceneGraph, so it lives with it. SceneGraph has
        # no target of its own to depend on.
        "MotionRetarget.cc",
        "SceneGraph.cc",
    ],
    hdrs = [
        "GLTFAdapter.hpp",
        "MotionRetarget.hpp",
        "SceneGraph.hpp",
    ],
    defines = SAMPLE_DEFINES,
    # GLTFAdapter.hpp includes DeviceResources.hpp and holds
    # ComPtr<ID3D12Resource> members; GLTFAdapter.cc includes pch.h.
    deps = [
        ":Camera",
        ":DeviceResources",
        ":Descriptors",
        ":GBuffer",
        ":pch",
        "@fastgltf//:fastgltf",
        "//third:stb",
        "@vcpkg_directxtk12//:imgui",
        ":shader_bytecode"
    ],
)

# The G-buffer layout. Its own target because three places have to agree on it:
# the shader that writes it, GLTFAdapter's PSO, and DeferredRenderer's targets.
cc_library(
    name = "GBuffer",
    hdrs = ["GBuffer.hpp"],
    defines = SAMPLE_DEFINES,
    includes = ["."],
    deps = [":pch"],
)

# The render graph: passes declare their resource accesses, the graph places the
# barriers and records the frame. No D3D12 device is needed to compile a plan,
# which is what //Engine:rendergraph-smoke exercises.
cc_library(
    name = "RenderGraph",
    srcs = ["RenderGraph.cc"],
    hdrs = ["RenderGraph.hpp"],
    defines = SAMPLE_DEFINES,
    includes = ["."],
    deps = [
        ":Descriptors",
        ":pch",
    ],
)

cc_binary(
    name = "rendergraph-smoke",
    srcs = ["rendergraph-smoke.cc"],
    defines = SAMPLE_DEFINES,
    deps = [
        ":RenderGraph",
        ":pch",
    ],
)

# Deferred shading on top of the render graph: the G-buffer targets, the
# lighting pipeline and the two passes. Depends on :GLTFAdapter only for
# ShaderLight out of SceneGraph.hpp -- the geometry itself arrives as a
# callback, so this does not know what it is drawing.
cc_library(
    name = "DeferredRenderer",
    srcs = ["DeferredRenderer.cc"],
    hdrs = ["DeferredRenderer.hpp"],
    defines = SAMPLE_DEFINES,
    includes = ["."],
    deps = [
        ":Descriptors",
        ":DeviceResources",
        ":GBuffer",
        ":GLTFAdapter",
        ":RenderGraph",
        ":pch",
        ":shader_bytecode",
        "@vcpkg_directxtk12//:directxtk12",
        "@vcpkg_directxtk12//:imgui",
    ],
)

cc_binary(
    name = "modelmanager-smoke",
    srcs = ["modelmanager-smoke.cc"],
    deps = [
        ":DeviceResources",
        ":ModelManager",
        ":pch",
        "@vcpkg_directxtk12//:onnx",
    ],
)

cc_binary(
    name = "ortdml-smoke",
    srcs = ["ortdml-smoke.cc"],
    deps = [
        ":pch",
        "//third:onnxruntime",
        "@vcpkg_directxtk12//:onnx",
    ],
)

# Measures which FloodDiffusion input shapes should be fixed at export time,
# using ORT free dimension overrides so no re-export is needed to find out.
cc_binary(
    name = "fd-probe",
    srcs = ["fd-probe.cc"],
    deps = ["//third:onnxruntime"],
)

cc_binary(
    name = "onnx-smoke",
    srcs = ["onnx-smoke.cc"],
    deps = ["@vcpkg_directxtk12//:onnx"],
)

cc_library(
    name = "OrtEnv",
    srcs = ["OrtEnv.cc"],
    hdrs = ["OrtEnv.hpp"],
    defines = SAMPLE_DEFINES,
    deps = [
        ":pch",
        "//third:onnxruntime",
    ],
)

# The FloodDiffusion text-to-motion pipeline: a Unigram tokenizer, the four ONNX
# sessions, the diffusion-forcing sampler and HumanML3D joint recovery.
#
# MotionFeatures and Tokenizer carry no D3D12 or ONNX Runtime dependency on
# purpose, so //Engine:motionfeatures-smoke can exercise the parts that are
# pure arithmetic without a device.
cc_library(
    name = "FloodDiffusion",
    srcs = [
        "FloodDiffusion.cc",
        "MotionFeatures.cc",
        "Tokenizer.cc",
    ],
    hdrs = [
        "FloodDiffusion.hpp",
        "MotionFeatures.hpp",
        "Tokenizer.hpp",
    ],
    defines = SAMPLE_DEFINES,
    deps = [
        ":OrtEnv",
        ":pch",
        "//third:onnxruntime",
    ],
)

cc_binary(
    name = "motionfeatures-smoke",
    srcs = ["motionfeatures-smoke.cc"],
    deps = [":FloodDiffusion"],
)

cc_binary(
    name = "flooddiffusion-smoke",
    srcs = ["flooddiffusion-smoke.cc"],
    data = ["//:media"],
    deps = [":FloodDiffusion"],
)

cc_library(
    name = "ModelManager",
    srcs = [
        "ModelManager.cc",
        "OnnxModel.cc",
        "OrtModel.cc",
    ],
    hdrs = [
        "DirectMLX.h",
        "ModelManager.hpp",
        "OnnxModel.hpp",
        "OrtModel.hpp",
    ],
    defines = SAMPLE_DEFINES,
    deps = [
        ":DeviceResources",
        ":Descriptors",
        ":OrtEnv",
        ":pch",
        # The ONNX protobuf schema, for OnnxModel.cc. Only the schema -- the
        # graph is translated to DirectML here rather than run by ONNX Runtime.
        "@vcpkg_directxtk12//:onnx",
        # ShowImgui lists the registered models.
        "@vcpkg_directxtk12//:imgui",
        # ONNX Runtime with the DirectML EP, for the OnnxRuntime backend.
        "//third:onnxruntime",
    ],
)

cc_library(
    name = "Application",
    srcs = [
        "Application.cc",
        # The view over the FloodDiffusion pipeline: playback, the skeleton
        # lines and the UI. Here rather than in :FloodDiffusion because it is
        # the only part that needs ImGui and DirectXTK12's PrimitiveBatch.
        "MotionStreamView.cc",
    ],
    hdrs = [
        "Application.hpp",
        "FindMedia.h",
        "MotionStreamView.hpp",
    ],
    defines = SAMPLE_DEFINES,
    deps = [
        ":Camera",
        ":DeferredRenderer",
        ":Descriptors",
        ":DeviceResources",
        ":FloodDiffusion",
        ":GLTFAdapter",
        ":ModelManager",
        ":RenderGraph",
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

cc_binary(
    name = "scenegraph-smoke",
    srcs = ["scenegraph-smoke.cc"],
    defines = SAMPLE_DEFINES,
    deps = [
        ":GLTFAdapter",
        ":pch",
    ],
)
