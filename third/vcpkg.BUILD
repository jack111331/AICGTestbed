# BUILD file for the whole vcpkg x64-windows tree produced by `vcpkg install`
# from third/vcpkg.json -- currently directxtk12 (+ its directx-headers and
# directxmath deps) and imgui.
#
# The repository root is third/vcpkg_installed/x64-windows, so every path below
# is relative to that directory:
#
#   include/                 DirectXMath headers (root), directx/, directxtk12/,
#                            dxguids/, imgui
#   lib/, bin/               release (/MD) libs + DLLs
#   debug/lib/, debug/bin/   debug (/MDd) libs + DLLs
#
# Dependencies that are NOT vcpkg packages live elsewhere: fastgltf in
# third/fastgltf.BUILD (a local source checkout) and DirectML in
# third/BUILD.bazel (it needs nothing but the Windows SDK).

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

# Bazel's MSVC toolchain compiles with /MDd in -c dbg and /MD otherwise, so the
# matching vcpkg CRT variant has to be picked to avoid runtime mismatches.
config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
)

cc_import(
    name = "directxtk12_prebuilt",
    interface_library = select({
        ":dbg_build": "debug/lib/DirectXTK12.lib",
        "//conditions:default": "lib/DirectXTK12.lib",
    }),
    shared_library = select({
        ":dbg_build": "debug/bin/DirectXTK12.dll",
        "//conditions:default": "bin/DirectXTK12.dll",
    }),
)

cc_library(
    name = "directxtk12",
    hdrs = glob(
        [
            "include/**/*.h",
            "include/**/*.hpp",
            "include/**/*.inl",
        ],
        # fastgltf and simdjson are no longer consumed from vcpkg -- they come
        # from the @fastgltf source checkout. A vcpkg_installed tree that has not
        # been re-synced still holds their old headers, and because this target
        # puts include/ on the search path (ahead of @fastgltf's, since include
        # path order follows dep order) those stale copies would silently shadow
        # the modified ones. Never claim them here.
        #
        # imgui and stb share this include/ directory but are owned by :imgui and
        # :stb_headers below; each package declaring only its own headers keeps
        # that ownership legible.
        exclude = [
            "include/fastgltf/**",
            "include/imconfig.h",
            "include/imgui*.h",
            "include/imstb_*.h",
            "include/simdjson.h",
            "include/stb_*.h",
            # onnx and its protobuf/abseil dependencies, owned by :onnx below.
            "include/absl/**",
            "include/google/**",
            "include/onnx/**",
            "include/utf8_range.h",
            "include/utf8_validity.h",
        ],
    ),
    # Mirrors DirectXTK12-targets.cmake:
    #   INTERFACE_COMPILE_DEFINITIONS "DIRECTX_TOOLKIT_IMPORT;USING_WINDOWS_GAMING_INPUT"
    # DIRECTX_TOOLKIT_IMPORT is required because vcpkg builds the toolkit as a
    # DLL -- without it the headers declare symbols without __declspec(dllimport).
    defines = [
        "DIRECTX_TOOLKIT_IMPORT",
        "USING_WINDOWS_GAMING_INPUT",
    ],
    # "include" gives <DirectXMath.h>, <directx/...>, <dxguids/...>;
    # "include/directxtk12" matches the toolkit's own
    # INTERFACE_INCLUDE_DIRECTORIES so <SpriteBatch.h> etc. resolve.
    includes = [
        "include",
        "include/directxtk12",
    ],
    linkopts = [
        "-DEFAULTLIB:d3d12.lib",
        "-DEFAULTLIB:dxgi.lib",
        "-DEFAULTLIB:dxguid.lib",
        "-DEFAULTLIB:uuid.lib",
        # WICTextureLoader / ScreenGrab
        "-DEFAULTLIB:windowscodecs.lib",
        "-DEFAULTLIB:ole32.lib",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    deps = [":directxtk12_prebuilt"],
)

# Static libs from the directx-headers port. Consumers of the DLL normally do
# NOT need these (the toolkit's public headers fall back to the Windows SDK
# because USING_DIRECTX_HEADERS is not defined), but they are here for code that
# wants the Agility-SDK style <directx/d3dx12.h> helpers.
cc_import(
    name = "directx_headers_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/DirectX-Headers.lib",
        "//conditions:default": "lib/DirectX-Headers.lib",
    }),
)

cc_import(
    name = "directx_guids_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/DirectX-Guids.lib",
        "//conditions:default": "lib/DirectX-Guids.lib",
    }),
)

cc_library(
    name = "directx_headers",
    hdrs = glob([
        "include/directx/**/*.h",
        "include/directx/**/*.hpp",
        "include/dxguids/**/*.h",
        "include/wsl/**/*.h",
    ]),
    defines = ["USING_DIRECTX_HEADERS"],
    includes = ["include"],
    target_compatible_with = ["@platforms//os:windows"],
    deps = [
        ":directx_guids_prebuilt",
        ":directx_headers_prebuilt",
    ],
)

# ---------------------------------------------------------------------------
# Dear ImGui, with the DX12 and Win32 backends.
#
# The port runs vcpkg_check_linkage(ONLY_STATIC_LIBRARY), so this is a static
# library on every triplet -- one static_library, no DLL to stage and no
# dllimport define to match. The port's CMakeLists sets CMAKE_DEBUG_POSTFIX "d",
# hence imguid.lib on the debug side, and passes IMGUI_SKIP_HEADERS=ON for the
# debug build, so the headers exist only under the release include/.
#
# imgui_impl_dx12.cpp and imgui_impl_win32.cpp are compiled into the library by
# the dx12-binding and win32-binding features, so there are no backend sources
# to add here -- just include <imgui_impl_dx12.h> / <imgui_impl_win32.h>.
# ---------------------------------------------------------------------------

cc_import(
    name = "imgui_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/imguid.lib",
        "//conditions:default": "lib/imgui.lib",
    }),
)

cc_library(
    name = "imgui",
    hdrs = glob([
        "include/imconfig.h",
        "include/imgui*.h",
        "include/imstb_*.h",
    ]),
    includes = ["include"],
    linkopts = [
        # D3D12SerializeRootSignature, used by the DX12 backend.
        "-DEFAULTLIB:d3d12.lib",
        "-DEFAULTLIB:dxgi.lib",
        # DwmIsCompositionEnabled / window handling in the Win32 backend.
        "-DEFAULTLIB:dwmapi.lib",
        "-DEFAULTLIB:user32.lib",
    ],
    target_compatible_with = ["@platforms//os:windows"],
    deps = [":imgui_prebuilt"],
)

# ---------------------------------------------------------------------------
# ONNX -- the protobuf schema and checker only, NOT a runtime.
#
# This is what lets Engine/OnnxModel.cc read a .onnx file; the graph is then
# translated into DirectML expressions by hand, so model work still records
# into the renderer's command list. ONNX Runtime would own execution instead
# and could not do that.
#
# The port builds protobuf and abseil as DLLs, which is why protobuf needs
# PROTOBUF_USE_DLLS (without it the headers declare symbols without
# __declspec(dllimport) and the link fails on protobuf globals) and why both
# appear as cc_import with a shared_library, so Bazel stages the DLLs next to
# the executable. onnx itself is static.
#
# ONNX_ML=1 is not optional here: the port installs only onnx-ml.pb.h, and
# onnx_pb.h includes onnx.pb.h without it. Both defines mirror
# share/onnx/ONNXTargets.cmake's INTERFACE_COMPILE_DEFINITIONS.
# ---------------------------------------------------------------------------

cc_import(
    name = "protobuf_prebuilt",
    interface_library = select({
        ":dbg_build": "debug/lib/libprotobufd.lib",
        "//conditions:default": "lib/libprotobuf.lib",
    }),
    shared_library = select({
        ":dbg_build": "debug/bin/libprotobufd.dll",
        "//conditions:default": "bin/libprotobuf.dll",
    }),
)

cc_import(
    name = "abseil_prebuilt",
    interface_library = select({
        ":dbg_build": "debug/lib/abseil_dll.lib",
        "//conditions:default": "lib/abseil_dll.lib",
    }),
    shared_library = select({
        ":dbg_build": "debug/bin/abseil_dll.dll",
        "//conditions:default": "bin/abseil_dll.dll",
    }),
)

cc_import(
    name = "onnx_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/onnx.lib",
        "//conditions:default": "lib/onnx.lib",
    }),
)

cc_import(
    name = "onnx_proto_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/onnx_proto.lib",
        "//conditions:default": "lib/onnx_proto.lib",
    }),
)

cc_import(
    name = "utf8_range_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/utf8_range.lib",
        "//conditions:default": "lib/utf8_range.lib",
    }),
)

cc_import(
    name = "utf8_validity_prebuilt",
    static_library = select({
        ":dbg_build": "debug/lib/utf8_validity.lib",
        "//conditions:default": "lib/utf8_validity.lib",
    }),
)

cc_library(
    name = "onnx",
    hdrs = glob([
        "include/onnx/**/*.h",
        "include/google/**/*.h",
        "include/google/**/*.inc",
        "include/absl/**/*.h",
        "include/absl/**/*.inc",
        "include/utf8_range.h",
        "include/utf8_validity.h",
    ]),
    defines = [
        "ONNX_NAMESPACE=onnx",
        "ONNX_ML=1",
        "PROTOBUF_USE_DLLS",
    ],
    includes = ["include"],
    target_compatible_with = ["@platforms//os:windows"],
    deps = [
        ":abseil_prebuilt",
        ":onnx_prebuilt",
        ":onnx_proto_prebuilt",
        ":protobuf_prebuilt",
        ":utf8_range_prebuilt",
        ":utf8_validity_prebuilt",
    ],
)

# ---------------------------------------------------------------------------
# stb -- headers only.
#
# The port just copies the repo's *.h into include/, so there is nothing to link.
# stb_image.h is both the interface and the implementation: exactly one
# translation unit in the whole binary must define STB_IMAGE_IMPLEMENTATION.
# That TU is //third:stb (third/stb_impl.cc), which is what callers depend on --
# this target only publishes the headers.
# ---------------------------------------------------------------------------

cc_library(
    name = "stb_headers",
    hdrs = glob(["include/stb_*.h"]),
    includes = ["include"],
    target_compatible_with = ["@platforms//os:windows"],
)
