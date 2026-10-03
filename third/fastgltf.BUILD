# BUILD file for a local fastgltf source checkout (C:/Code/fastgltf).
#
# Unlike the vcpkg packages in @vcpkg_directxtk12, this one is compiled from
# source by Bazel rather than consumed as a prebuilt DLL, so local edits to the
# checkout rebuild automatically with no CMake or vcpkg step in between.
#
# Mirrors the upstream CMakeLists.txt: the same three translation units, the
# same public include dir, and the same PUBLIC compile definitions.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# The amalgamated simdjson that fastgltf's cmake/dependencies.cmake downloads
# into deps/ (4.6.2). Built as a static library here, so simdjson.h must NOT see
# SIMDJSON_USING_WINDOWS_DYNAMIC_LIBRARY -- static linkage is its default.
cc_library(
    name = "simdjson",
    srcs = ["deps/simdjson/simdjson.cpp"],
    hdrs = ["deps/simdjson/simdjson.h"],
    copts = select({
        "@platforms//os:windows": [
            "/EHsc",
            "/utf-8",
        ],
        "//conditions:default": [],
    }),
    defines = ["SIMDJSON_THREADS_ENABLED=1"],
    includes = ["deps/simdjson"],
)

cc_library(
    name = "fastgltf",
    srcs = [
        "src/base64.cpp",
        "src/fastgltf.cpp",
        "src/io.cpp",
    ],
    hdrs = glob(["include/fastgltf/*.hpp"]),
    # fastgltf_compiler_flags() in cmake/compilers.cmake. /utf-8 matters: the
    # sources contain UTF-8 string literals, and MSVC otherwise reads them in
    # the system codepage.
    copts = select({
        "@platforms//os:windows": [
            "/EHsc",
            "/utf-8",
        ],
        "//conditions:default": [],
    }),
    # PUBLIC target_compile_definitions from CMakeLists.txt; every option
    # defaults to OFF, so each $<BOOL:...> is 0. They change public struct
    # layouts, so anything linking this must see the same values -- hence
    # `defines` rather than `local_defines`.
    defines = [
        "FASTGLTF_USE_CUSTOM_SMALLVECTOR=0",
        "FASTGLTF_DISABLE_CUSTOM_MEMORY_POOL=0",
        "FASTGLTF_USE_64BIT_FLOAT=0",
        "FASTGLTF_ENABLE_KHR_IMPLICIT_SHAPES=0",
        "FASTGLTF_ENABLE_KHR_PHYSICS_RIGID_BODIES=0",
    ],
    includes = ["include"],
    # PRIVATE in CMakeLists.txt: only fastgltf's own TUs need it.
    local_defines = ['SIMDJSON_TARGET_VERSION=\\"4.6.2\\"'],
    deps = [":simdjson"],
)
