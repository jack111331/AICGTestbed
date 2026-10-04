# Shared sample assets, mirroring the Media/ layout at the root of
# microsoft/Xbox-ATG-Samples. DX::FindMediaFile (Engine/FindMedia.h) looks for
# exactly these subdirectory names, walking up from the executable, so keeping
# them at the workspace root means they resolve from the Bazel execroot without
# depending on runfiles symlinks (which are off by default on Windows).
# Lets //:shaders.bzl drop shader optimisation in debug builds. The vcpkg repo
# declares its own copy of this; repos created by use_repo_rule cannot see
# targets in the root module, so the two cannot be shared.
config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "media",
    srcs = glob(["Media/**"]),
    visibility = ["//visibility:public"],
)
