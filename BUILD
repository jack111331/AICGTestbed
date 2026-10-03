# Shared sample assets, mirroring the Media/ layout at the root of
# microsoft/Xbox-ATG-Samples. DX::FindMediaFile (Engine/FindMedia.h) looks for
# exactly these subdirectory names, walking up from the executable, so keeping
# them at the workspace root means they resolve from the Bazel execroot without
# depending on runfiles symlinks (which are off by default on Windows).
filegroup(
    name = "media",
    srcs = glob(["Media/**"]),
    visibility = ["//visibility:public"],
)
