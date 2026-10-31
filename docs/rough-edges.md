# Rough edges

Known problems, listed so they do not surprise you. Back to the
[README](../README.md).

Known and worth fixing, listed so they do not surprise you:

- **`shaders.bzl` hardcodes an absolute `dxc.exe` path** under Windows SDK
  `10.0.26100.0`, so a machine with a different SDK build installed needs that
  string edited. It also prepends `"Engine/"` to shader sources to work around
  the genrule's working directory (there is a `TODO` in the file), which means
  the rule only works for shaders under `Engine/`.
- **The shader genrule needs `bash`** — it uses `&&` and `cat`. Bazel on Windows
  invokes MSYS2 bash (`c:\msys64\usr\bin\bash.exe`) for `genrule`. It now runs
  `dxc` twice per entry pair and concatenates the generated headers, so a file
  with more pairs costs more genrule steps but still produces one header.
- **`@vcpkg_directxtk12` is named after its first package** but now covers the
  whole vcpkg tree including imgui and stb. Renaming it would touch every
  `@vcpkg_directxtk12//:...` label.
