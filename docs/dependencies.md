# Dependencies

Where each third-party library comes from and which Bazel label names it. Back
to the [README](../README.md).

Four different mechanisms, because each library wants something different:

| Library | Source | Bazel label |
|---|---|---|
| DirectXTK12, DirectX-Headers, DirectXMath | vcpkg | `@vcpkg_directxtk12//:directxtk12`, `@vcpkg_directxtk12//:directx_headers` |
| Dear ImGui (+ DX12 and Win32 backends) | vcpkg | `@vcpkg_directxtk12//:imgui` |
| stb_image | vcpkg headers, one local implementation TU | `//third:stb` |
| DirectML | Windows SDK + in-box `directml.dll` | `//third:directml` |
| fastgltf (+ bundled simdjson) | sibling clone of [the `mylab-generative` fork](https://github.com/jack111331/fastgltf/tree/mylab-generative), compiled here | `@fastgltf//:fastgltf` |

Where the wiring lives:

- `third/vcpkg.json` — the vcpkg manifest.
- `third/vcpkg.BUILD` — build file for the whole installed vcpkg tree.
- `third/fastgltf.BUILD` — build file for the sibling source checkout.
- `third/BUILD.bazel` — dependencies needing no external repo (DirectML, stb).

A few non-obvious details are documented in comments at the point of use rather
than repeated here — in particular why `STB_IMAGE_IMPLEMENTATION` must be
`local_defines`, why `DML_TARGET_VERSION_USE_LATEST` is set, and why each
package declares only its own headers out of the shared vcpkg `include/`.

