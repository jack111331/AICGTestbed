# AICGTestbed

A Direct3D 12 testbed for integrating neural models into a glTF scene pipeline.

The renderer scaffold is Microsoft's
[`DirectXTKSimpleSamplePC12`](https://github.com/microsoft/Xbox-ATG-Samples/tree/main/PCSamples/IntroGraphics/DirectXTKSimpleSamplePC12)
(Win32 desktop, not the UWP variant), extended with glTF loading via a modified
fastgltf, DirectML for inference, and Dear ImGui for tooling UI. Built with
Bazel rather than MSBuild.

Windows x64 only — every dependency is either a Windows SDK component, a D3D12
API, or a vcpkg `x64-windows` package.

## Prerequisites

| Requirement | Notes |
|---|---|
| Visual Studio 2022 with the C++ workload | Bazel autodetects MSVC. Developed against MSVC 14.44. |
| Windows SDK 10.0.26100.0 | Required at that exact version, for two independent reasons: its `DirectML.h` is the first to reach `DML_TARGET_VERSION 0x6400`, which `Engine/directml-smoke.cc` enforces with a `static_assert` (the 22621 header caps at `0x5000`), and `shaders.bzl` hardcodes `dxc.exe` from its `bin` directory. |
| [Bazel](https://bazel.build) 9.x via [bazelisk](https://github.com/bazelbuild/bazelisk) | Bzlmod only — there is no `WORKSPACE` file. |
| [vcpkg](https://github.com/microsoft/vcpkg) | Used in manifest mode; see below. |
| A `fastgltf` checkout beside this repo | **Required.** See [fastgltf](#fastgltf-is-a-sibling-checkout). |

If you installed bazelisk with `winget`, it is not placed on `PATH`. Add its
package directory to `PATH` — otherwise every command below needs the full path
to `bazelisk.exe`, which lives under:

```
%LOCALAPPDATA%\Microsoft\WinGet\Packages\Bazel.Bazelisk_Microsoft.Winget.Source_8wekyb3d8bbwe\
```

### fastgltf is a sibling checkout

fastgltf is **not** taken from vcpkg. It comes from a fork carrying the
`MYLAB_generative` vendor extension — which adds `Asset::generators` and
`Node::generative` for describing a neural model's inputs, outputs, graph and
weights inside a glTF asset:

**[jack111331/fastgltf, branch `mylab-generative`](https://github.com/jack111331/fastgltf/tree/mylab-generative)**

Bazel compiles it from source rather than consuming a prebuilt library, so edits
to the checkout rebuild here with no CMake step in between. `MODULE.bazel`
resolves it as `path = "../fastgltf"`, relative to this workspace root, so clone
it as a sibling:

```bash
git clone -b mylab-generative https://github.com/jack111331/fastgltf.git ../fastgltf
```

giving this layout:

```
C:\Code\
  ├─ ptflio\        <- this repo (its folder name does not matter)
  └─ fastgltf\      <- must be named exactly "fastgltf"
```

Without it, `bazelisk build //...` fails to resolve `@fastgltf`.

## Build

### 1. Install the vcpkg packages

Bazel does **not** drive vcpkg. Run it once up front, and again after editing
`third/vcpkg.json`:

```bash
vcpkg install --x-manifest-root=third --triplet x64-windows
```

That populates `third/vcpkg_installed/` (git-ignored, ~90 MB) with directxtk12
and imgui. `third/vcpkg-configuration.json` pins the registry baseline, so the
result is reproducible.

### 2. Build

```bash
bazelisk build //...
```

### 3. Run

```bash
bazelisk run //Engine:hello-world
```

## Targets

| Target | What it is |
|---|---|
| `//Engine:hello-world` | The sample application — opens a window and renders the scene. |
| `//Engine:directxtk12-smoke` | Verifies DirectXTK12 headers, import library and DLL staging. |
| `//Engine:fastgltf-smoke` | Parses a `MYLAB_generative` asset; fails if the build regresses to stock fastgltf. |
| `//Engine:imgui-smoke` | Builds an ImGui frame headless; link-checks the DX12 and Win32 backends. |
| `//Engine:directml-smoke` | Creates a DirectML device and reports the runtime feature level. |
| `//Engine:stb-smoke` | Decodes a known PNG from memory and from disk, checking pixel values. |

The smoke tests are cheap and each one pins a property that is easy to break
silently, so they are worth running after any dependency change:

```bash
bazelisk run //Engine:fastgltf-smoke
```

### Compilation modes

`fastbuild` (default), `-c dbg` and `-c opt` all work. Bazel's MSVC toolchain
compiles with `/MDd` under `-c dbg` and `/MD` otherwise, and the vcpkg packages
ship both CRT variants, so `third/vcpkg.BUILD` uses `select()` to pick the
matching one. Mixing them would be a silent CRT mismatch, so prefer these three
modes over hand-rolled flags.

## Dependencies

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

## Assets

- `Media/` — fonts, meshes, sounds and textures for the ATG sample, laid out the
  way `DX::FindMediaFile` (`Engine/FindMedia.h`) expects. It searches `Media/`,
  `Media/Textures`, `Media/Fonts`, `Media/Meshes` and `Media/Sounds`, walking up
  from the executable. Keeping them at the workspace root means they resolve from
  the Bazel execroot without relying on runfiles symlinks, which are off by
  default on Windows.
- `resources/` — glTF test scenes and HLSL sources.

## Rough edges

Known and worth fixing, listed so they do not surprise you:

- **`shaders.bzl` hardcodes an absolute `dxc.exe` path** under Windows SDK
  `10.0.26100.0`, so a machine with a different SDK build installed needs that
  string edited. It also prepends `"Engine/"` to shader sources to work around
  the genrule's working directory (there is a `TODO` in the file), which means
  the rule only works for shaders under `Engine/`.
- **The shader genrule needs `bash`** — it uses `&&` and `cat`. Bazel on Windows
  invokes MSYS2 bash (`c:\msys64\usr\bin\bash.exe`) for `genrule`.
- **`@vcpkg_directxtk12` is named after its first package** but now covers the
  whole vcpkg tree including imgui and stb. Renaming it would touch every
  `@vcpkg_directxtk12//:...` label.
