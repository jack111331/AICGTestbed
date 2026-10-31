# AICGTestbed

A Direct3D 12 testbed for integrating neural models into a glTF scene pipeline.

## Summary

The renderer scaffold is Microsoft's
[`DirectXTKSimpleSamplePC12`](https://github.com/microsoft/Xbox-ATG-Samples/tree/main/PCSamples/IntroGraphics/DirectXTKSimpleSamplePC12)
(Win32 desktop, not the UWP variant), extended with glTF loading via a modified
fastgltf, DirectML for inference, and Dear ImGui for tooling UI. Built with
Bazel rather than MSBuild.

Windows x64 only — every dependency is either a Windows SDK component, a D3D12
API, or a vcpkg `x64-windows` package.

What is in it:

| | |
|---|---|
| **Deferred shading on a render graph** | Four G-buffer targets plus depth, passes declaring their reads and writes so the barriers are derived rather than hand-written, and a tonemap split that keeps linear radiance on one side of it. [Rendering](docs/rendering.md) |
| **glTF scenes, merged** | Several files in one scene graph from a JSON manifest, with per-file placement, visibility and animation playback, skins, punctual lights and per-pixel specular-glossiness conversion. [Scenes](docs/scenes.md), [Assets](docs/assets.md) |
| **Shadows** | Four cascades in one 4096² atlas with the static half cached across frames, plus an offline-unwrapped lightmap that bakes the sun's visibility into the surface. [Shadows](docs/shadows.md) |
| **Neural model work** | A name-keyed `ModelManager` over DirectML and ONNX Runtime, each on its own compute queue. [Compute queues](docs/compute-queues.md) |
| **Streaming text to motion** | FloodDiffusion's four ONNX models with the diffusion-forcing sampler on the host, decoded paced to the playhead and retargeted onto a rigged character. [Text to motion](docs/text-to-motion.md) |
| **Tooling** | One ImGui window of four tabs, thirteen debug views that replace the frame, and a smoke test per property that is easy to break silently. [The panel](docs/ui.md) |

### Documentation

| | |
|---|---|
| [Scenes and the scene manifest](docs/scenes.md) | `scenes.json`, merging several glTF files, pinning the camera |
| [Assets, the Bistro, the sky and materials](docs/assets.md) | staging the Bistro, what it takes to load, the sky, light units, specular-glossiness |
| [Rendering](docs/rendering.md) | why deferred, the G-buffer, the render graph, culling, descriptors, the BRDF |
| [Shadows and baked lightmaps](docs/shadows.md) | the cascades, the static cache, the lightmap bake |
| [Compute queues for model work](docs/compute-queues.md) | `ModelManager`'s queue, FloodDiffusion's own, what the fences cover |
| [Text to motion](docs/text-to-motion.md) | the pipeline, providers, latency, HumanML3D features, retargeting |
| [The ImGui panel](docs/ui.md) | the four tabs, mouse gating, the sun override, the debug views |
| [Dependencies](docs/dependencies.md) | where each library comes from and which label names it |
| [Rough edges](docs/rough-edges.md) | known problems |

## Usage

```bash
bazelisk run //Engine:AICGTestbed
```

First time through, see [Build](#build) — the vcpkg packages, ONNX Runtime and
the fastgltf sibling checkout all have to be in place. All three compilation
modes build and run the application; see [Compilation
modes](#compilation-modes) for what each costs.

Three environment variables steer what it loads:

| Variable | Effect |
|---|---|
| `PTFLIO_SCENE` | A `.json` [scene manifest](docs/scenes.md), or one or more `.gltf`/`.glb` paths, `;`-separated. Overrides everything else. |
| `PTFLIO_MAX_TEXTURE` | Longest texture edge in pixels; `0` removes the cap. Default 2048. |
| `PTFLIO_SKYBOX` | Equirectangular environment map for the sky. Defaults to the first `.hdr` beside the **first** scene. |

With none set, the app looks for a **scene manifest** -- `scenes.json` in the
working directory, then `resources/scenes.json` -- and failing that takes the
first `.gltf` or `.glb` it finds in `../niagara_bistro`, then
`resources/bistro/interior-wine`, `.../interior`, `.../exterior`, falling back
to the small authored microphone-and-character scene.

A manifest is how several glTF files are merged into one scene, and how a shot
is pinned so it survives a restart:

```bash
cp scenes.example.json scenes.json
bazelisk run //Engine:AICGTestbed
```

`scenes.json` is gitignored and is what the app loads when nothing names a
scene, so that is the whole of it — no environment variable. The format, the
per-file placement and animation controls, and how the **Scene** tab writes the
file back are in [Scenes and the scene manifest](docs/scenes.md).

### Controls

Drag controls: **left orbits, right dollies, middle pans**. `Escape` quits,
unless a text field has the keyboard. The camera and the panel share one mouse,
and the owner of a drag is decided when the first button goes down — see
[Sharing the mouse with the panel](docs/ui.md#sharing-the-mouse-with-the-panel).

One ImGui window holds four tabs:

| Tab | What is in it |
|---|---|
| **Render** | What the back buffer shows, the sun's direction, cascaded shadows, the sky, frustum culling |
| **Scene** | The camera, each file's transform, its animations, its hierarchy and images |
| **Neural** | The registered models, and the FloodDiffusion prompt and playback controls |
| **Performance** | Every cost and timing in the process |

The **Shown** combo at the top of **Render** replaces the frame with any one of
the thirteen buffers it was built from — the G-buffer channels, linear depth,
the shadow atlas, the lightmap and its sun depth map. Both are documented in
[The ImGui panel](docs/ui.md).

## Build

### Prerequisites

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

#### fastgltf is a sibling checkout

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

### 1. Install the vcpkg packages

Bazel does **not** drive vcpkg. Run it once up front, and again after editing
`third/vcpkg.json`:

```bash
vcpkg install --x-manifest-root=third --triplet x64-windows
```

That populates `third/vcpkg_installed/` (git-ignored, ~90 MB) with directxtk12
and imgui. `third/vcpkg-configuration.json` pins the registry baseline, so the
result is reproducible.

### 2. Fetch ONNX Runtime (DirectML build)

```bash
python third/fetch_onnxruntime.py
```

Writes `third/onnxruntime/` (18 MB, gitignored) from the
Microsoft.ML.OnnxRuntime.DirectML NuGet package, pinned to a version in the
script. Needed by `//third:onnxruntime`, which the `ModelManager` ONNX Runtime
backend uses.

This is vendored rather than taken from vcpkg because **the vcpkg
`onnxruntime` port cannot build the DirectML execution provider**: its portfile
maps `directml -> onnxruntime_USE_DML`, but no `directml` feature is declared in
its `vcpkg.json`, so the mapping never activates. The only other route is a
from-source build with `--use_dml`.

`DirectML.dll` is not fetched: `onnxruntime.dll` imports it and resolves against
the in-box `%WINDIR%\System32\DirectML.dll`. The script checks that copy is at
least the 1.15.4 the package was built against, and says so if it is not.

### 3. Stage the FloodDiffusion models (optional)

Only needed for the text-to-motion panel. Point the script at a directory
written by [FloodDiffusion](https://github.com/ShandaAI/FloodDiffusion)'s own
`export_onnx.py`:

```bash
python tools/prepare_flooddiffusion.py C:/Code/FloodDiffusion/onnx_models/tiny --link
```

That writes `resources/FloodDiffusion/` (gitignored): the denoiser, the two VAE
decoders, the text encoder, the tokenizer baked flat as `tokenizer.bin`, and
`pipeline.txt`. `--link` hard-links the `.onnx` files instead of copying them,
which saves 1.2 GB when the export is on the same volume.

Two things are converted rather than copied, so the engine needs no JSON
parser: `tokenizer.json` is 16 MB holding a 256k-entry Unigram vocabulary, and
`config.json` becomes `key value` lines. The shapes `config.json` does not
record -- the text feature width, the VAE cache count and each cache shape --
are read back out of the `.onnx` files by `tools/onnx_signature.py`, so they
cannot drift from the models they describe.

Without this the application still runs; the FloodDiffusion panel reports that
the models are not staged.

### 4. Build

```bash
bazelisk build //...
```

### Targets

| Target | What it is |
|---|---|
| `//Engine:AICGTestbed` | The sample application — opens a window and renders the scene. |
| `//Engine:directxtk12-smoke` | Verifies DirectXTK12 headers, import library and DLL staging. |
| `//Engine:fastgltf-smoke` | Parses a `MYLAB_generative` asset; fails if the build regresses to stock fastgltf. |
| `//Engine:imgui-smoke` | Builds an ImGui frame headless; link-checks the DX12 and Win32 backends. |
| `//Engine:directml-smoke` | Creates a DirectML device and reports the runtime feature level. |
| `//Engine:stb-smoke` | Decodes a known PNG from memory and from disk, checking pixel values. |
| `//Engine:rendergraph-smoke` | The render graph's pass compilation and barrier placement. Runs with no D3D12 device at all. |
| `//Engine:Lightmap` | Chart sizing and atlas packing for baked per-surface results. No D3D12. |
| `//Engine:dynamicdescriptorheap-smoke` | The batched descriptor commit, on a two-table root signature no shader here uses. Needs a device and the debug layer, no window. |
| `//Engine:scenegraph-smoke` | The glTF scene graph's transforms, skinning and animation sampling, plus frustum culling and the shadow cascade fit. No device needed. |
| `//Engine:motionfeatures-smoke` | HumanML3D feature decoding and the Unigram tokenizer. No device or model needed. |
| `//Engine:flooddiffusion-smoke` | Runs the real text-to-motion pipeline end to end on both providers and checks they agree; skips the model sections when `resources/FloodDiffusion/` is absent. |
| `//Engine:retarget-probe` | Not an assertion but a measurement: scores every plausible reading of a feature frame's 6D rotation block against the `ric` positions of the same frame, and poses the real character to measure it against the skeleton overlay. Run from the repository root, since it reads `resources/`. |

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

## Performance

Every figure here was measured on this machine — an RTX 5080 — and each one is
derived in the document it links to, together with what it does *not* say. The
reference scene throughout is the Amazon Lumberyard Bistro: 2909 drawn
primitives from 551 meshes, 4.21M triangles, 254 materials and 97 punctual
lights, with 1.17 GB of block-compressed texture. [Staging
it](docs/assets.md#staging-the-bistro).

**What the frame costs, and what made it cheaper.** The frame rate is not
capped: `DeviceResources` is created with `c_AllowTearing` and presents with a
sync interval of 0, so a vsync-pinned 16.667 ms is a measurement artefact
rather than a result.

| | | |
|---|---|---|
| **Material descriptor tables built once at load** | `CopyDescriptors` 4749 → **3** calls a frame, 23738 → 8 descriptors | [detail](docs/rendering.md#material-descriptor-tables-built-once) |
| | 0.45 ms a frame of geometry recording, about 5% | |
| **Caching the shadow cascades' static half** | geometry recording 11000 → **1976 us/frame**, 82% less | [detail](docs/shadows.md#caching-the-static-half) |
| | shadow draws 3326 → **13**; 98% of cascade-frames held while walking | |
| **Frustum culling, per primitive** | 9% culled from inside the street, 94% at a 30 degree field of view | [detail](docs/rendering.md#view-frustum-culling) |
| | 74% of the four cascades' 11636 shadow draws | [detail](docs/shadows.md#cascaded-shadow-maps) |
| **Baked lightmap** | one 4096² R8 atlas, 16 MB, 2902 charts at 8 texels/m, 54.3% used | [detail](docs/shadows.md#baking-the-shadow-into-the-surface) |

Two of those are the ones worth reading rather than quoting. The descriptor
work is **structurally** right and numerically a wash at this size — batching
the per-table copies changed nothing measurable, and what paid was deleting the
per-frame copies altogether. The cascade cache's 82% is CPU recording time, and
it is bought with 64 MB a frame of GPU copy traffic that has not been narrowed
yet.

**Text to motion**, at the shipped 60 latents with the VAE decoders on the CPU
provider. [Detail](docs/text-to-motion.md).

| | calls | mean | max | total |
|---|---|---|---|---|
| denoiser | 128 | 2.29 ms | 3.26 ms | 294 ms |
| VAE decoder | 60 | 1.43 ms | 2.11 ms | 86 ms |

Start-up latency is about **20 ms** to the first motion frame, and the sampler
runs about six times faster than playback consumes it. An 8-latent generation,
across the configurations that matter:

| | warmed | cold |
|---|---|---|
| CPU | 87 ms | 89 ms |
| DirectML, as exported | 966 ms | 1289 ms |
| DirectML, shapes pinned | **60 ms** | **171 ms** |

Pinned DirectML is the fastest configuration available — 16x the dynamic GPU
path — and it is **off by default**, because pinning `time` changes the sampler
rather than only its speed. The dynamic path's 966 ms is per-dispatch overhead,
not compute: 1226 of the graph's 1334 nodes are dispatched individually, and
pinning the shapes collapses the graph to one fused dispatch. Model weights are
1226.7 MB, of which 1.1 GB is the text encoder, which is why it loads on demand.

**Reaction latency** to a prompt change mid-stream is governed by the decoder's
backlog rather than by the model: uncapped it reached 127 frames, about 6.4
seconds, and `samplerLeadFrames` bounds it at 35.

## Profiling

Where the numbers come from, and the ways this has produced confident nonsense
before.

**The Performance tab** collects every cost in the process into one place, which
is the entire point of it — the reason to look at a cost is almost always to
compare it against another one. It opens with the frame's wall clock and a plot
of the last four seconds on a **fixed 33.3 ms scale**, so the plot does not
rescale itself the moment the frame gets slower, which is exactly when its shape
is being read. An 8 ms average over a frame alternating 2 and 14 ms is a stutter
and reads as 8 ms either way without the history.

| | |
|---|---|
| `PTFLIO_FRAME_REPORT=1` | one-shot stdout dump of p95/p99 frame time, video memory, render target and vsync state. Needs an **interactive** run: the UI loop does not tick when the app is launched from a tool, so nothing is sampled. |
| `DeferredRenderer::ShowImgui` | prints the barrier plan the current frame compiled to, which is cheaper to read than a capture when the question is only "did the transitions happen". |
| `DenoiseTiming()`, `DecodeTiming()` | calls, mean, max and total for each half of the motion pipeline. The mean says where a generation's time goes; the maximum says what a frame budget has to survive. |
| `FirstFrameMs()`, `PromptLatencyMs()` | wall time from `Begin` to the first motion frame, and from a prompt change to the first pose that could carry it. |
| the smoke tests | each prints its conditions beside its numbers — provider split, latent count, pinned shapes, which generation was measured, and that every figure is CPU wall clock with no GPU timestamps. |

**A frame hash is the check that matters for anything structural.** An FNV-1a
hash of the `R16G16B16A16_FLOAT` scene colour is how the descriptor work, the
cascade cache and the sun override were each shown to leave the image alone.
The failure mode of a change to descriptor binding or to a cache is not a crash
— it is every material sampling somebody else's textures, or a shadow that is
merely in the wrong place, and the debug layer is perfectly happy with both.

**Submission is not execution on the GPU-resident path.** A `Run` there records
and returns, so a CPU clock around it times how long it took to *ask*. D3D12
timestamp queries bracket the execution provider's work with one write before
the `Run` and one after, on submissions the sampler was making anyway:

| denoiser, 128 calls | mean | max | total |
|---|---|---|---|
| submission (CPU clock) | 0.48 ms | 2.59 ms | 62 ms |
| GPU execution (timestamps) | 2.05 ms | 2.92 ms | 262 ms |

Submission understated the work by about **4x**, which is the size of the error
every earlier reading on that path carried.

Three measurement traps, each of which cost real time:

- **Vsync pins everything.** A first attempt at timing a per-frame dispatch
  returned `16.667 ms` for all six runs, to three decimal places. Present with
  `c_AllowTearing` and a sync interval of 0 before timing a frame.
- **A reinstated "before" path can measure itself.** The first reading of the
  descriptor change came back 1.30 ms because the reinstated block called
  `GetDevice` twice per material change, which the old path did not; 0.85 ms of
  that figure was pure artefact.
- **Check for a stray process.** Timestamp numbers moved by 2x while an
  uncapped `AICGTestbed` from an earlier launch was still rendering in the
  background.

### Shader debugging in RenderDoc

Shaders are compiled with debug information embedded in the shader container, so
RenderDoc shows the HLSL source and can step through it with no extra setup --
no PDB search path and no source path mapping, because `dxc` embeds the source
text itself alongside the line tables and the original variable names.

`shaders.bzl` passes `-Zi -Qembed_debug` in every compilation mode, and adds
`-Od` under `-c dbg` only. That last flag is the one that matters for stepping:
with optimisation on, DXIL reorders instructions and folds locals away, so
RenderDoc's line highlighting drifts and variables show as optimised out. Use
`-c dbg` when you intend to debug a shader.

Note `-c dbg` is currently the only mode whose executable runs at all (see [Rough edges](docs/rough-edges.md)), so in practice it is the mode you capture in.

To confirm a build really carries the information, dump the compiled container:

```bash
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe" -T ps_6_0 -E PSStraight -Zi -Qembed_debug -Od -Fo ps.cso Engine/shaders/no_texture.fx
```

```bash
"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe" -dumpbin ps.cso
```

The dump should contain `DILocalVariable` entries and the shader source as a
metadata string. The same command without `-Zi` has neither.

The cost is container size, about 8 KB to 39 KB per shader for `no_texture.fx`,
which ends up in the generated `no_texture.h` byte arrays. If that ever matters,
move `-Zi -Qembed_debug` into the `dbg` branch of `_DXC_DEBUG_FLAGS`.

## Skills

`skills/` holds distilled notes meant to be read before starting a particular
kind of work, in the Claude Code skill format (`SKILL.md` with frontmatter).

- [`skills/onnx-directml-ep`](skills/onnx-directml-ep/SKILL.md) — pitfalls and
  fixes for putting an existing ONNX model onto ONNX Runtime's DirectML
  execution provider from C++: package selection, the session options that are
  *not* in fact required, two `Ort::` lifetime traps that masquerade as broken
  models, diagnosing `MLOperatorAuthorImpl` faults, why the GPU is often slower
  for small models and how to prove it, which input dimensions must be fixed at
  export time and which paddings are numerically exact, and sharing a device
  and queue with a renderer.

Known problems are listed in [Rough edges](docs/rough-edges.md).
