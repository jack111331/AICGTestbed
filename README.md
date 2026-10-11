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

### 5. Run

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
| `//Engine:rendergraph-smoke` | The render graph's pass compilation and barrier placement. Runs with no D3D12 device at all. |
| `//Engine:motionfeatures-smoke` | HumanML3D feature decoding and the Unigram tokenizer. No device or model needed. |
| `//Engine:flooddiffusion-smoke` | Runs the real text-to-motion pipeline end to end on both providers and checks they agree; skips the model sections when `resources/FloodDiffusion/` is absent. |

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

Note `-c dbg` is currently the only mode whose executable runs at all (see
Rough edges), so in practice it is the mode you capture in.

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
- `resources/FloodDiffusion/` — the staged text-to-motion models, written by
  `tools/prepare_flooddiffusion.py` and gitignored. 1.2 GB, almost all of it
  `text_encoder.onnx`, and the weights are not ours to redistribute.

## Rendering

Deferred shading, driven by a render graph.

### Why deferred

Forward shading runs the light loop once per pixel *per primitive covering it*,
so the cost is `lights x overdraw`. Deferred records each pixel's material
inputs once and then runs the light loop once per pixel, making the cost
`lights + overdraw`. That is the only reason to do it, and it is what makes a
scene with more than a handful of lights affordable: the forward path carried
four lights in a per-node constant buffer, the deferred path carries 64 in one
buffer uploaded once a frame.

What it costs in exchange:

- **Bandwidth.** Four render targets written and read back every frame.
- **One shading model.** Every pixel is shaded by one pixel shader, so a
  material has to be expressible in the G-buffer's channels.
- **No transparency.** A blended surface needs the lit colour of what is behind
  it, and one pixel of G-buffer holds one surface. Nothing in the scene is
  blended yet; when something is, it belongs in a forward pass after the
  lighting resolve, sharing the same depth buffer.

### The G-buffer

Four colour targets plus depth. Depth is `DeviceResources`' own depth buffer,
read back through an SRV rather than duplicated — which is why that resource is
created `R32_TYPELESS`: D3D12 will not put a shader resource view on a resource
created as `D32_FLOAT`, so the resource is typeless and each view names its own
typed interpretation (`DeviceResources::TypelessDepthFormat`).

| Target | Format | Contents |
|---|---|---|
| 0 | `R8G8B8A8_UNORM` | base colour rgb (linear), alpha in `a` |
| 1 | `R16G16B16A16_FLOAT` | world-space shading normal, after normal mapping |
| 2 | `R8G8B8A8_UNORM` | `r` metallic, `g` roughness, `b` occlusion |
| 3 | `R11G11B10_FLOAT` | emissive rgb |
| depth | `D32_FLOAT` | depth; the lighting pass rebuilds a world position from it |

Two of those formats are a judgement rather than an obvious choice, and
`Engine/GBuffer.hpp` carries the reasoning: normals get 16-bit float because an
8-bit UNORM normal bands visibly across a smooth specular highlight, and
emissive gets a float format because it is the one channel legitimately allowed
above 1.0. Target 0 is deliberately **not** `_SRGB` — it would spend its 8 bits
where the eye can see them, but it would also apply a transfer function on write
and undo it on read, silently, which is maddening to debug when a base colour
does not match its texture.

There is no position target. The lighting pass reconstructs a world position
from depth and the inverse view-projection, which saves a whole `RGBA16F`
target for the price of one matrix and a divide.

### The render graph

`Engine/RenderGraph.hpp`. A pass declares which resources it reads and writes
and in what capacity; the graph works out the resource barriers and records the
frame into one command list. The frame is currently:

| Pass | Reads | Writes |
|---|---|---|
| `GBuffer` | — | 4 G-buffer targets + depth |
| `DeferredLighting` | 4 G-buffer targets + depth | back buffer |
| `ForwardOverlay` | — | back buffer + depth |
| `ImGui` | — | back buffer |

The depth buffer is the reason this is worth having: it is written by the
geometry pass, **read** by the lighting pass, and written again by the overlays,
which is two transitions per frame on one resource in an order that depends on
which passes are enabled. Hand-written barriers get that wrong quietly — the
symptom of forgetting the transition back to `DEPTH_WRITE` is overlays that
z-fight, not a validation error.

`Compile()` produces the barrier plan **without a D3D12 device**, which is what
makes `//Engine:rendergraph-smoke` possible: it asserts on the transitions a
given frame structure produces, and on the mistakes `Compile()` has to refuse
(reading a target nothing wrote, reading and writing one resource in one pass,
using a colour target as depth, clearing a resource the pass did not declare).
A render graph that silently accepts a mis-ordered pass list is worse than none,
because the barriers it *does* emit look authoritative.

`DeferredRenderer::ShowImgui` prints the plan the current frame compiled to,
which is cheaper to read than a capture when the question is only "did the
transitions happen".

Deliberately not implemented yet, in rough order of value:

- **Transient memory aliasing.** Declared textures are committed resources that
  live until the next resize. Aliasing needs resource lifetime analysis plus
  aliasing barriers; the pass list is the hard part and it is already here.
- **Pass culling.** Every added pass runs. This only matters once passes are
  added conditionally.
- **Tiled or clustered light culling.** 64 lights in a flat per-pixel loop is
  where this stops scaling, and the answer is a pass of its own — which is
  most of why the light cap is 64 rather than the 1024 a constant buffer would
  hold.
- **A G-buffer debug view.** `hlsl_shader` now takes any number of entry pairs,
  so this is one more pair plus a pass.
- Split barriers, an async compute queue, multi-threaded recording per pass.

### Where the shading lives

`Engine/shaders/no_texture.fx` holds both entry pairs, so they share the BRDF
helpers, the light struct and the material constants. The split is exactly the
old forward pixel shader cut in half: `PSGBuffer` is its material gather
(texture sampling, normal mapping, occlusion, emissive) writing to four targets
instead of consuming the values, and `PSLighting` is its shading half reading
them back. Neither half was rewritten.

Their constant buffers sit at different registers (`b0` for the geometry pass,
`b3` for lighting) because both entry points live in one file and two `cbuffer`s
at `b0` collide at declaration even when no single entry point uses both. The
root signature decides what is actually bound, so the numbers only have to be
distinct.

## Text to motion

`//Engine:FloodDiffusion` streams
[FloodDiffusion](https://github.com/ShandaAI/FloodDiffusion) text-to-motion and
the application draws the result as skeleton lines, in the **FloodDiffusion
motion** panel.

It is four ONNX models with the sampling loop on the host:

| Model | Signature |
|---|---|
| `text_encoder.onnx` | token ids → `[tokens, 768]` features |
| `denoiser.onnx` | one diffusion-forcing denoising call |
| `vae_decoder_first.onnx` | first latent → 1 motion frame + 20 caches |
| `vae_decoder_step.onnx` | next latent + caches → 4 motion frames + 20 caches |

**Why it streams.** Diffusion forcing gives each latent frame its own noise
level, rising along the sequence, and every step shifts that ramp forward by a
fraction of a chunk. Latents at the front finish while the ones behind are
still noisy, so `FloodDiffusionPipeline::Step` denoises once and then decodes
whatever that finalised. Motion appears after roughly 8 of 128 steps and keeps
arriving, rather than all at the end. One step per rendered frame generates
about five times faster than playback consumes it.

**Either execution provider runs it, and the choice is per generation** —
`FloodDiffusionOptions::device` for the sampler, `textEncoderDevice` for the
prompt encoder, both switchable in the panel.

An earlier export only ran on the CPU: the DirectML EP faulted on the
attention mask, which the graph built from `context_lens` with a
`Range → Reshape → Less` chain. That export was replaced and DirectML now runs
every model. Measured on an RTX 5080, with both providers agreeing to within
**0.000004** on the largest joint coordinate:

| | DirectML | CPU |
|---|---|---|
| denoiser | 37–40 ms | 1.4–10.5 ms |
| VAE step | 3.3 ms | 1.3 ms |
| text encoder | 46 ms | 9.4 ms |
| 8-latent generation, warmed | 955 ms | 83 ms |

**The CPU is currently about 11× faster, and the reason is visible in the shape
of the numbers**: the denoiser costs the same 40 ms whether one latent is live
or sixty-four, while the CPU scales 1.4 → 10.5 ms over the same range. That is
per-dispatch overhead, not compute. ORT profiling confirms it — 1226 of the
graph's 1334 nodes are dispatched to DirectML individually, with 108 left on
the CPU. A 36 MB model with `hidden_dim` 256 and a classifier-free-guidance
batch of two does not fill a 5080; it starves it.

Two consequences worth knowing before choosing DirectML:

* **The first pass pays shape compilation.** The EP compiles per distinct input
  shape and the sampler presents a new sequence length every other step, so a
  cold 8-latent run took 1252 ms against 955 ms warmed.
* **It contends with rendering.** A denoising step measured 44 ms standalone
  but 91 ms inside the running application, because ORT's work goes on the
  same queue as the renderer. At one step per frame that is a visible hitch
  while generating. A worker thread, or fewer steps per frame, is the fix.

**The EP already has the compiler needed to fix this; it just cannot use it.**
ONNX Runtime's `DmlGraphFusionTransformer` compiles each DirectML-assignable
partition into one `IDMLCompiledOperator` and dispatches it as a single
`DmlFusedNode`. It is on by default, but it only engages when shapes are
statically known, because `IDMLDevice1::CompileGraph` needs concrete tensor
descriptions. Counting `Node` events in an ORT profile:

| session | dispatches per run | on DML | on CPU | fused nodes |
|---|---|---|---|---|
| as exported, all dims symbolic | **1334** | 1226 | 108 | none |
| `batch` and `time` pinned | **294** | 239 | 55 | 10 partitions |
| all four dims pinned | **1** | 1 | 0 | `DmlFusedNode_0_2` |

Fixing the shapes collapses the whole graph to one dispatch with no CPU
partitions at all, and `ep.dml.disable_graph_fusion` confirms the mechanism:
static shapes cost 2.0 ms with fusion and 10.2 ms without. (GPU-side tensor
binding is **not** the lever — that saves copies, and copies are not where the
time goes. See [`skills/onnx-directml-ep`](skills/onnx-directml-ep/SKILL.md)
§6–§7.)

**`FloodDiffusionOptions::staticShapes` turns this on without a re-export**, via
`AddFreeDimensionOverrideByName`, and it is the "pin shapes" checkbox in the
panel. An 8-latent generation:

| | warmed | cold |
|---|---|---|
| CPU | 87 ms | 89 ms |
| DirectML, as exported | 966 ms | 1289 ms |
| DirectML, shapes pinned | **60 ms** | **171 ms** |

So pinned DirectML is the fastest configuration available, 16× the dynamic GPU
path and 1.5× the CPU, and the cold penalty falls 7.5× because there is only
one shape left to compile. In the running application a denoising step went
from 91.5 ms to 14.66 ms.

Two things to know before switching it on, both deliberate:

* **It changes the sampler.** `time` has to be pinned too, so the whole latent
  buffer is submitted every step rather than the live prefix, with the noise
  ramp clipped to 1 over the part that is not live. The denoiser is non-causal,
  so that padding attends into the real frames: the motion differs (mean 0.11 m,
  max 0.52 m against the prefix version, on a figure about 1.6 m tall). Bone
  lengths and proportions hold, and it is arguably closer to how the model was
  trained, but it is a different sampler. An export taking a valid-length input
  and masking self-attention would make it exact. That is why the default is
  off.
* **The length is compiled in.** The pinned `time` is `latentFrames +
  chunk_size`, baked at load, so `Begin` refuses a different length and the
  panel's slider reloads the sessions.

The VAE decoders are left dynamic on purpose: pinning their `batch` would take
a decode from 1.48 ms to 0.47 ms, but it breaks the 20-tensor cache hand-off
between `vae_decoder_first` and `vae_decoder_step` on ORT 1.24.4 (the step
decoder reports a null input). The denoiser is where the time is.

**The output is not joint positions.** The VAE decodes to HumanML3D's
263-dimensional feature vector: a root trajectory as per-frame velocities plus
joint positions in a root-local frame. `Engine/MotionFeatures.cc` integrates
the root and un-rotates the joints. The subtlety worth knowing is that the
batch formulation shifts the velocity arrays by one frame before integrating,
so frame *i* is placed using frame *i-1*'s velocity; applying the current
frame's instead still produces plausible motion that drifts one frame ahead of
the pose. `//Engine:motionfeatures-smoke` pins this against
`tools/motion_reference.py`, an independent numpy transcription, and against
properties that hold either way — a constant velocity must integrate to a
straight line, a pure yaw must not change a bone's length.

**Reproducibility.** The same seed gives the same motion within the engine but
*not* the same motion as the Python reference: that seeds numpy's PCG64 with a
ziggurat normal, and this uses `std::mt19937`.

**Not done yet.** Retargeting onto a glTF character — the skeleton is drawn on
its own, and only a single prompt per generation is supported (the reference
also accepts several prompts with `--text-end` handing over at given frames,
which would widen the `segments` axis of `context`).

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

## Rough edges

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
