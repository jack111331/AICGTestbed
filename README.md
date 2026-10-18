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

All three compilation modes build and run the application; see
**Compilation modes** for what each costs.

Two environment variables steer what it loads:

| Variable | Effect |
|---|---|
| `PTFLIO_SCENE` | A `.json` scene manifest (below), or one or more `.gltf`/`.glb` paths, `;`-separated. Overrides everything else. |
| `PTFLIO_MAX_TEXTURE` | Longest texture edge in pixels; `0` removes the cap. Default 2048. |
| `PTFLIO_SKYBOX` | Equirectangular environment map for the sky. Defaults to the first `.hdr` beside the **first** scene. |

With none set, the app looks for a **scene manifest** -- `scenes.json` in the
working directory, then `resources/scenes.json` -- and failing that takes the
first `.gltf` or `.glb` it finds in `../niagara_bistro`, then
`resources/bistro/interior-wine`, `.../interior`, `.../exterior`, falling back
to the small authored microphone-and-character scene.

#### Loading more than one file

Several glTF files are **merged into one scene**. The manifest says which, and
where each one goes:

```bash
cp scenes.example.json scenes.json
bazelisk run //Engine:hello-world
```

`scenes.json` is gitignored and is what the app loads when nothing names a
scene, so that is the whole of it -- no environment variable:

```json
{
  "name": "Animated character standing in the shadow-test set",
  "maxTextureDimension": 2048,
  "camera": {
    "position": [3.0, 2.0, -7.0],
    "target": [0.0, 1.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fovDegrees": 45.0
  },
  "scenes": [
    "resources/shadow-test/shadow-test.gltf",
    {
      "path": "resources/microphone_and_animated_char/scene_only_animated_char.gltf",
      "translation": [0.0, 0.0, 0.0],
      "rotation": [0.0, 180.0, 0.0],
      "scale": 1.0,
      "visible": true
    }
  ]
}
```

```
scene manifest: scenes.json (2 file(s))
scene 0: shadow-test.gltf -- 4 node(s) at offset 0, 0 image(s) at offset 0, 0 animation(s)
scene 1: scene_only_animated_char.gltf -- 67 node(s) at offset 4, 4 image(s) at offset 0, 2 animation(s)
```

`scenes` is the only required field; an entry may be a bare path string, which
means the default transform.

A **`camera`** is used instead of framing the scene. Without one the app points
a raised three-quarter view at the scene's bounding sphere, which is right the
first time a scene is opened and wrong as soon as the shot matters -- the
character is the subject, not the bounding box. `position` and `target` are
required; `up` defaults to +Y, and `fovDegrees`, `near` and `far` are each left
at what the app would have used when absent, so a manifest can pin where the
camera stands without having an opinion about its lens. The scene is still
measured either way: the bounding radius drives the shadow cascade distance, the
lightmap extent and the mouse speeds, none of which is about where the view is. **Paths resolve against the manifest's own
directory**, not the working directory, so a manifest sits beside its assets and
the pair can be moved or checked out anywhere; an absolute path is left alone,
which is how a scene staged outside the repository is named. `skybox` and
`maxTextureDimension` are the two environment variables, per scene.

**Rotation is XYZ Euler angles in degrees**, not a quaternion, even though glTF
stores quaternions. This file is hand-written and the three numbers are the same
ones the UI's sliders show, so there is no conversion between the file, the
sliders and the matrix to be lossy or to disagree about a convention. Scale is
uniform, and applies inside the translation -- so the translation stays in world
units whatever the scale is. `//Engine:scenegraph-smoke` pins that order down:
the other one scales the position too, which makes a position slider behave
differently depending on the scale.

The parser is strict. An unknown field, a two-element translation, a scale of
zero and a `"rotation"` of four numbers are all errors naming the field, because
a manifest is typed by hand and the alternative is a typo that loads most of a
scene -- which reads as the renderer having lost a file.
`//Engine:scenemanifest-smoke` covers each of them.

The **Scene** tab writes the file back: drag the sliders, press
**Save to scenes.json**, and the next run starts from that placement. An entry
nobody has moved is written back as the bare string it was read as. **Save the
camera too** adds the current view, with its field of view and clip planes, so a
shot found by flying around with the mouse survives a restart; it starts ticked
only for a manifest that already pinned a camera, since pinning one changes what
the next run looks like.

`PTFLIO_SCENE` still takes a `;`-separated list of glTF paths, which is the
older form and needs no manifest. Semicolon rather than `:` or `,` because
these are Windows paths. A file that cannot be parsed is reported and skipped,
so one bad path does not stop the others loading.

This exists because **a skinned character wants to be its own file** while the
set it stands in is authored separately -- re-exporting one into the other on
every change is not a workflow. One merged graph rather than two renderers is
what makes them share the material descriptor heap, the lightmap atlas, the
shadow cascades and the light list, so the character is lit by the set's lights
and casts a shadow onto its floor.

Each file gets its own section in the **Scene** tab, and keeps its own:

- **Placement** -- position, XYZ rotation in degrees and a uniform scale,
  composed inside the whole scene's own three, so moving or turning the
  character does not move the set. The scale slider is logarithmic over 0.001 to
  1000 with a `1x` button beside it: an exporter that wrote centimetres is out
  by exactly 100, and that is not a value a linear slider can reach. These are
  the same values the manifest holds, and the save button writes them there.
- **Visibility** -- a checkbox that takes the file out of the draw order, out of
  both shadow halves and out of the light upload. Switching the character off is
  how you see the set it is standing in, and switching the set off is how you
  find a character that has landed somewhere unexpected.
- **Animation playback** -- clip, play/loop, speed and time. These are separate
  because a walking character in one file and a turning fan in another are two
  clips running at once, not a choice between them.

What is merged, and therefore offset on the way in, is every index that crosses
a file boundary: node parents and children, skin joints, animation channel
targets, and a material's texture indices. A missed offset there does not fail
-- it silently names something in the *other* file -- so
`//Engine:scenegraph-smoke` asserts each one, including against two copies of
the same asset, where a wrong index would still look right.

Whatever loads, the camera is **framed on it** at startup and the projection's
near and far planes are sized to its bounding sphere. This is not cosmetic: the
Bistro's bounding sphere has a radius of 83 units, so the previous fixed camera
at 1.5 units from the origin behind a far plane of 100 would have been inside
the geometry with most of the scene clipped away.

The **Camera** panel in the UI then exposes the rest: eye and target positions,
the up vector, vertical field of view, both clip planes, and the navigation
speeds. Three things there are worth knowing:

- **Move speed is scaled to the scene** when it frames, because the mouse
  controls move the camera in *world units* — a drag that crosses a
  two-metre scene in a moment barely stirs a hundred-and-sixty-metre one.
- **Near and far are shown with their ratio**, because that ratio is what
  governs depth precision: pushing the near plane in costs far more than
  pushing the far plane out.
- **Frame scene**, **Look at centre** (re-aims without moving the eye) and
  **Reset**. Easy to dolly past a large scene and lose it off the edge.

Drag controls: left orbits, right dollies, middle pans.

### Sharing the mouse with the panel

The camera and Dear ImGui read the same mouse, and the camera used to read it
unconditionally -- so dragging any slider also orbited the scene. Worse than the
spurious motion: pressing a button put the mouse in `MODE_RELATIVE`, which hides
the cursor and stops delivering positions, so the widget being dragged stopped
tracking as well.

Gating on `io.WantCaptureMouse` each frame does not fix it on its own, because
the camera's own drag is what breaks that flag. Once the cursor is captured and
hidden, `WM_MOUSEMOVE` stops arriving and ImGui's idea of where the pointer is
freezes wherever the drag began; if that happened to be over a panel the flag
goes true and the camera would stop halfway through an orbit.

So the owner of a drag is **decided once, when the first button goes down, and
held until every button is up**. That settles both directions: a drag that
starts on a widget never reaches the camera, and a drag that starts in the
viewport keeps the camera even as it sweeps across the UI window. Capturing the
cursor is likewise only something a camera-owned drag does.

`Escape` is gated too, on `io.WantCaptureKeyboard`. The FloodDiffusion prompt is
an `InputText`, and `Escape` there means "cancel this edit" to ImGui; it must
not also mean "quit".

The one wrinkle is ordering: `WantCaptureMouse` is computed inside
`ImGui::NewFrame`, which runs in `Render`, so `Update` reads the **previous**
frame's answer. That is the one frame of lag every ImGui integration lives with
and it is harmless here -- the pointer moves a few pixels in 16 ms, and an item
that was active last frame is still active now.

Checked by driving the two inputs the gate reads, 15 assertions:

```
a drag begun while ImGui wants the mouse
  the UI owns it                                 ok
  the cursor was NOT captured                    ok
  the camera did not move                        ok
  the UI still owns it after the flag drops      ok
  the cursor was never captured for the UI drag  ok
a drag begun while ImGui does NOT want the mouse
  the UI does NOT own it                         ok
  the camera keeps it when the flag rises        ok
  the cursor WAS captured for the camera drag    ok
```

The two rows about the flag changing *during* the drag are the ones that matter;
they are the latch, and both directions of it fail without it.

Two things about how that was measured are worth recording, because the first
two attempts at it produced confident nonsense:

- **Injecting `WM_MOUSEMOVE` to fake the hover does not work.** The first probe
  did that and could not be believed: `io.WantCaptureMouse` went on changing
  between steps where nothing at all was injected. The Win32 backend polls the
  real cursor in `NewFrame`, so a synthetic position is overwritten before the
  gate ever sees it. `ImGui::SetNextFrameWantCaptureMouse` states the flag
  directly instead, which is the actual input the gate is written against. The
  buttons still go in as real messages through the app's own `WndProc`, which
  is the path DirectXTK's `Mouse` listens on.
- **`Mouse::GetState().positionMode` can trail `SetMode` by a step or two in a
  harness like this**, because the pending change commits when a mouse message
  is processed and a probe that never moves a real mouse supplies those only
  when it injects one. Two assertions failed on that before the cause was
  found; they now ask whether the capture happened *at all*, and during which
  of the two drags, which is the thing worth asserting and is not
  timing-dependent.

### The panel

One window, four tabs, because everything used to be a single scrolling column
of collapsing headers and that column had the frame time at the bottom of it,
past the scene hierarchy:

| Tab | What is in it |
|---|---|
| **Render** | What the back buffer shows, the sun's direction, cascaded shadows, the sky, frustum culling |
| **Scene** | The camera, the asset's transform, its animations, its hierarchy and images |
| **Neural** | The registered models, and the FloodDiffusion prompt and playback controls |
| **Performance** | Every cost and timing in the process |

The split is by what you are doing rather than by which class owns the widget,
so **Performance** collects numbers from four different files and three of those
files also draw controls into one of the other tabs. `Engine/Ui.hpp` is the
whole mechanism: `ShowImgui` takes a `UiSection` and draws only what belongs
there, which is what lets the renderer's shadow sliders sit next to the sky's
while its cascade-cache counters sit next to the scene's draw counts.

Separating the two is worth more than tidiness. The reason to look at a cost is
almost always to compare it against another one, and before this no two costs in
the frame were ever on screen together -- the shadow cache's hit rate was behind
one header and the draw count it was meant to be weighed against was behind
another. **Performance** opens with the frame's own wall clock and a plot of the
last four seconds of it, on a fixed 33.3 ms scale so the plot does not rescale
itself the moment the frame gets slower, which is exactly when its shape is
being read. An 8 ms average over a frame alternating 2 and 14 ms is a stutter
and reads as 8 ms either way without the history.

### Steering the sun

The **Sun** section of **Render** overrides the asset's directional light with
two Euler angles. Two, not three: a direction has two degrees of freedom and a
third angle would be a roll about the direction itself, which changes nothing.
The resulting unit vector is shown below the sliders, derived, so there is
nothing ambiguous about what the angles mean.

Both angles describe the direction the light **travels** -- the light node's
`-Z` axis in world space, which is what glTF stores and what the shader
consumes. A sun in the sky shines *downwards*, so it has a **negative** pitch,
and the labels say so because that sign is the one genuinely confusing thing
here. Yaw is measured from `-Z`, the same zero the camera panel uses, so an
angle read off one means the same thing in the other and `(0, 0)` is glTF's
default `(0, 0, -1)`.

The override is applied in `DeferredRenderer::ResolveSun`, which runs before
anything in the frame reads the light, and **everything downstream takes its
answer from there** -- the cascade fit, the lightmap bake, and the direction
uploaded to the lighting pass. That last one is written unconditionally rather
than only when the override is on: with no override it is the same vector that
was just copied out of the light array, and one code path cannot drift from
itself. Two places deciding the sun's direction separately is exactly how the
shadows end up disagreeing with the shading.

It is a view-time override, not an edit to the scene.
`SceneGraph::UpdateTransforms` recomputes a light's direction from its node
every frame and would undo anything written into the `SceneLight`, so the Scene
tab's hierarchy still shows what the asset authored -- which the panel says.
Moving the sun also **invalidates the baked lightmap**, because a bake fixes the
light into the data; the renderer notices and requests another, which on a large
scene is a visible hitch on the next frame.

The conversions live in `Engine/ShadowCascades.{hpp,cc}` rather than in the
renderer, because that file is arithmetic with no device in it and so
`//Engine:scenegraph-smoke` can assert on them -- 16 checks, including a round
trip over 408 angles that recovers both angles to **0.0000 degrees**. That
matters for the checkbox: enabling the override seeds the sliders from wherever
the sun already is, and a lossy round trip would make the light jump the moment
it was switched on.

Checked at runtime by reading the frame back:

| `PTFLIO_SUN` | resolved direction | frame mean |
|---|---|---|
| *(no override)* | (0.7071, -0.7071, 0.0000) | 11.644 |
| `-20,0` | (0.0000, -0.3420, -0.9397) | 9.593 |
| `-60,0` | (0.0000, -0.8660, -0.5000) | 12.192 |
| `-20,90` | (0.9397, -0.3420, 0.0000) | 9.585 |
| `-20,-90` | (-0.9397, -0.3420, 0.0000) | 9.541 |
| `20,0` | (0.0000, 0.3420, -0.9397) | 3.553 |

Every vector is the closed form of its angles, every frame hashed differently,
and the no-override frame came back **bit-identical** to the hash recorded
before any of this existed, which is what says the default path is untouched.
The last row is the useful one: pitch `+20` points the light *upwards*, away
from the ground, and the frame drops to a third of its brightness -- so the sign
convention reaches the shading rather than merely reaching a variable.

The same two runs on the Bistro are what check the part the fixture cannot. It
carries **97 lights** and the sun is not the first of them, so the patch has to
find the right index:

| | resolved direction | frame mean |
|---|---|---|
| *(no override)* | (0.2362, -0.9302, -0.2809) | 4.358 |
| `-25,140` | (0.5826, -0.4226, 0.6943) | 2.086 |

Predicted, the second is (0.5828, -0.4226, 0.6942). The frame re-lit, and that
is the actual proof: patching the `direction` field of a *point* light would
have changed nothing, since point lights do not read it, and the frame would
have come back identical.

### Looking at an intermediate

The **Shown** combo at the top of **Render** replaces the frame with any one of
the buffers it was built from:

| | |
|---|---|
| Scene (tonemapped) | the real frame |
| Scene colour (no tonemap) | the same HDR target raw, so anything above 1.0 clips rather than rolling off |
| G-buffer: base colour, normal, metallic, roughness, occlusion, emissive | the four targets, one channel at a time |
| G-buffer: baked sun visibility | the lightmap's term where the G-buffer carries it, in the material target's alpha |
| Depth (linear, / far plane) | the scene depth buffer, as a fraction of the view distance |
| Shadow atlas | whichever half the lighting pass sampled this frame |
| Lightmap atlas, Lightmap sun depth | the bake's output and the map it tested against |

It **replaces** the display transform rather than drawing over it. A thumbnail
in the corner is resampled twice, and the thing one of these views is usually
opened to find is a single wrong texel. `AddTonemapPass` dispatches to one pass
or the other, so there is exactly one pass writing the back buffer either way
and the application does not have to know a debug view exists -- whatever it
adds afterwards, the screen overlays and the UI, lands on top of both.

One pixel shader serves all thirteen, because what differs between a normal
buffer and a roughness buffer is only which channels to show and how to map
them onto [0,1]. Three things are not just a channel swizzle:

- **Normals are renormalised before the remap.** The G-buffer holds an
  interpolated, normal-mapped direction; a short one would read as murky grey
  rather than as the direction it points.
- **Every scalar channel shares one window**, "Black at" and "White at", so the
  two sliders mean the same thing in all of them. An orthographic depth map is
  what needs them: its occupied range is often a few per cent wide and raw it is
  a white rectangle.
- **Depth is linearised against the far plane, and the usual advice is wrong
  here.** `1/z` crowds a raw depth buffer against 1.0, so the standard fix is a
  logarithmic mapping -- black at the near plane, white at the far plane, each
  doubling of distance worth the same number of grey levels. It was written
  that way first and it is measurably worse:

  | | near | far | surface at | linear | logarithmic |
  |---|---|---|---|---|---|
  | shadow fixture | 0.01 | 28.0 | 12.6 m | 0.450 (114) | 0.899 (229) |
  | Bistro | 0.083 | 325.5 | 80 m | 0.246 (62) | 0.830 (211) |

  Both backgrounds are the far-plane clear, so they are white either way and
  contrast is the whole question: the log mapping pushes the geometry to within
  26 grey levels of the background in one scene and 44 in the other. The reason
  is that it spends its range on the gap between the near plane and the scene,
  and **this application already sizes the near plane to the scene** when it
  frames the camera -- so that gap is empty and the linear range is not being
  wasted in the first place. The predictions above reproduce the measured frame
  means to within 0.2 of 255, which is what identified the cause rather than
  just the symptom.

A source that is not screen-sized is **fitted, not stretched**. A 4096² atlas
squeezed into a 16:9 window makes a chart look like it has the wrong aspect
ratio, which is one of the things the view exists to check, so the shader
letterboxes instead and the bars are 0.05 grey rather than black -- black is a
legitimate value in every one of these buffers and the edge of the image is
worth being able to see.

The whole thing was checked by walking all thirteen views and reading the back
buffer back after each: every view produced a distinct hash on the Bistro, and
the letterbox came out at **43.8% of the frame against a closed-form 1 - 720/1280
= 43.75%**. Two views matched on the quad fixture -- base colour and occlusion,
which are both uniformly white there -- and diverged on the Bistro, which is
what said the match was the fixture rather than the wiring. The lightmap atlas
view reads 4.4% black on the Bistro, so what it shows is baked shadow and not
the lit clear.

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
| `//Engine:Lightmap` | Chart sizing and atlas packing for baked per-surface results. No D3D12. |
| `//Engine:dynamicdescriptorheap-smoke` | The batched descriptor commit, on a two-table root signature no shader here uses. Needs a device and the debug layer, no window. |
| `//Engine:scenegraph-smoke` | The glTF scene graph's transforms, skinning and animation sampling, plus frustum culling and the shadow cascade fit. No device needed. |
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
- `resources/alpha-test/` — a six-quad fixture for alpha coverage and face
  culling, generated by `tools/make_alpha_fixture.py` and gitignored. It uses no
  textures at all, relying on the renderer's 1x1 opaque white fallback so the
  alpha cutoff is the only variable. Left to right the quads must read
  **visible, visible, missing, visible, visible, missing**; the two gaps are a
  MASK material whose cutoff the fallback alpha cannot beat, and a back-facing
  single-sided quad.
- `resources/bistro/` — staged Bistro exports, gitignored. See below.
- `resources/FloodDiffusion/` — the staged text-to-motion models, written by
  `tools/prepare_flooddiffusion.py` and gitignored. 1.2 GB, almost all of it
  `text_encoder.onnx`, and the weights are not ours to redistribute.

### Staging the Bistro

The Amazon Lumberyard Bistro is the large scene this renderer is tuned against.
It is not in the repository and cannot be: 3.5 GB, and it is not ours to
redistribute.

Clone it as a **sibling of this checkout**, the same arrangement fastgltf uses:

```
C:/Code/ptflio             <- this repository
C:/Code/niagara_bistro     <- the scene
```

and it is picked up automatically. `PTFLIO_SCENE` overrides the location.

The copy to use is the [niagara project's](https://github.com/zeux/niagara)
conversion of [NVIDIA's rtxdi-assets](https://github.com/NVIDIAGameWorks/rtxdi-assets),
which is MIT licensed and, unlike the ORCA original, already glTF. Avoid
re-exporting the ORCA FBX yourself if you can: FBX2glTF gets metallic wrong on
most materials and exports foliage as alpha blend rather than alpha clip.

**It is worth knowing why this scene.** `pbrt-v4-scenes` catalogues the Bistro,
but **that repository holds no glTF** — it is pbrt format throughout. Of its 27
scenes the Bistro is the only one that originated as a real-time game asset, so
it is the only one whose lights are already punctual. The rest were authored for
offline path tracing: measured BRDFs, BSSRDFs, area lights, environment maps,
3.1 billion instanced triangles.

#### What it takes to load

Three things in this asset are not what a plain glTF loader expects, and each
one fails in its own quiet way:

- **`MSFT_texture_dds`.** Every texture names a `.png` in its ordinary `source`
  that *was never shipped*, and the real BC7 file in the extension — 343 of
  each. Preferring the DDS source is the difference between having textures and
  having none. It also means `Options::LoadExternalImages` cannot be used:
  fastgltf loads every image eagerly under that flag and returns on the first
  file it cannot open, so 343 dead URIs fail the whole parse. Images are
  therefore read by `PrepareImage` itself, and a missing one is a warning.
- **`KHR_materials_pbrSpecularGlossiness`** on 234 of 254 materials. Converted
  to metallic-roughness **per pixel in `PSGBuffer`**, so the lighting pass only
  ever sees one material model. See below for why per pixel is not optional.
- **97 punctual lights** (96 point, one directional), against a cap that was 64.
  Silent truncation reads as "the scene is too dark", so the cap is now 128.

Only images a texture actually points at are read. An `MSFT_texture_dds` asset
lists two per texture — the `.dds` it uses and a `.png` it does not — and
decoding that second set costs **5.8 GB instead of 1.17 GB**. The `images:` line
reports how many were skipped as unreferenced.

Loading takes a couple of minutes, almost all of it uploading **1.17 GB** of
texture. That is with the DDS kept block compressed; decoding it to RGBA8 would
be roughly four times as much. The `images:` line on stderr reports the total,
and `PTFLIO_MAX_TEXTURE` maps onto DirectXTK's `maxsize`, which drops top mip
levels — the only way to shrink a block-compressed texture without
decompressing it.

### The sky

The `Skybox` pass turns each pixel back into a world-space view ray and looks
it up in an equirectangular environment map. It runs before the lighting
resolve and **owns the clear**: every pixel is covered, so lighting writes over
the ones with geometry and leaves the sky where it discards. With no
environment loaded the pass is not added at all and lighting clears instead
— a conditional pass list is what the render graph is for, and no barrier
anywhere else has to know the difference.

Two supported containers, both of which the renderer could already read:

| | |
|---|---|
| `.hdr` | Radiance RGBE, via stb_image's float loader, converted to `R16G16B16A16_FLOAT` with mips |
| `.dds` | anything DirectXTK12 understands, including BC6H |

The sky is tonemapped with everything else now, so it no longer reads 2.4x
darker than equivalently-lit geometry.

**`.exr` is not supported.** The common compression for a released HDRI is PIZ
(a wavelet plus Huffman scheme), decoding it means linking OpenEXR, and nothing
in this project does. `LoadEnvironment` says so explicitly rather than giving
you a black sky. Poly Haven publishes a `.hdr` beside every `.exr`, which is
the path of least resistance; failing that, convert with anything that reads
EXR — Blender, Krita, GIMP, `oiiotool`, or `pip install OpenEXR`.

The Bistro is set up this way: `braustuble_alley_2k.hdr` sits in
`../niagara_bistro/` next to `bistro.gltf` and is picked up with no
configuration at all.

### Light units

`KHR_lights_punctual` is **photometric**: point and spot intensity is in
candela (lm/sr), directional in lux (lm/m2). A renderer working in radiometric
units has to divide by the luminous efficacy, **683 lm/W**, which is what
`ToShaderLight` does on load — the same place and the same constant
`glTF-Blender-IO` uses (`PBR_WATTS_TO_LUMENS` in `com/conversion.py`, applied by
its spec-compliant import mode).

Skipping it does not look like "slightly too bright". The Bistro's sun is
authored at **6830 lux**, which is exactly 10.0 W/m2 after the division — the
value was clearly picked so a correct import lands on a round number. Passing
6830 through raw gave a white surface a diffuse radiance of 2174, and measured
at one camera position:

| | p05 | p25 | p50 | p75 | p95 | pixels >= 250 |
|---|---|---|---|---|---|---|
| raw | 24 | 79 | **251** | 254 | 255 | **57.4%** |
| /683 | 23 | 60 | **95** | 124 | 188 | **0.0%** |

Albedo stopped mattering: a 20:1 range of surface colours all mapped to
254-255.

**Exposure is measured, not guessed.** An HDRI's absolute scale is a free
choice of whoever produced it, so `LoadEnvironment` takes the median luminance
of the loaded image and maps it to 0.18, linear mid-grey. The median rather
than the mean because the sun is in these images: `braustuble_alley_2k.hdr`
peaks at 132 against a median of 0.187, so a mean-based key would be dragged
several stops off by a handful of pixels. That HDRI lands on exposure 0.963.

This does **not** fix the dynamic range. 22% of that sky is still above 1.0 and
clips, because nothing tonemaps yet. Auto-exposure only ensures the sky starts
somewhere usable; the `.dds` path cannot even do that, since the pixels go
straight from file to GPU with nothing on the CPU to take a median of.

The **Deferred shading** panel carries the rest of the sky's controls: an
enable toggle, the exposure (overriding the measured default) and a yaw,
because a capture faces whatever direction the photographer did and almost
never the one the scene wants. The yaw rotates the *ray*, not the image, so
there is no resampling and no seam.

`tools/make_env_fixture.py` writes a test environment whose colours encode
direction: facing down -Z the middle of the screen must be green, the right
red, the left blue and behind you yellow. That is how the mapping was checked
rather than assumed, and it is two trig calls and a sign away from a sky that
looks plausible while the sun is behind you.

#### Converting specular-glossiness

The extension's own conversion, from the archived
[`KHR_materials_pbrSpecularGlossiness` examples](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js)
(`three.pbrUtilities.js` for the solve, `three.colorExtensions.js` for
`getPerceivedBrightness`), run in `PSGBuffer` at G-buffer write time. The
`specularGlossinessTexture` rides in the **metallic-roughness slot** — free on
a spec-gloss material, since the two workflows are mutually exclusive — so this
needs no extra SRV, no wider descriptor table and no root signature change.

**It has to be per pixel.** The solve reads metalness out of the specular
*colour*, and a material that keeps that colour in a texture carries a factor
of (1,1,1) meaning "use the texture as-is" — which on its own solves to fully
metallic. Measured on this asset, that is exactly what `Plaster` and `Beams`
carry. A factor-only conversion turns the street to chrome; their `_spec`
texture is what makes them dielectric, and 108 of the 234 are in that case.

The 125 factor-only materials convert fine: 101 solve to metallic 0.0, and the
ones that go metal are named `Forge_Metal`, `metal_clean_01` and `metal`. The
result across the loaded scene:

```
metallic   p10   0  p50   0  p90  11  p99 255     <= 0.05: 97.1%   >= 0.5: 1.4%
roughness  p10  56  p50 125  p90 170  p99 245
```

A street of plaster, wood and fabric with metal fittings, and a roughness
spread that only exists because glossiness now comes from the texture's alpha.

`tools/make_specgloss_fixture.py` is the regression test: factor-only
spec-gloss materials whose converted metallic, roughness and base colour are
closed-form. Its materials also carry a deliberately absurd fallback
`pbrMetallicRoughness` block (base colour magenta), so if the extension is ever
silently ignored the fixture shows it immediately.

One note on colour space: the extension puts specular **colour** in RGB and
glossiness in **A**. A `_SRGB` DXGI format decodes RGB through the sRGB curve
and passes alpha through linearly, which is exactly the split this needs — so
the map is claimed as an sRGB slot even though it rides in the
metallic-roughness one.

#### Measured cost

| | |
|---|---|
| nodes drawn | 2909 |
| draws | 2909 (one per primitive, no instancing) |
| triangles | 4,209,006 |
| pipeline switches | 314 (single- vs double-sided) |
| descriptor stages | 2326 of 2909 draws |
| texture memory | 1.17 GB |

The 551 meshes are referenced by 2909 nodes, which is why 1.75M triangles in
the file become 4.2M drawn. The gap between 2326 descriptor stages and 2909
draws is what the material cache already saves; closing it means sorting draws
by material, which nothing does yet.

#### What still is not right

- **Glass and transmission.** 18 materials use `KHR_materials_transmission` and
  three are alpha BLEND. A G-buffer holds one surface per pixel, so they render
  opaque. This wants a forward pass after the lighting resolve.
- **Per-pixel gloss.** 109 materials carry a `specularGlossinessTexture`, which
  is deliberately not bound: its channels are RGB = specular, A = glossiness,
  while the shader reads the metallic-roughness slot as B = metallic,
  G = roughness. Binding it would feed specular blue in as metalness. Using it
  needs a decision in the pixel shader.
- **No ambient or sky.** The lighting pass has 97 punctual lights and a constant
  `float3(0.03)` term. An exterior scene's look is substantially sky-driven.

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
| `Skybox` | environment map | scene colour (HDR) |
| `DeferredLighting` | 4 G-buffer targets + depth | scene colour |
| `WorldOverlay` | — | scene colour + depth |
| `Tonemap` | scene colour | back buffer |
| `ScreenOverlay` | — | back buffer |
| `ImGui` | — | back buffer |

Everything above `Tonemap` writes **linear radiance** into an
`R16G16B16A16_FLOAT` scene colour target; everything below it is in display
space. That split is why the overlays are two passes: the grid, the motion
skeleton and the teapot are world-space geometry and belong on the scene's
side of the transform, while the sprite and caption are authored in display
space and would be wrong if tonemapped. A PSO bakes its render target format
in, so the two sets of DirectXTK effects are built from two different
`RenderTargetState`s.

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

### View frustum culling

Per primitive, before anything is bound or allocated for it: the point is to
skip the constant buffer, the descriptor staging and the draw, not just the
draw. On by default, with a toggle in the **Scene cost** panel, because the only
way to be sure culling is not eating something it should have kept is to turn it
off and look.

`ViewFrustum::FromViewProjection` extracts six world-space planes from the
combined view-projection. Written out rather than using DirectXMath's
`BoundingFrustum` because the planes fall straight out of the clip-space
inequalities (`-w <= x <= w`, `-w <= y <= w`, `0 <= z <= w`) and so do not
depend on a handedness flag being guessed right — `CreateFromMatrix` takes the
projection alone plus `rhcoords` and then needs transforming by the inverse
view, which is two more places to get a convention wrong.
`//Engine:scenegraph-smoke` pins all of it down against the same `CreateLookAt`
and `CreatePerspectiveFieldOfView` the application uses, including that a box
*behind* the camera is rejected — the assertion that catches a near/far swap,
which otherwise culls nothing at all.

Primitive bounds come from the POSITION accessor's mandatory min/max, and the
AABB transform takes the **absolute value** of the matrix's linear part: a
rotation turns the box, and without the absolute values the contributions
cancel instead of adding. A 45 degree rotation about Z must grow a unit cube's
x and y extents to sqrt(2), which the tests check.

**Skinned primitives are never culled.** Their vertices are moved by the joint
matrices, so the authored bounds do not contain the deformed mesh — the
retargeted character's hips are relocated outright. Bounding the posed mesh
would mean transforming the bounds by every joint, which is not worth it for
the handful of skinned primitives in these scenes. The panel reports how many
primitives were exempt for that reason or for having no bounds at all.

Measured on the Bistro's 2909 primitives:

| camera | draws | triangles | culled |
|---|---|---|---|
| auto-framed, whole scene in view | 2909 | 4.21M | 0 (0%) |
| inside the scene, aimed at its centre | 2635 | 3.62M | 274 (9%) |
| same position, 30 degree field of view | 187 | 1.36M | 2722 (94%) |
| same position, looking away | 69 | 0.39M | 2840 (98%) |

The first row is the sanity check, not a disappointment: the framing fits the
entire bounding sphere, so nothing *is* outside. The 9% is the honest worst
case for a street scene — a 70 degree frustum aimed at the middle of it
contains nearly all of it, and the far plane sits at 325 units for an 83-unit
radius so nothing is distance-culled either. Culling earns its keep once the
camera is actually in the street looking along it.

What would raise it further is not a better frustum test: it is occlusion
culling, since a street is mostly walls, and there is no spatial hierarchy here
so all 2909 primitives are tested every frame regardless.

#### The bug culling exposed

Turning culling on produced a one-frame artifact while the camera moved, and
turning it off made it go away. Culling was not the cause; it was the first
thing to make an existing race visible.

`DynamicDescriptorHeap::Reset()` did this:

```cpp
m_AvailableDescriptorHeaps = m_DescriptorHeapPool;
```

Every GPU-visible descriptor heap handed straight back for reuse, with no
synchronisation. With two back buffers the swap chain only waits for the frame
that last used the *next* back buffer index, so frame N+1 can still be
executing while frame N+2 records — and frame N+2 would copy its descriptors
into the very heap frame N+1's submitted command list is still sampling from.
Logging the heap pointers showed the same three objects handed out on every
single frame.

That is invisible as long as consecutive frames stage **identical** descriptors,
which is exactly what happened before: the same primitives, in the same order,
every frame. Frustum culling changes the drawn set as the camera moves, so the
overwrite started writing *different* descriptors — and the in-flight frame
sampled another frame's textures for one frame. Which is why it looked like
material data arriving late: it was, just the descriptors rather than the
G-buffer.

`Reset()` now retires a frame's heaps against `Descriptors::Context()`'s frame
counter and only returns them once the in-flight margin has passed, the same
way `ReleaseStaleDescriptors` already gated the CPU-side allocator. Verified by
the same logging: frames 1-4 each get distinct heaps and frame 5 is the first
to reuse frame 1's. The cost is four sets of heaps instead of one — about
1 MB for the geometry pass's 8192-descriptor heaps.

Diagnosing it also turned up a second defect with the same symptom. `m_view`
was built at the *top* of `Update`, before the mouse handling further down
moves the camera, so it described the previous frame's camera while
`BuildFrameGraph` read `m_camera->mEye` fresh for the lighting and the sky. A
frame mixed a stale view-projection with a fresh eye position, which disagree
by exactly that frame's camera motion. `Sample::UpdateViewMatrix` is called from
`Render` now, after the input has been applied, so every consumer reads one
instant.

#### Committing descriptors in one call

`CommitStagedDescriptors` issued one `CopyDescriptors` per stale descriptor
table. It now issues one for all of them: a destination range per table, and
the source handles behind them flattened into as few ranges as they coalesce
into.

What makes the batch legal is that `CopyDescriptors` flattens both sides into a
single stream and pairs them off descriptor by descriptor -- the Nth source
descriptor goes to the Nth destination descriptor, wherever the range
boundaries fall. Only the two totals have to agree. That is also why a run of
source descriptors may be coalesced *across* a table boundary without
disturbing which descriptor lands where.

**Measured on the Bistro, it is a wash**, and it is worth writing down why.
Per frame:

| | before | after |
|---|---|---|
| `CopyDescriptors` calls | 4749 | 4749 |
| destination ranges | 4749 | 4749 |
| source ranges | 23738 | 23733 |
| descriptors copied | 23738 | 23738 |

4749 calls for 4749 commits with one stale table each. Every root signature
here has exactly **one** CBV_SRV_UAV descriptor table -- the sampler table is a
different heap type, and the lighting signature's two ranges (`t5` x5 and
`t12`) are two ranges of one table, not two tables -- so there was never more
than one table in a commit to batch. And only 5 of 23738 source descriptors
follow the previous one in their own heap, because a material's five texture
SRVs are five independent allocations; coalescing has almost nothing to fold.

So the change is structural rather than an optimisation at this size: it is the
right shape, it costs nothing, and it pays the moment a root signature grows a
second SRV table. **The 4749 calls are not what this function can fix.** They
are 4749 separate commits, one per draw whose material differs from the last.
Cutting them meant batching across *draws* instead, which is the next section.

Since nothing here exercises a two-table commit,
`//Engine:dynamicdescriptorheap-smoke` builds a root signature that does: three
SRVs at `t0` and two at `t3`, staged so table A's handles cannot coalesce and
table B's must. It asserts both tables are bound and that table B starts exactly
one table past table A -- the batch's characteristic failure is that both tables
bind, both copies happen, and the second table reads the first one's
descriptors. It runs with the debug layer on and drains the info queue after
each case, because a flattened copy whose two totals disagree is precisely what
the debug layer catches and nothing else would.

That test also found a latent crash, which is why it has a second case. A heap
switch marks every table in the root signature stale, including tables nothing
has staged into -- and the first commit on a fresh heap always takes that
branch. With no staged handles, the source is the zeroed handle cache, so the
driver is handed null source descriptors. Upstream gets away with it only
because every caller here happens to stage every table before committing. The
stale mask is now intersected with a mask of tables that have actually been
staged. Removing that one `&` makes the test abort rather than merely report, so
it is a real negative control.

#### Material descriptor tables, built once

`GLTFAdapter` used to stage five descriptors into a `DynamicDescriptorHeap` and
commit them on every draw whose material differed from the last. It now builds
**one shader-visible heap at load** holding a five-descriptor block per material,
and a draw binds its block by offset.

What makes this possible is that there was never anything per-frame about those
descriptors. A material's table is exactly its five texture SRVs; the images are
created once by `PrepareImage` and the fallback once alongside them, and nothing
touches either again. The dynamic heap was copying the same bytes to the same
place every frame because that is what the upstream staging model does, not
because the data moved.

Blocks are deduplicated on the five-slot texture tuple rather than per
primitive, which matters more than it sounds: everything untextured shares the
one block that is five copies of the white fallback.

| | before | after |
|---|---|---|
| `CopyDescriptors` calls per frame | 4749 | **3** |
| descriptors copied per frame | 23738 | **8** |
| material table binds | — | 2337 of 2909 draws |
| descriptors held | — | 805 (161 blocks, 25.2 KB), built once |

The three remaining calls are the lighting, skybox and tonemap passes, whose
views are genuinely per-frame — they point at that frame's G-buffer and scene
colour. Six descriptors, one and one. `DynamicDescriptorHeap` is still the right
tool there and `GLTFAdapter` no longer uses it at all.

161 blocks for 2909 primitives is the number worth noticing. The old code
re-copied five descriptors per material change, 4674 times a frame, to express
161 distinct tables that had already been decided at load.

**What it is worth, measured.** Geometry recording for the Bistro's 2909
G-buffer draws plus 2985 shadow draws, averaged over 120 frames:

| | per frame |
|---|---|
| with the per-draw copies | 8420.7 us |
| without | 7972.4 us |

**0.45 ms a frame**, about 5% of recording. Two honest caveats on that figure.
The "before" row was produced by reinstating *only* the `CopyDescriptors` call
into the current code, so it does not also reinstate the five `StageDescriptors`
calls and the five-element comparison of `std::optional`s that the old path ran
per material change — the real saving is therefore somewhat larger than 0.45 ms.
And a first attempt at this measurement read 1.30 ms because the reinstated
block called `GetDevice` twice per material change, which the old path did not;
that 0.85 ms was pure measurement artifact. Worth recording because it is the
shape of mistake that makes an optimisation look three times better than it is.

The other ~8 ms is not descriptors. It is the rest of the per-draw work --
a constant-buffer allocation per node, `IASetVertexBuffers` and
`IASetIndexBuffer` per primitive, one `DrawIndexedInstanced` each -- which is
what instancing or sorting draws by material would go after.

Two smaller things fell out of it. The sampler table is now set **once per
pass** rather than once per draw: that ordering used to matter, because a
descriptor commit that outgrew its heap invalidated every bound root table, so
the sampler table had to be re-set *after* each commit. Nothing switches heaps
mid-pass any more. And the per-draw "has the material changed" test is now an
integer compare against a block index instead of a five-way comparison of
`std::optional<std::size_t>`.

The frame is byte-identical across all of this: the same FNV-1a hash of the
`R16G16B16A16_FLOAT` scene colour, `f888a12e20596feb`, before the batching work
and after it. That is the check worth running on a change to descriptor binding,
because the failure mode is not a crash -- it is every material sampling
somebody else's textures, which the debug layer is perfectly happy with.

### Cascaded shadow maps

Directional lights only, which is the one case the cascade scheme is for: a
light with no position has no natural frustum to render from, so the frustum
has to be invented from the camera's. Point and spot lights are unshadowed and
`SampleShadow` is never reached for them.

Four cascades of 2048 texels, in one 4096x4096 atlas rather than a texture
array -- the render graph creates single-slice views, so an atlas needs no graph
changes, one DSV and one SRV, and each cascade is placed with a viewport
instead of a slice index. 64 MB at `R32_TYPELESS`, declared like every other
graph resource and so cleared and transitioned by the same code. 32-bit and not
`D16_UNORM` on purpose: the far cascade covers tens of metres across one tile,
and 16 bits over that range puts the quantisation step close to the depth bias,
which shows up as acne that no bias setting clears.

`Engine/ShadowCascades.{hpp,cc}` is pure arithmetic with no D3D12 in it, so
`//Engine:scenegraph-smoke` can assert on all of it. That split matters more
here than it does elsewhere, because nearly every way of getting cascades wrong
still produces shadows -- just shadows in the wrong place, or that crawl when
the camera moves, or that vanish at a slice boundary. Two of the fourteen
assertions are the ones worth naming:

* **Each slice is fitted with a bounding sphere, not a bounding box.** A box
  fitted to the slice's eight corners changes size as the camera rotates, by up
  to 40%, so the shadow texels change size with it and every edge crawls while
  you turn. A sphere is rotation-invariant: *"yawing the camera does not change
  the texel size"* holds to one part in a thousand.
* **The fitted centre is snapped to whole texels.** Without it the projection
  slides by a fraction of a texel every frame and every shadow edge shimmers.
  The test nudges the camera by tenths of a texel and measures where a fixed
  world point lands in shadow-map texels; the worst sub-texel drift is 0.0001,
  i.e. the projection moves in whole texels or not at all.

The depth range of each fit comes from the **scene** bounds, not the slice: a
roof behind the near slice still has to be inside the light's view or its
shadow does not exist, which the tests also check (*"a caster above the slice is
still inside the light's depth range"*). The splits themselves blend uniform
and logarithmic with `splitLambda = 0.75`, and `shadowDistance` defaults to
twice the scene radius via `SetSceneScale` rather than to the camera's far
plane -- the Bistro's far plane is 325 units, and spreading four cascades over
that spends most of the resolution on geometry too far away to see a shadow
edge on.

Filtering is a comparison sampler plus a 3x3 PCF kernel, clamped to the
cascade's own atlas tile so a tap near a tile edge cannot read the neighbouring
cascade. Acne is handled mostly by a **normal offset** of 1.5 cascade texels --
moving the sample along the surface normal, scaled by that cascade's world texel
size -- with only 0.0005 of constant depth bias and `SlopeScaledDepthBias = 2`
on the shadow PSO. That ordering is deliberate: a constant bias large enough to
clear acne on its own visibly detaches a shadow from the thing casting it.

The shadow pass reuses the G-buffer's vertex shader (`VSShadow` is `VSGBuffer`,
light matrix in the camera matrix's place) and its pixel shader does nothing but
alpha-clip, so the Bistro's foliage casts leaf-shaped shadows rather than
rectangles. The PSO is depth-only, `NumRenderTargets = 0`.

Each cascade culls against its own frustum, which is where cascades get cheap:

| scene | camera draws | shadow draws | culled |
|---|---|---|---|
| shadow fixture, 2 primitives | 2 | 3 | 5 of 8 |
| Bistro, 2909 primitives | 2909 | 2985 | 8651 of 11636 (74%) |

Both rows add up exactly to primitives x 4 cascades, which is the arithmetic
worth checking: a cascade frustum that silently rejected everything, or
accepted everything, would still render a plausible-looking frame.

`tools/make_shadow_fixture.py` writes a floor, an occluder above it and one
directional light at 45 degrees, and prints the screen coordinates where the
shadow must and must not be. Looking straight down at it, with the camera's
view-projection reading back off the tonemapped frame:

```
             lit    lit   in shadow
shadows on   170    170          51
shadows off  170    170         170
```

which is the test that the whole chain is wired up, as against the cascade
arithmetic being right in isolation.

Two things this deliberately does not do yet. **Cascade selection is a hard
switch** on view depth, with no blend band between slices, so a boundary is
findable if you look for it -- the texel sizes either side differ by the split
ratio. Blending means sampling two cascades over a narrow depth range and
interpolating. And **transparent casters cast as though opaque or not at all**:
the shadow pass alpha-clips, which is right for foliage but wrong for the
Bistro's glass, which should tint rather than block.

#### Caching the static half

The cascades are refitted and re-rendered from scratch every frame above. They
no longer are: static geometry's depth is rendered once per cascade fit and kept,
and only what moves is redrawn.

**What had to change first.** The obvious cache -- reuse a cascade's shadow map
while its matrix is unchanged -- never hits. Snapping stabilises the texel
*grid*, not the matrix: the fitted centre's component along the light, the
radius and the light's eye position all slide continuously with the camera.
Measured on a Bistro-sized setup, with texels of 1.6 to 24 cm:

| | cascade 0 | cascade 1 | cascade 2 | cascade 3 |
|---|---|---|---|---|
| texel size | 0.016 m | 0.034 m | 0.072 m | 0.239 m |
| matrix changed after | 1 mm | 2 mm | 1 mm | 1 mm |
| or after a yaw of | 0.01 deg | 0.01 deg | 0.01 deg | 0.01 deg |

So `ComputeShadowCascades` now **holds** a fit instead of recomputing it. Each
slice is fitted to a sphere padded by `fitMargin`, and the fit is reused while
the new slice still sits inside the padded one -- a two-sphere containment test.
The margin is what the camera travels through before a refit, so it is the whole
reason the cache hits; it is paid for in resolution, since the same texel count
now covers a sphere that much larger. At `fitMargin = 0` the old
refit-every-frame behaviour comes back exactly, which is what the test asserts.

Walking 20 m in 5 cm steps with a 12% margin, **98% of cascade-frames are held**
(1573 of 1600). A stationary camera holds everything, and the camera here is
mouse-drag driven, so it is stationary most of the time.

**What counts as static.** `SceneNode::movesAtRuntime` is set at load for any
node an animation channel targets, any node under one, any skinned node and any
joint -- over *every* animation rather than the selected clip, so changing clips
cannot quietly invalidate the cache. On the Bistro that is **7 moving nodes of
2909**, which an independent pass over the asset's JSON agrees with: six
`subset_N` props, a wicker basket, and one orbiting point light. `DrawOrder()` is
partitioned into static and moving halves as the transforms are walked, so the
per-frame pass iterates what it draws rather than scanning the scene once per
cascade.

**The frame.** One shadow pass became three:

```
ShadowStatic     static depth, only for cascades refitted this frame
ShadowComposite  the cached atlas copied into the live one
ShadowDynamic    the moving primitives, on top
```

`ShadowStatic` deliberately does not use the graph's `clears`, which clear a
whole resource and would wipe the tiles of cascades still holding their fit; it
clears each dirty tile with a rect instead. The composite is what keeps the
result **bit-identical** to drawing everything in one pass: depth with a LESS
test is a per-texel minimum, and a minimum does not care what order its terms
arrive in. Sampling two atlases and taking the lesser of two PCF results would
*not* be identical, because filtering the minimum is not the minimum of the
filtered -- which is why there is a copy here rather than a second sampler.

When a scene has nothing moving at all, the composite and the dynamic pass are
both skipped and the lighting pass samples the cached atlas directly. The shadow
fixture renders with **no shadow passes in the frame at all** after the first.

**Measured on the Bistro**, same build and same camera, toggling the cache:

| | cache on | cache off |
|---|---|---|
| geometry recording | **1976 us/frame** | 11000 us/frame |
| shadow draws | **13** | 3326 |
| cascades refitted | 0 | 4 every frame |

**82% less CPU recording time.** The image is identical either way: the same
FNV-1a hash of the scene colour with the cache on and off, on the Bistro
(`6bc3c6042f2bbb0d`) and on the fixture (`280c2fc835a964bd`), where the shadowed
floor sample also still reads far darker than the two lit ones. That is the
check worth running on a cache, because stale depth does not crash -- it renders
a shadow that is merely in the wrong place.

What this does *not* do: the composite copies the whole 4096x4096 atlas every
frame a scene has movers, which is 64 MB of GPU copy traffic to restore the
footprint of seven small props. Narrowing it to the union of this frame's and
last frame's dynamic footprints is the obvious next step, and was left out
because the copy is GPU-side while the 9 ms recovered is CPU-side, so the
trade is already clearly worth it and the measurement to justify the extra
complexity has not been made.


### Baking the shadow into the surface

A shadow map is indexed by the **light's** view. A lightmap is indexed by the
**receiver's** surface. That is the whole difference, and it is what makes a
baked result view-independent: once baked, a surface's shadow is a property of
the surface rather than of where the camera is.

`//Engine:Lightmap` packs one chart per static instance; `VSLightmapBake` /
`PSLightmapBake` fill them; the G-buffer pass samples the atlas and hands the
term to the lighting pass in `Material.w`.

#### What the asset has to provide

Addressing a surface needs a parameterisation of it: a UV set that covers the
surface without overlapping itself. **`TEXCOORD_0` will not do.** It is a
texturing UV, and measured on the Bistro by reading the real vertex data,
**58% of sampled primitives have UVs outside [0,1]** -- they tile, so they
address the same texel from many places on the mesh. Written into, one wall's
shadow would appear on every wall sharing the material.

So the bake consumes `TEXCOORD_1`, glTF's second UV set, and the Bistro as
shipped has none:

```
lightmap: 0 chart(s) of 0 static instance(s), 0 m2 at 8 texels/m
lightmap: 2902 static instance(s) have no TEXCOORD_1 and cannot be baked
```

That is the engine reporting the one missing piece rather than failing quietly.
Generating those UVs is **Blender's job**, in two offline steps:

```bash
blender --background --python tools/blender_unwrap.py -- \
    C:/Code/niagara_bistro/bistro.gltf resources/lightmap-uv/bistro-uv.gltf

python tools/harvest_lightmap_uv.py \
    C:/Code/niagara_bistro/bistro.gltf \
    resources/lightmap-uv/bistro-uv.gltf \
    resources/lightmap-uv/bistro-baked.gltf
```

The second file is a complete glTF to load with `PTFLIO_SCENE`, and the engine
needs to know nothing about any of this -- it reads `TEXCOORD_1`, which is now
simply there:

```
lightmap: 2902 chart(s) of 2902 static instance(s), 92950 m2 at 8 texels/m,
          4096 x 4096 atlas (54.3% used)
images:   343 loaded, 1166.9 MB, 0 file(s) absent, 0 failed to decode
```

Note that charts are per **instance**, not per mesh: the Bistro instances 2909
nodes from 551 meshes, one of them 84 times, and the same mesh in two places
receives two different shadows. An unwrap can be shared; a chart cannot. That
is why Blender unwraps per mesh *datablock* -- its importer gives 2909 objects
sharing 551 of them, matching the asset's own structure -- while the atlas
packing above is per instance.

#### Why Blender, and why a harvest rather than a round trip

UVAtlas was the first choice and was measured against: a median **0.2 ms per
face** and worse on larger meshes, so the Bistro's 1.75 million unique faces
came to about **six minutes** single-threaded. Blender does the same work in
**two and a half minutes** (18.6 s import, 6.6 s export) and needs nothing in
the Bazel build, so the UVAtlas path was deleted rather than kept as a second
unwrapper.

What Blender must **not** do is export the whole asset. A full glTF round trip
does not add a UV set, it rewrites the scene: the Bistro's
`KHR_materials_pbrSpecularGlossiness` materials come back as metallic-roughness,
bypassing the per-pixel conversion above, and its `MSFT_texture_dds` references
are dropped. So `tools/blender_unwrap.py` exports **geometry and hierarchy
only** -- no materials, no images -- and `tools/harvest_lightmap_uv.py` puts the
one thing worth keeping back into the original. The 161 distinct material tables
and 343 images above are the same numbers the untouched asset reports, which is
what says the materials came through rather than being reconstructed.

**Matching meshes is the part that is easy to get wrong**, and its failure looks
exactly like a mangled export. Three matchers were tried before one was right:

| keyed on | reported |
|---|---|
| mesh name | 520 of 551 "broken" |
| list position | 551 of 551 "broken" |
| face count + bounding box, first candidate | 3 of 40 paired wrongly |

All three were the matcher, not Blender. glTF does not require mesh names to be
unique and this asset does not make them so -- the Bistro has many meshes called
`subset_0` -- an import may reorder, and an architectural scene repeats props, so
many meshes share a shape and the first candidate with it is often the wrong one.
The wrong pairing is obvious once measured: a worst corner distance of 8e-2 on a
mesh whose entire extent is 0.03.

What works is to bucket by **face count**, which is an integer and cannot drift,
and then *prove* each pairing by walking the corners. 39 of 60 sampled meshes
matched on the first candidate and **20 only after searching**.
`tools/compare_uv_export.py` does exactly this and prints its own verdict, so
the claim is checkable rather than asserted.

Positions come back **bit-identical** -- worst corner distance 0.000e+00 across
60 sampled meshes, despite the float32-to-double and Y-up-to-Z-up-and-back round
trip. The tolerance in the comparison is belt and braces, not a necessity.

**The one real wrinkle: Blender discards degenerate faces on import.** The
Bistro goes from 1,753,630 faces to 1,752,986, losing 644 (0.037%) concentrated
in **6 of 551 primitives** -- and because those are large meshes, they are 4.5%
of the asset's faces. Their face order no longer corresponds, so an index-keyed
harvest would put the wrong UVs on them. The harvest therefore falls back to
matching faces on their **corner positions**, which does not depend on order,
and recovers all six:

```
matched by face order:      545
matched by face positions:  6
not matched (left as-is):   0
primitives rewritten:       551 of 551
```

A primitive that matched neither way would simply keep no `TEXCOORD_1` and fall
back to the shadow map, which is why the harvest reports that count rather than
failing.

#### Sizing and packing

A chart's edge comes from how much world surface its instance covers:
`sqrt(area) * texelsPerMetre`, plus a two-texel guard band, rounded up to a
power of two and clamped. Powers of two are not an aesthetic choice -- they let
a row of equal charts tile the atlas exactly, so the shelf packer wastes
nothing inside a row. The guard band is what stops a bilinear tap at one
chart's edge from reading the next one.

The Bistro's bill, under that exact rule over its 2909 instances and 80,480 m2
of face area:

| texels/m | texel size | texels needed | atlas | R8 |
|---|---|---|---|---|
| 2 | 50 cm | 1.06 M | 2048² | 4 MB |
| 4 | 25 cm | 3.43 M | 4096² | 16 MB |
| **8** | **12.5 cm** | **8.82 M** | **4096²** | **16 MB** |
| 16 | 6 cm | 27.2 M | 8192² | 64 MB |

8 per metre is the default: 12.5 cm sits between the cascades' 1.6 cm near and
24 cm far, and it fits a 4096² atlas with room to spare.

`//Engine:scenegraph-smoke` asserts the packing, because the failure mode is
not a crash. **Two charts overlapping by one texel makes one object wear
another object's shadow**, which reads as a baking bug and is an allocation
bug. So it checks that no two charts overlap, that each UV transform stays
inside its own chart's content, that an atlas too small reports it rather than
papering over it, and that what did fit is still internally consistent when
something did not. It also prints occupancy -- 55.5% on a representative size
mix -- as a guard against a packer that quietly degenerates.

#### The bake

Two passes, once, on the first frame that has a sun and something to bake:

```
LightmapSunDepth   static casters, whole scene, one orthographic sun view
LightmapBake       each charted instance rasterised into its own chart
```

The sun view is **not** a cascade. Cascades exist because a camera sees near
things in more detail than far ones; a lightmap texel anywhere can need any
caster, so the depth it is tested against has to cover everything at once.
`SceneSunViewProjection` fits the scene's bounding sphere, and the test asserts
every corner of the scene lands inside it -- a cascade may legitimately exclude
geometry outside its slice, this may not exclude anything. At 2048 over the
Bistro's 166-unit diagonal that resolves 4.1 cm, finer than the 12.5 cm
lightmap texels it feeds, so the bake is limited by the atlas rather than by
the depth map.

`VSLightmapBake` is the interesting half: it rasterises into the **lightmap UV**
rather than into the camera, so the triangle is drawn where it lives in its own
parameterisation, and the pixel shader asks whether the sun reaches that point
of the surface. Culling is off and there is no depth test -- the chart has to be
covered whichever way the triangle faces, and two triangles overlapping in UV
would be a broken unwrap, not something to resolve with a depth buffer.

#### Combining it with the cascades

`min`, not a product:

```hlsl
float DirectionalShadow = BakedSunVisibility;
if (BakedSunVisibility > 0.0)
{
    DirectionalShadow = min(BakedSunVisibility,
                            SampleShadow(ShadowCascade, WorldPosition, Normal));
}
```

The two estimates see largely the **same** static occluders -- the bake and the
cascade are both looking at the building that casts the shadow -- so
multiplying them would darken one shadow twice. For a shared occluder the
lesser of two visibilities is right, and where only one of them sees an
occluder at all, which is what happens for a moving caster since those are
never baked, the lesser is that one. Where the bake already says fully
shadowed, the cascade's 3x3 comparison kernel is skipped outright.

What `min` costs is that at a static shadow's penumbra the result is the darker
of the two estimates rather than purely the baked one, so the cascade still
bounds the edge quality there. Making the bake the sole authority for static
casters means splitting the shadow atlas into a static half and a movers-only
half and multiplying those instead -- different occluders, so a product is then
correct. That also deletes the 64 MB-per-frame composite copy. It is the
obvious next step and is not done.

#### Verified against a fixture, not an unwrapper

`tools/make_shadow_fixture.py` emits `TEXCOORD_1` as well. For a single quad the
texturing UVs already are a non-overlapping parameterisation, which is exactly
why a quad fixture can exercise the whole bake without Blender in the loop at
all -- and exactly why a real asset's tiled UVs cannot be reused this way.

The fixture prints its expectation in lightmap space, which is independent of
where the camera is:

```
  its shadow spans x 1.0 .. 3.0  ->  u 0.600 .. 0.800
```

Reading the baked atlas back:

```
[bake] floor chart: 600 dark texels, u 0.597 .. 0.798, v 0.403 .. 0.597
```

Both axes match. The v band is 0.4 to 0.6 because the occluder is a 2x2 quad,
so its shadow is bounded across the floor as well as along it -- 25 by 25 of
the chart's 124 texels, which is the 600. The occluder's own chart is fully lit,
correctly: its top faces the sun.

Then the end-to-end check, with the cascades switched **off** so the bake is the
only thing that can cast anything:

```
[screen] bake + cascades          lit 14739 14723  shadowed 10158
[screen] bake only, cascades off  lit 14739 14723  shadowed 10158
```

The shadow survives, identically. That is the whole chain -- bake, atlas,
G-buffer sample, `Material.w`, lighting, screen -- confirmed by removing the
other explanation rather than by looking at it.

#### Memory, honestly

This **adds** memory rather than saving it: 16 MB of atlas and 16 MB of sun
depth map, on top of the cascades' 128 MB, which stay because a moving object
receiving a shadow from a static caster still needs them. An earlier estimate
of 56 MB "versus" 128 MB was wrong -- it assumed the lightmap replaces the
cascades, and the movers prevent that. What the lightmap buys is
view-independent quality on static surfaces and a single bilinear fetch in
place of nine comparison taps wherever the bake has already settled the answer.

A baked result is also tied to one light: the sun direction is in the data, so
moving it invalidates the bake. The renderer notices and requests another, and
there is a **Rebake the lightmap** button next to the shadow controls.

### Checking the BRDF against the reference

`PSLighting`'s specular term is `V_GGX * D_GGX`, matching
[glTF-Sample-Renderer's `BRDF_specularGGX`](https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/main/source/Renderer/shaders/brdf.glsl).
`V_GGX` is the Smith joint **visibility**, which already carries the
`1 / (4 NdotL NdotV)` that a Cook-Torrance denominator would otherwise supply
— so there is deliberately no denominator in the shader. Getting that wrong is
worth knowing about because of how it fails: with a `G` term and no divide, the
specular is 4x too bright facing a light, and *only on metals*, because a
dielectric's 0.04 Fresnel damps the error to nothing. A scene of dielectrics
looks fine while its metals blow out.

`tools/make_brdf_fixture.py` writes four quads and one point light whose shaded
values are closed-form, and prints them. Put the camera at (0, 0, 3) looking at
the origin and the quads must match:

| quad | expected | before the fix |
|---|---|---|
| metal, roughness 0.4 | 81 | 122 |
| dielectric, roughness 0.5 | 71 | 77 |
| metal, roughness 0.2 | 150 | 207 |
| dielectric, roughness 0.9 | 69 | 69 |

The script models the whole chain, including that metallic and roughness pass
through an 8-bit G-buffer and that `PSLighting` ends with Reinhard and a 1/2.2
gamma, so the numbers are byte values on screen rather than radiance.

`GeometrySmith` (Schlick-GGX with UE4's `k = (r+1)^2/8`) is kept in the file,
unused, for comparison: measured against `V_GGX * D_GGX` it agrees to 1.00x at
normal incidence but falls to 0.66x at 60 degrees and 0.31x at grazing, so it
darkens silhouettes.

### Where the shading lives

`Engine/shaders/no_texture.fx` holds every entry pair -- `GBuffer`,
`Lighting`, `Skybox`, `Tonemap`, `Shadow`, `LightmapBake` and `DebugView` -- so
they share the BRDF helpers,
the light struct and the material constants. The first split is exactly the old
forward pixel shader cut in half: `PSGBuffer` is its material gather (texture
sampling, normal mapping, occlusion, emissive) writing to four targets instead
of consuming the values, and `PSLighting` is its shading half reading them back.
Neither half was rewritten.

Their constant buffers sit at different registers (`b0` for the geometry pass,
`b3` for lighting) because every entry point lives in one file and two
`cbuffer`s at `b0` collide at declaration even when no single entry point uses
both. The root signature decides what is actually bound, so the numbers only
have to be distinct. `VSShadow` is `VSGBuffer` under another name, reading the
same `b0` with the light's matrix where the camera's would go.

Distinct *per entry point*, not globally: `t5` is the lightmap atlas to
`PSGBuffer` and the base colour target to `PSLighting`, and `b4` is the bake's
constants to one pair and the sky's to another. dxc compiles one entry point at
a time and only binds what that entry point can reach, so two declarations at
one register are a collision only if some single entry point reaches both. The
debug view took the next free numbers all the same -- `b5`, `t14`, `s5` -- since
nothing is gained by reusing one.

### The model work runs on its own compute queue

`ModelManager` creates an `ID3D12CommandQueue` of type
`D3D12_COMMAND_LIST_TYPE_COMPUTE` and both backends submit to it: the
`DirectMLGraph` runner's own command list, and the ONNX Runtime DirectML EP,
which accepts a DIRECT or a COMPUTE queue and so moves with it. The command
lists become `COMPUTE` lists to match, which they can because everything they
record -- DirectML dispatches, `CopyBufferRegion`, and barriers between
`COMMON`, `COPY_DEST`, `COPY_SOURCE` and `UNORDERED_ACCESS` -- is legal on a
compute queue. A render-target or pixel-shader-resource state would not be,
which is the constraint to remember if one of those lists ever has to touch a
texture the frame uses.

**What sharing the graphics queue was buying, and what it cost.** It bought
ordering for free: anything submitted after a model's work was guaranteed to
see its output, with no fence anywhere. It cost exactly what that guarantee is
worth -- model work and frame work could never overlap, because a dispatch sat
between two frames instead of beside one.

So the free ordering is gone and has to be asked for. `OrderAfterModelWork`
signals the manager's fence on the compute queue and makes the caller's queue
wait on it, GPU-side. Nothing in this project reads a model's output on the GPU
yet, which is the only reason the per-frame `EnqueueModel` in `Render` is
correct without it; the day a pass samples a model's buffer, that call is the
whole difference between a correct frame and a race that only appears under
load. `RecordModelDispatch` is deliberately left alone and still records into
the caller's list, because being ordered *with* the frame is the entire point
of folding a model into a render pass.

One fence for the manager rather than one per model: a wait is always "after
everything submitted so far", which is what a consumer of any model's output
needs, and a monotonic counter on one queue gives that for free. The destructor
drains the queue -- required, not tidy, because nothing else is going to wait
for it and releasing a resource a submitted list still references surfaces
later as a device removal with no obvious cause.

#### A latent bug the move exposed

`DmlGraphRunner::Enqueue` submitted its command list and never signalled its
fence. `BeginList` then had nothing to wait on, so the next `Enqueue` reset a
command allocator whose previous submission was still running. `Reset` returns
`S_OK` and corrupts the allocator; only the debug layer notices.

It was harmless while `Enqueue` was called about once per session. Calling it
every frame is not harmless, and the first thing the compute queue did was turn
it into 17 errors per run. The negative control -- removing the signal again on
purpose -- is what says the assertion that now covers it is load-bearing:

```
  no debug-layer errors from reusing the allocator MISMATCH
    ID3D12CommandAllocator::Reset: A command allocator ... is being reset
    before previous executions associated with the allocator have completed.
    (x17)
```

The allocators and lists are also named now, because that message identifies
the object and an unnamed one reads "Unnamed ID3D12CommandAllocator Object".
`OrtModelRunner` already drained correctly and already documented this exact
failure (`EXECUTION ERROR #552, COMMAND_ALLOCATOR_SYNC`), which is how the
cause was obvious the moment the errors appeared.

#### What this is and is not worth

Measured, and it is **not** a speed-up on anything here. Frame time with the
per-frame dispatch on the compute queue, on the graphics queue (the old
behaviour) and absent entirely:

| | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|
| no dispatch | 0.249 | 0.257 | 0.233 | 0.233-0.257 |
| graphics queue | 0.229 | 0.314 | 0.254 | 0.229-0.314 |
| compute queue | 0.243 | 0.227 | 0.294 | 0.227-0.294 |

All in milliseconds. Every column overlaps every other, each condition's own
run-to-run spread is wider than any difference between them, and the fastest
single run of the nine belongs to the *old* graphics-queue path. These are
noise. The reason is that the registered model is a single 3 x 256 x 256
convolution and the fixture frame is a quad: there is no contention to relieve.
What the numbers do say is that the extra queue and its fence cost nothing
measurable either.

Note that the first attempt at this measurement returned `16.667 ms` for all
six runs, to three decimal places. That is vsync, not a result; presentation
pins the frame and no sub-millisecond difference can survive it. The numbers
above come from presenting with `c_AllowTearing` and a sync interval of 0.

#### FloodDiffusion has its own queue too

`FloodDiffusionPipeline` creates and owns a second compute queue rather than
sharing `ModelManager`'s. Its submissions are a 70 ms generation; the
manager's are a per-frame dispatch, and queueing the second behind the first
would be most of what putting either on a compute queue was for. `Load` no
longer takes a command queue at all -- a caller cannot put this work on the
graphics queue, which is the point, because on the renderer's queue those
seventy milliseconds are frames.

**What the fence is, and is not, for.** Every tensor this pipeline binds is in
*host* memory, so `Step()` already returns with its results on the CPU: ONNX
Runtime does the copy and the wait itself, and no fence of ours could make the
floats appear sooner. `Frame()`, `FeatureFrame()` and the retargeting read
ordinary `std::vector`s and need no synchronisation whatever. Adding a fence
"to sync the output" would be theatre.

What the fence does cover is the two things queue order used to and no longer
does:

- **Lifetime.** `Load` is callable again to change provider, and it destroys
  every ONNX Runtime session -- releasing resources whose submissions may still
  be running. On the renderer's queue the frame's own drains hid that. Nothing
  drains a private queue, so `Load` and the destructor do, and the `Impl`
  declares the queue *above* the sessions so reverse-order destruction keeps it
  alive longer than they are.
- **Cross-queue ordering**, through `OrderAfterWork(queue)`, for whatever reads
  this on the GPU next. Nothing does today.

The queue is created once and kept across provider changes: a fence value
already handed out has to stay meaningful, and recreating the queue would reset
the timeline under anyone holding one. On the CPU path there is no queue, and
both calls are deliberately successful no-ops rather than failures every caller
would have to special-case.

**The output is unchanged**, which is the check that matters. The existing
CPU-against-DirectML comparison in `//Engine:flooddiffusion-smoke` runs the
same prompt and seed through both providers and now has DirectML on the compute
queue:

```
largest single-coordinate difference   0.000004
mean absolute difference               0.000000
largest coordinate magnitude           2.524
```

Sixteen further checks cover the queue itself: that it is `COMPUTE` and is not
the `DIRECT` queue the test still creates as a stand-in for the renderer's,
that a generation completes on it, that both synchronisation calls work and
refuse a null queue, that reloading twice over live sessions leaves the
pipeline usable and the queue identity intact, and that the whole thing is
silent on the debug layer -- which the test enables for exactly this, since
releasing a session mid-flight is otherwise invisible.

**It is not faster, and the queue type is not why it is slow.** A/B of the same
build on the same machine, reading the pinned-shape generation time:

| queue type | run 1 | run 2 | run 3 |
|---|---|---|---|
| DIRECT | 73.6 ms | 73.1 ms | 71.3 ms |
| COMPUTE | 73.4 ms | 71.5 ms | 71.3 ms |

Indistinguishable -- the pairs differ by less than the drift across runs as the
machine warms, and the third pair is identical to a tenth of a millisecond. (The 61 ms recorded further up this file was an
earlier measurement on a different machine state; the DirectML path's cost is
per-dispatch overhead, as the table at the top of `FloodDiffusion.hpp` sets
out, and which queue it submits to does not touch that.)

So the case for this change is structural rather than numeric -- it removes a
serialization point that would have had to be removed before any model big
enough to matter could overlap a frame -- and the correctness evidence is the
part that is real: every numeric inference assertion in
`//Engine:modelmanager-smoke` now runs on the compute queue and still agrees
with the arithmetic, including both backends against each other, and ~1500
frames of per-frame enqueuing in a `-c dbg` build with the debug layer's
break-on-error armed produce nothing.

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
