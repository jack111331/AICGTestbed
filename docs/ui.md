# The ImGui panel

The four tabs, how the camera and the panel share one mouse, the sun override,
and the debug views. Back to the [README](../README.md).

## The panel

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

## Sharing the mouse with the panel

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

## Steering the sun

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

## Looking at an intermediate

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

