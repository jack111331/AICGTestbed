# Rendering

The graph the AI nodes sit in. Deferred shading: the G-buffer, pass
compilation, frustum culling, descriptor management and the BRDF. Back to the
[README](../README.md); the write-up is [AI-Integrated Renderer](https://jack111331.github.io/BlogPage/ai_render_engine/).

## Why deferred

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

## The G-buffer

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

## The render graph

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

**Where the model work meets this list.** The motion pipeline is not a pass in
it, and the ordering is worth being precise about: `Update` advances the sampler
and `PoseCharacter` writes the retargeted pose as overrides on the scene graph,
then `Render` calls `UpdateSceneTransforms` and only then `BuildFrameGraph`. So
a model's output reaches the frame one step before the graph exists, as node
transforms, and `GBuffer` draws a posed character without knowing a model
produced it. The inference itself is off the graphics queue entirely.

A pass that *consumed* a model's buffer on the GPU is the other shape this
supports: `RecordModelDispatch` records into the frame's own command list, which
is what being ordered *with* the frame is for, and `OrderAfterModelWork` is the
GPU-side wait for work already submitted to the manager's own queue. Nothing
here needs the second shape yet, which is the only reason the per-frame
`EnqueueModel` in `Render` is correct without a fence — see [Compute
queues](compute-queues.md).

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

## View frustum culling

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

### The bug culling exposed

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

### Committing descriptors in one call

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

### Material descriptor tables, built once

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

## Checking the BRDF against the reference

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

## Where the shading lives

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

