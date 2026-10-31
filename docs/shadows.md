# Shadows and baked lightmaps

Cascaded shadow maps, the cached static half, and baking the sun's visibility
into the surface — the two shadow stages of the frame the models run in. Back
to the [README](../README.md); the write-up is [AI-Integrated Renderer](https://jack111331.github.io/BlogPage/ai_render_engine/).

## Cascaded shadow maps

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

### Caching the static half

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

## Baking the shadow into the surface

A shadow map is indexed by the **light's** view. A lightmap is indexed by the
**receiver's** surface. That is the whole difference, and it is what makes a
baked result view-independent: once baked, a surface's shadow is a property of
the surface rather than of where the camera is.

`//Engine:Lightmap` packs one chart per static instance; `VSLightmapBake` /
`PSLightmapBake` fill them; the G-buffer pass samples the atlas and hands the
term to the lighting pass in `Material.w`.

### What the asset has to provide

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

### Why Blender, and why a harvest rather than a round trip

UVAtlas was the first choice and was measured against: a median **0.2 ms per
face** and worse on larger meshes, so the Bistro's 1.75 million unique faces
came to about **six minutes** single-threaded. Blender does the same work in
**two and a half minutes** (18.6 s import, 6.6 s export) and needs nothing in
the Bazel build, so the UVAtlas path was deleted rather than kept as a second
unwrapper.

What Blender must **not** do is export the whole asset. A full glTF round trip
does not add a UV set, it rewrites the scene: the Bistro's
`KHR_materials_pbrSpecularGlossiness` materials come back as metallic-roughness,
bypassing the per-pixel conversion in [Assets](assets.md#converting-specular-glossiness), and its `MSFT_texture_dds` references
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

### Sizing and packing

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

### The bake

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

### Combining it with the cascades

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

### Verified against a fixture, not an unwrapper

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

### Memory, honestly

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

