# Scenes and the scene manifest

How several glTF files are merged into one scene, what `scenes.json` holds,
and how the camera is pinned. Back to the [README](../README.md).

## Merging several glTF files

Several glTF files are **merged into one scene**. The manifest says which, and
where each one goes:

```bash
cp scenes.example.json scenes.json
bazelisk run //Engine:AICGTestbed
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

