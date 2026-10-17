# Unwraps a glTF's meshes in Blender and writes a lightmap UV set.
#
# Run headless:
#
#     blender --background --python tools/blender_unwrap.py -- \
#         <in.gltf> <out.gltf> [--angle-limit 66] [--island-margin 0.02]
#
# A baked lightmap is addressed by the receiver's surface, so it needs a UV set
# that covers the surface without overlapping. glTF's TEXCOORD_0 is a texturing
# UV and on a converted scene it tiles, so it cannot serve. This adds a second
# UV layer, which the glTF exporter writes as TEXCOORD_1 -- exactly what the
# renderer reads.
#
# Blender's Smart UV Project is used rather than Lightmap Pack: the latter packs
# every face as its own island, which is robust but spends the atlas on seams,
# while Smart UV Project cuts by angle and keeps contiguous surfaces together.
#
# This writes a SEPARATE output file and never touches the input.
import os
import sys


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    if len(argv) < 2:
        raise SystemExit(
            "usage: blender --background --python blender_unwrap.py -- "
            "<in.gltf> <out.gltf> [--angle-limit D] [--island-margin M]")
    options = {
        "input": argv[0],
        "output": argv[1],
        "angle_limit": 66.0,
        "island_margin": 0.02,
    }
    rest = argv[2:]
    for i in range(0, len(rest) - 1, 2):
        if rest[i] == "--angle-limit":
            options["angle_limit"] = float(rest[i + 1])
        elif rest[i] == "--island-margin":
            options["island_margin"] = float(rest[i + 1])
    return options


def main():
    import bpy  # only available inside Blender
    import math

    options = parse_args(sys.argv)
    source = os.path.abspath(options["input"])
    target = os.path.abspath(options["output"])
    if os.path.normcase(source) == os.path.normcase(target):
        raise SystemExit("refusing to overwrite the input asset")

    # Start from genuinely empty, not from the startup cube.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    print("importing %s" % source)
    bpy.ops.import_scene.gltf(filepath=source)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    print("imported %d mesh object(s), %d mesh datablock(s)"
          % (len(meshes), len(bpy.data.meshes)))

    # Unwrap per mesh DATABLOCK, not per object: an instanced asset shares its
    # meshes between many objects, and unwrapping the shared data once is both
    # correct and far faster. The per-instance placement in the atlas is the
    # renderer's job, not Blender's.
    done = 0
    skipped = 0
    for mesh in bpy.data.meshes:
        if len(mesh.polygons) == 0:
            skipped += 1
            continue
        # A second layer, left non-active for rendering so nothing else in the
        # file starts using it as the texturing UV.
        if "Lightmap" in mesh.uv_layers:
            layer = mesh.uv_layers["Lightmap"]
        else:
            layer = mesh.uv_layers.new(name="Lightmap")
        mesh.uv_layers.active = layer

        # Smart UV Project works on the selected faces of the active object in
        # edit mode, so the datablock is driven through a temporary object.
        holder = bpy.data.objects.new("unwrap_holder", mesh)
        bpy.context.collection.objects.link(holder)
        bpy.context.view_layer.objects.active = holder
        for other in bpy.context.selected_objects:
            other.select_set(False)
        holder.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        try:
            bpy.ops.uv.smart_project(
                angle_limit=math.radians(options["angle_limit"]),
                island_margin=options["island_margin"],
                correct_aspect=False,
                scale_to_bounds=True)
        except RuntimeError as error:
            print("  smart_project failed on %s: %s" % (mesh.name, error))
            skipped += 1
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.data.objects.remove(holder)
        done += 1
        if done % 50 == 0:
            print("  unwrapped %d/%d" % (done, len(bpy.data.meshes)))

    print("unwrapped %d mesh(es), skipped %d" % (done, skipped))

    print("exporting %s" % target)
    # Geometry and hierarchy only. Materials and images are deliberately NOT
    # exported: the renderer keeps using the ORIGINAL asset for those, and this
    # file exists only to carry the extra UV set. A full round-trip would also
    # rewrite the Bistro's KHR_materials_pbrSpecularGlossiness materials as
    # metallic-roughness and drop its MSFT_texture_dds references, which is a
    # change to the scene rather than an addition to it.
    bpy.ops.export_scene.gltf(
        filepath=target,
        export_format="GLTF_SEPARATE",
        export_materials="NONE",
        export_image_format="NONE",
        export_texcoords=True,
        export_normals=True,
        export_tangents=False,
        export_skins=False,
        export_morph=False,
        export_cameras=False,
        export_lights=False,
        export_apply=False,
        use_selection=False)
    print("done")


if __name__ == "__main__":
    main()
