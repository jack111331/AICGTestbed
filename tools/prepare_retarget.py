"""Converts a KeeMap bone-mapping file into the flat form the engine reads.

    python tools/prepare_retarget.py <mapping.json> --out resources/retarget/mixamo.txt

The input is a bone mapping saved by the KeeMap Blender retargeting addon
(nkeeline/Keemap-Blender-Rig-ReTargeting-Addon). momask-codes ships one at
assets/mapping.json that maps a HumanML3D skeleton onto a Mixamo rig, and its
README uses exactly that addon to animate Mixamo characters from generated
motion:

    https://github.com/EricGuo5513/momask-codes/tree/main/assets

Two things this conversion settles, both of which matter and neither of which
is obvious from the file:

* **Which correction the addon actually uses.** Each bone carries BOTH Euler
  (CorrectionFactorX/Y/Z) and quaternion (QuatCorrectionFactor*) offsets, and
  the addon picks between them on the file's `bone_rotation_mode`. In EULER
  mode -- which momask's mapping uses -- it builds the correction from the
  EULER fields and ignores the quaternion ones entirely. Emitting the wrong
  one silently mis-orients every bone, so this script resolves it here and
  writes a single correction per bone.
* **Which source joint each bone is.** The addon addresses bones by name in a
  Blender armature. The engine has no armature for the source, only the 22
  HumanML3D joints, so the names are resolved to joint indices against the
  table below and an unknown name is reported rather than skipped.

Output is `key value` lines, matching tools/prepare_flooddiffusion.py, so the
engine needs no JSON parser.
"""
import argparse
import json
import math
import os
import sys

# The HumanML3D 22-joint skeleton, in the order the feature vector uses. Names
# follow the BVH that momask exports, which is what mapping.json's
# SourceBoneName refers to. The chains are utils/paramUtil.py's
# t2m_kinematic_chain.
JOINT_NAMES = [
    "Hips",          # 0
    "LeftUpLeg",     # 1
    "RightUpLeg",    # 2
    "Spine",         # 3
    "LeftLeg",       # 4
    "RightLeg",      # 5
    "Spine1",        # 6
    "LeftFoot",      # 7
    "RightFoot",     # 8
    "Spine2",        # 9
    "LeftToeBase",   # 10
    "RightToeBase",  # 11
    "Neck",          # 12
    "LeftShoulder",  # 13
    "RightShoulder", # 14
    "Head",          # 15
    "LeftArm",       # 16
    "RightArm",      # 17
    "LeftForeArm",   # 18
    "RightForeArm",  # 19
    "LeftHand",      # 20
    "RightHand",     # 21
]

CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

# Which joint each BONE points at.
#
# This is the distinction that decides whether a retarget works at all.
# HumanML3D's rotation at joint j orients the bone ENDING at j: forward
# kinematics computes position[j] = position[parent] + R(j) * offset[j]. A rig's
# bone named j orients the bone STARTING at j. So driving Mixamo's "LeftForeArm"
# needs the rotation HumanML3D stores at LeftHand, not at LeftForeArm.
#
# Measured against the generated motion: using each joint's own rotation leaves
# bone directions about 90 degrees out, i.e. uncorrelated; using the child's
# brings the legs and spine to within a few degrees, and adding rest-direction
# alignment brings every bone to 0.000 degrees.
#
# Two joints sit in more than one chain, and the choice is anatomical rather
# than whichever chain happens to be listed first: the bones Mixamo calls Hips
# and Spine2 point UP THE SPINE, not down a leg or out along an arm.
BONE_CHILD = {
    0: 3,    # Hips    -> Spine, not LeftUpLeg
    3: 6,    # Spine   -> Spine1
    6: 9,    # Spine1  -> Spine2
    9: 12,   # Spine2  -> Neck, not a shoulder
    12: 15,  # Neck    -> Head
    1: 4, 4: 7, 7: 10,       # left leg
    2: 5, 5: 8, 8: 11,       # right leg
    13: 16, 16: 18, 18: 20,  # left arm
    14: 17, 17: 19, 19: 21,  # right arm
    # 15 (Head) is a leaf: HumanML3D has no joint beyond it, so there is no
    # direction to point it at and it keeps its rest orientation.
}

# The rest direction of each bone in the SOURCE skeleton, from
# utils/paramUtil.py's t2m_raw_offsets. Needed because the source rests with its
# arms DOWN while a Mixamo character rests in a T-pose, and that difference has
# to be cancelled per bone.
RAW_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, -1, 0),
    (0, 1, 0), (0, -1, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1),
    (0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, -1, 0), (0, -1, 0),
    (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0),
]


def euler_xyz_to_quaternion(x, y, z):
    """Blender's Euler('XYZ').to_quaternion().

    Blender composes an XYZ Euler as R = Rz @ Ry @ Rx acting on column
    vectors, i.e. X is applied first. As quaternions that is qz * qy * qx.
    Returned as (w, x, y, z).
    """
    def axis(angle, ax):
        half = angle * 0.5
        s = math.sin(half)
        q = [math.cos(half), 0.0, 0.0, 0.0]
        q[ax + 1] = s
        return q

    def multiply(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]

    return multiply(multiply(axis(z, 2), axis(y, 1)), axis(x, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mapping", help="a KeeMap bone mapping .json")
    parser.add_argument("--out", default="resources/retarget/mixamo.txt")
    parser.add_argument("--source", default="momask-codes assets/mapping.json",
                        help="recorded in the output for provenance")
    args = parser.parse_args()

    with open(args.mapping, encoding="utf-8") as handle:
        spec = json.load(handle)

    mode = spec.get("bone_rotation_mode", "EULER")
    if mode not in ("EULER", "QUATERNION"):
        raise SystemExit("unknown bone_rotation_mode %r" % mode)

    index_of = {name: i for i, name in enumerate(JOINT_NAMES)}
    rows = []
    unknown = []
    for bone in spec.get("bones", []):
        source = bone["SourceBoneName"]
        if source not in index_of:
            unknown.append(source)
            continue
        if mode == "EULER":
            quat = euler_xyz_to_quaternion(bone["CorrectionFactorX"],
                                           bone["CorrectionFactorY"],
                                           bone["CorrectionFactorZ"])
        else:
            quat = [bone["QuatCorrectionFactorw"], bone["QuatCorrectionFactorx"],
                    bone["QuatCorrectionFactory"], bone["QuatCorrectionFactorz"]]
        rows.append({
            "joint": index_of[source],
            "source": source,
            "destination": bone["DestinationBoneName"],
            "quat": quat,
            "rotation": 1 if bone.get("set_bone_rotation", True) else 0,
            "position": 1 if bone.get("set_bone_position", False) else 0,
        })

    if unknown:
        raise SystemExit("these source bones are not HumanML3D joints: %s"
                         % ", ".join(sorted(set(unknown))))
    if not rows:
        raise SystemExit("the mapping produced no usable bones")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Written by tools/prepare_retarget.py from\n")
        handle.write("# %s\n" % args.source)
        handle.write("# Correction resolved from the %s fields, which is what the\n"
                     "# KeeMap addon uses in %s mode.\n"
                     % ("Euler" if mode == "EULER" else "quaternion", mode))
        handle.write("joints %d\n" % len(JOINT_NAMES))
        handle.write("rotation_mode %s\n" % mode)
        for parent_chain in CHAINS:
            handle.write("chain %s\n" % " ".join(str(j) for j in parent_chain))
        destination_of = {row["joint"]: row["destination"] for row in rows}
        for row in rows:
            child = BONE_CHILD.get(row["joint"], -1)
            # The child's node name. Taken from the mapping when the child is
            # itself mapped; otherwise built with the same prefix the parent's
            # destination uses, so nothing is assumed about naming beyond what
            # this file already states.
            child_destination = "-"
            offset = (0.0, 0.0, 0.0)
            if child >= 0:
                if child in destination_of:
                    child_destination = destination_of[child]
                elif row["destination"].endswith(row["source"]):
                    prefix = row["destination"][:len(row["destination"]) -
                                                len(row["source"])]
                    child_destination = prefix + JOINT_NAMES[child]
                else:
                    child_destination = JOINT_NAMES[child]
                offset = RAW_OFFSETS[child]
            # bone <jointIndex> <sourceName> <destinationName> <setRot> <setPos>
            #      <qw> <qx> <qy> <qz> <childJoint> <childDestination>
            #      <restDirX> <restDirY> <restDirZ>
            handle.write("bone %d %s %s %d %d %.9g %.9g %.9g %.9g %d %s "
                         "%g %g %g\n"
                         % (row["joint"], row["source"], row["destination"],
                            row["rotation"], row["position"],
                            row["quat"][0], row["quat"][1], row["quat"][2],
                            row["quat"][3], child, child_destination,
                            offset[0], offset[1], offset[2]))

    print("%s: %d bones, rotation mode %s" % (args.out, len(rows), mode))
    for row in rows:
        angle = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(row["quat"][0])))))
        child = BONE_CHILD.get(row["joint"], -1)
        print("  %-2d %-16s -> %-26s points at %-14s%s"
              % (row["joint"], row["source"], row["destination"],
                 JOINT_NAMES[child] if child >= 0 else "(leaf)",
                 "  +position" if row["position"] else ""))
        del angle
    return 0


if __name__ == "__main__":
    sys.exit(main())
