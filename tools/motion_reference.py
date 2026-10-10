"""Reference joint positions for //Engine:motionfeatures-smoke.

A second, independent transcription of FloodDiffusion's
utils/motion_process.py (recover_root_rot_pos and StreamJointRecovery263) in
numpy, so the C++ in Engine/MotionFeatures.cc is checked against something
other than itself. The expected numbers this prints are pasted into
Engine/motionfeatures-smoke.cc as literals; run it again after changing either
side.

    python tools/motion_reference.py

The synthetic input is deliberately made of small integers over 16, so every
value is exactly representable in float32 and C++ builds bit-identical input
from the same formula rather than needing a data file.
"""
import numpy as np

JOINTS = 22
MOTION_DIM = 4 + 9 * (JOINTS - 1) + 3 * JOINTS + 4  # 263


def feature(frame_index, component):
    """Matches Feature() in motionfeatures-smoke.cc."""
    return ((frame_index * MOTION_DIM + component) % 37 - 18) / 16.0


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def qrot(q, v):
    """utils/math/quaternion.py qrot, for a single quaternion and vector."""
    u = q[1:]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return (v + 2.0 * (q[0] * uv + uuv)).astype(np.float32)


def recover(frames):
    """StreamJointRecovery263 with smoothing off. frames: [N, 263] float32."""
    angle = 0.0                  # Python float, i.e. double
    position = np.zeros(3)       # float64, as in the reference
    previous_angular = 0.0
    previous_linear = np.zeros(2, dtype=np.float32)

    out = np.zeros((len(frames), JOINTS, 3), dtype=np.float32)
    for index, data in enumerate(frames):
        angular = np.float32(data[0])
        linear = data[1:3].astype(np.float32)

        angle += float(previous_angular)
        rotation = np.array([np.cos(angle), 0.0, np.sin(angle), 0.0], dtype=np.float32)
        inverse = qinv(rotation)

        velocity = np.array([previous_linear[0], 0.0, previous_linear[1]],
                            dtype=np.float32)
        position = position + qrot(inverse, velocity).astype(np.float64)

        root = np.array([position[0], data[3], position[2]], dtype=np.float32)
        out[index, 0] = root

        ric = data[4:4 + (JOINTS - 1) * 3].reshape(-1, 3).astype(np.float32)
        for joint in range(JOINTS - 1):
            world = qrot(inverse, ric[joint])
            out[index, joint + 1] = [world[0] + root[0], world[1], world[2] + root[2]]

        previous_angular = angular
        previous_linear = linear
    return out


def main():
    count = 12
    frames = np.array(
        [[feature(i, c) for c in range(MOTION_DIM)] for i in range(count)],
        dtype=np.float32)
    joints = recover(frames)

    print("// %d frames, %d joints; from tools/motion_reference.py" % (count, JOINTS))
    print("// frame, joint, x, y, z")
    for frame in (0, 1, 2, 5, 11):
        for joint in (0, 1, 9, 21):
            x, y, z = joints[frame, joint]
            print("    {%2d, %2d, %14.7ff, %14.7ff, %14.7ff},"
                  % (frame, joint, x, y, z))

    print()
    print("// root path, to show the integration is actually accumulating")
    for frame in range(count):
        x, y, z = joints[frame, 0]
        print("//   frame %2d root %9.5f %9.5f %9.5f" % (frame, x, y, z))

    # A property that holds regardless of either transcription: with no angular
    # velocity the heading never changes, so the root travels in a straight line
    # at the constant linear velocity, offset by one frame.
    straight = np.zeros((6, MOTION_DIM), dtype=np.float32)
    straight[:, 0] = 0.0      # no yaw velocity
    straight[:, 1] = 0.25     # constant X velocity
    straight[:, 2] = -0.5     # constant Z velocity
    straight[:, 3] = 1.0      # constant height
    path = recover(straight)[:, 0]
    print()
    print("// constant velocity (0.25, -0.5), no rotation: a straight line one "
          "frame behind")
    for frame in range(len(path)):
        print("//   frame %d root %7.4f %7.4f %7.4f" % (frame, *path[frame]))


if __name__ == "__main__":
    main()
