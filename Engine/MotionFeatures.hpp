#pragma once

#include <DirectXMath.h>

#include <cstddef>
#include <vector>

// Decoding HumanML3D motion features into joint positions.
//
// FloodDiffusion's VAE decodes to the 263-dimensional feature vector HumanML3D
// uses, which is NOT joint positions -- it is a root trajectory expressed as
// per-frame velocities plus joint positions in a root-local frame. Drawing a
// skeleton means integrating the root and un-rotating the joints, which is
// what StreamJointRecovery does.
//
// Deliberately free of D3D12 and ONNX Runtime so it can be tested on its own;
// //Engine:motionfeatures-smoke checks it against values computed independently
// in Python from FloodDiffusion's utils/motion_process.py.
namespace NeuralModelIntegrateTestbed {

// Where each block sits inside one feature vector. HumanML3D packs, in order:
//
//   [0]                    root angular velocity about Y, radians per frame
//   [1:3]                  root linear velocity in XZ, root-local
//   [3]                    root height (absolute, not a velocity)
//   [4 : 4+(J-1)*3]        "ric": joint positions relative to the root, in the
//                          root-local frame, excluding the root itself
//   then (J-1)*6           joint rotations as 6D vectors   -- unused here
//   then J*3               local velocities                -- unused here
//   then 4                 foot contact flags              -- unused here
//
// Only the first two blocks are needed for a skeleton: the rotation block is an
// alternative parameterisation of the same pose, and FloodDiffusion's own
// recover_joint_positions_263 uses ric as "the most direct way to get the
// skeleton for animation".
struct MotionFeatureLayout {
    int joints = 0;
    int motionDim = 0;

    static MotionFeatureLayout ForJoints(int joints) {
        MotionFeatureLayout layout;
        layout.joints = joints;
        layout.motionDim = 4 + 9 * (joints - 1) + 3 * joints + 4;
        return layout;
    }

    // Solves the width back to a joint count. False when `motionDim` is not a
    // HumanML3D width at all, which is worth catching: a wrong joint count
    // reads past the ric block and produces a skeleton of noise rather than an
    // error.
    static bool FromMotionDim(int motionDim, MotionFeatureLayout* out);

    int RicOffset() const { return 4; }
    int RicCount() const { return (joints - 1) * 3; }
};

// Bone connectivity, as five chains of joint indices: right leg, left leg,
// spine to head, right arm, left arm. Consecutive entries within a chain are
// one bone. From FloodDiffusion's utils/render_skeleton.py.
const std::vector<std::vector<int>>& HumanML3DChains();

// Turns one feature frame at a time into joint positions, carrying the root
// integration across calls.
//
// The subtlety this exists to get right: the batch formulation shifts the
// velocity arrays by one frame before integrating, so frame i is placed using
// frame i-1's velocity. A streaming implementation therefore has to hold the
// PREVIOUS frame's velocities and apply those, not the current frame's. Using
// the current frame's instead still yields plausible-looking motion that drifts
// one frame ahead of the pose, which is why this is spelled out rather than
// left to the reader.
//
// Equivalent to FloodDiffusion's StreamJointRecovery263.
class StreamJointRecovery {
public:
    // `smoothingAlpha` is an exponential moving average over output positions:
    // 1 passes frames through untouched, smaller values smooth more. It is
    // applied after recovery, exactly as the reference does, so it never feeds
    // back into the root integration.
    explicit StreamJointRecovery(int joints, float smoothingAlpha = 1.0f);

    void Reset();

    // `frame` holds motionDim floats; `out` receives `joints` positions, the
    // root first. Returns false and writes nothing if `frame` is null.
    bool ProcessFrame(const float* frame, DirectX::XMFLOAT3* out);

    // The accumulated root yaw after the last ProcessFrame, in radians. The
    // rotation block of the feature vector holds joint rotations RELATIVE to
    // the root, so anything that wants world-space rotations -- retargeting
    // onto a rigged character -- needs this same accumulator, and must not
    // start its own.
    float RootAngle() const { return static_cast<float>(m_rootAngle); }

    const MotionFeatureLayout& Layout() const { return m_layout; }
    float SmoothingAlpha() const { return m_smoothingAlpha; }
    void SetSmoothingAlpha(float alpha);

private:
    MotionFeatureLayout m_layout;
    float m_smoothingAlpha = 1.0f;

    // Integrated in double, matching the reference, where these live in Python
    // floats and a float64 numpy array while the quaternion itself is float32.
    // Keeping that split matters over a long stream: the accumulator is the one
    // place error compounds.
    double m_rootAngle = 0.0;
    double m_rootPosition[3] = {0.0, 0.0, 0.0};

    float m_previousAngularVelocity = 0.0f;
    float m_previousLinearVelocity[2] = {0.0f, 0.0f};

    std::vector<DirectX::XMFLOAT3> m_smoothed;
    bool m_hasSmoothed = false;
};

}  // namespace NeuralModelIntegrateTestbed
