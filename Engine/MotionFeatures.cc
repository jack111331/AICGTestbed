#include "pch.h"
#include "MotionFeatures.hpp"

#include <algorithm>
#include <cmath>

namespace NeuralModelIntegrateTestbed {

namespace {

// Quaternions here follow the reference's layout, (w, x, y, z), with the root
// rotation always about Y: q = (cos a, 0, sin a, 0).
struct Quaternion {
    float w, x, y, z;
};

Quaternion Conjugate(const Quaternion& q) {
    return {q.w, -q.x, -q.y, -q.z};
}

// v + 2 * (w * (u x v) + u x (u x v)), with u the vector part. This is the
// expansion utils/math/quaternion.py uses; writing it out rather than calling
// XMQuaternionRotation keeps the arithmetic identical to the reference.
void Rotate(const Quaternion& q, const float v[3], float out[3]) {
    const float ux = q.x, uy = q.y, uz = q.z;
    const float cx = uy * v[2] - uz * v[1];
    const float cy = uz * v[0] - ux * v[2];
    const float cz = ux * v[1] - uy * v[0];
    const float ccx = uy * cz - uz * cy;
    const float ccy = uz * cx - ux * cz;
    const float ccz = ux * cy - uy * cx;
    out[0] = v[0] + 2.0f * (q.w * cx + ccx);
    out[1] = v[1] + 2.0f * (q.w * cy + ccy);
    out[2] = v[2] + 2.0f * (q.w * cz + ccz);
}

}  // namespace

bool MotionFeatureLayout::FromMotionDim(int motionDim, MotionFeatureLayout* out) {
    if (out == nullptr || motionDim < 4 + 3 + 4) {
        return false;
    }
    // motionDim = 4 + 9*(J-1) + 3*J + 4  =>  J = (motionDim + 1) / 12
    const int joints = (motionDim + 1) / 12;
    const MotionFeatureLayout layout = MotionFeatureLayout::ForJoints(joints);
    if (layout.motionDim != motionDim) {
        return false;
    }
    *out = layout;
    return true;
}

const std::vector<std::vector<int>>& HumanML3DChains() {
    static const std::vector<std::vector<int>> chains = {
        {0, 2, 5, 8, 11},       // right leg
        {0, 1, 4, 7, 10},       // left leg
        {0, 3, 6, 9, 12, 15},   // spine, neck, head
        {9, 14, 17, 19, 21},    // right arm
        {9, 13, 16, 18, 20},    // left arm
    };
    return chains;
}

StreamJointRecovery::StreamJointRecovery(int joints, float smoothingAlpha)
    : m_layout(MotionFeatureLayout::ForJoints(joints)) {
    SetSmoothingAlpha(smoothingAlpha);
    Reset();
}

void StreamJointRecovery::SetSmoothingAlpha(float alpha) {
    m_smoothingAlpha = std::clamp(alpha, 0.0f, 1.0f);
}

void StreamJointRecovery::Reset() {
    m_rootAngle = 0.0;
    m_rootPosition[0] = m_rootPosition[1] = m_rootPosition[2] = 0.0;
    m_previousAngularVelocity = 0.0f;
    m_previousLinearVelocity[0] = m_previousLinearVelocity[1] = 0.0f;
    m_smoothed.assign(static_cast<std::size_t>(std::max(m_layout.joints, 0)),
                      DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f));
    m_hasSmoothed = false;
}

bool StreamJointRecovery::ProcessFrame(const float* frame, DirectX::XMFLOAT3* out) {
    if (frame == nullptr || out == nullptr || m_layout.joints <= 0) {
        return false;
    }

    // Read this frame's velocities before anything is integrated: they apply to
    // the NEXT frame, not this one.
    const float angularVelocity = frame[0];
    const float linearVelocity[2] = {frame[1], frame[2]};

    // The previous frame's angular velocity advances the heading first, so the
    // quaternion below is the one the batch form would use for this frame.
    m_rootAngle += static_cast<double>(m_previousAngularVelocity);

    const Quaternion rootRotation = {
        static_cast<float>(std::cos(m_rootAngle)), 0.0f,
        static_cast<float>(std::sin(m_rootAngle)), 0.0f};
    const Quaternion inverseRotation = Conjugate(rootRotation);

    // Root translation: the previous frame's XZ velocity, rotated out of the
    // root-local frame and accumulated.
    const float localVelocity[3] = {m_previousLinearVelocity[0], 0.0f,
                                    m_previousLinearVelocity[1]};
    float worldVelocity[3] = {0.0f, 0.0f, 0.0f};
    Rotate(inverseRotation, localVelocity, worldVelocity);
    m_rootPosition[0] += static_cast<double>(worldVelocity[0]);
    m_rootPosition[1] += static_cast<double>(worldVelocity[1]);
    m_rootPosition[2] += static_cast<double>(worldVelocity[2]);

    // Height is stored absolutely, so it replaces rather than accumulates. The
    // Y component of the integrated velocity is discarded here, exactly as the
    // reference discards it.
    const float rootX = static_cast<float>(m_rootPosition[0]);
    const float rootY = frame[3];
    const float rootZ = static_cast<float>(m_rootPosition[2]);

    out[0] = DirectX::XMFLOAT3(rootX, rootY, rootZ);

    const float* ric = frame + m_layout.RicOffset();
    for (int joint = 1; joint < m_layout.joints; ++joint) {
        const float local[3] = {ric[(joint - 1) * 3 + 0], ric[(joint - 1) * 3 + 1],
                                ric[(joint - 1) * 3 + 2]};
        float world[3] = {0.0f, 0.0f, 0.0f};
        Rotate(inverseRotation, local, world);
        // Only XZ picks up the root offset; ric already carries absolute height.
        out[joint] = DirectX::XMFLOAT3(world[0] + rootX, world[1], world[2] + rootZ);
    }

    if (m_smoothingAlpha < 1.0f) {
        const std::size_t count = static_cast<std::size_t>(m_layout.joints);
        if (!m_hasSmoothed) {
            std::copy(out, out + count, m_smoothed.begin());
            m_hasSmoothed = true;
        } else {
            const float alpha = m_smoothingAlpha;
            const float rest = 1.0f - alpha;
            for (std::size_t i = 0; i < count; ++i) {
                out[i].x = alpha * out[i].x + rest * m_smoothed[i].x;
                out[i].y = alpha * out[i].y + rest * m_smoothed[i].y;
                out[i].z = alpha * out[i].z + rest * m_smoothed[i].z;
                m_smoothed[i] = out[i];
            }
        }
    }

    m_previousAngularVelocity = angularVelocity;
    m_previousLinearVelocity[0] = linearVelocity[0];
    m_previousLinearVelocity[1] = linearVelocity[1];
    return true;
}

}  // namespace NeuralModelIntegrateTestbed
