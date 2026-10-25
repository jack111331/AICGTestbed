#include "pch.h"
#include "MotionRetarget.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

// NOTE on composition order: SimpleMath's Quaternion::Concatenate(a, b)
// applies B FIRST and then a -- the opposite of what its name suggests and
// of DirectXMath's own XMQuaternionMultiply. //Engine:scenegraph-smoke pins
// this down in TestConcatenateOrder, because getting it backwards does not
// crash or produce garbage: it leaves a retargeted character folded and
// floating, which is how it was found.

namespace NeuralModelIntegrateTestbed {

namespace {

using DirectX::SimpleMath::Matrix;
using DirectX::SimpleMath::Quaternion;
using DirectX::SimpleMath::Vector3;

void SetError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

// HumanML3D's cont6d_to_matrix: the six numbers are two 3-vectors that
// Gram-Schmidt turns into an orthonormal frame, whose COLUMNS are the
// rotation. Reconstructing rather than trusting them is the point of the
// representation -- a network's raw output is only approximately orthonormal.
Quaternion Cont6dToQuaternion(const float* c) {
    Vector3 xRaw(c[0], c[1], c[2]);
    Vector3 yRaw(c[3], c[4], c[5]);

    const float xLength = xRaw.Length();
    if (xLength < 1e-8f) {
        return Quaternion::Identity;
    }
    Vector3 x = xRaw / xLength;

    Vector3 z = x.Cross(yRaw);
    const float zLength = z.Length();
    if (zLength < 1e-8f) {
        // The two vectors are parallel, so they describe no frame. Identity is
        // the honest answer; silently producing a NaN rotation would propagate
        // through the whole chain.
        return Quaternion::Identity;
    }
    z /= zLength;
    const Vector3 y = z.Cross(x);

    // Columns [x, y, z] of a column-vector rotation. DirectXMath is
    // row-vector, so the transpose of that is the matrix whose ROWS are
    // x, y, z -- which is what building it row-wise here gives.
    Matrix m = Matrix::Identity;
    m._11 = x.x; m._12 = x.y; m._13 = x.z;
    m._21 = y.x; m._22 = y.y; m._23 = y.z;
    m._31 = z.x; m._32 = z.y; m._33 = z.z;
    return Quaternion::CreateFromRotationMatrix(m);
}

// The node's authored local rotation, for nodes a retarget does not drive.
Quaternion AuthoredRotation(const SceneNode& node) {
    if (node.hasTrs) {
        return node.baseTransform.rotation;
    }
    // Matrix-only node: recover the rotation. Decompose is non-const, so this
    // works on a copy.
    Matrix local = node.localTransform;
    Vector3 scale;
    Quaternion rotation;
    Vector3 translation;
    if (!local.Decompose(scale, rotation, translation)) {
        return Quaternion::Identity;
    }
    return rotation;
}

// Local transform from a node's AUTHORED transform, so measurements and the
// parent chain do not depend on whatever pose the graph currently holds.
Matrix AuthoredLocal(const SceneNode& node) {
    if (!node.hasTrs) {
        return node.localTransform;
    }
    return Matrix::CreateScale(node.baseTransform.scale) *
           Matrix::CreateFromQuaternion(node.baseTransform.rotation) *
           Matrix::CreateTranslation(node.baseTransform.translation);
}

}  // namespace

Quaternion MotionRetargeter::AlignDirections(const Vector3& from,
                                            const Vector3& to) {
    const float dot = from.Dot(to);
    if (dot > 0.999999f) {
        return Quaternion::Identity;
    }
    if (dot < -0.999999f) {
        // Opposed: a half turn about any perpendicular axis. Picking the axis
        // from whichever component is smallest keeps the cross product well
        // conditioned.
        Vector3 axis = std::fabs(from.x) < 0.9f ? Vector3::UnitX : Vector3::UnitY;
        axis = from.Cross(axis);
        axis.Normalize();
        return Quaternion::CreateFromAxisAngle(axis, DirectX::XM_PI);
    }
    Vector3 axis = from.Cross(to);
    axis.Normalize();
    return Quaternion::CreateFromAxisAngle(axis, std::acos(std::clamp(dot, -1.0f, 1.0f)));
}

Quaternion MotionRetargeter::FrameRotation(const Vector3& fromPrimary,
                                           const Vector3& fromSecondary,
                                           const Vector3& toPrimary,
                                           const Vector3& toSecondary) {
    // Built as "aim, then twist" rather than as the difference of two
    // orthonormal frames. The frame-difference form is the textbook one and it
    // needs a matrix-to-quaternion round trip, whose row-vs-column convention
    // is exactly the kind of thing this file has been bitten by before; this
    // form uses only AlignDirections and CreateFromAxisAngle, whose
    // conventions //Engine:scenegraph-smoke already pins.
    Vector3 from = fromPrimary;
    Vector3 to = toPrimary;
    if (from.Length() < 1e-6f || to.Length() < 1e-6f) {
        return Quaternion(0.0f, 0.0f, 0.0f, 0.0f);
    }
    from.Normalize();
    to.Normalize();

    // The secondaries only decide the roll about the primary, so each is
    // reduced to its component across it.
    Vector3 toSide = toSecondary - to * toSecondary.Dot(to);
    if (toSide.Length() < 1e-4f) {
        // Parallel to the primary, so it says nothing about the roll. A zero
        // quaternion is the caller's signal to fall back to a minimal arc --
        // identity would silently leave the bone unaimed.
        return Quaternion(0.0f, 0.0f, 0.0f, 0.0f);
    }
    toSide.Normalize();

    const Quaternion aim = AlignDirections(from, to);
    Vector3 fromSide = Vector3::Transform(fromSecondary, aim);
    fromSide -= to * fromSide.Dot(to);
    if (fromSide.Length() < 1e-4f) {
        return Quaternion(0.0f, 0.0f, 0.0f, 0.0f);
    }
    fromSide.Normalize();

    // Signed angle about the primary, which is what makes this a twist rather
    // than another shortest arc: the sign has to come from the axis, or a
    // half-turn one way is indistinguishable from the other.
    const float cosine = std::clamp(fromSide.Dot(toSide), -1.0f, 1.0f);
    const float sine = fromSide.Cross(toSide).Dot(to);
    const Quaternion twist =
        Quaternion::CreateFromAxisAngle(to, std::atan2(sine, cosine));

    // Aim first, then twist about the aimed direction, which leaves the
    // primary exactly where the aim put it.
    return Quaternion::Concatenate(twist, aim);
}

Matrix MotionRetargeter::AuthoredParentWorld(const SceneGraph& graph,
                                             std::size_t nodeIndex,
                                             const Matrix& scenePlacement,
                                             bool includeAssetPlacement) {
    const std::vector<SceneNode>& nodes = graph.Nodes();
    // The character's own file may be placed somewhere in the merged scene, and
    // that placement is part of the chain from the hips up to the world without
    // being a node in it -- UpdateTransforms applies it just inside the scene
    // placement, so this has to as well. Identity when only one file is loaded,
    // or when nothing has moved this one.
    const Matrix outer =
        includeAssetPlacement ? graph.PlacementForNode(nodeIndex) * scenePlacement
                              : scenePlacement;
    if (nodeIndex >= nodes.size() || !nodes[nodeIndex].parent.has_value()) {
        return outer;
    }

    // Collect the chain, then compose root-outward. Row-vector convention, so a
    // child's world transform is its local transform TIMES its parent's.
    std::vector<std::size_t> chain;
    std::optional<std::size_t> cursor = nodes[nodeIndex].parent;
    while (cursor.has_value()) {
        chain.push_back(*cursor);
        cursor = nodes[*cursor].parent;
    }

    Matrix world = outer;
    for (auto ancestor = chain.rbegin(); ancestor != chain.rend(); ++ancestor) {
        world = AuthoredLocal(nodes[*ancestor]) * world;
    }
    return world;
}

BindPoseMetrics MotionRetargeter::MeasureBindPose(const SceneGraph& graph,
                                                  std::size_t skinIndex) const {
    BindPoseMetrics metrics;
    if (skinIndex >= graph.Skins().size()) {
        return metrics;
    }
    const std::vector<std::size_t>& joints = graph.Skins()[skinIndex].joints;
    if (joints.empty()) {
        return metrics;
    }

    // In the character's own space: no scene placement, so the numbers describe
    // the rig rather than where it happens to be put.
    const Matrix identity = Matrix::Identity;
    const std::vector<SceneNode>& nodes = graph.Nodes();

    float lowest = 0.0f;
    float highest = 0.0f;
    bool first = true;
    for (const std::size_t joint : joints) {
        if (joint >= nodes.size()) {
            continue;
        }
        const Matrix world =
            AuthoredLocal(nodes[joint]) * AuthoredParentWorld(graph, joint, identity);
        const float y = world._42;
        if (first) {
            lowest = highest = y;
            first = false;
        } else {
            lowest = std::min(lowest, y);
            highest = std::max(highest, y);
        }
    }
    if (first) {
        return metrics;
    }

    // The hips are whichever node source joint 0 maps to, which is the root of
    // the bone hierarchy and the only bone carrying position.
    for (const RetargetBone& bone : m_bones) {
        if (bone.sourceJoint != 0 || !bone.destinationNode.has_value()) {
            continue;
        }
        const std::size_t hips = *bone.destinationNode;
        const Matrix world =
            AuthoredLocal(nodes[hips]) * AuthoredParentWorld(graph, hips, identity);
        metrics.hipsY = world._42;
        metrics.valid = true;
        break;
    }
    if (!metrics.valid) {
        return metrics;
    }

    metrics.lowestY = lowest;
    metrics.highestY = highest;
    metrics.hipsHeight = metrics.hipsY - lowest;
    return metrics;
}

bool MotionRetargeter::LoadConfig(const std::filesystem::path& path,
                                  std::string* error) {
    m_bones.clear();
    m_chains.clear();
    m_sourceJoints = 0;

    std::ifstream stream(path);
    if (!stream) {
        SetError(error, "cannot open " + path.string() +
                            "; run tools/prepare_retarget.py to produce it");
        return false;
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key == "joints") {
            fields >> m_sourceJoints;
        } else if (key == "chain") {
            std::vector<int> chain;
            int joint = 0;
            while (fields >> joint) {
                chain.push_back(joint);
            }
            if (chain.size() >= 2) {
                m_chains.push_back(std::move(chain));
            }
        } else if (key == "bone") {
            RetargetBone bone;
            int setRotation = 1;
            int setPosition = 0;
            float w = 1.0f;
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float dirX = 0.0f;
            float dirY = 0.0f;
            float dirZ = 0.0f;
            fields >> bone.sourceJoint >> bone.sourceName >> bone.destinationName >>
                setRotation >> setPosition >> w >> x >> y >> z >> bone.childJoint >>
                bone.childDestinationName >> dirX >> dirY >> dirZ;
            bone.sourceRestDirection = Vector3(dirX, dirY, dirZ);
            if (bone.childDestinationName == "-") {
                bone.childDestinationName.clear();
            }
            if (!fields && !fields.eof()) {
                SetError(error, path.string() + " has a malformed bone line: " + line);
                return false;
            }
            bone.setRotation = setRotation != 0;
            bone.setPosition = setPosition != 0;
            // The file stores (w, x, y, z); SimpleMath stores (x, y, z, w).
            bone.correction = Quaternion(x, y, z, w);
            bone.correction.Normalize();
            m_bones.push_back(std::move(bone));
        }
        // Unknown keys are ignored, so a newer preparation script can record
        // things this build does not read yet.
    }

    if (m_sourceJoints <= 1 || m_bones.empty() || m_chains.empty()) {
        SetError(error, path.string() + " is missing joints, chains or bones");
        return false;
    }
    for (const RetargetBone& bone : m_bones) {
        if (bone.sourceJoint < 0 || bone.sourceJoint >= m_sourceJoints) {
            SetError(error, "bone " + bone.sourceName + " names source joint " +
                                std::to_string(bone.sourceJoint) + ", outside the " +
                                std::to_string(m_sourceJoints) + " the file declares");
            return false;
        }
    }
    return true;
}

std::optional<std::size_t> MotionRetargeter::FindNode(
    const SceneGraph& graph, const std::string& name,
    const std::vector<std::size_t>* candidates) {
    const std::vector<SceneNode>& nodes = graph.Nodes();
    if (candidates != nullptr) {
        for (const std::size_t index : *candidates) {
            if (index < nodes.size() && nodes[index].name == name) {
                return index;
            }
        }
        return std::nullopt;
    }
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].name == name) {
            return i;
        }
    }
    return std::nullopt;
}

std::vector<RetargetTarget> MotionRetargeter::DiscoverTargets(
    const SceneGraph& graph) const {
    std::vector<RetargetTarget> targets;
    const std::vector<SceneSkin>& skins = graph.Skins();
    const std::vector<SceneNode>& nodes = graph.Nodes();

    for (std::size_t skinIndex = 0; skinIndex < skins.size(); ++skinIndex) {
        const SceneSkin& skin = skins[skinIndex];
        RetargetTarget target;
        target.skinIndex = skinIndex;
        target.jointCount = static_cast<int>(skin.JointCount());

        // The node instancing this skin carries the character's name.
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].skinIndex.has_value() && *nodes[i].skinIndex == skinIndex) {
                target.meshNode = i;
                if (!nodes[i].name.empty()) {
                    target.name = nodes[i].name;
                }
                break;
            }
        }
        if (target.name.empty()) {
            target.name = !skin.name.empty() ? skin.name
                                             : ("skin " + std::to_string(skinIndex));
        }

        for (const RetargetBone& bone : m_bones) {
            if (FindNode(graph, bone.destinationName, &skin.joints).has_value()) {
                ++target.matchedBones;
            }
        }
        targets.push_back(std::move(target));
    }
    return targets;
}

int MotionRetargeter::Bind(const SceneGraph& graph, std::vector<std::string>* missing) {
    m_boundSkin.reset();
    int bound = 0;
    for (RetargetBone& bone : m_bones) {
        bone.destinationNode = FindNode(graph, bone.destinationName, nullptr);
        if (bone.destinationNode.has_value()) {
            ++bound;
        } else if (missing != nullptr) {
            missing->push_back(bone.destinationName);
        }
    }
    if (bound > 0) {
        ResolveRestPose(graph);
    }
    return bound;
}

int MotionRetargeter::BindTo(const SceneGraph& graph, std::size_t skinIndex,
                             std::vector<std::string>* missing) {
    m_boundSkin.reset();
    for (RetargetBone& bone : m_bones) {
        bone.destinationNode.reset();
    }
    if (skinIndex >= graph.Skins().size()) {
        if (missing != nullptr) {
            missing->push_back("<no such skin>");
        }
        return 0;
    }

    const std::vector<std::size_t>& joints = graph.Skins()[skinIndex].joints;
    int bound = 0;
    for (RetargetBone& bone : m_bones) {
        bone.destinationNode = FindNode(graph, bone.destinationName, &joints);
        if (bone.destinationNode.has_value()) {
            ++bound;
        } else if (missing != nullptr) {
            missing->push_back(bone.destinationName);
        }
    }
    if (bound > 0) {
        m_boundSkin = skinIndex;
        ResolveRestPose(graph);
    }
    return bound;
}

void MotionRetargeter::ResolveRestPose(const SceneGraph& graph) {
    const std::vector<SceneNode>& nodes = graph.Nodes();
    const Matrix identity = Matrix::Identity;

    // Parent of each joint, from the chains. Needed to find the joints that
    // BRANCH off a bone's own joint, which is where the twist reference comes
    // from.
    std::vector<int> parentJoint(static_cast<std::size_t>(std::max(m_sourceJoints, 0)),
                                 -1);
    for (const std::vector<int>& chain : m_chains) {
        for (std::size_t k = 1; k < chain.size(); ++k) {
            if (chain[k] >= 0 && chain[k] < m_sourceJoints) {
                parentJoint[static_cast<std::size_t>(chain[k])] = chain[k - 1];
            }
        }
    }
    // The node a source joint drives, so a twist reference can be resolved to
    // the character's own joints as well as the motion's.
    const auto nodeForJoint = [&](int joint) -> std::optional<std::size_t> {
        for (const RetargetBone& other : m_bones) {
            if (other.sourceJoint == joint) {
                return other.destinationNode;
            }
        }
        return std::nullopt;
    };

    for (RetargetBone& bone : m_bones) {
        bone.restResolved = false;
        bone.twistResolved = false;
        bone.restRotation = Quaternion::Identity;
        bone.restAlignment = Quaternion::Identity;
        bone.restSide = Vector3::Zero;
        bone.twistLeftJoint = -1;
        bone.twistRightJoint = -1;
        bone.twistLeftNode.reset();
        bone.twistRightNode.reset();
        bone.childDestinationNode.reset();

        if (!bone.destinationNode.has_value() || bone.childJoint < 0 ||
            bone.childDestinationName.empty()) {
            continue;
        }
        bone.childDestinationNode =
            FindNode(graph, bone.childDestinationName, nullptr);
        if (!bone.childDestinationNode.has_value()) {
            continue;
        }

        // Both rest frames, from the AUTHORED transforms so the current pose
        // cannot leak in.
        const std::size_t node = *bone.destinationNode;
        const std::size_t child = *bone.childDestinationNode;
        const Matrix nodeWorld =
            AuthoredLocal(nodes[node]) *
            AuthoredParentWorld(graph, node, identity, false);
        const Matrix childWorld =
            AuthoredLocal(nodes[child]) *
            AuthoredParentWorld(graph, child, identity, false);

        Vector3 restDirection(childWorld._41 - nodeWorld._41,
                              childWorld._42 - nodeWorld._42,
                              childWorld._43 - nodeWorld._43);
        if (restDirection.Length() < 1e-6f) {
            continue;
        }
        restDirection.Normalize();

        Vector3 sourceDirection = bone.sourceRestDirection;
        if (sourceDirection.Length() < 1e-6f) {
            continue;
        }
        sourceDirection.Normalize();

        Matrix rotationOnly = nodeWorld;
        rotationOnly._41 = rotationOnly._42 = rotationOnly._43 = 0.0f;
        Vector3 scale;
        Quaternion rotation;
        Vector3 translation;
        if (!rotationOnly.Decompose(scale, rotation, translation)) {
            continue;
        }

        bone.restRotation = rotation;
        bone.restDirection = restDirection;
        bone.restAlignment = AlignDirections(restDirection, sourceDirection);
        bone.restResolved = true;

        // The twist reference: the joints that branch off this bone's own
        // joint, other than the one it points at. Two of them is a left and a
        // right -- the hips' two legs, the upper spine's two collars -- and
        // one or three is not a pair this can use.
        std::vector<int> branches;
        for (int joint = 0; joint < m_sourceJoints; ++joint) {
            if (joint != bone.childJoint &&
                parentJoint[static_cast<std::size_t>(joint)] == bone.sourceJoint) {
                branches.push_back(joint);
            }
        }
        if (branches.size() != 2) {
            continue;
        }
        // SMPL orders the sides left-then-right at every branch, so the lower
        // index is the left one. Only the CONSISTENCY matters here, since the
        // same pair is read from both skeletons, but getting it backwards
        // would mirror the twist rather than resolve it.
        const int leftJoint = std::min(branches[0], branches[1]);
        const int rightJoint = std::max(branches[0], branches[1]);
        const std::optional<std::size_t> leftNode = nodeForJoint(leftJoint);
        const std::optional<std::size_t> rightNode = nodeForJoint(rightJoint);
        if (!leftNode.has_value() || !rightNode.has_value()) {
            continue;
        }

        const Matrix leftWorld = AuthoredLocal(nodes[*leftNode]) *
                                 AuthoredParentWorld(graph, *leftNode, identity, false);
        const Matrix rightWorld =
            AuthoredLocal(nodes[*rightNode]) *
            AuthoredParentWorld(graph, *rightNode, identity, false);
        Vector3 side(rightWorld._41 - leftWorld._41, rightWorld._42 - leftWorld._42,
                     rightWorld._43 - leftWorld._43);
        // Orthogonalised here rather than at every frame: it is a property of
        // the rest pose, and keeping it perpendicular to restDirection is what
        // makes the frame a rotation.
        side -= restDirection * side.Dot(restDirection);
        if (side.Length() < 1e-5f) {
            continue;
        }
        side.Normalize();

        bone.twistLeftJoint = leftJoint;
        bone.twistRightJoint = rightJoint;
        bone.twistLeftNode = leftNode;
        bone.twistRightNode = rightNode;
        bone.restSide = side;
        bone.twistResolved = true;
    }
}

int MotionRetargeter::BoundCount() const {
    int bound = 0;
    for (const RetargetBone& bone : m_bones) {
        if (bone.destinationNode.has_value()) {
            ++bound;
        }
    }
    return bound;
}

bool MotionRetargeter::SourceWorldRotations(
    const float* feature, int motionDim, float rootAngle,
    std::vector<Quaternion>& out) const {
    if (feature == nullptr || m_sourceJoints <= 0) {
        return false;
    }
    const int rotationOffset = 4 + (m_sourceJoints - 1) * 3;
    if (motionDim < rotationOffset + (m_sourceJoints - 1) * 6) {
        return false;
    }

    out.assign(static_cast<std::size_t>(m_sourceJoints), Quaternion::Identity);

    // The root's rotation is not in the rotation block: it is the accumulated
    // yaw, the same quantity the position recovery integrates. HumanML3D stores
    // it as (cos a, 0, sin a, 0) in (w, x, y, z) -- and a unit quaternion whose
    // w is cos(a) turns by 2a, not by a. recover_from_rot seeds its forward
    // kinematics with that quaternion as it stands, so this has to be the same
    // rotation.
    //
    // It used to call CreateFromAxisAngle(UnitY, rootAngle), which is
    // (cos(a/2), 0, sin(a/2), 0) -- HALF the turn, and silently right whenever
    // the character walks in a straight line, because then a is near zero.
    // //Engine:retarget-probe measures the difference by checking the rotation
    // block against the ric positions of the same frame: over a take that turns
    // 72 degrees, the implied rest offsets scatter by 67.0 degrees under the old
    // seed and 51.7 under this one. It is the best of the 60 readings that probe
    // sweeps, which is how the factor of two was found rather than reasoned.
    const Quaternion root =
        Quaternion::CreateFromAxisAngle(Vector3::UnitY, 2.0f * rootAngle);
    out[0] = root;

    // Everything else is a LOCAL rotation relative to its parent, so the chains
    // accumulate from the root outward. Matches
    // Skeleton.forward_kinematics_cont6d, which seeds the running rotation with
    // the root and multiplies in each joint's own as it walks.
    for (const std::vector<int>& chain : m_chains) {
        // A chain may start at a joint another chain already resolved (the arm
        // chains both start at the upper spine), so seed from that joint rather
        // than from the root.
        Quaternion accumulated = out[static_cast<std::size_t>(chain.front())];
        for (std::size_t k = 1; k < chain.size(); ++k) {
            const int joint = chain[k];
            const float* c = feature + rotationOffset + (joint - 1) * 6;
            // A child's world rotation is its own local rotation followed by
            // the parent's. Concatenate takes those in the opposite order --
            // see the note at the top of this file.
            accumulated = Quaternion::Concatenate(accumulated, Cont6dToQuaternion(c));
            out[static_cast<std::size_t>(joint)] = accumulated;
        }
    }
    return true;
}

bool MotionRetargeter::SourceBoneDirections(const DirectX::XMFLOAT3* positions,
                                            std::vector<Vector3>& out) const {
    if (positions == nullptr || m_sourceJoints <= 0) {
        return false;
    }
    out.assign(static_cast<std::size_t>(m_sourceJoints), Vector3::Zero);

    // The chains are the connectivity: consecutive entries are one bone, and
    // the later one is the joint it ends at. Walking them rather than a
    // parent array because that is what the configuration carries, and it is
    // the same traversal SourceWorldRotations and //Engine:retarget-probe use,
    // so a bone is the same bone in all three.
    for (const std::vector<int>& chain : m_chains) {
        for (std::size_t k = 1; k < chain.size(); ++k) {
            const int joint = chain[k];
            const int parent = chain[k - 1];
            if (joint < 0 || joint >= m_sourceJoints || parent < 0 ||
                parent >= m_sourceJoints) {
                continue;
            }
            Vector3 bone(positions[joint].x - positions[parent].x,
                         positions[joint].y - positions[parent].y,
                         positions[joint].z - positions[parent].z);
            // A collapsed bone has no direction. Leaving it zero is what lets
            // ApplyFrame keep the character's authored orientation there,
            // rather than normalising a near-zero vector into noise.
            if (bone.Length() < 1e-6f) {
                continue;
            }
            bone.Normalize();
            out[static_cast<std::size_t>(joint)] = bone;
        }
    }
    return true;
}

bool MotionRetargeter::ApplyFrame(SceneGraph& graph, const float* feature,
                                  int motionDim,
                                  const DirectX::XMFLOAT3* positions,
                                  float rootAngle,
                                  const Vector3& hipsTarget,
                                  float characterScale,
                                  const Matrix& scenePlacement,
                                  std::string* error) {
    if (!IsLoaded()) {
        SetError(error, "no retarget configuration is loaded");
        return false;
    }
    const bool fromPositions = m_source == RetargetSource::Positions;
    m_positions = positions;
    if (fromPositions) {
        if (!SourceBoneDirections(positions, m_sourceDirection)) {
            SetError(error, "the position-driven retarget needs the frame's "
                            "recovered joint positions");
            return false;
        }
    } else if (!SourceWorldRotations(feature, motionDim, rootAngle, m_sourceWorld)) {
        SetError(error, "the feature frame does not match the retarget "
                        "configuration's joint count");
        return false;
    }

    const std::vector<SceneNode>& nodes = graph.Nodes();

    // Which bone, if any, drives each node. Built per call because Bind can
    // run again against a different character.
    m_boneForNode.assign(nodes.size(), -1);
    for (std::size_t i = 0; i < m_bones.size(); ++i) {
        const RetargetBone& bone = m_bones[i];
        // A bone whose rest pose did not resolve -- a leaf with no joint beyond
        // it, or a child node the character does not have -- is left at its
        // authored orientation rather than driven from a rotation that does not
        // describe it.
        if (bone.destinationNode.has_value() && bone.setRotation &&
            bone.restResolved && bone.childJoint >= 0 &&
            bone.childJoint < m_sourceJoints) {
            // Driving from positions needs a direction as well: a joint the
            // chains do not reach, or a bone that collapsed to a point in this
            // frame, has none, and SourceBoneDirections leaves it zero.
            if (fromPositions &&
                m_sourceDirection[static_cast<std::size_t>(bone.childJoint)]
                        .LengthSquared() < 1e-12f) {
                continue;
            }
            m_boneForNode[*bone.destinationNode] = static_cast<int>(i);
        }
    }

    m_nodeWorld.assign(nodes.size(), Quaternion::Identity);
    m_overrides.clear();

    // Parents before children, which is what the algorithm requires: a mapped
    // node's local rotation is derived from its parent's FINAL world rotation,
    // so the parent has to have been resolved first.
    //
    // Deliberately NOT SceneGraph::DrawOrder(), which skips subtrees hidden in
    // the UI and nodes outside the scene. That is right for drawing and wrong
    // here: a hidden node can still be an ancestor of a driven bone, and
    // leaving it out would silently drop its contribution to the chain.
    m_visitOrder.clear();
    m_visitOrder.reserve(nodes.size());
    m_stack.assign(graph.RootNodes().rbegin(), graph.RootNodes().rend());
    while (!m_stack.empty()) {
        const std::size_t index = m_stack.back();
        m_stack.pop_back();
        m_visitOrder.push_back(index);
        const std::vector<std::size_t>& children = nodes[index].children;
        for (auto child = children.rbegin(); child != children.rend(); ++child) {
            m_stack.push_back(*child);
        }
    }

    for (const std::size_t index : m_visitOrder) {
        const SceneNode& node = nodes[index];
        const Quaternion parentWorld =
            node.parent.has_value() ? m_nodeWorld[*node.parent] : Quaternion::Identity;

        const int boneIndex = m_boneForNode[index];
        if (boneIndex < 0) {
            // Not driven: keep what the asset authored, but still accumulate,
            // because a driven descendant needs this node's world rotation.
            m_nodeWorld[index] =
                Quaternion::Concatenate(parentWorld, AuthoredRotation(node));
            continue;
        }

        const RetargetBone& bone = m_bones[static_cast<std::size_t>(boneIndex)];

        // The one thing the two sources disagree about: where this bone's
        // desired WORLD rotation comes from. Everything after it -- the local
        // rotation against the parent's final world rotation, parents before
        // children, the override written through ApplyPoseOverrides -- is the
        // same either way.
        Quaternion desiredWorld;
        if (fromPositions) {
            const Vector3& measured =
                m_sourceDirection[static_cast<std::size_t>(bone.childJoint)];

            // The bone's direction expressed in its NODE's frame, which is
            // what a world rotation has to act on.
            Quaternion inverseRest = bone.restRotation;
            inverseRest.Conjugate();
            const Vector3 localDirection =
                Vector3::Transform(bone.restDirection, inverseRest);

            bool framed = false;
            if (bone.twistResolved && m_positions != nullptr) {
                // A FULL FRAME, so no twist is left free. The second direction
                // is the line between the two joints branching off this one --
                // hip to hip at the pelvis, shoulder to shoulder at the upper
                // spine -- read from the motion's positions and from the
                // character's rest pose the same way.
                //
                // This is what makes the character FACE where the motion
                // faces. Aiming the hips with a minimal arc cannot: the bone
                // points straight up, so a rotation about it is pure yaw and
                // the arc leaves it at the rest pose's. Measured over a take
                // that turns, that was 91.5 degrees of mean facing error with
                // every bone direction still exact.
                const Vector3 motionSide(
                    m_positions[bone.twistRightJoint].x -
                        m_positions[bone.twistLeftJoint].x,
                    m_positions[bone.twistRightJoint].y -
                        m_positions[bone.twistLeftJoint].y,
                    m_positions[bone.twistRightJoint].z -
                        m_positions[bone.twistLeftJoint].z);
                const Quaternion frame = FrameRotation(bone.restDirection,
                                                       bone.restSide, measured,
                                                       motionSide);
                framed = frame.LengthSquared() > 0.5f;
                if (framed) {
                    desiredWorld =
                        Quaternion::Concatenate(frame, bone.restRotation);
                }
            }
            if (!framed) {
                // No pair to read a twist from, so the bone INHERITS its
                // parent's and adds only the arc that aims it. Deliberately
                // against the parent's already-resolved rotation rather than
                // against the rest pose: an arc from rest would discard
                // whatever twist the frame above resolved, and the spine would
                // go back to facing the rest direction one bone below the hips
                // that were just turned to face the motion.
                //
                // WHAT IS STILL FREE is the roll about the bone's own axis
                // relative to the parent, which no position can supply. The
                // minimal arc adds none, so the character keeps its AUTHORED
                // roll there -- visible on the forearms, which pronate without
                // moving the wrist, and over the spine.
                const Quaternion inherited =
                    Quaternion::Concatenate(parentWorld, AuthoredRotation(node));
                const Vector3 aimed =
                    Vector3::Transform(localDirection, inherited);
                desiredWorld = Quaternion::Concatenate(
                    AlignDirections(aimed, measured), inherited);
            }
        } else {
            // W = R(child) * A * restRotation, in row-vector order, which
            // reads right to left: start from the character's rest orientation
            // for this bone, cancel the difference between the two skeletons'
            // rest directions, then apply the source's world rotation. The
            // source rotation comes from the CHILD joint because HumanML3D
            // stores at a joint the rotation that orients the bone ending
            // there.
            const Quaternion sourceRotation =
                m_sourceWorld[static_cast<std::size_t>(bone.childJoint)];
            desiredWorld = Quaternion::Concatenate(
                sourceRotation,
                Quaternion::Concatenate(bone.restAlignment, bone.restRotation));
        }
        desiredWorld.Normalize();

        // local = inverse(parentWorld) * desiredWorld, in row-vector order.
        Quaternion inverseParent = parentWorld;
        inverseParent.Conjugate();
        Quaternion local = Quaternion::Concatenate(inverseParent, desiredWorld);
        local.Normalize();

        SceneGraph::PoseOverride override;
        override.nodeIndex = index;
        override.rotation = local;
        if (bone.setPosition) {
            // The target is in the character's own file space, so it goes OUT
            // through that file's placement to reach world space and then back
            // through the parent chain into the space this node's translation
            // lives in. Both legs include the placement, so what the motion
            // sees is the file's frame: move the file and the character goes
            // with it, turn the file and the walk turns with it.
            const Matrix parentWorld =
                AuthoredParentWorld(graph, index, scenePlacement);
            const Vector3 worldTarget = Vector3::Transform(
                hipsTarget, graph.PlacementForNode(index) * scenePlacement);
            // A degenerate chain -- a zero scale somewhere above the hips --
            // has no inverse, and transforming through it would produce
            // infinities that spread to every descendant. Leaving the authored
            // translation is the recoverable outcome.
            if (std::fabs(parentWorld.Determinant()) > 1e-12f) {
                override.hasTranslation = true;
                override.translation =
                    Vector3::Transform(worldTarget, parentWorld.Invert());
            }

            // Resizing the root of the joint hierarchy resizes the whole rig,
            // and the skinned mesh with it. Rotation-only retargeting cannot
            // change proportions, so this is the only way to match the source
            // skeleton's size.
            if (characterScale > 0.0f) {
                override.hasScale = true;
                override.scale = Vector3(characterScale, characterScale,
                                         characterScale);
            }
        }
        m_overrides.push_back(override);

        m_nodeWorld[index] = desiredWorld;
    }

    // A bone that carries position but was not rotation-driven (its rest pose
    // did not resolve) still has to be placed, or the character would be posed
    // around an unmoved root.
    for (const RetargetBone& bone : m_bones) {
        if (!bone.setPosition || !bone.destinationNode.has_value()) {
            continue;
        }
        if (m_boneForNode[*bone.destinationNode] >= 0) {
            continue;  // already handled above
        }
        const Matrix parentWorld =
            AuthoredParentWorld(graph, *bone.destinationNode, scenePlacement);
        if (std::fabs(parentWorld.Determinant()) <= 1e-12f) {
            continue;
        }
        // Same two legs as above: out through the file's placement, back
        // through the parent chain.
        const Vector3 worldTarget =
            Vector3::Transform(hipsTarget, graph.PlacementForNode(*bone.destinationNode) *
                                               scenePlacement);
        SceneGraph::PoseOverride override;
        override.nodeIndex = *bone.destinationNode;
        override.rotation = AuthoredRotation(nodes[*bone.destinationNode]);
        override.hasTranslation = true;
        override.translation = Vector3::Transform(worldTarget, parentWorld.Invert());
        if (characterScale > 0.0f) {
            override.hasScale = true;
            override.scale = Vector3(characterScale, characterScale, characterScale);
        }
        m_overrides.push_back(override);
    }

    if (!graph.ApplyPoseOverrides(m_overrides)) {
        SetError(error, "a retarget override named a node outside the graph");
        return false;
    }
    return true;
}

}  // namespace NeuralModelIntegrateTestbed
