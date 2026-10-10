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

// Shortest rotation taking unit vector `from` to unit vector `to`. Used to
// cancel the difference between the two skeletons' rest bone directions, which
// is what momask's hand-tuned corrections were approximating.
Quaternion AlignVectors(const Vector3& from, const Vector3& to) {
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

Matrix MotionRetargeter::AuthoredParentWorld(const SceneGraph& graph,
                                             std::size_t nodeIndex,
                                             const Matrix& scenePlacement) {
    const std::vector<SceneNode>& nodes = graph.Nodes();
    if (nodeIndex >= nodes.size() || !nodes[nodeIndex].parent.has_value()) {
        return scenePlacement;
    }

    // Collect the chain, then compose root-outward. Row-vector convention, so a
    // child's world transform is its local transform TIMES its parent's.
    std::vector<std::size_t> chain;
    std::optional<std::size_t> cursor = nodes[nodeIndex].parent;
    while (cursor.has_value()) {
        chain.push_back(*cursor);
        cursor = nodes[*cursor].parent;
    }

    Matrix world = scenePlacement;
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

    for (RetargetBone& bone : m_bones) {
        bone.restResolved = false;
        bone.restRotation = Quaternion::Identity;
        bone.restAlignment = Quaternion::Identity;
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
            AuthoredLocal(nodes[node]) * AuthoredParentWorld(graph, node, identity);
        const Matrix childWorld =
            AuthoredLocal(nodes[child]) * AuthoredParentWorld(graph, child, identity);

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
        bone.restAlignment = AlignVectors(restDirection, sourceDirection);
        bone.restResolved = true;
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
    // it as (cos a, 0, sin a, 0) in (w, x, y, z), i.e. a rotation about Y.
    const Quaternion root = Quaternion::CreateFromAxisAngle(Vector3::UnitY, rootAngle);
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

bool MotionRetargeter::ApplyFrame(SceneGraph& graph, const float* feature,
                                  int motionDim, float rootAngle,
                                  const Vector3& hipsWorldTarget,
                                  float characterScale,
                                  const Matrix& scenePlacement,
                                  std::string* error) {
    if (!IsLoaded()) {
        SetError(error, "no retarget configuration is loaded");
        return false;
    }
    if (!SourceWorldRotations(feature, motionDim, rootAngle, m_sourceWorld)) {
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

        // W = R(child) * A * restRotation, in row-vector order, which reads
        // right to left: start from the character's rest orientation for this
        // bone, cancel the difference between the two skeletons' rest
        // directions, then apply the source's world rotation. The source
        // rotation comes from the CHILD joint because HumanML3D stores at a
        // joint the rotation that orients the bone ending there.
        const Quaternion sourceRotation =
            m_sourceWorld[static_cast<std::size_t>(bone.childJoint)];
        const Quaternion desiredWorld = Quaternion::Concatenate(
            sourceRotation,
            Quaternion::Concatenate(bone.restAlignment, bone.restRotation));

        // local = inverse(parentWorld) * desiredWorld, in row-vector order.
        Quaternion inverseParent = parentWorld;
        inverseParent.Conjugate();
        Quaternion local = Quaternion::Concatenate(inverseParent, desiredWorld);
        local.Normalize();

        SceneGraph::PoseOverride override;
        override.nodeIndex = index;
        override.rotation = local;
        if (bone.setPosition) {
            // The target is a WORLD position, so it has to be pulled back
            // through the parent chain -- scene placement included -- into the
            // space this node's translation actually lives in. Writing the
            // world position directly is what left the character offset by the
            // scene placement.
            const Matrix parentWorld =
                AuthoredParentWorld(graph, index, scenePlacement);
            // A degenerate chain -- a zero scale somewhere above the hips --
            // has no inverse, and transforming through it would produce
            // infinities that spread to every descendant. Leaving the authored
            // translation is the recoverable outcome.
            if (std::fabs(parentWorld.Determinant()) > 1e-12f) {
                override.hasTranslation = true;
                override.translation =
                    Vector3::Transform(hipsWorldTarget, parentWorld.Invert());
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
        SceneGraph::PoseOverride override;
        override.nodeIndex = *bone.destinationNode;
        override.rotation = AuthoredRotation(nodes[*bone.destinationNode]);
        override.hasTranslation = true;
        override.translation = Vector3::Transform(hipsWorldTarget, parentWorld.Invert());
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
