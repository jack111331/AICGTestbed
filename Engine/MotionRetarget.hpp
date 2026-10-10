#pragma once

#include "SceneGraph.hpp"

#include <DirectXMath.h>
#include <SimpleMath.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// Driving a rigged glTF character from HumanML3D motion.
//
// This is a port of the retargeting momask-codes documents in its README: the
// KeeMap Blender addon
// (nkeeline/Keemap-Blender-Rig-ReTargeting-Addon) driven by the bone mapping in
// momask's assets/mapping.json. tools/prepare_retarget.py converts that
// mapping into the flat file this loads.
//
// WHAT THE ADDON ACTUALLY DOES, since its implementation hides it behind
// Blender's pose-bone types. The one line that matters is
//
//     FinalQuat = dest.rotation_quaternion
//                 @ DestWS.rotation_difference(SourceWS)
//                 @ CorrectionQuat
//
// where `rotation_quaternion` is the destination bone's LOCAL pose rotation and
// the two WS quaternions are world-space. Because a bone's world rotation is
// its parent chain times that local rotation, substituting gives
//
//     W_dest' = W_source * Correction
//
// So the whole algorithm is: set each destination bone's WORLD rotation to the
// source joint's world rotation, times a fixed per-bone correction for the
// difference in rest orientation -- and do it parents-first, because each
// bone's local rotation is derived from its parent's already-final world
// rotation. glTF makes this simpler than Blender: a node's local transform IS
// its full TRS, with no separate rest matrix, so
//
//     local_rotation = inverse(parent_world_rotation) * W_source * Correction
//
// WHY momask's CORRECTION CONSTANTS ARE NOT USED. The addon's formula is
// right, but the constants beside it are not transferable. They were tuned
// against a Blender armature imported from momask's BVH export, whose bone
// axes are Blender's (Y along the bone) rather than HumanML3D's cont6d frames.
// Applied to this project's source rotations they leave bone directions about
// 90 degrees out -- a median of 91 degrees across the mapped bones, i.e. no
// correlation at all. So the correction is DERIVED here instead, from the two
// rest poses, which needs no tuning and is exact:
//
//     A_b  = the rotation taking the character's rest direction for bone b to
//            the source skeleton's rest direction for it (the source rests with
//            its arms down, a Mixamo character in a T-pose)
//     W_b  = R(child of b) * A_b * restRotation_b
//
// with R(child) rather than R(b) because HumanML3D's rotation at a joint
// orients the bone ENDING there while a rig's bone orients the one STARTING
// there. Measured over four frames and every mapped bone, this puts the
// character's bone directions 0.000 degrees from the motion's; the joint's own
// rotation gives ~90 degrees, and the child's rotation without rest alignment
// fixes the legs and spine but leaves the arms 84 degrees out.
//
// THE SOURCE ROTATIONS come from the feature vector's rotation block, not from
// the joint positions. HumanML3D's 263-dim vector carries both: positions
// (which Engine/MotionFeatures.cc decodes for the skeleton view) and
// (joints-1) six-dimensional LOCAL joint rotations. Using the rotations avoids
// solving an ill-posed inverse-kinematics problem from positions -- a bone
// direction fixes only two of three degrees of freedom, leaving the roll
// undetermined. Verified usable rather than assumed: running forward
// kinematics from the rotation block reproduces the position block to a mean
// of 1.45 cm on a 2.5 m motion, so the two encodings agree.
namespace NeuralModelIntegrateTestbed {

// One mapped bone from the KeeMap configuration.
struct RetargetBone {
    // Index into the HumanML3D skeleton, 0..joints-1.
    int sourceJoint = -1;
    std::string sourceName;

    // The glTF node to drive, by name, e.g. "mixamorig:LeftForeArm".
    std::string destinationName;

    bool setRotation = true;

    // Only the hips carry position in momask's mapping; every other bone is
    // rotation-only, which is what keeps the character's own proportions.
    bool setPosition = false;

    // momask's own rest-orientation correction, kept for reference and NOT
    // applied. Measured against this project's source rotations it leaves bone
    // directions about 90 degrees out, because it was tuned for a Blender
    // armature imported from momask's BVH, whose bone axes are not HumanML3D's
    // cont6d frames. The alignment computed at Bind replaces it; see the note
    // on MotionRetargeter.
    DirectX::SimpleMath::Quaternion correction;

    // The joint this BONE points at, and that joint's node. HumanML3D stores a
    // rotation that orients the bone ENDING at a joint, while a rig's bone
    // orients the one STARTING at it, so a bone is driven by its CHILD joint's
    // rotation. -1 for a leaf, which keeps its rest orientation.
    int childJoint = -1;
    std::string childDestinationName;

    // The child's rest direction in the SOURCE skeleton, from
    // t2m_raw_offsets.
    DirectX::SimpleMath::Vector3 sourceRestDirection = {0.0f, 0.0f, 0.0f};

    // Filled by Bind. Empty when the character has no node of that name.
    std::optional<std::size_t> destinationNode;
    std::optional<std::size_t> childDestinationNode;

    // Computed at Bind from the two rest poses: the character's bind-pose world
    // rotation for this bone, and the rotation taking the character's rest bone
    // direction to the source's. Together these make the transfer exact.
    DirectX::SimpleMath::Quaternion restRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    DirectX::SimpleMath::Quaternion restAlignment = {0.0f, 0.0f, 0.0f, 1.0f};

    // The character's own rest direction for this bone, in its rest world
    // space. Kept so a test can predict where the bone should point without
    // recomputing the bind pose.
    DirectX::SimpleMath::Vector3 restDirection = {0.0f, 0.0f, 0.0f};

    bool restResolved = false;
};

// A rig in the scene that motion can be retargeted onto. One per skin, since
// a skin is what owns a joint list and an inverse-bind pose; several nodes
// instancing the same skin would pose identically.
struct RetargetTarget {
    // What to show in a selector. The node carrying the skinned mesh, which is
    // what Mixamo names after the character ("Ch43"), falling back to the
    // skin's own name and then to its index.
    std::string name;

    std::size_t skinIndex = 0;
    std::optional<std::size_t> meshNode;
    int jointCount = 0;

    // How many of the configuration's bones this rig actually has, found
    // WITHOUT binding. Lets a selector show which characters are drivable and
    // which are rigged differently, rather than discovering it on selection.
    int matchedBones = 0;
};

// What a character's rest pose measures, in its own space (no scene
// placement), so a retarget can be sized to it.
struct BindPoseMetrics {
    bool valid = false;

    // Hips height above the lowest joint: the measure that decides leg length,
    // and so how tall the character stands.
    float hipsHeight = 0.0f;

    float hipsY = 0.0f;
    float lowestY = 0.0f;
    float highestY = 0.0f;
};

class MotionRetargeter {
public:
    // Loads the flat mapping written by tools/prepare_retarget.py.
    bool LoadConfig(const std::filesystem::path& path, std::string* error);

    bool IsLoaded() const { return !m_bones.empty(); }
    const std::vector<RetargetBone>& Bones() const { return m_bones; }
    const std::vector<std::vector<int>>& Chains() const { return m_chains; }
    int SourceJointCount() const { return m_sourceJoints; }

    // Every rig in the scene that could be driven, in skin order.
    std::vector<RetargetTarget> DiscoverTargets(const SceneGraph& graph) const;

    // Resolves every bone's destination name to a node index, searching the
    // whole scene. Returns how many bound; names with no node are left unbound
    // and reported in `missing` rather than failing, because a partial rig is
    // still worth driving.
    //
    // Prefer BindTo when the scene holds more than one character: Mixamo rigs
    // all name their bones "mixamorig:*", so an unscoped search would silently
    // bind to whichever character happens to come first.
    int Bind(const SceneGraph& graph, std::vector<std::string>* missing);

    // Binds only among the joints of one skin, which is what makes the choice
    // of character meaningful.
    int BindTo(const SceneGraph& graph, std::size_t skinIndex,
               std::vector<std::string>* missing);

    int BoundCount() const;

    // The skin BindTo last bound to, if any.
    std::optional<std::size_t> BoundSkin() const { return m_boundSkin; }

    // Measures one skin's rest pose from the nodes' AUTHORED transforms, so it
    // is unaffected by whatever pose the graph currently holds -- including a
    // retarget already applied to it.
    BindPoseMetrics MeasureBindPose(const SceneGraph& graph,
                                    std::size_t skinIndex) const;

    // World-space rotation of each source joint for one feature frame.
    // `rootAngle` is the accumulated root yaw, which the feature vector does
    // not carry per-frame -- take it from
    // FloodDiffusionPipeline::FrameRootAngle.
    //
    // `out` is resized to SourceJointCount(). False if `feature` is null or
    // the layout does not match the configuration.
    bool SourceWorldRotations(const float* feature, int motionDim, float rootAngle,
                              std::vector<DirectX::SimpleMath::Quaternion>& out) const;

    // Poses `graph` from one frame. Walks the hierarchy parents-first,
    // overriding the mapped nodes' local rotations and leaving everything else
    // at its authored transform, then writes the result through
    // SceneGraph::ApplyPoseOverrides.
    //
    // `hipsWorldTarget` is where the hips should end up in WORLD space, which
    // is the whole reason `scenePlacement` is a parameter: the translation
    // written onto the hips node is in its parent's space, and the placement is
    // the outermost link of that chain. Aiming in world space is what lets the
    // character stand on the ground plane whatever the scene placement is.
    //
    // `characterScale` is applied to the hips node, resizing the entire joint
    // hierarchy and so the skinned mesh with it. Rotation-only retargeting
    // cannot change a character's proportions, so this is what makes a
    // character match the source skeleton's size.
    //
    // Must run BEFORE SceneGraph::UpdateTransforms.
    bool ApplyFrame(SceneGraph& graph,
                    const float* feature,
                    int motionDim,
                    float rootAngle,
                    const DirectX::SimpleMath::Vector3& hipsWorldTarget,
                    float characterScale,
                    const DirectX::SimpleMath::Matrix& scenePlacement,
                    std::string* error);

private:
    // Computes each bone's rest rotation and rest-direction alignment from the
    // character's bind pose. Run by BindTo and Bind, since both change which
    // nodes the bones refer to.
    void ResolveRestPose(const SceneGraph& graph);

    // World transform of a node's parent chain, from the AUTHORED local
    // transforms plus `scenePlacement`. Authored rather than current, because
    // the chain above a joint hierarchy is not what a retarget drives, and
    // reading worldTransform would depend on a previous UpdateTransforms.
    static DirectX::SimpleMath::Matrix AuthoredParentWorld(
        const SceneGraph& graph, std::size_t nodeIndex,
        const DirectX::SimpleMath::Matrix& scenePlacement);

    // Resolves `name` to a node index, optionally restricted to `candidates`.
    static std::optional<std::size_t> FindNode(
        const SceneGraph& graph, const std::string& name,
        const std::vector<std::size_t>* candidates);

    std::vector<RetargetBone> m_bones;
    std::optional<std::size_t> m_boundSkin;
    std::vector<std::vector<int>> m_chains;
    int m_sourceJoints = 0;

    // Scratch, so a per-frame call does not allocate.
    mutable std::vector<DirectX::SimpleMath::Quaternion> m_sourceWorld;
    mutable std::vector<DirectX::SimpleMath::Quaternion> m_nodeWorld;
    mutable std::vector<int> m_boneForNode;
    mutable std::vector<SceneGraph::PoseOverride> m_overrides;
    mutable std::vector<std::size_t> m_visitOrder;
    mutable std::vector<std::size_t> m_stack;
};

}  // namespace NeuralModelIntegrateTestbed
