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
// Read that 0.000 for what it measures: the character agreeing with the
// FORWARD KINEMATICS OF THE ROTATION BLOCK, which is the only thing a
// rotation-driven transfer can be checked against from the rotations alone.
// Against the positions the same path lands 32.65 degrees out, mean over the
// mapped bones of five frames -- see the next note for why.
//
// WHICH OF THE TWO ENCODINGS DRIVES IT. HumanML3D's 263-dim vector carries the
// same pose twice: `ric` joint positions in the root-local frame (which
// Engine/MotionFeatures.cc decodes for the skeleton view) and (joints-1)
// six-dimensional LOCAL joint rotations. Rotations are the better input in
// principle -- a bone direction fixes only two of three degrees of freedom,
// so positions leave the roll undetermined -- and that is what this started
// out doing.
//
// It does not hold for the motion this engine generates. //Engine:retarget-probe
// scores a candidate reading of the rotation block by pulling each measured
// bone direction back through the rotation that is supposed to have produced
// it; the result is the joint's rest offset, a property of the skeleton, so it
// must be identical on every frame and a correct reading scores near zero
// scatter. Sweeping 60 readings -- local-accumulated against world, both
// product orders, the frame against its transpose, two block layouts, five
// root-angle conventions -- the BEST scores 51.7 degrees of mean scatter. One
// real bug came out of that sweep and is fixed (the root seed was half the
// rotation it should have been, 67.0 -> 51.7); the remaining ~52 degrees is
// the block not describing the pose `ric` describes, which no reading can
// repair.
//
// So RetargetSource::Positions is the default. For a bone whose source joint
// is p and child joint is c it takes
//
//     d_b  = normalise(position[c] - position[p])
//
// and aims the bone along it. Posed onto the real character, every bone then
// lands 0.00 degrees mean (0.03 worst) from the direction the skeleton overlay
// draws, against 32.65 mean (165.50 worst) for the rotation path.
//
// AIMING IS NOT POSING, which is the trap this fell into first and the reason
// the rest of RetargetBone exists. A direction fixes two of three degrees of
// freedom; the third is the twist about the bone, and at the HIPS that twist
// is which way the character faces, because the hips bone points straight up
// and a rotation about it is pure yaw. Aiming every bone with a minimal arc
// from its rest direction scored a perfect 0.00 on directions and left the
// character facing 91.5 degrees away from the motion on average, up to 172.9
// -- wrung around its own waist, torso at the rest facing and legs following
// the motion. That is a pose no direction test can see; see RetargetBone's
// twist fields for how the second direction resolves it, and the facing check
// in //Engine:scenegraph-smoke, which exists because the direction check read
// 0.00 throughout. It now measures 2.2 degrees mean on real motion. The rotation path is kept,
// not deleted, because a future export with a faithful rotation block would
// want it back -- and because it carries the roll that positions cannot.
namespace NeuralModelIntegrateTestbed {

// Which of the feature vector's two pose encodings drives the character.
enum class RetargetSource {
    // The 6D rotation block. Carries the roll about each bone's own axis,
    // which a direction cannot, but on this export it disagrees with the
    // positions by ~52 degrees; see the note above.
    Rotations,

    // The `ric` joint positions, via each bone's measured world direction.
    // Validated against Python by //Engine:motionfeatures-smoke, and the
    // encoding the skeleton overlay uses.
    Positions,
};

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

    // --- resolving the twist -------------------------------------------
    // A bone's direction fixes two of three degrees of freedom. The third,
    // the twist about the bone itself, is free -- and at the HIPS that free
    // degree of freedom is which way the character FACES, because the hips
    // bone points straight up and a rotation about it is pure yaw. Aiming it
    // with a minimal arc therefore leaves the body facing wherever the rest
    // pose faced while the legs follow the motion, measured at 91.5 degrees
    // mean and 172.9 worst by //Engine:retarget-probe -- a character wrung
    // around its own waist, with every bone still pointing exactly where it
    // was asked to.
    //
    // A SECOND direction resolves it, and the positions already carry one:
    // the line between the two joints that branch off this one. For the hips
    // that is hip to hip, for the upper spine shoulder to shoulder, which are
    // exactly the two places the twist is worth pinning. Both are found from
    // the chains, so no new configuration is needed: a joint with two
    // branches BESIDES the one this bone points at has a left and a right.
    int twistLeftJoint = -1;
    int twistRightJoint = -1;
    std::optional<std::size_t> twistLeftNode;
    std::optional<std::size_t> twistRightNode;

    // The character's own left-to-right direction at this bone, in the same
    // rest space as restDirection and already orthogonal to it.
    DirectX::SimpleMath::Vector3 restSide = {0.0f, 0.0f, 0.0f};

    bool twistResolved = false;
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

    // Which encoding ApplyFrame reads. Positions by default; see the note at
    // the top of this file for the measurement that decides it.
    void SetSource(RetargetSource source) { m_source = source; }
    RetargetSource Source() const { return m_source; }

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

    // World-space direction of the bone ENDING at each source joint, from the
    // recovered joint positions of one frame -- normalise(position[j] -
    // position[parent of j]). Zero for the root and for any joint the
    // configuration's chains do not reach, which a caller must treat as "no
    // direction" rather than as a direction of zero length.
    //
    // `positions` is one frame of FloodDiffusionPipeline::Frame, i.e. already
    // in the motion's world frame with the root yaw applied, so the directions
    // are world directions and need no root convention. That is the whole
    // reason this path cannot be wrong in the way the rotation block is: the
    // five things //Engine:retarget-probe sweeps over do not exist here.
    //
    // `out` is resized to SourceJointCount(). False if `positions` is null.
    bool SourceBoneDirections(
        const DirectX::XMFLOAT3* positions,
        std::vector<DirectX::SimpleMath::Vector3>& out) const;

    // Shortest rotation taking unit vector `from` to unit vector `to`, with
    // the anti-parallel case resolved explicitly instead of producing a NaN
    // from a zero-length cross product. Public because it is half of the
    // position-driven construction, and //Engine:retarget-probe scores that
    // construction rather than a re-implementation of it.
    static DirectX::SimpleMath::Quaternion AlignDirections(
        const DirectX::SimpleMath::Vector3& from,
        const DirectX::SimpleMath::Vector3& to);

    // Poses `graph` from one frame. Walks the hierarchy parents-first,
    // overriding the mapped nodes' local rotations and leaving everything else
    // at its authored transform, then writes the result through
    // SceneGraph::ApplyPoseOverrides.
    //
    // `hipsTarget` is where the hips should end up in the CHARACTER'S OWN
    // FILE space -- the space its glTF authored it in, before that file's
    // placement in the merged scene. It is carried out to world space here, by
    // the same placement every other node in that file goes through:
    //
    //     world = hipsTarget * assetPlacement * scenePlacement
    //
    // and then pulled back into the hips' parent space, because that is where
    // a node's translation lives.
    //
    // It used to be a world position, which made the motion the one thing in
    // the scene that did NOT follow its own file: moving that glTF left the
    // character behind, rotating it turned the body but not the direction the
    // motion walked in, and scaling it resized the rig but not its stride. The
    // body orientation always followed the placement -- the bone rotations are
    // composed onto the parent chain, which includes it -- so world-space
    // aiming made the translation disagree with the rotation.
    //
    // `scenePlacement` is the WHOLE scene's placement only. The file's own
    // comes from the graph, keyed on the node being placed, so a character and
    // the set it stands in can be moved independently.
    //
    // `characterScale` is applied to the hips node, resizing the entire joint
    // hierarchy and so the skinned mesh with it. Rotation-only retargeting
    // cannot change a character's proportions, so this is what makes a
    // character match the source skeleton's size.
    //
    // `positions` is the same frame's recovered joint positions, which
    // RetargetSource::Positions drives from; null is accepted and is an error
    // only in that mode. `feature`, `motionDim` and `rootAngle` are read only
    // by RetargetSource::Rotations.
    //
    // Must run BEFORE SceneGraph::UpdateTransforms.
    bool ApplyFrame(SceneGraph& graph,
                    const float* feature,
                    int motionDim,
                    const DirectX::XMFLOAT3* positions,
                    float rootAngle,
                    const DirectX::SimpleMath::Vector3& hipsTarget,
                    float characterScale,
                    const DirectX::SimpleMath::Matrix& scenePlacement,
                    std::string* error);

private:
    // Computes each bone's rest rotation and rest-direction alignment from the
    // character's bind pose. Run by BindTo and Bind, since both change which
    // nodes the bones refer to.
    void ResolveRestPose(const SceneGraph& graph);

    // World transform of a node's parent chain, from the AUTHORED local
    // transforms plus the node's own asset placement and `scenePlacement`.
    // Authored rather than current, because the chain above a joint hierarchy
    // is not what a retarget drives, and reading worldTransform would depend on
    // a previous UpdateTransforms.
    //
    // `includeAssetPlacement` is false for the rest pose, which has to be
    // resolved in the character's FILE space: that is the space ApplyFrame
    // composes in, since UpdateTransforms applies the file's placement on top
    // of what this writes. Resolving the rest pose in world space instead put
    // the placement into restRotation and then applied it a second time --
    // harmless while the only use of the rest pose was an arc between two
    // directions that both carried it, and not harmless once a full frame is
    // built from it.
    static DirectX::SimpleMath::Matrix AuthoredParentWorld(
        const SceneGraph& graph, std::size_t nodeIndex,
        const DirectX::SimpleMath::Matrix& scenePlacement,
        bool includeAssetPlacement = true);

    // Rotation taking one orthonormal frame onto another, each given as a
    // primary direction and a secondary that is orthogonalised against it.
    // This is the 3-DOF construction that AlignDirections cannot be: it
    // leaves no twist free.
    static DirectX::SimpleMath::Quaternion FrameRotation(
        const DirectX::SimpleMath::Vector3& fromPrimary,
        const DirectX::SimpleMath::Vector3& fromSecondary,
        const DirectX::SimpleMath::Vector3& toPrimary,
        const DirectX::SimpleMath::Vector3& toSecondary);

    // Resolves `name` to a node index, optionally restricted to `candidates`.
    static std::optional<std::size_t> FindNode(
        const SceneGraph& graph, const std::string& name,
        const std::vector<std::size_t>* candidates);

    std::vector<RetargetBone> m_bones;
    RetargetSource m_source = RetargetSource::Positions;
    std::optional<std::size_t> m_boundSkin;
    std::vector<std::vector<int>> m_chains;
    int m_sourceJoints = 0;

    // Scratch, so a per-frame call does not allocate.
    mutable std::vector<DirectX::SimpleMath::Quaternion> m_sourceWorld;
    mutable std::vector<DirectX::SimpleMath::Vector3> m_sourceDirection;

    // The frame ApplyFrame is in the middle of, so the bone loop can read a
    // twist reference pair without threading the array through it. Valid only
    // for the duration of that call.
    mutable const DirectX::XMFLOAT3* m_positions = nullptr;
    mutable std::vector<DirectX::SimpleMath::Quaternion> m_nodeWorld;
    mutable std::vector<int> m_boneForNode;
    mutable std::vector<SceneGraph::PoseOverride> m_overrides;
    mutable std::vector<std::size_t> m_visitOrder;
    mutable std::vector<std::size_t> m_stack;
};

}  // namespace NeuralModelIntegrateTestbed
