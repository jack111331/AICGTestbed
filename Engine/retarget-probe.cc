// Finds which convention a HumanML3D feature vector's 6D rotation block is
// written in, by checking it against the `ric` block of the same vector.
//
// Why this exists. A feature frame carries the same pose twice: `ric` as joint
// positions in the root-local frame, and a block of 6D rotations. The skeleton
// overlay is drawn from the first; retargeting onto a rigged character is
// driven by the second. So "the generated motion looks fine but the retargeted
// character looks weird" is exactly the symptom of the two disagreeing -- and a
// misread rotation block is invisible in the overlay, which never touches it.
//
// The test needs no reference data. HumanML3D's forward kinematics says
//
//     position[j] - position[parent] = worldRotation[j] * restOffset[j]
//
// and restOffset is a property of the SKELETON, not of the pose. So pulling the
// measured bone direction back through the rotation,
//
//     offset[j] = normalise(position[j] - position[parent]) * inverse(worldRotation[j])
//
// must come out the same on every frame. The spread of that over a whole take
// is therefore a score for a candidate reading, and a reading that is right
// scores near zero. Every wrong reading is wrong in a pose-dependent way, which
// is precisely what cannot be seen by eye.
//
// The sweep covers the three things that are easy to get backwards and hard to
// spot: whether the entries are local rotations to accumulate or world
// rotations to use as they are, which side of the product the parent goes on,
// and whether the reconstructed frame is the rotation or its transpose.
//
// AND ONE CANDIDATE THAT IS NOT A READING OF THAT BLOCK. Mode::Positions is
// what MotionRetargeter does under RetargetSource::Positions: take each bone's
// world direction from the `ric` positions and build the minimal arc that
// reaches it. Scored by the same measure, it must come out at zero -- the test
// pulls a direction back through a rotation built from that same direction, so
// anything but zero means the construction is wrong, not that the data is. It
// is here as the control the 60 readings are compared against: with the best
// of them at 51.7 degrees, a zero in this row is the argument for driving the
// character from positions.

#include "pch.h"

#include "FloodDiffusion.hpp"
#include "MotionFeatures.hpp"
#include "MotionRetarget.hpp"
#include "SceneGraph.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace NeuralModelIntegrateTestbed;
using DirectX::SimpleMath::Matrix;
using DirectX::SimpleMath::Quaternion;
using DirectX::SimpleMath::Vector3;

namespace {

const std::filesystem::path kDirectory = "resources/FloodDiffusion";
const std::filesystem::path kConfig = "resources/retarget/mixamo.txt";
const std::filesystem::path kCharacter =
    "resources/microphone_and_animated_char/scene_only_animated_char.gltf";
constexpr float kRadiansToDegrees = 57.2957795f;

// Joint names in the SMPL order HumanML3D uses. The sides are what the mapping
// file pairs with mixamo bones by name, so if index 1 is not the LEFT hip the
// character is driven mirrored.
const char* JointName(int joint) {
    static const char* names[22] = {
        "pelvis",   "L_hip",      "R_hip",      "spine1",   "L_knee",
        "R_knee",   "spine2",     "L_ankle",    "R_ankle",  "spine3",
        "L_foot",   "R_foot",     "neck",       "L_collar", "R_collar",
        "head",     "L_shoulder", "R_shoulder", "L_elbow",  "R_elbow",
        "L_wrist",  "R_wrist"};
    return (joint >= 0 && joint < 22) ? names[joint] : "?";
}

// The same reconstruction MotionRetarget.cc does. Repeated here so the probe
// can vary it; `transpose` is the variant where the six numbers describe the
// frame's rows rather than its columns, which inverts every rotation.
Quaternion Cont6dToQuaternion(const float* c, bool transpose) {
    Vector3 xRaw(c[0], c[1], c[2]);
    Vector3 yRaw(c[3], c[4], c[5]);
    const float xLength = xRaw.Length();
    if (xLength < 1e-8f) {
        return Quaternion::Identity;
    }
    const Vector3 x = xRaw / xLength;
    Vector3 z = x.Cross(yRaw);
    const float zLength = z.Length();
    if (zLength < 1e-8f) {
        return Quaternion::Identity;
    }
    z /= zLength;
    const Vector3 y = z.Cross(x);
    Matrix m = Matrix::Identity;
    if (transpose) {
        m._11 = x.x; m._21 = x.y; m._31 = x.z;
        m._12 = y.x; m._22 = y.y; m._32 = y.z;
        m._13 = z.x; m._23 = z.y; m._33 = z.z;
    } else {
        m._11 = x.x; m._12 = x.y; m._13 = x.z;
        m._21 = y.x; m._22 = y.y; m._23 = y.z;
        m._31 = z.x; m._32 = z.y; m._33 = z.z;
    }
    return Quaternion::CreateFromRotationMatrix(m);
}

// The rest offset of the bone ending at each joint, from HumanML3D's
// utils/paramUtil.py t2m_raw_offsets. A property of the skeleton, which is why
// the test below can treat it as the quantity that must not move. Mode::Positions
// needs the whole table rather than the subset resources/retarget/mixamo.txt
// carries, so that its score covers the same joints the rotation readings are
// scored on.
const Vector3 kRawOffsets[22] = {
    {0, 0, 0},  {1, 0, 0},  {-1, 0, 0}, {0, 1, 0},  {0, -1, 0}, {0, -1, 0},
    {0, 1, 0},  {0, -1, 0}, {0, -1, 0}, {0, 1, 0},  {0, 0, 1},  {0, 0, 1},
    {0, 1, 0},  {1, 0, 0},  {-1, 0, 0}, {0, 0, 1},  {0, -1, 0}, {0, -1, 0},
    {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
};

enum class Mode {
    LocalForward,  // accumulated as MotionRetarget.cc does
    LocalReverse,  // the same, with the product the other way round
    World,         // the entries used as they are
    Positions,     // not the block at all: the ric positions, as the retarget
                   // reads them under RetargetSource::Positions
};

const char* ModeName(Mode mode) {
    switch (mode) {
        case Mode::LocalForward: return "local, parent first";
        case Mode::LocalReverse: return "local, child first ";
        case Mode::World:        return "world, as-is       ";
        case Mode::Positions:    return "RIC POSITIONS      ";
    }
    return "?";
}

struct Candidate {
    Mode mode = Mode::LocalForward;
    bool transpose = false;
    float rootScale = 1.0f;
    // Where joint j's six numbers sit. By joint index, which is what the code
    // assumes, or in the order a walk of the chains visits the joints -- a
    // plausible difference between one export and another, and one that would
    // scramble the pose while leaving the block perfectly well formed.
    bool chainOrder = false;
};

// Slot per joint for the chain-ordered layout: each chain contributes its
// joints after the first, in order, skipping any already placed. That comes to
// exactly the 21 entries the block holds, which is why it is worth testing.
const std::vector<int>& ChainOrderSlots(
    int joints, const std::vector<std::vector<int>>& chains) {
    static std::vector<int> slots;
    if (!slots.empty()) {
        return slots;
    }
    slots.assign(static_cast<std::size_t>(joints), -1);
    int next = 0;
    for (const std::vector<int>& chain : chains) {
        for (std::size_t k = 1; k < chain.size(); ++k) {
            const int joint = chain[k];
            if (joint < joints && slots[static_cast<std::size_t>(joint)] < 0) {
                slots[static_cast<std::size_t>(joint)] = next++;
            }
        }
    }
    return slots;
}

void WorldRotations(const float* feature, int joints, const Candidate& candidate,
                    float rootAngle, const std::vector<std::vector<int>>& chains,
                    const MotionRetargeter& retargeter,
                    const DirectX::XMFLOAT3* positions,
                    std::vector<Quaternion>& out) {
    if (candidate.mode == Mode::Positions) {
        // Exactly the two steps MotionRetargeter::ApplyFrame takes under
        // RetargetSource::Positions, through the same two entry points, so
        // this scores the shipped construction rather than a copy of it. The
        // character's own rest direction is replaced by the SOURCE skeleton's
        // rest offset, which is what lets the probe do this without a scene --
        // and which is also what makes the expected score exactly zero, since
        // the rest offset is then the quantity the test recovers.
        static std::vector<Vector3> directions;
        out.assign(static_cast<std::size_t>(joints), Quaternion::Identity);
        if (!retargeter.SourceBoneDirections(positions, directions)) {
            return;
        }
        for (int joint = 1; joint < joints && joint < 22; ++joint) {
            const std::size_t j = static_cast<std::size_t>(joint);
            if (directions[j].LengthSquared() < 1e-12f) {
                continue;
            }
            Vector3 rest = kRawOffsets[joint];
            if (rest.Length() < 1e-6f) {
                continue;
            }
            rest.Normalize();
            out[j] = MotionRetargeter::AlignDirections(rest, directions[j]);
        }
        return;
    }

    const int rotationOffset = 4 + (joints - 1) * 3;
    const std::vector<int>& slots = ChainOrderSlots(joints, chains);
    const auto entry = [&](int joint) {
        const int slot = candidate.chainOrder
                             ? slots[static_cast<std::size_t>(joint)]
                             : joint - 1;
        return feature + rotationOffset + (slot < 0 ? 0 : slot) * 6;
    };
    out.assign(static_cast<std::size_t>(joints), Quaternion::Identity);
    out[0] = Quaternion::CreateFromAxisAngle(Vector3::UnitY,
                                             rootAngle * candidate.rootScale);
    if (candidate.mode == Mode::World) {
        for (int joint = 1; joint < joints; ++joint) {
            out[static_cast<std::size_t>(joint)] =
                Cont6dToQuaternion(entry(joint), candidate.transpose);
        }
        return;
    }
    for (const std::vector<int>& chain : chains) {
        Quaternion accumulated = out[static_cast<std::size_t>(chain.front())];
        for (std::size_t k = 1; k < chain.size(); ++k) {
            const int joint = chain[k];
            const Quaternion local =
                Cont6dToQuaternion(entry(joint), candidate.transpose);
            accumulated = candidate.mode == Mode::LocalForward
                              ? Quaternion::Concatenate(accumulated, local)
                              : Quaternion::Concatenate(local, accumulated);
            out[static_cast<std::size_t>(joint)] = accumulated;
        }
    }
}

struct Score {
    Candidate candidate;
    float worst = 0.0f;
    double mean = 0.0;
    std::vector<Vector3> offsets;
    std::vector<float> spreads;
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (!std::filesystem::exists(kDirectory / "pipeline.txt") ||
        !std::filesystem::exists(kConfig)) {
        std::printf("resources/FloodDiffusion or the retarget config is not "
                    "staged; nothing to probe\n");
        return 0;
    }

    MotionRetargeter retargeter;
    std::string error;
    if (!retargeter.LoadConfig(kConfig, &error)) {
        std::printf("loading %s failed: %s\n", kConfig.string().c_str(), error.c_str());
        return 1;
    }

    // The CPU provider: this is about numbers, and it needs no device.
    FloodDiffusionPipeline pipeline;
    FloodDiffusionOptions options;
    options.latentFrames = 20;
    options.seed = 1234;
    options.device = FloodDiffusionDevice::Cpu;
    options.textEncoderDevice = FloodDiffusionDevice::Cpu;
    options.vaeDevice = FloodDiffusionDevice::Cpu;
    // A prompt that turns, so a root-rotation error cannot hide: walking in a
    // straight line leaves the cumulative angle near zero, where every root
    // convention agrees.
    const char* prompt = "a person walks forward and turns around";
    if (!pipeline.Load(kDirectory, options, &error) ||
        !pipeline.Begin(prompt, options, &error)) {
        std::printf("the pipeline refused: %s\n", error.c_str());
        return 1;
    }
    int guard = 0;
    while (pipeline.IsRunning() && guard++ < 4000) {
        if (!pipeline.Step(&error)) {
            std::printf("Step failed: %s\n", error.c_str());
            return 1;
        }
    }

    const int frames = pipeline.FrameCount();
    const int joints = pipeline.JointCount();
    const int motionDim = pipeline.Config().motionDim;
    std::printf("prompt: \"%s\"\n%d frames, %d joints, motionDim %d\n", prompt, frames,
                joints, motionDim);
    if (frames < 4) {
        std::printf("not enough frames to compare\n");
        return 1;
    }

    float lowest = 1e9f;
    float highest = -1e9f;
    for (int frame = 0; frame < frames; ++frame) {
        lowest = std::min(lowest, pipeline.FrameRootAngle(frame));
        highest = std::max(highest, pipeline.FrameRootAngle(frame));
    }
    std::printf("root angle over the take: %.0f to %.0f deg\n",
                lowest * kRadiansToDegrees, highest * kRadiansToDegrees);

    // --- is the block well formed at all? ----------------------------------
    // Every reading below can only be WRONG if the block is a rotation of
    // something. A 6D rotation is two 3-vectors that Gram-Schmidt turns into a
    // frame; in training data that pair is exactly orthonormal and a network's
    // output is approximately so. Numbers far from 1 and 0 here would mean the
    // block is untrained output rather than a pose, and no reading would help.
    {
        const int rotationOffset = 4 + (joints - 1) * 3;
        double lengthSum = 0.0;
        double dotSum = 0.0;
        int samples = 0;
        for (int frame = 0; frame < frames; ++frame) {
            const float* feature = pipeline.FeatureFrame(frame);
            if (feature == nullptr) continue;
            for (int joint = 1; joint < joints; ++joint) {
                const float* c = feature + rotationOffset + (joint - 1) * 6;
                const Vector3 x(c[0], c[1], c[2]);
                const Vector3 y(c[3], c[4], c[5]);
                const float xLength = x.Length();
                const float yLength = y.Length();
                lengthSum += xLength;
                if (xLength > 1e-8f && yLength > 1e-8f) {
                    dotSum += std::fabs((x / xLength).Dot(y / yLength));
                }
                ++samples;
            }
        }
        std::printf("\nthe block as data: mean |x| %.4f (want 1), mean |x.y| %.4f "
                    "(want 0)\n",
                    samples ? lengthSum / samples : 0.0,
                    samples ? dotSum / samples : 0.0);
    }

    // --- the sweep ----------------------------------------------------------
    const std::vector<std::vector<int>>& chains = HumanML3DChains();
    const float rootScales[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    const Mode modes[] = {Mode::LocalForward, Mode::LocalReverse, Mode::World};

    std::vector<Candidate> candidates;
    for (const Mode mode : modes) {
        for (const bool transpose : {false, true}) {
            for (const bool chainOrder : {false, true}) {
                for (const float scale : rootScales) {
                    Candidate candidate;
                    candidate.mode = mode;
                    candidate.transpose = transpose;
                    candidate.rootScale = scale;
                    candidate.chainOrder = chainOrder;
                    candidates.push_back(candidate);
                }
            }
        }
    }
    // The control, which varies in none of those: there is nothing to sweep
    // over when the pose comes from the positions.
    {
        Candidate candidate;
        candidate.mode = Mode::Positions;
        candidates.push_back(candidate);
    }

    std::vector<Score> scores;
    std::vector<Quaternion> world;
    for (const Candidate& candidate : candidates) {
        std::vector<Vector3> sums(static_cast<std::size_t>(joints), Vector3::Zero);
        std::vector<int> counts(static_cast<std::size_t>(joints), 0);
        std::vector<std::vector<Vector3>> samples(
            static_cast<std::size_t>(joints));

        for (int frame = 0; frame < frames; ++frame) {
            const float* feature = pipeline.FeatureFrame(frame);
            const DirectX::XMFLOAT3* positions = pipeline.Frame(frame);
            if (feature == nullptr || positions == nullptr) continue;
            WorldRotations(feature, joints, candidate,
                           pipeline.FrameRootAngle(frame), chains, retargeter,
                           positions, world);
            for (const std::vector<int>& chain : chains) {
                for (std::size_t k = 1; k < chain.size(); ++k) {
                    const int joint = chain[k];
                    const int parent = chain[k - 1];
                    if (joint >= joints || parent >= joints) continue;
                    Vector3 bone(positions[joint].x - positions[parent].x,
                                 positions[joint].y - positions[parent].y,
                                 positions[joint].z - positions[parent].z);
                    if (bone.Length() < 1e-6f) continue;
                    bone.Normalize();
                    Quaternion inverse = world[static_cast<std::size_t>(joint)];
                    inverse.Conjugate();
                    const Vector3 offset = Vector3::Transform(bone, inverse);
                    sums[static_cast<std::size_t>(joint)] += offset;
                    ++counts[static_cast<std::size_t>(joint)];
                    samples[static_cast<std::size_t>(joint)].push_back(offset);
                }
            }
        }

        Score score;
        score.candidate = candidate;
        score.offsets.assign(static_cast<std::size_t>(joints), Vector3::Zero);
        score.spreads.assign(static_cast<std::size_t>(joints), 0.0f);
        double meanSum = 0.0;
        int counted = 0;
        for (int joint = 0; joint < joints; ++joint) {
            const std::size_t j = static_cast<std::size_t>(joint);
            if (counts[j] == 0) continue;
            Vector3 mean = sums[j] / static_cast<float>(counts[j]);
            if (mean.Length() > 1e-6f) mean.Normalize();
            score.offsets[j] = mean;
            for (const Vector3& sample : samples[j]) {
                const float dot =
                    std::max(-1.0f, std::min(1.0f, sample.Dot(mean)));
                score.spreads[j] = std::max(
                    score.spreads[j], std::acos(dot) * kRadiansToDegrees);
            }
            score.worst = std::max(score.worst, score.spreads[j]);
            meanSum += score.spreads[j];
            ++counted;
        }
        score.mean = counted ? meanSum / counted : 0.0;
        scores.push_back(std::move(score));
    }

    std::sort(scores.begin(), scores.end(),
              [](const Score& a, const Score& b) { return a.mean < b.mean; });

    std::printf("\nevery reading, best first. A reading that is RIGHT scores near\n"
                "zero: the rest offset it implies cannot depend on the pose.\n"
                "RIC POSITIONS is not a reading of the block at all: it is what\n"
                "RetargetSource::Positions does, and its score is the control.\n");
    for (const Score& score : scores) {
        if (score.candidate.mode == Mode::Positions) {
            std::printf("  %s  %-10s %-11s           "
                        "   mean %6.2f deg, worst %6.2f deg\n",
                        ModeName(score.candidate.mode), "n/a", "n/a", score.mean,
                        score.worst);
            continue;
        }
        std::printf("  %s  %-10s %-11s root %+4.1fa   mean %6.2f deg, "
                    "worst %6.2f deg\n",
                    ModeName(score.candidate.mode),
                    score.candidate.transpose ? "transposed" : "as-is",
                    score.candidate.chainOrder ? "chain order" : "joint order",
                    score.candidate.rootScale, score.mean, score.worst);
    }

    // The two numbers the choice of source turns on, side by side, so a future
    // export can be judged by re-running this rather than by eye.
    {
        const Score* positionScore = nullptr;
        const Score* bestRotation = nullptr;
        for (const Score& score : scores) {
            if (score.candidate.mode == Mode::Positions) {
                if (positionScore == nullptr) positionScore = &score;
            } else if (bestRotation == nullptr) {
                bestRotation = &score;
            }
        }
        if (positionScore != nullptr && bestRotation != nullptr) {
            std::printf("\nthe choice of source: positions %.2f deg of mean scatter,"
                        " best rotation reading %.2f deg\n",
                        positionScore->mean, bestRotation->mean);
            if (positionScore->mean > 0.5f) {
                std::printf("  THE POSITION PATH SHOULD SCORE ZERO BY"
                            " CONSTRUCTION. It does not, so the construction"
                            " itself is wrong -- look there before looking at"
                            " the character.\n");
            }
        }
    }

    // Deliberately the best ROTATION reading, not scores.front(): the position
    // control scores zero, and a per-joint table of it would only print
    // kRawOffsets back. What is worth reading per joint is where the block
    // disagrees.
    const Score* bestReading = nullptr;
    for (const Score& score : scores) {
        if (score.candidate.mode != Mode::Positions) {
            bestReading = &score;
            break;
        }
    }
    if (bestReading == nullptr) {
        return 0;
    }
    const Score& best = *bestReading;
    std::printf("\nbest reading, per joint (%s, %s, %s, root %+.1fa):\n",
                ModeName(best.candidate.mode),
                best.candidate.transpose ? "transposed" : "as-is",
                best.candidate.chainOrder ? "chain order" : "joint order",
                best.candidate.rootScale);
    for (int joint = 1; joint < joints; ++joint) {
        const std::size_t j = static_cast<std::size_t>(joint);
        std::printf("  %-11s offset (%+.3f, %+.3f, %+.3f)  spread %6.2f deg\n",
                    JointName(joint), best.offsets[j].x, best.offsets[j].y,
                    best.offsets[j].z, best.spreads[j]);
    }
    // --- the character against the overlay ---------------------------------
    // The measurement that stands in for looking at the screen. Everything
    // above scores the SOURCE pose; this poses the real rigged character
    // through the real ApplyFrame and asks the one question a person asks of
    // the app: does the character agree with the skeleton drawn beside it?
    //
    // Both are measured as bone directions, because that is all the two have
    // in common -- the character's proportions are its own, so the joints
    // cannot be compared position by position. Scale and placement do not
    // enter: a direction is unaffected by both.
    if (std::filesystem::exists(kCharacter)) {
        fastgltf::Parser parser(fastgltf::Extensions::KHR_lights_punctual);
        auto data = fastgltf::GltfDataBuffer::FromPath(kCharacter);
        auto asset = data.error() == fastgltf::Error::None
                         ? parser.loadGltf(data.get(), kCharacter.parent_path(),
                                           fastgltf::Options::LoadExternalBuffers)
                         : fastgltf::Expected<fastgltf::Asset>(data.error());
        if (asset.error() != fastgltf::Error::None) {
            std::printf("\ncannot read the character: %s\n",
                        fastgltf::getErrorMessage(asset.error()).data());
            return 0;
        }

        SceneGraph graph;
        const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> noBuffers;
        graph.Build(asset.get(), asset->defaultScene.value_or(0), noBuffers, 0);

        // The character's own file is PLACED in a real scene -- scenes.json
        // turns this one 180 degrees so it faces the camera -- and the rest
        // pose is resolved at Bind, after that placement is set. So the probe
        // has to bind under a placement too, or it measures a case the app
        // never runs.
        const char* placementName = std::getenv("PROBE_PLACEMENT");
        const float placementYaw = placementName ? std::atof(placementName) : 0.0f;
        if (placementYaw != 0.0f) {
            SceneTransform placed;
            placed.rotation[1] = placementYaw;
            graph.SetAssetPlacement(0, PlacementMatrix(placed));
            std::printf("\nasset placement: %.0f degrees of yaw\n", placementYaw);
        }
        const Matrix scenePlacement = Matrix::Identity;

        std::vector<std::string> missing;
        const int bound = retargeter.BindTo(graph, 0, &missing);
        std::printf("\nthe character: %d of %zu bones bound\n", bound,
                    retargeter.Bones().size());

        // Frames spread over the take rather than the first few, so a reading
        // that is right only near the start -- which is what a root-angle
        // error looks like -- cannot pass.
        const int sampled[] = {frames / 8, frames / 4, frames / 2,
                               (frames * 3) / 4, frames - 1};

        for (const RetargetSource source :
             {RetargetSource::Positions, RetargetSource::Rotations}) {
            retargeter.SetSource(source);
            double sum = 0.0;
            float worst = 0.0f;
            int counted = 0;
            for (const int frame : sampled) {
                const float* feature = pipeline.FeatureFrame(frame);
                const DirectX::XMFLOAT3* positions = pipeline.Frame(frame);
                if (feature == nullptr || positions == nullptr) continue;
                std::string error;
                if (!retargeter.ApplyFrame(graph, feature, motionDim, positions,
                                           pipeline.FrameRootAngle(frame),
                                           Vector3(0.0f, 0.9f, 0.0f), 1.0f,
                                           scenePlacement, &error)) {
                    std::printf("  frame %d refused: %s\n", frame,
                                error.c_str());
                    continue;
                }
                graph.UpdateTransforms(scenePlacement);

                for (const RetargetBone& bone : retargeter.Bones()) {
                    if (!bone.restResolved || !bone.destinationNode.has_value() ||
                        !bone.childDestinationNode.has_value() ||
                        bone.childJoint < 0 || bone.sourceJoint < 0) {
                        continue;
                    }
                    // The overlay's own direction for this bone: straight from
                    // the positions it is drawn from.
                    Vector3 overlay(
                        positions[bone.childJoint].x - positions[bone.sourceJoint].x,
                        positions[bone.childJoint].y - positions[bone.sourceJoint].y,
                        positions[bone.childJoint].z - positions[bone.sourceJoint].z);
                    if (overlay.Length() < 1e-6f) continue;
                    overlay.Normalize();
                    // The motion is authored in the character's FILE space and
                    // carried out through the placement, exactly as the hips
                    // target is, so the direction it asks for is the placed
                    // one. Comparing against the unplaced direction would
                    // report an error the app does not have.
                    overlay = Vector3::TransformNormal(
                        overlay, graph.PlacementForNode(*bone.destinationNode));
                    overlay.Normalize();

                    const Matrix& a =
                        graph.Nodes()[*bone.destinationNode].worldTransform;
                    const Matrix& b =
                        graph.Nodes()[*bone.childDestinationNode].worldTransform;
                    Vector3 got(b._41 - a._41, b._42 - a._42, b._43 - a._43);
                    if (got.Length() < 1e-6f) continue;
                    got.Normalize();

                    const float angle =
                        std::acos(std::max(-1.0f, std::min(1.0f, got.Dot(overlay)))) *
                        kRadiansToDegrees;
                    sum += angle;
                    worst = std::max(worst, angle);
                    ++counted;
                }
            }
            // THE DEGREE OF FREEDOM A DIRECTION CANNOT CARRY, measured where
            // it is most visible. The hips bone points UP in both skeletons,
            // so the minimal arc that aims it says nothing about yaw -- and
            // yaw about the vertical is which way the character FACES. The
            // check: the character's own hip-to-hip vector against the
            // motion's. A bone-direction test cannot see this at all, because
            // every bone still points exactly where it was asked to.
            {
                std::size_t leftLeg = 0;
                std::size_t rightLeg = 0;
                bool haveLegs = false;
                for (const RetargetBone& bone : retargeter.Bones()) {
                    if (bone.destinationName == "mixamorig:LeftUpLeg" &&
                        bone.destinationNode.has_value()) {
                        leftLeg = *bone.destinationNode;
                        haveLegs = rightLeg != 0;
                    }
                    if (bone.destinationName == "mixamorig:RightUpLeg" &&
                        bone.destinationNode.has_value()) {
                        rightLeg = *bone.destinationNode;
                        haveLegs = leftLeg != 0;
                    }
                }
                double facingSum = 0.0;
                float facingWorst = 0.0f;
                int facingCount = 0;
                for (const int frame : sampled) {
                    const float* feature = pipeline.FeatureFrame(frame);
                    const DirectX::XMFLOAT3* positions = pipeline.Frame(frame);
                    if (!haveLegs || feature == nullptr || positions == nullptr) {
                        continue;
                    }
                    std::string error;
                    if (!retargeter.ApplyFrame(graph, feature, motionDim, positions,
                                               pipeline.FrameRootAngle(frame),
                                               Vector3(0.0f, 0.9f, 0.0f), 1.0f,
                                               scenePlacement, &error)) {
                        continue;
                    }
                    graph.UpdateTransforms(scenePlacement);
                    const Matrix& l = graph.Nodes()[leftLeg].worldTransform;
                    const Matrix& r = graph.Nodes()[rightLeg].worldTransform;
                    Vector3 character(r._41 - l._41, r._42 - l._42, r._43 - l._43);
                    // Joint 1 is the LEFT hip and 2 the RIGHT, in SMPL order.
                    Vector3 motion(positions[2].x - positions[1].x,
                                   positions[2].y - positions[1].y,
                                   positions[2].z - positions[1].z);
                    if (character.Length() < 1e-6f || motion.Length() < 1e-6f) {
                        continue;
                    }
                    character.Normalize();
                    motion.Normalize();
                    motion = Vector3::TransformNormal(
                        motion, graph.PlacementForNode(leftLeg));
                    motion.Normalize();
                    const float angle =
                        std::acos(std::max(-1.0f, std::min(1.0f,
                                                           character.Dot(motion)))) *
                        kRadiansToDegrees;
                    facingSum += angle;
                    facingWorst = std::max(facingWorst, angle);
                    ++facingCount;
                }
                if (facingCount > 0) {
                    std::printf("  %-18s hip-to-hip off by mean %6.2f deg, "
                                "worst %6.2f deg\n",
                                source == RetargetSource::Positions
                                    ? "from positions"
                                    : "from rotations",
                                facingSum / facingCount, facingWorst);
                }
            }

            std::printf("  %-18s %d bone-frames, mean %6.2f deg, worst %6.2f deg\n",
                        source == RetargetSource::Positions ? "from positions"
                                                            : "from rotations",
                        counted, counted ? sum / counted : 0.0, worst);
        }
        std::printf("  The position row is the character agreeing with the overlay\n"
                    "  by construction, and must be near zero. The rotation row is\n"
                    "  how far the old path put it from what the overlay draws.\n");
    }

    std::printf("\nThe offsets should be cardinal for the t2m skeleton: +X for the\n"
                "left hip, -X for the right, -Y down each leg. Reversed sides there\n"
                "would mean the block is mirrored; a large spread under EVERY\n"
                "reading means no reading fits and the block does not describe the\n"
                "pose the ric block does.\n");
    return 0;
}
