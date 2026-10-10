// Smoke test for the HumanML3D feature decoding in Engine/MotionFeatures.cc
// and the Unigram tokenizer in Engine/Tokenizer.cc.
//
// Neither needs a device or a model, so this runs standalone and fast. It is
// where the arithmetic that a renderer cannot sanity-check for you is pinned
// down: a wrong quaternion or a one-frame shift in the root integration still
// produces a skeleton that moves, and looks plausible until you measure it.
//
// Two kinds of assertion, deliberately:
//
//   * Against tools/motion_reference.py, an independent numpy transcription of
//     FloodDiffusion's own utils/motion_process.py. That catches coding slips.
//   * Against properties that hold whatever either transcription says -- a
//     constant velocity must integrate to a straight line, a pure rotation
//     must not change a bone's length. Those catch a shared misreading of the
//     format, which agreeing transcriptions cannot.
#include "pch.h"

#include "MotionFeatures.hpp"
#include "Tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using NeuralModelIntegrateTestbed::HumanML3DChains;
using NeuralModelIntegrateTestbed::MotionFeatureLayout;
using NeuralModelIntegrateTestbed::StreamJointRecovery;
using NeuralModelIntegrateTestbed::UnigramTokenizer;

namespace {

int g_failures = 0;

void Expect(const char* what, bool condition) {
    std::printf("  %-52s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

void ExpectNear(const char* what, float got, float expected, float epsilon) {
    const bool ok = std::fabs(got - expected) <= epsilon;
    std::printf("  %-52s %11.6f expected %11.6f  %s\n", what, got, expected,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

constexpr int kJoints = 22;
constexpr int kMotionDim = 263;

// Must match feature() in tools/motion_reference.py. Small integers over 16 so
// every value is exact in float32 and both sides build identical input.
float Feature(int frameIndex, int component) {
    const int value = (frameIndex * kMotionDim + component) % 37 - 18;
    return static_cast<float>(value) / 16.0f;
}

void BuildFrame(int frameIndex, std::vector<float>& out) {
    out.resize(kMotionDim);
    for (int c = 0; c < kMotionDim; ++c) {
        out[static_cast<std::size_t>(c)] = Feature(frameIndex, c);
    }
}

// --- the reference table, from tools/motion_reference.py -------------------

struct Expected {
    int frame;
    int joint;
    float x, y, z;
};

constexpr Expected kReference[] = {
    { 0,  0,      0.0000000f,     -0.9375000f,      0.0000000f},
    { 0,  1,     -0.8750000f,     -0.8125000f,     -0.7500000f},
    { 0,  9,      0.6250000f,      0.6875000f,      0.7500000f},
    { 0, 21,      0.5625000f,      0.6250000f,      0.6875000f},
    { 1,  0,     -0.1106389f,     -0.6875000f,      1.4548764f},
    { 1,  1,     -0.1070670f,     -0.5625000f,      2.2552590f},
    { 1,  9,      0.1177825f,      0.9375000f,      0.1458888f},
    { 1, 21,      0.1084138f,      0.8750000f,      0.2337792f},
    { 2,  0,      0.9880483f,     -0.4375000f,      1.3302070f},
    { 2,  1,      1.4223652f,     -0.3125000f,      1.2098169f},
    { 2,  9,      1.0568018f,     -1.1250000f,      2.8761063f},
    { 2, 21,      1.1449549f,      1.1250000f,      2.8696585f},
    { 5,  0,      0.8368025f,      0.3125000f,      0.2615800f},
    { 5,  1,      1.1950065f,      0.4375000f,      0.7737469f},
    { 5,  9,      0.4099119f,     -0.3750000f,     -0.0652639f},
    { 5, 21,      0.3495200f,     -0.4375000f,     -0.1298032f},
    {11,  0,     -1.7243736f,     -0.5000000f,     -0.1260124f},
    {11,  1,     -1.8914795f,     -0.3750000f,      0.3850043f},
    {11,  9,     -3.1159747f,      1.1250000f,     -0.8027534f},
    {11, 21,     -0.9505605f,      1.0625000f,     -1.4170734f},
};

void TestLayout() {
    std::printf("feature layout\n");

    MotionFeatureLayout layout;
    Expect("263 features decode to 22 joints",
           MotionFeatureLayout::FromMotionDim(kMotionDim, &layout) &&
               layout.joints == kJoints);
    Expect("22 joints pack into 263 features",
           MotionFeatureLayout::ForJoints(kJoints).motionDim == kMotionDim);
    Expect("the ric block starts at 4 and spans 63",
           layout.RicOffset() == 4 && layout.RicCount() == 63);

    // A width that is not a HumanML3D vector has to be rejected, not rounded to
    // the nearest joint count: the recovery would read past the ric block and
    // the only symptom would be a skeleton of noise.
    Expect("262 features are rejected",
           !MotionFeatureLayout::FromMotionDim(262, &layout));
    Expect("264 features are rejected",
           !MotionFeatureLayout::FromMotionDim(264, &layout));
    Expect("0 features are rejected", !MotionFeatureLayout::FromMotionDim(0, &layout));

    const auto& chains = HumanML3DChains();
    bool chainsValid = chains.size() == 5;
    int bones = 0;
    for (const auto& chain : chains) {
        if (chain.size() < 2) chainsValid = false;
        for (const int joint : chain) {
            if (joint < 0 || joint >= kJoints) chainsValid = false;
        }
        bones += static_cast<int>(chain.size()) - 1;
    }
    Expect("five chains, all indices within 22 joints", chainsValid);
    Expect("the chains describe 21 bones", bones == 21);
}

void TestAgainstReference() {
    std::printf("\nagainst tools/motion_reference.py\n");

    StreamJointRecovery recovery(kJoints);
    std::vector<DirectX::XMFLOAT3> joints(kJoints);
    std::vector<float> frame;

    int cursor = 0;
    int checked = 0;
    float worst = 0.0f;
    for (int frameIndex = 0; frameIndex <= 11; ++frameIndex) {
        BuildFrame(frameIndex, frame);
        if (!recovery.ProcessFrame(frame.data(), joints.data())) {
            Expect("ProcessFrame accepted the frame", false);
            return;
        }
        while (cursor < static_cast<int>(std::size(kReference)) &&
               kReference[cursor].frame == frameIndex) {
            const Expected& expected = kReference[cursor];
            const DirectX::XMFLOAT3& got = joints[static_cast<std::size_t>(expected.joint)];
            worst = std::max(worst, std::fabs(got.x - expected.x));
            worst = std::max(worst, std::fabs(got.y - expected.y));
            worst = std::max(worst, std::fabs(got.z - expected.z));
            ++checked;
            ++cursor;
        }
    }
    std::printf("  %-52s %d\n", "sampled joint positions compared", checked);
    ExpectNear("largest disagreement with the numpy reference", worst, 0.0f, 2e-6f);
    Expect("every reference row was reached",
           cursor == static_cast<int>(std::size(kReference)));
}

void TestStraightLine() {
    std::printf("\nconstant velocity integrates to a straight line\n");

    // Holds whatever the reference says: with no angular velocity the heading
    // never changes, so the root advances by exactly the linear velocity each
    // frame -- and starts at the origin, because frame 0 applies the previous
    // frame's velocity, of which there is none.
    StreamJointRecovery recovery(kJoints);
    std::vector<float> frame(kMotionDim, 0.0f);
    frame[0] = 0.0f;    // no yaw velocity
    frame[1] = 0.25f;   // X velocity
    frame[2] = -0.5f;   // Z velocity
    frame[3] = 1.0f;    // height

    std::vector<DirectX::XMFLOAT3> joints(kJoints);
    bool exact = true;
    for (int frameIndex = 0; frameIndex < 6; ++frameIndex) {
        recovery.ProcessFrame(frame.data(), joints.data());
        const float expectedX = 0.25f * static_cast<float>(frameIndex);
        const float expectedZ = -0.5f * static_cast<float>(frameIndex);
        if (std::fabs(joints[0].x - expectedX) > 1e-6f ||
            std::fabs(joints[0].z - expectedZ) > 1e-6f ||
            std::fabs(joints[0].y - 1.0f) > 1e-6f) {
            exact = false;
        }
    }
    Expect("the root advances one velocity per frame, from the origin", exact);

    // The one-frame delay is the whole point: applying the CURRENT frame's
    // velocity instead would put the root at 0.25 on the very first frame.
    StreamJointRecovery single(kJoints);
    single.ProcessFrame(frame.data(), joints.data());
    ExpectNear("frame 0 is at the origin, not one step ahead", joints[0].x, 0.0f, 1e-6f);
}

void TestRotationPreservesBoneLengths() {
    std::printf("\nyaw rotation preserves distances\n");

    // A pure rotation cannot change how far a joint sits from the root. This
    // holds independently of the format reading, so it catches a wrong
    // quaternion -- including the easy mistake of rotating by the heading
    // instead of its inverse, which would still look like motion.
    StreamJointRecovery recovery(kJoints);
    std::vector<float> frame(kMotionDim, 0.0f);
    frame[0] = 0.3f;   // a steady yaw, so the heading accumulates
    frame[3] = 0.9f;   // height

    // An offset oblique to every axis, so a mistaken axis assignment shows up.
    for (int joint = 0; joint < kJoints - 1; ++joint) {
        frame[4 + joint * 3 + 0] = 0.3f + 0.01f * static_cast<float>(joint);
        frame[4 + joint * 3 + 1] = -0.7f;
        frame[4 + joint * 3 + 2] = 0.5f;
    }

    std::vector<DirectX::XMFLOAT3> joints(kJoints);
    std::vector<float> firstDistances;
    float worstDrift = 0.0f;

    // The control has to measure the heading itself, not the root position:
    // the linear velocity above is zero, so the root sits at the origin however
    // the heading turns, and watching it move would assert nothing. Joint 1's
    // offset from the root is what the heading actually rotates.
    float firstOffsetX = 0.0f;
    float firstOffsetZ = 0.0f;
    float largestSwing = 0.0f;
    for (int frameIndex = 0; frameIndex < 16; ++frameIndex) {
        recovery.ProcessFrame(frame.data(), joints.data());

        const float offsetX = joints[1].x - joints[0].x;
        const float offsetZ = joints[1].z - joints[0].z;
        if (frameIndex == 0) {
            firstOffsetX = offsetX;
            firstOffsetZ = offsetZ;
        } else {
            const float dx = offsetX - firstOffsetX;
            const float dz = offsetZ - firstOffsetZ;
            largestSwing = std::max(largestSwing, std::sqrt(dx * dx + dz * dz));
        }

        for (int joint = 1; joint < kJoints; ++joint) {
            const float dx = joints[static_cast<std::size_t>(joint)].x - joints[0].x;
            const float dy = joints[static_cast<std::size_t>(joint)].y - joints[0].y;
            const float dz = joints[static_cast<std::size_t>(joint)].z - joints[0].z;
            const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (frameIndex == 0) {
                firstDistances.push_back(distance);
            } else {
                worstDrift = std::max(
                    worstDrift,
                    std::fabs(distance -
                              firstDistances[static_cast<std::size_t>(joint - 1)]));
            }
        }
    }
    // Note the root's Y comes from the feature vector while the joints' Y comes
    // from ric, so the vertical component of the offset is included above only
    // because both are measured against the same root.
    // 0.3 rad per frame over 16 frames is more than a full turn, so the
    // offset has to swing by at least its own length at some point.
    Expect("the heading actually rotated the joint offset", largestSwing > 0.5f);
    ExpectNear("joint distance to the root is unchanged", worstDrift, 0.0f, 1e-5f);
}

void TestSmoothing() {
    std::printf("\noutput smoothing\n");

    std::vector<float> first(kMotionDim, 0.0f);
    first[3] = 1.0f;
    for (int joint = 0; joint < kJoints - 1; ++joint) {
        first[4 + joint * 3 + 0] = 1.0f;
    }
    std::vector<float> second = first;
    for (int joint = 0; joint < kJoints - 1; ++joint) {
        second[4 + joint * 3 + 0] = 3.0f;
    }

    std::vector<DirectX::XMFLOAT3> joints(kJoints);

    StreamJointRecovery unsmoothed(kJoints, 1.0f);
    unsmoothed.ProcessFrame(first.data(), joints.data());
    unsmoothed.ProcessFrame(second.data(), joints.data());
    ExpectNear("alpha 1 passes the second frame through", joints[1].x, 3.0f, 1e-6f);

    StreamJointRecovery smoothed(kJoints, 0.25f);
    smoothed.ProcessFrame(first.data(), joints.data());
    ExpectNear("the first smoothed frame is untouched", joints[1].x, 1.0f, 1e-6f);
    smoothed.ProcessFrame(second.data(), joints.data());
    // 0.25 * 3 + 0.75 * 1
    ExpectNear("alpha 0.25 blends a quarter of the new frame", joints[1].x, 1.5f, 1e-6f);

    StreamJointRecovery reset(kJoints, 1.0f);
    reset.ProcessFrame(second.data(), joints.data());
    reset.Reset();
    reset.ProcessFrame(second.data(), joints.data());
    ExpectNear("Reset returns the root to the origin", joints[0].x, 0.0f, 1e-6f);
}

void TestTokenizer() {
    std::printf("\nUnigram tokenizer\n");

    const std::filesystem::path path = "resources/FloodDiffusion/tokenizer.bin";
    if (!std::filesystem::exists(path)) {
        std::printf("  %-52s %s\n", "tokenizer.bin is absent", "skipped");
        std::printf("  run tools/prepare_flooddiffusion.py to exercise this section\n");
        return;
    }

    UnigramTokenizer tokenizer;
    std::string error;
    if (!tokenizer.Load(path, &error)) {
        std::printf("  loading failed: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    std::printf("  %-52s %zu\n", "vocabulary entries", tokenizer.VocabSize());

    // The empty prompt is what classifier-free guidance uses as its
    // unconditional branch, so it has to come out as exactly the
    // end-of-sequence token rather than nothing at all.
    const std::vector<int64_t> empty = tokenizer.Encode("", 128);
    Expect("the empty prompt is one end-of-sequence token",
           empty.size() == 1 && empty[0] == tokenizer.EndOfSequenceId());

    const std::string prompt = "a person walks forward and waves both hands";
    const std::vector<int64_t> ids = tokenizer.Encode(prompt, 128);
    std::printf("  %-52s %zu\n", "tokens for the sample prompt", ids.size());
    Expect("the prompt ends with end-of-sequence",
           !ids.empty() && ids.back() == tokenizer.EndOfSequenceId());

    bool inRange = true;
    int unknowns = 0;
    for (const int64_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= tokenizer.VocabSize()) {
            inRange = false;
        }
        if (id == tokenizer.UnknownId()) ++unknowns;
    }
    Expect("every id is within the vocabulary", inRange);
    Expect("an ordinary English prompt needs no unknown token", unknowns == 0);

    // The round trip is the real check on the segmentation: Unigram is
    // lossless, so concatenating the chosen tokens and turning the metaspace
    // back into spaces has to reproduce the prompt exactly, modulo the leading
    // space the metaspace scheme always prepends. If the Viterbi pass dropped
    // or duplicated a span, this is where it shows.
    const std::string decoded = tokenizer.Decode(ids);
    const std::string expected = " " + prompt;
    Expect("the tokens decode back to the prompt", decoded == expected);
    if (decoded != expected) {
        std::printf("    got      %s\n    expected %s\n", decoded.c_str(),
                    expected.c_str());
    }

    // Truncation has to leave room for the token the post-processor appends,
    // so a cap of 8 yields 7 pieces plus end-of-sequence.
    const std::vector<int64_t> capped = tokenizer.Encode(prompt, 8);
    Expect("a cap of 8 produces exactly 8 ids", capped.size() == 8);
    Expect("the capped sequence still ends with end-of-sequence",
           !capped.empty() && capped.back() == tokenizer.EndOfSequenceId());

    // Repeated whitespace must not produce empty pieces.
    const std::vector<int64_t> spaced = tokenizer.Encode("  a   person  ", 128);
    const std::vector<int64_t> plain = tokenizer.Encode("a person", 128);
    Expect("runs of whitespace tokenise like single spaces", spaced == plain);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    TestLayout();
    TestAgainstReference();
    TestStraightLine();
    TestRotationPreservesBoneLengths();
    TestSmoothing();
    TestTokenizer();

    std::printf("\n%s\n", g_failures == 0 ? "all checks passed"
                                          : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
