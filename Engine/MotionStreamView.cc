#include "pch.h"
#include "MotionStreamView.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace NeuralModelIntegrateTestbed {

namespace {

// One colour per kinematic chain, from FloodDiffusion's own
// utils/render_skeleton.py palette, so a skeleton here reads the same way as
// one rendered by their tooling.
constexpr DirectX::XMFLOAT4 kChainColours[] = {
    {254.0f / 255.0f, 178.0f / 255.0f, 26.0f / 255.0f, 1.0f},   // right leg
    {0.0f, 170.0f / 255.0f, 1.0f, 1.0f},                        // left leg
    {19.0f / 255.0f, 70.0f / 255.0f, 134.0f / 255.0f, 1.0f},    // spine to head
    {1.0f, 182.0f / 255.0f, 0.0f, 1.0f},                        // right arm
    {0.0f, 212.0f / 255.0f, 126.0f / 255.0f, 1.0f},             // left arm
};

constexpr DirectX::XMFLOAT4 kJointColour = {1.0f, 1.0f, 1.0f, 1.0f};
constexpr DirectX::XMFLOAT4 kTrailColour = {0.35f, 0.35f, 0.40f, 1.0f};

}  // namespace

bool MotionStreamView::Load(const std::filesystem::path& directory,
                            ID3D12Device* device,
                            IDMLDevice* dmlDevice,
                            std::string* error) {
    m_directory = directory;
    m_device = device;
    m_dmlDevice = dmlDevice;

    // On by default: it is the fastest configuration measured (60 ms against
    // the dynamic GPU path's 966 ms and the CPU's 87 ms for 8 latents). The
    // cost is that `time` becomes fixed, so the whole latent buffer is
    // submitted each step and the motion differs from the growing-prefix
    // sampler; the checkbox in ShowImgui turns it off.
    m_options.staticShapes = true;

    // Decoding follows the PLAYHEAD here, not the sampler. This view is the
    // streaming consumer the pipeline was built for: one vae_decoder_step is
    // four motion frames, which at 20 fps is 0.2 s of playback, so the stream
    // needs about five decodes a second however far ahead the sampler has run
    // -- and at 60 latents it runs a long way ahead, finishing 11.85 s of
    // motion in roughly 2.1 s. Eagerly that spends the whole 70 ms decode
    // budget inside the generating burst, on motion the viewer has not reached
    // and may never ask for.
    //
    // Twelve frames of lookahead is three latents, 0.6 s of slack: enough that
    // a dropped frame or two does not starve playback, small enough that the
    // catch-up at the start of a generation is three decodes rather than
    // sixty. Update calls DecodeAhead every tick, which is what makes this the
    // caller's job rather than Step's.
    m_options.decodeAheadFrames = 12;

    // And the stream runs until Stop rather than for a fixed length, which is
    // what diffusion forcing is for: the window slides instead of the sequence
    // ending, so there is no clip boundary. With the decode paced to the
    // playhead above, the two together regulate the whole thing to real time
    // -- the sampler cannot slide a latent out before the decoder has had it,
    // so a stream left running costs about what playback costs.
    m_options.continuous = true;

    // And the sampler is held a second and a half in front of the decoder.
    // Without this it runs ahead until the window stops it -- 60 latents of
    // already-decided motion, about twelve seconds -- and a prompt change
    // cannot be seen until all of it has played. Measured, the worst reaction
    // goes from 127 frames to 35, and it stops depending on the sampling step
    // count, which is the part that made it feel arbitrary.
    m_options.samplerLeadFrames = 30;

    if (!std::filesystem::exists(directory / "pipeline.txt")) {
        m_status = "not staged; run tools/prepare_flooddiffusion.py";
        m_hasStatusError = false;
        return false;
    }

    // Without a DirectML device there is no GPU provider to attach, so asking
    // for one would fail at session creation. Say so by falling back here,
    // where it can be reported, rather than at the first Generate.
    if (m_dmlDevice == nullptr) {
        m_options.device = FloodDiffusionDevice::Cpu;
        m_options.textEncoderDevice = FloodDiffusionDevice::Cpu;
    }

    std::string reason;
    if (!Reload(&reason)) {
        if (error != nullptr) {
            *error = reason;
        }
        return false;
    }
    return true;
}

void MotionStreamView::AttachScene(SceneGraph* scene,
                                   const std::filesystem::path& retargetConfig) {
    m_scene = scene;
    m_targets.clear();
    m_selectedTarget = -1;
    m_boundBones = 0;

    if (scene == nullptr) {
        m_retargetStatus = "no scene";
        return;
    }
    std::string error;
    if (!m_retargeter.LoadConfig(retargetConfig, &error)) {
        m_retargetStatus = error;
        return;
    }
    m_targets = m_retargeter.DiscoverTargets(*scene);
    if (m_targets.empty()) {
        m_retargetStatus = "the scene has no skinned character";
        return;
    }

    // Default to the first character whose rig the mapping actually covers,
    // rather than to whichever skin comes first -- a scene can hold skinned
    // props that match nothing.
    const int needed = static_cast<int>(m_retargeter.Bones().size());
    for (std::size_t i = 0; i < m_targets.size(); ++i) {
        if (m_targets[i].matchedBones == needed) {
            m_selectedTarget = static_cast<int>(i);
            break;
        }
    }
    RebindTarget();
}

void MotionStreamView::RebindTarget() {
    m_boundBones = 0;
    if (m_scene == nullptr || !m_retargeter.IsLoaded()) {
        return;
    }
    if (m_selectedTarget < 0 ||
        m_selectedTarget >= static_cast<int>(m_targets.size())) {
        m_retargetStatus = "no character selected";
        // Leave whatever pose the character already had; putting it back to
        // the bind pose here would fight the asset's own animation.
        return;
    }

    std::vector<std::string> missing;
    m_boundBones = m_retargeter.BindTo(
        *m_scene, m_targets[static_cast<std::size_t>(m_selectedTarget)].skinIndex,
        &missing);
    m_characterMetrics = m_retargeter.MeasureBindPose(
        *m_scene, m_targets[static_cast<std::size_t>(m_selectedTarget)].skinIndex);
    // Size it as soon as there is something to size against; otherwise wait
    // for the first decoded frames.
    m_fitPending = true;

    const int needed = static_cast<int>(m_retargeter.Bones().size());
    if (m_boundBones == needed) {
        m_retargetStatus = "bound " + std::to_string(m_boundBones) + " bones";
    } else {
        m_retargetStatus = "bound " + std::to_string(m_boundBones) + " of " +
                           std::to_string(needed) + " bones; first unmatched: " +
                           (missing.empty() ? std::string("none") : missing.front());
    }
}

float MotionStreamView::MotionHipsHeight() const {
    const int frames = m_pipeline.FrameCount();
    const int joints = m_pipeline.JointCount();
    if (frames == 0 || joints <= 0) {
        return 0.0f;
    }

    // Hips above the lowest joint, averaged: the same measure taken of the
    // character, so the ratio is a like-for-like leg length rather than a
    // comparison of two differently-defined heights.
    double total = 0.0;
    int counted = 0;
    for (int frame = 0; frame < frames; ++frame) {
        const DirectX::XMFLOAT3* pose = m_pipeline.Frame(frame);
        if (pose == nullptr) {
            continue;
        }
        float lowest = pose[0].y;
        for (int joint = 1; joint < joints; ++joint) {
            lowest = std::min(lowest, pose[joint].y);
        }
        total += static_cast<double>(pose[0].y - lowest);
        ++counted;
    }
    return counted > 0 ? static_cast<float>(total / counted) : 0.0f;
}

bool MotionStreamView::FitToMotion(std::string* summary) {
    const float motionHips = MotionHipsHeight();
    if (!m_characterMetrics.valid || m_characterMetrics.hipsHeight <= 1e-4f ||
        motionHips <= 1e-4f) {
        if (summary != nullptr) {
            *summary = "nothing to measure yet";
        }
        return false;
    }

    // How much bigger the MOTION is than the character, which is what
    // ApplyFrame divides its hips target by. It does not resize the character:
    // the rig keeps the size its file authored and its placement gives it, so
    // a character placed at 0.8 stays at 0.8 of its own size rather than 0.8
    // of the source skeleton's.
    m_retargetScale = motionHips / m_characterMetrics.hipsHeight;

    // The character's feet come out at the ground plane on their own once the
    // root height is expressed at the character's proportions: the motion's
    // own root height already puts its feet near zero, and dividing by the
    // ratio above keeps that true at the character's size. So the offset is
    // cleared rather than used to paper over a scale error.
    m_retargetOffset = {0.0f, 0.0f, 0.0f};

    char text[192];
    std::snprintf(text, sizeof(text),
                  "character hips %.3f m, motion hips %.3f m -> motion / %.3f",
                  m_characterMetrics.hipsHeight, motionHips, m_retargetScale);
    m_fitSummary = text;
    if (summary != nullptr) {
        *summary = m_fitSummary;
    }
    return true;
}

void MotionStreamView::PoseCharacter() {
    if (m_scene == nullptr || m_boundBones == 0 || m_selectedTarget < 0) {
        return;
    }
    const int frameCount = m_pipeline.FrameCount();
    if (frameCount == 0) {
        return;
    }
    const float* feature = m_pipeline.FeatureFrame(m_currentFrame);
    if (feature == nullptr) {
        return;
    }

    if (m_fitPending && m_autoFit) {
        if (FitToMotion(nullptr)) {
            m_fitPending = false;
        }
    }

    // In the CHARACTER'S OWN FILE space, not world space. ApplyFrame carries it
    // out through that file's placement, so the motion follows the glTF it is
    // retargeted onto: the character travels with the file when it is moved,
    // turns with it, and its stride scales with it.
    //
    // What that means for the offset below: it is a nudge within the character's
    // file, so a file already placed in the scene needs no offset to undo that
    // placement. The motion's own root height still places its feet at about
    // zero, so aiming the hips there puts the feet on the file's ground plane --
    // provided the height is expressed at the character's proportions, which is
    // what m_retargetScale arranges.
    //
    // ApplyFrame divides the whole target by that ratio, so the offset is
    // PRE-MULTIPLIED by it to come out the other side unchanged. Without this
    // a nudge of 0.1 m would land as 0.086 m against a 1.16 ratio, which is
    // not what a nudge in the character's own file space should mean.
    const float motionScale = m_retargetScale > 1e-6f ? m_retargetScale : 1.0f;
    DirectX::SimpleMath::Vector3 target(m_retargetOffset.x * motionScale,
                                        m_retargetOffset.y * motionScale,
                                        m_retargetOffset.z * motionScale);
    const DirectX::XMFLOAT3* joints = m_pipeline.Frame(m_currentFrame);
    if (joints == nullptr && m_retargetFromPositions) {
        // Driving from positions needs them. A frame whose feature vector
        // exists but whose positions have not been recovered yet is a frame to
        // skip, not a failure to unbind over.
        return;
    }
    if (joints != nullptr) {
        // Height always follows the motion; only travel is optional. Dropping
        // the height with the travel would leave the character sunk at the
        // origin instead of standing.
        target.y += joints[0].y;
        if (m_retargetRootMotion) {
            target.x += joints[0].x;
            target.z += joints[0].z;
        }
    }

    std::string error;
    m_retargeter.SetSource(m_retargetFromPositions ? RetargetSource::Positions
                                                   : RetargetSource::Rotations);
    if (!m_retargeter.ApplyFrame(*m_scene, feature, m_pipeline.Config().motionDim,
                                 joints,
                                 m_pipeline.FrameRootAngle(m_currentFrame), target,
                                 m_retargetScale, m_scenePlacement, &error)) {
        m_retargetStatus = error;
        m_boundBones = 0;  // stop retrying every frame with the same failure
    }
}

bool MotionStreamView::Reload(std::string* error) {
    // Load drops the sessions, the lazily-created text encoder and the cached
    // prompt features before rebuilding, so calling it again is how a provider
    // change takes effect.
    m_playbackTime = 0.0f;
    m_currentFrame = 0;

    std::string reason;
    if (!m_pipeline.Load(m_directory, m_options, &reason, m_device, m_dmlDevice)) {
        m_status = reason;
        m_hasStatusError = true;
        if (error != nullptr) {
            *error = reason;
        }
        return false;
    }
    // Seeded from the export rather than left at the zero that means "take the
    // export's default", so the sampling-steps control below shows the real
    // number the next generation will use instead of a 0 nobody can read.
    // Only when it has never been set, or reloading would discard the user's
    // choice every time a provider changed.
    if (m_options.denoiseSteps <= 0) {
        m_options.denoiseSteps = m_pipeline.Config().noiseSteps;
    }

    m_status = "loaded; press Generate";
    m_hasStatusError = false;
    return true;
}

bool MotionStreamView::Generate(std::string* error) {
    if (!IsLoaded()) {
        return false;
    }
    std::string reason;
    if (!m_pipeline.Begin(m_prompt, m_options, &reason)) {
        m_status = reason;
        m_hasStatusError = true;
        if (error != nullptr) {
            *error = reason;
        }
        return false;
    }
    m_playbackTime = 0.0f;
    m_currentFrame = 0;
    m_playing = true;
    m_fitPending = true;
    m_status = "generating";
    m_hasStatusError = false;
    return true;
}

void MotionStreamView::Update(float elapsedSeconds) {
    if (!IsLoaded()) {
        return;
    }

    // Generation first, so a frame decoded this tick can be displayed this
    // tick rather than on the next one.
    //
    // The budget is a RATE converted by the measured frame delta, so the
    // sampler advances at the same speed whatever the frame rate is -- the
    // whole point of m_stepsPerSecond. The leftover fraction is carried, which
    // is what lets a rate below the frame rate advance at all.
    if (m_pipeline.IsRunning()) {
        m_stepBudget += elapsedSeconds * std::max(m_stepsPerSecond, 0.0f);
        // A ceiling on the catch-up after a hitch. StepTimer already clamps
        // its delta, so this only bites when the rate itself is enormous; what
        // it prevents is one frame spending unbounded time denoising and
        // making the next hitch worse.
        constexpr float kMaxStepsPerFrame = 16.0f;
        m_stepBudget = std::min(m_stepBudget, kMaxStepsPerFrame);

        std::string error;
        while (m_stepBudget >= 1.0f && m_pipeline.IsRunning()) {
            m_stepBudget -= 1.0f;
            if (!m_pipeline.Step(&error)) {
                m_status = error;
                m_hasStatusError = true;
                break;
            }
        }
    } else {
        // So a generation does not begin with a budget banked from however
        // long the view sat idle.
        m_stepBudget = 0.0f;
    }

    // Then decoding, as far as the playhead needs and no further. Before the
    // frameCount check below rather than after, because until this runs there
    // is no motion at all: with decodeAheadFrames set, Step denoises and
    // decodes nothing.
    if (m_pipeline.IsRunning()) {
        std::string error;
        const int playhead = static_cast<int>(m_playbackTime *
                                              FloodDiffusionPipeline::kFrameRate);
        if (!m_pipeline.DecodeAhead(playhead, &error)) {
            m_status = error;
            m_hasStatusError = true;
        }
    }
    if (!m_pipeline.IsRunning() && !m_hasStatusError && m_pipeline.FrameCount() > 0) {
        m_status = "complete";
    } else if (m_pipeline.IsRunning() && !m_hasStatusError &&
               m_pipeline.StepsTaken() >= m_pipeline.TotalSteps()) {
        // The sampler is done and the stream is not. What remains is waiting
        // for the consumer rather than for the GPU, and while playback is
        // paused it will wait indefinitely -- so "generating" would be a lie.
        m_status = "streaming";
    }

    const int frameCount = m_pipeline.FrameCount();
    if (frameCount == 0) {
        m_currentFrame = 0;
        return;
    }

    if (m_playing) {
        m_playbackTime += elapsedSeconds * m_playbackSpeed;
    }

    const float duration = static_cast<float>(frameCount) /
                           FloodDiffusionPipeline::kFrameRate;
    if (m_playbackTime >= duration) {
        if (m_pipeline.IsRunning()) {
            // Playback has caught up with generation. Hold on the newest frame
            // rather than looping, which would otherwise snap back to the start
            // every time the sampler fell behind.
            m_playbackTime = duration;
        } else if (m_loop) {
            m_playbackTime = std::fmod(m_playbackTime, duration);
        } else {
            m_playbackTime = duration;
            m_playing = false;
        }
    }

    const int frame = static_cast<int>(m_playbackTime *
                                       FloodDiffusionPipeline::kFrameRate);
    m_currentFrame = std::clamp(frame, 0, frameCount - 1);

    PoseCharacter();
}

DirectX::XMFLOAT3 MotionStreamView::Placed(const DirectX::XMFLOAT3& joint) const {
    return DirectX::XMFLOAT3(joint.x * m_scale + m_offset.x,
                             joint.y * m_scale + m_offset.y,
                             joint.z * m_scale + m_offset.z);
}

void MotionStreamView::Draw(
    DirectX::PrimitiveBatch<DirectX::VertexPositionColor>& batch) const {
    if (!IsLoaded() || m_pipeline.FrameCount() == 0) {
        return;
    }
    if (m_hideSkeletonWhenRetargeting && m_boundBones > 0) {
        return;
    }
    const DirectX::XMFLOAT3* joints = m_pipeline.Frame(m_currentFrame);
    if (joints == nullptr) {
        return;
    }

    const auto& chains = HumanML3DChains();
    const int jointCount = m_pipeline.JointCount();

    // The root's path so far, which makes it obvious whether the motion is
    // travelling or marching in place.
    if (m_drawTrail) {
        for (int frame = 1; frame <= m_currentFrame; ++frame) {
            const DirectX::XMFLOAT3* previous = m_pipeline.Frame(frame - 1);
            const DirectX::XMFLOAT3* current = m_pipeline.Frame(frame);
            if (previous == nullptr || current == nullptr) {
                continue;
            }
            batch.DrawLine(
                DirectX::VertexPositionColor(Placed(previous[0]), kTrailColour),
                DirectX::VertexPositionColor(Placed(current[0]), kTrailColour));
        }
    }

    for (std::size_t chain = 0; chain < chains.size(); ++chain) {
        const DirectX::XMFLOAT4& colour =
            kChainColours[chain % std::size(kChainColours)];
        const auto& indices = chains[chain];
        for (std::size_t i = 0; i + 1 < indices.size(); ++i) {
            const int from = indices[i];
            const int to = indices[i + 1];
            if (from >= jointCount || to >= jointCount) {
                continue;
            }
            batch.DrawLine(
                DirectX::VertexPositionColor(Placed(joints[from]), colour),
                DirectX::VertexPositionColor(Placed(joints[to]), colour));
        }
    }

    // A small axis cross at each joint. Cheaper and clearer at this scale than
    // geometry, and it shows joints the chains do not connect.
    if (m_drawJoints) {
        const float size = m_jointSize;
        for (int joint = 0; joint < jointCount; ++joint) {
            const DirectX::XMFLOAT3 centre = Placed(joints[joint]);
            for (int axis = 0; axis < 3; ++axis) {
                DirectX::XMFLOAT3 a = centre;
                DirectX::XMFLOAT3 b = centre;
                float* const aComponent = axis == 0 ? &a.x : (axis == 1 ? &a.y : &a.z);
                float* const bComponent = axis == 0 ? &b.x : (axis == 1 ? &b.y : &b.z);
                *aComponent -= size;
                *bComponent += size;
                batch.DrawLine(DirectX::VertexPositionColor(a, kJointColour),
                               DirectX::VertexPositionColor(b, kJointColour));
            }
        }
    }
}

void MotionStreamView::ShowImgui(UiSection section) {
    switch (section) {
        case UiSection::Neural:      ShowNeuralUi(); break;
        case UiSection::Performance: ShowPerformanceUi(); break;
        case UiSection::Render:
        case UiSection::Scene:
            break;
    }
}

// Where the sampler runs and what a take cost to produce. Separate from the
// controls because these are the numbers the device and pin-shapes choices
// exist to move, and comparing two configurations means reading them against
// the rest of the frame's cost rather than against the prompt box.
void MotionStreamView::ShowPerformanceUi() {
    if (!ImGui::CollapsingHeader("FloodDiffusion", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    if (!IsLoaded()) {
        ImGui::TextUnformatted("not loaded");
        return;
    }
    const FloodDiffusionConfig& config = m_pipeline.Config();
    ImGui::Text("%d joints, chunk %d, %d steps, cfg %.1f", config.joints,
                config.chunkSize, config.noiseSteps, config.cfgScale);
    ImGui::Text("sampler on %s, prompt encoder on %s, VAE on %s%s",
                ToString(m_pipeline.Device()),
                ToString(m_pipeline.TextEncoderDevice()),
                ToString(m_pipeline.VaeDevice()),
                m_pipeline.StaticShapes() ? ", shapes pinned" : "");
    // Worth stating next to the timings: on the GPU path these milliseconds
    // are spent on a queue of the pipeline's own, so they are not frame time.
    if (m_pipeline.ComputeQueue() != nullptr) {
        ImGui::TextUnformatted("submitting to its own compute queue, not the "
                               "renderer's");
    }
    ImGui::Text("denoise %.2f ms, decode %.2f ms, total %.0f ms",
                m_pipeline.LastDenoiseMs(), m_pipeline.LastDecodeMs(),
                m_pipeline.GenerationMs());
    // Per call rather than summed, because the maximum is what a frame budget
    // has to survive and the mean cannot tell you about it. One slow call is
    // what drops a frame; a mean that fits comfortably says nothing.
    const FloodDiffusionPipeline::PhaseTiming denoise = m_pipeline.DenoiseTiming();
    const FloodDiffusionPipeline::PhaseTiming decode = m_pipeline.DecodeTiming();
    if (denoise.calls > 0 || decode.calls > 0) {
        ImGui::Text("denoiser  x%-4d mean %.2f ms  max %.2f ms", denoise.calls,
                    denoise.meanMs, denoise.maxMs);
        if (m_pipeline.GpuLatents()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(submission)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("With the latents on the GPU a denoiser Run "
                                  "records and returns, so this row measures "
                                  "submission rather than execution. The row "
                                  "below is the work itself.");
            }
            // The timestamp-query figure, which is the only one of these that
            // is the actual GPU work on this path. Shown immediately under the
            // submission row so the two cannot be read apart: on this machine
            // submission understates execution about fourfold.
            const FloodDiffusionPipeline::PhaseTiming gpu =
                m_pipeline.DenoiseGpuTiming();
            if (gpu.calls > 0) {
                ImGui::Text("  on GPU  x%-4d mean %.2f ms  max %.2f ms", gpu.calls,
                            gpu.meanMs, gpu.maxMs);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "GPU execution, from D3D12 timestamp queries bracketing "
                        "the provider's work on the shared compute queue.\n\n"
                        "Compare it with the host path's denoiser rather than "
                        "with the submission row above: the difference between "
                        "those two is the copies this path removes, and it is "
                        "small. This path is faster at not waiting, not at "
                        "denoising.\n\n"
                        "A few calls can be missing: a slot is read only once "
                        "its fence has retired, and nothing waits for one.");
                }
            } else {
                ImGui::TextDisabled("  on GPU  timestamp queries unavailable");
            }
        }
        ImGui::Text("decoder   x%-4d mean %.2f ms  max %.2f ms", decode.calls,
                    decode.meanMs, decode.maxMs);
        ImGui::Text("first frame after %.0f ms", m_pipeline.FirstFrameMs());
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Start-up latency: what a viewer waits through "
                              "before anything moves. The ramp has to climb "
                              "far enough to finalise the first latent, so "
                              "this is denoising however cheap the decode is.");
        }
        // Only after a prompt change has actually produced a pose, since
        // before that there is nothing to report and a zero would read as an
        // instant reaction rather than as no reaction yet.
        if (m_pipeline.PromptLatencyMs() > 0.0) {
            ImGui::Text("new prompt reached the motion in %.0f ms",
                        m_pipeline.PromptLatencyMs());
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("From the prompt change to the first pose "
                                  "that could carry it. Latents already "
                                  "finalised never will, so this waits only "
                                  "for the ones still in flight -- which is "
                                  "why it beats the start-up latency above.");
            }
        }
    }
    // GPU work issued, which answers a different question from time spent:
    // whether the EP is compiling the graph or dispatching it operator by
    // operator. One dispatch per Run means it compiled.
    const FloodDiffusionPipeline::GpuWorkCounts work = m_pipeline.GpuWork();
    if (work.ortRuns > 0) {
        ImGui::Text("GPU work: %d EP dispatches over %d Runs%s", work.epDispatches,
                    work.ortRuns,
                    m_pipeline.StaticShapes() ? "" : "  (shapes NOT pinned)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "One fused DirectML dispatch per Run once shapes are pinned, "
                "which is the graph compiler working. Unpinned it is 1226 per "
                "Run -- the same graph dispatched operator by operator.\n\n"
                "The decoders are on the CPU provider by default, so they "
                "issue no GPU dispatches at all and this counts the "
                "denoiser's.\n\n"
                "Counted from the Run calls; the per-Run figure comes from ORT "
                "profiling at these shapes, since the provider's dispatches "
                "are inside its own command lists.");
        }
        if (work.computeDispatches > 0 || work.submissions > 0) {
            ImGui::Text("  plus ours: %d shader dispatches, %d submissions",
                        work.computeDispatches, work.submissions);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("The latent-step compute shader and the "
                                  "command lists this pipeline submits to its "
                                  "own queue. Only with the latents on the "
                                  "GPU.");
            }
        }
    }
    if (m_pipeline.ModelBytes() > 0) {
        ImGui::Text("model weights %.0f MB%s",
                    static_cast<double>(m_pipeline.ModelBytes()) / (1024.0 * 1024.0),
                    m_pipeline.TextEncoderLoaded() ? "" : " (encoder not loaded yet)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The .onnx files opened so far. The prompt "
                              "encoder is 1.1 GB of this and is loaded on "
                              "first use, which is why it is called out.");
        }
    }
    ImGui::Text("%d motion frames (%.2f s)", m_pipeline.FrameCount(),
                static_cast<float>(m_pipeline.FrameCount()) /
                    FloodDiffusionPipeline::kFrameRate);
    // The two halves separately, because with decoding paced to the playhead
    // they are deliberately not in step: the sampler finishes a long way ahead
    // and the decoder trails it by the lookahead. Seeing them together is how
    // a stalled stream -- playback paused, so nothing to decode for -- tells
    // itself apart from a stalled sampler.
    if (m_options.decodeAheadFrames > 0) {
        ImGui::Text("sampler at latent %d, decoder at %d (%d frames ahead of "
                    "the playhead)",
                    m_pipeline.FinalisedLatents(), m_pipeline.DecodedLatents(),
                    m_pipeline.FrameCount() - m_currentFrame);
    }
    // Non-zero means the stream is in its steady state rather than still
    // ramping up, and the gap to the decoder is the backpressure working.
    if (m_pipeline.ShiftedLatents() > 0) {
        ImGui::Text("streaming: %d latents slid out of the window",
                    m_pipeline.ShiftedLatents());
    }
    // The lead the cap governs, which is also the stale-prompt backlog: a
    // prompt change has to wait for exactly this much motion to play out.
    if (m_options.samplerLeadFrames > 0 && m_pipeline.IsRunning()) {
        const int lead = m_pipeline.FinalisedLatents() - m_pipeline.DecodedLatents();
        ImGui::Text("sampler %d latents ahead of the decoder (%.1f s of motion)",
                    lead,
                    static_cast<float>(lead * m_pipeline.Config().FramesPerLatent()) /
                        FloodDiffusionPipeline::kFrameRate);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Also the stale-prompt backlog: a prompt change "
                              "waits for this much motion to play out first.");
        }
    }
}

void MotionStreamView::ShowNeuralUi() {
    if (!ImGui::CollapsingHeader("FloodDiffusion motion",
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (!IsLoaded()) {
        ImGui::TextWrapped("%s", m_status.c_str());
        ImGui::TextWrapped("Stage the models with:\n"
                           "python tools/prepare_flooddiffusion.py "
                           "<FloodDiffusion>/onnx_models/tiny");
        return;
    }

    // The joint count and chunk size are reported in the Performance tab; this
    // is the one place the controls need them, to say how long the take the
    // "latent frames" slider is asking for will actually be.
    const FloodDiffusionConfig& config = m_pipeline.Config();

    // Changing a provider rebuilds the sessions, so it is deliberately a
    // separate action from Generate.
    const bool canUseGpu = m_dmlDevice != nullptr;
    const char* const deviceNames[] = {"CPU", "DirectML"};
    int samplerDevice = m_options.device == FloodDiffusionDevice::DirectML ? 1 : 0;
    int encoderDevice =
        m_options.textEncoderDevice == FloodDiffusionDevice::DirectML ? 1 : 0;
    int vaeDevice = m_options.vaeDevice == FloodDiffusionDevice::DirectML ? 1 : 0;
    bool devicesChanged = false;
    if (!canUseGpu) {
        ImGui::TextDisabled("no DirectML device; CPU only");
    } else {
        if (ImGui::Combo("sampler device", &samplerDevice, deviceNames, 2)) {
            m_options.device = samplerDevice == 1 ? FloodDiffusionDevice::DirectML
                                                  : FloodDiffusionDevice::Cpu;
            devicesChanged = true;
        }
        if (ImGui::Combo("encoder device", &encoderDevice, deviceNames, 2)) {
            m_options.textEncoderDevice = encoderDevice == 1
                                              ? FloodDiffusionDevice::DirectML
                                              : FloodDiffusionDevice::Cpu;
            devicesChanged = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The prompt encoder is 1.1 GB and runs once per "
                              "prompt. On this export the CPU is about five "
                              "times faster for it and keeps the weights out "
                              "of video memory.");
        }
        if (ImGui::Combo("VAE device", &vaeDevice, deviceNames, 2)) {
            m_options.vaeDevice = vaeDevice == 1 ? FloodDiffusionDevice::DirectML
                                                 : FloodDiffusionDevice::Cpu;
            devicesChanged = true;
        }
        if (ImGui::IsItemHovered()) {
            // The one control here where moving work to the GPU makes the
            // generation slower, so the number is on the tooltip rather than
            // left to be discovered.
            ImGui::SetTooltip("The decoder runs 59 times per generation and is "
                              "about four times faster on the CPU (1.4 ms a "
                              "call against 5.4), which is most of what the "
                              "whole generation costs.");
        }
    }
    // Pinning the input dimensions is what lets the DirectML EP compile the
    // graph rather than dispatch it per operator, and it is the single biggest
    // lever on GPU speed here -- 16x on the GPU path, and faster than the CPU
    // provider too. On by default (see Load); the tooltip records that it also
    // changes the sampler.
    bool pinned = m_options.staticShapes;
    if (ImGui::Checkbox("pin shapes (compile the graph)", &pinned)) {
        m_options.staticShapes = pinned;
        devicesChanged = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Fixes every symbolic input dimension so DirectML compiles the\n"
            "graph: 1334 dispatches per denoising call become 1, and an\n"
            "8-latent generation went from 958 ms to 61 ms.\n\n"
            "It also submits the WHOLE latent buffer every step instead of\n"
            "the live prefix, which a non-causal denoiser reacts to, so the\n"
            "motion differs (mean 0.11 m, max 0.52 m in testing). An export\n"
            "that masks self-attention over the padding would make it exact.\n\n"
            "No effect on the CPU provider, which does not fuse.");
    }

    // Also needs pinned shapes, for the same reason: a sliding window is a
    // fixed submitted length.
    ImGui::BeginDisabled(!m_options.staticShapes);
    bool streaming = m_options.continuous;
    if (ImGui::Checkbox("stream until Stop", &streaming)) {
        m_options.continuous = streaming;
        devicesChanged = true;
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Generate until Stop instead of for a fixed length. The live\n"
            "region is always one chunk wide, so past the ramp the sampler is\n"
            "in a steady state -- it slides the window forward, dropping\n"
            "finalised latents off the front and appending fresh noise at the\n"
            "back, rather than stopping. The ramp-up is identical to a finite\n"
            "generation of the same length.\n\n"
            "A latent is only dropped once it has been decoded, so the\n"
            "sampler waits for the playhead and the stream regulates itself\n"
            "to real time. Needs pinned shapes.");
    }
    if (!m_options.staticShapes && m_options.continuous) {
        m_options.continuous = false;
        devicesChanged = true;
    }

    // Needs both of the above, so it is offered only when they hold rather
    // than accepted and then refused by Load.
    const bool canKeepLatentsOnGpu =
        m_options.device == FloodDiffusionDevice::DirectML && m_options.staticShapes;
    ImGui::BeginDisabled(!canKeepLatentsOnGpu);
    bool resident = m_options.gpuLatents;
    if (ImGui::Checkbox("keep the latents on the GPU", &resident)) {
        m_options.gpuLatents = resident;
        devicesChanged = true;
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Binds every denoiser tensor into D3D12 buffers and runs the\n"
            "guidance combine and the Euler step as a compute shader, so the\n"
            "latents never leave the GPU and a step sends 520 bytes.\n\n"
            "It does NOT make a generation faster: 508.7 ms against 506.9 at\n"
            "60 latents. Denoising drops from 318.9 to 53.8 ms because this\n"
            "path never waits, and that time reappears in the decoder's\n"
            "readback. The generation is bound by GPU work, not by copies.\n\n"
            "The motion is identical. Needs DirectML and pinned shapes.");
    }
    if (!canKeepLatentsOnGpu && m_options.gpuLatents) {
        // Turning off either precondition would otherwise leave an option set
        // that the next Load refuses outright.
        m_options.gpuLatents = false;
    }

    if (devicesChanged) {
        std::string reloadError;
        if (!Reload(&reloadError)) {
            std::printf("FloodDiffusion: %s\n", reloadError.c_str());
        }
    }

    ImGui::InputText("prompt", m_prompt, sizeof(m_prompt));
    // Applied to the generation in flight the moment the edit finishes --
    // Enter, or clicking away -- rather than waiting for the next Generate.
    // Not per keystroke: encoding is a synchronous text-encoder run of about
    // 9 ms, and the half-typed prompts on the way would each pay for one and
    // condition the model on nonsense. SetPrompt is a no-op when the text has
    // not actually changed, so this costs nothing when the field is merely
    // visited.
    if (ImGui::IsItemDeactivatedAfterEdit() && m_pipeline.IsRunning()) {
        std::string error;
        if (!m_pipeline.SetPrompt(m_prompt, &error)) {
            m_status = error;
            m_hasStatusError = true;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Changing this while a stream is running re-conditions "
                          "it in place: the latents keep their noise levels, so "
                          "the motion transitions into the new prompt over about "
                          "a window rather than cutting.");
    }
    const int requestedFrames = m_options.latentFrames;
    ImGui::SliderInt("sampler lead (frames)", &m_options.samplerLeadFrames, 0, 120);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "How far the sampler may finalise ahead of the decoder before it "
            "waits. 0 is unbounded.\n\n"
            "This is what makes a prompt change feel prompt. Latents already "
            "finalised carry the prompt that was in force when they were "
            "denoised, and the decoder cannot skip them, so a change is "
            "invisible until they have played. Uncapped the sampler runs "
            "ahead until the window stops it -- about twelve seconds of "
            "motion at the shipped window.\n\n"
            "Costs nothing: the sampler already runs about six times faster "
            "than playback, so holding it here loses no motion. Lower it for "
            "a quicker reaction, raise it for more slack against a stutter.");
    }

    ImGui::SliderInt(m_options.continuous ? "window (latents)" : "latent frames",
                     &m_options.latentFrames, 2, 240);
    if (ImGui::IsItemHovered() && m_options.continuous) {
        ImGui::SetTooltip("Streaming, so this is not the length -- it is how "
                          "much finalised motion the denoiser still sees "
                          "behind the live chunk. The stream runs until Stop.");
    }
    // With shapes pinned the sequence length is compiled into the session, so
    // moving this slider has to rebuild it rather than just change the next
    // generation.
    if (m_options.staticShapes && m_options.latentFrames != requestedFrames) {
        std::string reloadError;
        if (!Reload(&reloadError)) {
            std::printf("FloodDiffusion: %s\n", reloadError.c_str());
        }
    }
    ImGui::SameLine();
    ImGui::Text("%.1f s%s", static_cast<float>(
                                config.vaeFramesFirst +
                                config.FramesPerLatent() *
                                    (m_options.latentFrames - 1)) /
                                FloodDiffusionPipeline::kFrameRate,
                m_options.continuous ? " of context" : "");

    // Sampling steps per chunk: how many denoising steps each latent gets
    // before it is finalised. Read by Begin rather than Load, so unlike the
    // window above it needs no session rebuild -- it takes effect on the next
    // Generate, which is said on the control rather than left to be
    // discovered.
    ImGui::SliderInt("sampling steps", &m_options.denoiseSteps, 1, 50);
    if (ImGui::IsItemHovered()) {
        const int exported = config.noiseSteps;
        ImGui::SetTooltip(
            "Denoising steps each chunk of latents gets before it is final.\n"
            "Generation cost is close to linear in this: the schedule runs\n"
            "roughly latentFrames / chunk_size x this many steps.\n\n"
            "This export is DISTILLED for %d, which is what it was trained to\n"
            "reach -- fewer will degrade the motion and many more mostly buys\n"
            "time rather than quality. It does not have to divide chunk_size\n"
            "(%d); a value below it simply finalises several latents per step.\n\n"
            "Takes effect on the next Generate: the schedule is built there, "
            "and changing the noise ramp under latents already in flight would "
            "make them jump.",
            exported, config.chunkSize);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("export default")) {
        m_options.denoiseSteps = config.noiseSteps;
    }

    int seed = static_cast<int>(m_options.seed);
    if (ImGui::InputInt("seed", &seed)) {
        m_options.seed = static_cast<uint32_t>(std::max(seed, 0));
    }
    ImGui::SliderFloat("smoothing", &m_options.smoothingAlpha, 0.05f, 1.0f);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Exponential moving average over joint positions. "
                          "1 is off.");
    }

    if (ImGui::Button("Generate")) {
        if (!m_pipeline.TextEncoderLoaded()) {
            // Said before the call, not after: loading the text encoder blocks
            // for several seconds and the window will not redraw until it is
            // done.
            std::printf("loading the FloodDiffusion text encoder (about 1.1 GB); "
                        "the window will not respond until it is ready\n");
        }
        std::string error;
        if (!Generate(&error)) {
            std::printf("FloodDiffusion: %s\n", error.c_str());
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop") && m_pipeline.IsRunning()) {
        // Stops the SAMPLER. Playback keeps going, because what has already
        // been finalised still has motion to give and cutting it off would
        // throw away frames the stream has paid for; the pipeline reports
        // itself finished once the decoder has drained.
        m_pipeline.Stop();
    }

    if (m_hasStatusError) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", m_status.c_str());
    } else {
        ImGui::TextUnformatted(m_status.c_str());
    }

    const int frameCount = m_pipeline.FrameCount();
    if (m_pipeline.TotalSteps() > 0) {
        const float progress = static_cast<float>(m_pipeline.StepsTaken()) /
                               static_cast<float>(m_pipeline.TotalSteps());
        char overlay[64];
        std::snprintf(overlay, sizeof(overlay), "%d / %d steps",
                      m_pipeline.StepsTaken(), m_pipeline.TotalSteps());
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
    }

    ImGui::Text("%d motion frames (%.2f s), showing %d", frameCount,
                static_cast<float>(frameCount) / FloodDiffusionPipeline::kFrameRate,
                m_currentFrame);
    if (m_pipeline.PromptTokenCount() > 0) {
        ImGui::Text("%d tokens:%s", m_pipeline.PromptTokenCount(),
                    m_pipeline.TokenisedPrompt().c_str());
    }

    ImGui::SliderFloat("steps / second", &m_stepsPerSecond, 1.0f, 480.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered()) {
        // The rate that keeps up with playback exactly: noise_steps finalise
        // one chunk, and a chunk is chunkSize * framesPerLatent motion frames,
        // which at kFrameRate is that many seconds of motion. Derived rather
        // than written down, so it stays right if the export changes.
        const float secondsPerChunk =
            static_cast<float>(config.chunkSize * config.FramesPerLatent()) /
            FloodDiffusionPipeline::kFrameRate;
        const float realTime = secondsPerChunk > 0.0f
                                   ? static_cast<float>(config.noiseSteps) /
                                         secondsPerChunk
                                   : 0.0f;
        ImGui::SetTooltip("Denoising steps per SECOND, converted with the "
                          "measured frame delta -- so generation runs at this "
                          "speed whatever the frame rate is.\n\n"
                          "%.0f steps/s matches playback exactly; this is "
                          "%.1fx real time. Each step costs 2-8 ms, so the "
                          "rate is also what it costs per second of frame "
                          "time.",
                          realTime,
                          realTime > 0.0f ? m_stepsPerSecond / realTime : 0.0f);
    }

    ImGui::Checkbox("playing", &m_playing);
    ImGui::SameLine();
    ImGui::Checkbox("loop", &m_loop);
    ImGui::SliderFloat("speed", &m_playbackSpeed, 0.1f, 3.0f);

    if (frameCount > 1) {
        int frame = m_currentFrame;
        if (ImGui::SliderInt("frame", &frame, 0, frameCount - 1)) {
            m_currentFrame = frame;
            m_playbackTime = static_cast<float>(frame) /
                             FloodDiffusionPipeline::kFrameRate;
            m_playing = false;
        }
    }

    // --- retargeting ------------------------------------------------------
    ImGui::SeparatorText("retarget to a character");
    if (!m_retargeter.IsLoaded()) {
        ImGui::TextWrapped("%s", m_retargetStatus.c_str());
        ImGui::TextWrapped("Run: python tools/prepare_retarget.py "
                           "<momask>/assets/mapping.json");
    } else if (m_targets.empty()) {
        ImGui::TextWrapped("%s", m_retargetStatus.c_str());
    } else {
        const int needed = static_cast<int>(m_retargeter.Bones().size());

        // "none" first, so index 0 is always valid and selecting nothing is a
        // deliberate choice rather than an empty combo.
        std::string preview = "none (skeleton lines only)";
        if (m_selectedTarget >= 0 &&
            m_selectedTarget < static_cast<int>(m_targets.size())) {
            preview = m_targets[static_cast<std::size_t>(m_selectedTarget)].name;
        }
        if (ImGui::BeginCombo("character", preview.c_str())) {
            if (ImGui::Selectable("none (skeleton lines only)", m_selectedTarget < 0)) {
                m_selectedTarget = -1;
                RebindTarget();
            }
            for (std::size_t i = 0; i < m_targets.size(); ++i) {
                const RetargetTarget& target = m_targets[i];
                char label[160];
                std::snprintf(label, sizeof(label), "%s  (%d joints, %d/%d bones)",
                              target.name.c_str(), target.jointCount,
                              target.matchedBones, needed);
                const bool selected = m_selectedTarget == static_cast<int>(i);
                if (ImGui::Selectable(label, selected)) {
                    m_selectedTarget = static_cast<int>(i);
                    RebindTarget();
                }
                if (target.matchedBones != needed && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("This rig does not carry every bone the "
                                      "mapping names, so it is rigged "
                                      "differently from a stock Mixamo "
                                      "character. It can still be driven "
                                      "partially.");
                }
            }
            ImGui::EndCombo();
        }

        if (m_boundBones > 0) {
            ImGui::TextUnformatted(m_retargetStatus.c_str());
        } else if (m_selectedTarget >= 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s",
                               m_retargetStatus.c_str());
        }

        if (ImGui::Button("fit to skeleton")) {
            std::string summary;
            if (!FitToMotion(&summary)) {
                m_fitSummary = summary;
            }
        }
        ImGui::SameLine();
        ImGui::Checkbox("auto", &m_autoFit);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Refit whenever a new take is generated.");
        }
        if (!m_fitSummary.empty()) {
            ImGui::TextUnformatted(m_fitSummary.c_str());
        }

        ImGui::SliderFloat("motion / character", &m_retargetScale, 0.1f, 10.0f,
                           "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("How much bigger the motion's skeleton is than "
                              "this character. The root height and travel are "
                              "DIVIDED by it, so the motion is brought down to "
                              "the character.\n\n"
                              "It does not resize the character: the rig keeps "
                              "the size its file and its scene placement give "
                              "it. Raise it to make the character take smaller "
                              "steps and stand lower, lower it for the "
                              "opposite. The fit above sets it from the ratio "
                              "of hip heights.");
        }
        ImGui::DragFloat3("retarget offset", &m_retargetOffset.x, 0.01f);
        ImGui::Checkbox("root motion", &m_retargetRootMotion);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Off keeps the character in place and drives only "
                              "its pose.");
        }
        ImGui::SameLine();
        ImGui::Checkbox("hide skeleton", &m_hideSkeletonWhenRetargeting);

        ImGui::Checkbox("drive from positions", &m_retargetFromPositions);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("On drives the bones from the same ric joint "
                              "positions the skeleton overlay is drawn from: "
                              "bone directions land 0.00 degrees from it and "
                              "the body faces within 2.2. Off uses the feature "
                              "vector's 6D rotation block, which "
                              "//Engine:retarget-probe measures as 51.7 degrees "
                              "of mean scatter away from describing the same "
                              "pose -- but which is the only source that "
                              "carries a limb's roll about its own axis.");
        }
    }

    ImGui::SeparatorText("skeleton lines");
    ImGui::Checkbox("joints", &m_drawJoints);
    ImGui::SameLine();
    ImGui::Checkbox("root trail", &m_drawTrail);
    ImGui::DragFloat3("offset", &m_offset.x, 0.01f);
    ImGui::SliderFloat("scale", &m_scale, 0.1f, 4.0f);
}

}  // namespace NeuralModelIntegrateTestbed
