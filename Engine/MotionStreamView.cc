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
                            ID3D12CommandQueue* queue,
                            IDMLDevice* dmlDevice,
                            std::string* error) {
    m_directory = directory;
    m_device = device;
    m_queue = queue;
    m_dmlDevice = dmlDevice;

    // On by default: it is the fastest configuration measured (60 ms against
    // the dynamic GPU path's 966 ms and the CPU's 87 ms for 8 latents). The
    // cost is that `time` becomes fixed, so the whole latent buffer is
    // submitted each step and the motion differs from the growing-prefix
    // sampler; the checkbox in ShowImgui turns it off.
    m_options.staticShapes = true;

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

    m_retargetScale = motionHips / m_characterMetrics.hipsHeight;

    // The character's feet come out at the ground plane on their own once the
    // legs are the right length: the motion's own root height already puts its
    // feet near zero, and the hips are aimed at that height in world space. So
    // the offset is cleared rather than used to paper over a scale error.
    m_retargetOffset = {0.0f, 0.0f, 0.0f};

    char text[192];
    std::snprintf(text, sizeof(text),
                  "character hips %.3f m, motion hips %.3f m -> scale %.3f",
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

    // A WORLD-space target, which is what makes the character stand on the
    // ground plane: the motion's own root height already places its feet at
    // about zero, so aiming the hips there puts the feet there too -- provided
    // the legs are the right length, which the scale above arranges.
    DirectX::SimpleMath::Vector3 target(m_retargetOffset.x, m_retargetOffset.y,
                                        m_retargetOffset.z);
    const DirectX::XMFLOAT3* joints = m_pipeline.Frame(m_currentFrame);
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
    if (!m_retargeter.ApplyFrame(*m_scene, feature, m_pipeline.Config().motionDim,
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
    if (!m_pipeline.Load(m_directory, m_options, &reason, m_device, m_queue,
                         m_dmlDevice)) {
        m_status = reason;
        m_hasStatusError = true;
        if (error != nullptr) {
            *error = reason;
        }
        return false;
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
    if (m_pipeline.IsRunning()) {
        std::string error;
        for (int step = 0; step < m_stepsPerFrame && m_pipeline.IsRunning(); ++step) {
            if (!m_pipeline.Step(&error)) {
                m_status = error;
                m_hasStatusError = true;
                break;
            }
        }
        if (!m_pipeline.IsRunning() && !m_hasStatusError) {
            m_status = "complete";
        }
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

void MotionStreamView::ShowImgui() {
    if (!ImGui::CollapsingHeader("FloodDiffusion motion")) {
        return;
    }

    if (!IsLoaded()) {
        ImGui::TextWrapped("%s", m_status.c_str());
        ImGui::TextWrapped("Stage the models with:\n"
                           "python tools/prepare_flooddiffusion.py "
                           "<FloodDiffusion>/onnx_models/tiny");
        return;
    }

    const FloodDiffusionConfig& config = m_pipeline.Config();
    ImGui::Text("%d joints, chunk %d, %d steps, cfg %.1f", config.joints,
                config.chunkSize, config.noiseSteps, config.cfgScale);
    ImGui::Text("sampler on %s, prompt encoder on %s%s",
                ToString(m_pipeline.Device()),
                ToString(m_pipeline.TextEncoderDevice()),
                m_pipeline.StaticShapes() ? ", shapes pinned" : "");

    // Changing a provider rebuilds the sessions, so it is deliberately a
    // separate action from Generate.
    const bool canUseGpu = m_dmlDevice != nullptr;
    const char* const deviceNames[] = {"CPU", "DirectML"};
    int samplerDevice = m_options.device == FloodDiffusionDevice::DirectML ? 1 : 0;
    int encoderDevice =
        m_options.textEncoderDevice == FloodDiffusionDevice::DirectML ? 1 : 0;
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

    if (devicesChanged) {
        std::string reloadError;
        if (!Reload(&reloadError)) {
            std::printf("FloodDiffusion: %s\n", reloadError.c_str());
        }
    }

    ImGui::InputText("prompt", m_prompt, sizeof(m_prompt));
    const int requestedFrames = m_options.latentFrames;
    ImGui::SliderInt("latent frames", &m_options.latentFrames, 2, 240);
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
    ImGui::Text("%.1f s", static_cast<float>(
                              config.vaeFramesFirst +
                              config.FramesPerLatent() * (m_options.latentFrames - 1)) /
                              FloodDiffusionPipeline::kFrameRate);

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
        // Nothing to cancel mid-step: a step is synchronous. Pausing playback
        // and leaving the sampler to finish is the honest behaviour here.
        m_playing = false;
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
    ImGui::Text("denoise %.2f ms, decode %.2f ms, total %.0f ms",
                m_pipeline.LastDenoiseMs(), m_pipeline.LastDecodeMs(),
                m_pipeline.GenerationMs());
    if (m_pipeline.PromptTokenCount() > 0) {
        ImGui::Text("%d tokens:%s", m_pipeline.PromptTokenCount(),
                    m_pipeline.TokenisedPrompt().c_str());
    }

    ImGui::SliderInt("steps / frame", &m_stepsPerFrame, 1, 8);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Denoising steps per rendered frame. Each costs 2-8 ms "
                          "on the CPU and yields about 1.9 motion frames, so one "
                          "step already generates faster than playback.");
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

        ImGui::SliderFloat("retarget scale", &m_retargetScale, 0.1f, 10.0f,
                           "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Resizes the whole rig. Set by the fit above from "
                              "the ratio of hip heights, since rotation-only "
                              "retargeting cannot change proportions.");
        }
        ImGui::DragFloat3("retarget offset", &m_retargetOffset.x, 0.01f);
        ImGui::Checkbox("root motion", &m_retargetRootMotion);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Off keeps the character in place and drives only "
                              "its pose.");
        }
        ImGui::SameLine();
        ImGui::Checkbox("hide skeleton", &m_hideSkeletonWhenRetargeting);
    }

    ImGui::SeparatorText("skeleton lines");
    ImGui::Checkbox("joints", &m_drawJoints);
    ImGui::SameLine();
    ImGui::Checkbox("root trail", &m_drawTrail);
    ImGui::DragFloat3("offset", &m_offset.x, 0.01f);
    ImGui::SliderFloat("scale", &m_scale, 0.1f, 4.0f);
}

}  // namespace NeuralModelIntegrateTestbed
