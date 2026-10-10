#pragma once

#include "FloodDiffusion.hpp"
#include "MotionRetarget.hpp"

#include <DirectXMath.h>
#include <PrimitiveBatch.h>
#include <VertexTypes.h>

#include <filesystem>
#include <memory>
#include <string>

// Drives a FloodDiffusionPipeline from the frame loop and draws what it has
// produced as skeleton lines.
//
// Generation and playback are deliberately separate clocks. The pipeline is
// advanced a bounded number of denoising steps per frame, while playback walks
// the decoded frames at HumanML3D's 20 fps -- so the skeleton starts moving
// within a few frames of pressing Generate and keeps moving while the rest is
// still being sampled. One denoising step costs 2-8 ms on the CPU and yields
// about 1.9 motion frames, so a single step per rendered frame generates
// roughly five times faster than playback consumes it.
//
// This owns no GPU state: Draw takes the caller's PrimitiveBatch, the same one
// the grid uses, so there is one line pipeline in the sample rather than two.
namespace NeuralModelIntegrateTestbed {

class MotionStreamView {
public:
    // Loads resources/FloodDiffusion if it is staged. A missing directory is
    // NOT an error -- the models are 1.2 GB and gitignored, so a fresh clone
    // legitimately has none; IsLoaded() stays false and the UI says what to
    // run. `error` receives the reason when a staged directory fails to load.
    //
    // The three device pointers are what let the DirectML provider run on the
    // renderer's own device and queue. Pass the ones ModelManager already
    // created, so there is a single IDMLDevice on the adapter. Null pointers
    // restrict this to FloodDiffusionDevice::Cpu.
    bool Load(const std::filesystem::path& directory,
              ID3D12Device* device,
              ID3D12CommandQueue* queue,
              IDMLDevice* dmlDevice,
              std::string* error);

    bool IsLoaded() const { return m_pipeline.IsLoaded(); }

    // Starts generating for the current prompt. Loads the text encoder on the
    // first call, which takes a few seconds for a gigabyte of weights.
    bool Generate(std::string* error);

    // Gives the view the scene it can retarget onto, and the KeeMap bone
    // mapping to do it with. Call after the scene graph is built. A missing
    // mapping file is not an error -- the skeleton lines still draw, and the
    // character selector says what to run.
    void AttachScene(SceneGraph* scene, const std::filesystem::path& retargetConfig);

    // Where the scene sits, so retargeting can aim at a world position through
    // it. Pass GLTFAdapter::ScenePlacement() each frame; it is a UI-driven
    // value, so it is read rather than cached.
    void SetScenePlacement(const DirectX::SimpleMath::Matrix& placement) {
        m_scenePlacement = placement;
    }

    // One frame's worth of generation and playback. Also poses the selected
    // character, so it must run before whatever composes world transforms.
    void Update(float elapsedSeconds);

    // Draws the current pose. Must be called between Begin and End on `batch`.
    void Draw(DirectX::PrimitiveBatch<DirectX::VertexPositionColor>& batch) const;

    void ShowImgui();

private:
    // The joint positions being displayed, after the world placement below.
    DirectX::XMFLOAT3 Placed(const DirectX::XMFLOAT3& joint) const;

    // Recreates the sessions on the currently selected providers. Changing a
    // provider means new sessions, so this discards any generation in flight.
    bool Reload(std::string* error);

    // Binds the retargeter to the currently selected character, or unbinds
    // when none is selected.
    void RebindTarget();

    // Poses the selected character from the frame being displayed.
    void PoseCharacter();

    // Sizes the character to the generated skeleton by comparing hip heights,
    // and reports what it measured. Needs at least one decoded frame.
    bool FitToMotion(std::string* summary);

    // Mean hips height above the lowest joint over the decoded frames, which
    // is what the character is matched against. Zero when nothing is decoded.
    float MotionHipsHeight() const;

    // --- retargeting --------------------------------------------------------

    SceneGraph* m_scene = nullptr;
    MotionRetargeter m_retargeter;
    std::vector<RetargetTarget> m_targets;
    std::string m_retargetStatus;

    // -1 is "none": draw the skeleton lines and leave the characters alone.
    int m_selectedTarget = -1;
    int m_boundBones = 0;

    // HumanML3D is metres with the root about 0.9 m up. A Mixamo rig exported
    // from Blender is usually metres too, but the scene it sits in may be
    // scaled, so this is exposed rather than assumed.
    // Set by FitToMotion from the two rigs' hip heights rather than guessed.
    float m_retargetScale = 1.0f;
    DirectX::XMFLOAT3 m_retargetOffset = {0.0f, 0.0f, 0.0f};

    // Refit automatically once the first frames of a new take are decoded, so
    // the character is the right size without anyone pressing anything.
    bool m_autoFit = true;
    bool m_fitPending = false;
    std::string m_fitSummary;

    DirectX::SimpleMath::Matrix m_scenePlacement =
        DirectX::SimpleMath::Matrix::Identity;
    BindPoseMetrics m_characterMetrics;
    bool m_retargetRootMotion = true;
    bool m_hideSkeletonWhenRetargeting = false;

    FloodDiffusionPipeline m_pipeline;
    FloodDiffusionOptions m_options;

    // Kept so a provider change can rebuild the pipeline without the caller
    // having to hand the devices over again.
    std::filesystem::path m_directory;
    ID3D12Device* m_device = nullptr;
    ID3D12CommandQueue* m_queue = nullptr;
    IDMLDevice* m_dmlDevice = nullptr;

    // ImGui::InputText needs a fixed buffer.
    char m_prompt[256] = "a person walks forward";
    std::string m_status;
    bool m_hasStatusError = false;

    // --- generation pacing --------------------------------------------------

    // Denoising steps per rendered frame. One keeps the added cost inside a
    // frame while still outrunning playback; more finishes sooner at the cost
    // of a longer frame.
    int m_stepsPerFrame = 1;

    // --- playback -----------------------------------------------------------

    float m_playbackTime = 0.0f;
    float m_playbackSpeed = 1.0f;
    bool m_playing = true;
    bool m_loop = true;
    int m_currentFrame = 0;

    // --- placement and appearance -------------------------------------------

    // HumanML3D is metres with the root near the hips and the origin where the
    // motion starts, so a skeleton lands in the same place as the glTF scene.
    // These move it aside.
    DirectX::XMFLOAT3 m_offset = {0.0f, 0.0f, 0.0f};
    float m_scale = 1.0f;
    bool m_drawJoints = true;
    bool m_drawTrail = false;
    float m_jointSize = 0.02f;
};

}  // namespace NeuralModelIntegrateTestbed
