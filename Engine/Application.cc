//--------------------------------------------------------------------------------------
// Application.cc
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "Application.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include "Descriptors/DescriptorContext.hpp"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <directx/d3dx12.h>


#include "FindMedia.h"

extern void ExitSample() noexcept;

using namespace DirectX;

// The model ModelManager registers at startup and EnqueueModel submits each
// frame. One constant so the two cannot disagree -- a typo would simply stop
// dispatching, silently.
static constexpr const char* kNeuralModelName = "FloodDiffusion";
using namespace DirectX::SimpleMath;

using Microsoft::WRL::ComPtr;

Sample::Sample() noexcept(false)
{
    m_deviceResources = std::make_shared<DX::DeviceResources>();
    m_deviceResources->RegisterDeviceNotify(this);
}

Sample::~Sample()
{
    if (m_deviceResources)
    {
        m_deviceResources->WaitForGpu();
    }
}

namespace {

// Where to look for a scene, in order of preference. A directory rather than a
// file, because an FBX-to-glTF export names the .gltf after the source file and
// that name is whatever the exporter felt like -- so the first .gltf or .glb in
// the directory is taken.
//
// Bistro first, then the small authored scenes, so dropping an export into
// resources/bistro/ is all it takes to switch. See the README for how to get
// and stage it; it is not in the repository and cannot be, at 2.8M triangles
// and a CC-BY attribution requirement.
constexpr const char *kSceneDirectories[] = {
    // A sibling checkout, the same arrangement fastgltf uses. This is the
    // niagara project's conversion of NVIDIA's rtxdi-assets Bistro: glTF with
    // MSFT_texture_dds and specular-glossiness materials, MIT licensed.
    "../niagara_bistro",
    "resources/bistro/interior-wine",
    "resources/bistro/interior",
    "resources/bistro/exterior",
};

// The scene used when none of the above is staged.
constexpr const char *kFallbackScene =
    "resources/microphone_and_animated_char/scene_withlight_animated_char.gltf";

// Longest edge any texture may keep. Every image is decoded to RGBA8 with a
// full mip chain and no BC compression, so an uncapped large scene runs the
// GPU out of memory part way through loading. 2048 costs 22 MB per texture and
// leaves the authored scenes untouched.
constexpr std::uint32_t kDefaultMaxTextureDimension = 2048;

std::string FirstGltfIn(const std::filesystem::path &directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
        return {};
    }
    // Sorted, so a directory holding more than one export picks the same file
    // every run rather than depending on the filesystem's order.
    std::vector<std::string> candidates;
    for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) {
            continue;
        }
        const std::filesystem::path extension = entry.path().extension();
        if (extension == ".gltf" || extension == ".glb") {
            candidates.push_back(entry.path().generic_string());
        }
    }
    if (candidates.empty()) {
        return {};
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front();
}

// PTFLIO_SCENE overrides everything, so a scene staged anywhere can be loaded
// without a rebuild.
std::string ResolveScenePath() {
    if (const char *fromEnvironment = std::getenv("PTFLIO_SCENE")) {
        if (*fromEnvironment != '\0') {
            return fromEnvironment;
        }
    }
    for (const char *directory : kSceneDirectories) {
        const std::string found = FirstGltfIn(directory);
        if (!found.empty()) {
            return found;
        }
    }
    return kFallbackScene;
}

// Which environment map to use as the sky.
//
// PTFLIO_SKYBOX names one explicitly; otherwise the first .hdr beside the
// scene is taken. Deliberately only .hdr and not "any image beside the scene":
// the Bistro ships 390 .dds textures in the same tree and picking one of those
// at random would be worse than no sky.
std::string ResolveSkyboxPath(const std::string& scenePath) {
    if (const char *fromEnvironment = std::getenv("PTFLIO_SKYBOX")) {
        if (*fromEnvironment != '\0') {
            return fromEnvironment;
        }
    }
    std::error_code error;
    const std::filesystem::path directory =
        std::filesystem::path(scenePath).parent_path();
    if (directory.empty() || !std::filesystem::is_directory(directory, error)) {
        return {};
    }
    std::vector<std::string> candidates;
    for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) {
            continue;
        }
        if (entry.path().extension() == ".hdr") {
            candidates.push_back(entry.path().generic_string());
        }
    }
    if (candidates.empty()) {
        return {};
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front();
}

// PTFLIO_MAX_TEXTURE caps the longest texture edge; 0 removes the cap.
std::uint32_t ResolveMaxTextureDimension() {
    if (const char *fromEnvironment = std::getenv("PTFLIO_MAX_TEXTURE")) {
        try {
            const int parsed = std::stoi(fromEnvironment);
            if (parsed >= 0) {
                return static_cast<std::uint32_t>(parsed);
            }
        } catch (const std::exception &) {
            // Left at the default rather than failing to start over a typo.
        }
    }
    return kDefaultMaxTextureDimension;
}

}  // namespace

// Initialize the Direct3D resources required to run.
void Sample::Initialize(HWND window, int width, int height)
{
    m_gamePad = std::make_unique<GamePad>();

    m_keyboard = std::make_unique<Keyboard>();

    m_mouse = std::make_unique<Mouse>();
    m_mouse->SetWindow(window);
    m_camera = std::make_shared<Camera>();

    m_deviceResources->SetWindow(window, width, height);

    m_deviceResources->CreateDeviceResources();  	
    CreateDeviceDependentResources();

    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();

    // Create DirectXTK for Audio objects
    AUDIO_ENGINE_FLAGS eflags = AudioEngine_Default;
#ifdef _DEBUG
    eflags = eflags | AudioEngine_Debug;
#endif

    m_audEngine = std::make_unique<AudioEngine>(eflags);

    m_audioEvent = 0;
    m_audioTimerAcc = 10.f;
    m_retryDefault = false;

    wchar_t strFilePath[MAX_PATH];
    DX::FindMediaFile(strFilePath, MAX_PATH, L"adpcmdroid.xwb");
    m_waveBank = std::make_unique<WaveBank>(m_audEngine.get(), strFilePath);

    DX::FindMediaFile(strFilePath, MAX_PATH, L"MusicMono_adpcm.wav");
    m_soundEffect = std::make_unique<SoundEffect>(m_audEngine.get(), strFilePath);
    m_effect1 = m_soundEffect->CreateInstance();
    m_effect2 = m_waveBank->CreateInstance(10);

    m_effect1->Play(true);
    m_effect2->Play();

    // Which scene, and how much texture memory it is allowed. Both are
    // resolved rather than hardcoded so a large staged scene can be loaded
    // without editing this file -- see ResolveScenePath above.
    //
    // Reported on stderr, not stdout: this is a WINDOWS-subsystem binary, so
    // stdout goes nowhere when the app is launched normally.
    const std::string scenePath = ResolveScenePath();
    const std::uint32_t maxTextureDimension = ResolveMaxTextureDimension();
    std::fprintf(stderr, "scene: %s\n", scenePath.c_str());

    m_gltfAdapter.SetMaxTextureDimension(maxTextureDimension);
    m_gltfAdapter.Initialize(scenePath);
    m_gltfAdapter.SetCommonStates(m_states);
    m_gltfAdapter.AssignCamera(m_camera);
    m_gltfAdapter.PrepareBuffer(m_deviceResources);
    m_gltfAdapter.PrepareImage(m_deviceResources);
    m_gltfAdapter.BuildSceneGraph();
    m_gltfAdapter.PreparePSO(m_deviceResources);
    // After PreparePSO, because it needs the scene graph built and the
    // transforms composable.
    FrameSceneWithCamera();

    // The sky. A scene without one renders against the clear colour, so a
    // failure here is reported and then shrugged off rather than thrown.
    const std::string skyboxPath = ResolveSkyboxPath(scenePath);
    if (!skyboxPath.empty()) {
        std::string skyboxError;
        if (!m_deferred.LoadEnvironment(*m_deviceResources, skyboxPath, &skyboxError)) {
            std::fprintf(stderr, "skybox: %s: %s\n", skyboxPath.c_str(),
                         skyboxError.c_str());
            std::fflush(stderr);
        }
    }

    m_nnModelManager.Initialize(m_deviceResources);
    {
        // Models are registered by name and dispatched by that name, so adding
        // a second one is another AddModel / AddOnnxModel call rather than a
        // change to ModelManager.
        std::string modelError;
        if (!m_nnModelManager.AddModel(
                std::make_unique<NeuralModelIntegrateTestbed::FloodDiffusionNNModel>(
                    kNeuralModelName),
                &modelError)) {
            std::printf("model '%s' failed to load: %s\n", kNeuralModelName,
                        modelError.c_str());
        }

        // An ONNX file is registered the same way when one is present; absence
        // is not an error, since the repo ships no .onnx by default.
        //
        // ONNX Runtime's DirectML EP is the default backend: it has full
        // operator coverage, runs on this project's own device and queue, and
        // Run() records without stalling. The DirectMLGraph backend is still
        // there for a graph you want recorded into the render command list, or
        // for operators you want to express yourself.
        const std::filesystem::path onnxPath = "resources/Models/test.onnx";
        if (std::filesystem::exists(onnxPath)) {
            std::string onnxError;
            if (m_nnModelManager.AddOnnxModel("test.onnx", onnxPath,
                                              NeuralModelIntegrateTestbed::ModelBackend::OnnxRuntime,
                                              &onnxError)) {
                std::printf("loaded ONNX model 'test.onnx' (ONNX Runtime)\n");
            } else {
                std::printf("ONNX model '%s' failed to load: %s\n",
                            onnxPath.string().c_str(), onnxError.c_str());
            }
        }
    }

    // The FloodDiffusion text-to-motion pipeline. Its own object rather than a
    // ModelManager entry: it is four ONNX sessions driven by a host-side
    // sampling loop, not one model dispatched by name. A missing
    // resources/FloodDiffusion is not an error -- the weights are 1.2 GB and
    // gitignored.
    //
    // It is handed this renderer's device plus the IDMLDevice the ModelManager
    // already made, so the DirectML provider runs on the same adapter and
    // there is one IDMLDevice, not two. Hence this must follow
    // m_nnModelManager.Initialize above.
    //
    // NOT the renderer's command queue: the pipeline makes its own compute
    // queue, so a generation does not land in the middle of a frame.
    {
        // The retarget target lives in the glTF scene, so this has to follow
        // BuildSceneGraph above.
        m_motionStream.AttachScene(&m_gltfAdapter.Scene(),
                                   "resources/retarget/mixamo.txt");

        std::string motionError;
        if (m_motionStream.Load("resources/FloodDiffusion",
                                m_deviceResources->GetD3DDevice(),
                                m_nnModelManager.DmlDevice(),
                                &motionError)) {
            std::printf("FloodDiffusion motion pipeline ready\n");
        } else if (!motionError.empty()) {
            std::printf("FloodDiffusion motion pipeline unavailable: %s\n",
                        motionError.c_str());
        }
    }

    // Setup Dear ImGui context
    ImGui_ImplWin32_EnableDpiAwareness();
    float main_scale = ImGui_ImplWin32_GetDpiScaleForMonitor(::MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY));

    {
        static const int APP_SRV_HEAP_SIZE = 64;
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = APP_SRV_HEAP_SIZE;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        m_deviceResources->GetD3DDevice()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_imguiSrvDescHeap));
        g_pd3dSrvDescHeapAlloc.Create(m_deviceResources->GetD3DDevice(), m_imguiSrvDescHeap);
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (in docking branch: using io.ConfigDpiScaleFonts=true automatically overrides this for every window depending on the current monitor)

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(window);

    ImGui_ImplDX12_InitInfo init_info = {};
    init_info.Device = m_deviceResources->GetD3DDevice();
    init_info.CommandQueue = m_deviceResources->GetCommandQueue();
    init_info.NumFramesInFlight = 2;
    init_info.RTVFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
    init_info.DSVFormat = DXGI_FORMAT_UNKNOWN;
    // Allocating SRV descriptors (for textures) is up to the application, so we provide callbacks.
    // (current version of the backend will only allocate one descriptor, future versions will need to allocate more)
    init_info.SrvDescriptorHeap = m_imguiSrvDescHeap;
    init_info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle) { return g_pd3dSrvDescHeapAlloc.Alloc(out_cpu_handle, out_gpu_handle); };
    init_info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)            { return g_pd3dSrvDescHeapAlloc.Free(cpu_handle, gpu_handle); };
    ImGui_ImplDX12_Init(&init_info);

}

#pragma region Frame Update
// Executes basic render loop.
void Sample::Tick()
{
    m_timer.Tick([&]()
    {
        Update(m_timer);
    });

    // Only update audio engine once per frame
    if (!m_audEngine->IsCriticalError() && m_audEngine->Update())
    {
        // Setup a retry in 1 second
        m_audioTimerAcc = 1.f;
        m_retryDefault = true;
    }

    Render();
}

// Updates the world.
void Sample::Update(DX::StepTimer const& timer)
{
    PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

    static Vector3 eye(0.0f, 0.7f, 1.5f);
    static Vector3 at(0.0f, -0.1f, 0.0f);

    m_world = Matrix::CreateRotationY(float(timer.GetTotalSeconds() * XM_PIDIV4));

    m_lineEffect->SetWorld(Matrix::Identity);

    // The view matrix is NOT built here. It used to be, at the top of Update,
    // which is before the mouse handling further down moves the camera -- so it
    // described where the camera was on the PREVIOUS frame while
    // BuildFrameGraph read m_camera->mEye fresh for the lighting and the sky.
    // A frame then mixed a stale view-projection with a fresh eye position,
    // which disagree by exactly the frame's camera motion. UpdateViewMatrix is
    // called from Render instead, so every consumer reads one instant.

    // Before Render, which composes the posed nodes into world transforms and
    // the joint matrices the vertex shader skins with.
    m_gltfAdapter.UpdateAnimation((float)timer.GetElapsedSeconds());

    // Advances the sampler by a bounded number of denoising steps and walks
    // playback over whatever has been decoded, so the skeleton moves while the
    // rest of the take is still being generated.
    // The scene placement is a UI-driven value, so the retarget is told it
    // every frame rather than at attach time; it has to aim through the same
    // transform the adapter draws with.
    m_motionStream.SetScenePlacement(m_gltfAdapter.ScenePlacement());
    m_motionStream.Update((float)timer.GetElapsedSeconds());

    m_audioTimerAcc -= (float)timer.GetElapsedSeconds();
    if (m_audioTimerAcc < 0)
    {
        if (m_retryDefault)
        {
            m_retryDefault = false;
            if (m_audEngine->Reset())
            {
                // Restart looping audio
                m_effect1->Play(true);
            }
        }
        else
        {
            m_audioTimerAcc = 4.f;

            m_waveBank->Play(m_audioEvent++);

            if (m_audioEvent >= 11)
                m_audioEvent = 0;
        }
    }

    auto pad = m_gamePad->GetState(0);
    if (pad.IsConnected())
    {
        m_gamePadButtons.Update(pad);

        if (pad.IsViewPressed())
        {
            ExitSample();
        }
    }
    else
    {
        m_gamePadButtons.Reset();
    }

    // Everything below shares the mouse and keyboard with Dear ImGui, so it
    // has to ask first.
    //
    // These two flags are computed inside ImGui::NewFrame, which runs in
    // Render -- so what is read here is the PREVIOUS frame's answer. That is
    // the one frame of lag every ImGui integration lives with, and it is
    // harmless: the pointer moves a few pixels in 16 ms, and an item that was
    // active last frame is still active now.
    const ImGuiIO& io = ImGui::GetIO();

    auto kb = m_keyboard->GetState();
    m_keyboardButtons.Update(kb);

    // Not while a text field has the keyboard. The FloodDiffusion prompt is an
    // InputText, and Escape there means "cancel this edit" to ImGui -- it must
    // not also mean "quit the application".
    if (kb.Escape && !io.WantCaptureKeyboard)
    {
        ExitSample();
    }

    auto mouse = m_mouse->GetState();
    m_mouseTracker.Update( mouse );

    // Latched on the press. See m_mouseDragging in Application.hpp for why
    // this is not simply `if (!io.WantCaptureMouse)` on each frame.
    const bool anyButtonDown =
        mouse.leftButton || mouse.rightButton || mouse.middleButton;
    if (anyButtonDown && !m_mouseDragging)
    {
        m_mouseDragging = true;
        m_uiOwnsMouseDrag = io.WantCaptureMouse;
    }
    else if (!anyButtonDown)
    {
        m_mouseDragging = false;
        m_uiOwnsMouseDrag = false;
    }
    const bool cameraOwnsDrag = m_mouseDragging && !m_uiOwnsMouseDrag;

    if (cameraOwnsDrag && mouse.positionMode == Mouse::MODE_RELATIVE)
    {
        // Scaled by the speed multipliers, which FrameSceneWithCamera sizes to
        // the scene: MoveForward takes WORLD UNITS, so an unscaled drag that
        // crosses a two-metre scene in a moment barely stirs a street.
        const float elapsed = static_cast<float>(timer.GetElapsedSeconds());
        const float move = elapsed * m_cameraMoveSpeed;
        const float rotate = elapsed * m_cameraRotateSpeed;
        if (m_mouseTracker.leftButton == Mouse::ButtonStateTracker::HELD) {
            m_camera->RotateAroundYAxis(-mouse.x * rotate);
            m_camera->RotatePitch(-mouse.y * rotate);
        }
        if (m_mouseTracker.rightButton == Mouse::ButtonStateTracker::HELD) {
            m_camera->MoveForward(-mouse.y * move);
        }
        if (m_mouseTracker.middleButton == Mouse::ButtonStateTracker::HELD) {
            m_camera->MoveUpward(-mouse.y * move);
            m_camera->MoveRightward(-mouse.x * move);
        }
    }
    // Capturing the pointer is the camera's doing, so it only happens for a
    // drag the camera owns. Grabbing it for a drag that belongs to a widget is
    // worse than the spurious camera motion it caused: MODE_RELATIVE hides the
    // cursor and stops delivering positions, so the widget being dragged would
    // stop tracking too.
    if (cameraOwnsDrag &&
        (m_mouseTracker.leftButton == Mouse::ButtonStateTracker::PRESSED ||
         m_mouseTracker.rightButton == Mouse::ButtonStateTracker::PRESSED ||
         m_mouseTracker.middleButton == Mouse::ButtonStateTracker::PRESSED)) {
        m_mouse->SetMode(Mouse::MODE_RELATIVE);
    }
    else if (m_mouseTracker.leftButton == Mouse::ButtonStateTracker::ButtonState::RELEASED ||
             m_mouseTracker.rightButton == Mouse::ButtonStateTracker::ButtonState::RELEASED ||
             m_mouseTracker.middleButton == Mouse::ButtonStateTracker::ButtonState::RELEASED)
    {
        // Unconditional on purpose: a release has to restore the cursor
        // whoever owned the drag, and setting ABSOLUTE when it was never left
        // costs nothing.
        m_mouse->SetMode(Mouse::MODE_ABSOLUTE);
    }


    PIXEndEvent();
}


#pragma endregion

#pragma region Frame Render
// Draws the scene.












void Sample::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

    // Resets this frame's command allocator and list and transitions the back
    // buffer out of PRESENT. The list is left open; Present() closes it, so
    // everything below records into one list -- which is what lets the render
    // graph put a barrier between the geometry pass and the lighting pass that
    // reads what it wrote.
    m_deviceResources->Prepare();
    auto commandList = m_deviceResources->GetCommandList();


    // Before the graph: the ImGui pass only submits draw data, so the draw data
    // has to already exist. Panels that inspect this frame's graph therefore
    // show the PREVIOUS frame's plan, which is why the plan description is kept
    // from frame to frame rather than cleared.
    BuildUserInterface();

    // After Update has applied this frame's input, so the matrices and the eye
    // position every pass reads describe the same camera.
    UpdateViewMatrix();

    // Composes world transforms and places the lights. The lighting pass needs
    // both and runs inside the graph, so this cannot wait until recording.
    m_gltfAdapter.UpdateSceneTransforms();

    BuildFrameGraph();

    std::string graphError;
    if (m_frameGraph.Compile(&graphError))
    {
        m_deferred.SetPlanDescription(m_frameGraph.DescribePlan());
        if (!m_frameGraph.Execute(commandList, &graphError))
        {
            // Recording stopped part way, so the frame is incomplete -- but the
            // command list is still valid and Present() will submit what there
            // is. Reporting it in the UI beats throwing: a frame that cannot be
            // recorded is usually a pass that was just edited, and killing the
            // app loses the message.
            m_deferred.SetPlanDescription("execute failed: " + graphError);
        }
    }
    else
    {
        m_deferred.SetPlanDescription("compile failed: " + graphError);
    }

    // Outside the graph, and off the graphics queue entirely: EnqueueModel
    // submits to ModelManager's own compute queue, so this dispatch is
    // scheduled beside the frame rather than between two of them.
    //
    // That is only correct because nothing in the frame reads the result. The
    // moment a pass samples a model's output, this becomes a graph pass and
    // the frame needs ModelManager::OrderAfterModelWork ahead of it --
    // sharing the graphics queue used to make that ordering free, and it is
    // not free any more.
    //
    // Dispatched by name. An unregistered name submits nothing and returns
    // false rather than asserting, so a model that failed to load does not
    // take the frame down with it.
    m_nnModelManager.EnqueueModel(kNeuralModelName);

    // Show the new frame. Present closes and submits the command list.
    PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");
    m_deviceResources->Present();
    m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());
    PIXEndEvent(m_deviceResources->GetCommandQueue());

    // Advance the descriptor frame clock, then reclaim descriptors retired long
    // enough ago that the GPU cannot still be reading them. Present() has
    // already waited for the frame BACK_BUFFER_COUNT ago to complete, so
    // anything retired before that point is safe; subtracting the back buffer
    // count keeps a conservative margin rather than plumbing fence values in.
    auto& descriptorContext = NeuralModelIntegrateTestbed::Descriptors::Context();
    descriptorContext.AdvanceFrame();
    const uint64_t frameCount = descriptorContext.GetFrameCount();
    constexpr uint64_t kInFlightMargin = 3;  // back buffer count + 1
    if (frameCount > kInFlightMargin) {
        m_gltfAdapter.ReleaseStaleDescriptors(frameCount - kInFlightMargin);
    }
}

// The frame, as a pass list:
//
//   GBuffer           geometry -> four material targets + depth
//   DeferredLighting  those targets -> the back buffer, one light loop per pixel
//   ForwardOverlay    grid, skeleton, sprites, teapot, depth-tested against the
//                     deferred geometry because they share its depth buffer
//   ImGui             last, so the UI is never occluded
//
// Order here is execution order; the graph derives the barriers from the
// accesses each pass declares, not from this list.
void Sample::BuildFrameGraph()
{
    m_frameGraph.BeginFrame();

    // What shows wherever no geometry was drawn: the lighting pass clears the
    // back buffer and then discards on far-plane depth.
    const float clearColor[4] = {Colors::CornflowerBlue.f[0], Colors::CornflowerBlue.f[1],
                                 Colors::CornflowerBlue.f[2], Colors::CornflowerBlue.f[3]};
    const NeuralModelIntegrateTestbed::DeferredRenderer::FrameTargets targets =
        m_deferred.ImportFrameTargets(m_frameGraph, *m_deviceResources, clearColor);

    NeuralModelIntegrateTestbed::DeferredRenderer::FrameInputs inputs;
    inputs.view = m_view;
    inputs.projection = m_projection;
    XMStoreFloat3(&inputs.eyePosition, m_camera->mEye);

    // Up to kMaxDeferredLights of them, against the forward path's 4. Gathered
    // here because the lights belong to the scene, not to the renderer; stored
    // in a member because the pass holds a pointer to it until Execute runs.
    inputs.lightCount = m_gltfAdapter.Scene().GatherShaderLights(
        m_frameLights, NeuralModelIntegrateTestbed::kMaxDeferredLights);
    inputs.lights = m_frameLights;

    // The cascade fit needs to know how far the scene extends, so a caster
    // behind a slice is still inside the light's depth range. Cached rather
    // than recomputed: the bounds only change when the scene is reposed, and
    // SceneBounds walks every primitive.
    inputs.sceneMin = m_sceneBoundsMin;
    inputs.sceneMax = m_sceneBoundsMax;
    inputs.hasSceneBounds = m_hasSceneBounds;
    // Nothing moving means no compositing: the cached shadow atlas is the
    // frame's shadow map as it stands.
    inputs.hasMovingGeometry = m_gltfAdapter.HasMovingGeometry();

    m_deferred.AddPasses(
        m_frameGraph, targets, inputs,
        [this](ID3D12GraphicsCommandList* commandList,
               D3D12_CPU_DESCRIPTOR_HANDLE lightmapAtlas) {
            m_gltfAdapter.SetLightmapAtlasSrv(lightmapAtlas);
            m_gltfAdapter.RecordGeometry(commandList, m_view, m_projection);
        },
        [this](ID3D12GraphicsCommandList* commandList,
               const DirectX::SimpleMath::Matrix& lightViewProjection,
               NeuralModelIntegrateTestbed::DeferredRenderer::ShadowGeometry which) {
            m_gltfAdapter.RecordShadowGeometry(
                commandList, lightViewProjection,
                which == NeuralModelIntegrateTestbed::DeferredRenderer::
                             ShadowGeometry::Moving);
        },
        [this](ID3D12GraphicsCommandList* commandList,
               const DirectX::SimpleMath::Matrix& sunViewProjection,
               float sunTexelWorldSize, D3D12_CPU_DESCRIPTOR_HANDLE sunDepth) {
            m_gltfAdapter.RecordLightmapBake(commandList, sunViewProjection,
                                             sunTexelWorldSize, sunDepth);
        });

    {
        NeuralModelIntegrateTestbed::Render::RenderPassDesc overlay;
        overlay.name = "WorldOverlay";
        overlay.writes = {
            // The HDR target, not the back buffer: these are world-space draws
            // and belong on the same side of the display transform as the
            // scene they sit in.
            {m_deferred.SceneColor(),
             NeuralModelIntegrateTestbed::Render::Access::RenderTarget},
            // Depth-tested, so the graph has to transition depth back out of
            // the readable state the lighting pass left it in.
            {targets.depth, NeuralModelIntegrateTestbed::Render::Access::DepthWrite},
        };
        overlay.execute = [this](ID3D12GraphicsCommandList* commandList,
                                 const NeuralModelIntegrateTestbed::Render::PassResources&) {
            RecordWorldOverlays(commandList);
        };
        m_frameGraph.AddPass(std::move(overlay));
    }


    // Everything above wrote linear radiance; this is where it becomes an
    // image. Anything added after it is in display space.
    m_deferred.AddTonemapPass(m_frameGraph, targets.color);


/*
    {
        NeuralModelIntegrateTestbed::Render::RenderPassDesc screen;
        screen.name = "ScreenOverlay";
        screen.writes = {
            {targets.color, NeuralModelIntegrateTestbed::Render::Access::RenderTarget}};
        screen.execute = [this](ID3D12GraphicsCommandList* commandList,
                                const NeuralModelIntegrateTestbed::Render::PassResources&) {
            RecordScreenOverlays(commandList);
        };
        m_frameGraph.AddPass(std::move(screen));
    }
*/
    {
        NeuralModelIntegrateTestbed::Render::RenderPassDesc ui;
        ui.name = "ImGui";
        ui.writes = {
            {targets.color, NeuralModelIntegrateTestbed::Render::Access::RenderTarget}};
        // ImGui's D3D12 backend sets its own render target and descriptor heap,
        // so the graph only needs to have the back buffer in the right state --
        // which is why the write is still declared.
        ui.bindRenderTargets = false;
        ui.execute = [this](ID3D12GraphicsCommandList* commandList,
                            const NeuralModelIntegrateTestbed::Render::PassResources&) {
            commandList->OMSetRenderTargets(1, &m_deviceResources->GetRenderTargetView(),
                                            FALSE, nullptr);
            commandList->SetDescriptorHeaps(1, &m_imguiSrvDescHeap);
            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);
        };
        m_frameGraph.AddPass(std::move(ui));
    }
}

void Sample::RecordWorldOverlays(ID3D12GraphicsCommandList* commandList)
{
    // Draw procedurally generated dynamic grid
    const XMVECTORF32 xaxis = { 20.f, 0.f, 0.f };
    const XMVECTORF32 yaxis = { 0.f, 0.f, 20.f };
    // DrawGrid(xaxis, yaxis, g_XMZero, 20, 20, Colors::Gray);
    DrawMotionSkeleton();

    // The sampler heap, for the teapot's texture.
    ID3D12DescriptorHeap* heaps[] = { m_resourceDescriptors->Heap(), m_states->Heap() };
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);

    // Draw 3D object
    // PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Draw teapot");
    // XMMATRIX local = m_world * Matrix::CreateTranslation(-2.f, -2.f, -4.f);
    // m_shapeEffect->SetWorld(local);
    // m_shapeEffect->Apply(commandList);
    // m_shape->Draw(commandList);
    // PIXEndEvent(commandList);
}

void Sample::RecordScreenOverlays(ID3D12GraphicsCommandList* commandList)
{
    ID3D12DescriptorHeap* heaps[] = { m_resourceDescriptors->Heap(), m_states->Heap() };
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);

    // Draw sprite
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Draw sprite");
    m_sprites->Begin(commandList);
    m_sprites->Draw(m_resourceDescriptors->GetGpuHandle(Descriptors::WindowsLogo), GetTextureSize(m_texture2.Get()),
        XMFLOAT2(10, 75));

    m_font->DrawString(m_sprites.get(), L"DirectXTK Simple Sample", XMFLOAT2(100, 10), Colors::Yellow);
    m_sprites->End();
    PIXEndEvent(commandList);
}

void Sample::FrameSceneWithCamera()
{
    DirectX::XMFLOAT3 minimum = {};
    DirectX::XMFLOAT3 maximum = {};
    if (!m_gltfAdapter.SceneBounds(minimum, maximum))
    {
        return;
    }

    const Vector3 low(minimum.x, minimum.y, minimum.z);
    const Vector3 high(maximum.x, maximum.y, maximum.z);
    const Vector3 centre = (low + high) * 0.5f;
    const float radius = std::max(0.001f, (high - low).Length() * 0.5f);

    m_sceneCentre = centre;
    m_sceneRadius = radius;
    m_sceneBoundsMin = low;
    m_sceneBoundsMax = high;
    m_hasSceneBounds = true;
    // The shadow distance follows the scene, so the cascades are worth having
    // in a two-metre fixture and a hundred-and-sixty-metre street alike.
    m_deferred.SetSceneScale(radius);

    // Far enough back that the bounding sphere fits the vertical field of view,
    // with a little margin. 70 degrees is what CreateWindowSizeDependentResources
    // uses; a narrower window makes it wider still, so this is the tight case.
    const float fovY = m_fovDegrees * XM_PI / 180.0f;
    const float distance = radius / std::sin(fovY * 0.5f) * 1.1f;

    // A raised three-quarter view. Any fixed direction is a guess, but one that
    // is not axis-aligned at least shows depth rather than a flat elevation.
    Vector3 direction(-0.55f, 0.35f, 1.0f);
    direction.Normalize();

    m_camera->Set(centre + direction * distance, centre, Vector3::UnitY);

    // Depth precision is governed by far/near, not by either alone, so the near
    // plane scales with the scene. 1/1000 of the radius keeps the ratio near
    // 10^4, which D32_FLOAT handles comfortably; leaving it at 0.01 against a
    // far plane of 400 would be 40000 and z-fight in the distance.
    m_near = std::max(0.01f, radius * 0.001f);
    m_far = distance + radius * 2.0f;

    // Roughly "cross the scene in a second of dragging". Rotation is in radians
    // and has nothing to do with scene size, so it is left alone.
    m_cameraMoveSpeed = radius * 0.05f;

    UpdateProjection();

    std::fprintf(stderr,
                 "[camera] scene centre (%.2f, %.2f, %.2f) radius %.2f -> "
                 "eye distance %.2f, near %.4f, far %.1f\n",
                 centre.x, centre.y, centre.z, radius, distance, m_near, m_far);
    std::fflush(stderr);
}

void Sample::ShowCameraUI()
{
    if (!ImGui::CollapsingHeader("Camera"))
    {
        return;
    }

    // Drag steps are scaled to the scene, so one pixel of drag is a sensible
    // nudge whether the scene is two units across or a hundred and sixty.
    const float step = std::max(0.001f, m_sceneRadius * 0.002f);

    Vector3 eye = m_camera->mEye;
    Vector3 at = m_camera->mAt;
    Vector3 up = m_camera->mUp;
    bool changed = false;

    changed |= ImGui::DragFloat3("Position", &eye.x, step);
    changed |= ImGui::DragFloat3("Target", &at.x, step);
    changed |= ImGui::DragFloat3("Up", &up.x, 0.01f);

    if (changed)
    {
        // CreateLookAt produces NaNs when the eye and the target coincide, and
        // one NaN in the view matrix takes the whole frame with it -- including
        // the UI that would let you undo it. Dragging the target onto the eye
        // is an easy accident, so it is caught here.
        Vector3 forward = at - eye;
        if (forward.Length() < 1e-4f)
        {
            at = eye + Vector3(0.0f, 0.0f, -std::max(step, 1e-3f));
        }
        if (up.Length() < 1e-4f)
        {
            up = Vector3::UnitY;
        }
        m_camera->Set(eye, at, up);
    }

    ImGui::Separator();

    // Derived, so there is something to read back after flying around with the
    // mouse rather than only something to type into.
    Vector3 forward = Vector3(m_camera->mAt) - Vector3(m_camera->mEye);
    const float distance = forward.Length();
    if (distance > 1e-6f)
    {
        forward /= distance;
    }
    ImGui::Text("Distance to target: %.3f", distance);
    ImGui::Text("Forward: (%.3f, %.3f, %.3f)", forward.x, forward.y, forward.z);
    // Pitch is unambiguous. Yaw needs a zero, and -Z is the one this
    // right-handed projection looks down.
    ImGui::Text("Pitch %.1f deg, yaw %.1f deg (0 == -Z)",
                XMConvertToDegrees(std::asin(std::max(-1.0f, std::min(1.0f, forward.y)))),
                XMConvertToDegrees(std::atan2(forward.x, -forward.z)));

    ImGui::Separator();

    // The projection. Each of these rebuilds it, which also re-pushes it into
    // the DirectXTK effects the forward overlay pass draws with -- otherwise
    // the grid and the teapot would keep the old one and stop lining up.
    bool projectionChanged = false;
    projectionChanged |= ImGui::SliderFloat("Field of view", &m_fovDegrees, 20.0f, 120.0f,
                                            "%.1f deg");
    // Depth precision is governed by far/near rather than by either alone, so
    // the ratio is shown: pushing the near plane in is far more expensive than
    // pushing the far plane out.
    projectionChanged |= ImGui::DragFloat("Near plane", &m_near, step * 0.1f, 0.001f,
                                         m_far * 0.5f, "%.4f");
    projectionChanged |= ImGui::DragFloat("Far plane", &m_far, step * 10.0f,
                                         m_near * 2.0f, 100000.0f, "%.1f");
    ImGui::Text("far/near = %.0f", m_near > 0.0f ? m_far / m_near : 0.0f);
    if (projectionChanged)
    {
        UpdateProjection();
    }

    ImGui::Separator();

    ImGui::DragFloat("Move speed", &m_cameraMoveSpeed, step, 0.0001f, 10000.0f, "%.4f");
    ImGui::DragFloat("Rotate speed", &m_cameraRotateSpeed, 0.01f, 0.0001f, 100.0f, "%.4f");
    ImGui::TextUnformatted("drag: left orbit, right dolly, middle pan");

    ImGui::Separator();

    if (ImGui::Button("Frame scene"))
    {
        FrameSceneWithCamera();
    }
    ImGui::SameLine();
    if (ImGui::Button("Look at centre") && m_sceneRadius > 0.0f)
    {
        // Keeps the eye where it is and re-aims it, which is what you want after
        // dollying past the scene and losing it off the edge of the screen.
        m_camera->Set(m_camera->mEye, m_sceneCentre, Vector3::UnitY);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
        m_camera->Reset();
    }
    if (m_sceneRadius > 0.0f)
    {
        ImGui::Text("Scene: centre (%.2f, %.2f, %.2f), radius %.2f", m_sceneCentre.x,
                    m_sceneCentre.y, m_sceneCentre.z, m_sceneRadius);
    }
}



void Sample::ShowFrameTimingUI()
{
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("%.2f ms this frame, %.2f ms average (%.1f FPS)",
                io.DeltaTime * 1000.0f, 1000.0f / io.Framerate, io.Framerate);

    // The floor is fixed at two 60 Hz frames rather than fitted to the data, so
    // the plot does not rescale itself the moment the frame gets slower --
    // which is exactly when its shape is being read. It grows past that only
    // when something in the window genuinely exceeded it.
    float worst = 0.0f;
    for (const float sample : m_frameTimes)
    {
        worst = std::max(worst, sample);
    }
    const float ceiling = worst > 33.3f ? worst : 33.3f;
    char overlay[48];
    std::snprintf(overlay, sizeof(overlay), "worst %.1f ms", worst);
    // The cursor is where the NEXT sample goes, so plotting from there reads
    // oldest-to-newest and the newest sample is at the right edge.
    ImGui::PlotLines("##frametimes", m_frameTimes, kFrameTimeHistory,
                     m_frameTimeCursor, overlay, 0.0f, ceiling, ImVec2(0.0f, 60.0f));
    ImGui::Separator();
}

void Sample::BuildUserInterface()
{
    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // Sampled here rather than inside the Performance tab: ImGui skips the body
    // of a tab that is not selected, so recording it there would leave the
    // history full of whichever frames happened to be on that tab -- a plot
    // that only has data for the moments you were already looking at it.
    m_frameTimes[m_frameTimeCursor] = ImGui::GetIO().DeltaTime * 1000.0f;
    m_frameTimeCursor = (m_frameTimeCursor + 1) % kFrameTimeHistory;

    // One window, four tabs. Everything was in a single scrolling column of
    // collapsing headers before, which meant the frame time was at the bottom
    // past the scene hierarchy and no two costs were ever on screen together.
    //
    // The split is by what you are doing, not by which class owns the widget:
    // Performance collects every cost in the process from four different files,
    // and three of those files also draw controls into one of the other tabs.
    ImGui::Begin("ptflio");
    if (ImGui::BeginTabBar("sections"))
    {
        using NeuralModelIntegrateTestbed::UiSection;

        if (ImGui::BeginTabItem("Render"))
        {
            m_deferred.ShowImgui(UiSection::Render);
            m_gltfAdapter.ShowImgui(UiSection::Render);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Scene"))
        {
            ShowCameraUI();
            m_gltfAdapter.ShowImgui(UiSection::Scene);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Neural"))
        {
            m_nnModelManager.ShowImgui(UiSection::Neural);
            m_motionStream.ShowImgui(UiSection::Neural);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Performance"))
        {
            ShowFrameTimingUI();
            m_deferred.ShowImgui(UiSection::Performance);
            m_gltfAdapter.ShowImgui(UiSection::Performance);
            m_nnModelManager.ShowImgui(UiSection::Performance);
            m_motionStream.ShowImgui(UiSection::Performance);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

    // Rendering
    ImGui::Render();
}

void XM_CALLCONV Sample::DrawGrid(FXMVECTOR xAxis, FXMVECTOR yAxis, FXMVECTOR origin, size_t xdivs, size_t ydivs, GXMVECTOR color)
{
    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Draw grid");

    m_lineEffect->Apply(commandList);

    m_batch->Begin(commandList);

    xdivs = std::max<size_t>(1, xdivs);
    ydivs = std::max<size_t>(1, ydivs);

    for (size_t i = 0; i <= xdivs; ++i)
    {
        float fPercent = float(i) / float(xdivs);
        fPercent = (fPercent * 2.0f) - 1.0f;
        XMVECTOR vScale = XMVectorScale(xAxis, fPercent);
        vScale = XMVectorAdd(vScale, origin);

        VertexPositionColor v1(XMVectorSubtract(vScale, yAxis), color);
        VertexPositionColor v2(XMVectorAdd(vScale, yAxis), color);
        m_batch->DrawLine(v1, v2);
    }

    for (size_t i = 0; i <= ydivs; i++)
    {
        float fPercent = float(i) / float(ydivs);
        fPercent = (fPercent * 2.0f) - 1.0f;
        XMVECTOR vScale = XMVectorScale(yAxis, fPercent);
        vScale = XMVectorAdd(vScale, origin);

        VertexPositionColor v1(XMVectorSubtract(vScale, xAxis), color);
        VertexPositionColor v2(XMVectorAdd(vScale, xAxis), color);
        m_batch->DrawLine(v1, v2);
    }

    m_batch->End();

    PIXEndEvent(commandList);
}

void Sample::DrawMotionSkeleton()
{
    if (!m_motionStream.IsLoaded())
    {
        return;
    }

    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Draw motion skeleton");

    // Same effect and batch as the grid: vertex-coloured lines in world space,
    // and m_lineEffect already has this frame's view and projection.
    m_lineEffect->Apply(commandList);
    m_batch->Begin(commandList);
    m_motionStream.Draw(*m_batch);
    m_batch->End();

    PIXEndEvent(commandList);
}
#pragma endregion

#pragma region Message Handlers
// Message handlers
void Sample::OnActivated()
{
}

void Sample::OnDeactivated()
{
}

void Sample::OnSuspending()
{
    m_audEngine->Suspend();
}

void Sample::OnResuming()
{
    m_timer.ResetElapsedTime();
    m_gamePadButtons.Reset();
    m_keyboardButtons.Reset();
    m_audEngine->Resume();
}

void Sample::OnWindowSizeChanged(int width, int height)
{
    if (!m_deviceResources->WindowSizeChanged(width, height))
        return;

    CreateWindowSizeDependentResources();
}

void Sample::NewAudioDevice()
{
    if (m_audEngine && !m_audEngine->IsAudioDevicePresent())
    {
        // Setup a retry in 1 second
        m_audioTimerAcc = 1.f;
        m_retryDefault = true;
    }
}

// Properties
void Sample::GetDefaultSize(int& width, int& height) const
{
    width = 1280;
    height = 720;
}
#pragma endregion

#pragma region Direct3D Resources
// These are the resources that depend on the device.
void Sample::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

    // The frame graph and the G-buffer it owns. Declared here rather than per
    // frame: the targets survive across frames, and the graph remembers the
    // state each one was left in so the next frame's barriers come out right.
    m_frameGraph.SetDevice(device);
    m_deferred.CreateDeviceDependentResources(device, m_frameGraph);

    m_states = std::make_shared<CommonStates>(device);

    m_resourceDescriptors = std::make_shared<DescriptorHeap>(device, Descriptors::Count);

    m_batch = std::make_unique<PrimitiveBatch<VertexPositionColor>>(device);

    m_shape = GeometricPrimitive::CreateTeapot(4.f, 8);

    // SDKMESH has to use clockwise winding with right-handed coordinates, so textures are flipped in U
    wchar_t strFilePath[MAX_PATH] = {};
    DX::FindMediaFile(strFilePath, MAX_PATH, L"Tiny\\tiny.sdkmesh");

    wchar_t txtPath[MAX_PATH] = {};
    {
        wchar_t drive[_MAX_DRIVE];
        wchar_t path[_MAX_PATH];

        if (_wsplitpath_s(strFilePath, drive, _MAX_DRIVE, path, _MAX_PATH, nullptr, 0, nullptr, 0))
            throw std::exception("_wsplitpath_s");

        if (_wmakepath_s(txtPath, _MAX_PATH, drive, path, nullptr, nullptr))
            throw std::exception("_wmakepath_s");
    }

    m_model = Model::CreateFromSDKMESH(device, strFilePath);

    {
        ResourceUploadBatch resourceUpload(device);

        resourceUpload.Begin();

        m_model->LoadStaticBuffers(device, resourceUpload);

        DX::FindMediaFile(strFilePath, MAX_PATH, L"seafloor.dds");
        DX::ThrowIfFailed(
            CreateDDSTextureFromFile(device, resourceUpload, strFilePath, m_texture1.ReleaseAndGetAddressOf())
        );

        CreateShaderResourceView(device, m_texture1.Get(), m_resourceDescriptors->GetCpuHandle(Descriptors::SeaFloor));

        DX::FindMediaFile(strFilePath, MAX_PATH, L"windowslogo.dds");
        DX::ThrowIfFailed(
            CreateDDSTextureFromFile(device, resourceUpload, strFilePath, m_texture2.ReleaseAndGetAddressOf())
        );

        CreateShaderResourceView(device, m_texture2.Get(), m_resourceDescriptors->GetCpuHandle(Descriptors::WindowsLogo));

        // Two render target states now, because the overlays were split either
        // side of the tonemap. A PSO bakes its RTV format in, so a world-space
        // effect drawing into the HDR scene colour target cannot share one with
        // a screen-space effect drawing onto the back buffer.
        RenderTargetState rtState(m_deviceResources->GetBackBufferFormat(), m_deviceResources->GetDepthBufferFormat());
        RenderTargetState sceneState(NeuralModelIntegrateTestbed::kSceneColorFormat,
                                     m_deviceResources->GetDepthBufferFormat());

        {
            SpriteBatchPipelineStateDescription pd(rtState);

            m_sprites = std::make_unique<SpriteBatch>(device, resourceUpload, pd);
        }

        {
            // DepthRead, not DepthNone. The grid and the motion skeleton are
            // world-space geometry drawn in the forward overlay pass, which now
            // runs AFTER the deferred lighting resolve rather than before the
            // scene was drawn. Without a depth test they would paint over the
            // character instead of being hidden behind it -- the old ordering
            // got that for free by drawing the lines first and letting the
            // scene paint over them. Read rather than write, so the lines do
            // not occlude the overlays that follow.
            EffectPipelineStateDescription pd(
                &VertexPositionColor::InputLayout,
                CommonStates::Opaque,
                CommonStates::DepthRead,
                CommonStates::CullNone,
                sceneState,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);

            m_lineEffect = std::make_unique<BasicEffect>(device, EffectFlags::VertexColor, pd);
        }

        {
            EffectPipelineStateDescription pd(
                &GeometricPrimitive::VertexType::InputLayout,
                CommonStates::Opaque,
                CommonStates::DepthDefault,
                CommonStates::CullNone,
                sceneState);

            m_shapeEffect = std::make_unique<BasicEffect>(device, EffectFlags::PerPixelLighting | EffectFlags::Texture, pd);
            m_shapeEffect->EnableDefaultLighting();
            m_shapeEffect->SetTexture(m_resourceDescriptors->GetGpuHandle(Descriptors::SeaFloor), m_states->AnisotropicWrap());
        }

        m_modelResources = m_model->LoadTextures(device, resourceUpload, txtPath);

        {
            EffectPipelineStateDescription psd(
                nullptr,
                CommonStates::Opaque,
                CommonStates::DepthDefault,
                CommonStates::CullClockwise,    // Using RH coordinates, and SDKMESH is in LH coordiantes
                rtState);

            EffectPipelineStateDescription alphapsd(
                nullptr,
                CommonStates::NonPremultiplied, // Using straight alpha
                CommonStates::DepthRead,
                CommonStates::CullClockwise,    // Using RH coordinates, and SDKMESH is in LH coordiantes
                rtState);

            m_modelEffects = m_model->CreateEffects(psd, alphapsd, m_modelResources->Heap(), m_states->Heap());
        }

        DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_18.spritefont");
        m_font = std::make_unique<SpriteFont>(device, resourceUpload,
            strFilePath,
            m_resourceDescriptors->GetCpuHandle(Descriptors::SegoeFont),
            m_resourceDescriptors->GetGpuHandle(Descriptors::SegoeFont));

        // Upload the resources to the GPU.
        auto uploadResourcesFinished = resourceUpload.End(m_deviceResources->GetCommandQueue());

        // Wait for the upload thread to terminate
        uploadResourcesFinished.wait();
    }
}

// Allocate all memory resources that change on a window SizeChanged event.
void Sample::CreateWindowSizeDependentResources()
{
    auto size = m_deviceResources->GetOutputSize();

    // Recreates every G-buffer target at the new size, and rewrites the view
    // onto DeviceResources' depth buffer -- which is a different resource after
    // a resize, so the old descriptor would point at freed memory.
    m_frameGraph.Resize(static_cast<std::uint32_t>(size.right),
                        static_cast<std::uint32_t>(size.bottom));
    m_deferred.CreateWindowSizeDependentResources(*m_deviceResources);

    UpdateProjection();

    auto viewport = m_deviceResources->GetScreenViewport();
    m_sprites->SetViewport(viewport);
}

void Sample::UpdateViewMatrix()
{
    m_view = Matrix::CreateLookAt(m_camera->mEye, m_camera->mAt, m_camera->mUp);
    // The overlay effects hold their own copy, and they draw into the same
    // scene colour target as the deferred geometry, so they have to agree with
    // it about where the camera is.
    m_lineEffect->SetView(m_view);
    m_shapeEffect->SetView(m_view);
}

void Sample::UpdateProjection()
{
    auto size = m_deviceResources->GetOutputSize();
    float aspectRatio = float(size.right) / float(size.bottom);
    float fovAngleY = m_fovDegrees * XM_PI / 180.0f;

    // This is a simple example of change that can be made when the app is in
    // portrait or snapped view. Applied to a local, so the field of view the
    // panel shows stays the one that was asked for.
    if (aspectRatio < 1.0f)
    {
        fovAngleY *= 2.0f;
    }

    // This sample makes use of a right-handed coordinate system using row-major matrices.
    m_projection = Matrix::CreatePerspectiveFieldOfView(
        fovAngleY,
        aspectRatio,
        m_near,
        m_far
    );

    // The overlay effects hold their own copy, so without this the grid and the
    // teapot keep the previous projection and stop lining up with the scene.
    m_lineEffect->SetProjection(m_projection);
    m_shapeEffect->SetProjection(m_projection);
}

void Sample::OnDeviceLost()
{
    // Reset() drops the declared G-buffer targets and their views as well as
    // the pass list, so CreateDeviceDependentResources re-declares them. The
    // deferred renderer's own PSO and descriptors go with it.
    m_frameGraph.Reset();
    m_deferred.OnDeviceLost();

    m_texture1.Reset();
    m_texture2.Reset();

    m_font.reset();
    m_batch.reset();
    m_shape.reset();
    m_model.reset();
    m_lineEffect.reset();
    m_shapeEffect.reset();
    m_modelEffects.clear();
    m_modelResources.reset();
    m_sprites.reset();
    m_resourceDescriptors.reset();
    m_states.reset();
    m_graphicsMemory.reset();
}

void Sample::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion
