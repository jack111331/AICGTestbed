//--------------------------------------------------------------------------------------
// DirectXTKSimpleSample12.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "Application.hpp"
#include "Descriptors/DescriptorContext.hpp"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <directx/d3dx12.h>


#include "FindMedia.h"

extern void ExitSample() noexcept;

using namespace DirectX;

// The model ModelManager registers at startup and RecordModelDispatch runs each
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

    // m_gltfAdapter.Initialize("resources/microphone_gxl_066_bafhcteks/scene_withlight.gltf");
    m_gltfAdapter.Initialize("resources/microphone_and_animated_char/scene_withlight_animated_char.gltf");
    // m_gltfAdapter.Initialize("resources/TestGLTF/WithTexture.gltf");
    m_gltfAdapter.SetCommonStates(m_states);
    m_gltfAdapter.AssignCamera(m_camera);
    m_gltfAdapter.PrepareBuffer(m_deviceResources);
    m_gltfAdapter.PrepareImage(m_deviceResources);
    m_gltfAdapter.BuildSceneGraph();
    m_gltfAdapter.PreparePSO(m_deviceResources);

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
    // It is handed this renderer's device and queue plus the IDMLDevice the
    // ModelManager already made, so the DirectML provider runs on the same
    // adapter and queue as rendering and there is one IDMLDevice, not two.
    // Hence this must follow m_nnModelManager.Initialize above.
    {
        // The retarget target lives in the glTF scene, so this has to follow
        // BuildSceneGraph above.
        m_motionStream.AttachScene(&m_gltfAdapter.Scene(),
                                   "resources/retarget/mixamo.txt");

        std::string motionError;
        if (m_motionStream.Load("resources/FloodDiffusion",
                                m_deviceResources->GetD3DDevice(),
                                m_deviceResources->GetCommandQueue(),
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

    // m_view = Matrix::CreateLookAt(eye, at, Vector3::UnitY);
    m_view = Matrix::CreateLookAt(m_camera->mEye, m_camera->mAt, m_camera->mUp);

    m_world = Matrix::CreateRotationY(float(timer.GetTotalSeconds() * XM_PIDIV4));

    m_lineEffect->SetView(m_view);
    m_lineEffect->SetWorld(Matrix::Identity);

    m_shapeEffect->SetView(m_view);

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

    auto kb = m_keyboard->GetState();
    m_keyboardButtons.Update(kb);

    if (kb.Escape)
    {
        ExitSample();
    }

    auto mouse = m_mouse->GetState();
    m_mouseTracker.Update( mouse );
    if (mouse.positionMode == Mouse::MODE_RELATIVE)
    {
        if (m_mouseTracker.leftButton == Mouse::ButtonStateTracker::HELD) {
            m_camera->RotateAroundYAxis(-mouse.x * (float)timer.GetElapsedSeconds());
            m_camera->RotatePitch(-mouse.y * (float)timer.GetElapsedSeconds());
        }
        if (m_mouseTracker.rightButton == Mouse::ButtonStateTracker::HELD) {
            m_camera->MoveForward(-mouse.y * (float)timer.GetElapsedSeconds());
        }
        if (m_mouseTracker.middleButton == Mouse::ButtonStateTracker::HELD) {
            m_camera->MoveUpward(-mouse.y * (float)timer.GetElapsedSeconds());
            m_camera->MoveRightward(-mouse.x * (float)timer.GetElapsedSeconds());
        }
    }
    if ( m_mouseTracker.leftButton == Mouse::ButtonStateTracker::PRESSED ||
         m_mouseTracker.rightButton == Mouse::ButtonStateTracker::PRESSED ||
         m_mouseTracker.middleButton == Mouse::ButtonStateTracker::PRESSED  ) {
        m_mouse->SetMode(Mouse::MODE_RELATIVE);
    }
    else if (m_mouseTracker.leftButton == Mouse::ButtonStateTracker::ButtonState::RELEASED ||
             m_mouseTracker.rightButton == Mouse::ButtonStateTracker::ButtonState::RELEASED ||
             m_mouseTracker.middleButton == Mouse::ButtonStateTracker::ButtonState::RELEASED)
    {
        m_mouse->SetMode(Mouse::MODE_ABSOLUTE);
    }
    mouse;

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

    // Outside the graph: a compute dispatch on the same queue that touches
    // none of the frame's render targets. It would belong in the graph the
    // moment it produced something a pass reads.
    //
    // Dispatched by name. An unregistered name records nothing and returns
    // false rather than asserting, so a model that failed to load does not
    // take the frame down with it.
    m_nnModelManager.RecordModelDispatch(commandList, kNeuralModelName);

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

    m_deferred.AddPasses(m_frameGraph, targets, inputs,
                         [this](ID3D12GraphicsCommandList* commandList) {
                             m_gltfAdapter.RecordGeometry(commandList, m_view, m_projection);
                         });

    {
        NeuralModelIntegrateTestbed::Render::RenderPassDesc overlay;
        overlay.name = "ForwardOverlay";
        overlay.writes = {
            {targets.color, NeuralModelIntegrateTestbed::Render::Access::RenderTarget},
            // Depth-tested, so the graph has to transition depth back out of
            // the readable state the lighting pass left it in.
            {targets.depth, NeuralModelIntegrateTestbed::Render::Access::DepthWrite},
        };
        overlay.execute = [this](ID3D12GraphicsCommandList* commandList,
                                 const NeuralModelIntegrateTestbed::Render::PassResources&) {
            RecordForwardOverlays(commandList);
        };
        m_frameGraph.AddPass(std::move(overlay));
    }


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

void Sample::RecordForwardOverlays(ID3D12GraphicsCommandList* commandList)
{
    // Draw procedurally generated dynamic grid
    const XMVECTORF32 xaxis = { 20.f, 0.f, 0.f };
    const XMVECTORF32 yaxis = { 0.f, 0.f, 20.f };
    DrawGrid(xaxis, yaxis, g_XMZero, 20, 20, Colors::Gray);
    DrawMotionSkeleton();

    // Set the descriptor heaps
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

    // Draw 3D object
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Draw teapot");
    XMMATRIX local = m_world * Matrix::CreateTranslation(-2.f, -2.f, -4.f);
    m_shapeEffect->SetWorld(local);
    m_shapeEffect->Apply(commandList);
    m_shape->Draw(commandList);
    PIXEndEvent(commandList);
}

void Sample::BuildUserInterface()
{
    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    bool show_demo_window = true;
    float clear_color[3] = {0.0, .0, .0};
    // ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        m_deferred.ShowImgui();
        m_gltfAdapter.ShowImgui();
        m_nnModelManager.ShowImgui();
        m_motionStream.ShowImgui();

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();
    }

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

        RenderTargetState rtState(m_deviceResources->GetBackBufferFormat(), m_deviceResources->GetDepthBufferFormat());

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
                rtState,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);

            m_lineEffect = std::make_unique<BasicEffect>(device, EffectFlags::VertexColor, pd);
        }

        {
            EffectPipelineStateDescription pd(
                &GeometricPrimitive::VertexType::InputLayout,
                CommonStates::Opaque,
                CommonStates::DepthDefault,
                CommonStates::CullNone,
                rtState);

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

    float aspectRatio = float(size.right) / float(size.bottom);
    float fovAngleY = 70.0f * XM_PI / 180.0f;

    // This is a simple example of change that can be made when the app is in
    // portrait or snapped view.
    if (aspectRatio < 1.0f)
    {
        fovAngleY *= 2.0f;
    }

    // This sample makes use of a right-handed coordinate system using row-major matrices.
    m_projection = Matrix::CreatePerspectiveFieldOfView(
        fovAngleY,
        aspectRatio,
        0.01f,
        100.0f
    );

    m_lineEffect->SetProjection(m_projection);
    m_shapeEffect->SetProjection(m_projection);

    auto viewport = m_deviceResources->GetScreenViewport();
    m_sprites->SetViewport(viewport);
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
