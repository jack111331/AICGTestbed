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

    // Prepare the command list to render a new frame.
    // TODO since we render gltf first, we change the beforeState from D3D12_RESOURCE_STATE_PRESENT to D3D12_RESOURCE_STATE_RENDER_TARGET already
    m_deviceResources->Prepare();
    Clear();

    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render");

    // Draw procedurally generated dynamic grid
    const XMVECTORF32 xaxis = { 20.f, 0.f, 0.f };
    const XMVECTORF32 yaxis = { 0.f, 0.f, 20.f };
    DrawGrid(xaxis, yaxis, g_XMZero, 20, 20, Colors::Gray);

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

    PIXEndEvent(commandList);

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
        m_gltfAdapter.ShowImgui();
        m_nnModelManager.ShowImgui();

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

    DX::ThrowIfFailed(commandList->Close());
    m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    m_gltfAdapter.Render(m_deviceResources, m_world, m_view, m_projection);

    DX::ThrowIfFailed(commandList->Reset(m_deviceResources->GetCommandAllocator(), nullptr));

    // Render Dear ImGui graphics
    commandList->OMSetRenderTargets(1, &m_deviceResources->GetRenderTargetView(), FALSE, nullptr);
    commandList->SetDescriptorHeaps(1, &m_imguiSrvDescHeap);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);

    // Dispatched by name. An unregistered name records nothing and returns
    // false rather than asserting, so a model that failed to load does not
    // take the frame down with it.
    m_nnModelManager.RecordModelDispatch(commandList, kNeuralModelName);
    // commandList->Close();

    // m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

    // Show the new frame.
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

// Helper method to clear the back buffers.
void Sample::Clear()
{
    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Clear");

    // Clear the views.
    auto rtvDescriptor = m_deviceResources->GetRenderTargetView();
    auto dsvDescriptor = m_deviceResources->GetDepthStencilView();

    commandList->OMSetRenderTargets(1, &rtvDescriptor, FALSE, &dsvDescriptor);
    commandList->ClearRenderTargetView(rtvDescriptor, Colors::CornflowerBlue, 0, nullptr);
    commandList->ClearDepthStencilView(dsvDescriptor, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // Set the viewport and scissor rect.
    auto viewport = m_deviceResources->GetScreenViewport();
    auto scissorRect = m_deviceResources->GetScissorRect();
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissorRect);

    PIXEndEvent(commandList);
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
            EffectPipelineStateDescription pd(
                &VertexPositionColor::InputLayout,
                CommonStates::Opaque,
                CommonStates::DepthNone,
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
