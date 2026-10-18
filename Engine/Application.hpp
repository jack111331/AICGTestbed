//--------------------------------------------------------------------------------------
// Application.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "DeviceResources.hpp"
#include "StepTimer.hpp"
#include "DeferredRenderer.hpp"
#include "GLTFAdapter.hpp"
#include "ModelManager.hpp"
#include "MotionStreamView.hpp"
#include "RenderGraph.hpp"
#include "Camera.hpp"

// Simple free list based allocator
struct ExampleDescriptorHeapAllocator
{
    ID3D12DescriptorHeap*       Heap = nullptr;
    D3D12_DESCRIPTOR_HEAP_TYPE  HeapType = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES;
    D3D12_CPU_DESCRIPTOR_HANDLE HeapStartCpu;
    D3D12_GPU_DESCRIPTOR_HANDLE HeapStartGpu;
    UINT                        HeapHandleIncrement;
    std::vector<int>               FreeIndices;

    void Create(ID3D12Device* device, ID3D12DescriptorHeap* heap)
    {
        assert(Heap == nullptr && FreeIndices.empty());
        Heap = heap;
        D3D12_DESCRIPTOR_HEAP_DESC desc = heap->GetDesc();
        HeapType = desc.Type;
        HeapStartCpu = Heap->GetCPUDescriptorHandleForHeapStart();
        HeapStartGpu = Heap->GetGPUDescriptorHandleForHeapStart();
        HeapHandleIncrement = device->GetDescriptorHandleIncrementSize(HeapType);
        FreeIndices.reserve((int)desc.NumDescriptors);
        for (int n = desc.NumDescriptors; n > 0; n--)
            FreeIndices.push_back(n - 1);
    }
    void Destroy()
    {
        Heap = nullptr;
        FreeIndices.clear();
    }
    void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_desc_handle)
    {
        assert(FreeIndices.size() > 0);
        int idx = FreeIndices.back();
        FreeIndices.pop_back();
        out_cpu_desc_handle->ptr = HeapStartCpu.ptr + (idx * HeapHandleIncrement);
        out_gpu_desc_handle->ptr = HeapStartGpu.ptr + (idx * HeapHandleIncrement);
    }
    void Free(D3D12_CPU_DESCRIPTOR_HANDLE out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE out_gpu_desc_handle)
    {
        int cpu_idx = (int)((out_cpu_desc_handle.ptr - HeapStartCpu.ptr) / HeapHandleIncrement);
        int gpu_idx = (int)((out_gpu_desc_handle.ptr - HeapStartGpu.ptr) / HeapHandleIncrement);
        assert(cpu_idx == gpu_idx);
        FreeIndices.push_back(cpu_idx);
    }
};

static     ExampleDescriptorHeapAllocator g_pd3dSrvDescHeapAlloc; // TODO need refactor

// A basic sample implementation that creates a D3D12 device and
// provides a render loop.
class Sample final : public DX::IDeviceNotify
{
public:

    Sample() noexcept(false);
    ~Sample();

    Sample(Sample&&) = default;
    Sample& operator= (Sample&&) = default;

    Sample(Sample const&) = delete;
    Sample& operator= (Sample const&) = delete;

    // Initialization and management
    void Initialize(HWND window, int width, int height);

    // Basic render loop
    void Tick();

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // Messages
    void OnActivated();
    void OnDeactivated();
    void OnSuspending();
    void OnResuming();
    void OnWindowSizeChanged(int width, int height);
    void NewAudioDevice();

    // Properties
    void GetDefaultSize( int& width, int& height ) const;

    std::shared_ptr<DX::DeviceResources> GetDeviceResource() {return m_deviceResources;}

private:

    void Update(DX::StepTimer const& timer);
    void Render();

    // Builds this frame's pass list. Rebuilt every frame rather than once,
    // which is what lets a pass be added or dropped by a UI toggle without any
    // of the other passes' barriers needing to change.
    void BuildFrameGraph();

    // World-space forward draws: the grid, the motion skeleton and the
    // sample's teapot. They share the deferred depth buffer, so they occlude
    // against the deferred geometry, and they go into the HDR scene colour
    // target BEFORE the tonemap -- they are radiance like everything else.
    void RecordWorldOverlays(ID3D12GraphicsCommandList* commandList);

    // Screen-space draws: the sprite and the sample's caption. These run AFTER
    // the tonemap, straight onto the back buffer, because they are authored in
    // display space and tonemapping them would be wrong.
    void RecordScreenOverlays(ID3D12GraphicsCommandList* commandList);

    // Points the camera at the loaded scene and sizes the projection's near
    // and far planes to it.
    //
    // Not cosmetic. A converted street scene's contents sit hundreds of units
    // from the origin -- the Lumberyard Bistro is centred at (13.9, 510.6,
    // 593.2) with a 163-unit diagonal -- while the default camera sits 1.5
    // units from the origin behind a far plane of 100. Every triangle loads
    // correctly and nothing whatsoever is on screen.
    void FrameSceneWithCamera();

    // The scene's bounds, radius and the drag speeds derived from them, without
    // touching the camera. Split out because a manifest that pins the camera
    // still needs all of it: the radius sets the shadow cascade distance and
    // the lightmap's extent, neither of which is about where the view is.
    //
    // Returns false for a scene nothing can be measured from, which is a scene
    // with no drawable geometry.
    bool MeasureScene();

    // The camera a scene manifest pinned, used instead of framing the scene.
    // Measures the scene first, then overrides the eye, the target and whichever
    // of the three projection values the manifest named.
    void ApplySceneCamera(const NeuralModelIntegrateTestbed::SceneCamera &camera);

    // The camera and projection panel: eye, target, field of view, clip planes
    // and navigation speed. Lives here rather than on Camera because half of
    // what it edits -- the field of view and the clip planes -- belongs to the
    // projection, which is Sample's, and because Camera is a small maths class
    // with no UI dependency and is more useful kept that way.
    void ShowCameraUI();

    // The frame's own cost, at the top of the Performance tab: the wall clock
    // and its recent history. A number alone cannot distinguish a steady 8 ms
    // from an 8 ms average over a frame that alternates 2 and 14, and the
    // second is what a stutter is, so the history is the point rather than
    // decoration.
    void ShowFrameTimingUI();

    // Rebuilds the projection matrix and pushes it into the DirectXTK effects
    // the forward overlay pass draws with.
    //
    // Split out of CreateWindowSizeDependentResources so the camera panel can
    // change the field of view or a clip plane without the rest of it: that
    // function also rewrites the shader resource view onto the depth buffer,
    // and doing so from inside a frame would overwrite a descriptor a submitted
    // command list is still referencing.
    void UpdateProjection();

    // Rebuilds the view matrix from the camera and pushes it into the overlay
    // effects. Called from Render, after Update has applied this frame's input:
    // built at the top of Update instead it would describe the previous frame's
    // camera while the lighting and the sky read the eye position fresh.
    void UpdateViewMatrix();

    // Everything between ImGui::NewFrame and ImGui::Render. Called before the
    // graph records, because the draw data has to exist before the pass that
    // submits it runs.
    void BuildUserInterface();

    void CreateDeviceDependentResources();
    void CreateWindowSizeDependentResources();

    void XM_CALLCONV DrawGrid(DirectX::FXMVECTOR xAxis, DirectX::FXMVECTOR yAxis, DirectX::FXMVECTOR origin, size_t xdivs, size_t ydivs, DirectX::GXMVECTOR color);

    // The streamed FloodDiffusion skeleton, drawn with the same line batch as
    // the grid.
    void DrawMotionSkeleton();

    // Device resources.
    std::shared_ptr<DX::DeviceResources>        m_deviceResources;

    // Rendering loop timer.
    DX::StepTimer                               m_timer;

    // Input devices.
    std::unique_ptr<DirectX::GamePad>           m_gamePad;
    std::unique_ptr<DirectX::Keyboard>          m_keyboard;
    std::unique_ptr<DirectX::Mouse>             m_mouse;

    DirectX::GamePad::ButtonStateTracker        m_gamePadButtons;
    DirectX::Keyboard::KeyboardStateTracker     m_keyboardButtons;
    DirectX::Mouse::ButtonStateTracker          m_mouseTracker;

    // DirectXTK objects.
    std::unique_ptr<DirectX::GraphicsMemory>                                m_graphicsMemory;
    std::shared_ptr<DirectX::DescriptorHeap>                                m_resourceDescriptors;
    std::shared_ptr<DirectX::CommonStates>                                  m_states;
    std::unique_ptr<DirectX::BasicEffect>                                   m_lineEffect;
    std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>>  m_batch;
    std::unique_ptr<DirectX::BasicEffect>                                   m_shapeEffect;
    std::unique_ptr<DirectX::Model>                                         m_model;
    DirectX::Model::EffectCollection                                        m_modelEffects;
    std::unique_ptr<DirectX::EffectTextureFactory>                          m_modelResources;
    std::unique_ptr<DirectX::GeometricPrimitive>                            m_shape;
    std::unique_ptr<DirectX::SpriteBatch>                                   m_sprites;
    std::unique_ptr<DirectX::SpriteFont>                                    m_font;

    std::unique_ptr<DirectX::AudioEngine>                                   m_audEngine;
    std::unique_ptr<DirectX::WaveBank>                                      m_waveBank;
    std::unique_ptr<DirectX::SoundEffect>                                   m_soundEffect;
    std::unique_ptr<DirectX::SoundEffectInstance>                           m_effect1;
    std::unique_ptr<DirectX::SoundEffectInstance>                           m_effect2;

    Microsoft::WRL::ComPtr<ID3D12Resource>                                  m_texture1;
    Microsoft::WRL::ComPtr<ID3D12Resource>                                  m_texture2;

    uint32_t                                                                m_audioEvent;
    float                                                                   m_audioTimerAcc;

    bool                                                                    m_retryDefault;

    DirectX::SimpleMath::Matrix                                             m_world;
    DirectX::SimpleMath::Matrix                                             m_view;
    DirectX::SimpleMath::Matrix                                             m_projection;

    // Near and far planes, sized to the scene by FrameSceneWithCamera rather
    // than fixed, so one projection serves both a 2-metre character and a
    // 160-metre street. The ratio is what drives depth precision, so the near
    // plane is scaled with the scene instead of being left at 0.01.
    float                                                                   m_near = 0.01f;
    float                                                                   m_far = 100.0f;
    // The framing, kept so the UI can return to it after flying away.
    DirectX::SimpleMath::Vector3                                            m_sceneCentre;
    float                                                                   m_sceneRadius = 0.0f;
    // The scene's world bounds, for the shadow cascade fit. Cached because
    // SceneBounds walks every primitive and they only change when the scene is
    // reposed.
    DirectX::SimpleMath::Vector3                                            m_sceneBoundsMin;
    DirectX::SimpleMath::Vector3                                            m_sceneBoundsMax;
    bool                                                                    m_hasSceneBounds = false;

    // Vertical field of view. A member rather than a local in
    // CreateWindowSizeDependentResources so the UI can change it.
    float                                                                   m_fovDegrees = 70.0f;

    // A ring buffer of recent frame times, in milliseconds, for the plot in the
    // Performance tab. Four seconds at 60 Hz, which is long enough to see a
    // periodic hitch and short enough that the plot is still readable.
    static constexpr int                                                    kFrameTimeHistory = 240;
    float                                                                   m_frameTimes[kFrameTimeHistory] = {};
    int                                                                     m_frameTimeCursor = 0;

    // Who owns the drag in progress, latched on the press.
    //
    // The camera and Dear ImGui read the same mouse, and the camera used to
    // read it unconditionally: dragging any slider also orbited the scene.
    // Gating on io.WantCaptureMouse every frame is not enough on its own,
    // because the camera's drag puts the mouse in MODE_RELATIVE, which hides
    // the cursor and stops WM_MOUSEMOVE -- so ImGui's idea of where the
    // pointer is freezes, and if it froze over a panel the camera would stop
    // mid-drag. Deciding once, when the first button goes down, settles both
    // directions: a drag that starts on a widget never reaches the camera, and
    // a drag that starts in the viewport keeps the camera even as it sweeps
    // across the UI.
    bool                                                                    m_mouseDragging = false;
    bool                                                                    m_uiOwnsMouseDrag = false;

    // Multipliers on the mouse-driven camera motion. The raw controls move the
    // camera in world units per pixel of mouse travel per second, which is
    // usable in a two-metre scene and hopeless in a hundred-and-sixty-metre
    // one -- so FrameSceneWithCamera sizes these to the scene rather than
    // leaving the user to discover why dragging appears to do nothing.
    float                                                                   m_cameraMoveSpeed = 1.0f;
    float                                                                   m_cameraRotateSpeed = 1.0f;

    NeuralModelIntegrateTestbed::Render::RenderGraph m_frameGraph;
    NeuralModelIntegrateTestbed::DeferredRenderer m_deferred;
    // This frame's lights, gathered from the scene. A member rather than a
    // local in BuildFrameGraph because the lighting pass's callback holds a
    // pointer to it and does not run until the graph executes, by which time a
    // local would be gone.
    NeuralModelIntegrateTestbed::ShaderLight
        m_frameLights[NeuralModelIntegrateTestbed::kMaxDeferredLights] = {};
    NeuralModelIntegrateTestbed::GLTFAdapter m_gltfAdapter;
    NeuralModelIntegrateTestbed::ModelManager m_nnModelManager;
    NeuralModelIntegrateTestbed::MotionStreamView m_motionStream;
    std::shared_ptr<Camera>                                                 m_camera;

    ID3D12DescriptorHeap*        m_imguiSrvDescHeap = nullptr;

    // Descriptors
    enum Descriptors
    {
        WindowsLogo,
        SeaFloor,
        SegoeFont,
        Count = 65536
    };
};
