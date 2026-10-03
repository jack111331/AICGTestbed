// Smoke test for the vcpkg imgui package.
//
// Builds a real ImGui frame with no graphics device to confirm the core library
// links and produces draw data, then takes the address of the DX12 and Win32
// backend entry points. Those symbols only exist if the dx12-binding and
// win32-binding features actually compiled the backends into imgui.lib, so this
// fails at link time if the manifest features are ever dropped.
#include <windows.h>

#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_win32.h>

#include <iostream>

// imgui_impl_win32.h keeps this declaration inside an `#if 0` on purpose, so it
// does not drag <windows.h> into the header. Upstream's instruction is to copy
// the line into your own .cpp -- the real WndProc will need it too.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

int main() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(1280.0f, 720.0f);
  io.DeltaTime = 1.0f / 60.0f;
  io.BackendPlatformName = "ptflio-smoke";
  io.BackendRendererName = "ptflio-smoke";
  // 1.92 moved fonts to a dynamic texture system: the renderer declares that it
  // services ImTextureData requests, and ImGui then emits them through
  // ImDrawData::Textures instead of eagerly building an atlas the backend must
  // have uploaded. Stand in for a renderer here and simply ignore the requests.
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  io.Fonts->AddFontDefault();

  // Frame 1 only produces the font-texture request: ImGui cannot emit draw
  // commands that reference a texture the renderer has not created yet. Service
  // the request the way a real backend would, then render frame 2 for geometry.
  int serviced = 0;
  const ImDrawData* draw = nullptr;
  for (int frame = 0; frame < 2; ++frame) {
    ImGui::NewFrame();
    ImGui::Begin("ptflio");
    ImGui::Text("Hello from Bazel");
    ImGui::Button("Click");
    ImGui::End();
    ImGui::Render();
    draw = ImGui::GetDrawData();

    if (draw->Textures != nullptr) {
      for (ImTextureData* tex : *draw->Textures) {
        if (tex->Status == ImTextureStatus_WantCreate ||
            tex->Status == ImTextureStatus_WantUpdates) {
          // A real backend would upload tex->Pixels here and hand back its own
          // texture handle; any non-zero ID is enough to unblock rendering.
          tex->SetTexID(static_cast<ImTextureID>(1));
          tex->SetStatus(ImTextureStatus_OK);
          ++serviced;
        }
      }
    }
  }

  // Snapshot before DestroyContext(): ImDrawData is owned by the context and is
  // freed along with it.
  const int cmd_lists = draw->CmdListsCount;
  const int vertices = draw->TotalVtxCount;
  const int indices = draw->TotalIdxCount;

  std::cout << "imgui linked OK" << std::endl
            << "  version:    " << ImGui::GetVersion() << std::endl
            << "  tex served: " << serviced << std::endl
            << "  draw lists: " << cmd_lists << std::endl
            << "  vertices:   " << vertices << std::endl
            << "  indices:    " << indices << std::endl;

  // Force the linker to resolve the backend symbols. ImGui_ImplDX12_Init is
  // overloaded, so the function-pointer type selects the InitInfo overload.
  bool (*dx12_init)(ImGui_ImplDX12_InitInfo*) = &ImGui_ImplDX12_Init;
  void (*dx12_render)(ImDrawData*, ID3D12GraphicsCommandList*) =
      &ImGui_ImplDX12_RenderDrawData;
  bool (*win32_init)(void*) = &ImGui_ImplWin32_Init;
  LRESULT (*win32_wndproc)(HWND, UINT, WPARAM, LPARAM) =
      &ImGui_ImplWin32_WndProcHandler;

  std::cout << "backends present" << std::endl
            << "  dx12:  init=" << (dx12_init != nullptr)
            << " render=" << (dx12_render != nullptr) << std::endl
            << "  win32: init=" << (win32_init != nullptr)
            << " wndproc=" << (win32_wndproc != nullptr) << std::endl;

  ImGui::DestroyContext();

  if (cmd_lists <= 0 || vertices <= 0) {
    std::cerr << "no draw data produced" << std::endl;
    return 1;
  }
  return 0;
}
