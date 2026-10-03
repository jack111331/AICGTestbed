// Smoke test: proves the vcpkg DirectXTK12 headers resolve, the import library
// links, and the DLL is staged into runfiles. Deliberately avoids creating a
// D3D12 device so it can run headless.
#include <Keyboard.h>
#include <SimpleMath.h>

#include <iostream>

int main() {
  DirectX::Keyboard keyboard;
  const auto state = keyboard.GetState();

  DirectX::SimpleMath::Vector3 a(1.0f, 2.0f, 3.0f);
  DirectX::SimpleMath::Vector3 b(4.0f, 5.0f, 6.0f);
  const DirectX::SimpleMath::Vector3 c = a.Cross(b);

  std::cout << "DirectXTK12 linked OK\n"
            << "  Escape down: " << std::boolalpha << state.Escape << "\n"
            << "  cross(a, b): (" << c.x << ", " << c.y << ", " << c.z << ")\n";
  return 0;
}
