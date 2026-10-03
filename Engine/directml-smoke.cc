// Smoke test for the DirectML dependency (//third:directml).
//
// DirectML has no vcpkg port and nothing to download: DirectML.h and
// directml.lib come from the Windows SDK, and directml.dll is in-box. So the
// only things worth proving are that the link flag resolves, that
// DML_TARGET_VERSION_USE_LATEST really exposes the modern API surface, and that
// the in-box runtime can create a device.
#include <windows.h>

#include <d3d12.h>
#include <DirectML.h>
#include <wrl/client.h>

#include <cstdio>
#include <iostream>

using Microsoft::WRL::ComPtr;

namespace {

// Fails loudly at compile time if the target version ever silently falls back to
// whatever the NTDDI ladder in DirectML.h picks. Without
// DML_TARGET_VERSION_USE_LATEST that is 0x5000 (feature level 5.0) on this SDK,
// which compiles fine but hides every DirectML 6.x operator and interface.
static_assert(DML_TARGET_VERSION >= 0x6400,
              "DML_TARGET_VERSION too low -- is DML_TARGET_VERSION_USE_LATEST "
              "still set on //third:directml?");

const char* FeatureLevelName(DML_FEATURE_LEVEL level) {
  switch (level) {
    case DML_FEATURE_LEVEL_1_0: return "1.0";
    case DML_FEATURE_LEVEL_2_0: return "2.0";
    case DML_FEATURE_LEVEL_2_1: return "2.1";
    case DML_FEATURE_LEVEL_3_0: return "3.0";
    case DML_FEATURE_LEVEL_3_1: return "3.1";
    case DML_FEATURE_LEVEL_4_0: return "4.0";
    case DML_FEATURE_LEVEL_4_1: return "4.1";
    case DML_FEATURE_LEVEL_5_0: return "5.0";
    case DML_FEATURE_LEVEL_5_1: return "5.1";
    case DML_FEATURE_LEVEL_5_2: return "5.2";
    case DML_FEATURE_LEVEL_6_0: return "6.0";
    case DML_FEATURE_LEVEL_6_1: return "6.1";
    case DML_FEATURE_LEVEL_6_2: return "6.2";
    case DML_FEATURE_LEVEL_6_3: return "6.3";
    case DML_FEATURE_LEVEL_6_4: return "6.4";
    default: return "unknown";
  }
}

bool Failed(const char* what, HRESULT hr) {
  if (SUCCEEDED(hr)) return false;
  std::fprintf(stderr, "%s failed: 0x%08X\n", what, static_cast<unsigned>(hr));
  return true;
}

}  // namespace

int main() {
  ComPtr<ID3D12Device> d3d12;
  if (Failed("D3D12CreateDevice",
             D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                               IID_PPV_ARGS(&d3d12)))) {
    return 1;
  }

  ComPtr<IDMLDevice> dml;
  if (Failed("DMLCreateDevice",
             DMLCreateDevice(d3d12.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                             IID_PPV_ARGS(&dml)))) {
    return 1;
  }

  // Ask the runtime what it actually supports, rather than assuming the
  // compile-time surface is available.
  static const DML_FEATURE_LEVEL kLevels[] = {
      DML_FEATURE_LEVEL_1_0, DML_FEATURE_LEVEL_2_0, DML_FEATURE_LEVEL_2_1,
      DML_FEATURE_LEVEL_3_0, DML_FEATURE_LEVEL_3_1, DML_FEATURE_LEVEL_4_0,
      DML_FEATURE_LEVEL_4_1, DML_FEATURE_LEVEL_5_0, DML_FEATURE_LEVEL_5_1,
      DML_FEATURE_LEVEL_5_2, DML_FEATURE_LEVEL_6_0, DML_FEATURE_LEVEL_6_1,
      DML_FEATURE_LEVEL_6_2, DML_FEATURE_LEVEL_6_3, DML_FEATURE_LEVEL_6_4,
  };
  DML_FEATURE_QUERY_FEATURE_LEVELS query = {};
  query.RequestedFeatureLevelCount = static_cast<UINT>(std::size(kLevels));
  query.RequestedFeatureLevels = kLevels;
  DML_FEATURE_DATA_FEATURE_LEVELS levels = {};
  if (Failed("CheckFeatureSupport(FEATURE_LEVELS)",
             dml->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS, sizeof(query),
                                      &query, sizeof(levels), &levels))) {
    return 1;
  }

  // Build one trivial operator, to exercise the tensor/operator API rather than
  // only device creation.
  UINT sizes[4] = {1, 1, 2, 2};
  DML_BUFFER_TENSOR_DESC buffer = {};
  buffer.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  buffer.Flags = DML_TENSOR_FLAG_NONE;
  buffer.DimensionCount = 4;
  buffer.Sizes = sizes;
  buffer.TotalTensorSizeInBytes = 4 * sizeof(float);
  const DML_TENSOR_DESC tensor = {DML_TENSOR_TYPE_BUFFER, &buffer};

  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity = {};
  identity.InputTensor = &tensor;
  identity.OutputTensor = &tensor;
  const DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                     &identity};

  ComPtr<IDMLOperator> op;
  if (Failed("CreateOperator", dml->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
    return 1;
  }

  D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
  d3d12->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch));

  std::cout << "DirectML linked OK" << std::endl;
  std::printf("  compiled against: DML_TARGET_VERSION 0x%04X\n",
              DML_TARGET_VERSION);
  std::cout << "  runtime max level: "
            << FeatureLevelName(levels.MaxSupportedFeatureLevel) << std::endl
            << "  d3d12 uma:         " << std::boolalpha << (arch.UMA != 0)
            << std::endl
            << "  identity operator: created" << std::endl;
  return 0;
}
