// Smoke test for DynamicDescriptorHeap::CommitStagedDescriptors.
//
// Exists because the interesting path -- more than one descriptor table stale
// in the same commit, copied by a single CopyDescriptors call -- is not reached
// by any root signature in this project. Every one of ours has exactly one
// CBV_SRV_UAV table, so the batch has a single element and the flattening is
// never put to the test. This builds a two-table root signature on purpose.
//
// What makes it a real test rather than a call that returns: the D3D12 debug
// layer validates CopyDescriptors' arguments, and the totals on the two sides
// of a flattened copy disagreeing is exactly the mistake a batched version
// invites. The info queue is drained and asserted empty after each case, so a
// malformed call fails here instead of silently copying the wrong descriptors.
//
// Needs a real device, but no window and no swap chain.
#include "pch.h"

#include "Descriptors/DescriptorContext.hpp"
#include "Descriptors/DescriptorHeapBinder.hpp"
#include "Descriptors/DynamicDescriptorHeap.hpp"
#include "Descriptors/RootSignature.hpp"

#include <directx/d3d12.h>
#include <directx/d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <cstdio>
#include <crtdbg.h>
#include <cstdlib>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using namespace NeuralModelIntegrateTestbed::Descriptors;

namespace {

int g_failures = 0;

void Expect(const char* what, bool condition) {
    std::printf("  %-48s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

void ExpectEqual(const char* what, uint64_t got, uint64_t expected) {
    const bool ok = got == expected;
    std::printf("  %-48s %6llu  expected %6llu  %s\n", what,
                static_cast<unsigned long long>(got),
                static_cast<unsigned long long>(expected), ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

// What a commit did, captured through the setter CommitStagedDescriptors calls
// for each table it bound. This is the only externally visible part of the
// batch: which tables were bound, and where in the GPU heap each one landed.
struct BoundTable {
    UINT rootIndex;
    D3D12_GPU_DESCRIPTOR_HANDLE handle;
};

std::vector<BoundTable> g_bound;

void RecordBinding(ID3D12GraphicsCommandList*, UINT rootIndex,
                   D3D12_GPU_DESCRIPTOR_HANDLE handle) {
    g_bound.push_back({rootIndex, handle});
}

// Any error or corruption message the debug layer produced since the last
// drain. A well-formed CopyDescriptors produces none.
std::vector<std::string> DrainErrors(ID3D12InfoQueue* infoQueue) {
    std::vector<std::string> messages;
    if (infoQueue == nullptr) {
        return messages;
    }
    const UINT64 count = infoQueue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T length = 0;
        if (FAILED(infoQueue->GetMessage(i, nullptr, &length))) continue;
        std::vector<char> storage(length);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(infoQueue->GetMessage(i, message, &length))) continue;
        if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
            message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
            messages.emplace_back(message->pDescription, message->DescriptionByteLength);
        }
    }
    infoQueue->ClearStoredMessages();
    return messages;
}

void ExpectNoDebugErrors(const char* what, ID3D12InfoQueue* infoQueue) {
    const std::vector<std::string> errors = DrainErrors(infoQueue);
    Expect(what, errors.empty());
    for (const std::string& error : errors) {
        std::printf("    %s\n", error.c_str());
    }
}

// Two SRV tables: three descriptors at t0 and two at t3. The sizes differ so a
// commit that mixed up which destination range belongs to which table would
// put the second table at the wrong offset rather than merely reordering.
constexpr UINT kTableADescriptors = 3;
constexpr UINT kTableBDescriptors = 2;
constexpr UINT kTotalDescriptors = kTableADescriptors + kTableBDescriptors;

struct TwoTableSignature {
    D3D12_DESCRIPTOR_RANGE1 ranges[2] = {};
    D3D12_ROOT_PARAMETER1 parameters[2] = {};
    RootSignature signature;

    TwoTableSignature() {
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = kTableADescriptors;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[1].NumDescriptors = kTableBDescriptors;
        ranges[1].BaseShaderRegister = kTableADescriptors;
        ranges[1].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;

        for (int i = 0; i < 2; ++i) {
            parameters[i].ParameterType =
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            parameters[i].DescriptorTable.NumDescriptorRanges = 1;
            parameters[i].DescriptorTable.pDescriptorRanges = &ranges[i];
        }

        D3D12_ROOT_SIGNATURE_DESC1 desc = {};
        desc.NumParameters = 2;
        desc.pParameters = parameters;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        signature.SetRootSignatureDesc(desc, D3D_ROOT_SIGNATURE_VERSION_1_1);
    }
};

}  // namespace

int main() {
    // Unbuffered: this test can fault inside the runtime, and a buffered log
    // would lose the line saying where.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // A failed assert inside the descriptor code must abort, not open a modal
    // dialog: this runs headless and a dialog is an indefinite hang rather
    // than a failure.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);

    // The debug layer is the point of this test, so a device without it is a
    // failure rather than a fallback.
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
        debug->EnableDebugLayer();
    } else {
        std::printf("the D3D12 debug layer is not installed; "
                    "install the Graphics Tools optional feature\n");
        return 1;
    }

    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&device)))) {
        std::printf("no D3D12 device\n");
        return 1;
    }

    ComPtr<ID3D12InfoQueue> infoQueue;
    device->QueryInterface(IID_PPV_ARGS(&infoQueue));

    Context().Initialize(device.Get());

    // A command list, only so the binder and the commit have one. Nothing is
    // submitted: CopyDescriptors is a CPU-side operation on heap memory, and
    // the root table bindings are recorded by the test's own setter instead.
    ComPtr<ID3D12CommandAllocator> allocator;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              IID_PPV_ARGS(&allocator)))) {
        std::printf("no command allocator\n");
        return 1;
    }
    ComPtr<ID3D12GraphicsCommandList> commandList;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&commandList)))) {
        std::printf("no command list\n");
        return 1;
    }

    const UINT increment = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Source descriptors, in a CPU-visible heap so they can be staged. Null
    // SRVs with a fully specified description are legal descriptors and need no
    // resources behind them, which keeps this to descriptor plumbing only.
    constexpr UINT kSourceDescriptors = 8;
    D3D12_DESCRIPTOR_HEAP_DESC sourceDesc = {};
    sourceDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    sourceDesc.NumDescriptors = kSourceDescriptors;
    sourceDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ComPtr<ID3D12DescriptorHeap> sourceHeap;
    if (FAILED(device->CreateDescriptorHeap(&sourceDesc, IID_PPV_ARGS(&sourceHeap)))) {
        std::printf("no source descriptor heap\n");
        return 1;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    for (UINT i = 0; i < kSourceDescriptors; ++i) {
        const CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
            sourceHeap->GetCPUDescriptorHandleForHeapStart(), static_cast<INT>(i),
            increment);
        device->CreateShaderResourceView(nullptr, &srv, handle);
    }
    const auto sourceAt = [&](UINT index) {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(
            sourceHeap->GetCPUDescriptorHandleForHeapStart(),
            static_cast<INT>(index), increment);
    };

    ExpectNoDebugErrors("setup produced no debug errors", infoQueue.Get());

    TwoTableSignature twoTables;
    Expect("the root signature compiled",
           twoTables.signature.GetRootSignature() != nullptr);
    ExpectEqual("both tables are CBV_SRV_UAV tables",
                twoTables.signature.GetDescriptorTableBitMask(
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                0b11);

    HeapBinder binder;

    // --- two tables stale in one commit ----------------------------------
    std::printf("two tables in one commit\n");
    {
        // Exactly the total, so a second commit is forced onto a new heap --
        // used by the heap-switch case below.
        DynamicDescriptorHeap heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                   kTotalDescriptors);
        binder.Reset(commandList.Get());
        heap.ParseRootSignature(twoTables.signature);

        // Table A's three descriptors staged individually and deliberately out
        // of order in the source heap, so none of them coalesce. Table B's two
        // are staged as one contiguous range, which must coalesce into a single
        // source range.
        heap.StageDescriptors(0, 0, 1, sourceAt(5));
        heap.StageDescriptors(0, 1, 1, sourceAt(0));
        heap.StageDescriptors(0, 2, 1, sourceAt(3));
        heap.StageDescriptors(1, 0, kTableBDescriptors, sourceAt(6));

        g_bound.clear();
        heap.CommitStagedDescriptors(commandList.Get(), binder, &RecordBinding);

        ExpectEqual("both tables were bound", g_bound.size(), 2);
        ExpectNoDebugErrors("the batched copy is well-formed", infoQueue.Get());

        if (g_bound.size() == 2) {
            ExpectEqual("first binding is root 0", g_bound[0].rootIndex, 0);
            ExpectEqual("second binding is root 1", g_bound[1].rootIndex, 1);
            // The second table must start exactly where the first one ends.
            // Getting this wrong is the batch's characteristic failure: both
            // tables are bound, both copies happen, and the second table reads
            // the first one's descriptors.
            ExpectEqual("table B starts one table past table A",
                        g_bound[1].handle.ptr - g_bound[0].handle.ptr,
                        static_cast<uint64_t>(kTableADescriptors) * increment);
        }

        // --- a second commit with nothing stale --------------------------
        g_bound.clear();
        heap.CommitStagedDescriptors(commandList.Get(), binder, &RecordBinding);
        ExpectEqual("committing again with nothing stale binds nothing",
                    g_bound.size(), 0);

        // --- re-staging one table restages only that one -----------------
        heap.StageDescriptors(1, 0, kTableBDescriptors, sourceAt(1));
        g_bound.clear();
        heap.CommitStagedDescriptors(commandList.Get(), binder, &RecordBinding);
        // The heap held exactly kTotalDescriptors and the first commit used all
        // of them, so this cannot fit and must switch heaps -- which makes
        // BOTH tables stale again even though only one was restaged.
        ExpectEqual("a heap switch rebinds every staged table", g_bound.size(), 2);
        ExpectNoDebugErrors("the copy after a heap switch is well-formed",
                            infoQueue.Get());
        if (g_bound.size() == 2) {
            ExpectEqual("table B still follows table A on the new heap",
                        g_bound[1].handle.ptr - g_bound[0].handle.ptr,
                        static_cast<uint64_t>(kTableADescriptors) * increment);
        }
    }

    // --- a table nothing ever staged into --------------------------------
    //
    // A heap switch marks every table in the root signature stale. A table with
    // no staged handles has nothing to copy from, and copying from the zeroed
    // handle cache would hand the driver null source descriptors -- which the
    // debug layer reports and which is a device-removal in a real frame, not a
    // missing texture.
    std::printf("a table nothing staged into\n");
    {
        // Sized to the root signature's total, which ParseRootSignature
        // requires, so the switch below has to be forced by filling it rather
        // than by starting small: three descriptors committed twice is six,
        // which does not fit in five.
        DynamicDescriptorHeap heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                   kTotalDescriptors);
        binder.Reset(commandList.Get());
        heap.ParseRootSignature(twoTables.signature);

        heap.StageDescriptors(0, 0, kTableADescriptors, sourceAt(0));
        g_bound.clear();
        heap.CommitStagedDescriptors(commandList.Get(), binder, &RecordBinding);
        ExpectEqual("only the staged table is bound", g_bound.size(), 1);

        // Restaging the same table needs three more descriptors and only two
        // are left, so this commit switches heaps. Table B is still untouched.
        heap.StageDescriptors(0, 0, kTableADescriptors, sourceAt(2));
        g_bound.clear();
        heap.CommitStagedDescriptors(commandList.Get(), binder, &RecordBinding);
        ExpectEqual("a heap switch does not copy the unstaged table",
                    g_bound.size(), 1);
        ExpectNoDebugErrors("no null source descriptors were copied",
                            infoQueue.Get());
    }

    Context().Shutdown();

    if (g_failures != 0) {
        std::printf("\n%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
