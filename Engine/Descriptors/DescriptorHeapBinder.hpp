#pragma once

// Replaces the slice of DX12Lib's CommandList that the ported
// DynamicDescriptorHeap depends on.
//
// D3D12 takes every shader-visible heap in a single SetDescriptorHeaps call, and
// each call replaces the whole binding -- so a DynamicDescriptorHeap that
// switched to a fresh CBV_SRV_UAV heap mid-frame cannot just bind its own heap
// without unbinding the sampler heap. DX12Lib solves this by having CommandList
// remember one heap per type and rebind them together; this does the same for a
// raw ID3D12GraphicsCommandList.
//
// It also lets the application register heaps it owns itself (the DirectXTK12
// CommonStates sampler heap, say) so they survive a dynamic-heap switch.

#include <directx/d3d12.h>

namespace NeuralModelIntegrateTestbed::Descriptors {

class HeapBinder {
public:
    // Point the binder at the command list for this frame and forget all
    // previously bound heaps. Call once after resetting the command list,
    // because descriptor heap bindings do not survive a command list reset.
    void Reset(ID3D12GraphicsCommandList* commandList);

    // Bind `heap` for `type`, replacing whatever was bound for that type.
    // Re-issues SetDescriptorHeaps only when the set actually changes, so
    // calling this every draw is cheap. Only CBV_SRV_UAV and SAMPLER are valid;
    // RTV and DSV heaps are not shader-visible and must never be passed here.
    void SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                           ID3D12DescriptorHeap* heap);

    ID3D12DescriptorHeap* GetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type) const;

    ID3D12GraphicsCommandList* GetCommandList() const { return m_commandList; }

private:
    void BindDescriptorHeaps();

    ID3D12GraphicsCommandList* m_commandList = nullptr;
    ID3D12DescriptorHeap* m_heaps[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] = {};
};

}  // namespace NeuralModelIntegrateTestbed::Descriptors
