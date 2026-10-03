#include "pch.h"

#include "Descriptors/DescriptorHeapBinder.hpp"

#include <cassert>

namespace NeuralModelIntegrateTestbed::Descriptors {

namespace {

bool IsShaderVisibleType(D3D12_DESCRIPTOR_HEAP_TYPE type) {
    return type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ||
           type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
}

}  // namespace

void HeapBinder::Reset(ID3D12GraphicsCommandList* commandList) {
    m_commandList = commandList;
    for (auto& heap : m_heaps) {
        heap = nullptr;
    }
}

void HeapBinder::SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type,
                                   ID3D12DescriptorHeap* heap) {
    // Upstream loops over all four heap types when rebinding; feeding it an RTV
    // or DSV heap would produce an invalid SetDescriptorHeaps call, so reject
    // those here rather than letting the debug layer catch it later.
    assert(IsShaderVisibleType(type) &&
           "only CBV_SRV_UAV and SAMPLER heaps are shader-visible");
    if (!IsShaderVisibleType(type)) {
        return;
    }

    if (m_heaps[type] != heap) {
        m_heaps[type] = heap;
        BindDescriptorHeaps();
    }
}

ID3D12DescriptorHeap* HeapBinder::GetDescriptorHeap(
    D3D12_DESCRIPTOR_HEAP_TYPE type) const {
    assert(type >= 0 && type < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES);
    return m_heaps[type];
}

void HeapBinder::BindDescriptorHeaps() {
    if (m_commandList == nullptr) {
        return;
    }

    UINT numHeaps = 0;
    ID3D12DescriptorHeap* heaps[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] = {};
    for (int i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i) {
        if (m_heaps[i] != nullptr) {
            heaps[numHeaps++] = m_heaps[i];
        }
    }

    if (numHeaps > 0) {
        m_commandList->SetDescriptorHeaps(numHeaps, heaps);
    }
}

}  // namespace NeuralModelIntegrateTestbed::Descriptors
