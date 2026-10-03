#include "pch.h"

#include "Descriptors/DescriptorContext.hpp"

#include <cassert>

namespace NeuralModelIntegrateTestbed::Descriptors {

void DescriptorContext::Initialize(ID3D12Device* device) {
    assert(device != nullptr);
    m_device = device;
    // Cached because DescriptorAllocatorPage and DynamicDescriptorHeap ask for
    // these in their constructors, and the value never changes for a device.
    for (int i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i) {
        m_handleIncrementSizes[i] = device->GetDescriptorHandleIncrementSize(
            static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(i));
    }
}

void DescriptorContext::Shutdown() {
    m_device = nullptr;
    // Deliberately not resetting m_frameCount: descriptors retired before a
    // device reset must not look newer than ones retired after it.
}

uint32_t DescriptorContext::GetDescriptorHandleIncrementSize(
    D3D12_DESCRIPTOR_HEAP_TYPE type) const {
    assert(type >= 0 && type < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES);
    return m_handleIncrementSizes[type];
}

DescriptorContext& Context() {
    static DescriptorContext s_context;
    return s_context;
}

}  // namespace NeuralModelIntegrateTestbed::Descriptors
