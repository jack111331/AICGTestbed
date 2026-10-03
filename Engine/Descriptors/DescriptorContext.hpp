#pragma once

// Stands in for DX12Lib's Application singleton, which the ported descriptor
// classes reach for whenever they need the device or the current frame number.
//
// Rather than thread an ID3D12Device* and a frame counter through every
// constructor (DescriptorAllocation's destructor needs the frame number, and it
// has no other way to reach one), this keeps the upstream shape: one
// process-wide context, initialised once at device creation.

#include <directx/d3d12.h>

#include <cstdint>

namespace NeuralModelIntegrateTestbed::Descriptors {

class DescriptorContext {
public:
    // Called once, after the D3D12 device exists and again after a device reset.
    void Initialize(ID3D12Device* device);
    void Shutdown();

    ID3D12Device* GetDevice() const { return m_device; }
    bool IsInitialized() const { return m_device != nullptr; }

    uint32_t GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE type) const;

    // Monotonically increasing. Descriptors freed during frame N are only
    // recycled once the caller has confirmed frame N is off the GPU, so this
    // must be advanced exactly once per frame -- see AdvanceFrame().
    uint64_t GetFrameCount() const { return m_frameCount; }
    void AdvanceFrame() { ++m_frameCount; }

private:
    ID3D12Device* m_device = nullptr;
    uint32_t m_handleIncrementSizes[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] = {};
    uint64_t m_frameCount = 1;
};

// The single context the ported classes use.
DescriptorContext& Context();

}  // namespace NeuralModelIntegrateTestbed::Descriptors
