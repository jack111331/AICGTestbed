#pragma once

/*
 *  Copyright(c) 2018 Jeremiah van Oosten
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files(the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions :
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 *  IN THE SOFTWARE.
 */

// ---------------------------------------------------------------------------
// Adapted for ptflio from LearningDirectX12 v0.0.3 (MIT, (c) 2018 Jeremiah van
// Oosten -- see Descriptors/LICENSE-LearningDirectX12.txt). The algorithm is
// unchanged; only the couplings to the DX12Lib framework were replaced:
//
//   Application::Get().GetDevice()  ->  Descriptors::Context().GetDevice()
//   Application::GetFrameCount()    ->  Descriptors::Context().GetFrameCount()
//   CommandList&                    ->  ID3D12GraphicsCommandList* + HeapBinder
//
// and everything was moved into the NeuralModelIntegrateTestbed::Descriptors
// namespace.
// ---------------------------------------------------------------------------

/**
 *  @file DynamicDescriptorHeap.h
 *  @date October 22, 2018
 *  @author Jeremiah van Oosten
 *
 *  @brief The DynamicDescriptorHeap is a GPU visible descriptor heap that allows for
 *  staging of CPU visible descriptors that need to be uploaded before a Draw
 *  or Dispatch command is executed.
 *  The DynamicDescriptorHeap class is based on the one provided by the MiniEngine:
 *  https://github.com/Microsoft/DirectX-Graphics-Samples
 */

#include <directx/d3dx12.h>

#include <wrl.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>

namespace NeuralModelIntegrateTestbed::Descriptors {

class HeapBinder;
class RootSignature;

class DynamicDescriptorHeap
{
public:
    DynamicDescriptorHeap(
        D3D12_DESCRIPTOR_HEAP_TYPE heapType,
        uint32_t numDescriptorsPerHeap = 1024,
        // Frames a heap must sit out after use. The default matches
        // Application.cc's kInFlightMargin; raise it alongside the back buffer
        // count.
        uint64_t framesInFlight = 3);

    virtual ~DynamicDescriptorHeap();

    /**
     * Stages a contiguous range of CPU visible descriptors.
     * Descriptors are not copied to the GPU visible descriptor heap until
     * the CommitStagedDescriptors function is called.
     */
    void StageDescriptors(uint32_t rootParameterIndex, uint32_t offset, uint32_t numDescriptors, const D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptors);

    /**
     * Copy all of the staged descriptors to the GPU visible descriptor heap and
     * bind the descriptor heap and the descriptor tables to the command list.
     * The passed-in function object is used to set the GPU visible descriptors
     * on the command list. Two possible functions are:
     *   * Before a draw    : ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable
     *   * Before a dispatch: ID3D12GraphicsCommandList::SetComputeRootDescriptorTable
     * 
     * Since the DynamicDescriptorHeap can't know which function will be used, it must
     * be passed as an argument to the function.
     */
    void CommitStagedDescriptors( ID3D12GraphicsCommandList* commandList, HeapBinder& binder, std::function<void(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE)> setFunc );
    void CommitStagedDescriptorsForDraw(ID3D12GraphicsCommandList* commandList, HeapBinder& binder);
    void CommitStagedDescriptorsForDispatch(ID3D12GraphicsCommandList* commandList, HeapBinder& binder);

    /**
     * Copies a single CPU visible descriptor to a GPU visible descriptor heap.
     * This is useful for the
     *   * ID3D12GraphicsCommandList::ClearUnorderedAccessViewFloat
     *   * ID3D12GraphicsCommandList::ClearUnorderedAccessViewUint
     * methods which require both a CPU and GPU visible descriptors for a UAV 
     * resource.
     * 
     * @param commandList The command list is required in case the GPU visible
     * descriptor heap needs to be updated on the command list.
     * @param cpuDescriptor The CPU descriptor to copy into a GPU visible 
     * descriptor heap.
     * 
     * @return The GPU visible descriptor.
     */
    D3D12_GPU_DESCRIPTOR_HANDLE CopyDescriptor( ID3D12GraphicsCommandList* commandList, HeapBinder& binder, D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptor);

    /**
     * Parse the root signature to determine which root parameters contain
     * descriptor tables and determine the number of descriptors needed for
     * each table.
     */
    void ParseRootSignature( const RootSignature& rootSignature);

    /**
     * Reset used descriptors. This should only be done if any descriptors
     * that are being referenced by a command list has finished executing on the 
     * command queue.
     */
    // Call once per frame, in step with the command list this heap records
    // into. Retires the heaps this frame used rather than making them
    // immediately available again: a submitted command list keeps referencing
    // its descriptor heap until the GPU has finished with it, and overwriting
    // one in flight shows up as a single frame drawn with another frame's
    // descriptors -- the wrong textures on the wrong geometry.
    //
    // Reclaims heaps retired at least m_FramesInFlight frames ago, using
    // Descriptors::Context()'s frame counter, so this must be advanced once per
    // frame (Application.cc does it after Present).
    void Reset();

protected:

private:
    // Request a descriptor heap if one is available.
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> RequestDescriptorHeap();
    // Create a new descriptor heap of no descriptor heap is available.
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap();

    // Compute the number of stale descriptors that need to be copied
    // to GPU visible descriptor heap.
    uint32_t ComputeStaleDescriptorCount() const;

    /**
     * The maximum number of descriptor tables per root signature.
     * A 32-bit mask is used to keep track of the root parameter indices that
     * are descriptor tables.
     */
    static const uint32_t MaxDescriptorTables = 32;

    /**
     * A structure that represents a descriptor table entry in the root signature.
     */
    struct DescriptorTableCache
    {
        DescriptorTableCache()
            : NumDescriptors(0)
            , BaseDescriptor(nullptr)
        {}

        // Reset the table cache.
        void Reset()
        {
            NumDescriptors = 0;
            BaseDescriptor = nullptr;
        }

        // The number of descriptors in this descriptor table.
        uint32_t NumDescriptors;
        // The pointer to the descriptor in the descriptor handle cache.
        D3D12_CPU_DESCRIPTOR_HANDLE* BaseDescriptor;
    };

    // Describes the type of descriptors that can be staged using this 
    // dynamic descriptor heap.
    // Valid values are:
    //   * D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
    //   * D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
    // This parameter also determines the type of GPU visible descriptor heap to 
    // create.
    D3D12_DESCRIPTOR_HEAP_TYPE m_DescriptorHeapType;

    // The number of descriptors to allocate in new GPU visible descriptor heaps.
    uint32_t m_NumDescriptorsPerHeap;

    // The increment size of a descriptor.
    uint32_t m_DescriptorHandleIncrementSize;

    // The descriptor handle cache.
    std::unique_ptr<D3D12_CPU_DESCRIPTOR_HANDLE[]> m_DescriptorHandleCache;

    // Descriptor handle cache per descriptor table.
    DescriptorTableCache m_DescriptorTableCache[MaxDescriptorTables];

    // Each bit in the bit mask represents the index in the root signature
    // that contains a descriptor table.
    uint32_t m_DescriptorTableBitMask;
    // Each bit set in the bit mask represents a descriptor table
    // in the root signature that has changed since the last time the 
    // descriptors were copied.
    uint32_t m_StaleDescriptorTableBitMask;
    // Each bit set represents a descriptor table that has had descriptors
    // staged into it at least once since the root signature was parsed, and so
    // has source handles worth copying. Only consulted when a heap switch
    // forces every table to be recopied.
    uint32_t m_StagedDescriptorTableBitMask;

    // Flattened source ranges for CommitStagedDescriptors' one CopyDescriptors
    // call, sized by ParseRootSignature to the total across all tables.
    std::unique_ptr<D3D12_CPU_DESCRIPTOR_HANDLE[]> m_SrcRangeStarts;
    std::unique_ptr<UINT[]> m_SrcRangeSizes;
    uint32_t m_ScratchRangeCapacity = 0;

    using DescriptorHeapPool = std::queue< Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> >;

    // Every heap this object has ever created; owns them.
    DescriptorHeapPool m_DescriptorHeapPool;
    // Heaps the GPU is known to be finished with.
    DescriptorHeapPool m_AvailableDescriptorHeaps;

    // Heaps handed out since the last Reset, and heaps retired by earlier
    // frames with the frame number that retired them.
    //
    // Upstream's Reset() did `m_AvailableDescriptorHeaps = m_DescriptorHeapPool`,
    // handing every heap straight back with no synchronisation at all. With two
    // back buffers the swap chain only waits for the frame that last used the
    // NEXT back buffer index, so frame N+1 can still be executing while frame
    // N+2 records -- and frame N+2 would then copy new descriptors into the
    // very heap frame N+1's submitted command list is still sampling from.
    //
    // That is invisible as long as consecutive frames stage identical
    // descriptors, which is why it survived until frustum culling started
    // changing the drawn set as the camera moves. See the note on Reset().
    DescriptorHeapPool m_UsedThisFrame;
    std::queue< std::pair<uint64_t, Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> > >
        m_RetiredDescriptorHeaps;

    // How many frames a retired heap has to sit out before it is handed out
    // again. Back buffer count + 1, matching kInFlightMargin in Application.cc,
    // which gates the CPU descriptor allocator the same way.
    uint64_t m_FramesInFlight;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_CurrentDescriptorHeap;
    CD3DX12_GPU_DESCRIPTOR_HANDLE m_CurrentGPUDescriptorHandle;
    CD3DX12_CPU_DESCRIPTOR_HANDLE m_CurrentCPUDescriptorHandle;

    uint32_t m_NumFreeHandles;
    
};
}  // namespace NeuralModelIntegrateTestbed::Descriptors
