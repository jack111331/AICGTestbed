#include "pch.h"

#include "Descriptors/DynamicDescriptorHeap.hpp"

#include <cstdio>

#include "Descriptors/DescriptorContext.hpp"
#include "Descriptors/DescriptorHeapBinder.hpp"
#include "Descriptors/RootSignature.hpp"

namespace NeuralModelIntegrateTestbed::Descriptors {

DynamicDescriptorHeap::DynamicDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE heapType, uint32_t numDescriptorsPerHeap, uint64_t framesInFlight)
    : m_DescriptorHeapType(heapType)
    , m_NumDescriptorsPerHeap(numDescriptorsPerHeap)
    , m_FramesInFlight(framesInFlight)
    , m_DescriptorTableBitMask(0)
    , m_StaleDescriptorTableBitMask(0)
    , m_StagedDescriptorTableBitMask(0)
    , m_CurrentCPUDescriptorHandle(D3D12_DEFAULT)
    , m_CurrentGPUDescriptorHandle(D3D12_DEFAULT)
    , m_NumFreeHandles(0)
{
    m_DescriptorHandleIncrementSize = Context().GetDescriptorHandleIncrementSize(heapType);

    // Allocate space for staging CPU visible descriptors.
    m_DescriptorHandleCache = std::make_unique<D3D12_CPU_DESCRIPTOR_HANDLE[]>(m_NumDescriptorsPerHeap);
}

DynamicDescriptorHeap::~DynamicDescriptorHeap()
{}

void DynamicDescriptorHeap::ParseRootSignature(const RootSignature& rootSignature)
{
    // If the root signature changes, all descriptors must be (re)bound to the
    // command list.
    m_StaleDescriptorTableBitMask = 0;
    // The table layout is about to be rebuilt, so which tables hold staged
    // handles is no longer a meaningful question.
    m_StagedDescriptorTableBitMask = 0;

    const auto& rootSignatureDesc = rootSignature.GetRootSignatureDesc();

    // Get a bit mask that represents the root parameter indices that match the 
    // descriptor heap type for this dynamic descriptor heap.
    m_DescriptorTableBitMask = rootSignature.GetDescriptorTableBitMask(m_DescriptorHeapType);
    uint32_t descriptorTableBitMask = m_DescriptorTableBitMask;

    uint32_t currentOffset = 0;
    DWORD rootIndex;
    while (_BitScanForward(&rootIndex, descriptorTableBitMask) && rootIndex < rootSignatureDesc.NumParameters)
    {
        uint32_t numDescriptors = rootSignature.GetNumDescriptors(rootIndex);

        DescriptorTableCache& descriptorTableCache = m_DescriptorTableCache[rootIndex];
        descriptorTableCache.NumDescriptors = numDescriptors;
        descriptorTableCache.BaseDescriptor = m_DescriptorHandleCache.get() + currentOffset;

        currentOffset += numDescriptors;

        // Flip the descriptor table bit so it's not scanned again for the current index.
        descriptorTableBitMask ^= (1 << rootIndex);
    }

    // Scratch for CommitStagedDescriptors' single CopyDescriptors call. One
    // source range per descriptor is the worst case -- when none of them
    // coalesce -- so the total descriptor count across every table is enough,
    // and it is usually a handful rather than the whole heap.
    if (currentOffset > m_ScratchRangeCapacity)
    {
        m_SrcRangeStarts = std::make_unique<D3D12_CPU_DESCRIPTOR_HANDLE[]>(currentOffset);
        m_SrcRangeSizes = std::make_unique<UINT[]>(currentOffset);
        m_ScratchRangeCapacity = currentOffset;
    }

    // Make sure the maximum number of descriptors per descriptor heap has not been exceeded.
    assert(currentOffset <= m_NumDescriptorsPerHeap && "The root signature requires more than the maximum number of descriptors per descriptor heap. Consider increasing the maximum number of descriptors per descriptor heap.");
}

void DynamicDescriptorHeap::StageDescriptors(uint32_t rootParameterIndex, uint32_t offset, uint32_t numDescriptors, const D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptor)
{
    // Cannot stage more than the maximum number of descriptors per heap.
    // Cannot stage more than MaxDescriptorTables root parameters.
    if (numDescriptors > m_NumDescriptorsPerHeap || rootParameterIndex >= MaxDescriptorTables )
    {
        throw std::bad_alloc();
    }

    DescriptorTableCache& descriptorTableCache = m_DescriptorTableCache[rootParameterIndex];

    // Check that the number of descriptors to copy does not exceed the number
    // of descriptors expected in the descriptor table.
    if ( (offset + numDescriptors) > descriptorTableCache.NumDescriptors)
    {
        throw std::length_error("Number of descriptors exceeds the number of descriptors in the descriptor table.");
    }

    D3D12_CPU_DESCRIPTOR_HANDLE* dstDescriptor = (descriptorTableCache.BaseDescriptor + offset);
    for (uint32_t i = 0; i < numDescriptors; ++i)
    {
        dstDescriptor[i] = CD3DX12_CPU_DESCRIPTOR_HANDLE(srcDescriptor, i, m_DescriptorHandleIncrementSize);
    }

    // Set the root parameter index bit to make sure the descriptor table 
    // at that index is bound to the command list.
    m_StaleDescriptorTableBitMask |= (1 << rootParameterIndex);
    // Remembered for the lifetime of the root signature, not just until the
    // next commit: a heap switch has to recopy every table that has source
    // handles behind it, and this is how CommitStagedDescriptors tells those
    // apart from a table nothing has filled in yet.
    m_StagedDescriptorTableBitMask |= (1 << rootParameterIndex);
}

uint32_t DynamicDescriptorHeap::ComputeStaleDescriptorCount() const
{
    uint32_t numStaleDescriptors = 0;
    DWORD i;
    DWORD staleDescriptorsBitMask = m_StaleDescriptorTableBitMask;

    while ( _BitScanForward( &i, staleDescriptorsBitMask ) )
    {
        numStaleDescriptors += m_DescriptorTableCache[i].NumDescriptors;
        staleDescriptorsBitMask ^= ( 1 << i );
    }

    return numStaleDescriptors;
}

Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> DynamicDescriptorHeap::RequestDescriptorHeap()
{
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    if (!m_AvailableDescriptorHeaps.empty())
    {
        descriptorHeap = m_AvailableDescriptorHeaps.front();
        m_AvailableDescriptorHeaps.pop();
    }
    else
    {
        descriptorHeap = CreateDescriptorHeap();
        m_DescriptorHeapPool.push(descriptorHeap);
    }

    // Remembered so Reset can retire it against this frame rather than handing
    // it straight back while the GPU may still be reading it.
    m_UsedThisFrame.push(descriptorHeap);

    return descriptorHeap;
}

Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> DynamicDescriptorHeap::CreateDescriptorHeap()
{
    auto device = Context().GetDevice();

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = m_DescriptorHeapType;
    descriptorHeapDesc.NumDescriptors = m_NumDescriptorsPerHeap;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    DX::ThrowIfFailed(device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&descriptorHeap)));

    return descriptorHeap;
}

void DynamicDescriptorHeap::CommitStagedDescriptors(ID3D12GraphicsCommandList* commandList, HeapBinder& binder, std::function<void(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE)> setFunc)
{
    // Compute the number of descriptors that need to be copied
    uint32_t numDescriptorsToCommit = ComputeStaleDescriptorCount();

    if ( numDescriptorsToCommit == 0 )
    {
        return;
    }

    auto device = Context().GetDevice();
    assert(commandList != nullptr);

    if ( !m_CurrentDescriptorHeap || m_NumFreeHandles < numDescriptorsToCommit )
    {
        m_CurrentDescriptorHeap = RequestDescriptorHeap();
        m_CurrentCPUDescriptorHandle = m_CurrentDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        m_CurrentGPUDescriptorHandle = m_CurrentDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        m_NumFreeHandles = m_NumDescriptorsPerHeap;

        binder.SetDescriptorHeap(m_DescriptorHeapType, m_CurrentDescriptorHeap.Get());

        // When updating the descriptor heap on the command list, all descriptor
        // tables must be (re)recopied to the new descriptor heap (not just
        // the stale descriptor tables).
        //
        // Intersected with the tables something has actually staged into. A
        // table that is in the root signature but has never been staged has no
        // source handles behind it -- m_DescriptorHandleCache is zeroed -- and
        // copying from null handles is a device-removal, not a missing texture.
        // Upstream copies unconditionally and gets away with it only because
        // every caller happens to stage every table before committing.
        m_StaleDescriptorTableBitMask = m_DescriptorTableBitMask & m_StagedDescriptorTableBitMask;

        // Recounted because the mask just grew: the figure above only covered
        // the tables that were stale before the heap switch.
        numDescriptorsToCommit = ComputeStaleDescriptorCount();
        if ( numDescriptorsToCommit == 0 )
        {
            return;
        }
    }

    // One CopyDescriptors call for every stale table at once, rather than one
    // call per table.
    //
    // CopyDescriptors flattens both sides into a single stream and pairs them
    // off descriptor by descriptor -- the Nth source descriptor goes to the Nth
    // destination descriptor, regardless of where the range boundaries fall.
    // Only the two totals have to agree. That is what makes the batch legal,
    // and it is also why a source run may be coalesced ACROSS a table boundary
    // below without disturbing which descriptor lands where.
    //
    // Destinations are one range per table. They happen to be contiguous in
    // the heap, so they could be expressed as a single range, but a range per
    // table costs nothing and keeps this correct if the allocation strategy
    // ever stops being a bump pointer.
    D3D12_CPU_DESCRIPTOR_HANDLE destRangeStarts[MaxDescriptorTables];
    UINT destRangeSizes[MaxDescriptorTables];
    UINT numDestRanges = 0;

    // The root tables to bind, collected so they can be bound after the copy
    // instead of between copies. CopyDescriptors writes heap memory on the CPU
    // immediately while the command list only records a GPU handle, so the two
    // are independent and the order does not matter.
    UINT boundRootIndex[MaxDescriptorTables];
    D3D12_GPU_DESCRIPTOR_HANDLE boundGpuHandle[MaxDescriptorTables];

    UINT numSrcRanges = 0;
    assert(m_SrcRangeStarts && m_SrcRangeSizes &&
           "ParseRootSignature must run before descriptors are committed");

    DWORD rootIndex;
    uint32_t staleDescriptorTableBitMask = m_StaleDescriptorTableBitMask;
    // Scan from LSB to MSB for a bit set in staleDescriptorsBitMask
    while ( _BitScanForward( &rootIndex, staleDescriptorTableBitMask ) )
    {
        staleDescriptorTableBitMask ^= (1 << rootIndex);

        const UINT numSrcDescriptors = m_DescriptorTableCache[rootIndex].NumDescriptors;
        const D3D12_CPU_DESCRIPTOR_HANDLE* pSrcDescriptorHandles = m_DescriptorTableCache[rootIndex].BaseDescriptor;
        if ( numSrcDescriptors == 0 )
        {
            continue;
        }

        destRangeStarts[numDestRanges] = m_CurrentCPUDescriptorHandle;
        destRangeSizes[numDestRanges] = numSrcDescriptors;
        boundRootIndex[numDestRanges] = rootIndex;
        boundGpuHandle[numDestRanges] = m_CurrentGPUDescriptorHandle;
        ++numDestRanges;

        // Sources are staged as individual handles, so the obvious encoding is
        // one single-descriptor range each. Handles that continue the previous
        // one in their own heap are folded into one longer range instead,
        // which is the only way to hand the driver a block copy rather than a
        // walk of one-descriptor ranges. Worth doing because it is nearly free;
        // how much it actually folds depends entirely on where the source
        // descriptors were allocated.
        for ( UINT i = 0; i < numSrcDescriptors; ++i )
        {
            if ( numSrcRanges != 0 )
            {
                const SIZE_T runEnd = m_SrcRangeStarts[numSrcRanges - 1].ptr +
                    static_cast<SIZE_T>(m_SrcRangeSizes[numSrcRanges - 1]) * m_DescriptorHandleIncrementSize;
                if ( pSrcDescriptorHandles[i].ptr == runEnd )
                {
                    ++m_SrcRangeSizes[numSrcRanges - 1];
                    continue;
                }
            }
            m_SrcRangeStarts[numSrcRanges] = pSrcDescriptorHandles[i];
            m_SrcRangeSizes[numSrcRanges] = 1;
            ++numSrcRanges;
        }

        // Offset current CPU and GPU descriptor handles.
        m_CurrentCPUDescriptorHandle.Offset(numSrcDescriptors, m_DescriptorHandleIncrementSize);
        m_CurrentGPUDescriptorHandle.Offset(numSrcDescriptors, m_DescriptorHandleIncrementSize);
    }

    m_NumFreeHandles -= numDescriptorsToCommit;

    // Copy the staged CPU visible descriptors to the GPU visible descriptor heap.
    device->CopyDescriptors(numDestRanges, destRangeStarts, destRangeSizes,
        numSrcRanges, m_SrcRangeStarts.get(), m_SrcRangeSizes.get(), m_DescriptorHeapType);

    // Set the descriptors on the command list using the passed-in setter function.
    for ( UINT i = 0; i < numDestRanges; ++i )
    {
        setFunc(commandList, boundRootIndex[i], boundGpuHandle[i]);
    }

    // Clear the stale bits so the descriptor tables are not recopied again unless they are updated with a new descriptor.
    m_StaleDescriptorTableBitMask = 0;
}

void DynamicDescriptorHeap::CommitStagedDescriptorsForDraw(ID3D12GraphicsCommandList* commandList, HeapBinder& binder)
{
    CommitStagedDescriptors(commandList, binder, &ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable);
}

void DynamicDescriptorHeap::CommitStagedDescriptorsForDispatch(ID3D12GraphicsCommandList* commandList, HeapBinder& binder)
{
    CommitStagedDescriptors(commandList, binder, &ID3D12GraphicsCommandList::SetComputeRootDescriptorTable);
}

D3D12_GPU_DESCRIPTOR_HANDLE DynamicDescriptorHeap::CopyDescriptor(ID3D12GraphicsCommandList* commandList, HeapBinder& binder, D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptor)
{
    if (!m_CurrentDescriptorHeap || m_NumFreeHandles < 1)
    {
        m_CurrentDescriptorHeap = RequestDescriptorHeap();
        m_CurrentCPUDescriptorHandle = m_CurrentDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        m_CurrentGPUDescriptorHandle = m_CurrentDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
        m_NumFreeHandles = m_NumDescriptorsPerHeap;

        binder.SetDescriptorHeap(m_DescriptorHeapType, m_CurrentDescriptorHeap.Get());

        // When updating the descriptor heap on the command list, all descriptor
        // tables must be (re)recopied to the new descriptor heap (not just
        // the stale descriptor tables).
        m_StaleDescriptorTableBitMask = m_DescriptorTableBitMask;
    }

    auto device = Context().GetDevice();

    D3D12_GPU_DESCRIPTOR_HANDLE hGPU = m_CurrentGPUDescriptorHandle;
    device->CopyDescriptorsSimple(1, m_CurrentCPUDescriptorHandle, cpuDescriptor, m_DescriptorHeapType);

    m_CurrentCPUDescriptorHandle.Offset(1, m_DescriptorHandleIncrementSize);
    m_CurrentGPUDescriptorHandle.Offset(1, m_DescriptorHandleIncrementSize);
    m_NumFreeHandles -= 1;

    return hGPU;
}

void DynamicDescriptorHeap::Reset()
{
    // Reclaim anything retired long enough ago that the GPU cannot still be
    // reading it, then retire what this frame used. Upstream assigned the whole
    // pool straight back to the available queue here, which is the bug this
    // replaces: a heap still referenced by a submitted command list was handed
    // out again and overwritten.
    const uint64_t frame = Context().GetFrameCount();
    while (!m_RetiredDescriptorHeaps.empty() &&
           m_RetiredDescriptorHeaps.front().first + m_FramesInFlight <= frame)
    {
        m_AvailableDescriptorHeaps.push(m_RetiredDescriptorHeaps.front().second);
        m_RetiredDescriptorHeaps.pop();
    }
    while (!m_UsedThisFrame.empty())
    {
        m_RetiredDescriptorHeaps.emplace(frame, m_UsedThisFrame.front());
        m_UsedThisFrame.pop();
    }

    m_CurrentDescriptorHeap.Reset();
    m_CurrentCPUDescriptorHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(D3D12_DEFAULT);
    m_CurrentGPUDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(D3D12_DEFAULT);
    m_NumFreeHandles = 0;
    m_DescriptorTableBitMask = 0;
    m_StaleDescriptorTableBitMask = 0;
    // The table cache below is cleared too, so nothing has staged handles any
    // more. The scratch arrays are kept: their size follows the root
    // signature, not the frame.
    m_StagedDescriptorTableBitMask = 0;

    // Reset the table cache
    for (int i = 0; i < MaxDescriptorTables; ++i)
    {
        m_DescriptorTableCache[i].Reset();
    }
}

}  // namespace NeuralModelIntegrateTestbed::Descriptors
