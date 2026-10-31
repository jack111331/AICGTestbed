#include "pch.h"
#include "FloodDiffusion.hpp"

#include "OrtEnv.hpp"
#include "Tokenizer.hpp"

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

// The guidance-and-Euler compute shader, for the GPU-resident sampler.
#include <latent_step.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

// TODO: If I want to optimize its execution speed, I need to first distill the denoiser to 1-2 steps denoiser, avoiding the 128 steps denoising, which may save about x64 time and can do it in 60 FPS
// The second one may be inspecting what layers can be computed using bfloat16 instead of float32

namespace NeuralModelIntegrateTestbed {

namespace {

double Milliseconds(std::chrono::steady_clock::time_point start) {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Ort::MemoryInfo CpuMemoryInfo() {
    return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
}

// Trims an ONNX Runtime message down to the part that identifies the problem.
// Its exceptions carry an absolute path into Microsoft's build machine plus a
// repeated copy of the status, none of which helps a caller.
std::string Shorten(const char* what) {
    std::string message = what == nullptr ? std::string() : std::string(what);
    const std::size_t cut = message.find(" Status Message:");
    if (cut != std::string::npos) {
        message = message.substr(0, cut);
    }
    if (message.size() > 300) {
        message.resize(300);
    }
    return message;
}

}  // namespace

const char* ToString(FloodDiffusionDevice device) {
    switch (device) {
        case FloodDiffusionDevice::Cpu: return "CPU";
        case FloodDiffusionDevice::DirectML: return "DirectML";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// FloodDiffusionConfig
// ---------------------------------------------------------------------------

bool FloodDiffusionConfig::Load(const std::filesystem::path& path, std::string* error) {
    const auto fail = [error](std::string message) {
        SetOrtError(error, std::move(message));
        return false;
    };

    std::ifstream stream(path);
    if (!stream) {
        return fail("cannot open " + path.string() +
                    "; run tools/prepare_flooddiffusion.py to produce it");
    }

    *this = FloodDiffusionConfig();

    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key == "chunk_size") {
            fields >> chunkSize;
        } else if (key == "noise_steps") {
            fields >> noiseSteps;
        } else if (key == "cfg_scale") {
            fields >> cfgScale;
        } else if (key == "prediction_type") {
            fields >> predictionType;
        } else if (key == "text_len") {
            fields >> textLen;
        } else if (key == "text_clean") {
            fields >> textClean;
        } else if (key == "latent_dim") {
            fields >> latentDim;
        } else if (key == "motion_dim") {
            fields >> motionDim;
        } else if (key == "text_dim") {
            fields >> textDim;
        } else if (key == "joints") {
            fields >> joints;
        } else if (key == "vae_num_caches") {
            fields >> vaeNumCaches;
        } else if (key == "vae_upsample_factor") {
            fields >> vaeUpsampleFactor;
        } else if (key == "vae_frames_first") {
            fields >> vaeFramesFirst;
        } else if (key == "vae_frames_step") {
            fields >> vaeFramesStep;
        } else if (key == "has_text_encoder") {
            int value = 0;
            fields >> value;
            hasTextEncoder = value != 0;
        } else if (key == "source_config") {
            fields >> sourceConfig;
        } else if (key == "vae_cache_shape") {
            std::string name;
            int d1 = 0;
            int d2 = 0;
            fields >> name >> d1 >> d2;
            vaeCacheShapes.emplace_back(d1, d2);
        }
        // Unknown keys are ignored on purpose: a newer preparation script may
        // record things this build does not read yet.
    }

    if (chunkSize <= 0 || noiseSteps <= 0 || latentDim <= 0 || motionDim <= 0 ||
        textDim <= 0 || joints <= 1 || vaeFramesStep <= 0) {
        return fail(path.string() + " is missing required keys; regenerate it "
                                    "with tools/prepare_flooddiffusion.py");
    }
    if (static_cast<int>(vaeCacheShapes.size()) != vaeNumCaches) {
        return fail(path.string() + " declares " + std::to_string(vaeNumCaches) +
                    " VAE caches but lists " +
                    std::to_string(vaeCacheShapes.size()) + " shapes");
    }
    if (predictionType != "vel" && predictionType != "x0" && predictionType != "noise") {
        return fail("prediction_type " + predictionType + " is not one of vel, x0, noise");
    }
    MotionFeatureLayout layout;
    if (!MotionFeatureLayout::FromMotionDim(motionDim, &layout) ||
        layout.joints != joints) {
        return fail("motion_dim " + std::to_string(motionDim) + " does not describe " +
                    std::to_string(joints) + " HumanML3D joints");
    }
    if (noiseSteps % chunkSize != 0) {
        // The reference asserts this: the ramp advances by chunk_size latents
        // per noise_steps steps, and a remainder would desynchronise them.
        return fail("noise_steps " + std::to_string(noiseSteps) +
                    " is not a multiple of chunk_size " + std::to_string(chunkSize));
    }
    return true;
}

// ---------------------------------------------------------------------------
// DiffusionForcingSchedule
// ---------------------------------------------------------------------------

DiffusionForcingSchedule::DiffusionForcingSchedule(int latentFrames, int chunkSize,
                                                   int denoiseSteps, bool continuous)
    : m_chunkSize(chunkSize), m_denoiseSteps(denoiseSteps), m_continuous(continuous) {
    if (latentFrames <= 0 || chunkSize <= 0 || denoiseSteps <= 0) {
        return;
    }
    m_generatedLatents = latentFrames;
    m_totalLatents = latentFrames + chunkSize;

    // Mirrors the reference exactly, including the arithmetic types. max_t and
    // dt are doubles and the step count truncates their quotient, which is not
    // always what rounding would give: 8 latents yields 2.4 / 0.1 =
    // 23.999999999999996 and so 23 steps, not 24.
    const double maxT = 1.0 + static_cast<double>(latentFrames - 1) /
                                  static_cast<double>(chunkSize);
    m_dt = 1.0 / static_cast<double>(denoiseSteps);
    m_totalSteps = static_cast<int>(maxT / m_dt);

    // base[i] = 1 + i / chunk, in float32, because the noise level derived from
    // it is a float32 model input.
    m_base.resize(static_cast<std::size_t>(m_totalLatents));
    for (int i = 0; i < m_totalLatents; ++i) {
        m_base[static_cast<std::size_t>(i)] =
            1.0f + static_cast<float>(i) / static_cast<float>(chunkSize);
    }
}

double DiffusionForcingSchedule::TimeAt(int step) const {
    return static_cast<double>(step) * m_dt;
}

int DiffusionForcingSchedule::StartAt(int step) const {
    if (!IsValid()) {
        return 0;
    }
    // Truncation toward zero is deliberate and matches Python's int(): for
    // t < 1 the product is negative and int(-4.5) is -4, not -5.
    const double t = TimeAt(step);
    const int start = static_cast<int>(static_cast<double>(m_chunkSize) * (t - 1.0)) + 1;
    return std::max(0, start);
}

int DiffusionForcingSchedule::EndAt(int step) const {
    if (!IsValid()) {
        return 0;
    }
    const double t = TimeAt(step);
    const int end = static_cast<int>(static_cast<double>(m_chunkSize) * t) + 1;
    // The clamp is what makes a FINITE schedule stop producing: the sequence
    // is m_totalLatents long and the ramp cannot run past its end. A
    // continuous one has no end to run past.
    return m_continuous ? end : std::min(end, m_totalLatents);
}

void DiffusionForcingSchedule::NoiseLevels(int step, std::vector<float>& out) const {
    WindowNoiseLevels(step, 0, std::max(EndAt(step), 0), out);
}

void DiffusionForcingSchedule::WindowNoiseLevels(int step, int origin, int count,
                                                 std::vector<float>& out) const {
    out.resize(static_cast<std::size_t>(std::max(count, 0)));
    if (count <= 0) {
        return;
    }
    // t and origin/chunk both grow with the stream and their difference does
    // not, so they are combined in DOUBLE and once, before anything is cast to
    // the float32 the model wants. m_base carries the per-slot part, which is
    // small and exact.
    const double rebased = TimeAt(step) - static_cast<double>(origin) /
                                              static_cast<double>(m_chunkSize);
    const float t = static_cast<float>(rebased);
    for (int i = 0; i < count; ++i) {
        // m_base is sized for a finite sequence; past its end the ramp
        // continues by the same rule, which is what a sliding window reads.
        const float base = i < static_cast<int>(m_base.size())
                               ? m_base[static_cast<std::size_t>(i)]
                               : 1.0f + static_cast<float>(i) /
                                            static_cast<float>(m_chunkSize);
        out[static_cast<std::size_t>(i)] = std::clamp(base - t, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// FloodDiffusionPipeline
// ---------------------------------------------------------------------------

// --- the GPU-resident sampler ----------------------------------------------
//
// Keeps every denoiser tensor in a D3D12 buffer, binds them once with
// Ort::IoBinding, and runs the guidance combine and the Euler step as a
// compute shader writing the latents in place. Per step the only thing that
// crosses the bus is the noise ramp, 520 bytes.
//
// There is no CPU synchronisation in the loop at all. The DirectML EP submits
// to the same compute queue this dispatch goes on, so queue order alone
// sequences denoise -> step -> denoise. The one place a wait is unavoidable is
// decoding, which needs four floats of a finalised latent on the host; that
// happens about 60 times a generation rather than 128.
//
// WHAT IT IS AND IS NOT WORTH is measured in FloodDiffusionOptions::
// gpuLatents, and the short version is that it does not make a generation
// faster: 508.7 ms against the host path's 506.9 at 60 latents. It makes the
// DENOISING six times cheaper in the accounting (318.9 -> 53.8 ms) by never
// waiting, and that time reappears in the decoder's readback, which does. The
// generation is bound by GPU execution rather than by copies.
//
// So read the structure here as the device-resident half of a pipeline whose
// other half -- the VAE decoder, 60 calls and most of the wall time -- is
// still on host tensors. That is where the next measurable win is.
struct GpuSampler {
    // One tensor living in a D3D12 buffer. Ownership order matters on
    // teardown: the Ort::Value points into the allocation, which points at the
    // buffer, so they are released in that order.
    struct Tensor {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
        void* allocation = nullptr;
        std::unique_ptr<Ort::Value> value;
        std::uint64_t bytes = 0;
    };

    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    const OrtDmlApi* dmlApi = nullptr;

    // What this class issues to the queue, counted rather than reasoned about.
    // `dispatches` is the latent-step compute shader; the execution
    // provider's own dispatches are not ours to count and come from an ORT
    // profile instead (one fused node per Run once shapes are pinned).
    int computeDispatches = 0;
    int submissions = 0;

    std::unique_ptr<Ort::MemoryInfo> memoryInfo;
    Tensor x;           // [rows, time, latentDim]  the latent state
    Tensor t;           // [rows, time]             the noise ramp
    Tensor context;     // [rows, 1, contextLen, textDim]
    Tensor lens;        // [rows, 1] int64
    Tensor textIdx;     // [rows, time] int64
    Tensor prediction;  // [rows, time, latentDim]  the denoiser's output
    std::unique_ptr<Ort::IoBinding> binding;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    std::uint64_t fenceValue = 0;

    // A RING of allocators, not one. An allocator may not be reset while the
    // GPU is still executing commands recorded into it, and nothing in the
    // step loop waits -- so reusing a single one reset it under the GPU's feet
    // and produced "the GPU will not respond to more commands", which is a TDR
    // rather than an error any API call returns. The command LIST is safe to
    // reset as soon as it has been submitted; only the allocator has to be
    // left alone, so one list cycling over several allocators is enough.
    //
    // Four, against two submissions per step: deep enough that the wait below
    // is never reached in practice, shallow enough to be a rounding error of
    // memory.
    static constexpr std::uint32_t kAllocators = 4;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocators[kAllocators];
    std::uint64_t allocatorFence[kAllocators] = {};
    std::uint32_t allocatorCursor = 0;
    std::uint32_t openAllocator = 0;

    // The per-step noise ramp is uploaded through a ring rather than one
    // buffer: writing a slot the GPU has not finished copying from would
    // corrupt the step in flight, and waiting for it every step would
    // reintroduce exactly the stall this class exists to remove. Eight slots
    // of 520 bytes, each remembering the fence value that retires it.
    static constexpr std::uint32_t kRingSlots = 8;
    Microsoft::WRL::ComPtr<ID3D12Resource> ring;
    std::uint64_t ringSlotFence[kRingSlots] = {};
    std::uint64_t ringSlotBytes = 0;
    std::uint32_t ringCursor = 0;

    // The sliding window's scratch. D3D12 forbids a CopyBufferRegion whose
    // source and destination are the same resource, so shifting the latents
    // left cannot be done in place: they go out to this and come back at an
    // offset. Allocated only for a continuous generation, which is the only
    // thing that slides.
    Microsoft::WRL::ComPtr<ID3D12Resource> slideScratch;
    // And a ring for the fresh noise appended at the back, for the same reason
    // the ramp has one: a slide happens every other step and must not wait on
    // the one before it. chunk_size latents per slide, so 80 bytes a slot at
    // the shipped shapes.
    Microsoft::WRL::ComPtr<ID3D12Resource> slideRing;
    std::uint64_t slideSlotBytes = 0;
    std::uint64_t slideSlotFence[kRingSlots] = {};
    std::uint32_t slideCursor = 0;

    // --- GPU execution timing ----------------------------------------------
    //
    // What a submission measurement cannot tell you. On this path a denoiser
    // Run records and returns, so timing it on the CPU measures how long it
    // took to ASK -- which is why every denoise figure from here has carried a
    // caveat. Timestamps answer the real question.
    //
    // The execution provider records onto its OWN command lists, so there is
    // nothing of ours to put a query inside. What makes this work anyway is
    // that the queue is in-order and the EP submits to the same one: a
    // timestamp written by the list BEFORE the Run and another by the list
    // AFTER it bracket the EP's work, whatever lists it used in between. The
    // two markers ride along on the noise upload and the latent-step dispatch
    // rather than needing submissions of their own -- which matters, because
    // extra submissions would inflate the very submission number this is being
    // compared against.
    //
    // A ring, and nothing ever waits for it: a slot is read only once its
    // fence has retired, so a step reads the result of a step several back.
    // The last few slots of a generation are drained by the caller.
    static constexpr std::uint32_t kTimerSlots = 8;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> timestampHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> timestampReadback;
    std::uint64_t timestampFrequency = 0;
    std::uint64_t timerFence[kTimerSlots] = {};
    bool timerPending[kTimerSlots] = {};

    // One buffer big enough for whichever constant input is largest, used only
    // by Begin. Released afterwards would save 786 KB and cost a reallocation
    // every generation, so it stays.
    Microsoft::WRL::ComPtr<ID3D12Resource> staging;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;

    int rows = 1;
    int time = 0;
    int latentDim = 0;

    ~GpuSampler() { Destroy(); }

    void Destroy() {
        // The binding references the values; the values reference the
        // allocations. Anything else is a use-after-free inside ORT.
        binding.reset();
        for (Tensor* tensor : {&x, &t, &context, &lens, &textIdx, &prediction}) {
            tensor->value.reset();
            if (tensor->allocation != nullptr && dmlApi != nullptr) {
                dmlApi->FreeGPUAllocation(tensor->allocation);
                tensor->allocation = nullptr;
            }
            tensor->buffer.Reset();
        }
        memoryInfo.reset();
    }

    bool MakeBuffer(std::uint64_t bytes, D3D12_HEAP_TYPE heapType,
                    Microsoft::WRL::ComPtr<ID3D12Resource>& out) const {
        const auto heap = CD3DX12_HEAP_PROPERTIES(heapType);
        // DirectML requires its tensor buffers to allow unordered access, and
        // so does the compute shader that writes the latents.
        const auto flags = heapType == D3D12_HEAP_TYPE_DEFAULT
                               ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                               : D3D12_RESOURCE_FLAG_NONE;
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);
        const D3D12_RESOURCE_STATES state =
            heapType == D3D12_HEAP_TYPE_UPLOAD    ? D3D12_RESOURCE_STATE_GENERIC_READ
            : heapType == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                   : D3D12_RESOURCE_STATE_COMMON;
        return SUCCEEDED(device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out)));
    }

    bool MakeTensor(Tensor& tensor, const std::vector<int64_t>& shape,
                    ONNXTensorElementDataType type, std::uint64_t elementBytes) {
        std::uint64_t elements = 1;
        for (const int64_t dim : shape) {
            elements *= static_cast<std::uint64_t>(dim);
        }
        tensor.bytes = elements * elementBytes;
        if (!MakeBuffer(tensor.bytes, D3D12_HEAP_TYPE_DEFAULT, tensor.buffer)) {
            return false;
        }
        if (dmlApi->CreateGPUAllocationFromD3DResource(tensor.buffer.Get(),
                                                       &tensor.allocation) != nullptr) {
            return false;
        }
        try {
            // "DML" memory means the tensor's data is the allocation above
            // rather than anything in CPU address space, which is what makes
            // the Run copy-free.
            tensor.value = std::make_unique<Ort::Value>(Ort::Value::CreateTensor(
                *memoryInfo, tensor.allocation, static_cast<std::size_t>(tensor.bytes),
                shape.data(), shape.size(), type));
        } catch (const Ort::Exception&) {
            return false;
        }
        return true;
    }

    bool Create(ID3D12Device* d3d, ID3D12CommandQueue* computeQueue,
                const OrtDmlApi* api, Ort::Session& denoiser, int rowCount,
                int submitted, int dim, int contextLen, int textDim,
                std::string* error) {
        device = d3d;
        queue = computeQueue;
        dmlApi = api;
        rows = rowCount;
        time = submitted;
        latentDim = dim;

        const auto fail = [&](const char* what) {
            SetOrtError(error, std::string("the GPU-resident sampler: ") + what);
            Destroy();
            return false;
        };

        try {
            memoryInfo = std::make_unique<Ort::MemoryInfo>("DML", OrtDeviceAllocator, 0,
                                                           OrtMemTypeDefault);
        } catch (const Ort::Exception& e) {
            return fail(Shorten(e.what()).c_str());
        }

        const int64_t r = rows; // batch
        const int64_t tm = time;
        const int64_t ld = latentDim;
        if (!MakeTensor(x, {r, tm, ld}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                        sizeof(float)) ||
            !MakeTensor(t, {r, tm}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                        sizeof(float)) ||
            !MakeTensor(context, {r, 1, contextLen, textDim},
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)) ||
            !MakeTensor(lens, {r, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                        sizeof(int64_t)) ||
            !MakeTensor(textIdx, {r, tm}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                        sizeof(int64_t)) ||
            !MakeTensor(prediction, {r, tm, ld}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                        sizeof(float))) {
            return fail("could not create the device tensors");
        }

        ringSlotBytes = t.bytes;
        if (!MakeBuffer(ringSlotBytes * kRingSlots, D3D12_HEAP_TYPE_UPLOAD, ring) ||
            !MakeBuffer(context.bytes, D3D12_HEAP_TYPE_UPLOAD, staging) ||
            !MakeBuffer(x.bytes, D3D12_HEAP_TYPE_READBACK, readback)) {
            return fail("could not create the upload and readback buffers");
        }

        // Two timestamps per slot, and a readback big enough for all of them.
        // Not fatal if any of it fails: the timings are a diagnostic, so the
        // sampler carries on without them rather than refusing to run.
        D3D12_QUERY_HEAP_DESC queryDesc = {};
        queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queryDesc.Count = kTimerSlots * 2;
        if (SUCCEEDED(device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&timestampHeap))) &&
            MakeBuffer(sizeof(std::uint64_t) * kTimerSlots * 2,
                       D3D12_HEAP_TYPE_READBACK, timestampReadback)) {
            // Ticks per second for THIS queue. A compute queue always supports
            // timestamps; only copy queues need a capability check.
            if (FAILED(queue->GetTimestampFrequency(&timestampFrequency))) {
                timestampFrequency = 0;
            }
        }
        if (timestampFrequency == 0) {
            timestampHeap.Reset();
            timestampReadback.Reset();
        }

        // Root constants and three root descriptors: three raw buffers need no
        // descriptor heap, so this dispatch cannot disturb whatever heaps the
        // renderer left bound on its own queue.
        //
        // Must stay in lockstep with shaders/latent_step.hlsl. Kept only here
        // rather than also in the shader, unlike the graphics signatures in
        // this project: the shader is compiled without a root signature
        // attribute, so there is no second copy to drift from.
        CD3DX12_ROOT_PARAMETER1 parameters[4] = {};
        parameters[0].InitAsConstants(8, 0);
        parameters[1].InitAsUnorderedAccessView(0);
        parameters[2].InitAsUnorderedAccessView(1);
        parameters[3].InitAsUnorderedAccessView(2);
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(static_cast<UINT>(std::size(parameters)), parameters, 0, nullptr,
                      D3D12_ROOT_SIGNATURE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3DBlob> serialized;
        Microsoft::WRL::ComPtr<ID3DBlob> blobError;
        if (FAILED(D3DX12SerializeVersionedRootSignature(
                &desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &serialized, &blobError)) ||
            FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                               serialized->GetBufferSize(),
                                               IID_PPV_ARGS(&rootSignature)))) {
            return fail("could not create the latent-step root signature");
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
        pso.pRootSignature = rootSignature.Get();
        pso.CS = {g_latent_step_bytecode, sizeof(g_latent_step_bytecode)};
        if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)))) {
            return fail("could not create the latent-step pipeline");
        }
        pipeline->SetName(L"FloodDiffusion latent step");

        // COMPUTE, matching the queue: a command list's type has to equal its
        // queue's, and this list only dispatches and copies buffers.
        for (std::uint32_t i = 0; i < kAllocators; ++i) {
            if (FAILED(device->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    IID_PPV_ARGS(&commandAllocators[i])))) {
                return fail("could not create the latent-step command allocators");
            }
        }
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             commandAllocators[0].Get(), nullptr,
                                             IID_PPV_ARGS(&commandList))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
            return fail("could not create the latent-step command list");
        }
        commandList->Close();

        try {
            binding = std::make_unique<Ort::IoBinding>(denoiser);
            binding->BindInput("x", *x.value);
            binding->BindInput("t", *t.value);
            binding->BindInput("context", *context.value);
            binding->BindInput("context_lens", *lens.value);
            binding->BindInput("text_idx", *textIdx.value);
            binding->BindOutput("prediction", *prediction.value);
        } catch (const Ort::Exception& e) {
            return fail(Shorten(e.what()).c_str());
        }
        return true;
    }


    // Blocks until `target` has retired. Used only where a wait is genuinely
    // required: the readback, teardown, the ring having wrapped, and the VAE
    // decoder's motion output.
    bool WaitFor(std::uint64_t target) {
        if (target == 0 || fence->GetCompletedValue() >= target) {
            return true;
        }
        Microsoft::WRL::Wrappers::Event event(
            CreateEventEx(nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE));
        if (!event.IsValid() ||
            FAILED(fence->SetEventOnCompletion(target, event.Get()))) {
            return false;
        }
        WaitForSingleObject(event.Get(), INFINITE);
        return true;
    }

    // One list, submitted and left for the queue to run. Everything in the
    // step loop goes through this, so nothing in the loop waits.
    bool Submit() {
        commandList->Close();
        ID3D12CommandList* lists[] = {commandList.Get()};
        queue->ExecuteCommandLists(1, lists);
        ++submissions;
        const std::uint64_t target = ++fenceValue;
        if (FAILED(queue->Signal(fence.Get(), target))) {
            return false;
        }
        // The allocator this list recorded into is now in flight, and must not
        // be reset before this value retires.
        allocatorFence[openAllocator] = target;
        return true;
    }

    bool BeginList() {
        const std::uint32_t slot = allocatorCursor % kAllocators;
        allocatorCursor = (allocatorCursor + 1) % kAllocators;
        if (!WaitFor(allocatorFence[slot])) {
            return false;
        }
        if (FAILED(commandAllocators[slot]->Reset()) ||
            FAILED(commandList->Reset(commandAllocators[slot].Get(), nullptr))) {
            return false;
        }
        openAllocator = slot;
        return true;
    }

    // Copies host bytes into a DEFAULT-heap buffer through the staging buffer,
    // and waits. Only Begin calls this: the constant inputs go up once per
    // generation, which is the whole point of binding them.
    bool UploadOnce(ID3D12Resource* destination, const void* source,
                    std::uint64_t bytes) {
        void* mapped = nullptr;
        if (FAILED(staging->Map(0, nullptr, &mapped))) {
            return false;
        }
        std::memcpy(mapped, source, static_cast<std::size_t>(bytes));
        staging->Unmap(0, nullptr);
        if (!BeginList()) {
            return false;
        }
        commandList->CopyBufferRegion(destination, 0, staging.Get(), 0, bytes);
        // Waited on, unlike the step loop: the caller is about to overwrite
        // the staging buffer with the next input.
        return Submit() && WaitFor(fenceValue);
    }

    // The noise ramp, through the ring, with no wait unless the ring has
    // wrapped onto a slot still in flight.
    bool UploadNoiseLevels(const float* levels, int count, int timerSlot) {
        const std::uint32_t slot = ringCursor % kRingSlots;
        ringCursor = (ringCursor + 1) % kRingSlots;
        if (!WaitFor(ringSlotFence[slot])) {
            return false;
        }

        const std::uint64_t offset = static_cast<std::uint64_t>(slot) * ringSlotBytes;
        void* mapped = nullptr;
        const D3D12_RANGE range = {static_cast<SIZE_T>(offset),
                                   static_cast<SIZE_T>(offset + ringSlotBytes)};
        if (FAILED(ring->Map(0, &range, &mapped))) {
            return false;
        }
        float* slotBase = reinterpret_cast<float*>(static_cast<std::uint8_t*>(mapped) +
                                                   offset);
        // Every guidance row sees the same ramp, which is what makes the two
        // predictions comparable.
        for (int row = 0; row < rows; ++row) {
            std::memcpy(slotBase + static_cast<std::size_t>(row) * count, levels,
                        static_cast<std::size_t>(count) * sizeof(float));
        }
        ring->Unmap(0, &range);

        if (!BeginList()) {
            return false;
        }
        commandList->CopyBufferRegion(t.buffer.Get(), 0, ring.Get(), offset, t.bytes);
        // The opening bracket, riding along rather than paying for a
        // submission of its own. Everything the queue runs between here and
        // the closing bracket in DispatchStep is the execution provider's.
        if (timerSlot >= 0 && timestampHeap != nullptr) {
            commandList->EndQuery(timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                  static_cast<UINT>(timerSlot) * 2);
        }
        if (!Submit()) {
            return false;
        }
        ringSlotFence[slot] = fenceValue;
        return true;
    }

    // The guidance combine and the Euler step, dispatched after the Run whose
    // prediction it reads. Same queue, so no barrier between them is needed
    // beyond the one queue order already gives.
    bool DispatchStep(int start, int end, float cfgScale, float dt,
                      std::uint32_t predictionType, int timerSlot) {
        if (end <= start) {
            return true;
        }
        if (!BeginList()) {
            return false;
        }
        // The closing bracket, before this list's own work so the interval
        // holds only the provider's.
        if (timerSlot >= 0 && timestampHeap != nullptr) {
            commandList->EndQuery(timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                  static_cast<UINT>(timerSlot) * 2 + 1);
            commandList->ResolveQueryData(
                timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                static_cast<UINT>(timerSlot) * 2, 2, timestampReadback.Get(),
                static_cast<UINT64>(timerSlot) * 2 * sizeof(std::uint64_t));
        }
        const std::uint32_t constants[8] = {
            static_cast<std::uint32_t>(start),
            static_cast<std::uint32_t>(end),
            static_cast<std::uint32_t>(latentDim),
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(time),
            // Floats through the same 32-bit root constants, reinterpreted
            // rather than converted.
            *reinterpret_cast<const std::uint32_t*>(&cfgScale),
            *reinterpret_cast<const std::uint32_t*>(&dt),
            predictionType,
        };
        commandList->SetComputeRootSignature(rootSignature.Get());
        commandList->SetPipelineState(pipeline.Get());
        commandList->SetComputeRoot32BitConstants(0, 8, constants, 0);
        commandList->SetComputeRootUnorderedAccessView(
            1, x.buffer->GetGPUVirtualAddress());
        commandList->SetComputeRootUnorderedAccessView(
            2, prediction.buffer->GetGPUVirtualAddress());
        commandList->SetComputeRootUnorderedAccessView(
            3, t.buffer->GetGPUVirtualAddress());
        const std::uint32_t lanes =
            static_cast<std::uint32_t>(end - start) * static_cast<std::uint32_t>(latentDim);
        commandList->Dispatch((lanes + 63) / 64, 1, 1);
        ++computeDispatches;
        if (!Submit()) {
            return false;
        }
        if (timerSlot >= 0 && timestampHeap != nullptr) {
            timerFence[timerSlot] = fenceValue;
            timerPending[timerSlot] = true;
        }
        return true;
    }

    // Milliseconds of GPU execution for a slot, if the GPU has got that far.
    // False means "not yet", never "failed" -- the caller tries again later
    // rather than waiting, which is what keeps this path free of stalls.
    bool TryReadTimer(int slot, double* ms) {
        if (slot < 0 || !timerPending[slot] || timestampReadback == nullptr) {
            return false;
        }
        if (fence->GetCompletedValue() < timerFence[slot]) {
            return false;
        }
        const std::uint64_t base =
            static_cast<std::uint64_t>(slot) * 2 * sizeof(std::uint64_t);
        void* mapped = nullptr;
        const D3D12_RANGE range = {
            static_cast<SIZE_T>(base),
            static_cast<SIZE_T>(base + 2 * sizeof(std::uint64_t))};
        if (FAILED(timestampReadback->Map(0, &range, &mapped))) {
            return false;
        }
        const std::uint64_t* stamps = reinterpret_cast<const std::uint64_t*>(
            static_cast<std::uint8_t*>(mapped) + base);
        const std::uint64_t first = stamps[0];
        const std::uint64_t second = stamps[1];
        const D3D12_RANGE empty = {0, 0};
        timestampReadback->Unmap(0, &empty);
        timerPending[slot] = false;
        if (second <= first || timestampFrequency == 0) {
            return false;
        }
        *ms = static_cast<double>(second - first) * 1000.0 /
              static_cast<double>(timestampFrequency);
        return true;
    }

    // Allocated on demand, because only a continuous generation slides.
    bool EnsureSlideBuffers(int chunkSize) {
        if (slideScratch != nullptr) {
            return true;
        }
        slideSlotBytes = static_cast<std::uint64_t>(chunkSize) *
                         static_cast<std::uint64_t>(latentDim) * sizeof(float);
        return MakeBuffer(x.bytes, D3D12_HEAP_TYPE_DEFAULT, slideScratch) &&
               MakeBuffer(slideSlotBytes * kRingSlots, D3D12_HEAP_TYPE_UPLOAD,
                          slideRing);
    }

    // Slides the latent window left by `by` latents and writes `fresh` into
    // the `by` slots that opens at the back. Per row, because x is
    // [rows, time, latentDim] and a shift that ignored the row boundary would
    // drag one guidance row's tail into the next row's head.
    //
    // No wait: three copies on the same queue, ordered against the denoise
    // either side of them by queue order alone, like everything else here.
    bool Slide(int by, const float* fresh, int chunkSize) {
        if (by <= 0) {
            return true;
        }
        if (!EnsureSlideBuffers(chunkSize)) {
            return false;
        }
        const std::uint64_t rowBytes = static_cast<std::uint64_t>(time) *
                                       static_cast<std::uint64_t>(latentDim) *
                                       sizeof(float);
        const std::uint64_t byBytes = static_cast<std::uint64_t>(by) *
                                      static_cast<std::uint64_t>(latentDim) *
                                      sizeof(float);
        if (byBytes > slideSlotBytes || byBytes >= rowBytes) {
            return false;
        }

        const std::uint32_t slot = slideCursor % kRingSlots;
        slideCursor = (slideCursor + 1) % kRingSlots;
        if (!WaitFor(slideSlotFence[slot])) {
            return false;
        }
        const std::uint64_t offset = static_cast<std::uint64_t>(slot) * slideSlotBytes;
        void* mapped = nullptr;
        const D3D12_RANGE range = {static_cast<SIZE_T>(offset),
                                   static_cast<SIZE_T>(offset + byBytes)};
        if (FAILED(slideRing->Map(0, &range, &mapped))) {
            return false;
        }
        std::memcpy(static_cast<std::uint8_t*>(mapped) + offset, fresh,
                    static_cast<std::size_t>(byBytes));
        slideRing->Unmap(0, &range);

        if (!BeginList()) {
            return false;
        }
        // Out whole, back per row at an offset, then the new tail. Reading the
        // scratch while writing x is safe because they are different
        // resources, which is the entire reason the scratch exists.
        commandList->CopyBufferRegion(slideScratch.Get(), 0, x.buffer.Get(), 0, x.bytes);
        // x is a copy SOURCE above and a copy DEST below, in one list, so the
        // implicit promotion each buffer gets at first use is not enough -- the
        // debug layer rejects the second copy with "Missing State:
        // D3D12_RESOURCE_STATE_COPY_DEST". Both resources swap roles here, so
        // both are transitioned. No transition back: buffers decay to COMMON
        // when the list finishes executing, which is what lets the next
        // dispatch promote x to UAV again.
        const D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(x.buffer.Get(),
                                                 D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_COPY_DEST),
            CD3DX12_RESOURCE_BARRIER::Transition(slideScratch.Get(),
                                                 D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_COPY_SOURCE),
        };
        commandList->ResourceBarrier(static_cast<UINT>(std::size(barriers)), barriers);
        for (int row = 0; row < rows; ++row) {
            const std::uint64_t base = static_cast<std::uint64_t>(row) * rowBytes;
            commandList->CopyBufferRegion(x.buffer.Get(), base, slideScratch.Get(),
                                          base + byBytes, rowBytes - byBytes);
            // Every guidance row gets the SAME fresh noise, for the same
            // reason they get the same ramp: the two predictions are only
            // comparable if the latents they see are identical.
            commandList->CopyBufferRegion(x.buffer.Get(), base + rowBytes - byBytes,
                                          slideRing.Get(), offset, byBytes);
        }
        if (!Submit()) {
            return false;
        }
        slideSlotFence[slot] = fenceValue;
        return true;
    }

    // Four floats per latent, for the decoder. The only wait in a generation
    // that is not teardown, and it happens per decode batch rather than per
    // step.
    bool ReadLatents(int firstIndex, int count, float* out) {
        if (count <= 0) {
            return true;
        }
        const std::uint64_t offset = static_cast<std::uint64_t>(firstIndex) *
                                     static_cast<std::uint64_t>(latentDim) * sizeof(float);
        const std::uint64_t bytes = static_cast<std::uint64_t>(count) *
                                    static_cast<std::uint64_t>(latentDim) * sizeof(float);
        if (!BeginList()) {
            return false;
        }
        // Row 0 only: the rows hold the same latents.
        commandList->CopyBufferRegion(readback.Get(), 0, x.buffer.Get(), offset, bytes);
        if (!Submit() || !WaitFor(fenceValue)) {
            return false;
        }
        void* mapped = nullptr;
        const D3D12_RANGE range = {0, static_cast<SIZE_T>(bytes)};
        if (FAILED(readback->Map(0, &range, &mapped))) {
            return false;
        }
        std::memcpy(out, mapped, static_cast<std::size_t>(bytes));
        const D3D12_RANGE empty = {0, 0};
        readback->Unmap(0, &empty);
        return true;
    }
};

struct FloodDiffusionPipeline::Impl {
    FloodDiffusionConfig config;
    FloodDiffusionDevice device = FloodDiffusionDevice::Cpu;
    std::filesystem::path directory;

    UnigramTokenizer tokenizer;

    // One set of options per device, because the prompt encoder can sit on a
    // different provider from the sampler. ORT copies what it needs out of
    // SessionOptions at session creation, but these are cheap to keep and the
    // text encoder is created lazily, long after Load returns.
    // The denoiser and the VAE decoders need SEPARATE option sets once
    // dimensions are pinned: the denoiser runs a classifier-free-guidance pair
    // (batch 2) while the decoders run one latent at a time (batch 1), and a
    // free dimension override is by NAME, so a single shared set would pin
    // `batch` to the wrong value for one of them.
    std::unique_ptr<Ort::SessionOptions> denoiserOptions;
    std::unique_ptr<Ort::SessionOptions> vaeOptions;
    std::unique_ptr<Ort::SessionOptions> textOptions;
    FloodDiffusionDevice textEncoderDevice = FloodDiffusionDevice::Cpu;
    FloodDiffusionDevice vaeDevice = FloodDiffusionDevice::Cpu;

    bool staticShapes = false;
    int staticLatentFrames = 0;
    int staticTime = 0;   // the pinned `time`: latentFrames + chunkSize

    // Non-null when the sampler keeps its latents in GPU memory; see
    // GpuSampler. Requires DirectML and pinned shapes, both checked at Load.
    std::unique_ptr<GpuSampler> gpu;


    // Declared BEFORE the sessions on purpose: members are destroyed in
    // reverse declaration order, and the queue an execution provider submits
    // to has to outlive the sessions that submit to it.
    //
    // COMPUTE rather than DIRECT because that is the whole point -- a separate
    // hardware engine, so a 61 ms generation is scheduled beside the frame
    // instead of ahead of it. The DirectML EP documents DIRECT or COMPUTE.
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> computeQueue;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    std::uint64_t fenceValue = 0;

    std::unique_ptr<Ort::Session> denoiser;
    std::unique_ptr<Ort::Session> vaeFirst;
    std::unique_ptr<Ort::Session> vaeStep;
    std::unique_ptr<Ort::Session> textEncoder;

    // Kept so the text encoder can be created on demand, long after Load.
    ID3D12Device* d3dDevice = nullptr;
    IDMLDevice* dmlDevice = nullptr;

    // One entry per distinct prompt, including the empty one used as the
    // unconditional branch. Encoding is the only part of the pipeline that
    // needs the gigabyte-scale text encoder, and a stream reuses the same
    // prompt every step, so caching is what keeps that model off the hot path.
    struct TextFeatures {
        std::vector<float> values;  // tokens * textDim
        int tokens = 0;
    };
    std::map<std::string, TextFeatures> textCache;

    // --- generation state ---------------------------------------------------

    std::string prompt;
    int promptTokens = 0;
    std::vector<int64_t> promptIds;

    DiffusionForcingSchedule schedule;
    int step = 0;
    bool running = false;

    // [totalLatents, latentDim], updated in place by each step.
    std::vector<float> latents;
    std::vector<float> noiseLevels;

    // Context assembled once per generation: [rows, 1, contextLen, textDim].
    std::vector<float> context;
    std::vector<int64_t> contextLens;
    int rows = 1;
    int contextLen = 0;

    // VAE state. `caches` holds the 20 tensors threaded from one decode to the
    // next; `decodedLatents` is how far the decoder has consumed.
    std::vector<Ort::Value> caches;
    int decodedLatents = 0;
    // Zero decodes inside Step, as it always did; non-zero hands decoding to
    // DecodeAhead. See FloodDiffusionOptions::decodeAheadFrames.
    int decodeAheadFrames = 0;
    // See FloodDiffusionOptions::samplerLeadFrames. Zero is unbounded.
    int samplerLeadFrames = 0;

    // --- the sliding window, for FloodDiffusionOptions::continuous ----------
    //
    // `shifted` is how many latents have scrolled off the front, so absolute
    // latent j sits at window slot j - shifted. It stays 0 for a finite
    // generation and for a continuous one until the ramp has run up, which is
    // what makes the two identical step for step until then.
    bool continuous = false;
    int shifted = 0;
    bool stopRequested = false;
    // Fresh noise for the latents appended at the back of the window, and the
    // generator it comes from -- held across steps so a stream does not
    // restart the sequence, which would repeat the same noise every slide.
    std::mt19937 noise;
    std::vector<float> freshLatents;

    std::unique_ptr<StreamJointRecovery> recovery;
    std::vector<DirectX::XMFLOAT3> frames;  // frameCount * joints
    // The decoded feature vectors behind those positions, frameCount *
    // motionDim. Kept for retargeting, which needs the rotation block. About
    // 250 KB for a twelve-second take.
    std::vector<float> features;
    // The accumulated root yaw per frame, from the recovery's own accumulator.
    std::vector<float> rootAngles;
    int frameCount = 0;

    double lastDenoiseMs = 0.0;
    double lastDecodeMs = 0.0;
    double generationMs = 0.0;
    // Summed over the generation. The pair is what localises the cost: the
    // denoiser runs 128 times and the decoder 60, and it is the decoder that
    // dominates.
    double denoiseTotalMs = 0.0;
    double decodeTotalMs = 0.0;

    // Per-call, for the mean and the maximum. The decode pair is kept
    // separately from decodeTotalMs on purpose: that one times whole
    // DecodeUpTo batches and so includes the latent readback the GPU path
    // waits on, while these time the Run itself.
    int denoiseCalls = 0;
    double denoiseMaxMs = 0.0;
    // GPU EXECUTION, from timestamp queries, and so only on the gpuLatents
    // path. Separate from the figures above because they measure a different
    // thing there: those time the submission, these time the work.
    int denoiseGpuCalls = 0;
    double denoiseGpuTotalMs = 0.0;
    double denoiseGpuMaxMs = 0.0;
    int decodeCalls = 0;
    double decodeCallTotalMs = 0.0;
    double decodeMaxMs = 0.0;

    // Begin's clock, and the wall time at which motion first existed.
    std::chrono::steady_clock::time_point genBegan{};
    double firstFrameMs = 0.0;

    // The last SetPrompt, and the first latent that could carry it. A latent
    // already finalised when the prompt changed cannot: it will never be
    // denoised again. So the first NEW pose is the first frame decoded from a
    // latent at or past the frontier at that moment, and the wait for it is
    // what a prompt change feels like.
    std::chrono::steady_clock::time_point promptChangedAt{};
    int promptChangeLatent = -1;
    double promptLatencyMs = 0.0;

    // Total bytes of the .onnx files this pipeline opened, for the question of
    // what the models cost against the renderer.
    std::uint64_t modelBytes = 0;

    // Scratch, reused across steps so a step does not allocate.
    std::vector<float> batchedLatents;
    std::vector<float> batchedTimes;
    std::vector<int64_t> batchedTextIdx;

    std::vector<std::string> cacheInNames;
    std::vector<std::string> cacheOutNames;
    std::vector<const char*> vaeFirstOutputNames;
    std::vector<const char*> vaeStepInputNames;
    std::vector<const char*> vaeStepOutputNames;

    struct DimensionOverride {
        const char* name;
        int64_t value;
    };

    bool MakeOptions(FloodDiffusionDevice forDevice,
                     const std::vector<DimensionOverride>& overrides,
                     std::unique_ptr<Ort::SessionOptions>& out,
                     std::string* error);
    bool EncodeText(const std::string& text, std::string* error);
    // Encodes `newPrompt` (and the empty prompt guidance needs) and rebuilds
    // `context`/`contextLens` from it. Host side only; UploadContext is what
    // puts it on the device.
    bool ConditionOn(const std::string& newPrompt, std::string* error);
    bool UploadContext(std::string* error);
    bool EnsureTextEncoder(std::string* error);
    bool Denoise(std::string* error);

    // Drains whatever timestamp slots the GPU has finished with into the
    // execution stats. Cheap and non-blocking: a slot that is not ready is
    // left for next time.
    void HarvestGpuTimers() {
        if (gpu == nullptr) {
            return;
        }
        for (int slot = 0; slot < static_cast<int>(GpuSampler::kTimerSlots); ++slot) {
            double ms = 0.0;
            if (gpu->TryReadTimer(slot, &ms)) {
                ++denoiseGpuCalls;
                denoiseGpuTotalMs += ms;
                denoiseGpuMaxMs = std::max(denoiseGpuMaxMs, ms);
            }
        }
    }
    bool DecodeUpTo(int latentLimit, std::string* error);
    bool DecodeOne(int latentIndex, std::string* error);
    void AppendMotion(const float* motion, int frameCountInBatch);

    // Slides the window forward by `by` latents: drops that many off the
    // front and appends that many of fresh noise at the back. Only ever
    // called with `by` latents that have already been decoded, which
    // SlideBudget is what enforces.
    bool SlideWindow(int by, std::string* error);

    // True when the sampler has finalised as much as the lead allows and must
    // wait for the decoder. Expressed in latents, because that is the unit
    // both halves count in; the option is in frames because that is the unit a
    // person thinks in.
    bool SamplerAtLeadLimit() const {
        if (samplerLeadFrames <= 0) {
            return false;
        }
        const int perLatent = config.FramesPerLatent();
        if (perLatent <= 0) {
            return false;
        }
        // Rounded up, so a cap smaller than one latent still permits one --
        // a cap that permitted none would never produce anything.
        const int allowed = std::max(1, (samplerLeadFrames + perLatent - 1) / perLatent);
        return FinalisedLatents() - decodedLatents >= allowed;
    }

    // How far the window may slide right now: enough to keep the finalised
    // frontier `latentFrames` into the window, but never past what the decoder
    // has consumed. Zero means the sampler has to wait for the decoder.
    int SlideBudget() const;

    // Latents no remaining step will touch again. `start` never decreases, so
    // the current step's start is exactly that boundary -- except for a finite
    // sampler that has run out of steps, where everything left is as denoised
    // as it is going to get. A continuous one never reaches that case.
    int FinalisedLatents() const {
        if (!schedule.IsValid()) {
            return 0;
        }
        if (continuous || step < schedule.TotalSteps()) {
            return schedule.StartAt(step);
        }
        return schedule.GeneratedLatents();
    }
};

bool FloodDiffusionPipeline::Impl::MakeOptions(
    FloodDiffusionDevice forDevice,
    const std::vector<DimensionOverride>& overrides,
    std::unique_ptr<Ort::SessionOptions>& out, std::string* error) {
    out = std::make_unique<Ort::SessionOptions>();

    // Pinning a symbolic dimension by name is what lets shape inference
    // resolve statically, which is in turn what lets the DirectML EP compile
    // the graph rather than dispatch it per operator. Harmless on the CPU
    // provider, so it is applied either way and the session simply refuses a
    // shape that disagrees -- which is the behaviour we want if a caller ever
    // feeds the wrong length.
    for (const DimensionOverride& entry : overrides) {
        out->AddFreeDimensionOverrideByName(entry.name, entry.value);
    }

    if (forDevice == FloodDiffusionDevice::Cpu) {
        return true;
    }

    if (d3dDevice == nullptr || dmlDevice == nullptr) {
        SetOrtError(error, "FloodDiffusionDevice::DirectML needs the D3D12 device "
                           "and the DirectML device; pass them to Load");
        return false;
    }
    if (computeQueue == nullptr) {
        SetOrtError(error, "the pipeline's compute queue was not created");
        return false;
    }
    const OrtDmlApi* dmlApi = SharedDmlApi(error);
    if (dmlApi == nullptr) {
        return false;
    }
    // DirectML's documented session configuration: the EP supports neither
    // memory-pattern optimisation (which assumes fixed shapes) nor parallel
    // execution. MEASURED on ORT 1.24.4, though: a session created with
    // neither of these set still builds AND runs, producing identical output,
    // so they are advisable rather than load-bearing. Newer ORT appears to
    // settle them for the EP itself. Kept because the documentation asks for
    // them and they cost nothing.
    out->DisableMemPattern();
    out->SetExecutionMode(ORT_SEQUENTIAL);
    // The pipeline's own compute queue, not the caller's. Every session shares
    // it, so their submissions are ordered against each other without a fence
    // -- which the denoiser and the two VAE decoders rely on, since a decode
    // reads what a denoise step produced.
    const OrtStatus* status = dmlApi->SessionOptionsAppendExecutionProvider_DML1(
        *out, dmlDevice, computeQueue.Get());
    if (status != nullptr) {
        std::string message = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
        SetOrtError(error, "could not attach the DirectML execution provider: " + message);
        return false;
    }
    return true;
}

FloodDiffusionPipeline::FloodDiffusionPipeline() : m_impl(std::make_unique<Impl>()) {}

FloodDiffusionPipeline::~FloodDiffusionPipeline() {
    // Explicit rather than defaulted: the sessions have to be released while
    // the queue they submitted to is still alive AND idle. Member order gets
    // the first half; this drain is the second. Skipping it does not fault
    // here -- it surfaces later as a device removal with no obvious cause.
    WaitForWork(nullptr);
    m_impl->textEncoder.reset();
    m_impl->vaeStep.reset();
    m_impl->vaeFirst.reset();
    m_impl->denoiser.reset();
}

ID3D12CommandQueue* FloodDiffusionPipeline::ComputeQueue() const {
    return m_impl->computeQueue.Get();
}

bool FloodDiffusionPipeline::OrderAfterWork(ID3D12CommandQueue* queue,
                                            std::string* error) {
    Impl& impl = *m_impl;
    if (impl.computeQueue == nullptr || impl.fence == nullptr) {
        // The CPU path submits nothing, so there is nothing to order against
        // and saying so as a failure would make every caller special-case it.
        return true;
    }
    if (queue == nullptr) {
        SetOrtError(error, "null queue");
        return false;
    }
    ++impl.fenceValue;
    if (FAILED(impl.computeQueue->Signal(impl.fence.Get(), impl.fenceValue))) {
        SetOrtError(error, "fence signal failed");
        return false;
    }
    if (FAILED(queue->Wait(impl.fence.Get(), impl.fenceValue))) {
        SetOrtError(error, "cross-queue wait failed");
        return false;
    }
    return true;
}

bool FloodDiffusionPipeline::WaitForWork(std::string* error) {
    // Draining is exactly when the last timestamp slots become readable, so
    // this is where a generation's final steps get counted.
    m_impl->HarvestGpuTimers();
    Impl& impl = *m_impl;
    if (impl.computeQueue == nullptr || impl.fence == nullptr) {
        return true;
    }
    ++impl.fenceValue;
    if (FAILED(impl.computeQueue->Signal(impl.fence.Get(), impl.fenceValue))) {
        SetOrtError(error, "fence signal failed");
        return false;
    }
    if (impl.fence->GetCompletedValue() >= impl.fenceValue) {
        return true;
    }
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) {
        SetOrtError(error, "could not create a wait event");
        return false;
    }
    impl.fence->SetEventOnCompletion(impl.fenceValue, event);
    // Generous: a long generation's last submission can still be running, and
    // the alternative to waiting is releasing a session underneath it.
    const DWORD waited = WaitForSingleObject(event, 60000);
    CloseHandle(event);
    if (waited != WAIT_OBJECT_0) {
        SetOrtError(error, "timed out waiting for the compute queue");
        return false;
    }
    return true;
}

bool FloodDiffusionPipeline::Load(const std::filesystem::path& directory,
                                  const FloodDiffusionOptions& options,
                                  std::string* error,
                                  ID3D12Device* device,
                                  IDMLDevice* dmlDevice) {
    Impl& impl = *m_impl;

    // Before anything is released: a provider change destroys sessions whose
    // submissions may still be running on the compute queue, and on a queue of
    // our own nothing else is going to wait for them.
    if (!WaitForWork(error)) {
        return false;
    }

    // Load is callable again to change provider, so everything a provider owns
    // has to go first. The text encoder matters most: it is created lazily, so
    // a stale one would quietly keep serving prompts from the OLD provider.
    // The cached features go with it -- they are just floats, but they were
    // produced by the session being replaced.
    // Before the sessions, because the sampler holds an Ort::IoBinding onto
    // the denoiser: an OrtIoBinding outliving its session is a use-after-free
    // inside ORT.
    impl.gpu.reset();
    impl.denoiser.reset();
    impl.vaeFirst.reset();
    impl.vaeStep.reset();
    impl.textEncoder.reset();
    impl.textCache.clear();
    impl.caches.clear();
    impl.modelBytes = 0;
    impl.running = false;
    impl.step = 0;
    impl.decodedLatents = 0;
    impl.frames.clear();
    impl.frameCount = 0;
    impl.promptIds.clear();
    impl.promptTokens = 0;
    impl.features.clear();

    impl.directory = directory;
    impl.device = options.device;
    impl.d3dDevice = device;
    impl.dmlDevice = dmlDevice;

    // The pipeline's own queue, created once and kept across provider changes:
    // a fence value already handed out has to stay meaningful, and recreating
    // the queue would reset the timeline out from under anyone holding one.
    if (device != nullptr && impl.computeQueue == nullptr) {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        if (FAILED(device->CreateCommandQueue(&queueDesc,
                                              IID_PPV_ARGS(&impl.computeQueue)))) {
            SetOrtError(error, "could not create the pipeline's compute queue");
            return false;
        }
        impl.computeQueue->SetName(L"FloodDiffusion compute");
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(&impl.fence)))) {
            SetOrtError(error, "could not create the pipeline's fence");
            return false;
        }
        impl.fence->SetName(L"FloodDiffusion fence");
    }

    if (!impl.config.Load(directory / "pipeline.txt", error)) {
        return false;
    }
    if (!impl.tokenizer.Load(directory / "tokenizer.bin", error)) {
        return false;
    }
    impl.textEncoderDevice = options.textEncoderDevice;
    impl.vaeDevice = options.vaeDevice;
    impl.staticShapes = options.staticShapes;
    impl.staticLatentFrames = options.latentFrames;
    impl.staticTime = options.latentFrames + impl.config.chunkSize;

    // Classifier-free guidance submits a [conditional, unconditional] pair, so
    // the denoiser's batch is 2 whenever the exported cfg_scale is not 1.
    const int64_t guidanceRows = impl.config.cfgScale != 1.0f ? 2 : 1;

    std::vector<Impl::DimensionOverride> denoiserDims;
    std::vector<Impl::DimensionOverride> vaeDims;
    if (impl.staticShapes) {
        if (options.latentFrames <= 0) {
            SetOrtError(error, "staticShapes needs a positive latentFrames at Load, "
                               "because the pinned sequence length is baked into "
                               "the session");
            return false;
        }
        denoiserDims = {
            {"batch", guidanceRows},
            {"time", impl.staticTime},
            {"segments", 1},
            // Padding the context up to text_len is exact: context_lens already
            // masks the padding out of cross-attention.
            {"ctx_len", impl.config.textLen},
        };
        // The cache_out_9's ConstantOfShape may come from padding operations in decoding? (see CausalConv1d in https://github.com/AlayaLab/FloodDiffusion/blob/b611d6c4e24357a90066034935ed445179723b37/models/tools/wan_vae_1d.py#L32)
        // The VAE decoders are left DYNAMIC, and the reason has changed twice,
        // so it is worth recording what was actually measured rather than only
        // the conclusion. //Engine:fd-probe section 11 is the apparatus.
        //
        // ONE. Pinning `batch` on vae_decoder_FIRST makes the DirectML EP
        // return a NULL for one of its 20 cache outputs -- cache_out_9 -- so
        // the next decode fails with "Unexpected input data type. Actual:
        // (((null))) , expected: ((tensor(float)))".
        //
        // Root-caused under a debugger against a Debug build of ORT 1.31, and
        // it is not a shape problem at all. cache_out_9 is the only one of the
        // twenty produced by a ConstantOfShape; pinning `batch` makes its
        // shape input static, so ORT's ConstantFolding evaluates it and
        // replaces the output with a graph INITIALIZER of the same name. The
        // DirectML EP's graph fusion then finds that initializer unused by the
        // fused partition -- its only consumer, the Identity behind
        // cache_out_14, was folded away too -- and deletes it with
        // Graph::RemoveInitializedTensor, which does not check whether the
        // name is a graph output. The output then has neither a producer nor
        // an initializer behind it. Setting
        // optimization.disable_specified_optimizers to ConstantFolding avoids
        // it, and is not worth what the rest of constant folding is worth: the
        // pinned DirectML decode is still 4.45 ms a call against the CPU's
        // 1.36. skills/onnx-directml-ep/SKILL.md section 5.0 has the stack.
        //
        // TWO. So Ort::IoBinding with the caches in D3D12 buffers was built,
        // on the theory that removing the output-to-input hand-off would
        // remove the refusal. It does not: IoBinding on vae_decoder_step fails
        // in every configuration -- `batch` pinned or symbolic, fusion on or
        // off, caches in device OR host memory -- while the identical values
        // through the ordinary Run API work in all of them. The binding is the
        // problem, not the memory. That code was written, measured and
        // removed.
        //
        // THREE. Pinning only vae_decoder_STEP does work -- no null, hand-off
        // intact -- and buys nothing measurable: 5.42 against 5.50 ms per
        // call, which is run-to-run noise. So there is no configuration of the
        // pin worth a third SessionOptions set. What did move the decode was
        // FloodDiffusionOptions::vaeDevice, about 4x, by leaving the EP that
        // all of the above is a property of.
        vaeDims = {};
    }

    // A sliding window IS a fixed submitted length, so continuous mode and
    // pinned shapes are the same requirement from two directions. Without the
    // pin, `time` would grow for as long as the stream ran and the sessions
    // would be rebuilt every other step.
    if (options.continuous && !impl.staticShapes) {
        SetOrtError(error, "continuous needs staticShapes: a sliding window has a "
                           "fixed submitted length, and without pinning `time` "
                           "would grow for as long as the stream ran");
        return false;
    }

    if (!impl.MakeOptions(impl.device, denoiserDims, impl.denoiserOptions, error)) {
        return false;
    }
    if (!impl.MakeOptions(impl.vaeDevice, vaeDims, impl.vaeOptions, error)) {
        return false;
    }
    if (!impl.MakeOptions(impl.textEncoderDevice, {}, impl.textOptions, error)) {
        return false;
    }

    const auto open = [&](const char* name, const Ort::SessionOptions& sessionOptions,
                          std::unique_ptr<Ort::Session>& out) {
        const std::filesystem::path path = directory / name;
        if (!std::filesystem::exists(path)) {
            SetOrtError(error, path.string() + " is missing; run "
                               "tools/prepare_flooddiffusion.py");
            return false;
        }
        try {
            out = std::make_unique<Ort::Session>(SharedOrtEnv(), path.c_str(),
                                                 sessionOptions);
        } catch (const Ort::Exception& e) {
            SetOrtError(error, std::string("loading ") + name + ": " + Shorten(e.what()));
            return false;
        }
        // Weights on disk, which is the floor on what the models cost in
        // memory. Not the same as what they occupy once loaded -- the EP
        // preprocesses and may pad -- so it is reported as a file size and
        // called one.
        std::error_code ignored;
        const std::uintmax_t bytes = std::filesystem::file_size(path, ignored);
        if (!ignored) {
            impl.modelBytes += bytes;
        }
        return true;
    };

    if (!open("denoiser.onnx", *impl.denoiserOptions, impl.denoiser) ||
        !open("vae_decoder_first.onnx", *impl.vaeOptions, impl.vaeFirst) ||
        !open("vae_decoder_step.onnx", *impl.vaeOptions, impl.vaeStep)) {
        return false;
    }

    // The VAE's cache tensors are addressed by name on every decode, so the
    // name arrays and the pointer arrays ORT wants are built once.
    const int cacheCount = impl.config.vaeNumCaches;
    impl.cacheInNames.clear();
    impl.cacheOutNames.clear();
    for (int i = 0; i < cacheCount; ++i) {
        impl.cacheInNames.push_back("cache_in_" + std::to_string(i));
        impl.cacheOutNames.push_back("cache_out_" + std::to_string(i));
    }
    // The GPU-resident sampler, once the denoiser session exists to bind to.
    // Both preconditions are refused rather than worked around: without
    // DirectML there is no device tensor to make, and without pinned shapes
    // the x buffer would have to be rebuilt every other step.
    impl.gpu.reset();
    if (options.gpuLatents) {
        if (impl.device != FloodDiffusionDevice::DirectML) {
            SetOrtError(error, "gpuLatents needs FloodDiffusionDevice::DirectML");
            return false;
        }
        if (!impl.staticShapes) {
            SetOrtError(error, "gpuLatents needs staticShapes: the device buffers are "
                               "created once with the pinned shapes");
            return false;
        }
        // guidanceRows, not impl.rows: the latter is settled by Begin from
        // the prompt, and these buffers are created here. Both come from the
        // exported cfg_scale, so they agree -- and the denoiser session was
        // just created with `batch` pinned to this very value, so a mismatch
        // would be refused by Run rather than silently batched wrongly.
        const int contextLen = impl.config.textLen;
        auto sampler = std::make_unique<GpuSampler>();
        if (!sampler->Create(device, impl.computeQueue.Get(), SharedDmlApi(error),
                             *impl.denoiser, static_cast<int>(guidanceRows),
                             impl.staticTime, impl.config.latentDim, contextLen,
                             impl.config.textDim, error)) {
            return false;
        }
        impl.gpu = std::move(sampler);
    }

    impl.vaeFirstOutputNames.assign({"motion"});
    impl.vaeStepInputNames.assign({"z"});
    impl.vaeStepOutputNames.assign({"motion"});
    for (int i = 0; i < cacheCount; ++i) {
        impl.vaeFirstOutputNames.push_back(impl.cacheOutNames[i].c_str());
        impl.vaeStepInputNames.push_back(impl.cacheInNames[i].c_str());
        impl.vaeStepOutputNames.push_back(impl.cacheOutNames[i].c_str());
    }

    impl.recovery = std::make_unique<StreamJointRecovery>(impl.config.joints,
                                                          options.smoothingAlpha);
    return true;
}

bool FloodDiffusionPipeline::IsLoaded() const {
    return m_impl->denoiser != nullptr && m_impl->vaeFirst != nullptr &&
           m_impl->vaeStep != nullptr;
}

const FloodDiffusionConfig& FloodDiffusionPipeline::Config() const {
    return m_impl->config;
}

FloodDiffusionDevice FloodDiffusionPipeline::Device() const {
    return m_impl->device;
}

FloodDiffusionDevice FloodDiffusionPipeline::TextEncoderDevice() const {
    return m_impl->textEncoderDevice;
}

FloodDiffusionDevice FloodDiffusionPipeline::VaeDevice() const {
    return m_impl->vaeDevice;
}

bool FloodDiffusionPipeline::StaticShapes() const { return m_impl->staticShapes; }
bool FloodDiffusionPipeline::GpuLatents() const { return m_impl->gpu != nullptr; }

double FloodDiffusionPipeline::DenoiseTotalMs() const {
    return m_impl->denoiseTotalMs;
}
FloodDiffusionPipeline::PhaseTiming FloodDiffusionPipeline::DenoiseTiming() const {
    const Impl& impl = *m_impl;
    PhaseTiming out;
    out.calls = impl.denoiseCalls;
    out.totalMs = impl.denoiseTotalMs;
    out.meanMs = impl.denoiseCalls > 0
                     ? impl.denoiseTotalMs / static_cast<double>(impl.denoiseCalls)
                     : 0.0;
    out.maxMs = impl.denoiseMaxMs;
    return out;
}

FloodDiffusionPipeline::PhaseTiming FloodDiffusionPipeline::DecodeTiming() const {
    const Impl& impl = *m_impl;
    PhaseTiming out;
    out.calls = impl.decodeCalls;
    out.totalMs = impl.decodeCallTotalMs;
    out.meanMs = impl.decodeCalls > 0
                     ? impl.decodeCallTotalMs / static_cast<double>(impl.decodeCalls)
                     : 0.0;
    out.maxMs = impl.decodeMaxMs;
    return out;
}

double FloodDiffusionPipeline::FirstFrameMs() const { return m_impl->firstFrameMs; }

FloodDiffusionPipeline::PhaseTiming FloodDiffusionPipeline::DenoiseGpuTiming() const {
    const Impl& impl = *m_impl;
    PhaseTiming out;
    out.calls = impl.denoiseGpuCalls;
    out.totalMs = impl.denoiseGpuTotalMs;
    out.meanMs = impl.denoiseGpuCalls > 0
                     ? impl.denoiseGpuTotalMs / static_cast<double>(impl.denoiseGpuCalls)
                     : 0.0;
    out.maxMs = impl.denoiseGpuMaxMs;
    return out;
}

FloodDiffusionPipeline::GpuWorkCounts FloodDiffusionPipeline::GpuWork() const {
    const Impl& impl = *m_impl;
    GpuWorkCounts out;
    out.ortRuns = impl.denoiseCalls + impl.decodeCalls;
    // One fused dispatch per Run on DirectML with shapes pinned; a Run on the
    // CPU provider costs the GPU nothing, which is why the decoders moving
    // there took the decode out of this count entirely. Without pinning the
    // denoiser would be 1226 per Run instead of 1, and saying so is the point
    // of reporting it.
    const int perDirectMlRun = impl.staticShapes ? 1 : 1226;
    if (impl.device == FloodDiffusionDevice::DirectML) {
        out.epDispatches += impl.denoiseCalls * perDirectMlRun;
    }
    if (impl.vaeDevice == FloodDiffusionDevice::DirectML) {
        out.epDispatches += impl.decodeCalls * perDirectMlRun;
    }
    if (impl.gpu != nullptr) {
        out.computeDispatches = impl.gpu->computeDispatches;
        out.submissions = impl.gpu->submissions;
    }
    return out;
}

double FloodDiffusionPipeline::PromptLatencyMs() const {
    return m_impl->promptLatencyMs;
}

std::uint64_t FloodDiffusionPipeline::ModelBytes() const { return m_impl->modelBytes; }

double FloodDiffusionPipeline::DecodeTotalMs() const {
    return m_impl->decodeTotalMs;
}

int FloodDiffusionPipeline::StaticLatentFrames() const {
    return m_impl->staticLatentFrames;
}

bool FloodDiffusionPipeline::Impl::EnsureTextEncoder(std::string* error) {
    if (textEncoder != nullptr) {
        return true;
    }
    if (!config.hasTextEncoder) {
        SetOrtError(error, "this export was staged with --skip-text-encoder, so a "
                           "prompt cannot be encoded");
        return false;
    }
    const std::filesystem::path path = directory / "text_encoder.onnx";
    if (!std::filesystem::exists(path)) {
        SetOrtError(error, path.string() + " is missing; re-run "
                           "tools/prepare_flooddiffusion.py without "
                           "--skip-text-encoder");
        return false;
    }
    try {
        textEncoder = std::make_unique<Ort::Session>(SharedOrtEnv(), path.c_str(),
                                                     *textOptions);
    } catch (const Ort::Exception& e) {
        SetOrtError(error, "loading text_encoder.onnx: " + Shorten(e.what()));
        return false;
    }
    // Counted now rather than at Load, because this one is loaded on demand --
    // and it is 1.1 GB, so whether it has been loaded is most of the answer to
    // what these models cost.
    std::error_code ignored;
    const std::uintmax_t bytes = std::filesystem::file_size(path, ignored);
    if (!ignored) {
        modelBytes += bytes;
    }
    return true;
}

bool FloodDiffusionPipeline::Impl::EncodeText(const std::string& text,
                                              std::string* error) {
    if (textCache.count(text) != 0) {
        return true;
    }
    if (!EnsureTextEncoder(error)) {
        return false;
    }

    std::vector<int64_t> ids = tokenizer.Encode(text, config.textLen);
    if (ids.empty()) {
        SetOrtError(error, "the tokenizer produced no tokens");
        return false;
    }

    // attention_mask is all ones: nothing is padded, because each prompt is
    // encoded on its own rather than batched to a common length.
    std::vector<int64_t> mask(ids.size(), 1);
    const int64_t shape[] = {1, static_cast<int64_t>(ids.size())};
    Ort::MemoryInfo cpu = CpuMemoryInfo();

    const char* inputNames[] = {"input_ids", "attention_mask"};
    const char* outputNames[] = {"text_features"};
    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, ids.data(), ids.size(),
                                                           shape, 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, mask.data(), mask.size(),
                                                           shape, 2));
        std::vector<Ort::Value> outputs =
            textEncoder->Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(),
                             inputs.size(), outputNames, 1);
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto outShape = info.GetShape();
        if (outShape.size() != 3 || outShape[2] != config.textDim) {
            SetOrtError(error, "text_encoder.onnx returned an unexpected feature width");
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(outShape[1] * outShape[2]);
        TextFeatures entry;
        entry.tokens = static_cast<int>(outShape[1]);
        entry.values.assign(outputs[0].GetTensorData<float>(),
                            outputs[0].GetTensorData<float>() + count);
        textCache.emplace(text, std::move(entry));
    } catch (const Ort::Exception& e) {
        SetOrtError(error, "encoding text: " + Shorten(e.what()));
        return false;
    }
    return true;
}

bool FloodDiffusionPipeline::Begin(const std::string& prompt,
                                   const FloodDiffusionOptions& options,
                                   std::string* error) {
    Impl& impl = *m_impl;
    if (!IsLoaded()) {
        SetOrtError(error, "the FloodDiffusion pipeline is not loaded");
        return false;
    }

    const int denoiseSteps = options.denoiseSteps > 0 ? options.denoiseSteps
                                                      : impl.config.noiseSteps;
    DiffusionForcingSchedule schedule(options.latentFrames, impl.config.chunkSize,
                                      denoiseSteps, options.continuous);
    if (!schedule.IsValid()) {
        SetOrtError(error, "latentFrames must be positive");
        return false;
    }

    // A pinned sequence length is baked into the session, so a different
    // latentFrames cannot be fed at all. Refuse it here, before any state is
    // replaced, rather than failing inside Run with a shape mismatch.
    if (impl.staticShapes && options.latentFrames != impl.staticLatentFrames) {
        SetOrtError(error, "this pipeline was loaded with staticShapes for " +
                               std::to_string(impl.staticLatentFrames) +
                               " latent frames and cannot run " +
                               std::to_string(options.latentFrames) +
                               "; load again to change the length");
        return false;
    }

    if (!impl.ConditionOn(prompt, error)) {
        return false;
    }

    // Starting noise. std::mt19937 rather than numpy's PCG64: the stream will
    // not match the Python reference for the same seed, which is called out in
    // FloodDiffusionOptions.
    impl.schedule = schedule;
    impl.latents.assign(static_cast<std::size_t>(schedule.TotalLatents()) *
                            static_cast<std::size_t>(impl.config.latentDim),
                        0.0f);
    // Seeded here and then KEPT: a sliding window draws the latents it
    // appends from the same stream, so a stream does not see this same noise
    // again on every slide. A finite generation consumes exactly the same
    // numbers it always did.
    impl.noise.seed(options.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : impl.latents) {
        value = normal(impl.noise);
    }

    // Everything the GPU path keeps resident goes up here, once: the context
    // and its lengths, the all-zero text_idx, and the seeded latents
    // duplicated across the guidance rows. After this, a step sends 520 bytes.
    if (impl.gpu != nullptr) {
        // Per generation, like every other statistic here. Reset before the
        // uploads below rather than with the counters further down, so this
        // generation's own setup submissions are counted in it.
        impl.gpu->computeDispatches = 0;
        impl.gpu->submissions = 0;

        const std::size_t perRow = static_cast<std::size_t>(impl.staticTime) *
                                   static_cast<std::size_t>(impl.config.latentDim);
        impl.batchedLatents.assign(perRow * static_cast<std::size_t>(impl.rows), 0.0f);
        for (int row = 0; row < impl.rows; ++row) {
            // The latent buffer holds TotalLatents, which in static mode is
            // exactly staticTime, so this is a whole-buffer copy per row.
            std::copy(impl.latents.begin(),
                      impl.latents.begin() + static_cast<std::ptrdiff_t>(
                                                 std::min(perRow, impl.latents.size())),
                      impl.batchedLatents.begin() +
                          static_cast<std::ptrdiff_t>(perRow * row));
        }
        impl.batchedTextIdx.assign(static_cast<std::size_t>(impl.staticTime) *
                                       static_cast<std::size_t>(impl.rows),
                                   0);
        if (!impl.gpu->UploadOnce(impl.gpu->x.buffer.Get(), impl.batchedLatents.data(),
                                  impl.gpu->x.bytes) ||
            !impl.gpu->UploadOnce(impl.gpu->context.buffer.Get(), impl.context.data(),
                                  impl.gpu->context.bytes) ||
            !impl.gpu->UploadOnce(impl.gpu->lens.buffer.Get(), impl.contextLens.data(),
                                  impl.gpu->lens.bytes) ||
            !impl.gpu->UploadOnce(impl.gpu->textIdx.buffer.Get(),
                                  impl.batchedTextIdx.data(),
                                  impl.gpu->textIdx.bytes)) {
            SetOrtError(error, "uploading the generation's inputs to the GPU failed");
            return false;
        }
    }

    impl.step = 0;
    impl.running = true;
    impl.continuous = options.continuous;
    impl.shifted = 0;
    impl.stopRequested = false;
    impl.caches.clear();
    impl.decodedLatents = 0;
    impl.frames.clear();
    impl.features.clear();
    impl.rootAngles.clear();
    impl.frameCount = 0;
    impl.lastDenoiseMs = 0.0;
    impl.lastDecodeMs = 0.0;
    impl.generationMs = 0.0;
    impl.denoiseTotalMs = 0.0;
    impl.decodeTotalMs = 0.0;
    impl.denoiseCalls = 0;
    impl.denoiseMaxMs = 0.0;
    impl.denoiseGpuCalls = 0;
    impl.denoiseGpuTotalMs = 0.0;
    impl.denoiseGpuMaxMs = 0.0;
    impl.decodeCalls = 0;
    impl.decodeCallTotalMs = 0.0;
    impl.decodeMaxMs = 0.0;
    impl.firstFrameMs = 0.0;
    impl.genBegan = std::chrono::steady_clock::now();
    impl.promptChangeLatent = -1;
    impl.promptLatencyMs = 0.0;
    impl.decodeAheadFrames = std::max(0, options.decodeAheadFrames);
    impl.samplerLeadFrames = std::max(0, options.samplerLeadFrames);
    impl.recovery->SetSmoothingAlpha(options.smoothingAlpha);
    impl.recovery->Reset();
    return true;
}

bool FloodDiffusionPipeline::Impl::Denoise(std::string* error) {
    const int end = schedule.EndAt(step);
    const int start = schedule.StartAt(step);
    if (end <= 0) {
        return true;
    }

    // How much of the sequence is submitted. Normally just the live prefix,
    // which is all the sampler needs -- but that makes the shape change every
    // other step and denies the DirectML EP any chance to compile the graph.
    // With pinned shapes the whole buffer goes every time; `end` still decides
    // what is READ back, so the extra frames only cost GPU work, not meaning.
    const int submitted = staticShapes ? staticTime : end;

    // The ramp over the submitted WINDOW, indexed from the window's own start
    // rather than from the beginning of the sequence. Past `end` it clips to
    // 1, i.e. pure noise, which is what the model saw for not-yet-denoised
    // frames in training; before the window it does not exist, which is what
    // `shifted` says. For a finite generation `shifted` is 0 and this is the
    // ramp it always had.
    schedule.WindowNoiseLevels(step, shifted, submitted, noiseLevels);

    // Everything below indexes the WINDOW, so the absolute range the schedule
    // reports has to be rebased onto it. Identical to the absolute range until
    // a continuous generation starts sliding.
    const int windowStart = start - shifted;
    const int windowEnd = end - shifted;
    const int latentDim = config.latentDim;
    const std::size_t perRow = static_cast<std::size_t>(submitted) *
                               static_cast<std::size_t>(latentDim);

    // The GPU-resident path. The latents are already in the x buffer from the
    // previous step's dispatch, the context has not changed since Begin, and
    // the prediction is written into a buffer this process owns -- so the only
    // thing to send is the ramp, and nothing comes back.
    if (gpu != nullptr) {
        // One slot per step, round robin. Harvested before being reused, so a
        // slot's result is read several steps after it was written and nothing
        // ever waits on the GPU to produce it.
        const int timerSlot = step % static_cast<int>(GpuSampler::kTimerSlots);
        HarvestGpuTimers();

        const auto began = std::chrono::steady_clock::now();
        if (!gpu->UploadNoiseLevels(noiseLevels.data(), submitted, timerSlot)) {
            SetOrtError(error, "uploading the noise ramp failed");
            return false;
        }

        try {
            denoiser->Run(Ort::RunOptions{nullptr}, *gpu->binding);
        } catch (const Ort::Exception& e) {
            SetOrtError(error, "denoising: " + Shorten(e.what()));
            return false;
        }
        // "vel" needs neither the current latent nor the level, but the shader
        // takes all three so the two paths cannot diverge on the other two
        // prediction types.
        const std::uint32_t predictionType = config.predictionType == "x0"   ? 1u
                                             : config.predictionType == "noise" ? 2u
                                                                                : 0u;
        if (!gpu->DispatchStep(windowStart, windowEnd, config.cfgScale,
                               static_cast<float>(schedule.Dt()), predictionType,
                               timerSlot)) {
            SetOrtError(error, "dispatching the latent step failed");
            return false;
        }
        // Wall time only, and deliberately not a GPU time: nothing in this
        // path waits, so what this measures is how long the submission took.
        // The honest end-to-end number is GenerationMs.
        lastDenoiseMs = Milliseconds(began);
        denoiseTotalMs += lastDenoiseMs;
        ++denoiseCalls;
        denoiseMaxMs = std::max(denoiseMaxMs, lastDenoiseMs);
        return true;
    }

    // Every guidance row sees the same latents and noise levels; only the text
    // context differs, which is what makes the two predictions comparable.
    batchedLatents.resize(perRow * static_cast<std::size_t>(rows));
    batchedTimes.resize(static_cast<std::size_t>(submitted) * static_cast<std::size_t>(rows));
    batchedTextIdx.assign(static_cast<std::size_t>(submitted) * static_cast<std::size_t>(rows), 0);
    for (int row = 0; row < rows; ++row) {
        std::copy(latents.begin(), latents.begin() + static_cast<std::ptrdiff_t>(perRow),
                  batchedLatents.begin() + static_cast<std::ptrdiff_t>(perRow * row));
        std::copy(noiseLevels.begin(), noiseLevels.begin() + submitted,
                  batchedTimes.begin() + static_cast<std::ptrdiff_t>(
                                             static_cast<std::size_t>(submitted) * row));
    }

    const int64_t rowCount = rows;
    const int64_t time = submitted;
    const int64_t xShape[] = {rowCount, time, latentDim};
    const int64_t tShape[] = {rowCount, time};
    const int64_t contextShape[] = {rowCount, 1, contextLen, config.textDim};
    const int64_t lensShape[] = {rowCount, 1};

    const char* inputNames[] = {"x", "t", "context", "context_lens", "text_idx"};
    const char* outputNames[] = {"prediction"};
    Ort::MemoryInfo cpu = CpuMemoryInfo();

    const auto began = std::chrono::steady_clock::now();
    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu, batchedLatents.data(), batchedLatents.size(), xShape, 3));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu, batchedTimes.data(), batchedTimes.size(), tShape, 2));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu, context.data(), context.size(), contextShape, 4));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu, contextLens.data(), contextLens.size(), lensShape, 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            cpu, batchedTextIdx.data(), batchedTextIdx.size(), tShape, 2));

        std::vector<Ort::Value> outputs =
            denoiser->Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(),
                          inputs.size(), outputNames, 1);
        lastDenoiseMs = Milliseconds(began);
        denoiseTotalMs += lastDenoiseMs;
        ++denoiseCalls;
        denoiseMaxMs = std::max(denoiseMaxMs, lastDenoiseMs);

        const float* prediction = outputs[0].GetTensorData<float>();
        const float dt = static_cast<float>(schedule.Dt());
        const float cfg = config.cfgScale;

        for (int index = windowStart; index < windowEnd; ++index) {
            for (int component = 0; component < latentDim; ++component) {
                // Row-major [rows, submitted, latentDim]; `perRow` already
                // accounts for how much was submitted, so row 1 is at perRow.
                const std::size_t offset =
                    static_cast<std::size_t>(index) * static_cast<std::size_t>(latentDim) +
                    static_cast<std::size_t>(component);

                // Classifier-free guidance: push away from what the model would
                // predict with no prompt at all.
                float value = prediction[offset];
                if (rows == 2) {
                    const float unconditional = prediction[perRow + offset];
                    value = cfg * value - (cfg - 1.0f) * unconditional;
                }

                const float level = noiseLevels[static_cast<std::size_t>(index)];
                float velocity = value;
                if (config.predictionType == "x0") {
                    velocity = (value - latents[offset]) / level;
                } else if (config.predictionType == "noise") {
                    velocity = (latents[offset] - value) / (1.0f + dt - level);
                }
                latents[offset] += velocity * dt;
            }
        }
    } catch (const Ort::Exception& e) {
        SetOrtError(error, "denoising: " + Shorten(e.what()));
        return false;
    }
    return true;
}

void FloodDiffusionPipeline::Impl::AppendMotion(const float* motion, int frameCountInBatch) {
    // The instant the stream first has something to show, which is the latency
    // a viewer waits through. Recorded before the copying below rather than
    // after, because the frames exist as soon as the decoder produced them.
    if (frameCount == 0 && frameCountInBatch > 0) {
        firstFrameMs = Milliseconds(genBegan);
    }
    const int joints = config.joints;
    const int motionDim = config.motionDim;
    frames.resize(static_cast<std::size_t>(frameCount + frameCountInBatch) *
                  static_cast<std::size_t>(joints));
    features.resize(static_cast<std::size_t>(frameCount + frameCountInBatch) *
                    static_cast<std::size_t>(motionDim));
    rootAngles.reserve(static_cast<std::size_t>(frameCount + frameCountInBatch));
    for (int frame = 0; frame < frameCountInBatch; ++frame) {
        const float* feature = motion + static_cast<std::size_t>(frame) *
                                            static_cast<std::size_t>(motionDim);
        std::copy(feature, feature + motionDim,
                  features.begin() + static_cast<std::ptrdiff_t>(
                                         static_cast<std::size_t>(frameCount) *
                                         static_cast<std::size_t>(motionDim)));
        recovery->ProcessFrame(feature,
                               frames.data() + static_cast<std::size_t>(frameCount) *
                                                   static_cast<std::size_t>(joints));
        rootAngles.push_back(recovery->RootAngle());
        ++frameCount;
    }
}

bool FloodDiffusionPipeline::Impl::DecodeOne(int latentIndex, std::string* error) {
    // Scoped so every return path records the call, including the ones that
    // fail -- a decode that threw still cost its time, and a maximum that
    // quietly excluded the slow failures would be the wrong number.
    struct Timed {
        Impl& impl;
        std::chrono::steady_clock::time_point began =
            std::chrono::steady_clock::now();
        ~Timed() {
            const double ms = Milliseconds(began);
            ++impl.decodeCalls;
            impl.decodeCallTotalMs += ms;
            impl.decodeMaxMs = std::max(impl.decodeMaxMs, ms);
        }
    } timed{*this};

    const int64_t zShape[] = {1, 1, config.latentDim};
    // latentIndex is absolute and `latents` is the window, so a continuous
    // generation that has slid reads further back than the index says.
    const float* z = latents.data() +
                     static_cast<std::size_t>(latentIndex - shifted) *
                         static_cast<std::size_t>(config.latentDim);
    // CreateTensor wants non-const memory even though Run only reads it.
    std::vector<float> zCopy(z, z + config.latentDim);
    Ort::MemoryInfo cpu = CpuMemoryInfo();
    const char* zName[] = {"z"};

    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(cpu, zCopy.data(), zCopy.size(),
                                                         zShape, 3));
        std::vector<Ort::Value> outputs;
        if (latentIndex == 0) {
            outputs = vaeFirst->Run(Ort::RunOptions{nullptr}, zName, inputs.data(), 1,
                                    vaeFirstOutputNames.data(),
                                    vaeFirstOutputNames.size());
        } else {
            if (caches.size() != static_cast<std::size_t>(config.vaeNumCaches)) {
                SetOrtError(error, "the VAE caches are missing; the first latent must "
                                   "be decoded before any other");
                return false;
            }
            // The caches move INTO the input list and the fresh ones move back
            // out of the results, so the decoder is strictly causal: each latent
            // is decoded exactly once, in order.
            for (Ort::Value& cache : caches) {
                inputs.push_back(std::move(cache));
            }
            caches.clear();
            outputs = vaeStep->Run(Ort::RunOptions{nullptr}, vaeStepInputNames.data(),
                                   inputs.data(), inputs.size(),
                                   vaeStepOutputNames.data(),
                                   vaeStepOutputNames.size());
        }

        const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3 || shape[2] != config.motionDim) {
            SetOrtError(error, "the VAE decoder returned an unexpected motion shape");
            return false;
        }
        AppendMotion(outputs[0].GetTensorData<float>(), static_cast<int>(shape[1]));

        caches.clear();
        for (std::size_t i = 1; i < outputs.size(); ++i) {
            caches.push_back(std::move(outputs[i]));
        }

        // The first pose that could carry the new prompt has just been
        // decoded. Measured here rather than in AppendMotion because only
        // this function knows which latent the frames came from.
        if (promptChangeLatent >= 0 && latentIndex >= promptChangeLatent) {
            promptLatencyMs = Milliseconds(promptChangedAt);
            promptChangeLatent = -1;
        }
    } catch (const Ort::Exception& e) {
        SetOrtError(error, "decoding latent " + std::to_string(latentIndex) + ": " +
                               Shorten(e.what()));
        return false;
    }
    return true;
}

bool FloodDiffusionPipeline::Impl::DecodeUpTo(int latentLimit, std::string* error) {
    // GeneratedLatents is the finite sequence's length and is not a bound on a
    // continuous one, which has none.
    const int limit = continuous ? latentLimit
                                 : std::min(latentLimit, schedule.GeneratedLatents());
    const auto began = std::chrono::steady_clock::now();

    // On the GPU path the authoritative latents are in the x buffer, so the
    // ones about to be decoded have to come back first. This is the only wait
    // in a generation: it happens once per decode batch -- about 60 times --
    // rather than once per step, and it brings back four floats per latent.
    if (gpu != nullptr && limit > decodedLatents) {
        const int count = limit - decodedLatents;
        if (!gpu->ReadLatents(decodedLatents - shifted, count,
                              latents.data() +
                                  static_cast<std::size_t>(decodedLatents - shifted) *
                                      static_cast<std::size_t>(config.latentDim))) {
            SetOrtError(error, "reading the finalised latents back failed");
            return false;
        }
    }
    bool decoded = false;
    while (decodedLatents < limit) {
        if (!DecodeOne(decodedLatents, error)) {
            return false;
        }
        ++decodedLatents;
        decoded = true;
    }
    if (decoded) {
        lastDecodeMs = Milliseconds(began);
        decodeTotalMs += lastDecodeMs;
    }
    return true;
}

int FloodDiffusionPipeline::Impl::SlideBudget() const {
    if (!continuous) {
        return 0;
    }
    // Keep the finalised frontier exactly staticLatentFrames into the window.
    // That is not a tuning choice: the live region is chunk_size wide, so with
    // the frontier there the step's write range ends precisely at the window's
    // end, and the window is full rather than overrun or short.
    const int wanted = schedule.StartAt(step) - staticLatentFrames - shifted;
    if (wanted <= 0) {
        return 0;
    }
    // Never past the decoder. A latent dropped off the front before it has
    // been decoded is motion that never existed, so this is the one place the
    // sampler is allowed to be held up by its consumer.
    return std::min(wanted, std::max(0, decodedLatents - shifted));
}

bool FloodDiffusionPipeline::Impl::SlideWindow(int by, std::string* error) {
    if (by <= 0) {
        return true;
    }
    const int latentDim = config.latentDim;
    const std::size_t stride = static_cast<std::size_t>(latentDim);
    const std::size_t window = latents.size() / stride;
    if (static_cast<std::size_t>(by) >= window) {
        SetOrtError(error, "the window cannot slide by its own width");
        return false;
    }

    // Drawn from the generation's own generator rather than a fresh one, so a
    // stream does not see the same noise again on every slide.
    freshLatents.resize(static_cast<std::size_t>(by) * stride);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : freshLatents) {
        value = normal(noise);
    }

    if (gpu != nullptr && !gpu->Slide(by, freshLatents.data(), config.chunkSize)) {
        SetOrtError(error, "sliding the latent window on the GPU failed");
        return false;
    }
    // Slid on the host either way: on the GPU path this array is where
    // ReadLatents puts the latents the decoder reads, so its indexing has to
    // track the window too.
    std::memmove(latents.data(), latents.data() + static_cast<std::size_t>(by) * stride,
                 (window - static_cast<std::size_t>(by)) * stride * sizeof(float));
    std::copy(freshLatents.begin(), freshLatents.end(),
              latents.begin() + static_cast<std::ptrdiff_t>(
                                    (window - static_cast<std::size_t>(by)) * stride));
    shifted += by;
    return true;
}

bool FloodDiffusionPipeline::Impl::ConditionOn(const std::string& newPrompt,
                                               std::string* error) {
    // The conditional prompt and the unconditional (empty) one. Both are
    // encoded before anything is replaced, because a failure here should leave
    // the conditioning that is in force untouched rather than half-changed --
    // which matters more for SetPrompt on a running stream than it did for
    // Begin.
    if (!EncodeText(newPrompt, error)) {
        return false;
    }
    const bool guided = config.cfgScale != 1.0f;
    if (guided && !EncodeText(std::string(), error)) {
        return false;
    }

    const TextFeatures& conditional = textCache.at(newPrompt);
    const TextFeatures* unconditional =
        guided ? &textCache.at(std::string()) : nullptr;

    const int rowCount = guided ? 2 : 1;
    int width = conditional.tokens;
    if (unconditional != nullptr) {
        width = std::max(width, unconditional->tokens);
    }
    if (staticShapes) {
        // The session pinned ctx_len to text_len, so the context has to be fed
        // at that width. Exact rather than approximate: context_lens below
        // still reports the real token counts, and the model masks the rest
        // out of cross-attention. Measured difference against an unpadded run:
        // 0.000004.
        width = config.textLen;
        if (conditional.tokens > width ||
            (unconditional != nullptr && unconditional->tokens > width)) {
            SetOrtError(error, "the prompt encoded to more tokens than text_len");
            return false;
        }
    }

    // Everything above could still fail; from here nothing can, so this is
    // where the live conditioning is allowed to change.
    prompt = newPrompt;
    // Kept for the UI. Done here rather than in EncodeText, which returns
    // early for a prompt it has already cached and would leave these stale.
    promptIds = tokenizer.Encode(newPrompt, config.textLen);
    promptTokens = static_cast<int>(promptIds.size());
    rows = rowCount;
    contextLen = width;

    // context is [rows, segments=1, contextLen, textDim], zero-padded, with
    // context_lens telling the model how much of each row is real. Only one
    // segment: a single prompt covers the whole sequence, so text_idx is all
    // zeros. Several prompts over time would widen the segments axis -- which
    // is a different feature from SetPrompt, where one prompt replaces another
    // for everything still in flight rather than taking over at a frame.
    const std::size_t stride = static_cast<std::size_t>(contextLen) *
                               static_cast<std::size_t>(config.textDim);
    context.assign(stride * static_cast<std::size_t>(rows), 0.0f);
    contextLens.assign(static_cast<std::size_t>(rows), 0);

    std::copy(conditional.values.begin(), conditional.values.end(), context.begin());
    contextLens[0] = conditional.tokens;
    if (unconditional != nullptr) {
        std::copy(unconditional->values.begin(), unconditional->values.end(),
                  context.begin() + static_cast<std::ptrdiff_t>(stride));
        contextLens[1] = unconditional->tokens;
    }
    return true;
}

bool FloodDiffusionPipeline::Impl::UploadContext(std::string* error) {
    if (gpu == nullptr) {
        return true;
    }
    // Only reachable with pinned shapes, where contextLen is text_len and the
    // buffer was created at exactly that width -- so a mismatch here would
    // mean the pin and the context had come apart, and reading the host array
    // for the buffer's length would run off the end of it.
    if (context.size() * sizeof(float) != gpu->context.bytes ||
        contextLens.size() * sizeof(int64_t) != gpu->lens.bytes) {
        SetOrtError(error, "the prompt context does not match the device buffer "
                           "it was created for");
        return false;
    }
    // Ordered after anything already submitted, because it goes on the same
    // queue the execution provider uses -- so a Run still reading the old
    // context cannot see this copy land halfway. UploadOnce waits, which costs
    // a few milliseconds once per prompt rather than once per step.
    if (!gpu->UploadOnce(gpu->context.buffer.Get(), context.data(),
                         gpu->context.bytes) ||
        !gpu->UploadOnce(gpu->lens.buffer.Get(), contextLens.data(),
                         gpu->lens.bytes)) {
        SetOrtError(error, "uploading the prompt context to the GPU failed");
        return false;
    }
    return true;
}

bool FloodDiffusionPipeline::SetPrompt(const std::string& prompt, std::string* error) {
    Impl& impl = *m_impl;
    if (!IsLoaded()) {
        SetOrtError(error, "the FloodDiffusion pipeline is not loaded");
        return false;
    }
    // Cheap enough to call on every edit: the common case is that nothing
    // changed, and the next-cheapest is a prompt already in the cache.
    if (prompt == impl.prompt) {
        return true;
    }
    if (!impl.ConditionOn(prompt, error) || !impl.UploadContext(error)) {
        return false;
    }
    // Armed only for a RUNNING generation: with nothing in flight there is no
    // pose on the way and the next Begin would reset this anyway.
    if (impl.running) {
        impl.promptChangedAt = std::chrono::steady_clock::now();
        impl.promptChangeLatent = impl.FinalisedLatents();
        impl.promptLatencyMs = 0.0;
    }
    return true;
}

void FloodDiffusionPipeline::Stop() { m_impl->stopRequested = true; }

int FloodDiffusionPipeline::ShiftedLatents() const { return m_impl->shifted; }

bool FloodDiffusionPipeline::Step(std::string* error) {
    Impl& impl = *m_impl;
    if (!impl.running) {
        return true;
    }
    if (!IsLoaded()) {
        SetOrtError(error, "the FloodDiffusion pipeline is not loaded");
        return false;
    }

    const auto began = std::chrono::steady_clock::now();

    // Out of steps: everything that remains is as denoised as it is going to
    // get. Eagerly, that means decode the rest and stop. Paced, the sampler is
    // simply finished and the generation is not -- running stays true until
    // DecodeAhead has drained the backlog, because a stream is over when the
    // motion exists, not when the latents do.
    impl.HarvestGpuTimers();
    // A continuous generation has no step to run out of: past the ramp it
    // slides the window instead, and only Stop() ends the sampling.
    const bool sampling = impl.continuous
                              ? !impl.stopRequested
                              : impl.step < impl.schedule.TotalSteps();

    // The lead cap, checked before anything else the sampler would do. A
    // stall here falls THROUGH to the decode below rather than returning, so
    // the half that releases it still runs -- decoding eagerly that is the
    // only thing that could, and returning early would deadlock.
    const bool atLeadLimit = sampling && impl.SamplerAtLeadLimit();

    if (sampling && !atLeadLimit) {
        // Before denoising, not after: the slide is what makes room for what
        // this step is about to write, and the arithmetic in SlideBudget only
        // lands the write range exactly at the window's end if it happens
        // first.
        const int wanted =
            impl.continuous
                ? impl.schedule.StartAt(impl.step) - impl.staticLatentFrames - impl.shifted
                : 0;
        if (wanted > 0) {
            const int budget = impl.SlideBudget();
            if (budget < wanted) {
                // The decoder is behind, so there is nowhere to put this
                // step's output yet. Doing nothing and saying so is the whole
                // of the backpressure; the caller's next DecodeAhead is what
                // releases it.
                impl.generationMs += Milliseconds(began);
                return true;
            }
            if (!impl.SlideWindow(wanted, error)) {
                impl.running = false;
                return false;
            }
        }
        if (!impl.Denoise(error)) {
            impl.running = false;
            return false;
        }
        ++impl.step;
    }

    if (impl.decodeAheadFrames == 0) {
        // A latent is final once no remaining step will touch it again.
        // Because `start` never decreases, the next step's start is exactly
        // that boundary.
        const int finalised = impl.FinalisedLatents();
        if (!impl.DecodeUpTo(finalised, error)) {
            impl.running = false;
            return false;
        }
    }

    // Over when the MOTION exists, not when the latents do -- the same rule in
    // both modes. For a continuous stream the sampler's end is Stop() rather
    // than running out of steps.
    if (!sampling && impl.decodedLatents >= impl.FinalisedLatents()) {
        impl.running = false;
    }
    impl.generationMs += Milliseconds(began);
    return true;
}

bool FloodDiffusionPipeline::DecodeAhead(int consumedFrame, std::string* error) {
    Impl& impl = *m_impl;
    if (impl.decodeAheadFrames == 0 || !impl.schedule.IsValid()) {
        return true;
    }
    if (!IsLoaded()) {
        SetOrtError(error, "the FloodDiffusion pipeline is not loaded");
        return false;
    }

    // Frames wanted, rounded up to whole latents because a latent is the unit
    // the decoder produces. The first decodes to vaeFramesFirst and the rest
    // to vaeFramesStep, so the arithmetic is over the frames already in hand
    // rather than over latents times four.
    const int wanted = std::max(0, consumedFrame) + impl.decodeAheadFrames;
    const int perLatent = impl.config.FramesPerLatent();
    if (perLatent <= 0) {
        SetOrtError(error, "the VAE decodes no frames per latent");
        return false;
    }

    const auto began = std::chrono::steady_clock::now();
    const int finalised = impl.FinalisedLatents();
    int target = impl.decodedLatents;
    for (int frames = impl.frameCount; frames < wanted && target < finalised; ++target) {
        frames += target == 0 ? impl.config.vaeFramesFirst : perLatent;
    }
    if (target > impl.decodedLatents && !impl.DecodeUpTo(target, error)) {
        impl.running = false;
        return false;
    }

    // The same end condition Step applies, so a generation whose sampler
    // finished long ago still reports itself over on the tick that decodes its
    // last latent rather than on the next Step.
    const bool sampling = impl.continuous
                              ? !impl.stopRequested
                              : impl.step < impl.schedule.TotalSteps();
    if (!sampling && impl.decodedLatents >= impl.FinalisedLatents()) {
        impl.running = false;
    }
    impl.generationMs += Milliseconds(began);
    return true;
}

bool FloodDiffusionPipeline::IsRunning() const { return m_impl->running; }
bool FloodDiffusionPipeline::Done() const {
    return !m_impl->running && m_impl->frameCount > 0;
}
int FloodDiffusionPipeline::StepsTaken() const { return m_impl->step; }
int FloodDiffusionPipeline::TotalSteps() const { return m_impl->schedule.TotalSteps(); }

int FloodDiffusionPipeline::FrameCount() const { return m_impl->frameCount; }
int FloodDiffusionPipeline::DecodedLatents() const { return m_impl->decodedLatents; }

int FloodDiffusionPipeline::FinalisedLatents() const {
    return m_impl->FinalisedLatents();
}

int FloodDiffusionPipeline::JointCount() const { return m_impl->config.joints; }

const DirectX::XMFLOAT3* FloodDiffusionPipeline::Frame(int index) const {
    const Impl& impl = *m_impl;
    if (index < 0 || index >= impl.frameCount) {
        return nullptr;
    }
    return impl.frames.data() + static_cast<std::size_t>(index) *
                                    static_cast<std::size_t>(impl.config.joints);
}

const float* FloodDiffusionPipeline::FeatureFrame(int index) const {
    const Impl& impl = *m_impl;
    if (index < 0 || index >= impl.frameCount) {
        return nullptr;
    }
    return impl.features.data() + static_cast<std::size_t>(index) *
                                      static_cast<std::size_t>(impl.config.motionDim);
}

float FloodDiffusionPipeline::FrameRootAngle(int index) const {
    const Impl& impl = *m_impl;
    if (index < 0 || index >= static_cast<int>(impl.rootAngles.size())) {
        return 0.0f;
    }
    return impl.rootAngles[static_cast<std::size_t>(index)];
}

const std::string& FloodDiffusionPipeline::Prompt() const { return m_impl->prompt; }
int FloodDiffusionPipeline::PromptTokenCount() const { return m_impl->promptTokens; }
double FloodDiffusionPipeline::LastDenoiseMs() const { return m_impl->lastDenoiseMs; }
double FloodDiffusionPipeline::LastDecodeMs() const { return m_impl->lastDecodeMs; }
double FloodDiffusionPipeline::GenerationMs() const { return m_impl->generationMs; }
bool FloodDiffusionPipeline::TextEncoderLoaded() const {
    return m_impl->textEncoder != nullptr;
}

std::string FloodDiffusionPipeline::TokenisedPrompt() const {
    return m_impl->tokenizer.Decode(m_impl->promptIds);
}

}  // namespace NeuralModelIntegrateTestbed
