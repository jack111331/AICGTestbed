// Measures which input shapes the FloodDiffusion ONNX export should fix, so
// ONNX Runtime's DirectML EP can resolve shapes statically and stop
// dispatching the graph one operator at a time.
//
// It answers the question without needing a re-export, by using ORT's free
// dimension overrides (AddFreeDimensionOverrideByName) to pin named symbolic
// dimensions at session creation -- the same state a static export produces
// for shape inference. Re-run it after changing the export to check the
// prediction held.
//
// It also settles the two correctness questions a static export raises:
//
//   * Padding `ctx_len` up to text_len is EXACT, because context_lens already
//     masks the padding out of cross-attention. Section 3 measures the
//     difference against an unpadded run.
//   * Padding `time` is NOT exact. The denoiser is non-causal (causal: false),
//     so self-attention is bidirectional and the not-yet-denoised tail
//     attends into the live prefix. Section 4 measures how far the result
//     moves, which is why a fixed `time` needs a valid-length input and a
//     self-attention mask rather than padding alone.
//
// Findings as of the 2026-10-10 export, on an RTX 5080:
//
//   denoiser, symbolic              6.6-7.1 ms
//   pinning any ONE dimension       no change
//   pinning batch AND time          2.0 ms
//   pinning all four                1.7 ms
//   VAE first, batch symbolic       1.48 ms
//   VAE first, batch pinned         0.47 ms
//
#include <d3d12.h>
#include <wrl/client.h>

#include <DirectML.h>

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

ComPtr<ID3D12Device> g_device;
ComPtr<ID3D12CommandQueue> g_queue;
ComPtr<IDMLDevice> g_dmlDevice;
const OrtDmlApi* g_dmlApi = nullptr;

constexpr int64_t kLatentDim = 4;
constexpr int64_t kTextDim = 768;
constexpr int64_t kTextLen = 128;   // config.json text_len
constexpr int64_t kBatch = 2;       // classifier-free guidance
constexpr int64_t kSegments = 1;    // one prompt

double Since(std::chrono::steady_clock::time_point began) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - began).count();
}

std::string Shorten(const char* what) {
    std::string message = what == nullptr ? std::string() : std::string(what);
    if (message.size() > 220) message.resize(220);
    return message;
}

// Deterministic, so every configuration below sees identical values at
// identical positions and the outputs are comparable.
float Latent(int64_t index, int64_t component) {
    return static_cast<float>((index * kLatentDim + component) % 29 - 14) / 16.0f;
}
float ContextValue(int64_t token, int64_t channel) {
    return static_cast<float>((token * kTextDim + channel) % 23 - 11) / 64.0f;
}

struct Result {
    double milliseconds = 0.0;
    std::vector<float> prediction;  // [batch, time, latentDim]
    bool ok = false;
    std::string error;
};

// One denoiser call. `paddedContextLen` is the ctx_len axis actually fed;
// `realContextLen` is what context_lens declares, so the difference is
// padding the model should be masking away.
Result RunDenoiser(Ort::Session& session, int64_t time, int64_t paddedContextLen,
                   int64_t realContextLen) {
    Result result;
    Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<float> x(static_cast<std::size_t>(kBatch * time * kLatentDim));
    for (int64_t row = 0; row < kBatch; ++row) {
        for (int64_t i = 0; i < time; ++i) {
            for (int64_t c = 0; c < kLatentDim; ++c) {
                x[static_cast<std::size_t>((row * time + i) * kLatentDim + c)] =
                    Latent(i, c);
            }
        }
    }
    // A noise ramp like the sampler's: 1 + i/chunk - t, clipped.
    std::vector<float> t(static_cast<std::size_t>(kBatch * time));
    for (int64_t row = 0; row < kBatch; ++row) {
        for (int64_t i = 0; i < time; ++i) {
            const float level = 1.0f + static_cast<float>(i) / 5.0f - 1.5f;
            t[static_cast<std::size_t>(row * time + i)] =
                level < 0.0f ? 0.0f : (level > 1.0f ? 1.0f : level);
        }
    }
    std::vector<int64_t> textIdx(static_cast<std::size_t>(kBatch * time), 0);

    std::vector<float> context(
        static_cast<std::size_t>(kBatch * kSegments * paddedContextLen * kTextDim), 0.0f);
    for (int64_t row = 0; row < kBatch; ++row) {
        for (int64_t token = 0; token < realContextLen; ++token) {
            for (int64_t channel = 0; channel < kTextDim; ++channel) {
                context[static_cast<std::size_t>(
                    (row * paddedContextLen + token) * kTextDim + channel)] =
                    ContextValue(token, channel);
            }
        }
    }
    std::vector<int64_t> lens(static_cast<std::size_t>(kBatch * kSegments),
                              realContextLen);

    const int64_t xShape[] = {kBatch, time, kLatentDim};
    const int64_t tShape[] = {kBatch, time};
    const int64_t contextShape[] = {kBatch, kSegments, paddedContextLen, kTextDim};
    const int64_t lensShape[] = {kBatch, kSegments};

    const char* inputNames[] = {"x", "t", "context", "context_lens", "text_idx"};
    const char* outputNames[] = {"prediction"};

    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(cpu, x.data(), x.size(), xShape, 3));
        inputs.push_back(Ort::Value::CreateTensor<float>(cpu, t.data(), t.size(), tShape, 2));
        inputs.push_back(Ort::Value::CreateTensor<float>(cpu, context.data(),
                                                         context.size(), contextShape, 4));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, lens.data(), lens.size(),
                                                           lensShape, 2));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, textIdx.data(),
                                                           textIdx.size(), tShape, 2));
        const auto began = std::chrono::steady_clock::now();
        auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(),
                                   inputs.size(), outputNames, 1);
        result.milliseconds = Since(began);
        const float* out = outputs[0].GetTensorData<float>();
        const std::size_t count = static_cast<std::size_t>(kBatch * time * kLatentDim);
        result.prediction.assign(out, out + count);
        result.ok = true;
    } catch (const Ort::Exception& e) {
        result.error = Shorten(e.what());
    }
    return result;
}

// Largest absolute difference over the first `time` positions of row 0.
float Difference(const Result& a, const Result& b, int64_t time) {
    float worst = 0.0f;
    const std::size_t count = static_cast<std::size_t>(time * kLatentDim);
    if (a.prediction.size() < count || b.prediction.size() < count) {
        return -1.0f;
    }
    for (std::size_t i = 0; i < count; ++i) {
        worst = std::max(worst, std::fabs(a.prediction[i] - b.prediction[i]));
    }
    return worst;
}

struct Override {
    const char* name;
    int64_t value;
};

struct Config {
    const char* key;
    const char* value;
};

Ort::SessionOptions MakeOptions(bool directml, const std::vector<Override>& overrides,
                                const std::vector<Config>& configs = {}) {
    Ort::SessionOptions options;
    for (const Override& entry : overrides) {
        options.AddFreeDimensionOverrideByName(entry.name, entry.value);
    }
    // The DirectML EP's own session config keys. None of these appear in the
    // public headers; they are discoverable as strings in onnxruntime.dll:
    //   ep.dml.disable_graph_fusion, ep.dml.enable_graph_capture,
    //   ep.dml.enable_graph_serialization, ep.dml.disable_memory_arena,
    //   ep.dml.enable_cpu_sync_spinning
    for (const Config& entry : configs) {
        options.AddConfigEntry(entry.key, entry.value);
    }
    if (directml) {
        options.DisableMemPattern();
        options.SetExecutionMode(ORT_SEQUENTIAL);
        Ort::ThrowOnError(g_dmlApi->SessionOptionsAppendExecutionProvider_DML1(
            options, g_dmlDevice.Get(), g_queue.Get()));
    }
    return options;
}

// --- device-resident inputs ------------------------------------------------
//
// Section 10's apparatus. The sampler feeds host-memory tensors, so ONNX
// Runtime copies five inputs up and one output back inside every Run. This
// measures what that costs by doing the same work with the tensors already in
// D3D12 buffers, which is also the entry fee for graph capture.

// Written out rather than via the d3dx12 helpers, so this probe keeps its one
// dependency on //third:onnxruntime.
Microsoft::WRL::ComPtr<ID3D12Resource> MakeBuffer(uint64_t bytes,
                                                  D3D12_HEAP_TYPE heapType) {
    Microsoft::WRL::ComPtr<ID3D12Resource> out;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = heapType;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // DirectML requires its tensor buffers to allow unordered access.
    desc.Flags = heapType == D3D12_HEAP_TYPE_DEFAULT
                     ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                     : D3D12_RESOURCE_FLAG_NONE;
    const D3D12_RESOURCE_STATES state = heapType == D3D12_HEAP_TYPE_UPLOAD
                                            ? D3D12_RESOURCE_STATE_GENERIC_READ
                                            : D3D12_RESOURCE_STATE_COMMON;
    if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                 state, nullptr, IID_PPV_ARGS(&out)))) {
        return nullptr;
    }
    return out;
}

// One input or output living in a D3D12 buffer, wrapped as an Ort::Value.
struct DeviceTensor {
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    void* allocation = nullptr;
    std::unique_ptr<Ort::Value> value;
    uint64_t bytes = 0;

    ~DeviceTensor() {
        // The order matters: the Value points into the allocation.
        value.reset();
        if (allocation != nullptr && g_dmlApi != nullptr) {
            g_dmlApi->FreeGPUAllocation(allocation);
        }
    }

    bool Create(const Ort::MemoryInfo& memoryInfo, const std::vector<int64_t>& shape,
                ONNXTensorElementDataType type, uint64_t elementBytes) {
        uint64_t elements = 1;
        for (const int64_t dim : shape) {
            elements *= static_cast<uint64_t>(dim);
        }
        bytes = elements * elementBytes;
        buffer = MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT);
        if (!buffer) {
            return false;
        }
        if (g_dmlApi->CreateGPUAllocationFromD3DResource(buffer.Get(), &allocation) !=
            nullptr) {
            return false;
        }
        try {
            value = std::make_unique<Ort::Value>(Ort::Value::CreateTensor(
                memoryInfo, allocation, static_cast<std::size_t>(bytes), shape.data(),
                shape.size(), type));
        } catch (const Ort::Exception&) {
            return false;
        }
        return true;
    }
};

// Copies host bytes into each device tensor once, through one upload buffer per
// tensor and one command list. Not timed: the point of section 10 is what a
// Run costs once the data is already there.
bool UploadAll(const std::vector<std::pair<DeviceTensor*, const void*>>& tensors) {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    if (FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&allocator))) ||
        FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           allocator.Get(), nullptr,
                                           IID_PPV_ARGS(&list))) ||
        FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        return false;
    }
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> staging;
    for (const auto& entry : tensors) {
        DeviceTensor* tensor = entry.first;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload =
            MakeBuffer(tensor->bytes, D3D12_HEAP_TYPE_UPLOAD);
        if (!upload) {
            return false;
        }
        void* mapped = nullptr;
        if (FAILED(upload->Map(0, nullptr, &mapped))) {
            return false;
        }
        std::memcpy(mapped, entry.second, static_cast<std::size_t>(tensor->bytes));
        upload->Unmap(0, nullptr);
        list->CopyBufferRegion(tensor->buffer.Get(), 0, upload.Get(), 0, tensor->bytes);
        staging.push_back(std::move(upload));
    }
    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(fence.Get(), 1);
    // Blocking is fine here; this runs once per configuration.
    while (fence->GetCompletedValue() < 1) {
    }
    return true;
}

// Median of several runs, so one scheduling hiccup does not set the number.
double Median(Ort::Session& session, int64_t time, int64_t padded, int64_t real,
              int runs, Result* last) {
    std::vector<double> samples;
    for (int i = 0; i < runs; ++i) {
        Result r = RunDenoiser(session, time, padded, real);
        if (!r.ok) {
            std::printf("      FAILED: %s\n", r.error.c_str());
            return -1.0;
        }
        samples.push_back(r.milliseconds);
        if (last != nullptr) *last = std::move(r);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "shape-probe");
    const wchar_t* path = L"resources/FloodDiffusion/denoiser.onnx";

    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_device)))) {
        std::printf("no D3D12 device\n");
        return 1;
    }
    D3D12_COMMAND_QUEUE_DESC desc = {};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    g_device->CreateCommandQueue(&desc, IID_PPV_ARGS(&g_queue));
    if (FAILED(DMLCreateDevice(g_device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                               IID_PPV_ARGS(&g_dmlDevice)))) {
        std::printf("no DirectML device\n");
        return 1;
    }
    Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
                                          reinterpret_cast<const void**>(&g_dmlApi));
    if (g_dmlApi == nullptr) {
        std::printf("onnxruntime.dll is not the DirectML build\n");
        return 1;
    }

    constexpr int kRuns = 7;
    constexpr int64_t kTime = 16;
    constexpr int64_t kRealContext = 12;

    // ---------------------------------------------------------------- 1 ----
    // The dynamic export as it stands, both providers, as the baseline.
    Result dynamicReference;
    std::printf("1. as exported, all dimensions symbolic\n");
    for (const bool directml : {true, false}) {
        Ort::Session session(env, path, MakeOptions(directml, {}));
        Result last;
        const double ms = Median(session, kTime, kRealContext, kRealContext, kRuns, &last);
        std::printf("   %-10s time=%lld ctx=%lld  %7.2f ms   prediction[0]=%.6f\n",
                    directml ? "DirectML" : "CPU", static_cast<long long>(kTime),
                    static_cast<long long>(kRealContext), ms,
                    last.ok ? last.prediction[0] : 0.0f);
        if (!directml) dynamicReference = std::move(last);
    }

    // ---------------------------------------------------------------- 2 ----
    // Pin the dimensions one group at a time, to see which one matters.
    std::printf("\n2. DirectML with free dimensions pinned (no re-export)\n");
    const struct {
        const char* label;
        std::vector<Override> overrides;
    } configurations[] = {
        {"batch=2",                    {{"batch", kBatch}}},
        {"batch, segments",            {{"batch", kBatch}, {"segments", kSegments}}},
        {"batch, segments, ctx_len",   {{"batch", kBatch}, {"segments", kSegments},
                                        {"ctx_len", kRealContext}}},
        // The isolation that decides how invasive a re-export has to be: if
        // `time` alone buys the speedup, nothing else needs fixing.
        {"time=16 ALONE",              {{"time", kTime}}},
        {"time, batch",                {{"time", kTime}, {"batch", kBatch}}},
        {"all four, time=16",          {{"batch", kBatch}, {"segments", kSegments},
                                        {"ctx_len", kRealContext}, {"time", kTime}}},
    };
    for (const auto& configuration : configurations) {
        try {
            Ort::Session session(env, path, MakeOptions(true, configuration.overrides));
            Result last;
            const double ms = Median(session, kTime, kRealContext, kRealContext, kRuns,
                                     &last);
            if (ms < 0.0) continue;
            const float drift = last.ok ? Difference(last, dynamicReference, kTime) : -1.0f;
            std::printf("   %-28s %7.2f ms   max diff vs CPU %.6f\n",
                        configuration.label, ms, drift);
        } catch (const Ort::Exception& e) {
            std::printf("   %-28s session failed: %s\n", configuration.label,
                        Shorten(e.what()).c_str());
        }
    }

    // ---------------------------------------------------------------- 3 ----
    // Is padding ctx_len to text_len exact? context_lens already masks it, so
    // a padded run should agree with an unpadded one to float noise.
    std::printf("\n3. padding ctx_len from %lld to text_len %lld\n",
                static_cast<long long>(kRealContext), static_cast<long long>(kTextLen));
    {
        Ort::Session padded(env, path,
                            MakeOptions(true, {{"batch", kBatch},
                                               {"segments", kSegments},
                                               {"ctx_len", kTextLen},
                                               {"time", kTime}}));
        Result last;
        const double ms = Median(padded, kTime, kTextLen, kRealContext, kRuns, &last);
        const float drift = last.ok ? Difference(last, dynamicReference, kTime) : -1.0f;
        std::printf("   DirectML, ctx padded to %lld  %7.2f ms   max diff vs "
                    "unpadded CPU %.6f\n", static_cast<long long>(kTextLen), ms, drift);
        std::printf("   (near zero means context_lens masks the padding, so a "
                    "fixed ctx_len is exact)\n");
    }

    // ---------------------------------------------------------------- 4 ----
    // Is padding `time` exact? The model is non-causal, so feeding the whole
    // buffer instead of the live prefix lets the noisy tail attend into it.
    std::printf("\n4. feeding the whole latent buffer instead of the live prefix\n");
    {
        const int64_t total = 65;  // 60 latents + one chunk of lookahead
        Ort::Session full(env, path, MakeOptions(false, {}));
        Result prefix = RunDenoiser(full, kTime, kRealContext, kRealContext);
        Result whole = RunDenoiser(full, total, kRealContext, kRealContext);
        if (prefix.ok && whole.ok) {
            const float drift = Difference(prefix, whole, kTime);
            std::printf("   CPU, same first %lld latents: max diff %.6f\n",
                        static_cast<long long>(kTime), drift);
            std::printf("   (large means a fixed `time` CHANGES the result and needs "
                        "a self-attention\n    mask over the padding, not just "
                        "padding)\n");
        } else {
            std::printf("   failed: %s%s\n", prefix.error.c_str(), whole.error.c_str());
        }
    }

    // ---------------------------------------------------------------- 5 ----
    // If `time` must be fixed, what does one fixed length cost per call, and
    // what would a whole 128-step generation cost?
    std::printf("\n5. DirectML fully static, at each candidate fixed `time`\n");
    for (const int64_t time : {16, 32, 65}) {
        try {
            Ort::Session session(env, path,
                                 MakeOptions(true, {{"batch", kBatch},
                                                    {"segments", kSegments},
                                                    {"ctx_len", kTextLen},
                                                    {"time", time}}));
            const double ms = Median(session, time, kTextLen, kRealContext, kRuns, nullptr);
            std::printf("   time=%-3lld ctx=%lld  %7.2f ms per call  -> %6.0f ms "
                        "for 128 steps\n", static_cast<long long>(time),
                        static_cast<long long>(kTextLen), ms, ms * 128.0);
        } catch (const Ort::Exception& e) {
            std::printf("   time=%-3lld session failed: %s\n",
                        static_cast<long long>(time), Shorten(e.what()).c_str());
        }
    }

    // ---------------------------------------------------------------- 6 ----
    // The VAE decoders run once per latent, 60 times a generation, and their
    // only symbolic dimension is `batch`. Everything else is already fixed, so
    // pinning batch=1 should be the whole story for them.
    std::printf("\n6. VAE decoder (first), DirectML, batch symbolic vs pinned\n");
    for (const bool pinned : {false, true}) {
        std::vector<Override> overrides;
        if (pinned) overrides.push_back({"batch", 1});
        try {
            Ort::Session first(env, L"resources/FloodDiffusion/vae_decoder_first.onnx",
                               MakeOptions(true, overrides));
            Ort::Session step(env, L"resources/FloodDiffusion/vae_decoder_step.onnx",
                              MakeOptions(true, overrides));

            std::vector<std::string> inNames, outNames;
            for (int i = 0; i < 20; ++i) {
                inNames.push_back("cache_in_" + std::to_string(i));
                outNames.push_back("cache_out_" + std::to_string(i));
            }
            std::vector<const char*> firstOut{"motion"}, stepIn{"z"}, stepOut{"motion"};
            for (int i = 0; i < 20; ++i) {
                firstOut.push_back(outNames[i].c_str());
                stepIn.push_back(inNames[i].c_str());
                stepOut.push_back(outNames[i].c_str());
            }

            Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                             OrtMemTypeCPU);
            const int64_t zShape[] = {1, 1, kLatentDim};
            std::vector<float> z(kLatentDim, 0.25f);
            const char* zName[] = {"z"};

            // vae_decoder_first only, run repeatedly. Threading the 20 caches
            // through vae_decoder_step is what the pipeline does, but it is
            // also extra plumbing that can fail for reasons unrelated to the
            // shapes; this isolates the shape effect.
            (void)step;
            std::vector<double> samples;
            float last = 0.0f;
            for (int s = 0; s < 16; ++s) {
                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(cpu, z.data(), z.size(),
                                                                 zShape, 3));
                const auto began = std::chrono::steady_clock::now();
                auto outputs = first.Run(Ort::RunOptions{nullptr}, zName, inputs.data(), 1,
                                         firstOut.data(), firstOut.size());
                samples.push_back(Since(began));
                last = outputs[0].GetTensorData<float>()[0];
            }
            std::sort(samples.begin(), samples.end());
            std::printf("   batch %-9s %6.2f ms per call -> %5.0f ms for 60 latents"
                        "   motion[0]=%.6f\n", pinned ? "pinned" : "symbolic",
                        samples[samples.size() / 2], samples[samples.size() / 2] * 60.0,
                        last);
        } catch (const Ort::Exception& e) {
            std::printf("   batch %-9s failed: %s\n", pinned ? "pinned" : "symbolic",
                        Shorten(e.what()).c_str());
        }
    }

    // ---------------------------------------------------------------- 7 ----
    // Is DmlGraphFusion what makes the static case fast? The EP has a
    // DmlGraphFusionTransformer that compiles each DirectML-assignable
    // partition into one IDMLCompiledOperator, dispatched as a single
    // DmlFusedNode_<n>. If that is the mechanism, turning it off should take
    // the static configuration back to roughly the dynamic cost.
    std::printf("\n7. DmlGraphFusion, on vs off, DirectML\n");
    {
        const std::vector<Override> staticDims = {{"batch", kBatch},
                                                  {"segments", kSegments},
                                                  {"ctx_len", kRealContext},
                                                  {"time", kTime}};
        const struct {
            const char* label;
            std::vector<Override> overrides;
            std::vector<Config> configs;
        } cases[] = {
            {"dynamic shapes, fusion on",  {},         {}},
            {"dynamic shapes, fusion OFF", {},         {{"ep.dml.disable_graph_fusion", "1"}}},
            {"static shapes, fusion on",   staticDims, {}},
            {"static shapes, fusion OFF",  staticDims, {{"ep.dml.disable_graph_fusion", "1"}}},
        };
        for (const auto& entry : cases) {
            try {
                Ort::Session session(env, path,
                                     MakeOptions(true, entry.overrides, entry.configs));
                const double ms = Median(session, kTime, kRealContext, kRealContext,
                                         kRuns, nullptr);
                std::printf("   %-30s %7.2f ms\n", entry.label, ms);
            } catch (const Ort::Exception& e) {
                std::printf("   %-30s session failed: %s\n", entry.label,
                            Shorten(e.what()).c_str());
            }
        }
        std::printf("   (if OFF is much slower only in the static row, fusion is "
                    "the mechanism\n    and it needs static shapes to engage)\n");
    }

    // ---------------------------------------------------------------- 8 ----
    // Does the profile show fused nodes once shapes are static? This also
    // tells us whether the earlier per-node profile -- 1226 individual
    // DirectML nodes -- was describing the real execution or an artifact of
    // profiling suppressing fusion.
    std::printf("\n8. provider assignment from the profile, dynamic vs static\n");
    const struct {
        const wchar_t* prefix;
        const char* label;
        std::vector<Override> overrides;
    } profiles[] = {
        {L"fd_dynamic", "dynamic", {}},
        // The minimum an export would have to fix, from section 2.
        {L"fd_bt", "batch+time", {{"batch", kBatch}, {"time", kTime}}},
        {L"fd_static", "all four", {{"batch", kBatch}, {"segments", kSegments},
                                    {"ctx_len", kRealContext}, {"time", kTime}}},
    };
    for (const auto& profile : profiles) {
        Ort::SessionOptions options = MakeOptions(true, profile.overrides);
        options.EnableProfiling(profile.prefix);
        try {
            Ort::Session session(env, path, options);
            RunDenoiser(session, kTime, kRealContext, kRealContext);
            RunDenoiser(session, kTime, kRealContext, kRealContext);
            const std::string file = session.EndProfilingAllocated(
                Ort::AllocatorWithDefaultOptions()).get();
            std::printf("   %-11s profile -> %s\n", profile.label, file.c_str());
        } catch (const Ort::Exception& e) {
            std::printf("   %-11s profiling failed: %s\n", profile.label,
                        Shorten(e.what()).c_str());
        }
    }

    // ---------------------------------------------------------------- 9 ----
    // ep.dml.enable_graph_capture records the whole GPU sequence once and
    // replays it, which is the mechanism that removes per-dispatch cost
    // outright. Report what it does here, including whether it refuses:
    // capture normally requires static shapes AND inputs and outputs at fixed
    // device addresses, which means IoBinding rather than host-memory tensors.
    std::printf("\n9. ep.dml.enable_graph_capture, with host-memory tensors\n");
    {
        const std::vector<Override> staticDims = {{"batch", kBatch},
                                                  {"segments", kSegments},
                                                  {"ctx_len", kRealContext},
                                                  {"time", kTime}};
        try {
            Ort::Session session(env, path,
                                 MakeOptions(true, staticDims,
                                             {{"ep.dml.enable_graph_capture", "1"}}));
            const double ms = Median(session, kTime, kRealContext, kRealContext, kRuns,
                                     nullptr);
            std::printf("   static shapes, capture on      %7.2f ms\n", ms);
            std::printf("   (no error means the session accepted it; whether it "
                        "actually captured\n    needs IoBinding to device memory "
                        "-- see the skill)\n");
        } catch (const Ort::Exception& e) {
            std::printf("   REFUSED: %s\n", Shorten(e.what()).c_str());
        }
    }

    // --------------------------------------------------------------- 10 ----
    // What the host-memory tensors cost. The sampler keeps its latents on the
    // CPU, so every Run copies five inputs up and one output back; this is the
    // same call with the tensors already in D3D12 buffers, bound by
    // Ort::IoBinding, which is also what graph capture requires.
    //
    // Measured at both sizes on purpose. The probe's own 16x12 is small enough
    // that the copies could be dismissed as noise; the shipped sampler submits
    // time=60+chunk and a context padded to text_len, and almost all of the
    // bytes are that context -- which is CONSTANT for a whole generation and
    // re-uploaded 128 times by the host-memory path.
    std::printf("\n10. host-memory tensors vs device-resident (IoBinding)\n");
    const struct {
        const char* label;
        int64_t time;
        int64_t ctx;
    } bindingSizes[] = {
        {"probe size", kTime, kRealContext},
        {"shipped size", 65, kTextLen},
    };
    for (const auto& size : bindingSizes) {
        const int64_t time = size.time;
        const int64_t ctx = size.ctx;
        const std::vector<Override> staticDims = {{"batch", kBatch},
                                                  {"segments", kSegments},
                                                  {"ctx_len", ctx},
                                                  {"time", time}};
        const uint64_t perStepBytes =
            static_cast<uint64_t>(kBatch * time * kLatentDim) * sizeof(float) +
            static_cast<uint64_t>(kBatch * time) * sizeof(float) +
            static_cast<uint64_t>(kBatch * kSegments * ctx * kTextDim) * sizeof(float) +
            static_cast<uint64_t>(kBatch * kSegments) * sizeof(int64_t) +
            static_cast<uint64_t>(kBatch * time) * sizeof(int64_t);
        const uint64_t contextBytes =
            static_cast<uint64_t>(kBatch * kSegments * ctx * kTextDim) * sizeof(float);
        std::printf("   %s: time=%lld ctx_len=%lld, %.1f KB per call "
                    "(%.0f%% of it context)\n",
                    size.label, static_cast<long long>(time),
                    static_cast<long long>(ctx),
                    static_cast<double>(perStepBytes) / 1024.0,
                    100.0 * static_cast<double>(contextBytes) /
                        static_cast<double>(perStepBytes));

        Result hostReference;
        double hostMs = -1.0;
        try {
            Ort::Session session(env, path, MakeOptions(true, staticDims));
            hostMs = Median(session, time, ctx, ctx, kRuns, &hostReference);
            std::printf("      host memory          %7.2f ms\n", hostMs);
        } catch (const Ort::Exception& e) {
            std::printf("      host memory failed: %s\n", Shorten(e.what()).c_str());
        }

        // The same inputs the host path builds, so the predictions can be
        // compared rather than merely timed.
        std::vector<float> x(static_cast<std::size_t>(kBatch * time * kLatentDim));
        for (int64_t row = 0; row < kBatch; ++row) {
            for (int64_t i = 0; i < time; ++i) {
                for (int64_t c = 0; c < kLatentDim; ++c) {
                    x[static_cast<std::size_t>((row * time + i) * kLatentDim + c)] =
                        Latent(i, c);
                }
            }
        }
        std::vector<float> t(static_cast<std::size_t>(kBatch * time));
        for (int64_t row = 0; row < kBatch; ++row) {
            for (int64_t i = 0; i < time; ++i) {
                const float level = 1.0f + static_cast<float>(i) / 5.0f - 1.5f;
                t[static_cast<std::size_t>(row * time + i)] =
                    level < 0.0f ? 0.0f : (level > 1.0f ? 1.0f : level);
            }
        }
        std::vector<int64_t> textIdx(static_cast<std::size_t>(kBatch * time), 0);
        std::vector<float> context(
            static_cast<std::size_t>(kBatch * kSegments * ctx * kTextDim), 0.0f);
        for (int64_t row = 0; row < kBatch; ++row) {
            for (int64_t token = 0; token < ctx; ++token) {
                for (int64_t channel = 0; channel < kTextDim; ++channel) {
                    context[static_cast<std::size_t>(
                        (row * ctx + token) * kTextDim + channel)] =
                        ContextValue(token, channel);
                }
            }
        }
        std::vector<int64_t> lens(static_cast<std::size_t>(kBatch * kSegments), ctx);

        // With capture off, then on: capture is the reason to bind in the first
        // place, so measuring one without the other would miss the point.
        for (const bool capture : {false, true}) {
            std::vector<Config> configs;
            if (capture) {
                configs.push_back({"ep.dml.enable_graph_capture", "1"});
            }
            try {
                Ort::Session session(env, path, MakeOptions(true, staticDims, configs));
                const Ort::MemoryInfo memoryInfo("DML", OrtDeviceAllocator, 0,
                                                 OrtMemTypeDefault);
                DeviceTensor dx, dt, dcontext, dlens, dtextIdx, dprediction;
                const bool made =
                    dx.Create(memoryInfo, {kBatch, time, kLatentDim},
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)) &&
                    dt.Create(memoryInfo, {kBatch, time},
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)) &&
                    dcontext.Create(memoryInfo, {kBatch, kSegments, ctx, kTextDim},
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)) &&
                    dlens.Create(memoryInfo, {kBatch, kSegments},
                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)) &&
                    dtextIdx.Create(memoryInfo, {kBatch, time},
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                    sizeof(int64_t)) &&
                    dprediction.Create(memoryInfo, {kBatch, time, kLatentDim},
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                       sizeof(float));
                if (!made) {
                    std::printf("      %-20s could not create the device tensors\n",
                                capture ? "device + capture" : "device-resident");
                    continue;
                }
                if (!UploadAll({{&dx, x.data()},
                                {&dt, t.data()},
                                {&dcontext, context.data()},
                                {&dlens, lens.data()},
                                {&dtextIdx, textIdx.data()}})) {
                    std::printf("      upload failed\n");
                    continue;
                }

                Ort::IoBinding binding(session);
                binding.BindInput("x", *dx.value);
                binding.BindInput("t", *dt.value);
                binding.BindInput("context", *dcontext.value);
                binding.BindInput("context_lens", *dlens.value);
                binding.BindInput("text_idx", *dtextIdx.value);
                binding.BindOutput("prediction", *dprediction.value);

                // A real completion wait, on the queue the EP submits to.
                //
                // SynchronizeOutputs is NOT one here: with the output already
                // in device memory there is nothing for it to copy back, so it
                // returns as soon as the work is submitted and the timer then
                // measures submission rather than execution. Measured that
                // way this configuration looked 5x faster than host memory,
                // which the end-to-end generation in
                // //Engine:flooddiffusion-smoke flatly contradicted -- the
                // host path waits, because reading the prediction back forces
                // it to.
                Microsoft::WRL::ComPtr<ID3D12Fence> fence;
                if (FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                                 IID_PPV_ARGS(&fence)))) {
                    std::printf("      could not create a fence\n");
                    continue;
                }
                uint64_t fenceValue = 0;
                std::vector<double> samples;
                for (int i = 0; i < kRuns; ++i) {
                    const auto began = std::chrono::steady_clock::now();
                    session.Run(Ort::RunOptions{nullptr}, binding);
                    binding.SynchronizeOutputs();
                    g_queue->Signal(fence.Get(), ++fenceValue);
                    while (fence->GetCompletedValue() < fenceValue) {
                    }
                    samples.push_back(Since(began));
                }
                std::sort(samples.begin(), samples.end());
                const double ms = samples[samples.size() / 2];
                std::printf("      %-20s %7.2f ms", capture ? "device + capture"
                                                            : "device-resident", ms);
                if (hostMs > 0.0) {
                    std::printf("   %+.2f ms (%+.0f%%) vs host memory", ms - hostMs,
                                100.0 * (ms - hostMs) / hostMs);
                }
                std::printf("\n");
            } catch (const Ort::Exception& e) {
                std::printf("      %-20s REFUSED: %s\n",
                            capture ? "device + capture" : "device-resident",
                            Shorten(e.what()).c_str());
            }
        }
    }

    // --------------------------------------------------------------- 11 ----
    // The VAE decode, which is where this pipeline's wall time actually is:
    // vae_decoder_step runs 59 times of 60 per generation against the
    // denoiser's 128, and each call costs several times as much.
    //
    // Section 6 left two things wrong about it, and this section is what
    // corrected them, so it measures each claim rather than asserting any.
    //
    //  * Section 6 timed vae_decoder_FIRST, which runs ONCE. Its 1.48 -> 0.47
    //    ms from pinning `batch` is worth a millisecond a generation, not the
    //    sixty it was read as. The 59-call cost is vae_decoder_step's.
    //  * "A statically shaped session cannot hand its outputs to another" is
    //    not the failure. Pinning vae_decoder_first makes the DirectML EP
    //    return a NULL for cache_out_9; the hand-off is the innocent party,
    //    and `first non-tensor` below says so directly.
    //
    // Three groups, in the order they were needed: the hand-off through the
    // ordinary Run API with each session pinned or not and on either provider;
    // the step graph alone, to show the pin is not what breaks; and
    // Ort::IoBinding, which was the plan and does not work here at all.
    std::printf("\n11. the VAE decode: pinning, providers, and IoBinding\n");
    {
        constexpr int kCaches = 20;
        constexpr int64_t kMotionDim = 263;
        constexpr int64_t kFramesFirst = 1;
        constexpr int64_t kFramesStep = 4;
        // From pipeline.txt: cache 0 is 4x2, the other nineteen are 512x2.
        const auto cacheShape = [](int i) -> std::vector<int64_t> {
            return {1, i == 0 ? 4 : 512, 2};
        };
        const std::vector<Override> pinned = {{"batch", 1}};

        std::vector<std::string> inNames, outNames;
        for (int i = 0; i < kCaches; ++i) {
            inNames.push_back("cache_in_" + std::to_string(i));
            outNames.push_back("cache_out_" + std::to_string(i));
        }

        // The hand-off itself, through the ORDINARY Run API, both sessions
        // pinned. This is what section 6 skipped and what the pipeline's
        // comment recorded as impossible; it reports which of the twenty cache
        // outputs is not a tensor, if any, rather than only that one is not.
        //
        // Which SESSION is pinned turns out to decide it, and the two are not
        // equally worth pinning: vae_decoder_step runs 59 times a generation
        // and vae_decoder_first exactly once.
        // The CPU rows are here because the header's one-off numbers claimed
        // the VAE step is faster on the CPU provider than on DirectML, and if
        // that holds over a threaded sequence it is a far bigger lever than
        // either pin.
        //
        // The last two rows are the ROOT-CAUSE test rather than a shippable
        // configuration. `cache_out_9` is the only one of the twenty produced
        // by a ConstantOfShape rather than a Slice, and `cache_out_14` is an
        // Identity of it. Pinning `batch` makes its shape input static, so
        // ORT's ConstantFolding evaluates the node and replaces its output
        // with a graph INITIALIZER -- and an initializer that is also a graph
        // output is the case that comes back empty. If that is the mechanism,
        // disabling that one optimizer has to make the nulls go away.
        const struct {
            const char* label;
            bool pinFirst;
            bool pinStep;
            bool directml;
            bool noConstantFolding;
        } handOffs[] = {
            {"neither",      false, false, true,  false},
            {"both",         true,  true,  true,  false},
            {"step only",    false, true,  true,  false},
            {"first only",   true,  false, true,  false},
            {"CPU, none",    false, false, false, false},
            {"CPU, both",    true,  true,  false, false},
            {"both, no CF",  true,  true,  true,  true},
            {"first, no CF", true,  false, true,  true},
        };
        for (const auto& handOff : handOffs) {
            try {
                std::vector<Config> handOffConfigs;
                if (handOff.noConstantFolding) {
                    handOffConfigs.push_back(
                        {"optimization.disable_specified_optimizers", "ConstantFolding"});
                }
                Ort::Session first(env,
                                   L"resources/FloodDiffusion/vae_decoder_first.onnx",
                                   MakeOptions(handOff.directml,
                                               handOff.pinFirst
                                                   ? pinned
                                                   : std::vector<Override>{},
                                               handOffConfigs));
                Ort::Session step(env,
                                  L"resources/FloodDiffusion/vae_decoder_step.onnx",
                                  MakeOptions(handOff.directml,
                                              handOff.pinStep
                                                  ? pinned
                                                  : std::vector<Override>{},
                                              handOffConfigs));
                Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                                 OrtMemTypeCPU);
                const int64_t zShape[] = {1, 1, kLatentDim};
                std::vector<float> zHost(kLatentDim, 0.25f);
                std::vector<const char*> firstOut{"motion"}, stepIn{"z"},
                    stepOut{"motion"};
                for (int i = 0; i < kCaches; ++i) {
                    firstOut.push_back(outNames[i].c_str());
                    stepIn.push_back(inNames[i].c_str());
                    stepOut.push_back(outNames[i].c_str());
                }
                const char* zName[] = {"z"};
                Ort::Value zValue = Ort::Value::CreateTensor<float>(
                    cpu, zHost.data(), zHost.size(), zShape, 3);
                auto firstOutputs =
                    first.Run(Ort::RunOptions{nullptr}, zName, &zValue, 1,
                              firstOut.data(), firstOut.size());
                // EVERY empty output, by name rather than index: the Identity
                // of cache_out_9 should come back empty too, and a second name
                // in this list is what ties the two together.
                std::string empties;
                for (std::size_t i = 0; i < firstOutputs.size(); ++i) {
                    if (!firstOutputs[i].IsTensor()) {
                        if (!empties.empty()) empties += ",";
                        empties += firstOut[i];
                    }
                }
                std::printf("   hand-off, pinned %-12s first gave %zu outputs, "
                            "empty: %s\n", handOff.label, firstOutputs.size(),
                            empties.empty() ? "none" : empties.c_str());

                std::vector<Ort::Value> stepInputs;
                stepInputs.push_back(Ort::Value::CreateTensor<float>(
                    cpu, zHost.data(), zHost.size(), zShape, 3));
                for (std::size_t i = 1; i < firstOutputs.size(); ++i) {
                    stepInputs.push_back(std::move(firstOutputs[i]));
                }
                std::vector<double> samples;
                std::vector<Ort::Value> stepOutputs;
                for (int run = 0; run < kRuns; ++run) {
                    const auto began = std::chrono::steady_clock::now();
                    stepOutputs = step.Run(Ort::RunOptions{nullptr}, stepIn.data(),
                                           stepInputs.data(), stepInputs.size(),
                                           stepOut.data(), stepOut.size());
                    samples.push_back(Since(began));
                    // Thread them, exactly as the pipeline does.
                    for (std::size_t i = 1; i < stepOutputs.size(); ++i) {
                        stepInputs[i] = std::move(stepOutputs[i]);
                    }
                }
                std::sort(samples.begin(), samples.end());
                std::printf("   hand-off, pinned %-12s step %6.2f ms per call "
                            "-> %5.0f ms for 59 calls\n",
                            handOff.label, samples[samples.size() / 2],
                            samples[samples.size() / 2] * 59.0);
            } catch (const Ort::Exception& e) {
                std::printf("   hand-off, pinned %-12s REFUSED: %s\n",
                            handOff.label, Shorten(e.what()).c_str());
            }
        }

        // Is the hand-off the problem, or the pinned step graph itself?
        // Decisive, because it needs no second session at all: feed the step
        // decoder twenty ordinary host-memory cache tensors it has never seen
        // and see whether it still loses one.
        for (const bool pinStep : {false, true}) {
            for (const bool fusion : {true, false}) {
                if (!pinStep && !fusion) continue;  // nothing to fuse anyway
                std::vector<Config> configs;
                if (!fusion) configs.push_back({"ep.dml.disable_graph_fusion", "1"});
                try {
                    Ort::Session step(env,
                                      L"resources/FloodDiffusion/vae_decoder_step.onnx",
                                      MakeOptions(true, pinStep ? pinned
                                                                : std::vector<Override>{},
                                                  configs));
                    std::vector<std::vector<float>> hostCaches;
                    for (int i = 0; i < kCaches; ++i) {
                        const std::vector<int64_t> shape = cacheShape(i);
                        hostCaches.emplace_back(
                            static_cast<std::size_t>(shape[1] * shape[2]), 0.1f);
                    }
                    Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                                     OrtMemTypeCPU);
                    const int64_t zShape[] = {1, 1, kLatentDim};
                    std::vector<float> zHost(kLatentDim, 0.25f);
                    std::vector<Ort::Value> inputs;
                    inputs.push_back(Ort::Value::CreateTensor<float>(
                        cpu, zHost.data(), zHost.size(), zShape, 3));
                    for (int i = 0; i < kCaches; ++i) {
                        const std::vector<int64_t> shape = cacheShape(i);
                        inputs.push_back(Ort::Value::CreateTensor<float>(
                            cpu, hostCaches[static_cast<std::size_t>(i)].data(),
                            hostCaches[static_cast<std::size_t>(i)].size(), shape.data(),
                            shape.size()));
                    }
                    std::vector<const char*> stepIn{"z"}, stepOut{"motion"};
                    for (int i = 0; i < kCaches; ++i) {
                        stepIn.push_back(inNames[i].c_str());
                        stepOut.push_back(outNames[i].c_str());
                    }
                    const auto began = std::chrono::steady_clock::now();
                    step.Run(Ort::RunOptions{nullptr}, stepIn.data(), inputs.data(),
                             inputs.size(), stepOut.data(), stepOut.size());
                    std::printf("   step alone, batch %-8s fusion %-3s %6.2f ms\n",
                                pinStep ? "pinned" : "symbolic", fusion ? "on" : "off",
                                Since(began));
                } catch (const Ort::Exception& e) {
                    std::printf("   step alone, batch %-8s fusion %-3s REFUSED: %s\n",
                                pinStep ? "pinned" : "symbolic", fusion ? "on" : "off",
                                Shorten(e.what()).c_str());
                }
            }
        }

        // Fusion on and off, because the refusal below names DmlFusedNode_0_0
        // and the question is whether the fused node is what loses a
        // bound input.
        // Ort::IoBinding, which was the plan: bind the twenty caches to
        // D3D12 buffers both sessions share, so nothing is handed between them
        // and ORT never sees an output become an input. It does not work, in
        // any configuration, and the control row is what says why -- the same
        // twenty values in HOST memory, bound rather than passed, fail
        // identically, so it is the binding and not the memory.
        //
        // The graphs are all float32 and every dimension is concrete once
        // `batch` is pinned: z is [1,1,4], motion [1,1,263] and [1,4,263],
        // cache 0 is [1,4,2] and the other nineteen [1,512,2]. Nothing about
        // the shapes explains the refusals below.
        const struct { const char* label; bool pin; bool fusion; } bindCases[] = {
            {"batch pinned, fusion on",  true,  true},
            {"batch pinned, fusion off", true,  false},
            {"batch symbolic",           false, true},
        };
        for (const auto& bindCase : bindCases) {
        std::vector<Config> bindConfigs;
        if (!bindCase.fusion) {
            bindConfigs.push_back({"ep.dml.disable_graph_fusion", "1"});
        }
        const std::vector<Override> bindDims =
            bindCase.pin ? pinned : std::vector<Override>{};
        std::printf("   --- %s ---\n", bindCase.label);
        try {
            Ort::Session first(env, L"resources/FloodDiffusion/vae_decoder_first.onnx",
                               MakeOptions(true, bindDims, bindConfigs));
            Ort::Session step(env, L"resources/FloodDiffusion/vae_decoder_step.onnx",
                              MakeOptions(true, bindDims, bindConfigs));

            const Ort::MemoryInfo memoryInfo("DML", OrtDeviceAllocator, 0,
                                             OrtMemTypeDefault);
            DeviceTensor dz;
            DeviceTensor dmotionFirst, dmotionStep;
            std::vector<std::unique_ptr<DeviceTensor>> setA, setB;
            bool made = dz.Create(memoryInfo, {1, 1, kLatentDim},
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)) &&
                        dmotionFirst.Create(memoryInfo, {1, kFramesFirst, kMotionDim},
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                            sizeof(float)) &&
                        dmotionStep.Create(memoryInfo, {1, kFramesStep, kMotionDim},
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           sizeof(float));
            for (int i = 0; i < kCaches && made; ++i) {
                setA.push_back(std::make_unique<DeviceTensor>());
                setB.push_back(std::make_unique<DeviceTensor>());
                made = setA.back()->Create(memoryInfo, cacheShape(i),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           sizeof(float)) &&
                       setB.back()->Create(memoryInfo, cacheShape(i),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           sizeof(float));
            }
            if (!made) {
                std::printf("   could not create the device tensors\n");
                continue;
            }
            std::vector<float> z(kLatentDim, 0.25f);
            if (!UploadAll({{&dz, z.data()}})) {
                std::printf("   upload failed\n");
                continue;
            }

            Microsoft::WRL::ComPtr<ID3D12Fence> fence;
            g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
            uint64_t fenceValue = 0;
            const auto wait = [&] {
                g_queue->Signal(fence.Get(), ++fenceValue);
                while (fence->GetCompletedValue() < fenceValue) {
                }
            };

            // One configuration at a time, so a refusal names the tensor class
            // responsible rather than just "a binding".
            //
            //   caches only      outputs ORT allocates for motion
            //   caches + motion  everything device-resident
            //   host z           nothing device-resident at all
            //   device z         only z, every output left to ORT
            //   caches only      outputs ORT allocates for motion
            //   caches + motion  everything device-resident
            enum Which { kHostZ, kDeviceZ, kCaches, kCachesMotion, kHostCaches };
            Ort::MemoryInfo cpuInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                                 OrtMemTypeCPU);
            const int64_t zShape[] = {1, 1, kLatentDim};
            // Host-memory cache tensors bound through IoBinding, as the
            // control: the same twenty values the ordinary Run above accepted,
            // delivered the other way.
            std::vector<std::vector<float>> hostCacheValues;
            for (int i = 0; i < kCaches; ++i) {
                const std::vector<int64_t> shape = cacheShape(i);
                hostCacheValues.emplace_back(
                    static_cast<std::size_t>(shape[1] * shape[2]), 0.1f);
            }
            for (const Which which :
                 {kHostZ, kDeviceZ, kCaches, kCachesMotion, kHostCaches}) {
                const char* label = which == kHostZ      ? "host z"
                                    : which == kDeviceZ  ? "device z"
                                    : which == kCaches   ? "caches only"
                                    : which == kCachesMotion ? "caches + motion"
                                                         : "host caches, bound";
                const bool bindCaches = which == kCaches ||
                                        which == kCachesMotion ||
                                        which == kHostCaches;
                const bool bindMotion = which == kCachesMotion;
                Ort::Value hostZ = Ort::Value::CreateTensor<float>(
                    cpuInfo, z.data(), z.size(), zShape, 3);
                try {
                    Ort::IoBinding bindFirst(first);
                    bindFirst.BindInput("z", which == kHostZ ? hostZ : *dz.value);
                    if (bindMotion) {
                        bindFirst.BindOutput("motion", *dmotionFirst.value);
                    } else {
                        bindFirst.BindOutput("motion", memoryInfo);
                    }
                    for (int i = 0; i < kCaches; ++i) {
                        if (bindCaches) {
                            bindFirst.BindOutput(outNames[i].c_str(), *setA[i]->value);
                        } else {
                            bindFirst.BindOutput(outNames[i].c_str(), memoryInfo);
                        }
                    }
                    const auto began = std::chrono::steady_clock::now();
                    try {
                        first.Run(Ort::RunOptions{nullptr}, bindFirst);
                    } catch (const Ort::Exception& e) {
                        // Which Run refuses matters: the whole point is
                        // whether the FIRST decoder can write device buffers
                        // or only the STEP decoder cannot read them.
                        std::printf("   %-16s FIRST refused: %s\n", label, e.what());
                        continue;
                    }
                    wait();
                    const double firstMs = Since(began);

                    Ort::IoBinding bindStep(step);
                    bindStep.BindInput("z", which == kHostZ ? hostZ : *dz.value);
                    // Without device cache buffers there is nothing to feed
                    // the step session, which is the original problem; the
                    // outputs of the first Run are what it would have to take,
                    // and IoBinding does not expose them as inputs. So the
                    // step call is only attempted once the caches are bound.
                    if (!bindCaches) {
                        std::printf("   %-16s first %6.2f ms   (step needs bound "
                                    "caches)\n", label, firstMs);
                        continue;
                    }
                    // Every input before any output, in the model's own
                    // order. Interleaving them had the step Run report
                    // "Missing Input: cache_in_10".
                    std::vector<Ort::Value> hostCacheTensors;
                    for (int i = 0; i < kCaches; ++i) {
                        if (which == kHostCaches) {
                            const std::vector<int64_t> shape = cacheShape(i);
                            hostCacheTensors.push_back(Ort::Value::CreateTensor<float>(
                                cpuInfo,
                                hostCacheValues[static_cast<std::size_t>(i)].data(),
                                hostCacheValues[static_cast<std::size_t>(i)].size(),
                                shape.data(), shape.size()));
                            bindStep.BindInput(inNames[i].c_str(),
                                               hostCacheTensors.back());
                        } else {
                            bindStep.BindInput(inNames[i].c_str(), *setA[i]->value);
                        }
                    }
                    if (bindMotion) {
                        bindStep.BindOutput("motion", *dmotionStep.value);
                    } else {
                        bindStep.BindOutput("motion", memoryInfo);
                    }
                    for (int i = 0; i < kCaches; ++i) {
                        bindStep.BindOutput(outNames[i].c_str(), *setB[i]->value);
                    }
                    std::vector<double> samples;
                    for (int s = 0; s < kRuns; ++s) {
                        const auto stepBegan = std::chrono::steady_clock::now();
                        try {
                            step.Run(Ort::RunOptions{nullptr}, bindStep);
                        } catch (const Ort::Exception& e) {
                            std::printf("   %-16s STEP refused: %s\n", label, e.what());
                            samples.clear();
                            break;
                        }
                        wait();
                        samples.push_back(Since(stepBegan));
                    }
                    if (samples.empty()) {
                        continue;
                    }
                    std::sort(samples.begin(), samples.end());
                    std::printf("   %-16s first %6.2f ms   step %6.2f ms per call\n",
                                label, firstMs, samples[samples.size() / 2]);
                } catch (const Ort::Exception& e) {
                    std::printf("   %-16s REFUSED: %s\n", label,
                                Shorten(e.what()).c_str());
                }
            }
        } catch (const Ort::Exception& e) {
            std::printf("   sessions failed: %s\n", Shorten(e.what()).c_str());
        }
        }
    }

    return 0;
}
