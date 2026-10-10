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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
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

    return 0;
}
