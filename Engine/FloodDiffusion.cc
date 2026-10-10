#include "pch.h"
#include "FloodDiffusion.hpp"

#include "OrtEnv.hpp"
#include "Tokenizer.hpp"

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <sstream>

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
                                                   int denoiseSteps)
    : m_chunkSize(chunkSize), m_denoiseSteps(denoiseSteps) {
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
    return std::min(end, m_totalLatents);
}

void DiffusionForcingSchedule::NoiseLevels(int step, std::vector<float>& out) const {
    const int end = EndAt(step);
    out.resize(static_cast<std::size_t>(std::max(end, 0)));
    const float t = static_cast<float>(TimeAt(step));
    for (int i = 0; i < end; ++i) {
        const float level = m_base[static_cast<std::size_t>(i)] - t;
        out[static_cast<std::size_t>(i)] = std::clamp(level, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// FloodDiffusionPipeline
// ---------------------------------------------------------------------------

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

    bool staticShapes = false;
    int staticLatentFrames = 0;
    int staticTime = 0;   // the pinned `time`: latentFrames + chunkSize

    std::unique_ptr<Ort::Session> denoiser;
    std::unique_ptr<Ort::Session> vaeFirst;
    std::unique_ptr<Ort::Session> vaeStep;
    std::unique_ptr<Ort::Session> textEncoder;

    // Kept so the text encoder can be created on demand, long after Load.
    ID3D12Device* d3dDevice = nullptr;
    ID3D12CommandQueue* queue = nullptr;
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
    bool EnsureTextEncoder(std::string* error);
    bool Denoise(std::string* error);
    bool DecodeUpTo(int latentLimit, std::string* error);
    bool DecodeOne(int latentIndex, std::string* error);
    void AppendMotion(const float* motion, int frameCountInBatch);
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

    if (d3dDevice == nullptr || queue == nullptr || dmlDevice == nullptr) {
        SetOrtError(error, "FloodDiffusionDevice::DirectML needs the D3D12 device, "
                           "command queue and DirectML device; pass them to Load");
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
    const OrtStatus* status = dmlApi->SessionOptionsAppendExecutionProvider_DML1(
        *out, dmlDevice, queue);
    if (status != nullptr) {
        std::string message = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
        SetOrtError(error, "could not attach the DirectML execution provider: " + message);
        return false;
    }
    return true;
}

FloodDiffusionPipeline::FloodDiffusionPipeline() : m_impl(std::make_unique<Impl>()) {}
FloodDiffusionPipeline::~FloodDiffusionPipeline() = default;

bool FloodDiffusionPipeline::Load(const std::filesystem::path& directory,
                                  const FloodDiffusionOptions& options,
                                  std::string* error,
                                  ID3D12Device* device,
                                  ID3D12CommandQueue* queue,
                                  IDMLDevice* dmlDevice) {
    Impl& impl = *m_impl;

    // Load is callable again to change provider, so everything a provider owns
    // has to go first. The text encoder matters most: it is created lazily, so
    // a stale one would quietly keep serving prompts from the OLD provider.
    // The cached features go with it -- they are just floats, but they were
    // produced by the session being replaced.
    impl.denoiser.reset();
    impl.vaeFirst.reset();
    impl.vaeStep.reset();
    impl.textEncoder.reset();
    impl.textCache.clear();
    impl.caches.clear();
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
    impl.queue = queue;
    impl.dmlDevice = dmlDevice;

    if (!impl.config.Load(directory / "pipeline.txt", error)) {
        return false;
    }
    if (!impl.tokenizer.Load(directory / "tokenizer.bin", error)) {
        return false;
    }
    impl.textEncoderDevice = options.textEncoderDevice;
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
        // The VAE decoders are deliberately left DYNAMIC, even though pinning
        // their `batch` would take a decode from 1.48 ms to 0.47 ms.
        //
        // Pinning it breaks the cache hand-off between the two sessions on ORT
        // 1.24.4: vae_decoder_first runs fine on its own, but feeding its 20
        // cache outputs into vae_decoder_step then fails with "Unexpected input
        // data type. Actual: (((null))) , expected: ((tensor(float)))". A
        // statically-shaped session's outputs apparently cannot be handed
        // straight to another statically-shaped session's inputs here. The
        // denoiser is where the time is -- it runs 128 times a generation
        // against the decoders' 60, and it is the one carrying 1334 nodes -- so
        // this costs little and keeps the decoder path working.
        vaeDims = {};
    }

    if (!impl.MakeOptions(impl.device, denoiserDims, impl.denoiserOptions, error)) {
        return false;
    }
    if (!impl.MakeOptions(impl.device, vaeDims, impl.vaeOptions, error)) {
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

bool FloodDiffusionPipeline::StaticShapes() const { return m_impl->staticShapes; }

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
                                      denoiseSteps);
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

    impl.prompt = prompt;
    // Kept for the UI. Done here rather than in EncodeText, which returns early
    // for a prompt it has already cached and would leave these stale.
    impl.promptIds = impl.tokenizer.Encode(prompt, impl.config.textLen);
    impl.promptTokens = static_cast<int>(impl.promptIds.size());

    // The conditional prompt and the unconditional (empty) one. Both are needed
    // before anything else, because a failure here should leave the previous
    // generation untouched rather than half-replaced.
    if (!impl.EncodeText(prompt, error)) {
        return false;
    }
    const bool guided = impl.config.cfgScale != 1.0f;
    if (guided && !impl.EncodeText(std::string(), error)) {
        return false;
    }

    const Impl::TextFeatures& conditional = impl.textCache.at(prompt);
    const Impl::TextFeatures* unconditional =
        guided ? &impl.textCache.at(std::string()) : nullptr;

    impl.rows = guided ? 2 : 1;
    impl.contextLen = conditional.tokens;
    if (unconditional != nullptr) {
        impl.contextLen = std::max(impl.contextLen, unconditional->tokens);
    }
    if (impl.staticShapes) {
        // The session pinned ctx_len to text_len, so the context has to be fed
        // at that width. Exact rather than approximate: context_lens below
        // still reports the real token counts, and the model masks the rest out
        // of cross-attention. Measured difference against an unpadded run:
        // 0.000004.
        impl.contextLen = impl.config.textLen;
        if (conditional.tokens > impl.contextLen ||
            (unconditional != nullptr && unconditional->tokens > impl.contextLen)) {
            SetOrtError(error, "the prompt encoded to more tokens than text_len");
            return false;
        }
    }

    // context is [rows, segments=1, contextLen, textDim], zero-padded, with
    // context_lens telling the model how much of each row is real. Only one
    // segment: a single prompt covers the whole sequence, so text_idx is all
    // zeros. Several prompts over time would widen the segments axis.
    const std::size_t stride =
        static_cast<std::size_t>(impl.contextLen) * static_cast<std::size_t>(impl.config.textDim);
    impl.context.assign(stride * static_cast<std::size_t>(impl.rows), 0.0f);
    impl.contextLens.assign(static_cast<std::size_t>(impl.rows), 0);

    std::copy(conditional.values.begin(), conditional.values.end(), impl.context.begin());
    impl.contextLens[0] = conditional.tokens;
    if (unconditional != nullptr) {
        std::copy(unconditional->values.begin(), unconditional->values.end(),
                  impl.context.begin() + static_cast<std::ptrdiff_t>(stride));
        impl.contextLens[1] = unconditional->tokens;
    }

    // Starting noise. std::mt19937 rather than numpy's PCG64: the stream will
    // not match the Python reference for the same seed, which is called out in
    // FloodDiffusionOptions.
    impl.schedule = schedule;
    impl.latents.assign(static_cast<std::size_t>(schedule.TotalLatents()) *
                            static_cast<std::size_t>(impl.config.latentDim),
                        0.0f);
    std::mt19937 generator(options.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : impl.latents) {
        value = normal(generator);
    }

    impl.step = 0;
    impl.running = true;
    impl.caches.clear();
    impl.decodedLatents = 0;
    impl.frames.clear();
    impl.features.clear();
    impl.rootAngles.clear();
    impl.frameCount = 0;
    impl.lastDenoiseMs = 0.0;
    impl.lastDecodeMs = 0.0;
    impl.generationMs = 0.0;
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

    // The ramp over the full buffer: beyond `end` it clips to 1, i.e. pure
    // noise, which is what the model saw for not-yet-denoised frames in
    // training. NoiseLevels sizes itself to EndAt(step), so the tail is filled
    // in explicitly here.
    schedule.NoiseLevels(step, noiseLevels);
    if (submitted > static_cast<int>(noiseLevels.size())) {
        const float t = static_cast<float>(schedule.TimeAt(step));
        const int wasSized = static_cast<int>(noiseLevels.size());
        noiseLevels.resize(static_cast<std::size_t>(submitted));
        for (int i = wasSized; i < submitted; ++i) {
            const float level = 1.0f + static_cast<float>(i) /
                                           static_cast<float>(config.chunkSize) - t;
            noiseLevels[static_cast<std::size_t>(i)] = std::clamp(level, 0.0f, 1.0f);
        }
    }

    const int latentDim = config.latentDim;
    const std::size_t perRow = static_cast<std::size_t>(submitted) *
                               static_cast<std::size_t>(latentDim);

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

        const float* prediction = outputs[0].GetTensorData<float>();
        const float dt = static_cast<float>(schedule.Dt());
        const float cfg = config.cfgScale;

        for (int index = start; index < end; ++index) {
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
    const int64_t zShape[] = {1, 1, config.latentDim};
    const float* z = latents.data() + static_cast<std::size_t>(latentIndex) *
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
    } catch (const Ort::Exception& e) {
        SetOrtError(error, "decoding latent " + std::to_string(latentIndex) + ": " +
                               Shorten(e.what()));
        return false;
    }
    return true;
}

bool FloodDiffusionPipeline::Impl::DecodeUpTo(int latentLimit, std::string* error) {
    const int limit = std::min(latentLimit, schedule.GeneratedLatents());
    const auto began = std::chrono::steady_clock::now();
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
    }
    return true;
}

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

    if (impl.step >= impl.schedule.TotalSteps()) {
        // Out of steps: everything that remains is as denoised as it is going
        // to get, so decode the rest and stop.
        if (!impl.DecodeUpTo(impl.schedule.GeneratedLatents(), error)) {
            impl.running = false;
            return false;
        }
        impl.running = false;
        impl.generationMs += Milliseconds(began);
        return true;
    }

    if (!impl.Denoise(error)) {
        impl.running = false;
        return false;
    }
    ++impl.step;

    // A latent is final once no remaining step will touch it again. Because
    // `start` never decreases, the next step's start is exactly that boundary.
    const int finalised = impl.step < impl.schedule.TotalSteps()
                              ? impl.schedule.StartAt(impl.step)
                              : impl.schedule.GeneratedLatents();
    if (!impl.DecodeUpTo(finalised, error)) {
        impl.running = false;
        return false;
    }

    if (impl.step >= impl.schedule.TotalSteps() &&
        impl.decodedLatents >= impl.schedule.GeneratedLatents()) {
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
