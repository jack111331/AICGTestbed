#pragma once

#include "MotionFeatures.hpp"

#include <DirectXMath.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Text-to-motion streaming with the FloodDiffusion ONNX export.
//
// FloodDiffusion is not one model but four, and the sampling loop that drives
// them stays on the host:
//
//   text_encoder.onnx       prompt token ids  -> [tokens, 768] features
//   denoiser.onnx           one diffusion-forcing denoising call
//   vae_decoder_first.onnx  first latent      -> 1 motion frame  + 20 caches
//   vae_decoder_step.onnx   next latent + caches -> 4 motion frames + 20 caches
//
// "Diffusion forcing" is what makes this streamable. Rather than denoising a
// whole clip uniformly, each latent frame sits at its own noise level, rising
// along the sequence, and every step shifts that ramp forward by a fraction of
// a chunk. The consequence worth understanding: latents at the FRONT finish
// while the ones behind are still noisy, so motion can be decoded and shown
// while generation continues, and the stream never has to stop at a clip
// boundary. Step() exposes exactly that -- one denoising step, then whatever
// latents it finalised get decoded and appended.
//
// Either execution provider runs these models, and the choice is per
// generation (FloodDiffusionOptions::device, and textEncoderDevice for the
// prompt encoder separately).
//
// The first export could only run on the CPU: the DirectML EP faulted on the
// attention mask, which was built inside the graph from context_lens by a
// Range -> Reshape -> Less chain. That export was replaced, and DirectML now
// runs every model. The honest numbers on an RTX 5080, with both providers
// agreeing to within float noise:
//
//                      DirectML        CPU
//   denoiser           37-40 ms        1.4-10.5 ms
//   VAE step            3.3 ms          1.3 ms
//   text encoder       46 ms           9.4 ms
//
// So DirectML WORKS but is currently slower, and the reason is visible in the
// shape of the numbers: the denoiser costs the same 40 ms whether one latent
// is live or sixty-four, while the CPU scales 1.4 -> 10.5 ms with the same
// work. That is per-dispatch overhead, not compute. ORT profiling confirms it
// -- 1226 of the graph's 1334 nodes are dispatched to DirectML individually,
// with 108 left on the CPU, and the GPU spends its time waiting between tiny
// operators rather than working. A 36 MB model with hidden_dim 256 and a
// classifier-free-guidance batch of two simply does not fill a 5080.
//
// What would change that is fewer, larger dispatches: the 108 CPU-assigned
// nodes are mostly shape arithmetic (Shape, Squeeze, Range, Slice) and each
// one is a partition boundary, so an export that resolves shapes statically
// would let the EP fuse far more of the graph into one DirectML submission.
// Binding the tensors into GPU memory instead of host memory would NOT fix
// this -- it saves copies, and copies are not where the time goes.
// Only needed when FloodDiffusionDevice::DirectML is selected; forward
// declared so this header stays free of d3d12.h and DirectML.h.
struct ID3D12Device;
struct ID3D12CommandQueue;
struct IDMLDevice;

namespace NeuralModelIntegrateTestbed {

enum class FloodDiffusionDevice {
    // ONNX Runtime's CPU execution provider. Currently the faster of the two
    // for these model sizes -- see the note above.
    Cpu,

    // ONNX Runtime's DirectML EP, given this project's own ID3D12Device,
    // command queue and IDMLDevice, so its work lands on the same queue as
    // rendering and is ordered against it without a fence.
    DirectML,
};

const char* ToString(FloodDiffusionDevice device);

// The pipeline's own parameters, written by tools/prepare_flooddiffusion.py as
// `key value` lines. The shapes in here are read back out of the .onnx files by
// that script rather than copied from config.json, so they cannot drift from
// the models they describe.
struct FloodDiffusionConfig {
    int chunkSize = 0;
    int noiseSteps = 0;
    float cfgScale = 1.0f;
    std::string predictionType;  // "vel", "x0" or "noise"
    int textLen = 0;
    std::string textClean;       // "none" or "whitespace"
    int latentDim = 0;
    int motionDim = 0;
    int textDim = 0;
    int joints = 0;
    int vaeNumCaches = 0;
    int vaeUpsampleFactor = 0;
    int vaeFramesFirst = 0;
    int vaeFramesStep = 0;
    bool hasTextEncoder = false;
    std::string sourceConfig;

    // Per cache, the two trailing dimensions of [batch, d1, d2]. Only used to
    // seed the first step and to report sizes.
    std::vector<std::pair<int, int>> vaeCacheShapes;

    bool Load(const std::filesystem::path& path, std::string* error);

    // Motion frames one latent frame decodes to, after the first.
    int FramesPerLatent() const { return vaeFramesStep; }
};

// The noise schedule of one generation. Pure arithmetic with no model
// involved, kept separate because it is the part most easily got wrong: the
// step count comes out of a floating-point division whose truncation genuinely
// matters (60 latents gives exactly 128 steps, 8 latents gives 23 rather than
// 24, because 2.4 / 0.1 lands just under 24).
class DiffusionForcingSchedule {
public:
    DiffusionForcingSchedule() = default;
    DiffusionForcingSchedule(int latentFrames, int chunkSize, int denoiseSteps);

    bool IsValid() const { return m_totalSteps > 0; }

    int TotalSteps() const { return m_totalSteps; }

    // Latents the sampler holds, which is the generated length plus one chunk.
    // That trailing chunk is never fully denoised -- it is the lookahead the
    // ramp needs -- so it is not part of the output.
    int TotalLatents() const { return m_totalLatents; }
    int GeneratedLatents() const { return m_generatedLatents; }

    double TimeAt(int step) const;

    // The half-open latent range [Start, End) this step writes. End also gives
    // how much of the sequence is handed to the denoiser.
    int StartAt(int step) const;
    int EndAt(int step) const;

    // Per-latent noise level for `step`, resized to EndAt(step). Computed in
    // float32 throughout, matching the reference.
    void NoiseLevels(int step, std::vector<float>& out) const;

    double Dt() const { return m_dt; }

private:
    int m_chunkSize = 0;
    int m_denoiseSteps = 0;
    int m_totalLatents = 0;
    int m_generatedLatents = 0;
    int m_totalSteps = 0;
    double m_dt = 0.0;
    std::vector<float> m_base;
};

struct FloodDiffusionOptions {
    // Latent frames to generate. Each decodes to vaeFramesStep motion frames
    // (four here), and HumanML3D motion is 20 fps, so 60 latents is about three
    // seconds.
    int latentFrames = 60;

    // Seeds the starting noise. NOTE this will not reproduce the Python
    // reference for the same number: that uses numpy's PCG64 and a ziggurat
    // normal, neither of which is reproduced here. Same seed, same motion
    // within this engine; not across the two implementations.
    uint32_t seed = 1234;

    // Denoising steps per chunk. Zero takes the exported default.
    int denoiseSteps = 0;

    // Exponential moving average over output joint positions. 1 is off.
    float smoothingAlpha = 1.0f;

    // Where the denoiser and the VAE decoders run. DirectML needs the device
    // pointers to have been passed to Load.
    FloodDiffusionDevice device = FloodDiffusionDevice::DirectML;

    // Where the prompt encoder runs, kept separate because it is a different
    // kind of workload: it runs once per prompt rather than once per step, it
    // is by far the largest of the four models, and on this export it is both
    // five times faster on the CPU and 1.1 GB of weights that would otherwise
    // sit in video memory. FloodDiffusion's own onnx_inference.py exposes the
    // same split for the same reason (its --text-encoder-device exists to keep
    // the 11 GB umT5-XXL encoder off a small GPU).
    FloodDiffusionDevice textEncoderDevice = FloodDiffusionDevice::Cpu;

    // Pin every symbolic input dimension so ONNX Runtime's DirectML EP can
    // compile the graph instead of dispatching it operator by operator.
    //
    // This is what the dispatch count actually turns on. The EP's
    // DmlGraphFusionTransformer compiles each DirectML-assignable partition
    // into one IDMLCompiledOperator, but only when shapes are statically
    // known; measured on the denoiser, pinning the dimensions takes it from
    // 1334 dispatches per call to 1, and 9.5 ms to 2.0 ms. It is applied with
    // AddFreeDimensionOverrideByName, so it needs no re-export.
    //
    // TWO CONSEQUENCES, both deliberate and both visible from here:
    //
    //  * `time` has to be fixed too, and the sampler would otherwise feed a
    //    growing prefix. So in this mode the WHOLE latent buffer is submitted
    //    every step, with the noise ramp clipped to 1 over the part that is
    //    not live yet. The denoiser is non-causal, so that is NOT numerically
    //    identical to feeding the prefix -- the padding attends into the live
    //    frames. It is arguably closer to how the model was trained, where a
    //    full sequence carries a ramp, but it is a different sampler and the
    //    motion it produces should be looked at before trusting it. An export
    //    that takes a valid-length input and masks self-attention would make
    //    this exact; see skills/onnx-directml-ep/SKILL.md.
    //  * The pinned `time` is latentFrames + chunkSize, baked at Load. Begin
    //    then refuses a different latentFrames rather than feeding a shape the
    //    session cannot accept, so changing the length means loading again.
    //
    // No effect on the CPU provider, which does not fuse.
    bool staticShapes = false;
};

// Load once, then Begin a prompt and Step it. Holds the four sessions, the
// tokenizer, the sampler state and the decoded motion.
class FloodDiffusionPipeline {
public:
    FloodDiffusionPipeline();
    ~FloodDiffusionPipeline();

    FloodDiffusionPipeline(const FloodDiffusionPipeline&) = delete;
    FloodDiffusionPipeline& operator=(const FloodDiffusionPipeline&) = delete;

    // Loads pipeline.txt, tokenizer.bin, the denoiser and the two VAE decoders
    // from `directory`. The text encoder is NOT loaded here: it is over a
    // gigabyte and only needed when a prompt is first encoded, so it is loaded
    // on demand by Begin.
    //
    // `options.device` is the only option read at load time; the rest are
    // per-generation and taken by Begin.
    //
    // The three device pointers are used only by FloodDiffusionDevice::
    // DirectML, where ONNX Runtime is handed this project's own device and
    // queue so its work is ordered against rendering without a fence. Pass
    // nullptr for the CPU path.
    bool Load(const std::filesystem::path& directory,
              const FloodDiffusionOptions& options,
              std::string* error,
              ID3D12Device* device = nullptr,
              ID3D12CommandQueue* queue = nullptr,
              IDMLDevice* dmlDevice = nullptr);

    bool IsLoaded() const;
    const FloodDiffusionConfig& Config() const;

    // What Load actually settled on. These can differ from what was asked for
    // only by failing, never by silently falling back.
    FloodDiffusionDevice Device() const;
    FloodDiffusionDevice TextEncoderDevice() const;

    // True when the sessions were created with their input dimensions pinned.
    bool StaticShapes() const;

    // The latent-frame count this pipeline was loaded for. Only meaningful in
    // static-shape mode, where Begin refuses anything else.
    int StaticLatentFrames() const;

    // Starts a generation. Encodes `prompt` (and the empty prompt, which
    // classifier-free guidance needs), loading the text encoder on first use,
    // then reseeds the latents and clears the decoded motion.
    bool Begin(const std::string& prompt,
               const FloodDiffusionOptions& options,
               std::string* error);

    // One denoising step, followed by decoding any latents that step finalised.
    // Returns false only on a real failure; reaching the end is reported by
    // Done() rather than as an error, so a frame loop can call this blindly.
    bool Step(std::string* error);

    bool IsRunning() const;
    bool Done() const;
    int StepsTaken() const;
    int TotalSteps() const;

    // --- decoded motion -----------------------------------------------------

    // Frames available so far. Each is JointCount() positions, root first.
    int FrameCount() const;
    int JointCount() const;

    // Null when `index` is out of range. Valid until the next Step or Begin.
    const DirectX::XMFLOAT3* Frame(int index) const;

    // The raw HumanML3D feature vector for a frame: Config().motionDim floats,
    // retained because the joint positions above are only part of what it
    // carries. Retargeting onto a rigged character needs the ROTATION block
    // (Config().joints - 1 six-dimensional local rotations starting at
    // 4 + (joints-1)*3), which Frame() has already thrown away. Null when
    // `index` is out of range.
    const float* FeatureFrame(int index) const;

    // The accumulated root yaw for a frame, in radians. Needed alongside
    // FeatureFrame to turn the vector's root-relative rotations into world
    // rotations. Zero for an out-of-range index.
    float FrameRootAngle(int index) const;

    // Motion frames per second, for playback. HumanML3D is 20 fps.
    static constexpr float kFrameRate = 20.0f;

    // --- diagnostics --------------------------------------------------------

    const std::string& Prompt() const;
    int PromptTokenCount() const;
    double LastDenoiseMs() const;
    double LastDecodeMs() const;
    double GenerationMs() const;
    bool TextEncoderLoaded() const;

    // Ids the prompt tokenised to, and the text they decode back to. For the
    // UI, and for checking a prompt was understood rather than silently turned
    // into unknown tokens.
    std::string TokenisedPrompt() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace NeuralModelIntegrateTestbed
