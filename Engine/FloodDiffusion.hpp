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
// generation and per model group: FloodDiffusionOptions::device for the
// denoiser, textEncoderDevice for the prompt encoder, vaeDevice for the two
// decoders. They are separate because the answer is genuinely different for
// each -- the denoiser wants DirectML, the other two want the CPU.
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
// Those were one-off per-call numbers and the VAE row has since been measured
// properly, over a threaded sequence at the shipped length: 5.4 ms a call on
// DirectML against 1.4 on the CPU. It held, and it is why the decoders now
// default to the CPU provider -- see FloodDiffusionOptions::vaeDevice.
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

    // ONNX Runtime's DirectML EP, given this project's own ID3D12Device and
    // IDMLDevice. It submits to a COMPUTE queue the pipeline creates and owns,
    // not the renderer's graphics queue, so a generation is scheduled beside
    // the frame instead of between frames -- see OrderAfterWork for what that
    // costs in ordering.
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

    // Motion frames to keep decoded ahead of the frame a consumer has reached.
    // Non-zero takes decoding OUT of Step and makes it the caller's, through
    // DecodeAhead; zero decodes every finalised latent inside Step, which is
    // what a batch generation wants.
    //
    // This is the difference between generating a clip and streaming one. The
    // sampler finishes long before playback does -- at 60 latents it produces
    // 11.85 s of motion in about 2.1 s -- and eager decoding therefore pays
    // all sixty decodes during that burst, whether or not anyone watches the
    // motion. Paced, a stream needs one vae_decoder_step per four motion
    // frames, which at 20 fps is five decodes a second however far ahead the
    // sampler has got.
    //
    // It does not reduce the total work for a clip watched to the end; what it
    // changes is WHEN the work happens, and that decoding a latent nobody
    // reaches never happens at all. Both halves stay strictly in order, so
    // this defers decoding and never skips or reorders it.
    //
    // The number is the slack that absorbs a slow frame. Four frames is one
    // latent, so it has to be more than that; twelve is 0.6 s of motion, and
    // the burst at the start is bounded by it -- three decodes, once.
    //
    // WITH THIS SET, Step DECODES NOTHING, so a caller that never calls
    // DecodeAhead will denoise to the end and then report IsRunning() forever,
    // because the generation is not finished until the motion exists.
    int decodeAheadFrames = 0;

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

    // Where the two VAE decoders run, kept separate from `device` for the same
    // reason as the prompt encoder: it is a different workload, and the answer
    // came out the other way round.
    //
    // CPU BY DEFAULT, and this is the single largest saving in the pipeline.
    // The decode is where a generation's wall time is -- vae_decoder_step runs
    // 59 times of 60, against the denoiser's 128 -- and the two providers are
    // not close on it. Measured per call over a threaded sequence of 59,
    // which is what //Engine:fd-probe section 11 does:
    //
    //   DirectML    5.4 ms per call    about 320 ms for 59
    //   CPU         1.4 ms per call    about  80 ms for 59
    //
    // So it is roughly 4x, and it is the opposite of the denoiser's answer --
    // which is why the two are separate options. End to end at 60 latents it
    // takes the decode total from 188.5 ms to 69.2 ms and the whole generation
    // from 512.3 ms to 354.1 ms.
    //
    // THE REASON IS NOT the small graph failing to fill the GPU, though that
    // is true of it. It is that the CPU is a SECOND PROCESSOR and this
    // pipeline's compute queue is not: every session shares that one queue,
    // because that is what orders a decode after the denoising step that
    // produced its latent without a fence. A DirectML decode therefore queues
    // behind the denoiser and the denoiser queues behind it -- moving the
    // decoders back to DirectML raises the decode from 69 to 185 ms AND the
    // denoising from 281 to 327. On the CPU the decode runs on cores that were
    // idle anyway, concurrently with the GPU, and costs the critical path
    // almost nothing.
    //
    // Which is why the per-call figures above are the weaker evidence. A
    // from-source ORT 1.31 build takes the DirectML decode to 1.62 ms a call,
    // and 0.80 ms with `batch` pinned -- beating the CPU outright -- and the
    // generation measured 513 ms against this path's 348 all the same. See
    // skills/onnx-directml-ep/SKILL.md section 5.1.
    //
    // Pinning the decoders' `batch` is NOT the lever it looked like, and the
    // write-up is on vaeDims in Load: the 1.48 -> 0.47 ms in fd-probe section
    // 6 was measured on vae_decoder_first, which runs ONCE per generation, so
    // it is worth a millisecond rather than sixty.
    //
    // It costs nothing in plumbing. The decoder has always read its latent
    // from host memory, so a CPU session is what it was already being fed.
    FloodDiffusionDevice vaeDevice = FloodDiffusionDevice::Cpu;

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

    // Keep the latents in GPU memory for the whole generation, and run the
    // guidance combine and the Euler step there as a compute shader, instead
    // of reading the prediction back and uploading the latents again.
    //
    // OFF BY DEFAULT, AND THE REASON IS WORTH READING BEFORE TURNING IT ON.
    //
    // The round trip looks like it should be expensive. With host-memory
    // tensors ONNX Runtime copies all five denoiser inputs up and the
    // prediction back on every Run, and at the shipped shapes that is 772 KB
    // per step of which effectively all is the text context -- a buffer that
    // does not change for a whole generation and is re-sent 128 times.
    //
    // Measured end to end at 60 latents on an RTX 5080, with the decoders on
    // DirectML as they were when this was written, it bought nothing:
    //
    //   host round trip          506.9 ms     denoising 318.9  decode 187.3
    //   latents kept on the GPU  508.7 ms     denoising  53.8  decode 454.7
    //
    // The accounting moves and the total does not. The denoising half gets six
    // times cheaper because this path never waits -- Run records and returns
    // -- so the time reappears at the one place that does wait, the readback
    // of a finalised latent for the decoder. The generation is bound by GPU
    // execution, not by the copies, which is what
    // skills/onnx-directml-ep/SKILL.md already said about binding and which
    // this confirms at the shipped shapes rather than the probe's small ones.
    //
    // WITH vaeDevice ON THE CPU, which is now the default, the same pair reads
    //
    //   host round trip          354.1 ms     denoising 284.1  decode  69.2
    //   latents kept on the GPU  330.5 ms     denoising  54.3  decode 276.0
    //
    // so it is now worth a few percent of the total as well. The resident
    // column's decode is still most of its time for the reason above: that is
    // where the generation waits for the denoising the GPU is actually doing.
    //
    // It would be worth having even at parity, for three reasons that are not
    // speed:
    //   * the sampler's arithmetic is where the data is, so the only thing
    //     crossing the bus per step is a 520-byte noise ramp;
    //   * device-resident tensors are the entry fee for
    //     ep.dml.enable_graph_capture, which host memory cannot use at all;
    //   * it is the half of a fully device-resident pipeline that is done.
    //     The other half will not be: the VAE decoders are faster on the CPU
    //     provider than on DirectML, and Ort::IoBinding refuses
    //     vae_decoder_step outright. See vaeDevice and vaeDims in Load.
    //
    // Needs DirectML and staticShapes: the buffers are created once with the
    // pinned shapes, and a growing `time` would mean rebuilding them and the
    // IoBinding every step. Load fails rather than falling back silently.
    //
    // The motion is identical, not merely close: measured at 0.00000 m of
    // joint difference over a whole take, because the shader does the same
    // arithmetic in the same order. //Engine:flooddiffusion-smoke asserts it
    // to within a centimetre, which is the tolerance that would catch a
    // transposed index rather than float noise.
    bool gpuLatents = false;
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
    // The two device pointers are used only by FloodDiffusionDevice::DirectML.
    // Pass nullptr for the CPU path.
    //
    // No command queue: the pipeline creates its OWN compute queue on the
    // device it is given and hands that to the execution provider. A caller
    // cannot put this work on the graphics queue, which is the point -- a
    // generation is tens of milliseconds of denoising, and on the renderer's
    // queue those milliseconds are frames.
    //
    // Calling Load again changes provider. It drains the compute queue first,
    // because it destroys the ONNX Runtime sessions and their submissions may
    // still be running on it.
    bool Load(const std::filesystem::path& directory,
              const FloodDiffusionOptions& options,
              std::string* error,
              ID3D12Device* device = nullptr,
              IDMLDevice* dmlDevice = nullptr);

    bool IsLoaded() const;
    const FloodDiffusionConfig& Config() const;

    // --- the compute queue --------------------------------------------------
    //
    // What the fence is, and is not, for.
    //
    // Every tensor this pipeline binds is in HOST memory, so Step() already
    // comes back with its results on the CPU: ONNX Runtime's Run does the
    // copy and the wait itself, and no fence of ours could make the floats
    // appear any sooner. Frame(), FeatureFrame() and the retargeting all read
    // ordinary std::vectors and need no synchronisation at all.
    //
    // The fence exists for the two things queue order used to cover for free
    // and no longer does:
    //
    //   * LIFETIME. Destroying an ONNX Runtime session, which Load does on
    //     every provider change, releases resources its submissions may still
    //     be reading. On the renderer's queue the frame's own drains hid this.
    //     Nothing drains a private queue, so Load and the destructor do.
    //   * CROSS-QUEUE ORDERING, for whatever samples this on the GPU next. The
    //     motion is CPU-side today; the moment a pass reads one of these
    //     tensors as a buffer, OrderAfterWork is what makes the read safe.

    // Makes `queue` wait until everything the pipeline has submitted so far
    // has finished. GPU-side: returns immediately, and costs the waiting queue
    // nothing unless it arrives early. False only on a real failure; with no
    // compute queue (the CPU path) there is nothing to order and it succeeds.
    bool OrderAfterWork(ID3D12CommandQueue* queue, std::string* error = nullptr);

    // Blocks the CPU until the same point. For teardown, provider changes and
    // tests.
    bool WaitForWork(std::string* error = nullptr);

    // The queue the DirectML provider submits to, or null on the CPU path.
    // Exposed so a test can assert what type it is and that it is not the
    // renderer's.
    ID3D12CommandQueue* ComputeQueue() const;

    // What Load actually settled on. These can differ from what was asked for
    // only by failing, never by silently falling back.
    FloodDiffusionDevice Device() const;
    FloodDiffusionDevice TextEncoderDevice() const;
    FloodDiffusionDevice VaeDevice() const;

    // True when the sessions were created with their input dimensions pinned.
    bool StaticShapes() const;

    // True when the latents live in GPU memory for the whole generation. See
    // FloodDiffusionOptions::gpuLatents.
    bool GpuLatents() const;

    // The latent-frame count this pipeline was loaded for. Only meaningful in
    // static-shape mode, where Begin refuses anything else.
    int StaticLatentFrames() const;

    // Starts a generation. Encodes `prompt` (and the empty prompt, which
    // classifier-free guidance needs), loading the text encoder on first use,
    // then reseeds the latents and clears the decoded motion.
    bool Begin(const std::string& prompt,
               const FloodDiffusionOptions& options,
               std::string* error);

    // One denoising step, followed by decoding any latents that step finalised
    // -- or, when FloodDiffusionOptions::decodeAheadFrames is set, no decoding
    // at all, because then it is DecodeAhead's. Returns false only on a real
    // failure; reaching the end is reported by Done() rather than as an error,
    // so a frame loop can call this blindly.
    //
    // At the shipped schedule this decodes at most ONE latent: 60 latents
    // finalise over 128 steps, one every other step, so eager decoding is
    // already spread rather than bunched. What decodeAheadFrames changes is
    // that the spreading follows playback instead of the sampler.
    bool Step(std::string* error);

    // Decodes as far as a consumer that has reached `consumedFrame` needs,
    // plus the lookahead the generation was begun with, and no further. Call
    // it once per tick of whatever is consuming the motion; it is a no-op
    // without decodeAheadFrames, where Step has already done the decoding.
    //
    // Bounded by what the sampler has finalised, so it cannot run ahead of
    // generation, and it never decodes a latent twice -- FrameCount() says how
    // much motion came of it. False only on a real failure.
    bool DecodeAhead(int consumedFrame, std::string* error = nullptr);

    bool IsRunning() const;
    bool Done() const;
    int StepsTaken() const;
    int TotalSteps() const;

    // --- decoded motion -----------------------------------------------------

    // Frames available so far. Each is JointCount() positions, root first.
    int FrameCount() const;
    int JointCount() const;

    // How far each half of the pipeline has got, in latents. The pair is what
    // shows pacing working: eager, they move together; paced, the decoder sits
    // a fixed distance behind the sampler.
    int DecodedLatents() const;
    int FinalisedLatents() const;

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

    // Where a generation's wall time went, summed over the whole thing rather
    // than the last call. Worth having as a pair: this is what localised the
    // decode as the pipeline's real cost and then showed the fix working. On
    // an RTX 5080 at the shipped 60 latents, with the latents on the host, the
    // decode was 188.5 ms of a 512.3 ms generation with the decoders on
    // DirectML and is 69.2 ms of 354.1 ms with them on the CPU. See
    // FloodDiffusionOptions::vaeDevice and ::gpuLatents.
    double DenoiseTotalMs() const;
    double DecodeTotalMs() const;

    // Ids the prompt tokenised to, and the text they decode back to. For the
    // UI, and for checking a prompt was understood rather than silently turned
    // into unknown tokens.
    std::string TokenisedPrompt() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace NeuralModelIntegrateTestbed
