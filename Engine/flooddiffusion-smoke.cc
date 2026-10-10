// End-to-end smoke test for the FloodDiffusion text-to-motion pipeline.
//
// Runs the real models, so it needs resources/FloodDiffusion/ staged by
// tools/prepare_flooddiffusion.py; without it the model sections are skipped
// rather than failed, because the weights are not in the repository.
//
// The assertions that matter are the ones a renderer cannot show you:
//
//   * the sampling schedule, whose step count comes out of a floating-point
//     truncation that is genuinely load-bearing (60 latents gives exactly 128
//     steps, 8 latents gives 23 and not 24);
//   * that motion arrives WHILE generation is still running, which is the
//     whole point of diffusion forcing and the thing a finished-clip API would
//     quietly fail to provide;
//   * that bone lengths stay constant across frames. A skeleton with the wrong
//     root integration or a mistaken quaternion still animates plausibly; limbs
//     changing length is what gives it away.
#include "pch.h"

#include "FloodDiffusion.hpp"

#include <d3d12.h>
#include <wrl/client.h>

#include <DirectML.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using NeuralModelIntegrateTestbed::DiffusionForcingSchedule;
using NeuralModelIntegrateTestbed::FloodDiffusionConfig;
using NeuralModelIntegrateTestbed::FloodDiffusionDevice;
using NeuralModelIntegrateTestbed::FloodDiffusionOptions;
using NeuralModelIntegrateTestbed::FloodDiffusionPipeline;
using NeuralModelIntegrateTestbed::HumanML3DChains;

namespace {

int g_failures = 0;

void Expect(const char* what, bool condition) {
    std::printf("  %-54s %s\n", what, condition ? "ok" : "MISMATCH");
    if (!condition) ++g_failures;
}

void ExpectInt(const char* what, long long got, long long expected) {
    const bool ok = got == expected;
    std::printf("  %-54s %8lld expected %8lld  %s\n", what, got, expected,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++g_failures;
}

const std::filesystem::path kDirectory = "resources/FloodDiffusion";

// The D3D12 and DirectML devices the GPU provider needs. Created once here;
// the application passes in the renderer's own instead.
struct Devices {
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<IDMLDevice> dml;

    bool Create() {
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&device)))) {
            return false;
        }
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)))) {
            return false;
        }
        return SUCCEEDED(DMLCreateDevice(device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                         IID_PPV_ARGS(&dml)));
    }
};

// What one generation produced, so two providers can be compared.
struct Run {
    int frames = 0;
    int joints = 0;
    std::vector<DirectX::XMFLOAT3> positions;  // frames * joints
    double milliseconds = 0.0;
    bool ok = false;
};

// --- the schedule, which needs no models ----------------------------------

void TestSchedule() {
    std::printf("diffusion-forcing schedule\n");

    // Expected values come from running the reference formula in
    // onnx_inference.py's generate_latents; see tools/prepare_flooddiffusion.py
    // for where the parameters are read from.
    const DiffusionForcingSchedule sixty(60, 5, 10);
    ExpectInt("60 latents, chunk 5, 10 steps -> total steps", sixty.TotalSteps(), 128);
    ExpectInt("  latents held, including the noise lookahead",
              sixty.TotalLatents(), 65);
    ExpectInt("  latents actually generated", sixty.GeneratedLatents(), 60);

    // 2.4 / 0.1 is 23.999999999999996, so this truncates to 23. Rounding here
    // would run a step past the end of the ramp.
    const DiffusionForcingSchedule eight(8, 5, 10);
    ExpectInt("8 latents -> 23 steps, not 24", eight.TotalSteps(), 23);

    ExpectInt("first step writes [0, 1)", sixty.StartAt(0), 0);
    ExpectInt("  and ends at 1", sixty.EndAt(0), 1);
    ExpectInt("step 2 ends at 2", sixty.EndAt(2), 2);
    ExpectInt("step 127 starts at 59", sixty.StartAt(127), 59);
    ExpectInt("  and ends at 64", sixty.EndAt(127), 64);
    ExpectInt("step 22 of the 8-latent run starts at 7", eight.StartAt(22), 7);
    ExpectInt("  and ends at 12", eight.EndAt(22), 12);

    // start must never decrease: the pipeline relies on that to decide a latent
    // will not be revisited, and so to decode it.
    bool monotonic = true;
    int previousStart = 0;
    int previousEnd = 0;
    for (int step = 0; step < sixty.TotalSteps(); ++step) {
        const int start = sixty.StartAt(step);
        const int end = sixty.EndAt(step);
        if (start < previousStart || end < previousEnd || start > end ||
            end > sixty.TotalLatents()) {
            monotonic = false;
        }
        previousStart = start;
        previousEnd = end;
    }
    Expect("start and end never decrease and stay in range", monotonic);

    std::vector<float> levels;
    sixty.NoiseLevels(0, levels);
    Expect("at step 0 the first latent is fully noised",
           levels.size() == 1 && std::fabs(levels[0] - 1.0f) < 1e-6f);

    sixty.NoiseLevels(20, levels);
    bool ramped = levels.size() == static_cast<std::size_t>(sixty.EndAt(20));
    for (std::size_t i = 0; i + 1 < levels.size(); ++i) {
        // The ramp rises along the sequence: earlier latents are further along.
        if (levels[i] > levels[i + 1] + 1e-6f) ramped = false;
        if (levels[i] < 0.0f || levels[i] > 1.0f) ramped = false;
    }
    Expect("the noise level rises along the sequence, within [0, 1]", ramped);

    const DiffusionForcingSchedule invalid(0, 5, 10);
    Expect("zero latents is rejected", !invalid.IsValid());
}

// --- the configuration ----------------------------------------------------

bool TestConfig(FloodDiffusionConfig* config) {
    std::printf("\npipeline.txt\n");
    std::string error;
    if (!config->Load(kDirectory / "pipeline.txt", &error)) {
        std::printf("  %s\n", error.c_str());
        return false;
    }
    std::printf("  %-54s %s\n", "exported from", config->sourceConfig.c_str());
    ExpectInt("joints", config->joints, 22);
    ExpectInt("motion feature width", config->motionDim, 263);
    ExpectInt("latent width", config->latentDim, 4);
    ExpectInt("text feature width", config->textDim, 768);
    ExpectInt("VAE caches", config->vaeNumCaches, 20);
    ExpectInt("motion frames per latent after the first",
              config->FramesPerLatent(), 4);
    Expect("the prediction type is one this build handles",
           config->predictionType == "vel" || config->predictionType == "x0" ||
               config->predictionType == "noise");
    Expect("noise steps divide into chunks evenly",
           config->noiseSteps % config->chunkSize == 0);
    return true;
}

// --- the pipeline ---------------------------------------------------------

// Largest change in any bone's length across the whole take, and the mean
// length, so the figure can be read as a proportion.
void MeasureBones(const FloodDiffusionPipeline& pipeline, float* worstDrift,
                  float* meanLength) {
    const auto& chains = HumanML3DChains();
    std::vector<float> first;
    std::vector<float> lengths;
    *worstDrift = 0.0f;
    double total = 0.0;
    int samples = 0;

    for (int frame = 0; frame < pipeline.FrameCount(); ++frame) {
        const DirectX::XMFLOAT3* joints = pipeline.Frame(frame);
        lengths.clear();
        for (const auto& chain : chains) {
            for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
                const DirectX::XMFLOAT3& a = joints[chain[i]];
                const DirectX::XMFLOAT3& b = joints[chain[i + 1]];
                const float dx = b.x - a.x;
                const float dy = b.y - a.y;
                const float dz = b.z - a.z;
                lengths.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
        if (frame == 0) {
            first = lengths;
        } else {
            for (std::size_t i = 0; i < lengths.size(); ++i) {
                *worstDrift = std::max(*worstDrift, std::fabs(lengths[i] - first[i]));
            }
        }
        for (const float length : lengths) {
            total += length;
            ++samples;
        }
    }
    *meanLength = samples > 0 ? static_cast<float>(total / samples) : 0.0f;
}

void TestPipeline(const FloodDiffusionConfig& config, FloodDiffusionDevice device,
                  const Devices& devices, Run* run, bool staticShapes = false) {
    std::printf("\nloading the models on %s%s\n", ToString(device),
                staticShapes ? ", shapes pinned" : "");

    FloodDiffusionPipeline pipeline;
    FloodDiffusionOptions options;
    options.latentFrames = 8;  // 29 motion frames; enough to exercise streaming
    options.seed = 1234;
    options.device = device;
    options.staticShapes = staticShapes;
    // Pinned rather than left at its default, so the two runs below differ in
    // exactly one thing: where the sampler runs.
    options.textEncoderDevice = FloodDiffusionDevice::Cpu;

    std::string error;
    if (!pipeline.Load(kDirectory, options, &error, devices.device.Get(),
                       devices.queue.Get(), devices.dml.Get())) {
        std::printf("  %s\n", error.c_str());
        ++g_failures;
        return;
    }
    Expect("the denoiser and both VAE decoders loaded", pipeline.IsLoaded());
    Expect("the text encoder is NOT loaded yet", !pipeline.TextEncoderLoaded());
    Expect("the sampler is on the requested provider", pipeline.Device() == device);
    Expect("the prompt encoder is on the CPU as asked",
           pipeline.TextEncoderDevice() == FloodDiffusionDevice::Cpu);
    Expect("static-shape mode reports itself",
           pipeline.StaticShapes() == staticShapes);
    if (staticShapes) {
        ExpectInt("  pinned for the requested latent count",
                  pipeline.StaticLatentFrames(), options.latentFrames);

        // A pinned sequence length is baked into the session, so a different
        // length has to be refused rather than fed.
        FloodDiffusionOptions other = options;
        other.latentFrames = options.latentFrames + 1;
        std::string refusal;
        Expect("  a different latent count is refused",
               !pipeline.Begin("a person walks forward", other, &refusal));
        Expect("  and the error says to load again",
               refusal.find("load again") != std::string::npos);
    }

    std::printf("\nencoding a prompt (loads the 1.1 GB text encoder)\n");
    const std::string prompt = "a person walks forward";
    if (!pipeline.Begin(prompt, options, &error)) {
        std::printf("  %s\n", error.c_str());
        ++g_failures;
        return;
    }
    Expect("the text encoder loaded on demand", pipeline.TextEncoderLoaded());
    std::printf("  %-54s %d\n", "prompt tokens", pipeline.PromptTokenCount());
    std::printf("  %-54s \"%s\"\n", "tokens decode back to",
                pipeline.TokenisedPrompt().c_str());
    Expect("the prompt tokenised to something", pipeline.PromptTokenCount() > 1);
    ExpectInt("total steps for 8 latents", pipeline.TotalSteps(), 23);

    std::printf("\nstreaming\n");
    int firstFrameStep = -1;
    int stepsWhenFirstFrameArrived = -1;
    int guard = 0;
    while (pipeline.IsRunning() && guard++ < 1000) {
        if (!pipeline.Step(&error)) {
            std::printf("  step %d failed: %s\n", pipeline.StepsTaken(), error.c_str());
            ++g_failures;
            return;
        }
        if (firstFrameStep < 0 && pipeline.FrameCount() > 0) {
            firstFrameStep = pipeline.FrameCount();
            stepsWhenFirstFrameArrived = pipeline.StepsTaken();
        }
    }
    Expect("the generation terminated", !pipeline.IsRunning());
    Expect("it finished without exhausting the guard", guard < 1000);

    // The streaming property. If motion only appeared once everything had
    // finished, this would be equal to the total rather than well short of it.
    std::printf("  %-54s %d of %d\n", "steps taken when the first frame arrived",
                stepsWhenFirstFrameArrived, pipeline.TotalSteps());
    Expect("motion arrived before generation finished",
           stepsWhenFirstFrameArrived > 0 &&
               stepsWhenFirstFrameArrived < pipeline.TotalSteps());
    Expect("the first delivery was a single frame", firstFrameStep == 1);

    // 1 frame from the first latent, then 4 per latent after it.
    const int expectedFrames = config.vaeFramesFirst +
                               config.FramesPerLatent() * (options.latentFrames - 1);
    ExpectInt("motion frames produced", pipeline.FrameCount(), expectedFrames);
    ExpectInt("joints per frame", pipeline.JointCount(), config.joints);
    Expect("a frame past the end is null",
           pipeline.Frame(pipeline.FrameCount()) == nullptr);
    Expect("a negative frame is null", pipeline.Frame(-1) == nullptr);

    std::printf("\nthe motion itself\n");
    bool finite = true;
    float lowestY = 1e9f;
    float highestY = -1e9f;
    float largestExtent = 0.0f;
    for (int frame = 0; frame < pipeline.FrameCount(); ++frame) {
        const DirectX::XMFLOAT3* joints = pipeline.Frame(frame);
        for (int joint = 0; joint < pipeline.JointCount(); ++joint) {
            const DirectX::XMFLOAT3& p = joints[joint];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                finite = false;
            }
            lowestY = std::min(lowestY, p.y);
            highestY = std::max(highestY, p.y);
            largestExtent = std::max(largestExtent,
                                     std::max(std::fabs(p.x), std::fabs(p.z)));
        }
    }
    Expect("every joint position is finite", finite);
    std::printf("  %-54s %.3f to %.3f\n", "height range over the take", lowestY,
                highestY);
    std::printf("  %-54s %.3f\n", "largest horizontal distance from the origin",
                largestExtent);

    // HumanML3D is in metres with the root near the hips, so a standing figure
    // spans roughly a metre and a half. This is a sanity band, not a tight
    // bound: it is here to catch output that is scaled or garbage, which is the
    // realistic failure, rather than to pin the model's behaviour.
    Expect("the figure is roughly human-sized",
           highestY - lowestY > 0.3f && highestY - lowestY < 4.0f);

    float worstDrift = 0.0f;
    float meanLength = 0.0f;
    MeasureBones(pipeline, &worstDrift, &meanLength);
    std::printf("  %-54s %.4f\n", "mean bone length", meanLength);
    std::printf("  %-54s %.4f\n", "largest bone-length change across frames",
                worstDrift);
    // The VAE reconstructs each frame independently, so bone lengths are not
    // exactly constant the way a rigged skeleton's would be. They should still
    // stay well within a fraction of their own size; a broken root integration
    // or quaternion shows up here as drift comparable to the bones themselves.
    Expect("bone lengths hold to within half their own length",
           meanLength > 0.01f && worstDrift < meanLength * 0.5f);

    // Hand the frames back for the cross-provider comparison.
    if (run != nullptr) {
        run->frames = pipeline.FrameCount();
        run->joints = pipeline.JointCount();
        run->milliseconds = pipeline.GenerationMs();
        run->positions.clear();
        for (int frame = 0; frame < pipeline.FrameCount(); ++frame) {
            const DirectX::XMFLOAT3* joints = pipeline.Frame(frame);
            run->positions.insert(run->positions.end(), joints,
                                  joints + pipeline.JointCount());
        }
        run->ok = true;
    }

    std::printf("\ntiming\n");
    std::printf("  %-54s %.1f ms\n", "whole generation", pipeline.GenerationMs());
    std::printf("  %-54s %.2f ms\n", "last denoising step", pipeline.LastDenoiseMs());
    std::printf("  %-54s %.2f ms\n", "last VAE decode", pipeline.LastDecodeMs());
    const double seconds = pipeline.FrameCount() / FloodDiffusionPipeline::kFrameRate;
    std::printf("  %-54s %.2f s\n", "motion generated", seconds);
    std::printf("  %-54s %.1fx\n", "faster than playback, cold",
                seconds * 1000.0 / std::max(pipeline.GenerationMs(), 1e-6));

    // The cold figure is NOT asserted for DirectML. The EP compiles per
    // distinct input shape, and the sampler presents a new sequence length
    // every other step, so a first pass pays that compilation for roughly a
    // dozen shapes -- measured at about 1.1 s of the 2.1 s first run here. The
    // warmed figure after the re-run below is the one that describes a stream.
    if (device == FloodDiffusionDevice::Cpu) {
        Expect("generation outruns playback", pipeline.GenerationMs() < seconds * 1000.0);
    } else {
        std::printf("  (not asserted cold: the DirectML EP compiles per input "
                    "shape)\n");
    }

    // A second generation has to reuse the cached text features rather than
    // re-running the text encoder, which is what keeps a prompt change cheap.
    std::printf("\nre-running the same prompt\n");
    const double firstRun = pipeline.GenerationMs();
    if (!pipeline.Begin(prompt, options, &error)) {
        std::printf("  %s\n", error.c_str());
        ++g_failures;
        return;
    }
    ExpectInt("the frame count reset", pipeline.FrameCount(), 0);
    guard = 0;
    while (pipeline.IsRunning() && guard++ < 1000) {
        if (!pipeline.Step(&error)) {
            std::printf("  %s\n", error.c_str());
            ++g_failures;
            return;
        }
    }
    ExpectInt("the same number of frames came out", pipeline.FrameCount(),
              expectedFrames);
    std::printf("  %-54s %.1f ms (first was %.1f ms)\n", "second generation",
                pipeline.GenerationMs(), firstRun);
    std::printf("  %-54s %.1fx\n", "faster than playback, warmed",
                seconds * 1000.0 / std::max(pipeline.GenerationMs(), 1e-6));

    // This is the property a stream actually depends on: once the shapes are
    // compiled, generation has to stay ahead of 20 fps playback. It holds on
    // both providers, though with far less headroom on DirectML.
    Expect("warmed generation outruns playback",
           pipeline.GenerationMs() < seconds * 1000.0);

    if (run != nullptr) {
        run->milliseconds = pipeline.GenerationMs();
    }
}

// Both providers must produce the same motion from the same prompt and seed.
// This is what separates "the GPU path runs" from "the GPU path is correct":
// a provider that silently mis-executes an operator still yields a skeleton
// that moves.
void CompareProviders(const Run& cpu, const Run& gpu) {
    std::printf("\nCPU against DirectML, same prompt and seed\n");
    if (!cpu.ok || !gpu.ok) {
        std::printf("  one of the runs did not complete; nothing to compare\n");
        ++g_failures;
        return;
    }
    ExpectInt("the same number of frames", gpu.frames, cpu.frames);
    ExpectInt("the same number of joints", gpu.joints, cpu.joints);
    if (gpu.positions.size() != cpu.positions.size()) {
        Expect("the position arrays are the same size", false);
        return;
    }

    float worst = 0.0f;
    double total = 0.0;
    float scale = 0.0f;
    for (std::size_t i = 0; i < cpu.positions.size(); ++i) {
        const DirectX::XMFLOAT3& a = cpu.positions[i];
        const DirectX::XMFLOAT3& b = gpu.positions[i];
        const float dx = std::fabs(a.x - b.x);
        const float dy = std::fabs(a.y - b.y);
        const float dz = std::fabs(a.z - b.z);
        worst = std::max(worst, std::max(dx, std::max(dy, dz)));
        total += dx + dy + dz;
        scale = std::max(scale, std::max(std::fabs(a.x),
                                         std::max(std::fabs(a.y), std::fabs(a.z))));
    }
    const double mean = total / (cpu.positions.size() * 3);
    std::printf("  %-54s %.6f\n", "largest single-coordinate difference", worst);
    std::printf("  %-54s %.6f\n", "mean absolute difference", mean);
    std::printf("  %-54s %.3f\n", "largest coordinate magnitude", scale);

    // Not bit-exact, and should not be expected to be: the providers use
    // different kernels and accumulation orders, and the root is INTEGRATED
    // over the take, so a small per-frame difference compounds along the
    // trajectory. The bound is therefore relative to the motion's own scale.
    const float tolerance = std::max(0.05f, scale * 0.05f);
    const bool agree = worst <= tolerance;
    std::printf("  %-54s %.4f  %s\n", "within 5% of the motion's scale", tolerance,
                agree ? "ok" : "MISMATCH");
    if (!agree) ++g_failures;

    std::printf("  %-54s %.1f ms vs %.1f ms (%.1fx)\n",
                "CPU vs DirectML, warmed", cpu.milliseconds, gpu.milliseconds,
                gpu.milliseconds / std::max(cpu.milliseconds, 1e-6));
}

// Pinned shapes against the growing prefix, on the same provider, prompt and
// seed. These are NOT expected to agree closely: submitting the whole buffer
// lets the not-yet-denoised tail attend into the live frames, which a
// non-causal denoiser will act on. What matters is that the result is still
// motion -- finite, human-sized, bones holding together -- and how much faster
// it is. Making it exact needs an export that masks self-attention over the
// padding.
void ComparePinned(const Run& prefix, const Run& pinned) {
    std::printf("\ngrowing prefix against pinned shapes, same provider and seed\n");
    if (!prefix.ok || !pinned.ok) {
        std::printf("  one of the runs did not complete\n");
        ++g_failures;
        return;
    }
    ExpectInt("the same number of frames", pinned.frames, prefix.frames);

    float worst = 0.0f;
    double total = 0.0;
    for (std::size_t i = 0; i < std::min(prefix.positions.size(), pinned.positions.size());
         ++i) {
        const DirectX::XMFLOAT3& a = prefix.positions[i];
        const DirectX::XMFLOAT3& b = pinned.positions[i];
        const float d = std::max(std::fabs(a.x - b.x),
                                 std::max(std::fabs(a.y - b.y), std::fabs(a.z - b.z)));
        worst = std::max(worst, d);
        total += d;
    }
    std::printf("  %-54s %.4f\n", "largest joint difference", worst);
    std::printf("  %-54s %.4f\n", "mean joint difference",
                total / std::max<std::size_t>(prefix.positions.size(), 1));
    std::printf("  %-54s %.1f ms vs %.1f ms (%.2fx)\n", "prefix vs pinned, warmed",
                prefix.milliseconds, pinned.milliseconds,
                prefix.milliseconds / std::max(pinned.milliseconds, 1e-6));
    std::printf("  (a large difference is expected and is a quality question; see\n"
                "   skills/onnx-directml-ep/SKILL.md on masking the padding)\n");

    Expect("pinning shapes made it faster", pinned.milliseconds < prefix.milliseconds);
}

void TestFailures() {
    std::printf("\nrefusals\n");
    FloodDiffusionPipeline pipeline;
    FloodDiffusionOptions options;
    std::string error;

    options.device = FloodDiffusionDevice::Cpu;
    Expect("a missing directory is refused",
           !pipeline.Load("resources/NoSuchPipeline", options, &error));
    Expect("  with an error that names the file",
           error.find("pipeline.txt") != std::string::npos);
    Expect("an unloaded pipeline refuses Begin",
           !pipeline.Begin("anything", options, &error));

    FloodDiffusionPipeline other;
    options.device = FloodDiffusionDevice::Cpu;
    options.textEncoderDevice = FloodDiffusionDevice::Cpu;
    if (std::filesystem::exists(kDirectory / "pipeline.txt") &&
        other.Load(kDirectory, options, &error)) {
        options.latentFrames = 0;
        Expect("zero latent frames is refused",
               !other.Begin("a person walks", options, &error));
    }

    // DirectML needs the devices. Asking for it without them has to fail
    // loudly rather than silently fall back to the CPU.
    FloodDiffusionPipeline directml;
    FloodDiffusionOptions dmlOptions;
    dmlOptions.device = FloodDiffusionDevice::DirectML;
    Expect("DirectML without a device is refused",
           !directml.Load(kDirectory, dmlOptions, &error));
    Expect("  with an error that says why",
           error.find("DirectML") != std::string::npos);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    TestSchedule();

    Devices devices;
    const bool hasGpu = devices.Create();
    std::printf("\nDirectML device: %s\n", hasGpu ? "created" : "UNAVAILABLE");

    FloodDiffusionConfig config;
    if (std::filesystem::exists(kDirectory / "pipeline.txt")) {
        if (TestConfig(&config)) {
            Run cpu;
            Run gpu;
            TestPipeline(config, FloodDiffusionDevice::Cpu, devices, &cpu);
            if (hasGpu) {
                TestPipeline(config, FloodDiffusionDevice::DirectML, devices, &gpu);
                CompareProviders(cpu, gpu);

                // Shapes pinned, which is what lets the DirectML EP compile the
                // graph instead of dispatching per operator. It also changes the
                // sampler: the whole latent buffer is submitted every step, and
                // the denoiser is non-causal, so the comparison below is a
                // QUALITY question rather than a correctness one.
                Run pinned;
                TestPipeline(config, FloodDiffusionDevice::DirectML, devices, &pinned,
                             true);
                ComparePinned(gpu, pinned);
            } else {
                std::printf("\nno DirectML device; skipping the GPU sections\n");
            }
        }
    } else {
        std::printf("\n%s is not staged; skipping the model sections\n",
                    kDirectory.string().c_str());
        std::printf("run: python tools/prepare_flooddiffusion.py "
                    "<FloodDiffusion>/onnx_models/tiny\n");
    }
    TestFailures();

    std::printf("\n%s\n", g_failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
