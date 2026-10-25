// The sampler's arithmetic, on the GPU, so the latents never leave it.
//
// The host path reads the denoiser's `prediction` back, combines the two
// classifier-free-guidance rows, takes an Euler step and uploads the latents
// again for the next call. This shader is the same arithmetic written where
// the data already is, dispatched onto the same compute queue the DirectML
// execution provider submits to, so queue order alone sequences it after the
// denoise it reads -- and nothing in the step loop has to wait.
//
// It does not make a generation faster; see FloodDiffusionOptions::gpuLatents
// for the measurement and for what it is actually for.
//
// It MUST stay numerically identical to the loop at the end of
// FloodDiffusionPipeline::Impl::Denoise -- //Engine:flooddiffusion-smoke runs
// the two paths against each other and compares the motion.

// Mirrors LatentStepConstants in FloodDiffusion.cc. Root constants, so there
// is no constant buffer to allocate or align.
cbuffer LatentStepConstants : register(b0) {
    // The half-open latent range this step writes, as the schedule reports it.
    // Only the live window moves; everything outside it keeps its value, which
    // is what makes the ramp a ramp.
    uint Start;
    uint End;
    uint LatentDim;
    // 1, or 2 when classifier-free guidance is on.
    uint Rows;
    // The submitted sequence length, which is the per-row stride.
    uint Time;
    float CfgScale;
    float Dt;
    // 0 = vel, 1 = x0, 2 = noise. The exported pipeline is "vel"; the other
    // two are here because the host loop has them and the two paths have to
    // agree whatever the export says.
    uint PredictionType;
};

// Root descriptors rather than a descriptor table: three raw buffers need no
// heap, and a dispatch with no heap to bind cannot disturb whatever the
// renderer left bound on its own queue.
RWStructuredBuffer<float> Latents : register(u0);     // [Rows, Time, LatentDim]
RWStructuredBuffer<float> Prediction : register(u1);  // [Rows, Time, LatentDim]
RWStructuredBuffer<float> NoiseLevels : register(u2); // [Rows, Time]

[numthreads(64, 1, 1)]
void CSLatentStep(uint3 id : SV_DispatchThreadID) {
    const uint count = (End - Start) * LatentDim;
    if (id.x >= count) {
        return;
    }

    const uint index = Start + id.x / LatentDim;
    const uint component = id.x % LatentDim;
    // Row-major [Rows, Time, LatentDim]; row 1 begins one whole row in.
    const uint perRow = Time * LatentDim;
    const uint offset = index * LatentDim + component;

    // Classifier-free guidance: push away from what the model would predict
    // with no prompt at all.
    float value = Prediction[offset];
    if (Rows == 2) {
        const float unconditional = Prediction[perRow + offset];
        value = CfgScale * value - (CfgScale - 1.0f) * unconditional;
    }

    const float level = NoiseLevels[index];
    const float current = Latents[offset];
    float velocity = value;
    if (PredictionType == 1) {
        velocity = (value - current) / level;
    } else if (PredictionType == 2) {
        velocity = (current - value) / (1.0f + Dt - level);
    }
    const float updated = current + velocity * Dt;

    // Every guidance row sees the same latents -- only the text context
    // differs -- so the new value goes into both. Writing one and copying
    // would be a second dispatch for the same bytes.
    for (uint row = 0; row < Rows; ++row) {
        Latents[row * perRow + offset] = updated;
    }
}
