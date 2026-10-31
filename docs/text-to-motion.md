# Text to motion

The one model integrated so far, and the first step toward characters that
react to what is around them: the FloodDiffusion streaming pipeline, its
execution providers and measured cost, latency and backpressure, the HumanML3D
features, and retargeting onto a rigged character. Back to the
[README](../README.md); the write-up is [AI-Integrated Renderer](https://jack111331.github.io/BlogPage/ai_render_engine/).

`//Engine:FloodDiffusion` streams
[FloodDiffusion](https://github.com/ShandaAI/FloodDiffusion) text-to-motion and
the application draws the result as skeleton lines, in the **FloodDiffusion
motion** panel.

It is four ONNX models with the sampling loop on the host:

| Model | Signature |
|---|---|
| `text_encoder.onnx` | token ids → `[tokens, 768]` features |
| `denoiser.onnx` | one diffusion-forcing denoising call |
| `vae_decoder_first.onnx` | first latent → 1 motion frame + 20 caches |
| `vae_decoder_step.onnx` | next latent + caches → 4 motion frames + 20 caches |

**Why it streams.** Diffusion forcing gives each latent frame its own noise
level, rising along the sequence, and every step shifts that ramp forward by a
fraction of a chunk. Latents at the front finish while the ones behind are
still noisy, so `FloodDiffusionPipeline::Step` denoises once and then decodes
whatever that finalised. Motion appears after roughly 8 of 128 steps and keeps
arriving, rather than all at the end. One step per rendered frame generates
about five times faster than playback consumes it.

**Either execution provider runs it, and the choice is per generation** —
`FloodDiffusionOptions::device` for the sampler, `textEncoderDevice` for the
prompt encoder, both switchable in the panel.

An earlier export only ran on the CPU: the DirectML EP faulted on the
attention mask, which the graph built from `context_lens` with a
`Range → Reshape → Less` chain. That export was replaced and DirectML now runs
every model. Measured on an RTX 5080, with both providers agreeing to within
**0.000004** on the largest joint coordinate:

| | DirectML | CPU |
|---|---|---|
| denoiser | 37–40 ms | 1.4–10.5 ms |
| VAE step | 3.3 ms | 1.3 ms |
| text encoder | 46 ms | 9.4 ms |
| 8-latent generation, warmed | 955 ms | 83 ms |

**The CPU is currently about 11× faster, and the reason is visible in the shape
of the numbers**: the denoiser costs the same 40 ms whether one latent is live
or sixty-four, while the CPU scales 1.4 → 10.5 ms over the same range. That is
per-dispatch overhead, not compute. ORT profiling confirms it — 1226 of the
graph's 1334 nodes are dispatched to DirectML individually, with 108 left on
the CPU. A 36 MB model with `hidden_dim` 256 and a classifier-free-guidance
batch of two does not fill a 5080; it starves it.

Two consequences worth knowing before choosing DirectML:

* **The first pass pays shape compilation.** The EP compiles per distinct input
  shape and the sampler presents a new sequence length every other step, so a
  cold 8-latent run took 1252 ms against 955 ms warmed.
* **It contends with rendering.** A denoising step measured 44 ms standalone
  but 91 ms inside the running application, because ORT's work goes on the
  same queue as the renderer. At one step per frame that is a visible hitch
  while generating. A worker thread, or a lower step rate, is the fix.

**The EP already has the compiler needed to fix this; it just cannot use it.**
ONNX Runtime's `DmlGraphFusionTransformer` compiles each DirectML-assignable
partition into one `IDMLCompiledOperator` and dispatches it as a single
`DmlFusedNode`. It is on by default, but it only engages when shapes are
statically known, because `IDMLDevice1::CompileGraph` needs concrete tensor
descriptions. Counting `Node` events in an ORT profile:

| session | dispatches per run | on DML | on CPU | fused nodes |
|---|---|---|---|---|
| as exported, all dims symbolic | **1334** | 1226 | 108 | none |
| `batch` and `time` pinned | **294** | 239 | 55 | 10 partitions |
| all four dims pinned | **1** | 1 | 0 | `DmlFusedNode_0_2` |

Fixing the shapes collapses the whole graph to one dispatch with no CPU
partitions at all, and `ep.dml.disable_graph_fusion` confirms the mechanism:
static shapes cost 2.0 ms with fusion and 10.2 ms without. (GPU-side tensor
binding is **not** the lever — that saves copies, and copies are not where the
time goes. See [`skills/onnx-directml-ep`](../skills/onnx-directml-ep/SKILL.md)
§6–§7.)

**`FloodDiffusionOptions::staticShapes` turns this on without a re-export**, via
`AddFreeDimensionOverrideByName`, and it is the "pin shapes" checkbox in the
panel. An 8-latent generation:

| | warmed | cold |
|---|---|---|
| CPU | 87 ms | 89 ms |
| DirectML, as exported | 966 ms | 1289 ms |
| DirectML, shapes pinned | **60 ms** | **171 ms** |

So pinned DirectML is the fastest configuration available, 16× the dynamic GPU
path and 1.5× the CPU, and the cold penalty falls 7.5× because there is only
one shape left to compile. In the running application a denoising step went
from 91.5 ms to 14.66 ms.

Per denoising call the write-up puts the same change at **42.45 ms mean and
98.43 ms worst dynamic against 5.54 ms and 6.42 ms pinned**, at 10 sampling
steps on an RTX 5080 — 7.7× on the mean and 15.3× on the worst. The worst case
is the number that decides whether a step fits inside a frame, which is why
pinning is what made streaming generation affordable at all rather than merely
faster.

Two things to know before switching it on, both deliberate:

* **It changes the sampler.** `time` has to be pinned too, so the whole latent
  buffer is submitted every step rather than the live prefix, with the noise
  ramp clipped to 1 over the part that is not live. The denoiser is non-causal,
  so that padding attends into the real frames: the motion differs (mean 0.11 m,
  max 0.52 m against the prefix version, on a figure about 1.6 m tall). Bone
  lengths and proportions hold, and it is arguably closer to how the model was
  trained, but it is a different sampler. An export taking a valid-length input
  and masking self-attention would make it exact. That is why the default is
  off.
* **The length is compiled in.** The pinned `time` is `latentFrames +
  chunk_size`, baked at load, so `Begin` refuses a different length and the
  panel's slider reloads the sessions.

**The VAE decoders run on the CPU provider, and that is the pipeline's single
biggest saving.** `vae_decoder_step` runs 59 of a generation's 60 decodes, and
moving them off DirectML takes the decode total from 188.5 ms to 69.2 and the
generation from 512.3 ms to 354.1 at 60 latents.
`FloodDiffusionOptions::vaeDevice` selects it; the Performance panel reports
which provider each of the three model groups ended up on.

The reason is not that the decoder is slow on the GPU — it is that **the CPU is
a second processor and the compute queue is not.** Every session here shares one
queue, which is what orders a decode after the denoising step that produced its
latent without a fence, so a DirectML decode queues behind the denoiser and the
denoiser behind it: putting the decoders back on DirectML raises the decode from
69 to 185 ms *and* the denoising from 281 to 327. On the CPU the decode runs
concurrently on cores that were idle anyway. A from-source ORT 1.31 build makes
the DirectML decode faster per call than the CPU's (0.80 ms against 1.3) and
the generation still measured 513 ms against 348, which is the clearest
statement of the point.

Their `batch` is left dynamic, and the reason is not the one previously
recorded here. Pinning it on `vae_decoder_first` makes the DirectML EP return a
**null for one of its 21 outputs** (`cache_out_9`), which is what the hand-off
failure actually was — not "a statically shaped session cannot hand its outputs
to another". `cache_out_9` is the only one produced by a `ConstantOfShape`, so
pinning lets constant folding turn it into a graph initializer, and the EP's
graph fusion then deletes that initializer without noticing it is also a graph
output — root-caused under a debugger and written up with the call stack in
`skills/onnx-directml-ep/SKILL.md` §5.0, along with the one-config-entry
workaround that is not worth taking. Pinning only `vae_decoder_step` works and
buys nothing measurable.
Binding the caches to D3D12 buffers to remove the hand-off does not help
either: `Ort::IoBinding` on `vae_decoder_step` fails in every configuration on
ORT 1.24.4, while the identical tensors through the ordinary `Run` API work —
host memory included, so it is the binding and not the memory.
`//Engine:fd-probe` section 11 keeps all of it measured.

**The stream runs until Stop.** `FloodDiffusionOptions::continuous` is what
diffusion forcing is actually for, and it needs almost no new machinery,
because the sampler is already in a steady state: the live region is exactly
`chunk_size` latents wide at every step (`EndAt - StartAt` is constant), so
once the ramp has run up, each `denoiseSteps` steps consume `chunk_size` fresh
noise latents and finalise `chunk_size`. A finite run spends all but its first
moments there and then simply declines to carry on. Continuous mode keeps it
there: the window slides, dropping finalised latents off the front and
appending fresh noise at the back. `latentFrames` becomes the window — how much
finalised context the denoiser still sees — and the ramp-up is identical, step
for step, to a finite generation of the same length, which
`//Engine:flooddiffusion-smoke` asserts bit-for-bit.

It needs `staticShapes`, because a sliding window *is* a fixed submitted length.
The GPU-resident path slides with buffer copies through a scratch resource
rather than a `memmove`: D3D12 forbids a `CopyBufferRegion` within one
resource, and `x` goes source-then-dest in one command list, so it also needs
an explicit barrier between the two — the debug layer rejects it otherwise with
`Missing State: D3D12_RESOURCE_STATE_COPY_DEST`.

The part worth knowing is the **backpressure**. A latent may only be dropped
once it has been decoded, or its motion would be lost, so when the decoder is
behind the sampler stalls — `Step` returns true having done nothing, which is
not an error. Together with the paced decode below that regulates the whole
thing to real time, so a stream left running costs about what playback costs.
`Stop()` stops the sampler and not the generation: what it already finalised
still has to become motion, so `IsRunning()` stays true until the decoder has
drained. Motion accumulates for as long as the stream runs, about 94 MB an
hour, and nothing trims it yet.

**Sampling steps are a control, not a constant.** `denoiseSteps` — steps each
chunk of latents gets before it is final — is on a slider in the Neural tab,
with a button back to the export's own value. It is read by `Begin` rather than
`Load`, so unlike the window it needs no session rebuild; it takes effect on
the next Generate, because building a new noise ramp under latents already in
flight would make them jump.

It does **not** have to divide `chunk_size`, which is the thing the slider
would otherwise invite someone to get wrong: the config loader rejects an
*export* whose `noise_steps` is not a multiple, so the natural assumption is
that a hand-picked count must be one too. It need not. A count below
`chunk_size` is the interesting case — the frontier then finalises several
latents per step instead of creeping — and it holds because the frontier
advances by at most `ceil(chunk_size / steps)` latents per step, which never
exceeds `chunk_size`, and `chunk_size` is exactly what the GPU sampler's slide
ring is sized for. `//Engine:flooddiffusion-smoke` runs 3, 7, 10 and 20, plus a
sliding window at 3.

Measured at 8 latents, and the cost is linear in the schedule rather than in
the control: 3 steps is 7 schedule steps and 55.6 ms, 7 is 16 and 67.9 ms, the
exported 10 is 23 and 112.5 ms, 20 is 47 and 142.3 ms. The ~40 ms that does not
scale is the eight decodes and the setup; the rest is 2.2 ms a denoising step,
which is the figure from the table below. Quality is not something these
numbers can speak to — the export is **distilled for 10**, so fewer degrades
the motion and many more mostly buys time, and that needs eyes rather than an
assertion.

**Reaction latency, and the backlog that actually governs it.** `PromptLatencyMs()`
measures from a prompt change to the first pose that could carry it, and the
definition is half the measurement: a latent already finalised when the prompt
changed will never be denoised again, so the first *new* pose is the first
frame decoded from a latent at or past the frontier at that moment. Timing to
the next frame of any kind would report a couple of milliseconds and mean
nothing.

Decoding **eagerly** it is **7 ms, two denoising steps** — against about 20 ms
to start from nothing, because it only finishes latents already in flight
rather than climbing the ramp. That is the number this section used to quote,
and on its own it was misleading: the panel runs decoding **paced** to the
playhead, and there the picture inverts.

Paced, the sampler runs ahead of the decoder and leaves finalised latents it
has not reached. Every one carries the old prompt and the decoder is strictly
causal, so all of them must be decoded and played before the change is visible.
Measured at a 30-latent window, changing the prompt early:

| sampling steps | backlog (latents) | frames to react |
|---|---|---|
| 3 | 31 | 127 |
| 10 | 10 | 43 |
| 20 | 1 | 5 |

**A lower step count reacts slower**, which is the opposite of the eager
ordering and looks like a bug until the backlog column explains it: fewer steps
finish the ramp sooner and so run further ahead. In the application a frame is
a rendered frame against a real-time playhead, so 127 of them is about **6.4
seconds**. The backlog saturates at the window, which is therefore the ceiling:
at the shipped `latentFrames` of 60 that is 60 latents of old prompt, about 12
seconds.

**`FloodDiffusionOptions::samplerLeadFrames` fixes it** by bounding the
sampler's lead over the *decoder* rather than only over the window — continuous
mode's backpressure only stops the sampler overwriting an undecoded latent,
which at a 60-latent window is far too late. The panel holds it at 30 frames,
a second and a half, on a slider. Same measurement with the cap on:

| sampling steps | uncapped | capped at 30 frames |
|---|---|---|
| 3 | 127 | 35 |
| 10 | 127 | 35 |
| 20 | 5 | 5 |

The worst case improves 3.6x, and the two low counts come out **identical** —
the reaction stopped being a function of the step count, which is the part that
made it feel arbitrary. Twenty steps is quicker still because it never reaches
the cap on its own; this is a ceiling, not a floor.

It costs nothing in throughput. The sampler already runs about six times faster
than playback, so holding it a second and a half in front of the decoder loses
no motion — it only stops it banking work nobody has asked to see. The stall
shows up as `Step` returning true having done nothing, like the window
backpressure it sits in front of, and it is inert when decoding eagerly because
`Step` decodes to the frontier before returning: the 60-latent benchmark above
is unchanged to within noise.

Per motion frame produced, the denoiser runs 0.54 times and the decoder 0.25 —
**1.85 frames per denoise call, 3.95 per decode call**. That is the shape of
the pipeline in one line, and the 3.95 is `vae_frames_step` with the first
latent's single frame averaged in. Model weights are **1226.7 MB** of `.onnx`
with the text encoder loaded, of which 1.1 GB is the encoder — which is why it
is loaded on demand and why it defaults to the CPU provider.

The smoke test prints its conditions with the numbers, because none of them
mean anything without: provider split, latent count, pinned shapes, which
generation was measured, that the renderer is not running, and that every
figure is **CPU wall clock around each `Run` with no GPU timestamps**.

Two of these are instrumented but have no numbers here. **p95/p99 frame time,
video memory, render target and vsync state** are in the Performance overlay
and in a one-shot stdout dump behind `PTFLIO_FRAME_REPORT=1`, but they need an
interactive run — the UI loop does not tick when the app is launched from a
tool, so nothing is drawn and nothing is sampled. And **FP32 against FP16**,
and **the original step count against the 10-step distilled export**, cannot be
measured at all yet: only the distilled FP32 export is staged, so both need new
assets rather than new code and should not be quoted until someone exports
them.

**The GPU-resident denoiser, measured as execution rather than submission.**
Every earlier figure for that path carried a caveat — a `Run` there records and
returns, so a CPU clock around it times how long it took to *ask*. D3D12
timestamp queries remove the caveat. The execution provider records onto its
own command lists, so there is nothing of ours to put a query inside; what
makes it work is that the queue is in-order and the EP submits to the same one,
so a timestamp written by the list **before** the `Run` and another by the list
**after** it bracket the EP's work whatever lists it used in between. Both
markers ride along on submissions the sampler was making anyway, so measuring
does not inflate the submission figure it is being compared against, and a
slot is read only once its fence has retired, so nothing ever waits.

| denoiser, 128 calls | mean | max | total |
|---|---|---|---|
| submission (CPU clock) | 0.48 ms | 2.59 ms | 62 ms |
| GPU execution (timestamps) | 2.05 ms | 2.92 ms | 262 ms |

**Submission understated the work by about 4x**, which is the size of the error
every earlier reading carried. It also reconciles two numbers that never
agreed: the host path's CPU-side denoiser is 2.22 ms a call — honest, because
reading the prediction back forces it to wait — against 2.05 ms of GPU
execution, and the 0.17 ms between them is the host↔device copies the device
path removes. Real, and nothing like the 4x the submission figure implied. **The
device path was never faster at denoising; it was faster at not waiting**, which
is what the earlier end-to-end totals had been saying all along.

A measurement-hygiene note, because it cost a confused half hour: these numbers
moved by 2x while a stray uncapped `AICGTestbed` from an earlier launch was
still rendering in the background. Check for one before trusting any of this.

**Per-call cost, and the start-up latency.** `DenoiseTiming()` and
`DecodeTiming()` report calls, mean, max and total for each half, and
`FirstFrameMs()` reports the wall time from `Begin` to the first motion frame.
The pair matters because the mean and the maximum answer different questions:
the mean says where a generation's time goes, the maximum says what a frame
budget has to survive, and one slow call is what drops a frame. Measured on an
RTX 5080 at the shipped 60 latents, host latents, decoders on the CPU:

| | calls | mean | max | total |
|---|---|---|---|---|
| denoiser | 128 | 2.29 ms | 3.26 ms | 294 ms |
| VAE decoder | 60 | 1.43 ms | 2.11 ms | 86 ms |

Start-up latency is **about 20 ms** — the ramp has to climb far enough to
finalise the first latent before there is anything to decode, so it is
denoising however cheap the decode becomes. The maxima sit at roughly 1.4x the
means, so neither half has a tail worth worrying about, which is what makes a
step budgetable at all.

The device-resident path is where these numbers earn their keep. Its decoder is
the same 1.4–1.5 ms a call, but `DecodeTotalMs` reads ~287 ms against ~90 ms of
actual decoder calls: **the ~200 ms difference is the latent readback waiting
for denoising the GPU is really doing**, which is the first direct measurement
of the accounting artefact described under `gpuLatents` rather than an
inference from the totals. Its denoiser row reads 0.54 ms mean because on that
path a `Run` records and returns — submission, not execution — and the smoke
test labels it as such so the number cannot be misread.

**The frame rate is not capped, and nothing moves faster because of it.**
`DeviceResources` is created with `c_AllowTearing` and presents with a sync
interval of 0, so the rate is the machine's rather than the display's; the
profiling overlay has a vsync checkbox to put the wait back, and
`CreateDeviceResources` clears the flag and forces the wait on an adapter that
cannot tear. `DX::StepTimer` was already variable (`m_isFixedTimeStep` is
false), so everything that reads `GetElapsedSeconds` was correct to begin with
— playback advances `m_playbackTime` by the measured delta and derives the
motion frame from it, so the retargeted character moves at 20 fps whatever the
renderer is doing.

Two things *were* frame-rate dependent and are not now:

* **The sampler's budget was per frame.** `steps / frame` meant the sampler ran
  five times faster at 300 FPS and could not keep playback fed at 20. It is now
  `steps / second`, converted with the measured delta and carrying the leftover
  fraction so a rate below the frame rate still advances. The default of 60 is
  what one step per frame meant at 60 FPS; the rate that matches playback
  exactly is `noise_steps` per second of motion, which is 10 at the shipped
  shapes, so the default generates about six times faster than real time.
* **The camera scaled mouse deltas by the frame time.** A `MODE_RELATIVE`
  mouse reports a per-frame *displacement*, so multiplying by the delta counted
  the frame twice and the same physical drag moved the camera less the faster
  the frame rate got — invisible at a pinned 60 Hz, obvious uncapped. The
  speeds are now per pixel of drag, with the defaults divided by a 60 Hz frame
  so the feel on that display is unchanged.

Not addressed: `PoseCharacter` re-runs the full retarget every rendered frame
even though the posed frame only changes at 20 fps, so most of that work is
redundant uncapped. It is tens of microseconds against a two-millisecond
frame, which is why it was left alone rather than guarded with a cache that
could go stale.

**The prompt can change mid-stream.** `SetPrompt` re-conditions a generation
that is already running: it encodes the new prompt and replaces `context`, and
nothing else moves — the latents keep their noise levels, the schedule keeps
its step, the window keeps its position, and the motion already decoded stays
exactly as it was. The live chunk and the noise behind it are then denoised
toward the new prompt from wherever they had got to, so the motion
*transitions* into it over about a window rather than cutting. That is the
other thing diffusion forcing buys and a clip sampler cannot do at all: there
is no boundary to restart at, and `Begin` would reseed the latents and discard
the motion. `//Engine:flooddiffusion-smoke` asserts both halves against a
control stream that never switches — the frames decoded before the switch are
bit-identical, and the frames after it differ.

The panel applies it when the prompt field's edit finishes, on Enter or on
clicking away, not per keystroke: encoding is a synchronous text-encoder run of
about 9 ms, and the half-typed prompts on the way would each pay for one. It is
cached per prompt, so returning to one already used is free.

**Decoding follows the playhead, not the sampler.** The sampler finishes 11.85 s
of motion in about 2.1 s, so decoding eagerly spends the whole decode budget
inside that burst, on motion nobody has reached yet.
`FloodDiffusionOptions::decodeAheadFrames` takes decoding out of `Step` and
gives it to `DecodeAhead(consumedFrame)`, which decodes just far enough ahead
of a consumer — one `vae_decoder_step` is four motion frames, 0.2 s at 20 fps,
so a stream needs about five decodes a second however far ahead the sampler has
run. `MotionStreamView` uses a 12-frame lookahead. The motion is bit-identical
to eager decoding, because pacing defers decoding without reordering it.

**The output is not joint positions.** The VAE decodes to HumanML3D's
263-dimensional feature vector: a root trajectory as per-frame velocities plus
joint positions in a root-local frame. `Engine/MotionFeatures.cc` integrates
the root and un-rotates the joints. The subtlety worth knowing is that the
batch formulation shifts the velocity arrays by one frame before integrating,
so frame *i* is placed using frame *i-1*'s velocity; applying the current
frame's instead still produces plausible motion that drifts one frame ahead of
the pose. `//Engine:motionfeatures-smoke` pins this against
`tools/motion_reference.py`, an independent numpy transcription, and against
properties that hold either way — a constant velocity must integrate to a
straight line, a pure yaw must not change a bone's length.

**Reproducibility.** The same seed gives the same motion within the engine but
*not* the same motion as the Python reference: that seeds numpy's PCG64 with a
ziggurat normal, and this uses `std::mt19937`.

**Retargeting onto a rigged character is driven by the positions, not the
rotations.** A feature frame carries the same pose twice — `ric` joint
positions in the root-local frame, and a block of (J-1) six-dimensional local
joint rotations — and `Engine/MotionRetarget.cc` can drive a Mixamo glTF rig
from either (`RetargetSource`, the **drive from positions** checkbox in the
Neural tab). Rotations are the better input in principle, since a bone
direction fixes only two of three degrees of freedom and so cannot carry the
roll about the bone's own axis. They are not the better input here.

`//Engine:retarget-probe` is what decided it. Forward kinematics says
`position[j] - position[parent] = worldRotation[j] * restOffset[j]`, and
`restOffset` is a property of the skeleton rather than of the pose — so
pulling each measured bone direction back through the rotation that is supposed
to have produced it must give the same vector on every frame, and the scatter
of that over a take scores a candidate reading. The probe sweeps 60 of them:
local-accumulated against world, both product orders, the reconstructed frame
against its transpose, two block layouts, five root-angle conventions. One real
bug came out of the sweep — the root was seeded with `cos(a/2)` where
HumanML3D stores `cos(a)`, i.e. half the rotation, which is invisible while the
character walks in a straight line — and fixing it took the best score from
67.0 to **51.7 degrees** of mean scatter. A correct reading would score near
zero, so the remaining 52 degrees is the block not describing the pose `ric`
describes, and no reading can repair that.

Driving from the positions instead scores **0.03 degrees** under the same
measure, which is float noise: the direction is taken from the positions and
the rotation is built to reach it, so there is no convention left to get wrong.
Posed onto the real character, each bone lands **0.00 degrees** mean (0.03
worst) from the direction the skeleton overlay draws, against **32.65** mean
(165.50 worst) for the rotation path — which is the "the skeleton looks fine
but the character looks weird" report, as a number.

**The character's size wins, and the motion is brought down to it.** A
HumanML3D skeleton is not the size of any particular glTF character — this
one's hips measure 0.774 m against the source's ~0.9 — and rotation-only
retargeting cannot change proportions, so something has to give. It used to be
the character: the ratio of hip heights was written onto the hips node as a
*scale*, resizing the whole joint hierarchy and the skinned mesh with it. That
is wrong in a way that only shows up once a character has a size of its own.
It silently discarded whatever the file and the scene manifest had given it, so
a character deliberately placed at `"scale": 0.8` came out at 0.8 of the
*motion's* size rather than 0.8 of its own — about 16% larger — and only when
generated motion drove it, because glTF animation writes no scale channel here
and left the authored size alone.

Now `ApplyFrame` takes `motionScale` — how much bigger the motion is than the
character — and **divides the hips target by it**. The rig keeps the size its
file and its placement give it; the root height and the stride arrive at the
character's proportions, which is what keeps the feet on the floor without
resizing anything. `//Engine:scenegraph-smoke` asserts the hips land at half
the requested height at a ratio of 2, and that the hips node's scale is
untouched.

`SceneGraph::ApplyPoseOverrides` was fixed with it: an override carrying a
scale now **composes** with the authored one instead of replacing it. That is
deliberately unlike a glTF animation scale channel, and the reason is the same
— an override is a pose laid on a character that the file and the manifest have
already sized, so replacing its scale throws that away. A caller wanting an
absolute size divides by the authored scale itself, which it can see.

**Aiming is not posing**, and this is the part worth reading before trusting a
direction test. A bone direction fixes two of three degrees of freedom. The
third is the twist about the bone, and at the hips that twist is *which way the
character faces*, because the hips bone points straight up and a rotation about
it is pure yaw. The first version aimed every bone with a minimal arc from its
rest direction: it scored a perfect 0.00 on directions and left the character
facing **91.5 degrees** away from the motion on average, up to 172.9 — torso
at its rest facing, legs following the motion, the mesh wrung around the waist.
Nothing in a per-bone direction check can see that.

A **second direction** resolves it, and the positions already carry one: the
line between the two joints that branch off the bone's own joint. At the hips
that is hip to hip, at the upper spine shoulder to shoulder, which are exactly
the two places the twist is worth pinning; both are found from the chains, so
no new configuration is needed. Those two bones get a full 3-DOF frame, and
every other bone *inherits* its parent's resolved twist and adds only the
minimal arc that aims it — deliberately against the parent's already-final
rotation rather than against the rest pose, or the spine would snap back to the
rest facing one bone below the hips that were just turned. Facing error is now
**2.2 degrees** mean on real motion, with bone directions unchanged at 0.00.
`//Engine:scenegraph-smoke` asserts the facing separately from the directions,
because the direction assertion read 0.00 through the whole bug.

What is still free is the roll about a limb's own axis relative to its parent,
which no position can supply, so **the character keeps its authored roll
there** — visible on the forearms, which pronate without moving the wrist.
The rotation path is kept rather than deleted, because an export with a
faithful rotation block would want it back, and it is the only source that
carries that roll.

**Not done yet.** Only a single prompt per generation is supported (the
reference also accepts several prompts with `--text-end` handing over at given
frames, which would widen the `segments` axis of `context`).

