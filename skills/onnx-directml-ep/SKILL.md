---
name: onnx-directml-ep
description: "Read this before putting an existing ONNX model onto ONNX Runtime's DirectML execution provider from C++, or when a DirectML EP session fails to build, faults inside MLOperatorAuthorImpl, silently falls back to the CPU, turns out slower than the CPU provider, or breaks a D3D12 renderer it shares a device with. Covers package selection, the required and NOT-required session options, two C++ API lifetime traps that look like broken models, diagnosing graph-level faults, why GPU is often slower for small models and how to prove it, which input dimensions must be fixed at export time and which paddings are numerically exact, sharing a device and queue with a renderer, and how to validate a provider change. Not about CUDA or TensorRT, and not about writing DirectML operators by hand (that is DirectMLX)."
---

# Adopting an existing model onto the ONNX DirectML EP

Hard-won notes from putting a four-model text-to-motion pipeline
(FloodDiffusion) onto ONNX Runtime's DirectML execution provider inside a
Direct3D 12 renderer. Every number here was measured on an RTX 5080 with ONNX
Runtime 1.24.4; re-measure rather than trusting the figures, but the *shapes*
of the problems generalise.

The single most useful habit: **the DirectML EP fails and degrades quietly.**
It faults with access violations rather than clear errors, falls back to the
CPU per-node without telling you, and can be an order of magnitude slower than
the CPU provider while looking like it works. Assume nothing; measure.

---

## 1. Before you start: can you even have this provider?

**One `onnxruntime.dll` per process, and the redists are different builds of
it.** `Microsoft.ML.OnnxRuntime.DirectML` contains the DML EP;
`Microsoft.ML.OnnxRuntime.Gpu.Windows` contains CUDA and TensorRT. Neither
contains the other. So DirectML and CUDA are an either/or decision for the
whole process, not a per-session one — and if some other component in your app
already needs CUDA, you cannot also have DirectML.

**Check the package is still alive.** As of 2026-10 the DirectML redist's
latest stable was **1.24.4**, published 2026-03-17, while core ORT was at
**1.31.0** (2026-10-09) and had shipped 1.25 through 1.31 in between. A frozen
package is a signal about where effort is going. Query it rather than assuming:

```
https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/index.json
```

**And check what "upgrade" would even mean, because for DirectML there is no
prebuilt newer anything.** Four things to confirm before planning one, all
cheap and all decisive:

* The **core** package does not carry the DML EP as a fallback. Read the
  nupkg's zip central directory with a ranged request rather than downloading
  160 MB of it: `Microsoft.ML.OnnxRuntime` 1.31.0 ships `cpu_provider_factory.h`
  and no `dml_provider_factory.h`, and `runtimes/win-x64/native/` holds only
  `onnxruntime.dll` and `onnxruntime_providers_shared.dll`.
* The **GitHub releases** carry no DirectML asset either, at any recent tag.
* The **plugin EP** scheme that ORT 1.24 introduced, and that 1.31's notes
  defer provider detail to, has published CUDA and WebGPU only
  (`plugin-ep-cuda/v0.3.0`, `plugin-ep-webgpu/v0.4.0`). There is no DirectML
  plugin EP.
* The EP is nonetheless **still in the source tree and still being fixed** —
  1.31.0's notes mention validating DirectML upload-heap allocations. So the
  only route to a newer DML EP is a from-source build with `--use_dml`.

The official position matches: the EP docs say "DirectML is in sustained
engineering" — supported, but new feature work has moved to WinML, which wraps
the same ORT APIs and **chooses** the provider for you. That last part is the
catch for a renderer: this project hands ORT its own `ID3D12Device`,
`IDMLDevice` and a private compute queue through
`SessionOptionsAppendExecutionProvider_DML1` (§8), and an EP that selects
itself is not obviously able to take them. Price that before treating WinML as
the migration path.

**We built 1.31.0 from source and it was not worth adopting.** Recorded so the
next person does not spend the build on it. `v1.31.0` with

```
python tools\ci_build\build.py --build_dir build\Windows --config Release ^
  --build_shared_lib --parallel --use_dml --skip_tests --skip_submodule_sync ^
  --compile_no_warning_as_error ^
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_POLICY_VERSION_MINIMUM=3.5
```

builds clean on VS 17.14 and CMake 4.2.7, restores `Microsoft.AI.DirectML
1.15.4` itself, and drops a 24.6 MB `onnxruntime.dll` with the DML EP linked
in. Staging it needs the headers from `include/onnxruntime/core/session/*.h`
plus the two provider factories and `core/framework/provider_options.h` — 1.31
added `onnxruntime_error_code.h` and others the 1.24.4 redist did not have, so
copy the directory rather than the old file list. Our code compiled against it
unchanged.

What it bought: **both bugs in §5 reproduce exactly** — the pinned
`vae_decoder_first` still returns a null for `cache_out_9`, and `IoBinding` on
`vae_decoder_step` still fails with the same three errors in the same three
configurations. The denoiser is unchanged (1.64 ms static against 1.7). The
whole generation was 348.3 ms against 354.1, which is noise. Not worth trading
a two-constant redist fetch for a local build.

The one real difference is a **trap**, and it is the lesson worth keeping:
`vae_decoder_step` in isolation went **5.42 → 1.62 ms** a call, and 0.80 ms
with its `batch` pinned — suddenly beating the CPU's 1.3 ms and apparently
reversing §5.1. It does not survive the pipeline. Measured end to end with the
decoders moved back to DirectML, the generation went **348 → 513 ms**: the
decode total came out at 185 ms rather than the 96 the per-call figure
predicted, *and denoising rose from 281 to 327 ms*. See §5.1 for why.

**`DirectML.dll` is not in the package.** `onnxruntime.dll` imports it and
resolves against the in-box `%WINDIR%\System32\DirectML.dll`. The nuspec
declares the minimum version it was built against (1.15.4 for ORT 1.24.4);
check the in-box copy meets it and say so at fetch time, rather than letting a
load failure surface later on someone else's machine.

**Confirm the DLL beside your executable really is the DirectML build** before
blaming your model:

```cpp
for (const std::string& name : Ort::GetAvailableProviders()) { /* expect DmlExecutionProvider */ }
```

If `GetExecutionProviderApi("DML", ORT_API_VERSION, ...)` hands back null, you
have the wrong redist staged. Say that in the error message — it is a build
configuration problem, and a message about the model will send the reader the
wrong way.

**A release-only redist in a debug build is fine here.** ORT exposes a flat C
ABI and its C++ header is inline wrappers, so no CRT object crosses the DLL
boundary. This is unlike most vcpkg packages, where debug and release must be
paired.

---

## 2. Session options: what is actually required

DirectML's documentation asks for two options, and you should pass them —
they are free and documented:

```cpp
options.DisableMemPattern();              // the EP assumes fixed shapes
options.SetExecutionMode(ORT_SEQUENTIAL); // the EP has no parallel execution
options.DisableMemPattern();
Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML1(
    options, dmlDevice, commandQueue));
```

**But do not believe — as this project's notes wrongly recorded for a while —
that session creation *fails* without them.** Measured on ORT 1.24.4, a session
created with neither option set, and with `ORT_PARALLEL` explicitly requested,
still built, ran, and produced bit-identical output:

```
  neither set (ORT defaults)   ran, motion[0]=0.001035
  ORT_SEQUENTIAL only          ran, motion[0]=0.001035
  DisableMemPattern only       ran, motion[0]=0.001035
  both, as the docs say        ran, motion[0]=0.001035
```

Newer ORT appears to settle them for the EP itself. The practical consequence:
**never diagnose a DirectML failure as "the options were missing."** That
wastes time on a non-cause. Write a four-line check like the one above rather
than inheriting a claim.

---

## 3. Two C++ API lifetime traps that look exactly like broken models

Both of these cost real debugging time, and both present as the *model* being
wrong. They are worth checking first whenever a session misbehaves.

### `Ort::Env` must outlive the session

```cpp
Ort::Session session(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "app"), path, options); // BROKEN
```

The `Env` is a temporary; it owns the logger and dies at the end of the
statement. The session then fails at the first node that tries to log:

```
Non-zero status code returned while running Mul node. Name:'node_mul_14'
Status Message: ... LoggingManager::DefaultLogger
Attempt to use DefaultLogger but none has been registered.
```

That names an arbitrary node and reads exactly like an unsupported operator. I
concluded from this that a provider could not run a model; it could. Hold the
`Env` in a named local — one per process is correct, since ORT keys its logger
and thread pools off it.

### `Ort::TypeInfo::GetTensorTypeAndShapeInfo()` returns a non-owning view

```cpp
const auto info = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo(); // BROKEN
if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { /* garbage */ }
```

`TypeInfo` owns the `OrtTypeInfo`; `GetTensorTypeAndShapeInfo` returns a
`ConstTensorTypeAndShapeInfo` that merely points into it. Combining them in one
expression leaves the view dangling. The read-after-free returned the right
dtype for weeks and then started reporting a non-FLOAT32 tensor, rejecting
every model. Hold the `TypeInfo`:

```cpp
const Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
const auto info = typeInfo.GetTensorTypeAndShapeInfo();
```

Note `Ort::Value`'s identically-named method *does* return an owning object.
Only `TypeInfo`'s is a view.

---

## 4. When the EP faults on the graph

The DirectML EP's failures are access violations wearing an operator's name:

```
Non-zero status code returned while running Reshape node. Name:'node_Reshape_1920'
Status Message: ...\MLOperatorAuthorImpl.cpp(2597)... Exception(2) tid(284) 8007023E
{Application Error}
```

**Localise the node before theorising.** Walk the graph and print the named
node with its neighbours and producers. A minimal protobuf scanner is enough
and avoids a dependency on the `onnx` Python package — you only need
`ModelProto.graph` (field 7), `GraphProto.node` (1), and each `NodeProto`'s
input (1), output (2), name (3) and op_type (4).

In our case `node_Reshape_1920` reshaped the output of a `Range` feeding a
`Less` against `context_lens` — the attention mask, built *inside* the graph
from a runtime length. The pattern to look for is **shape arithmetic on
runtime values**: `Range`, `Shape`, `Squeeze`, `Slice`, `NonZero`. Exporting
with PyTorch's `dynamo=True` tends to produce exactly these chains.

**Establish whether it is the EP or your inputs.** Run the identical feeds on
the CPU provider as a control. If CPU succeeds, it is the EP. We confirmed the
fault at *every* graph optimisation level (`ORT_DISABLE_ALL` through
`ORT_ENABLE_ALL`) and every sequence length, which ruled out both optimisation
and dynamic shapes as the trigger.

**Fixes, cheapest first:**

1. **Check for a newer redist.** Sometimes this is a fixed EP bug. (Not
   available to us — the package was frozen.)
2. **Re-export so the pattern never appears.** Computing the mask on the host
   and passing it as an input, or resolving shapes statically, deletes the
   offending subgraph. This is what fixed it for us.
3. Per-node EP assignment is *not* a practical lever — ORT offers no clean
   per-node override, and the EP claims `Reshape` as supported, so CPU fallback
   never engages.

---

## 5. The big one: GPU is often slower, and you must prove which

Our first working DirectML run was **11× slower** than the CPU provider. This
is the normal outcome for small models and the most likely surprise of the
whole exercise.

| | DirectML | CPU |
|---|---|---|
| denoiser (36 MB, 1353 nodes) | 37–40 ms | 1.4–10.5 ms |
| VAE step | 3.3 ms | 1.3 ms |
| text encoder (1.1 GB) | 46 ms | 9.4 ms |

The VAE row held up when measured properly, over a threaded sequence at the
shipped shapes rather than one call: 5.4 ms against 1.4. Acting on it was the
pipeline's biggest win — §5.1.

### The diagnostic that tells you why, in one table

**Sweep the workload size and watch how the time responds.**

* Time that **scales** with the work is compute-bound.
* Time that is **flat** is overhead-bound — you are paying per dispatch, and
  the GPU is idle between operators.

Ours was flat: 40 ms whether 1 sequence position was live or 64, while the CPU
scaled 1.4 → 10.5 ms over the same range. That one observation localises the
problem before any profiling.

### Confirming it with ORT profiling

```cpp
options.EnableProfiling(L"prefix");
// ... run twice ...
const std::string file = session.EndProfilingAllocated(Ort::AllocatorWithDefaultOptions()).get();
```

Then count the `"cat": "Node"` events by `args.provider`. We saw **1226 of 1334
nodes dispatched to DirectML individually**, with 108 left on the CPU. Each
CPU-assigned node is a partition boundary with a copy and a sync.

This count is trustworthy. I initially assumed profiling would suppress the
EP's graph fusion in order to time nodes individually, and **it does not** —
the same profile on a statically-shaped session reports a single fused node, so
the event count is a faithful dispatch counter. Use it as one.

One real caveat: the per-node durations over-count against wall time (ours
summed to ~116 ms against a 40 ms measured call), so read the counts and the
provider split, not the microseconds.

A single warning in the log is the free version of this signal:

```
Some nodes were not assigned to the preferred execution providers ...
e.g. ORT explicitly assigns shape related ops to CPU to improve perf
```

### What does *not* help

**Binding tensors into GPU memory, *as a cure for dispatch overhead*.** It is
the obvious move and it was wrong for us: it saves host↔device copies, and
copies were not where the time went. The fix for dispatch overhead is fusion
(§6). Binding is the right tool for two other jobs — when a render pass
consumes the output on the GPU (`OrtDmlApi::CreateGPUAllocationFromD3DResource`
wraps your own `ID3D12Resource` into an `Ort::Value`), and as a *prerequisite*
for graph capture, which does attack the remaining per-submission cost. So do
not reach for it first, and do not rule it out either.

**Confirmed end to end, later.** The sampler was then built both ways -- host
tensors against every denoiser tensor in a D3D12 buffer with `Ort::IoBinding`
plus the guidance combine and Euler step as a compute shader -- and measured
over a whole 60-latent generation:

| | total | denoising | VAE decode |
|---|---|---|---|
| host round trip | 506.9 ms | 318.9 ms | 187.3 ms |
| latents on the GPU | 508.7 ms | 53.8 ms | 454.7 ms |

The accounting moves and the total does not. The denoising half gets six times
cheaper because the device path never waits -- `Run` records and returns -- and
that time reappears at the one place that does wait, reading a finalised latent
back for the decoder. **The generation is bound by GPU execution, not by the
copies**, exactly as the paragraph above predicted.

Both rows had the VAE decoders on DirectML. Moving them to the CPU provider
(§5.1) later took the host row to 354.1 ms and the device row to 330.5, so the
binding is worth a few percent once the thing it was competing with is gone --
still not what the per-call copy saving suggested.

### Timing a device-resident Run: `SynchronizeOutputs` is not a wait

The trap that nearly reversed the conclusion above. With the output already in
device memory there is nothing for `IoBinding::SynchronizeOutputs()` to copy
back, so it returns as soon as the work is **submitted**:

```cpp
session.Run(Ort::RunOptions{nullptr}, binding);
binding.SynchronizeOutputs();       // records nothing; does NOT wait
const double ms = Since(began);     // measures submission, not execution
```

Measured that way, device-resident looked 5x faster than host memory (2.56
against 13.12 ms). The host path waits because reading the prediction back
forces it to, so the comparison was a wait against no wait. Signal a fence on
the queue the EP submits to and wait for it:

```cpp
queue->Signal(fence.Get(), ++value);
while (fence->GetCompletedValue() < value) {}
```

The same comparison then came out 3.0 against 2.0 ms -- a real but much
smaller difference -- and the whole-generation numbers above showed even that
not reaching the total. If a device-resident configuration looks several times
faster, check what you are timing before believing it.

### Where the time actually was

Worth recording because it was not where anyone looked: of a 507 ms generation,
the VAE decoder was about 190 ms across 60 calls while the denoiser was about
320 ms across 128.

**The fix was not a better binding. It was the other provider, for that model
group only** — see §5.1. Pinning the decoder's `batch` looked like the lead and
was not, and the three measurements that disposed of it are worth more than the
conclusion, because each one is a mistake that is easy to repeat.

**Measure the call that runs many times, not the one that is convenient.** The
`1.48 -> 0.47 ms per call` from pinning `batch` (§2) was measured on
`vae_decoder_first`, which runs **once** per generation. The 59 other calls are
`vae_decoder_step`, and pinning *its* `batch` is worth nothing measurable: 5.42
against 5.50 ms a call, which is run-to-run noise. A 3× win on 1/60th of the
work is a 1 ms win.

**Read the error as a statement about the EP, not about your code.** Pinning
`batch` on `vae_decoder_first` makes the DML EP return a **null for one of its
21 outputs** — `cache_out_9` — so the next session's `Run` reports:

```
Unexpected input data type. Actual: (((null))) , expected: ((tensor(float)))
```

That was long read as "a statically-shaped session cannot hand its outputs to
another". It is narrower and worse: a tensor is lost inside the fused graph.
Check your outputs with `IsTensor()` and the message stops being mysterious.

### 5.0 The null output, root-caused: the EP deletes an initializer that is a graph output

Chased to the line under `cdb` against a Debug `--use_dml` build of 1.31.0,
because it reproduces there too. Every link below was checked, not inferred.

1. **The model.** `cache_out_9` is the only one of the twenty cache outputs
   produced by a `ConstantOfShape`; the other nineteen are `Slice`s.
   `cache_out_14` is an `Identity` of it.
2. **Constant folding converts it to an initializer.** Pinning `batch` makes
   the `ConstantOfShape`'s shape input (`val_48`) statically computable, so
   ORT's `ConstantFolding` evaluates the node and replaces its output with a
   graph initializer *of the same name* — `constant_folding.cc`,
   `graph.AddInitializedOrtValue`. Nothing there checks that the folded output
   is also a **graph output**. That is legal on its own: an initializer that is
   a graph output is valid ONNX, and the CPU EP serves it correctly.
3. **The DirectML EP then deletes it.** Graph fusion collects partition inputs
   that are initializers into `isInitializerTransferable`, marked removable
   (`DmlGraphFusionTransformer.cpp`), and
   `DmlGraphFusionHelper::ProcessInputData` hits the branch commented *"If the
   input isn't actually used by the graph, nothing ever needs to be bound"* —
   `if (!inputsUsed[i]) { graph.RemoveInitializedTensor(...) }`. It is unused
   because the one consumer, the `Identity` behind `cache_out_14`, was folded
   away too. The stack is unambiguous:

   ```
   onnxruntime!onnxruntime::Graph::RemoveInitializedTensor
   onnxruntime!Dml::DmlGraphFusionHelper::ProcessInputData+0xe2b
   onnxruntime!Dml::DmlGraphFusionHelper::FusePartitionAndRegisterKernel+0x6ea
   onnxruntime!Dml::DmlGraphFusionTransformer::ApplyImplHelper+0x1ed7
   onnxruntime!onnxruntime::InferenceSession::TransformGraph+0x2f72
   ```
4. **Nothing guards it.** `Graph::RemoveInitializedTensor` erases the entry
   from `name_to_initial_tensor_` and from the proto with no check that the
   name is in `graph.GetOutputs()`.
5. So the session output has neither a producer node nor an initializer, and
   the fetched `OrtValue` is empty. `IsTensor()` is false, and `Get<Tensor>()`
   throws the `(((null)))` message above.

**The defect is step 3** — the EP may not delete an initializer that is a graph
output — and it is latent because it needs all of: folding that produces an
initializer, that initializer also being a graph output, and the fused
partition not consuming it.

Two things fall out that are worth having:

* **The workaround is one config entry**, and it does unblock the pin:
  `optimization.disable_specified_optimizers` = `ConstantFolding` (the key is
  `kOrtSessionOptionsDisableSpecifiedOptimizers`). With it, no `cache_out_*` is
  ever passed to `RemoveInitializedTensor` and all 21 outputs come back. We do
  not use it: it costs the rest of constant folding, and the pinned DirectML
  decode is still 4.45 ms a call against the CPU's 1.36 (§5.1).
* **How to confirm this class of bug quickly.** Break on
  `onnxruntime!onnxruntime::Graph::RemoveInitializedTensor` and print the name;
  if anything your session is supposed to output appears there, this is your
  bug. In a Debug build the STL string has a container-proxy pointer first, so
  the characters are at `+0x08` and the capacity at `+0x20`, not `+0x00` and
  `+0x18`:

  ```
  bm onnxruntime!onnxruntime::Graph::RemoveInitializedTensor ".block {
       .if (poi(@rdx+0x20) > 0xf) { r $t0 = poi(@rdx+0x8) } .else { r $t0 = @rdx+0x8 };
       .printf \"REMOVED %ma\\n\", @$t0; gc }"
  ```

  Use `bm`, not `bp`: at the initial break the symbol may not be resolvable
  yet. Drive `cdb` with `-cf <script>` from a shell that does not rewrite
  quotes — PowerShell mangles the nested `\"` and the breakpoint silently never
  arms.

**`Ort::IoBinding` is not universally available.** The plan was to remove the
hand-off entirely by binding the 20 caches to D3D12 buffers both sessions
share. `IoBinding` on `vae_decoder_step` fails in **every** configuration —
`batch` pinned or symbolic, fusion on or off — while the identical values
through the ordinary `Run(names, values, ...)` API work in all of them:

| configuration | `IoBinding` | ordinary `Run` |
|---|---|---|
| pinned, fusion on | `Missing Input: cache_in_10` | works |
| pinned, fusion off | access violation in `Concat` | works |
| symbolic | `The parameter is incorrect` in `Shape` | works |

The control that settles it: bind the same twenty tensors in **host** memory
and it fails identically. So it is the binding, not the memory, and no amount
of device residency gets round it. `vae_decoder_first` — one input, 21 outputs
— binds fine, so "many bound outputs" is not the trigger; many bound *inputs*
is. Keep an ordinary-`Run` path working before you build on `IoBinding`.

**This one is NOT root-caused.** It reproduces on 1.31.0 unchanged and we have
only narrowed it: the pinned+fusion symptom is the same *shape* of failure as
§5.0 — `OpKernelContext::Input` finds an `OrtValue` that is present but holds
no tensor — so it may well be another lost tensor rather than a binding
mismatch. What is already ruled out: the model (`cache_in_10` is consumed by a
plain `Concat`, exactly like the other nineteen), the memory kind, and fusion
(it fails with fusion off too, differently). The apparatus in §5.0 is what to
point at it next.

### 5.1 Pick the provider per model group, not per pipeline

The denoiser wants DirectML and the two decoders want the CPU, in the same
process, on the same hardware, at the same time:

| | DirectML | CPU |
|---|---|---|
| `vae_decoder_step`, per call, threaded ×59 | 5.4 ms | 1.4 ms |

End to end at 60 latents that is a generation of **512.3 ms → 354.1 ms**, with
the decode total going **188.5 ms → 69.2 ms**. It is the largest single saving
found in this pipeline, it needed no re-export, no binding and no new
synchronisation, and it took one extra `SessionOptions`.

Part of the mechanism is the one §5 names — the decoder is a small graph run
with a batch of one, so it never fills the GPU — but that is **not the main
reason, and believing it will mislead you.** The from-source 1.31 build proved
it: there, the decoder's isolated per-call cost beat the CPU's (0.80 ms against
1.3) and moving it to DirectML still made the generation 165 ms *slower*.

The real reason is that **the CPU is a second processor and the GPU queue is
not.** Every session in this pipeline shares one compute queue, because that is
what orders a decode after the denoising step that produced its latent without
a fence (§8). So a DirectML decode does not merely cost its own time: it queues
behind the denoiser, and the denoiser then queues behind it. That is visible in
the end-to-end pair — moving the decoders to DirectML raised the DECODE from 69
to 185 ms *and the denoising from 281 to 327*, neither of which a per-call
benchmark on an idle queue can show. On the CPU provider the decode runs on
cores that were otherwise idle, concurrently with GPU denoising, and costs the
critical path almost nothing.

Two things follow. **A per-call benchmark of one session predicts nothing about
a pipeline that shares a queue** — measure the whole generation, both halves
reported separately, which is what localises this. And the provider is worth
making a per-model decision with its own measurement: a model may belong on the
CPU because the GPU is busy with something more valuable, not because it is
slow there. Ours has three such decisions: sampler, prompt encoder, decoders.

---

## 6. Yes, there is a compiler: DmlGraphFusion

The EP does not have to dispatch operator by operator. It carries a
`DmlGraphFusionTransformer` that takes each contiguous DirectML-assignable
partition, compiles it with `IDMLDevice1::CompileGraph` into one
`IDMLCompiledOperator`, and dispatches it as a single node named
`DmlFusedNode_<partition>_<size>`. **It is on by default.** The reason you may
never see it working is that **it only engages when shapes are statically
known** — `CompileGraph` needs concrete tensor descriptions, and a symbolic
dimension denies it those.

Measured on one 1334-node transformer, counting `"cat": "Node"` events from an
ORT profile:

| session | dispatches per run | on DML | on CPU | fused nodes |
|---|---|---|---|---|
| as exported, all dims symbolic | **1334** | 1226 | 108 | none |
| `batch` and `time` pinned | **294** | 239 | 55 | 10 partitions |
| all four dims pinned | **1** | 1 | 0 | `DmlFusedNode_0_2` |

The whole graph collapsing to a *single* dispatch, with **zero** CPU nodes, is
the thing to aim at: fixing the shapes also constant-folds away all the shape
arithmetic that was forcing CPU partitions.

Prove fusion is what you are seeing by switching it off, since the EP's own
knobs are reachable only as session config entries:

```cpp
options.AddConfigEntry("ep.dml.disable_graph_fusion", "1");
```

| | fusion on | fusion off |
|---|---|---|
| dynamic shapes | 9.5 ms | 13.8 ms |
| static shapes | **2.0 ms** | 10.2 ms |

Static-with-fusion is 5× static-without. Dynamic shapes barely respond, because
there was nothing to fuse in the first place.

A nuance worth knowing before optimising further: **294 dispatches and 1
dispatch cost almost the same wall time** (2.02 vs 2.01 ms). The cliff is
between "no fusion" and "some fusion"; squeezing ten partitions down to one
bought nothing measurable. So once fusion engages, stop and re-measure rather
than chasing the last symbolic dimension.

### The DML EP's session config keys

None of these are in the public headers. They are discoverable as strings in
`onnxruntime.dll` (`grep -aoE "ep\.dml[A-Za-z0-9_.]*"`), which is also how to
check whether the build you shipped supports one:

| key | what it does |
|---|---|
| `ep.dml.disable_graph_fusion` | turns the above off; for A/B proof |
| `ep.dml.enable_graph_capture` | record the GPU sequence once, replay it |
| `ep.dml.enable_graph_serialization` | cache compiled graphs across runs |
| `ep.dml.disable_memory_arena` | arena allocator off |
| `ep.dml.enable_cpu_sync_spinning` | spin rather than block on sync |

### Graph capture, for the cost fusion leaves behind

`ep.dml.enable_graph_capture` is the stronger mechanism: it records the whole
submission once and replays it, removing the residual per-`Run` CPU work that
fusion does not. It is also stricter. Enabling it with ordinary host-memory
`Ort::Value`s was accepted at session creation and then failed at `Run`:

```
OrtValue::Get IsTensor() was false. Trying to get a Tensor, but got: (null)
```

Capture needs static shapes *and* every input and output at a fixed device
address, which means `Ort::IoBinding` with device-resident tensors rather than
host memory. That is why binding is not simply "an optimisation that did not
help" — it is the entry fee for this feature.

Reach for capture only when fusion has already landed and the remaining time
is submission overhead rather than GPU work: for us, fusion took a call to
~2 ms of genuine compute, and capture had little left to win.

## 7. Input shapes: the lever that actually works

Static shapes are the precondition for §6's fusion, and therefore the lever
that actually changes the dispatch count. The payoff for us was **4×**, taking
DirectML from slower than the CPU to faster.

### Test it without re-exporting

This is the most valuable trick in these notes. ORT can pin a named symbolic
dimension at session creation, producing the same state a static export would:

```cpp
options.AddFreeDimensionOverrideByName("batch", 2);
options.AddFreeDimensionOverrideByName("time", 65);
```

Measure with those in place before asking anyone to change an export. Our
results, denoiser at `time=16`:

| pinned | per call |
|---|---|
| nothing | 6.6–7.1 ms |
| `batch` alone | 6.9 ms — no change |
| `time` alone | 6.9 ms — no change |
| `batch` **and** `time` | **2.0 ms** |
| all four symbolic dims | 1.7 ms |

**Pinning one dimension bought nothing.** Fusion cannot compile a partition
until every dimension feeding it is known, so partial fixes look like failures
and will mislead you into abandoning the approach. Fix the batch and sequence
axes together, then measure again — that is the point where fusion starts
engaging at all (1334 dispatches down to 294).

For a convolutional decoder whose other dims were already static, pinning
`batch` alone was the whole story: 1.48 → 0.47 ms.

### Which paddings are numerically exact

To fix a dimension you must pad to it, and padding is only safe when the model
masks what you added.

* **Exact when an existing input already masks it.** Padding a text-context
  axis up to the model's maximum changed our output by **0.000004**, because a
  `context_lens` input already masked the padding out of cross-attention. Same
  argument covers a token axis guarded by `attention_mask`. Free win, no
  interface change.
* **Not exact otherwise.** Padding the sequence axis of a **non-causal**
  model moved our output by **1.99**. Bidirectional self-attention lets the
  padding attend into the real data. Fixing such an axis requires a new
  valid-length input and a self-attention mask, not padding alone.

So the question to ask of every axis is not "can I pad this" but **"what masks
the padding?"** If the answer is nothing, fixing that axis is a model change,
not an export flag.

### You may not need a re-export at all

`AddFreeDimensionOverrideByName` is not only a measurement trick — it is a
shipping strategy. Pinning the dimensions at session creation gives you fusion
today, provided the application can actually *feed* those fixed shapes. Whether
it can is decided by §"which paddings are exact" above: pad the masked axes
freely, and for an unmasked one decide whether the altered semantics are
acceptable.

Doing this to one real pipeline took an 8-step generation from 966 ms to 60 ms
— past the CPU provider — and the cold cost from 1289 ms to 171 ms, because
only one shape remained to compile. In the host application a step fell from
91.5 ms to 14.66 ms.

Two consequences to design for:

* **A pinned length is compiled into the session.** Changing it means building
  a new session, so the entry point should refuse a mismatched length with a
  clear message rather than letting `Run` fail on a shape it cannot take.
* **Pin per model, not per process.** A free dimension override is matched by
  NAME, so one shared `SessionOptions` will pin `batch` wrongly for a model
  that runs a different batch. In our pipeline the denoiser runs a
  classifier-free-guidance pair (batch 2) while the decoders run one item at a
  time (batch 1); they need separate option sets.

And one genuine limitation found by trying it: pinning the shapes of two
sessions that hand tensors to each other broke the hand-off on ORT 1.24.4. A
decoder pair threading 20 cache tensors from one session into the next failed
with `Unexpected input data type. Actual: (((null))) , expected:
((tensor(float)))` once both were static, while each ran correctly alone. If
you chain sessions, pin the expensive one and leave the chain dynamic.

### Choosing the fixed size

Larger fixed sizes cost more per call (1.9 ms at 16, 2.1 at 32, 2–4 at 65 for
us), so a single generous size is simple but not free; a few bucketed exports
are cheaper if early work is small. And fixing an axis **caps** what the
application can ask for — reconcile that with your UI limits.

---

## 8. Sharing a device and queue with a renderer

`SessionOptionsAppendExecutionProvider_DML1` takes *your* `ID3D12Device`,
`ID3D12CommandQueue` and `IDMLDevice`, which is the main attraction: inference
is ordered against your rendering by queue order alone, with no fence.

**Create one `IDMLDevice` per adapter** and pass it to every consumer, rather
than letting each component make its own.

**Expect contention.** The same inference step measured **44 ms standalone and
91 ms inside the running renderer**, because ORT's work queues behind the
frame. At one step per frame that is a visible hitch. Budget for it, move the
work to a worker thread, or step less often.

**Ordering is not completion.** Queue order guarantees your pass runs after
ORT's; it does not mean ORT's work has *finished*. Anything that requires
completion — above all resetting a command allocator — needs a fence wait.

**`ID3D12CommandAllocator::Reset()` returns `S_OK` while the allocator is still
in use.** Testing its HRESULT and draining only on failure therefore never
drains. The only witness is the debug layer:

```
D3D12 ERROR: ID3D12CommandAllocator::Reset: A command allocator ... is being reset
before previous executions associated with the allocator have completed.
[ EXECUTION ERROR #552: COMMAND_ALLOCATOR_SYNC]
```

Drain unconditionally before reusing an allocator.

**`SetBreakOnSeverity(ERROR, true)` makes debug-layer errors fatal only when no
debugger is attached.** This produces the worst debugging signature there is: a
program that passes under `cdb` and dies without it, having lost its buffered
`stdout`. If something works under a debugger and not outside one, suspect the
debug layer and read its message under the debugger.

---

## 9. Validating a provider change

**Run the same inputs through both providers and compare element-wise.** This
is the only check that separates "the GPU path runs" from "the GPU path is
correct" — a provider that mis-executes one operator still produces plausible
output. Ours agreed to **0.000004** on the largest coordinate, which is what
made the switch trustworthy.

Build the comparison into the test suite with the provider as a parameter, and
pin every other variable (seed, prompt, and which device each model runs on) so
the two runs differ in exactly one thing.

Pick a tolerance with a reason. Different kernels and accumulation orders mean
bit-equality is the wrong bar; and if your pipeline *integrates* anything over
time, small per-step differences compound, so scale the tolerance to the
output's own magnitude.

Keep the per-provider device choice in your options rather than hard-coding
one, and report which provider a session actually settled on. Where a model is
large, runs rarely, or is faster on the CPU, let it differ from the rest — our
1.1 GB prompt encoder stayed on the CPU, where it was 5× faster and kept its
weights out of video memory, while the sampler ran on the GPU.

If a provider can be changed at runtime, make the load path **idempotent**:
drop every session a provider owns — including any created lazily — or a stale
one keeps quietly serving the old provider.

---

## 10. Measurement hygiene

**The EP compiles per input shape, and a cold pass pays for it.** A cold run
cost us 1252 ms against 955 ms warmed, for identical work. Report cold and warm
separately, and never assert a cold timing in a test.

**Take a median of several runs**, and give each shape its own session when
comparing shapes — a session that has seen many shapes behaves differently from
one that has seen a single shape.

**Windows exit codes are not small integers.** `bash` reports them mod 256, so
a real `2170` (`0x87A`) appears as the meaningless `122`. Read `$LASTEXITCODE`
in PowerShell when a code looks like nonsense.

**A GUI subsystem binary has no stdout.** `printf` diagnostics vanish. Write to
a file when verifying something inside a windowed application.

---

## Checklist

1. Confirm the staged `onnxruntime.dll` offers `DmlExecutionProvider`, and that
   DirectML and CUDA are not both required in one process.
2. Hold `Ort::Env` and `Ort::TypeInfo` in named locals.
3. Get it running on the CPU provider first; keep that as the control and the
   correctness reference.
4. On a fault, localise the node, then look for shape arithmetic on runtime
   values.
5. Sweep workload size. Flat timing means dispatch overhead.
6. Count dispatches from an ORT profile, and remember the EP's fusion compiler
   is already on but needs static shapes. Prove it with
   `ep.dml.disable_graph_fusion`.
7. Use free dimension overrides to price static shapes before requesting a
   re-export, and fix the batch and sequence axes together.
8. For every axis you want to fix, name what masks the padding.
9. Compare providers element-wise on identical inputs before trusting the
   switch.
10. Drain the GPU before resetting an allocator; expect queue contention with
   rendering.
11. Separate cold from warm timings, and keep the provider selectable per
   model *group* — measure each one, because the answer differs within one
   pipeline.
12. Time the call that runs many times. A 3x win on the once-per-generation
   model is worth a millisecond.
13. Before building on `Ort::IoBinding`, check the same tensors go through the
   ordinary `Run` API, and keep that path working.
