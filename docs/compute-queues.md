# Compute queues for model work

How a model runs on D3D12 here without stalling graphics work: why
`ModelManager` and `FloodDiffusionPipeline` each own a dedicated compute queue,
and what their fences are and are not for. Back to the
[README](../README.md); the write-up is [AI-Integrated Renderer](https://jack111331.github.io/BlogPage/ai_render_engine/).

## The model work runs on its own compute queue

`ModelManager` creates an `ID3D12CommandQueue` of type
`D3D12_COMMAND_LIST_TYPE_COMPUTE` and both backends submit to it: the
`DirectMLGraph` runner's own command list, and the ONNX Runtime DirectML EP,
which accepts a DIRECT or a COMPUTE queue and so moves with it. The command
lists become `COMPUTE` lists to match, which they can because everything they
record -- DirectML dispatches, `CopyBufferRegion`, and barriers between
`COMMON`, `COPY_DEST`, `COPY_SOURCE` and `UNORDERED_ACCESS` -- is legal on a
compute queue. A render-target or pixel-shader-resource state would not be,
which is the constraint to remember if one of those lists ever has to touch a
texture the frame uses.

**What sharing the graphics queue was buying, and what it cost.** It bought
ordering for free: anything submitted after a model's work was guaranteed to
see its output, with no fence anywhere. It cost exactly what that guarantee is
worth -- model work and frame work could never overlap, because a dispatch sat
between two frames instead of beside one.

So the free ordering is gone and has to be asked for. `OrderAfterModelWork`
signals the manager's fence on the compute queue and makes the caller's queue
wait on it, GPU-side. Nothing in this project reads a model's output on the GPU
yet, which is the only reason the per-frame `EnqueueModel` in `Render` is
correct without it; the day a pass samples a model's buffer, that call is the
whole difference between a correct frame and a race that only appears under
load. `RecordModelDispatch` is deliberately left alone and still records into
the caller's list, because being ordered *with* the frame is the entire point
of folding a model into a render pass.

One fence for the manager rather than one per model: a wait is always "after
everything submitted so far", which is what a consumer of any model's output
needs, and a monotonic counter on one queue gives that for free. The destructor
drains the queue -- required, not tidy, because nothing else is going to wait
for it and releasing a resource a submitted list still references surfaces
later as a device removal with no obvious cause.

### A latent bug the move exposed

`DmlGraphRunner::Enqueue` submitted its command list and never signalled its
fence. `BeginList` then had nothing to wait on, so the next `Enqueue` reset a
command allocator whose previous submission was still running. `Reset` returns
`S_OK` and corrupts the allocator; only the debug layer notices.

It was harmless while `Enqueue` was called about once per session. Calling it
every frame is not harmless, and the first thing the compute queue did was turn
it into 17 errors per run. The negative control -- removing the signal again on
purpose -- is what says the assertion that now covers it is load-bearing:

```
  no debug-layer errors from reusing the allocator MISMATCH
    ID3D12CommandAllocator::Reset: A command allocator ... is being reset
    before previous executions associated with the allocator have completed.
    (x17)
```

The allocators and lists are also named now, because that message identifies
the object and an unnamed one reads "Unnamed ID3D12CommandAllocator Object".
`OrtModelRunner` already drained correctly and already documented this exact
failure (`EXECUTION ERROR #552, COMMAND_ALLOCATOR_SYNC`), which is how the
cause was obvious the moment the errors appeared.

### What this is and is not worth

Measured, and it is **not** a speed-up on anything here. Frame time with the
per-frame dispatch on the compute queue, on the graphics queue (the old
behaviour) and absent entirely:

| | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|
| no dispatch | 0.249 | 0.257 | 0.233 | 0.233-0.257 |
| graphics queue | 0.229 | 0.314 | 0.254 | 0.229-0.314 |
| compute queue | 0.243 | 0.227 | 0.294 | 0.227-0.294 |

All in milliseconds. Every column overlaps every other, each condition's own
run-to-run spread is wider than any difference between them, and the fastest
single run of the nine belongs to the *old* graphics-queue path. These are
noise. The reason is that the registered model is a single 3 x 256 x 256
convolution and the fixture frame is a quad: there is no contention to relieve.
What the numbers do say is that the extra queue and its fence cost nothing
measurable either.

Note that the first attempt at this measurement returned `16.667 ms` for all
six runs, to three decimal places. That is vsync, not a result; presentation
pins the frame and no sub-millisecond difference can survive it. The numbers
above come from presenting with `c_AllowTearing` and a sync interval of 0.

### FloodDiffusion has its own queue too

`FloodDiffusionPipeline` creates and owns a second compute queue rather than
sharing `ModelManager`'s. Its submissions are a 70 ms generation; the
manager's are a per-frame dispatch, and queueing the second behind the first
would be most of what putting either on a compute queue was for. `Load` no
longer takes a command queue at all -- a caller cannot put this work on the
graphics queue, which is the point, because on the renderer's queue those
seventy milliseconds are frames.

**What the fence is, and is not, for.** Every tensor this pipeline binds is in
*host* memory, so `Step()` already returns with its results on the CPU: ONNX
Runtime does the copy and the wait itself, and no fence of ours could make the
floats appear sooner. `Frame()`, `FeatureFrame()` and the retargeting read
ordinary `std::vector`s and need no synchronisation whatever. Adding a fence
"to sync the output" would be theatre.

What the fence does cover is the two things queue order used to and no longer
does:

- **Lifetime.** `Load` is callable again to change provider, and it destroys
  every ONNX Runtime session -- releasing resources whose submissions may still
  be running. On the renderer's queue the frame's own drains hid that. Nothing
  drains a private queue, so `Load` and the destructor do, and the `Impl`
  declares the queue *above* the sessions so reverse-order destruction keeps it
  alive longer than they are.
- **Cross-queue ordering**, through `OrderAfterWork(queue)`, for whatever reads
  this on the GPU next. Nothing does today.

The queue is created once and kept across provider changes: a fence value
already handed out has to stay meaningful, and recreating the queue would reset
the timeline under anyone holding one. On the CPU path there is no queue, and
both calls are deliberately successful no-ops rather than failures every caller
would have to special-case.

**The output is unchanged**, which is the check that matters. The existing
CPU-against-DirectML comparison in `//Engine:flooddiffusion-smoke` runs the
same prompt and seed through both providers and now has DirectML on the compute
queue:

```
largest single-coordinate difference   0.000004
mean absolute difference               0.000000
largest coordinate magnitude           2.524
```

Sixteen further checks cover the queue itself: that it is `COMPUTE` and is not
the `DIRECT` queue the test still creates as a stand-in for the renderer's,
that a generation completes on it, that both synchronisation calls work and
refuse a null queue, that reloading twice over live sessions leaves the
pipeline usable and the queue identity intact, and that the whole thing is
silent on the debug layer -- which the test enables for exactly this, since
releasing a session mid-flight is otherwise invisible.

**It is not faster, and the queue type is not why it is slow.** A/B of the same
build on the same machine, reading the pinned-shape generation time:

| queue type | run 1 | run 2 | run 3 |
|---|---|---|---|
| DIRECT | 73.6 ms | 73.1 ms | 71.3 ms |
| COMPUTE | 73.4 ms | 71.5 ms | 71.3 ms |

Indistinguishable -- the pairs differ by less than the drift across runs as the
machine warms, and the third pair is identical to a tenth of a millisecond. (The 61 ms recorded in an earlier note was measured on a different
machine state; the DirectML path's cost is
per-dispatch overhead, as the table at the top of `FloodDiffusion.hpp` sets
out, and which queue it submits to does not touch that.)

So the case for this change is structural rather than numeric -- it removes a
serialization point that would have had to be removed before any model big
enough to matter could overlap a frame -- and the correctness evidence is the
part that is real: every numeric inference assertion in
`//Engine:modelmanager-smoke` now runs on the compute queue and still agrees
with the arithmetic, including both backends against each other, and ~1500
frames of per-frame enqueuing in a `-c dbg` build with the debug layer's
break-on-error armed produce nothing.

