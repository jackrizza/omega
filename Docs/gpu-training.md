# GPU training on Intel Arc A770

Raw benchmark JSON paths below refer to local captures excluded from Git.

Assistant-path update (2026-09-30): [assistant execution](assistant-execution.md)
documents chat manifests, assistant-target resume schemas 5/6 and the explicit
`train-stage` command. Those new paths require separate A770 qualification; the
hardware results below describe the earlier plain-text training implementation.

Omega has an optional Burn 0.18 Vulkan/SPIR-V backend for f32 training,
evaluation and generation. CPU remains the default. The shared trainer retains
sampling, bounded padded batches, clipping/warmup, validation, token caches,
transactional numerical checks and periodic/cooperative-stop checkpoints.

The qualified hardware is Linux x86_64 with an Intel Core i7-13700K and a
16 GB Arc A770, Mesa 26.1.8-arch1.1. Other devices/platforms are unqualified.
Vulkan adapter discovery alone does not establish training compatibility.
Burn's WGSL path failed vector boolean shader compilation in the mask probe;
the selected feature uses SPIR-V instead. Burn was not upgraded.

## Build and select the GPU

Run from `Code/Rust`:

```sh
cargo build -p omega-training --bin omega-training --release --features gpu --locked --jobs 2
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- devices
```

Use the uniquely named `omega-training` executable for new workflows. The
legacy `--bin main` remains available, but other packages share that filename.
Do not build different feature sets concurrently with commands using the same
output executable. GPU support is opt-in: builds without `--features gpu`
return an actionable error for Vulkan requests, without initializing a GPU.

`devices` lists discrete Vulkan adapters and per-buffer limits as JSON. Indices
are zero-based among discrete adapters, excluding integrated/software devices.
Use `--backend vulkan --device 0` with train/resume/generate/evaluate. Selection
is verified against the initialized adapter; there is no automatic CPU fallback.
`--device` without Vulkan is rejected. Explicit CPU thread flags cannot be
combined with Vulkan and do not control GPU parallelism. Inherited environment
may still affect host-side work such as tokenization.

## Start a separate GPU training run

This command uses the existing tokenizer and corpus, trains a fresh Omega model,
and saves after a bounded 20-update segment under a new run name:

```sh
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- train \
  --backend vulkan --device 0 \
  --name macro-invest-gpu --dataset investment/macro \
  --tokenizer qwen-3-1.7-base.json --epochs 10 \
  --max-updates 20 --save-every-updates 10
```

The tokenizer JSON supplies vocabulary/preprocessing only. This does not load
pretrained Qwen weights. Vocabulary 151665 makes the output projection/loss
substantial even with default width 32, context 64, two layers and FF width 128.
At batch 1/context 64 the logits alone occupy about 37 MiB in f32.

To continue the saved GPU run toward ten total epochs:

```sh
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- resume \
  --backend vulkan --device 0 --latest-run macro-invest-gpu \
  --name macro-invest-gpu --epochs 10 --save-every-updates 100
```

`resume --epochs` is the total target, not an additional epoch count. Add
`--max-updates` for another bounded segment. `train` always starts fresh.
Use separate run names for concurrent jobs. Existing user training jobs and
checkpoints are not stopped, converted or modified by installing GPU support.

```sh
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- generate \
  --backend vulkan --device 0 --latest-run macro-invest-gpu --prompt "Economic policy" --max-new-tokens 8
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- evaluate \
  --backend vulkan --device 0 --latest-run macro-invest-gpu --dataset investment/macro
```

Evaluation of training documents is not held-out evaluation. Use the original
split recipe or separate held-out documents for validation claims. Generation
still recomputes the full prefix; KV caching and streaming are separate tasks.

The Bash wrapper also supports GPU selection while retaining its four arguments:

```sh
bash Scripts/training_run.sh macro-invest-gpu investment/macro qwen-3-1.7-base.json "Economic policy" --backend vulkan --device 0
```

Run that example from the repository root. The wrapper uses release builds and
the uniquely named binary, trains ten epochs and then generates; it is not the
bounded segment example above.

## Checkpoints and compatibility

Inference records remain portable between supported CPU and GPU backends.
Loading the same weights on a different backend is distinct from exact resume;
small floating-point differences can affect generated tokens near ties.

CPU training continues to write resume schema 3. GPU training writes schema 4
with required backend/precision, adapter index/name/vendor/device, driver,
host kernel, compiler, target, build profile, optimization/debug settings and
Rust flags. GPU resume requires that identity and saved training settings to
match. Debug-to-release, CPU-to-GPU,
Windows-to-Linux and changed-device/driver resumes are rejected. Adapter index
is conservative provenance, not a stable physical hardware UUID; replacing an
identical card at the same index cannot be detected by this profile. Unsupported
external backend feature unification is outside the qualified build contract.

The device-reduction trainer uses execution kernel
`wgpu-vulkan-f32-device-checks-v2`. Earlier `wgpu-vulkan-f32-checked-v1`
checkpoints remain readable for catalog/inference, but exact training resume
requires the original executable/runtime. Do not relabel a v1 checkpoint as v2:
GPU reduction order changes reported norms and can change clipped updates.
CPU arithmetic and execution identity are unchanged. No checkpoint schema or
dependency upgrade accompanies this optimization.

Existing schema-1/2/3 CPU checkpoints retain their load/migration rules. A
narrow migration permits the exact pre-GPU lockfile digest only for the reviewed
successor lockfile; CPU dependency versions/checksums stayed unchanged. Other
runtime, OS and CPU-profile checks still apply. In particular, the historical
Windows `macro_invest-*` snapshots cannot resume on this Linux GPU. Never edit
metadata to bypass a compatibility check. There is no weights-only training
restart/import command in this scope.

Model, Adam state, sampler/cursor and partial epoch metrics are saved before
`COMPLETE`; hashes and numbered reservations are retained. Catalog/latest stays
metadata-only and works in CPU-only builds, including discovery of GPU saves.
A failed selected load never falls back to an older checkpoint.

Linux SIGINT/SIGTERM request cooperative stopping. The current update finishes
with its checks, then the committed state is saved and the command returns
nonzero if the epoch target was not reached. Forced termination/device loss
cannot guarantee a final checkpoint. Earlier completed periodic saves survive.

## Memory and performance boundaries

Device preflight checks arithmetic and known individual model/activation buffer
sizes against adapter storage/buffer limits (about 2 GiB per buffer on this A770).
This is not an aggregate VRAM bound: gradients, Adam moments, candidate copies,
backend temporaries and allocations overlap. Batch/context/vocabulary increases
can still exhaust memory. The CLI never truncates examples to fit GPU memory.

GPU checks reduce gradients and candidate model/optimizer state on the device.
Only three f32 scalars per gradient tensor and one candidate-validity flag are
read back. Shapes, parameter IDs and Adam counters are still checked, and invalid
candidates never commit. IEEE bit checks reject NaN/infinity and negative second
moments, including subnormals; negative zero is allowed. Scaled sum-of-squares
reductions avoid overflow and reconstruct subnormal gradients without relying on
GPU denormal arithmetic. A tiny clipping multiplier is applied in two safe steps.
CPU training retains the original elementwise host checks and f64 accumulation.

The GPU norm uses deterministic single-axis FP32 reductions (avoiding Burn's
global floating-point atomic sum), with f64 combination of tensor summaries,
not the old elementwise f64 sum. Qualification uses a relative norm tolerance
of 2e-5; unclipped updates and same-runtime resume are checked bitwise. The loss,
gradient summary and candidate flag still synchronize with the host. All updates
remain ordered with one optimizer. Utilization alone is not a throughput measure.

The `gpu_benchmark` example uses synthetic IDs with representative dimensions,
shared CPU-materialized initial weights and the production checked trainer. It
reports a parameter hash, synchronized warm timings, real targets/sec, host peak
RSS and raw DRM memory samples; DRM samples are not peak GPU memory. Build time,
initialization and warmup are separate from timed updates:

```sh
cargo build -p omega-training --example gpu_benchmark --release --features gpu --locked --jobs 2
timeout 30s cargo run -p omega-training --example gpu_benchmark --release --features gpu --locked -- --backend vulkan --batch-size 1 --max-seconds 25
```

Add `--profile` to report `stage_seconds` for preparation, forward/loss,
backward, gradient validation, optimizer and candidate validation. These are host
wall times: queued GPU work can complete in a later readback stage, so they are
not isolated kernel timings. Whole updates synchronize before and after timing.
Compare `real_targets_per_second` at the same model/context and batch settings;
do not run performance cases beside another training job or compiler workload.

For measurements, invoke the built example directly under a supervisor after
compilation to exclude Cargo overhead. Cargo target location may be overridden
by `.cargo/config.toml`. Run cases sequentially and record competing CPU jobs.

## Validation-readback comparison — 2026-09-30

The T20.G6 report (`benchmarks/t20-g6-gpu-checks.json`, local capture) records ten sequential
release cases on the A770. The user's run was gracefully stopped and its saved
checkpoint hashes verified first. All cases use the same initial parameter hash,
synthetic sequences, vocabulary 151665, context 64, width 32, two layers and FF
width 128. Each case has two warmup and seven timed updates; host/device checks
each have two repeats per batch in alternating order. Rates below aggregate
targets divided by timed seconds across repeats.

| Batch | Original host checks, targets/sec | Device checks, targets/sec | Improvement |
|---|---:|---:|---:|
| 2 | 212.65 | 345.20 | 62% |
| 4 | 363.83 | 538.03 | 48% |

Candidate-validation wall time fell from about 0.19–0.20 seconds to 0.023 seconds
per update. Gradient-validation stage time fell from 0.29 to 0.23 seconds, but
that stage also waits for queued backward work; it is not pure validation cost.
These results measure useful targets/sec, not GPU utilization or model quality.
Full corpus preparation, held-out evaluation and checkpoint I/O are excluded.

An isolated Burn fusion experiment retained the original host checks to measure
fusion alone. It produced 210.18/351.48 targets/sec at batches 2/4 (one repeat
each), showing no improvement in this small workload. Fusion remains disabled;
fusion combined with the new device checks was not qualified or measured. The
experiment required a larger compiler recursion limit and made no changes to the
workspace manifest, lockfile, dependency versions or production backend.

Validation includes numerical extremes/subnormals, invalid gradients and Adam
moments, clipping, unchanged unclipped updates, exact CPU/GPU checkpoint
continuation within each supported runtime, and Linux signal/subprocess tests.
The task board records commands and results; other GPUs remain unqualified.

## Measured representative workload — 2026-09-28

The 13-case report (`benchmarks/t20-gpu-a770.json`, local capture) completed in 74.74 seconds,
without timeouts. Every case used the same initial parameter hash, and the
executable remained unchanged during the sweep. Each case used two warmup
updates and three synchronized timed updates, retaining all production checks.

| Batch | Release CPU targets/sec | A770 targets/sec | GPU/CPU ratio |
|---|---:|---:|---:|
| 1 | 74.8 | 127.6 | 1.71x |
| 2 | 73.1 | 211.6 | 2.90x |
| 4 | 92.2 | 347.1 | 3.77x |

CPU rows use Rayon 8/matmul 4, the highest aggregate throughput among the
batch-1 thread cases. All batch-1 CPU settings were close (73.8–74.8 targets/sec),
so this does not establish a meaningful thread-setting advantage. Defaults remain
unchanged. Host peak RSS for batch 4 was approximately 754 MiB CPU and 394 MiB
GPU; the latter excludes device memory, so it is not a total-memory comparison.

These measurements use synthetic sequences with vocabulary 151665 and the
recorded model dimensions, not the user corpus. The existing CPU debug training
job remained active, making background load a confounder. More representative
samples and an idle host would strengthen small-difference claims. Batch 4
improved token throughput but changes optimizer-step count and training behavior;
it is an explicit option, not an automatic replacement for batch 1. No two-day
to exact-finish-time conversion is claimed. Save/evaluation/corpus I/O costs are
outside these update timings.

The measurements preceded final device-identity/profile and legacy-header
preflight hardening; the checked training mathematics and kernels were unchanged.
The report retains the measured executable digest and runtime profile.

Detailed profiling of projection/loss, validation, synchronization, candidate
copies and allocations remains open in T20.2/T20.4. The sweep measures whole
checked updates; it does not attribute their costs to individual operations or
measure checkpoint I/O latency.

## Validation

Hardware regressions are explicitly ignored in ordinary tests so machines
without a GPU can run CPU checks. Execute them on qualified hardware:

```sh
cargo test --workspace --locked --jobs 1
cargo test --workspace --features omega-training/gpu --locked --jobs 1
cargo test -p omega-training --features gpu --test gpu_resume --locked -- --ignored --test-threads=1
cargo test -p omega-training --features gpu --test gpu_cli --locked -- --ignored --test-threads=1
cargo test -p omega-training --features gpu --test gpu_subprocess --locked -- --ignored --test-threads=1
cargo test -p omega-training --features gpu --lib gpu_artifact_failure_never_publishes_complete_and_preserves_numbering --locked -- --ignored --test-threads=1
cargo run -p omega-training --features gpu --example gpu_probe --locked
cargo clippy --workspace --all-targets --features omega-training/gpu --locked --jobs 1 -- -D warnings
```

The hardware probe compares shared CPU/GPU logits, loss, all parameter gradients
and post-Adam output using absolute/relative tolerance 0.003, and checks causal
and padding invariance. This is a bounded f32 parity test, not cross-backend
bitwise equivalence. See [TASKS.md](../TASKS.md) for actual acceptance evidence,
remaining limitations and measured results.
