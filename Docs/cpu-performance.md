# CPU performance and threading

Raw benchmark JSON paths below refer to local captures excluded from Git.

These are historical Windows/Ryzen measurements. For the later Linux i7-13700K
and Arc A770 implementation, see [GPU training](gpu-training.md).

T20 keeps the Burn 0.18 NdArray f32 CPU backend. Intel Arc A770 support is a
separate, later GPU qualification and implementation phase. The current benchmark
host is Windows x86_64 with an AMD Ryzen AI 9 365 (10 cores, 20 logical processors),
using Rust 1.93.0. Results on tiny synthetic models do not predict larger-model
training speed or language-model quality.

## Reproduce the baseline

From `Code/Rust`, build once, then measure the executable in fresh processes:

```sh
cargo build -p omega-training --example cpu_benchmark --release --locked --jobs 1
python scripts/benchmark_cpu.py --output cpu-baseline.json
```

The output must be a new file. No corpus or checkpoint is read or written.
The supervisor limits each process to ten seconds and the initial sweep to five
minutes, including setup. Compilation is separate. Timeouts, incomplete samples
and unavailable memory measurements are reported rather than counted as success.
Do not change source or rebuild the executable while a sweep runs.

The default staged sweep tests backend defaults, then Rayon counts 1/2/4/8/10/20
with matrix multiplication at one, then matrix-multiply counts 2/4 with Rayon at
one. Every child disables tokenizer parallelism; the synthetic workload performs
no tokenization. Only child-process environment is changed. The deprecated
`RAYON_RS_NUM_CPUS` variable is cleared in these controlled runs too.

Training cases use batches 1/4 and contexts 32/128. Generation cases use one prompt
and eight generated tokens, recomputing the prefix each time. All use vocabulary
64, width 32, four heads, one decoder layer and feed-forward width 64. Setup and
mandatory warmup are excluded from sample times. Training measurements include
the real persistent trainer and its numerical checks. Reports contain samples,
median/range, real targets or generated tokens per second, source/build identity
and correctness fingerprints. Windows peak working set is a whole-process memory
measurement, not tensor allocation or a memory limit.

For a focused comparison:

```sh
python scripts/benchmark_cpu.py --output cpu-focused.json --threads 1:1 --threads 4:1 --case train:4:128 --case forward:4:128
```

Forward-only timing is a diagnostic workload; subtracting it from training time
does not isolate backward or validation cost. The optional
`TrainingSession::step_profiled` API measures stages within the same checked update
implementation. Normal `step()` takes the untimed path. Profiling includes clock
overhead and excludes final assignment/counter bookkeeping; it does not remove
checks or weaken transactional failure behavior.

## Backend thread pools

The current feature graph already enables Rayon and
`matrixmultiply/threading`. Burn dispatches batched matrix multiplication through
Rayon; ndarray matrix multiplication uses the separately initialized
matrixmultiply pool. Tokenizer batch/BPE parallel work shares the resolved Rayon
pool. These are existing capabilities, not new threads added by T20.

Matrixmultiply 0.3.11 reads `MATMUL_NUM_THREADS` once, defaults to physical core
count and clamps its limit to 1–4. Its splitting policy uses one, two or four
threads and may choose fewer for small matrices. Rayon uses `RAYON_NUM_THREADS`,
then its deprecated alias, then automatic logical concurrency. A pool limit is
not a cap on total process threads or a claim that all those threads are active.
More threads can add scheduling overhead or compete for resources; choose settings
from measurements for the actual workload.

The baseline does not enable Burn's optional SIMD feature. Kernel experiments must
record their feature graph separately, retain numerical checks and compare the
same workload and thread settings. No Burn upgrade or external BLAS installation
is implied by this work.

## Measured baseline on this host

The initial 54-case sweep (`benchmarks/t20-cpu-baseline.json`, local capture),
stage profile (`benchmarks/t20-cpu-profile.json`, local capture) and
18-case repeated sweep (`benchmarks/t20-cpu-repeat.json`, local capture) completed without
timeouts or source/executable changes. The repeated sweep used 15 samples of four
iterations each, following one warmup. Median throughput was:

| Workload | Backend defaults | Rayon 1 / matmul 1 | Ratio |
|---|---:|---:|---:|
| Train, batch 1, context 32 | 4,620 targets/s | 5,618 targets/s | 1.22× |
| Train, batch 1, context 128 | 10,368 targets/s | 10,975 targets/s | 1.06× |
| Train, batch 4, context 32 | 10,435 targets/s | 13,458 targets/s | 1.29× |
| Train, batch 4, context 128 | 14,981 targets/s | 15,739 targets/s | 1.05× |
| Generate, context 32 | 1,035 tokens/s | 1,794 tokens/s | 1.73× |
| Generate, context 128 | 371 tokens/s | 571 tokens/s | 1.54× |

These are process-local comparisons on a development laptop, with ordinary
background load and no power/affinity changes. Small differences can be noise;
the raw reports retain ranges and individual samples. Default settings remain
unchanged because larger workloads and other machines have not been measured.
Initial/final training logits hashes and final losses matched across the nine
initial thread configurations for each case. This bounded result does not promise
cross-platform or arbitrary thread-count bitwise equivalence.

The stage profile identifies candidate model/Adam validation (~2 ms) and gradient
validation (~0.7 ms) as material parts of a small update. Forward/backward dominate
the larger batch/context case. Source preparation is minor in this in-memory
benchmark; it provides no evidence for parallel corpus prefetch or cache-I/O changes.

Baseline build: `cargo build -p omega-training --example cpu_benchmark --release
--locked --jobs 1`, default features, no custom target-CPU flags. Baseline executable
SHA256: `05f9a387297f4c862f2f73c2895b9ddc6f35cf98357bea6d5cbe54b4f62946fd`.
The reports include the exact source and lockfile hashes. Build time is excluded
from performance claims; the first release dependency compilation took much longer
than the bounded measurement sweeps.

## Final retained build

The final retained build (`benchmarks/t20-cpu-final.json`, local capture) repeated the same
18-case comparison after removing the slice experiment. All numerical fingerprints
matched the saved baseline, with no timeout or source/executable changes. The
sweep took 19.34 seconds. Median results for default versus Rayon 1 / matmul 1:

| Workload | Backend defaults | Rayon 1 / matmul 1 | Ratio |
|---|---:|---:|---:|
| Train, batch 1, context 32 | 4,264 targets/s | 5,750 targets/s | 1.35× |
| Train, batch 1, context 128 | 9,573 targets/s | 9,681 targets/s | 1.01× |
| Train, batch 4, context 32 | 9,595 targets/s | 12,420 targets/s | 1.29× |
| Train, batch 4, context 128 | 13,933 targets/s | 13,721 targets/s | 0.98× |
| Generate, context 32 | 1,014 tokens/s | 1,540 tokens/s | 1.52× |
| Generate, context 128 | 355 tokens/s | 501 tokens/s | 1.41× |

The repeated small-context gains support making the controls available, while
the large training case shows why a universal one-thread default is unwarranted.
Peak process working set for batch 4 / context 128 was 16.05 MiB at defaults and
13.61 MiB at 1/1; this is a single small-process measurement. The report retains
all sample ranges and memory observations. Final executable SHA256:
`3fdf471879c1dbb0fb3fd39b9c069a633fcb0e5cdb922bfa9dceea3cb8c48912`.

## Explicit startup controls

The training CLI accepts global `--cpu-threads N` (Rayon, 1–256) and
`--matmul-threads N` (1, 2 or 4), before or after the subcommand. For example:

```sh
cargo run -p omega-training --bin main --release --locked -- --cpu-threads 1 --matmul-threads 1 train --name cpu-test --dataset examples --epochs 2
cargo run -p omega-training --bin main --release --locked -- --cpu-threads 1 --matmul-threads 1 generate --checkpoint cpu-test-1 --prompt hello --max-new-tokens 3
```

Setting both to one selects one worker for each compute pool, not a single OS
thread for the whole process. Defaults remain unchanged: omitted controls inherit
the environment/backend policy. Explicit CPU counts set both Rayon environment
variables in the child; explicit matmul counts override its variable. There are no
new concurrent optimizer updates, corpus-prefetch workers or tokenizer ID changes.
The NN smoke CLI can still use environment variables set before startup; the new
flags belong to the training orchestration CLI.

The CLI starts a fresh process before any tokenizer/tensor work. Unix replaces
the process; Windows waits for a child sharing its console and keeps the parent
alive during console interruption so the child can finish its existing graceful
save. A signal aimed only at the waiting parent, forced termination or loss of the
console is not a cooperative-stop guarantee. Libraries can use
`cpu::CpuThreadSettings::apply_to_command` on a new command; it never mutates the
current environment or reconfigures initialized pools.

Call `cpu::execution_profile()` before initializing backend/tokenizer/custom pools
in library integrations. The trainer captures it before source callbacks and model
initialization; later captures reject changed settings. It records raw environment
values and available logical parallelism, not observed worker counts. Externally
initialized/custom pools cannot be certified by this API. Do not change execution
settings during a session. Automatic matrixmultiply physical-core detection still
requires the same host/default policy for supported continuation.

## Checkpoint compatibility

New training saves use resume schema 3, adding the required `cpu_execution`
profile to schema 2's training state. Resuming requires the same raw pool settings,
recorded logical concurrency and supported kernel/build profile; even an irrelevant
deprecated alias difference is rejected conservatively. Repeat the original flags
or startup environment. Inference weight loading is independent of these controls.
Missing/changed profiles fail before model/optimizer restoration. Existing saves
are never modified by loading or migration.

Schema 1/2 snapshots did not record pool settings. Their migration retains the
existing narrow build/lockfile policy and is permitted only with all three pool
variables absent and the default kernels. This assumes the historical run used its
documented default environment on the same host; it cannot establish unknown old
thread settings. Explicit controls, including `1`, cannot be applied to that
migration. Subsequent saves use schema 3. Catalog inspection recognizes all three
schemas but remains metadata-only and does not certify current-runtime compatibility.

## Hot-path and kernel experiments

The checked f32 slice experiment replaced boxed host iterators in gradient and
candidate-state validation while retaining every check. The
paired baseline (`benchmarks/t20-cpu-baseline-paired.json`, local capture) and
experimental build (`benchmarks/t20-cpu-optimized.json`, local capture) each completed 18 cases
with 15 samples of four iterations. All correctness fingerprints matched.
Training throughput ratios ranged from 0.864 to 1.033, with no consistent gain;
whole-process peak working sets were approximately 8–17 MiB across these cases.
Unchanged generation also varied, indicating substantial background variability.

The reversed-order experimental stage run (`benchmarks/t20-cpu-optimized-profile.json`, local capture)
and baseline stage run (`benchmarks/t20-cpu-baseline-paired-profile.json`, local capture) confirmed
slightly shorter validation scans but no end-to-end improvement. At Rayon 1 /
matmul 1, median gradient validation fell from 0.746 to 0.693 ms for batch 1 /
context 32, while total update throughput fell from 5,993 to 5,369 targets/s.
Candidate validation changed from 2.089 to 2.023 ms. The experiment was removed;
files named `optimized` are retained evidence of a rejected candidate, not the
shipping implementation. No validation check or rollback behavior was removed.
Future optimization should measure representative model/data sizes, including
cache I/O when relevant, before adding preprocessing concurrency.

An isolated numerical probe (`benchmarks/t20-kernel-probe.json`, local capture) compared these
commands from `Code/Rust`:

```sh
cargo run -p omega-nn --example cpu_kernel_probe --locked --jobs 1
cargo run -p omega-nn --example cpu_kernel_probe --features burn/simd --locked --jobs 1
```

Both debug builds ran successfully. For 64 copies of `3.0`, reciprocal bits were
`3eaaaaab` by default and `3eaaaa80` with SIMD. The seeded tiny model's 128 logits
were bit-identical in this probe; that does not establish parity for other
operations, model shapes, training or hardware. This was a numerical compatibility
experiment, not a speed benchmark. SIMD remains disabled in the supported build;
no manifest, dependency version or lockfile changed. Arbitrary dependency feature
unification, custom target-CPU flags and externally configured pools are outside
the supported exact-resume contract; the recorded kernel label cannot attest those
build choices. Adopting them requires an explicit kernel/build identity and
broader numerical/portability validation.

The Windows fresh-process supervisor was also tested in an isolated console:
Ctrl+Break after the first update allowed the in-flight second update to finish,
published one completed schema-3 checkpoint and exited nonzero. No child processes
remained. Native Unix startup/signal behavior has not been exercised here.


## Source-read probe

The separate `cpu_source_probe` example measures eager example cloning against
real indexed cache reads, including per-read document length/checksum validation
and decoding. It uses 16 temporary synthetic documents of 128 fixture tokens,
context 32, and 64 examples with 2,032 real targets per pass. It verifies ordered
identity, counts and every example before/after the five alternating warmed
passes. Setup, warmup and cleanup are outside read timings; no model or optimizer
runs. Its fixture differs from the tensor benchmark and rates are not training
throughput or cold-storage measurements.

```sh
cargo build -p omega-training --example cpu_source_probe --release --locked --jobs 1
cargo run -p omega-training --example cpu_source_probe --release --locked
```

The executable enforces a cooperative five-second total limit and returns errors
on overruns, parity failures or owned temporary cleanup failures. Use a supervising
process deadline for a hard cap; filesystem operations themselves are not
interruptible at the cooperative boundary. It never modifies a user corpus or
checkpoint.

The measured source probe (`benchmarks/t20-source-probe.json`, local capture), supervised with a
ten-second process limit, completed in 0.166 seconds including setup and cleanup.
All parity checks passed and the owned temporary directory was removed. Median
64-example read time was 0.0098 ms eager (range 0.0053–0.0187 ms) and 9.403 ms cached
(8.719–10.030 ms); peak whole-process working set was 7.75 MiB. Cached reads cost
about 0.147 ms per example here, including repeated whole-document verification.
This isolates the existing I/O/copy tradeoff; it does not establish end-to-end
training gains from prefetching or justify weakening cache integrity checks.
