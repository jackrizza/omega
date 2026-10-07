# Omega implementation status and remaining work

## Post-training implementation — 2026-10-07

Omega now has schema-2 post-training configuration, read-only parent/data
readiness, versioned conversational suites, baseline/candidate loss reports,
bounded SFT segments, persisted recovery, human scoring and explicit promotion.
Training mechanics remain in `omega-training`; Omega owns budgets and sequencing.
New-stage transfer supports resumable schemas 3–8 across builds, while exact
resume retains strict runtime/backend checks. See [the usage guide](post-training.md)
and FT01–FT12 in [TASKS.md](../TASKS.md) for current evidence and remaining gates.
The production dataset/parent/budgets and actual model-quality acceptance are
unresolved; temporary software fixtures do not satisfy them. The active remote
base run was not interrupted or replaced.

The [Omega application](omega.md) adds a persistent Linux terminal workspace, detached workers, whole-workflow projects and experimental CUDA. See OA01 in the task board for current validation. CUDA uses resume schemas 7/8; CPU/Vulkan formats are preserved.

## Terminal workspace redesign — 2026-10-07

UI01 replaces the monolithic job text dump with a responsive pane layout,
navigation sidebar, formatted training metrics and a recent-loss chart. Separate
views expose log history, checkpoint paths and worker results. Forms show current
values, and the TOML editor scrolls horizontally without wrapping source lines.
Dashboard data is read in bounded batches during polling rather than rendering.
Worker controls, job/checkpoint schemas and exact-resume mechanics are unchanged.
See [the application guide](omega.md) for controls, a sample-data preview and
history/ETA limits, and UI01 in [TASKS.md](../TASKS.md) for validation evidence.

## Dataset training-time estimates — 2026-10-05

`omega-benchmark` adds a library and `training-time` CLI for CPU and optional
Vulkan. It measures disposable checked updates on real batches across a selected
dataset, weights those times by coverage and projects a fixed-order epoch count.
It supports ordinary text/JSONL, published base/chat partitions and existing base
caches using shared training APIs. Setup and warmup are reported separately;
checkpoint/evaluation overhead, quality targets and shuffled/weighted schedules
are outside the estimate. See the [benchmark README](../Code/Rust/omega-benchmark/README.md)
and OB01 in [TASKS.md](../TASKS.md) for validation and limitations.

## GPU validation reductions — 2026-09-30

T20.G6 moves gradient norm and candidate numerical validation onto the qualified
Vulkan backend, with compact host summaries and unchanged transactional commit
guards. CPU arithmetic is unchanged. The new GPU execution identity preserves
legacy inference/catalog access while rejecting cross-version exact resume.
The GPU benchmark accepts `--profile` for host stage timing alongside synchronized
targets/sec; see [GPU training](gpu-training.md) and T20.G6 for qualification and
measurements. This does not establish model quality or change the ZH14 scope.

## Prepared release ingestion — 2026-09-30

`omega-training` recognizes completed `omega-datasets` base/chat partitions,
infers their format, verifies the manifest and selected payload, and rejects
published held-out fitting or automatic re-splitting. Ordinary-folder formats,
tokenizer/checkpoint identity and existing eager/cache limits remain explicit.
OD02 in [TASKS.md](../TASKS.md) records integration validation and compatibility;
this does not accept a corpus or certify a trained model.

## Container tooling — 2026-09-30

CT02 configures the Python runner's default pipeline for the existing omega-alpha
base release, explicit container roots, auto format and a new chat-capable Omega
tokenizer before a two-update CPU pilot. `all --skip-dataset-build` reuses prepared
data. Preparation still reads the full train partition; this is not an automatic
corpus-size cap. See [container commands](../Containers/README.md) and CT02's
validation record. Actual Docker execution remains pending engine availability.

[Containers](../Containers/README.md) adds builder, dataset and CPU training image
targets with one Python runner in `Scripts/run_containers.py`. Binary artifacts
are separated by package, and datasets/checkpoints stay in host bind mounts.
The configurable sequential pipeline supports dataset acquisition, tokenizer
preparation and training; no corpus or model acceptance follows from this tooling.
CT01 in [TASKS.md](../TASKS.md) records actual validation and pending Docker/Linux
runtime checks.

## Assistant implementation update — 2026-09-30

See [assistant execution](assistant-execution.md) for the new conversation software
path and the actual data/runtime gates. ZH03/ZH06/ZH07/ZH12 add protocol tokenization,
assistant-target masks, new-stage initialization and local chat; ZH02/ZH05 tooling
supports data preparation and recorded experiments. Production corpus acceptance,
current A770 qualification, base/SFT training and held-out assistant acceptance
are separate unfinished deliverables. Validation evidence is recorded in TASKS.md.
The historical snapshot below predates these changes.

**Source review: 2026-09-27.** This is a dated implementation snapshot; task
approval and acceptance evidence live in [TASKS.md](../TASKS.md). See the
[training guide](omega-training.md) for commands and precise contracts.

## GPU implementation update — 2026-09-28

Optional Burn 0.18 Vulkan/SPIR-V training now shares the backend-generic trainer
with CPU. Explicit device selection, GPU inference/evaluation, schema-4 GPU
checkpoints and same-runtime continuation are implemented. CPU wrappers/defaults
and schema-3 saves remain. See [GPU training](gpu-training.md) and the latest task
handoff for current validation and performance; the snapshot below predates GPU
integration. GPU kernels/driver behavior on other hardware remains unqualified.

## Current capabilities (2026-09-27 snapshot)

Omega is a small Rust CPU GPT training foundation, not a pretrained/chat model
or a scalable training service. Evaluation metrics and passing tests alone do
not demonstrate useful language-model quality.

| Area | Implemented | Boundary |
|---|---|---|
| [Tokenizer](../Code/Rust/omega-tokenizer/README.md) | Saved full pipelines, encode/decode/lookups, case-preserving byte BPE training, new-path saving and explicit unknown-ID coverage | Freeze trained IDs; no deterministic retraining guarantee, chat templates or guessed special tokens. Fixture remains 13 entries. |
| [Model](../Code/Rust/omega-nn/README.md) | CPU/autodiff causal GPT, checked model/config contracts, greedy-default or seeded sampled generation; explicit right-padding attention and masked target loss APIs | Training orchestrates optional masked minibatches; no GPU selection, KV cache or streaming generation. |
| [Documents](../Code/Rust/omega-training/src/dataset.rs) | Safe text/JSONL selection, document IDs, split-before-chunking and complete local target coverage; bounded indexed [token caches](../Code/Rust/omega-training/src/cache.rs) | Default eager loading; caches cap individual inputs/index metadata and verify sources/settings. No duplicate-content detection or arbitrary-size streaming tokenizer. |
| [Trainer](../Code/Rust/omega-training/src/trainer.rs) | Eager/cached sources, seeded shuffle/weighted sampling, masked minibatches, transactional Adam, global clipping/linear warmup and [disk resume](../Code/Rust/omega-training/src/resume.rs) | CPU f32; fixed order/batch1/disabled controls remain defaults. No accumulation or validation-based early stopping. Backend RNG is not restored; ordering uses saved portable sampler state. |
| [Evaluation](../Code/Rust/omega-training/src/evaluation.rs) | Fixed-model target-weighted cross entropy, perplexity and numerical errors; split/evaluate CLI | Arbitrary supplied data is not certified held-out. Chunk context affects metrics. |
| [Metrics](../Code/Rust/omega-training/src/metrics.rs) | Live progress, quiet mode, configurable schema-1 JSONL writer and propagated logging errors | Events describe committed updates; partial final lines are possible. Terminal CLI events follow a successful final checkpoint save. |
| [Checkpoints](../Code/Rust/omega-training/src/checkpoint.rs) | Numbered marker-last schema-1 inference saves; resume schema 3 with explicit schema-1/2 training migration; periodic/cooperative-stop saves and read-only catalog/latest selection | Catalog inspects metadata, not tensor payloads. Integrity hashes are not signatures or power-loss guarantees. Inference-only checkpoints cannot resume; no retention/deletion executor. |
| [CLI](../Code/Rust/omega-training/src/bin/main.rs) | Explicit roots, train/resume/prepare-cache/generate/evaluate/train-tokenizer/coverage, checkpoints list/latest, explicit latest-run loading, bounded update segments and CPU/matmul thread controls | Omitted thread controls preserve inherited defaults. Paths refer to compile-time checkout; all binaries are named `main`, so select package with `-p`. |

Byte BPE includes every byte, preserves case/whitespace/UTF-8, and has no
normalizer, injected prefix space, unknown token, BOS/EOS, padding or truncation.
A zero unknown rate does not measure segmentation efficiency or model quality.
Saved tokenizer IDs must match embeddings/output heads.

JSONL records remain separate through splitting/chunking. Logical IDs append
`/@record-N` to the relative source path (physical 1-based line). Different
records from one file may be in different partitions; source file lists are
deduplicated. All within-document adjacent targets occur once, including short
remainders, without targets crossing documents.

CLI manifests record dataset selections/format, split recipe, document partition
membership and SHA256 of complete little-endian u32 token IDs (not raw file
bytes), training settings and available build identity. The separate tokenizer
hash covers its complete effective pipeline. Missing provenance is explicit null.
New checkpoints require their manifest and versioned marker; empty-marker legacy
checkpoints remain loadable. This is integrity checking, not authentication of
arbitrary caller-supplied historical model/tokenizer relationships.

## Remaining work

The board retains non-READY work until explicitly promoted; completing a
prerequisite is not automatic authorization.

- T20's CPU work adds bounded benchmarks, optional checked-update stage timings,
  explicit Rayon/matmul startup controls and resume schema 3's CPU profile.
  See [CPU performance](cpu-performance.md) for measurements and compatibility.
  Fewer threads improved the measured tiny workloads; defaults remain unchanged.
  A validation-scan experiment showed no consistent total speedup and was removed;
  optional SIMD changed reciprocal results and remains disabled.
  Intel Arc A770 is the later GPU target, with backend,
  driver/memory and hardware validation still pending. T21.1 token sampling
  and T22.1 discovery are complete; their parent tasks remain
  unfinished. T21.2/T21.3 KV caching/streaming and T22.2–T22.4 stronger storage and
  retention previews still require assignment under the task-board plan.
- Broader optimization controls such as gradient accumulation and decay schedules
  beyond linear warmup remain unimplemented; T19's bounded scope is clipping,
  warmup and numerical validation with exact continuation.

The update loop checks loss and all gradients, applies optional global clipping,
then checks candidate model/Adam state before committing it with counters.
Rejected numerical updates leave the previous state intact. Snapshot saving
validates restoration state; validation checks fixed-model logits when enabled.
Tokenizers 0.23 can omit overflow for configured truncation; disable it for
corpus preparation. Coverage and token-cache creation reject configured truncation outright.

T07 continuation records exact f64 state bits, checks Adam parameter IDs/ranks/
shapes/counters, model/optimizer/state hashes, ordered source identity and saved
runtime settings. Only the current CPU path is supported; global Burn RNG state
is not serialized. Schema 3 retains schema 2's sampling, batching and optimization
controls and adds required CPU execution settings. Exact continuation rejects a
changed profile. Schema-1/2 migration requires unset thread-pool environment and
default kernels under the documented historical assumption. Existing eager callers keep
their APIs through `TrainingSession<S = TrainingSet>`. T14 indexed cache reads
use that same trainer loop and can continue equivalent eager checkpoints. Cache
construction/read memory is bounded by explicit source/document/index limits;
tokenizer/model allocations remain outside those bounds. Session target lengths,
sampler group metadata and one index per epoch draw add memory; padded batches
have a separate input-position cap. Each read rehashes its token document,
trading I/O for integrity; no performance benchmark is claimed. Weighted counter
validation replays completed epoch orders and can become costly for long runs.

## Validation and portability

Run from `Code/Rust`:

```sh
cargo fmt --all --check
cargo build --workspace --locked --jobs 1
cargo test --workspace --locked --jobs 1
cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings
cargo run -p omega-training --bin main --locked -- train --help
cargo run -p omega-training --bin main --locked -- generate --help
```

The T20 final production source passed 199 workspace harness tests (40 NN,
16 tokenizer, 143 training), workspace build, formatting and strict all-target
Clippy. Both benchmark example tests also passed. The existing Windows symlink
privilege limitation below still applies. Train/generate help was checked for the
actual training commands and both CPU flags after forcing a relink to overcome
the pre-existing shared `main.exe` collision. Performance and rejected experiment
evidence are in [CPU performance](cpu-performance.md).

The T21.1/T22.1 pass passed formatting, locked workspace build/tests, strict
workspace Clippy and ten CLI help checks. The harness reported 185 passes (40 NN,
16 tokenizer, 129 training), including one Windows symlink test that returned
early for missing privilege; its symlink assertions were not exercised. A separate
temporary Windows junction smoke verified candidate/root rejection and no fallback.
The task-board handoff records the exact commands and completion evidence.
Focused suites cover tokenizer save/reload/unknown behavior, manifest tampering
and legacy compatibility, evaluation immutability/weighting/overflow, metrics
partial/failure records, JSONL schema/identities/splits and padding/loss gradients.
CLI tests use bounded CPU models and temporary roots, including real subprocess
argument parsing, train/save/reload/evaluate, provenance and overwrite failures.
New regressions cover sampler vectors, bounded collation, numerical rollback,
periodic/final-save deduplication and exact continuation with enabled controls.
Interruption regressions set the atomic stop notification and verify the saved
boundary and subsequent resume. T20 additionally verified real Windows Ctrl+Break
delivery to the fresh-process CPU supervisor/child in an isolated console: the
in-flight update completed, a completed schema-3 checkpoint was saved, and the
process exited nonzero with no remaining descendants. Native Unix execution is
unverified. Legacy migration tests reconstruct schema-1/2 manifests rather than
run a historical binary.
Generation regressions cover fixed sampling vectors, original greedy equivalence,
nonfinite logits and isolated sampler RNG. Catalog regressions cover bounds,
aliases, publication races, corrupt/conflicting metadata, and no fallback from
payload/runtime/source failures through the CLI. Catalog itself inspects metadata
only and does not certify these payloads or exact training-resume compatibility.

Rust 1.93.0 on Windows is the verified toolchain, not a declared MSRV. Native Unix
binaries and Unix-only symlink tests have not run in this pass. Earlier Windows
binaries also exercised reservation/case-alias behavior on WSL case-sensitive
storage; that historical result is not a current rerun. Shared `main.exe` filename
warnings remain; serial workspace build/test avoids competing Windows linker
writes. An overlapping NN build temporarily replaced the training executable
during a help check; after workers finished, explicitly rebuilding the training
binary resolved it. Avoid cross-package builds during CLI execution in a shared
target directory. Burn RNG is process-shared, so deterministic tests clone initialized
state or isolate processes rather than assuming parallel seeds are independent.

## Repository/build policy

[Cargo.lock](../Code/Rust/Cargo.lock) is intended to be tracked for CLI builds.
T06 added SHA256 support through `sha2` 0.10 and its lockfile entries; T07 enables
serde_json's `float_roundtrip` feature to preserve decimal provenance alongside
exact state bits. T08 adds ctrlc 3.5 with termination handling and its target-specific
dependencies; existing versions and Burn/backends are unchanged. A lockfile fixes dependencies, not cross-platform
floating-point results. Only load trusted compatible checkpoints.

The preserved WordLevel fixture is `Code/Rust/test-fixtures/wordlevel.json`.
`datasets/test.json`, `datasets/omega-alpha/` and weights are local-only. The empty
crate-local tokenizer JSON is an unused placeholder retained as existing data.
User corpora and checkpoint files must not be overwritten. This checkout lacks
Git metadata; coordinator snapshots outside the repository allow change review.
No branches, commits, pushes or deployments are part of this implementation pass.
