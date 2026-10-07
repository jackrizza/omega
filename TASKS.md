# Omega implementation tasks

## Post-training roadmap — 2026-10-07

### Execution assignment — 2026-10-07

User requested sub-agent execution of the prerequisites and post-training tasks.
Coordinator owns integration/configuration/workflow/CLI/TUI, shared exports,
manifests and documentation on `dev`. Existing staged changes and the active
production run must be preserved. Production acceptance remains separate from
software fixture evidence; ZH14 and AH01–AH04 remain deferred.

First bounded wave (implemented; integration evidence below):
- corpus: read-only ZH01–ZH05/ZH08/ZH09/FT01 data and artifact audit; no downloads,
  production compute, credential output, mutations or acceptance claims.
- training: FT02/FT07 training-library compatibility and segment outcome work;
  exclusive writes to training `resume.rs`, `operations.rs`, `run_control.rs`
  and new `tests/post_training_stage.rs`. Preserve existing staged edits.
- protocol: FT04/FT05 suite/report engine; exclusive writes to new training
  `src/conversation_evaluation.rs` and `tests/conversation_evaluation.rs`.
- coordinator: agree interfaces, implement Omega configuration and workflow, and
  integrate/test sequentially. No task is complete until acceptance checks pass.

Second bounded wave (2026-10-07, implemented): corpus owns training `readiness.rs`,
`tests/readiness.rs`, Omega `src/main.rs`, `tests/post_training.rs`, and
`Docs/post-training-audit.md`; training additionally owns training `evaluation.rs`
for cooperative metric deadlines; protocol owns Omega `src/tui.rs`,
`src/tui/view.rs`, and `src/tui/tests.rs` for post-training actions. Coordinator
owns `omega/src/post_training/`, configuration, jobs/pipeline wiring and shared
documentation. API agreements: verified new-stage transfer is separate from exact
resume; `execute_segment_to` writes to an explicit output root; evaluation reports
distinguish completed checks from quality acceptance. Windows focused training
tests passed initially (6 conversation, 8 readiness, 1 stage); final evidence is
recorded below and in Docs/post-training-validation.md.

FT01 is active for inventory/profile definition. Its real data, thresholds and
budgets are not assumed accepted; independent software scaffolding/tests may
proceed against explicit temporary fixtures while those inputs are resolved.

Planning baseline preserved on `dev`; implementation evidence follows. See
[the post-training roadmap](Docs/post-training-roadmap.md) for the approved
workflow, compatibility decisions, resource boundaries and validation gates.
FT02–FT10 now have implemented software and temporary-fixture evidence. Their
production use remains gated by FT01; fixture inputs do not close its real
experiment decisions. FT11 qualification and FT12 model acceptance are separate.
The real pilot needs separately approved data/budgets. AH01–AH04 remain deferred.

| ID | Task | Status | Dependencies | Owner |
|---|---|---|---|---|
| FT01 | Define the post-training experiment | blocked: real inputs | — | user/coordinator |
| FT02 | Validate parent checkpoints and SFT data | software done | FT01 | coordinator/team |
| FT03 | Add post-training configuration | software done | FT02 | coordinator/team |
| FT04 | Implement conversational test suites | software done | FT02 | coordinator/team |
| FT05 | Produce baseline and comparison reports | software done | FT04 | coordinator/team |
| FT06 | Add human quality review | software done | FT05 | coordinator/team |
| FT07 | Implement bounded train/evaluate segments | software done | FT03, FT05 | coordinator/team |
| FT08 | Persist the automated workflow | software done | FT07 | coordinator/team |
| FT09 | Integrate TUI and CLI | software done | FT06, FT08 | coordinator/team |
| FT10 | Rank and promote candidates | software done | FT06, FT08 | coordinator/team |
| FT11 | Qualify the implementation | partial: idle A770 pending | FT09, FT10 | coordinator |
| FT12 | Run the real pilot and acceptance process | blocked | FT11; separately approved data/budgets | — |

Reuse the implemented ZH06/ZH07 assistant path and ZH12 chat; do not reimplement
them. FT01/FT04–FT06/FT08/FT10 extend ZH05. FT12 supplies execution evidence for
ZH10, ZH11 and ZH13 without replacing those historical acceptance gates.
Existing worker/checkpoint state and the current training run remain untouched.

Before assigning code work, read the affected crate README, manifest, source and
tests; record one owner and a bounded write set here. Model operations/evaluation
stay in omega-training; workflow state, budgets and UI stay in Omega. Shared
exports/manifests/CLI wiring and documentation remain coordinator-owned unless
explicitly assigned. Each task requires focused tests and documentation; FT11
owns integrated validation. No Python runtime dependency is introduced.

### FT01 — Define the post-training experiment

Status: blocked on actual parent/data/scope/budgets. Owner: user/coordinator. Dependencies: —.

- [ ] Record parent identity, conversational scope, partitions, suite, human rubric, thresholds, and compute/storage limits. Separate software and quality acceptance.
- [ ] Freeze actual parent/checkpoint identity, language/domain, response/context expectations, suite/rubric versions, required thresholds and resource limits; unresolved production inputs must stay explicit.
- [x] Separate software regression acceptance from human-reviewed model quality. Do not approve a corpus, quality threshold or compute budget by substituting a toy fixture.

### FT02 — Validate parent checkpoints and SFT data

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT01.

- [x] Verify completeness, hashes, architecture, tokenizer/protocol, roles, assistant targets, context lengths, provenance and detectable overlap; return an actionable report without changing inputs.
- [x] Check checkpoint payload/header agreement, frozen chat controls, source groups, held-out membership, detectable overlap, conversation ordering and context lengths; report detection limits and preserve inputs.
- [x] Define and test a separate supported-format new-stage weights-transfer path across builds with fresh optimizer/sampler and preserved lineage. Keep exact-resume runtime/backend checks strict; reject unsupported parents without fallback.

### FT03 — Add post-training configuration

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT02.

- [x] Configure an existing parent, assistant-specific settings, segment size, evaluation cadence, budgets and outputs; freeze resolved settings without rerunning base training.
- [x] Introduce project schema 2 with explicit comment-preserving migration and schema-1 reading; do not rewrite old job snapshots. Parent identity, assistant settings and paths must be resolved in the immutable launch snapshot.
- [x] Remove unrelated base-data/base-training requirements for the new workflow. Validate update/time/evaluation/storage limits and stage-specific settings before launch; preserve the existing standalone command behaviour.

### FT04 — Implement conversational test suites

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT02.

- [x] Version single/multi-turn cases with fixed generation settings, expected-answer/JSON checks, termination, role leakage, empty output, repetition and context-limit checks; preserve outputs and failures.
- [x] Expose typed versioned suites and case results in omega-training. Preserve prompts, generated replies, generation settings, completion reasons and individual check failures, including multi-turn history.
- [x] Test expected-answer and JSON checks, reply termination, role leakage, empty/repetitive output, context overflow, timeouts and malformed cases. Do not equate deterministic checks with semantic correctness.

### FT05 — Produce baseline and comparison reports

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT04.

- [x] Measure assistant validation loss, base-language regression and conversation results; compare compatible configurations with full artifact/runtime identity; export JSON and Markdown.
- [x] Expose typed reports with checkpoint/data/tokenizer/protocol/suite/runtime identity, target counts, metrics, outputs, timings and completion state. Reject incompatible comparisons and preserve partial reports.
- [x] Retain a fixed-parent baseline, assistant development results and a separate base-language regression measurement. Export JSON plus readable Markdown; never use sealed-test results to drive automatic tuning.

### FT06 — Add human quality review

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT05.

- [x] Score instruction-following, correctness, relevance and coherence against a versioned rubric; retain notes and responses; missing review remains pending.
- [x] Tie scores/notes and rubric version to exact saved responses/report hashes. Changed output, suite or rubric cannot silently reuse an old review.
- [x] Test score validation, missing/partial review and persistence. Required missing scores leave candidates pending; the model must not grade or approve itself.

### FT07 — Implement bounded train/evaluate segments

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT03, FT05.

- [x] Initialize SFT once, resume the same stage between segments, verify saves before evaluation, and distinguish segment completion, budget exhaustion, user stop and failure.
- [x] Add typed outcomes distinguishing segment completion, total budget exhaustion, user interruption and failure. Start fresh SFT once, then restore the same optimizer/data/settings for every continuation.
- [x] Verify a complete save before evaluation; account for training/evaluation time and output capacity with safe-save headroom. Cooperative limits may finish the current operation/save and must report overruns; do not delete artifacts.

### FT08 — Persist the automated workflow

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT07.

- [x] Sequence baseline, training and evaluation under one compute lock; journal transitions and reports; recover without duplicate work or false passing results.
- [x] Journal phase transitions, consumed budgets and verified checkpoint/report references under the existing detached-worker and host compute-lock model. Execute training and evaluation sequentially with one approved configuration.
- [x] Test crash recovery at every phase, duplicate prevention, missing/changed artifacts, unavailable storage and failed evaluation. User stop cancels remaining stages; reconnect does not restart work; reboot recovery remains explicit.

### FT09 — Integrate TUI and CLI

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT06, FT08.

- [x] Expose parent selection, readiness, review, progress, results, comparisons and human scoring through shared Rust APIs in the single binary.
- [x] Expose one shared implementation through functional TUI screens and headless operations: parent/readiness/configuration review, workflow progress, retained reports, candidate comparisons and human review.
- [x] Test configuration round trips, errors/nonzero exits, navigation/resizing, detach and reconnect. Keep worker completion, quality status and human approval visibly distinct.

### FT10 — Rank and promote candidates

Status: software done; real experiment acceptance remains gated by FT01. Owner: coordinator/team. Dependencies: FT06, FT08.

- [x] Filter failed/pending candidates, rank eligible checkpoints by validation assistant loss then fewer updates, and record human-approved selection with immutable hashes/evidence.
- [x] Apply required gates first; incomplete reports or required missing human scores remain pending. Rank eligible candidates by lower validation assistant loss, then fewer updates; display other metrics separately.
- [x] Require explicit human approval to create an immutable selection manifest with checkpoint/report/review hashes. Refuse overwrite, never mutate checkpoints and never infer best from latest. Final sealed-test acceptance is a separate state.

### FT11 — Qualify the implementation

Status: partial; CPU and tiny WSL CUDA qualified, idle A770 qualification pending. Owner: coordinator. Dependencies: FT09, FT10.

- [ ] Pass offline end-to-end, persistence, compatibility, failure and backend tests, including parent preservation and exact SFT continuation.
- [x] Run tiny offline end-to-end and failure tests: parent preservation, fresh-stage state, compatible transfer, unchanged exact-resume rejection, masked loss, segmented equivalence, budgets, partial reports and human review.
- [x] Run Linux subprocess/PTY detach, kill, hangup, reconnect, stop and phase-recovery tests. Run workspace formatting/tests/strict Clippy and relevant feature builds, recording fresh results.
- [ ] Qualify bounded Vulkan fixtures on an idle A770 and CUDA fixtures on available WSL hardware; report hardware evidence separately. No production-run interruption or infrastructure rental is authorized.

### FT12 — Run the real pilot and acceptance process

Status: blocked. Owner: unassigned. Dependencies: FT11; separately approved data/budgets.

- [ ] Compare a bounded real pilot with its base, review responses, select a candidate, then perform final sealed-test acceptance and local model packaging.
- [ ] Obtain separate approval of real data, frozen experiment thresholds and pilot/storage budgets. Run a bounded SFT pilot against its base; retain failed gates and human scores as well as successes.
- [ ] Continue only the accepted configuration within budget. Explicitly select/promote a candidate, run sealed-test acceptance separately, and record failures honestly.
- [ ] Produce the model card, exact artifact identities, reports and reproducible local-use/recovery instructions. Supply evidence to ZH10/ZH11/ZH13; no public upload or deletion follows from task completion.

### Implementation evidence — 2026-10-07

FT02–FT10 software is implemented in the training readiness/evaluation/transfer
APIs and Omega post_training/configuration/worker/TUI/CLI integration. Three
sub-agents supplied disjoint modules, tests and audits; coordinator integrated
and reviewed them. Fresh evidence: Windows and Linux CPU workspace tests pass;
final readiness 13/13 and workflow 8/8 pass on both. Formatting and strict
workspace/all-target CPU/Vulkan/CUDA Clippy pass; Linux CPU/all-backend builds
pass. The new Linux subprocess/PTY suite passes baseline/training/evaluation
crashes, detach/reconnect, compute exclusion, verified stop, retained-executable
recovery, frozen-input rejection and conservative budget reservations. Tiny WSL
CUDA post-training passes in 92.4 seconds (schema 8, two SFT segments, full reports,
unchanged parent). See [full validation](Docs/post-training-validation.md) for
commands, test limits and initial test-harness corrections.

FT01 remains blocked on actual parent/data/scope/rubric/thresholds and budgets;
FT12 remains blocked on those approvals and FT11. The local audit found empty
conversation partitions, unresolved permitted use, no matching local parent,
and blank pilot budget fields. These are not replaced by fixture data. Earlier
ZH01–ZH05/ZH08–ZH11/ZH13 acceptance boxes retain their historical status. FT11's
remaining hardware gate is the new workflow on an idle A770; the current remote
run was not interrupted. The discovered 33 MiB legacy tokenizer preparation
receipt versus 8 MiB job-reader limit is recorded in the audit; that unrelated
base-preparation defect was not silently fixed or counted as accepted evidence.
No production training, downloads, publication, commits or pushes occurred.

### Deferred helper tasks

The initial helper authority is **propose, then approve**. It uses structured
reports/actions and does not replace deterministic validation or resource limits.
All helper tasks remain unchecked until the post-training implementation is
qualified and the future work is explicitly assigned.

### AH01 — Define the helper action interface

Status: deferred. Owner: unassigned. Dependencies: FT11; explicit future assignment.

- [ ] Define structured project summaries, readiness/report inputs and allowed proposals. Route actions through deterministic validation and the existing approval screen; no arbitrary shell execution.

### AH02 — Prepare the Omega helper corpus

Status: deferred. Owner: unassigned. Dependencies: AH01.

- [ ] Curate Omega documentation, troubleshooting examples and explicitly approved workflow examples. Remove credentials/private content, keep evaluation cases separate and do not automatically use user conversations as training data.

### AH03 — Train and evaluate a separate tiny helper

Status: deferred. Owner: unassigned. Dependencies: AH02; approved data/budgets.

- [ ] Train and evaluate the separate tiny helper for explanation and valid next-step proposals. Measure memory, latency, factual accuracy, action validity and abstention. Concurrent assistance requires separate resource qualification.

### AH04 — Integrate proposal-and-approval assistance

Status: deferred. Owner: unassigned. Dependencies: AH03.

- [ ] Integrate contextual help across configuration, data, training, evaluation and recovery. Show proposed changes, reasons and resource impact before approval. Keep every workflow usable without the helper.

### Initial planning validation and scope (historical)

Documentation-only changes: created the roadmap/task entries and refreshed
zero-to-hero's implemented/missing distinctions against current source. No FT/AH
implementation, training, download, release, commit or push was performed.
Validation passed: 19 local links/anchors, balanced Markdown fences, 16 unique
unchecked tasks with consistent acyclic dependencies, preserved prior staged
task history, and Git whitespace checks. No Cargo or hardware tests were run for
this documentation-only change. These checks do not establish future software
or model acceptance. LoRA, preference optimization, autonomous tuning, external
model judges and ZH14 remain excluded.


## UI01 — Pane-based terminal workspace — 2026-10-07

Status: done; uncommitted. Owner: coordinator. Branch: `dev`.
Scope: Omega TUI rendering, navigation and bounded dashboard presentation; focused
render/control tests and application documentation. Use the openapi-tui demo as
visual inspiration for a persistent sidebar, labelled panes and selection colours.
Keep worker APIs, checkpoint formats and running executables unchanged. Preserve
the staged RP02 changes. Acceptance: readable metrics and durations, recent loss
history, separate log/checkpoint/result views, responsive layout, existing launch
and detach controls, package/workspace checks and Linux persistence regression.

Implemented in `Code/Rust/omega/src/tui.rs` and its new `view.rs`, `dashboard.rs`
and `tests.rs` submodules. Added navigation shortcuts, project/form detail panes,
four job views, formatted metrics/durations, recent-loss chart, bounded log
following/history and checkpoint paths. Rendering performs no dashboard file
reads; polling reads bounded tails and caches the checkpoint index by metadata.
Worker controls and schemas are unchanged. Updated the crate README, application
guide and implementation status. `Docs/images/omega-dashboard.png` is an export
of the actual Ratatui TestBackend with test data. The PTY test now opens the
reconnected worker monitor and switches views; its RP02 fixture change remains.

Validation: 280 Linux CPU workspace tests passed. After final editor/chart-label
adjustments, all 10 Omega tests passed again. All four new UI tests also passed
on Windows. Strict workspace/all-target Clippy with CPU/Vulkan/CUDA features
passed; strict all-feature Omega Clippy passed again after final adjustments.
Formatting, Git whitespace and documentation-link checks passed. Inspected the
140x44 render; tests also exercise every screen/dialog at 80x24, 24x8, 8x4 and 1x1,
unknown metrics, log scrolling, stage changes and partial event records.

The final Linux PTY suite passed detach, abrupt TUI kill, terminal hangup,
reconnect/view switching without duplicate workers, competing-launch rejection,
checkpoint-and-stop, retained-executable resume, worker death/failure, stale
process identity and terminal restoration checks. Only temporary CPU projects
were used. An initial invalid test job ID was corrected before the passing runs.

No production worker or installed executable was replaced; no GPU training,
release, commit or push was performed. Rebuild Omega to use the new interface,
then reconnect through Jobs. Existing jobs retain their original worker binary.


## RP02 — Keep local datasets and weights out of Git — 2026-10-05

Status: done; changes staged, not committed. Owner: coordinator. Removed
`datasets/omega-alpha/model.toml`, `datasets/test.json` and `weights/.gitkeep`
from the index. Ignore rules exclude all of datasets/omega-alpha and both
weights/Weights directory spellings, with no recipe or marker-file exceptions.
Local originals remain byte-identical; existing checkpoint files were untouched.
The user's existing first commit was not amended or rewritten.

The unchanged 13-entry tokenizer fixture now lives at
`Code/Rust/test-fixtures/wordlevel.json`. Rust unit/integration tests, the CPU
source probe and the Linux PTY test use that location. The NN CLI tests explicitly
select the fixture rather than relying on ignored local data. CLI runtime paths
and token IDs are unchanged. Updated fixture/contribution guides and local-only
recipe/checkpoint documentation describe the new Git boundary.

Validation: a clean index export excludes every requested path and passes all
276 Linux CPU workspace tests and formatting. Strict all-target Clippy with both
Vulkan/CUDA features passed on Windows. Index/ignore assertions, original-file
hash comparisons and documentation link checks passed. An initial clean-export
failure exposed the NN CLI test's implicit default path; that dependency was
fixed before the successful full run. No production training or release rebuild
was needed for this fixture/path-only change.


## RP01 — Initial Git push and main release builds — 2026-10-05

Status: done. Owner: coordinator. The initial file set is staged: 140 source,
documentation, configuration, tooling and fixture files (about 1.9 MB). No commit,
push, remote creation or GitHub release publication was performed.

Changes: `.gitignore` now includes required fixtures, tooling and Cargo.lock while
excluding credentials, downloaded data, checkpoints, target directories, local
benchmark captures and the host-specific Cargo target path. `.gitattributes`
normalizes text to LF; shell helpers have executable Git modes. Nested, unborn
Git metadata from omega-nn/omega-tokenizer was preserved outside the checkout;
both crates are ordinary source files in the root index, with no gitlinks.
An exact-path local Git safe.directory entry permits this SMB checkout's owner
mapping; no wildcard trust setting was added.
Local benchmark references are labelled as excluded captures, and fresh-checkout
container documentation no longer assumes prepared datasets are included.

`.github/workflows/release.yml` now builds on pushes to main, v* tags and manual
runs. Main artifacts have a commit-qualified archive name and 14-day retention;
tagged releases publish only after build and clean-runtime checks succeed.
CI disables incremental/debug build output to reduce hosted-runner disk use.
Installation and first-push instructions are in [GitHub builds](Docs/github-builds.md).

Validation: candidate size/credential scans and explicit ignore assertions passed;
no files above 5 MB or populated local credential values entered the candidate
set. Actionlint 1.7.12 passed (external ShellCheck disabled); main/tag packaging
and SHA-256 verification passed. A clean export of the staged source had LF
scripts, the required tokenizer fixture, no local Cargo config or credentials,
and passed workspace formatting, all 276 Linux CPU tests and strict all-target
Clippy. Documentation links resolve within the staged file set. The GitHub-hosted
workflow has not run yet; the first push to main will trigger it.


## OA01 — Persistent Omega application — 2026-10-05

Status: implementation complete; final A770 hardware gate pending. Owner:
coordinator. Implements the approved Linux x64 Ratatui application, versioned
whole-workflow model.toml, detached same-binary workers, retained executable and
configuration snapshots, local IPC/history, explicit checkpoint-and-stop, library
workflows, CPU/Vulkan and experimental CUDA, and release CI.

- OA01.A — done: versioned configuration, legacy recipe import, explicit paths,
  comment-preserving forms, advanced TOML editor, validation and frozen launches.
- OA01.B — done: typed shared training operations/progress/cooperative controls,
  preserved standalone CLIs, CUDA f32/host checks and resume schemas 7/8.
- OA01.C — done: detached workers, per-user compute exclusion, local sockets,
  journals, boot/start identity, retained executable selection for exact resume,
  verified checkpoint history and interruption recovery.
- OA01.D — done: onboarding, directory/recent-project navigation, dataset and
  tokenizer workflows, pipeline review, benchmark/train/assistant/evaluation,
  checkpoint browsing, generation/chat continuation, reconnecting dashboard.
- OA01.E — implementation and local validation done; A770 gate pending:
  Ubuntu 22.04 pinned-CUDA release workflow, local optimized all-backend preview,
  installation/runtime/recovery documentation and software/hardware tests.
  The workflow has not been triggered and no GitHub release was published.

All implementation owned by coordinator. Existing datasets, checkpoints,
executables and active training runs were preserved; no paid compute or production
training was launched. The user authorized the local WSL NVIDIA host for tests.

Implementation paths: `Code/Rust/omega/{Cargo.toml,src/,tests/,README.md}`;
workspace manifests/lockfile; `omega-training/{Cargo.toml,build.rs,src/operations.rs,
src/cuda.rs,src/lib.rs,src/bin/main.rs,src/checkpoint.rs,src/checkpoint_catalog.rs,
src/resume.rs,src/gpu.rs,tests/assistant_cli.rs,tests/checkpoint_catalog.rs,
tests/cuda_training.rs}`; benchmark backend dispatch/features;
`omega-datasets/src/lib.rs` release verification; `.github/workflows/release.yml`;
affected READMEs and `Docs/{omega.md,omega-validation.md,omega-training.md,
implementation-status.md}`. No training mechanics moved into Omega; it owns
project sequencing, process lifecycle and UI state.

Compatibility: CPU/Vulkan schemas and legacy inference readers are preserved.
CUDA adds schemas 7/8 and rejects cross-backend or incompatible execution identity.
Build provenance now identifies compiler/target/source; old exact resumes require
the original executable. Omega retains each worker executable by SHA-256 and uses
it for subsequent exact resumes, including earlier periodic checkpoints. Dataset
recipes and tokenizer IDs remain unchanged. Credentials stay in the environment.

Validation: Windows workspace tests (273 passed), workspace formatting and strict
all-target CPU/Vulkan/CUDA Clippy passed. Linux all-backend workspace tests (276 passed, 14 opt-in hardware tests
ignored) and strict Clippy passed. The optimized release executable was built in the pinned
CUDA 12.5.1 Ubuntu 22.04 image. PTY tests passed detach, forced TUI exit, terminal
hangup, reconnect/no duplicates, concurrent launch rejection, frozen configuration,
checkpoint-and-stop, retained-version resume, worker crash/failure and stale
identity. Clean Ubuntu 22.04 execution without Rust/GPU libraries passed startup,
tokenizer preparation, CPU training and complete-checkpoint verification.

NVIDIA qualification: RTX 5070 Laptop GPU through WSL, driver 591.59, temporary
NVRTC 12.8.61 selection with installed toolkit headers. Both final CUDA hardware
tests passed. The final release binary passed tokenizer, synchronized benchmark,
base/assistant training, schemas 7/8, exact resume and repeated separate-worker
continuation with identical model bytes; missing-header diagnostics passed.
CUDA remains experimental. Installed NVRTC 13.1 is incompatible with these
Burn 0.18 bindings; the driver/toolkit installation was not changed.

Remaining acceptance gate: qualify the final Vulkan executable on the A770 once
idle. Read-only inspection confirmed its existing training process is still
active; it was not interrupted or used for competing work. No automatic reboot
recovery, provisioning, multi-GPU training or separate-desktop control is claimed.
See [application guide](Docs/omega.md) and [validation evidence](Docs/omega-validation.md).


## OB01 — Dataset-based training-time benchmark library and CLI — 2026-10-05

Status: done (software; GPU runtime unverified). Owner: coordinator. User requested `omega-benchmark`, a library
and CLI whose first benchmark estimates training time for a selected dataset.
Scope: reuse checked training/corpus APIs; CPU and optional Vulkan; bounded
warmup/timed real-data updates, spread across the dataset; exact dataset/batch
counts and explicit compute-time extrapolation with setup/exclusions; text,
JSONL, chat and existing base caches; model/tokenizer settings, human/JSON output.
No checkpoints, production training, corpus downloads, backend changes or model
quality claims. Dependencies flow benchmark -> training/nn/tokenizer.
Owned writes: new crate, workspace manifest/lock entry, shared dataset-format
resolver extracted from training CLI, focused tests and relevant docs. Preserve
all existing data, executables and checkpoint compatibility guards.
Acceptance: estimator arithmetic/sampling/budget/failure tests; real tiny CLI
CPU smoke and release-format validation; workspace fmt/tests/strict Clippy;
GPU build check, with hardware execution only if an idle host is available.

Implementation: `Code/Rust/omega-benchmark/{Cargo.toml,src/lib.rs,src/main.rs,
tests/training_time.rs,README.md}` adds `training_time`, coverage-weighted
projection and the `training-time` subcommand. Workspace `Cargo.toml`/`Cargo.lock`
register the package. `omega-training/src/selection.rs` now owns the existing
format/release guards, exported through `src/lib.rs` and reused by its CLI in
`src/bin/main.rs`; the training behavior is preserved. Documentation links and
scope are updated in `readme.md`, `Docs/omega-training.md`,
`Docs/implementation-status.md` and the training README.

Compatibility: no dataset/tokenizer/checkpoint schemas change. The added
workspace lock entry changes exact-resume build identity; keep original training
executables for existing runs. No compatibility checks are bypassed. Benchmark
output is new-file-only JSON, schema 1. Budget applies between warmup/timed
updates, not preparation; no ETA on incomplete timing. Fixed-order estimates
exclude saves/evaluation/logging, use fresh weights and do not predict quality.

Hardware status (2026-10-05): read-only remote inspection found an active
`omega-training train --backend vulkan` process (PID 684546, batch 2, six epochs).
No hardware benchmark was launched and the run/executable/checkpoints were not
modified. The new ignored Vulkan fixture test is provided for an idle host.

Validation (2026-10-05, Windows, current final source): `cargo test --workspace
--locked --jobs 1` passed **275 tests**, including eight new benchmark integration
tests; `cargo fmt --all --check` and `cargo clippy --workspace --all-targets
--locked -- -D warnings` passed. Benchmark GPU `cargo check --features gpu
--all-targets --locked` and strict all-target GPU Clippy passed. Benchmark CLI
help/human/JSON output, explicit CPU pool startup, no-clobber reports and error
exits passed; training `train` and `generate` help passed after resolver extraction.
Affected documentation's local links resolve. Commands used explicit local target
directories outside the SMB checkout. Existing legacy `main` output-name warnings
remain; the new binary has the unique name `omega-benchmark`. No GPU runtime
or representative production-corpus accuracy claim is made from CPU fixtures.

## T20.G6 — Reduce GPU validation readbacks — 2026-09-30

Status: done. Owner: assistant-training coordinator. User authorized profiling
and device-side numerical checks/norm calculation after live A770 inspection.
Scope: bounded before/after profiling; GPU reductions with compact host results;
preserve CPU arithmetic and transactional rejection, clipping, shape/ID/counter
checks; assess fusion separately against measured evidence. Do not alter the
active executable or interrupt training without the user's explicit choice.
Owned files: training numerical-validation module, optimization/trainer wiring,
GPU benchmark and focused tests, execution-profile compatibility, relevant docs.
Acceptance: nonfinite/negative-moment and extreme-norm regressions, unchanged
unclipped updates, clipping tolerance and rejected-update rollback; measured
batch-2/4 targets/sec and stage timings on an idle A770; workspace validation.
Existing checkpoints/data are preserved; any changed GPU arithmetic requires an
explicit execution-profile compatibility decision before use.

Delivered: new `omega-training/src/numerical.rs` with IEEE-bit finite/sign checks
and scaled deterministic per-axis norm reductions; compact gradient summaries
and one candidate-validity readback. Integration in `src/optimization.rs`,
`src/trainer.rs`, `src/lib.rs`; GPU execution revision in `src/gpu.rs`;
`examples/gpu_benchmark.rs --profile`; regression coverage in these modules and
`tests/gpu_resume.rs`. CPU arithmetic is unchanged. GPU v1 headers remain
readable, but v2 exact resume rejects v1 execution profiles; no metadata rewrite,
weights-only migration, manifest/lockfile edit or dependency upgrade. Old
checkpoints require the original executable/runtime to continue exactly.

Two GPU edge failures found during development are fixed and covered: reciprocal
overflow/denormal flushing at extreme norms, and nondeterministic global atomic
sums affecting exact-resume events. Integer views preserve f32 bits/strides;
subnormal mantissas are reconstructed at a normal scale. Small clip multipliers
use two normal-range factors. Invalid candidates still leave model, Adam,
counters and loss bookkeeping untouched.

Measured report: `Docs/benchmarks/t20-g6-gpu-checks.json`. Ten idle-host sequential
release cases with matching initial parameters, two warmups/seven timed updates,
alternating before/after repeats. Batch 2: 212.65 -> 345.20 targets/sec (+62%);
batch 4: 363.83 -> 538.03 (+48%). Candidate validation: about 0.19 -> 0.023 s.
An isolated fusion/host-check experiment showed no benefit (210.18/351.48
targets/sec); production fusion stays off. Combined fusion/device checks are
not qualified. Stage times include deferred GPU execution, not kernel-only cost.

Validation: Windows `cargo test --workspace` (267 passed), focused final numeric
test, `cargo clippy --workspace --all-targets -- -D warnings`, GPU-feature
all-target Clippy, and `cargo fmt --all --check`. Linux release A770: four ignored
library qualification tests plus six ignored integration tests across
`gpu_resume`, `gpu_masked_training`, `gpu_subprocess` and `assistant_cli`, all
passed; includes clipped exact resume, new assistant stage, negative/nonfinite
rejection, transactional rollback and real SIGINT/SIGTERM handling. The final
benchmark reran all four library qualification tests before measurement.

Operational handoff: user explicitly authorized a graceful stop. Saved and
verified `weights/gpu-pilot-3` at update 687 / 75,364 targets. Training remains
stopped. Original executable preserved at
`/home/jack/rust/omega-g6-baseline/omega-training-v1`; optimized executable is in
the separate `/home/jack/rust/omega-g6-build/release/omega-training` target.
The original build target, corpus, tokenizer and existing checkpoints were not
overwritten. Temporary benchmarks/checkpoint tests used isolated outputs.
Updated training README, GPU guide and implementation status. No commits; ZH14
and unrelated task ownership/status are unchanged.

## CT02 — Match container flags to dataset/training CLIs — 2026-09-30

Status: flag configuration complete; container execution blocked under CT01.
Owner: container-workflow coordinator. User requested checking
both CLIs and configuring the dataset-build/training flags. Prerequisites: OD01
producer and OD02 checked release ingestion. Scope: concrete omega-alpha pipeline,
explicit container roots, auto format and frozen chat-capable tokenizer before
base initialization; explicit reuse of an existing dataset release; runner tests
and documentation. Preserve all existing corpus/tokenizer/checkpoint files and
Rust APIs. Do not launch full-corpus preparation or training as a flag check.
Acceptance: options verified against actual CLI parsers/source; correct published
train-only partition and tokenizer identity; no replacement of existing release;
bounded model updates; tests for skip behavior and command construction.

Changed paths: `Scripts/run_containers.py`, `Containers/pipeline.example.json`,
new `Containers/pipeline.omega-alpha.json`, `Containers/tests/test_runner.py`,
`Containers/README.md`, `Scripts/README.md`, `Docs/implementation-status.md`,
and this board. `all` now defaults to the concrete omega-alpha plan; the optional
`--skip-dataset-build` reuses its existing release without rebuilding/overwriting.
Explicit `--plan` remains supported. Both plans use explicit container roots and
auto format; tokenizers register Omega chat controls before base initialization.
The concrete plan writes a new `omega-tokenizer-v1.json`, preserving the existing
151643-entry base BPE vocabulary/tokenizer file. No SFT is configured for empty
chat partitions. The CPU pilot is capped at two updates; full eager preparation
and tokenizer fitting still process the roughly 405 MiB train file.

Validation: reviewed both manifests, READMEs, CLIs, corpus/cache/release loaders,
builder new-output guards and producer-to-trainer integration tests. Configured
options were cross-checked against the actual CLI source, including auto format,
model/thread options and train-only release guards. The existing COMPLETE marker
matches the manifest hash; selected payloads were not rehashed in this setup pass
(Rust does this on load). New tokenizer output is unused. `python -B -m unittest
discover -s Containers/tests -v`: 14 passed, including new default-plan/tokenizer
alignment and skip-only-dataset tests. Runner help, dry runs and 8 local doc links
passed. Actual `all --skip-dataset-build` fails cleanly at Docker preflight because
the Linux engine is unavailable. No corpus download, tokenizer fitting or training
ran; no Rust code, existing release, tokenizer, checkpoint or recipe was modified.
Cargo build/test/Clippy were not rerun for this Python/config/documentation change;
CT01's Linux image/runtime acceptance remains outstanding.

## OD02 — Consume omega-datasets releases — 2026-09-30

Status: done. Owner: assistant-training coordinator. User requested reviewing
`omega-datasets` output and updating `omega-training` to fit it. Prerequisites:
OD01 output schema and ZH conversation contracts are implemented and reviewed.
Scope: recognize explicit published base/chat partition directories, validate
completion/manifest/selected payload integrity, infer format while retaining
legacy text defaults, prevent release-parent and held-out training selection or
re-splitting prepared groups, and prove the actual producer-to-trainer workflow.
No downloads, production training, tokenizer replacement or checkpoint migration.

Ownership: `corpus` exclusively owns new `omega-training/src/release.rs` and
`tests/release.rs`; `training` owns new `tests/dataset_release_cli.rs` using the
actual dataset builder with an offline Hub. Coordinator owns exports, discovery,
CLI, manifests/lockfile, shared docs and board. A test-only `omega-datasets`
dependency is deliberate; production dependency direction stays unchanged.
Lockfile identity changes retain strict existing resume rules.
Acceptance: valid base/chat release train/tokenizer/cache/evaluate/stage/resume;
corrupt/incomplete/empty/wrong-stage and leakage failures; legacy behavior; full
workspace fmt/test/strict Clippy and focused GPU-feature compilation.

Delivered:

- `Code/Rust/omega-training/src/release.rs`: identify exact published partitions;
  bounded marker/manifest checks; streamed selected JSONL hash/count; incomplete,
  empty, extra-file, parent-directory and symlink/reparse rejection. Exported by
  `src/lib.rs`, integrated into shared `src/dataset.rs` discovery and cache reads.
- `Code/Rust/omega-training/src/bin/main.rs`: default `auto` resolves published
  base/chat formats, leaves ordinary folders in text mode, rejects incompatible
  explicit formats and mixed stages/partitions, restricts fitting to published
  train partitions, and rejects re-splitting prepared groups. Resume keeps saved
  resolved formats and normal source/runtime checks.
- `Code/Rust/omega-training/tests/release.rs` and `tests/dataset_release_cli.rs`:
  integrity/platform tests and actual producer-to-tokenizer/cache/base/evaluation/
  SFT/resume coverage using an offline Hub and temporary files.
- `Code/Rust/omega-training/Cargo.toml` and `Code/Rust/Cargo.lock`: only test-only
  `anyhow`/`omega-datasets` dependency links added; no package versions changed.
- Usage/compatibility updates: `Code/Rust/omega-training/README.md`,
  `Docs/omega-datasets.md`, `Docs/omega-training.md`,
  `Docs/implementation-status.md` and this board.

Validation on Windows/Rust 1.93.0, current four-crate workspace:

- `cargo test --workspace --locked --jobs 1`: 266 passed. The earlier package run
  found a Windows junction fixture passing a slash-containing path to `mklink`;
  corrected to native path components and verified by the full rerun.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings` and
  `cargo fmt --all --check`: passed on the final files.
- `cargo check -p omega-training --features gpu --all-targets --locked --jobs 1`:
  passed. Actual GPU execution and Unix symlink tests were not rerun for OD02;
  the current SSH fields were blank, so validation ran locally.
- `cargo run -p omega-training --bin main --locked -- train --help` and
  `generate --help`: passed; auto is documented as the default. CLI regressions
  also verify nonzero failures and the unique training binary.
- 62 local documentation links checked before the final handoff update; no
  broken links. Snapshot: external `omega-od02-before-yve0x8ec/before.zip`.

Limits/compatibility: only selected payloads are verified, not raw/membership or
unselected data. Hashes are consistency checks, not authenticity or corpus
acceptance. Existing eager/chat/cache/context bounds remain; no truncation or
automatic sampling. Library callers constructing their own examples retain
partition/objective responsibility. Lockfile identity changed, so old exact
resumes may require their original build; no migration guard was widened. No
corpus download, production training, existing dataset/checkpoint changes,
branches or commits were made.

## CT01 — Container workflow — 2026-09-30

Status: implementation ready; validation incomplete (not done).
Owner: container-workflow coordinator in this chat.
User scope: containers in `Containers/` to compile every Rust binary, prepare
datasets with `omega-datasets`, and run `omega-training`; one Python runner in
`Scripts/` orchestrates the stages. Prerequisites: existing Rust CLI contracts,
locked workspace, and OD01's dataset CLI (under implementation at assignment).
Exclusive implementation writes: new container files and Python runner/tests;
additive documentation/task updates only. Do not edit Rust, corpora or weights.
Acceptance: isolated binary outputs despite shared `main` names; persistent
dataset/checkpoint mounts; explicit sequential workflow with failure propagation;
dry-run and regression coverage; documented commands and actual validation.
Docker/Linux execution and workspace checks must be reported before completion.

Delivered: `Containers/Dockerfile`, `Containers/Dockerfile.dockerignore`,
`Containers/build-rust.sh`, `Containers/pipeline.example.json`,
`Containers/README.md`, `Containers/tests/test_runner.py`, and
`Scripts/run_containers.py`. Additive navigation/usage updates in
`Scripts/README.md`, `readme.md`, `Docs/omega-training.md`,
`Docs/implementation-status.md` and this board. No Rust source/manifests,
lockfile, existing datasets, tokenizer IDs or checkpoints were changed by CT01.

Decisions: three targets in one multi-stage Dockerfile; Linux CPU release builds
with the existing Rust 1.93.0 toolchain and locked dependencies; Cargo metadata
discovers all binaries, each package has its own cached target directory and
artifact directory. Runtime images reuse the unique dataset/training executables.
The runner uses argument arrays, validates the pipeline structure before work,
bind-mounts persistent roots, forwards HF_TOKEN by name only to dataset processes,
and stops on any error. `all` performs dataset build then ordered training CLI
steps, including optional tokenizer preparation. The example uses only base/train
and at most two tiny CPU updates. No actual download or training was performed.

Validation on 2026-09-30:

- `python -B -m unittest discover -s Containers/tests -v`: 12 passed. Mocked
  Docker/Cargo cover colliding binaries, failure at every pipeline stage, literal
  arguments, roots with spaces, invalid plans, token handling, daemon errors and
  timeouts, interruption, successful sequencing and arbitrary invocation directory.
- Runner help and `--dry-run all --plan Containers/pipeline.example.json` passed;
  56 local Markdown links checked. Build helper shell syntax passed `sh -n`
  through WSL (WSL warned that it could not translate the host Z: working directory).
- `cargo metadata --locked --no-deps --format-version 1` succeeded.
  `cargo fmt --all --check` passed on the final check; the first attempt found
  formatting in concurrent OD01 edits, which CT01 did not modify.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`
  passed in 3m43s, checking all four crates and their default-feature targets.
- `cargo test --workspace --locked --jobs 1` was interrupted after approximately
  ten minutes compiling dependencies; it had not reached test execution. No
  passing Rust test result is claimed.
- `python -B Scripts/run_containers.py build` failed cleanly with a nonzero engine
  preflight error. Starting Docker Desktop exposed a host backend crash while
  opening its `dockerInference` socket (invalid/inaccessible path). No images
  could be compiled or containers run. Docker configuration/data were not reset.

Remaining: complete the workspace test suite; build all three images on a
working Docker Linux engine; verify CLI help, dataset init/check and tiny training
with temporary host mounts. Linux linking, actual container signals/UID handling
and Windows mapped-network-drive bind mounts remain unverified. Runtime paths
may need local host dataset/weights overrides for this Z: checkout. This handoff
does not claim completed CT01 acceptance or a qualified training environment.

Planning baseline: **2026-09-27**, from [implementation status](Docs/implementation-status.md)
and [the training guide](Docs/omega-training.md). Follow [AGENTS.md](AGENTS.md).
Verify current source before claiming a task; the review is a historical snapshot.

## From-scratch conversational assistant roadmap — 2026-09-30

### Execution assignment — 2026-09-30

The user authorized sub-agents and execution of ZH01–ZH13, excluding ZH14.
Coordinator owns integration, shared manifests/exports/CLI/docs and this board.
Initial prerequisite review is read-only: `protocol` reviews ZH03's tokenizer/chat
contract; `training` reviews ZH06/ZH07 interfaces and persistence; `corpus` reviews
ZH01/ZH02 local data and preparation needs. These workers have no write ownership
until the coordinator records a bounded assignment and agreed interface here.
Coordinator inventories the actual runtime/hardware and requests missing scope,
data and training budgets. Real training gates remain unmet until measured.

Implementation wave 1: prerequisite **tooling portions** are assigned now; final
corpus, frozen tokenizer and measured training gates remain open. Shared API:
`omega-chat-v1`, four registered role/end-turn control IDs, optional initial system
then alternating user/assistant messages. Complete records end assistant; prompts
end user. Assistant content/end-turn targets only; mask aligns to `ids[1..]`.
Base text retains no automatic document-end insertion. Existing fixtures unchanged.

- `protocol`: exclusive writes `Code/Rust/omega-tokenizer/src/chat.rs`,
  `src/training.rs` and `tests/chat.rs`; opt-in BPE + safe literal content encoding.
  Coordinator alone exports modules and updates README/manifest if needed.
- `training`: exclusive writes `Code/Rust/omega-training/src/dataset.rs`,
  `batching.rs`, `trainer.rs`, `evaluation.rs`, `tests/masked_training.rs`,
  `tests/gpu_masked_training.rs`. Extend `ExampleSource` with default all-target
  objective and optional aligned mask; retain legacy all-target behavior/identity.
  New assistant objective is `assistant-next-token-omega-chat-v1`.
- `corpus`: exclusive writes `Code/Python/prepare_assistant_data.py` and
  `Code/Python/test_prepare_assistant_data.py`; explicit reviewed source/group
  partition spec, bounded duplicate audit, immutable new-path outputs, no source
  corpus changes or fabricated demonstrations.
- Coordinator: checkpoint/resume schema and lineage, conversation loader, CLI,
  shared exports/docs, experiment specification/workflow and integration tests.
  Snapshot before code changes: external temp `omega-zh-implementation-fpyfdfe7`.
  Do not format another owner's files or build competing `main` binaries.

Wave 1 follow-up for `corpus` after its preparation tests: exclusive new files
`Code/Python/assistant_experiment.py` and `test_assistant_experiment.py` for ZH05's
bounded subprocess/report tooling. Immutable run recipes, explicit checkpoint
selection, timeout/failure handling and manual rubric records; no actual training
or acceptance claims from mocked results. Coordinator retains docs and run gates.

Wave 1 follow-up for `protocol` after its 8 focused tests passed: exclusive new
`Code/Rust/omega-training/tests/assistant_cli.rs` for temporary-root unique-binary
CLI regressions covering tokenizer/base/stage/resume/evaluation/chat failure
contracts. Coordinator retains CLI implementation and fixes review findings.

Integration follow-up for `training`: exclusive test-only ownership of
`Code/Rust/omega-training/tests/assistant.rs` (released by coordinator) and its
existing `tests/gpu_masked_training.rs` to extend parent failure/reset and GPU
stage regressions. No production writes. Concurrent separate chat is creating
`omega-datasets`; its manifest/new crate/lockfile changes are preserved. Until
that workspace stabilizes, original three crates are validated from an external
copy with matching source and pre-concurrent workspace manifest/lockfile.

Compatibility follow-up: `protocol` owns `omega-tokenizer/src/chat.rs` and
`tests/chat.rs` for `ChatProtocol::has_registered_controls(&Tokens) -> bool`;
only registered special controls imply chat intent. `training` owns
`omega-training/tests/assistant.rs` for a plain-vocabulary checkpoint regression.
Coordinator owns the `ChatArtifact` integration. Ordinary vocabulary spellings
must preserve plain schema-1 saves; partial special registration still fails.

Remote follow-up: user supplied `.env` SSH settings; Linux A770 and shared source
are reachable. Coordinator owns bounded remote qualification. `protocol` owns
only `omega-training/tests/assistant_cli.rs` for an explicitly ignored GPU chat
integration test (feature gated, actual backend request, temporary artifacts).
No production corpus or long training is authorized by these test fixtures.

Linux portability follow-up: `corpus` owns only
`Code/Python/test_assistant_experiment.py` to resolve the test interpreter's
physical executable path (Linux `/usr/bin/python3` is commonly a symlink) while
preserving the recorder's explicit symlink rejection. Coordinator runs both
platform suites and retains documentation ownership.

Four-crate integration follow-up: coordinator owns the test-only correction in
`omega-training/src/resume.rs` for the new workspace lockfile. Retain the existing
production migration allowlist and platform checks; do not grant compatibility
to an unrelated lockfile merely because the previous test expected migration.

ZH software handoff: formatter/tokenizer opt-in, bounded conversation loading,
assistant-only loss/evaluation/accounting, schema-2 chat manifests, schema-5/6
continuation, explicit fresh-Adam `train-stage`, parent lineage and local `chat`
are implemented. Python preparation and immutable bounded experiment/report
tools are implemented. ZH06/ZH07/ZH12 pass CPU and actual A770 regressions; ZH02,
ZH03 and ZH05 still require real accepted artifacts/evidence. ZH04's hardware
checks pass but its representative capacity sweep and run budget remain open.
Validation and runtime details: [assistant execution](Docs/assistant-execution.md).
Final current four-crate Linux checks: 254 tests passed, strict all-target Clippy
passed, formatting passed. Actual A770: two assistant training/stage tests and
two CLI chat/stage tests passed. Python: 18 preparation + 14 experiment tests
passed on Linux; Windows preparation has one symlink-privilege skip. Isolated
Windows three-crate regression and strict CPU/GPU Clippy checks also passed.
The user supplied SSH access through ignored `.env`; corpus/scope/rubric and
training-budget fields remain blank. No production tokenizer/model, long training,
sealed-test acceptance, release or ZH14 optimization is claimed.

- ZH00 / coordinator / done. User-requested documentation assignment: create
  `zero-to-hero.md`, add actionable tasks here, and link the guide from `readme.md`.
  Scope is documentation only; no training, downloads, code changes or commits.
  Confirmed target: a conversational assistant trained from random weights on one
  16 GB Intel Arc A770, beginning with bounded pilots. Operating system, corpus,
  language/domain, quality thresholds and total training-time budget remain open.
  Delivered the guide, 14 dependent tasks with acceptance criteria, and README
  navigation. Checked 23 local Markdown links, task IDs/dependency cycles, current
  CLI examples against source, balanced code fences, and preservation of all
  historical task-board content. No builds/tests/training were run for this
  documentation-only change; no implementation task is marked complete.

See [zero-to-hero.md](zero-to-hero.md) for the staged workflow, current command
examples and limits. The tasks below are planning items, not authorization to
download data, implement every item or launch training. Existing accepted work
T01–T19 and T21.1/T22.1 is reused; GPU functionality is implemented under T20.
The new tasks add the assistant-specific gaps and experiment deliverables.
`ready` means eligible for assignment, not started; owners remain unassigned.

| ID | Deliverable / type | Status | Owner | Depends on |
|---|---|---|---|---|
| ZH01 | Assistant specification and pilot budget / planning | active | coordinator | None |
| ZH02 | Curated, deduplicated base/chat corpus releases / data and tooling | active | corpus / coordinator; tooling tested, data gate open | ZH01 |
| ZH03 | Frozen tokenizer and versioned chat protocol / implementation and artifact | active | protocol / coordinator; software tested, artifact gate open | ZH02 |
| ZH04 | A770 capacity, runtime qualification and run budget / measured experiment | active | coordinator; remote hardware checks underway, corpus/budget gates open | ZH03; reuse T20 hardware checks |
| ZH05 | Reproducible run/evaluation workflow / tooling | active | corpus / coordinator | ZH02, ZH03 |
| ZH06 | Conversation examples and assistant-target training/evaluation / implementation | done | training / coordinator; CPU and A770 regressions passed | ZH03 protocol contract implemented; production artifact still open |
| ZH07 | Explicit base-checkpoint-to-SFT stage initialization / implementation | done | coordinator; CPU and A770 library/CLI regressions passed | ZH06 |
| ZH08 | Bounded from-scratch base pilot and recovery / experiment | blocked | — | ZH04, ZH05 |
| ZH09 | Budgeted base pretraining and checkpoint selection / training | blocked | — | ZH08 |
| ZH10 | Bounded SFT pilot and base comparison / experiment | blocked | — | ZH05, ZH07, ZH09 |
| ZH11 | Budgeted assistant training and checkpoint selection / training | blocked | — | ZH10 |
| ZH12 | Local multi-turn chat CLI / implementation | done | coordinator / protocol; CPU and A770 CLI regressions passed | ZH03 protocol contract implemented; reuses existing generation |
| ZH13 | Sealed-test acceptance, model card and recoverable local release / evaluation | blocked | — | ZH11, ZH12 |
| ZH14 | Evidence-driven optimization extensions / conditional implementation | deferred | — | ZH04; measured need and bounded assignment |

T20 detailed profiling, T21.2/T21.3 inference caching/streaming and T22.2–T22.4
storage improvements retain their existing scopes/statuses. They are not blanket
prerequisites for the first small assistant. Model sizing and throughput must be
measured on the actual host; no model size, quality or completion date is promised.

### ZH01 — Agree the first assistant and bounded experiments

- [ ] Record language, domain, expected prompt/answer lengths and multi-turn
  behavior. Confirm actual A770 host OS/driver and available host RAM/disk.
- [ ] Define development and sealed-test prompts, human rubric, numerical pass
  thresholds and acceptable language-model regression before training comparisons.
  Include answer relevance, instruction following, context use, repetition,
  uncertainty, role leakage and correct reply termination.
- [ ] Record maximum pilot updates, wall time and output storage, plus stop/fail
  conditions. Record the separate decision gate for longer pretraining and SFT.
- [ ] Deliver an experiment specification; no large job or data acquisition is
  implied by completing the specification. Set dependent tasks ready only after
  their prerequisites are actually accepted.

### ZH02 — Prepare and freeze two corpus releases

Read first: training `src/dataset.rs`, `src/cache.rs`, tokenizer training,
`Code/Python/README.md`. Proposed preparation tooling paths require assignment;
do not rewrite source corpora or treat current JSONL as a conversation schema.

- [ ] Inventory permitted data sources for base text and assistant demonstrations;
  record source/group IDs, provenance, language, extraction and filtering rules.
  Review samples and report malformed, blank, too-short and oversized records.
- [ ] Detect exact duplicates and review near-duplicate/source-family groups;
  split groups before tokenization/chunking. Audit base/SFT overlap against both
  validation and sealed tests. Path deduplication alone does not satisfy this.
- [ ] Produce versioned train/validation/test directories and a machine-readable
  membership/hash/count report in new paths. Keep raw originals unchanged and
  never recursively select a parent containing validation/test for training.
- [x] Test deterministic preparation, group isolation, duplicate leakage, invalid
  input and overwrite rejection with temporary fixtures. Document human review
  and any unmeasured near-duplicate coverage; do not imply perfect detection.

### ZH03 — Freeze a tokenizer and shared conversation protocol

Read first: tokenizer `src/{lib,training}.rs`, tokenizer tests, NN encoding and
generation, training provenance. Agree cross-crate ownership before editing.

- [ ] Compare a bounded set of BPE candidates on training data only; report actual
  vocabulary, segmentation/length statistics, Unicode round trips and memory.
  Preserve the 13-entry fixture and existing no-special-token default API.
- [x] Define roles, role order, turn/end semantics, assistant prefix and literal
  delimiter escaping. Implement one versioned formatter for training/inference,
  with opt-in special-token registration before model initialization. IDs must
  come from the saved tokenizer; no guesses, retroactive addition or resizing.
- [x] Decide explicitly whether base documents receive an end token and implement
  any opted-in insertion with target-boundary coverage tests; retain legacy text
  behavior. Special-token reservation alone does not teach document endings.
- [x] Persist and validate tokenizer/protocol identity with model artifacts;
  agree schema and legacy policy before changes. Old plain-text checkpoints
  remain usable for their original operations; chat must not guess their format.
- [ ] Test serialization parity, special-ID stability, malformed roles, delimiter
  literals, no duplicate insertion and mismatched protocol rejection. Publish the
  frozen artifact and protocol contract before starting ZH06/ZH12 consumers.

### ZH04 — Qualify the actual A770 and measure the budget

Read first: `Docs/gpu-training.md`, T20 handoff, GPU probes/benchmarks and buffer
preflight. Reuse T20 tools; avoid duplicating detailed profiling work.

- [x] Record hardware/OS/driver/compiler/build and run relevant real-device
  regressions. Linux acceptance in the historical handoff does not certify a
  Windows GPU. Unavailable hardware means this gate stays unverified.
- [ ] Measure a bounded release sweep around an explicitly selected tiny model,
  context and batches on representative prepared data. Report actual parameters,
  real/padded tokens, throughput, host memory, device-memory measurement limits,
  failures, checkpoint size/save/load time and validation cost.
- [ ] Derive unique-token/exposure/update counts and realistic elapsed/disk bounds;
  distinguish SFT supervised targets from all input tokens. Include overhead and
  repeated-data exposure; do not extrapolate from unrelated synthetic benchmarks.
- [ ] Record the chosen architecture/tokenizer/runtime and pilot budget. Avoid
  changing a running job or upgrading dependencies as incidental performance work.

### ZH05 — Reproducible experiments and evaluation

Read first: metrics/evaluation/checkpoint catalog implementations and current CLI.
Start with a wrapper/report workflow where possible; do not invent unsupported
CLI flags or a second trainer. Conversation scoring uses the ZH03 formatter;
assistant-target loss integration is completed with ZH06.

- [ ] Save an immutable run recipe, hashes/runtime identity, exact commands and
  unique logs/output paths. Distinguish fresh train, exact resume and new stage.
- [ ] Evaluate an explicit saved checkpoint on external validation folders between
  bounded segments. The current CLI has no separate validation-folder train flag.
  Never select validation/test folders for model or tokenizer training.
- [ ] Retain fixed prompt outputs and rubric scores alongside loss, throughput,
  target exposures and elapsed time. Compare loss only with matched tokenizer,
  context and target semantics; do not tune against sealed-test results.
- [ ] Select checkpoints from validation evidence and record exact names/hashes;
  numeric latest is not best. Test failure propagation, partial logs, wrong paths,
  overwrite refusal and no fallback after a selected artifact fails loading.

### ZH06 — Assistant-only targets throughout the training path

Read first: training `src/{dataset,batching,trainer,evaluation,cache,resume,metrics}.rs`
and NN `src/loss.rs`. Agree the example/source API and exclusive shared-file owners.

- [x] Add an explicit versioned conversation format; preserve existing text/JSONL
  semantics. Keep whole conversations/source groups in one split. Validate roles,
  ordering and required assistant targets. Reject oversized conversations clearly
  in the first version; no silent truncation or context-free answer fragments.
- [x] Represent input validity separately from assistant-target eligibility.
  Preserve causal user/system context; supervise designated assistant content and
  terminators only, shift once, and exclude all padding targets.
- [x] Apply masking in batch-one and padded paths, fixed-model evaluation, target
  totals, weighted metrics and sampler accounting. Reuse the existing masked-loss
  primitive; ignored logits have zero direct loss gradient while context remains
  available to influence assistant predictions.
- [x] Include IDs, loss masks, formatting/objective version and source order in
  identity/provenance/resume validation. Version changed schemas with explicit
  legacy all-target loading. Either add verified mask-aware caches or reject SFT
  cache requests explicitly and document the eager-data bound for the first run.
- [x] Test boundary alignment, causal influence, zero ignored-logit gradients,
  all-ignored errors, short/final batches, changed-mask rejection and exact SFT
  continuation on CPU and qualified A770. Keep base-training regression coverage.

### ZH07 — Start a new SFT stage from our own base checkpoint

Read first: checkpoint/resume loaders, session constructors, CLI and run control.
Agree explicit operation/API names and schema policy before implementation.

- [x] Add an explicit weights-initialized stage operation using a completed Omega
  checkpoint with identical architecture, tokenizer and protocol; verify parent
  payloads/identity and preserve its files. No third-party import or vocab resize.
- [x] Copy the base model into a new stage, initialize fresh Adam/sampler/counters,
  accept explicit new data/settings and record parent name/hash and objective.
  Do not describe this as exact continuation of the base optimizer.
- [x] Keep existing `train` random initialization and `resume` strict restoration
  semantics. A saved SFT stage must resume its own data, masks and settings without
  overrides; no metadata editing or runtime-compatibility bypass.
- [x] Test initial weight equivalence, fresh optimizer state, parent immutability,
  invalid/incomplete/mismatched parents and split/uninterrupted SFT equivalence.
  Check CLI help, nonzero error exits and CPU/qualified-GPU workflows.

### ZH08 — Run a bounded base pilot

- [ ] Use only the accepted training partition, frozen tokenizer/protocol and
  qualified runtime; start from random weights. Record a tiny overfit diagnostic
  separately from a representative held-out pilot.
- [ ] Exercise periodic save, cooperative interruption, fresh-process exact resume,
  reload/evaluate/generate and backup restoration on owned pilot outputs.
- [ ] Compare validation and fixed generations at multiple saved boundaries;
  require ZH01 thresholds and ZH04 budget feasibility, not merely lower toy loss.
  Report failures and stop/revise rather than automatically extending the run.
- [ ] Deliver an accepted base-run recipe and measured budget, or leave this gate
  incomplete with the specific corpus/configuration/runtime problem to resolve.

### ZH09 — Train and select the base model

- [ ] Record the agreed larger-run budget and stop criteria; use the accepted
  recipe in bounded segments with metrics, periodic saves and storage checks.
- [ ] Preserve tokenizer/data/runtime identity for exact resume. Reconfiguration
  starts a separate run; never alter saved manifests to force continuation.
- [ ] Select a base checkpoint using validation and retained generation evidence;
  record exposures, elapsed time, failures and artifact hashes. Keep test data
  unused. Verify an independent backup before handing the base to SFT.

### ZH10 — Prove the conversation-training stage in a pilot

- [ ] Initialize from ZH09 with ZH07, fresh optimizer and a bounded ZH02 chat
  training subset; use the exact ZH03 protocol and ZH06 objective.
- [ ] Compare fixed development prompts before/after SFT, assistant-target
  validation loss, turn endings and base-language regression against ZH01 limits.
  A memorized training dialogue is a diagnostic, not conversational acceptance.
- [ ] Measure stage throughput/memory (including prompt tokens), exercise exact
  SFT resume and agree the longer-stage budget/settings only after the gate passes.

### ZH11 — Train and select the assistant

- [ ] Run the accepted SFT recipe within its agreed token/update/time/disk budget;
  use bounded segments, validation comparisons and verified periodic saves.
- [ ] Select the best validation checkpoint by the predeclared rubric and loss
  criteria, report regressions and preserve the base/parent lineage and backups.
- [ ] Freeze the candidate and inference settings before sealed-test evaluation;
  do not claim useful quality if the agreed criteria remain unmet.

### ZH12 — Local chat using the trained protocol

Read first: NN generation, training CLI, frozen ZH03 formatter. An interactive
terminal command is the first target; web UI/serving are not prerequisites.

- [x] Format system/user/history and assistant prefix with the same protocol used
  in SFT. Resolve stop IDs from the artifact, stop at the defined reply terminator,
  and display only newly generated assistant text without control delimiters.
- [x] Keep explicit bounded history; initially reject over-budget requests with
  an actionable reset path. Do not silently drop turns or promise long-term memory.
- [x] Preserve existing generate behavior/options and reject missing/mismatched
  chat metadata. Test role formatting, EOS, empty replies, repeated turns, Unicode,
  history reset, context overflow and greedy/seeded behavior on supported backends.
  KV cache/streaming remain T21 scopes, not required for this first chat command.

### ZH13 — Final evaluation and local release

- [ ] Evaluate the frozen assistant once on sealed data using the ZH01 rubric,
  including multi-turn behavior, unsupported answers, stopping and measured latency.
  Publish failures as well as scores; further tuning requires a new held-out plan.
- [ ] Package exact weights/tokenizer/protocol/config/provenance, a model card,
  training/evaluation report and reproducible local chat command. State training
  scope, context/resource limits and unsupported uses without quality inflation.
- [ ] Verify reload/chat on the supported runtime and restore from a separate
  backup. Preserve existing artifacts; no public upload, deployment or checkpoint
  deletion is implied. Mark done only when ZH01 acceptance thresholds are met.

### ZH14 — Conditional training improvements

This is deferred, not required to finish the first measured pilot. Assign bounded
subtasks only when evidence shows they are needed. Reuse T20 for profiling.

- [ ] If assigned, add a specified learning-rate decay schedule with persisted
  schedule/horizon and update accounting; default constant/warmup behavior stays
  compatible. Test boundary rates, save/resume and invalid settings.
- [ ] If assigned, add gradient accumulation with real-target weighting across
  microbatches, explicit optimizer-update versus microstep counters and defined
  partial-accumulation save/stop behavior. Test unequal-target batches against an
  equivalent combined objective, numerical rejection and exact continuation.
- [ ] Any further memory/kernel/precision change needs a separate measured design,
  hardware tests and migration policy; 16 GB is not permission to weaken checks,
  upgrade Burn or promise a large model fits. Update documentation for accepted work.

## Script utility assignment

- SCRIPT01 / coordinator / done. User-requested train-then-generate wrapper.
  Scope: `Scripts/training_run.sh`, script usage/tests, and this entry. Preserve
  Rust CLI behavior; accept multiple datasets, stop on failure, and resolve the
  latest completed checkpoint for the supplied run name. Bash syntax checks and
  mocked-Cargo tests passed for argument quoting, repeated datasets, working-directory
  resolution, help/invalid inputs, and stopping after training failure. No real
  training was launched. Concurrent jobs must use distinct run names.

## How to use this board

- This is a backlog, not authorization to implement everything. The coordinator
  selects a bounded task matching the user's request.
- Status: `ready` = no backlog prerequisite; `blocked` = listed prerequisites not
  done; `active` = assigned and in progress; `review` = handoff awaiting integration;
  `done` = accepted and validated; `deferred` = needs explicit scope/design approval.
- `ready` does not mean another agent's write set is available. Check conflicts.
- Replace the owner dash with an agent/session identifier when assigning work.
  Record exact allowed files and interface decisions in the assignment log below.
- Dependencies are **hard implementation prerequisites** unless labeled advisory.
  Read-only research may proceed earlier, but dependent code must not invent a
  competing interface. All unchecked acceptance criteria remain outstanding.
- Only the coordinator updates shared status, integration files, and documentation
  unless exclusive ownership is explicitly delegated.
- A completed task needs tests, documentation, a compatibility decision where
  relevant, and the validation/handoff required by `AGENTS.md`.

## Existing baseline — do not reimplement

The workspace already has saved-tokenizer encode/decode, a causal CPU GPT,
single-sequence optimization, document-local chunking across selected text
folders, one persistent optimizer within each training run, numbered inference
checkpoints, and saved-model greedy generation. These are not new backlog tasks.

The CLI now supports CPU training resume, bounded token caches, seeded shuffle
and weighted sampling, masked minibatches, clipping/warmup, and periodic or
cooperative-stop saves. It does not stream arbitrary-size documents. Evaluation,
tokenizer preparation, metrics and versioned provenance were added by the second
READY pass below. The 13-entry WordLevel fixture is not a production vocabulary.
See the docs for precise limitations.

## Task board

Priority indicates recommended sequencing, not permission to broaden scope.

| ID | Priority | Deliverable | Status | Owner | Depends on |
|---|---|---|---|---|---|
| T01 | P1 | Tokenizer and encoding regression coverage | done | worker-tokenizer | None |
| T02 | P1 | Model/configuration and tensor contracts | done | worker-model | None |
| T03 | P1 | Document-level train/validation splitting | done | worker-dataset | None |
| T04 | P1 | Held-out loss and perplexity | done | worker-dataset | T03 |
| T05 | P1 | Explicit persistent trainer state | done | worker-model | None |
| T06 | P1 | Versioned checkpoint manifest and provenance | done | worker-model | T02 |
| T07 | P1 | Training-state save/load and exact resume | done | worker-model | T05, T06 |
| T08 | P1 | Periodic saves and graceful interruption | done | coordinator | T07 |
| T09 | P1 | Live progress and structured metrics | done | coordinator | T05 |
| T10 | P1 | Tokenizer training and coverage tooling | done | worker-tokenizer | T01 |
| T11 | P2 | Reproducible shuffling and dataset mixing | done | worker-dataset | T03, T07 |
| T12 | P2 | Padding-aware attention and loss contracts | done | worker-tokenizer | T02 |
| T13 | P2 | Minibatch collation and training | done | worker-tokenizer | T05, T07, T12 |
| T14 | P2 | Streaming/token-cache design and implementation | done | worker-dataset | T03, T06 |
| T15 | P2 | Explicit additional dataset formats | done | worker-dataset | T03 |
| T16 | P2 | Cross-case checkpoint reservation safety | done | coordinator | None |
| T17 | P2 | Explicit CLI roots and end-to-end tests | done | worker-dataset | None |
| T18 | P2 | Reproducible build and documentation navigation | done | coordinator | None |
| T19 | P2 | Stateful optimization controls and numerical checks | done | worker-model | T07 |
| T20 | P2 | CPU optimization and multithreading; Arc A770 GPU training | active | coordinator | Core GPU backend qualified; detailed profiling remains |
| T21 | Later | Sampling and incremental generation | deferred | — | T02, approved inference scope |
| T22 | Later | Checkpoint lifecycle and stronger storage guarantees | deferred | — | T06, T16, approved retention/durability policy |

T08/T11/T13/T19 are accepted after the fourth READY pass. Explicitly assigned
T21.1 and T22.1 are now done; the parent T21/T22 tasks remain deferred because
their later phases are unassigned. T20.C1/C2/C3 are complete. The user subsequently
authorized GPU implementation with workers. A770 feasibility passed and the optional
backend is integrated; hardware regressions and final release validation passed.
Detailed performance profiling remains open. See the GPU phase table and implementation
handoff below; historical CPU results remain unchanged.

## First parallel wave

A useful first wave is **T01, T02, and T03**, with exclusive file scopes agreed
before delegation:

- T01: tokenizer source/tests and a new NN encoding integration-test file; avoid
  writing NN `src/lib.rs` while T02 owns it. Hand implementation defects to the
  coordinator if a fix needs that file.
- T02: NN model/helper implementation and its own tests, with exclusive ownership
  of NN `src/lib.rs` including its exports during this wave. The coordinator
  integrates any additional edits to that file only after T02 hands it back.
- T03: training dataset implementation and its own tests.
- Coordinator: manifests, other module exports, shared CLI wiring, docs, and this
  board. No concurrent writes to files exclusively delegated to an agent.

Do **not** blindly start every `ready` task at once:

- T05 and T09 change trainer state/reporting; T04, T07, T08, T11, T13, and T19 also
  need integration with the training loop. Serialize shared edits or first agree
  a modular interface and assign nonoverlapping modules.
- T06 and T16 both touch checkpoint code. Give that file to one owner at a time.
- T17 touches the same training CLI as evaluation, resume, and metrics work.
- T18 owns shared build/docs files only when the coordinator explicitly delegates
  them. Package-wide formatting is a write conflict during parallel work.

The source areas below are **starting points, not blanket write permission**.
`training` means `Code/Rust/omega-training`, `nn` means `Code/Rust/omega-nn`, and
`tokenizer` means `Code/Rust/omega-tokenizer`. Proposed files are labeled *new*;
they are not assumed to exist, and final module/API design belongs to integration.

## P1 acceptance criteria

### T01 — Tokenizer and encoding regression coverage

Read: `tokenizer/src/lib.rs`, `tokenizer/src/bin/main.rs`, `nn/src/lib.rs`.
Suggested writes: tokenizer tests and `nn/tests/encoding.rs` (*new*). Coordinate
any production fix touching NN helpers or tokenizer CLI.

- [x] Test saved special-token processing, tokenizer save/load, malformed/missing
  files, empty text, and vocabulary lookups without altering `datasets/test.json`.
- [x] Test sparse/empty vocabulary handling, actual padding rejection, and reported
  truncation-overflow rejection in NN helpers using separately generated fixtures.
- [x] Cover tokenizer CLI action conflicts and nonzero error exits with temporary
  files; distinguish library-relative and CLI dataset-relative paths.
- [x] Preserve the loaded pipeline; do not add guessed BOS/EOS IDs to pass tests.

### T02 — Model/configuration and tensor contracts

Read: `nn/src/model.rs`, `nn/src/lib.rs`, and checkpoint/generation call sites.
Suggested writes: NN model/helper code; coordinator integrates checkpoint callers.

- [x] Define a reliable model architecture accessor or equivalent invariant so
  generation and saving can reject mismatched configurations before tensor work.
- [x] Preserve existing valid APIs where practical; document any intentional break
  and how existing callers migrate. Keep orchestration out of the model crate.
- [x] Add tests for mismatch rejection, token bounds, empty shapes, context limits,
  and explicit next-token alignment; retain causal prefix-invariance coverage.
- [x] Distinguish checked helper errors from documented low-level tensor panics.

### T03 — Document-level train/validation splitting

Read: `training/src/dataset.rs` and current `TrainingSet` consumers.
Suggested writes: dataset module and focused tests; exports/CLI are coordinator-owned.

- [x] Retain stable document identity through selection and splitting, and split
  **before chunking**. Agree the resulting data structures with T04/T06 consumers.
- [x] Expose a deterministic seeded split policy with explicit handling of zero
  validation, insufficient documents, empty partitions, and invalid ratios/counts.
- [x] Ensure document membership is disjoint, overlapping folder selections remain
  deduplicated, and selection order does not accidentally change split membership.
- [x] Preserve every within-document adjacent target pair exactly once in its
  partition, including short remainders; never create cross-file targets.
- [x] Preserve the current all-training behavior when splitting is not requested.

### T04 — Held-out loss and perplexity

Depends on T03. Read: dataset outputs, training loop, and NN forward/label handling.
Suggested module: `training/src/evaluation.rs` (*new*); coordinator wires exports/CLI.

- [x] Evaluate a fixed inference model without parameter or optimizer updates.
- [x] Aggregate cross entropy by target count, not by unweighted batch/chunk means;
  report target count and perplexity with explicit non-finite/overflow behavior.
- [x] Handle empty/invalid validation inputs with actionable errors and document
  the distinction from pre-update training-loss summaries.
- [x] Test known logits/targets, different chunk lengths, absence of model mutation,
  and split isolation; expose a documented library and CLI evaluation workflow.

### T05 — Explicit persistent trainer state

Read: `training/src/lib.rs`, `nn::train_on_tokens`, and Burn 0.18 optimizer APIs.
Suggested module: `training/src/trainer.rs` (*new*) with coordinator-owned wrappers.

- [x] Extract model, Adam state, counters, and agreed configuration into a session
  that can advance updates without resetting the optimizer or initialization.
- [x] Define update/epoch boundaries and an event or callback interface usable by
  evaluation, metrics, and checkpoint orchestration without circular dependencies.
- [x] Preserve the existing `train` behavior through a wrapper or documented migration.
- [x] Test that chunked calls to the session preserve state relative to an equivalent
  uninterrupted sequence; isolate shared RNG usage in comparisons.
- [x] This task does not claim disk resume; hand off the concrete state API to T07.

### T06 — Versioned checkpoint manifest and provenance

Depends on T02. Read: `training/src/checkpoint.rs`, dataset identity, model metadata.
Suggested writes: checkpoint implementation and manifest schema; shared dependency
and CLI changes require coordinator ownership.

- [x] Define schema version, authoritative model configuration, tokenizer identity,
  dataset fingerprints/selections, training hyperparameters, and build identity.
  Specify how unavailable metadata is represented; never fabricate it.
- [x] Validate model/config compatibility and tokenizer identity, not size alone.
- [x] Preserve current numbered saves and marker-last completion. Metadata must
  be written before `COMPLETE`, not patched into an already completed checkpoint.
- [x] Keep existing unversioned inference checkpoints loadable or provide an
  explicit, tested migration path; reject unknown schemas clearly.
- [x] Test round trips, tampering/mismatches, missing fields/files, unsupported
  versions, and legacy compatibility using temporary directories.

### T07 — Training-state save/load and exact resume

Depends on T05 and T06. Read: trainer state, checkpoint schema, Burn recorder APIs.
Suggested split after API agreement: state serialization module and trainer restore
adapter; one coordinator owns shared checkpoint/CLI edits.

- [x] Persist/restore model, optimizer, completed update count, epoch/data cursor,
  and all ordering/RNG state affecting continuation. Investigate actual Burn RNG
  support; do not claim exact resume if required state cannot be restored.
- [x] Reject incompatible tokenizer, data, architecture, or state versions before
  an update; define which runtime overrides, if any, are safe.
- [x] Add an explicit resume workflow distinct from new training and inference
  loading. Legacy weights-only checkpoints remain valid for inference, not resume.
- [x] Compare interrupted-save-load-continue with uninterrupted training on tiny
  CPU data, including optimizer state and model outputs within justified tolerances.
- [x] Test mid-epoch and epoch-boundary resumes, corrupt/missing state, and errors
  without modifying the original checkpoint. Keep unsupported cases blocked.

### T08 — Periodic saves and graceful interruption

Depends on T07. Read: trainer boundary events and resume/checkpoint contracts.

- [x] Support documented step/epoch save intervals; preserve a final save and avoid
  ambiguous duplicate saves when interval/final boundaries coincide.
- [x] Save at consistent update boundaries with the correct next-data cursor.
- [x] Handle interruption using a safe signal/notification mechanism; do not perform
  complex checkpoint I/O directly inside a signal handler.
- [x] Test interval selection, interruption and subsequent resume, write failures,
  numbering, and rejection of incomplete saves. Document supported platforms.

### T09 — Live progress and structured metrics

Depends on T05. Read: trainer event interface and current CLI output.
Suggested module: metrics/progress observer (*new*); no second training loop.

- [x] Emit progress during training, with epoch/update/target counts and accurately
  labeled training losses. Integrate validation metrics only once T04 is available.
- [x] Support a stable machine-readable metrics format and error policy for logging
  failures; avoid claiming pre-update loss is final-model validation loss.
- [x] Keep library output configurable rather than hardcoding stdout writes.
- [x] Test event order, aggregation, partial-run records, and CLI output behavior.

### T10 — Tokenizer training and coverage tooling

Depends on T01. Read: tokenizer wrapper and saved-tokenizer/model identity contracts.
Suggested ownership: tokenizer training/inspection code; corpus selection stays in
training or a coordinator-agreed shared interface, not duplicated path logic.

- [x] Agree a supported tokenizer family and normalization/BOS/EOS policy before
  implementation; use compatible existing `tokenizers` APIs where practical.
- [x] Train/save to an explicit new path with overwrite protection and configurable
  vocabulary settings. No downloads or large jobs as part of implementation tests.
- [x] Report coverage/unknown-token statistics on supplied text with clear metric
  definitions; document limitations of unknown-rate metrics for byte-level models.
- [x] Test save/reload stability and encoding/decoding expectations using tiny local
  corpora; keep all existing fixture IDs and tests unchanged.

## P2 acceptance criteria

### T11 — Reproducible shuffling and dataset mixing

Depends on T03 and T07. Read: document identity, trainer cursor, saved resume state.

- [x] Define per-epoch shuffling and optional dataset-weighted sampling without
  mixing validation into training. Preserve deterministic defaults unless changed
  explicitly through a documented option.
- [x] Persist ordering/sampler state, test seed repeatability and resume equivalence,
  and document whether an epoch covers each example once or samples with replacement.

### T12 — Padding-aware attention and loss contracts

Depends on T02. Read: `nn/src/model.rs` and loss computation call sites.

- [x] Agree tensor shapes and semantics for attention masks and ignored targets;
  combine padding and causal masks without allowing future-token visibility.
- [x] Define behavior for fully masked rows/batches and avoid NaNs or zero-denominator
  metrics. Preserve the existing unpadded path.
- [x] Test padded/unpadded valid-token equivalence, unchanged causal behavior,
  zero contribution of padded targets, and shape/error validation.

### T13 — Minibatch collation and training

Depends on T05, T07, and T12. Read: dataset examples, trainer state, mask API,
and the implemented resume schema.

- [x] Implement bounded batch collation with explicit padding/packing policy and
  correct document boundaries; do not drop remainders or silently truncate.
- [x] Count real targets for loss/metrics, define optimizer-step and resume cursor
  semantics, and preserve batch-size-one behavior.
- [x] Test variable lengths, final partial batches, masks, and train/resume behavior.
  Coordinate any T07 schema/state extensions; do not silently invalidate snapshots.

### T14 — Streaming/token-cache design and implementation

Depends on T03 and T06. Read: dataset pipeline, fingerprints, and trainer iteration.

- [x] Agree iterator/cache interfaces and memory bounds before introducing a second
  dataset representation. Preserve deterministic document identity and chunk pairs.
- [x] Include tokenizer identity, preprocessing configuration, and source identity
  in cache invalidation; reject incomplete caches and avoid overwriting user data.
- [x] Test streaming/eager equivalence, stale-cache invalidation, corruption, and
  resume cursor implications. Document actual memory limits, not assumed scalability.

### T15 — Explicit additional dataset formats

Depends on T03. Read: document abstraction and selection validation.

- [x] Select one format per assignment (for example JSONL), define text-field schema,
  document boundaries, missing-field behavior, and actionable file/record errors.
- [x] Preserve `.txt` behavior and path safety. Test malformed records, empty data,
  UTF-8 handling, split isolation, and target-pair coverage.

### T16 — Cross-case checkpoint reservation safety

Read: `training/src/checkpoint.rs`, especially scan/reserve logic and tests.
Do not overlap T06's checkpoint write set.

- [x] Reproduce mixed-case concurrent writers on a case-sensitive filesystem; agree
  whether canonicalized names or shared locking establishes one numbering series.
- [x] Keep existing mixed-case checkpoints discoverable; preserve max-present-suffix,
  occupied-file handling, overflow errors, and incomplete-directory reservations.
- [x] Test concurrent case variants and same-name writers without overwriting data;
  document legacy naming behavior and any platform-specific constraints.

### T17 — Explicit CLI roots and end-to-end tests

Read: `training/src/bin/main.rs`, `project_root`, and safe dataset/checkpoint paths.
Suggested writes: CLI plus `training/tests/cli.rs` (*new*); coordinate CLI ownership.

- [x] Add explicit dataset/weights-root options while keeping compile-time defaults;
  preserve absolute tokenizer support and safe relative dataset selections.
- [x] Test full tiny train/save/generate cycles with temporary roots, repeated saves,
  different working directories, argument/help behavior, and nonzero failure exits.
- [x] Tests must not leave checkpoints in repository `weights/` or mutate fixtures.

### T18 — Reproducible build and documentation navigation

Coordinator-owned unless the exact shared files are delegated.

- [x] Decide and document workspace `Cargo.lock` tracking for reproducible CLI builds;
  adjust ignore rules only for that decision, not unrelated artifacts.
- [x] Add root navigation to crate READMEs, training/status docs, and this task board.
  Clarify crate-local versus workspace-wide capability claims.
- [x] Explain the empty crate-local tokenizer JSON without deleting user data;
  distinguish verified toolchain version from a declared MSRV.
- [x] Verify all added paths/commands; do not advertise unimplemented backlog items
  as available or repeat dated test counts as current validation.

### T19 — Stateful optimization controls and numerical checks

Depends on T07. Read: trainer update flow, optimizer configuration, resume schema.

- [x] Break work into bounded assignments (for example gradient clipping first,
  then warmup/scheduling, then accumulation); agree interaction with resume state.
- [x] Persist scheduler/accumulation state if needed; define update versus microstep
  accounting. Test interrupted/uninterrupted equivalence for enabled controls.
- [x] Detect/report numerical failures including final-update validity, with no
  silently saved successful checkpoint after a known failure.
- [x] Keep defaults compatible and benchmark only tiny authorized workloads.

## Phased work: only explicitly assigned phases may start

The following implementation plan was requested on 2026-09-27 and includes the
user's subsequent CPU-first T20 scope revision. Parent task IDs are retained.
Phase labels such as T21.1 are bounded planning references; readiness is explicit
in the parent board and phase status. An explicit phase or readiness
review can assign a bounded phase using these contracts; do not implement an
entire parent task merely because its first phase is eligible. Keep the parent
unfinished until its agreed acceptance criteria are verified.

Recommended sequence:

1. T21.1/T22.1 and the assigned CPU phases T20.C1/C2/C3 are complete.
   Future implementation needs an explicit assignment from the phases below.
2. T21.2/T21.3 and T22.2–T22.4 remain unassigned; do not start those tracks as
   incidental CPU work. In particular, KV caching remains T21.2's separate scope.
3. Intel Arc A770 GPU work follows the updated T20.1–T20.5 plan below.
   A770 backend qualification passed; implementation and final validation are assigned.
4. Coordinator integrates CLI, exports, manifests/lockfile, docs and task status
   serially. At most three workers plus the coordinator run concurrently here.

### T20 — CPU optimization and multithreading; Arc A770 GPU training

The original CPU-first assignment is complete. The user authorized workers and
GPU implementation on 2026-09-28. T07/T13 are complete. The selected optional
backend is Burn 0.18 WGPU/Vulkan with SPIR-V, using shared backend-generic
orchestration, explicit devices and a separate GPU resume contract. No Burn
upgrade or remote compute is part of this assignment.

**Historical source/hardware evidence (2026-09-27):** the then-current Windows host reports
AMD Ryzen AI 9 365, 10 physical cores and 20 logical processors. Its detected
adapters are an NVIDIA RTX 5070 Laptop GPU and AMD Radeon 880M; an Arc A770 is not
present in this host inventory. These observations describe the benchmark host,
not the final deployment hardware. A770 OS, driver, memory variant and test access
will be recorded in T20.1; do not infer them or substitute the current NVIDIA GPU.

The locked Burn 0.18 NdArray `std` feature already enables Rayon and
`matrixmultiply/threading`; this is not a single-thread-only stack. Burn's batched
matmul uses Rayon around matrix multiplication, and matrixmultiply 0.3.11 reads
`MATMUL_NUM_THREADS` once at pool initialization and clamps it to 1..=4. Multiple
thread pools can compete; actual concurrency and gains must be measured. The
current resolved feature tree does not enable Burn NdArray's optional `simd`
feature. This is a candidate to investigate, not proof that all existing kernels
lack vectorization or that enabling it will improve the workload.

**Scope and compatibility:** retain NdArray f32 CPU as the default and preserve
existing public CPU wrappers, fixed-seed behavior, causal/padding contracts and
transactional numerical checks. Do not parallelize optimizer steps: model/Adam
updates must remain ordered, with one persistent optimizer. Keep corpus/sampling
order deterministic and bounded; no Hogwild/shared mutable model training.

**T20.C1 — CPU performance baseline and thread audit**

Phase status: **done** — coordinator reviewed and measured; see C1 acceptance below.

- [x] Add a reproducible, bounded release-mode benchmark harness using synthetic
  in-memory token data and tiny CPU models. Measure full training updates and
  full-prefix generation separately; isolate setup/lazy initialization, data
  preparation, forward/backward, numerical validation, optimizer work and saving
  where instrumentation permits. Do not compare debug builds or timing dominated
  by compilation, initialization, console logging or checkpoint I/O.
- [x] Measure single-example and minibatch workloads with identical model/data
  settings. Start with batch sizes 1/4 and context lengths 32/128; use a capped
  small model and record its dimensions/vocabulary. Warm up and report repeated
  samples, median/range, real targets/sec or generated tokens/sec, elapsed time,
  available process-memory measurements and the exact source/build/feature profile.
- [x] Audit Rayon, matrixmultiply and tokenizer pools before adding threads.
  Run thread configurations in fresh processes, setting child-process environment
  before pools initialize. Compare Rayon counts 1/2/4/8/10/20 and matmul counts
  1/2/4 in a bounded staged sweep on this host; start with one pool at a time and
  expand only promising combinations. Report requested versus known effective
  limits; high CPU utilization is not itself a success criterion.
- [x] Cap each timed case at 10 seconds and total timed work at 5 minutes for
  the initial sweep; stop/record overruns rather than launching larger training.
  Build time is separate. Do not change user environment variables globally,
  power plans, affinity/security settings or user datasets/checkpoints.
- [x] Return measured bottlenecks, a reproducible baseline and a ranked T20.C2/C3
  work list. No speedup claim, thread default change or new dependency is accepted
  solely from the existence of a parallel/SIMD feature.

**T20.C2 — controlled CPU threading** (after T20.C1)

Phase status: **done** — integrated tests, strict Clippy, CLI help and Windows
interruption passed; native Unix execution remains unverified.

- [x] Based on the audit, expose explicit CPU compute-thread controls with a
  working one-thread mode and documented precedence over inherited environment
  settings. Keep Rayon and matmul controls distinguishable; do not promise that
  one flag caps every process thread or imply matmul can use all 20 logical cores.
  Leave existing defaults intact until the benchmark supports a different default.
- [x] Initialize pools before tensor/tokenizer work. Prefer a verified scoped
  runtime where possible; otherwise use a safe CLI startup/child-process boundary.
  Never mutate process environment from a library after threads start. Library
  calls must reject or document attempts to change already-initialized global pools.
- [x] Avoid nested oversubscription across backend, preprocessing and tokenizer
  pools. Keep output/document/example order stable. Add bounded preprocessing or
  prefetch workers only if C1 identifies a bottleneck, with error propagation,
  cancellation, bounded queues and no concurrent parameter updates.
- [x] Test invalid thread counts, startup/override behavior, one/multiple thread
  parity, causality/masks and interrupted resume with clipping/warmup/sampling.
  Decide and test whether thread/kernel settings affect exact continuation; record
  or reject incompatible execution settings before changing the resume contract.
  Do not silently declare cross-thread-count floating-point results identical.

**T20.C3 — measured CPU hot-path improvements** (after T20.C1; coordinate C2)

Phase status: **done** - measured investigation accepted. Typed scans showed no
reproducible total gain and were removed; SIMD changed reciprocal results and
remains disabled. C2 thread controls are the retained performance choice.

- [x] Profile allocation/copying, tensor layout/matmul, host reads, gradient checks,
  model/Adam candidate validation and cached-source reads. Optimize only measured
  hotspots; removing failure detection or rollback semantics is not an optimization.
- [x] Evaluate compatible existing SIMD/kernel options in an isolated feature
  experiment before making a production dependency/default change. Preserve a
  portable build; a local native-CPU benchmark is not a distributable binary policy.
  Alternative BLAS backends require a separate compatibility/build decision, not
  an incidental Burn upgrade or unchecked dependency replacement.
- [x] Compare against C1 with identical workloads, thread settings and validation
  enabled. Report speed/memory tradeoffs and variability; keep changes with a
  reproducible benefit and no unjustified regression on the smaller workload.
  Test numerical parity and exact supported resume behavior after each change.

CPU ownership proposal: a measurement worker owns a dedicated benchmark/example
and fixture module, without editing the trainer initially. Once C1 reports,
assign disjoint runtime-threading and measured hot-path files. Coordinator owns
CLI/exports, feature/lockfile changes, resume compatibility, metrics and docs.
Do not start C2/C3 production edits in parallel before measurement/API agreement.

**GPU training plan — updated 2026-09-28**

Goal: use the user's Intel Arc A770 for Omega training and compare it with a
properly optimized CPU baseline on the i7-13700K. The user requested this task
plan, then explicitly authorized workers and implementation. Keep T20's completed
CPU criteria and historical results intact. Current implementation evidence and
remaining profiling work are recorded in the handoff below.

Verified local evidence from read-only inspection on 2026-09-28:

- Host: Linux x86_64, kernel `7.1.8-arch1-3`, Intel Core i7-13700K, 24 logical
  CPUs. This is a different host from the earlier Windows/Ryzen CPU benchmarks.
- `vulkaninfo --summary` reports an Intel Arc A770 (device `0x56a0`), Mesa
  `26.1.8-arch1.1`, Vulkan device API `1.4.354`. Its device-local heap is
  approximately 15.91 GiB, identifying the 16 GB variant. Vulkan enumeration
  establishes local device access, not successful Burn training or performance.
- The observed active command was `debug/main train --name macro-invest
  --dataset investment/macro --tokenizer qwen-3-1.7-base.json --epochs 10`.
  It had 29 OS threads but averaged roughly one core of CPU usage; no explicit
  Rayon/matmul environment overrides were present. Thread count is not evidence
  of parallel execution of the dominant kernels. Do not stop/restart this job
  or alter its files as part of implementing or benchmarking GPU support.
- Existing `weights/macro_invest-1` and `macro_invest-2` are separately named
  historical Windows schema-2 snapshots, not verified snapshots of that active
  Linux `macro-invest` process. Their metadata records vocabulary 151665, context
  64, width 32, four heads, two layers, FF width 128, batch size 1, 1559 examples
  and 99719 targets per epoch. Recheck corpus identity before reusing these counts.
- The Qwen JSON supplies tokenizer IDs only; Omega trains its own model from
  scratch. Full logits at batch 1/context 64/vocabulary 151665 occupy about
  37 MiB in f32, excluding gradients, other activations and optimizer state.
  The earlier vocabulary-64 benchmarks do not establish performance here.

**Scope and compatibility:** add exactly one optional GPU backend, initially
Burn 0.18 WGPU using Vulkan, subject to T20.1 verification. Retain CPU defaults
and f32; no implicit CPU fallback, Burn upgrade, mixed precision, distributed
training, pretrained Qwen import or tokenizer replacement. The model is already
backend-generic; generalize orchestration through one training loop and preserve
existing CPU APIs. GPU weight transfer is distinct from exact training resume.

| Phase | Deliverable | Planning state | Depends on |
|---|---|---|---|
| T20.1 | A770/Burn feasibility and execution contract | done; actual A770 probe passed | T07, T13 (done) |
| T20.2 | Backend-generic trainer and explicit device selection | implemented; detailed cost profiling open | T20.1 accepted |
| T20.3 | GPU checkpoints, continuation and correctness | done; A770 continuation and failure tests passed | T20.2; schema agreement before persistence edits |
| T20.4 | Representative release CPU/GPU benchmark and tuning | partial; 13-case sweep complete, detailed profiling open | T20.3 correctness; CPU baseline may be measured earlier after assignment |
| T20.5 | Operational CLI, regression and documentation handoff | done for operational backend; profiling follow-up recorded | T20.3, T20.4 |

**T20.1 — hardware and API feasibility gate**

Read: crate manifests/lockfile, NN model/mask/loss APIs, trainer/optimization,
resume records and the exact locked Burn 0.18 backend APIs. Suggested artifact:
a separately bounded GPU probe; no production backend switch in this phase.

- [x] Record the current OS, CPU, A770 memory variant, Vulkan driver/runtime and
  local device enumeration above. Revalidate them at implementation time.
- [x] Verify the exact Burn 0.18 WGPU/Vulkan feature set, adapter enumeration and
  explicit device selection. Distinguish the A770 from integrated/software
  adapters; identify required device limits for the large vocabulary buffers.
  Record the resolved dependency/features and adapter information. If a Burn
  upgrade is required, report the blocker instead of upgrading incidentally.
- [x] Compile and run a tiny f32 probe on the A770 covering forward, causal and
  padding masks, cross entropy, backward, parameter-gradient extraction, Adam
  update, synchronization, and model/optimizer record round trips. Verify actual
  GPU execution; successful Vulkan enumeration alone does not pass this gate.
- [x] Establish record/parameter-ID compatibility and host/device transfer APIs.
  Declare numerical tolerances before CPU/GPU comparisons, starting from shared
  materialized parameters rather than independently seeded initializations.
- [x] Agree device-index semantics, adapter identity, memory measurement sources,
  backend feature name and generic public APIs before T20.2. Keep compilation
  separate from timing; bound execution with an external deadline and temporary
  output roots. No user corpus training or checkpoint writes in this probe.

**T20.2 — generic orchestration and explicit device selection**

Suggested areas: NN helpers/generation/loss; training trainer, optimization,
evaluation, batching and CLI. Agree generic type/default parameters before edits.

- [x] Introduce backend/device-aware internals while retaining existing CPU
  constructors, wrappers and default type behavior. Preserve `ExampleSource`
  eager/cache inputs, events and one persistent optimizer in the shared loop.
  Pass devices explicitly instead of constructing defaults inside helpers.
- [x] Add an opt-in Cargo GPU feature and proposed `--backend cpu|vulkan` plus
  `--device INDEX` on train/resume/evaluate/generate. Final names follow T20.1.
  Provide read-only device listing; report selected adapter/backend in startup
  output. Invalid devices, missing features and unavailable GPUs fail clearly.
  CPU-only builds must not initialize Vulkan; never silently use CPU for a GPU run.
- [x] Define interaction with existing CPU thread flags: they cannot select GPU
  compute concurrency. Document any supported host-side effect or reject irrelevant
  combinations. Include backend/device information in structured run metadata.
- [x] Stage GPU inference and CPU-created weight loading before full training.
  Verify fixed/weighted/shuffled ordering, right-padded batches, real-target loss,
  clipping/warmup and validation through the same orchestration code.
- [x] Preserve transactional updates: validate loss, gradients, candidate model
  and Adam state before committing counters/state. Backend/device failure must
  not emit successful completion or publish an invalid checkpoint. GPU candidate
  rollback and injected checkpoint publication failures are tested.
- [ ] Measure synchronization and candidate-copy costs separately; checks remain
  enabled in the whole-update benchmark. Detailed attribution is still open.

**T20.3 — checkpoint policy, continuation and correctness**

Suggested areas: checkpoint, resume, catalog, run control and dedicated regression
tests. Agree schema and compatibility policy before introducing new writers.

- [x] Define a versioned backend/precision/runtime profile for new training saves,
  including relevant adapter/driver/kernel/build identity. Preserve schema-1/2/3
  CPU compatibility through explicit tested migration/load rules. Account for GPU
  dependencies changing the lockfile hash without broadly allowlisting old builds.
- [x] Test CPU-to-GPU and GPU-to-CPU inference weight transfer, tokenizer identity,
  architecture checks and output tolerances. Preserve marker-last publication,
  numbered reservations, hashes, catalog/latest behavior and no-fallback errors.
- [x] Reject CPU-to-GPU and Windows-to-Linux exact resume under existing contracts.
  The historical `macro_invest-*` snapshots must not be presented as directly
  resumable GPU runs. Do not edit checkpoint metadata to bypass validation.
  Any weights-only training restart needs a separate explicit design with fresh
  optimizer/counters and provenance; it is not implicit in inference transfer.
- [x] Implement and test same-backend GPU save/load continuation: model, Adam
  moments/parameter IDs/counters, sampler, cursor, partial loss, batching and
  optimization options. Compare interrupted/uninterrupted execution across
  mid-epoch and epoch boundaries. State measured reproducibility precisely;
  tolerance agreement alone is not bitwise exact resume. Unsupported GPU resume
  must fail explicitly and leaves this criterion unfinished.
- [x] Exercise periodic/final-save deduplication and real Linux SIGINT/SIGTERM
  delivery in an isolated subprocess. Finish/synchronize the in-flight committed
  update before saving. Verify subsequent continuation, nonzero interrupted exit,
  no orphan process, and incomplete-save rejection on injected failures.
- [x] Compare CPU/GPU logits, loss, gradients and optimizer updates using shared
  parameters and declared tolerances. Retain causality/padding tests and add
  nonfinite/rollback regressions, incompatible-device/runtime rejection and
  generated-token tests that account for near-tie numerical differences.

**T20.4 — representative release CPU/GPU benchmark and tuning**

The performance target is the user's large-tokenizer workload, not the old tiny
vocabulary benchmark. A two-day estimate is an observation to investigate, not a
baseline measurement or a promised acceleration ratio.

- [x] Establish an optimized `--release --locked` CPU baseline on the i7-13700K.
  Use fresh processes to compare bounded Rayon/matmul combinations (initially
  Rayon 1/4/8/16/24 with matmul 1, then promising pairs with matmul 2/4).
  Record effective configuration and competing load; do not change global pools,
  power settings or the active user's training process. Avoid timing concurrent
  benchmark processes; flag existing background training as a confounder.
- [x] Use synthetic token IDs with the recorded model dimensions and vocabulary
  151665, context 64, batches 1/2/4, identical starting parameters and fixed update
  counts. Preserve the small fixture case for regression comparison. Account for
  real targets per second and altered optimizer-step counts when comparing batches.
- [x] Compare CPU and A770 release runs including mandatory numerical checks.
  Synchronize GPU work at timing boundaries; separate compilation, first-use
  kernel setup/autotuning, warmup, source preparation, updates and checkpoint I/O.
  Report repeated medians/ranges, wall time, host RSS and measured GPU allocation
  or heap usage with the method and limitations, not just GPU utilization.
- [x] Bound the initial sweep to five minutes of execution with a 30-second child
  deadline; record overruns rather than increasing workload automatically. Build
  time is separate. Start with batch 1 and check memory/device limits before larger
  cases; use temporary synthetic data and checkpoints only.
- [ ] Profile the large vocabulary projection/loss, gradient and Adam validation,
  transfers and allocations. Retain only measured improvements with correctness
  evidence. Do not change tokenizer IDs, weaken checks or assume linear scaling.
  Any later real-corpus segment must be bounded, separately assigned, read-only
  on sources, and write only a new isolated output root.
- [x] Publish exact commands, build/features, hardware/driver identity, results
  and qualified throughput estimates. Recommend CPU/GPU and batch/thread settings
  from measurements; do not claim a speedup before results or extrapolate kernel
  timing alone into a full-run completion time.

**T20.5 — operational CLI, regression and documentation handoff**

- [x] Run formatting, locked workspace build/tests and strict all-target Clippy
  for CPU-only and selected GPU-feature builds. Run hardware tests on the actual
  A770; distinguish skipped/unavailable GPU tests from passing execution. Recheck
  CPU compatibility after generic refactoring and lockfile changes.
- [x] Add temporary-root CLI workflows covering GPU train/save/generate/evaluate,
  supported resume, help/errors, adapter selection, missing GPU feature/device,
  cooperative interruption and CPU/GPU inference transfer. Keep all existing
  regression coverage; never test against user checkpoint outputs.
- [x] Update root/crate READMEs, training/status/performance docs and task handoff
  with release build commands, Cargo feature/device flags, measured settings,
  memory limits and resume/migration policy. Address the existing shared `main`
  executable collision in reproducible commands.
- [x] Update `Scripts/training_run.sh`, its README and mocked-Cargo tests to use
  release builds and accept optional backend/device controls while preserving
  its existing four positional arguments, repeated dataset handling and failure
  propagation. CPU remains the default; prevent shared-target build collisions
  from selecting the wrong executable. Do not launch a full user run in tests.
- [x] Report supported fresh GPU training and continuation separately. Do not mark
  the full GPU plan complete while required GPU continuation or hardware checks
  remain unverified. No model downloads, remote compute, checkpoint deletion or
  automatic conversion/restart of the current training job.

Integration ownership: agree bounded file ownership before implementation.
Coordinator owns manifests/lockfile, module exports, CLI wiring, schema decisions,
shared scripts/docs and task status unless explicitly delegated. Serialize trainer
and resume edits; coordinate NN changes with T21.2 and storage changes with T22.
The subsequent user assignment authorizes the GPU workers recorded below.
T20.C1/C2/C3 remain complete; later T21/T22 phases remain unassigned.

### T21 — Sampling and incremental generation

After T02, assign sampling, KV cache, and streaming as separate bounded tasks.
Sampling must preserve greedy defaults and validate parameters; KV cache tests
must compare cached/uncached outputs and position/context behavior. Pretrained
import, chat templates, quantization, and architecture changes are separate designs.

**Scope and compatibility:** CPU f32, one prompt per request, unchanged tokenizer
IDs/decoder and learned absolute positions. Retain the existing `generate` API as
a greedy wrapper, including prompt-plus-output, generated-EOS inclusion, and
up-front rejection when prompt plus requested tokens exceeds context length.
No sliding window, batching, generation-state persistence or checkpoint changes.

**T21.1 — validated deterministic token selection**

Phase status: **done** — accepted implementation and integrated validation below.

- [x] Add a separate inference generation module/options API with greedy default
  and explicit sampling mode. Sampling uses finite temperature > 0, optional
  top-k in 1..=vocab size and top-p in (0, 1], plus a request-local seed (default
  42). Do not change the training sampler or consume Burn's global RNG.
- [x] Define selection as stable temperature softmax, descending probability
  ordering with token-ID tie breaking, top-k then smallest nonempty top-p prefix,
  renormalization and one draw. Reject nonfinite logits and invalid options;
  document the PRNG/version and greedy tie behavior. Greedy mode rejects supplied
  sampling-only flags instead of silently ignoring them.
- [x] Add fixed-vector tests for filtering, ties, normalization, extreme finite
  logits, invalid parameters and seeded repeatability. Compare default generation
  against the current implementation; cover EOS, zero budget, overflow, invalid
  IDs and context limits. CLI/library failures must be actionable and nonzero.

**T21.2 — bounded KV cache and incremental forward** (after T21.1)

- [ ] Add request-local per-layer key/value caches, separate from model parameters
  and serialized records. Prefill once, then process one new token at its absolute
  position. Cached decoding must use the same selection and stopping logic.
- [ ] Bind cache lifetime to its immutable model/request; reject incompatible
  shapes/devices and invalid positions. Preserve causal attention with the correct
  offset. No cache reuse across models, prompts or parameter updates.
- [ ] Check cache-size arithmetic before allocation, cap positions by context,
  and expose a cache byte budget (proposed default 64 MiB). Count both K/V buffers;
  document transient/backend allocations outside this cap. Exceeding the budget
  returns an error rather than truncating or silently falling back.
- [ ] Compare every position's cached/uncached logits and greedy tokens on tiny
  models, varied prefill lengths and context edges; test reset/isolation, EOS,
  byte-budget overflow and unchanged full-forward causality. Establish and record
  justified CPU f32 tolerances before accepting differences; include shared-logit
  sampler tests to separate sampling behavior from near-tie numerical effects.
- [ ] Make caching opt-in initially (`--kv-cache`); measure bounded cached versus
  uncached token latency and retained cache bytes without claiming a speedup in
  advance. Do not change the default until parity and measurements are reviewed.

**T21.3 — incremental output** (after T21.2)

- [ ] Expose a fallible iterator/callback over generated token IDs, using the same
  generation engine for collecting and streaming APIs, with stop/cancellation
  checked between tokens. No background inference worker or server is required.
- [ ] Add opt-in `--stream` JSONL: one token event with index/ID followed by a
  successful completion event containing final decoded text and stop reason.
  Keep default text output unchanged. Do not decode individual token IDs as
  independently valid UTF-8 text; final decoding uses the saved full pipeline.
- [ ] Test streamed/collected token equivalence in both greedy and seeded modes,
  cached and uncached paths, generated EOS, zero budget, UTF-8 byte BPE, cancellation
  and failing output writers. Partial output must not end with a success event
  after an error; CLI returns nonzero on interrupted/failed generation.

Ownership proposal: one worker owns the new NN generation module/tests for T21.1;
after release, assign NN `src/model.rs` and dedicated cache tests for T21.2.
T21.3 reuses that generation module under exclusive ownership. Coordinator wires
NN exports, both affected CLIs, help/regression tests and docs. Exact API names
are agreed before assignment; no separate implementations in the two CLIs.

### T22 — Checkpoint lifecycle and stronger storage guarantees

After T06/T16, agree discovery/`latest`, retention, checksum, durability, and load
resource-limit requirements. Never delete user checkpoints without explicit
policy/authorization. Decide whether deleted historical numbers must remain
reserved; do not silently replace the current highest-present-suffix contract.

**Scope and numbering decision:** preserve maximum-present-suffix-plus-one,
including files, mixed-case names and incomplete reservations. No historical
high-water registry or tombstones in this plan. If a user externally removes the
highest entries, those numbers may be reused under the existing contract; document
this explicitly. Retention defaults to keeping everything and this implementation
plan adds previews only. Any deletion command needs a separate authorized scope.

**T22.1 — read-only discovery and explicit latest selection**

Phase status: **done** — accepted with the platform limits recorded below.

- [x] Add a bounded inventory under an explicit weights root/run name. Report
  complete inference, resumable, legacy/unverified, incomplete and invalid entries
  with reasons. Do not initialize tensors, modify files or clean reservations.
- [x] Define `latest` as the highest numeric suffix eligible for the requested
  operation within one run; ignore modification time. Report skipped incomplete
  reservations/inference-only entries and fail on ambiguous equal-number aliases
  or no eligible checkpoint. A completed candidate with corrupt metadata or
  payloads must produce an error, not select an older apparently healthy save.
  Resume must still apply all source/runtime checks after selection; never silently
  fall back to an older checkpoint because the selected one is incompatible.
- [x] Add list/latest CLI workflows with human-readable and JSON output, keeping
  explicit checkpoint-path loading unchanged. Define inspection depth/integrity
  status so header inspection never claims payload verification.
- [x] Test mixed case, leading zeros, occupied files, gaps, incomplete saves,
  corrupt metadata, overflow, empty roots, symlink/root escape rejection and a
  concurrent save whose COMPLETE marker appears during discovery. Revalidate
  selected entries on load; discovery is not an atomic filesystem snapshot.

**T22.2 — payload integrity and bounded loading** (after T22.1)

- [ ] Introduce configurable checked limits for directory entries, metadata and
  tokenizer bytes, model/optimizer bytes, total artifact bytes, dimensions and
  parameter counts. Record default values and override semantics before coding;
  reject limits before unbounded reads, record decoding or tensor allocation.
- [ ] Review recorder allocation behavior: encoded file size alone does not bound
  declared tensor shapes or decompression/decoder allocations. Add structural
  checks/bounded decoding where supported and report remaining resource limits
  precisely; do not describe a file-size cap as protection from arbitrary inputs.
- [ ] Extend inference integrity to all required payloads using a versioned hash
  manifest published before COMPLETE; coordinate with existing resume hashes.
  Require fixed safe artifact names, sizes and hashes without duplicate entries;
  bind the manifest digest from the versioned completion marker without a hash
  cycle. Reuse bounded reads and incremental hashing for inference and resume.
  New schema loads reject missing/tampered artifacts; legacy explicit loads remain
  available with clearly reported missing integrity guarantees. Never modify old
  checkpoints in place or claim hashes authenticate their author.
- [ ] Test every required artifact, malformed lengths/shapes, oversized/sparse
  files, limit boundaries, unknown schemas and old inference/resume compatibility.
  Reuse one validation path across discovery, inference and resume where applicable.

**T22.3 — explicit durable publication** (after T22.2)

- [ ] Preserve current logical marker-last mode; add opt-in durable saving only
  for platforms/filesystems whose required primitives can be implemented and
  tested. Sync payloads/metadata before publishing COMPLETE, then sync the marker
  and checkpoint/weights-root directories as supported. Unsupported guarantees
  must return an explicit error.
- [ ] Specify the failure state after every stage, including failure after marker
  publication: a save may exist despite an error, and retries reserve a new number.
  Keep failed reservations and existing successful checkpoints untouched.
- [ ] Test injected write/flush/sync/marker failures and concurrent reservations
  in temporary roots. Document local filesystem guarantees separately from cloud
  synchronization/network storage; fault injection alone does not prove power-loss
  survival. Never advertise OneDrive synchronization as durability confirmation.

**T22.4 — retention preview only** (after T22.1 and T22.2)

- [ ] Add a deterministic dry-run plan with explicit run and positive keep-last-N
  policy. Protect the newest complete checkpoint, newest resumable checkpoint,
  highest occupied suffix, and all incomplete/invalid/unverified entries; report
  retained/candidate paths, reasons and validated byte totals. No automatic policy.
  Support explicitly protected names and report when protections prevent the
  requested keep count; never relax protections to satisfy the target count.
- [ ] Reject paths outside the selected root, aliases and ambiguous inventories;
  test mixed legacy/new entries, no candidates and filesystem changes. Label the
  output advisory, never an authorization token for a future deletion operation.
- [ ] Ensure the command cannot delete/rename files. Future destructive retention
  must separately define confirmation, revalidation against concurrent changes
  and whether numbering semantics need migration; no such change is implicit here.

Ownership proposal: one worker owns a new training inventory module and tests
for T22.1; coordinator owns exports/CLI. Assign `src/checkpoint.rs`, `src/resume.rs`
and focused integrity/durability tests exclusively for T22.2/T22.3, serializing
with T20.3. The retention planner can follow in its own module after inventory
contracts stabilize; all tests use temporary roots, never repository weights.

### Integration and completion for T20–T22

- Before each phase, record exact write ownership, agreed public API and any
  schema/legacy policy in this board. Workers return changed paths, actual test
  results and limitations; only the coordinator marks acceptance.
- Run affected-package tests/Clippy, then from `Code/Rust` run
  `cargo fmt --all --check`, `cargo test --workspace --locked --jobs 1`,
  `cargo build --workspace --locked --jobs 1`, and
  `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`.
  Check affected CLI help, parsing, errors and temporary-root end-to-end workflows.
- T20 additionally requires the selected feature build and real hardware checks;
  T22 durable mode requires its platform checks. An unavailable environment is
  recorded as unverified, not accepted based on CPU or mocked tests alone.
- Update crate READMEs and both Docs guides with defaults, compatibility, limits
  and measured results. Preserve old accepted task criteria and existing data.
  No model downloads, remote costs, commits, deployment or checkpoint deletion.

Distributed training, serving APIs, instruction/preference tuning, and containers
remain unassigned future directions, not implied additions to these tasks.

## Assignment and handoff log

### PDF text preparation — user-requested utility

- PDF01 / coordinator / done. Independent of Rust implementation tasks.
  Scope: `Code/Python/` only, plus this handoff entry. Add folder-to-folder PDF
  text extraction, dependency/usage documentation, and focused tests. Preserve
  source PDFs and existing outputs; report extraction failures and OCR limits.
  Delivered: `pdf_to_text.py`, requirements, README, and temporary-file tests.
  Validation: all 7 Python unit tests and CLI help passed using Windows `py`
  (Python 3.14.2). `python3` was unavailable. No OCR or recursive scan is included;
  existing outputs and source PDFs are preserved.

### 2026-09-27 — first implementation wave

- T16 / coordinator / active. No prerequisites; T06 remains unapproved/blocked.
  Owns training `src/checkpoint.rs`, shared with coordinator's T02 integration.
  Canonicalize new run directories to ASCII lowercase while scanning all legacy
  casing. Validate concurrent same-name/case-variant reservations on Windows and
  an existing WSL case-sensitive filesystem without changing filesystem settings.

- T18 / coordinator / active. No prerequisites. Owns root `.gitignore`,
  `readme.md`, crate READMEs, `Docs/`, and workspace lockfile policy. Keep the
  existing resolved lockfile and allow tracking; validate locked workspace builds
  and all added documentation links. No dependency upgrades or MSRV declaration.

- T01 / worker-tokenizer / active. No prerequisites. Owns tokenizer `src/lib.rs`,
  `src/bin/main.rs`, `tests/`, and NN `tests/encoding.rs`; preserve fixture IDs and
  loaded pipelines. Coordinator owns manifests/docs. Validate tokenizer tests and
  Clippy plus NN encoding tests. Request coordination for NN implementation fixes.
- T02 / worker-model / active. No prerequisites. Owns NN `src/model.rs` and
  `src/lib.rs`; add architecture validation without changing valid helper APIs or
  legacy model records. Coordinator integrates checkpoint callers and docs.
  Validate NN tests/Clippy, including causality and next-token alignment.
- T03 / worker-dataset / active. No prerequisites. Owns training `src/dataset.rs`.
  Preserve `TrainingSet` and the all-training wrapper; add document identities and
  deterministic split-before-chunking API. Coordinator owns exports/docs/CLI.
  Validate dataset tests and training Clippy; no evaluation or CLI split scope.
- Coordinator owns integration, this board, manifests, lockfile, and shared docs.
  Existing files were snapshotted outside the repository before edits because this
  checkout has no Git metadata; no branches or commits will be created.

```text
Task/subtask ID:
Owner/session:
Status and date:
Dependencies verified:
Agreed API and compatibility policy:
Allowed writes:
Shared integration owner:
Required validation:
Handoff (paths, behavior, commands/results, remaining risks):
```

For a task that becomes blocked, record the precise missing decision, dependency,
or failing validation and the next bounded action. Do not mark it done merely
because a sub-agent returned a summary.

### 2026-09-27 — second implementation wave

- T02/T03 handed back their source ownership; coordinator reviewed APIs and
  integrated checkpoint validation/dataset exports. T03's 12 dataset tests and
  T02/T16's 10 checkpoint tests pass, including checkpoint tests on WSL-backed
  case-sensitive storage. Final workspace validation/documentation remains.
- T05 / worker-model / active. No prerequisites. Exclusively owns training
  `src/trainer.rs` (new) and `src/lib.rs` including exports/wrapper. Preserve
  `train` and `TrainingSet`; define persistent in-memory Adam/model/counter/event
  state, with no disk-resume claim. Validate split-call equivalence with shared
  initialization (avoid RNG races), invalid inputs, events, package tests/Clippy.
- T17 / worker-dataset / active. No prerequisites. Exclusively owns training
  `src/bin/main.rs` and `tests/cli.rs` (new). Add explicit roots preserving
  defaults and safe selections; tiny subprocess train/save/generate tests use
  temporary roots only. Validate argument/help/error paths and repeated saves.
  Coordinator retains all other shared files, manifests, docs, and board.

- T01 handoff and coordinator package tests pass (9 tokenizer tests, 21 NN tests).
  No fixture or dependency changes. CLI subprocess harness avoids shared binary
  replacement. Documented dependency limitation: truncation can omit overflow.
- T17 handed back CLI ownership; package tests/help/Clippy passed. Await final
  integration checks. An early workspace check during edits found formatting and
  include-order lint issues, now fixed; parallel linking of shared `main.exe`
  failed on Windows, so final workspace build/tests use `--jobs 1`.
- T03 review found possible Windows case-alias selection duplication/leakage.
  Reassigned only `src/dataset.rs` to worker-dataset to reproduce, canonicalize
  validated discovered paths, and regression-test before final acceptance.

### 2026-09-27 — accepted integration and resumption notes

All seven originally `ready` tasks are now `done`; all worker assignments have
been collected and released. No other task was promoted or implemented.

| Task | Accepted changes and evidence |
|---|---|
| T01 | Tokenizer `src/bin/main.rs`, `tests/cli.rs`, `tests/pipeline.rs`, `tests/support/mod.rs`; NN `tests/encoding.rs`. Saved pipeline, lookup/error, sparse vocabulary, padding/overflow and subprocess CLI regressions pass; fixtures unchanged. |
| T02 | NN `src/model.rs`, `src/lib.rs`; training `src/checkpoint.rs`. Architecture/accessor validation, generation and pre-reservation save mismatch checks; target alignment and tensor-contract tests pass. Existing public helpers and model record fields preserved. |
| T03 | Training `src/dataset.rs`, exports in `src/lib.rs`. Whole-document seeded partitions/provenance and all-training wrapper pass 13 dataset tests. Case-alias leakage reproduced and fixed by canonicalization after symlink validation; sensitive and insensitive filesystem branches passed. |
| T05 | Training `src/trainer.rs`, `src/lib.rs`. Session owns model/Adam/fixed data/config/counters/cursor and partial epoch loss; post-update events and compatible wrapper. Five new tests prove state continuity, weighted events, observer recovery, invalid inputs and nonfinite-loss no-commit behavior. |
| T16 | Training `src/checkpoint.rs`. Canonical lowercase names preserve legacy scanning and suffix rules. Old race reproduced: 32 reservations yielded 16 suffixes. All 10 checkpoint tests pass on Windows and WSL case-sensitive storage. Old concurrent binaries remain outside the new guarantee. |
| T17 | Training `src/bin/main.rs`, `tests/cli.rs`. Explicit roots preserve defaults/path validation; temporary-root tiny train/save/generate, repeated saves, help and failure-exit tests pass. Actual executable smoke also passed. |
| T18 | Root `.gitignore`, `readme.md`, all three crate READMEs, `Docs/omega-training.md`, `Docs/implementation-status.md`. Existing lockfile resolution unchanged, tracking exception added, navigation/contracts updated; 47 local Markdown links verified. No dependency upgrade or declared MSRV. |

Coordinator validation from `Code/Rust` (Rust 1.93.0, Windows):

- `cargo fmt --all --check`: passed.
- `cargo build --workspace --locked --jobs 1`: passed; existing shared `main`
  filename warnings remain.
- `cargo test --workspace --locked --jobs 1`: 70 passed, zero failed/ignored.
  Count includes subprocess drivers and parser tests reused by the CLI harness.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.
- `cargo run -p omega-training --bin main --locked -- train --help` and
  `... generate --help`: passed.
- Actual `cargo run -p omega-training --bin main --locked -- train` used explicit
  temporary roots, one three-token document, one epoch, context 4, width 4, one
  head/layer and FF width 8; saved `smoke-1`. Matching `generate` with one new
  token passed. Automated repeated-save cycles also passed in temporary roots.
- Case-sensitive reruns set process-local TEMP/TMP to an existing WSL temporary
  directory and ran checkpoint tests plus the dataset case-alias test executable;
  all passed without changing filesystem/security settings. Native Linux binaries
  and the Unix-only symlink test were not run.
- Baseline hash comparison confirmed fixture corpora/tokenizer, `AGENTS.md`,
  `Cargo.lock` and repository `weights/.gitkeep` unchanged; no repository weights
  were created. An unrelated new `datasets/llama-3.json` appeared during work and
  was left untouched.

Review notes: independent reviews found and resolved the T03 case-alias leak;
no outstanding findings for T02/T05/T16. Legacy records still trust saved head
metadata, tokenizer identity is not fingerprinted, and tokenizers 0.23 can omit
truncation overflow. In-memory continuation is not disk resume. These limits are
documented; no unapproved feature was added to resolve them.

Temporary review artifacts remain outside the repository under `%TEMP%`:
`omega-baseline-ede67bb13994461aa08afae0f6592910` (pre-edit source snapshot and
race harness) and `omega-cli-smoke-bb56f6332e6e49efa5f50f2732604d9c` (manual smoke).
The initial combined smoke/recursive-cleanup command was rejected by automatic
approval review (reported only as "blocked by policy"); the smoke was safely
completed without cleanup. Automated test temporary directories cleaned up.

At the end of that implementation pass, no eligible `ready` tasks remained;
promotion of newly unblocked work awaited explicit authorization. The subsequent
readiness review below supersedes that resumption state.

### 2026-09-27 — readiness promotion review

User authorized reviewing and promoting eligible tasks. Promoted the following
tasks from `blocked` to `ready`, with owners left unassigned and acceptance
criteria unchanged:

| Task | Completed prerequisite and available interface |
|---|---|
| T04 | T03: document corpus, deterministic splits and partition provenance are implemented. |
| T06 | T02: model architecture inspection and configuration validation are implemented. |
| T09 | T05: persistent session and committed update/epoch events are implemented. |
| T10 | T01: saved-tokenizer/encoding regression coverage is implemented. |
| T12 | T02: model/configuration and tensor contracts are implemented. |
| T15 | T03: document identity and split-before-chunking interfaces are implemented. |

Readiness removes backlog prerequisites; existing implementation agreements still
apply. T10 must first agree its tokenizer family and special-token policy, T12
its mask/target semantics, and T15 one dataset format/schema. T09 can implement
training metrics now; validation metrics integrate only when T04 is available.
Coordinate shared CLI, exports, manifests and document code before assignment.

Remaining blockers are unchanged: T07 needs T06; T08, T11 and T19 need T07;
T13 needs T07 and T12; T14 needs T06. A prerequisite being `ready` does not make
it complete. T20–T22 remain deferred, including their explicit design approvals.

Verified task IDs/dependency states against the accepted handoffs and current
source interfaces/regression tests. This update changes only task status/notes;
no implementation was started and no build or test suite was rerun.

### 2026-09-27 — second READY implementation pass, assignment wave 1

- T04 / worker-dataset / active; T03 done. Owns training `src/evaluation.rs`
  (new) and `tests/evaluation.rs` if needed. Fixed inference, target-weighted
  loss/perplexity, no mutations; coordinator owns exports/CLI/docs.
- T06 / worker-model / active; T02 done. Owns training `src/checkpoint.rs` and
  `tests/checkpoint_manifest.rs` if needed. Versioned manifest with explicit
  unavailable metadata, canonical tokenizer identity, legacy load compatibility;
  coordinator owns dependencies/provenance assembly/CLI/docs.
- T10 / worker-tokenizer / active; T01 done. Owns tokenizer `src/lib.rs`, new
  `src/training.rs`, `tests/training.rs`. Agreed family: byte-level BPE, all 256
  byte alphabet entries, case-preserving/no normalization, no automatic BOS/EOS
  or other special tokens. Library accepts supplied texts; corpus selection stays
  in training. New-path-only saving and explicit unknown-ID coverage semantics.
  Add immutable `Tokens::to_json` for T06 identity; coordinate its signature.
- T09 / coordinator / active; T05 done. Owns training `src/metrics.rs` (new),
  shared exports, CLI/tests, manifests/lockfile and docs. Versioned JSONL records
  and configurable writer over existing committed events; propagate write errors.
- All workers must return tests/Clippy results, paths, interfaces and limitations;
  no board/instruction/manifest edits, no nested agents, and no unrelated files.
  T12/T15 wait for available worker slots; no blocked/deferred task is promoted.

### 2026-09-27 — second READY pass, assignment wave 2

- T04/T06/T10 library ownership released after focused tests; status review until
  coordinator CLI/docs and integrated validation are accepted.
- T12 / worker-tokenizer / active; T02 done. Exclusively owns NN `src/model.rs`,
  `src/lib.rs`, new `src/loss.rs` and `tests/masking.rs`. Right-padding boolean
  mask [batch, sequence], true=valid, nonempty prefix per row; causal/key masks
  combined, padded logits zero. Differentiable target-mask loss excludes ignored
  indices, rejects no valid targets. Existing unpadded APIs/record fields retained.
- T15 / worker-dataset / active; T03 done. Exclusively owns training
  `src/dataset.rs`. Explicit JSONL objects with required string `text`, one
  document per nonblank record, physical 1-based line errors and identities
  `relative.jsonl/@record-N`. Shared raw-text loader supports tokenizer CLI;
  old loader retains Text default. Preserve source paths and selection safety.
- Coordinator owns all shared CLI/exports/docs/manifest integration and T09.
  Workers cannot edit board/instructions, spawn agents or expand write sets.
- T06/T04/T09/T10/T15 documentation integration: worker-model exclusively owns
  training `README.md` after read-only evaluation/metrics review (no defects;
  coordinator adding missing CLI regressions). Other shared docs stay coordinator-owned.
- T12 final review extension / worker-tokenizer: exclusively owns NN
  `tests/masking.rs` to compare padded and unpadded parameter gradients from
  shared initialization. Production files released; no production defect found.
- T15 ownership released; 18 Windows dataset tests and strict package Clippy pass.
  Independent T06 review found no defect; corrected UTF-8 documentation to
  distinguish file-only decoding errors from file-and-line JSON/schema errors.

### 2026-09-27 — second READY pass accepted integration

All six selected tasks are done. Coordinator inspected implementations, integrated
CLI/exports/docs, collected every worker and independently ran final workspace
validation. Read-only reviews found no outstanding implementation defect; the
T12 parameter-gradient review gap was closed with a passing regression.

| Task | Accepted changes and evidence |
|---|---|
| T04 | Training `src/evaluation.rs`, exports, CLI/tests. Fixed inference loss/perplexity is target-weighted, rejects invalid/nonfinite/overflow cases and preserves optimizer/model state; six evaluator tests plus held-out CLI/reloaded-model comparisons pass. |
| T06 | Training `src/checkpoint.rs`, manifest/lockfile and CLI provenance. Schema-1 authoritative config, full canonical tokenizer SHA256, tokenized-document/partition/run/build metadata, explicit null unknowns, marker-last saves and tested empty-marker legacy loads. Eighteen checkpoint tests plus CLI provenance pass. |
| T09 | Training `src/metrics.rs`, shared CLI/tests. Live committed progress, quiet mode, configurable schema-1 JSONL writer, target weighting, partial/failure semantics and protected log creation; three metrics tests and CLI integration pass. |
| T10 | Tokenizer `src/lib.rs`, `src/training.rs`, `tests/training.rs`; training CLI/tests. Approved byte BPE policy, full-byte UTF-8 round trips, new-file saving, configurable vocabulary/frequency and explicit unknown-ID coverage. Seven new library integration tests and tokenizer CLI workflow pass. |
| T12 | NN `src/model.rs`, `src/lib.rs`, `src/loss.rs`, `tests/masking.rs`. Checked right-padding attention and differentiable selected-target loss; eight mask tests cover causality, valid-logit equivalence, errors, zero ignored gradients and every parameter's padded/unpadded gradient equivalence. Unpadded APIs/records retained. |
| T15 | Training `src/dataset.rs`, exports, CLI/tests. Explicit JSONL text schema, physical-line logical record IDs, shared raw discovery, source-file deduplication and split/pair isolation. Eighteen dataset tests plus end-to-end JSONL train/evaluate/tokenizer commands pass; old text defaults preserved. |

Shared integration files: training `src/lib.rs`, `src/bin/main.rs`, `tests/cli.rs`,
`Cargo.toml`; workspace `Cargo.lock`; root `readme.md`, all crate READMEs,
`Docs/omega-training.md`, `Docs/implementation-status.md`, and this board.
Only `sha2` 0.10 and its six transitive packages were added to the lockfile;
existing package versions and Burn/backend selection are unchanged.

Final coordinator checks from `Code/Rust` on Windows (Rust 1.93.0):

- `cargo fmt --all --check`: passed.
- `cargo build --workspace --locked --jobs 1`: passed.
- `cargo test --workspace --locked --jobs 1`: **110 passed**, zero failed/ignored;
  29 NN, 16 tokenizer, 65 training tests including subprocess/parser harnesses.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`: passed.
- `cargo run -p omega-training --bin main --locked -- <command> --help`:
  train, generate, evaluate, train-tokenizer and coverage all passed.
- Forty-five local Markdown links verified; documented paths/flags checked
  against source. Snapshot comparison found no removed baseline files and no
  changes to existing datasets, instruction files or `weights/.gitkeep`.

Intermediate failures were resolved: in-progress test error formatting and
`unwrap_err` Debug bounds, Clippy style, boolean-mask broadcasting, and a CLI
test expecting `line 2` instead of the correct `file:2` diagnostic. No acceptance
criterion was weakened. All binaries still use `main`: an overlapping NN build
temporarily replaced the executable during help checks. After workers finished,
`cargo rustc -p omega-training --bin main --locked -- -C debuginfo=2` forced the
intended binary build and all package-specific help checks passed. Avoid running
different package builds concurrently with CLI use in the same target directory.

Platform/behavior limits: Unix-only symlink tests, native Unix binaries and
non-CPU/device-mismatch behavior were not exercised. CPU masks synchronize to
the host; no minibatch training was added. Legacy identity cannot be verified;
manifests do not authenticate arbitrary historical weights or checksum model
bytes. Tokenizer retraining is not guaranteed deterministic. There is no disk
resume, periodic save, or new guarantee for final-update numerical validity.

All tests wrote only temporary roots. Unrelated new `Code/Python/` files,
`datasets/raw/` PDFs and `weights/foo-1` appeared during this pass and were left
untouched. Existing Qwen/Llama tokenizer files were preserved. Source snapshot:
`%TEMP%/omega-ready2-baseline-b5dfdd22b8de4d5c98ccadf322dac8e4`.

Resumption: re-read the board; no `ready`, `active` or `review` tasks remain.
T07 and T14 now have completed prerequisites but remain unpromoted as instructed;
an explicit readiness review/assignment is needed next. T08/T11/T19 still need
T07; T13 still needs T07 (T12 is done). T20–T22 retain deferred design approvals.
These notes supersede earlier prerequisite blockers without authorizing backlog
implementation. No commits, pushes, deployments or instruction edits were made.

### 2026-09-27 — readiness promotion after second READY pass

User authorized another readiness review. Promoted T07 and T14 from `blocked`
to `ready`, leaving owners unassigned and acceptance criteria unchanged:

| Task | Completed prerequisites and next implementation agreement |
|---|---|
| T07 (P1) | T05 provides persistent model/Adam/cursor/event state; T06 provides versioned manifests, identity checks and legacy compatibility. Agree state serialization/restore APIs and investigate Burn RNG restoration before claiming exact disk resume. |
| T14 (P2) | T03 provides stable document identities and split/chunk contracts; T06 provides tokenizer identity and dataset fingerprint metadata. Agree iterator/cache interfaces, invalidation rules and memory bounds before adding a second dataset representation. |

T07 has priority. T14 is eligible by its listed prerequisites, but coordinate its
data/cursor interfaces with T07 and serialize overlapping trainer/checkpoint/CLI
edits. Readiness does not establish exclusive write ownership or resolve these
implementation design obligations.

T08, T11, T13 and T19 remain blocked by unfinished T07. T20 still needs T07/T13
and hardware/backend approval; T21 and T22 retain their explicit inference and
storage-policy design approvals. Completed code prerequisites alone do not
authorize those deferred scopes.

Checked all task dependencies against current statuses, accepted handoffs and
source interfaces/regression-test definitions. Only this board changed; no code
implementation or new build/test run occurred. Existing PDF01 handoff preserved.
This review supersedes the previous no-READY resumption note.

### 2026-09-27 — third READY pass assignments

- T07 / worker-model: prerequisites T05/T06 done. First propose serialization,
  restore validation and fixed-order RNG contract; then exclusively own training
  `src/trainer.rs`, new `src/resume.rs`, and `tests/resume.rs` after agreement.
- T14 / worker-dataset: prerequisites T03/T06 done. First propose bounded raw
  reading/cache/index interfaces compatible with T07; then exclusively own
  training `src/dataset.rs`, new `src/cache.rs`, and `tests/cache.rs` after agreement.
- worker-tokenizer: read-only Burn Adam recorder/RNG investigation for T07;
  no source ownership or edits. Return evidence and supported continuation limits.
- Coordinator owns checkpoint integration, CLI/tests, exports, manifests/lockfile,
  docs and this board. No worker may edit these, spawn agents, or broaden scope.
  Preserve all existing user/agent work; this checkout still lacks Git metadata.
  Source snapshot taken outside the repository before this pass.
- Agreed source contract: `ExampleSource` supplies example count, fallible target
  count/indexed reads and ordered-example identity; `TrainingSession<S=TrainingSet>`
  preserves eager callers and supports bounded cached reads through one loop.
  Identity hashes ordered chunk lengths/IDs, not absolute paths; cache validity
  separately pins source bytes, tokenizer/settings and document identities.
- T07 uses full-precision model/Adam records, restored parameter IDs, strict state
  checks and fixed-order CPU/no-stochastic-update policy. Burn exposes no public
  RNG-state restore; current updates require none. Extra resume files precede
  the checkpoint completion marker. Legacy inference loads remain supported.
- T14 uses new-directory-only schema-1 caches, bounded discovery/source reads,
  document token files/checksums and indexed partitions. Explicit limits bound
  source/index/token storage, while tokenizer/backend scratch is not hard-bounded.
- T07/T14 CLI regression assignment / worker-tokenizer: exclusively owns training
  `tests/cli.rs` after RNG research release. Test bounded train/resume equivalence,
  cache creation/use/stale rejection and new help/errors. Coordinator owns CLI
  implementation; no production/shared file changes by this test worker.

### 2026-09-27 — third READY pass accepted handoff

T07 and T14 are done after coordinator source review and integrated validation.
All three native workers returned results and released ownership. No assignments
or commands remain outstanding. An independent CLI/checkpoint/metrics review
found no substantive defect; its cache-only limit clarification is documented.

- T07: training `src/trainer.rs`, new `src/resume.rs`, checkpoint extension hook,
  CLI and regression tests. Full-precision Adam/model records preserve parameter
  IDs, cursor/counters and partial epoch loss. Strict identity/state checks and
  hashes reject incompatible/corrupt inputs before updates. Explicit `resume`
  and bounded `--max-updates` save new numbered snapshots; inference-only and
  legacy loads retain their compatibility policy. Eight resume tests compare
  exact continued events, logits and semantic optimizer state across boundaries.
- T14: training `src/dataset.rs`, new `src/cache.rs` and `tests/cache.rs`.
  Shared indexed examples preserve eager APIs and drive training/evaluation
  without a second loop. New-directory caches pin raw sources, full tokenizer,
  preprocessing/split/settings and token integrity. Six cache tests verify
  eager text/JSONL equivalence, cursor behavior, corruption and all input limits.
- Shared integration: training `src/lib.rs`, `src/checkpoint.rs`,
  `src/evaluation.rs`, `src/metrics.rs`, `src/bin/main.rs`, `tests/cli.rs`,
  `Cargo.toml`, training/root READMEs, both Docs guides and this board.
  `serde_json/float_roundtrip` preserves provenance floats; Cargo.lock and
  dependency versions are unchanged. Sixteen CLI tests cover real subprocess
  continuation/cache workflows, validation metrics, compatibility and errors.

Final coordinator commands from `Code/Rust` on Windows:

- `cargo fmt --all --check`: passed after formatting the integration exports.
- `cargo test --workspace --locked --jobs 1`: **132 passed**, zero failed/ignored
  (29 NN, 16 tokenizer, 87 training, including parser/subprocess harnesses).
- `cargo build --workspace --locked --jobs 1`: passed.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`: passed.
- `cargo run -p omega-training --bin main --locked -- <command> --help`: train,
  resume, prepare-cache, generate, evaluate, train-tokenizer and coverage passed,
  with command-specific flags checked. Shared binary-name collisions initially
  showed NN help; `cargo rustc -p omega-training --bin main --locked -- -C debuginfo=2`
  rebuilt the intended binary before these final checks.
- Forty-five local documentation links verified. Snapshot comparison found no
  removed baseline files or unrelated changes. New source files are only
  `src/resume.rs`, `src/cache.rs` and `tests/cache.rs` in omega-training.

Resolved intermediate validation issues included export formatting and a CLI
test's legacy fixture retaining a versioned marker; the fixture now uses the
original empty marker. Acceptance criteria were not weakened. Tests used only
temporary roots; user datasets/checkpoints and instruction files were untouched.
Baseline: `%TEMP%/omega-ready3-baseline-914bb69780cb40a7a2b45ddcac5a46a2`.

Limits: exact continuation is supported on the current fixed-order CPU f32 path
with a compatible unchanged build/platform. Burn RNG is unused during updates,
not restored. Available build metadata is checked, but null compiler/revision
fields do not prove binary equivalence. Unix-only symlink tests, native Unix and
cross-platform numerical equivalence were not exercised. Cache limits bound
inputs/counts, not process RSS: JSONL retains one capped file's raw records,
token limits apply after encoding, and metadata parsing/tokenizer/backend add
allocations. Each chunk rehashes its token document; no throughput claim is made.

Resumption: no `ready`, `active` or `review` tasks remain. T08, T11, T13 and T19
now have satisfied code prerequisites but retain their unpromoted board status;
they need an explicit readiness review before implementation. This supersedes
their earlier T07 prerequisite blockers. T20 still needs T13 and hardware/backend
approval; T21/T22 retain their deferred design approvals. No backlog tasks were
automatically promoted, and no commits, pushes or deployments were made.

### 2026-09-27 — readiness promotion after third READY pass

User authorized reviewing and promoting eligible tasks. Promoted T08, T11, T13
and T19 from `blocked` to `ready`; owners remain unassigned, and descriptions,
dependencies and acceptance criteria are unchanged.

| Task | Completed prerequisites and implementation agreement needed |
|---|---|
| T08 (P1) | T07 provides verified training snapshots and continuation; T05 update/epoch callbacks expose committed boundaries. Agree save intervals, duplicate-final-save handling and supported interruption signals before wiring the CLI. |
| T11 (P2) | T03 provides stable document partitions; T07 provides explicit cursor/state restoration. Define sampler/order state and reproducible RNG independently of Burn's non-restorable global RNG, preserving current fixed-order defaults. |
| T13 (P2) | T05 provides persistent sessions; T07 provides resume state; T12 provides checked padding masks and differentiable masked loss. Agree bounded collation, real-target weighting and batch/update cursor semantics while preserving batch-size-one behavior. |
| T19 (P2) | T07 provides validated optimizer/session persistence. Split controls into bounded assignments, agree scheduler/accumulation accounting and persisted state, and extend numerical checks beyond the existing snapshot validation. |

Start with P1 T08. T11/T13/T19 are eligible by their listed prerequisites, but
their trainer, resume schema, CLI and metrics changes overlap. Serialize those
write sets or agree compatible interfaces and exclusive ownership first. Schema
changes must explicitly preserve or migrate existing fixed-order snapshots;
readiness does not waive compatibility tests or authorize all controls at once.

T20 remains deferred: T13 is not complete and hardware/backend design approval
is still required. T21 and T22 have completed code prerequisites but retain their
explicit inference and storage-policy design requirements. This readiness review
does not supply those design assignments or authorize retention/deletion policies.

Checked all board dependencies, accepted handoffs, current session/resume/source
and mask APIs, and their regression-test definitions. This was a task-board-only
review: no implementation began and no build/test suite was rerun. The checkout
still lacks Git metadata; a pre-edit board copy was retained in the system temp
directory. This review supersedes the preceding no-READY resumption note.

### 2026-09-27 — fourth READY pass assignments

- T08 / coordinator: prerequisites done. Owns shared CLI, checkpoint orchestration,
  metrics, exports, manifests/lockfile, documentation and board. Implement periodic
  and final saves plus safe interruption at committed update boundaries.
- T11 / worker-dataset: prerequisites T03/T07 done. Propose deterministic sampling
  contract, then exclusively own new training `src/sampling.rs` and inline tests.
  No trainer/resume/CLI writes; request integration through coordinator.
- T13 / worker-tokenizer: prerequisites T05/T07/T12 done. Propose bounded padded
  collation contract, then exclusively own new training `src/batching.rs` and
  inline tests. No trainer/resume/CLI writes; request integration through coordinator.
- T19 / worker-model: T07 done. First investigate bounded clipping/warmup controls
  and numerical checks, proposing state and integration contracts. No code writes
  until agreed; trainer/resume integration will be assigned exclusively after
  cross-task APIs agree. Accumulation is a separate optional control, not implicit.
- Workers may not edit board/instructions/shared files, spawn agents or expand
  scope. Source snapshot taken before editing; preserve user data and other work.
- Interfaces agreed: sampling uses saved fixed/shuffle/integer-weight group policies
  with deterministic epoch orders; batching uses bounded right-padding and target
  masks. worker-model now exclusively owns trainer/resume plus new optimization
  module to integrate both and implement transactional global-norm clipping,
  linear warmup and numerical checks. No accumulation in this bounded T19 scope.
  Resume schema 2 will explicitly migrate schema-1 defaults; only the recorded
  prior lockfile digest is allowed for legacy build compatibility, with other
  runtime checks preserved. Coordinator owns the ctrlc dependency and exports.
- Sampler and collation modules released after five focused tests each and strict
  Clippy. worker-tokenizer now exclusively owns training `tests/cli.rs` for T08,
  T11, T13 and T19 integration regressions. worker-dataset performs read-only
  trainer/resume review; worker-model retains its integration files through tests.

### 2026-09-27 — fourth READY pass accepted handoff

T08, T11, T13 and T19 are done after coordinator implementation review and final
integrated validation. All three native workers returned results and released
ownership; no assignments or commands remain outstanding. Independent sampler
and trainer/resume/optimization reviews found no outstanding implementation defect.

| Task | Accepted changes and evidence |
|---|---|
| T08 | New training `src/run_control.rs`, CLI and metrics integration. Cumulative update/epoch intervals, coincident final-save deduplication and atomic stop notifications save only committed boundaries. Three controller tests and CLI regressions cover interval selection, stop/save/resume, failure propagation and numbered saves; existing incomplete-save tests still pass. Signal handlers only set a flag; checkpoint I/O runs on the training thread. |
| T11 | New training `src/sampling.rs` with five tests, trainer/resume and CLI integration. Versioned portable SplitMix64/Fisher–Yates orders; fixed/shuffle visit each training example once, weighted groups sample with replacement for an explicit draw count. Validation membership stays excluded; strict policies, saved seed/options and deterministic epoch reconstruction preserve exact continuation. |
| T13 | New training `src/batching.rs` with five tests, trainer/resume and CLI integration. Explicit right-padding, attention/loss masks and checked padded-input-position limits preserve document-local targets and partial batches. Loss/metrics count real targets; one batch is one optimizer update, with the cursor at the next draw. Default one-example arithmetic is retained. |
| T19 | New training `src/optimization.rs`, trainer/resume and CLI integration. Optional global gradient clipping and linear warmup persist through resume; every candidate parameter and Adam moment is checked before committing model/optimizer/counters. Rejection tests compare complete Adam state. Enabled-control resume tests compare continued events, logits and semantic optimizer state exactly. No accumulation in the agreed bounded scope. |

Shared changed paths: training `src/lib.rs`, `src/trainer.rs`, `src/resume.rs`,
`src/metrics.rs`, `src/bin/main.rs`, `tests/cli.rs`, `Cargo.toml`, `README.md`;
workspace `Cargo.lock`; NN `README.md`; root `readme.md`; both Docs guides; this
board. New modules are the four listed above. Only ctrlc 3.5.2 and five supporting
packages were added; existing dependency versions and Burn/backend are unchanged.

Compatibility: resume schema 2 requires saved sampling/batching/optimization
options and algorithm identity. Explicit schema-1 migration restores defaults;
only the recorded predecessor lock digest is allowlisted while other runtime
identity checks remain exact. Schema-2 snapshots require the current runtime
identity. Inference manifests remain schema 1 and weights-only checkpoints remain
non-resumable. Terminal CLI metrics now follow a successful final save; earlier
valid periodic saves remain available if a later update or logging operation fails.

Final coordinator validation from `Code/Rust`, Windows/Rust 1.93.0:

- `cargo fmt --all --check`: passed.
- `cargo test --workspace --locked --jobs 1`: **155 passed**, zero failed/ignored
  (29 NN, 16 tokenizer, 110 training including 20 subprocess CLI tests).
- `cargo build --workspace --locked --jobs 1`: passed; existing shared `main`
  output-name warnings remain.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`: passed.
- `cargo rustc -p omega-training --bin main --locked -- -C debuginfo=2`: passed,
  rebuilding the intended binary after the workspace build.
- `cargo run -p omega-training --bin main --locked -- <command> --help`: train,
  resume, prepare-cache, generate, evaluate, train-tokenizer and coverage passed;
  command-specific flags were checked, including every new training control.
- Forty-eight local Markdown links verified. Snapshot comparison found no removed
  baseline files or unrelated edits; instruction files and user data were preserved.

Resolved intermediate issues: strict unknown-field decoding for unit sampling
variants, absolute paths in temporary CLI fixtures, Clippy's large enum warning,
and a help-check harness expecting `--cache` instead of prepare-cache's existing
`--output`. No acceptance criterion or regression assertion was weakened. Tests
used temporary roots and tiny CPU models. Baseline source snapshot:
`%TEMP%/omega-ready4-baseline-312f7dc63a95497aace96d505f3809e0`.

Limits: actual Windows OS signal delivery and Unix execution were not exercised;
atomic stop notification, boundary saving and subsequent resume were tested.
Schema-1 fixtures reconstruct the legacy manifest, not a preserved historical
binary artifact. Exact continuation remains limited to compatible CPU f32 builds;
null compiler/revision metadata does not prove binary equivalence. Source target
lengths and epoch orders consume memory beyond padded-buffer limits; weighted
counter validation replays earlier epochs. No process-RSS bound, accumulation,
cross-platform numerical equivalence or power-loss guarantee is claimed.

Resumption: re-read the board; no ready, active or review tasks remain. T20's code
prerequisites are now complete, but hardware/backend design approval is still
required. T21/T22 retain their inference and storage-policy design requirements.
No deferred task was promoted. No commits, pushes, deployments or instruction
changes were made.

### 2026-09-27 — readiness review after fourth READY pass

User requested another readiness review. T01–T19 are done; no remaining task
can be promoted yet. All code dependencies of T20–T22 are satisfied, but their
explicit design prerequisites remain unresolved:

| Task | Completed prerequisites | Required before promotion |
|---|---|---|
| T20 | T07, T13 | Select target hardware and a Burn-compatible backend; agree device selection, checkpoint portability and numerical/performance validation scope. |
| T21 | T02 | Assign bounded inference work for sampling, KV caching and streaming, preserving greedy defaults; define sampling parameters and cached/uncached equivalence requirements. |
| T22 | T06, T16 | Agree discovery/latest selection, retention, integrity/durability and load limits, including whether deleted checkpoint numbers remain reserved. No deletion policy is authorized. |

Statuses remain `deferred`, with owners unassigned. This promotion review does
not itself choose the missing hardware, inference or storage policies. The next
step is a concrete design assignment for one of these scopes, followed by its
readiness update; there is no unfinished code prerequisite to implement first.

Checked the board and acceptance handoffs against current manifests, CPU/backend
aliases, generation, checkpoint reservation and resume compatibility code, plus
both Docs guides. Only this review note changed; task IDs, descriptions, statuses,
dependencies and acceptance criteria are preserved. The checkout has no Git
metadata; a pre-edit board copy was saved in the system temporary directory.
No implementation or build/test run was needed for this documentation-only review.

### 2026-09-27 — requested T20–T22 implementation plan

User requested a plan and TASKS.md update. Added bounded phases, sequencing,
ownership proposals, compatibility decisions and acceptance checklists under the
three existing deferred task IDs. Native worker-model reviewed T20 and
worker-dataset reviewed T22 read-only; the coordinator reviewed T21 and integrated
the plans. Both workers returned their findings and released the assignments.

The first proposed implementation wave is T21.1 plus T22.1, with T20.1 hardware
qualification independent. GPU hardware/backend selection is still open. T22
keeps current numbering and limits retention to previews; deletion is outside
this plan. T21 preserves greedy defaults and stages sampling, cache, then streaming.
All three parents remain deferred: this turn defines their implementation plan
without promoting phases or claiming implementation. A readiness assignment can
now reference these bounded phases rather than inventing their scope at dispatch.

Validation: checked plans against current CPU aliases, generic model/tensor APIs,
generation/context behavior, checkpoint numbering/publication and resume identity
checks. Verified task IDs/dependencies and preserved prior requirements/history.
Only TASKS.md changed; a pre-edit copy was saved outside the repository. No code,
build, test run, dependency change, hardware benchmark or data operation occurred.

### 2026-09-27 — T21.1 / T22.1 implementation assignments

User explicitly requested native workers for T21.1 and T22.1. Both phases are
active; T02/T06/T16 prerequisites are complete. Other phases remain unassigned.

- T21.1 / worker-model: exclusively owns NN `src/generation.rs` (new),
  `src/lib.rs` for exports and greedy wrapper delegation, `src/bin/main.rs`,
  and dedicated `tests/generation.rs` / `tests/generation_cli.rs` (new).
  Implement CPU deterministic token selection and sampling CLI for NN only;
  no model/cache/streaming/checkpoint changes. Coordinator owns training CLI.
- T22.1 / worker-dataset: exclusively owns training `src/checkpoint_catalog.rs`
  (new), `src/checkpoint.rs` for shared name/header validation helpers, and
  `tests/checkpoint_catalog.rs` (new). Implement bounded read-only inventory and
  latest selection; no schema, retention, durability or resume implementation edits.
- Coordinator owns all remaining integration: training exports/CLI/tests,
  manifests/lockfile if needed, documentation and this board. Workers may not
  edit instructions/board, spawn agents or expand scope. Agree API before callers
  are wired; shared Cargo output means no concurrent package CLI invocations.
- Required handoff: exact changed paths, API/behavior decisions, package tests
  and strict Clippy results, platform limitations and blockers. Coordinator reviews
  and runs integrated workspace validation before accepting either phase.
- T21.1/T22.1 CLI regressions / worker-tokenizer: exclusively owns training
  `tests/cli.rs` after reading existing subprocess fixtures. Coordinator implements
  CLI; this worker requests any production changes and does not edit other files.
- Agreed APIs: `GenerationOptions::{Greedy, Sample(SamplingOptions)}` and
  `generate_with_options`, retaining the old greedy wrapper. Sampling applies
  top-p to normalized mass after top-k and uses versioned request-local SplitMix64.
  Catalog exposes `CatalogLimits`, `CatalogMode`, `discover_checkpoints` and
  `CheckpointCatalog::latest`; metadata inspection does not certify payloads or
  resume runtime/source compatibility. Limits are 100000 root entries and 16 MiB
  aggregate header bytes per candidate by default, with explicit catalog overrides.
- Coordinator wires `--sample`/sampling controls into training generation and
  `checkpoints list/latest`, plus mutually exclusive `--checkpoint`/`--latest-run`
  on generate/evaluate/resume. Selected loader failures never retry older saves.
  Manifests and Cargo.lock remain unchanged; no new dependency was needed.

### 2026-09-27 — T21.1 / T22.1 accepted integration

Both explicitly assigned phases are done after coordinator source review and
integrated validation. All three native workers returned results and released
their files. Parent T21/T22 remain deferred; their other phase checkboxes are
unchanged. No cache/streaming, backend, schema, retention or durability work began.

- T21.1: new NN `src/generation.rs`, `tests/generation.rs` and
  `tests/generation_cli.rs`; NN `src/lib.rs` and `src/bin/main.rs`. Shared options
  and `generate_with_options` support validated temperature/top-k/top-p sampling.
  `generate` retains the greedy path and its tie behavior. The versioned
  SplitMix64 sampler uses one upper-53-bit draw per generated token, seed 42 by
  default; top-p is measured after top-k normalization. Both CLIs expose the
  controls and reject sampling-only flags without `--sample`.
- T22.1: new training `src/checkpoint_catalog.rs` and
  `tests/checkpoint_catalog.rs`; shared parsers in `src/checkpoint.rs` preserve
  explicit loading and reservation contracts. Metadata-only inventory is bounded,
  classifies statuses/reasons and detects unstable completion markers. Numeric
  latest selection rejects ambiguous aliases and corrupt completed headers;
  selected load failures never retry an older checkpoint.
- Shared integration: training `src/lib.rs`, `src/bin/main.rs`, `tests/cli.rs`;
  root/NN/training READMEs, both Docs guides and this board. Added checkpoints
  list/latest (text/JSON) and `--latest-run` on generate/evaluate/resume. No
  dependency, lockfile, instruction, model architecture or checkpoint schema change.

Coordinator commands from `Code/Rust`, Windows/Rust 1.93.0:

- `cargo check -p omega-training --all-targets --locked --jobs 1`: passed.
- `cargo fmt --all --check`: passed after final source changes.
- `cargo test --workspace --locked --jobs 1`: passed after the review fix;
  **185 harness-reported passes**, zero failures (40 NN, 16 tokenizer, 129
  training). One of those reported passes conditionally returned without running
  its Windows symlink assertions because privilege was unavailable; 184 tests
  ran without that conditional skip. Focused catalog output explicitly records it.
- `cargo build --workspace --locked --jobs 1`: passed; existing shared `main`
  filename warnings remain.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`: passed.
- Rebuilt each CLI with `cargo rustc -p <package> --bin main --locked -- -C debuginfo=2`
  before package-specific `cargo run ... --help`. NN train help and all nine
  training command paths (including checkpoints list/latest) passed; new flags
  were verified. No competing package builds ran during those checks.
- Actual Windows junction smoke: catalog classified a junction candidate invalid,
  latest refused it, and a junction weights root was rejected. Only newly created
  temporary paths were used and removed with exact nonrecursive cleanup.
- Forty-eight local documentation links resolve. Source snapshot review found
  no removed baseline files or unrelated edits. Existing user data was preserved.

Independent review found one header consistency defect: discovery initially
omitted the loader's seed/learning-rate provenance comparison. The fix and two
separate regressions passed, followed by a fresh integrated workspace run. The
initial RNG-isolation test was corrected to materialize Burn's existing lazy
model parameters before probing token selection; production sampling was unchanged.
The Windows symlink test now recognizes exact privilege error 1314; no permissions
were changed. No acceptance criterion was weakened.

Limits: actual Windows file/directory symlink assertions and native Unix execution
remain unverified; Windows junction/reparse rejection was exercised separately.
Catalog checks header bytes/semantics, not tokenizer/tensor payloads, complete
source-dependent counters or runtime compatibility; normal loaders retain those
checks. Limits are not process-RSS guarantees, scans are not atomic snapshots,
and adversarial path replacement is outside the trusted-root contract. Windows
cloud placeholders may be excluded because all reparse points are rejected.
Identical sampler logits/options repeat, but cross-platform floating-point token
equivalence is not promised. Fresh model initialization can still consume Burn RNG.

Resumption: T21.2 and T22.2 are the next planned phases, requiring explicit
assignment; do not auto-promote or implement them. T21.3 follows T21.2, and T22.3/
T22.4 follow their listed dependencies. T20 still requires hardware/backend
qualification. No workers or validation commands remain outstanding. No commits,
pushes, deployments, user checkpoint deletion or permission changes were made.
Pre-edit snapshot: `%TEMP%/omega-phases-baseline-eeed30c126294874b971c9ade61c0dac`.

### 2026-09-27 — T20 CPU-first scope update

User prioritized CPU optimization and multithreading, with Intel Arc A770 GPU
support later. Added CPU phases T20.C1 (baseline/thread audit), T20.C2 (thread
controls) and T20.C3 (measured hot-path improvements). Retained T20.1–T20.3 for
the later GPU work. T20 is ready only for C1; no worker is assigned yet. C2/C3
depend on measurement, and GPU qualification does not block CPU progress.

Verified the current host inventory, manifests, locked feature tree and local
Burn/Rayon/matrixmultiply source. The backend already enables multithreading;
thread-pool coordination and performance remain to be measured. The current host
does not report an Arc A770. Its future driver, memory variant and backend
compatibility remain unverified.

Read-only validation from `Code/Rust`: both
`cargo tree -p omega-nn -e features -i matrixmultiply --locked` and
`cargo tree -p omega-nn -e features -i burn-ndarray --locked` passed. Reviewed
phase dependencies, source paths and documentation consistency. Only TASKS.md
and Docs/implementation-status.md changed in this scope update; no production
code, dependency, instruction, dataset or checkpoint changes. No benchmarks,
builds or tests ran, and no speedup is claimed. Next assignment: T20.C1's bounded
release-mode measurement harness and baseline. No new workers were launched.
Pre-edit documentation snapshot:
`%TEMP%/omega-cpu-plan-d71e5939dca848f7bb43db1ae6eb2a6d`.

### 2026-09-27 — T20 CPU implementation assignment

User authorized T20 implementation within the CPU-first scope. T07/T13 are done.
Worker-model owns only `Code/Rust/omega-training/examples/cpu_benchmark.rs`
(new): bounded release measurement harness using public APIs. Worker-dataset
performs a read-only threading/runtime/resume audit. Neither worker may edit
instructions, TASKS.md, shared integration files, or spawn workers. Coordinator
owns baseline execution, shared files, API decisions and integration. CPU C2/C3
production assignments follow reviewed C1 measurements; GPU phases remain later.
- Worker-tokenizer additionally owns only `Code/Rust/scripts/benchmark_cpu.py`
  (new): fresh-process benchmark supervisor, bounded deadlines and JSON results.
  This is independent of the worker-model Rust harness; their CLI contract is
  coordinated before execution. No global environment mutations or user data IO.
- C1 profiling extension: after its read-only audit, worker-dataset exclusively
  owns `omega-training/src/trainer.rs` for optional stage timing only (no
  optimization/semantic changes). Add `step_profiled` alongside unchanged `step`
  and a tiny equivalence regression. No other file writes. Coordinator approves
  the API before benchmark integration. This measures production checks directly.
- Worker-tokenizer released the benchmark supervisor after helper/deadline tests;
  it now performs a read-only optional SIMD/portability audit. Production SIMD
  changes remain gated on C1 measurements and controlled comparison.

### 2026-09-27 — T20.C1 accepted; CPU runtime and measured optimization assignments

Coordinator reviewed the harness, subprocess supervisor and shared timing path.
Release build, example regression (1), strict example Clippy and profiling parity
regression (1) passed. Initial JSON macro recursion failure was fixed in the
example only. All workers returned their C1 files and released ownership.
The 54-case baseline, four profile cases and 18-case repeated comparison all
completed with stable sources/executables, no deadlines exceeded. Raw reports
are in Docs/benchmarks/t20-cpu-{baseline,profile,repeat}.json. Initial sweep wall
9.15 seconds; repeated sweep 16.97 seconds, well inside the five-minute bound.
Training initial/final logits hashes and final loss matched across all nine
initial pool configurations for each workload. This is bounded same-host evidence,
not a promise of cross-machine/thread-count bitwise equality in general.

Fewer Rayon threads improved these tiny workloads; preserve inherited defaults
and expose explicit controls. Profiled small updates spend material time validating
candidate state (~2 ms) and gradients (~0.7 ms); larger batches are dominated by
forward/backward. Investigate typed f32 host scans before changing tensor math.
No evidence supports adding corpus-prefetch workers from this in-memory benchmark.

C2/C3 are authorized as the next bounded CPU phases by the user's T20 assignment:
- Worker-model: exclusively new `omega-training/src/cpu.rs`; pure validated
  child-command thread configuration and frozen execution profile, with unit tests.
- Worker-dataset: exclusively `omega-training/src/trainer.rs` and
  `omega-training/src/optimization.rs`; experiment with typed f32 validation scans,
  preserving every shape/ID/finiteness/moment/counter check and f64 norm order.
  Retain only after coordinator A/B measurement and regression validation.
- Worker-tokenizer: exclusively `omega-training/tests/cli.rs`; runtime flags,
  inherited override, parity and resume compatibility regressions after API agreement.
- Coordinator: CLI, exports, resume schema/migration, catalog compatibility,
  manifests/lockfile, documentation, baseline comparisons and this board. No
  worker may edit outside its write set or spawn another worker. No Burn upgrade,
  default kernel/thread change, GPU work or user-data operation is authorized.
- C2/C3 implementation handoffs collected and write ownership released. Current
  integrated library tests: 91 passed; strict library Clippy passed. CLI tests:
  31 passed; strict CLI Clippy passed. Coordinator reviewed typed scans and CPU
  APIs; worker-model's independent startup/resume/catalog review found no issues.
  Full workspace validation and optimized A/B performance acceptance remain pending.
- Worker-model now owns only uniquely created temporary validation files for a
  bounded, isolated hidden-console Windows Ctrl-Break test of the CPU supervisor.
  It may signal only its own test process group and must collect/clean its children.
- Coordinator added `omega-nn/examples/cpu_kernel_probe.rs` for an isolated
  default-versus-Burn-SIMD numerical experiment (64-element reciprocal and fixed
  tiny model logits). It is not a speed benchmark or supported resume profile;
  optional SIMD remains disabled in production manifests pending compatibility.
- C3 measured slice scans showed no reproducible end-to-end benefit and were
  removed after paired/reversed-order runs; all numerical checks remain. SIMD
  probe changed reciprocal bits and is not adopted. Worker-model now exclusively
  owns new `omega-training/examples/cpu_source_probe.rs`: a bounded release-only
  eager/cached source-read comparison using temporary synthetic files, exact
  source/example parity and existing APIs. No production/cache API edits. This
  closes the cached-read profiling criterion before final C3 review.
- Final production source (after removing the slice experiment) passed
  `cargo test --workspace --locked --jobs 1`: 199 harness passes (40 NN,
  16 tokenizer, 143 training), with the existing Windows symlink privilege
  limitation. `cargo build --workspace --locked --jobs 1` and formatting passed.
  Cargo still warns that all three binaries are named `main`; package-specific
  CLI help checks will follow. C2/C3 acceptance awaits the source probe, final
  all-target Clippy and current release measurement. All numerical/rollback tests
  remain enabled. Manifests, Cargo.lock and instruction files are unchanged.

- Both final benchmark example tests passed (2 total), including cache/eager parity
  and owned temporary cleanup. Source-probe ownership released. Final workspace
  all-target Clippy and formatting passed. The first package-qualified train help
  invocation exposed Cargo's existing shared `main.exe` collision: it printed NN
  help despite `-p omega-training`. Coordinator is forcing the training binary
  relink before checking actual train/generate help. No test assertion is bypassed.
- The relink (`cargo rustc -p omega-training --bin main --locked -- -C debuginfo=2`)
  resolved the existing binary collision. Both requested `cargo run -p
  omega-training --bin main --locked -- train|generate --help` checks then passed
  with correct command-specific options and both CPU flags. Windows Ctrl+Break
  smoke passed: two committed updates, one interruption event, completed schema-3
  save, nonzero exit and zero remaining descendants. Automatic approval review
  rejected recursive cleanup of its owned scratch directory as "blocked by policy";
  no bypass was attempted. Retained artifacts:
  `%TEMP%/omega-cpu-ctrlbreak-36aa2c82366d4d5e889f16a08e340b7c`.

### 2026-09-27 — T20 CPU phases accepted

Completed T20.C1/C2/C3 in the assigned CPU-first scope. Coordinator reviewed all
worker files and experimental changes. Safe fresh-process controls expose
`--cpu-threads 1..256` and `--matmul-threads 1|2|4`, preserving inherited defaults.
Library helpers configure child commands without global environment mutation.
New resume schema 3 requires the frozen CPU profile; mismatches fail before
restoration/updates. Schema-1/2 migration preserves the prior build policy and
requires unset pool variables/default kernels under the documented same-host
historical assumption. Inference loading remains independent of thread settings.

C1/C3 deliverables: bounded release train/forward/generation harness and Python
supervisor; optional checked-update stage timings; numerical kernel probe;
eager/cached read probe; raw reports and reproduction in Docs/cpu-performance.md.
Typed host-scan experiment was removed after no reproducible end-to-end gain;
all original numerical/rollback checks remain. Optional SIMD changed reciprocal
bits (`3eaaaaab` to `3eaaaa80`), so no feature/dependency/default changes were made.

The final 18-case release sweep completed in 19.34 seconds with all fingerprints
matching the saved C1 baseline. At Rayon1/matmul1, small-context training improved
1.29–1.35x and generation 1.41–1.52x; larger training was approximately unchanged
or slightly slower. These tiny-model laptop measurements are not general speedup
claims. Five warmed eager/cache read pairs verified exact identities/examples;
median 64-example read times were 0.0098/9.403 ms, total probe 0.166 seconds and
peak process working set 7.75 MiB. No cache integrity check or user data changed.

Final validation from Code/Rust:

- `cargo fmt --all --check`: passed.
- `cargo test --workspace --locked --jobs 1`: 199 harness passes; 0 failures.
- `cargo build --workspace --locked --jobs 1`: passed; existing main-name warnings.
- `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`: passed.
- `cargo test -p omega-training --example cpu_source_probe --example cpu_benchmark
  --locked --jobs 1`: 2 passed; strict source-probe example Clippy also passed.
- `cargo build -p omega-training --examples --release --locked --jobs 1`: passed.
- `python -B scripts/benchmark_cpu.py --output ../../Docs/benchmarks/t20-cpu-final.json
  --threads default --threads 1:1 --threads 1:2 --samples 15 --iterations 4`: all
  18 cases complete; no timeout or source/executable change. Earlier 116 baseline,
  profile and experiment cases also completed; rejected candidate reports are labeled.
- `target/release/examples/cpu_source_probe.exe`: passed under the supervisor's
  ten-second process deadline, with successful owned scratch cleanup.
- `cargo run -p omega-nn --example cpu_kernel_probe --locked --jobs 1`, with and
  without `--features burn/simd`: both probes ran; numerical difference recorded.
- Actual training CLI train/generate help passed after the documented relink.
  Default feature tree confirms SIMD remains disabled; manifests/lock unchanged.

Changed production paths: omega-training/src/{cpu,lib,trainer,resume,
checkpoint_catalog}.rs and src/bin/main.rs. Regressions: tests/{cli,
checkpoint_catalog}.rs. Tooling: omega-training/examples/{cpu_benchmark,
cpu_source_probe}.rs, omega-nn/examples/cpu_kernel_probe.rs and
Code/Rust/scripts/benchmark_cpu.py. Documentation: training README.md,
Docs/{cpu-performance,omega-training,implementation-status}.md, benchmark reports
and this board. No instruction files, manifests, dependencies, user corpora or
checkpoints were changed. Pre-T20 source snapshot remains in
`%TEMP%/omega-t20-baseline-2c1afb220fe24a7cacb3a71e2e63a20f`.

Limits/resumption: native Unix startup/signal behavior and Windows symlink
privilege assertions remain unverified. Resume profiles record requested raw
settings, not observed worker counts or externally initialized pools; unsupported
feature/target-CPU builds need an explicit compatibility policy. T20.1–T20.3 GPU
work remains deferred for A770 hardware/driver/backend qualification. T21/T22
later phases remain unassigned. All workers have returned/released ownership and
no commands remain running. The Ctrl+Break scratch cleanup rejection and retained
path are recorded above. No commits, pushes, deployments or permission changes.

### 2026-09-28 — Arc A770 GPU training task plan

User requested the tasks needed for GPU training. Expanded the existing T20.1–3
plan and added T20.4–5 for representative performance measurement and operational
handoff. Recorded read-only hardware/workload evidence from the preceding
investigation: Linux/i7-13700K, Vulkan-visible 16 GB A770, debug-mode active run,
and large-vocabulary historical snapshot settings. Hardware discovery does not
complete the Burn feasibility gate. Earlier Windows/Ryzen evidence stays historical.

Only TASKS.md changed. T20 remains deferred/unassigned for implementation; no
phase was promoted, no GPU support was claimed, and no active job, source,
dataset or checkpoint was modified. Checked task IDs/dependencies, source paths,
existing CPU/schema contracts and preservation of accepted task history. This
was a documentation change; no build, test suite or benchmark was run.

### 2026-09-28 — GPU implementation assignment

User authorized workers and implementation. T20.1 is active; subsequent GPU
phases proceed only after its hardware/API gate passes. Prerequisites T07/T13
are complete. Existing CPU training process and user data remain untouched.

- gpu-probe: owns only new `Code/Rust/omega-training/examples/gpu_probe.rs`
  and new `Code/Rust/omega-training/src/gpu.rs`; qualify Burn 0.18 Vulkan on
  A770, then implement explicit adapter/profile helpers after API agreement.
- trainer: initially read-only generic trainer/optimization/evaluation audit;
  proposed exclusive write set is training `src/trainer.rs`, `src/optimization.rs`,
  `src/evaluation.rs`, `src/run_control.rs`, `src/metrics.rs`. Production edits
  wait for feasibility/API agreement. Preserve CPU wrapper signatures.
- persistence: initially read-only checkpoint/resume and generic NN helper audit;
  propose schemas and compatibility. No writes until coordinator assigns exact
  files after feasibility.
- Coordinator owns manifests/lockfile, shared exports/CLI, docs, scripts, tests
  and TASKS.md unless later explicitly delegated. Do not run concurrent GPU
  workloads or package binaries sharing the same target filename. Workers return
  exact changes, APIs, validation and limitations; no nested agents.

Validation: affected packages and strict Clippy, then CPU/GPU feature workspace
checks and bounded real A770 tests in temporary roots. Model/device transfers
are not exact resume. No automatic active-job restart or checkpoint conversion.

- T20.1 gate passed on A770 using Burn 0.18 Vulkan/SPIR-V: masked forward/loss,
  backward with 21 finite parameter gradients, Adam updates, optimizer record
  round trips and CPU/GPU model transfer passed the bounded probe. WGSL failed
  vector boolean shader compilation and is not selected. Production T20.2/3
  writes are now assigned: trainer owns its five proposed files; persistence owns
  training checkpoint.rs, resume.rs, checkpoint_catalog.rs, tests/checkpoint_catalog.rs
  and new tests/gpu_resume.rs. Coordinator owns NN generation.rs/lib.rs plus CLI.
  CPU wrappers remain, generic session backend follows source type with default
  TrainingBackend; explicit-device constructors use the same update loop. CPU
  saves retain schema 3; GPU state uses schema 4 with strict execution identity.

- After T20.1 gate, gpu-probe worker released gpu.rs/probe and owns only new
  examples/gpu_benchmark.rs for synthetic representative benchmark preparation.
  Coordinator owns benchmark execution/results; no concurrent GPU measurements.
  Trainer adds an explicit fresh-model constructor for shared initial parameters
  with a new optimizer/cursor; this is not a checkpoint resume/import CLI.

- Persistence worker released production files and now exclusively owns new
  tests/gpu_subprocess.rs: same-snapshot cross-process continuation and actual
  Linux SIGINT/SIGTERM save/resume tests, isolated temporary roots with child
  deadlines. Coordinator serializes hardware execution and owns all production
  integration. GPU helper/profile review fixes and buffer preflight are integrated.

### 2026-09-28 — GPU backend implementation and hardware acceptance

The user authorized workers and implementation. Coordinator integrated three
bounded workers after the A770 feasibility gate; all worker files are released.
T20.1/T20.3 are accepted. T20.2's working backend and T20.4's bounded benchmark
are accepted, with detailed performance attribution still open. T20.5 operational
handoff is accepted after final release/build validation below; the full T20
plan remains open for profiling. This is operational GPU training and
continuation, not merely device enumeration or an inference-only backend.

Implementation and compatibility:

- Optional `omega-training/gpu` selects Burn 0.18 Vulkan/SPIR-V, f32; CPU default
  APIs and one shared checked training loop remain. WGSL failed mask shader
  compilation in the feasibility experiment and was not selected. No existing
  dependency package version/checksum changed; nine optional compiler packages
  were added to the lockfile.
- Discrete Vulkan adapter indices exclude integrated/software adapters. Explicit
  initialization verifies the selected adapter, with no CPU fallback. Known
  individual model/activation buffers are checked against device limits before
  CLI allocation; this is not an aggregate VRAM guarantee.
- Generic session/evaluation/generation APIs accept explicit devices. CLI adds
  `--backend cpu|vulkan`, `--device INDEX` and `devices`; CPU-only requests for
  Vulkan and irrelevant CPU thread flags fail clearly. The unique
  `--bin omega-training` avoids the legacy cross-package `main` collision.
- CPU saves remain schema 3; GPU training saves use schema 4, including adapter,
  driver, kernel, precision, compiler/target, build profile, optimization/debug
  settings and Rust flags. Exact resume requires matching execution identity.
  CPU/GPU inference weights transfer, but exact cross-backend or Windows/Linux
  resume is rejected. No implicit weights-only training import was introduced.
- The exact pre-GPU lock digest `04991746127d1994da4354c1d969ed436a799b8ef6ac9e549b473a43b4782696`
  maps only to reviewed successor `55fc999ee31d21418f04f8f4006d69a849c27cb75da94e9c51adee98bc159201`
  for existing CPU resume schemas. Other execution checks remain. Historical
  Windows `macro_invest-*` snapshots are not directly resumable GPU checkpoints.

Hardware evidence (actual Linux Arc A770/Mesa 26.1.8-arch1.1):

- `gpu_probe` passed masked/causal forward/loss, all 21 parameter gradients,
  Adam/record round trips, synchronization and CPU/GPU transfer from shared
  parameters, absolute/relative tolerance 0.003. Actual probe losses were
  2.9991803 and 2.980272; these are correctness observations, not quality claims.
- Explicit ignored `gpu_resume` test passed in 15.04 s. Fixed/shuffled/weighted
  order, padded batches, clipping/warmup, mid/end-epoch continuation compare
  model values, canonical Adam IDs/moments/counters and progress exactly on this
  profile. Injected nonfinite candidate updates preserve committed state;
  nonfinite generation errors, clear greedy maxima agree, and near-ties accept
  either numerically tied token rather than asserting cross-backend equality.
- Explicit `gpu_cli` workflow passed in 7.54 s: temporary-root GPU training,
  save/resume/generate/evaluate, CPU/GPU inference transfer, invalid selection,
  cross-backend rejection and preflight of legacy checkpoint headers.
- Explicit Linux `gpu_subprocess` test passed in 22.03 s: fresh-process split
  continuation equals uninterrupted saved-state continuation, periodic/final
  saves deduplicate, real SIGINT and SIGTERM yield completed resumable saves and
  nonzero interrupted exits. Every owned child has a 30 s deadline and is reaped.
- `gpu_artifact_failure_never_publishes_complete_and_preserves_numbering` passed
  in 0.14 s: deterministic post-model artifact failure leaves no COMPLETE,
  rejects loading, preserves the prior save and reserves the failed suffix.

Performance: raw 13-case report (`Docs/benchmarks/t20-gpu-a770.json`, local capture) and
[commands/limitations](Docs/gpu-training.md). Sequential release sweep completed
in 74.74 s, under the five-minute/30-second-child bounds, with identical initial
parameter hashes and unchanged executable. Model dimensions match the recorded
151665-vocabulary workload; IDs are synthetic. Two warmups and three timed
checked updates per case give GPU/CPU targets/s: batch 1 127.6/74.8 (1.71x),
batch 2 211.6/73.1 (2.90x), batch 4 347.1/92.2 (3.77x). CPU thread settings had
little measured impact. Batch 4 changes optimizer-step count; defaults remain.
Peak host RSS and raw DRM samples are reported separately; the latter are not
peak VRAM. Compilation, warmup and update timings are separate; checkpoint I/O
is excluded, not measured. The active CPU debug job is a recorded confounder.
No full-corpus completion-time extrapolation or quality improvement is claimed.

Remaining T20 work: detailed projection/loss, gradient/Adam validation, transfer,
synchronization, candidate-copy and allocation attribution, plus checkpoint-I/O
latency. No kernel/check-removal optimization was accepted without measurements.
Other GPUs, drivers and operating systems remain unqualified; adapter index is
not a physical device UUID and unsupported external feature unification is not
covered by the execution profile. T21/T22 later phases remain unassigned.

Changed implementation paths under `Code/Rust`:

- `Cargo.lock`; `omega-training/Cargo.toml`, `omega-training/build.rs`.
- `omega-training/src/{gpu,trainer,optimization,evaluation,run_control,metrics,checkpoint,resume,checkpoint_catalog,lib}.rs`.
- `omega-training/src/bin/{main,omega-training}.rs`.
- `omega-training/examples/{gpu_probe,gpu_benchmark}.rs`.
- `omega-training/tests/{gpu_cli,gpu_resume,gpu_subprocess,checkpoint_catalog}.rs`.
- `omega-nn/src/{generation,lib}.rs` and both affected crate READMEs.
- Root `readme.md`, `TASKS.md`; `Docs/{gpu-training,omega-training,implementation-status,cpu-performance}.md`, benchmark JSON; `Scripts/{training_run.sh,test_training_run.sh,README.md}`.

No user corpus/checkpoint was used for writes, and PID 2881718 was left running.
Pre-edit snapshot: `/tmp/omega-gpu-implementation-b9clc8d9`. No Git metadata was
present and no commit was created.

Final integrated validation (2026-09-28):

- `cargo test --workspace --locked --jobs 2`: passed CPU suites.
- `cargo test --workspace --features omega-training/gpu --locked --jobs 1`:
  passed ordinary suites; hardware-only tests were then explicitly executed
  sequentially as recorded above. Ordinary ignored status is not the hardware
  acceptance evidence.
- Locked CPU and `omega-training/gpu` workspace builds passed with `--jobs 1`.
- Strict `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`
  and the equivalent `--features omega-training/gpu` command passed. The first
  GPU Clippy pass found one needless CLI path borrow, which was fixed; final
  passes include the additional GPU failure/generation regressions.
- `cargo fmt --all --check` passed. Bash wrapper mocked-Cargo tests passed;
  modified Markdown local links resolve and benchmark JSON parses.
- `cargo build -p omega-training --bin omega-training --example gpu_benchmark --release --features gpu --locked --jobs 2`
  passed (final incremental release build 2m28s). Final optimized executable
  listed the A770 and passed train/generate help checks. A temporary-root smoke
  test then performed release GPU train, schema-4 save, fresh-process GPU resume,
  GPU generation and CPU inference transfer, each with a 30-second subprocess
  deadline. Saved profile is release/opt3/debugfalse; COMPLETE was present.
  Temporary smoke artifacts were removed.
- Workspace builds still warn about the pre-existing three legacy `main` target
  names; the new unique training binary and updated wrapper avoid that collision
  for GPU workflows. No automatic CPU-job restart or user-data writes occurred.

## OD01 — Hugging Face dataset builder — 2026-09-30

Status: done (tooling only). Owner: current omega-datasets coordinator. User explicitly requested
an independent Rust library/CLI configured by datasets/<name>/model.toml.
Prerequisites reviewed: zero-to-hero ZH02 corpus contract, current text and schema-1
conversation loaders. This is acquisition/preparation tooling, not acceptance of
ZH01/ZH02 or authorization for a large download/training run.
Exclusive scope: new Code/Rust/omega-datasets subtree; workspace member/dependency
lockfile integration; additive navigation/docs/task-board updates. Existing ongoing
assistant/tokenizer/trainer/Python changes are outside this assignment.
Acceptance: validated TOML; pinned Hugging Face file acquisition; base/chat field
mapping; deterministic grouped train/validation/test splits; exact duplicate and
cross-stage exact-text overlap handling; immutable output and provenance; offline
HTTP/fixture regression tests; package and workspace fmt/test/clippy checks.
No branches or commits. Snapshot of shared files saved outside the repository.

OD01 integration progress: omega-datasets source, workspace manifest and Cargo.lock
are now stable for integrated checks. Package tests: 14 passed; strict package
Clippy passed. Offline HTTP fixtures cover pinned paths/auth/errors; live metadata-only
`plan` resolved the example shard successfully, with no corpus download. The lockfile
adds the new crate/dependencies without removing prior package versions or changing
existing checksums. Existing exact-resume lockfile checks are preserved; old-build
resume compatibility is not claimed. Full four-crate validation is in progress using
an isolated CARGO_TARGET_DIR, avoiding concurrent training executable collisions.

OD01 full-workspace validation found one integration failure on the new lockfile:
omega-training::resume::tests::pre_gpu_cpu_lock_migration_is_narrow_and_keeps_platform_checks
unwraps successful migration at src/resume.rs:893, but production deliberately requires
current lock == GPU_COMPATIBLE_LOCK. The new lock correctly fails that allowlist.
All 14 omega-datasets tests passed under workspace feature unification; the training
library reported 94 passed/1 failed before Cargo stopped. Production resume guards
must stay unchanged; adapting the historical test expectation is being coordinated
with the current owner of resume.rs. OD01 remains active pending integration checks.

OD01 compatibility resolution: the assistant-training coordinator owns the test-only
change in omega-training/src/resume.rs. It now permits PRE_GPU_LOCK only on the exact
old build or its reviewed GPU_COMPATIBLE_LOCK successor; other current locks assert
rejection. Platform rejection is checked after restoring the current lock so it is
independent of the lock guard. Production migration code and allowlists are unchanged.
Windows full-workspace strict Clippy now passed with this correction present.
The assistant coordinator reports full four-crate test/Clippy/fmt passes on Linux
Rust 1.97.1 with the current shared checkout, plus a separate successful historical
allowlisted positive-branch test in the preserved Windows three-crate checkout.
Final current-lock Windows full-suite rerun is in progress.

### OD01 final handoff — 2026-09-30

Accepted as acquisition/preparation tooling only. ZH02 corpus selection, human
quality/near-duplicate review and a frozen real release remain unaccepted.

Changed paths owned by this assignment:
- Code/Rust/Cargo.toml and Code/Rust/Cargo.lock
- Code/Rust/omega-datasets/Cargo.toml
- Code/Rust/omega-datasets/README.md
- Code/Rust/omega-datasets/examples/model.toml
- Code/Rust/omega-datasets/src/config.rs
- Code/Rust/omega-datasets/src/hub.rs
- Code/Rust/omega-datasets/src/lib.rs
- Code/Rust/omega-datasets/src/main.rs
- Code/Rust/omega-datasets/src/records.rs
- Code/Rust/omega-datasets/tests/preparation.rs
- Docs/omega-datasets.md, readme.md, and this OD01 TASKS.md section

Coordinated integration change owned by the assistant-training coordinator:
Code/Rust/omega-training/src/resume.rs historical-lock unit test only, as described
above. No production migration policy, allowlist, checkpoint or tokenizer changed.
No branch/commit was made; this checkout has no Git metadata.

API/behavior: independent Rust library + uniquely named omega-datasets executable;
Config::load/validate, Hub transport trait, HuggingFace client, hub::plan, build,
dataset_folder and init. CLI init/check/plan/build reads datasets/<name>/model.toml.
JSONL/gzip, JSON arrays and Parquet become schema-compatible base text or schema-1
chat JSONL. Pinned commits, raw-file hashes, normalized exact deduplication, explicit
source families and cross-stage exact-message overlap precede deterministic grouped
splits. Fixed partition conflicts fail. New releases include raw originals, replay
config, source/membership/output hashes and a marker written last. Existing outputs
are never replaced. Library paths explicit; CLI defaults preserve checkout rooting.

Direct Windows/Rust 1.93.0 validation, using isolated CARGO_TARGET_DIR:
- cargo test -p omega-datasets --locked --jobs 2: 14 tests passed.
- cargo clippy -p omega-datasets --all-targets --locked --jobs 2 -- -D warnings: passed.
- cargo test --workspace --locked --jobs 1: final rerun passed all 251 harness tests;
  the initial historical-lock test failure was resolved with the scoped test fix.
- cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings: passed.
- cargo fmt --all --check: passed after the final test rerun.
- CLI init/check/error exits and all four subcommand help screens covered by tests.
- Live metadata-only plan resolved the TinyStories example's full commit SHA and
  248731111-byte shard; no dataset source bytes were downloaded.
- Four local documentation links and balanced Markdown fences verified.
- Existing shared-main filename warnings remain; serial tests passed. New CLI has
  a unique executable name. New Windows junction assertions were exercised.

Independent coordinated evidence: assistant-training coordinator reports current
four-crate test/strict-Clippy/fmt passes on Linux/Rust 1.97.1, including the new
Unix-gated tests. Those are coordinator-run results, not locally rerun Linux tests.
GPU training was not part of OD01. No real corpus was downloaded and no non-test
training job was launched; regression tests used tiny temporary fixtures.

Limits: normalized records/indexes are retained in bounded memory; parser scratch
and Parquet pages are not a hard RSS bound. No download resume/retries, near-duplicate
or semantic matching, license/privacy certification, tokenizer/context checks or
automatic corpus acceptance. Full upstream files are processed; exceeded budgets
fail without silently sampling/truncating. The starter shard exceeds its conservative
record cap, documented in the template and guides. Existing strict exact-resume
identity can reject prior builds because Cargo.lock changed; use the original
compatible runtime rather than widening the guard. No unresolved OD01 test failures.
