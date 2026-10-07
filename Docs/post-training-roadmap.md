# After base training: fine-tuning and automated evaluation

Planning baseline: **2026-10-07**, branch **dev**. The software workflow is now
implemented and undergoing integrated qualification; the real experiment and
model acceptance remain pending. Execution status
and acceptance checklists live in [TASKS.md](../TASKS.md#post-training-roadmap--2026-10-07).
The user reports an active base-training run; this work does not inspect,
interrupt, replace or certify that run.

## Goal and current baseline

For implemented APIs, schema-2 configuration, commands and recovery limits, see
[the post-training guide](post-training.md). The table below records the original
planning gap; current evidence and unresolved pilot inputs are in
[the prerequisite audit](post-training-audit.md) and TASKS.md.

Build a complete workflow around Omega's existing full-model supervised
fine-tuning (SFT) support:

**Select base checkpoint → validate conversation data → establish baseline →
fine-tune in bounded segments → evaluate and compare → human review → promote.**

After approving a workflow, the user can detach while training and evaluation
continue within its approved budget. Promotion remains an explicit user decision.
The first target is a conversational assistant. A separate small in-app helper
comes later and initially proposes actions for approval.

| Capability | Current source | Remaining work |
|---|---|---|
| Conversation training | Frozen `omega-chat-v1` tokenizer protocol, assistant-only loss, padded masked batches, `train-stage` with fresh Adam and parent lineage | Parent/data readiness report and independently configured post-training workflow |
| Persistence | Complete-checkpoint verification, compatible exact stage resume, detached workers and host compute lock | Durable segment/evaluation sequencing, recovery and aggregate budgets |
| Evaluation | Fixed-checkpoint, target-weighted loss/perplexity, including assistant targets | Versioned conversation suites, baseline comparison, human scoring and selection gates |
| Interaction | Bounded local chat, TUI checkpoint selection and job monitoring | Guided fine-tuning, comparison/review screens and explicit promotion |
| Compatibility | Project schema 1; the assistant initializer uses the normal strict resume metadata reader | Explicit project migration and a separately validated new-stage weights-transfer contract |

See [assistant execution](assistant-execution.md), [the application guide](omega.md)
and [the training README](../Code/Rust/omega-training/README.md) for existing use.
The relevant implementation is in training `assistant`, `evaluation`, `resume`,
`run_control` and `operations`, plus Omega `config`, `jobs` and `pipeline`.
Read those sources before implementation; historical test counts are not new evidence.

Today the combined assistant pipeline requires base training, standalone assistant
launch validation still checks the general dataset selection, and bounded training
can stop the overall pipeline. The new workflow must support an already-trained
parent and distinguish a finished segment from a request to stop all remaining work.

## Roadmap and dependencies

The original dependency order remains the acceptance order. Software can be
implemented and tested with explicitly synthetic fixtures while FT01's real
parent/data/quality thresholds and budgets remain unresolved. Passing those
software tests does not accept a production dataset or a trained model. FT12
remains blocked until qualification and separate pilot approval.

| ID | Task | Dependencies | Required outcome |
|---|---|---|---|
| FT01 | Define the post-training experiment | — | Record parent identity, conversational scope, partitions, suite, human rubric, thresholds, and compute/storage limits. Separate software and quality acceptance. |
| FT02 | Validate parent checkpoints and SFT data | FT01 | Verify completeness, hashes, architecture, tokenizer/protocol, roles, assistant targets, context lengths, provenance and detectable overlap; return an actionable report without changing inputs. |
| FT03 | Add post-training configuration | FT02 | Configure an existing parent, assistant-specific settings, segment size, evaluation cadence, budgets and outputs; freeze resolved settings without rerunning base training. |
| FT04 | Implement conversational test suites | FT02 | Version single/multi-turn cases with fixed generation settings, expected-answer/JSON checks, termination, role leakage, empty output, repetition and context-limit checks; preserve outputs and failures. |
| FT05 | Produce baseline and comparison reports | FT04 | Measure assistant validation loss, base-language regression and conversation results; compare compatible configurations with full artifact/runtime identity; export JSON and Markdown. |
| FT06 | Add human quality review | FT05 | Score instruction-following, correctness, relevance and coherence against a versioned rubric; retain notes and responses; missing review remains pending. |
| FT07 | Implement bounded train/evaluate segments | FT03, FT05 | Initialize SFT once, resume the same stage between segments, verify saves before evaluation, and distinguish segment completion, budget exhaustion, user stop and failure. |
| FT08 | Persist the automated workflow | FT07 | Sequence baseline, training and evaluation under one compute lock; journal transitions and reports; recover without duplicate work or false passing results. |
| FT09 | Integrate TUI and CLI | FT06, FT08 | Expose parent selection, readiness, review, progress, results, comparisons and human scoring through shared Rust APIs in the single binary. |
| FT10 | Rank and promote candidates | FT06, FT08 | Filter failed/pending candidates, rank eligible checkpoints by validation assistant loss then fewer updates, and record human-approved selection with immutable hashes/evidence. |
| FT11 | Qualify the implementation | FT09, FT10 | Pass offline end-to-end, persistence, compatibility, failure and backend tests, including parent preservation and exact SFT continuation. |
| FT12 | Run the real pilot and acceptance process | FT11; separately approved data/budgets | Compare a bounded real pilot with its base, review responses, select a candidate, then perform final sealed-test acceptance and local model packaging. |

Delivery order:

1. **Readiness:** FT01–FT03 establish the experiment and safe launch inputs.
2. **Measurement:** FT04–FT06 establish what improvement means before automation.
3. **Automation:** FT07–FT10 sequence work and expose review/selection controls.
4. **Qualification and use:** FT11 proves software behaviour; FT12 supplies actual
   model-quality evidence. A failed gate stays failed; more epochs are not an
   automatic remedy.

### Relationship to the existing roadmap

| Existing task | Connection |
|---|---|
| ZH05 | FT01, FT04–FT06, FT08 and FT10 supply the native experiment/evaluation workflow; retain existing experiment-recorder history. |
| ZH06, ZH07 | Reuse implemented masking and stage initialization; FT02/FT03 extend readiness and compatibility deliberately. |
| ZH10 | FT12's bounded SFT pilot supplies real baseline-comparison evidence after FT11. |
| ZH11 | FT12's approved continuation and candidate selection supply assistant-training evidence. |
| ZH12 | Reuse existing chat for inference; passing chat regression tests does not prove conversational quality. |
| ZH13 | FT12's sealed-test report, model card and recoverable local artifacts supply final acceptance evidence. |

Do not mark ZH10, ZH11 or ZH13 complete merely because their supporting software
exists. Keep ZH14 outside this milestone.

## Implementation contracts

### Experiment and data

FT01 records a concrete parent path and hashes, language/subject scope, expected
conversation lengths, exact train/development/base-regression/sealed-test
partitions, suite and rubric versions, generation settings, required gates and
resource caps. Missing production choices remain unresolved, not guessed from
test fixtures or an active job. Freeze the profile before comparing candidates.

Keep tokenizer IDs, architecture and the saved conversation protocol unchanged.
Reject missing chat controls or oversized conversations with actionable errors;
never resize vocabulary, guess token IDs or silently truncate answers. Retain
whole conversation/source-group split membership. Verify available release
provenance and report detectable overlap and limits of detection; absence of an
exact match is not proof that semantic leakage is impossible.

Development data drives comparisons. Sealed tests are excluded from automatic
loops and training; final acceptance is a separate explicit operation. Failed
final acceptance is retained honestly, and further tuning requires a fresh
held-out evaluation plan.

### APIs and compatibility

Model execution/evaluation belongs in `omega-training`; sequencing, budgets,
persistence and presentation belong in Omega. Reuse dataset/tokenizer libraries
and typed operations. No sibling executable, shell or Python runtime dependency.

Introduce typed evaluation suites, case results, reports and operation outcomes.
Outcomes distinguish segment completion from user interruption and budget
exhaustion. Workflow state refers to verified checkpoints and completed reports.
Keep process success separate from quality-gate success and human approval.

Introduce **project schema 2** for new workflow settings. Keep schema-1 readers
and provide explicit comment-preserving migration. Never rewrite existing job
snapshots. These are project schema numbers, not checkpoint/resume schema changes.
Existing jobs continue using their retained worker executable.

Add a separately tested **new-stage weights transfer** path for supported Omega
checkpoint formats across application builds. Preserve format, tensor structure,
integrity and tokenizer/protocol checks; initialize fresh optimizer/sampler state
and record the original parent and new build in lineage. Reject unsupported
formats instead of falling back. This is not exact resume, external model import
or permission to weaken exact-resume runtime/backend checks. Record the supported
format/compatibility matrix with tests before enabling this path.

### Evaluation and human review

Record fixed generation settings, inputs, actual outputs, completion reasons,
per-check outcomes and latency for each case. Keep single-turn and multi-turn
tests, expected-answer checks and structured-output checks distinct from human
judgments. Hold model weights fixed during evaluation. Preserve partial/error
reports as incomplete; never substitute omitted cases with passing scores.

Compare assistant-target loss only under matched tokenizer, context, target
policy, partition and suite settings. Show base-language regression separately.
Persist immutable report identities and tie human scores/notes to the exact
response and rubric being reviewed. Changed outputs or suite versions invalidate
reuse of the old review for a new candidate.

Apply required gates before ranking. Rank eligible candidates by lower validation
assistant loss, breaking ties by fewer updates; show other metrics separately.
Missing required human scores or incomplete reports leave a candidate pending.
Explicit promotion writes a new immutable selection manifest with checkpoint,
report and review hashes. It never modifies weights, silently chooses latest,
or implies a public release. Candidate promotion and final sealed-test acceptance
remain distinguishable states.

### Budgets, sequencing and recovery

Require explicit update, elapsed-time, evaluation and output-storage limits before
automated launch. Include baseline/evaluation in the appropriate total budgets.
Time limits are cooperative at safe boundaries: the current operation and
necessary checkpoint save may finish. Report that overrun honestly. Storage
preflight must account for safe-save headroom; stop with an actionable result
rather than deleting existing artifacts when capacity is insufficient.

Use one approved configuration, not a parameter search. Initialize SFT once and
resume its exact state between segments. Verify a complete checkpoint before
each evaluation. Run compute operations sequentially under the existing
one-active-pipeline-per-user-per-host lock. UI detach never controls progress;
user stop cancels the remaining workflow, including later evaluation/training.

Persist phase transitions, budgets consumed, artifact hashes and completed
reports. On explicit recovery, verify referenced artifacts, avoid repeating
completed work, and use compatible retained runtimes for exact continuation.
An interrupted report stays incomplete. Host reboot does not automatically
restart a job; no daemon, machine provisioning or remote-control service is added.

## Future roadmap: the small in-app helper

AH01–AH04 are **deferred**, not part of FT01–FT12 implementation. The helper is a
separate Omega-owned tiny model, initially CPU-first when compute is idle. It
consumes structured reports and proposes allowed actions; it does not replace
deterministic validation, resource limits or checkpoint checks.

| ID | Dependencies | Required outcome |
|---|---|---|
| AH01 | FT11; explicit future assignment | Define structured project summaries, readiness/report inputs and allowed proposals. Route actions through deterministic validation and the existing approval screen; no arbitrary shell execution. |
| AH02 | AH01 | Curate Omega documentation, troubleshooting examples and explicitly approved workflow examples. Remove credentials/private content, keep evaluation cases separate and do not automatically use user conversations as training data. |
| AH03 | AH02; approved data/budgets | Train and evaluate the separate tiny helper for explanation and valid next-step proposals. Measure memory, latency, factual accuracy, action validity and abstention. Concurrent assistance requires separate resource qualification. |
| AH04 | AH03 | Integrate contextual help across configuration, data, training, evaluation and recovery. Show proposed changes, reasons and resource impact before approval. Keep every workflow usable without the helper. |

No base-model, assistant or helper training run follows automatically from these
task entries. The helper's initial authority is **propose, then approve**.

## Validation and completion gates

- **Training:** verify parent immutability, fresh SFT optimizer, target masks,
  supported stage transfer and unchanged rejection of incompatible exact resume.
- **Evaluation:** test reproducible cases, multi-turn history, target weighting,
  malformed output, timeouts, partial reports, changed suite hashes and missing
  human scores. Failing outputs must remain visible even if the process succeeds.
- **Automation:** exercise detach, terminal loss, recovery at every phase,
  duplicate prevention, insufficient storage, budget exhaustion and user stop
  during training and evaluation. Reconnection must not launch another worker.
- **Interfaces:** test migration/comment round trips, TUI navigation/resizing,
  human review and report inspection, headless parity and nonzero error exits.
- **Hardware:** tiny CPU tests in CI; bounded Vulkan qualification only on an
  idle A770; CUDA checks on available WSL hardware. Record hardware evidence
  separately and never interrupt a current production run to obtain it.
- **Repository:** run formatting, workspace tests and strict Clippy from
  `Code/Rust`, plus relevant feature builds. Record fresh commands/results and
  unresolved failures before changing task status.
- **Model acceptance:** FT12 additionally requires approved real data/budgets,
  threshold evidence, human review, sealed-test results, a model card and verified
  reload/recovery instructions. Toy-model regression success is insufficient.

LoRA, preference optimization, autonomous parameter search, external model judges
and ZH14 optimizations are outside this milestone. This roadmap authorizes
implementation planning, not production training, downloads, public deployment
or changes to the current run.
