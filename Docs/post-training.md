# Post-training workflow

Implementation work: 2026-10-07. See [TASKS](../TASKS.md) for current qualification
and [the real-data audit](post-training-audit.md) for unresolved pilot inputs.
Software fixture results do not accept a conversational model for real use.

Omega can run a baseline, full-model supervised fine-tuning segments and local
development evaluation under one detached worker and the existing host compute
lock. Every segment saves a complete checkpoint. The first segment transfers
weights into a **fresh optimizer**; subsequent segments resume that SFT state
exactly. No production training or downloads are implicit in configuration.

## Define the experiment

Record the conversational purpose/language, parent, approved data sources and
permissions, development and sealed partitions, human rubric, quality thresholds,
and update/time/evaluation/storage budgets. Review source-family isolation and
near duplicates separately; automated exact-overlap detection cannot prove them.
The current real pilot has not supplied these inputs or passed acceptance.

Existing schema-1 projects and saved jobs remain readable. Explicitly migrate a
project before adding post-training configuration; migration preserves comments
and does not alter old job snapshots:

```sh
omega post-training migrate /work/my-project/model.toml
```

Add `[post_training]` using the TUI TOML editor or a text editor. This illustrative
configuration is **not an approved production budget or quality threshold**:

```toml
omega_schema_version = 2 # existing root key, before any table

[post_training]
parent = "weights/base-12"
output = "experiments/sft-001" # new directory; create its parent first
purpose = "Answer short questions within the approved domain"
language = "English"
training = ["assistant/train"] # relative to paths.datasets
validation = ["assistant/validation"]
base_validation = ["base/validation"]
sealed = ["assistant/test"]
suite = "evaluation/development.json"
rubric_version = "conversation-v1"
epochs = 2
learning_rate = 0.0001
seed = 42
batch_size = 1
max_batch_tokens = 1024
segment_updates = 10
max_updates = 20
max_seconds = 300.0
evaluation_max_seconds = 30.0 # total per suite + both loss passes
max_evaluations = 3 # includes the baseline
max_output_bytes = 1073741824
min_test_pass_rate = 1.0
max_assistant_loss = 5.0
max_base_loss_increase = 0.10 # relative increase from baseline
min_human_score = 3 # each of the four dimensions, each case
```

The parent supplies architecture, tokenizer and chat controls. No base-training
selection, tokenizer refit or new base run is needed. Training backend/device and
CPU thread profile still come from `[training]`. SFT learning rate, batching,
seed, epochs and budgets come from `[post_training]`. This first workflow uses
fixed-order full-model SFT without hyperparameter search. Other ordinary pipeline
flags do not run when launching post-training.

Paths are resolved from the selected project directory. Outputs, datasets and
checkpoints may be on a mounted share; worker sockets, executable copies and job
records remain host-local. Do not put credentials into configuration or suite
files; worker credentials remain inherited environment variables.

## Development suites and readiness

A minimal versioned development suite:

```json
{
  "schema_version": 1,
  "name": "conversation-development-v1",
  "generation": { "max_new_tokens": 32, "strategy": { "kind": "greedy" } },
  "cases": [{
    "id": "greeting",
    "system": "Answer briefly.",
    "turns": [{
      "prompt": "Hello. What can you help me with?",
      "checks": [
        { "kind": "non_empty" },
        { "kind": "end_turn" },
        { "kind": "no_role_leakage" }
      ]
    }]
  }]
}
```

Check the source suite enum for the exact supported JSON names before extending
this example. Cases support multiple turns with actual generated history, exact
answers, substrings, JSON field expectations, termination, forbidden role tokens,
empty output, repeated n-grams, and an intentional context-overflow check. Seeded
sampling has fixed temperature/top-k/top-p/seed settings. Limits bound suite size,
turns and generated tokens. These checks cannot certify open-ended correctness.

```sh
omega post-training ready /work/my-project/model.toml
omega post-training plan /work/my-project/model.toml
omega post-training start /work/my-project/model.toml --yes
```

Readiness is read-only: completeness and hashes, model/tokenizer/chat agreement,
conversation roles, assistant targets, context limits, release partition roles,
and detectable exact content/group overlap. All four selections are required.
Missing chat controls require another compatible parent; Omega never mutates its
tokenizer. Readiness currently caps each partition's overlap audit at 64 MiB raw
input, 100,000 records and 200,000 units; oversized selections fail actionably.

New-stage transfer supports resumable checkpoint schemas 3–8 across builds,
preserving structural/integrity checks. Legacy schemas 1/2 require their existing
migration path; arbitrary model formats are unsupported. Exact SFT continuation
still rejects incompatible runtime/backend/build identity. CUDA remains
experimental until its separate hardware qualification passes.

## Detach, stop and recover

The TUI's post-training review launches the same worker as the CLI. Its normal job
dashboard displays update/loss/throughput/checkpoint events. **Detach** leaves the
worker running; reopening the Jobs screen reconnects without launching again.
**Save checkpoint and stop** stops the remaining workflow at a safe boundary.
During evaluation, cancellation stops between replies or metric examples and
retains an incomplete report instead of passing it.

```sh
omega jobs
omega status JOB_ID
omega stop JOB_ID
omega post-training recover /work/my-project/experiments/sft-001 --yes
```

Recovery is explicit, uses the retained executable and frozen settings, and
verifies registered reports/checkpoints and current input hashes. A verified
checkpoint receipt is persisted before the next phase. An interrupted training
attempt retains its reserved update allowance; downtime after a crashed active
operation is charged conservatively against elapsed time. Budgets are never
reset by recovery. Unregistered/incomplete evaluations never become passing
results; a recovered evaluation is charged as another attempt. Completed or
budget-exhausted experiments require a separately reviewed new experiment.
The original project configuration path must still exist for the launcher; its
edited contents are not substituted for the retained approval snapshot. Immutable
`approval.json` and `readiness.json` preserve the approved settings and data
identities. Active input hashes/inventories are rechecked between operations;
sealed payloads are excluded from those repeated checks.

Time limits are cooperative, not hard process deadlines: model/data loading, an
in-flight generation/update, and necessary saves may finish after the nominal
limit. Actual elapsed time and overruns are recorded. Storage admission reserves
estimated checkpoint headroom and checks measured usage/report sizes; unexpected
filesystem or quota failures fail the job. No artifact is automatically deleted.
Allow room for complete optimizer snapshots and reports. A reboot cannot
automatically restart a worker.

## Compare, score and select

```sh
omega post-training reports /work/my-project/experiments/sft-001
```

Every evaluation has JSON and Markdown under `reports/`, including actual model
responses, identities, settings and failures. Baseline and candidates measure
assistant-target validation loss and target-weighted base-language regression on
separate selections. Comparison requires matching suite/runtime/tokenizer/model
configuration and frozen input identity. Sealed data is structurally audited for
isolation but never trained on or evaluated in the automatic development loop.

For each case, a human scores instruction following, correctness, relevance and
coherence from 0 to 4: **0** unusable, **1** major failures, **2** mixed, **3** meets
the stated expectation with minor issues, **4** fully meets it. Document concrete
domain-specific anchors under the experiment's rubric version before the pilot.
Save this review JSON yourself; software never invents a human reviewer or scores:

```json
{
  "schema_version": 1,
  "report_sha256": "SHA256_OF_THE_EXACT_EVALUATION_JSON",
  "rubric_version": "conversation-v1",
  "reviewer": "Your name",
  "cases": [{
    "case_id": "greeting",
    "instruction_following": 3,
    "correctness": 3,
    "relevance": 3,
    "coherence": 3,
    "notes": "Explain the actual judgment here."
  }]
}
```

```sh
omega post-training review /work/my-project/experiments/sft-001 /work/my-project/experiments/sft-001/reports/evaluation-2.json /work/review.json
omega post-training promote /work/my-project/experiments/sft-001 /work/my-project/experiments/sft-001/reports/evaluation-2.json --approved-by "Your name" --output /work/selection.json --yes
```

Submit only a complete review; edit incomplete drafts before submission. Submitted
reviews are immutable. Pending/missing/failed reviews or automated gates exclude
candidates. Eligible candidates are ordered by assistant validation loss, then
fewer updates; other scores remain visible separately. Promotion verifies the
selected checkpoint again and creates a new immutable manifest containing hashes
and evidence. It never replaces a checkpoint or silently selects the latest.
Sealed-test final acceptance, model card, backup/restore evidence and actual
production quality review remain FT12 work requiring separately approved inputs.

The future helper, LoRA, preference optimization, external model judges,
hyperparameter search and ZH14 optimizations remain outside this implementation.

## Terminal UI actions

Open a project and scroll past the existing dataset inspection action to the seven
post-training actions. **Migrate project to schema 2** displays the exact file for
confirmation, then opens the TOML editor. Enter the explicit `[post_training]`
settings above and use Ctrl+S to validate and save. **Readiness** displays its
report. **Review and start** freezes the reviewed configuration; Enter starts the
detached worker. Esc cancels a review or confirmation without starting it.

**Ranked reports and results** accepts the workflow directory, initially populated
from `post_training.output`. It displays state, eligibility reasons and the full
reports with actual prompts and replies. Arrow keys and PageUp/PageDown scroll.
Use F3 for the existing jobs dashboard and F2 to return to the project.

**Submit human review** accepts a saved request JSON file path:

```json
{
  "workflow": "experiments/sft-001",
  "report": "experiments/sft-001/reports/evaluation-2.json",
  "scores": "human-review.json"
}
```

The scores file uses the complete `HumanReview` format above. Relative paths in
this request resolve beside the request file, not beside the scores file. Omega
loads and displays the exact scores, reviewer, report hash and rubric version.
Enter on the confirmation screen submits that snapshot; merely opening the file
does not submit or accept it. The workflow validates the report identity and every
case before saving an immutable review.

**Promote candidate** accepts a separate saved request JSON file path:

```json
{
  "workflow": "experiments/sft-001",
  "report": "experiments/sft-001/reports/evaluation-2.json",
  "approved_by": "Your name",
  "output": "selection.json"
}
```

These paths also resolve beside the request file. The UI requires an eligible
candidate and displays the approver, report and new output path. Enter explicitly
confirms promotion; the workflow checks the gates again before writing. The
output must not already exist. Unknown request fields are rejected.

**Recover workflow** accepts an existing workflow directory and displays its
frozen configuration and retained executable identity. Enter confirms recovery
using that executable and the remaining budget. This does not reset budgets or
make incomplete evaluations pass.

The terminal UI currently uses the TOML editor for post-training settings and
saved JSON files for human scoring and promotion; it has no dedicated score-entry
grid or graphical report browser. Brief metadata operations hold navigation until
their result arrives; q or Ctrl+C still detaches. Detached workers require Linux;
rendering and configuration tests on Windows do not qualify Linux worker recovery
or GPU execution. No UI action supplies missing real-data acceptance evidence.
