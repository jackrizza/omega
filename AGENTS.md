# Agent instructions for Omega

These instructions apply throughout this repository. Follow the user's assigned
scope; the backlog is not permission to implement unrelated features.

## Start here

1. Read [TASKS.md](TASKS.md) for task status, dependencies, and acceptance criteria.
2. Read [the training guide](Docs/omega-training.md) and
   [implementation status](Docs/implementation-status.md).
3. Read the affected crate's README, manifest, implementation, tests, and call
   sites before editing. The docs describe a dated review; verify current source
   rather than treating historical test counts or missing-feature lists as live.
4. Check for existing user/agent changes and narrower instructions. Do not
   overwrite work outside your assignment. Do not create branches or commits
   unless explicitly requested.

## Project map

The Cargo workspace is **`Code/Rust`**, not the repository root. All three crates
currently use Rust edition 2024 and have a binary named `main`.

| Area | Ownership |
|---|---|
| `Code/Rust/omega-tokenizer` | Saved tokenizer pipeline, vocabulary inspection, encode/decode, tokenizer tooling |
| `Code/Rust/omega-nn` | GPT architecture, tensor/model contracts, optimization primitives, generation |
| `Code/Rust/omega-training` | Corpus preparation, training orchestration, evaluation, checkpoints, training CLI |
| `datasets/` | Tokenizer and text fixtures; user corpora may also live here |
| `weights/` | Generated checkpoints; treat existing contents as user data |
| `Docs/` | Usage guide and implementation-status snapshot |

Dependencies flow from `omega-training` to `omega-nn` and `omega-tokenizer`, and
from `omega-nn` to `omega-tokenizer`. Avoid circular dependencies. Keep orchestration
in `omega-training`, not inside the tokenizer or model architecture.

Burn 0.18 with NdArray/autodiff is the current CPU stack. Check the actual
manifests and compatible APIs before adding dependencies; do not upgrade Burn or
switch backends as an incidental cleanup. Rust 1.93.0 on Windows was verified in
the documentation review; this is not a declared minimum supported version.

## Contracts to preserve

Unless an assigned task explicitly changes one of these contracts, preserve it.
Any intentional change needs tests, documentation, and a compatibility decision.

- **Tokenizer identity:** token IDs must match model embeddings and the output
  head. Equal vocabulary size alone does not prove compatibility. Do not change
  IDs, add guessed special tokens, or silently replace the saved pipeline.
- **Fixtures:** `datasets/test.json` is the 13-entry WordLevel test fixture, not a
  corpus or production tokenizer. Preserve it and existing fixture expectations;
  place alternative tokenizers in separately named files.
- **Causality:** a prediction cannot see future tokens. Shift next-token labels
  exactly once. Preserve prefix-invariance tests when changing attention.
- **Documents:** current corpus chunks share one boundary token, cover adjacent
  target pairs once, keep short remainders, and never create cross-file targets.
  Split train/validation by document before chunking to prevent leakage.
- **Padding:** the current model has only a causal mask. Do not introduce padded
  minibatches without attention and loss masking. Do not silently truncate data.
- **Training state:** retain one optimizer across examples/epochs in a run.
  Current CLI training starts fresh; numbered saves are not resumable snapshots.
  Never call a weights-only reload an exact training resume.
- **Checkpoints:** preserve existing weights, highest-present-suffix numbering,
  collision handling, and incomplete-save rejection. Write `COMPLETE` last.
  Changes to schemas require an explicit legacy-load/migration policy.
- **Paths:** current CLI roots refer to the compile-time checkout; library paths
  are explicit. Preserve safe dataset selection and use `/` in documented nested
  dataset names, including on Windows.
- **Errors:** library errors should be actionable, and CLI failures should return
  nonzero status. Do not hide numerical failures or claim training success from
  lower toy loss alone.

Existing limitations such as cross-case concurrent naming, missing resume state,
and configuration/model mismatch checks are recorded in `TASKS.md`. Do not
mistake the contract above for a claim that those gaps are already fixed.

## Sub-agent coordination

The coordinating agent owns task allocation and integration.

1. Choose one bounded task or subtask. Confirm its prerequisites and record its
   owner/status in `TASKS.md` before starting code changes.
2. Send the agent a self-contained assignment with: task ID, goal, source paths,
   agreed API, allowed write set, forbidden files, acceptance tests, and expected
   handoff. Agents do not automatically share conversation context.
3. Assign **disjoint write sets**, not merely different feature names. Two tasks
   touching `omega-training/src/lib.rs` are not independent implementations.
4. Agree cross-crate interfaces first. Proposed new modules in the backlog are
   design suggestions, not existing files or permission to invent public APIs.
5. The coordinator owns shared integration files by default: workspace/crate
   manifests, lockfile policy, module exports, CLI wiring, `AGENTS.md`, `TASKS.md`,
   shared READMEs, and `Docs/`. Delegate an individual file only with exclusive
   ownership. Avoid simultaneous formatter writes to another agent's files.
6. If a task needs another owner's file or an API change, stop that portion and
   request coordination. Do not work around it with duplicated implementations.
7. Reuse an existing agent session for follow-ups. Independent read-only reviews
   can run in parallel; avoid duplicated implementation work.
8. Integrate and run broader validation before marking the task done. Only the
   coordinator updates the shared task board unless explicitly delegated.

Recommended assignment template:

```text
Task ID / bounded subtask:
Goal and non-goals:
Prerequisites and agreed interface:
Read first:
Allowed writes (exact files or explicitly bounded subtree):
Do not edit:
Required tests and commands:
Handoff: changed paths, API/behavior changes, results, limitations, follow-ups
```

Do not leave placeholder implementations, TODO-only features, or disabled tests
and mark the task complete. Partial work should be handed off with an explicit
remaining checklist and status.

## Validation and safe execution

Run Cargo commands from `Code/Rust`. Start with the affected package, for example:

```sh
cargo test -p omega-training
cargo clippy -p omega-training --all-targets -- -D warnings
```

Use `omega-nn` or `omega-tokenizer` instead when those are the affected packages.
Before integration is complete:

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

A formatter check is safe during parallel work; coordinate formatting that writes
files. Add focused regression tests for changed behavior. Use temporary directories
for dataset/checkpoint tests and keep CPU models tiny. Burn seeding uses shared
backend RNG state; isolate deterministic comparisons rather than assuming parallel
seeded tests are independent. Gate platform-specific tests and report untested
platforms explicitly.

For CLI changes, verify argument parsing, error exits, and help output:

```sh
cargo run -p omega-training --bin main -- train --help
cargo run -p omega-training --bin main -- generate --help
```

Always specify `-p` for these binaries. Run end-to-end training only when needed,
with bounded runtime and tiny data. Do not launch large training jobs, download
corpora/models, or incur remote compute costs without task authorization.

Never overwrite or delete user datasets or checkpoints. Prefer temporary output
roots for tests; if a smoke test creates repository artifacts, track the exact
paths and remove only artifacts created by that test. Do not globally reset or
clean the working tree. Generated weights must not be added to version control.

For documentation-only changes, verify paths, commands, task IDs/dependencies,
and consistency with source; a full training run is unnecessary. Report what was
actually checked, failures/timeouts, and checks not run. Never reuse an older test
result as proof that the current changes passed.

## Definition of done and handoff

- The assigned acceptance criteria are met without unrelated changes.
- Tests cover the new behavior and relevant failure/compatibility paths.
- Affected code compiles; required validation passes or unresolved limitations
  are explicitly reported and the task is not marked done.
- CLI/library behavior and checkpoint/data compatibility are documented.
- The coordinator updates relevant READMEs, `Docs/`, and task status without
  presenting future work as implemented.
- Handoff lists exact changed paths, API decisions, commands/results, and remaining
  risks. Do not commit automatically.

Keep user-facing summaries concise and honor the user's requested response or
commit-message format. Sub-agent handoffs must still contain the technical detail
needed for safe integration.
