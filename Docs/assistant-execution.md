# Assistant implementation and execution gates

Updated 2026-09-30. This report accompanies [zero-to-hero.md](../zero-to-hero.md)
and the ZH task board in [TASKS.md](../TASKS.md). The user authorized ZH01–ZH13
implementation with sub-agents; ZH14 remains excluded.

## Actual environment and data inventory

The accessible executor is Windows, Rust/Cargo 1.93.0, with approximately 31 GiB
host RAM, an NVIDIA RTX 5070 Laptop GPU and AMD integrated graphics. The checkout
at `Z:/Omega` is a network share. This Windows executor has no A770. The supplied
SSH settings now reach the intended Linux host, which sees the same source files
on SMB, an Intel Arc A770 (`8086:56a0`), approximately 64 GB RAM and Rust/Cargo
1.97.1. Its kernel is `7.1.8-arch1-3`. Current assistant-path hardware regression
results are recorded below separately from the earlier T20 qualification.

The user confirmed that the A770 is on a remote desktop with access to this SMB
share. The root `.env` now provides local connection, remote repository-mount,
output-path and bounded-pilot fields. It is ignored by Git. SSH is preferred for
remote command execution; if only RDP is available, identify it with the transport
field so access can be assessed. Creating this file does not establish a remote
connection by itself; the subsequent SSH inspection succeeded. Values are
configuration data, not shell code; credentials must not
be copied into reports or command output. This is an agent handoff template, not
automatic `.env` loading by the Rust CLI or the Python experiment recorder.

The existing nonfixture corpus contains two economics texts, totaling 475,531
bytes and approximately 74,509 whitespace words, plus two source PDFs. There are
also two tiny test texts and two tokenizer JSONs. No conversation demonstrations,
source-permission manifest or reviewed train/validation/test release was found.
Two source documents alone cannot establish three disjoint source-group
partitions. No user corpus or checkpoint was modified by implementation work.

## Decisions still required before actual model training

- First assistant language, subject scope and expected conversation lengths.
- Permitted corpus and conversation demonstrations, including source-family
  assignments and data-use review. Existing texts are not assumed licensed merely
  because they are in the checkout.
- Confirm the selected A770 runtime through current hardware regression tests.
- Maximum pilot elapsed time/update count, checkpoint storage, and a separately
  recorded total budget for base pretraining and conversation training.
- Development/closed-test prompt sets, human scoring thresholds and acceptable
  regression criteria, frozen before comparing trained models.

These are unresolved acceptance inputs, not completed ZH01/ZH02 deliverables.
Tiny generated test fixtures validate software only. They are never substituted
for a production corpus, held-out quality evidence or a trained assistant release.

## Implemented software contracts

### Tokenizer and conversation records

`train-tokenizer --chat-protocol` is opt-in, reserves four versioned control
tokens and requires vocabulary at least 260 including those controls. Default
byte BPE and the 13-entry fixture remain unchanged. Freeze the tokenizer before
base initialization; token additions/resizing after pretraining are unsupported.

`omega-chat-v1` uses system/user/assistant role IDs and a shared end-turn ID.
An optional initial system message is followed by alternating user/assistant
messages. Training records end assistant; inference history ends user and the
formatter appends the assistant role. Literal control-token spellings inside
content encode as ordinary text, not role boundaries. No automatic base-document
end token or additional end-of-conversation token is inserted.

Use `--dataset-format chat` for explicit conversation JSONL:

```json
{"schema_version":1,"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello!"}]}
```

Each complete record is one example. Over-context records fail; answers are not
silently truncated or detached from their prompt. Nonzero automatic chat splits
are rejected: prepare reviewed source-group partitions and evaluate the external
validation folder separately. Ordinary text-field JSONL retains its old meaning.
Chat input is eager, capped at 16 MiB/file, 64 MiB aggregate and 100,000 records;
these are input bounds, not a process-memory guarantee. `--cache` is rejected for
chat because token-only caches cannot preserve the new target masks.

User/system and prior assistant tokens remain causal context. Loss supervises
assistant content and end-turn targets only, including in batch size one. Input
padding validity is independent of target eligibility. Evaluation/counters use
supervised targets; these counts should not be mistaken for all compute tokens.

### Persistence and stage transitions

Plain inference manifests remain schema 1. Chat-capable tokenizer saves use
manifest schema 2 with required protocol version and resolved IDs, checked against
the full saved tokenizer identity. Existing legacy/plain loaders remain supported;
the chat command rejects missing or mismatched protocol metadata.

Unmasked CPU/GPU continuation remains resume schema 3/4. Assistant-target sources
use resume schema 5/6 with an explicit objective; ordered source identity includes
IDs, masks, source record IDs and protocol/objective. Old resume schemas cannot
be used to restore an assistant-target source. Cache and legacy identities remain
unchanged. Catalog recognizes new schemas but still inspects metadata only.

`train-stage --checkpoint NAME` starts fresh Adam, sampler and counters from our
verified base weights on explicitly selected chat data. It checks the parent's
training-state checksum and model checksum, requires a compatible chat protocol,
and records parent name/model/manifest/resume hashes. It preserves parent files.
The initial implementation requires parent metadata accepted by the normal
current-runtime reader; it is not a compatibility bypass or third-party import.
`resume` continues the saved stage's data/settings and retains its parent lineage.

### Local interaction

`chat --checkpoint NAME --prompt TEXT` produces one reply; omit `--prompt` for
interactive input. `/reset` clears history and `/exit` ends the interaction.
`--system` supplies an optional initial instruction; generation sampling flags
retain their current meaning. Output displays assistant text only. End-turn
stops generation; an unexpected generated role token is an error. Context overflow
is rejected without mutating history. A token-budget cutoff is reported. There is
no KV cache, streaming or unlimited memory in this command.

## Example commands after the data/runtime gates pass

Run Cargo from `Code/Rust`, use the uniquely named binary, and never rebuild a
binary while another process is using it. These are usage examples, not completed
training runs. Prepared data must exist; choose new tokenizer/run/log names.

```sh
cargo run -p omega-training --bin omega-training --release --locked -- train-tokenizer --dataset assistant-v1/base/train --chat-protocol --vocab-size 4096 --output ../../datasets/assistant-v1/tokenizer.json
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- train-stage --backend vulkan --device 0 --checkpoint BASE_CHECKPOINT --name assistant-sft-pilot --dataset assistant-v1/chat/train --epochs 1 --max-updates 20 --learning-rate 0.0001 --save-every-updates 10 --metrics-jsonl sft-pilot-01.jsonl
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- evaluate --backend vulkan --device 0 --checkpoint SFT_CHECKPOINT --dataset assistant-v1/chat/validation --dataset-format chat
cargo run -p omega-training --bin omega-training --release --features gpu --locked -- chat --backend vulkan --device 0 --checkpoint SFT_CHECKPOINT --prompt "Hello" --max-new-tokens 32
```

Use the roadmap's bounded base-training recipe before `train-stage`, with the
final chat tokenizer. `BASE_CHECKPOINT`/`SFT_CHECKPOINT` are explicit replacement
names. Model capability and supported hardware must be demonstrated separately
from command parsing or software regression success.

## Validation and remaining execution work

The final **current four-crate checkout** passed 254 tests on Linux/Rust 1.97.1:
`cargo test --workspace --locked --jobs 2`,
`cargo clippy --workspace --all-targets --locked --jobs 2 -- -D warnings`, and
`cargo fmt --all --check`. The new dataset-crate lockfile correctly rejects the
old pre-GPU migration; its test now reflects that restricted compatibility while
independently checking platform mismatch. Production migration guards did not
change. The reviewed-successor positive branch also passed in the preserved
three-crate Windows copy. Older saved runs are not automatically compatible with
the new workspace lockfile; never edit manifests to force continuation.

The `omega-datasets` coordinator separately reported the final Windows/Rust
1.93.0 four-crate rerun: 251 harness tests passed, strict all-target workspace
Clippy passed, and formatting passed. This is that chat's integration evidence;
the Linux and A770 results above were run directly by the assistant coordinator.

Current Windows validation used a source-matched copy of the original three Rust
crates and their pre-`omega-datasets` workspace manifest/lockfile, avoiding another
chat's concurrent workspace integration. `cargo test --workspace --locked --jobs 1`
passed 237 tests; `cargo clippy --workspace --all-targets --locked --jobs 1 -- -D warnings`
and `cargo fmt --all --check` passed. The GPU-feature all-target compile also passed.
A Windows canonical-path assertion was corrected, and compatibility tests verify
that ordinary vocabulary spellings of chat controls still save plain schema-1
checkpoints. After the later Vulkan CLI routing fix, all three CPU assistant CLI
tests and `cargo clippy -p omega-training --features gpu --all-targets --locked
--jobs 1 -- -D warnings` passed. Do not treat the isolated three-crate results as
the new four-crate workspace results.

On the actual Linux A770, with the current four-crate lockfile, both tests in
`gpu_masked_training` passed (8.29 seconds): assistant-target batching/evaluation/
exact resume and base-to-conversation-stage initialization/resume/parent retention.
Adapter index 0 reports Intel Arc A770, Mesa `26.1.8-arch1.1`. These tiny debug
regressions establish those software paths, not capacity or useful model quality.
The first actual GPU chat CLI test exposed an omitted command in the global backend
allowlist. Both `chat` and `train-stage` are now routed to Vulkan; the new ignored
CLI tests cover both paths. Their A770 rerun passed both tests (9.35 seconds),
including real CLI base training, new-stage initialization, evaluation, exact
stage resume and chat stopping/reset/context/role checks.

Python preparation passed 17 tests on Windows with one symlink-privilege skip,
and all 18 on Linux. Experiment recording passed 14 Windows tests, including
descendant cleanup. Its first Linux run exposed symlinked interpreter paths in
the test fixtures; the fixtures now resolve physical paths without relaxing
production symlink rejection. The corrected Linux suite passed all 14 tests,
including timeout/descendant cleanup. Both platforms are now covered.

The user restored the temporarily blank `.env` connection values; SSH and the
hardware reruns succeeded. No credentials are reproduced here. Production data
selections, scope/rubric and pilot/longer-run budgets are still unprovided.
No production tokenizer, base model, trained assistant or sealed-test report has
been produced. ZH14 has not been implemented.

## Implementation paths

- Tokenizer: `Code/Rust/omega-tokenizer/src/{lib,chat,training}.rs` and
  `tests/chat.rs`.
- Training: `Code/Rust/omega-training/src/{lib,assistant,dataset,batching,trainer,
  evaluation,checkpoint,resume,checkpoint_catalog}.rs` and `src/bin/main.rs`.
- Training regressions: `Code/Rust/omega-training/tests/{assistant,assistant_cli,
  checkpoint_catalog,masked_training,gpu_masked_training}.rs`.
- Preparation/experiment tooling: `Code/Python/prepare_assistant_data.py`,
  `test_prepare_assistant_data.py`, `assistant_experiment.py`, and
  `test_assistant_experiment.py`.
- Documentation/status: `zero-to-hero.md`, `TASKS.md`, this report,
  `Docs/{omega-training,implementation-status,gpu-training}.md`, and the Python,
  tokenizer and training READMEs. Root `readme.md` links the original roadmap.
- Local access handoff: root `.env` and its `.gitignore` exclusions. The dataset
  crate, its manifest/lock integration and container work belong to other chats.

No commits or branches were created. Existing datasets and checkpoints were
preserved; all test model/data outputs used temporary directories.
