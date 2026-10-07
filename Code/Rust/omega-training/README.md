# Omega Training

Post-training library APIs: `readiness` inspects parent/data without executing a
model; `conversation_evaluation` runs versioned local suites and returns typed
complete/partial reports; `operations::execute_segment_to` trains a bounded
segment into an explicit output root; `execute_evaluation` adds cooperative
metric cancellation/deadlines. `resume::read_stage_parent` and the separate
weights-transfer initializer permit supported new stages across builds without
weakening exact resume. Project sequencing, budgets, review and promotion live
in Omega. See [the post-training guide](../../../Docs/post-training.md).

The [Omega application](../../../Docs/omega.md) adds a persistent Linux terminal workspace, detached workers, whole-workflow projects and experimental CUDA. See OA01 in the task board for current validation. CUDA uses resume schemas 7/8; CPU/Vulkan formats are preserved.

For a bounded training-time estimate before a run, see
[omega-benchmark](../omega-benchmark/README.md). It reuses this crate's dataset,
format-selection and checked update APIs and discards the benchmark weights.

## Conversational assistant implementation

Opt-in chat tokenizer preparation, explicit `--dataset-format chat`, assistant-only
loss/evaluation, `train-stage` from a verified Omega base checkpoint, exact stage
resume and a bounded local `chat` command are implemented. See
[assistant execution](../../../Docs/assistant-execution.md) for formats, limits,
schema-2 chat metadata and schema-5/6 assistant continuation. Existing text and
CPU/GPU resume defaults remain. Software implementation does not establish a
qualified A770 run, accepted corpus or useful trained assistant; those gates are
tracked separately in [TASKS.md](../../../TASKS.md).

Prepare local corpora and tokenizers, train a small CPU or optional Vulkan GPU GPT, evaluate fixed
weights, and save numbered inference and training checkpoints. This crate owns orchestration;
[omega-nn](../omega-nn/README.md) owns the model and
[omega-tokenizer](../omega-tokenizer/README.md) owns tokenizer pipelines.
See the [root README](../../../readme.md), [training guide](../../../Docs/omega-training.md),
[implementation status](../../../Docs/implementation-status.md), and
[task board](../../../TASKS.md) for broader context.

## GPU training

See [Arc A770 GPU training](../../../Docs/gpu-training.md) for release-mode
commands, device listing, schema-4 GPU continuation and hardware validation.
Build with `--features gpu` and choose `--backend vulkan --device 0`.
The unique `--bin omega-training` avoids the legacy `main` filename collision.
CPU remains the default; cross-backend exact resume is rejected.

GPU numerical validation now uses compact device reductions automatically with
`--features gpu`; no additional train flag is needed. Norm reduction order has
its own execution identity, so checkpoints from the earlier host-check GPU
trainer require their original executable for exact resume. See the GPU guide
for compatibility and `gpu_benchmark --profile` stage/targets-per-second reports.

## Quick start

### Prepared `omega-datasets` releases

Select an exact published partition, for example
`--dataset assistant-v1/release-v1/base/train`. The default `--dataset-format auto`
uses text-field JSONL for published `base` partitions and the conversation format
for published `chat` partitions. Ordinary folders still default to `.txt`; ordinary
JSONL/chat folders still need their explicit format. `model.toml` is a corpus
recipe, not a GPT model configuration; train from its completed release output.

Training, tokenizer fitting, new stages and resume accept only published `train`
partitions. Evaluation can explicitly select `validation` or `test` (reserve test
for final acceptance). Published groups must not be re-split using validation
count/ratio flags. Do not select the recipe, release or stage parent directory,
mix stages/partitions, or mix published partitions with ordinary folders.

Loading verifies `COMPLETE`, the manifest checksum/schema, and the selected
`data.jsonl` checksum/record count. Incomplete, empty, modified or augmented
partitions fail before training. Cache loading rechecks the selected source too.
Raw downloads, membership payloads and unselected partitions are not rehashed;
these checks establish selected-input consistency, not quality or authenticity.
Keep the release immutable while a command reads it. Existing checkpoints retain
their normal tokenizer/source/runtime checks and saved resolved format.

```sh
cargo run -p omega-training --bin omega-training -- train-tokenizer --dataset assistant-v1/release-v1/base/train --chat-protocol --vocab-size 4096 --output ../../datasets/assistant-v1/tokenizer.json
cargo run -p omega-training --bin omega-training -- train --dataset assistant-v1/release-v1/base/train --tokenizer assistant-v1/tokenizer.json --name base-pilot --epochs 1 --max-updates 2
cargo run -p omega-training --bin omega-training -- evaluate --dataset assistant-v1/release-v1/base/validation --checkpoint base-pilot-1
cargo run -p omega-training --bin omega-training -- train-stage --dataset assistant-v1/release-v1/chat/train --checkpoint base-pilot-1 --name chat-pilot --epochs 1 --max-updates 2
```

Use the actual checkpoint names printed by your run and explicitly chosen model
dimensions/context; these commands illustrate the connection, not a qualified
training recipe. Preparation does not guarantee that conversations fit the model's
context. Chat loading retains its 16 MiB/file, 64 MiB aggregate and 100,000-record
bounds and rejects over-context records. Chat caches remain unsupported. Base token
caches default to 16 MiB/source; their existing `--max-source-bytes` can be set
explicitly. A producer's larger preparation limits do not override trainer limits.
No silent sampling, truncation or tokenizer resizing is introduced.

The producer-to-trainer integration tests add test-only workspace dependencies.
The workspace lockfile identity therefore changes even though dependency versions
do not. Existing exact-resume guards still require a compatible recorded build;
keep the original build for older runs. This adds no checkpoint migration or
runtime compatibility bypass.

### Ordinary folders

Run from `Code/Rust`; every package names its binary `main`, so include `-p`:

```sh
cargo run -p omega-training --bin main -- train --name foo --dataset examples/greetings --dataset examples/omega --tokenizer test.json --epochs 10
cargo run -p omega-training --bin main -- generate --checkpoint foo-1 --prompt "hello" --max-new-tokens 5
```

Use the checkpoint name printed by training if `foo-1` already existed. Each
`train` invocation initializes a fresh model and optimizer; numbering does not
resume training. The included WordLevel tokenizer and text are toy fixtures.
Use the separate `resume` command to continue a new training snapshot.
Generation defaults to greedy and returns the prompt plus generated tokens. Optional
`--eos-token-id` stops at a newly generated ID; no special-token IDs are guessed.

`--dataset` accepts a folder below the dataset root; repeat it to combine folders.
Use `/` in nested selections. `--tokenizer` accepts a dataset-relative JSON path
or an absolute path. Default roots refer to the checkout at compile time. Override
with `--datasets-root PATH` and `--weights-root PATH` where applicable; relative
root overrides resolve from the working directory. `generate` and `evaluate`
take either a single `--checkpoint` name or `--latest-run` run name below the
weights root. `resume` supports the same mutually exclusive selection flags.

Model options include `--context-length`, `--d-model`, `--heads`, `--layers`,
`--d-ff`, `--epochs`, `--learning-rate`, and `--seed`. Each subcommand provides
`--help`. CPU initialization uses Burn's shared RNG; concurrent initialization
and different platforms/builds are not guaranteed to reproduce results.

## Generation sampling and checkpoint discovery

```sh
cargo run -p omega-training --bin main -- generate --checkpoint foo-1 --prompt "hello" --max-new-tokens 5 --sample --temperature 0.8 --top-k 5 --top-p 0.9 --sampling-seed 42
cargo run -p omega-training --bin main -- checkpoints list --run foo --json
cargo run -p omega-training --bin main -- checkpoints latest --run foo --mode resume --json
cargo run -p omega-training --bin main -- generate --latest-run foo --prompt "hello" --max-new-tokens 5
```

Sampling requires `--sample`. Temperature defaults to 1 and must be finite and
positive; top-k is optional and within 1..=vocabulary size; top-p is optional and
in (0, 1]. The request-local seed defaults to 42. Stable softmax probabilities are
sorted descending with token-ID tie breaking, filtered by top-k, then top-p
relative to the retained probability mass, and normalized for a draw. This does
not consume training or Burn RNG state. All sampling-only flags are rejected in
greedy mode; nonfinite logits are errors. KV caching and streaming are not added.

Inventory commands are read-only and inspect metadata without initializing tensors.
They report complete inference, resumable, legacy/unverified, incomplete and invalid
entries with reasons. `--max-entries` defaults to 100000 and counts all root entries;
`--max-metadata-bytes` defaults to 16 MiB aggregate header bytes per candidate.
These are catalog limits, not general load or process-memory limits.

Latest selection uses the highest numeric suffix for one run and operation, not
timestamps. It reports skipped incomplete entries and (for resume) inference-only
entries, rejects equal-number aliases and corrupt completed candidates, and never
retries an older save after loading fails. `generate`, `evaluate` and `resume`
revalidate the selected path through their normal loaders. Their `--latest-run`
uses the default catalog limits. Existing explicit checkpoint loading is retained.

The JSON `inspection` field reports `metadata_only_payloads_unverified`: catalog
inspection does not verify tokenizer/model payloads or certify resume runtime/data
compatibility. A corrupt payload can pass catalog selection and fail loading.
Symlink candidates are rejected; Windows checks reject all reparse points,
including some cloud placeholders. The weights root must remain trusted during
inspection/loading. Discovery is not an atomic filesystem snapshot. There is no
checkpoint deletion, schema change or new durability guarantee in these commands.

## Documents, splits, and evaluation

`train`, `evaluate`, and `train-tokenizer` accept `--dataset-format auto|text|jsonl|chat`
(tokenizer fitting requires text, not conversation records). On ordinary folders,
the default `auto` resolves to `text`, selecting exact lowercase `.txt` files, one document per file.
`jsonl` selects exact lowercase `.jsonl` files, with one JSON object per physical
line and a required string `text` field:

```json
{"text":"hello world", "source":"optional metadata"}
{"text":"this is a test"}
```

Extra fields are ignored. Blank lines and blank text are skipped; malformed JSON,
missing text, or non-string text fails with the source path and physical line.
Each JSONL record is a separate document with logical ID
`relative/file.jsonl/@record-N`, where `N` is the one-based physical line. These
IDs are not filesystem paths. Adding/reordering lines changes identities; records
from one file can belong to different split partitions.

Selections are sorted and deduplicated, including overlapping folders. Every
selected folder must contain a file of the requested format. Paths with traversal
or symlinks are rejected, and text must be UTF-8. Model training/evaluation require
each nonblank document to encode to at least two tokens. No text is silently
trimmed to fit the model. Actual padding and reported truncation overflow are
rejected; disable tokenizer truncation because upstream overflow reporting does
not detect every truncation case.

Chunks contain at most `context_length + 1` tokens and share one boundary token.
Every within-document adjacent target pair appears once; short remainders stay,
and targets never cross document boundaries. Context resets at each chunk.
Training defaults to fixed order and one example per update, with optional seeded
shuffle, weighted sampling and masked minibatches. Adam persists across updates.
The default keeps examples in memory; optional
token caches supply indexed examples without retaining the whole token corpus.

Use either `--validation-count N` or `--validation-ratio R`, plus `--split-seed`,
to split whole documents before chunking. Ratios use `floor(documents * R)`;
positive ratios producing no validation documents and splits leaving no training
documents fail. No split option, or an explicit zero, keeps all documents in
training. Selection order and chunk length do not change seeded membership.
Document identity does not detect duplicated content under different IDs.

```sh
cargo run -p omega-training --bin main -- train --name heldout --dataset examples --epochs 2 --validation-count 1 --split-seed 42 --metrics-jsonl training-metrics.jsonl
cargo run -p omega-training --bin main -- evaluate --checkpoint heldout-1 --dataset examples --validation-count 1 --split-seed 42
```

Training evaluates the validation partition after each epoch. `evaluate` performs
no optimization; without split options it evaluates all selected documents.
With split options it evaluates only the requested validation partition, rejecting
an empty one. Reuse the original corpus and split recipe to reproduce membership;
the command does not establish that arbitrary supplied data was held out.

Evaluation reports target-weighted mean cross entropy and perplexity using one
fixed model. Non-finite logits/loss and perplexity overflow are errors. Training
losses instead describe the model immediately before each update. Neither a lower
toy training loss nor a low unknown-token rate establishes model quality.

## Progress and metrics

Live output reports epoch, completed updates/targets, pre-update loss, and weighted
epoch summaries. `--quiet` suppresses progress/loss output but still prints the
saved checkpoint path. `--metrics-jsonl PATH` creates a new file relative to the
working directory or at an absolute path; existing files are never overwritten.

JSON Lines schema 1 emits `training_started`, committed `update` records,
per-epoch `validation` records when enabled, and `training_complete`. The last
update in an epoch includes its target-weighted loss summary. Each record is
flushed. Logging failure stops the run after any already-committed update and
prevents the later checkpoint save. An interrupted/failed log may have a partial
last line and no completion event. `training_complete` means training/evaluation
finished; CLI terminal events follow the final save. Earlier valid checkpoints
remain available if subsequent metrics writing fails.
Bounded runs stopping before their target use `segment_complete`; resumed runs
start with `training_resumed` and retain cumulative update/target counters.
Cooperative stops emit `training_interrupted`. Update records add example count,
effective learning rate, global gradient norm and whether clipping occurred.

## Train and inspect a tokenizer

```sh
cargo run -p omega-training --bin main -- train-tokenizer --dataset examples --output ../../datasets/byte-bpe.json --vocab-size 512 --min-frequency 2
cargo run -p omega-training --bin main -- coverage --tokenizer byte-bpe.json --text "Hello, Omega!"
```

Tokenizer training uses case-preserving byte-level BPE with the full 256-byte
alphabet and a ByteLevel decoder. It adds no normalization, prefix space,
BOS/EOS, unknown token, padding, or truncation. Vocabulary size must be at least
256 and is an upper bound; a small corpus may yield fewer entries. Minimum
frequency must be positive. Text documents are supplied separately, and training
statistics remain in memory. Repeated training is not promised to assign the same
IDs: save and freeze a tokenizer before training compatible model weights.

`--output` is a new path relative to the working directory or absolute; its parent
must exist. Existing entries are never replaced, including the fixture tokenizer.
A failed write can leave a partial new file. To train a GPT with the new pipeline,
pass `--tokenizer byte-bpe.json` to a fresh `train` run.

`coverage` prints schema-1 JSON with `token_count`, `unknown_count`, and
`unknown_rate`. Supply `--unknown-token-id` only when its meaning is known for
that tokenizer. Without it, unknown metrics are null; an empty encoding has a
null rate. Coverage rejects configured truncation and actual padding. Full byte
coverage does not measure segmentation efficiency or language-model quality.

## Saving and loading

New checkpoints contain `model.mpk`, `config.json`, `tokenizer.json`,
`manifest.json`, and `COMPLETE`. The schema-1 manifest is authoritative for model
dimensions and stores the full effective tokenizer pipeline's canonical-JSON
SHA256 identity. Loading rejects conflicting config files, changed tokenizer
pipelines (even at equal vocabulary size), malformed metadata and unknown schemas.

The CLI records selected folders, format, document/partition IDs, split policy
and seed, training settings, and available build identity. Document fingerprints
are SHA256 over each complete unchunked token sequence encoded as little-endian
u32 IDs (`token-ids-le-u32-v1`); they are not hashes of raw source bytes. Tokenizer
identity separately pins the encoding pipeline. Build identity includes the
workspace lockfile hash; unavailable values are explicit null. Library saves
without provenance retain null dataset/training/build metadata.

Metadata is written before the versioned completion marker, which is written
last. Losing the manifest cannot silently downgrade a new checkpoint. Legacy
checkpoints with an empty `COMPLETE` and no manifest remain loadable with no
historical identity/provenance verification. Existing model record fields are
unchanged. Supplied metadata does not authenticate an arbitrary model's historical
relationship to a tokenizer. Inference-only saves have no model checksum;
training snapshots add integrity hashes, but neither format has a signature.

Numbering chooses the maximum existing suffix plus one. New names use ASCII
lowercase (`Foo` saves as `foo-N`); mixed-case legacy names and occupied files
still count. Atomic directory creation coordinates updated concurrent writers.
Older binaries may race on case-sensitive storage. Failed saves can leave an
incomplete numbered reservation, which still occupies its suffix and will not
load. Existing checkpoints are never overwritten or renamed.

The inference save APIs retain that format. New CLI training saves add optimizer
and cursor state as described below. Use trusted compatible checkpoints.
Completion is not a power-loss durability guarantee; generated weights are ignored.

## Bounded runs and resume

Global `--cpu-threads N` and `--matmul-threads 1|2|4` configure fresh-process
compute pools; omitted values inherit existing defaults. Resume schema 3 records
their execution profile and rejects mismatches. Legacy schema-1/2 migration
requires unset pool variables and default kernels. See
[CPU controls and compatibility](../../../Docs/cpu-performance.md#explicit-startup-controls)
for the exact policy and benchmarks.

```sh
cargo run -p omega-training --bin main -- train --name partial --dataset examples --epochs 2 --max-updates 1
cargo run -p omega-training --bin main -- resume --checkpoint partial-1 --name continued --epochs 2
```

`--max-updates` is a positive cap for this invocation, followed by one final save.
`resume --epochs` specifies total target epochs including previous progress. It
restores the saved dataset/split recipe, tokenizer, configuration, learning rate,
model, Adam, cursor and partial epoch loss; it allows no optimizer/model overrides.
Dataset and weights roots may be supplied explicitly. Original snapshots are
never changed; `--name` chooses a new numbered output. Old inference-only
checkpoints cannot resume.

Training snapshots include `optimizer.mpk`, `resume.json`, and `resume.sha256` before
`COMPLETE`. Integrity checks cover model, optimizer and state bytes; restored
parameter IDs, moments/shapes/step counters and source/cursor consistency are
validated. Exact f64 state uses bit representations. Continuation supports the
NdArray f32 CPU path on a compatible build/platform, and optional schema-4
Vulkan continuation on the qualified A770 with matching execution identity.
See the [GPU guide](../../../Docs/gpu-training.md) for that separate contract.
Sampling policy,
batching and optimization settings are saved; a separate versioned PRNG regenerates
epoch orders. No backend RNG state is restored. Schema 3 additionally requires the
CPU execution profile. Schema-1 snapshots migrate to default controls with a
narrowly allowed prior lockfile identity; schema-1/2 migration requires unset
thread-pool environment variables and the default kernels. See the guide for the
exact compatibility policy. Cross-platform/toolchain equivalence is not promised.

## Save intervals and training controls

`train` and `resume` accept positive `--save-every-updates` and `--save-every-epochs`
intervals, using cumulative counts. They retain a final save and deduplicate
coincident boundaries. Ctrl+C/Ctrl+Break on Windows and SIGINT/SIGTERM/SIGHUP on
Unix set a stop flag. The training thread finishes the current update/observer,
saves the committed state and exits nonzero if the target was not reached.
Forced termination cannot guarantee a final save. Existing periodic saves survive
later errors; incomplete writes are never loaded.

New training accepts `--shuffle`, or repeated `--dataset-weight FOLDER=INTEGER`
for every selected, nonoverlapping training folder. Weighted sampling draws a
group by positive weight then a member uniformly, with replacement; optional
`--samples-per-epoch` sets draw count. Validation is excluded. Shuffle visits
each example once. Both use `--seed`, with saved policy/epoch/cursor for resume.

`--batch-size` groups examples without crossing their target boundaries. Final
partial batches remain; `--max-batch-tokens` caps padded input positions. Explicit
right-padding masks attention and loss, and losses count only real targets.
`--gradient-clip-norm` applies one global norm bound; `--warmup-updates` linearly
increases learning rate to its base value. Defaults disable both controls.
One minibatch is one optimizer update, with no gradient accumulation. Gradients
and candidate parameters/Adam state are checked before committing any state.
Resume restores all controls without overrides. Sampling retains epoch indices
and per-source target lengths; weighted-state validation replays elapsed epochs.

## Bounded token caches

```sh
cargo run -p omega-training --bin main -- prepare-cache --dataset examples --context-length 64 --output ../../example-cache
cargo run -p omega-training --bin main -- train --name cached --dataset examples --context-length 64 --cache ../../example-cache --epochs 2
```

Creation requires a new directory with an existing parent. Opening verifies raw
source bytes/membership, tokenizer/preprocessing identity, context/split settings,
limits, completion and token-file hashes. Stale or incomplete caches fail and are
never rebuilt automatically. `resume --cache PATH` supports the same ordered
examples as an eager snapshot. Relocation with identical relative sources works.

Explicit limits cap source bytes, tokens per document, document/visited-directory
entry counts and manifest bytes; see `prepare-cache --help`. Preparation retains
one capped source file including parsed JSONL record text, one token document and
capped metadata. Reads retain one capped token document and a chunk, rechecking
the file hash. Token limits are checked after encoding; serialized-manifest limits
exclude parsing overhead. These are input/count limits, not an RSS guarantee.
Tokenizer scratch/model memory is not hard-bounded. Large individual
inputs are rejected, never truncated; there is no arbitrary-length streaming
tokenizer. Limits/settings must match when opening. Cache data must not be edited
during use; per-read integrity errors stop before the corresponding update.
The limit flags apply to train/resume only with `--cache`; eager loading ignores them.

## Library and validation

Library callers can use `release::inspect_partition` to verify and identify a
published partition. Shared source discovery checks release integrity and rejects
release-parent traversal, including cache loads. CLI format inference and fitting
policy are orchestration rules; callers constructing their own `DocumentCorpus`
or `ExampleSource` remain responsible for retaining prepared group/partition
boundaries and selecting the proper objective.

The bounded release benchmark and CPU thread-pool audit are documented in
[CPU performance](../../../Docs/cpu-performance.md). Use measurements from the
current build; debug tests and toy loss reductions are not performance evidence.

- `dataset::load_text_documents` and `load_document_corpus_with_format` share
  corpus selection for raw text and encoded documents. `load_document_corpus`
  and `build_training_set` preserve the original text-format defaults.
- `split_document_corpus` retains document identities and partition chunk ranges.
  `TrainingSet.files` counts unique source paths, not JSONL document records.
- `TrainingSession` owns one model, Adam state and immutable dataset across
  `step()`/`advance_updates()` calls. Events occur after committed updates; an
  observer error leaves that update committed and resumes at the next example.
  Zero updates is a no-op. `inference_model()` snapshots current weights.
- `train` preserves the convenience API, copying its borrowed set into a fresh
  session. Cloning a session forks in-memory state. `TrainingSession<S=TrainingSet>`
  accepts indexed `ExampleSource`s through `from_source`, using the same loop.
- `evaluate` returns fixed-model `EvaluationMetrics`; `JsonlMetrics` writes events
  to a caller-supplied writer without hardcoded library stdout.
- `checkpoint::save_checkpoint_with_metadata` adds explicit provenance;
  `save_checkpoint` retains its signature with null provenance. `load_checkpoint`
  returns model/config/tokenizer; `checkpoint::read_checkpoint_manifest` reads
  metadata without loading tensors and returns `None` for legacy checkpoints.
- `save_training_checkpoint`/`load_training_checkpoint` persist and validate
  training state separately from inference-only loading.
- `cache::{create_token_cache, open_token_cache}` return indexed partitions with
  fallible examples and `iter_from` cursors; eager and cached ordered-example
  identity uses the same length-prefixed token-ID digest.

Run validation from `Code/Rust`:

```sh
cargo test -p omega-training --locked --jobs 1
cargo clippy -p omega-training --all-targets --locked --jobs 1 -- -D warnings
```

See the [training guide](../../../Docs/omega-training.md) for detailed contracts
and the [task board](../../../TASKS.md) for current implementation and verification
status. These commands are validation instructions, not a claim about a run.
