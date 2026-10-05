# Hugging Face corpus preparation

`omega-datasets` prepares the base and conversation partitions described in
[zero-to-hero.md](../zero-to-hero.md). It is acquisition/preparation tooling;
human corpus review and the ZH02 release gate remain separate work.

## Folder workflow

Create `datasets/assistant-v1/model.toml`, or generate a starter with:

```sh
# From Code/Rust
cargo run -p omega-datasets --locked -- init assistant-v1
cargo run -p omega-datasets --locked -- check assistant-v1
cargo run -p omega-datasets --locked -- plan assistant-v1
```

Edit the recipe before invoking `build`. `check` performs local validation;
`plan` reads Hub metadata but does not download corpus files. The CLI uses the
[Hub API](https://huggingface.co/docs/hub/api) and commit-pinned
[repository file downloads](https://huggingface.co/docs/hub/datasets-downloading).
It does not run a repository's Python dataset loader. Select repository files
explicitly using glob patterns; a Hub dataset configuration/split generally
corresponds to a directory or filename prefix in its Files tab.

```toml
schema_version = 1
output = "release-v1"
seed = 42

[split]
train = 0.90
validation = 0.05
test = 0.05

[[sources]]
id = "base-text"
repo = "YOUR_ORG/YOUR_DATASET"
revision = "main"
files = ["data/train-*.parquet"]
format = "parquet"
language = "en"
permitted_use = "Record the reviewed license and intended permitted use"
group_column = "document_id"
mapping = { kind = "text", column = "text" }

[[sources]]
id = "conversations"
repo = "YOUR_ORG/YOUR_CHAT_DATASET"
revision = "main"
files = ["data/train-*.jsonl.gz"]
format = "jsonl"
language = "en"
permitted_use = "Record the reviewed license and intended permitted use"
group_column = "conversation_family"
mapping = { kind = "messages", column = "messages" }
```

The placeholders above must be replaced with your selected repositories and
their actual column/file names. Every glob must match; overlapping globs within
one source are deduplicated and files are sorted. `*` stays within a directory;
`**` can cross directories. `revision` defaults to `main`; it may be a tag, branch
or full commit. A build resolves a repo/ref once and pins subsequent sources using
that same reference. A new `plan` or build may see a newer branch head: use the
saved `model.lock.toml` to replay the pinned files and commits.

Supported formats: `parquet` (including common Snappy/Zstd/Gzip/LZ4/Brotli codecs),
`jsonl` (one object per nonblank physical line), and `json` (an array of objects).
JSON/JSONL filenames ending in `.gz` are decompressed. CSV, Arrow IPC, text dumps,
ZIP/TAR archives, remote code execution, images, audio and tool-call messages are
not supported. Parquet rows use the Apache parquet crate's
[record reader](https://docs.rs/parquet/57.3.1/parquet/record/struct.Row.html).

## Mapping records

Fields are **top-level, case-sensitive names**, not JSONPath expressions. Extra
upstream fields are ignored. Output content keeps its original Unicode, casing
and whitespace. Base records with blank text are counted and skipped. All other
missing/wrongly typed fields and malformed records fail with source/row context.

- `kind = "text"`: `column` is a required string. Output is `{"text":"..."}`.
- `kind = "messages"`: `column` is an array of role/content messages. Output is
  `{"schema_version":1,"messages":[{"role":"user","content":"..."},...]}`.
- `kind = "instruction"`: `prompt_column` and `response_column` are strings.
  Optional `input_column` appends a nonblank input to the prompt with two newlines;
  optional `system` supplies a constant initial system instruction.

For ShareGPT-style conversations:

```toml
mapping = { kind = "messages", column = "conversations", role_field = "from", content_field = "value", user_role = "human", assistant_role = "gpt", system_role = "system" }
```

For instruction/input/output data:

```toml
mapping = { kind = "instruction", prompt_column = "instruction", input_column = "input", response_column = "output" }
```

Chat validation permits an optional initial system message, followed by alternating
user/assistant turns, ending with an assistant. Empty messages and unknown roles
are rejected. The output matches the current schema-1 conversation loader in
`omega-training/src/assistant.rs`; text-field JSONL remains unchanged. Preparation
does not insert chat delimiters, assign tokenizer IDs, truncate conversations or
check token/context lengths. Freeze a compatible chat tokenizer and check lengths
with the training tools afterward.

## Groups, overlap and partition assignment

Ratios must be finite, nonnegative, sum to 1, and include positive `train`.
Each complete document/conversation is a unit. `group_column` identifies its source
family using a nonblank string or number; alternatively, `group` puts the entire
source in one named family. Group strings and numeric values are distinct.
`group_namespace` defaults to the repository name; explicitly share it across
repositories that use the same family IDs. If no grouping is supplied, independent
records remain separate unless exact content overlap connects them.

Before assigning any partitions, preparation builds connected groups across all
base/chat sources using:

1. Explicit source-family identifiers.
2. Equal full normalized record hashes. Duplicate records within each stage are
   removed, retaining the first in recipe/file/row order and recording its aliases.
3. Equal normalized base text, full user or assistant messages, or all non-system
   conversation messages joined together. Related records stay together even when
   their schemas differ; base and chat records are retained in their own stages.

Audit normalization is Unicode NFKC followed by whitespace collapsing. It does
not change saved payloads. This conservative policy can join many records through
common short replies/prompts. Generic system messages alone do not join groups.
Substring, near-duplicate, paraphrase and semantic overlap are not detected.
Review source families, quality samples and base/SFT evaluation overlap before
accepting a release; the tool cannot infer all related editions or passages.

For automatic splits, each connected group is assigned by SHA256 of a versioned
prefix, the seed as little-endian u64, and its smallest normalized record hash.
The first 53 hash bits produce a value in `[0,1)` compared to cumulative ratios.
Ratios are expected group proportions, not exact record counts or stratification.
The same recipe and source bytes reproduce membership across machines. Changing
data can merge groups or change their hash, so create a new release and re-audit.

To preserve an upstream held-out split, create a source with its own `files` and
`partition = "validation"` or `"test"`; use `partition = "train"` for its training
source. A fixed partition applies to the whole connected group. Conflicting fixed
partitions fail, including conflicts introduced by a duplicate that would otherwise
be removed. No held-out record is silently reassigned. For each represented stage,
every positive-ratio partition must contain at least one retained record; otherwise
the build fails. Add independent groups, adjust the recipe or deliberately change
the ratios/seed. No group is broken merely to fill an empty partition.

## Limits and publication

The optional `[limits]` table has these defaults:

| Field | Default | Meaning |
|---|---:|---|
| `max_download_bytes` | 1073741824 | Aggregate downloaded bytes, counting repeated selections across sources |
| `max_file_bytes` | 268435456 | Per downloaded file; also expanded JSON/JSONL input bytes |
| `max_record_bytes` | 1048576 | JSONL physical line, compact parsed row and normalized output |
| `max_records` | 100000 | All parsed rows across sources, before blank/duplicate removal |
| `max_normalized_bytes` | 536870912 | Aggregate compact normalized output before duplicate removal |
| `max_files` | 100 | Total selected files across sources |
| `timeout_seconds` | 300 | Per HTTP request, including download; connect timeout is 30 seconds |

All are positive hard input/count limits, not sample sizes. A limit breach fails;
no rows are silently truncated. `plan` can check only known file sizes/counts, not
row counts, expanded size or quality. The starter TinyStories shard is a real
example but exceeds the default record budget: edit file selection or raise budgets
explicitly before downloading. No source data was downloaded to choose that example.

Downloads stream to disk. Normalized records, group indexes and hashes stay in
memory within the configured record/content limits; JSON arrays are parsed as a
whole, and Parquet decoding can allocate pages/rows before size checks. These
limits are not a process RSS guarantee. This first implementation targets bounded
corpus preparation, not arbitrary multi-terabyte streaming. There is no download
resume, retry loop, shared cache or automatic deletion of failed output.

```sh
cargo run -p omega-datasets --release --locked -- build assistant-v1
```

Produces a new directory:

```text
datasets/assistant-v1/
  model.toml
  release-v1/
    model.lock.toml
    raw/<source-id>/000000.download
    base/train/data.jsonl
    base/validation/data.jsonl
    base/test/data.jsonl
    chat/train/data.jsonl
    chat/validation/data.jsonl
    chat/test/data.jsonl
    membership.jsonl
    manifest.json
    COMPLETE
```

All six partition files exist; an unrepresented stage has empty files. Raw files
retain original bytes with numbered local names; `manifest.json` maps them to their
repository paths and pinned commits. The manifest includes source permissions,
language, mappings, counts and SHA256 hashes. `membership.jsonl` records each
nonblank source row's ID, connected group, split, normalized/payload hashes,
duplicate representative and output row. Blank rows are aggregate-counted.

`COMPLETE` contains a schema label and manifest hash and is written last, after
flushing output files. Failed builds retain an incomplete directory and never
receive a success marker. Existing output directories and files are never replaced;
choose a fresh `output` name to retry. This is not crash durability, cryptographic
authentication, license approval or quality certification. Keep filesystem paths
stable during a build. Preserve completed releases and their recipes/hashes.

To reproduce, copy `model.lock.toml` into a fresh dataset folder as `model.toml`
and build there (or set a fresh output name). It pins source commits and files;
keep the tool version and compare source/output hashes too.

## Training compatibility

Select only `assistant-v1/release-v1/base/train` for base training/tokenizer
training. `omega-training` now defaults to `--dataset-format auto`: published
base partitions resolve to `jsonl` and chat partitions resolve to `chat`; ordinary
folders retain the text default. Explicit matching format flags remain valid.
Evaluate external `base/validation` or `chat/validation` separately. The training
CLI rejects published held-out data for fitting, mixed stages/partitions, parent
selection and nonzero automatic validation splits on prepared groups.

The consumer requires a valid `COMPLETE`/manifest checksum/schema and a nonempty,
unchanged selected `data.jsonl` matching the manifest hash/count. Its partition
directory must contain only that file. It does not rehash raw downloads,
membership payloads or unselected partitions. Do not move partition files out of
their release and assume those checks still apply. Details and commands are in
the [training README](../Code/Rust/omega-training/README.md#prepared-omega-datasets-releases).

Preparation budgets do not establish training capacity. In particular the eager
chat loader still caps input at 16 MiB/file, 64 MiB aggregate and 100,000 records,
and rejects conversations exceeding model context. Base caches have configurable
source limits; chat caches remain unsupported. Choose a bounded release and a
compatible frozen tokenizer/model; nothing is silently sampled or truncated.
This command does not train a tokenizer/model, create a checkpoint or mark ZH02
accepted.

Adding this crate changes the workspace `Cargo.lock` hash recorded by exact
training resume. Existing strict runtime compatibility checks remain in force;
old snapshots may require their original build/lockfile. No resume bypass,
checkpoint migration or tokenizer identity change is part of this addition.
