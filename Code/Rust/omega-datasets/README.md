# omega-datasets

A Rust library and uniquely named `omega-datasets` CLI for the corpus preparation
stage in [zero to hero](../../../zero-to-hero.md). A folder's `model.toml` selects
Hugging Face data files, maps their fields, and describes train/validation/test
splits. This configuration describes a **dataset recipe**, not GPT architecture.

Run from `Code/Rust`:

```sh
cargo run -p omega-datasets --locked -- init assistant-v1
# Edit ../../datasets/assistant-v1/model.toml.
cargo run -p omega-datasets --locked -- check assistant-v1
cargo run -p omega-datasets --locked -- plan assistant-v1
cargo run -p omega-datasets --release --locked -- build assistant-v1
```

You may instead create `datasets/assistant-v1` and `model.toml` yourself.
`init` is optional and never replaces existing files. `check` is offline;
`plan` retrieves repository metadata and prints pinned SHAs, filenames and known
download sizes, without downloading source files. `build` downloads and prepares
the selected files. Only run it after choosing files and appropriate limits.
Progress starts on stderr; final partition counts and release path go to stdout.

All commands accept `--datasets-root PATH`. The default refers to this checkout
at compile time; an explicit relative root is relative to the current directory.
Use `/` in nested dataset names. Traversal, symlinks and Windows junctions are
rejected. Private/gated repositories use `HF_TOKEN` from the environment; acquire
access through Hugging Face first. The token is not stored in output files.

See [examples/model.toml](examples/model.toml) for the starter recipe and
[the configuration guide](../../../Docs/omega-datasets.md) for all fields,
mapping examples, output contracts and limitations. The starter references a
real TinyStories training shard as a syntax example, not an approved corpus.
That shard exceeds the default record budget: choose smaller files or explicitly
raise the relevant limits before building. Limits cause errors, never sampling.

## Library

```rust,no_run
use std::path::Path;
use omega_datasets::{build, config::Config, hub::HuggingFace};

let folder = Path::new("../../datasets/assistant-v1");
let config = Config::load(&folder.join("model.toml"))?;
let hub = HuggingFace::new(std::env::var("HF_TOKEN").ok(), config.limits.timeout_seconds)?;
let report = build(folder, &config, &hub)?;
println!("{:?}", report.counts);
# Ok::<(), anyhow::Error>(())
```

`hub::plan` resolves repository references to commits. `Hub` can be implemented
for offline fixtures or a caller-owned transport/cache. `build` validates the
configuration and writes only a newly reserved child directory. An injected
transport must enforce the supplied download limit. No dependency on Burn,
`omega-training`, or tokenizer IDs is introduced. Output format compatibility
is described below; no existing data, tokenizers or checkpoints are migrated.

## Validation

```sh
cargo test -p omega-datasets --locked --jobs 1
cargo clippy -p omega-datasets --all-targets --locked -- -D warnings
cargo fmt --all --check
```

Tests use temporary directories, an injected in-memory Hub and local HTTP servers.
They cover real Parquet decoding, JSON/gzip, auth/status/size handling, pinned
download paths, deterministic splits, transitive grouping, duplicate removal,
cross-stage overlap, failure markers, quotas, paths and CLI parsing/error exits.
No live dataset is downloaded by the test suite. Unix symlink tests and Windows
junction tests are platform-gated.

`verify_release(path)` validates the completion marker and all six published
payload hashes/counts, including intentionally empty partitions. Omega uses this
for verified pipeline reuse. Training still requires a populated explicit partition.
