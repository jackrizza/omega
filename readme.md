# Omega

The [Omega application](Docs/omega.md) adds a persistent Linux terminal workspace, detached workers, whole-workflow projects and experimental CUDA. See OA01 in the task board for current validation. CUDA uses resume schemas 7/8; CPU/Vulkan formats are preserved.

A small Rust GPT workspace with CPU, Vulkan and experimental CUDA training: prepare byte BPE tokenizers, build text/JSONL
datasets, train with held-out metrics, cache tokens, save resumable CPU training
snapshots with periodic/cooperative-stop saving, train with optional sampling,
masked minibatches and clipping/warmup, inspect checkpoint inventories, and
generate greedily or with seeded token sampling from checkpoints.
The included data and vocabulary are toy fixtures, not pretrained weights.

## Start here

- [Omega application](Docs/omega.md): install and use the persistent training TUI.
- [GitHub builds](Docs/github-builds.md): initial push, main-branch artifacts and tagged releases.

- [Container workflow](Containers/README.md): build all Rust binaries, prepare datasets and train through one Python runner.

- [Zero to hero](zero-to-hero.md): roadmap and remaining tasks for training a conversational assistant from scratch on one 16 GB Arc A770.
- [Arc A770 GPU training](Docs/gpu-training.md): release builds, device selection and resume compatibility.
- [Training guide](Docs/omega-training.md): commands, paths, and model contracts.
- [Training-time benchmarks](Code/Rust/omega-benchmark/README.md): measure a bounded dataset sample and estimate epoch/run duration on CPU or Vulkan.
- [Implementation status](Docs/implementation-status.md): capabilities and limits.
- [Task board](TASKS.md): approved task states, dependencies, and acceptance criteria.
- [Agent instructions](AGENTS.md): contribution and verification rules.

## Workspace

- [omega](Code/Rust/omega/README.md): main Linux Ratatui application with persistent background training workers.

- [omega-datasets](Code/Rust/omega-datasets/README.md): Hugging Face downloads and
  grouped base/chat corpus splits configured by `datasets/<name>/model.toml`.
- [omega-tokenizer](Code/Rust/omega-tokenizer/README.md): saved tokenizer pipelines.
- [omega-nn](Code/Rust/omega-nn/README.md): GPT tensors, optimization, and generation.
- [omega-training](Code/Rust/omega-training/README.md): document datasets, corpus
  training, and checkpoints. These orchestration features belong to this crate.
- [omega-benchmark](Code/Rust/omega-benchmark/README.md): library and CLI for dataset-based training-time estimates using the training pipeline.

Run Cargo commands from `Code/Rust`. Use `-p omega-training --bin omega-training` for the uniquely named training
executable. The legacy `main` binaries remain available; select their package with `-p`. Keep [Cargo.lock](Code/Rust/Cargo.lock) in version control for
the CLI workspace and use `--locked` so builds do not silently change dependency
resolution. Update it deliberately when changing dependencies. A lockfile fixes
dependency resolution, not cross-platform numerical behavior.

```sh
cargo build --workspace --locked --jobs 1
cargo test --workspace --locked --jobs 1
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check
```

Serial build jobs avoid competing Windows linker writes for the shared `main`
binary filename. Cargo can still warn about that filename; select a package for
running a binary.

The workspace uses edition 2024. Rust 1.93.0 on Windows was verified in the
2026-09-27 source review; this is not a declared minimum supported Rust version.
The empty `Code/Rust/omega-tokenizer/test.json` is an unused placeholder, retained
as existing data. The documented tokenizer fixture is `datasets/test.json`;
tokenizer CLI `-f test.json` resolves there, while library paths are used literally.
