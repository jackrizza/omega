# Omega application validation — 2026-10-05

OA01 adds the Linux terminal application and worker runtime. This record covers
software and bounded hardware qualification, not model quality or large-model
throughput. No existing training run, executable, dataset or checkpoint was replaced.

## Software checks

- Windows, Rust 1.93.0: `cargo test --workspace --locked` passed **273 tests**.
- Windows: workspace formatting and strict Clippy with
  `--features omega/gpu,omega/cuda --all-targets` passed.
- Linux: strict all-target Clippy with both backend features passed.
- Separate Vulkan-only and CUDA-only all-target `omega` feature checks passed;
  the default CPU build and combined optimized release build also passed.
- Ubuntu 24.04 WSL: the all-backend workspace test run passed **276 tests** with
  **14 opt-in hardware tests ignored** on the final source. Focused CUDA hardware
  tests were run separately, including the final source/header identity changes.
- Linux all-backend compilation succeeded. Configuration tests cover comments,
  implicit defaults, TOML edits, recipe import, path resolution, invalid values
  and conflicting edits. Temporary pipelines cover tokenizer reuse receipts,
  benchmarks, training, assistant stages, evaluation, generation and resume.
- Offline release preparation/reuse verifies all six payloads, including empty
  held-out partitions, and rejects corrupted data. Training still rejects an
  empty selected partition.
- Ratatui tests render onboarding, forms, editor, review, dashboard, jobs,
  checkpoints and tokenizer screens at normal and small terminal sizes. The
  shared chat operation test verifies continued accepted conversation history.

The pinned Ratatui 0.30.2 and Crossterm 0.29 dependencies match the
[official installation guidance](https://ratatui.rs/installation/). Dynamic CUDA
loading follows the existing
[CubeCL 0.6 dependency configuration](https://docs.rs/crate/cubecl-cuda/0.6.0/source/Cargo.toml.orig).

## Process and terminal checks

`omega/tests/persistence.py` drives the executable through Linux PTYs using only
temporary projects and state. It verifies:

- Updates continue after Detach, forced TUI termination and terminal hangup.
- A reopened UI retains the same worker identity and creates no duplicate job.
- Concurrent launch fails, including when another state directory is selected.
- Editing `model.toml` does not change the frozen running configuration.
- Checkpoint-and-stop publishes a complete checkpoint before reporting stopped.
- Resuming an earlier periodic checkpoint uses the retained executable.
- Missing tokenizer failures never claim a completed job or saved checkpoint.
- Forced termination of a test worker records failure and retains its latest
  complete checkpoint.
- Stale boot/start identity is rejected even when the PID belongs to a live process.
- Normal exit restores terminal settings. The executable operates outside the
  checkout with Rust tools removed from PATH.

Rust tests additionally cover partial journal tails, invalid IDs and unavailable
storage. `omega/tests/clean_runtime.sh` exercises startup, tokenizer preparation,
CPU training and checkpoint completion in a clean container without Rust or GPU
libraries. The final optimized all-backend executable passed both the PTY suite
and clean Ubuntu 22.04 container test. Ubuntu 24.04 clean-container execution also
passed during development.

## NVIDIA qualification

The user-provided local WSL host exposes an **RTX 5070 Laptop GPU, 8 GB**, with
Windows driver **591.59** and CUDA driver API **13.1**. The installed NVRTC 13.1
library failed because Burn 0.18/cudarc 0.16.6 expects `nvrtcGetNVVM`, which that
library does not export. CPU operation remains available.

An official NVIDIA NVRTC **12.8.61** package was extracted into the temporary
`/tmp/omega-cuda12-test/runtime` directory and selected using `LD_LIBRARY_PATH`.
The installed driver/toolkit was not changed. CUDA headers came from the installed
CUDA 13.1 toolkit. Burn requires these headers for runtime kernel compilation;
`CUDA_PATH` can select a complete toolkit include tree. CUDA 12.8 introduced
Blackwell support ([NVIDIA release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)).

The final two opt-in `cuda_training` tests passed, covering masked batches,
evaluation, schemas 7/8, continuous versus restored updates, CPU/CUDA resume
rejection, incompatible source identity, and transactional rejection of invalid
updates. No Vulkan-specific float-bit reinterpretation is used for CUDA.

`omega/tests/backend_pipeline.py BINARY cuda` provides the executable-level
qualification passed on the final optimized executable: tokenizer creation, synchronized benchmark, base and assistant
workers, verified checkpoints, exact assistant resume, and repeated continuation
in separate worker processes produced identical model bytes. Missing-header
diagnostics also passed.

CUDA remains experimental: these tiny f32 runs qualify the tested paths on this
host, not every NVIDIA architecture, driver or training scale.

## Distribution and remaining gates

The release workflow pins Ubuntu 22.04, CUDA 12.5.1 by image digest and Rust 1.93.0.
It tests, builds all backends, runs PTY/clean-runtime checks, and packages the
executable, archive, checksums and runtime guide. Publishing is tag-triggered;
no release has been published by this implementation task.

The final executable was also built locally in that exact pinned Ubuntu 22.04
CUDA image with all three backends enabled. Its tiny CPU and CUDA pipelines ran
outside the checkout; the clean Ubuntu 22.04 container had no Rust installation,
CUDA libraries, or NVIDIA driver. These checks exercise the actual release
executable, not only unit-test binaries. The GitHub workflow itself has not been
triggered during this task.

Local preview artifacts are in `Code/Rust/target/omega-preview-20261005/`:
the standalone `omega` executable, versioned `.tar.gz`, `RUNTIME.md` and
`SHA256SUMS`. The executable is 64,106,096 bytes with SHA-256
`4fec4e8e5ac16fe8fd137a302fa194ba552298866b21225ab155b00e4e88f4f2`.
This is a local preview, not a published GitHub release.
The adjacent `validation/` directory retains the build, workspace, Clippy, CUDA,
PTY and clean-runtime command logs.

The A770 host still has an existing `omega-training` process running. The final
Vulkan hardware qualification remains pending until that host is idle. It was
not interrupted or used for competing benchmark/training work.
