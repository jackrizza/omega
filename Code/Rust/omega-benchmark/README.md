# Omega Benchmark

The [Omega application](../../../Docs/omega.md) adds a persistent Linux terminal workspace, detached workers, whole-workflow projects and experimental CUDA. See OA01 in the task board for current validation. CUDA uses resume schemas 7/8; CPU/Vulkan formats are preserved.

Library and `omega-benchmark` binary for bounded performance measurements. The
first benchmark, `training-time`, measures real checked training updates on
selected dataset batches and estimates the time for a fixed number of epochs.
It runs on the machine where you launch it; use the A770 host for a GPU estimate.

## CLI

Run from `Code/Rust`, using a **release build** for useful timing. Copy the dataset,
tokenizer, dimensions, batch size, optimizer options and epoch count from your
intended training command. Defaults match `omega-training train` dimensions and
learning rate; the benchmark defaults to **one epoch**, training defaults to ten.
`--tokenizer` is required to avoid accidentally benchmarking the tiny test vocabulary.

For the existing Omega Alpha base partition on the A770:

```sh
cargo run -p omega-benchmark --release --locked --features gpu -- training-time --backend vulkan --device 0 --dataset omega-alpha/release-v1/base/train --tokenizer omega-alpha/tokenizer.json --batch-size 2 --epochs 6 --samples 12 --max-seconds 60
```

Run when the GPU is idle. Competing training or graphics work changes the result.
Increase `--max-seconds` if warmup and all requested samples do not fit. This is a
cooperative budget checked **between updates**, not a hard process timeout: one
update can exceed it. Dataset preparation, cache verification, model allocation
and initial device setup are outside that budget. There is no background job.

For ordinary text folders on CPU, with explicit model dimensions:

```sh
cargo run -p omega-benchmark --release --locked -- training-time --datasets-root /path/to/datasets --dataset my-corpus --tokenizer my-tokenizer.json --context-length 64 --d-model 32 --heads 4 --layers 2 --d-ff 128 --batch-size 2 --epochs 6 --cpu-threads 4 --matmul-threads 1
```

Use your actual paths. Relative `--datasets-root` paths resolve from the working
directory; the default is the compile-time checkout's `datasets/`. Tokenizer
paths are absolute or relative to that root. Nested dataset selections use `/`
on Windows too. Repeat `--dataset` to combine ordinary folders.

```sh
cargo run -p omega-benchmark --locked -- training-time --help
```

Add `--json` for a JSON report on stdout, or `--output new-report.json` to also
save it. Existing output files are rejected. Progress/errors go to stderr and
failures return nonzero. No weights, datasets or tokenizers are written, and no
data is downloaded. Warmup and sampled updates use one disposable model/Adam
session whose weights are discarded afterwards.

## Dataset and training compatibility

- Published `omega-datasets` releases: select exactly `base/train` or `chat/train`
  beneath the release. Shared training format detection validates completion,
  hashes and partition membership. Held-out partitions, release parents, mixed
  stages/partitions and automatic re-splitting of published data are rejected.
- Ordinary folders: default `--dataset-format auto` means text; choose `jsonl`
  for base `{"text":"..."}` records, or `chat` for Omega conversation JSONL.
  Chat requires a compatible saved protocol tokenizer, preserves assistant-only
  loss masks and uses the same conversation length/size checks as training.
- `--validation-count` or `--validation-ratio`, with `--split-seed`, reserves whole
  ordinary documents before chunking. Only the resulting training partition is
  estimated. Default: no holdout, as in `train`.
- `--cache /path/to/existing-cache` reuses a verified **base** token cache. Format,
  tokenizer, source contents, context, split and cache limits must match its
  creation options. The five `--max-source-bytes`, `--max-document-tokens`,
  `--max-documents`, `--max-directory-entries`, `--max-manifest-bytes` flags apply
  to cache verification only; eager preparation retains training's existing
  limits. The library exposes the same settings in `DatasetOptions.cache_limits`.
  This command does not create caches or support chat caches.
- Context chunking, final partial batches, attention/loss padding masks,
  `--max-batch-tokens`, clipping and `--warmup-updates` reuse `omega-training`.
  `--warmup` is separate: it controls untimed benchmark updates (default two).
- CPU is default. `--features gpu --backend vulkan` explicitly selects a discrete
  Vulkan adapter, with no CPU fallback. The GPU path uses the existing f32
  training backend and synchronizes each measured update. CPU thread flags start
  a fresh child before tokenizer/backend pools initialize.

## What the estimate means

Preparation reads the **entire selected training partition**, checks every batch
and counts supervised targets, inputs, padding and updates. Without a cache,
eager corpus tokenization can take substantial time and memory; `--samples` does
not limit preparation to a prefix. Cache verification still reads source data.

The epoch's batches are divided into contiguous intervals. The midpoint batch
of each interval is timed with its actual shape, including the final partial
batch when selected. The estimate is:

```text
epoch seconds = sum(sample seconds * batches represented by that sample)
training seconds = epoch seconds * epochs
```

Samples are capped to one epoch (default twelve, maximum 1,000); interval weights
cover all its batches. Warmup uses the first batch. A single partial batch cycles
through the same trainer for warmup and measurement. An incomplete timed plan
returns an error with **no estimate**.

The human summary and schema-1 JSON report include measured targets/second,
sample positions, interval weights and timings, exact counts, backend/runtime and
build identity, tokenizer/source identity, model and optimizer options, observed
setup and warmup durations, and projected epoch/run durations. The
`padded_positions_per_epoch` count is total allocated input positions **including**
real tokens; subtract `input_positions_per_epoch` to find padding overhead.

`estimated_training_seconds` estimates update work only. The separate
`estimated_setup_plus_training_seconds` adds observed preparation/model setup.
Both exclude Cargo build time, initial benchmark warmup, checkpoint writing,
validation, logging and other CLI overhead. Unseen GPU shapes can still compile
during timed samples. It is an approximation, with no statistical confidence
interval; use more samples and repeat on an idle device for heterogeneous data.
Fresh random weights do not reproduce the numerical state of an existing run.

This does not estimate time to reach a quality target, memory capacity, shuffled
or weighted sampling, distributed training, or exact checkpoint continuation.
It does not select a model size or change training/checkpoint behavior.

Adding this workspace crate changes `Cargo.lock`, part of exact-resume build
identity. Keep the original executable for continuing an existing run. The
benchmark never bypasses those compatibility checks or replaces that executable.

## Library

```rust
use omega_benchmark::{DatasetOptions, TrainingTimeConfig, training_time};

fn main() -> Result<(), String> {
    let config = TrainingTimeConfig {
        dataset: DatasetOptions {
            root: "/path/to/datasets".into(),
            selections: vec!["my-corpus".into()],
            tokenizer: "my-tokenizer.json".into(),
            ..Default::default()
        },
        epochs: 6,
        ..Default::default()
    };
    let report = training_time(&config)?;
    println!("Estimated update seconds: {}", report.estimated_training_seconds);
    Ok(())
}
```

Library callers configure CPU pools before calling. `training_time` returns a
serializable `TrainingTimeReport` or actionable `String` error. `sampling_plan`
and `project_seconds` are available for inspecting the projection separately.

## Validation

```sh
cargo test -p omega-benchmark --locked
cargo clippy -p omega-benchmark --all-targets --locked -- -D warnings
cargo check -p omega-benchmark --features gpu --all-targets --locked
```

On an explicitly idle discrete GPU, the ignored tiny fixture smoke test is:

```sh
cargo test -p omega-benchmark --release --features gpu --locked --test training_time vulkan_training_time_smoke -- --ignored --exact
```

Tests use temporary fixtures and cover weighted projection/overflow, real CPU
updates, padding and remainder counts, cache integrity, published base/chat
selection, assistant masks, time budgets, output protection and CLI failures.

## Experimental CUDA

Build with `--features cuda` and select `--backend cuda --device 0`. CUDA uses
f32 host numerical checks and synchronized timings with no backend fallback.
Burn 0.18 requires compatible CUDA 12 NVRTC libraries; CUDA 13 NVRTC is not
supported. See the [Omega runtime guide](../../../Docs/omega.md).
