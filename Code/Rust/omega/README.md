# Omega

The main Linux terminal application. The `omega` binary contains the dataset,
tokenizer, neural-network, training and benchmark libraries and can launch itself
as a detached worker. The terminal never owns a training session.

From `Code/Rust`:

```sh
cargo run -p omega -- --help
cargo build -p omega --release --features gpu,cuda
./target/release/omega doctor --backend cuda
./target/release/omega
```

Cargo's configured target directory may differ; use `--target-dir` to select it.
CUDA is experimental. Burn 0.18 uses f32 without fusion; CUDA uses host numerical
checks. Drivers, CUDA toolkit headers and compatible CUDA/NVRTC libraries must already be installed. The default
build contains CPU only. `gpu` adds Vulkan and `cuda` adds CUDA independently.

See [the application guide](../../../Docs/omega.md) for installation, project
configuration, keyboard controls, persistence and recovery. Existing standalone
CLIs remain available.

Public APIs: `ProjectConfig` validates the workflow schema; `JobSpec` freezes a
launch; `JobStatus` and `JobEvent` describe persistent worker state. Project
sequencing is in `pipeline`, local process/IPC management in `jobs`, and terminal
state in `tui`. Shared compute operations remain in `omega-training::operations`.

```sh
cargo test -p omega
cargo clippy -p omega --all-targets -- -D warnings
# Linux only; bounded temporary CPU projects, no user data:
python3 omega/tests/persistence.py /absolute/path/to/omega
# Opt-in NVIDIA hardware qualification:
cargo test -p omega-training --features cuda --test cuda_training -- --ignored --test-threads=1
```

The Python script tests terminal and subprocess behavior; Python is **not** a
runtime dependency of the application. It must run on an idle Omega host because
the application deliberately enforces one compute pipeline per user.
