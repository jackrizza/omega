# Scripts

## Docker workflow

Prepared datasets are excluded from Git. On a fresh checkout, review and build a
dataset recipe before using the `--skip-dataset-build` examples below.

[`run_containers.py`](run_containers.py) is the single Python entry point for
building all Rust binaries, creating datasets with `omega-datasets`, and running
`omega-training` in Linux CPU containers:

```sh
python Scripts/run_containers.py build
python Scripts/run_containers.py datasets -- init my-corpus
# Edit datasets/my-corpus/model.toml and the pipeline for your selected sources.
python Scripts/run_containers.py --dry-run all --plan Containers/pipeline.example.json
python Scripts/run_containers.py all --plan Containers/pipeline.example.json
```

See [container instructions](../Containers/README.md) for setup, persistent host
paths, individual stages, tokenizer preparation, failure handling and validation.
Python 3.10+ and a running Docker Linux engine are required; host Cargo is not.

The default `all` plan is now configured for the existing `omega-alpha` recipe.
Since its `release-v1` has already been built, run:

```sh
python Scripts/run_containers.py --dry-run all --skip-dataset-build
python Scripts/run_containers.py all --skip-dataset-build
```

This creates a separate Omega tokenizer and runs at most two CPU model updates.
Tokenizer fitting/loading still reads the complete corpus. The configured flags
and subsequent evaluation/generation commands are in the container instructions.

## Train and generate with one command

From the repository root, using Bash (for example Git Bash on Windows):

```sh
bash Scripts/training_run.sh foo "examples/greetings,examples/omega" test.json "hello world"
```

The wrapper uses `--release --locked` and the uniquely named `omega-training`
executable. Optional trailing GPU flags are supported:

```sh
bash Scripts/training_run.sh gpu-run examples test.json "hello" --backend vulkan --device 0
```

This enables the Cargo `gpu` feature for both commands. CPU remains the default.
Device indices enumerate discrete Vulkan adapters; see [GPU training](../Docs/gpu-training.md).

Arguments, in order:

1. Training run name.
2. Dataset folder name, or a **quoted comma-separated list** of folder names.
   Do not add spaces around commas unless they are part of a folder name.
   Commas and newlines within dataset names are not supported.
3. Tokenizer JSON path, relative to `datasets/` or absolute.
4. Quoted generation prompt.

The script finds `Code/Rust` relative to its own location, so it can be invoked
from another directory using the script's full path. Cargo must be on `PATH`.
Use Bash, not `sh`, because the script uses Bash arrays. Shell arrays cannot be
passed directly as a single process argument; the comma-separated string is
converted into repeated `--dataset` arguments internally.

It runs training for 10 epochs, then generates up to 5 tokens from the latest
completed checkpoint under the supplied run name. The prompt plus generation
budget must fit the trained model's context. Training failure prevents generation;
either command's failure produces a nonzero script exit code. Training starts a
fresh run, not a resume, and saves checkpoints using the CLI's normal numbering.

**Concurrency:** do not run jobs with the same run name concurrently when using
this wrapper. `--latest-run` selects the newest completed checkpoint at generation
time, which could be a different job's save. Use separate names for concurrent jobs.

```sh
bash Scripts/training_run.sh --help
bash Scripts/test_training_run.sh
```

The script tests mock Cargo to check command arguments, working-directory
resolution, and failure handling without training or writing weights.
