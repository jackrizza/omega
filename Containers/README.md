# Omega containers

A fresh Git checkout includes the example recipe, but no downloaded or prepared
datasets. The existing `omega-alpha` release described below is local development
data. Review your recipe and prepare its data before using `--skip-dataset-build`;
that flag requires an already completed release on your machine.

One [Dockerfile](Dockerfile) provides three image targets:

| Image | Purpose |
|---|---|
| `omega-builder:local` | Compile every workspace binary with Rust 1.93.0, `--release --locked`; retain outputs under `/opt/omega/bin/<package>/<binary>` and an inventory at `/opt/omega/binaries.json`. |
| `omega-datasets:local` | Run `omega-datasets init`, `check`, `plan` and `build`. |
| `omega-training:local` | Run `omega-training`, including tokenizer preparation, train, resume, evaluation, generation and chat. |

These are Linux **CPU** images. The optional Rust Vulkan feature is not enabled;
passing `--backend vulkan` does not enable GPU support in these images. The host
needs Python 3.10+ and Docker with a running Linux engine and BuildKit. No host
Rust installation or Python packages are required. The first build downloads
base images, OS packages and locked Cargo dependencies. Package-specific Cargo
cache directories keep all three legacy `main` binaries distinct. Building the
other targets reuses the same builder layer.

The Dockerfile-specific ignore file excludes datasets, weights, host build outputs,
Git directories, host `.cargo` settings and `.env` files from the build context. Existing corpora and
checkpoints are mounted only when running a container. This uses Docker's
[multi-stage targets](https://docs.docker.com/build/building/multi-stage/) and
[bind mounts](https://docs.docker.com/engine/storage/bind-mounts/).

## Run with one Python script

### Configured omega-alpha run

The default [omega-alpha pipeline](pipeline.omega-alpha.json) now uses the actual
recipe and published output in this checkout. The manifest reports 458,803 base
training records, 25,472 validation records, 25,350 test records and no chat data.
`release-v1` already has a completion marker, so explicitly reuse it:

```sh
python Scripts/run_containers.py --dry-run all --skip-dataset-build
python Scripts/run_containers.py all --skip-dataset-build
```

The first command only prints commands. The second builds images and performs
tokenizer preparation and base training. `--skip-dataset-build` skips dataset
execution; `--skip-build` before `all` separately skips Docker image builds.
The Rust loader still verifies the selected release's marker, manifest and payload.

| Operation | Configured flags |
|---|---|
| Dataset build | `build omega-alpha --datasets-root /omega/datasets`; the recipe's `output` is `release-v1`. |
| Tokenizer fitting | `--dataset omega-alpha/release-v1/base/train --dataset-format auto --vocab-size 4096 --min-frequency 2 --chat-protocol` |
| New tokenizer | `--output /omega/datasets/omega-alpha/omega-tokenizer-v1.json` |
| Base training | Same train partition, `--dataset-format auto`, `--tokenizer omega-alpha/omega-tokenizer-v1.json`, explicit dataset/weights roots. |
| Pilot budget | `--epochs 1 --max-updates 2 --backend cpu --cpu-threads 1 --matmul-threads 1` |
| Model | Context 128, width 32, four heads, two layers, feed-forward width 128, batch size 1; learning rate 0.003 and seed 42. |

The existing `omega-alpha/tokenizer.json` is preserved. The pipeline creates its
own smaller tokenizer and freezes its IDs before model initialization. Registering
Omega chat controls now permits later SFT without changing the vocabulary; no SFT
is configured because this release has no chat records. Tokenizer creation needs
a new path. Once it exists, invoke `training -- train ...` directly for subsequent
runs instead of repeating tokenizer creation.

The training JSONL is about 405 MiB. **The two-update cap does not limit tokenizer
training or corpus loading**: preparation still processes the full partition and
can need substantial RAM/time. This plan uses eager loading. Cache flags do not
bound eager loading; a cache would need raised source/document/index/manifest
limits, identical at creation and consumption. Its default 16 MiB source and
100,000-document caps do not fit this release. No data is truncated or sampled.

`python Scripts/run_containers.py all` includes dataset building and will reject
the existing release. To build another release, choose a new recipe `output` and
update every matching path in a copied plan. Source/split/download settings belong
in `model.toml`, not training flags. Do not use validation ratio/count flags to
re-split a prepared release. Evaluate validation separately with the actual saved
checkpoint (replace the example suffix if necessary):

```sh
python Scripts/run_containers.py --skip-build training -- evaluate --datasets-root /omega/datasets --weights-root /omega/weights --checkpoint omega-alpha-base-pilot-1 --dataset omega-alpha/release-v1/base/validation --dataset-format auto --cpu-threads 1 --matmul-threads 1
python Scripts/run_containers.py --skip-build training -- generate --weights-root /omega/weights --checkpoint omega-alpha-base-pilot-1 --prompt "Once upon a time" --max-new-tokens 16 --cpu-threads 1 --matmul-threads 1
```

Evaluation scans the whole selected partition; it is not automatically included
in the pilot. Test records remain reserved for final acceptance.

### Other corpus recipes

From the repository root:

```sh
python Scripts/run_containers.py build
python Scripts/run_containers.py datasets -- init my-corpus
```

Edit the newly created `datasets/my-corpus/model.toml` to select your sources,
field mappings, reviewed permitted use and download/record limits. Its `output`
must match the release name in your pipeline (the example uses `release-v1`).
The generated source template is an example requiring review, not a selected
production corpus. Inspect it with the dataset CLI before the full run:

```sh
python Scripts/run_containers.py --skip-build datasets -- check my-corpus
python Scripts/run_containers.py --skip-build datasets -- plan my-corpus
python Scripts/run_containers.py --dry-run all --plan Containers/pipeline.example.json
python Scripts/run_containers.py all --plan Containers/pipeline.example.json
```

`check` is local; `plan` contacts Hugging Face for metadata; `all` downloads the
configured sources. The [example pipeline](pipeline.example.json) builds all
three images, builds `my-corpus`, trains a new 512-entry BPE tokenizer on
`my-corpus/release-v1/base/train`, then performs at most two tiny CPU model
updates and saves a checkpoint. It demonstrates execution, not model quality.
No real corpus is chosen or downloaded merely by creating these container files.

Copy/edit the JSON plan for your run. `datasets` is one array of arguments
beginning with `build`; `training` is an ordered list of argument arrays. It can
include `train-tokenizer`, `train`, `evaluate`, `train-stage`, `resume`, or other
training CLI commands. At least one step must be `train`, `train-stage` or `resume`.
Put each CLI subcommand first and its flags after it. All entries are validated
as argument arrays before execution; the Rust CLIs validate their own flags.
There is no shell expansion or interpolation of plan strings.

For assistant training, use a tokenizer created with `--chat-protocol` from the
start, `chat/train` with `--dataset-format chat`, and the existing stage/checkpoint
contracts in [assistant execution](../Docs/assistant-execution.md). Select only
the intended partition. Selecting the release parent recursively would include
validation/test data; the example deliberately selects `base/train` only.

Each step must succeed before the next starts. Dataset releases and tokenizers
use new-path writes: repeating `all` with the same output paths fails instead
of replacing data. To continue after preparation, invoke `training` separately.
Use the Rust `resume` subcommand for a compatible training snapshot; running
`train` again starts fresh. Windows-native snapshots are not promised to resume
exactly in Linux containers. Keep the same image/toolchain/runtime for continuation.

## Individual stages and paths

```sh
python Scripts/run_containers.py training -- train --help
python Scripts/run_containers.py training -- generate --help
python Scripts/run_containers.py --skip-build training -- train --name container-smoke --dataset examples --tokenizer test.json --epochs 1 --max-updates 1 --context-length 8 --d-model 8 --heads 2 --layers 1 --d-ff 16 --cpu-threads 1 --matmul-threads 1
python Scripts/run_containers.py --skip-build training -- generate --checkpoint container-smoke-1 --prompt hello --max-new-tokens 2
```

For generation, replace `container-smoke-1` with the checkpoint name actually
printed by training. The fixture tokenizer is only for this smoke example.
Omitting CLI arguments displays the container's help. Runtime commands rebuild
their image through Docker's cache by default; `--skip-build` uses an existing
image. `build` explicitly builds all three images and compiles all workspace bins.

Runner options go **before** the command; Rust arguments follow `--`:

```sh
python Scripts/run_containers.py --datasets-root /absolute/data --weights-root /absolute/checkpoints --jobs 2 --tag experiment training -- train --help
```

The script locates the checkout relative to itself and can be called from any
directory. Default host roots are repository `datasets/` and `weights/`.
Explicit relative host roots resolve from your current directory. The dataset
root must exist; the weights root is created if needed. Paths with spaces are
supported; paths with commas, double quotes or newlines are rejected.

Inside the containers, roots are `/omega/datasets` and `/omega/weights`, matching
the compile-time checkout layout; working directory is `/omega`. Use container
paths in forwarded Rust arguments. Relative tokenizer paths are dataset-relative.
Use `/omega/datasets/...` for tokenizer/cache outputs and `/omega/weights/...`
for metrics that must persist. Other writes in the container disappear with
`--rm`. The dataset container mounts only datasets; training mounts both roots
read/write to allow tokenizers, caches, metrics and numbered checkpoints.

On Linux/macOS the runner supplies the caller's UID/GID to avoid root-owned host
outputs. On Windows, Docker Desktop controls bind-mount access. Host directories
must be accessible to the Docker engine: network shares/mapped drives such as
`Z:` may need local dataset/weights overrides or host file-sharing configuration.
A remote Docker context requires paths on that remote host; the runner does not
copy datasets to a remote engine.

If needed, set `HF_TOKEN` in the host environment; it is forwarded by variable
name only to `omega-datasets` and never included in an image, plan or printed
command. Ctrl+C stops the active named container, allowing up to 120 seconds for
training's cooperative checkpoint save. Forced termination cannot guarantee a
save. Failed/interrupted stages return nonzero and prevent later steps.

## Docker without the runner

Run builds with the repository root as context:

```sh
docker build -f Containers/Dockerfile --target builder -t omega-builder:local .
docker build -f Containers/Dockerfile --target datasets -t omega-datasets:local .
docker build -f Containers/Dockerfile --target training -t omega-training:local .
docker run --rm omega-datasets:local --help
docker run --rm omega-training:local train --help
docker run --rm omega-builder:local cat /opt/omega/binaries.json
```

Supply the same bind mounts shown by the Python runner's `--dry-run` for real
data/training commands. Executables in the builder are Linux binaries; they do
not run directly as Windows `.exe` files. The default image tag and OS package
repositories are not immutable digests; `Cargo.lock` fixes Rust dependencies,
not the complete OS image or cross-platform numerical results.

## Validation

```sh
python -B -m unittest discover -s Containers/tests -v
python Scripts/run_containers.py --dry-run all --plan Containers/pipeline.example.json
```

Offline tests cover bin discovery/collision isolation, mounts, spaces and literal
arguments, execution ordering, all-stage failure propagation, invalid plans,
token handling, interruption and invocation outside the checkout. They mock
Docker and Cargo and do not prove Linux compilation or container execution.
Current build/runtime acceptance and any environment blockers are recorded under
CT01 in [TASKS.md](../TASKS.md).
