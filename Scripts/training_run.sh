#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash %s <run_name> <dataset1,dataset2,...> <tokenizer> <test_prompt> [--backend cpu|vulkan] [--device INDEX]\n' "$0"
    printf 'Example: bash %s foo "examples/greetings,examples/omega" test.json "hello world"\n' "$0"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$#" -lt 4 ]]; then
    usage >&2
    exit 2
fi

run_name="$1"
dataset_list="$2"
tokenizer="$3"
prompt="$4"
shift 4
backend=cpu
device=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
            backend="$2"; shift 2 ;;
        --device)
            [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
            [[ "$2" =~ ^[0-9]+$ ]] || { printf 'Error: device must be a nonnegative integer index.\n' >&2; exit 2; }
            device="$2"; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ "$backend" != cpu && "$backend" != vulkan ]]; then
    printf 'Error: backend must be cpu or vulkan.\n' >&2
    exit 2
fi
if [[ -n "$device" && ( "$backend" != vulkan || ! "$device" =~ ^[0-9]+$ ) ]]; then
    printf 'Error: device requires Vulkan and a nonnegative integer index.\n' >&2
    exit 2
fi
cargo_features=()
backend_args=(--backend "$backend")
if [[ "$backend" == vulkan ]]; then
    cargo_features=(--features gpu)
fi
if [[ -n "$device" ]]; then
    backend_args+=(--device "$device")
fi

if [[ -z "$run_name" || -z "$dataset_list" || -z "$tokenizer" || -z "$prompt" ]]; then
    printf 'Error: all four arguments must be nonempty.\n' >&2
    exit 2
fi
if [[ "$dataset_list" == ,* || "$dataset_list" == *, || "$dataset_list" == *,,* || "$dataset_list" == *$'\n'* ]]; then
    printf 'Error: supply comma-separated dataset names without empty entries or newlines.\n' >&2
    exit 2
fi

IFS=',' read -r -a datasets <<< "$dataset_list"
dataset_args=()
for dataset in "${datasets[@]}"; do
    dataset_args+=(--dataset "$dataset")
done

# Locate the Cargo workspace independently of the caller's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/../Code/Rust"

cargo run -p omega-training --bin omega-training --release --locked "${cargo_features[@]}" -- train "${backend_args[@]}" \
    --name "$run_name" "${dataset_args[@]}" --tokenizer "$tokenizer" --epochs 10

# Only reached if training succeeds. Do not run concurrent jobs with this name:
# latest-run resolves the newest completed checkpoint, not a specific process's save.
cargo run -p omega-training --bin omega-training --release --locked "${cargo_features[@]}" -- generate "${backend_args[@]}" \
    --latest-run "$run_name" --prompt "$prompt" --max-new-tokens 5
