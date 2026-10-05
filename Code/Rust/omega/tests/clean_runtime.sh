#!/bin/sh
# Acceptance test in an ephemeral Linux container, with no Rust or GPU libraries.
set -eu
binary=$1
root=$(mktemp -d)
cd "$root"
export OMEGA_STATE_DIR="$root/state"
"$binary" --help >/dev/null
"$binary" doctor --backend cpu
if "$binary" doctor --backend cuda >cuda.out 2>cuda.err; then
    echo 'Unexpected CUDA availability in clean runtime' >&2
    exit 1
fi
mkdir -p datasets/base
printf 'Hello world this is a tiny corpus.\n' >datasets/base/a.txt
cat >model.toml <<'TOML'
omega_schema_version = 1
name = "clean-host"
[dataset]
selections = ["base"]
[tokenizer]
vocab_size = 260
min_frequency = 1
[model]
context_length = 32
d_model = 4
heads = 1
layers = 1
d_ff = 8
[training]
epochs = 1
cpu_threads = 1
matmul_threads = 1
[pipeline]
prepare_tokenizer = true
train = true
TOML
job=$("$binary" start model.toml --yes)
count=0
while ! grep -q '"status": "completed"' "$OMEGA_STATE_DIR/jobs/$job/status.json"; do
    if grep -q '"status": "failed"' "$OMEGA_STATE_DIR/jobs/$job/status.json"; then
        cat "$OMEGA_STATE_DIR/jobs/$job/worker.log" >&2
        exit 1
    fi
    count=$((count + 1))
    if [ "$count" -gt 90 ]; then
        "$binary" stop "$job" || true
        echo 'Tiny clean-host CPU pipeline timed out' >&2
        exit 1
    fi
    sleep 1
done
test -f weights/clean-host-1/COMPLETE
echo 'PASS clean runtime: startup without CUDA, tokenizer preparation, CPU training, complete checkpoint'
