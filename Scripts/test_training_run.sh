#!/usr/bin/env bash
# Tests use a mocked Cargo function: no models are trained or checkpoints written.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$script_dir/training_run.sh"
test_dir="$(mktemp -d)"
trap 'rm -rf -- "$test_dir"' EXIT
export CARGO_TRACE="$test_dir/cargo.log"
export FAIL_TRAIN=0

cargo() {
    [[ "$PWD" == */Code/Rust ]] || return 99
    printf '%s\n' 'CALL' "$@" >> "$CARGO_TRACE"
    if [[ "$*" == *' -- train '* && "$FAIL_TRAIN" == 1 ]]; then
        return 17
    fi
}
export -f cargo

# Execute from outside the repository and preserve arguments containing spaces.
cd -- "$test_dir"
bash "$runner" foo 'examples/greetings,folder with spaces' 'my tokenizer.json' 'hello world'
printf '%s\n' \
    CALL run -p omega-training --bin omega-training --release --locked -- train --backend cpu --name foo \
    --dataset examples/greetings --dataset 'folder with spaces' \
    --tokenizer 'my tokenizer.json' --epochs 10 \
    CALL run -p omega-training --bin omega-training --release --locked -- generate --backend cpu --latest-run foo \
    --prompt 'hello world' --max-new-tokens 5 > expected.log
diff -u expected.log "$CARGO_TRACE"

# GPU controls are forwarded to both operations with the optional Cargo feature.
: > "$CARGO_TRACE"
bash "$runner" gpu examples/greetings test.json hello --backend vulkan --device 0
[[ "$(grep -c '^gpu$' "$CARGO_TRACE")" == 4 ]]
[[ "$(grep -c '^--features$' "$CARGO_TRACE")" == 2 ]]
[[ "$(grep -c '^--device$' "$CARGO_TRACE")" == 2 ]]
[[ "$(grep -c '^vulkan$' "$CARGO_TRACE")" == 2 ]]

# Generation must not run after failed training; preserve its failure status.
: > "$CARGO_TRACE"
export FAIL_TRAIN=1
status=0
bash "$runner" foo examples/greetings test.json hello || status=$?
[[ "$status" == 17 ]]
[[ "$(grep -c '^CALL$' "$CARGO_TRACE")" == 1 ]]

# Invalid arguments and help must not invoke Cargo.
: > "$CARGO_TRACE"
status=0
bash "$runner" foo 2>/dev/null || status=$?
[[ "$status" == 2 ]]
for datasets in '' ',one' 'one,' 'one,,two'; do
    status=0
    bash "$runner" foo "$datasets" test.json hello 2>/dev/null || status=$?
    [[ "$status" == 2 ]]
done
for args in '--backend cuda' '--device 0' '--backend vulkan --device invalid' '--unknown flag'; do
    status=0
    read -r -a options <<< "$args"
    bash "$runner" foo examples test.json hello "${options[@]}" 2>/dev/null || status=$?
    [[ "$status" == 2 ]]
done
bash "$runner" --help > /dev/null
[[ ! -s "$CARGO_TRACE" ]]
printf 'All training_run.sh tests passed.\n'
