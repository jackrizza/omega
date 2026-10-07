# Post-training validation — 2026-10-07

Branch: `dev`; uncommitted implementation. Existing staged RP02 and UI01 changes
were preserved. No production dataset/checkpoint was changed, no remote training
run was stopped, and no release, commit or push was made.

## Software evidence

- Windows CPU workspace tests passed. Linux WSL CPU workspace tests passed
  (314 tests on the integrated snapshot). After the final input-size and immutable
  approval regressions, focused readiness **13/13** and workflow **8/8** passed
  again on both platforms.
- Conversation evaluation **6/6**, integrated transfer/segment scenario **1/1**,
  cooperative metric evaluation **1/1**, save/control **4/4** and resume **12/12**
  passed. The single segment scenario contains multiple tiny runs and verifies
  fresh optimizer state, parent immutability, cross-build transfer, strict resume
  rejection, continuous versus segmented weights, budgets and failure propagation.
- Omega's TUI **8/8**, CLI parsing/approval **2/2**, and existing application
  integration **6/6** passed. New workflow tests also cover schema migration,
  frozen inputs, changed suite/data, immutable budgets, full human scoring,
  checkpoint/report hashes, threshold gates and non-overwriting promotion.
- `cargo fmt --all --check` and strict workspace/all-target Clippy with
  CPU/Vulkan/CUDA feature edges passed. Linux CPU and combined CPU/Vulkan/CUDA
  binaries built successfully. No dependency version or lockfile change was needed.
- `post_training_persistence.py` passed with a stripped CPU test executable:
  single-binary offline parent preparation; PTY detach/reconnect; host-wide
  concurrent launch rejection; verified checkpoint-stop; recovery after deleting
  the original launcher; missing-input rejection with unchanged workflow state;
  crashes during baseline, training and evaluation; conservative budget charging;
  completed report hashes and unchanged parent bytes.
- The existing `persistence.py` also passed after its review-label assertion was
  updated: normal detach, abrupt TUI kill, terminal hangup, reconnect to the same
  worker, retained exact resume, worker termination/failure and stale identities.
  Its executable runs outside the checkout with Rust removed from `PATH`.

Reproduction from `Code/Rust` (select a suitable local target directory):

```sh
cargo test --workspace --locked --jobs 2
cargo test -p omega -p omega-training --test post_training --test readiness --locked --jobs 2
cargo fmt --all --check
cargo clippy --workspace --all-targets --features omega/gpu,omega/cuda --locked --jobs 2 -- -D warnings
cargo build -p omega --features gpu,cuda --locked --jobs 2
python3 omega/tests/post_training_persistence.py /absolute/omega
python3 omega/tests/persistence.py /absolute/omega
python3 omega/tests/post_training_backend.py /absolute/omega cuda
```

The Python scripts are test harnesses only; Omega has no Python runtime
dependency. Use release or stripped test executables for the subprocess tests.
The first attempt using a 260 MiB debug executable exceeded the harness's
30-second launch deadline while hashing/copying it; a separate stripped copy
passed. No installed/production executable was replaced. An existing PTY test's
expected review label was updated from “pipeline” to “operation” to match the
generalized launch screen.

Cargo still warns about the pre-existing shared `main` binary output names in
legacy crates. They were not renamed as part of this milestone.

## CUDA hardware evidence

The new post-training backend harness passed on the local WSL **RTX 5070 Laptop
GPU** in **92.4 seconds**. It ran one tiny base update, baseline evaluation, two
one-update SFT segments with exact continuation, and three complete development
reports. It verified schema **8**, source/settings/optimizer lineage, unchanged
parent bytes, model/report hashes and pending human acceptance. The overflow
suite fixture deliberately tests mechanics, not conversational quality.

Runtime: Rust **1.93.0**, WSL Linux **6.6.114.1**, CUDA driver API **13010**,
NVRTC **12.8**, installed CUDA **13.1** headers, f32 host numerical checks,
fusion disabled. The compatible NVRTC package already present under
`/tmp/omega-cuda12-test/runtime` was selected via `LD_LIBRARY_PATH`; nothing was
downloaded or installed. Both GPU features were compiled into the tested binary.
CUDA remains experimental; this bounded run does not qualify every NVIDIA
architecture, driver or training scale.

## Remaining acceptance

**FT11 is partial:** the new workflow still needs bounded Vulkan qualification on
an idle A770. The reported active remote run was preserved; historical A770 tests
are not reused as fresh evidence for this change.

**FT01/FT12 remain open:** a compatible actual parent, approved conversation data,
scope, rubric/quality thresholds and budgets are unresolved. The `.env` pilot
budget fields remain blank. Local base data has empty conversation partitions
and an unresolved permitted-use note. See [the audit](post-training-audit.md).
Real human scoring, sealed-test final acceptance, model card and verified
backup/restore remain required. The future helper and ZH14 are deferred.

Recovery intentionally requires the original config path to exist, although it
uses the frozen configuration contents and retained executable. Time limits act
at safe boundaries and record overruns; storage admission uses conservative
checkpoint headroom and measured report/output usage. No automatic deletion or
quality promotion occurs. These are local software/hardware tests, not clean-host
release certification or a production model-quality claim.
