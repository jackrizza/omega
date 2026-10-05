#!/bin/sh
set -eu

# Python is available only in the builder. Discover every workspace bin instead
# of assuming there is one per crate, or copying the last shared `main` output.
python3 - <<'PY'
import json
import os
from pathlib import Path
import shutil
import subprocess

jobs = int(os.environ.get("BUILD_JOBS", "2"))
if jobs < 1:
    raise SystemExit("BUILD_JOBS must be positive")
metadata = json.loads(subprocess.check_output([
    "cargo", "metadata", "--locked", "--no-deps", "--format-version", "1",
], text=True))
members = set(metadata["workspace_members"])
inventory = []
for package in sorted(metadata["packages"], key=lambda item: item["name"]):
    if package["id"] not in members:
        continue
    binaries = [target for target in package["targets"] if "bin" in target["kind"]]
    if not binaries:
        continue
    name = package["name"]
    # Separate target directories prevent Cargo from reusing another crate's main.
    target_dir = Path("/build") / name
    subprocess.run([
        "cargo", "build", "--release", "--locked", "--package", name,
        "--bins", "--jobs", str(jobs), "--target-dir", str(target_dir),
    ], check=True)
    destination = Path("/opt/omega/bin") / name
    destination.mkdir(parents=True, exist_ok=True)
    for binary in binaries:
        output = destination / binary["name"]
        shutil.copy2(target_dir / "release" / binary["name"], output)
        inventory.append({"package": name, "binary": binary["name"], "path": str(output)})
Path("/opt/omega/binaries.json").write_text(json.dumps(inventory, indent=2) + "\n")
PY
