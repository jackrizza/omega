"""Opt-in bounded backend/worker qualification: script ABSOLUTE_BINARY cuda|cpu.

Uses temporary data, a four-wide model, and a handful of updates. A compatible
runtime must already be selected in the environment. No runtime is installed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

binary = str(Path(sys.argv[1]).resolve())
backend = sys.argv[2]
assert backend in ("cpu", "cuda", "vulkan")

with tempfile.TemporaryDirectory(prefix="omega-backend-") as tmp:
    root = Path(tmp)
    state = root / "state"
    env = {**os.environ, "OMEGA_STATE_DIR": str(state)}
    (root / "datasets/base").mkdir(parents=True)
    (root / "datasets/chat").mkdir(parents=True)
    (root / "datasets/base/a.txt").write_text("Hello world! Hi ok")
    (root / "datasets/chat/a.jsonl").write_text(json.dumps({"schema_version": 1, "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "ok"}]}))
    config = root / "model.toml"
    config.write_text(f'''omega_schema_version = 1
name = "hardware"
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
backend = "{backend}"
epochs = 1
batch_size = 2
save_every_updates = 1
[pipeline]
prepare_tokenizer = true
benchmark = true
train = true
[benchmark]
samples = 1
warmup = 1
max_seconds = 120.0
[assistant]
selections = ["chat"]
epochs = 1
learning_rate = 0.001
''')
    if backend == "cuda":
        missing = subprocess.run([binary, "doctor", "--backend", backend], env={**env, "CUDA_PATH": str(root / "missing-toolkit")}, capture_output=True)
        assert missing.returncode != 0 and b"headers" in missing.stderr, missing.stderr
    subprocess.run([binary, "doctor", "--backend", backend, "--device", "0"], env=env, check=True)

    def command(*args):
        result = subprocess.run([binary, *map(str, args)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def finished(job):
        directory = state / "jobs" / job
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = json.loads((directory / "status.json").read_text())
            if status["status"] in ("completed", "stopped", "failed"):
                assert status["status"] == "completed", (status, (directory / "worker.log").read_text())
                return directory, status
            time.sleep(0.2)
        command("stop", job)
        raise AssertionError("Tiny backend pipeline exceeded 180 seconds")

    job = command("start", config, "--yes")
    try:
        directory, status = finished(job)
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        reports = [e["data"] for e in events if e["kind"] == "result" and "batches_per_epoch" in e["data"]]
        assert len(reports) == 1 and reports[0]["backend"] == backend
        assert all(s["seconds"] > 0 for s in reports[0]["samples"])
        checkpoints = json.loads((directory / "checkpoints.json").read_text())
        schemas = {json.loads((Path(p) / "resume.json").read_text())["schema_version"] for p in checkpoints}
        expected = {"cpu": {3, 5}, "vulkan": {4, 6}, "cuda": {7, 8}}[backend]
        assert schemas == expected, schemas
        assert (Path(status["checkpoint"]) / "COMPLETE").is_file()
        print(f"PASS {backend}: tokenizer, synchronized benchmark, base/assistant workers, schemas {sorted(schemas)}", flush=True)
        # Wait for the worker's terminal record to be flushed and lock released.
        time.sleep(1)
        config.write_text(config.read_text().replace("epochs = 1", "epochs = 2"))
        job = command("resume", config, status["checkpoint"], "--yes")
        _, resumed = finished(job)
        assert resumed["checkpoint"] != status["checkpoint"]
        print(f"PASS {backend}: same-binary assistant exact resume and verified final checkpoint", flush=True)
        time.sleep(1)
        job = command("resume", config, status["checkpoint"], "--yes")
        _, repeated = finished(job)
        first_hash = json.loads((Path(resumed["checkpoint"]) / "resume.json").read_text())["model_sha256"]
        second_hash = json.loads((Path(repeated["checkpoint"]) / "resume.json").read_text())["model_sha256"]
        assert first_hash == second_hash, "Identical checkpoint continuations diverged across worker processes"
        print(f"PASS {backend}: repeated separate-process continuation produced identical model bytes", flush=True)
    finally:
        subprocess.run([binary, "stop", job], env=env, capture_output=True)
