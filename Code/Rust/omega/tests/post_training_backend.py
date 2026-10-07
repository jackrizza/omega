"""Opt-in Linux qualification: python3 post_training_backend.py BINARY cpu|cuda|vulkan.

Only stdlib, existing runtime, tiny temporary fixtures and one supplied executable.
The 180-second deadline includes cleanup (10 seconds reserved). This verifies
software state transitions, not conversational quality or production acceptance.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    return {str(p.relative_to(path)): digest(p) for p in path.rglob("*") if p.is_file()}


def process_matches(identity):
    if not identity:
        return False
    try:
        stat = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(") ", 1)[1].split()
        return (stat[0] != "Z" and stat[19] == identity["start"]
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot"])
    except FileNotFoundError:
        return False


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("cpu", "cuda", "vulkan"):
        raise SystemExit("Usage: post_training_backend.py ABSOLUTE_BINARY cpu|cuda|vulkan")
    if not sys.platform.startswith("linux"):
        raise SystemExit("This opt-in detached-worker test requires Linux")
    binary = Path(sys.argv[1]).resolve(strict=True)
    backend = sys.argv[2]
    start = time.monotonic()
    deadline = start + 170.0  # reserve ten seconds for owned-worker cleanup
    root = Path(tempfile.mkdtemp(prefix="omega-post-training-backend-")).resolve()
    parent_directory = root.parent
    sentinel = uuid.uuid4().hex
    (root / "TEST_OWNER").write_text(sentinel)
    state = root / "state"
    env = {**os.environ, "OMEGA_STATE_DIR": str(state)}

    def remaining():
        seconds = deadline - time.monotonic()
        assert seconds > 0, "Tiny backend workflow exceeded its overall deadline"
        return seconds

    def command(*args):
        result = subprocess.run([str(binary), *map(str, args)], env=env,
                                capture_output=True, text=True, timeout=remaining())
        assert result.returncode == 0, (args, result.stdout, result.stderr)
        return result.stdout.strip()

    def finished(job):
        directory = state / "jobs" / job
        assert directory.resolve().parent == (state / "jobs").resolve(), "Invalid job path"
        while True:
            remaining()
            status = read(directory / "status.json")
            if status["status"] in ("completed", "stopped", "failed"):
                assert status["status"] == "completed", (status, (directory / "worker.log").read_text())
                # A terminal status precedes worker shutdown and compute-lock release.
                if not process_matches(status.get("process")):
                    return directory, status
            time.sleep(min(0.1, remaining()))

    def put(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def chat(prompt, answer):
        return json.dumps({"schema_version": 1, "messages": [
            {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]})

    def cleanup():
        # Never signal a recycled PID or an executable outside this owned state root.
        stop_deadline = start + 179.0
        records = list((state / "jobs").glob("*/status.json"))
        for path in records:
            status = read(path)
            identity = status.get("process")
            if not process_matches(identity):
                continue
            spec = read(path.parent / "job.json")
            retained = Path(spec["executable"]).resolve(strict=True)
            assert retained.is_relative_to(state.resolve()), "Worker executable is not test-owned"
            assert digest(retained) == spec["executable_sha256"], "Retained executable changed"
            actual = Path(f'/proc/{identity["pid"]}/exe').resolve(strict=True)
            assert actual == retained, "Worker executable identity mismatch; preserving temporary files"
            timeout = max(0.05, min(1.0, stop_deadline - time.monotonic()))
            try:
                subprocess.run([str(binary), "stop", status["id"]], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            grace = min(time.monotonic() + 1.0, stop_deadline)
            while process_matches(identity) and time.monotonic() < grace:
                time.sleep(0.05)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not process_matches(identity):
                    break
                try:
                    os.kill(identity["pid"], sig)
                except ProcessLookupError:
                    break
                grace = min(time.monotonic() + 0.5, stop_deadline)
                while process_matches(identity) and time.monotonic() < grace:
                    time.sleep(0.05)
            assert not process_matches(identity), "Owned worker remains alive; preserve temporary root"
        assert root.parent == parent_directory and root.resolve() == root and not root.is_symlink()
        assert (root / "TEST_OWNER").read_text() == sentinel, "Temporary directory identity changed"
        shutil.rmtree(root)

    try:
        put("datasets/base/text.txt", "tiny base text")
        put("datasets/train/data.jsonl", chat("a", "b") + "\n" + chat("c", "d") + "\n")
        put("datasets/validation/data.jsonl", chat("e", "f") + "\n")
        put("datasets/sealed/data.jsonl", chat("g", "h") + "\n")
        put("datasets/regression/text.txt", "ijklm")
        config = root / "model.toml"
        common = f'''name = "backend-fixture"
[dataset]
selections = ["base"]
[tokenizer]
vocab_size = 260
min_frequency = 1
chat_protocol = true
[model]
context_length = 32
d_model = 4
heads = 1
layers = 1
d_ff = 8
[training]
backend = "{backend}"
device = 0
epochs = 1
learning_rate = 0.001
batch_size = 1
max_batch_tokens = 32
max_updates = 1
save_every_updates = 1
[pipeline]
prepare_tokenizer = true
train = true
'''
        config.write_text("omega_schema_version = 1\n" + common, encoding="utf-8")
        command("doctor", "--backend", backend, "--device", "0")
        _, base = finished(command("start", config, "--yes"))
        parent = Path(base["checkpoint"])
        original = snapshot(parent)
        base_resume = read(parent / "resume.json")
        assert base_resume["schema_version"] == {"cpu": 3, "vulkan": 4, "cuda": 7}[backend]
        put("suite.json", json.dumps({"schema_version": 1, "name": "software overflow fixture",
            "generation": {"max_new_tokens": 1, "strategy": {"kind": "greedy"}},
            "cases": [{"id": "overflow", "system": None, "turns": [{"prompt": "x" * 40,
                "checks": [{"kind": "context_overflow"}]}]}]}))
        post = f'''
[post_training]
parent = {json.dumps(str(parent))}
output = "workflow"
purpose = "Bounded software fixture; no model quality acceptance"
language = "English"
training = ["train"]
validation = ["validation"]
base_validation = ["regression"]
sealed = ["sealed"]
suite = "suite.json"
rubric_version = "fixture-v1"
epochs = 1
learning_rate = 0.001
seed = 42
batch_size = 1
max_batch_tokens = 32
segment_updates = 1
max_updates = 2
max_seconds = 120.0
evaluation_max_seconds = 45.0
max_evaluations = 3
max_output_bytes = 67108864
min_test_pass_rate = 1.0
max_assistant_loss = 100.0
max_base_loss_increase = 100.0
min_human_score = 3
'''
        config.write_text("omega_schema_version = 2\n" + common + post, encoding="utf-8")
        assert json.loads(command("post-training", "ready", config))["passed"]
        directory, _ = finished(command("post-training", "start", config, "--yes"))
        workflow = read(root / "workflow/workflow.json")
        assert workflow["phase"] == "completed", workflow
        assert (workflow["completed_updates"], workflow["charged_updates"], workflow["evaluations"]) == (2, 2, 3)
        assert workflow["training_complete"] and not workflow["pending_evaluation"]
        assert not workflow.get("incomplete_reports", [])
        assert snapshot(parent) == original, "Base checkpoint changed during SFT"
        assert len(workflow["reports"]) == 3
        checkpoints = []
        for index, value in enumerate(workflow["reports"]):
            path = Path(value)
            report = read(path)
            assert digest(path) == workflow["report_hashes"][value]
            assert path.with_suffix(".md").is_file()
            assert report["baseline"] == (index == 0) and report["completed_updates"] == index
            assert report["complete"] and report["error"] is None
            assert all(math.isfinite(report[key]) for key in ("assistant_loss", "base_loss"))
            conversation = report["conversation"]
            assert conversation["status"] == "complete"
            assert all(case["status"] == "passed" for case in conversation["cases"])
            checkpoint = Path(conversation["checkpoint"]["path"])
            assert digest(checkpoint / "model.mpk") == conversation["checkpoint"]["model_sha256"]
            if index:
                checkpoints.append(checkpoint)
        assert len(set(checkpoints)) == 2
        resumes = [read(p / "resume.json") for p in checkpoints]
        expected = {"cpu": 5, "vulkan": 6, "cuda": 8}[backend]
        lineage = {"name": parent.name, "manifest_sha256": digest(parent / "manifest.json"),
                   "model_sha256": digest(parent / "model.mpk"), "resume_sha256": digest(parent / "resume.json")}
        for index, (checkpoint, resume) in enumerate(zip(checkpoints, resumes), 1):
            assert (checkpoint / "COMPLETE").is_file()
            assert resume["schema_version"] == expected and resume["parent"] == lineage
            assert resume["progress"]["completed_updates"] == index
            assert resume["progress"]["completed_targets"] == index * 2
            assert resume["objective"] == "assistant-next-token-omega-chat-v1"
            assert digest(checkpoint / "model.mpk") == resume["model_sha256"]
            assert digest(checkpoint / "optimizer.mpk") == resume["optimizer_sha256"]
        assert resumes[0]["progress"]["next_example_index"] == 1
        assert resumes[1]["progress"]["completed_epochs"] == 1
        for key in ("parent", "source_identity", "runtime", "options", "tokenizer", "adam"):
            assert resumes[0][key] == resumes[1][key], f"Segment continuation changed {key}"
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        segments = [event for event in events if event["kind"] == "stage"
                    and event["data"].get("name") == "Fine-tuning segment"]
        assert len(segments) == 2, "Expected two separately orchestrated SFT segments"
        ranking = json.loads(command("post-training", "reports", root / "workflow"))
        assert len(ranking) == 3 and all(not item["eligible"] for item in ranking)
        print(f"PASS {backend}: unchanged parent, baseline + two one-update SFT segments, "
              f"schema {expected}, optimizer/progress lineage, complete metrics/report hashes; "
              f"human acceptance pending ({time.monotonic() - start:.1f}s)", flush=True)
    finally:
        try:
            cleanup()
        except Exception:
            print(f"Cleanup could not safely finish; temporary evidence retained at {root}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
