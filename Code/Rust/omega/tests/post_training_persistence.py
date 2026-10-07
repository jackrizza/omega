"""Linux subprocess/PTY acceptance: python3 post_training_persistence.py /path/omega.

Uses only temporary offline fixtures, a copied launcher, and identity-checked own
workers. Python is a test dependency only. No checkpoint establishes model quality.
Use a release or stripped test binary so executable hashing fits launch deadlines.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def matching(identity):
    if not identity:
        return False
    try:
        fields = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(") ", 1)[1].split()
        return (fields[0] != "Z" and fields[19] == identity["start"]
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot"])
    except (FileNotFoundError, ProcessLookupError):
        return False


def own_signal(identity, number):
    assert matching(identity), f"Own worker identity no longer live: {identity}"
    os.kill(identity["pid"], number)


def drain(master, duration=0.25):
    end = time.monotonic() + duration
    output = b""
    while time.monotonic() < end:
        if select.select([master], [], [], 0.02)[0]:
            try:
                output += os.read(master, 65536)
            except OSError:
                break
    return output


def main():
    assert sys.platform == "linux", "This test requires Linux workers and PTYs"
    binary = Path(sys.argv[1]).resolve(strict=True)
    deadline = time.monotonic() + 180
    with tempfile.TemporaryDirectory(prefix="omega-post-training-persistence-") as temporary:
        root = Path(temporary)
        state = root / "state"
        project = root / "project"
        project.mkdir()
        launcher = root / "omega-launcher"
        shutil.copy2(binary, launcher)
        env = {**os.environ, "OMEGA_STATE_DIR": str(state), "TERM": "xterm-256color",
               "PATH": "/usr/bin:/bin"}
        active_ui = []
        paused = []

        def cli(*arguments, ok=True, extra_env=None, executable=None):
            remaining = deadline - time.monotonic()
            assert remaining > 0, "Overall 180-second acceptance deadline exceeded"
            result = subprocess.run([str(executable or binary), *map(str, arguments)],
                                    cwd=project, env={**env, **(extra_env or {})},
                                    capture_output=True, timeout=min(30, remaining))
            if ok:
                assert result.returncode == 0, result.stderr.decode(errors="replace")
            else:
                assert result.returncode != 0, result.stdout.decode(errors="replace")
            return result

        def wait_for(predicate, label, seconds=45):
            end = min(deadline, time.monotonic() + seconds)
            while time.monotonic() < end:
                value = predicate()
                if value:
                    return value
                time.sleep(0.005)
            statuses = {p.parent.name: read(p) for p in (state / "jobs").glob("*/status.json")}
            raise AssertionError(f"Timeout waiting for {label}: {statuses}")

        def status(job):
            return read(state / "jobs" / job / "status.json")

        def spec(job):
            return read(state / "jobs" / job / "job.json")

        def terminal(job):
            result = wait_for(lambda: status(job) if status(job).get("status") in
                              ["completed", "stopped", "failed"] else None, f"job {job} completion")
            wait_for(lambda: not matching(result.get("process")), "worker lock release", 10)
            return result

        def pause_at(job, workflow, phase, require_updates=False):
            def selected():
                journal = read(workflow / "workflow.json")
                record = status(job)
                progress = (record.get("progress") or {}).get("completed_updates", 0)
                if journal.get("phase") == phase and (not require_updates or progress > 1):
                    return record.get("process")
                assert record.get("status") not in ["failed", "completed"], record
                return None
            identity = wait_for(selected, phase)
            own_signal(identity, signal.SIGSTOP)
            paused.append(identity)
            return identity

        def resume_signal(identity):
            own_signal(identity, signal.SIGCONT)
            paused.remove(identity)

        def crash(identity):
            # Resume before kill so a failed assertion cannot strand a stopped worker.
            if identity in paused:
                resume_signal(identity)
            own_signal(identity, signal.SIGKILL)
            wait_for(lambda: not matching(identity), "owned worker death", 10)
            cli("jobs")  # Reconcile stale running records through the public API.

        def verified_reports(workflow):
            journal = read(workflow / "workflow.json")
            for report in journal["reports"] + journal["incomplete_reports"]:
                assert digest(Path(report)) == journal["report_hashes"][report]
            complete = [read(Path(p)) for p in journal["reports"]]
            assert all(r["complete"] for r in complete)
            assert sum(r["baseline"] for r in complete) == 1
            return journal, complete

        def ui_detach(config):
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 120, 0, 0))
            before = termios.tcgetattr(slave)
            def controlling_terminal():
                os.setsid()
                fcntl.ioctl(0, termios.TIOCSCTTY, 0)
            child = subprocess.Popen([str(binary), "--project", str(config)], stdin=slave,
                                     stdout=slave, stderr=slave, cwd=project, env=env,
                                     preexec_fn=controlling_terminal)
            active_ui.append(child)
            try:
                output = drain(master, 0.5)
                assert child.poll() is None, output.decode(errors="replace")
                os.write(master, b"q")
                drain(master, 0.3)
                assert child.wait(timeout=10) == 0
                assert termios.tcgetattr(slave) == before, "PTY terminal settings not restored"
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                active_ui.remove(child)
                os.close(master)
                os.close(slave)

        data = project / "datasets"
        for folder in ["base", "train", "validation", "sealed", "regression"]:
            (data / folder).mkdir(parents=True)
        (data / "base/a.txt").write_text("Offline base fixture words for tokenization. " * 5)
        def conversations(folder, prefix, count):
            records = [{"schema_version": 1, "messages": [
                {"role": "user", "content": f"{prefix}{i}"},
                {"role": "assistant", "content": f"answer {prefix}{i}"}]} for i in range(count)]
            (data / folder / "data.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
        conversations("train", "t", 128)
        conversations("validation", "v", 64)
        conversations("sealed", "s", 2)
        (data / "regression/a.txt").write_text("Held out base language regression fixture.")
        suite = {"schema_version": 1, "name": "offline context boundary fixture",
                 "generation": {"max_new_tokens": 1, "strategy": {"kind": "greedy"}},
                 "cases": [{"id": "context", "system": None, "turns": [
                     {"prompt": "x" * 100, "checks": [{"kind": "context_overflow"}]}]}]}
        (project / "suite.json").write_text(json.dumps(suite))
        base_config = project / "base.toml"
        base_text = '''omega_schema_version = 2
name = "seed"
[dataset]
selections = ["base"]
format = "text"
[tokenizer]
path = "tokenizer.json"
vocab_size = 260
min_frequency = 1
chat_protocol = true
[model]
context_length = 32
d_model = 8
heads = 1
layers = 1
d_ff = 16
[training]
epochs = 1
max_updates = 1
cpu_threads = 1
matmul_threads = 1
[pipeline]
prepare_tokenizer = true
prepare_dataset = false
prepare_cache = false
benchmark = false
train = true
evaluate = false
'''
        base_config.write_text(base_text)
        try:
            base_job = cli("start", base_config, "--yes", executable=launcher).stdout.decode().strip()
            built = terminal(base_job)
            assert built["checkpoint"], built
            parent = Path(built["checkpoint"])
            assert (parent / "COMPLETE").is_file()
            parent_hashes = {p.name: digest(p) for p in parent.iterdir() if p.is_file()}
            print("PASS offline chat tokenizer and base checkpoint via single Omega binary", flush=True)

            def make_config(name):
                config = project / f"{name}.toml"
                config.write_text(base_text.replace('name = "seed"', f'name = "{name}"') + f'''
[post_training]
parent = {json.dumps(str(parent))}
output = {json.dumps(str(project / name))}
purpose = "Offline persistence software acceptance only"
language = "English"
training = ["train"]
validation = ["validation"]
base_validation = ["regression"]
sealed = ["sealed"]
suite = "suite.json"
rubric_version = "fixture-v1"
epochs = 1
learning_rate = 0.001
seed = 7
batch_size = 1
max_batch_tokens = 32
segment_updates = 32
max_updates = 128
max_seconds = 120.0
evaluation_max_seconds = 30.0
max_evaluations = 10
max_output_bytes = 67108864
min_test_pass_rate = 1.0
max_assistant_loss = 100.0
max_base_loss_increase = 100.0
min_human_score = 3
''')
                return config, project / name

            config, workflow = make_config("graceful")
            job = cli("post-training", "start", config, "--yes", executable=launcher).stdout.decode().strip()
            identity = pause_at(job, workflow, "training")
            snapshot = spec(job)
            ui_detach(config)
            ui_detach(config)  # Reconnect and detach a second frontend.
            assert status(job)["process"] == identity and matching(identity)
            competing, _ = make_config("competing")
            denied = cli("post-training", "start", competing, "--yes", ok=False,
                         extra_env={"OMEGA_STATE_DIR": str(root / "other-state")})
            assert b"already running" in denied.stderr, denied.stderr
            resume_signal(identity)
            cli("stop", job)
            stopped = terminal(job)
            assert stopped["status"] == "stopped", stopped
            assert (Path(stopped["checkpoint"]) / "COMPLETE").is_file()
            journal, _ = verified_reports(workflow)
            assert journal["phase"] == "stopped"
            print("PASS PTY detach/reconnect, host-wide compute exclusion and verified stop", flush=True)

            frozen = (workflow / "workflow.json").read_bytes()
            (data / "validation").rename(data / "validation-away")
            cli("post-training", "reports", workflow)
            cli("post-training", "recover", workflow, "--yes", ok=False)
            assert (workflow / "workflow.json").read_bytes() == frozen
            (data / "validation-away").rename(data / "validation")
            launcher.unlink()  # The original copied launcher is no longer available.
            resumed = cli("post-training", "recover", workflow, "--yes").stdout.decode().strip()
            assert spec(resumed)["executable"] == snapshot["executable"]
            assert digest(Path(spec(resumed)["executable"])) == snapshot["executable_sha256"]
            result = terminal(resumed)
            assert result["status"] == "completed", result
            journal, reports = verified_reports(workflow)
            assert journal["phase"] == "completed" and journal["completed_updates"] == 128, journal
            assert len({r["conversation"]["checkpoint"]["path"] for r in reports}) == len(reports)
            print("PASS frozen-input rejection, report access without data, retained-executable recovery", flush=True)

            config, workflow = make_config("crashed")
            job = cli("post-training", "start", config, "--yes").stdout.decode().strip()
            identity = pause_at(job, workflow, "training", require_updates=True)
            reservation = read(workflow / "workflow.json")["charged_updates"]
            crash(identity)
            assert status(job)["status"] == "failed"
            job = cli("post-training", "recover", workflow, "--yes").stdout.decode().strip()
            identity = pause_at(job, workflow, "evaluation")
            before = read(workflow / "workflow.json")
            checkpoint = Path(before["checkpoint"])
            checkpoint_hash = digest(checkpoint / "resume.json")
            assert before["charged_updates"] >= reservation
            crash(identity)
            recovered = cli("post-training", "recover", workflow, "--yes").stdout.decode().strip()
            result = terminal(recovered)
            assert result["status"] == "completed", result
            journal, reports = verified_reports(workflow)
            assert journal["phase"] in ["completed", "budget_exhausted"], journal
            assert before["charged_updates"] <= journal["charged_updates"] <= 128
            assert journal["completed_updates"] <= journal["charged_updates"]
            assert journal["evaluations"] >= before["evaluations"] and journal["evaluations"] <= 10
            assert digest(checkpoint / "resume.json") == checkpoint_hash
            assert len({r["conversation"]["checkpoint"]["path"] for r in reports}) == len(reports)
            assert {p.name: digest(p) for p in parent.iterdir() if p.is_file()} == parent_hashes
            print("PASS training/evaluation crash recovery, conservative reservations and unchanged parent", flush=True)

            config, workflow = make_config("baseline-crash")
            job = cli("post-training", "start", config, "--yes").stdout.decode().strip()
            identity = pause_at(job, workflow, "baseline")
            before = read(workflow / "workflow.json")
            assert not before["reports"] and before["evaluations"] == 1
            crash(identity)
            resumed = cli("post-training", "recover", workflow, "--yes").stdout.decode().strip()
            assert terminal(resumed)["status"] == "completed"
            journal, reports = verified_reports(workflow)
            assert journal["phase"] == "completed" and journal["evaluations"] >= 6
            assert len(reports) == 5 and journal["completed_updates"] == 128
            print("PASS baseline crash recovery reserves failed attempt without accepting a partial baseline", flush=True)
        finally:
            for child in active_ui:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
            # Only signal workers whose PID, start time and boot identity still
            # match this test's private job records. Resume any paused own worker.
            for path in (state / "jobs").glob("*/status.json"):
                identity = read(path).get("process")
                if not matching(identity):
                    continue
                os.kill(identity["pid"], signal.SIGCONT)
                try:
                    subprocess.run([str(binary), "stop", path.parent.name], env=env,
                                   capture_output=True, timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                end = time.monotonic() + 5
                while matching(identity) and time.monotonic() < end:
                    time.sleep(0.05)
                if matching(identity):
                    os.kill(identity["pid"], signal.SIGKILL)


if __name__ == "__main__":
    main()
