"""Linux PTY acceptance test. Run with python3 persistence.py /absolute/omega.

Only temporary projects and their own workers are touched. No third-party Python
packages are needed; Python is a test dependency, never an Omega runtime dependency.
"""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

BINARY = str(Path(sys.argv[1]).resolve())
FIXTURE = Path(__file__).resolve().parents[2] / "test-fixtures/wordlevel.json"


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for worker state")


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def ui(config, env):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 110, 0, 0))
    before = termios.tcgetattr(slave)
    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    child = subprocess.Popen([BINARY, "--project", str(config)], stdin=slave,
                             stdout=slave, stderr=slave, env=env, preexec_fn=controlling_terminal)
    return child, master, slave, before


def drain(master, duration=0.7):
    deadline = time.monotonic() + duration
    result = b""
    while time.monotonic() < deadline:
        if select.select([master], [], [], 0.05)[0]:
            try:
                result += os.read(master, 65536)
            except OSError:
                break
    return result


with tempfile.TemporaryDirectory(prefix="omega-persistence-") as temporary:
    root = Path(temporary)
    state = root / "state"
    env = {**os.environ, "OMEGA_STATE_DIR": str(state), "TERM": "xterm-256color"}
    # Deliberately run outside the checkout, with no Rust tools on PATH.
    env["PATH"] = "/usr/bin:/bin"
    project = root / "project"
    project.mkdir()
    (project / "datasets/base").mkdir(parents=True)
    (project / "datasets/base/a.txt").write_text("hello world " * 60)
    (project / "tokenizer.json").write_bytes(FIXTURE.read_bytes())
    config = project / "model.toml"
    config.write_text('''omega_schema_version = 1
name = "persistence"
[dataset]
selections = ["base"]
format = "text"
[tokenizer]
path = "tokenizer.json"
[model]
context_length = 3
d_model = 4
heads = 1
layers = 1
d_ff = 8
[training]
epochs = 1000
max_updates = 3000
cpu_threads = 1
matmul_threads = 1
save_every_updates = 10
''')
    owned = []
    active_ui = None
    try:
        for ending in ["detach", "kill", "hangup"]:
            child, master, slave, before = ui(config, env)
            active_ui = child
            initial_screen=drain(master)
            assert child.poll() is None, initial_screen.decode(errors="replace")
            previous = set((state / "jobs").glob("*/job.json"))
            os.write(master, b"\r")  # review pipeline
            output = b""
            def reviewed():
                global output
                output += drain(master, 0.2)
                return b"START reviewed operation" in output
            try:
                wait_for(reviewed)
            except AssertionError:
                print(output[-12000:].decode(errors="replace"), flush=True)
                raise

            os.write(master, b"\r")  # launch reviewed configuration
            def launched():
                global output
                output += drain(master, 0.1)
                return next(iter(set((state / "jobs").glob("*/job.json")) - previous), None)
            try:
                job_file = wait_for(launched)
            except AssertionError:
                print(output[-12000:].decode(errors="replace"), flush=True)
                raise

            job_id = job_file.parent.name
            owned.append(job_id)
            status = job_file.with_name("status.json")
            wait_for(lambda: (read(status).get("progress") or {}).get("completed_updates", 0) > 1)
            snapshot = read(job_file)
            original_config = config.read_text()
            config.write_text(original_config.replace("epochs = 1000", "epochs = 999"))
            identity = read(status)["process"]
            initial = read(status)["progress"]["completed_updates"]
            drain(master)
            if ending == "detach":
                os.write(master, b"q")
                drain(master)
                assert child.wait(timeout=10) == 0
                assert termios.tcgetattr(slave) == before, "Terminal settings were not restored"
                os.close(master)
                os.close(slave)
            elif ending == "kill":
                child.kill()
                child.wait(timeout=10)
                os.close(master)
                os.close(slave)
            else:
                os.close(master)
                os.close(slave)
                child.send_signal(signal.SIGHUP)
                child.wait(timeout=10)
            active_ui = None
            wait_for(lambda: (read(status).get("progress") or {}).get("completed_updates", 0) > initial + 2)
            assert read(status)["process"] == identity
            # Competing launches are rejected even with another XDG state root.
            other = subprocess.run([BINARY, "start", str(config), "--yes"], env={**env, "OMEGA_STATE_DIR": str(root / "other")}, capture_output=True)
            assert other.returncode != 0 and b"already running" in other.stderr, other.stderr
            reconnect, m, s, _ = ui(config, env)
            active_ui = reconnect
            drain(m)
            os.write(m, b"j")
            drain(m)
            os.write(m, b"\r")
            monitor = drain(m)
            assert b"Training monitor" in monitor, "Jobs/Enter did not open the worker monitor"
            os.write(m, b"\t")
            drain(m)
            os.write(m, b"1")
            drain(m)
            os.write(m, b"q")
            drain(m)
            assert reconnect.wait(timeout=10) == 0
            os.close(m)
            os.close(s)
            active_ui = None
            assert read(status)["process"] == identity
            assert len(list((state / "jobs").glob("*/job.json"))) == len(owned)
            assert read(job_file) == snapshot, "Launch snapshot changed"
            config.write_text(original_config)
            subprocess.run([BINARY, "stop", job_id], env=env, check=True, capture_output=True)
            result = wait_for(lambda: read(status) if read(status).get("status") in ["stopped", "failed", "completed"] else None)
            assert result["status"] == "stopped", result
            assert (Path(result["checkpoint"]) / "COMPLETE").is_file()
            # Locks must be released before starting the next scenario.
            wait_for(lambda: not Path(f'/proc/{identity["pid"]}').exists() or Path(f'/proc/{identity["pid"]}/stat').read_text().split(") ")[1].startswith("Z"))
            print(f"PASS {ending}: updates continued, same worker reconnected, stop checkpoint verified", flush=True)
        # Resume from an earlier periodic checkpoint through the retained executable.
        checkpoints = read(job_file.with_name("checkpoints.json"))
        earlier = checkpoints[0]
        config.write_text(original_config.replace("max_updates = 3000", "max_updates = 2"))
        resume_launch = subprocess.run([BINARY, "resume", str(config), earlier, "--yes"], env=env, capture_output=True)
        assert resume_launch.returncode == 0, resume_launch.stderr.decode()
        resumed = resume_launch.stdout.decode().strip()
        owned.append(resumed)
        resume_status = state / "jobs" / resumed / "status.json"
        result = wait_for(lambda: read(resume_status) if read(resume_status).get("status") in ["stopped", "failed", "completed"] else None)
        assert result["status"] == "stopped", result
        resumed_spec = read(resume_status.with_name("job.json"))
        assert resumed_spec["executable"] == snapshot["executable"]
        assert (Path(result["checkpoint"]) / "COMPLETE").is_file()
        print("PASS exact resume uses retained executable, including earlier periodic checkpoint", flush=True)
        identity = result["process"]
        wait_for(lambda: not Path(f'/proc/{identity["pid"]}').exists() or Path(f'/proc/{identity["pid"]}/stat').read_text().split(") ")[1].startswith("Z"))
        # Unexpected worker death retains the last verified periodic checkpoint.
        config.write_text(original_config)
        crash_launch = subprocess.run([BINARY, "start", str(config), "--yes"], env=env, capture_output=True, check=True)
        crashed = crash_launch.stdout.decode().strip()
        owned.append(crashed)
        crash_status = state / "jobs" / crashed / "status.json"
        running = wait_for(lambda: read(crash_status) if read(crash_status).get("checkpoint") else None)
        identity = running["process"]
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot"]
        fields = Path(f'/proc/{identity["pid"]}/stat').read_text().split(") ")[1].split()
        assert fields[19] == identity["start"]
        os.kill(identity["pid"], signal.SIGKILL)
        def interrupted():
            subprocess.run([BINARY, "jobs"], env=env, capture_output=True, check=True)
            result = read(crash_status)
            return result if result.get("status") == "failed" else None
        result = wait_for(interrupted)
        assert (Path(result["checkpoint"]) / "COMPLETE").is_file()
        print("PASS unexpected worker termination retains a complete checkpoint and records failure", flush=True)
        # A missing tokenizer must yield failure, never completion/checkpoint success.
        config.write_text(original_config.replace('path = "tokenizer.json"', 'path = "missing.json"'))
        failed = subprocess.run([BINARY, "start", str(config), "--yes"], env=env, capture_output=True)
        assert failed.returncode == 0 or b"Worker failed to start" in failed.stderr, failed.stderr
        history = wait_for(lambda: json.loads(subprocess.run([BINARY, "jobs"], env=env, capture_output=True, check=True).stdout))
        latest = history[0]
        failed_status = state / "jobs" / latest["id"] / "status.json"
        result = wait_for(lambda: read(failed_status) if read(failed_status).get("status") == "failed" else None)
        assert result["checkpoint"] is None
        print("PASS worker failure is recorded without a successful checkpoint claim", flush=True)
        # A PID alone cannot establish ownership, even when it is a live process.
        record = read(status)
        record["status"] = "running"
        record["process"] = {"pid": os.getpid(), "boot": "wrong-boot", "start": "0"}
        status.write_text(json.dumps(record))
        history = subprocess.run([BINARY, "jobs"], env=env, capture_output=True, check=True)
        assert all(row["status"] != "running" for row in json.loads(history.stdout))
        print("PASS stale process identity rejected; executable operates outside checkout without Rust", flush=True)
    finally:
        if active_ui and active_ui.poll() is None:
            active_ui.kill()
            active_ui.wait(timeout=10)
        for job_id in owned:
            subprocess.run([BINARY, "stop", job_id], env=env, capture_output=True)
