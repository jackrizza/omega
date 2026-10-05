#!/usr/bin/env python3
"""Run the prebuilt CPU benchmark in fresh, deadline-limited processes.

No builds, downloads, global environment changes, or dataset/checkpoint access.
The default sweep varies one thread pool at a time; tokenizer parallelism is
disabled in every child, including the backend-default baseline. Results are
written exclusively to a new JSON file. Example, from Code/Rust:

  python scripts/benchmark_cpu.py --output cpu-results.json
  python scripts/benchmark_cpu.py --output focused.json --threads 1:1 \
      --threads 4:1 --case forward:1:128 --case train:4:128

Process deadlines include initialization and measured work; kernel scheduling
and kill/reap overhead can slightly exceed a deadline. A cleanup reserve prevents
starting work at the total deadline. Windows memory is the kernel's process peak
working set, not tensor memory; other platforms explicitly report unavailable.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


ENVIRONMENT_KEYS = (
    "RAYON_NUM_THREADS",
    "RAYON_RS_NUM_CPUS",
    "MATMUL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)
DEFAULT_THREADS = ("default", "1:1", "2:1", "4:1", "8:1", "10:1", "20:1", "1:2", "1:4")
DEFAULT_CASES = (
    "train:1:32", "train:1:128", "train:4:32", "train:4:128",
    "generate:1:32", "generate:1:128",
)
CLEANUP_RESERVE_SECONDS = 0.25


def bounded_integer(minimum, maximum):
    def parse(value):
        try:
            number = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected an integer") from error
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"must be in {minimum}..={maximum}")
        return number
    return parse


def positive_seconds(maximum):
    def parse(value):
        try:
            number = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected seconds") from error
        if not math.isfinite(number) or not 0 < number <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and in (0, {maximum}]")
        return number
    return parse


def parse_threads(value):
    if value == "default":
        return value
    try:
        rayon, matmul = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected default or RAYON:MATMUL") from error
    if not 1 <= rayon <= 20 or not 1 <= matmul <= 4:
        raise argparse.ArgumentTypeError("Rayon must be 1..20 and matmul 1..4")
    return f"{rayon}:{matmul}"


def parse_case(value):
    try:
        mode, batch, context = value.split(":")
        batch, context = int(batch), int(context)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected MODE:BATCH:CONTEXT") from error
    if mode not in ("train", "generate", "forward") or batch not in (1, 4) or context not in (32, 128):
        raise argparse.ArgumentTypeError("use train|generate|forward, batch 1|4, context 32|128")
    return f"{mode}:{batch}:{context}"


def parser():
    workspace = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, required=True, help="new JSON path; parent must exist")
    result.add_argument("--executable", type=Path, default=workspace / "target/release/examples" / ("cpu_benchmark.exe" if os.name == "nt" else "cpu_benchmark"))
    result.add_argument("--threads", action="append", type=parse_threads, help="repeat default or RAYON:MATMUL to replace the staged sweep")
    result.add_argument("--case", action="append", type=parse_case, help="repeat MODE:BATCH:CONTEXT to replace the six default workloads")
    result.add_argument("--samples", type=bounded_integer(1, 20), default=5)
    result.add_argument("--iterations", type=bounded_integer(1, 8), default=1)
    result.add_argument("--warmup", type=bounded_integer(1, 3), default=1)
    result.add_argument("--new-tokens", type=bounded_integer(1, 16), default=8)
    result.add_argument("--profile", action="store_true", help="request stage timings; requires explicitly selected train cases only")
    result.add_argument("--max-seconds", type=bounded_integer(1, 10), default=5, help="cooperative child work budget; one operation may exceed it")
    result.add_argument("--timeout-seconds", type=positive_seconds(10), default=10.0, help="hard child deadline including startup, at most 10 seconds")
    result.add_argument("--total-seconds", type=positive_seconds(300), default=300.0, help="total process-sweep deadline, at most 300 seconds")
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints(workspace):
    """Only explicit workspace manifests, Rust source trees and this harness."""
    candidates = [workspace / "Cargo.toml", workspace / "Cargo.lock", Path(__file__).resolve()]
    for crate in ("omega-nn", "omega-training", "omega-tokenizer"):
        candidates.append(workspace / crate / "Cargo.toml")
        source_root = workspace / crate / "src"
        for directory, subdirectories, files in os.walk(source_root, followlinks=False):
            subdirectories[:] = sorted(name for name in subdirectories if not (Path(directory) / name).is_symlink())
            candidates.extend(Path(directory) / name for name in sorted(files) if name.endswith(".rs"))
    candidates.append(workspace / "omega-training/examples/cpu_benchmark.rs")
    for directory in (workspace, workspace.parent.parent):
        for name in ("rust-toolchain", "rust-toolchain.toml"):
            if (directory / name).is_file():
                candidates.append(directory / name)
    repository = workspace.parent.parent
    fingerprints = {}
    for path in sorted(set(candidates)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular repository source file: {path}")
        resolved = path.resolve()
        relative = resolved.relative_to(repository).as_posix()
        fingerprints[relative] = sha256_file(resolved)
    encoded = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": fingerprints}


def child_environment(inherited, threads):
    environment = inherited.copy()
    for name in ENVIRONMENT_KEYS:
        environment.pop(name, None)
    if threads != "default":
        rayon, matmul = threads.split(":")
        environment["RAYON_NUM_THREADS"] = rayon
        environment["MATMUL_NUM_THREADS"] = matmul
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


class PeakMemory:
    """Use the already-owned Windows process handle, without opening other PIDs."""

    def __init__(self):
        self.query = None
        self.reason = "portable per-process peak working-set measurement unavailable"
        if os.name != "nt":
            return
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]
        try:
            self.query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            self.query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            self.query.restype = wintypes.BOOL
            self.counters = Counters
            self.reason = None
        except (AttributeError, OSError) as error:
            self.reason = str(error)

    def measure(self, process):
        if self.query is None:
            return None, self.reason
        handle = getattr(process, "_handle", None)
        if handle is None:
            return None, "Python runtime does not expose its owned Windows process handle"
        counters = self.counters()
        counters.cb = ctypes.sizeof(counters)
        if not self.query(int(handle), ctypes.byref(counters), counters.cb):
            return None, f"GetProcessMemoryInfo failed with Windows error {ctypes.get_last_error()}"
        return int(counters.PeakWorkingSetSize), None


def run_process(command, environment, directory, timeout, memory):
    """Run one child, sample memory, and kill/reap on deadline or interruption."""
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    peak = None
    memory_error = memory.reason
    timed_out = False
    try:
        while True:
            measured, error = memory.measure(process)
            if measured is not None:
                peak = max(peak or 0, measured)
                memory_error = None
            elif peak is None:
                memory_error = error
            remaining = started + timeout - time.monotonic()
            if remaining <= 0:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()
                break
            try:
                stdout, stderr = process.communicate(timeout=min(0.025, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        measured, error = memory.measure(process)
        if measured is not None:
            peak = max(peak or 0, measured)
            memory_error = None
        elif peak is None:
            memory_error = error
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate()
        raise
    return {
        "status": "timeout" if timed_out else ("exited" if process.returncode == 0 else "error"),
        "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
        "timeout_seconds": timeout, "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "peak_working_set_bytes": peak,
        "memory_measurement": "GetProcessMemoryInfo.PeakWorkingSetSize" if peak is not None else None,
        "memory_unavailable_reason": memory_error,
    }


def reject_constant(value):
    raise ValueError(f"nonfinite JSON value {value}")


def parse_payload(record, mode, batch, context):
    record["parsed"] = None
    if record["status"] != "exited":
        return
    try:
        payload = json.loads(record["stdout"], parse_constant=reject_constant)
        # JSON exponents such as 1e400 can overflow even without NaN/Infinity literals.
        json.dumps(payload, allow_nan=False)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Expected benchmark JSON object with schema_version 1")
        if (payload.get("mode"), payload.get("batch_size"), payload.get("context_length")) != (mode, batch, context):
            raise ValueError("Benchmark output does not match requested workload")
        if payload.get("status") not in ("complete", "budget_exhausted") or not isinstance(payload.get("samples"), list):
            raise ValueError("Unknown benchmark status or missing samples")
        record["parsed"] = payload
        record["status"] = payload["status"]
    except (ValueError, TypeError) as error:
        record["status"] = "error"
        record["error"] = f"Invalid benchmark JSON: {error}"


def main(argv=None):
    arguments = parser().parse_args(argv)
    workspace = Path(__file__).resolve().parents[1]
    executable = arguments.executable.resolve(strict=True)
    if not executable.is_file():
        raise ValueError(f"Benchmark executable is not a file: {executable}")
    sources = source_fingerprints(workspace)
    executable_hash = sha256_file(executable)
    inherited = os.environ.copy()
    threads = arguments.threads or list(DEFAULT_THREADS)
    cases = arguments.case or list(DEFAULT_CASES)
    if arguments.profile and any(not case.startswith("train:") for case in cases):
        raise ValueError("--profile requires train-only --case selections")
    plan = [(pool, case) for pool in threads for case in cases]
    report = {
        "schema_version": 1, "started_at": utc_now(), "status": "running",
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                 "python": sys.version},
        "workspace": str(workspace),
        "executable": {"path": str(executable), "sha256": executable_hash, "bytes": executable.stat().st_size},
        "source_fingerprints": sources,
        "cargo_lock_sha256": sources["files"]["Code/Rust/Cargo.lock"],
        "build_identity_source": "each parsed Rust case includes its available build metadata; source hashes do not attest executable/source correspondence",
        "inherited_thread_environment": {name: inherited.get(name) for name in ENVIRONMENT_KEYS},
        "baseline_policy": "backend-default pool variables unset, including legacy Rayon alias; tokenizer parallelism disabled for every child",
        "plan": {"threads": threads, "cases": cases, "case_count": len(plan),
                 "samples": arguments.samples, "iterations": arguments.iterations,
                 "warmup": arguments.warmup, "new_tokens": arguments.new_tokens,
                 "profile": arguments.profile,
                 "child_max_seconds": arguments.max_seconds,
                 "timeout_seconds": arguments.timeout_seconds, "total_seconds": arguments.total_seconds},
        "results": [],
    }
    # Reserve exclusively before launching expensive work; never replace reports.
    with arguments.output.open("x", encoding="utf-8") as destination:
        started = time.monotonic()
        deadline = started + arguments.total_seconds
        memory = PeakMemory()
        try:
            for pool, case in plan:
                mode, batch, context = case.split(":")
                batch, context = int(batch), int(context)
                environment = child_environment(inherited, pool)
                command = [str(executable), "--mode", mode, "--batch-size", str(batch),
                           "--context-length", str(context), "--samples", str(arguments.samples),
                           "--iterations", str(arguments.iterations), "--warmup", str(arguments.warmup),
                           "--new-tokens", str(arguments.new_tokens), "--max-seconds", str(arguments.max_seconds)]
                if arguments.profile:
                    command.append("--profile")
                record = {"threads": pool, "case": case, "command": command,
                          "environment": {name: environment.get(name) for name in ENVIRONMENT_KEYS},
                          "status": "running"}
                report["results"].append(record)
                remaining = deadline - time.monotonic() - CLEANUP_RESERVE_SECONDS
                if remaining <= 0:
                    record["status"] = "skipped_total_budget"
                    continue
                try:
                    # Begin kill/reap before the requested hard cap, reserving a
                    # little time for cleanup even on very short focused tests.
                    child_reserve = min(CLEANUP_RESERVE_SECONDS, arguments.timeout_seconds / 10)
                    record.update(run_process(command, environment, workspace,
                                              min(arguments.timeout_seconds - child_reserve, remaining), memory))
                    parse_payload(record, mode, batch, context)
                except (OSError, ValueError) as error:
                    record.update(status="error", error=str(error))
                print(f"{pool} {case}: {record['status']}", file=sys.stderr, flush=True)
            statuses = {record["status"] for record in report["results"]}
            report["status"] = "error" if statuses & {"error", "timeout"} else (
                "partial" if statuses != {"complete"} else "complete")
        except KeyboardInterrupt:
            report["status"] = "interrupted"
            if report["results"] and report["results"][-1]["status"] == "running":
                report["results"][-1]["status"] = "interrupted"
        finally:
            report["process_sweep_wall_seconds"] = time.monotonic() - started
            report["finished_at"] = utc_now()
            try:
                after = source_fingerprints(workspace)
                report["sources_changed_during_run"] = after != sources
                report["executable_changed_during_run"] = sha256_file(executable) != executable_hash
                if report["sources_changed_during_run"] or report["executable_changed_during_run"]:
                    report["status"] = "error"
                    report["error"] = "Sources or executable changed during measurement; comparisons are not valid"
            except (OSError, ValueError) as error:
                report.update(status="error", error=f"Cannot verify final provenance: {error}")
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
            destination.flush()
    print(f"Saved {report['status']} benchmark report: {arguments.output}")
    return 0 if report["status"] == "complete" else (2 if report["status"] == "partial" else 1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
