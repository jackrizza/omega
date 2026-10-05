"""Record bounded local Omega experiments and explicit human validation choices.

Commands: hash PATH; run --recipe FILE --output NEW_DIR; report --scores FILE
--checkpoint DIRECTORY --output NEW_DIR. No shell or external judge is used.
Recipes use schema_version 1, executable (path), optional launcher_files (for a
script interpreter), runtime (nonempty supplied metadata), inputs (name/path/
sha256), and commands (id/stage/args/timeout_seconds). Paths resolve relative to
the recipe. Omega stage names are train/resume/train-stage/evaluate/generate/chat.
Training requires --max-updates; loading requires one exact --checkpoint.
Loading also requires an explicit --weights-root; the selected completed artifact
is hashed immediately before its command, including artifacts from earlier steps.
Use {run_dir} inside an argv value for logs or new checkpoint paths in the new
run directory. Inputs are hash-checked before launch. No commands are inferred.

Score files require schema_version 1, partition 'validation', selected_checkpoint,
selection_reason, rubric, and records (prompt_id/checkpoint/criterion/score/max_score/
notes). Reports preserve supplied scores without inventing an overall quality
judgment. They hash the explicitly selected completed checkpoint. A report does
not verify model payload compatibility or prove that held-out data was protected.
Per-command stdout+stderr limits default to 16 MiB and are sampled every 0.1 s;
logs can overshoot during detection and the two-second cooperative-stop grace.
Checkpoint/model outputs are not disk-quota limited. Each artifact hash is bounded
to 1 GiB and 100,000 directory entries. Subprocess failure retains incomplete logs.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from prepare_assistant_data import encoded, parse_json, reject_links


STAGES = {"train", "resume", "train-stage", "evaluate", "generate", "chat"}
TRAINING = {"train", "resume", "train-stage"}
LOADING = {"resume", "train-stage", "evaluate", "generate", "chat"}
MAX_HASH_BYTES = 1024 * 1024 * 1024
MAX_HASH_FILES = 100_000


def path_from(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Paths must be nonempty strings")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    reject_links(path)
    return path.resolve(strict=True)


def artifact_hash(path, max_bytes=MAX_HASH_BYTES, max_files=MAX_HASH_FILES):
    """Hash bytes or a deterministic directory listing including every regular file."""
    path = Path(path).absolute()
    reject_links(path)
    budget = 0
    entries = []

    def file_hash(file):
        nonlocal budget
        result = hashlib.sha256()
        count = 0
        with file.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                budget += len(block)
                count += len(block)
                if budget > max_bytes:
                    raise ValueError(f"Hash byte limit {max_bytes} exceeded: {path}")
                result.update(block)
        return result.hexdigest(), count

    if path.is_file():
        checksum, size = file_hash(path)
        return {"sha256": checksum, "bytes": size, "files": 1, "kind": "file"}
    if not path.is_dir():
        raise ValueError(f"Artifact is not a file or directory: {path}")
    visited = 0
    for directory, folders, files in os.walk(path, followlinks=False):
        for name in sorted(folders + files):
            visited += 1
            if visited > max_files:
                raise ValueError(f"Hash entry limit {max_files} exceeded: {path}")
            item = Path(directory) / name
            reject_links(item)
            if item.is_file():
                checksum, size = file_hash(item)
                entries.append({"path": item.relative_to(path).as_posix(),
                                "sha256": checksum, "bytes": size})
            elif not item.is_dir():
                raise ValueError(f"Nonregular artifact entry: {item}")
    entries.sort(key=lambda item: item["path"])
    return {"sha256": hashlib.sha256(encoded(entries)).hexdigest(), "bytes": budget,
            "files": len(entries), "kind": "directory", "entries": entries}


def load_json(path):
    reject_links(path)
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError(f"JSON input exceeds 4 MiB: {path}")
    return parse_json(raw), raw


def reserve_output(output):
    output = Path(output).absolute()
    reject_links(output)
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent does not exist: {output.parent}")
    return output


def option(args, flag):
    values = []
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 == len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {flag}")
            values.append(args[index + 1])
        elif arg.startswith(flag + "="):
            values.append(arg[len(flag) + 1:])
    return values


def validate_recipe(recipe_path, output):
    recipe, raw = load_json(recipe_path)
    required = {"schema_version", "executable", "runtime", "inputs", "commands"}
    if (not isinstance(recipe, dict) or not required <= set(recipe)
            or set(recipe) - required - {"launcher_files"}
            or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1):
        raise ValueError("Recipe requires schema_version 1, executable, runtime, inputs, commands")
    if not isinstance(recipe["runtime"], dict) or not recipe["runtime"]:
        raise ValueError("Runtime must contain supplied host/build/backend metadata")
    root = recipe_path.parent
    executable = path_from(root, recipe["executable"])
    if not executable.is_file():
        raise ValueError("Executable must be a regular file")
    launchers = recipe.get("launcher_files", [])
    if not isinstance(launchers, list):
        raise ValueError("launcher_files must be a list of explicit script paths")
    prefix = [executable] + [path_from(root, item) for item in launchers]
    if any(not path.is_file() for path in prefix):
        raise ValueError("Executable and launcher_files must be regular files")
    binaries = [{"path": str(path), **artifact_hash(path)} for path in prefix]
    inputs = recipe["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("At least one hashed corpus/tokenizer or checkpoint input is required")
    verified, input_names = [], set()
    for entry in inputs:
        if not isinstance(entry, dict) or set(entry) != {"name", "path", "sha256"}:
            raise ValueError("Each input requires name, path, sha256")
        if not isinstance(entry["name"], str) or not entry["name"] or entry["name"] in input_names:
            raise ValueError("Input names must be unique nonempty strings")
        input_names.add(entry["name"])
        path = path_from(root, entry["path"])
        if output == path or output.is_relative_to(path):
            raise ValueError("Run output must not be inside a hashed input")
        actual = artifact_hash(path)
        if entry["sha256"] != actual["sha256"]:
            raise ValueError(f"Input hash mismatch: {entry['name']}")
        verified.append({"name": entry["name"], "path": str(path), **actual})
    commands = recipe["commands"]
    if not isinstance(commands, list) or not commands or len(commands) > 100:
        raise ValueError("Recipe requires 1..100 explicit commands")
    resolved, ids = [], set()
    for command in commands:
        required_command = {"id", "stage", "args", "timeout_seconds"}
        if (not isinstance(command, dict) or not required_command <= set(command)
                or set(command) - required_command - {"max_output_bytes"}):
            raise ValueError("Commands require id, stage, args, timeout_seconds")
        identifier, stage = command["id"], command["stage"]
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
                or identifier.casefold() in ids
                or re.fullmatch(r"(?i:complete|con|prn|aux|nul|com[1-9]|lpt[1-9])", identifier)):
            raise ValueError("Command IDs must be unique ASCII identifiers")
        ids.add(identifier.casefold())
        if not isinstance(stage, str) or stage not in STAGES:
            raise ValueError(f"Unknown Omega stage: {stage}")
        args = command["args"]
        if not isinstance(args, list) or any(not isinstance(arg, str) or "\0" in arg for arg in args):
            raise ValueError("Arguments must be explicit strings without NUL")
        if any(arg == "--latest-run" or arg.startswith("--latest-run=") for arg in args):
            raise ValueError("Experiments require exact checkpoint names; --latest-run is forbidden")
        if stage in LOADING:
            checkpoints = option(args, "--checkpoint")
            if len(checkpoints) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+-[0-9]+", checkpoints[0]):
                raise ValueError(f"{stage} requires one exact numbered --checkpoint name")
            weights = option(args, "--weights-root")
            if len(weights) != 1 or not weights[0].strip():
                raise ValueError(f"{stage} requires one explicit --weights-root for checkpoint provenance")
        if stage in TRAINING:
            updates = option(args, "--max-updates")
            if len(updates) != 1 or not updates[0].isascii() or not updates[0].isdigit() or int(updates[0]) <= 0:
                raise ValueError(f"{stage} requires one positive --max-updates bound")
        if stage == "chat" and len(option(args, "--prompt")) != 1:
            raise ValueError("Recorded chat requires one --prompt; interactive stdin is disabled")
        for dataset in option(args, "--dataset") + option(args, "-d"):
            components = dataset.replace("\\", "/").lower().split("/")
            if any(part in ("test", "sealed", "sealed-test") for part in components):
                raise ValueError("Sealed-test execution is excluded from the tuning recorder")
            if stage in TRAINING and "validation" in components:
                raise ValueError("Training cannot select a validation partition")
        timeout = command["timeout_seconds"]
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError("Each timeout_seconds must be finite and within (0, 86400]")
        output_limit = command.get("max_output_bytes", 16 * 1024 * 1024)
        if type(output_limit) is not int or not 0 < output_limit <= MAX_HASH_BYTES:
            raise ValueError("max_output_bytes must be a positive integer at most 1 GiB")
        resolved.append({**command, "max_output_bytes": output_limit,
                         "argv": [str(path) for path in prefix] + [stage]
                         + [arg.replace("{run_dir}", str(output)) for arg in args]})
    return raw, {"schema_version": 1, "runtime_supplied": recipe["runtime"],
                 "recorder_runtime": {"platform": sys.platform, "python": sys.version},
                 "executables": binaries, "inputs": verified, "commands": resolved,
                 "recipe_sha256": hashlib.sha256(raw).hexdigest()}


class WindowsJob:
    """Kill-on-close job keeps descendants from outliving an experiment command."""

    def __init__(self, process):
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                        ("max_working", ctypes.c_size_t), ("processes", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def stop_process(process, job):
    details = {"cooperative_requested": False, "hard_stop": False}
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        details["cooperative_requested"] = True
    except (OSError, ProcessLookupError) as error:
        details["cooperative_error"] = str(error)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        details["hard_stop"] = True
    if job is not None:
        job.close()
    elif os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    return details


def execute(command, directory, cwd):
    directory.mkdir()
    started = time.monotonic()
    result = {"id": command["id"], "stage": command["stage"], "argv": command["argv"],
              "status": "failed", "timed_out": False}
    process, job = None, None
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        try:
            if command["stage"] in LOADING:
                name = option(command["argv"], "--checkpoint")[0]
                root = Path(option(command["argv"], "--weights-root")[0])
                root = root if root.is_absolute() else cwd / root
                checkpoint = root / name
                reject_links(checkpoint)
                if not checkpoint.is_dir() or not (checkpoint / "COMPLETE").is_file():
                    raise ValueError(f"Selected checkpoint is missing or incomplete: {checkpoint}")
                result["checkpoint"] = {"name": name, "path": str(checkpoint.resolve()),
                                        **artifact_hash(checkpoint)}
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            process = subprocess.Popen(command["argv"], cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, shell=False, **kwargs)
            if os.name == "nt":
                job = WindowsJob(process)
            while True:
                remaining = command["timeout_seconds"] - (time.monotonic() - started)
                output_bytes = os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size
                if remaining <= 0 or output_bytes > command["max_output_bytes"]:
                    result.update({"status": "timed_out" if remaining <= 0 else "output_limit",
                                   "timed_out": remaining <= 0})
                    result["termination"] = stop_process(process, job)
                    result["returncode"] = process.returncode
                    break
                try:
                    result["returncode"] = process.wait(timeout=min(0.1, remaining))
                    output_bytes = os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size
                    result["status"] = ("output_limit" if output_bytes > command["max_output_bytes"] else
                                        "succeeded" if result["returncode"] == 0 else "failed")
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result["error"] = str(error)
            if process is not None:
                result["termination"] = stop_process(process, job)
                result["returncode"] = process.returncode
        finally:
            if job is not None:
                job.close()
            elif process is not None and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    result["elapsed_seconds"] = time.monotonic() - started
    for name in ("stdout", "stderr"):
        try:
            result[name] = artifact_hash(directory / (name + ".log"))
        except ValueError as error:
            result[name] = {"hash_error": str(error)}
            result["status"] = "failed"
    with (directory / "result.json").open("xb") as stream:
        stream.write(encoded(result))
    return result


def run(recipe_path, output):
    recipe_path = Path(recipe_path).absolute()
    output = reserve_output(output)
    raw, resolved = validate_recipe(recipe_path, output)
    output.mkdir()
    for name, value in (("recipe.json", raw), ("resolved.json", encoded(resolved))):
        with (output / name).open("xb") as stream:
            stream.write(value)
    results = []
    for command in resolved["commands"]:
        result = execute(command, output / command["id"], recipe_path.parent)
        results.append(result)
        if result["status"] != "succeeded":
            break
    summary = {"schema_version": 1, "status": "succeeded" if len(results) == len(resolved["commands"])
               and all(result["status"] == "succeeded" for result in results) else "failed",
               "results": results, "unexecuted_commands": [command["id"] for command in resolved["commands"][len(results):]],
               "quality_acceptance": "not_evaluated"}
    with (output / "results.json").open("xb") as stream:
        stream.write(encoded(summary))
    if summary["status"] == "succeeded":
        with (output / "COMPLETE").open("xb") as stream:
            stream.write(encoded({"schema_version": 1, "results_sha256": artifact_hash(output / "results.json")["sha256"]}))
    return summary


def report(scores_path, checkpoint, output):
    scores_path, checkpoint = Path(scores_path).absolute(), Path(checkpoint).absolute()
    output = reserve_output(output)
    scores, raw = load_json(scores_path)
    required = {"schema_version", "partition", "selected_checkpoint", "selection_reason", "rubric", "records"}
    if (not isinstance(scores, dict) or set(scores) != required
            or type(scores["schema_version"]) is not int or scores["schema_version"] != 1
            or scores["partition"] != "validation"):
        raise ValueError("Scores require schema_version 1 and explicit validation selection fields")
    if scores["selected_checkpoint"] != checkpoint.name or not re.fullmatch(r"[A-Za-z0-9_-]+-[0-9]+", checkpoint.name):
        raise ValueError("Selected checkpoint name must match the exact supplied numbered directory")
    if not isinstance(scores["selection_reason"], str) or not scores["selection_reason"].strip():
        raise ValueError("Provide the human validation-based selection reason")
    if not isinstance(scores["rubric"], dict) or not scores["rubric"]:
        raise ValueError("Provide the predeclared rubric")
    if not isinstance(scores["records"], list) or not scores["records"]:
        raise ValueError("Provide actual human score records")
    selected_seen, record_ids = False, set()
    for row in scores["records"]:
        if not isinstance(row, dict) or set(row) != {"prompt_id", "checkpoint", "criterion", "score", "max_score", "notes"}:
            raise ValueError("Scores require prompt_id/checkpoint/criterion/score/max_score/notes")
        for field in ("prompt_id", "checkpoint", "criterion", "notes"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Score field {field} must be nonempty text")
        key = (row["prompt_id"], row["checkpoint"], row["criterion"])
        if key in record_ids:
            raise ValueError("Duplicate prompt/checkpoint/criterion score")
        record_ids.add(key)
        if row["criterion"] not in scores["rubric"]:
            raise ValueError("Score criterion is absent from rubric")
        if any(type(row[field]) not in (int, float) or not math.isfinite(row[field]) for field in ("score", "max_score")):
            raise ValueError("Scores must be finite numbers")
        if not 0 <= row["score"] <= row["max_score"] or row["max_score"] <= 0:
            raise ValueError("Score must lie between zero and positive max_score")
        selected_seen |= row["checkpoint"] == checkpoint.name
    if not selected_seen:
        raise ValueError("Selected checkpoint has no validation scores")
    if not checkpoint.is_dir() or not (checkpoint / "COMPLETE").is_file():
        raise ValueError("Selected checkpoint must be a completed directory")
    if output == checkpoint or output.is_relative_to(checkpoint):
        raise ValueError("Report output must not be inside checkpoint")
    selection = {"schema_version": 1, "selected_checkpoint": checkpoint.name,
                 "checkpoint_path": str(checkpoint), "checkpoint": artifact_hash(checkpoint),
                 "scores_sha256": hashlib.sha256(raw).hexdigest(), "scores": scores,
                 "inspection": "artifact_hashes_only; model load and quality acceptance unverified"}
    output.mkdir()
    with (output / "selection.json").open("xb") as stream:
        stream.write(encoded(selection))
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="operation", required=True)
    hash_parser = commands.add_parser("hash")
    hash_parser.add_argument("path", type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--recipe", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--scores", required=True, type=Path)
    report_parser.add_argument("--checkpoint", required=True, type=Path)
    report_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "hash":
            print(json.dumps(artifact_hash(args.path)))
        elif args.operation == "run":
            result = run(args.recipe, args.output)
            print(json.dumps({"status": result["status"], "output": str(args.output)}))
            return 0 if result["status"] == "succeeded" else 1
        else:
            result = report(args.scores, args.checkpoint, args.output)
            print(json.dumps({"selected_checkpoint": result["selected_checkpoint"], "output": str(args.output)}))
    except (OSError, ValueError, RecursionError, subprocess.SubprocessError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
