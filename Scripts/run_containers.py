#!/usr/bin/env python3
"""Build Omega images and run dataset/tokenizer/training stages with Docker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


ROOT = Path(os.path.abspath(__file__)).parent.parent
STAGES = ("builder", "datasets", "training")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-run", action="store_true", help="print JSON argument arrays; perform no work")
    result.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    result.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    result.add_argument("--tag", default="local", help="image tag (default: local)")
    result.add_argument("--jobs", type=positive, default=2, help="Cargo jobs per package (default: 2)")
    result.add_argument("--skip-build", action="store_true", help="use previously built images")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="build all Rust binaries and all three images")
    for name in ("datasets", "training"):
        command = commands.add_parser(name, help=f"run omega-{name}; pass its arguments after --")
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    all_command = commands.add_parser("all", help="build images, create datasets, then run training steps")
    all_command.add_argument(
        "--plan", type=Path, default=ROOT / "Containers" / "pipeline.omega-alpha.json",
        help="JSON pipeline (default: Containers/pipeline.omega-alpha.json)",
    )
    all_command.add_argument(
        "--skip-dataset-build", action="store_true",
        help="reuse a prepared release; training still verifies its integrity",
    )
    return result


def argument_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item or "\0" in item for item in value
    ):
        raise ValueError(f"{label} must be a nonempty array of nonempty strings")
    return value


def load_plan(path: Path) -> list[tuple[str, list[str]]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "datasets", "training"}:
        raise ValueError("plan requires exactly schema_version, datasets and training")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("plan schema_version must be 1")
    dataset_args = argument_list(plan["datasets"], "datasets")
    if dataset_args[0] != "build":
        raise ValueError("all requires datasets arguments beginning with build")
    training = plan["training"]
    if not isinstance(training, list) or not training:
        raise ValueError("training must be a nonempty array of argument arrays")
    steps = [("datasets", dataset_args)]
    for index, arguments in enumerate(training):
        steps.append(("training", argument_list(arguments, f"training[{index}]")))
    if not any(args[0] in {"train", "train-stage", "resume"} for _, args in steps[1:]):
        raise ValueError("all requires a train, train-stage or resume step")
    return steps


def image_name(stage: str, tag: str) -> str:
    return f"omega-{stage}:{tag}"


def build_command(stage: str, args: argparse.Namespace) -> list[str]:
    return [
        "docker", "build", "--file", str(ROOT / "Containers" / "Dockerfile"),
        "--target", stage, "--tag", image_name(stage, args.tag),
        "--build-arg", f"BUILD_JOBS={args.jobs}", str(ROOT),
    ]


def mount(path: Path, destination: str) -> str:
    # --mount is comma-delimited; spaces and Windows drive colons are safe as
    # subprocess arguments, but embedded commas/quotes need an explicit error.
    if any(character in str(path) for character in ',"\r\n'):
        raise ValueError(f"Docker mount paths cannot contain commas, quotes or newlines: {path}")
    return f"type=bind,source={path},target={destination}"


def run_command(stage: str, arguments: list[str], args: argparse.Namespace) -> list[str]:
    command = [
        "docker", "run", "--rm", "--init", "--interactive", "--stop-timeout", "120",
        "--name", f"omega-{stage}-{uuid.uuid4().hex[:12]}",
        "--mount", mount(args.datasets_root, "/omega/datasets"),
    ]
    if stage == "training":
        command += ["--mount", mount(args.weights_root, "/omega/weights")]
    elif "HF_TOKEN" in os.environ:
        # Pass only the variable name; never print/store the token in a command.
        command += ["--env", "HF_TOKEN"]
    if hasattr(os, "getuid"):
        command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    if sys.stdin.isatty():
        command += ["--tty"]
    return command + [image_name(stage, args.tag), *arguments]


def execute(command: list[str], dry_run: bool) -> None:
    print(json.dumps(command, ensure_ascii=False), flush=True)
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        if command[:2] == ["docker", "run"]:
            name = command[command.index("--name") + 1]
            print(f"Stopping {name}; allowing up to 120 seconds for checkpoint saving.", file=sys.stderr)
            subprocess.run(["docker", "stop", "--time", "120", name], check=False, timeout=135)
        raise


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", args.tag):
            raise ValueError("invalid Docker image tag")
        if args.command == "build" and args.skip_build:
            raise ValueError("build cannot be combined with --skip-build")
        # Preserve Windows mapped-drive spellings instead of converting to UNC.
        args.datasets_root = Path(os.path.abspath(args.datasets_root.expanduser()))
        args.weights_root = Path(os.path.abspath(args.weights_root.expanduser()))
        if args.command == "all":
            steps = load_plan(args.plan)
            if args.skip_dataset_build:
                steps = steps[1:]
        elif args.command == "build":
            steps = []
        else:
            arguments = args.arguments
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            steps = [(args.command, arguments or ["--help"])]
        # Validate the entire plan and mount syntax before any build or mutation.
        run_commands = [run_command(stage, arguments, args) for stage, arguments in steps]
        images = STAGES if args.command in {"all", "build"} else (args.command,)
        commands = ([] if args.skip_build else [build_command(stage, args) for stage in images]) + run_commands
        if not args.dry_run:
            if shutil.which("docker") is None:
                raise ValueError("Docker CLI was not found; install Docker with Linux containers")
            try:
                check = subprocess.run(
                    ["docker", "info", "--format", "{{.OSType}}"],
                    capture_output=True, text=True, timeout=30,
                )
            except subprocess.TimeoutExpired as error:
                raise ValueError(
                    "Docker engine did not respond within 30 seconds; inspect Docker Desktop or the Docker service"
                ) from error
            if check.returncode != 0:
                raise ValueError("Docker engine is unavailable; start Docker Desktop or the Docker service")
            if check.stdout.strip() != "linux":
                raise ValueError("Omega images require Docker's Linux container engine")
            if steps:
                if not args.datasets_root.is_dir():
                    raise ValueError(f"datasets root must be an existing directory: {args.datasets_root}")
                if any(stage == "training" for stage, _ in steps):
                    args.weights_root.mkdir(parents=True, exist_ok=True)
        for command in commands:
            execute(command, args.dry_run)
        return 0
    except subprocess.CalledProcessError as error:
        print(f"Docker command failed (exit {error.returncode}); later stages were not run.", file=sys.stderr)
        return error.returncode if 1 <= error.returncode <= 255 else 1
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; later stages were not run.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
