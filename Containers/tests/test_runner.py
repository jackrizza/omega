"""Offline regression tests; no Docker engine, training or downloads required."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("omega_containers", ROOT / "Scripts/run_containers.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="omega-container-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data with spaces"
        self.data.mkdir()
        self.weights = self.root / "new weights"
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps({
            "schema_version": 1,
            "datasets": ["build", "corpus"],
            "training": [
                ["train-tokenizer", "--dataset", "corpus/release/base/train"],
                ["train", "--name", "tiny", "--dataset", "corpus/release/base/train"],
            ],
        }), encoding="utf-8")
        self.flags = ["--datasets-root", str(self.data), "--weights-root", str(self.weights)]

    def invoke(self, arguments, **kwargs):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = runner.main(self.flags + arguments, **kwargs)
        return code, output.getvalue(), errors.getvalue()

    def test_dry_run_orders_all_stages_without_side_effects(self):
        with patch.object(runner.subprocess, "run") as run:
            code, output, _ = self.invoke(["--dry-run", "all", "--plan", str(self.plan)])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertFalse(self.weights.exists())
        commands = [json.loads(line) for line in output.splitlines()]
        self.assertEqual([cmd[cmd.index("--target") + 1] for cmd in commands[:3]], list(runner.STAGES))
        self.assertEqual(commands[3][-2:], ["build", "corpus"])
        self.assertIn("train-tokenizer", commands[4])
        self.assertIn("train", commands[5])
        self.assertIn(f"type=bind,source={self.data},target=/omega/datasets", commands[3])
        self.assertNotIn("target=/omega/weights", " ".join(commands[3]))
        self.assertIn(f"type=bind,source={self.weights},target=/omega/weights", commands[5])

    def test_arguments_are_not_shell_interpreted(self):
        prompt = 'hello "world"; $(do-not-run) & goodbye'
        code, output, _ = self.invoke(["--dry-run", "--skip-build", "training", "--", "generate", "--prompt", prompt])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)[-1], prompt)

    def test_existing_release_skips_only_dataset_execution(self):
        code, output, _ = self.invoke([
            "--dry-run", "all", "--plan", str(self.plan), "--skip-dataset-build",
        ])
        self.assertEqual(code, 0)
        commands = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(len(commands), 5)
        self.assertEqual([cmd[cmd.index("--target") + 1] for cmd in commands[:3]], list(runner.STAGES))
        self.assertTrue(all("omega-training:local" in cmd for cmd in commands[3:]))
        self.assertIn("train-tokenizer", commands[3])
        self.assertIn("train", commands[4])

    def test_default_plan_uses_alpha_train_partition_and_new_tokenizer(self):
        code, output, _ = self.invoke(["--dry-run", "--skip-build", "all"])
        self.assertEqual(code, 0)
        commands = [json.loads(line) for line in output.splitlines()]
        self.assertIn("omega-alpha", commands[0])
        tokenizer, training = commands[1:]
        self.assertIn("--chat-protocol", tokenizer)
        for command in (tokenizer, training):
            self.assertEqual(command[command.index("--dataset") + 1], "omega-alpha/release-v1/base/train")
            self.assertEqual(command[command.index("--dataset-format") + 1], "auto")
            self.assertNotIn("--validation-ratio", command)
            self.assertNotIn("--validation-count", command)
        output_path = tokenizer[tokenizer.index("--output") + 1]
        input_path = training[training.index("--tokenizer") + 1]
        self.assertEqual(output_path, "/omega/datasets/" + input_path)
        self.assertNotEqual(input_path, "omega-alpha/tokenizer.json")
        self.assertEqual(training[training.index("--max-updates") + 1], "2")

    def test_failure_stops_pipeline_and_preserves_exit_code(self):
        # Each image build and each runtime stage can fail independently.
        for failure in range(6):
            with self.subTest(failure=failure):
                calls = []
                def run(command, **kwargs):
                    if command[1] == "info":
                        return subprocess.CompletedProcess(command, 0, "linux\n", "")
                    calls.append(command)
                    if len(calls) == failure + 1:
                        raise subprocess.CalledProcessError(17, command)
                    return subprocess.CompletedProcess(command, 0)
                with patch.object(runner.shutil, "which", return_value="docker"), patch.object(runner.subprocess, "run", side_effect=run):
                    code, _, _ = self.invoke(["all", "--plan", str(self.plan)])
                self.assertEqual(code, 17)
                self.assertEqual(len(calls), failure + 1)

    def test_plan_errors_precede_docker_or_directory_creation(self):
        invalid = [
            {},
            {"schema_version": True, "datasets": ["build", "x"], "training": [["train"]]},
            {"schema_version": 1, "datasets": ["plan", "x"], "training": [["train"]]},
            {"schema_version": 1, "datasets": ["build", "x"], "training": [["train"], [42]]},
            {"schema_version": 1, "datasets": ["build", "x"], "training": [["coverage"]]},
        ]
        for value in invalid:
            self.plan.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(runner.subprocess, "run") as run:
                code, _, _ = self.invoke(["all", "--plan", str(self.plan)])
            self.assertEqual(code, 1)
            run.assert_not_called()
            self.assertFalse(self.weights.exists())

    def test_daemon_error_is_actionable(self):
        with patch.object(runner.shutil, "which", return_value="docker"), patch.object(
            runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "unavailable")
        ) as run:
            code, _, errors = self.invoke(["build"])
        self.assertEqual(code, 1)
        self.assertIn("start Docker", errors)
        self.assertEqual(run.call_count, 1)

    def test_token_is_forwarded_by_name_only_and_only_to_datasets(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret-value-never-print"}):
            code, output, _ = self.invoke(["--dry-run", "all", "--plan", str(self.plan)])
        self.assertEqual(code, 0)
        self.assertNotIn("secret-value-never-print", output)
        commands = [json.loads(line) for line in output.splitlines()]
        self.assertIn("HF_TOKEN", commands[3])
        self.assertNotIn("HF_TOKEN", commands[4])

    def test_unresponsive_engine_times_out_before_any_build(self):
        with patch.object(runner.shutil, "which", return_value="docker"), patch.object(
            runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("docker", 30)
        ) as run:
            code, _, errors = self.invoke(["build"])
        self.assertEqual(code, 1)
        self.assertIn("did not respond within 30 seconds", errors)
        self.assertEqual(run.call_count, 1)

    def test_success_creates_weights_and_runs_all_steps(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "linux\n", "")
        with patch.object(runner.shutil, "which", return_value="docker"), patch.object(runner.subprocess, "run", side_effect=run):
            code, _, _ = self.invoke(["all", "--plan", str(self.plan)])
        self.assertEqual(code, 0)
        self.assertTrue(self.weights.is_dir())
        self.assertEqual(len(calls), 7)  # engine probe + three images + three steps

    def test_interruption_stops_named_container(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[1] == "info":
                return subprocess.CompletedProcess(command, 0, "linux\n", "")
            if command[1] == "run":
                raise KeyboardInterrupt
            return subprocess.CompletedProcess(command, 0)
        with patch.object(runner.shutil, "which", return_value="docker"), patch.object(runner.subprocess, "run", side_effect=run):
            code, _, _ = self.invoke(["--skip-build", "training", "--", "train", "--name", "tiny"])
        self.assertEqual(code, 130)
        self.assertEqual(calls[-1][:4], ["docker", "stop", "--time", "120"])
        self.assertEqual(calls[-1][-1], calls[-2][calls[-2].index("--name") + 1])

    def test_mount_delimiter_rejected(self):
        with self.assertRaisesRegex(ValueError, "commas"):
            runner.mount(Path("bad,path"), "/omega/datasets")

    def test_running_from_another_directory(self):
        result = subprocess.run([
            os.sys.executable, "-B", str(ROOT / "Scripts/run_containers.py"),
            "--dry-run", "build",
        ], cwd=self.root, capture_output=True, text=True, check=True)
        command = json.loads(result.stdout.splitlines()[0])
        self.assertTrue(Path(command[-1]).samefile(ROOT))


class BuilderTests(unittest.TestCase):
    def test_discovers_all_bins_and_preserves_colliding_names(self):
        script = (ROOT / "Containers/build-rust.sh").read_text()
        body = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        metadata = {
            "workspace_members": ["one", "two"],
            "packages": [
                {"id": "one", "name": "one", "targets": [{"name": "main", "kind": ["bin"]}]},
                {"id": "two", "name": "two", "targets": [
                    {"name": "main", "kind": ["bin"]},
                    {"name": "unique", "kind": ["bin"]},
                    {"name": "example", "kind": ["example"]},
                ]},
                {"id": "dependency", "name": "external", "targets": [{"name": "main", "kind": ["bin"]}]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_path(value):
                return root / value.lstrip("/")
            def build(command, **kwargs):
                name = command[command.index("--package") + 1]
                target = Path(command[command.index("--target-dir") + 1]) / "release"
                target.mkdir(parents=True)
                for binary in ("main", "unique"):
                    (target / binary).write_text(name + "/" + binary)
                self.assertIn("--locked", command)
                self.assertIn("--bins", command)
            with patch("pathlib.Path", side_effect=fake_path), patch("subprocess.check_output", return_value=json.dumps(metadata)), patch("subprocess.run", side_effect=build) as run:
                exec(compile(body, "build-rust.sh", "exec"), {})
            self.assertEqual(run.call_count, 2)
            self.assertEqual((root / "opt/omega/bin/one/main").read_text(), "one/main")
            self.assertEqual((root / "opt/omega/bin/two/main").read_text(), "two/main")
            inventory = json.loads((root / "opt/omega/binaries.json").read_text())
            self.assertEqual(len(inventory), 3)


if __name__ == "__main__":
    unittest.main()
