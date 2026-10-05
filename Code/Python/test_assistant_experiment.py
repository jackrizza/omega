"""Mock-only local experiment tests; no Omega training is launched."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import assistant_experiment as experiment


MOCK = """
import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
print(json.dumps(args), flush=True)
print('mock stderr', file=sys.stderr, flush=True)
if '--fail' in args or '--unknown-omega-option' in args:
    raise SystemExit(7)
if '--flood' in args:
    print('X' * 10000, flush=True)
if '--sleep' in args:
    time.sleep(30)
if '--descendant' in args:
    destination = args[args.index('--descendant') + 1]
    subprocess.Popen([sys.executable, '-c',
        'import pathlib,time; time.sleep(5); pathlib.Path(' + repr(destination) + ').write_text("leaked")'])
    time.sleep(30)
"""


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.mock = self.root / "mock.py"
        self.mock.write_text(MOCK, encoding="utf-8")
        self.input = self.root / "input.txt"
        self.input.write_text("immutable corpus fixture", encoding="utf-8")
        self.recipe_path = self.root / "recipe.json"
        self.output = self.root / "run"
        self.weights = self.root / "weights"
        self.weights.mkdir()
        for name in ("base-1", "sft-1"):
            checkpoint = self.weights / name
            checkpoint.mkdir()
            (checkpoint / "COMPLETE").write_text("test marker")
            (checkpoint / "model.bin").write_bytes(b"mock weights")
        # System Python launchers (for example /usr/bin/python3) may be symlinks.
        # Fixtures use the real executable; production still rejects aliases.
        self.recipe = {"schema_version": 1, "executable": str(Path(sys.executable).resolve()),
                       "launcher_files": [str(self.mock)], "runtime": {"backend": "mock-only", "host": "unit-test"},
                       "inputs": [{"name": "corpus", "path": str(self.input),
                                   "sha256": experiment.artifact_hash(self.input)["sha256"]}],
                       "commands": [{"id": "pilot", "stage": "train",
                                     "args": ["--max-updates", "2", "--dataset", "base/train"],
                                     "timeout_seconds": 5}]}

    def write_recipe(self):
        self.recipe_path.write_text(json.dumps(self.recipe), encoding="utf-8")

    def run_recipe(self):
        self.write_recipe()
        return experiment.run(self.recipe_path, self.output)

    def test_success_records_exact_argv_hashes_logs_and_unmodified_recipe(self):
        self.recipe["commands"][0]["args"].extend(["--name", "name with spaces", "--metrics-jsonl", "{run_dir}/metrics.jsonl"])
        result = self.run_recipe()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual((self.output / "recipe.json").read_bytes(), self.recipe_path.read_bytes())
        resolved = json.loads((self.output / "resolved.json").read_bytes())
        argv = resolved["commands"][0]["argv"]
        self.assertEqual(argv[-3], "name with spaces")
        self.assertEqual(argv[-1], str(self.output) + "/metrics.jsonl")
        observed = json.loads((self.output / "pilot/stdout.log").read_text().splitlines()[0])
        self.assertEqual(observed, argv[2:])
        self.assertEqual(resolved["inputs"][0]["sha256"], experiment.artifact_hash(self.input)["sha256"])
        self.assertEqual(len(resolved["executables"]), 2)
        self.assertEqual(result["quality_acceptance"], "not_evaluated")
        self.assertTrue((self.output / "COMPLETE").is_file())

    def test_failure_keeps_logs_and_prevents_following_commands(self):
        self.recipe["commands"][0]["args"].append("--fail")
        self.recipe["commands"].append({"id": "evaluation", "stage": "evaluate",
                                       "args": ["--checkpoint", "base-1", "--weights-root", str(self.weights), "--dataset", "base/validation"], "timeout_seconds": 5})
        result = self.run_recipe()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["results"][0]["returncode"], 7)
        self.assertEqual(result["unexecuted_commands"], ["evaluation"])
        self.assertFalse((self.output / "evaluation").exists())
        self.assertFalse((self.output / "COMPLETE").exists())
        self.assertIn("mock stderr", (self.output / "pilot/stderr.log").read_text())

    def test_timeout_returns_failure_and_preserves_partial_logs(self):
        self.recipe["commands"][0]["args"].append("--sleep")
        self.recipe["commands"][0]["timeout_seconds"] = 0.3
        before = time.monotonic()
        result = self.run_recipe()
        self.assertLess(time.monotonic() - before, 15)
        self.assertTrue(result["results"][0]["timed_out"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.output / "COMPLETE").exists())
        self.assertTrue((self.output / "pilot/stdout.log").is_file())

    def test_timeout_cleans_descendant_process(self):
        leaked = self.root / "leaked.txt"
        self.recipe["commands"][0]["args"].extend(["--descendant", str(leaked)])
        self.recipe["commands"][0]["timeout_seconds"] = 0.4
        result = self.run_recipe()
        self.assertTrue(result["results"][0]["timed_out"])
        time.sleep(5.2)
        self.assertFalse(leaked.exists(), "descendant survived timeout cleanup")

    def test_output_limit_fails_without_claiming_complete(self):
        self.recipe["commands"][0]["args"].append("--flood")
        self.recipe["commands"][0]["max_output_bytes"] = 512
        result = self.run_recipe()
        self.assertEqual(result["results"][0]["status"], "output_limit")
        self.assertFalse((self.output / "COMPLETE").exists())

    def test_wrong_input_hash_and_existing_output_prevent_execution(self):
        self.input.write_text("changed corpus", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_recipe()
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            experiment.run(self.recipe_path, self.output)
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")

    def test_invalid_recipe_arguments_fail_before_launch(self):
        cases = [
            ("evaluate", ["--latest-run", "base"]),
            ("evaluate", ["--checkpoint", "../base-1"]),
            ("generate", ["--checkpoint", "base-1", "--checkpoint=base-2"]),
            ("train", ["--max-updates", "0"]),
            ("train", ["--max-updates", "1", "--dataset", "base/validation"]),
            ("evaluate", ["--checkpoint", "base-1", "--weights-root", str(self.weights), "--dataset=base/test"]),
            ("chat", ["--checkpoint", "assistant-1"]),
            ("not-a-stage", []),
        ]
        for stage, args in cases:
            with self.subTest(stage=stage, args=args):
                self.recipe["commands"][0].update(stage=stage, args=args)
                with self.assertRaises(ValueError):
                    self.run_recipe()
                self.assertFalse(self.output.exists())

    def test_unknown_omega_flag_is_saved_and_propagates_cli_failure(self):
        self.recipe["commands"][0]["args"].append("--unknown-omega-option")
        self.write_recipe()
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            status = experiment.main(["run", "--recipe", str(self.recipe_path), "--output", str(self.output)])
        self.assertEqual(status, 1)
        result = json.loads((self.output / "pilot/result.json").read_bytes())
        self.assertIn("--unknown-omega-option", result["argv"])
        self.assertEqual(result["returncode"], 7)

    def test_directory_hash_is_stable_and_detects_content_and_membership_changes(self):
        directory = self.root / "artifact"
        directory.mkdir()
        (directory / "one").write_text("one")
        first = experiment.artifact_hash(directory)
        self.assertEqual(first, experiment.artifact_hash(directory))
        (directory / "one").write_text("changed")
        second = experiment.artifact_hash(directory)
        self.assertNotEqual(first["sha256"], second["sha256"])
        (directory / "two").write_text("two")
        self.assertNotEqual(second["sha256"], experiment.artifact_hash(directory)["sha256"])
        with self.assertRaisesRegex(ValueError, "byte limit"):
            experiment.artifact_hash(directory, max_bytes=1)
        with self.assertRaisesRegex(ValueError, "entry limit"):
            experiment.artifact_hash(directory, max_files=1)

    def test_reserved_or_case_alias_command_ids_and_invalid_timeout_rejected(self):
        for identifier in ("COMPLETE", "con", "../escape"):
            with self.subTest(identifier=identifier):
                self.recipe["commands"][0]["id"] = identifier
                with self.assertRaises(ValueError):
                    self.run_recipe()
        self.recipe["commands"][0]["id"] = "pilot"
        self.recipe["commands"].append({**self.recipe["commands"][0], "id": "PILOT"})
        with self.assertRaises(ValueError):
            self.run_recipe()
        self.recipe["commands"].pop()
        for timeout in (0, -1, True, 86401):
            self.recipe["commands"][0]["timeout_seconds"] = timeout
            with self.assertRaises(ValueError):
                self.run_recipe()
        self.assertFalse(self.output.exists())

    def score_fixture(self):
        checkpoint = self.root / "assistant-3"
        checkpoint.mkdir()
        (checkpoint / "COMPLETE").write_text("marker")
        (checkpoint / "model.bin").write_bytes(b"mock weights")
        scores = {"schema_version": 1, "partition": "validation", "selected_checkpoint": "assistant-3",
                  "selection_reason": "Human-selected using the retained development replies",
                  "rubric": {"relevance": "0 incorrect; 1 partially relevant; 2 relevant"},
                  "records": [{"prompt_id": "dev-01", "checkpoint": "assistant-3", "criterion": "relevance",
                               "score": 1, "max_score": 2, "notes": "Partial answer; no quality claim"}]}
        path = self.root / "scores.json"
        path.write_text(json.dumps(scores), encoding="utf-8")
        return path, checkpoint, scores

    def test_selection_report_preserves_actual_human_scores_and_exact_hash(self):
        path, checkpoint, scores = self.score_fixture()
        selection = experiment.report(path, checkpoint, self.output)
        self.assertEqual(selection["scores"], scores)
        self.assertEqual(selection["checkpoint"]["sha256"], experiment.artifact_hash(checkpoint)["sha256"])
        self.assertIn("unverified", selection["inspection"])
        self.assertTrue((self.output / "selection.json").is_file())
        with self.assertRaisesRegex(ValueError, "already exists"):
            experiment.report(path, checkpoint, self.output)

    def test_sealed_selection_and_fabricated_missing_scores_rejected(self):
        path, checkpoint, scores = self.score_fixture()
        for field, value in (("partition", "test"), ("records", []), ("selected_checkpoint", "assistant-2")):
            with self.subTest(field=field):
                changed = {**scores, field: value}
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    experiment.report(path, checkpoint, self.output)
                self.assertFalse(self.output.exists())

    def test_stage_and_prompt_chat_are_supported(self):
        self.recipe["commands"] = [
            {"id": "sft", "stage": "train-stage", "args": ["--checkpoint", "base-1", "--weights-root", str(self.weights), "--max-updates", "1"], "timeout_seconds": 5},
            {"id": "reply", "stage": "chat", "args": ["--checkpoint", "sft-1", "--weights-root", str(self.weights), "--prompt", "Hello"], "timeout_seconds": 5},
        ]
        result = self.run_recipe()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["checkpoint"]["sha256"], experiment.artifact_hash(self.weights / "base-1")["sha256"])

    def test_missing_selected_checkpoint_fails_without_fallback(self):
        self.recipe["commands"] = [{"id": "evaluate", "stage": "evaluate",
                                     "args": ["--checkpoint", "base-99", "--weights-root", str(self.weights)],
                                     "timeout_seconds": 5}]
        result = self.run_recipe()
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing or incomplete", result["results"][0]["error"])
        self.assertEqual((self.output / "evaluate/stdout.log").read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
