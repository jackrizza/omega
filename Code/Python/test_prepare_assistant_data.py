"""Preparation regressions; all data and releases live in temporary directories."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_assistant_data as preparation


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = self.root / "sources.json"
        self.output = self.root / "release"
        self.sources = []

    def source(self, identifier, text, partition="train", group=None, kind="text", path=None):
        relative = path or identifier + (".txt" if kind == "text" else ".jsonl")
        location = self.root / relative
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(text, encoding="utf-8")
        self.sources.append({"id": identifier, "path": relative, "format": kind,
                             "group": group or identifier, "partition": partition,
                             "language": "en", "provenance": "Temporary unit-test fixture",
                             "permitted_use": "Test author grants fixture use",
                             "extraction": "Created as UTF-8"})

    def write_spec(self):
        self.spec.write_text(json.dumps({"schema_version": 1, "release": "fixture-v1",
                                         "sources": self.sources}), encoding="utf-8")

    def run_preparation(self, **kwargs):
        self.write_spec()
        return preparation.prepare(self.spec, self.output, **kwargs)

    @staticmethod
    def chat(user="Name this color.", assistant="Blue.", system=None):
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.extend([{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}])
        return json.dumps({"schema_version": 1, "messages": messages})

    def assert_unpublished_failure(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_preparation()
        self.assertFalse(self.output.exists())

    def test_deterministic_outputs_and_preserved_unicode_text(self):
        self.source("book", "  Café Ω\n  text stays exactly.\n")
        self.source("dialogue", self.chat(), kind="chat", partition="validation")
        first = self.run_preparation()
        other = self.root / "other"
        second = preparation.prepare(self.spec, other)
        self.assertEqual(first, second)
        files = sorted(path.relative_to(self.output) for path in self.output.rglob("*") if path.is_file())
        self.assertEqual(files, sorted(path.relative_to(other) for path in other.rglob("*") if path.is_file()))
        for file in files:
            self.assertEqual((self.output / file).read_bytes(), (other / file).read_bytes())
        book = next(member for member in first["records"] if member["source"] == "book")
        self.assertEqual((self.output / book["output"]).read_text(encoding="utf-8"),
                         "  Café Ω\n  text stays exactly.\n")
        self.assertFalse(first["audit"]["all_stage_partitions_nonempty"])
        self.assertEqual(first["audit"]["human_review"], "required")
        marker = json.loads((self.output / "COMPLETE").read_bytes())
        self.assertEqual(marker["manifest_sha256"], preparation.digest((self.output / "manifest.json").read_bytes()))

    def test_duplicate_removal_retains_every_membership(self):
        self.source("one", "same   contents")
        self.source("two", "same\ncontents")
        manifest = self.run_preparation()
        self.assertEqual(manifest["counts"]["base"]["train"], 1)
        self.assertEqual(len(manifest["records"]), 2)
        self.assertEqual(manifest["records"][1]["duplicate_of"], "one")
        self.assertEqual(len(manifest["exact_overlaps"]), 1)

    def test_source_family_must_stay_in_one_partition_across_stages(self):
        self.source("book", "some valid independent text", group="family")
        self.source("chat", self.chat(), group="family", partition="test", kind="chat")
        self.assert_unpublished_failure("group .* crosses partitions")

    def test_exact_cross_stage_message_leakage_rejected(self):
        self.source("book", "Blue.", partition="train")
        self.source("chat", self.chat(), partition="test", kind="chat")
        self.assert_unpublished_failure("Exact content leakage")

    def test_message_overlap_without_identical_conversations_is_rejected(self):
        self.source("first", self.chat("Question A", "Answer is shared."), kind="chat")
        self.source("second", self.chat("Question B", "Answer is  shared."), kind="chat", partition="validation")
        self.assert_unpublished_failure("Exact content leakage")

    def test_near_containment_is_reported_unresolved(self):
        self.source("book", "A long sentence containing a shared sequence of useful words here and then more.")
        self.source("chat", self.chat("Explain these words", "a shared sequence of useful words here"),
                    kind="chat", partition="test")
        manifest = self.run_preparation()
        candidates = manifest["near_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["cross_partition"])
        self.assertTrue(candidates[0]["cross_stage"])
        self.assertEqual(candidates[0]["review"], "unresolved")

    def test_invalid_chat_and_physical_line_diagnostics(self):
        self.source("bad", '\n{"schema_version":1,"messages":[{"role":"assistant","content":"Wrong first role"}]}', kind="chat")
        self.assert_unpublished_failure("bad/@record-2: message 1 must have role user")

    def test_blank_and_short_records_reported_without_silent_filtering(self):
        self.source("base", '\n{"text":"   "}\n{"text":"a"}\n{"text":"usable text"}\n', kind="jsonl")
        manifest = self.run_preparation()
        self.assertEqual(manifest["counts"]["base"]["train"], 1)
        self.assertEqual(manifest["rejected"], [{"id": "base/@record-2", "reason": "blank"},
                                                {"id": "base/@record-3", "reason": "too_short"}])

    def test_overwrite_refusal_and_sources_unchanged(self):
        self.source("book", "original source contents")
        source_bytes = (self.root / "book.txt").read_bytes()
        self.run_preparation()
        marker_bytes = (self.output / "COMPLETE").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            preparation.prepare(self.spec, self.output)
        self.assertEqual(marker_bytes, (self.output / "COMPLETE").read_bytes())
        self.assertEqual(source_bytes, (self.root / "book.txt").read_bytes())

    def test_traversal_absolute_and_backslash_paths_rejected(self):
        self.source("book", "valid contents")
        for path in ("../book.txt", str(self.root / "book.txt"), "raw\\book.txt", "./book.txt"):
            with self.subTest(path=path):
                self.sources[0]["path"] = path
                self.assert_unpublished_failure("Source path")

    def test_repeated_physical_input_rejected(self):
        self.source("book", "valid contents")
        self.sources.append({**self.sources[0], "id": "alias", "group": "other"})
        self.assert_unpublished_failure("Source alias")

    def test_hardlink_alias_rejected(self):
        self.source("book", "valid contents")
        try:
            os.link(self.root / "book.txt", self.root / "alias.txt")
        except OSError as error:
            self.skipTest(f"Hardlinks unavailable: {error}")
        self.sources.append({**self.sources[0], "id": "alias", "path": "alias.txt"})
        self.assert_unpublished_failure("Source alias")

    def test_symlink_rejected_if_supported(self):
        self.source("book", "valid contents")
        try:
            (self.root / "alias.txt").symlink_to(self.root / "book.txt")
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        self.sources[0]["path"] = "alias.txt"
        self.assert_unpublished_failure("Symlink/reparse")

    def test_input_record_total_and_pair_limits(self):
        self.source("one", "some useful words in this source")
        self.source("two", "another distinct document")
        self.source("three", "third unique document")
        self.write_spec()
        for limits in (preparation.Limits(max_source_bytes=1), preparation.Limits(max_total_bytes=1),
                       preparation.Limits(max_record_bytes=1), preparation.Limits(max_records=1),
                       preparation.Limits(max_audit_units=1), preparation.Limits(max_pairs=1)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                preparation.prepare(self.spec, self.output, limits)
            self.assertFalse(self.output.exists())

    def test_failed_publication_has_no_complete_marker(self):
        self.source("book", "valid contents")
        self.write_spec()
        real_open = Path.open

        def fail_manifest(path, *args, **kwargs):
            if path.name == "manifest.json":
                raise OSError("simulated disk full")
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_manifest), self.assertRaisesRegex(OSError, "disk full"):
            preparation.prepare(self.spec, self.output)
        self.assertTrue(self.output.exists())
        self.assertFalse((self.output / "COMPLETE").exists())

    def test_invalid_permission_metadata_and_cli_failure(self):
        self.source("book", "valid contents")
        self.sources[0]["permitted_use"] = ""
        self.write_spec()
        with redirect_stderr(StringIO()) as errors, redirect_stdout(StringIO()):
            status = preparation.main(["--spec", str(self.spec), "--output", str(self.output)])
        self.assertEqual(status, 1)
        self.assertIn("permitted_use", errors.getvalue())
        self.assertFalse(self.output.exists())

    def test_nonfinite_duplicate_keys_utf8_and_role_errors(self):
        self.source("bad", "valid contents", kind="chat")
        bad_records = [
            '{"schema_version":1,"schema_version":2,"messages":[]}',
            '{"schema_version":NaN,"messages":[]}',
            '{"schema_version":true,"messages":[]}',
            '{"schema_version":1,"messages":[{"role":"user","content":"missing answer"}]}',
            self.chat(assistant="   "),
        ]
        self.write_spec()
        for record in bad_records:
            with self.subTest(record=record):
                (self.root / "bad.jsonl").write_text(record, encoding="utf-8")
                with self.assertRaises(ValueError):
                    preparation.prepare(self.spec, self.output)
                self.assertFalse(self.output.exists())
        (self.root / "bad.jsonl").write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            preparation.prepare(self.spec, self.output)

    def test_complete_base_chat_partitions_are_reported(self):
        for partition, text in zip(preparation.PARTITIONS, ("planet", "mountain", "river")):
            self.source("base-" + partition, text, partition=partition)
            self.source("chat-" + partition, self.chat("Question about " + text, "Reply about " + text),
                        partition=partition, kind="chat")
        manifest = self.run_preparation()
        self.assertTrue(manifest["audit"]["all_stage_partitions_nonempty"])
        self.assertEqual(len(manifest["records"]), 6)
        self.assertEqual(manifest["audit"]["human_review"], "required")


if __name__ == "__main__":
    unittest.main()
