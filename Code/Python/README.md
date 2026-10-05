# Python utilities

## Assistant corpus preparation

```sh
python Code/Python/prepare_assistant_data.py --spec sources.json --output NEW_RELEASE_DIRECTORY
```

The schema-1 specification has `release` and `sources`. Each source records `id`,
relative `/`-separated `path`, `format` (`text`, `jsonl`, or `chat`), `group`,
explicit `partition` (`train`, `validation`, `test`), `language`, `provenance`,
`permitted_use`, and `extraction`. Source paths resolve relative to the spec.

```json
{
  "schema_version": 1,
  "release": "assistant-v1",
  "sources": [{
    "id": "document-a", "path": "raw/document.txt", "format": "text",
    "group": "document-family", "partition": "train", "language": "en",
    "provenance": "Record the actual source", "permitted_use": "Record reviewed permission",
    "extraction": "UTF-8 source with reviewed extraction"
  }]
}
```

This illustrates the schema, not an accepted corpus. Keep raw inputs unchanged
and supply reviewed groups/partitions. The tool creates base/chat partition
directories and records source/output hashes, exact duplicate membership, blank
record skips and bounded near-duplicate candidates. Cross-partition exact overlap
between base documents and individual chat messages fails, including repeated
system messages; review data rather than bypassing that conservative rule.
Near-duplicate detection is incomplete and candidates require human review.
`COMPLETE` means preparation finished, not permission or quality certification.

Chat input uses `{"schema_version":1,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}`.
An optional initial system message is followed by alternating user/assistant turns.
Training records end assistant. Preparation rejects blank messages; the runtime
formatter still permits empty generated replies. Limits bound source/aggregate/
record bytes, record/audit counts and comparisons; exceeded limits fail explicitly.
Run `--help` for controls. Symlink/reparse inputs, source aliases, traversal and
existing output paths are rejected. Failed publication may leave an incomplete
new release; originals are never overwritten or removed.

## Bounded experiment recording

```sh
python Code/Python/assistant_experiment.py hash PREPARED_INPUT_PATH
python Code/Python/assistant_experiment.py run --recipe recipe.json --output NEW_RUN_DIRECTORY
python Code/Python/assistant_experiment.py report --scores scores.json --checkpoint EXACT_CHECKPOINT_DIRECTORY --output NEW_REPORT_DIRECTORY
```

Recipes use `schema_version: 1`, an `executable` path, supplied `runtime` metadata,
`inputs` with `name`/`path`/expected `sha256`, and `commands` with `id`, `stage`,
`args` and `timeout_seconds`. Stages are `train`, `resume`, `train-stage`,
`evaluate`, `generate` and noninteractive `chat --prompt`. Supply complete CLI
argument arrays, not shell strings; `{run_dir}` can place outputs in the new run.
Training commands require `--max-updates`; loading commands require exact numbered
`--checkpoint` and explicit `--weights-root`. No latest fallback is permitted.

The recorder checks input hashes, preserves the recipe/argv/executable identity,
hashes selected checkpoints before use, records stdout/stderr/results and stops
on the first failure. It attempts cooperative stopping at timeout, then cleans up
children. Native Windows Job Object and Linux process-group cleanup are tested.
Recognized validation/test path guards are not proof of corpus separation. Logs
have a sampled limit (default 16 MiB; polling/grace can overshoot); checkpoints
have no enforced disk quota. Hash operations cap each artifact at 1 GiB/100,000
entries. `COMPLETE` means command success, not assistant quality.

Human-score files use schema 1, `partition: "validation"`, `selected_checkpoint`,
`selection_reason`, a named `rubric`, and `records` containing `prompt_id`,
`checkpoint`, `criterion`, `score`, `max_score`, `notes`. Reports retain these
actual supplied scores and hash the explicit selected checkpoint; no scores are
invented and sealed-test data must not be used for selection. See
[assistant execution](../../Docs/assistant-execution.md) for unresolved training gates.

The assistant tools use Python's standard library. Test with:

```sh
python -m unittest discover -s Code/Python -p "test_prepare_assistant_data.py" -v
python -m unittest discover -s Code/Python -p "test_assistant_experiment.py" -v
```

## PDF to text

`pdf_to_text.py` converts every PDF directly inside an input folder into a
same-named UTF-8 `.txt` file in an output folder. Requires Python 3.10 or newer.

From the repository root on Windows:

```sh
py -m pip install -r Code/Python/requirements.txt
py Code/Python/pdf_to_text.py --input "C:/Documents/PDFs" --output "datasets/books"
```

On systems where the interpreter is named `python` or `python3`, substitute that
command for `py`. Folder paths are relative to the current working directory
unless absolute. Quote paths containing spaces. Short flags `-i` and `-o` are
also available. The output folder is created if needed.

For example, `report.pdf` becomes `report.txt`. Both `.pdf` and `.PDF` extensions
are accepted. Subfolders are not scanned. Existing output files are never
overwritten: they are reported as failures, and other files continue converting.
Use a new output folder for a full rerun.

Extraction runs locally with `pypdf`; the script does not upload documents.
Installing dependencies requires package-index access. It does not modify PDFs.
Pages are separated by blank lines. PDF reading order, tables, headers, and
layout may need manual cleanup before the text is used for training.

**No OCR:** scanned/image-only PDFs without an embedded text layer cannot be
converted by this script. Documents with no extractable text fail without
creating an empty output file. In mixed text/image PDFs only embedded text is
extracted; image-only pages do not receive OCR. Password-protected documents
must be decrypted separately before conversion.

Exit codes:

- `0`: every discovered PDF converted successfully.
- `1`: missing dependency, invalid folders, no PDFs, or one or more failed files.
  Successful outputs are retained when other files fail.
- `2`: invalid or missing CLI arguments.

Files produced under `datasets/books` can be selected by Omega Training with
`--dataset books`. Choose a suitable tokenizer first: the bundled WordLevel test
fixture is not appropriate for general PDF text. Omega's training loader also
requires each nonblank document to encode to at least two tokens.

## Tests

After installing requirements, from the repository root:

```sh
py -m unittest discover -s Code/Python -v
py Code/Python/pdf_to_text.py --help
```

Tests use temporary folders and locally generated PDFs; they do not modify user
PDFs or training datasets.
