# Versioned fixtures and local data

This repository includes only:

- `examples/greetings/train.txt` and `examples/omega/train.txt`: tiny text fixtures.

The shared tokenizer fixture is now
[`Code/Rust/test-fixtures/wordlevel.json`](../Code/Rust/test-fixtures/wordlevel.json).
`datasets/test.json`, all of `datasets/omega-alpha/`, and all of `weights/` are
local-only and excluded from Git, including recipe files and directory markers.

Downloaded corpora, generated tokenizers, caches, prepared dataset releases and
other local project configurations are ignored. They are not required to build
Omega or run the offline tests. Existing local data is never deleted by these
ignore rules. Use `omega-datasets init` to create a separate recipe, or the Omega
application to create a whole-workflow project.
