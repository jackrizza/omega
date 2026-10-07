# Shared offline test fixtures

`wordlevel.json` is the unchanged 13-entry WordLevel tokenizer used by the Rust
workspace tests and Linux persistence tests. Its token IDs are part of the test
contract. It is not a production tokenizer or training corpus.

Keep test inputs here instead of depending on ignored local datasets. Tests must
write outputs to temporary directories. The legacy `datasets/test.json` copy,
if present on a developer machine, is local data and excluded from Git.
