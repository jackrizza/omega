# Versioned fixtures and local data

This repository includes only:

- `test.json`: the 13-entry WordLevel tokenizer used by offline tests.
- `examples/greetings/train.txt` and `examples/omega/train.txt`: tiny text fixtures.
- `omega-alpha/model.toml`: an example dataset recipe; review its source and
  permitted-use settings before downloading anything.

Downloaded corpora, generated tokenizers, caches, prepared dataset releases and
other local project configurations are ignored. They are not required to build
Omega or run the offline tests. Existing local data is never deleted by these
ignore rules. Use `omega-datasets init` to create a separate recipe, or the Omega
application to create a whole-workflow project.
