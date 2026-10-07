# Omega Tokenizer

## Opt-in assistant protocol

`train_chat_byte_bpe` adds the four versioned `omega-chat-v1` role/end-turn tokens
before model initialization (total vocabulary minimum 260). The ordinary BPE
API/defaults and saved fixtures remain unchanged. `chat::ChatProtocol` formats
completed conversations or user-ending prompts and returns aligned assistant
target masks. Literal control spellings remain ordinary byte text. Freeze the
tokenizer and protocol with model weights; no retroactive vocabulary resizing.
See [assistant execution](../../../Docs/assistant-execution.md) for schema,
compatibility, CLI usage and remaining real-data/hardware gates.

Loads a Hugging Face `tokenizer.json` and runs its saved normalization,
pre-tokenization, model, post-processing, and decoding pipeline.

## CLI

From `Code/Rust`:

```sh
cp test-fixtures/wordlevel.json ../../datasets/test.json
cargo run -p omega-tokenizer --bin main -- -f test.json --test-string "Hello world!"
cargo run -p omega-tokenizer --bin main -- -f test.json --decode-ids 2,3,11
cargo test -p omega-tokenizer
```

Relative file names resolve under the repository's `datasets` directory, using
the repository location at compile time. Use an absolute path when running a
built binary outside that checkout.

The empty crate-local `test.json` is an unused placeholder retained as existing
data. The versioned fixture is `Code/Rust/test-fixtures/wordlevel.json`; the copy
command creates an ignored local `datasets/test.json` for the legacy CLI examples.
Library paths are literal and do not use the CLI dataset root.

`--add-special-tokens` enables the loaded post-processor's special-token rules.
It does not automatically add BOS/EOS if no such rules are saved in the file.
`--skip-special-tokens` removes registered special tokens during decoding.
Failures return a nonzero exit code.

## Library

```rust,no_run
use omega_tokenizer::Tokens;

fn main() -> Result<(), String> {
    let tokenizer = Tokens::new("/path/to/tokenizer.json")?;
    let encoding = tokenizer.encode("Hello world!", true)?;
    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let decoded = tokenizer.decode(input_ids, true)?;
    let batch = tokenizer.encode_batch(&["Hello", "Another prompt"], true)?;
    Ok(())
}
```

Library paths are used as provided, unlike the CLI's dataset-relative paths.
Calls borrow the tokenizer so it can be reused. `Encoding` exposes IDs, attention
masks, special-token masks, offsets, and overflow encodings. Batch lengths may
vary unless the saved tokenizer enables padding. Saved truncation settings also
apply; inspect overflow encodings if truncation would otherwise lose input.

## Using a GPT model

- For pretrained weights, load the **exact tokenizer for those weights**. Token
  IDs index model embeddings; another vocabulary is not interchangeable.
- `Code/Rust/test-fixtures/wordlevel.json` is a tiny WordLevel smoke-test fixture, not a production
  GPT tokenizer or training corpus. It lowercases text and maps unknown words to
  `[UNK]`, so decoding is not lossless. Keep it unchanged for the fixture tests;
  save real model tokenizers under a different name.
- A GPT trained from scratch needs a tokenizer trained on representative text,
  then frozen before model training. `train_byte_bpe` trains byte BPE from supplied
  texts; folder selection and CLI orchestration live in `omega-training`.
- Chat templates and BOS/EOS policies are model-specific and may live outside
  `tokenizer.json`. Format prompts and configure special-token insertion using
  that model's documentation. Do not prepend BERT `[CLS]`/`[SEP]` tokens or guess
  their IDs. Avoid duplicating special tokens already present in a prompt.
- Respect the model's context length and padding requirements. The wrapper
  preserves saved settings rather than inventing a maximum length or PAD token.
- Convert input IDs and masks to the tensor types expected by the model. The
  attention mask identifies padding; the model must separately enforce causal
  attention. Next-token labels, label shifting, and ignoring padding in the loss
  belong to the training pipeline (some model APIs shift labels internally).
- Decode generated IDs using the same tokenizer. Token strings alone are not a
  substitute for decoding byte-level/BPE output.

The earlier standalone `normalize`, `pre_token`, and BERT-style `post_process`
methods are replaced by `encode`: manually running those stages can bypass or
conflict with the loaded tokenizer configuration.

## Training and coverage

`train_byte_bpe(&[&str], &ByteBpeConfig { vocab_size, min_frequency })` returns a
`Tokens`. Vocabulary size is an upper bound (minimum 256); frequency must be
positive. The pipeline includes every byte, preserves case/whitespace/UTF-8,
and has no normalizer, injected prefix space, BOS/EOS, padding or truncation.
It holds statistics in memory and does not promise deterministic retraining.

`Tokens::save_new(path)` reserves a new file without overwriting any existing
entry. Parent directories must exist; failed writes may leave a partial new file.
The existing `save` method retains its overwrite behavior. `to_json` serializes
the full pipeline but is not itself a canonical fingerprint.

`coverage(text, unknown_id)` reports token count and optional unknown count/rate.
The optional ID must exist; the caller establishes that it means unknown. No ID
means unknown metrics unavailable; empty text has no defined ratio. It rejects
configured truncation and actual padding. Byte BPE has no unknown token, and
complete byte coverage says nothing about segmentation efficiency or model quality.

From `Code/Rust`, using a new output name:

```sh
cargo run -p omega-training --bin main -- train-tokenizer --dataset examples --output ../../datasets/byte-bpe.json --vocab-size 512
cargo run -p omega-training --bin main -- coverage --tokenizer byte-bpe.json --text "Hello world!"
```

See the [training guide](../../../Docs/omega-training.md) for corpus formats and
paths. Save/freeze the tokenizer before model training; never substitute its IDs
for a different checkpoint's tokenizer.
