# Omega NN

A small decoder-only GPT built with Burn 0.18 and integrated with
`omega-tokenizer`. CPU execution uses NdArray; training uses its autodiff wrapper.
`generate_with_options_on_device` supports explicit f32 backends; optional Vulkan
training/orchestration is documented in [GPU training](../../../Docs/gpu-training.md).
This is a runnable learning/testing foundation, not a pretrained language model.

## Try it

Run from `Code/Rust`. Select `-p omega-nn` because the tokenizer also has a binary
named `main`. Model/tokenizer options go before the subcommand.

```sh
cp test-fixtures/wordlevel.json ../../datasets/test.json
cargo run -p omega-nn --bin main -- -f test.json forward --text "hello world!"
cargo run -p omega-nn --bin main -- -f test.json train --text "hello world ! hello world ! hello world !" --prompt "hello" --steps 100 --learning-rate 0.003 --max-new-tokens 5
cargo run -p omega-nn --bin main -- --d-model 16 --heads 2 --layers 1 --d-ff 32 train --text "hello world !" --prompt "hello" --steps 50 --max-new-tokens 2
cargo test -p omega-nn
```

Relative tokenizer paths resolve under the repository's `datasets` directory at
its compile-time location. Absolute paths work outside that checkout.
`--help` and `train --help` list options. Invalid input exits with a failure code.

`forward` prints input IDs, logits dimensions, and predictions from **random
weights**. Those predictions are not meaningful language generation.

`train` initializes new weights, overfits one literal text, prints first/last
pre-update cross-entropy losses, and generates using the trained in-memory model.
**Weights are not saved between runs.** `--seed` controls initialization for a
standalone run; backend RNG state is shared within the process.

## Architecture and API

- Learned token and absolute-position embeddings.
- Pre-layer-normalized Transformer blocks with causal multi-head attention,
  GELU feed-forward layers, and residual connections.
- Final layer normalization and an untied vocabulary projection producing raw
  logits `[batch, sequence, vocabulary]`.
- Adam optimization with next-token cross entropy. For tokens `[a,b,c,d]`,
  inputs are `[a,b,c]` and targets `[b,c,d]`; the target shift happens exactly once.
- Greedy-default decoding or explicit seeded token sampling, with optional
  `--eos-token-id`; no EOS ID is guessed.

`GptConfig::init<B>` creates a backend-generic `Gpt<B>`. `Gpt::forward` accepts
integer tensors `[batch, sequence]`. `token_tensor` constructs a validated
single-sequence tensor. Direct tensor callers must enforce token bounds,
nonempty batches/sequences, and the context limit themselves.

`validate_tokenizer` checks for a dense vocabulary, `encode_text` uses the saved
tokenization pipeline, `train_on_tokens` returns an inference model plus loss
history, and `generate` returns prompt plus generated IDs. Pass the model's exact
configuration to `generate`.

`Gpt::architecture()` derives a checked `GptConfig` from parameter shapes and
attention metadata. `Gpt::validate_config(&config)` rejects mismatches before
generation or checkpoint saving, including zero-token generation requests.
Existing helper signatures and Burn record layouts are unchanged. Legacy records
do not store head counts, so loading still trusts the saved configuration for
that metadata; this check does not authenticate checkpoint provenance or tokenizer
identity. Direct `forward` calls can panic for invalid shapes/IDs; checked helpers
return actionable errors.

## Token sampling

`generate` remains the compatible greedy wrapper. `generate_with_options` accepts
`GenerationOptions::Greedy` or `GenerationOptions::Sample(SamplingOptions)`;
sampling defaults are temperature 1, no top-k/top-p filter, seed 42. The NN `train`
subcommand and training crate's `generate` command expose `--sample`,
`--temperature`, `--top-k`, `--top-p` and `--sampling-seed`. Sampling-only flags
require `--sample`; NN's existing `--seed` still controls model initialization.

Temperature must be finite and positive, top-k in 1..=vocabulary size, and top-p
in (0, 1]. Stable softmax probabilities use descending order with token-ID ties;
top-p is measured against the mass retained after top-k, then probabilities are
renormalized for a request-local portable PRNG draw. This generator consumes no
Burn/training RNG state. A newly initialized Burn model can still materialize lazy
parameters on its first forward; that existing initialization is separate from
token selection. Nonfinite logits fail in both modes. Repeatability applies
to identical logits/options, not arbitrary builds or platforms. Greedy retains
the existing backend argmax tie behavior. No KV cache or streaming is implemented.

## Padding and masked loss

`Gpt::forward_masked(tokens, valid)` accepts integer IDs and a boolean validity
mask, both `[batch, sequence]`, with true for real tokens. Each row must have a
nonempty valid prefix followed only by right padding. All IDs, including padding,
must be in range; choose any valid padding ID. It combines causal and padding-key
masks, preserves valid-token logits and zeros padded output logits. Fully masked
rows, left padding, gaps, mismatched shapes/devices and invalid IDs are errors.

`masked_cross_entropy(logits, targets, valid)` takes `[batch, sequence, vocab]`
logits and `[batch, sequence]` targets/boolean mask. Labels must already be aligned;
the helper does not shift them. Its target mask may have gaps or entirely ignored
rows. Ignored IDs/logits may be arbitrary: selected rows are gathered before loss,
giving ignored logits zero gradient. An entirely ignored batch is an error.
The returned `MaskedLoss` contains differentiable mean `loss` and `target_count`.
Use the appropriate target mask separately from the input-attention mask.

These checked APIs synchronize data to the host for validation and reject
nonfinite valid outputs/loss. Backend failures may still panic. Existing `forward`
and model record fields remain unchanged. Tests cover NdArray/autodiff and the
optional Vulkan backend on the qualified A770; see the
[GPU guide](../../../Docs/gpu-training.md) for tolerances and device limits.
Minibatch collation and
training orchestration use these APIs in [omega-training](../omega-training/README.md);
the NN convenience training helpers retain their unpadded behavior.

## Boundaries

- Existing training/generation helpers use unpadded inputs. Encoding rejects
  actual padding and reported truncation overflow. Saved truncation can omit
  overflow in tokenizers 0.23; disable it to avoid undetected truncation.
- Training accepts 2 through `context_length + 1` tokens. Generation requires
  prompt length plus the requested new tokens to fit the context, even if EOS
  might stop early. No sliding window, KV cache, or streaming generation.
- This crate has no dataset loader or checkpointing; folder training and
  inference checkpoints are provided by [omega-training](../omega-training/README.md).
  Minibatch training, dropout, chat templates, automatic BOS/EOS
  insertion, and pretrained-weight import are not implemented.
- The sample WordLevel tokenizer lowercases text and has only 13 entries. Unknown
  words become `[UNK]`; use its known words for smoke tests. For real training,
  choose/train and freeze a suitable tokenizer first. The model's embeddings and
  output head must use exactly that tokenizer's IDs.
- The CPU defaults are intentionally tiny. Attention memory grows quadratically
  with sequence length. Increasing CLI dimensions can exhaust memory.

Tests cover shapes, finite logits, causal masking, invalid input, token tensor
construction, tokenizer integration, generation/EOS, and decreasing training loss.
