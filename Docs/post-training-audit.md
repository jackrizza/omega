# Post-training prerequisite audit — 2026-10-07

This is a bounded, read-only inventory of the local checkout before production
post-training. It is not acceptance evidence for a model. No remote training host
was inspected, no credentials were read, and no production corpus or checkpoint
was modified. Large payloads were not hashed or decoded for this inventory.

## Dataset evidence

`datasets/omega-alpha/release-v1/manifest.json` records a TinyStories source pinned
to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The recorded raw shard size is
248,731,111 bytes. It reports 529,930 records seen, 55 blank records and 20,250
duplicates. The prepared partitions are:

| Partition | Base records | Base payload bytes | Chat records |
| --- | ---: | ---: | ---: |
| Train | 458,803 | 424,689,113 | 0 |
| Validation | 25,472 | 23,514,130 | 0 |
| Test | 25,350 | 23,485,822 | 0 |

The 3,072-byte manifest hashes to
`1f190c5aaabfd89255a345b4bb87112cbfe596d2daa3fb90aa1c7a7f82597186`, matching its
83-byte `COMPLETE` marker. This audit did not recompute the large partition or
membership hashes. Presence of a corpus is not evidence of permission to use it.
The source's `permitted_use` is still the literal
`REVIEW REQUIRED: record the license and your permitted use`. Its audit also does
not certify near/semantic deduplication, license, privacy or human quality review.

Only the first 100 training records were sampled, totaling 79,676 bytes. Their
character lengths were minimum 398, median 778 and maximum 1,213; whitespace word
counts were 80, 149 and 242 respectively. These are not token counts or population
estimates. Held-out contents were not sampled. The named directories under
`datasets/assistant-v1/{base,chat}/{train,validation,test}` were empty.

There is therefore local base-language train/validation/test infrastructure, but
no populated conversational training/development/sealed-test release in these
audited locations. A conversational release needs explicit source permission,
reviewed source-family partitioning, duplicate review and suitable completed
system/user/assistant records before production SFT.

## Active project and parent compatibility

At audit time, `datasets/omega_tui_test/model.toml` was a schema-1 configuration
named `Omega_Tui_Test`, selecting the base training partition. It specified a
context of 64 tokens, width 32, four attention heads, two layers, feed-forward
width 128, Vulkan device 0, ten epochs, learning rate 0.003 and batch size one.
Evaluation was disabled with no evaluation selections. It did not specify
training update, elapsed-time or storage caps. The sample above suggests context
length requires a measured token-length study before treating this configuration
as a useful conversational target.

The saved tokenizer in `datasets/omega_tui_test/tokenizers/` was 363,794 bytes:
8,188 ordinary BPE tokens plus four explicit chat controls, with IDs system 8188,
user 8189, assistant 8190 and end-turn 8191. It had no padding, truncation or
normalizer. Its receipt recorded canonical identity
`62e1ffe7f4646dd71cca878f5aa111bf125a4934a56ef1c66b350b756e1a4cd6`.
The receipt file was 33,034,145 bytes, exceeding the then-current 8 MiB generic
JSON reader bound used for receipt reuse; this was reported separately as a
tooling issue, not evidence that the tokenizer itself was invalid.

The project's local `weights/` directory was empty. The 25 checkpoint directories
under repository-root `weights/` instead used a 151,665-token tokenizer and
context 64/width 32/four heads/two layers/feed-forward width 128. They cannot be
paired with the new 8,192-token tokenizer. Some were legacy or inference-only
artifacts. Bounded header reads, including `macro-invest-gpu-18` and `gpu-pilot-3`,
were insufficient to establish a compatible, complete selected parent. The
active remote training run was deliberately left untouched and uninspected.

## Evidence still needed

No production baseline conversational report, complete human rubric review,
selected-candidate manifest, model card, or backup/restore evidence was found in
the scoped local datasets, weights and documentation locations. Historical
performance benchmark JSON files establish neither conversational quality nor
production acceptance. Temporary fixture tests establish software behavior only.

Before starting the production workflow, record:

- The exact completed parent checkpoint and compatible saved tokenizer/protocol,
  target runtime, and a verified recovery copy.
- Intended language, subject scope, context length, expected answer length and
  limitations of the assistant.
- Permitted conversational sources, reviewed train/development/sealed-test
  partitions, and a separate base-language regression validation selection.
- A fixed development suite, decoding settings, versioned human rubric, acceptance
  thresholds, and the person responsible for reviewing actual generated answers.
- Segment and total update limits, elapsed-time and evaluation budgets, storage
  allowance, stop/recovery behavior, and a separately authorized sealed-test pass.

The executable readiness checker can verify structural compatibility, bounded
record eligibility, selected release integrity and exact overlap. It cannot
resolve source permission, semantic duplication, answer quality, or adequacy of
the production training budget. Re-run readiness against the selected production
inputs after these decisions are recorded; this dated inventory is not a substitute.
