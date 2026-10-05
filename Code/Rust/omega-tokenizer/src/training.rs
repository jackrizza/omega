//! Byte-level BPE training from supplied texts and explicit coverage inspection.

use std::{fs::OpenOptions, io::Write, path::Path};

use tokenizers::{
    Tokenizer,
    models::{
        TrainerWrapper,
        bpe::{BPE, BpeTrainer},
    },
    pre_tokenizers::byte_level::ByteLevel,
};

use crate::Tokens;

/// Byte BPE vocabulary settings. Every byte is included before learning merges.
/// The requested vocabulary is an upper bound; a small corpus may yield fewer
/// merges. Retraining creates new token IDs, incompatible with existing weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteBpeConfig {
    pub vocab_size: usize,
    pub min_frequency: u64,
}

impl Default for ByteBpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            min_frequency: 2,
        }
    }
}

impl ByteBpeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size < 256 || u32::try_from(self.vocab_size).is_err() {
            return Err("Byte BPE vocab_size must be at least 256 and fit in u32".into());
        }
        if self.min_frequency == 0 {
            return Err("Byte BPE min_frequency must be greater than zero".into());
        }
        Ok(())
    }
}

/// Train a case-preserving byte-level BPE tokenizer on the supplied documents.
/// Texts are passed separately without trimming or concatenating boundaries.
/// Requires at least one nonblank text. Corpus discovery belongs to the caller;
/// this trainer holds token/pair statistics in memory and is not streaming.
///
/// The pipeline has no normalizer, prefix-space injection, added special tokens,
/// BOS/EOS, padding, or truncation. A full byte alphabet and ByteLevel decoder
/// preserve arbitrary UTF-8 text, including bytes absent from the corpus. There
/// is no unknown-token ID; complete byte coverage does not measure vocabulary
/// efficiency or language-model quality. No deterministic retraining guarantee
/// is made; save and freeze the trained tokenizer with its model weights.
pub fn train_byte_bpe(texts: &[&str], config: &ByteBpeConfig) -> Result<Tokens, String> {
    config.validate()?;
    if !texts.iter().any(|text| !text.trim().is_empty()) {
        return Err("Tokenizer training requires at least one nonblank text".into());
    }
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, true)));
    let mut trainer = TrainerWrapper::from(
        BpeTrainer::builder()
            .vocab_size(config.vocab_size)
            .min_frequency(config.min_frequency)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .show_progress(false)
            .build(),
    );
    tokenizer
        .train(&mut trainer, texts.iter().copied())
        .map_err(|error| format!("Failed to train byte BPE tokenizer: {error}"))?;
    Ok(Tokens { t: tokenizer })
}

/// Train a new v1 chat tokenizer with four reserved controls. The vocabulary
/// bound includes those controls (minimum 260). Existing pipelines and the
/// ordinary `train_byte_bpe` default are unchanged; no weights may precede this
/// vocabulary choice. Freeze the resulting pipeline before model initialization.
pub fn train_chat_byte_bpe(texts: &[&str], config: &ByteBpeConfig) -> Result<Tokens, String> {
    config.validate()?;
    let controls = crate::chat::CHAT_CONTROL_TOKENS;
    if config.vocab_size < 256 + controls.len() {
        return Err("Chat byte BPE vocab_size must be at least 260 including controls".into());
    }
    let mut tokenizer = train_byte_bpe(
        texts,
        &ByteBpeConfig {
            vocab_size: config.vocab_size - controls.len(),
            min_frequency: config.min_frequency,
        },
    )?;
    for spelling in controls {
        if tokenizer.token_to_id(spelling).is_some() {
            return Err(format!(
                "Reserved chat control spelling already exists in learned vocabulary: {spelling}"
            ));
        }
    }
    let specials: Vec<_> = controls
        .into_iter()
        .map(|spelling| tokenizers::AddedToken::from(spelling, true).normalized(false))
        .collect();
    tokenizer
        .t
        .add_special_tokens(specials)
        .map_err(|error| format!("Cannot register chat controls: {error}"))?;
    crate::chat::ChatProtocol::from_tokenizer(&tokenizer)?;
    Ok(tokenizer)
}

/// Counts from encoding the supplied text without requesting added specials.
/// `unknown_count` is unavailable unless the caller supplies the tokenizer's
/// known unknown-token ID. Its existence is checked, its meaning is the caller's
/// responsibility. The ratio uses all emitted tokens as denominator and is
/// unavailable for empty encodings. Zero unknowns do not imply efficient byte
/// BPE segmentation or a good language model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageReport {
    pub token_count: usize,
    pub unknown_count: Option<usize>,
    pub unknown_rate: Option<f64>,
}

impl Tokens {
    /// Save to a new explicit path. Atomic create-new reservation rejects every
    /// existing entry, including symlinks; existing data is never overwritten.
    /// Serialization precedes reservation. Missing parent directories are errors.
    /// A failed write can leave a partial new file for inspection; this is not an
    /// atomic publication or power-loss durability guarantee.
    pub fn save_new(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let json = self.to_json()?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)
            .map_err(|error| format!("Cannot create new tokenizer {} (existing paths are never overwritten): {error}", path.display()))?;
        file.write_all(json.as_bytes())
            .and_then(|()| file.flush())
            .map_err(|error| format!("Cannot write new tokenizer {}: {error}", path.display()))
    }

    /// Inspect a full text without guessing its unknown token. Actual padding
    /// and configured truncation are rejected: tokenizers 0.23 can truncate
    /// without reporting overflow, so its absence cannot prove full coverage.
    pub fn coverage(&self, text: &str, unknown_id: Option<u32>) -> Result<CoverageReport, String> {
        if let Some(id) = unknown_id
            && self.id_to_token(id).is_none()
        {
            return Err(format!(
                "Unknown-token ID {id} is not in the tokenizer vocabulary"
            ));
        }
        if self.t.get_truncation().is_some() {
            return Err(
                "Coverage requires truncation to be disabled to inspect the full text".into(),
            );
        }
        let encoding = self.encode(text, false)?;
        if !encoding.get_overflowing().is_empty() {
            return Err("Coverage encoding has truncation overflow; disable truncation".into());
        }
        if encoding.get_attention_mask().contains(&0) {
            return Err("Coverage encoding contains padding; disable padding".into());
        }
        let token_count = encoding.get_ids().len();
        let unknown_count = unknown_id.map(|unknown| {
            encoding
                .get_ids()
                .iter()
                .filter(|&&id| id == unknown)
                .count()
        });
        let unknown_rate = unknown_count
            .and_then(|count| (token_count > 0).then(|| count as f64 / token_count as f64));
        Ok(CoverageReport {
            token_count,
            unknown_count,
            unknown_rate,
        })
    }
}
