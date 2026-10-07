use std::path::Path;
use tokenizers::Tokenizer;

pub use tokenizers::Encoding;

pub mod chat;
pub mod training;
pub use training::{ByteBpeConfig, CoverageReport, train_byte_bpe, train_chat_byte_bpe};

/// Runs the pipeline stored in a tokenizer JSON without changing its vocabulary
/// or special-token conventions. These must match the GPT model's weights.
pub struct Tokens {
    t: Tokenizer,
}

impl Tokens {
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self, String> {
        let path = file_name.as_ref();
        Tokenizer::from_file(path)
            .map(|t| Self { t })
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", path.display(), e))
    }

    /// Runs normalization, pre-tokenization, tokenization, and post-processing.
    /// Special tokens are added only as defined by the loaded post-processor;
    /// this does not invent BOS/EOS tokens or a model-specific chat template.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, String> {
        self.t
            .encode(text, add_special_tokens)
            .map_err(|e| format!("Failed to encode text: {}", e))
    }

    /// Uses the saved padding/truncation settings, if any. Without padding,
    /// returned sequences may have different lengths.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, String> {
        self.t
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| format!("Failed to encode batch: {}", e))
    }

    /// Uses the saved decoder, which is important for byte-level/BPE models.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.t
            .decode(ids, skip_special_tokens)
            .map_err(|e| format!("Failed to decode token IDs: {}", e))
    }

    pub fn vocab_size(&self) -> usize {
        self.t.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.t.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.t.id_to_token(id)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        self.t
            .save(path, true)
            .map_err(|e| format!("Failed to save tokenizer to {}: {}", path.display(), e))
    }

    /// Serialize the complete saved pipeline without changing its configuration.
    /// This is JSON, not a canonical fingerprint; identity callers should parse
    /// and canonicalize its object keys before hashing.
    pub fn to_json(&self) -> Result<String, String> {
        self.t
            .to_string(false)
            .map_err(|error| format!("Failed to serialize tokenizer: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokens {
        Tokens {
            t: Tokenizer::from_bytes(include_bytes!("../../test-fixtures/wordlevel.json"))
                .expect("valid test tokenizer"),
        }
    }

    #[test]
    fn encodes_using_the_loaded_pipeline() {
        let tokenizer = tokenizer();
        let encoding = tokenizer.encode("Hello world!", false).unwrap();
        assert_eq!(encoding.get_ids(), &[2, 3, 11]);
        assert_eq!(encoding.get_attention_mask(), &[1, 1, 1]);
        assert_eq!(encoding.get_offsets(), &[(0, 5), (6, 11), (11, 12)]);
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), true).unwrap(),
            "hello world !"
        );
        assert_eq!(tokenizer.encode("test", true).unwrap().get_ids(), &[1]);
    }

    #[test]
    fn handles_unknown_tokens_and_special_token_decoding() {
        let tokenizer = tokenizer();
        let encoding = tokenizer.encode("unlisted", false).unwrap();
        assert_eq!(encoding.get_ids(), &[0]);
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "[UNK]"
        );
        assert_eq!(tokenizer.decode(encoding.get_ids(), true).unwrap(), "");
        assert_eq!(tokenizer.token_to_id("test"), Some(1));
        assert_eq!(tokenizer.id_to_token(1).as_deref(), Some("test"));
        assert_eq!(tokenizer.vocab_size(), 13);
    }

    #[test]
    fn batch_matches_individual_encodings_including_empty_input() {
        let tokenizer = tokenizer();
        let texts = ["hello world", "test", ""];
        let batch = tokenizer.encode_batch(&texts, false).unwrap();
        assert_eq!(batch.len(), texts.len());
        for (encoding, text) in batch.iter().zip(texts) {
            assert_eq!(
                encoding.get_ids(),
                tokenizer.encode(text, false).unwrap().get_ids()
            );
        }
    }
}
