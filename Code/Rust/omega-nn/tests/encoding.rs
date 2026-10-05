use omega_nn::{encode_text, validate_tokenizer};
use omega_tokenizer::Tokens;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);

impl Fixture {
    fn new(vocabulary: &str, padding: &str, truncation: &str) -> Self {
        Self::with_model(
            &format!(r#"{{"type":"WordLevel", "vocab":{vocabulary}, "unk_token":"[UNK]"}}"#),
            padding,
            truncation,
        )
    }

    fn with_model(model: &str, padding: &str, truncation: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = loop {
            let path = std::env::temp_dir().join(format!(
                "omega-nn-encoding-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create temporary fixture directory: {error}"),
            }
        };
        let fixture = Self(directory);
        let json = format!(
            r#"{{
            "version":"1.0", "truncation":{truncation}, "padding":{padding},
            "added_tokens":[], "normalizer":null,
            "pre_tokenizer":{{"type":"Whitespace"}}, "post_processor":null, "decoder":null,
            "model":{model}
        }}"#
        );
        fs::write(fixture.0.join("tokenizer.json"), json).unwrap();
        fixture
    }

    fn tokenizer(&self) -> Tokens {
        Tokens::new(self.0.join("tokenizer.json")).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove this test's temporary fixture");
    }
}

const VOCABULARY: &str = r#"{"[UNK]":0,"hello":1,"world":2,"[PAD]":3}"#;

#[test]
fn vocabulary_validation_rejects_empty_and_sparse_ids() {
    let empty = Fixture::new("{}", "null", "null");
    assert_eq!(empty.tokenizer().vocab_size(), 0);
    assert!(
        validate_tokenizer(&empty.tokenizer())
            .unwrap_err()
            .contains("must not be empty")
    );
    let sparse = Fixture::new(r#"{"[UNK]":0,"hello":2}"#, "null", "null");
    assert_eq!(sparse.tokenizer().vocab_size(), 2);
    assert_eq!(sparse.tokenizer().token_to_id("hello"), Some(2));
    assert!(
        validate_tokenizer(&sparse.tokenizer())
            .unwrap_err()
            .contains("missing ID 1")
    );
    let dense = Fixture::new(VOCABULARY, "null", "null");
    assert_eq!(validate_tokenizer(&dense.tokenizer()).unwrap(), 4);
}

#[test]
fn actual_padding_is_rejected_but_unpadded_encodings_are_accepted() {
    let fixture = Fixture::new(
        VOCABULARY,
        r#"{"strategy":{"Fixed":3},"direction":"Right","pad_to_multiple_of":null,"pad_id":3,"pad_type_id":0,"pad_token":"[PAD]"}"#,
        "null",
    );
    let tokenizer = fixture.tokenizer();
    let padded = tokenizer.encode("hello", false).unwrap();
    assert_eq!(padded.get_ids(), &[1, 3, 3]);
    assert_eq!(padded.get_attention_mask(), &[1, 0, 0]);
    assert!(
        encode_text(&tokenizer, "hello")
            .unwrap_err()
            .contains("padded")
    );
    assert_eq!(
        encode_text(&tokenizer, "hello world hello").unwrap(),
        [1, 2, 1]
    );
}

#[test]
fn reported_truncation_overflow_is_rejected_but_short_text_is_accepted() {
    let fixture = Fixture::with_model(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"h":1,"##e":2,"##l":3,"##o":4},"unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100}"###,
        "null",
        // One pre-token must expand beyond the limit to report overflow:
        // tokenizers 0.23 may stop between pre-tokens without reporting it.
        r#"{"direction":"Right","max_length":2,"strategy":"LongestFirst","stride":0}"#,
    );
    let tokenizer = fixture.tokenizer();
    let truncated = tokenizer.encode("hello", false).unwrap();
    assert_eq!(truncated.get_ids(), &[1, 2]);
    assert_eq!(truncated.get_overflowing().len(), 2);
    assert_eq!(truncated.get_overflowing()[0].get_ids(), &[3, 3]);
    assert_eq!(truncated.get_overflowing()[1].get_ids(), &[4]);
    assert!(
        encode_text(&tokenizer, "hello")
            .unwrap_err()
            .contains("truncated")
    );
    assert_eq!(encode_text(&tokenizer, "he").unwrap(), [1, 2]);
}

#[test]
fn empty_text_is_rejected_without_inventing_special_tokens() {
    let fixture = Fixture::new(VOCABULARY, "null", "null");
    let tokenizer = fixture.tokenizer();
    assert!(tokenizer.encode("", false).unwrap().get_ids().is_empty());
    assert!(
        encode_text(&tokenizer, "")
            .unwrap_err()
            .contains("at least one token")
    );
    assert_eq!(encode_text(&tokenizer, "hello world").unwrap(), [1, 2]);
}
