mod support;

use omega_tokenizer::{ByteBpeConfig, CoverageReport, Tokens, train_byte_bpe};
use support::{TempDir, save_pipeline};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

fn config(vocab_size: usize, min_frequency: u64) -> ByteBpeConfig {
    ByteBpeConfig {
        vocab_size,
        min_frequency,
    }
}

#[test]
fn full_byte_alphabet_preserves_unseen_unicode_case_and_whitespace() {
    let tokenizer = train_byte_bpe(&["hello hello", "Hello!"], &config(256, 1)).unwrap();
    assert_eq!(tokenizer.vocab_size(), 256);
    for id in 0..256 {
        assert!(tokenizer.id_to_token(id).is_some());
    }
    assert_eq!(tokenizer.token_to_id("[UNK]"), None);
    for text in ["hello", "Hello", "\t Café 世界 😀\r\n  ", "a\0b", "", "  "] {
        let encoding = tokenizer.encode(text, false).unwrap();
        assert_eq!(tokenizer.decode(encoding.get_ids(), false).unwrap(), text);
        assert_eq!(tokenizer.decode(encoding.get_ids(), true).unwrap(), text);
        assert_eq!(
            tokenizer.encode(text, true).unwrap().get_ids(),
            encoding.get_ids()
        );
        assert!(encoding.get_attention_mask().iter().all(|&mask| mask == 1));
        assert!(
            encoding
                .get_special_tokens_mask()
                .iter()
                .all(|&mask| mask == 0)
        );
        assert!(encoding.get_overflowing().is_empty());
    }
    assert_ne!(
        tokenizer.encode("Hello", false).unwrap().get_ids(),
        tokenizer.encode("hello", false).unwrap().get_ids()
    );
    let serialized = tokenizer.to_json().unwrap();
    let pipeline = Tokenizer::from_bytes(serialized.as_bytes()).unwrap();
    assert!(pipeline.get_normalizer().is_none());
    assert!(pipeline.get_post_processor().is_none());
    assert!(pipeline.get_padding().is_none());
    assert!(pipeline.get_truncation().is_none());
    assert_eq!(
        pipeline.get_vocab_size(false),
        pipeline.get_vocab_size(true)
    );
}

#[test]
fn vocabulary_and_frequency_control_learned_merges() {
    let texts = ["hello hello hello hello", "Hello world! hello world!"];
    let merged = train_byte_bpe(&texts, &config(280, 1)).unwrap();
    let bytes = train_byte_bpe(&texts, &config(280, 100)).unwrap();
    assert!(merged.vocab_size() > 256 && merged.vocab_size() <= 280);
    assert_eq!(bytes.vocab_size(), 256);
    assert!(
        merged.encode("hello", false).unwrap().len() < bytes.encode("hello", false).unwrap().len()
    );
    assert_eq!(
        merged
            .decode(merged.encode("hello", false).unwrap().get_ids(), false)
            .unwrap(),
        "hello"
    );
}

#[test]
fn save_reload_preserves_ids_and_refuses_existing_or_invalid_paths() {
    let temp = TempDir::new();
    let tokenizer =
        train_byte_bpe(&["Hi! café hi Hi!", "\t Omega 世界\n"], &config(280, 1)).unwrap();
    let path = temp.path().join("trained.json");
    tokenizer.save_new(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let reloaded = Tokens::new(&path).unwrap();
    for id in 0..tokenizer.vocab_size() as u32 {
        assert_eq!(tokenizer.id_to_token(id), reloaded.id_to_token(id));
    }
    for text in ["Hi! café hi Hi!", " unseen 🦀\t世界\r\n"] {
        let expected = tokenizer.encode(text, false).unwrap();
        let actual = reloaded.encode(text, false).unwrap();
        assert_eq!(expected.get_ids(), actual.get_ids());
        assert_eq!(reloaded.decode(actual.get_ids(), false).unwrap(), text);
    }
    assert!(
        tokenizer
            .save_new(&path)
            .unwrap_err()
            .contains("never overwritten")
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert!(tokenizer.save_new(temp.path()).is_err());
    let missing = temp.path().join("missing-parent/tokenizer.json");
    assert!(
        tokenizer
            .save_new(&missing)
            .unwrap_err()
            .contains("Cannot create new tokenizer")
    );
    assert!(!missing.exists());
}

#[test]
fn concurrent_saves_reserve_exactly_one_new_file() {
    let temp = TempDir::new();
    let path = temp.path().join("trained.json");
    let tokenizer = train_byte_bpe(&["hello world"], &config(256, 1)).unwrap();
    let barrier = std::sync::Barrier::new(8);
    let successes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    tokenizer.save_new(&path).is_ok()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|&saved| saved)
            .count()
    });
    assert_eq!(successes, 1);
    assert_eq!(Tokens::new(path).unwrap().vocab_size(), 256);
}

#[test]
fn invalid_configs_and_empty_training_fail_before_training() {
    for invalid in [
        config(0, 1),
        config(255, 1),
        config(256, 0),
        config(usize::MAX, 1),
    ] {
        if invalid.vocab_size == usize::MAX && usize::BITS == 32 {
            continue;
        }
        assert!(invalid.validate().is_err());
        assert!(train_byte_bpe(&["hello"], &invalid).is_err());
    }
    assert!(ByteBpeConfig::default().validate().is_ok());
    for texts in [vec![], vec![""], vec![" ", "\n\t"]] {
        assert!(
            train_byte_bpe(&texts, &config(256, 1))
                .err()
                .unwrap()
                .contains("nonblank text")
        );
    }
}

#[test]
fn coverage_uses_explicit_unknown_id_and_documents_empty_metrics() {
    let temp = TempDir::new();
    let path = temp.path().join("pipeline.json");
    save_pipeline(&path);
    let tokenizer = Tokens::new(path).unwrap();
    assert_eq!(
        tokenizer.coverage("hello unlisted", Some(0)).unwrap(),
        CoverageReport {
            token_count: 2,
            unknown_count: Some(1),
            unknown_rate: Some(0.5),
        }
    );
    assert_eq!(
        tokenizer.coverage("hello unlisted", None).unwrap(),
        CoverageReport {
            token_count: 2,
            unknown_count: None,
            unknown_rate: None,
        }
    );
    assert_eq!(
        tokenizer
            .coverage("hello world", Some(0))
            .unwrap()
            .unknown_rate,
        Some(0.0)
    );
    assert_eq!(
        tokenizer.coverage("", Some(0)).unwrap(),
        CoverageReport {
            token_count: 0,
            unknown_count: Some(0),
            unknown_rate: None,
        }
    );
    assert!(
        tokenizer
            .coverage("hello", Some(100))
            .unwrap_err()
            .contains("not in the tokenizer vocabulary")
    );
    let byte_bpe = train_byte_bpe(&["hello"], &config(256, 1)).unwrap();
    let coverage = byte_bpe.coverage("🦀", None).unwrap();
    assert_eq!(coverage.token_count, "🦀".len());
    assert_eq!(coverage.unknown_count, None);
    assert_eq!(coverage.unknown_rate, None);
}

#[test]
fn coverage_rejects_padding_and_truncation_that_would_distort_metrics() {
    let temp = TempDir::new();
    let source = temp.path().join("source.json");
    save_pipeline(&source);
    let mut padding = Tokenizer::from_file(&source).unwrap();
    padding.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(4),
        ..Default::default()
    }));
    let padded = temp.path().join("padded.json");
    padding.save(&padded, true).unwrap();
    let tokenizer = Tokens::new(padded).unwrap();
    assert!(
        tokenizer
            .coverage("hello", Some(0))
            .unwrap_err()
            .contains("padding")
    );
    assert_eq!(
        tokenizer
            .coverage("hello world hello world", Some(0))
            .unwrap()
            .token_count,
        4
    );

    let mut truncation = Tokenizer::from_file(source).unwrap();
    truncation
        .with_truncation(Some(TruncationParams {
            max_length: 3,
            ..Default::default()
        }))
        .unwrap();
    let truncated = temp.path().join("truncated.json");
    truncation.save(&truncated, true).unwrap();
    let tokenizer = Tokens::new(truncated).unwrap();
    for text in ["hello", "hello world hello world"] {
        assert!(
            tokenizer
                .coverage(text, Some(0))
                .unwrap_err()
                .contains("truncation to be disabled")
        );
    }
}

#[cfg(unix)]
#[test]
fn save_new_rejects_dangling_and_existing_symlinks() {
    use std::os::unix::fs::symlink;
    let temp = TempDir::new();
    let tokenizer = train_byte_bpe(&["hello"], &config(256, 1)).unwrap();
    let target = temp.path().join("target.json");
    let link = temp.path().join("link.json");
    symlink(&target, &link).unwrap();
    assert!(tokenizer.save_new(&link).is_err());
    assert!(!target.exists());
    std::fs::write(&target, "existing data").unwrap();
    assert!(tokenizer.save_new(&link).is_err());
    assert_eq!(std::fs::read_to_string(target).unwrap(), "existing data");
}
