mod support;

use omega_tokenizer::Tokens;
use support::{TempDir, save_pipeline};

#[test]
fn saved_special_token_pipeline_survives_round_trip() {
    let temp = TempDir::new();
    let original = temp.path().join("pipeline.json");
    save_pipeline(&original);
    let tokenizer = Tokens::new(&original).unwrap();
    let saved = temp.path().join("saved.json");
    tokenizer.save(&saved).unwrap();
    let reloaded = Tokens::new(&saved).unwrap();

    for tokenizer in [&tokenizer, &reloaded] {
        let encoding = tokenizer.encode("Hello world!", true).unwrap();
        assert_eq!(encoding.get_ids(), &[13, 2, 3, 11, 14]);
        assert_eq!(encoding.get_special_tokens_mask(), &[1, 0, 0, 0, 1]);
        assert_eq!(encoding.get_attention_mask(), &[1; 5]);
        assert_eq!(
            encoding.get_offsets(),
            &[(0, 0), (0, 5), (6, 11), (11, 12), (0, 0)]
        );
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), true).unwrap(),
            "hello world !"
        );
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "<start> hello world ! <end>"
        );
        assert_eq!(
            tokenizer.encode("Hello world!", false).unwrap().get_ids(),
            &[2, 3, 11]
        );
        assert!(tokenizer.encode("", false).unwrap().get_ids().is_empty());
        assert_eq!(tokenizer.encode("", true).unwrap().get_ids(), &[13, 14]);
        assert_eq!(tokenizer.decode(&[], true).unwrap(), "");
        assert_eq!(tokenizer.vocab_size(), 15);
        assert_eq!(tokenizer.token_to_id("<start>"), Some(13));
        assert_eq!(tokenizer.id_to_token(14).as_deref(), Some("<end>"));
        assert_eq!(tokenizer.token_to_id("not in vocabulary"), None);
        assert_eq!(tokenizer.id_to_token(15), None);
        let batch = tokenizer.encode_batch(&["Hello world!", ""], true).unwrap();
        assert_eq!(batch[0].get_ids(), encoding.get_ids());
        assert_eq!(batch[1].get_ids(), &[13, 14]);
        assert!(tokenizer.encode_batch(&[], true).unwrap().is_empty());
    }
}

#[test]
fn file_errors_identify_load_or_save_and_the_requested_path() {
    let temp = TempDir::new();
    for (name, contents) in [
        ("empty.json", ""),
        ("malformed.json", "{broken"),
        ("wrong-schema.json", "{}"),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, contents).unwrap();
        let error = Tokens::new(&path)
            .err()
            .expect("invalid tokenizer must fail");
        assert!(error.contains("Failed to load tokenizer"), "{error}");
        assert!(error.contains(&path.display().to_string()), "{error}");
    }
    let missing = temp.path().join("missing.json");
    let error = Tokens::new(&missing)
        .err()
        .expect("missing tokenizer must fail");
    assert!(error.contains("Failed to load tokenizer"), "{error}");
    assert!(error.contains(&missing.display().to_string()), "{error}");

    let source = temp.path().join("valid.json");
    save_pipeline(&source);
    let tokenizer = Tokens::new(source).unwrap();
    let destination = temp.path().join("absent-parent/saved.json");
    let error = tokenizer.save(&destination).unwrap_err();
    assert!(error.contains("Failed to save tokenizer"), "{error}");
    assert!(
        error.contains(&destination.display().to_string()),
        "{error}"
    );
}
