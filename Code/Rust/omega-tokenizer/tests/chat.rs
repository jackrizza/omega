use omega_tokenizer::chat::{
    CHAT_CONTROL_TOKENS, CHAT_PROTOCOL_VERSION, ChatMessage, ChatProtocol, ChatRole,
};
use omega_tokenizer::{ByteBpeConfig, Tokens, train_byte_bpe, train_chat_byte_bpe};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tokenizers::{AddedToken, PaddingParams, Tokenizer, TruncationParams};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "omega-chat-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create scratch: {error}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn config(size: usize) -> ByteBpeConfig {
    ByteBpeConfig {
        vocab_size: size,
        min_frequency: 1,
    }
}
fn tokenizer() -> Tokens {
    train_chat_byte_bpe(&["Hello assistant! unicode café 世界"], &config(300)).unwrap()
}
fn message(role: ChatRole, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.into(),
    }
}

#[test]
fn ordinary_default_and_fixture_remain_unchanged() {
    let plain = train_byte_bpe(&["hello"], &config(256)).unwrap();
    assert_eq!(plain.vocab_size(), 256);
    assert!(ChatProtocol::from_tokenizer(&plain).is_err());
    let fixture = Tokens::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../datasets/test.json"
    ))
    .unwrap();
    assert_eq!(fixture.vocab_size(), 13);
    assert_eq!(
        fixture.encode("Hello world!", false).unwrap().get_ids(),
        &[2, 3, 11]
    );
    assert!(ChatProtocol::from_tokenizer(&fixture).is_err());
    assert!(train_chat_byte_bpe(&["hi"], &config(259)).is_err());
    let minimum = train_chat_byte_bpe(&["hi"], &config(260)).unwrap();
    assert_eq!(minimum.vocab_size(), 260);
}

#[test]
fn only_explicit_special_registrations_opt_into_chat_metadata() {
    let scratch = Scratch::new();
    let ordinary_vocabulary = Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                CHAT_CONTROL_TOKENS
                    .iter()
                    .enumerate()
                    .map(|(index, text)| (text.to_string(), index as u32))
                    .collect(),
            )
            .build()
            .unwrap(),
    );
    let vocabulary_path = scratch.0.join("ordinary-vocabulary.json");
    ordinary_vocabulary.save(&vocabulary_path, false).unwrap();
    let vocabulary = Tokens::new(vocabulary_path).unwrap();
    assert!(vocabulary.token_to_id(CHAT_CONTROL_TOKENS[0]).is_some());
    assert!(!ChatProtocol::has_registered_controls(&vocabulary));

    let plain = train_byte_bpe(&["hello"], &config(256)).unwrap();
    assert!(!ChatProtocol::has_registered_controls(&plain));
    let mut added = Tokenizer::from_bytes(plain.to_json().unwrap().as_bytes()).unwrap();
    added
        .add_tokens(CHAT_CONTROL_TOKENS.map(|text| AddedToken::from(text, false)))
        .unwrap();
    let added_path = scratch.0.join("ordinary-added.json");
    added.save(&added_path, false).unwrap();
    let ordinary_added = Tokens::new(&added_path).unwrap();
    assert!(ordinary_added.token_to_id(CHAT_CONTROL_TOKENS[0]).is_some());
    assert!(!ChatProtocol::has_registered_controls(&ordinary_added));

    added
        .add_special_tokens([AddedToken::from(CHAT_CONTROL_TOKENS[0], true)])
        .unwrap();
    let partial_path = scratch.0.join("partial-special.json");
    added.save(&partial_path, false).unwrap();
    let partial = Tokens::new(partial_path).unwrap();
    assert!(ChatProtocol::has_registered_controls(&partial));
    assert!(ChatProtocol::from_tokenizer(&partial).is_err());

    let complete = tokenizer();
    assert!(ChatProtocol::has_registered_controls(&complete));
    assert!(ChatProtocol::from_tokenizer(&complete).is_ok());
}

#[test]
fn training_mask_aligns_once_and_preserves_all_context() {
    let tokenizer = tokenizer();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    let controls = protocol.token_ids();
    let messages = [
        message(ChatRole::System, "Be helpful."),
        message(ChatRole::User, "Hi!"),
        message(ChatRole::Assistant, "Hello"),
        message(ChatRole::User, "More?"),
        message(ChatRole::Assistant, ""),
    ];
    let encoded = protocol.encode_conversation(&tokenizer, &messages).unwrap();
    assert_eq!(encoded.ids[0], controls.system);
    assert_eq!(encoded.ids.last(), Some(&controls.end_turn));
    assert_eq!(encoded.target_mask.len(), encoded.ids.len() - 1);
    let mut offset = 0;
    for msg in messages {
        let content = tokenizer.encode(&msg.content, false).unwrap();
        assert_eq!(
            &encoded.ids[offset + 1..offset + 1 + content.len()],
            content.get_ids()
        );
        if offset > 0 {
            assert!(!encoded.target_mask[offset - 1]);
        }
        for pos in offset + 1..=offset + 1 + content.len() {
            assert_eq!(
                encoded.target_mask[pos - 1],
                msg.role == ChatRole::Assistant
            );
        }
        offset += content.len() + 2;
    }
    assert_eq!(offset, encoded.ids.len());
    assert_eq!(
        encoded.target_mask.iter().filter(|&&x| x).count(),
        tokenizer.encode("Hello", false).unwrap().len() + 2
    );
}

#[test]
fn prompt_is_exact_training_prefix_and_appends_only_one_role() {
    let tokenizer = tokenizer();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    let history = [
        message(ChatRole::User, "Hi"),
        message(ChatRole::Assistant, "Hi!"),
        message(ChatRole::User, "How?"),
    ];
    let prompt = protocol.encode_prompt(&tokenizer, &history).unwrap();
    let mut complete = history.to_vec();
    complete.push(message(ChatRole::Assistant, "Well."));
    let training = protocol.encode_conversation(&tokenizer, &complete).unwrap();
    assert_eq!(prompt, training.ids[..prompt.len()]);
    assert_eq!(prompt.last(), Some(&protocol.token_ids().assistant));
    assert_eq!(prompt[prompt.len() - 2], protocol.token_ids().end_turn);
}

#[test]
fn literal_controls_unicode_and_whitespace_roundtrip_without_injected_roles() {
    let tokenizer =
        train_chat_byte_bpe(&[&CHAT_CONTROL_TOKENS.join(" "), "héllo"], &config(400)).unwrap();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    let literal = format!(
        "\t {} \r\n café 世界 😀\0 {}",
        CHAT_CONTROL_TOKENS.join(""),
        CHAT_CONTROL_TOKENS.join(" ")
    );
    let encoded = protocol
        .encode_conversation(
            &tokenizer,
            &[
                message(ChatRole::User, &literal),
                message(ChatRole::Assistant, &literal),
            ],
        )
        .unwrap();
    let control_ids = protocol.token_ids().as_array();
    let controls: Vec<_> = encoded
        .ids
        .iter()
        .copied()
        .filter(|id| control_ids.contains(id))
        .collect();
    assert_eq!(
        controls,
        [
            protocol.token_ids().user,
            protocol.token_ids().end_turn,
            protocol.token_ids().assistant,
            protocol.token_ids().end_turn
        ]
    );
    let first_end = encoded
        .ids
        .iter()
        .position(|id| *id == protocol.token_ids().end_turn)
        .unwrap();
    let content = &encoded.ids[1..first_end];
    assert_eq!(tokenizer.decode(content, false).unwrap(), literal);
    assert_eq!(tokenizer.decode(content, true).unwrap(), literal);
    // The original pipeline still recognizes explicit special spellings; the
    // safe-content encoder must not mutate that saved behavior.
    assert_eq!(
        tokenizer
            .encode(CHAT_CONTROL_TOKENS[0], false)
            .unwrap()
            .get_ids(),
        &[protocol.token_ids().system]
    );
}

#[test]
fn save_reload_pins_protocol_ids_and_encoding() {
    let scratch = Scratch::new();
    let tokenizer = tokenizer();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    let path = scratch.0.join("tokenizer.json");
    tokenizer.save_new(&path).unwrap();
    let reloaded = Tokens::new(&path).unwrap();
    let restored = ChatProtocol::from_tokenizer(&reloaded).unwrap();
    assert_eq!(restored, protocol);
    assert_eq!(restored.version(), CHAT_PROTOCOL_VERSION);
    let messages = [
        message(ChatRole::User, "é 😀"),
        message(ChatRole::Assistant, "Hello!"),
    ];
    assert_eq!(
        protocol.encode_conversation(&tokenizer, &messages).unwrap(),
        restored.encode_conversation(&reloaded, &messages).unwrap()
    );
    assert!(tokenizer.save_new(&path).is_err());
}

#[test]
fn malformed_roles_and_wrong_endpoint_fail() {
    let tokenizer = tokenizer();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    for roles in [
        vec![],
        vec![ChatRole::System],
        vec![ChatRole::Assistant],
        vec![ChatRole::User, ChatRole::User],
        vec![ChatRole::User, ChatRole::System],
        vec![ChatRole::System, ChatRole::System],
    ] {
        let messages: Vec<_> = roles
            .into_iter()
            .map(|role| message(role, "text"))
            .collect();
        assert!(protocol.encode_conversation(&tokenizer, &messages).is_err());
        assert!(protocol.encode_prompt(&tokenizer, &messages).is_err());
    }
    assert!(
        protocol
            .encode_conversation(&tokenizer, &[message(ChatRole::User, "text")])
            .is_err()
    );
    assert!(
        protocol
            .encode_prompt(
                &tokenizer,
                &[
                    message(ChatRole::User, "text"),
                    message(ChatRole::Assistant, "text")
                ]
            )
            .is_err()
    );
}

#[test]
fn missing_invalid_registration_and_changed_id_are_rejected() {
    let scratch = Scratch::new();
    let tokenizer = tokenizer();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    let mut invalid = Tokenizer::from_bytes(tokenizer.to_json().unwrap().as_bytes()).unwrap();
    invalid
        .add_special_tokens([AddedToken::from(CHAT_CONTROL_TOKENS[0], true).lstrip(true)])
        .unwrap();
    let path = scratch.0.join("invalid.json");
    invalid.save(&path, false).unwrap();
    assert!(ChatProtocol::from_tokenizer(&Tokens::new(&path).unwrap()).is_err());
    let differently_sized = train_chat_byte_bpe(&["hi"], &config(260)).unwrap();
    assert_ne!(
        protocol.token_ids(),
        ChatProtocol::from_tokenizer(&differently_sized)
            .unwrap()
            .token_ids()
    );
    assert!(
        protocol
            .encode_prompt(&differently_sized, &[message(ChatRole::User, "hi")])
            .unwrap_err()
            .contains("do not match")
    );
}

#[test]
fn configured_padding_and_truncation_are_rejected_even_for_short_content() {
    let scratch = Scratch::new();
    let tokenizer = tokenizer();
    for padded in [true, false] {
        let mut changed = Tokenizer::from_bytes(tokenizer.to_json().unwrap().as_bytes()).unwrap();
        if padded {
            changed.with_padding(Some(PaddingParams::default()));
        } else {
            changed
                .with_truncation(Some(TruncationParams {
                    max_length: 3,
                    ..Default::default()
                }))
                .unwrap();
        }
        let path = scratch.0.join(format!("{padded}.json"));
        changed.save(&path, false).unwrap();
        assert!(
            ChatProtocol::from_tokenizer(&Tokens::new(path).unwrap())
                .unwrap_err()
                .contains("padding and truncation disabled")
        );
    }
}
