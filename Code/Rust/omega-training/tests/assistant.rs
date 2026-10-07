use burn::{
    optim::{
        Adam, Optimizer,
        adaptor::OptimizerAdaptor,
        record::{AdaptorRecord, AdaptorRecordV1},
    },
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use omega_nn::{Gpt, GptConfig, TrainingBackend, token_tensor};
use omega_tokenizer::{ByteBpeConfig, Tokens, train_chat_byte_bpe};
use omega_training::{
    TrainingSession, ValidationSplit,
    assistant::prepare_conversations,
    checkpoint::{CheckpointMetadata, load_checkpoint, read_checkpoint_manifest},
    dataset::ExampleSource,
    resume::{
        initialize_assistant_stage_on_device, load_training_checkpoint, read_resume_manifest,
        save_training_checkpoint, save_training_checkpoint_with_parent,
    },
    trainer::SessionOptions,
};
use std::{fs, path::Path};

fn tokenizer() -> Tokens {
    train_chat_byte_bpe(
        &["Hi ok hello world"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap()
}
fn put(root: &Path, name: &str, text: &str) {
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}
fn chat(answer: &str) -> String {
    serde_json::json!({"schema_version":1,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":answer}]}).to_string()
}

type AdamRecord = <OptimizerAdaptor<Adam, Gpt<TrainingBackend>, TrainingBackend> as Optimizer<
    Gpt<TrainingBackend>,
    TrainingBackend,
>>::Record;

fn optimizer_record(path: &Path) -> AdamRecord {
    <_ as Recorder<TrainingBackend>>::load(
        &NamedMpkBytesRecorder::<FullPrecisionSettings>::default(),
        fs::read(path.join("optimizer.mpk")).unwrap(),
        &Default::default(),
    )
    .unwrap()
}

fn assert_parent_has_moments(path: &Path) {
    let record = optimizer_record(path);
    assert!(!record.is_empty());
    let mut nonzero = false;
    for entry in record.values() {
        let (time, moment_1, moment_2) = match entry {
            AdaptorRecord::V1(AdaptorRecordV1::Rank1(_)) => {
                let state = entry.clone().into_state::<1>();
                (
                    state.momentum.time,
                    state.momentum.moment_1.into_data().to_vec::<f32>().unwrap(),
                    state.momentum.moment_2.into_data().to_vec::<f32>().unwrap(),
                )
            }
            AdaptorRecord::V1(AdaptorRecordV1::Rank2(_)) => {
                let state = entry.clone().into_state::<2>();
                (
                    state.momentum.time,
                    state.momentum.moment_1.into_data().to_vec::<f32>().unwrap(),
                    state.momentum.moment_2.into_data().to_vec::<f32>().unwrap(),
                )
            }
            _ => panic!("Unexpected GPT optimizer rank"),
        };
        assert_eq!(time, 1);
        assert!(moment_1.iter().all(|value| value.is_finite()));
        assert!(
            moment_2
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        nonzero |=
            moment_1.iter().any(|value| *value != 0.0) && moment_2.iter().any(|value| *value > 0.0);
    }
    assert!(nonzero, "Parent must have learned nonzero Adam moments");
}

#[test]
fn conversations_are_bounded_masked_and_identity_checked() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tok = tokenizer();
    put(
        root,
        "train/a.jsonl",
        &format!("\n{}\n{}", chat("ok"), chat("hello")),
    );
    let source =
        prepare_conversations(root, &["train".into()], &tok, 64, ValidationSplit::None, 42)
            .unwrap();
    assert_eq!(source.training.example_count(), 2);
    assert!(source.training.target_count().unwrap() < source.training_tokens - 2);
    let identity = source.training.identity().unwrap();
    assert_eq!(
        source.training.target_mask(0).unwrap().unwrap().len(),
        source.training.example(0).unwrap().len() - 1
    );
    assert_eq!(source.provenance.documents[0].id, "train/a.jsonl/@record-2");
    assert!(
        prepare_conversations(root, &["train".into()], &tok, 2, ValidationSplit::None, 42).is_err()
    );
    assert!(
        prepare_conversations(
            root,
            &["train".into()],
            &tok,
            64,
            ValidationSplit::Count(1),
            42
        )
        .is_err()
    );
    put(
        root,
        "train/a.jsonl",
        &format!("\n{}\n{}", chat("no"), chat("hello")),
    );
    let changed =
        prepare_conversations(root, &["train".into()], &tok, 64, ValidationSplit::None, 42)
            .unwrap();
    assert_ne!(identity, changed.training.identity().unwrap());
    put(
        root,
        "train/a.jsonl",
        "{\"text\":\"ordinary JSONL is not chat\"}",
    );
    assert!(
        prepare_conversations(root, &["train".into()], &tok, 64, ValidationSplit::None, 42)
            .is_err()
    );
}

#[test]
fn new_stage_preserves_parent_weights_resets_progress_and_resumes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tok = tokenizer();
    put(root, "train/a.jsonl", &chat("ok"));
    let config = GptConfig {
        vocab_size: tok.vocab_size(),
        context_length: 32,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let base_set = omega_training::TrainingSet {
        examples: vec![vec![1, 2, 3, 4]],
        files: vec![],
        token_count: 4,
    };
    let mut base = TrainingSession::new(&config, base_set, 0.001, 42).unwrap();
    base.step().unwrap();
    let parent_path =
        save_training_checkpoint(root, "base", &base, &tok, &CheckpointMetadata::default())
            .unwrap();
    let parent_bytes = fs::read(parent_path.join("model.mpk")).unwrap();
    let manifest = read_checkpoint_manifest(&parent_path).unwrap().unwrap();
    assert_eq!(manifest.schema_version, 2);
    assert!(manifest.chat.is_some());
    assert_eq!(
        read_resume_manifest(&parent_path).unwrap().schema_version,
        3
    );
    let data = prepare_conversations(root, &["train".into()], &tok, 32, ValidationSplit::None, 42)
        .unwrap()
        .training;
    let (mut stage, saved_tok, parent) =
        initialize_assistant_stage_on_device::<_, TrainingBackend>(
            &parent_path,
            data.clone(),
            0.0001,
            7,
            SessionOptions::default(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(stage.progress().completed_updates, 0);
    let ids = || {
        token_tensor::<omega_nn::Cpu>(&[1, 2], config.vocab_size, 32, &Default::default()).unwrap()
    };
    assert_eq!(
        base.inference_model().forward(ids()).into_data(),
        stage.inference_model().forward(ids()).into_data()
    );
    assert_parent_has_moments(&parent_path);
    let initialized = save_training_checkpoint_with_parent(
        root,
        "initialized",
        &stage,
        &saved_tok,
        &CheckpointMetadata::default(),
        Some(&parent),
    )
    .unwrap();
    // Burn creates Adam state lazily: an empty map means no inherited moments
    // and no per-parameter step counters, rather than allocated zero tensors.
    assert!(optimizer_record(&initialized).is_empty());
    assert_eq!(
        read_resume_manifest(&initialized).unwrap().progress,
        Default::default()
    );
    let mut fresh_restored =
        load_training_checkpoint(&initialized, data.clone(), &config, &tok).unwrap();
    assert_eq!(stage.step().unwrap(), fresh_restored.step().unwrap());
    assert_eq!(
        stage.inference_model().forward(ids()).into_data(),
        fresh_restored.inference_model().forward(ids()).into_data()
    );
    let path = save_training_checkpoint_with_parent(
        root,
        "sft",
        &stage,
        &saved_tok,
        &CheckpointMetadata::default(),
        Some(&parent),
    )
    .unwrap();
    let header = read_resume_manifest(&path).unwrap();
    assert_eq!(header.schema_version, 5);
    assert_eq!(header.parent.as_ref(), Some(&parent));
    let mut restored = load_training_checkpoint(&path, data, &config, &tok).unwrap();
    assert_eq!(stage.step().unwrap(), restored.step().unwrap());
    assert_eq!(
        stage.inference_model().forward(ids()).into_data(),
        restored.inference_model().forward(ids()).into_data()
    );
    assert_eq!(
        fs::read(parent_path.join("model.mpk")).unwrap(),
        parent_bytes
    );
    let catalog =
        omega_training::checkpoint_catalog::discover_checkpoints(root, "sft", &Default::default())
            .unwrap();
    assert_eq!(
        catalog
            .latest(omega_training::checkpoint_catalog::CatalogMode::Resume)
            .unwrap()
            .path,
        path.canonicalize().unwrap()
    );
    let (loaded, loaded_config, _) = load_checkpoint(&path).unwrap();
    assert_eq!(loaded_config, config);
    assert!(loaded.validate_config(&config).is_ok());
    fs::write(parent_path.join("model.mpk"), b"corrupt").unwrap();
    let data = prepare_conversations(root, &["train".into()], &tok, 32, ValidationSplit::None, 42)
        .unwrap()
        .training;
    assert!(
        initialize_assistant_stage_on_device::<_, TrainingBackend>(
            &parent_path,
            data,
            0.0001,
            7,
            SessionOptions::default(),
            &Default::default()
        )
        .is_err()
    );
}

#[test]
fn new_stage_rejects_incomplete_plain_and_mismatched_parent_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tok = tokenizer();
    put(root, "train/a.jsonl", &chat("ok"));
    let source =
        prepare_conversations(root, &["train".into()], &tok, 32, ValidationSplit::None, 42)
            .unwrap()
            .training;
    let config = GptConfig {
        vocab_size: tok.vocab_size(),
        context_length: 32,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let make = |tokenizer: &Tokens| {
        let configuration = GptConfig {
            vocab_size: tokenizer.vocab_size(),
            ..config.clone()
        };
        let set = omega_training::TrainingSet {
            examples: vec![vec![1, 2, 3]],
            files: vec![],
            token_count: 3,
        };
        let session = TrainingSession::new(&configuration, set, 0.001, 42).unwrap();
        save_training_checkpoint(
            root,
            "parent",
            &session,
            tokenizer,
            &CheckpointMetadata::default(),
        )
        .unwrap()
    };
    let error = |path: &Path| {
        initialize_assistant_stage_on_device::<_, TrainingBackend>(
            path,
            source.clone(),
            0.001,
            42,
            SessionOptions::default(),
            &Default::default(),
        )
        .err()
        .expect("Invalid parent must not initialize a stage")
    };

    let incomplete = make(&tok);
    fs::remove_file(incomplete.join("COMPLETE")).unwrap();
    assert!(error(&incomplete).contains("Incomplete checkpoint"));

    let plain = omega_tokenizer::train_byte_bpe(
        &["Hi ok"],
        &ByteBpeConfig {
            vocab_size: 256,
            min_frequency: 1,
        },
    )
    .unwrap();
    assert!(error(&make(&plain)).contains("no frozen chat protocol"));

    let config_mismatch = make(&tok);
    let file = config_mismatch.join("config.json");
    let mut changed: serde_json::Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    changed["context_length"] = 33.into();
    fs::write(&file, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(error(&config_mismatch).contains("config"));

    let tokenizer_mismatch = make(&tok);
    let file = tokenizer_mismatch.join("tokenizer.json");
    let mut changed: serde_json::Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    changed["normalizer"] = serde_json::json!({"type":"Lowercase"});
    fs::write(&file, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(error(&tokenizer_mismatch).contains("okenizer"));

    let training_header_mismatch = make(&tok);
    let file = training_header_mismatch.join("resume.json");
    let mut changed: serde_json::Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    changed["model"]["d_ff"] = 12.into();
    let bytes = serde_json::to_vec(&changed).unwrap();
    fs::write(&file, &bytes).unwrap();
    fs::write(
        training_header_mismatch.join("resume.sha256"),
        omega_training::checkpoint::sha256_bytes(&bytes),
    )
    .unwrap();
    assert!(error(&training_header_mismatch).contains("headers disagree"));
}

#[test]
fn chat_metadata_cannot_be_missing_mismatched_or_downgraded() {
    let temp = tempfile::tempdir().unwrap();
    let tok = tokenizer();
    let config = GptConfig {
        vocab_size: tok.vocab_size(),
        context_length: 4,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let set = omega_training::TrainingSet {
        examples: vec![vec![1, 2]],
        files: vec![],
        token_count: 2,
    };
    let session = TrainingSession::new(&config, set, 0.001, 42).unwrap();
    let path = save_training_checkpoint(
        temp.path(),
        "base",
        &session,
        &tok,
        &CheckpointMetadata::default(),
    )
    .unwrap();
    let manifest = path.join("manifest.json");
    let original: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("chat");
    fs::write(&manifest, serde_json::to_vec(&missing).unwrap()).unwrap();
    assert!(read_checkpoint_manifest(&path).is_err());
    let mut wrong = original.clone();
    wrong["chat"]["token_ids"][0] = 0.into();
    fs::write(&manifest, serde_json::to_vec(&wrong).unwrap()).unwrap();
    assert!(load_checkpoint(&path).is_err());
    let mut downgraded = original;
    downgraded["schema_version"] = 1.into();
    fs::write(&manifest, serde_json::to_vec(&downgraded).unwrap()).unwrap();
    assert!(load_checkpoint(&path).is_err());
}

#[test]
fn ordinary_control_spellings_preserve_plain_checkpoint_behavior() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let fixture_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/wordlevel.json");
    let original: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_path).unwrap()).unwrap();
    let spelling = omega_tokenizer::chat::CHAT_CONTROL_TOKENS[0];
    let mut ordinary = original;
    ordinary["model"]["vocab"][spelling] = 13.into();
    let config = GptConfig {
        vocab_size: 14,
        context_length: 4,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    for (name, added) in [("ordinary", false), ("non-special", true)] {
        let mut pipeline = ordinary.clone();
        if added {
            pipeline["added_tokens"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "id":13,"content":spelling,"single_word":false,"lstrip":false,
                    "rstrip":false,"normalized":false,"special":false,
                }));
        }
        let tokenizer_path = root.join(format!("{name}.json"));
        fs::write(&tokenizer_path, serde_json::to_vec(&pipeline).unwrap()).unwrap();
        let tokenizer = Tokens::new(&tokenizer_path).unwrap();
        assert_eq!(tokenizer.token_to_id(spelling), Some(13));
        assert_eq!(tokenizer.vocab_size(), 14);
        let model = config.init::<omega_nn::Cpu>(&Default::default()).unwrap();
        let input = token_tensor::<omega_nn::Cpu>(&[13, 2], 14, 4, &Default::default()).unwrap();
        let before = model.forward(input.clone()).into_data();
        let checkpoint =
            omega_training::save_checkpoint(root, name, model, &config, &tokenizer).unwrap();
        let manifest = read_checkpoint_manifest(&checkpoint).unwrap().unwrap();
        assert_eq!(manifest.schema_version, 1);
        assert!(manifest.chat.is_none());
        let (loaded, loaded_config, loaded_tokenizer) = load_checkpoint(&checkpoint).unwrap();
        assert_eq!(loaded_config, config);
        assert_eq!(loaded_tokenizer.token_to_id(spelling), Some(13));
        assert_eq!(
            omega_training::checkpoint::tokenizer_identity(&loaded_tokenizer).unwrap(),
            omega_training::checkpoint::tokenizer_identity(&tokenizer).unwrap()
        );
        assert_eq!(loaded.forward(input).into_data(), before);
    }

    // One explicit special is an incomplete opt-in, so it must still fail.
    let mut partial = ordinary;
    partial["normalizer"] = serde_json::Value::Null;
    partial["added_tokens"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "id":13,"content":spelling,"single_word":false,"lstrip":false,
            "rstrip":false,"normalized":false,"special":true,
        }));
    let tokenizer_path = root.join("partial.json");
    fs::write(&tokenizer_path, serde_json::to_vec(&partial).unwrap()).unwrap();
    let tokenizer = Tokens::new(&tokenizer_path).unwrap();
    let model = config.init::<omega_nn::Cpu>(&Default::default()).unwrap();
    let error =
        omega_training::save_checkpoint(root, "partial", model, &config, &tokenizer).unwrap_err();
    assert!(
        error.contains("Missing") && error.contains("control token"),
        "{error}"
    );
    assert!(!root.join("partial-1").exists());
}
