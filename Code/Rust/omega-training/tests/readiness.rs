use omega_nn::GptConfig;
use omega_tokenizer::{ByteBpeConfig, train_byte_bpe, train_chat_byte_bpe};
use omega_training::{
    CheckpointMetadata, TrainingSession, TrainingSet,
    checkpoint::sha256_bytes,
    readiness::{ReadinessReport, inspect, verify_active_inputs},
    resume::save_training_checkpoint,
};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};

struct Fixture {
    directory: tempfile::TempDir,
    parent: PathBuf,
}

fn put(root: &Path, relative: &str, value: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, value).unwrap();
}

fn conversation(question: &str, answer: &str) -> String {
    json!({"schema_version":1,"messages":[
        {"role":"system","content":"Common boilerplate"},
        {"role":"user","content":question},
        {"role":"assistant","content":answer}
    ]})
    .to_string()
}

fn fixture(chat: bool) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let config = ByteBpeConfig {
        vocab_size: if chat { 260 } else { 256 },
        min_frequency: 1,
    };
    let tokenizer = if chat {
        train_chat_byte_bpe(&["some text"], &config)
    } else {
        train_byte_bpe(&["some text"], &config)
    }
    .unwrap();
    let config = GptConfig {
        vocab_size: tokenizer.vocab_size(),
        context_length: 128,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let source = TrainingSet {
        examples: vec![vec![4, 5, 6]],
        files: vec![],
        token_count: 3,
    };
    let mut session = TrainingSession::new(&config, source, 0.001, 42).unwrap();
    session.step().unwrap();
    let parent = save_training_checkpoint(
        root,
        "parent",
        &session,
        &tokenizer,
        &CheckpointMetadata::default(),
    )
    .unwrap();
    put(
        root,
        "train/data.jsonl",
        &conversation("Question A?", "Answer A."),
    );
    put(
        root,
        "validation/data.jsonl",
        &conversation("Question B?", "Answer B."),
    );
    put(
        root,
        "sealed/data.jsonl",
        &conversation("Question C?", "Answer C."),
    );
    put(
        root,
        "base_validation/text.txt",
        "Separate language regression document.",
    );
    Fixture { directory, parent }
}

fn check(fixture: &Fixture) -> ReadinessReport {
    inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["train".into()],
        &["validation".into()],
        &["base_validation".into()],
        &["sealed".into()],
    )
    .unwrap()
}

#[test]
fn readiness_reports_masked_counts_and_artifacts_without_mutation_or_approval() {
    let fixture = fixture(true);
    let before = fs::read(fixture.parent.join("model.mpk")).unwrap();
    let report = check(&fixture);
    assert!(report.passed, "{:?}", report.errors);
    assert_eq!(report.production_acceptance, "not_evaluated");
    assert_eq!(report.partitions.len(), 4);
    let training = report
        .partitions
        .iter()
        .find(|part| part.role == "train")
        .unwrap();
    assert_eq!(training.examples, 1);
    assert!(training.eligible_targets.unwrap() < training.input_tokens.unwrap());
    assert!(training.max_input_positions.unwrap() <= 128);
    assert!(
        report
            .identities
            .iter()
            .any(|item| item.role == "parent/model.mpk")
    );
    assert!(
        report
            .findings
            .iter()
            .any(|line| line.contains("ordinary folder"))
    );
    assert_eq!(before, fs::read(fixture.parent.join("model.mpk")).unwrap());
    let json = serde_json::to_vec(&report).unwrap();
    let loaded: ReadinessReport = serde_json::from_slice(&json).unwrap();
    assert_eq!(loaded.parent.unwrap().model.context_length, 128);
}

#[test]
fn exact_record_and_individual_message_overlap_fail_without_echoing_content() {
    let fixture = fixture(true);
    put(
        fixture.directory.path(),
        "validation/data.jsonl",
        &conversation("Other prompt", "Answer  A."),
    );
    let report = check(&fixture);
    assert!(!report.passed);
    assert!(
        report
            .errors
            .iter()
            .any(|line| line.contains("Exact whole-record/message"))
    );
    assert!(!report.errors.iter().any(|line| line.contains("Answer A")));
    put(
        fixture.directory.path(),
        "sealed/data.jsonl",
        &conversation("Question A?", "Answer A."),
    );
    let report = check(&fixture);
    assert!(
        report
            .checks
            .iter()
            .any(|check| check.category == "overlap/train/sealed" && !check.passed)
    );
}

#[test]
fn base_regression_answer_overlap_and_shared_partition_paths_fail() {
    let fixture = fixture(true);
    put(
        fixture.directory.path(),
        "base_validation/text.txt",
        "Answer A.",
    );
    let report = check(&fixture);
    assert!(
        report
            .checks
            .iter()
            .any(|check| check.category == "overlap/train/base_validation" && !check.passed)
    );
    let report = inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["train".into()],
        &["train".into()],
        &["base_validation".into()],
        &["sealed".into()],
    )
    .unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.contains("Shared physical source"))
    );
}

#[test]
fn context_and_invalid_roles_are_actionable_and_data_remains_unchanged() {
    let fixture = fixture(true);
    let oversized = conversation("Question A?", &"long content ".repeat(100));
    put(fixture.directory.path(), "train/data.jsonl", &oversized);
    let report = check(&fixture);
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.contains("context is 128"))
    );
    assert_eq!(
        fs::read_to_string(fixture.directory.path().join("train/data.jsonl")).unwrap(),
        oversized
    );
    put(
        fixture.directory.path(),
        "train/data.jsonl",
        "{\"schema_version\":1,\"messages\":[{\"role\":\"tool\",\"content\":\"wrong\"}]}",
    );
    assert!(
        check(&fixture)
            .errors
            .iter()
            .any(|error| error.contains("Unsupported conversation role"))
    );
}

#[test]
fn missing_parent_controls_and_corrupt_or_incomplete_payloads_fail() {
    let plain = fixture(false);
    assert!(!check(&plain).passed);
    let fixture = fixture(true);
    fs::write(fixture.parent.join("model.mpk"), "corrupt").unwrap();
    assert!(!check(&fixture).passed);
    fs::remove_file(fixture.parent.join("COMPLETE")).unwrap();
    assert!(
        check(&fixture)
            .errors
            .iter()
            .any(|line| line.contains("COMPLETE") || line.contains("completed"))
    );
}

#[test]
fn all_partitions_required_and_traversal_rejected() {
    let fixture = fixture(true);
    let report = inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["../train".into()],
        &["validation".into()],
        &[],
        &[],
    )
    .unwrap();
    assert!(!report.passed);
    for category in ["train", "base_validation", "sealed"] {
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.category == category && !check.passed)
        );
    }
}

fn release(root: &Path) {
    let train = conversation("Train source", "Train response");
    let validation = conversation("Heldout source", "Heldout response");
    put(root, "chat/train/data.jsonl", &train);
    put(root, "chat/validation/data.jsonl", &validation);
    put(root, "model.lock.toml", "# temporary fixture");
    let raw = serde_json::to_vec(
        &json!({"schema_version":1,"membership_sha256":sha256_bytes(b"unused"),
        "counts":{"chat/train":1,"chat/validation":1},
        "output_hashes":{"chat/train/data.jsonl":sha256_bytes(train.as_bytes()),
                         "chat/validation/data.jsonl":sha256_bytes(validation.as_bytes())}}),
    )
    .unwrap();
    fs::write(root.join("manifest.json"), &raw).unwrap();
    fs::write(
        root.join("COMPLETE"),
        format!("omega-datasets-v1\n{}\n", sha256_bytes(&raw)),
    )
    .unwrap();
}

#[test]
fn published_heldout_training_and_partition_role_confusion_are_rejected() {
    let fixture = fixture(true);
    release(&fixture.directory.path().join("release"));
    let report = inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["release/chat/validation".into()],
        &["validation".into()],
        &["base_validation".into()],
        &["sealed".into()],
    )
    .unwrap();
    assert!(report.errors.iter().any(|error| error.contains("held-out")));
    let report = inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["train".into()],
        &["release/chat/train".into()],
        &["base_validation".into()],
        &["sealed".into()],
    )
    .unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.contains("requires published validation"))
    );
}

#[test]
fn oversized_eager_chat_file_is_rejected_without_reading_it_all() {
    let fixture = fixture(true);
    fs::File::create(fixture.directory.path().join("train/data.jsonl"))
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    assert!(
        check(&fixture)
            .errors
            .iter()
            .any(|error| error.contains("16 MiB/file"))
    );
}

#[test]
fn active_input_verification_skips_sealed_payloads_and_rejects_failed_reports() {
    let fixture = fixture(true);
    let report = check(&fixture);
    assert!(report.passed, "{:?}", report.errors);
    verify_active_inputs(&report, fixture.directory.path()).unwrap();
    // Removing the sealed payload proves that the automatic phase check does
    // not reopen it, discover its inventory, tokenize it or repeat overlap work.
    fs::remove_file(fixture.directory.path().join("sealed/data.jsonl")).unwrap();
    verify_active_inputs(&report, fixture.directory.path()).unwrap();
    let mut failed = report.clone();
    failed.passed = false;
    assert!(verify_active_inputs(&failed, fixture.directory.path()).is_err());
    let mut missing_parent = report;
    missing_parent
        .identities
        .retain(|item| item.role != "parent/model.mpk");
    assert!(verify_active_inputs(&missing_parent, fixture.directory.path()).is_err());
}

#[test]
fn active_input_verification_detects_each_active_corpus_and_parent_mutation() {
    let fixture = fixture(true);
    let report = check(&fixture);
    assert!(report.passed, "{:?}", report.errors);
    for item in report
        .identities
        .iter()
        .filter(|item| item.role != "sealed")
    {
        let original = fs::read(&item.path).unwrap();
        let mut changed = original.clone();
        changed.push(b' ');
        fs::write(&item.path, changed).unwrap();
        let error = verify_active_inputs(&report, fixture.directory.path()).unwrap_err();
        assert!(error.contains("input changed"), "{}: {error}", item.role);
        fs::write(&item.path, original).unwrap();
    }
    verify_active_inputs(&report, fixture.directory.path()).unwrap();
}

#[test]
fn active_input_verification_rejects_added_removed_and_oversized_sources() {
    let fixture = fixture(true);
    let report = check(&fixture);
    assert!(report.passed, "{:?}", report.errors);
    for (role, relative) in [
        ("train", "train/nested/additional.jsonl"),
        ("validation", "validation/nested/additional.jsonl"),
        ("base_validation", "base_validation/nested/additional.txt"),
    ] {
        put(fixture.directory.path(), relative, "new source");
        let error = verify_active_inputs(&report, fixture.directory.path()).unwrap_err();
        assert!(
            error.contains(&format!("Frozen {role} file inventory changed")),
            "{error}"
        );
        fs::remove_file(fixture.directory.path().join(relative)).unwrap();
    }
    let path = fixture.directory.path().join("train/data.jsonl");
    let original = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert!(verify_active_inputs(&report, fixture.directory.path()).is_err());
    fs::File::create(&path)
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    assert!(
        verify_active_inputs(&report, fixture.directory.path())
            .unwrap_err()
            .contains("raw bytes")
    );
    fs::write(path, original).unwrap();
    verify_active_inputs(&report, fixture.directory.path()).unwrap();
}

#[test]
fn active_input_verification_freezes_release_metadata_even_when_new_marker_is_valid() {
    let fixture = fixture(true);
    let release_root = fixture.directory.path().join("release");
    release(&release_root);
    let report = inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["release/chat/train".into()],
        &["validation".into()],
        &["base_validation".into()],
        &["sealed".into()],
    )
    .unwrap();
    assert!(report.passed, "{:?}", report.errors);
    verify_active_inputs(&report, fixture.directory.path()).unwrap();
    let path = release_root.join("manifest.json");
    let mut changed = fs::read(&path).unwrap();
    changed.push(b'\n');
    fs::write(path, &changed).unwrap();
    fs::write(
        release_root.join("COMPLETE"),
        format!("omega-datasets-v1\n{}\n", sha256_bytes(&changed)),
    )
    .unwrap();
    let error = verify_active_inputs(&report, fixture.directory.path()).unwrap_err();
    assert!(
        error.contains("Frozen release_metadata input changed"),
        "{error}"
    );
}

#[test]
fn active_input_size_preflight_rejects_grown_published_payload_before_hash_validation() {
    let fixture = fixture(true);
    let release_root = fixture.directory.path().join("release");
    release(&release_root);
    let report = inspect(
        &fixture.parent,
        fixture.directory.path(),
        &["release/chat/train".into()],
        &["validation".into()],
        &["base_validation".into()],
        &["sealed".into()],
    )
    .unwrap();
    assert!(report.passed, "{:?}", report.errors);
    fs::OpenOptions::new()
        .write(true)
        .open(release_root.join("chat/train/data.jsonl"))
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    let error = verify_active_inputs(&report, fixture.directory.path()).unwrap_err();
    assert!(
        error.contains("raw bytes before release verification"),
        "{error}"
    );
}
