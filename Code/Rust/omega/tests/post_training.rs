//! Offline workflow contracts only: these tiny fixtures prove no model quality.
use omega::{
    ProjectConfig,
    config::migrate_post_training,
    jobs::{self, Action, JobRecord, JobSpec, JobStatus, Reporter},
    post_training::{
        self, CandidateReport, CaseScore, HumanReview, PostTrainingConfig, WorkflowState,
    },
};
use omega_nn::GptConfig;
use omega_tokenizer::{ByteBpeConfig, train_chat_byte_bpe};
use omega_training::{
    CheckpointMetadata, TrainingSession, TrainingSet,
    checkpoint::sha256_bytes,
    conversation_evaluation::{
        EvaluationCase, EvaluationCheck, EvaluationSuite, EvaluationTurn, GenerationSettings,
        GenerationStrategy,
    },
    resume::save_training_checkpoint,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

struct Fixture {
    temp: tempfile::TempDir,
    config: ProjectConfig,
    parent: PathBuf,
}

fn put(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn chat(prompt: &str, answer: &str) -> String {
    json!({"schema_version":1,"messages":[
        {"role":"user","content":prompt},
        {"role":"assistant","content":answer}
    ]})
    .to_string()
}

fn fixture() -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tokenizer = train_chat_byte_bpe(
        &["fixture"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap();
    let model = GptConfig {
        vocab_size: tokenizer.vocab_size(),
        context_length: 32,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let mut session = TrainingSession::new(
        &model,
        TrainingSet {
            examples: vec![vec![4, 5, 6]],
            files: vec![],
            token_count: 3,
        },
        0.001,
        42,
    )
    .unwrap();
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
        "data/train/data.jsonl",
        &format!("{}\n{}\n", chat("a", "b"), chat("c", "d")),
    );
    put(root, "data/validation/data.jsonl", &chat("e", "f"));
    put(root, "data/sealed/data.jsonl", &chat("g", "h"));
    put(root, "data/base/text.txt", "ijklm");
    // An expected overflow checks the actual evaluator without relying on a
    // randomly initialized model to produce semantically correct responses.
    let suite = EvaluationSuite {
        schema_version: 1,
        name: "software overflow fixture".into(),
        generation: GenerationSettings {
            max_new_tokens: 1,
            strategy: GenerationStrategy::Greedy,
        },
        cases: ["overflow-a", "overflow-b"]
            .into_iter()
            .map(|id| EvaluationCase {
                id: id.into(),
                system: None,
                turns: vec![EvaluationTurn {
                    prompt: "x".repeat(40),
                    checks: vec![EvaluationCheck::ContextOverflow],
                }],
            })
            .collect(),
    };
    fs::write(
        root.join("suite.json"),
        serde_json::to_vec_pretty(&suite).unwrap(),
    )
    .unwrap();
    let mut config = ProjectConfig {
        omega_schema_version: 2,
        name: "fixture".into(),
        ..ProjectConfig::default()
    };
    config.paths.datasets = "data".into();
    config.paths.weights = "weights".into();
    config.post_training = Some(PostTrainingConfig {
        parent: parent.clone(),
        output: "workflow".into(),
        purpose: "Software fixture only".into(),
        language: "English".into(),
        training: vec!["train".into()],
        validation: vec!["validation".into()],
        base_validation: vec!["base".into()],
        sealed: vec!["sealed".into()],
        suite: "suite.json".into(),
        rubric_version: "fixture-v1".into(),
        epochs: 1,
        learning_rate: 0.001,
        seed: 42,
        batch_size: 1,
        max_batch_tokens: 32,
        segment_updates: 1,
        max_updates: 2,
        max_seconds: 120.0,
        evaluation_max_seconds: 30.0,
        max_evaluations: 3,
        max_output_bytes: 64 * 1024 * 1024,
        min_test_pass_rate: 1.0,
        max_assistant_loss: 100.0,
        max_base_loss_increase: 100.0,
        min_human_score: 3,
        frozen: None,
    });
    Fixture {
        temp,
        config,
        parent,
    }
}

impl Fixture {
    fn root(&self) -> &Path {
        self.temp.path()
    }
    fn output(&self) -> PathBuf {
        self.root().join("workflow")
    }
    fn spec(&self) -> JobSpec {
        let (config, plan) = post_training::prepare(&self.config, self.root()).unwrap();
        assert!(plan.contains("human review"));
        assert!(
            !self.output().exists(),
            "Planning must not create run output"
        );
        let executable = self.root().join("retained-worker.fixture");
        fs::write(&executable, b"offline retained executable identity fixture").unwrap();
        JobSpec {
            schema_version: 1,
            id: "fixture-job".into(),
            project: self.root().to_path_buf(),
            config_path: self.root().join("model.toml"),
            config,
            action: Action::PostTrain,
            checkpoint: None,
            prompt: None,
            executable: executable.clone(),
            executable_sha256: sha256_bytes(&fs::read(executable).unwrap()),
            created_ms: jobs::now_ms(),
        }
    }
    fn reporter(&self) -> Arc<Reporter> {
        let directory = self.root().join("job");
        fs::create_dir_all(&directory).unwrap();
        Arc::new(Reporter::new(
            directory,
            JobRecord {
                id: "fixture-job".into(),
                status: JobStatus::Running,
                process: None,
                project: self.root().to_path_buf(),
                stage: "fixture".into(),
                started_ms: jobs::now_ms(),
                updated_ms: jobs::now_ms(),
                progress: json!({}),
                total_updates: None,
                checkpoint: None,
                error: None,
                result: None,
            },
        ))
    }
    fn execute(&self, spec: &JobSpec, stop: bool) -> anyhow::Result<()> {
        post_training::execute(spec, self.reporter(), Arc::new(AtomicBool::new(stop)))
    }
}

fn phase(state: &WorkflowState) -> Value {
    serde_json::to_value(&state.phase).unwrap()
}

fn snapshot(directory: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn collect(root: &Path, path: &Path, result: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(root, &path, result);
            } else {
                result.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(path).unwrap(),
                );
            }
        }
    }
    let mut result = BTreeMap::new();
    collect(directory, directory, &mut result);
    result
}

fn review(report: &Path, score: u8) -> HumanReview {
    let data: CandidateReport = jobs::read_json(report).unwrap();
    HumanReview {
        schema_version: 1,
        report_sha256: sha256_bytes(&fs::read(report).unwrap()),
        rubric_version: "fixture-v1".into(),
        reviewer: "Fixture reviewer; not production evidence".into(),
        cases: data
            .conversation
            .suite
            .cases
            .iter()
            .map(|case| CaseScore {
                case_id: case.id.clone(),
                instruction_following: score,
                correctness: score,
                relevance: score,
                coherence: score,
                notes: "Synthetic fixture score only".into(),
            })
            .collect(),
    }
}

#[test]
fn schema_one_migration_preserves_comments_and_schema_two_round_trips() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.toml");
    let original =
        "# Keep this project note\nomega_schema_version = 1 # legacy\nname = \"legacy\"\n";
    fs::write(&path, original).unwrap();
    migrate_post_training(&path).unwrap();
    let changed = fs::read_to_string(&path).unwrap();
    assert!(changed.contains("# Keep this project note"));
    assert!(changed.contains("# legacy"));
    let migrated = ProjectConfig::parse(&changed).unwrap();
    assert_eq!(migrated.omega_schema_version, 2);
    assert!(migrated.post_training.is_none());
    assert!(migrate_post_training(&path).is_err());
    let f = fixture();
    let text = toml::to_string_pretty(&f.config).unwrap();
    let parsed = ProjectConfig::parse(&text).unwrap();
    assert_eq!(
        serde_json::to_value(&parsed).unwrap(),
        serde_json::to_value(&f.config).unwrap()
    );
    let spec = f.spec();
    let frozen = ProjectConfig::parse(&toml::to_string_pretty(&spec.config).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(frozen).unwrap(),
        serde_json::to_value(spec.config).unwrap()
    );
}

#[test]
fn two_segments_preserve_parent_and_require_complete_human_review_before_selection() {
    let f = fixture();
    let before = snapshot(&f.parent);
    let spec = f.spec();
    f.execute(&spec, false).unwrap();
    let state = WorkflowState::load(&f.output()).unwrap();
    assert_eq!(phase(&state), "completed");
    assert_eq!(
        (
            state.completed_updates,
            state.charged_updates,
            state.evaluations
        ),
        (2, 2, 3)
    );
    assert!(state.training_complete);
    assert_eq!(state.reports.len(), 3);
    assert!(state.incomplete_reports.is_empty());
    assert_eq!(snapshot(&f.parent), before);
    for (i, path) in state.reports.iter().enumerate() {
        let report: CandidateReport = jobs::read_json(path).unwrap();
        assert_eq!(report.baseline, i == 0);
        assert_eq!(report.completed_updates, i);
        assert!(report.complete && report.error.is_none());
        assert!(report.assistant_loss.unwrap().is_finite());
        assert!(report.base_loss.unwrap().is_finite());
        assert_eq!(report.conversation.pass_rate(), Some(1.0));
        assert!(path.with_extension("md").is_file());
    }
    assert!(
        post_training::rank(&f.output())
            .unwrap()
            .iter()
            .all(|candidate| !candidate.eligible)
    );
    let candidate = &state.reports[2];
    let selection = f.root().join("selected.json");
    assert!(post_training::promote(&f.output(), candidate, "Reviewer", &selection).is_err());
    assert!(!selection.exists());
    let mut partial = review(candidate, 4);
    partial.cases.pop();
    assert!(post_training::save_review(&f.output(), candidate, &partial).is_err());
    assert!(!candidate.with_extension("review.json").exists());
    let mut bad_hash = review(candidate, 4);
    bad_hash.report_sha256 = "0".repeat(64);
    assert!(post_training::save_review(&f.output(), candidate, &bad_hash).is_err());
    post_training::save_review(
        &f.output(),
        &state.reports[1],
        &review(&state.reports[1], 2),
    )
    .unwrap();
    post_training::save_review(&f.output(), candidate, &review(candidate, 4)).unwrap();
    assert!(post_training::save_review(&f.output(), candidate, &review(candidate, 4)).is_err());
    let ranked = post_training::rank(&f.output()).unwrap();
    assert_eq!(ranked.iter().filter(|c| c.eligible).count(), 1);
    assert_eq!(&ranked[0].report, candidate);
    assert!(ranked.iter().any(|c| {
        c.reasons
            .iter()
            .any(|reason| reason.contains("Human quality"))
    }));
    assert!(post_training::promote(&f.output(), candidate, " ", &selection).is_err());
    let selected = post_training::promote(
        &f.output(),
        candidate,
        "Explicit fixture approver",
        &selection,
    )
    .unwrap();
    assert!(selected.final_acceptance.contains("Pending"));
    assert_eq!(
        fs::canonicalize(&selected.checkpoint).unwrap(),
        fs::canonicalize(state.checkpoint.unwrap()).unwrap()
    );
    let selection_bytes = fs::read(&selection).unwrap();
    assert!(post_training::promote(&f.output(), candidate, "Someone else", &selection).is_err());
    assert_eq!(fs::read(selection).unwrap(), selection_bytes);
    assert_eq!(snapshot(&f.parent), before);
    assert!(post_training::recover_spec(&f.output()).is_err());
    let candidate_checkpoint = &selected.checkpoint;
    let model_path = candidate_checkpoint.join("model.mpk");
    let model_bytes = fs::read(&model_path).unwrap();
    fs::write(&model_path, b"corrupted fixture payload").unwrap();
    assert!(
        post_training::promote(
            &f.output(),
            candidate,
            "Reviewer",
            &f.root().join("corrupt-selection.json")
        )
        .is_err()
    );
    fs::write(&model_path, model_bytes).unwrap();
    fs::write(candidate, b"{}").unwrap();
    assert!(WorkflowState::load(&f.output()).is_err());
    assert!(post_training::rank(&f.output()).is_err());
}

#[test]
fn positive_human_scores_do_not_override_frozen_metric_thresholds() {
    let mut f = fixture();
    f.config.post_training.as_mut().unwrap().max_assistant_loss = f64::MIN_POSITIVE;
    let spec = f.spec();
    f.execute(&spec, false).unwrap();
    let state = WorkflowState::load(&f.output()).unwrap();
    for report in &state.reports[1..] {
        post_training::save_review(&f.output(), report, &review(report, 4)).unwrap();
    }
    let ranked = post_training::rank(&f.output()).unwrap();
    assert!(ranked.iter().all(|candidate| !candidate.eligible));
    assert!(ranked.iter().all(|candidate| {
        candidate
            .reasons
            .iter()
            .any(|reason| reason.contains("Assistant loss"))
    }));
    assert!(
        post_training::promote(
            &f.output(),
            &state.reports[1],
            "Reviewer",
            &f.root().join("selection.json")
        )
        .is_err()
    );
}

#[test]
fn approved_data_and_suite_changes_are_rejected_before_output_creation() {
    for change_suite in [false, true] {
        let f = fixture();
        let spec = f.spec();
        if change_suite {
            let path = f.root().join("suite.json");
            let mut suite: Value = jobs::read_json(&path).unwrap();
            suite["name"] = json!("Changed after plan");
            fs::write(path, serde_json::to_vec(&suite).unwrap()).unwrap();
        } else {
            put(
                f.root(),
                "data/train/data.jsonl",
                &chat("new prompt", "new answer"),
            );
        }
        let error = f.execute(&spec, false).unwrap_err().to_string();
        assert!(error.contains("Approved inputs changed"), "{error}");
        assert!(!f.output().exists());
    }
}

#[test]
fn explicit_stop_preserves_recovery_identity_and_resumes_with_original_budget() {
    let f = fixture();
    let spec = f.spec();
    f.execute(&spec, true).unwrap();
    let state = WorkflowState::load(&f.output()).unwrap();
    assert_eq!(phase(&state), "stopped");
    assert_eq!(state.completed_updates, 0);
    assert!(state.reports.is_empty());
    let mut recovered = post_training::recover_spec(&f.output()).unwrap();
    let executable_bytes = fs::read(&spec.executable).unwrap();
    fs::write(&spec.executable, b"changed").unwrap();
    assert!(
        post_training::recover_spec(&f.output())
            .unwrap_err()
            .to_string()
            .contains("executable")
    );
    fs::write(&spec.executable, executable_bytes).unwrap();
    let data_path = f.root().join("data/train/data.jsonl");
    let original_data = fs::read(&data_path).unwrap();
    fs::write(&data_path, chat("changed prompt", "changed answer")).unwrap();
    assert!(
        post_training::recover_spec(&f.output())
            .unwrap_err()
            .to_string()
            .contains("Frozen")
    );
    fs::write(&data_path, original_data).unwrap();
    recovered.action = Action::PostTrainRecover;
    recovered.checkpoint = Some(f.output());
    f.execute(&recovered, false).unwrap();
    let finished = WorkflowState::load(&f.output()).unwrap();
    assert_eq!(phase(&finished), "completed");
    assert_eq!(finished.completed_updates, 2);
    assert!(finished.elapsed_seconds >= state.elapsed_seconds);
}

#[test]
fn recovery_after_final_checkpoint_receipt_evaluates_without_another_segment() {
    let mut f = fixture();
    // Spare update capacity ensures recovery is stopped by epoch completion,
    // rather than accidentally appearing correct due to an exhausted budget.
    f.config.post_training.as_mut().unwrap().max_updates = 3;
    let spec = f.spec();
    f.execute(&spec, false).unwrap();
    let mut receipt = WorkflowState::load(&f.output()).unwrap();
    assert!(receipt.training_complete);
    let checkpoint = receipt.checkpoint.clone().unwrap();
    let checkpoint_bytes = snapshot(&f.output().join("checkpoints"));
    // Reconstruct the durable final save receipt at the crash boundary before
    // the final evaluation. Only this test's temporary report files are removed.
    let final_report = receipt.reports.pop().unwrap();
    receipt.report_hashes.remove(&final_report);
    fs::remove_file(final_report.with_extension("md")).unwrap();
    fs::remove_file(final_report).unwrap();
    receipt.evaluations -= 1;
    receipt.phase = post_training::Phase::Training;
    receipt.pending_evaluation = true;
    receipt.active_since_ms = Some(jobs::now_ms().saturating_sub(50));
    jobs::atomic_json(&f.output().join("workflow.json"), &receipt).unwrap();
    let mut recovery = post_training::recover_spec(&f.output()).unwrap();
    recovery.action = Action::PostTrainRecover;
    recovery.checkpoint = Some(f.output());
    f.execute(&recovery, false).unwrap();
    let finished = WorkflowState::load(&f.output()).unwrap();
    assert_eq!(phase(&finished), "completed");
    assert_eq!(finished.completed_updates, 2);
    assert_eq!(finished.charged_updates, 2);
    assert_eq!(finished.checkpoint.as_ref(), Some(&checkpoint));
    assert_eq!((finished.evaluations, finished.reports.len()), (3, 3));
    assert!(!finished.pending_evaluation);
    assert!(finished.elapsed_seconds >= receipt.elapsed_seconds + 0.05);
    assert_eq!(snapshot(&f.output().join("checkpoints")), checkpoint_bytes);
}

#[test]
fn update_evaluation_storage_and_time_caps_stop_without_promoting() {
    for cap in ["updates", "evaluations", "storage", "time"] {
        let mut f = fixture();
        let p = f.config.post_training.as_mut().unwrap();
        match cap {
            "updates" => p.max_updates = 1,
            "evaluations" => p.max_evaluations = 2,
            "storage" => p.max_output_bytes = 1,
            "time" => {
                p.max_seconds = f64::MIN_POSITIVE;
                p.evaluation_max_seconds = f64::MIN_POSITIVE;
            }
            _ => unreachable!(),
        }
        let spec = f.spec();
        f.execute(&spec, false).unwrap();
        let state = WorkflowState::load(&f.output()).unwrap();
        assert_eq!(
            phase(&state),
            "budget_exhausted",
            "{cap}: {}",
            state.message
        );
        assert!(!state.training_complete);
        let expected_updates = usize::from(cap == "updates" || cap == "evaluations");
        assert_eq!(state.completed_updates, expected_updates, "{cap}");
        assert!(state.charged_updates <= state.settings().unwrap().max_updates);
        assert!(state.evaluations <= state.settings().unwrap().max_evaluations);
        assert!(post_training::recover_spec(&f.output()).is_err());
        assert!(!f.output().join("selected.json").exists());
    }
}

#[test]
fn stopped_workflow_cannot_expand_its_immutable_approved_budget() {
    let f = fixture();
    let spec = f.spec();
    f.execute(&spec, true).unwrap();
    WorkflowState::load(&f.output()).unwrap();
    let approval_path = f.output().join("approval.json");
    let approval = fs::read(&approval_path).unwrap();
    let state_path = f.output().join("workflow.json");
    let mut changed: Value = jobs::read_json(&state_path).unwrap();
    changed["spec"]["config"]["post_training"]["max_updates"] = json!(3);
    jobs::atomic_json(&state_path, &changed).unwrap();
    let error = WorkflowState::load(&f.output()).unwrap_err().to_string();
    assert!(error.to_lowercase().contains("approval"), "{error}");
    assert!(post_training::recover_spec(&f.output()).is_err());
    assert_eq!(fs::read(approval_path).unwrap(), approval);
}
