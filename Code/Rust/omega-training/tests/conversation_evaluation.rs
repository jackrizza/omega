use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use burn::{
    module::{Module, ModuleMapper, ParamId},
    tensor::{Tensor, TensorData},
};
use omega_nn::{Cpu, GptConfig};
use omega_tokenizer::{ByteBpeConfig, chat::ChatProtocol, train_chat_byte_bpe};
use omega_training::{
    conversation_evaluation::{
        CaseStatus, CompletionReason, EvaluationCase, EvaluationCheck, EvaluationReport,
        EvaluationStatus, EvaluationSuite, EvaluationTurn, GenerationSettings, GenerationStrategy,
        check_response, run_suite,
    },
    operations::{BackendChoice, OperationControl},
};

fn control() -> OperationControl {
    OperationControl {
        stop: Arc::new(AtomicBool::new(false)),
        observer: Box::new(|_| Ok(())),
    }
}

fn turn(prompt: &str, checks: Vec<EvaluationCheck>) -> EvaluationTurn {
    EvaluationTurn {
        prompt: prompt.into(),
        checks,
    }
}

fn suite() -> EvaluationSuite {
    EvaluationSuite {
        schema_version: 1,
        name: "tiny fixture only".into(),
        generation: GenerationSettings {
            max_new_tokens: 4,
            strategy: GenerationStrategy::Greedy,
        },
        cases: vec![EvaluationCase {
            id: "end".into(),
            system: Some("Be brief".into()),
            turns: vec![
                turn(
                    "café 世界",
                    vec![
                        EvaluationCheck::Exact {
                            expected: "".into(),
                        },
                        EvaluationCheck::EndTurn,
                        EvaluationCheck::NoRoleLeakage,
                    ],
                ),
                turn("Again?", vec![EvaluationCheck::EndTurn]),
            ],
        }],
    }
}

struct OutputToken {
    vocab: usize,
    token: usize,
}
impl ModuleMapper<Cpu> for OutputToken {
    fn map_float<const D: usize>(&mut self, _: ParamId, tensor: Tensor<Cpu, D>) -> Tensor<Cpu, D> {
        let dims = tensor.dims();
        let mut values = vec![if D == 1 { 1.0 } else { 0.0 }; dims.iter().product()];
        if dims.as_slice() == [4, self.vocab] {
            for row in 0..4 {
                values[row * self.vocab + self.token] = 1.0;
            }
        }
        Tensor::from_data(TensorData::new(values, dims), &tensor.device())
    }
}

fn checkpoint(root: &Path, role: bool) -> std::path::PathBuf {
    let tokenizer = train_chat_byte_bpe(
        &["hello"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    let config = GptConfig {
        vocab_size: tokenizer.vocab_size(),
        context_length: 64,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let token = if role {
        protocol.token_ids().user
    } else {
        protocol.token_ids().end_turn
    };
    let model = config
        .init::<Cpu>(&Default::default())
        .unwrap()
        .map(&mut OutputToken {
            vocab: config.vocab_size,
            token: token as usize,
        });
    omega_training::save_checkpoint(root, "fixture", model, &config, &tokenizer).unwrap()
}

#[test]
fn pure_checks_cover_exact_contains_json_stopping_empty_repetition_and_roles() {
    let checks = vec![
        EvaluationCheck::Exact {
            expected: "Hello".into(),
        },
        EvaluationCheck::Contains { text: "ell".into() },
        EvaluationCheck::EndTurn,
        EvaluationCheck::NoRoleLeakage,
        EvaluationCheck::NonEmpty,
        EvaluationCheck::NoRepetition {
            ngram_size: 1,
            max_occurrences: 1,
        },
    ];
    assert!(
        check_response(&checks, "Hello", CompletionReason::EndTurn, true, false)
            .iter()
            .all(|c| c.passed)
    );
    let failed = check_response(
        &checks,
        "hello hello",
        CompletionReason::TokenLimit,
        false,
        true,
    );
    assert!(!failed[0].passed);
    assert!(failed[1].passed);
    assert!(!failed[2].passed);
    assert!(!failed[3].passed);
    assert!(!failed[5].passed);
    let json = vec![EvaluationCheck::JsonFields {
        fields: BTreeMap::from([("answer".into(), serde_json::json!({"n":4}))]),
    }];
    assert!(
        check_response(
            &json,
            r#"{"answer":{"n":4},"other":true}"#,
            CompletionReason::EndTurn,
            true,
            false
        )[0]
        .passed
    );
    for invalid in ["invalid", "[]", r#"{"answer":{"n":5}}"#] {
        assert!(!check_response(&json, invalid, CompletionReason::EndTurn, true, false)[0].passed);
    }
    assert!(
        !check_response(
            &[EvaluationCheck::NonEmpty],
            " \n",
            CompletionReason::EndTurn,
            true,
            false
        )[0]
        .passed
    );
    let repeat = [EvaluationCheck::NoRepetition {
        ngram_size: 2,
        max_occurrences: 1,
    }];
    assert!(
        !check_response(
            &repeat,
            "a b a b",
            CompletionReason::TokenLimit,
            false,
            false
        )[0]
        .passed
    );
    assert!(
        check_response(
            &repeat,
            "a b a c",
            CompletionReason::TokenLimit,
            false,
            false
        )[0]
        .passed
    );
    assert!(
        check_response(
            &[EvaluationCheck::ContextOverflow],
            "",
            CompletionReason::ContextOverflow,
            false,
            false
        )[0]
        .passed
    );
    assert!(
        check_response(
            &checks,
            "Hello",
            CompletionReason::GenerationError,
            false,
            false
        )
        .iter()
        .all(|c| !c.passed)
    );
}

#[test]
fn suites_reject_unknown_versions_fields_and_invalid_bounds() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("suite.json");
    let valid = suite();
    fs::write(&path, serde_json::to_vec(&valid).unwrap()).unwrap();
    assert_eq!(EvaluationSuite::load(&path).unwrap(), valid);
    let mut unknown = serde_json::to_value(&valid).unwrap();
    unknown["unknown"] = true.into();
    fs::write(&path, serde_json::to_vec(&unknown).unwrap()).unwrap();
    assert!(EvaluationSuite::load(&path).is_err());
    let mut invalid = valid.clone();
    invalid.schema_version = 99;
    assert!(invalid.validate().is_err());
    let mut invalid = valid.clone();
    invalid.cases.push(invalid.cases[0].clone());
    assert!(invalid.validate().is_err());
    let mut invalid = valid.clone();
    invalid.cases[0].turns[0].prompt.clear();
    assert!(invalid.validate().is_err());
    let mut invalid = valid.clone();
    invalid.cases[0].turns[0].checks.clear();
    assert!(invalid.validate().is_err());
    let mut invalid = valid.clone();
    invalid.generation.max_new_tokens = 0;
    assert!(invalid.validate().is_err());
    let mut invalid = valid.clone();
    invalid.cases[0].turns[0].checks = vec![EvaluationCheck::ContextOverflow];
    assert!(invalid.validate().is_err());
    let mut invalid = valid.clone();
    invalid.generation.strategy = GenerationStrategy::Sample {
        temperature: 0.0,
        top_k: None,
        top_p: None,
        seed: 1,
    };
    assert!(invalid.validate().is_err());
    let mut invalid = valid;
    invalid.generation.max_new_tokens = 4096;
    invalid.cases[0].turns = vec![invalid.cases[0].turns[0].clone(); 9];
    assert!(invalid.validate().is_err());
    fs::write(&path, vec![b' '; 1024 * 1024 + 1]).unwrap();
    assert!(
        EvaluationSuite::load(&path)
            .unwrap_err()
            .contains("byte limit")
    );
}

#[test]
fn real_cpu_suite_retains_history_context_results_identity_and_exports() {
    let temp = tempfile::tempdir().unwrap();
    let path = checkpoint(temp.path(), false);
    let original = fs::read(path.join("model.mpk")).unwrap();
    let mut suite = suite();
    suite.cases.push(EvaluationCase {
        id: "context".into(),
        system: None,
        turns: vec![turn(
            &"x".repeat(70),
            vec![EvaluationCheck::ContextOverflow],
        )],
    });
    let report = run_suite(&path, &suite, BackendChoice::Cpu, 0, 30.0, &control()).unwrap();
    assert!(report.is_complete());
    assert!(report.all_passed());
    assert_eq!(report.pass_rate(), Some(1.0));
    assert_eq!(report.generated_tokens, 2);
    assert_eq!(report.passed_checks(), report.total_checks());
    assert_eq!(report.suite_hash, suite.fingerprint().unwrap());
    let second = &report.cases[0].turns[1];
    assert_eq!(second.history.len(), 4);
    assert_eq!(second.history[2].role, "assistant");
    assert_eq!(second.history[2].content, "");
    assert_eq!(second.history[3].content, "Again?");
    assert_eq!(second.completion_reason, CompletionReason::EndTurn);
    assert_eq!(
        report.cases[1].turns[0].completion_reason,
        CompletionReason::ContextOverflow
    );
    assert_eq!(report.runtime.backend, "cpu");
    assert_eq!(fs::read(path.join("model.mpk")).unwrap(), original);
    let rendered = report.to_markdown().unwrap();
    assert!(rendered.contains("café 世界"));
    assert!(rendered.contains("semantic correctness"));
    let report_path = temp.path().join("report.json");
    fs::write(&report_path, report.to_json().unwrap()).unwrap();
    assert_eq!(EvaluationReport::load(&report_path).unwrap(), report);
    let mut peer = report.clone();
    peer.checkpoint.model_sha256 = "a".repeat(64);
    report.compare_compatible(&peer).unwrap();
    peer.runtime.device = 5;
    assert!(report.compare_compatible(&peer).is_err());
    let mut corrupt = report.clone();
    corrupt.cases[0].turns[0].checks[0].passed = false;
    assert!(corrupt.validate().is_err());
    let mut sampled = suite;
    sampled.generation.strategy = GenerationStrategy::Sample {
        temperature: 1.0,
        top_k: Some(1),
        top_p: Some(1.0),
        seed: 7,
    };
    let sampled = run_suite(&path, &sampled, BackendChoice::Cpu, 0, 30.0, &control()).unwrap();
    assert!(sampled.all_passed());
    for (actual, expected) in sampled.cases.iter().zip(&report.cases) {
        for (actual, expected) in actual.turns.iter().zip(&expected.turns) {
            assert_eq!(actual.reply_ids, expected.reply_ids);
            assert_eq!(actual.history, expected.history);
            assert_eq!(actual.completion_reason, expected.completion_reason);
        }
    }
}

#[test]
fn failed_checks_and_role_rejection_are_completed_failures_not_passes() {
    let temp = tempfile::tempdir().unwrap();
    let path = checkpoint(temp.path(), true);
    let report = run_suite(&path, &suite(), BackendChoice::Cpu, 0, 30.0, &control()).unwrap();
    assert!(report.is_complete());
    assert!(!report.all_passed());
    assert_eq!(report.cases[0].status, CaseStatus::Failed);
    assert_eq!(report.cases[0].turns.len(), 1);
    let first = &report.cases[0].turns[0];
    assert_eq!(first.completion_reason, CompletionReason::RoleLeakage);
    assert!(first.raw_reply.contains("omega_v1_user"));
    assert!(first.reply.is_empty());
    assert!(report.pass_rate().unwrap() < 1.0);
}

#[test]
fn numerical_generation_failure_is_incomplete_and_preserves_failure() {
    struct NonFinite;
    impl ModuleMapper<Cpu> for NonFinite {
        fn map_float<const D: usize>(
            &mut self,
            _: ParamId,
            tensor: Tensor<Cpu, D>,
        ) -> Tensor<Cpu, D> {
            let dims = tensor.dims();
            Tensor::from_data(
                TensorData::new(vec![f32::NAN; dims.iter().product()], dims),
                &tensor.device(),
            )
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let original = checkpoint(temp.path(), false);
    let (model, config, tokenizer) = omega_training::load_checkpoint(&original).unwrap();
    let path = omega_training::save_checkpoint(
        temp.path(),
        "nonfinite",
        model.map(&mut NonFinite),
        &config,
        &tokenizer,
    )
    .unwrap();
    let report = run_suite(&path, &suite(), BackendChoice::Cpu, 0, 30.0, &control()).unwrap();
    assert_eq!(report.status, EvaluationStatus::GenerationError);
    assert!(!report.is_complete());
    assert!(!report.all_passed());
    assert_eq!(report.pass_rate(), None);
    assert_eq!(
        report.cases[0].turns[0].completion_reason,
        CompletionReason::GenerationError
    );
    assert!(report.error.as_ref().unwrap().contains("non-finite"));
    assert!(report.to_json().is_ok());
}

#[test]
fn interruption_timeout_and_observer_failure_preserve_partial_reports() {
    let temp = tempfile::tempdir().unwrap();
    let path = checkpoint(temp.path(), false);
    let mut observer = control();
    observer.stop.store(true, Ordering::Relaxed);
    let interrupted = run_suite(&path, &suite(), BackendChoice::Cpu, 0, 30.0, &observer).unwrap();
    assert_eq!(interrupted.status, EvaluationStatus::Interrupted);
    assert_eq!(interrupted.pass_rate(), None);
    assert_eq!(interrupted.generated_tokens, 0);
    assert!(!interrupted.is_complete());
    let timeout = run_suite(
        &path,
        &suite(),
        BackendChoice::Cpu,
        0,
        f64::MIN_POSITIVE,
        &control(),
    )
    .unwrap();
    assert_eq!(timeout.status, EvaluationStatus::TimedOut);
    assert!(timeout.budget_overrun_seconds > 0.0);
    observer.stop.store(false, Ordering::Relaxed);
    let stop = observer.stop.clone();
    observer.observer = Box::new(move |_| {
        stop.store(true, Ordering::Relaxed);
        Ok(())
    });
    let partial = run_suite(&path, &suite(), BackendChoice::Cpu, 0, 30.0, &observer).unwrap();
    assert_eq!(partial.status, EvaluationStatus::Interrupted);
    assert_eq!(partial.cases[0].turns.len(), 1);
    assert_eq!(partial.generated_tokens, 1);
    assert_eq!(partial.total_checks(), 4);
    assert!(partial.to_json().unwrap().contains("interrupted"));
    let mut observer = control();
    observer.observer = Box::new(|_| Err("fixture observer failure".into()));
    let failed = run_suite(&path, &suite(), BackendChoice::Cpu, 0, 30.0, &observer).unwrap();
    assert_eq!(failed.status, EvaluationStatus::ObserverError);
    assert_eq!(failed.cases[0].turns.len(), 1);
    assert_eq!(failed.pass_rate(), None);
    assert!(partial.compare_compatible(&failed).is_err());
    assert!(run_suite(&path, &suite(), BackendChoice::Cpu, 0, f64::NAN, &control()).is_err());
}
