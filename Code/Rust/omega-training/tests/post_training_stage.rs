//! Bounded CPU software acceptance; no production corpus or device qualification.
use burn::{
    optim::{Adam, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use clap::Parser;
use omega_nn::{Gpt, GptConfig, TrainingBackend};
use omega_tokenizer::{ByteBpeConfig, Tokens, train_chat_byte_bpe};
use omega_training::{
    TrainingSession, TrainingSet, ValidationSplit,
    assistant::{ConversationSet, prepare_conversations},
    checkpoint::{CheckpointMetadata, sha256_bytes},
    operations::{
        Args, OperationControl, OperationStopReason, execute, execute_evaluation, execute_segment,
        execute_segment_to,
    },
    resume::{
        initialize_assistant_stage_on_device, initialize_assistant_stage_transfer_on_device,
        load_training_checkpoint, read_resume_manifest, read_stage_parent,
        save_training_checkpoint, save_training_checkpoint_with_parent,
    },
    trainer::SessionOptions,
};
use std::{
    fs,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

type AdamRecord = <OptimizerAdaptor<Adam, Gpt<TrainingBackend>, TrainingBackend> as Optimizer<
    Gpt<TrainingBackend>,
    TrainingBackend,
>>::Record;

fn write_resume(path: &Path, value: &serde_json::Value) {
    let bytes = serde_json::to_vec(value).unwrap();
    fs::write(path.join("resume.sha256"), sha256_bytes(&bytes)).unwrap();
    fs::write(path.join("resume.json"), bytes).unwrap();
}

fn args(root: &Path, command: &str, name: &str, checkpoint: &Path, limit: usize) -> Args {
    let mut argv = vec![
        "omega-training".to_string(),
        command.into(),
        "--datasets-root".into(),
        root.display().to_string(),
        "--weights-root".into(),
        root.display().to_string(),
        "--checkpoint".into(),
        checkpoint.file_name().unwrap().to_str().unwrap().into(),
        "--name".into(),
        name.into(),
        "--epochs".into(),
        "2".into(),
        "--max-updates".into(),
        limit.to_string(),
        "--quiet".into(),
    ];
    if command == "train-stage" {
        argv.extend(
            [
                "--dataset",
                "chat",
                "--seed",
                "7",
                "--learning-rate",
                "0.0001",
                "--shuffle",
            ]
            .map(str::to_owned),
        );
    }
    Args::try_parse_from(argv).unwrap()
}

fn control(stop: bool) -> OperationControl {
    OperationControl {
        stop: Arc::new(AtomicBool::new(stop)),
        observer: Box::new(|_| Ok(())),
    }
}

fn evaluation_args(root: &Path, checkpoint: &Path) -> Args {
    Args::try_parse_from([
        "omega-training",
        "evaluate",
        "--datasets-root",
        root.to_str().unwrap(),
        "--weights-root",
        checkpoint.parent().unwrap().to_str().unwrap(),
        "--checkpoint",
        checkpoint.file_name().unwrap().to_str().unwrap(),
        "--dataset",
        "chat",
        "--dataset-format",
        "chat",
    ])
    .unwrap()
}

fn transfer(
    path: &Path,
    source: ConversationSet,
) -> Result<
    (
        TrainingSession<ConversationSet>,
        Tokens,
        omega_training::resume::ParentCheckpoint,
    ),
    String,
> {
    initialize_assistant_stage_transfer_on_device::<_, TrainingBackend>(
        path,
        source,
        0.0001,
        7,
        SessionOptions::default(),
        &Default::default(),
    )
}

// Keep all seeded model creation/comparisons in one test: Burn's RNG is global.
#[test]
fn transfer_integrity_fresh_state_and_segment_continuation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tokenizer = train_chat_byte_bpe(
        &["Hi ok yes"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap();
    fs::create_dir(root.join("chat")).unwrap();
    let conversation = |answer| {
        serde_json::json!({"schema_version":1,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":answer}]}).to_string()
    };
    fs::write(
        root.join("chat/data.jsonl"),
        format!("{}\n{}\n", conversation("ok"), conversation("yes")),
    )
    .unwrap();
    let config = GptConfig {
        vocab_size: tokenizer.vocab_size(),
        context_length: 32,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let base_source = || TrainingSet {
        examples: vec![vec![1, 2, 3, 4]],
        files: vec![],
        token_count: 4,
    };
    let mut base = TrainingSession::new(&config, base_source(), 0.001, 42).unwrap();
    base.step().unwrap();
    let parent = save_training_checkpoint(
        root,
        "parent",
        &base,
        &tokenizer,
        &CheckpointMetadata::default(),
    )
    .unwrap();
    let source = prepare_conversations(
        root,
        &["chat".into()],
        &tokenizer,
        32,
        ValidationSplit::None,
        42,
    )
    .unwrap()
    .training;

    // A changed historical application build/platform/thread count is acceptable
    // only to explicit weights transfer, never exact resume or the legacy API.
    let mut historical: serde_json::Value =
        serde_json::from_slice(&fs::read(parent.join("resume.json")).unwrap()).unwrap();
    historical["runtime"]["build"]["revision"] = "historical-application-build".into();
    historical["runtime"]["operating_system"] = "historical-platform".into();
    historical["cpu_execution"]["rayon_num_threads"] = "2".into();
    write_resume(&parent, &historical);
    assert!(read_resume_manifest(&parent).is_err());
    assert!(load_training_checkpoint(&parent, base_source(), &config, &tokenizer).is_err());
    assert!(
        initialize_assistant_stage_on_device::<_, TrainingBackend>(
            &parent,
            source.clone(),
            0.0001,
            7,
            SessionOptions::default(),
            &Default::default()
        )
        .is_err()
    );
    let parent_snapshot: Vec<_> = [
        "COMPLETE",
        "manifest.json",
        "config.json",
        "tokenizer.json",
        "resume.json",
        "resume.sha256",
        "model.mpk",
        "optimizer.mpk",
    ]
    .into_iter()
    .map(|name| (name, fs::read(parent.join(name)).unwrap()))
    .collect();
    read_stage_parent(&parent).unwrap();
    let (stage, stage_tokenizer, lineage) = transfer(&parent, source.clone()).unwrap();
    assert_eq!(stage.progress().completed_updates, 0);
    assert_eq!(stage.progress().completed_epochs, 0);
    assert_eq!(stage.progress().next_example_index, 0);
    let zero = save_training_checkpoint_with_parent(
        root,
        "fresh",
        &stage,
        &stage_tokenizer,
        &CheckpointMetadata::default(),
        Some(&lineage),
    )
    .unwrap();
    assert_eq!(
        fs::read(parent.join("model.mpk")).unwrap(),
        fs::read(zero.join("model.mpk")).unwrap()
    );
    let optimizer: AdamRecord = <_ as Recorder<TrainingBackend>>::load(
        &NamedMpkBytesRecorder::<FullPrecisionSettings>::default(),
        fs::read(zero.join("optimizer.mpk")).unwrap(),
        &Default::default(),
    )
    .unwrap();
    assert!(
        optimizer.is_empty(),
        "Fresh Adam has no inherited moments or steps"
    );
    assert_eq!(read_resume_manifest(&zero).unwrap().parent, Some(lineage));

    for name in [
        "COMPLETE",
        "tokenizer.json",
        "config.json",
        "model.mpk",
        "optimizer.mpk",
        "resume.sha256",
    ] {
        let original = fs::read(parent.join(name)).unwrap();
        fs::write(parent.join(name), b"corrupt").unwrap();
        assert!(
            read_stage_parent(&parent).is_err(),
            "accepted corrupt {name}"
        );
        fs::write(parent.join(name), original).unwrap();
    }
    for field in ["tokenizer", "model", "schema"] {
        let mut changed = historical.clone();
        match field {
            "tokenizer" => changed["tokenizer"]["sha256"] = "0".repeat(64).into(),
            "model" => changed["model"]["d_ff"] = 16.into(),
            _ => changed["schema_version"] = 2.into(),
        }
        write_resume(&parent, &changed);
        assert!(
            read_stage_parent(&parent).is_err(),
            "accepted mismatched {field}"
        );
    }
    write_resume(&parent, &historical);

    let normal = control(false);
    assert!(
        execute(
            args(root, "train-stage", "legacy", &parent, 1),
            Some(&normal)
        )
        .is_err()
    );
    let whole = execute_segment(
        args(root, "train-stage", "whole", &parent, 4),
        &normal,
        120.0,
    )
    .unwrap();
    assert_eq!(whole.reason, OperationStopReason::Completed);
    assert_eq!((whole.completed_updates, whole.total_updates), (4, 4));
    let output_root = root.join("workflow-output/checkpoints");
    let first = execute_segment_to(
        args(root, "train-stage", "split", &parent, 1),
        &normal,
        120.0,
        &output_root,
    )
    .unwrap();
    assert_eq!(first.reason, OperationStopReason::SegmentLimit);
    assert_eq!(first.completed_updates, 1);
    assert_eq!(first.checkpoint.parent(), Some(output_root.as_path()));
    assert!(!root.join("split-1").exists());
    assert!(
        !normal.stop.load(Ordering::SeqCst),
        "Segment limit is not a user stop"
    );
    let mut resume_args = args(root, "resume", "split", &first.checkpoint, 3);
    if let omega_training::operations::Command::Resume { weights_root, .. } =
        &mut resume_args.command
    {
        *weights_root = Some(output_root.clone());
    }
    let resumed = execute_segment_to(resume_args, &normal, 120.0, &output_root).unwrap();
    assert_eq!(resumed.reason, OperationStopReason::Completed);
    assert_eq!(
        read_resume_manifest(&resumed.checkpoint).unwrap().progress,
        read_resume_manifest(&whole.checkpoint).unwrap().progress
    );
    assert_eq!(
        fs::read(resumed.checkpoint.join("model.mpk")).unwrap(),
        fs::read(whole.checkpoint.join("model.mpk")).unwrap(),
        "Segmenting must retain exact Adam and sampler continuation"
    );

    let stopped = control(true);
    let stop_result = execute_segment(
        args(root, "train-stage", "stopped", &parent, 4),
        &stopped,
        120.0,
    )
    .unwrap();
    assert_eq!(stop_result.reason, OperationStopReason::UserStop);
    assert_eq!(stop_result.completed_updates, 0);
    assert!(stop_result.checkpoint.join("COMPLETE").is_file());
    assert!(stopped.stop.load(Ordering::SeqCst));
    let timed = execute_segment(
        args(root, "train-stage", "timed", &parent, 4),
        &normal,
        f64::MIN_POSITIVE,
    )
    .unwrap();
    assert_eq!(timed.reason, OperationStopReason::TimeBudget);
    assert_eq!(timed.completed_updates, 0);
    assert!(timed.budget_overrun_seconds > 0.0);
    assert!(timed.checkpoint.join("COMPLETE").is_file());
    assert!(!normal.stop.load(Ordering::SeqCst));
    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            execute_segment(
                args(root, "train-stage", "invalid", &parent, 1),
                &normal,
                seconds
            )
            .is_err()
        );
    }
    // Observer errors propagate as failures, never as a fabricated successful outcome.
    let failure = OperationControl {
        stop: Arc::new(AtomicBool::new(false)),
        observer: Box::new(|event| {
            if event.kind == "checkpoint" {
                Err("observer unavailable".into())
            } else {
                Ok(())
            }
        }),
    };
    assert!(
        execute_segment(
            args(root, "train-stage", "observer-failure", &parent, 1),
            &failure,
            120.0
        )
        .unwrap_err()
        .contains("observer unavailable")
    );

    let metrics = Arc::new(Mutex::new(Vec::new()));
    let captured = metrics.clone();
    let evaluator = OperationControl {
        stop: Arc::new(AtomicBool::new(false)),
        observer: Box::new(move |event| {
            captured.lock().unwrap().push(event);
            Ok(())
        }),
    };
    execute_evaluation(evaluation_args(root, &whole.checkpoint), &evaluator, 120.0).unwrap();
    let complete = metrics.lock().unwrap().pop().unwrap();
    assert_eq!(complete.kind, "evaluation");
    assert!(complete.data["cross_entropy"].as_f64().unwrap().is_finite());
    assert!(metrics.lock().unwrap().is_empty());
    evaluator.stop.store(true, Ordering::SeqCst);
    assert!(
        execute_evaluation(evaluation_args(root, &whole.checkpoint), &evaluator, 120.0)
            .unwrap_err()
            .contains("stopped by user")
    );
    evaluator.stop.store(false, Ordering::SeqCst);
    assert!(
        execute_evaluation(
            evaluation_args(root, &whole.checkpoint),
            &evaluator,
            f64::MIN_POSITIVE
        )
        .unwrap_err()
        .contains("time budget")
    );
    assert!(
        metrics.lock().unwrap().is_empty(),
        "Incomplete evaluation must not emit metrics"
    );
    let dispatch_stop = Arc::new(AtomicBool::new(false));
    let signal = dispatch_stop.clone();
    let stop_after_dispatch = OperationControl {
        stop: dispatch_stop,
        observer: Box::new(move |_| {
            signal.store(true, Ordering::SeqCst);
            Ok(())
        }),
    };
    assert!(
        execute_evaluation(
            evaluation_args(root, &whole.checkpoint),
            &stop_after_dispatch,
            120.0
        )
        .unwrap_err()
        .contains("stopped by user")
    );
    for (name, bytes) in parent_snapshot {
        assert_eq!(
            fs::read(parent.join(name)).unwrap(),
            bytes,
            "modified parent {name}"
        );
    }
}
