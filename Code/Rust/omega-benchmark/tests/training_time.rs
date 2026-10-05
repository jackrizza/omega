use std::{fs, path::Path, process::Command};

use omega_benchmark::{
    Backend, DatasetFormat, DatasetOptions, ModelDimensions, TimedSample, TrainingTimeConfig,
    project_seconds, sampling_plan, training_time,
};
use omega_tokenizer::{ByteBpeConfig, Tokens, train_chat_byte_bpe};
use omega_training::{
    ValidationSplit,
    batching::BatchConfig,
    cache::{CacheOptions, create_token_cache},
    checkpoint::sha256_bytes,
};
use serde_json::{Value, json};

fn fixture() -> (tempfile::TempDir, TrainingTimeConfig) {
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir(temp.path().join("corpus")).unwrap();
    fs::write(
        temp.path().join("tokenizer.json"),
        include_bytes!("../../../../datasets/test.json"),
    )
    .unwrap();
    // 3 examples: lengths 4, 2, 1 input positions; 7 total targets, no cross-file pair.
    fs::write(
        temp.path().join("corpus/a.txt"),
        "hello world this is a test omega",
    )
    .unwrap();
    fs::write(temp.path().join("corpus/b.txt"), "hello world").unwrap();
    let config = TrainingTimeConfig {
        dataset: DatasetOptions {
            root: temp.path().into(),
            selections: vec!["corpus".into()],
            tokenizer: "tokenizer.json".into(),
            ..Default::default()
        },
        model: ModelDimensions {
            context_length: 4,
            d_model: 4,
            heads: 1,
            layers: 1,
            d_ff: 8,
        },
        batching: BatchConfig {
            batch_size: 2,
            max_batch_tokens: 8,
        },
        epochs: 3,
        warmup: 1,
        samples: 2,
        max_seconds: 60.0,
        ..Default::default()
    };
    (temp, config)
}

#[test]
fn plans_cover_intervals_and_projection_weights_different_costs() {
    let plan = sampling_plan(7, 3).unwrap();
    assert_eq!(
        plan.iter().map(|p| p.batch_index).collect::<Vec<_>>(),
        [1, 3, 5]
    );
    assert_eq!(
        plan.iter()
            .map(|p| p.represented_batches)
            .collect::<Vec<_>>(),
        [2, 2, 3]
    );
    let mut samples: Vec<_> = plan
        .into_iter()
        .zip([1.0, 2.0, 10.0])
        .map(|(point, seconds)| TimedSample {
            point,
            seconds,
            examples: 2,
            supervised_targets: 8,
        })
        .collect();
    assert_eq!(project_seconds(7, 2, &samples).unwrap(), 72.0);
    assert!(project_seconds(7, 2, &samples[..2]).is_err());
    assert!(project_seconds(7, 0, &samples).is_err());
    samples[0].seconds = f64::NAN;
    assert!(project_seconds(7, 2, &samples).is_err());
    samples[0].seconds = 0.0;
    assert!(project_seconds(7, 2, &samples).is_err());
    samples[0].seconds = f64::MAX;
    assert!(project_seconds(7, 2, &samples).is_err());
    let point = sampling_plan(1, 2).unwrap().remove(0);
    let sample = TimedSample {
        point,
        seconds: 1.0,
        examples: 1,
        supervised_targets: 1,
    };
    assert!(project_seconds(1, 1, &[sample.clone(), sample]).is_err());
    for (batches, requested) in [(1, 12), (17, 7), (usize::MAX, 1000)] {
        let points = sampling_plan(batches, requested).unwrap();
        assert_eq!(
            points.iter().map(|p| p.represented_batches).sum::<usize>(),
            batches
        );
        assert!(
            points
                .iter()
                .all(|p| p.batch_index < batches && p.represented_batches > 0)
        );
        assert!(
            points
                .windows(2)
                .all(|p| p[0].batch_index < p[1].batch_index)
        );
    }
    assert!(sampling_plan(0, 1).is_err());
    assert!(sampling_plan(1, 0).is_err());
    assert!(sampling_plan(1, 1001).is_err());
}

#[test]
fn real_cpu_updates_count_padding_tail_and_all_epochs_without_writes() {
    let (temp, config) = fixture();
    let before = fs::read(temp.path().join("corpus/a.txt")).unwrap();
    let report = training_time(&config).unwrap();
    assert_eq!(report.examples_per_epoch, 3);
    assert_eq!(report.supervised_targets_per_epoch, 7);
    assert_eq!(report.input_positions_per_epoch, 7);
    assert_eq!(report.padded_positions_per_epoch, 9);
    assert_eq!(report.batches_per_epoch, 2);
    assert_eq!(report.total_updates, 6);
    assert_eq!(
        report
            .samples
            .iter()
            .map(|s| (s.examples, s.supervised_targets))
            .collect::<Vec<_>>(),
        [(2, 6), (1, 1)]
    );
    assert!(
        (report.estimated_training_seconds - report.estimated_epoch_seconds * 3.0).abs() < 1e-9
    );
    assert!(report.measured_targets_per_second > 0.0);
    assert_eq!(fs::read(temp.path().join("corpus/a.txt")).unwrap(), before);
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
    assert!(
        serde_json::to_value(&report).unwrap()["estimated_training_seconds"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn smaller_than_one_batch_cycles_warmup_without_changing_sample_shape() {
    let (temp, mut config) = fixture();
    config.samples = 1;
    let tail = training_time(&config).unwrap();
    assert_eq!(tail.samples[0].point.batch_index, 1);
    assert_eq!(tail.samples[0].examples, 1);
    assert_eq!(tail.samples[0].supervised_targets, 1);
    fs::remove_file(temp.path().join("corpus/a.txt")).unwrap();
    config.warmup = 3;
    config.samples = 10;
    let report = training_time(&config).unwrap();
    assert_eq!(report.samples.len(), 1);
    assert_eq!(report.samples[0].examples, 1);
    assert_eq!(report.samples[0].supervised_targets, 1);
    assert_eq!(report.warmup_updates, 3);
}

#[test]
fn cached_partition_matches_eager_counts_and_rejects_changed_source() {
    let (temp, mut config) = fixture();
    config.dataset.validation = ValidationSplit::Count(1);
    let eager = training_time(&config).unwrap();
    let tokenizer = Tokens::new(temp.path().join("tokenizer.json")).unwrap();
    let path = temp.path().join("cache");
    create_token_cache(
        &path,
        temp.path(),
        &config.dataset.selections,
        &tokenizer,
        &CacheOptions {
            format: omega_training::dataset::DatasetFormat::Text,
            context_length: 4,
            validation: config.dataset.validation,
            split_seed: config.dataset.split_seed,
            limits: config.dataset.cache_limits.clone(),
        },
    )
    .unwrap();
    config.dataset.cache = Some(path);
    let cached = training_time(&config).unwrap();
    assert!(cached.cache_used);
    assert_eq!(cached.examples_per_epoch, eager.examples_per_epoch);
    assert_eq!(
        cached.supervised_targets_per_epoch,
        eager.supervised_targets_per_epoch
    );
    fs::write(temp.path().join("corpus/b.txt"), "hello world omega").unwrap();
    assert!(training_time(&config).is_err());
}

fn release(root: &Path) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("model.lock.toml"), "# test recipe").unwrap();
    let mut counts = serde_json::Map::new();
    let mut hashes = serde_json::Map::new();
    for stage in ["base", "chat"] {
        for partition in ["train", "validation", "test"] {
            let key = format!("{stage}/{partition}");
            fs::create_dir_all(root.join(&key)).unwrap();
            let data = if stage == "base" { json!({"text":"hello world omega"}) }
                else { json!({"schema_version":1,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"ok"}]}) }.to_string();
            fs::write(root.join(&key).join("data.jsonl"), &data).unwrap();
            counts.insert(key.clone(), json!(1));
            hashes.insert(
                format!("{key}/data.jsonl"),
                json!(sha256_bytes(data.as_bytes())),
            );
        }
    }
    let manifest = serde_json::to_vec(&json!({"schema_version":1,"counts":counts,"output_hashes":hashes,"membership_sha256":sha256_bytes(b"fixture"),"tool_version":"fixture","config":{}})).unwrap();
    fs::write(root.join("manifest.json"), &manifest).unwrap();
    fs::write(
        root.join("COMPLETE"),
        format!("omega-datasets-v1\n{}\n", sha256_bytes(&manifest)),
    )
    .unwrap();
}

#[test]
fn published_base_and_chat_infer_format_preserve_masks_and_guard_heldout_data() {
    let (temp, mut config) = fixture();
    release(&temp.path().join("release"));
    config.dataset.selections = vec!["release/base/train".into()];
    let base = training_time(&config).unwrap();
    assert_eq!(base.dataset_format, "jsonl");
    assert_eq!(base.supervised_targets_per_epoch, 2);
    for selection in [
        "release",
        "release/base",
        "release/base/validation",
        "release/chat/test",
    ] {
        config.dataset.selections = vec![selection.into()];
        assert!(training_time(&config).is_err(), "{selection}");
    }
    config.dataset.selections = vec!["release/base/train".into()];
    config.dataset.format = DatasetFormat::Text;
    assert!(
        training_time(&config)
            .unwrap_err()
            .contains("format mismatch")
    );
    config.dataset.format = DatasetFormat::Auto;
    config.dataset.validation = ValidationSplit::Count(1);
    assert!(training_time(&config).unwrap_err().contains("re-split"));
    config.dataset.validation = ValidationSplit::None;
    train_chat_byte_bpe(
        &["Hi ok hello world"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap()
    .save(temp.path().join("chat.json"))
    .unwrap();
    config.dataset.tokenizer = "chat.json".into();
    config.dataset.selections = vec!["release/chat/train".into()];
    config.model.context_length = 32;
    config.batching.max_batch_tokens = 64;
    let chat = training_time(&config).unwrap();
    assert_eq!(chat.dataset_format, "chat");
    assert!(chat.supervised_targets_per_epoch < chat.input_positions_per_epoch);
    assert_eq!(
        chat.samples[0].supervised_targets,
        chat.supervised_targets_per_epoch
    );
    config.dataset.cache = Some(temp.path().join("unsupported"));
    assert!(
        training_time(&config)
            .unwrap_err()
            .contains("Chat token caches")
    );
}

#[test]
fn limits_fail_without_returning_misleading_estimates() {
    let (_temp, mut config) = fixture();
    config.max_seconds = f64::MIN_POSITIVE;
    assert!(
        training_time(&config)
            .unwrap_err()
            .contains("no full-run estimate")
    );
    config.max_seconds = 60.0;
    config.batching.max_batch_tokens = 1;
    assert!(training_time(&config).is_err());
    config.batching.max_batch_tokens = 8;
    config.dataset.selections = vec!["../escape".into()];
    assert!(training_time(&config).is_err());
    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, 3601.0] {
        config.max_seconds = seconds;
        assert!(config.validate().is_err());
    }
}

fn command(root: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_omega-benchmark"));
    cmd.args([
        "training-time",
        "--dataset",
        "corpus",
        "--tokenizer",
        "tokenizer.json",
        "--datasets-root",
    ])
    .arg(root)
    .args([
        "--context-length",
        "4",
        "--d-model",
        "4",
        "--heads",
        "1",
        "--layers",
        "1",
        "--d-ff",
        "8",
        "--warmup",
        "1",
        "--samples",
        "1",
    ]);
    cmd
}

#[test]
fn cli_help_errors_json_thread_settings_and_output_no_clobber() {
    let help = Command::new(env!("CARGO_BIN_EXE_omega-benchmark"))
        .args(["training-time", "--help"])
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("--max-seconds"));
    let (temp, _) = fixture();
    let output = temp.path().join("report.json");
    let result = command(temp.path())
        .args([
            "--json",
            "--cpu-threads",
            "1",
            "--matmul-threads",
            "1",
            "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["runtime"]["cpu"]["rayon_num_threads"], "1");
    assert_eq!(report["runtime"]["cpu"]["matmul_num_threads"], "1");
    let saved = fs::read(&output).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&saved).unwrap(), report);
    assert!(
        !command(temp.path())
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(fs::read(&output).unwrap(), saved);
    for args in [
        vec!["--samples", "0"],
        vec!["--batch-size", "0"],
        vec!["--device", "1"],
        vec!["--epochs", "0"],
        vec!["--validation-count", "1", "--validation-ratio", "0.2"],
        vec!["--cpu-threads", "0"],
    ] {
        assert!(
            !command(temp.path())
                .args(args)
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    let human = command(temp.path()).output().unwrap();
    assert!(human.status.success());
    assert!(String::from_utf8_lossy(&human.stdout).contains("Estimated training updates"));
    let failed_path = temp.path().join("budget.json");
    assert!(
        !command(temp.path())
            .args(["--max-seconds", "0.000000000000001", "--output"])
            .arg(&failed_path)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(!failed_path.exists());
}

#[cfg(not(feature = "gpu"))]
#[test]
fn unavailable_gpu_never_silently_falls_back() {
    let (temp, mut config) = fixture();
    config.backend = Backend::Vulkan;
    assert!(
        training_time(&config)
            .unwrap_err()
            .contains("--features gpu")
    );
    assert!(
        !command(temp.path())
            .args(["--backend", "vulkan"])
            .output()
            .unwrap()
            .status
            .success()
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires an explicitly selected idle discrete Vulkan GPU"]
fn vulkan_training_time_smoke() {
    let (_temp, mut config) = fixture();
    config.backend = Backend::Vulkan;
    config.max_seconds = 120.0;
    let report = training_time(&config).unwrap();
    assert_eq!(report.backend, Backend::Vulkan);
    assert_eq!(report.samples.len(), 2);
    assert_eq!(report.supervised_targets_per_epoch, 7);
    assert!(
        report.estimated_training_seconds.is_finite() && report.estimated_training_seconds > 0.0
    );
    assert!(!report.runtime["gpu"].is_null());
}
