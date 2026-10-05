use std::{fs, path::Path};

use omega_training::{
    checkpoint::{
        BuildIdentity, CheckpointManifest, CheckpointMetadata, ModelConfig, TokenizerIdentity,
        reserve_run_directory, sha256_bytes,
    },
    checkpoint_catalog::{CatalogLimits, CatalogMode, CatalogStatus, discover_checkpoints},
    resume::{AdamSettings, ResumeManifest, ResumeRuntime},
    sampling::SAMPLING_ALGORITHM,
    trainer::{SessionOptions, TrainingProgress},
};

fn header(path: &Path, resumable: bool) {
    fs::create_dir(path).unwrap();
    let manifest = CheckpointManifest {
        chat: None,
        schema_version: 1,
        model: ModelConfig {
            vocab_size: 13,
            context_length: 4,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        },
        tokenizer: TokenizerIdentity {
            algorithm: "sha256-canonical-json-v1".into(),
            sha256: "a".repeat(64),
        },
        metadata: CheckpointMetadata::default(),
    };
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&manifest.model).unwrap(),
    )
    .unwrap();
    fs::write(
        path.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    // Intentionally invalid payload bytes: cataloging must never decode payloads.
    fs::write(path.join("tokenizer.json"), b"not a tokenizer").unwrap();
    fs::write(path.join("model.mpk"), b"not tensors").unwrap();
    if resumable {
        let resume = ResumeManifest {
            objective: None,
            parent: None,
            schema_version: 2,
            execution_mode: "epoch-order-cpu-ndarray-f32-v2".into(),
            rng_policy: "unused-after-initialization".into(),
            runtime: ResumeRuntime {
                build: BuildIdentity::current(),
                operating_system: "other-os".into(),
                architecture: "other-architecture".into(),
            },
            gpu_execution: None,
            cuda_execution: None,
            cpu_execution: omega_training::cpu::CpuExecutionProfile {
                kernel: "ndarray-f32-checked-v1".into(),
                rayon_num_threads: None,
                rayon_rs_num_cpus: None,
                matmul_num_threads: None,
                available_parallelism: None,
            },
            model: manifest.model,
            tokenizer: manifest.tokenizer,
            source_identity: "b".repeat(64),
            example_count: 1,
            targets_per_epoch: 1,
            learning_rate_bits: 0.01_f64.to_bits(),
            initialization_seed: 42,
            adam: AdamSettings::default(),
            options: SessionOptions::default(),
            sampling_algorithm: SAMPLING_ALGORITHM.into(),
            progress: TrainingProgress::default(),
            epoch_weighted_loss_bits: 0.0_f64.to_bits(),
            model_sha256: "c".repeat(64),
            optimizer_sha256: "d".repeat(64),
        };
        let mut value = serde_json::to_value(resume).unwrap();
        value.as_object_mut().unwrap().remove("cpu_execution");
        write_resume(path, &value);
        fs::write(path.join("optimizer.mpk"), b"not optimizer tensors").unwrap();
    }
    fs::write(path.join("COMPLETE"), b"omega-checkpoint-v1\n").unwrap();
}

fn write_resume(path: &Path, value: &serde_json::Value) {
    let bytes = serde_json::to_vec(value).unwrap();
    fs::write(path.join("resume.sha256"), sha256_bytes(&bytes)).unwrap();
    fs::write(path.join("resume.json"), bytes).unwrap();
}

#[test]
fn gpu_metadata_catalogs_without_gpu_initialization_or_runtime_matching() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run-1");
    header(&path, true);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("resume.json")).unwrap()).unwrap();
    value["schema_version"] = 4.into();
    value["execution_mode"] = omega_training::resume::GPU_EXECUTION_MODE.into();
    value["cpu_execution"] = serde_json::json!({
        "kernel":"ndarray-f32-checked-v1", "rayon_num_threads":null,
        "rayon_rs_num_cpus":null, "matmul_num_threads":null, "available_parallelism":null
    });
    value["gpu_execution"] = serde_json::json!({
        "schema_version":1, "kernel":"wgpu-vulkan-f32-checked-v1", "precision":"f32",
        "adapter_name":"different GPU", "adapter_index":0, "vendor":1, "device":2, "device_type":"DiscreteGpu",
        "backend":"Vulkan", "driver":"different-driver", "driver_info":"different-version",
        "host_kernel":"different-kernel", "rustc":"different-rustc", "target":"different-target",
        "features":"gpu;spirv;no-fusion", "build_profile":"release", "rustflags":""
    });
    write_resume(&path, &value);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Resumable);
    assert_eq!(
        catalog.latest(CatalogMode::Resume).unwrap().path,
        path.canonicalize().unwrap()
    );
    let original = value.clone();
    for field in ["gpu_execution", "cpu_execution"] {
        let mut invalid = original.clone();
        invalid.as_object_mut().unwrap().remove(field);
        write_resume(&path, &invalid);
        let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
        assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
    }
    value = original;
    value["schema_version"] = 3.into();
    value["execution_mode"] = "epoch-order-cpu-ndarray-f32-v2".into();
    write_resume(&path, &value);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
}

#[test]
fn schema_three_cpu_metadata_is_checked_without_current_runtime_comparison() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run-1");
    header(&path, true);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("resume.json")).unwrap()).unwrap();
    value["schema_version"] = 3.into();
    value["cpu_execution"] = serde_json::json!({
        "kernel":"ndarray-f32-checked-v1", "rayon_num_threads":"7",
        "rayon_rs_num_cpus":null, "matmul_num_threads":"4",
        "available_parallelism":999
    });
    write_resume(&path, &value);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Resumable);
    value["cpu_execution"]
        .as_object_mut()
        .unwrap()
        .remove("matmul_num_threads");
    write_resume(&path, &value);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
    assert!(catalog.latest(CatalogMode::Resume).is_err());
}

#[test]
fn numeric_inventory_and_selection_preserve_reservations_and_report_skips() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    header(&root.join("RUN-0002"), true);
    header(&root.join("Run-10"), false);
    fs::create_dir(root.join("run-11")).unwrap();
    fs::write(root.join("RUN-12"), b"occupied").unwrap();
    fs::create_dir(root.join("unrelated-99")).unwrap();
    fs::create_dir(root.join("run-x")).unwrap();
    let catalog = discover_checkpoints(root, "rUn", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.run_name, "run");
    assert_eq!(
        catalog.entries.iter().map(|e| e.number).collect::<Vec<_>>(),
        [2, 10, 11, 12]
    );
    assert_eq!(catalog.entries[0].status, CatalogStatus::Resumable);
    assert_eq!(catalog.entries[1].status, CatalogStatus::CompleteInference);
    let inference = catalog.latest(CatalogMode::Inference).unwrap();
    assert_eq!(inference.number, 10);
    assert_eq!(
        inference
            .skipped
            .iter()
            .map(|e| e.number)
            .collect::<Vec<_>>(),
        [12, 11]
    );
    let resume = catalog.latest(CatalogMode::Resume).unwrap();
    assert_eq!(resume.number, 2);
    assert_eq!(
        resume.skipped.iter().map(|e| e.number).collect::<Vec<_>>(),
        [12, 11, 10]
    );
    assert!(resume.inspection.contains("payloads_unverified"));
    assert!(resume.path.is_absolute());
    let json = serde_json::to_value(&catalog).unwrap();
    assert_eq!(json["entries"][0]["status"], "resumable");
    // Only this explicit reservation mutates the root; discovery filled no gaps.
    assert_eq!(fs::read_dir(root).unwrap().count(), 6);
    assert!(
        reserve_run_directory(root, "RUN")
            .unwrap()
            .ends_with("run-13")
    );
}

#[test]
fn equal_number_aliases_are_visible_but_never_auto_selected() {
    let temp = tempfile::tempdir().unwrap();
    header(&temp.path().join("run-01"), false);
    header(&temp.path().join("RUN-1"), true);
    header(&temp.path().join("run-2"), true);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries.len(), 3);
    for mode in [CatalogMode::Inference, CatalogMode::Resume] {
        assert!(catalog.latest(mode).unwrap_err().contains("Ambiguous"));
    }
    assert!(
        reserve_run_directory(temp.path(), "run")
            .unwrap()
            .ends_with("run-3")
    );
}

#[test]
fn completed_corrupt_metadata_never_falls_back() {
    for broken in [
        "manifest.json",
        "config.json",
        "resume.json",
        "resume.sha256",
        "COMPLETE",
    ] {
        let temp = tempfile::tempdir().unwrap();
        header(&temp.path().join("run-1"), true);
        let newer = temp.path().join("run-2");
        header(&newer, true);
        fs::write(newer.join(broken), b"corrupt").unwrap();
        let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
        assert_eq!(
            catalog.entries[1].status,
            CatalogStatus::Invalid,
            "{broken}"
        );
        for mode in [CatalogMode::Inference, CatalogMode::Resume] {
            assert!(
                catalog
                    .latest(mode)
                    .unwrap_err()
                    .contains("refusing fallback")
            );
        }
    }
}

#[test]
fn missing_payloads_and_partial_resume_artifacts_are_invalid_headers() {
    for missing in [
        "tokenizer.json",
        "model.mpk",
        "optimizer.mpk",
        "resume.sha256",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("run-1");
        header(&path, true);
        fs::remove_file(path.join(missing)).unwrap();
        let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
        assert_eq!(
            catalog.entries[0].status,
            CatalogStatus::Invalid,
            "{missing}"
        );
    }
}

fn assert_training_provenance_conflict(seed: u64, learning_rate: f64) {
    let temp = tempfile::tempdir().unwrap();
    header(&temp.path().join("run-1"), true);
    let path = temp.path().join("run-2");
    header(&path, true);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap();
    manifest["metadata"]["training"] = serde_json::json!({
        "epochs": 1,
        "learning_rate": learning_rate,
        "seed": seed,
        "optimizer": "adam-default-burn-0.18"
    });
    fs::write(
        path.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[1].status, CatalogStatus::Invalid);
    assert!(catalog.entries[1].reason.contains("provenance"));
    for mode in [CatalogMode::Inference, CatalogMode::Resume] {
        assert!(
            catalog
                .latest(mode)
                .unwrap_err()
                .contains("refusing fallback")
        );
    }
}

#[test]
fn conflicting_training_seed_is_invalid_metadata() {
    assert_training_provenance_conflict(43, 0.01);
}

#[test]
fn conflicting_training_learning_rate_is_invalid_metadata() {
    assert_training_provenance_conflict(42, 0.02);
}

#[test]
fn legacy_and_old_resume_headers_keep_their_explicit_limits() {
    let temp = tempfile::tempdir().unwrap();
    let legacy = temp.path().join("run-1");
    header(&legacy, false);
    fs::remove_file(legacy.join("manifest.json")).unwrap();
    fs::write(legacy.join("COMPLETE"), []).unwrap();
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(
        catalog.latest(CatalogMode::Inference).unwrap().status,
        CatalogStatus::LegacyUnverified
    );
    assert!(catalog.latest(CatalogMode::Resume).is_err());
    let old_resume = temp.path().join("run-2");
    header(&old_resume, true);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(old_resume.join("resume.json")).unwrap()).unwrap();
    value["schema_version"] = 1.into();
    value["execution_mode"] = "fixed-order-cpu-ndarray-f32-v1".into();
    value.as_object_mut().unwrap().remove("options");
    value.as_object_mut().unwrap().remove("sampling_algorithm");
    write_resume(&old_resume, &value);
    let original = fs::read(old_resume.join("resume.json")).unwrap();
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.latest(CatalogMode::Resume).unwrap().number, 2);
    assert_eq!(fs::read(old_resume.join("resume.json")).unwrap(), original);
    value["model"]["context_length"] = 3.into();
    write_resume(&old_resume, &value);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert!(catalog.latest(CatalogMode::Resume).is_err());
}

#[test]
fn invalid_unknown_schemas_and_resume_fields_fail_without_runtime_loading() {
    for field in ["schema_version", "options", "source_identity"] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("run-1");
        header(&path, true);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("resume.json")).unwrap()).unwrap();
        match field {
            "schema_version" => value[field] = 999.into(),
            "source_identity" => value[field] = "bad".into(),
            _ => {
                value.as_object_mut().unwrap().remove(field);
            }
        }
        write_resume(&path, &value);
        let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
        assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
    }
}

#[test]
fn bounded_scans_and_aggregate_metadata_reject_without_partial_success() {
    let temp = tempfile::tempdir().unwrap();
    header(&temp.path().join("run-1"), false);
    fs::write(temp.path().join("unrelated"), []).unwrap();
    let limits = CatalogLimits {
        max_entries: 1,
        ..Default::default()
    };
    assert!(
        discover_checkpoints(temp.path(), "run", &limits)
            .unwrap_err()
            .contains("max_entries")
    );
    let limits = CatalogLimits {
        max_metadata_bytes: 1,
        ..Default::default()
    };
    let catalog = discover_checkpoints(temp.path(), "run", &limits).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
    assert!(catalog.entries[0].reason.contains("byte limit"));
    let path = temp.path().join("run-1");
    let total: usize = ["manifest.json", "config.json", "COMPLETE"]
        .iter()
        .map(|n| fs::metadata(path.join(n)).unwrap().len() as usize)
        .sum();
    // Every individual file fits, but the aggregate (including marker re-read) does not.
    let limits = CatalogLimits {
        max_metadata_bytes: total - 1,
        ..Default::default()
    };
    let catalog = discover_checkpoints(temp.path(), "run", &limits).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
    fs::OpenOptions::new()
        .write(true)
        .open(path.join("manifest.json"))
        .unwrap()
        .set_len(1_000_000)
        .unwrap();
    let limits = CatalogLimits {
        max_metadata_bytes: 1024,
        ..Default::default()
    };
    assert!(
        discover_checkpoints(temp.path(), "run", &limits)
            .unwrap()
            .entries[0]
            .reason
            .contains("byte limit")
    );
    assert!(
        discover_checkpoints(
            temp.path(),
            "run",
            &CatalogLimits {
                max_entries: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn resume_semantics_are_checked_without_declared_source_sized_allocations() {
    for scenario in ["batch", "clip", "loss", "cursor", "weighted"] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("run-1");
        header(&path, true);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("resume.json")).unwrap()).unwrap();
        match scenario {
            "batch" => value["options"]["batching"]["batch_size"] = 0.into(),
            "clip" => value["options"]["optimization"]["gradient_clip_norm"] = (-1).into(),
            "loss" => value["epoch_weighted_loss_bits"] = f64::NAN.to_bits().into(),
            "cursor" => value["progress"]["next_example_index"] = 1.into(),
            "weighted" => {
                value["example_count"] = usize::MAX.into();
                value["targets_per_epoch"] = usize::MAX.into();
                value["options"]["sampling"] = serde_json::json!({
                    "kind": "weighted", "samples_per_epoch": 1,
                    "groups": [{"name":"one", "indices":[0], "weight":1}]
                });
            }
            _ => unreachable!(),
        }
        write_resume(&path, &value);
        let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
        assert_eq!(
            catalog.entries[0].status,
            CatalogStatus::Invalid,
            "{scenario}"
        );
    }
}

#[test]
fn empty_missing_invalid_names_and_u64_overflow_are_read_only() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing");
    assert!(discover_checkpoints(&missing, "run", &CatalogLimits::default()).is_err());
    assert!(!missing.exists());
    assert!(discover_checkpoints(temp.path(), "../bad", &CatalogLimits::default()).is_err());
    let empty = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert!(empty.entries.is_empty());
    assert!(empty.latest(CatalogMode::Inference).is_err());
    fs::write(temp.path().join("run-18446744073709551616"), []).unwrap();
    assert!(
        discover_checkpoints(temp.path(), "run", &CatalogLimits::default())
            .unwrap_err()
            .contains("exceeds u64")
    );
    assert!(
        reserve_run_directory(temp.path(), "run")
            .unwrap_err()
            .contains("exceeds u64")
    );
}

#[cfg(any(unix, windows))]
#[test]
fn symlink_roots_candidates_and_artifacts_are_never_followed() {
    let temp = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    header(&outside.path().join("real"), false);
    let link = temp.path().join("run-2");
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(outside.path().join("real"), &link);
    #[cfg(windows)]
    let result = std::os::windows::fs::symlink_dir(outside.path().join("real"), &link);
    if let Err(error) = result {
        #[cfg(windows)]
        if error.kind() == std::io::ErrorKind::PermissionDenied
            || error.raw_os_error() == Some(1314)
        {
            eprintln!("Symlink regression unavailable: Windows privilege denied");
            return;
        }
        panic!("Cannot create test symlink: {error}");
    }
    header(&temp.path().join("run-1"), false);
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[1].status, CatalogStatus::Invalid);
    assert!(catalog.latest(CatalogMode::Inference).is_err());
    assert!(discover_checkpoints(&link, "run", &CatalogLimits::default()).is_err());
    let artifact = temp.path().join("run-1/model.mpk");
    fs::remove_file(&artifact).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.path().join("real/model.mpk"), &artifact).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(outside.path().join("real/model.mpk"), &artifact).unwrap();
    let catalog = discover_checkpoints(temp.path(), "run", &CatalogLimits::default()).unwrap();
    assert_eq!(catalog.entries[0].status, CatalogStatus::Invalid);
    assert_eq!(
        fs::read(outside.path().join("real/model.mpk")).unwrap(),
        b"not tensors"
    );
}
