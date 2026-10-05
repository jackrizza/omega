//! Hardware qualification is opt-in: run --features gpu --test gpu_resume -- --ignored.
#![cfg(feature = "gpu")]

use std::{fs, path::Path};

use burn::{
    module::{Module, ModuleMapper, ParamId},
    optim::{Adam, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Record, Recorder},
    tensor::{Tensor, TensorData, backend::Backend},
};
use omega_nn::{Cpu, Gpt, GptConfig, token_tensor};
use omega_tokenizer::Tokens;
use omega_training::{
    TrainingSession, TrainingSet,
    batching::BatchConfig,
    checkpoint::{
        CheckpointMetadata, load_checkpoint, load_checkpoint_on_device, save_checkpoint,
        sha256_bytes,
    },
    checkpoint_catalog::{CatalogLimits, CatalogMode, CatalogStatus, discover_checkpoints},
    gpu::{Gpu, GpuTraining, initialize_vulkan},
    optimization::OptimizationOptions,
    resume::{
        load_gpu_training_checkpoint, load_training_checkpoint, read_resume_manifest,
        save_gpu_training_checkpoint, save_training_checkpoint,
    },
    sampling::{SamplingGroup, SamplingPolicy},
    trainer::SessionOptions,
};

fn config() -> GptConfig {
    GptConfig {
        vocab_size: 13,
        context_length: 3,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    }
}
fn source() -> TrainingSet {
    TrainingSet {
        examples: vec![vec![1, 2], vec![2, 3, 4], vec![4, 5, 6, 7]],
        files: vec![],
        token_count: 9,
    }
}
fn logits<B: Backend<FloatElem = f32>>(model: &Gpt<B>, device: &B::Device) -> Vec<f32> {
    model
        .forward(token_tensor::<B>(&[1, 2, 3], 13, 3, device).unwrap())
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}
fn model_record(model: Gpt<Gpu>) -> Vec<u8> {
    <_ as Recorder<Gpu>>::record(
        &NamedMpkBytesRecorder::<FullPrecisionSettings>::default(),
        model.into_record(),
        (),
    )
    .unwrap()
}
fn optimizer_value(path: &Path, device: &<Gpu as Backend>::Device) -> serde_json::Value {
    type Optim = OptimizerAdaptor<Adam, Gpt<GpuTraining>, GpuTraining>;
    type OptimRecord = <Optim as Optimizer<Gpt<GpuTraining>, GpuTraining>>::Record;
    let record: OptimRecord = <_ as Recorder<GpuTraining>>::load(
        &NamedMpkBytesRecorder::<FullPrecisionSettings>::default(),
        fs::read(path.join("optimizer.mpk")).unwrap(),
        device,
    )
    .unwrap();
    // Adam stores parameters in a HashMap. Compare IDs/tensor bytes/counters,
    // not nondeterministic map serialization order.
    serde_json::to_value(<OptimRecord as Record<GpuTraining>>::into_item::<
        FullPrecisionSettings,
    >(record))
    .unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 0.003 + 0.003 * a.abs(), "{a} != {b}");
    }
}
fn rewrite(path: &Path, value: &serde_json::Value) {
    let bytes = serde_json::to_vec_pretty(value).unwrap();
    fs::write(path.join("resume.sha256"), sha256_bytes(&bytes)).unwrap();
    fs::write(path.join("resume.json"), bytes).unwrap();
}

fn nonfinite_candidate_preserves_committed_gpu_state(
    root: &Path,
    tokenizer: &Tokens,
    selected: &omega_training::gpu::VulkanDevice,
) {
    let device = selected.device();
    // Finite f64 configuration, finite model/loss/gradients, but the effective
    // Adam rate overflows f32. The invalid candidate must never be committed.
    let mut run = TrainingSession::<_, GpuTraining>::from_source_on_device(
        &config(),
        source(),
        f64::MAX,
        42,
        device,
    )
    .unwrap();
    let metadata = CheckpointMetadata::default();
    let before = save_gpu_training_checkpoint(
        root,
        "rollback-before",
        &run,
        tokenizer,
        &metadata,
        selected,
    )
    .unwrap();
    let before_model = model_record(run.inference_model());
    let before_progress = run.progress();
    let error = run.step().unwrap_err();
    assert!(error.contains("Candidate update rejected"), "{error}");
    assert_eq!(run.progress(), before_progress);
    assert_eq!(model_record(run.inference_model()), before_model);
    let after =
        save_gpu_training_checkpoint(root, "rollback-after", &run, tokenizer, &metadata, selected)
            .unwrap();
    assert_eq!(
        optimizer_value(&before, device),
        optimizer_value(&after, device)
    );
    let a = read_resume_manifest(&before).unwrap();
    let b = read_resume_manifest(&after).unwrap();
    assert_eq!(a.progress, b.progress);
    assert_eq!(a.epoch_weighted_loss_bits, b.epoch_weighted_loss_bits);
    // A rejected candidate remains rejected on retry; no hidden Adam/cursor drift.
    assert!(
        run.step()
            .unwrap_err()
            .contains("Candidate update rejected")
    );
    assert_eq!(run.progress(), before_progress);
    assert_eq!(model_record(run.inference_model()), before_model);

    struct Nonfinite;
    impl ModuleMapper<Gpu> for Nonfinite {
        fn map_float<const D: usize>(
            &mut self,
            _id: ParamId,
            tensor: Tensor<Gpu, D>,
        ) -> Tensor<Gpu, D> {
            Tensor::full(tensor.shape(), f32::NAN, &tensor.device())
        }
    }
    let invalid = run.inference_model().map(&mut Nonfinite);
    let error = omega_nn::generation::generate_with_options_on_device(
        &invalid,
        &config(),
        &[1],
        1,
        None,
        &omega_nn::GenerationOptions::Greedy,
        device,
    )
    .unwrap_err();
    assert!(error.contains("finite"), "{error}");
}

fn generation_margin_contract(root: &Path, tokenizer: &Tokens, device: &<Gpu as Backend>::Device) {
    struct ControlledHead {
        second: f32,
    }
    impl ModuleMapper<Cpu> for ControlledHead {
        fn map_float<const D: usize>(
            &mut self,
            _id: ParamId,
            tensor: Tensor<Cpu, D>,
        ) -> Tensor<Cpu, D> {
            let dims = tensor.dims();
            let mut values = vec![if D == 1 { 1.0 } else { 0.0 }; dims.iter().product()];
            if dims.as_slice() == [4, 13] {
                for row in 0..4 {
                    values[row * 13] = 1.0;
                    values[row * 13 + 1] = self.second;
                }
            }
            Tensor::from_data(TensorData::new(values, dims), &tensor.device())
        }
    }
    for second in [0.0, 1.0 + f32::EPSILON] {
        let cpu = config()
            .init::<Cpu>(&Default::default())
            .unwrap()
            .map(&mut ControlledHead { second });
        let path =
            save_checkpoint(root, "generation-margin", cpu.clone(), &config(), tokenizer).unwrap();
        let (gpu, _, _) = load_checkpoint_on_device::<Gpu>(&path, device).unwrap();
        let a = logits(&cpu, &Default::default());
        let b = logits(&gpu, device);
        close(&a, &b);
        let cpu_tokens = omega_nn::generate_with_options(
            &cpu,
            &config(),
            &[1, 2],
            1,
            None,
            &omega_nn::GenerationOptions::Greedy,
        )
        .unwrap();
        let gpu_tokens = omega_nn::generation::generate_with_options_on_device(
            &gpu,
            &config(),
            &[1, 2],
            1,
            None,
            &omega_nn::GenerationOptions::Greedy,
            device,
        )
        .unwrap();
        if second == 0.0 {
            assert_eq!(cpu_tokens, vec![1, 2, 0]);
            assert_eq!(
                gpu_tokens, cpu_tokens,
                "unambiguous maximum must preserve greedy selection"
            );
        } else {
            // Backend roundoff may swap nearly tied maxima. Require close logits
            // and selection from the tied candidate set, not identical token IDs.
            assert!([0, 1].contains(cpu_tokens.last().unwrap()));
            assert!([0, 1].contains(gpu_tokens.last().unwrap()));
        }
    }
}

#[test]
#[ignore = "requires a real discrete Vulkan GPU; run explicitly under a timeout"]
fn gpu_training_transfer_continuation_and_rejection_contracts() {
    let selected = initialize_vulkan(0).unwrap();
    let device = selected.device();
    let config = config();
    let tokenizer =
        Tokens::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"))
            .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let metadata = CheckpointMetadata::default();
    // Inference transfer starts with identical serialized CPU parameters.
    let cpu = config.init::<Cpu>(&Default::default()).unwrap();
    let path = save_checkpoint(temp.path(), "cpu", cpu.clone(), &config, &tokenizer).unwrap();
    let (gpu, _, _) = load_checkpoint_on_device::<Gpu>(&path, device).unwrap();
    close(&logits(&cpu, &Default::default()), &logits(&gpu, device));
    let gpu_path = save_checkpoint(temp.path(), "gpu", gpu.clone(), &config, &tokenizer).unwrap();
    let (roundtrip, _, _) = load_checkpoint(&gpu_path).unwrap();
    assert_eq!(
        logits(&cpu, &Default::default()),
        logits(&roundtrip, &Default::default())
    );
    let cpu_session = TrainingSession::new(&config, source(), 0.003, 42).unwrap();
    let cpu_path = save_training_checkpoint(
        temp.path(),
        "cpu-state",
        &cpu_session,
        &tokenizer,
        &metadata,
    )
    .unwrap();
    assert!(
        load_gpu_training_checkpoint(&cpu_path, source(), &config, &tokenizer, &selected).is_err()
    );

    nonfinite_candidate_preserves_committed_gpu_state(temp.path(), &tokenizer, &selected);
    generation_margin_contract(temp.path(), &tokenizer, device);

    let policies = [
        SamplingPolicy::Fixed,
        SamplingPolicy::Shuffle,
        SamplingPolicy::Weighted {
            groups: vec![SamplingGroup {
                name: "all".into(),
                indices: vec![0, 1, 2],
                weight: 1,
            }],
            samples_per_epoch: 3,
        },
    ];
    for policy in policies {
        let options = SessionOptions {
            sampling: policy,
            batching: BatchConfig {
                batch_size: 2,
                max_batch_tokens: 6,
            },
            optimization: OptimizationOptions {
                gradient_clip_norm: Some(0.5),
                warmup_updates: 3,
            },
        };
        let initial = TrainingSession::<_, GpuTraining>::from_source_with_options_on_device(
            &config,
            source(),
            0.003,
            42,
            options,
            device,
        )
        .unwrap();
        // One update is mid-epoch; two updates complete the three-draw epoch.
        for boundary in [1, 2] {
            let mut continuous = initial.clone();
            continuous.advance_updates(boundary, |_, _| Ok(())).unwrap();
            let saved = save_gpu_training_checkpoint(
                temp.path(),
                "state",
                &continuous,
                &tokenizer,
                &metadata,
                &selected,
            )
            .unwrap();
            let manifest = read_resume_manifest(&saved).unwrap();
            assert_eq!(manifest.schema_version, 4);
            assert_eq!(manifest.gpu_execution.as_ref(), Some(selected.profile()));
            assert!(load_training_checkpoint(&saved, source(), &config, &tokenizer).is_err());
            let catalog =
                discover_checkpoints(temp.path(), "state", &CatalogLimits::default()).unwrap();
            let latest = catalog.latest(CatalogMode::Resume).unwrap();
            assert_eq!(latest.path, saved);
            assert_eq!(latest.status, CatalogStatus::Resumable);
            let mut restored =
                load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                    .unwrap();
            assert_eq!(continuous.progress(), restored.progress());
            assert_eq!(
                model_record(continuous.inference_model()),
                model_record(restored.inference_model())
            );
            for _ in 0..3 {
                // Assert bitwise equality, stronger than CPU/GPU tolerance comparison.
                assert_eq!(continuous.step().unwrap(), restored.step().unwrap());
            }
            assert_eq!(continuous.progress(), restored.progress());
            assert_eq!(
                model_record(continuous.inference_model()),
                model_record(restored.inference_model())
            );
            let a = save_gpu_training_checkpoint(
                temp.path(),
                "control",
                &continuous,
                &tokenizer,
                &metadata,
                &selected,
            )
            .unwrap();
            let b = save_gpu_training_checkpoint(
                temp.path(),
                "restored",
                &restored,
                &tokenizer,
                &metadata,
                &selected,
            )
            .unwrap();
            assert_eq!(optimizer_value(&a, device), optimizer_value(&b, device));

            let original: serde_json::Value =
                serde_json::from_slice(&fs::read(saved.join("resume.json")).unwrap()).unwrap();
            let mut legacy_checks = original.clone();
            legacy_checks["gpu_execution"]["kernel"] = "wgpu-vulkan-f32-checked-v1".into();
            rewrite(&saved, &legacy_checks);
            assert!(
                load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                    .err()
                    .unwrap()
                    .contains("execution profile differs")
            );
            for field in [
                "driver_info",
                "adapter_name",
                "host_kernel",
                "rustc",
                "features",
                "build_profile",
                "rustflags",
            ] {
                let mut changed = original.clone();
                changed["gpu_execution"][field] = "other".into();
                rewrite(&saved, &changed);
                assert!(
                    load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                        .is_err()
                );
            }
            let mut changed = original.clone();
            changed["gpu_execution"]["adapter_index"] =
                (selected.profile().adapter_index + 1).into();
            rewrite(&saved, &changed);
            assert!(
                load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                    .is_err()
            );
            let mut changed = original.clone();
            changed["runtime"]["operating_system"] = "other-os".into();
            rewrite(&saved, &changed);
            assert!(
                load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                    .is_err()
            );
            changed = original.clone();
            changed["runtime"]["build"]["cargo_lock_sha256"] =
                "04991746127d1994da4354c1d969ed436a799b8ef6ac9e549b473a43b4782696".into();
            rewrite(&saved, &changed);
            assert!(
                load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                    .is_err()
            );
            rewrite(&saved, &original);
            fs::remove_file(saved.join("COMPLETE")).unwrap();
            assert!(
                load_gpu_training_checkpoint(&saved, source(), &config, &tokenizer, &selected)
                    .is_err()
            );
        }
    }
}
