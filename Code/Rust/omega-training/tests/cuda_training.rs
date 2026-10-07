//! Explicit hardware qualification; never silently substitutes the CPU.
#![cfg(feature = "cuda")]

type Gpu = burn::backend::Cuda<f32, i32>;

use omega_nn::{GptConfig, token_tensor};
use omega_training::{
    ExampleSource, TrainingSession,
    batching::BatchConfig,
    checkpoint::{CheckpointMetadata, sha256_bytes},
    cuda::{CudaTraining as GpuTraining, initialize},
    dataset::ASSISTANT_TARGETS_OBJECTIVE,
    evaluation::evaluate_source_on_device,
    resume::{load_cuda_training_checkpoint, save_cuda_training_checkpoint},
    trainer::SessionOptions,
};

#[test]
#[ignore = "requires NVIDIA hardware; run explicitly on an idle host"]
fn base_resume_and_rejected_update_preserve_cuda_state() {
    use omega_training::{
        TrainingSet,
        checkpoint::CheckpointMetadata,
        resume::{
            load_cuda_training_checkpoint, load_training_checkpoint, read_resume_manifest,
            save_cuda_training_checkpoint,
        },
    };
    let selected = initialize(0).unwrap();
    let config = GptConfig {
        vocab_size: 13,
        context_length: 3,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let tokens = omega_tokenizer::Tokens::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/wordlevel.json"),
    )
    .unwrap();
    let source = || TrainingSet {
        examples: vec![vec![1, 2, 3], vec![3, 4, 5]],
        files: vec![],
        token_count: 6,
    };
    let root = tempfile::tempdir().unwrap();
    let mut run = TrainingSession::<_, GpuTraining>::from_source_on_device(
        &config,
        source(),
        0.01,
        42,
        &selected.device,
    )
    .unwrap();
    run.step().unwrap();
    let checkpoint = save_cuda_training_checkpoint(
        root.path(),
        "base",
        &run,
        &tokens,
        &CheckpointMetadata::default(),
        &selected,
        None,
    )
    .unwrap();
    assert_eq!(read_resume_manifest(&checkpoint).unwrap().schema_version, 7);
    let mut incompatible_profile = selected.profile.clone();
    incompatible_profile.source_sha256 = "0".repeat(64);
    let incompatible = omega_training::cuda::CudaDevice {
        device: selected.device.clone(),
        profile: incompatible_profile,
    };
    assert!(
        load_cuda_training_checkpoint(&checkpoint, source(), &config, &tokens, &incompatible)
            .is_err()
    );

    assert!(load_training_checkpoint(&checkpoint, source(), &config, &tokens).is_err());
    let mut restored =
        load_cuda_training_checkpoint(&checkpoint, source(), &config, &tokens, &selected).unwrap();
    for _ in 0..3 {
        assert_eq!(run.step().unwrap(), restored.step().unwrap());
    }
    let mut invalid = TrainingSession::<_, GpuTraining>::from_source_on_device(
        &config,
        source(),
        f64::MAX,
        42,
        &selected.device,
    )
    .unwrap();
    let before = invalid.progress();
    let input = token_tensor::<Gpu>(&[1, 2], 13, 3, &selected.device).unwrap();
    let values = invalid.inference_model().forward(input.clone()).into_data();
    assert!(
        invalid
            .step()
            .unwrap_err()
            .contains("Candidate update rejected")
    );
    assert_eq!(invalid.progress(), before);
    assert_eq!(invalid.inference_model().forward(input).into_data(), values);
}

struct Conversations {
    changed: bool,
}
impl ExampleSource for Conversations {
    fn example_count(&self) -> usize {
        3
    }
    fn target_count(&self) -> Result<usize, String> {
        Ok(4)
    }
    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        [vec![1, 2, 3, 4], vec![4, 5], vec![6, 7, 8]]
            .get(index)
            .cloned()
            .ok_or("bad index".into())
    }
    fn target_mask(&self, index: usize) -> Result<Option<Vec<bool>>, String> {
        [
            if self.changed {
                vec![true, false, true]
            } else {
                vec![false, true, true]
            },
            vec![true],
            vec![false, true],
        ]
        .get(index)
        .cloned()
        .map(Some)
        .ok_or("bad index".into())
    }
    fn objective(&self) -> &str {
        ASSISTANT_TARGETS_OBJECTIVE
    }
    fn identity(&self) -> Result<String, String> {
        Ok(sha256_bytes(
            format!("gpu-masked-test-v1/{}", self.changed).as_bytes(),
        ))
    }
}

#[test]
#[ignore = "requires NVIDIA hardware; run with --features cuda --test cuda_training -- --ignored"]
fn assistant_batches_evaluate_and_resume_exactly_on_selected_gpu() {
    let selected = initialize(0).unwrap();
    let device = &selected.device;
    let config = GptConfig {
        vocab_size: 13,
        context_length: 4,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let tokenizer = omega_tokenizer::Tokens::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/wordlevel.json"),
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    for batch_size in [1, 2] {
        let options = SessionOptions {
            batching: BatchConfig {
                batch_size,
                max_batch_tokens: 8,
            },
            ..Default::default()
        };
        let mut continuous = TrainingSession::<_, GpuTraining>::from_source_with_options_on_device(
            &config,
            Conversations { changed: false },
            0.01,
            42,
            options,
            device,
        )
        .unwrap();
        continuous.step().unwrap();
        let path = save_cuda_training_checkpoint(
            root.path(),
            "assistant",
            &continuous,
            &tokenizer,
            &CheckpointMetadata::default(),
            &selected,
            None,
        )
        .unwrap();
        assert_eq!(
            omega_training::resume::read_resume_manifest(&path)
                .unwrap()
                .schema_version,
            8
        );
        assert!(
            load_cuda_training_checkpoint(
                &path,
                Conversations { changed: true },
                &config,
                &tokenizer,
                &selected
            )
            .is_err()
        );
        let mut resumed = load_cuda_training_checkpoint(
            &path,
            Conversations { changed: false },
            &config,
            &tokenizer,
            &selected,
        )
        .unwrap();
        let metrics = evaluate_source_on_device(
            &resumed.inference_model(),
            &config,
            &Conversations { changed: false },
            device,
        )
        .unwrap();
        assert_eq!(metrics.target_count, 4);
        for _ in 0..4 {
            assert_eq!(continuous.step().unwrap(), resumed.step().unwrap());
        }
        assert_eq!(continuous.progress(), resumed.progress());
        let input = token_tensor::<Gpu>(&[1, 2, 3], 13, 4, device).unwrap();
        assert_eq!(
            continuous
                .inference_model()
                .forward(input.clone())
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            resumed
                .inference_model()
                .forward(input)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
    }
}
