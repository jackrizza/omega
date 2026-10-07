//! Explicit hardware qualification; never silently substitutes the CPU.
#![cfg(feature = "gpu")]

use omega_nn::{GptConfig, token_tensor};
use omega_training::{
    ExampleSource, TrainingSession,
    batching::BatchConfig,
    checkpoint::{CheckpointMetadata, sha256_bytes},
    dataset::ASSISTANT_TARGETS_OBJECTIVE,
    evaluation::evaluate_source_on_device,
    gpu::{Gpu, GpuTraining, initialize_vulkan},
    resume::{load_gpu_training_checkpoint, save_gpu_training_checkpoint},
    trainer::SessionOptions,
};

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
#[ignore = "requires explicitly qualified Vulkan GPU; run with --features gpu --test gpu_masked_training -- --ignored"]
fn assistant_batches_evaluate_and_resume_exactly_on_selected_gpu() {
    let selected = initialize_vulkan(0).unwrap();
    let device = selected.device();
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
        let path = save_gpu_training_checkpoint(
            root.path(),
            "assistant",
            &continuous,
            &tokenizer,
            &CheckpointMetadata::default(),
            &selected,
        )
        .unwrap();
        assert!(
            load_gpu_training_checkpoint(
                &path,
                Conversations { changed: true },
                &config,
                &tokenizer,
                &selected
            )
            .is_err()
        );
        let mut resumed = load_gpu_training_checkpoint(
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

#[test]
#[ignore = "requires explicitly qualified Vulkan GPU; run with --features gpu --test gpu_masked_training -- --ignored"]
fn gpu_base_initializes_fresh_conversation_stage_and_resumes_with_lineage() {
    use omega_training::{
        TrainingSet, ValidationSplit,
        assistant::prepare_conversations,
        resume::{
            initialize_assistant_stage_on_device, read_resume_manifest,
            save_gpu_training_checkpoint_with_parent,
        },
    };
    let selected = initialize_vulkan(0).unwrap();
    let device = selected.device();
    let root = tempfile::tempdir().unwrap();
    let tokenizer = omega_tokenizer::train_chat_byte_bpe(
        &["Hi ok"],
        &omega_tokenizer::ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
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
    std::fs::create_dir(root.path().join("chat")).unwrap();
    std::fs::write(root.path().join("chat/train.jsonl"), serde_json::json!({"schema_version":1,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"ok"}]}).to_string()).unwrap();
    let source = prepare_conversations(
        root.path(),
        &["chat".into()],
        &tokenizer,
        32,
        ValidationSplit::None,
        42,
    )
    .unwrap()
    .training;
    let base_set = TrainingSet {
        examples: vec![vec![1, 2, 3]],
        files: vec![],
        token_count: 3,
    };
    let mut base = TrainingSession::<_, GpuTraining>::from_source_on_device(
        &config, base_set, 0.001, 42, device,
    )
    .unwrap();
    base.step().unwrap();
    let metadata = CheckpointMetadata::default();
    let parent_path =
        save_gpu_training_checkpoint(root.path(), "base", &base, &tokenizer, &metadata, &selected)
            .unwrap();
    let parent_artifacts: Vec<_> = [
        "model.mpk",
        "optimizer.mpk",
        "manifest.json",
        "resume.json",
        "resume.sha256",
        "tokenizer.json",
        "COMPLETE",
    ]
    .into_iter()
    .map(|name| (name, std::fs::read(parent_path.join(name)).unwrap()))
    .collect();
    let (mut stage, saved_tokenizer, parent) =
        initialize_assistant_stage_on_device::<_, GpuTraining>(
            &parent_path,
            source.clone(),
            0.0001,
            7,
            SessionOptions::default(),
            device,
        )
        .unwrap();
    assert_eq!(stage.progress(), Default::default());
    let input =
        token_tensor::<Gpu>(&[1, 2], config.vocab_size, config.context_length, device).unwrap();
    assert_eq!(
        base.inference_model().forward(input.clone()).into_data(),
        stage.inference_model().forward(input.clone()).into_data()
    );
    let stage_path = save_gpu_training_checkpoint_with_parent(
        root.path(),
        "stage",
        &stage,
        &saved_tokenizer,
        &metadata,
        &selected,
        Some(&parent),
    )
    .unwrap();
    let header = read_resume_manifest(&stage_path).unwrap();
    assert_eq!(header.schema_version, 6);
    assert_eq!(header.parent.as_ref(), Some(&parent));
    assert_eq!(header.progress, Default::default());
    let mut restored =
        load_gpu_training_checkpoint(&stage_path, source, &config, &saved_tokenizer, &selected)
            .unwrap();
    for _ in 0..3 {
        assert_eq!(stage.step().unwrap(), restored.step().unwrap());
    }
    assert_eq!(
        stage.inference_model().forward(input.clone()).into_data(),
        restored.inference_model().forward(input).into_data()
    );
    let continued = save_gpu_training_checkpoint_with_parent(
        root.path(),
        "continued",
        &restored,
        &saved_tokenizer,
        &metadata,
        &selected,
        Some(&parent),
    )
    .unwrap();
    assert_eq!(
        read_resume_manifest(&continued).unwrap().parent.as_ref(),
        Some(&parent)
    );
    for (name, bytes) in parent_artifacts {
        assert_eq!(
            std::fs::read(parent_path.join(name)).unwrap(),
            bytes,
            "parent {name} changed"
        );
    }
}
