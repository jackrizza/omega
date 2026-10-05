//! Assistant eligibility is separate from visible context and padding.
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{Bool, Tensor, TensorData},
};
use omega_nn::{Cpu, GptConfig, TrainingBackend, masked_cross_entropy, token_tensor};
use omega_training::{
    ExampleSource, TrainingSession,
    batching::{BatchConfig, collate},
    checkpoint::{CheckpointMetadata, sha256_bytes},
    dataset::ASSISTANT_TARGETS_OBJECTIVE,
    evaluation::evaluate_source,
    resume::{load_training_checkpoint, save_training_checkpoint},
    sampling::{SamplingGroup, SamplingPolicy},
    trainer::SessionOptions,
};

#[derive(Clone)]
struct Masked {
    ids: Vec<Vec<u32>>,
    masks: Vec<Vec<bool>>,
}

impl ExampleSource for Masked {
    fn example_count(&self) -> usize {
        self.ids.len()
    }
    fn target_count(&self) -> Result<usize, String> {
        Ok(self
            .masks
            .iter()
            .flatten()
            .filter(|included| **included)
            .count())
    }
    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        self.ids.get(index).cloned().ok_or("bad index".into())
    }
    fn target_mask(&self, index: usize) -> Result<Option<Vec<bool>>, String> {
        self.masks
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
            &serde_json::to_vec(&(self.objective(), &self.ids, &self.masks)).unwrap(),
        ))
    }
}

fn source() -> Masked {
    Masked {
        ids: vec![vec![1, 2, 3, 4], vec![5, 6], vec![7, 8, 9]],
        masks: vec![vec![false, true, true], vec![true], vec![false, true]],
    }
}
fn config() -> GptConfig {
    GptConfig {
        vocab_size: 13,
        context_length: 4,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    }
}
fn tokenizer() -> omega_tokenizer::Tokens {
    omega_tokenizer::Tokens::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"),
    )
    .unwrap()
}
fn record(model: omega_nn::Gpt<Cpu>) -> Vec<u8> {
    <_ as Recorder<Cpu>>::record(
        &NamedMpkBytesRecorder::<FullPrecisionSettings>::default(),
        model.into_record(),
        (),
    )
    .unwrap()
}

#[test]
fn collation_shifts_mask_once_keeps_context_and_excludes_padding() {
    let batch = collate(
        &source(),
        &[0, 1],
        &config(),
        &BatchConfig {
            batch_size: 2,
            max_batch_tokens: 6,
        },
        0,
    )
    .unwrap();
    assert_eq!(batch.target_count(), 3);
    let tensors = batch.into_tensors::<Cpu>(&Default::default());
    assert_eq!(
        tensors.targets.into_data().to_vec::<i64>().unwrap(),
        [2, 3, 4, 6, 0, 0]
    );
    assert_eq!(
        tensors.valid.into_data().to_vec::<bool>().unwrap(),
        [true, true, true, true, false, false]
    );
    assert_eq!(
        tensors.loss_valid.into_data().to_vec::<bool>().unwrap(),
        [false, true, true, true, false, false]
    );
}

#[test]
fn single_row_loss_matches_selected_logits_and_context_remains_visible() {
    let device = Default::default();
    let mut session = TrainingSession::from_source(&config(), source(), 0.01, 42).unwrap();
    let batch = collate(&source(), &[0], &config(), &BatchConfig::default(), 0)
        .unwrap()
        .into_tensors::<TrainingBackend>(&device);
    let logits = session
        .model()
        .forward_masked(batch.inputs, batch.valid)
        .unwrap();
    let expected = masked_cross_entropy(logits.clone(), batch.targets, batch.loss_valid)
        .unwrap()
        .loss
        .into_scalar();
    let before = logits.into_data().to_vec::<f32>().unwrap();
    let changed = session
        .model()
        .forward(token_tensor::<TrainingBackend>(&[12, 2, 3], 13, 4, &device).unwrap())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_ne!(
        &before[13..],
        &changed[13..],
        "Ignored-target context must still influence assistant predictions"
    );
    let event = session.step().unwrap();
    assert_eq!(event.target_count, 2);
    assert_eq!(event.pre_update_loss, expected);
}

#[test]
fn ignored_logits_have_zero_direct_gradient_in_source_aligned_loss() {
    let device = Default::default();
    let batch = collate(&source(), &[0], &config(), &BatchConfig::default(), 0)
        .unwrap()
        .into_tensors::<TrainingBackend>(&device);
    let logits = Tensor::<TrainingBackend, 3>::zeros([1, 3, 13], &device).require_grad();
    let loss = masked_cross_entropy(logits.clone(), batch.targets, batch.loss_valid).unwrap();
    let grad = logits
        .grad(&loss.loss.backward())
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(&grad[..13], &[0.0; 13]);
    assert!(grad[13..].iter().any(|value| *value != 0.0));
}

#[test]
fn fixed_evaluation_weights_only_assistant_targets_without_changing_weights() {
    let model = config().init::<Cpu>(&Default::default()).unwrap();
    let before = record(model.clone());
    let metrics = evaluate_source(&model, &config(), &source()).unwrap();
    let mut sum = 0.0;
    for index in 0..source().example_count() {
        let all = source();
        let one = Masked {
            ids: vec![all.ids[index].clone()],
            masks: vec![all.masks[index].clone()],
        };
        let item = evaluate_source(&model, &config(), &one).unwrap();
        sum += item.mean_cross_entropy * item.target_count as f64;
    }
    assert_eq!(metrics.target_count, 4);
    assert!((metrics.mean_cross_entropy - sum / 4.0).abs() < 1e-12);
    assert_eq!(record(model), before);
}

#[test]
fn invalid_masks_and_objectives_fail_before_training_or_evaluation() {
    let model = config().init::<Cpu>(&Default::default()).unwrap();
    for mask in [vec![], vec![true], vec![false; 3], vec![true; 4]] {
        let mut invalid = source();
        invalid.masks[0] = mask;
        assert!(TrainingSession::from_source(&config(), invalid.clone(), 0.01, 42).is_err());
        assert!(evaluate_source(&model, &config(), &invalid).is_err());
        assert!(collate(&invalid, &[0], &config(), &BatchConfig::default(), 0).is_err());
    }
    struct Missing(&'static str);
    impl ExampleSource for Missing {
        fn example_count(&self) -> usize {
            1
        }
        fn target_count(&self) -> Result<usize, String> {
            Ok(1)
        }
        fn example(&self, _: usize) -> Result<Vec<u32>, String> {
            Ok(vec![1, 2])
        }
        fn identity(&self) -> Result<String, String> {
            Ok("0".repeat(64))
        }
        fn objective(&self) -> &str {
            self.0
        }
        fn target_mask(&self, _: usize) -> Result<Option<Vec<bool>>, String> {
            Ok((self.0 == omega_training::dataset::ALL_TARGETS_OBJECTIVE).then(|| vec![true]))
        }
    }
    assert!(
        TrainingSession::from_source(&config(), Missing(ASSISTANT_TARGETS_OBJECTIVE), 0.01, 42)
            .err()
            .unwrap()
            .contains("requires a target mask")
    );
    assert!(
        TrainingSession::from_source(&config(), Missing("unknown-objective"), 0.01, 42)
            .err()
            .unwrap()
            .contains("Unsupported source objective")
    );
    assert!(
        TrainingSession::from_source(
            &config(),
            Missing(omega_training::dataset::ALL_TARGETS_OBJECTIVE),
            0.01,
            42,
        )
        .err()
        .unwrap()
        .contains("must not supply an explicit target mask")
    );
}

#[test]
fn partial_batches_weighted_sampler_and_disk_resume_keep_supervised_counts() {
    let root = tempfile::tempdir().unwrap();
    for sampling in [
        SamplingPolicy::Fixed,
        SamplingPolicy::Shuffle,
        SamplingPolicy::Weighted {
            groups: vec![SamplingGroup {
                name: "all".into(),
                indices: vec![0, 1, 2],
                weight: 1,
            }],
            samples_per_epoch: 5,
        },
    ] {
        for batch_size in [1, 2] {
            let options = SessionOptions {
                sampling: sampling.clone(),
                batching: BatchConfig {
                    batch_size,
                    max_batch_tokens: 16,
                },
                ..Default::default()
            };
            let mut uninterrupted =
                TrainingSession::from_source_with_options(&config(), source(), 0.01, 42, options)
                    .unwrap();
            let first = uninterrupted.step().unwrap();
            assert!(first.target_count > 0);
            let path = save_training_checkpoint(
                root.path(),
                "masked",
                &uninterrupted,
                &tokenizer(),
                &CheckpointMetadata::default(),
            )
            .unwrap();
            let parent = std::fs::read(path.join("model.mpk")).unwrap();
            let mut resumed =
                load_training_checkpoint(&path, source(), &config(), &tokenizer()).unwrap();
            let mut changed = source();
            changed.masks[0] = vec![true, false, true]; // same total, different eligibility
            assert!(
                load_training_checkpoint(&path, changed, &config(), &tokenizer())
                    .err()
                    .unwrap()
                    .contains("identity")
            );
            for _ in 0..5 {
                assert_eq!(uninterrupted.step().unwrap(), resumed.step().unwrap());
            }
            assert_eq!(uninterrupted.progress(), resumed.progress());
            assert_eq!(
                record(uninterrupted.inference_model()),
                record(resumed.inference_model())
            );
            assert_eq!(std::fs::read(path.join("model.mpk")).unwrap(), parent);
        }
    }
}

#[test]
fn mask_selected_loss_is_equal_to_explicit_gather() {
    let device = Default::default();
    let logits = Tensor::<Cpu, 3>::from_data(
        TensorData::new(
            (0..39).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            [1, 3, 13],
        ),
        &device,
    );
    let full = masked_cross_entropy(
        logits.clone(),
        token_tensor::<Cpu>(&[2, 3, 4], 13, 4, &device).unwrap(),
        Tensor::<Cpu, 2, Bool>::from_data([[false, true, true]], &device),
    )
    .unwrap();
    let selected = masked_cross_entropy(
        logits.slice([0..1, 1..3, 0..13]),
        token_tensor::<Cpu>(&[3, 4], 13, 4, &device).unwrap(),
        Tensor::<Cpu, 2, Bool>::from_data([[true, true]], &device),
    )
    .unwrap();
    assert_eq!(full.loss.into_scalar(), selected.loss.into_scalar());
}
