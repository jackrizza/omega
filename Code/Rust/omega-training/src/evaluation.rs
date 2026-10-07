//! Fixed-model next-token evaluation, independent of optimization.

use burn::module::Module;
use omega_nn::{Cpu, Gpt, GptConfig, token_tensor};

use crate::{
    TrainingSet,
    dataset::{ExampleSource, checked_target_mask},
};

/// Metrics for one pass over the supplied examples using unchanged weights.
/// Unlike training's pre-update losses, every target sees the same model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvaluationMetrics {
    /// Mean negative log likelihood in natural-log units, weighted by targets.
    pub mean_cross_entropy: f64,
    pub target_count: usize,
    /// exp(mean_cross_entropy); non-finite/overflowing results are errors.
    pub perplexity: f64,
}

/// Evaluate an inference model without changing its weights or training state.
///
/// Each example supplies inputs `ids[..len-1]` and targets `ids[1..]`, with no
/// padding and exactly one next-token shift. Cross entropy is summed in f64 over
/// eligible targets, then divided by their count; short chunks have proportional
/// weight. Source target masks never hide input context. Chunk boundaries reset
/// context/positions, as they do during training.
/// Consequently changing chunk length can change the metric.
///
/// Prefer a document-disjoint validation partition from `split_document_corpus`.
/// This function evaluates exactly the supplied examples; it cannot establish
/// that they were excluded from training or verify tokenizer identity.
///
/// Rejects empty sets, malformed examples, IDs outside the vocabulary, model/
/// configuration mismatches, non-finite logits/loss, target count overflow, and
/// perplexity overflow. Metadata fields `files` and `token_count` are not used
/// as prediction counts. No model initialization, optimizer or RNG is involved.
pub fn evaluate(
    model: &Gpt<Cpu>,
    config: &GptConfig,
    set: &TrainingSet,
) -> Result<EvaluationMetrics, String> {
    evaluate_source(model, config, set)
}

/// Bounded indexed evaluation for eager or cached examples. The model and
/// source are borrowed; only one example and its logits are retained at a time.
pub fn evaluate_source<S: ExampleSource>(
    model: &Gpt<Cpu>,
    config: &GptConfig,
    set: &S,
) -> Result<EvaluationMetrics, String> {
    evaluate_source_on_device(model, config, set, &Default::default())
}

/// Evaluate on an explicitly selected device, retaining host f64 loss accumulation.
pub fn evaluate_on_device<B: burn::tensor::backend::Backend<FloatElem = f32>>(
    model: &Gpt<B>,
    config: &GptConfig,
    set: &TrainingSet,
    device: &B::Device,
) -> Result<EvaluationMetrics, String> {
    evaluate_source_on_device(model, config, set, device)
}

/// Evaluate an indexed source on an explicitly selected device.
pub fn evaluate_source_on_device<
    S: ExampleSource,
    B: burn::tensor::backend::Backend<FloatElem = f32>,
>(
    model: &Gpt<B>,
    config: &GptConfig,
    set: &S,
    device: &B::Device,
) -> Result<EvaluationMetrics, String> {
    evaluate_source_controlled_on_device(model, config, set, device, || Ok(()))
}

/// Cooperative evaluation over the same engine as the unbounded API. The
/// callback runs before/after each source-validation and forward-pass example,
/// and before returning metrics. An error discards partial aggregate metrics;
/// an in-flight source read or forward pass always finishes before the check.
pub fn evaluate_source_controlled_on_device<
    S: ExampleSource,
    B: burn::tensor::backend::Backend<FloatElem = f32>,
>(
    model: &Gpt<B>,
    config: &GptConfig,
    set: &S,
    device: &B::Device,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<EvaluationMetrics, String> {
    check()?;
    model.validate_config(config)?;
    if model.devices().iter().any(|current| current != device) {
        return Err("Evaluation model does not match the selected device".into());
    }
    if set.example_count() == 0 {
        return Err("Evaluation set must contain at least one example".into());
    }
    let mut expected_targets = 0usize;
    for index in 0..set.example_count() {
        check()?;
        let ids = set.example(index)?;
        if ids.len() < 2 || ids.len() - 1 > config.context_length {
            return Err(format!(
                "Evaluation example {index} must contain 2..=context_length+1 tokens"
            ));
        }
        if ids
            .iter()
            .any(|&id| u64::from(id) >= config.vocab_size as u64)
        {
            return Err(format!(
                "Evaluation example {index} contains an ID outside the model vocabulary"
            ));
        }
        let mask = checked_target_mask(set, index, ids.len())?;
        let supervised = mask.as_ref().map_or(ids.len() - 1, |mask| {
            mask.iter().filter(|included| **included).count()
        });
        expected_targets = expected_targets
            .checked_add(supervised)
            .ok_or("Evaluation target count overflows usize")?;
        check()?;
    }
    if expected_targets != set.target_count()? {
        return Err("Evaluation source target count does not match its examples".into());
    }
    check()?;

    let mut aggregate = LossAccumulator::default();
    for index in 0..set.example_count() {
        check()?;
        let ids = set.example(index)?;
        let mask = checked_target_mask(set, index, ids.len())?;
        let inputs = token_tensor::<B>(
            &ids[..ids.len() - 1],
            config.vocab_size,
            config.context_length,
            device,
        )?;
        let logits = model
            .forward(inputs)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("Cannot read evaluation logits for example {index}: {e:?}"))?;
        aggregate
            .add_masked_logits(&logits, &ids[1..], config.vocab_size, mask.as_deref())
            .map_err(|e| format!("Evaluation example {index}: {e}"))?;
        check()?;
    }
    debug_assert_eq!(aggregate.target_count, expected_targets);
    let metrics = aggregate.finish()?;
    check()?;
    Ok(metrics)
}

#[derive(Default)]
struct LossAccumulator {
    negative_log_likelihood: f64,
    target_count: usize,
}

impl LossAccumulator {
    #[cfg(test)]
    fn add_logits(
        &mut self,
        logits: &[f32],
        targets: &[u32],
        vocab_size: usize,
    ) -> Result<(), String> {
        self.add_masked_logits(logits, targets, vocab_size, None)
    }

    fn add_masked_logits(
        &mut self,
        logits: &[f32],
        targets: &[u32],
        vocab_size: usize,
        mask: Option<&[bool]>,
    ) -> Result<(), String> {
        let expected = targets
            .len()
            .checked_mul(vocab_size)
            .ok_or("Evaluation logit shape overflows usize")?;
        if vocab_size == 0
            || logits.len() != expected
            || targets.is_empty()
            || mask.is_some_and(|mask| mask.len() != targets.len())
        {
            return Err(
                "Evaluation logits must have one nonempty vocabulary row per target".into(),
            );
        }
        for (index, (row, &target)) in logits.chunks_exact(vocab_size).zip(targets).enumerate() {
            if u64::from(target) >= vocab_size as u64 {
                return Err(format!(
                    "Target {index} is outside the evaluation vocabulary"
                ));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(format!("Non-finite evaluation logits at target {index}"));
            }
            if mask.is_some_and(|mask| !mask[index]) {
                continue;
            }
            let max = row
                .iter()
                .copied()
                .map(f64::from)
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = row
                .iter()
                .map(|&value| (f64::from(value) - max).exp())
                .sum();
            // Subtract the target before adding log(sum), avoiding cancellation
            // when every logit has a large common offset.
            let loss = (max - f64::from(row[target as usize])) + sum.ln();
            let total = self.negative_log_likelihood + loss;
            if !loss.is_finite() || !total.is_finite() {
                return Err("Non-finite accumulated evaluation loss".into());
            }
            let count = self
                .target_count
                .checked_add(1)
                .ok_or("Evaluation target count overflows usize")?;
            self.negative_log_likelihood = total;
            self.target_count = count;
        }
        Ok(())
    }

    fn finish(self) -> Result<EvaluationMetrics, String> {
        if self.target_count == 0 {
            return Err("Evaluation requires at least one prediction target".into());
        }
        let mean_cross_entropy = self.negative_log_likelihood / self.target_count as f64;
        if !mean_cross_entropy.is_finite() || mean_cross_entropy < 0.0 {
            return Err("Non-finite or negative mean evaluation cross entropy".into());
        }
        let perplexity = mean_cross_entropy.exp();
        if !perplexity.is_finite() {
            return Err(format!(
                "Evaluation perplexity overflows f64 for mean cross entropy {mean_cross_entropy}"
            ));
        }
        Ok(EvaluationMetrics {
            mean_cross_entropy,
            target_count: self.target_count,
            perplexity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TrainingSession, ValidationSplit, load_document_corpus, split_document_corpus};
    use burn::{
        module::{Module, ModuleMapper, ParamId},
        tensor::Tensor,
    };
    use std::{fs, path::Path};

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 4,
            context_length: 4,
            d_model: 4,
            num_heads: 1,
            num_layers: 1,
            d_ff: 8,
        }
    }

    fn set(examples: Vec<Vec<u32>>) -> TrainingSet {
        TrainingSet {
            examples,
            files: vec![],
            token_count: usize::MAX,
        }
    }

    fn logits(model: &Gpt<Cpu>) -> Vec<f32> {
        model
            .forward(token_tensor::<Cpu>(&[0, 1, 2], 4, 4, &Default::default()).unwrap())
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn controlled_evaluation_discards_partial_metrics_and_preserves_weights() {
        struct ObservedSource {
            reads: std::cell::Cell<usize>,
            source: TrainingSet,
        }
        impl ExampleSource for ObservedSource {
            fn example_count(&self) -> usize {
                self.source.example_count()
            }
            fn target_count(&self) -> Result<usize, String> {
                self.source.target_count()
            }
            fn example(&self, index: usize) -> Result<Vec<u32>, String> {
                self.reads.set(self.reads.get() + 1);
                self.source.example(index)
            }
            fn identity(&self) -> Result<String, String> {
                self.source.identity()
            }
        }
        let config = config();
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let before = logits(&model);
        let source = ObservedSource {
            reads: Default::default(),
            source: set(vec![vec![0, 1], vec![2, 3]]),
        };
        let error = evaluate_source_controlled_on_device(
            &model,
            &config,
            &source,
            &Default::default(),
            || {
                // Two validation reads followed by the first model-evaluation read.
                if source.reads.get() > source.example_count() {
                    Err("stop after first evaluation example".into())
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error, "stop after first evaluation example");
        assert_eq!(
            source.reads.get(),
            3,
            "No second forward example after stop"
        );
        assert_eq!(logits(&model), before);
        let unbounded =
            evaluate_source_on_device(&model, &config, &source, &Default::default()).unwrap();
        let controlled = evaluate_source_controlled_on_device(
            &model,
            &config,
            &source,
            &Default::default(),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(controlled, unbounded);
    }

    #[test]
    fn known_logits_and_targets_use_stable_cross_entropy_and_target_weighting() {
        let mut aggregate = LossAccumulator::default();
        // One easy target versus three uncertain targets: average by 4, not 2.
        aggregate
            .add_logits(&[0.25_f32.ln(), 0.75_f32.ln()], &[1], 2)
            .unwrap();
        aggregate.add_logits(&[0.0; 6], &[0, 1, 0], 2).unwrap();
        let metrics = aggregate.finish().unwrap();
        let expected = (-0.75_f64.ln() + 3.0 * 2.0_f64.ln()) / 4.0;
        assert_eq!(metrics.target_count, 4);
        assert!((metrics.mean_cross_entropy - expected).abs() < 1e-8);
        assert!((metrics.perplexity - expected.exp()).abs() < 1e-8);
        let mut extreme = LossAccumulator::default();
        extreme.add_logits(&[f32::MAX, f32::MAX], &[0], 2).unwrap();
        assert_eq!(extreme.finish().unwrap().perplexity, 2.0);
    }

    #[test]
    fn unequal_examples_match_separate_target_weighted_passes_without_mutation() {
        let config = config();
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let before = logits(&model);
        let short = evaluate(&model, &config, &set(vec![vec![0, 1]])).unwrap();
        let long = evaluate(&model, &config, &set(vec![vec![2, 3, 0, 1]])).unwrap();
        let combined = evaluate(&model, &config, &set(vec![vec![0, 1], vec![2, 3, 0, 1]])).unwrap();
        assert_eq!(combined.target_count, 4);
        let expected = (short.mean_cross_entropy + 3.0 * long.mean_cross_entropy) / 4.0;
        assert!((combined.mean_cross_entropy - expected).abs() < 1e-12);
        assert_eq!(logits(&model), before);
        assert_eq!(
            evaluate(&model, &config, &set(vec![vec![0, 1]])).unwrap(),
            short
        );
    }

    #[test]
    fn evaluating_session_snapshot_preserves_optimizer_and_cursor_continuation() {
        let config = config();
        let mut session =
            TrainingSession::new(&config, set(vec![vec![0, 1, 2]]), 0.01, 42).unwrap();
        session.step().unwrap();
        let mut control = session.clone();
        let progress = session.progress();
        evaluate(&session.inference_model(), &config, &set(vec![vec![2, 3]])).unwrap();
        assert_eq!(session.progress(), progress);
        assert_eq!(session.step().unwrap(), control.step().unwrap());
        assert_eq!(
            logits(&session.inference_model()),
            logits(&control.inference_model())
        );
    }

    #[test]
    fn validation_partition_evaluates_only_its_document_targets() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("text")).unwrap();
        fs::write(
            temp.path().join("text/one.txt"),
            "hello world omega tokenizer this is",
        )
        .unwrap();
        fs::write(temp.path().join("text/two.txt"), "hello world").unwrap();
        let tokenizer = omega_tokenizer::Tokens::new(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/wordlevel.json"),
        )
        .unwrap();
        let corpus = load_document_corpus(temp.path(), &["text".into()], &tokenizer).unwrap();
        let split = split_document_corpus(&corpus, 4, ValidationSplit::Count(1), 42).unwrap();
        assert_ne!(
            split.training.documents[0].id,
            split.validation.documents[0].id
        );
        let config = GptConfig {
            vocab_size: 13,
            ..config()
        };
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let metrics = evaluate(&model, &config, &split.validation.set).unwrap();
        let source = corpus
            .documents
            .iter()
            .find(|doc| doc.id == split.validation.documents[0].id)
            .unwrap();
        assert_eq!(metrics.target_count, source.tokens.len() - 1);
        assert_ne!(metrics.target_count, 6); // Whole corpus has six targets.
    }

    #[test]
    fn rejects_empty_invalid_examples_configurations_and_nonfinite_models() {
        let config = config();
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        for examples in [
            vec![],
            vec![vec![]],
            vec![vec![0]],
            vec![vec![0, 4]],
            vec![vec![4, 0]],
            vec![vec![0; 6]],
        ] {
            assert!(evaluate(&model, &config, &set(examples)).is_err());
        }
        let data = set(vec![vec![0, 1]]);
        let mut mismatch = config.clone();
        mismatch.context_length += 1;
        assert!(evaluate(&model, &mismatch, &data).is_err());
        mismatch.vocab_size = 0;
        assert!(evaluate(&model, &mismatch, &data).is_err());
        struct Nonfinite;
        impl ModuleMapper<Cpu> for Nonfinite {
            fn map_float<const D: usize>(
                &mut self,
                _id: ParamId,
                tensor: Tensor<Cpu, D>,
            ) -> Tensor<Cpu, D> {
                tensor.mul_scalar(f32::NAN)
            }
        }
        let nonfinite = model.map(&mut Nonfinite);
        assert!(
            evaluate(&nonfinite, &config, &data)
                .unwrap_err()
                .contains("Non-finite evaluation logits")
        );
    }

    #[test]
    fn rejects_nonfinite_logits_overflow_and_invalid_reducer_shapes() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                LossAccumulator::default()
                    .add_logits(&[value, 0.0], &[0], 2)
                    .is_err()
            );
        }
        assert!(
            LossAccumulator::default()
                .add_logits(&[0.0], &[0], 2)
                .is_err()
        );
        assert!(LossAccumulator::default().add_logits(&[], &[0], 0).is_err());
        assert!(
            LossAccumulator::default()
                .add_logits(&[0.0, 0.0], &[2], 2)
                .is_err()
        );
        assert!(LossAccumulator::default().finish().is_err());
        let mut overflow = LossAccumulator {
            target_count: usize::MAX,
            negative_log_likelihood: 0.0,
        };
        assert!(
            overflow
                .add_logits(&[0.0, 0.0], &[0], 2)
                .unwrap_err()
                .contains("count overflows")
        );
        let mut huge_loss = LossAccumulator::default();
        huge_loss.add_logits(&[1000.0, 0.0], &[1], 2).unwrap();
        assert!(
            huge_loss
                .finish()
                .unwrap_err()
                .contains("perplexity overflows")
        );
        assert!(
            LossAccumulator {
                target_count: 1,
                negative_log_likelihood: f64::INFINITY
            }
            .finish()
            .is_err()
        );
    }
}
