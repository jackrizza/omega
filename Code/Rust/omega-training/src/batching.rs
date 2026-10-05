//! Bounded right-padding of indexed, document-local next-token examples.
//!
//! Selection/order belongs to the caller. Collation never joins examples, drops
//! remainders, truncates tokens, or guesses tokenizer special IDs. The padded
//! position limit bounds this module's retained buffers, not an ExampleSource's
//! transient allocation, attention matrices, or other model/backend memory.

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use omega_nn::GptConfig;

use crate::dataset::{ExampleSource, checked_target_mask};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchConfig {
    /// Maximum examples in an update; a smaller final batch is retained.
    pub batch_size: usize,
    /// Maximum padded input positions: rows times longest input length.
    /// Targets and validity use the same shape; this is not a byte limit.
    pub max_batch_tokens: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            max_batch_tokens: 65_536,
        }
    }
}

impl BatchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 || self.max_batch_tokens == 0 {
            return Err("batch_size and max_batch_tokens must be greater than zero".into());
        }
        Ok(())
    }
}

/// Already shifted, checked host buffers. Private fields preserve shape and
/// nonempty-right-prefix invariants until conversion to backend tensors.
#[derive(Debug)]
pub struct Batch {
    inputs: Vec<i64>,
    targets: Vec<i64>,
    valid: Vec<bool>,
    loss_valid: Vec<bool>,
    batch_size: usize,
    sequence_length: usize,
    target_count: usize,
}

impl Batch {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn target_count(&self) -> usize {
        self.target_count
    }

    pub fn into_tensors<B: Backend>(self, device: &B::Device) -> BatchTensors<B> {
        let shape = [self.batch_size, self.sequence_length];
        BatchTensors {
            inputs: Tensor::from_data(TensorData::new(self.inputs, shape), device),
            targets: Tensor::from_data(TensorData::new(self.targets, shape), device),
            valid: Tensor::from_data(TensorData::new(self.valid, shape), device),
            loss_valid: Tensor::from_data(TensorData::new(self.loss_valid, shape), device),
            target_count: self.target_count,
        }
    }
}

/// All tensors are `[batch, sequence]`. `valid` marks real inputs for attention;
/// `loss_valid` marks supervised next-token targets and excludes padding.
#[derive(Debug)]
pub struct BatchTensors<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub valid: Tensor<B, 2, Bool>,
    pub loss_valid: Tensor<B, 2, Bool>,
    pub target_count: usize,
}

fn filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|error| format!("Cannot allocate batch buffer: {error}"))?;
    values.resize(length, value);
    Ok(values)
}

/// Collate exactly the selected examples, preserving supplied order and any
/// intentional repeated indices (for sampling with replacement). Each example
/// must contain 2..=context_length+1 IDs; labels are shifted exactly once here.
///
/// `pad_id` must be an existing vocabulary ID, used only as ignored storage
/// filler; it need not be a tokenizer padding/special token. Every input and
/// target ID is validated. Oversized batches fail without partial collation.
/// Examples are read one at a time and the padded bound is checked immediately,
/// before retaining another example or allocating padded buffers. Backend tensor
/// creation is separate and may still panic on backend allocation failures.
pub fn collate<S: ExampleSource + ?Sized>(
    source: &S,
    indices: &[usize],
    model: &GptConfig,
    config: &BatchConfig,
    pad_id: u32,
) -> Result<Batch, String> {
    model.validate()?;
    config.validate()?;
    let rows = indices.len();
    if rows == 0 || rows > config.batch_size {
        return Err("Selected batch must contain 1..=batch_size examples".into());
    }
    if rows > config.max_batch_tokens {
        return Err("Even one input position per example exceeds max_batch_tokens".into());
    }
    if u64::from(pad_id) >= model.vocab_size as u64 {
        return Err(format!(
            "Batch padding ID {pad_id} is outside the model vocabulary"
        ));
    }
    for &index in indices {
        if index >= source.example_count() {
            return Err(format!("Batch example index {index} is out of bounds"));
        }
    }
    let mut examples = Vec::new();
    examples
        .try_reserve_exact(rows)
        .map_err(|error| format!("Cannot allocate batch example index: {error}"))?;
    let mut sequence_length = 0;
    let mut target_count = 0usize;
    for &index in indices {
        let ids = source
            .example(index)
            .map_err(|error| format!("Cannot read batch example {index}: {error}"))?;
        if ids.len() < 2 || ids.len() - 1 > model.context_length {
            return Err(format!(
                "Batch example {index} must contain 2..=context_length+1 tokens"
            ));
        }
        sequence_length = sequence_length.max(ids.len() - 1);
        let positions = rows
            .checked_mul(sequence_length)
            .ok_or("Padded batch size overflows usize")?;
        if positions > config.max_batch_tokens {
            return Err(format!(
                "Padded batch requires {positions} positions, exceeding max_batch_tokens {}",
                config.max_batch_tokens
            ));
        }
        for (position, &id) in ids.iter().enumerate() {
            if u64::from(id) >= model.vocab_size as u64 {
                return Err(format!(
                    "Batch example {index} token {position} ID {id} is outside the model vocabulary"
                ));
            }
        }
        let mask = checked_target_mask(source, index, ids.len())?;
        let supervised = mask.as_ref().map_or(ids.len() - 1, |mask| {
            mask.iter().filter(|included| **included).count()
        });
        target_count = target_count
            .checked_add(supervised)
            .ok_or("Batch target count overflows usize")?;
        examples.push((ids, mask));
    }
    // Multiplication was checked for this final longest length above.
    let positions = rows * sequence_length;
    let mut inputs = filled(positions, i64::from(pad_id))?;
    let mut targets = filled(positions, i64::from(pad_id))?;
    let mut valid = filled(positions, false)?;
    let mut loss_valid = filled(positions, false)?;
    for (row, (ids, mask)) in examples.iter().enumerate() {
        for (column, pair) in ids.windows(2).enumerate() {
            let position = row * sequence_length + column;
            inputs[position] = i64::from(pair[0]);
            targets[position] = i64::from(pair[1]);
            valid[position] = true;
            loss_valid[position] = mask.as_ref().is_none_or(|mask| mask[column]);
        }
    }
    Ok(Batch {
        inputs,
        targets,
        valid,
        loss_valid,
        batch_size: rows,
        sequence_length,
        target_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TokenizedDocument;
    use crate::{DocumentCorpus, TrainingSet, ValidationSplit, split_document_corpus};
    use std::cell::Cell;

    fn model() -> GptConfig {
        GptConfig {
            vocab_size: 20,
            context_length: 4,
            d_model: 4,
            num_heads: 1,
            num_layers: 1,
            d_ff: 8,
        }
    }

    fn source(examples: Vec<Vec<u32>>) -> TrainingSet {
        TrainingSet {
            examples,
            files: vec![],
            token_count: 0,
        }
    }

    fn config() -> BatchConfig {
        BatchConfig {
            batch_size: 3,
            max_batch_tokens: 12,
        }
    }

    #[test]
    fn variable_lengths_shift_once_and_right_pad_with_explicit_filler() {
        // The filler ID is also a real token; validity comes from lengths only.
        let source = source(vec![vec![1, 2], vec![3, 4, 5, 6], vec![19, 8, 19]]);
        // Six real targets still require nine padded positions.
        let tight = BatchConfig {
            max_batch_tokens: 6,
            ..config()
        };
        assert!(
            collate(&source, &[1, 0, 2], &model(), &tight, 19)
                .unwrap_err()
                .contains("9 positions")
        );
        let batch = collate(&source, &[1, 0, 2], &model(), &config(), 19).unwrap();
        assert_eq!(
            (
                batch.batch_size(),
                batch.sequence_length(),
                batch.target_count()
            ),
            (3, 3, 6)
        );
        let tensors = batch.into_tensors::<omega_nn::Cpu>(&Default::default());
        assert_eq!(tensors.inputs.dims(), [3, 3]);
        assert_eq!(tensors.targets.dims(), [3, 3]);
        assert_eq!(tensors.valid.dims(), [3, 3]);
        assert_eq!(tensors.target_count, 6);
        assert_eq!(
            tensors.inputs.into_data().to_vec::<i64>().unwrap(),
            [3, 4, 5, 1, 19, 19, 19, 8, 19]
        );
        assert_eq!(
            tensors.targets.into_data().to_vec::<i64>().unwrap(),
            [4, 5, 6, 2, 19, 19, 8, 19, 19]
        );
        assert_eq!(
            tensors.valid.into_data().to_vec::<bool>().unwrap(),
            [true, true, true, true, false, false, true, true, false]
        );
    }

    #[test]
    fn chunk_pairs_and_final_partial_batch_are_retained_without_crossing_documents() {
        let root = tempfile::tempdir().unwrap();
        let documents = [
            vec![1, 2, 3, 4, 5, 6, 7],
            vec![10, 11],
            vec![12, 13, 14, 15],
        ];
        let corpus = DocumentCorpus {
            documents: documents
                .iter()
                .enumerate()
                .map(|(index, tokens)| TokenizedDocument {
                    id: format!("{index}.txt").into(),
                    path: root.path().join(format!("{index}.txt")),
                    tokens: tokens.clone(),
                })
                .collect(),
        };
        let set = split_document_corpus(&corpus, 4, ValidationSplit::None, 42)
            .unwrap()
            .training
            .set;
        assert_eq!(set.example_count(), 4);
        let mut actual_pairs = Vec::new();
        let mut row_counts = Vec::new();
        for indices in (0..set.example_count()).collect::<Vec<_>>().chunks(3) {
            let batch = collate(&set, indices, &model(), &config(), 0).unwrap();
            row_counts.push(batch.batch_size());
            for ((input, target), valid) in
                batch.inputs.iter().zip(&batch.targets).zip(&batch.valid)
            {
                if *valid {
                    actual_pairs.push([*input as u32, *target as u32]);
                }
            }
        }
        let expected_pairs: Vec<_> = documents
            .iter()
            .flat_map(|ids| ids.windows(2).map(|pair| [pair[0], pair[1]]))
            .collect();
        assert_eq!(actual_pairs, expected_pairs);
        assert_eq!(row_counts, [3, 1]);
    }

    #[test]
    fn repeated_indices_preserve_requested_sampling_and_batch_one_has_no_padding() {
        let source = source(vec![vec![1, 2, 3], vec![4, 5]]);
        let batch = collate(&source, &[1, 0, 1], &model(), &config(), 0).unwrap();
        assert_eq!(batch.inputs, [4, 0, 1, 2, 4, 0]);
        assert_eq!(batch.target_count(), 4);
        let single = collate(&source, &[0], &model(), &BatchConfig::default(), 0).unwrap();
        assert_eq!(single.inputs, [1, 2]);
        assert_eq!(single.targets, [2, 3]);
        assert_eq!(single.valid, [true, true]);
    }

    #[test]
    fn invalid_configuration_selection_lengths_and_ids_are_errors() {
        let good = source(vec![vec![1, 2]]);
        for config in [
            BatchConfig {
                batch_size: 0,
                ..config()
            },
            BatchConfig {
                max_batch_tokens: 0,
                ..config()
            },
        ] {
            assert!(collate(&good, &[0], &model(), &config, 0).is_err());
        }
        for indices in [vec![], vec![1], vec![0; 4]] {
            assert!(collate(&good, &indices, &model(), &config(), 0).is_err());
        }
        for ids in [vec![], vec![1], vec![1; 6], vec![20, 1], vec![1, 20]] {
            assert!(collate(&source(vec![ids]), &[0], &model(), &config(), 0).is_err());
        }
        assert!(collate(&good, &[0], &model(), &config(), 20).is_err());
        let mut invalid_model = model();
        invalid_model.context_length = 0;
        assert!(collate(&good, &[0], &invalid_model, &config(), 0).is_err());
        assert!(
            serde_json::from_str::<BatchConfig>(
                r#"{"batch_size":1,"max_batch_tokens":4,"extra":0}"#
            )
            .is_err()
        );
        assert_eq!(
            serde_json::from_str::<BatchConfig>(&serde_json::to_string(&config()).unwrap())
                .unwrap(),
            config()
        );
    }

    struct ObservedSource {
        reads: Cell<usize>,
        fail: bool,
    }

    impl ExampleSource for ObservedSource {
        fn example_count(&self) -> usize {
            3
        }
        fn target_count(&self) -> Result<usize, String> {
            Ok(12)
        }
        fn identity(&self) -> Result<String, String> {
            Ok("unused".into())
        }
        fn example(&self, _: usize) -> Result<Vec<u32>, String> {
            self.reads.set(self.reads.get() + 1);
            if self.fail {
                Err("simulated cache read failure".into())
            } else {
                Ok(vec![1; 5])
            }
        }
    }

    #[test]
    fn padded_budget_is_checked_before_reading_more_examples_and_errors_propagate() {
        let source = ObservedSource {
            reads: Cell::new(0),
            fail: false,
        };
        let mut limit = config();
        limit.max_batch_tokens = 2;
        assert!(collate(&source, &[0, 1, 2], &model(), &limit, 0).is_err());
        assert_eq!(source.reads.get(), 0);
        limit.max_batch_tokens = 11;
        assert!(
            collate(&source, &[0, 1, 2], &model(), &limit, 0)
                .unwrap_err()
                .contains("12 positions")
        );
        assert_eq!(source.reads.get(), 1);
        // Equality at the configured bound is accepted.
        assert_eq!(
            collate(&source, &[0, 1, 2], &model(), &config(), 0)
                .unwrap()
                .target_count(),
            12
        );
        let failing = ObservedSource {
            reads: Cell::new(0),
            fail: true,
        };
        let error = collate(&failing, &[2], &model(), &config(), 0).unwrap_err();
        assert!(error.contains("example 2") && error.contains("simulated cache read failure"));
        assert_eq!(failing.reads.get(), 1);
    }
}
