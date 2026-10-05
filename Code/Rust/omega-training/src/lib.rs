//! Folder-based training sets and numbered model checkpoints for Omega GPTs.

pub mod assistant;
pub mod batching;
pub mod cache;
pub mod checkpoint;
pub mod checkpoint_catalog;
pub mod cpu;
pub mod cuda;
pub mod dataset;
pub mod evaluation;
pub mod gpu;
pub mod metrics;
mod numerical;
pub mod operations;
pub mod optimization;
pub mod release;
pub mod resume;
pub mod run_control;
pub mod sampling;
pub mod selection;
pub mod trainer;

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub(crate) fn host_kernel() -> Result<String, String> {
    let system = std::fs::read_to_string("/proc/sys/kernel/ostype")
        .map_err(|e| format!("Cannot identify Linux GPU host kernel: {e}"))?;
    let release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map_err(|e| format!("Cannot identify Linux GPU host release: {e}"))?;
    Ok(format!("{} {}", system.trim(), release.trim()))
}

pub use evaluation::{
    EvaluationMetrics, evaluate, evaluate_on_device, evaluate_source, evaluate_source_on_device,
};
pub use metrics::JsonlMetrics;
pub use resume::{
    ResumeManifest, load_training_checkpoint, read_resume_manifest, save_training_checkpoint,
};

pub use checkpoint::{
    CheckpointManifest, CheckpointMetadata, load_checkpoint, read_checkpoint_manifest,
    reserve_run_directory, save_checkpoint, save_checkpoint_with_metadata,
};
pub use dataset::{
    DatasetFormat, DatasetSplit, DocumentCorpus, DocumentInfo, DocumentPartition, ExampleSource,
    TextDocument, TokenizedDocument, TrainingSet, ValidationSplit, build_training_set,
    load_document_corpus, load_document_corpus_with_format, load_text_documents,
    split_document_corpus,
};

use std::path::PathBuf;

use omega_nn::{Cpu, Gpt, GptConfig};
pub use optimization::OptimizationOptions;
pub use trainer::{EpochSummary, SessionOptions, TrainingProgress, TrainingSession, UpdateEvent};

/// Default paths refer to the source checkout at compile time.
pub fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

pub struct TrainingResult {
    pub model: Gpt<Cpu>,
    /// Token-weighted mean of pre-update losses for each epoch, not validation loss.
    pub epoch_losses: Vec<f32>,
}

/// Train one model across all examples, keeping the same Adam state throughout.
/// Each epoch visits examples in deterministic file/chunk order with batch size 1.
/// The entire set is held in memory; no padding or cross-document targets are used.
/// This convenience wrapper copies the set into a fresh `TrainingSession`. Use
/// the session directly to transfer ownership and advance training across calls.
pub fn train(
    config: &GptConfig,
    set: &TrainingSet,
    epochs: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<TrainingResult, String> {
    if epochs == 0 {
        return Err("epochs must be greater than zero".into());
    }
    let updates = epochs
        .checked_mul(set.examples.len())
        .ok_or("Training update count overflows usize")?;
    let owned_set = TrainingSet {
        examples: set.examples.clone(),
        files: set.files.clone(),
        token_count: set.token_count,
    };
    let mut session = TrainingSession::new(config, owned_set, learning_rate, seed)?;
    let mut epoch_losses = Vec::new();
    session.advance_updates(updates, |_, event| {
        if let Some(summary) = event.epoch_summary {
            epoch_losses.push(summary.mean_pre_update_loss);
        }
        Ok(())
    })?;
    Ok(TrainingResult {
        model: session.inference_model(),
        epoch_losses,
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 4,
            context_length: 4,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        }
    }

    fn set(examples: Vec<Vec<u32>>) -> TrainingSet {
        TrainingSet {
            examples,
            files: Vec::new(),
            token_count: 0,
        }
    }

    #[test]
    fn trains_across_multiple_examples() {
        let result = train(
            &config(),
            &set(vec![vec![0, 1, 2], vec![2, 3, 0, 1]]),
            25,
            0.01,
            42,
        )
        .unwrap();
        assert_eq!(result.epoch_losses.len(), 25);
        assert!(result.epoch_losses.iter().all(|loss| loss.is_finite()));
        assert!(result.epoch_losses[24] < result.epoch_losses[0]);
    }

    #[test]
    fn rejects_invalid_training_inputs() {
        for examples in [vec![], vec![vec![0]], vec![vec![0, 4]], vec![vec![0; 6]]] {
            assert!(train(&config(), &set(examples), 1, 0.01, 42).is_err());
        }
        let data = set(vec![vec![0, 1]]);
        assert!(train(&config(), &data, 0, 0.01, 42).is_err());
        for rate in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            assert!(train(&config(), &data, 1, rate, 42).is_err());
        }
    }
}
