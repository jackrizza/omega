use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Explicit experiment inputs. No production budgets or quality gates are inferred.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostTrainingConfig {
    pub parent: PathBuf,
    pub output: PathBuf,
    pub purpose: String,
    pub language: String,
    pub training: Vec<String>,
    pub validation: Vec<String>,
    pub base_validation: Vec<String>,
    pub sealed: Vec<String>,
    pub suite: PathBuf,
    pub rubric_version: String,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub batch_size: usize,
    pub max_batch_tokens: usize,
    pub segment_updates: usize,
    pub max_updates: usize,
    pub max_seconds: f64,
    pub evaluation_max_seconds: f64,
    pub max_evaluations: usize,
    pub max_output_bytes: u64,
    pub min_test_pass_rate: f64,
    pub max_assistant_loss: f64,
    pub max_base_loss_increase: f64,
    pub min_human_score: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen: Option<FrozenInputs>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenInputs {
    pub readiness_sha256: String,
    pub suite_sha256: String,
}

impl PostTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, text) in [
            ("purpose", &self.purpose),
            ("language", &self.language),
            ("rubric_version", &self.rubric_version),
        ] {
            ensure!(
                !text.trim().is_empty() && text.len() <= 4096,
                "Post-training {name} must contain 1..4096 bytes"
            );
        }
        for (name, selections) in [
            ("training", &self.training),
            ("validation", &self.validation),
            ("base_validation", &self.base_validation),
            ("sealed", &self.sealed),
        ] {
            ensure!(
                !selections.is_empty() && selections.len() <= 1000,
                "Post-training {name} selections are required (maximum 1000)"
            );
            for s in selections {
                omega_datasets::config::safe_relative(s)?;
            }
        }
        for path in [&self.parent, &self.output, &self.suite] {
            ensure!(
                !path.as_os_str().is_empty(),
                "Post-training parent/output/suite paths are required"
            );
        }
        ensure!(
            self.epochs > 0 && self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "Post-training epochs and learning rate must be positive"
        );
        ensure!(
            self.batch_size > 0 && self.max_batch_tokens > 0,
            "Post-training batching limits must be positive"
        );
        ensure!(
            self.segment_updates > 0 && self.max_updates >= self.segment_updates,
            "Require 0 < segment_updates <= max_updates"
        );
        ensure!(
            self.max_seconds.is_finite()
                && self.max_seconds > 0.0
                && self.evaluation_max_seconds.is_finite()
                && self.evaluation_max_seconds > 0.0
                && self.evaluation_max_seconds <= self.max_seconds,
            "Require positive finite evaluation_max_seconds <= max_seconds"
        );
        ensure!(
            (2..=10000).contains(&self.max_evaluations) && self.max_output_bytes > 0,
            "Require 2..10000 evaluations (including baseline) and a positive output byte budget"
        );
        ensure!(
            self.min_test_pass_rate.is_finite() && (0.0..=1.0).contains(&self.min_test_pass_rate),
            "min_test_pass_rate must be in [0,1]"
        );
        ensure!(
            self.max_assistant_loss.is_finite()
                && self.max_assistant_loss > 0.0
                && self.max_base_loss_increase.is_finite()
                && self.max_base_loss_increase >= 0.0,
            "Loss gates must be finite: positive assistant loss and nonnegative relative base regression"
        );
        ensure!(self.min_human_score <= 4, "min_human_score must be 0..4");
        if let Some(f) = &self.frozen {
            for hash in [&f.readiness_sha256, &f.suite_sha256] {
                ensure!(
                    hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
                    "Invalid frozen input hash"
                );
            }
        }
        Ok(())
    }
}
