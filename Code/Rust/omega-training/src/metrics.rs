//! Version 1 JSON Lines metrics over committed trainer events.
//!
//! Each call writes and flushes one complete line. I/O errors are returned to
//! the observer, which stops training after the already-committed update. A
//! failed write may leave a partial final line; consumers should reject that
//! line, not invent a completion event. This module never writes to stdout.

use std::io::Write;

use serde::Serialize;

use crate::{EvaluationMetrics, TrainingProgress, UpdateEvent};

#[derive(Serialize)]
struct Envelope<T: Serialize> {
    schema_version: u32,
    #[serde(flatten)]
    record: T,
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Record {
    Execution {
        backend: String,
        device: String,
    },
    TrainingInterrupted {
        completed_updates: usize,
        completed_epochs: usize,
        completed_targets: usize,
        next_example_index: usize,
    },
    TrainingResumed {
        target_epochs: usize,
        completed_updates: usize,
        completed_epochs: usize,
        completed_targets: usize,
        next_example_index: usize,
    },
    /// An explicitly bounded run stopped before its planned total epochs.
    SegmentComplete {
        completed_updates: usize,
        completed_epochs: usize,
        completed_targets: usize,
        next_example_index: usize,
    },
    TrainingStarted {
        epochs: usize,
        examples_per_epoch: usize,
        targets_per_epoch: usize,
    },
    Update {
        epoch: usize,
        example_index: usize,
        completed_updates: usize,
        completed_targets: usize,
        target_count: usize,
        example_count: usize,
        effective_learning_rate: f64,
        gradient_norm: f64,
        clipped: bool,
        pre_update_loss: f32,
        epoch_mean_pre_update_loss: Option<f32>,
        epoch_target_count: Option<usize>,
    },
    Validation {
        epoch: Option<usize>,
        completed_updates: usize,
        target_count: usize,
        mean_cross_entropy: f64,
        perplexity: f64,
    },
    /// Training/evaluation finished; this is not a checkpoint-save assertion.
    TrainingComplete {
        completed_updates: usize,
        completed_epochs: usize,
        completed_targets: usize,
    },
}

pub struct JsonlMetrics<W: Write> {
    writer: W,
}

impl<W: Write> JsonlMetrics<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Record the selected compute backend and adapter identity.
    pub fn execution(&mut self, backend: &str, device: &str) -> Result<(), String> {
        if backend.trim().is_empty() || device.trim().is_empty() {
            return Err("Execution backend and device must be nonempty".into());
        }
        self.emit(Record::Execution {
            backend: backend.into(),
            device: device.into(),
        })
    }

    pub fn resumed(
        &mut self,
        target_epochs: usize,
        progress: TrainingProgress,
    ) -> Result<(), String> {
        self.emit(Record::TrainingResumed {
            target_epochs,
            completed_updates: progress.completed_updates,
            completed_epochs: progress.completed_epochs,
            completed_targets: progress.completed_targets,
            next_example_index: progress.next_example_index,
        })
    }

    pub fn segment_complete(&mut self, progress: TrainingProgress) -> Result<(), String> {
        self.emit(Record::SegmentComplete {
            completed_updates: progress.completed_updates,
            completed_epochs: progress.completed_epochs,
            completed_targets: progress.completed_targets,
            next_example_index: progress.next_example_index,
        })
    }

    pub fn interrupted(&mut self, progress: TrainingProgress) -> Result<(), String> {
        self.emit(Record::TrainingInterrupted {
            completed_updates: progress.completed_updates,
            completed_epochs: progress.completed_epochs,
            completed_targets: progress.completed_targets,
            next_example_index: progress.next_example_index,
        })
    }

    fn emit(&mut self, record: Record) -> Result<(), String> {
        let mut bytes = serde_json::to_vec(&Envelope {
            schema_version: 1,
            record,
        })
        .map_err(|error| format!("Cannot serialize metrics: {error}"))?;
        bytes.push(b'\n');
        self.writer
            .write_all(&bytes)
            .and_then(|()| self.writer.flush())
            .map_err(|error| format!("Cannot write metrics: {error}"))
    }

    pub fn started(
        &mut self,
        epochs: usize,
        examples_per_epoch: usize,
        targets_per_epoch: usize,
    ) -> Result<(), String> {
        if epochs == 0 || examples_per_epoch == 0 || targets_per_epoch == 0 {
            return Err("Metrics training counts must be positive".into());
        }
        self.emit(Record::TrainingStarted {
            epochs,
            examples_per_epoch,
            targets_per_epoch,
        })
    }

    /// Forward the trainer's weighted epoch summary; never average chunk means.
    pub fn update(&mut self, event: &UpdateEvent) -> Result<(), String> {
        if !event.pre_update_loss.is_finite()
            || !event.effective_learning_rate.is_finite()
            || event.effective_learning_rate <= 0.0
            || !event.gradient_norm.is_finite()
            || event.gradient_norm < 0.0
            || event
                .epoch_summary
                .is_some_and(|summary| !summary.mean_pre_update_loss.is_finite())
        {
            return Err("Cannot log non-finite training metrics".into());
        }
        self.emit(Record::Update {
            epoch: event.epoch,
            example_index: event.example_index,
            completed_updates: event.completed_updates,
            completed_targets: event.completed_targets,
            target_count: event.target_count,
            example_count: event.example_count,
            effective_learning_rate: event.effective_learning_rate,
            gradient_norm: event.gradient_norm,
            clipped: event.clipped,
            pre_update_loss: event.pre_update_loss,
            epoch_mean_pre_update_loss: event.epoch_summary.map(|s| s.mean_pre_update_loss),
            epoch_target_count: event.epoch_summary.map(|s| s.target_count),
        })
    }

    pub fn validation(
        &mut self,
        epoch: Option<usize>,
        completed_updates: usize,
        metrics: &EvaluationMetrics,
    ) -> Result<(), String> {
        if metrics.target_count == 0
            || !metrics.mean_cross_entropy.is_finite()
            || !metrics.perplexity.is_finite()
        {
            return Err("Cannot log empty or non-finite validation metrics".into());
        }
        self.emit(Record::Validation {
            epoch,
            completed_updates,
            target_count: metrics.target_count,
            mean_cross_entropy: metrics.mean_cross_entropy,
            perplexity: metrics.perplexity,
        })
    }

    /// An explicit final training boundary; dropping a writer emits nothing.
    pub fn training_complete(&mut self, progress: TrainingProgress) -> Result<(), String> {
        self.emit(Record::TrainingComplete {
            completed_updates: progress.completed_updates,
            completed_epochs: progress.completed_epochs,
            completed_targets: progress.completed_targets,
        })
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TrainingSession, TrainingSet};
    use omega_nn::GptConfig;

    fn session() -> TrainingSession {
        TrainingSession::new(
            &GptConfig {
                vocab_size: 3,
                context_length: 3,
                d_model: 4,
                num_heads: 1,
                num_layers: 1,
                d_ff: 8,
            },
            TrainingSet {
                examples: vec![vec![0, 1], vec![1, 2, 0, 1]],
                files: vec![],
                token_count: 6,
            },
            0.01,
            42,
        )
        .unwrap()
    }

    fn records(writer: JsonlMetrics<Vec<u8>>) -> Vec<serde_json::Value> {
        String::from_utf8(writer.into_inner())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn execution_records_selected_backend_and_device_without_changing_progress() {
        let mut writer = JsonlMetrics::new(Vec::new());
        assert!(writer.execution("", "adapter").is_err());
        assert!(writer.execution("vulkan", " ").is_err());
        writer
            .execution("vulkan", "Intel Arc A770, device 0")
            .unwrap();
        let rows = records(writer);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["schema_version"], 1);
        assert_eq!(rows[0]["event"], "execution");
        assert_eq!(rows[0]["backend"], "vulkan");
        assert_eq!(rows[0]["device"], "Intel Arc A770, device 0");
    }

    #[test]
    fn committed_event_order_and_weighted_summary_are_preserved() {
        let mut session = session();
        let mut writer = JsonlMetrics::new(Vec::new());
        writer.started(1, 2, 4).unwrap();
        session
            .advance_updates(2, |_, event| writer.update(event))
            .unwrap();
        let evaluation = crate::evaluate(
            &session.inference_model(),
            session.config(),
            session.training_set(),
        )
        .unwrap();
        writer.validation(Some(1), 2, &evaluation).unwrap();
        writer.training_complete(session.progress()).unwrap();
        let rows = records(writer);
        assert_eq!(rows.len(), 5);
        assert!(rows.iter().all(|row| row["schema_version"] == 1));
        let names: Vec<_> = rows
            .iter()
            .map(|row| row["event"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "training_started",
                "update",
                "update",
                "validation",
                "training_complete"
            ]
        );
        assert!(rows[1]["epoch_mean_pre_update_loss"].is_null());
        assert_eq!(rows[2]["epoch_target_count"], 4);
        let expected = (rows[1]["pre_update_loss"].as_f64().unwrap()
            + 3.0 * rows[2]["pre_update_loss"].as_f64().unwrap())
            / 4.0;
        assert!((rows[2]["epoch_mean_pre_update_loss"].as_f64().unwrap() - expected).abs() < 1e-6);
        assert_eq!(rows[3]["target_count"], 4);
        assert_eq!(rows[4]["completed_updates"], 2);
    }

    #[test]
    fn partial_run_has_only_observed_updates_and_no_completion() {
        let mut session = session();
        let mut writer = JsonlMetrics::new(Vec::new());
        writer.started(1, 2, 4).unwrap();
        session
            .advance_updates(1, |_, event| writer.update(event))
            .unwrap();
        let rows = records(writer);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["completed_targets"], 1);
        assert!(rows[1]["epoch_target_count"].is_null());
    }

    #[test]
    fn write_and_flush_failures_stop_after_committed_update() {
        struct Broken(bool);
        impl Write for Broken {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.0 {
                    Ok(bytes.len())
                } else {
                    Err(std::io::Error::other("write failed"))
                }
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("flush failed"))
            }
        }
        for flush_failure in [false, true] {
            let mut session = session();
            let mut writer = JsonlMetrics::new(Broken(flush_failure));
            let error = session
                .advance_updates(2, |_, event| writer.update(event))
                .unwrap_err();
            assert!(error.contains("Cannot write metrics"));
            assert_eq!(session.progress().completed_updates, 1);
        }
    }
}
