//! Periodic saves and cooperative stopping at committed optimizer boundaries.
//! Signal handlers should only set the supplied atomic flag. All model/file work
//! runs on the training thread; an update already in progress finishes first.

use crate::{ExampleSource, TrainingSession, UpdateEvent};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct SaveSchedule {
    pub every_updates: Option<usize>,
    pub every_epochs: Option<usize>,
}

impl SaveSchedule {
    pub fn validate(&self) -> Result<(), String> {
        if self.every_updates == Some(0) || self.every_epochs == Some(0) {
            return Err("Checkpoint intervals must be greater than zero".into());
        }
        Ok(())
    }

    fn due(&self, event: &UpdateEvent) -> bool {
        self.every_updates
            .is_some_and(|n| event.completed_updates.is_multiple_of(n))
            || event.epoch_summary.is_some_and(|summary| {
                self.every_epochs
                    .is_some_and(|n| summary.epoch.is_multiple_of(n))
            })
    }
}

#[derive(Debug)]
pub struct RunOutcome {
    pub interrupted: bool,
    pub checkpoint: PathBuf,
}

/// Advance at most `updates`, save at cumulative update/epoch intervals and once
/// at the final boundary. Coincident intervals/final/stop boundaries save once.
/// A stop already requested saves the current (possibly zero-update) state.
/// Observer/update/save failures propagate; no subsequent final save claims
/// success. Previously completed periodic checkpoints remain available.
pub fn advance_controlled<
    S: ExampleSource,
    B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>,
>(
    session: &mut TrainingSession<S, B>,
    updates: usize,
    stop: &AtomicBool,
    schedule: SaveSchedule,
    mut observe: impl FnMut(&TrainingSession<S, B>, &UpdateEvent) -> Result<(), String>,
    mut save: impl FnMut(&TrainingSession<S, B>) -> Result<PathBuf, String>,
) -> Result<RunOutcome, String> {
    schedule.validate()?;
    let mut last_save = None;
    for _ in 0..updates {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let event = session.step()?;
        observe(session, &event)?;
        if schedule.due(&event) {
            last_save = Some((event.completed_updates, save(session)?));
        }
    }
    let checkpoint = match last_save {
        Some((update, path)) if update == session.progress().completed_updates => path,
        _ => save(session)?,
    };
    Ok(RunOutcome {
        interrupted: stop.load(Ordering::SeqCst),
        checkpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CheckpointMetadata, TrainingSet, load_training_checkpoint, save_training_checkpoint,
    };
    use omega_nn::GptConfig;
    use omega_tokenizer::Tokens;

    fn make_session() -> TrainingSession {
        TrainingSession::new(
            &GptConfig {
                vocab_size: 13,
                context_length: 3,
                d_model: 4,
                num_heads: 1,
                num_layers: 1,
                d_ff: 8,
            },
            TrainingSet {
                examples: vec![vec![1, 2], vec![3, 4, 5]],
                files: vec![],
                token_count: 5,
            },
            0.003,
            42,
        )
        .unwrap()
    }

    #[test]
    fn intervals_use_cumulative_counts_and_coincident_final_is_not_duplicated() {
        let mut session = make_session();
        session.step().unwrap();
        let mut saved = Vec::new();
        advance_controlled(
            &mut session,
            5,
            &AtomicBool::new(false),
            SaveSchedule {
                every_updates: Some(2),
                every_epochs: Some(1),
            },
            |_, _| Ok(()),
            |s| {
                saved.push(s.progress().completed_updates);
                Ok(PathBuf::from("checkpoint"))
            },
        )
        .unwrap();
        assert_eq!(saved, [2, 4, 6]);
        let mut saved = Vec::new();
        advance_controlled(
            &mut session,
            1,
            &AtomicBool::new(false),
            SaveSchedule {
                every_updates: Some(4),
                every_epochs: None,
            },
            |_, _| Ok(()),
            |s| {
                saved.push(s.progress().completed_updates);
                Ok(PathBuf::from("final"))
            },
        )
        .unwrap();
        assert_eq!(saved, [7]);
        let mut epochs_only = make_session();
        let mut saved = Vec::new();
        advance_controlled(
            &mut epochs_only,
            5,
            &AtomicBool::new(false),
            SaveSchedule {
                every_updates: None,
                every_epochs: Some(2),
            },
            |_, _| Ok(()),
            |s| {
                saved.push(s.progress().completed_updates);
                Ok(PathBuf::from("epoch"))
            },
        )
        .unwrap();
        assert_eq!(saved, [4, 5]);
    }

    #[test]
    fn requested_stop_saves_exact_boundary_and_disk_resume_matches_continuation() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer = Tokens::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"),
        )
        .unwrap();
        let mut expected = make_session();
        let mut interrupted = expected.clone();
        let stop = AtomicBool::new(false);
        let mut saves = 0;
        let outcome = advance_controlled(
            &mut interrupted,
            10,
            &stop,
            SaveSchedule {
                every_updates: Some(1),
                every_epochs: Some(1),
            },
            |_, _| {
                stop.store(true, Ordering::SeqCst);
                Ok(())
            },
            |s| {
                saves += 1;
                save_training_checkpoint(
                    temp.path(),
                    "stopped",
                    s,
                    &tokenizer,
                    &CheckpointMetadata::default(),
                )
            },
        )
        .unwrap();
        assert!(outcome.interrupted);
        assert_eq!(saves, 1);
        assert_eq!(interrupted.progress().next_example_index, 1);
        let source = TrainingSet {
            examples: vec![vec![1, 2], vec![3, 4, 5]],
            files: vec![],
            token_count: 5,
        };
        let mut restored =
            load_training_checkpoint(&outcome.checkpoint, source, expected.config(), &tokenizer)
                .unwrap();
        expected.step().unwrap();
        for _ in 0..3 {
            assert_eq!(expected.step().unwrap(), restored.step().unwrap());
        }
        assert!(outcome.checkpoint.join("COMPLETE").is_file());
    }

    #[test]
    fn pre_requested_stop_invalid_intervals_and_failures_do_not_advance_further() {
        let mut session = make_session();
        let outcome = advance_controlled(
            &mut session,
            10,
            &AtomicBool::new(true),
            SaveSchedule::default(),
            |_, _| panic!("no update expected"),
            |_| Ok(PathBuf::from("zero")),
        )
        .unwrap();
        assert!(outcome.interrupted);
        assert_eq!(session.progress().completed_updates, 0);
        assert!(
            advance_controlled(
                &mut session,
                10,
                &AtomicBool::new(false),
                SaveSchedule {
                    every_updates: Some(0),
                    every_epochs: None
                },
                |_, _| Ok(()),
                |_| panic!("invalid schedule must not save")
            )
            .is_err()
        );
        let error = advance_controlled(
            &mut session,
            10,
            &AtomicBool::new(false),
            SaveSchedule {
                every_updates: Some(1),
                every_epochs: None,
            },
            |_, _| Ok(()),
            |_| Err("injected save failure".into()),
        )
        .unwrap_err();
        assert!(error.contains("save failure"));
        assert_eq!(session.progress().completed_updates, 1);
        let error = advance_controlled(
            &mut session,
            10,
            &AtomicBool::new(false),
            SaveSchedule::default(),
            |_, _| Err("injected observer failure".into()),
            |_| panic!("observer failure must not save"),
        )
        .unwrap_err();
        assert!(error.contains("observer failure"));
        assert_eq!(session.progress().completed_updates, 2);
    }
}
