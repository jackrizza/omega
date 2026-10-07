//! Explicit, bounded post-training workflow and evidence-based human selection.
mod config;
mod review;
mod workflow;
pub use config::{FrozenInputs, PostTrainingConfig};
pub use review::{
    CandidateReport, CaseScore, HumanReview, SelectionManifest, promote, rank, read_report,
    save_review,
};
pub use workflow::{Phase, WorkflowState, execute, prepare, readiness, recover_spec};
