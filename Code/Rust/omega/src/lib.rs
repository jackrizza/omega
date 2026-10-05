//! Project configuration, persistent jobs and terminal application for Omega.
pub mod config;
pub mod jobs;
pub mod pipeline;
pub mod tui;
pub use config::ProjectConfig;
pub use jobs::{JobEvent, JobSpec, JobStatus};
