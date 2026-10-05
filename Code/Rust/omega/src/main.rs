use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};
#[derive(Parser)]
#[command(
    version,
    about = "Persistent model training workspace. Without a command, open the terminal UI."
)]
struct Cli {
    #[arg(long)]
    project: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    /// Check this executable's selected compute backend; CUDA is experimental
    Doctor {
        #[arg(long, value_enum, default_value = "cpu")]
        backend: omega_benchmark::Backend,
        #[arg(long)]
        device: Option<usize>,
    },
    /// Create a new whole-workflow project, never overwrite
    Init { path: PathBuf },
    /// Inspect a project and resolve its download plan without launching it
    Plan { config: PathBuf },
    /// Explicitly start the reviewed pipeline in the background
    Start {
        config: PathBuf,
        #[arg(long)]
        yes: bool,
    },
    /// Continue a complete snapshot using its retained executable when known
    Resume {
        config: PathBuf,
        checkpoint: PathBuf,
        #[arg(long)]
        yes: bool,
    },
    /// Read persistent job history
    Jobs,
    /// Request a checkpoint boundary and stop; unlike closing the UI
    Stop { id: String },
    /// Reconnect to a live worker's status
    Status { id: String },
    #[command(name = "__worker", hide = true)]
    Worker {
        #[arg(long)]
        state: PathBuf,
        #[arg(long)]
        id: String,
    },
}
fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Some(Command::Resume {
            config,
            checkpoint,
            yes,
        }) => {
            anyhow::ensure!(
                yes,
                "Review the checkpoint/configuration, then pass --yes to resume"
            );
            let root = omega::jobs::state_root()?;
            let checkpoint = std::fs::canonicalize(checkpoint)?;
            let retained = omega::jobs::retained_for_checkpoint(&root, &checkpoint)?;
            let spec = omega::jobs::launch(
                &root,
                &config,
                omega::jobs::Action::Resume,
                Some(checkpoint),
                None,
                retained.as_deref(),
                None,
            )?;
            println!("{}", spec.id);
        }
        Some(Command::Doctor { backend, device }) => println!(
            "{}",
            serde_json::to_string_pretty(&match device {
                Some(index) => omega::pipeline::preflight(backend, index)?,
                None => omega::pipeline::doctor(backend)?,
            })?
        ),
        Some(Command::Init { path }) => omega::ProjectConfig::default().create(&path)?,
        Some(Command::Plan { config }) => {
            let path = std::fs::canonicalize(config)?;
            let c = omega::ProjectConfig::load(&path)?;
            println!(
                "{}",
                omega::pipeline::review(
                    &c,
                    path.parent().context("Project has no directory")?,
                    &omega::jobs::Action::Pipeline
                )?
                .1
            );
        }
        Some(Command::Start { config, yes }) => {
            anyhow::ensure!(
                yes,
                "Inspect `omega plan PATH` first, then use `omega start PATH --yes` to launch"
            );
            let path = std::fs::canonicalize(config)?;
            let c = omega::ProjectConfig::load(&path)?;
            let (approved, _) = omega::pipeline::review(
                &c,
                path.parent().context("Project has no directory")?,
                &omega::jobs::Action::Pipeline,
            )?;
            let job = omega::jobs::launch(
                &omega::jobs::state_root()?,
                &path,
                omega::jobs::Action::Pipeline,
                None,
                None,
                None,
                Some(approved),
            )?;
            println!("{}", job.id);
        }
        Some(Command::Jobs) => println!(
            "{}",
            serde_json::to_string_pretty(&omega::jobs::list(&omega::jobs::state_root()?)?)?
        ),
        Some(Command::Stop { id }) => println!(
            "{}",
            serde_json::to_string_pretty(&omega::jobs::request(
                &omega::jobs::state_root()?,
                &id,
                "stop"
            )?)?
        ),
        Some(Command::Status { id }) => println!(
            "{}",
            serde_json::to_string_pretty(&omega::jobs::request(
                &omega::jobs::state_root()?,
                &id,
                "status"
            )?)?
        ),
        Some(Command::Worker { state, id }) => omega::jobs::worker(&state, &id)?,
        None => omega::tui::run(cli.project)?,
    }
    Ok(())
}
fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
