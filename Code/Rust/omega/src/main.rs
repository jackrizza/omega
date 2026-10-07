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
    /// Inspect, run, review and explicitly promote a post-training experiment
    PostTraining {
        #[command(subcommand)]
        command: PostTrainingCommand,
    },
    #[command(name = "__worker", hide = true)]
    Worker {
        #[arg(long)]
        state: PathBuf,
        #[arg(long)]
        id: String,
    },
}

#[derive(Subcommand)]
enum PostTrainingCommand {
    /// Check parent and partition readiness without training or evaluation
    Ready { config: PathBuf },
    /// Resolve a frozen experiment and show its budgets before launch
    Plan { config: PathBuf },
    /// Start the approved bounded post-training workflow in the background
    Start {
        config: PathBuf,
        #[arg(long)]
        yes: bool,
    },
    /// Explicitly recover a workflow using its retained executable/settings
    Recover {
        workflow: PathBuf,
        #[arg(long)]
        yes: bool,
    },
    /// List candidate reports, gates, pending reviews and validation ranking
    Reports { workflow: PathBuf },
    /// Save human scores tied to an exact completed report and rubric
    Review {
        workflow: PathBuf,
        report: PathBuf,
        scores: PathBuf,
    },
    /// Record explicit human promotion; sealed-test acceptance remains separate
    Promote {
        workflow: PathBuf,
        report: PathBuf,
        #[arg(long)]
        approved_by: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        yes: bool,
    },
    /// Explicitly migrate a project to the post-training configuration schema
    Migrate { config: PathBuf },
}

fn post_training(command: PostTrainingCommand) -> Result<()> {
    match command {
        PostTrainingCommand::Ready { config } => {
            let path = std::fs::canonicalize(config)?;
            let config = omega::ProjectConfig::load(&path)?;
            let report = omega::post_training::readiness(
                &config,
                path.parent().context("Project has no directory")?,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            anyhow::ensure!(
                report.passed,
                "Post-training readiness checks failed; inspect the report"
            );
        }
        PostTrainingCommand::Plan { config } => {
            let path = std::fs::canonicalize(config)?;
            let config = omega::ProjectConfig::load(&path)?;
            let (_, summary) = omega::post_training::prepare(
                &config,
                path.parent().context("Project has no directory")?,
            )?;
            println!("{summary}");
        }
        PostTrainingCommand::Start { config, yes } => {
            anyhow::ensure!(
                yes,
                "Inspect `omega post-training plan PATH`, then pass --yes to start"
            );
            let path = std::fs::canonicalize(config)?;
            let config = omega::ProjectConfig::load(&path)?;
            let (approved, _) = omega::post_training::prepare(
                &config,
                path.parent().context("Project has no directory")?,
            )?;
            let spec = omega::jobs::launch(
                &omega::jobs::state_root()?,
                &path,
                omega::jobs::Action::PostTrain,
                None,
                None,
                None,
                Some(approved),
            )?;
            println!("{}", spec.id);
        }
        PostTrainingCommand::Recover { workflow, yes } => {
            anyhow::ensure!(
                yes,
                "Inspect the workflow and retained settings, then pass --yes to recover"
            );
            let workflow = std::fs::canonicalize(workflow)?;
            let retained = omega::post_training::recover_spec(&workflow)?;
            let spec = omega::jobs::launch(
                &omega::jobs::state_root()?,
                &retained.config_path,
                omega::jobs::Action::PostTrainRecover,
                Some(workflow),
                None,
                Some(&retained.executable),
                Some(retained.config),
            )?;
            println!("{}", spec.id);
        }
        PostTrainingCommand::Reports { workflow } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&omega::post_training::rank(&workflow)?)?
            );
        }
        PostTrainingCommand::Review {
            workflow,
            report,
            scores,
        } => {
            let review: omega::post_training::HumanReview = omega::jobs::read_json(&scores)?;
            println!(
                "{}",
                omega::post_training::save_review(&workflow, &report, &review)?.display()
            );
        }
        PostTrainingCommand::Promote {
            workflow,
            report,
            approved_by,
            output,
            yes,
        } => {
            anyhow::ensure!(
                yes,
                "Review candidate gates and human scores, then pass --yes to promote"
            );
            let selection =
                omega::post_training::promote(&workflow, &report, &approved_by, &output)?;
            println!("{}", serde_json::to_string_pretty(&selection)?);
        }
        PostTrainingCommand::Migrate { config } => {
            omega::config::migrate_post_training(&config)?;
            println!("{}", config.display());
        }
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Some(Command::PostTraining { command }) => post_training(command)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_training_subcommands_parse_explicit_paths_and_options() {
        for operation in ["ready", "plan", "start", "migrate"] {
            assert!(
                Cli::try_parse_from(["omega", "post-training", operation, "model.toml"]).is_ok()
            );
        }
        for operation in ["recover", "reports"] {
            assert!(Cli::try_parse_from(["omega", "post-training", operation, "workflow"]).is_ok());
        }
        assert!(
            Cli::try_parse_from([
                "omega",
                "post-training",
                "review",
                "workflow",
                "report.json",
                "scores.json"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "omega",
                "post-training",
                "promote",
                "workflow",
                "report.json",
                "--approved-by",
                "Human reviewer",
                "--output",
                "selection.json",
                "--yes"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "omega",
                "post-training",
                "promote",
                "workflow",
                "report.json"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from(["omega", "post-training", "start", "model.toml", "--unknown"])
                .is_err()
        );
    }

    #[test]
    fn launching_and_promotion_require_yes_before_accessing_inputs() {
        for command in [
            PostTrainingCommand::Start {
                config: "missing".into(),
                yes: false,
            },
            PostTrainingCommand::Recover {
                workflow: "missing".into(),
                yes: false,
            },
            PostTrainingCommand::Promote {
                workflow: "missing".into(),
                report: "missing".into(),
                approved_by: "Human".into(),
                output: "unused".into(),
                yes: false,
            },
        ] {
            assert!(
                post_training(command)
                    .unwrap_err()
                    .to_string()
                    .contains("--yes")
            );
        }
    }
}
