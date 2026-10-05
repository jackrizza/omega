use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use omega_datasets::{config::Config, hub::HuggingFace};

#[derive(Parser)]
#[command(
    version,
    about = "Create reproducible base/chat corpora from Hugging Face using datasets/<name>/model.toml"
)]
struct Cli {
    /// Explicit root; default is the compile-time checkout's datasets directory.
    #[arg(long, global = true)]
    datasets_root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a folder and example model.toml without overwriting an existing file.
    Init { dataset: String },
    /// Validate local model.toml without network access or output creation.
    Check { dataset: String },
    /// Resolve repositories, commit SHAs and file globs; print JSON, no downloads.
    Plan { dataset: String },
    /// Download, normalize, deduplicate and split into a NEW release directory.
    Build { dataset: String },
}

fn run(cli: Cli) -> Result<()> {
    let root = cli
        .datasets_root
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../datasets"));
    if let Command::Init { dataset } = &cli.command {
        println!("{}", omega_datasets::init(&root, dataset)?.display());
        return Ok(());
    }
    let dataset = match &cli.command {
        Command::Check { dataset } | Command::Plan { dataset } | Command::Build { dataset } => {
            dataset
        }
        Command::Init { .. } => unreachable!(),
    };
    let folder = omega_datasets::dataset_folder(&root, dataset)?;
    let config = Config::load(&folder.join("model.toml"))?;
    if matches!(cli.command, Command::Check { .. }) {
        println!(
            "Valid: {} sources; output {}",
            config.sources.len(),
            folder.join(&config.output).display()
        );
        return Ok(());
    }
    let token = match std::env::var("HF_TOKEN") {
        Ok(value) if !value.trim().is_empty() => Some(value),
        Ok(_) | Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(e).context("HF_TOKEN must be valid Unicode"),
    };
    let hub = HuggingFace::new(token, config.limits.timeout_seconds)?;
    match cli.command {
        Command::Plan { .. } => println!(
            "{}",
            serde_json::to_string_pretty(&omega_datasets::hub::plan(&config, &hub)?)?
        ),
        Command::Build { .. } => {
            eprintln!(
                "Building {} (HF_TOKEN is used only if set)",
                folder.join(&config.output).display()
            );
            let report = omega_datasets::build(&folder, &config, &hub)?;
            println!("{}", serde_json::to_string_pretty(&report.counts)?);
            println!("Complete: {}", folder.join(&config.output).display());
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run(Cli::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omega-datasets: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
