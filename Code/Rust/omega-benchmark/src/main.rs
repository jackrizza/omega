use std::{fs::OpenOptions, io::Write, path::PathBuf, process::ExitCode};

use clap::{Args, Parser, Subcommand};
use omega_benchmark::{
    Backend, DatasetFormat, DatasetOptions, ModelDimensions, TrainingTimeConfig, training_time,
};
use omega_training::{
    ValidationSplit, batching::BatchConfig, cpu::CpuThreadSettings,
    optimization::OptimizationOptions,
};

#[derive(Parser)]
#[command(
    version,
    about = "Bounded benchmarks using Omega's real training pipeline"
)]
struct Cli {
    #[command(flatten)]
    cpu: CpuThreadSettings,
    #[command(subcommand)]
    command: Benchmark,
}

#[derive(Subcommand)]
enum Benchmark {
    /// Time disposable updates on real data and estimate fixed-order training duration
    TrainingTime(Box<TrainingTimeArgs>),
}

#[derive(Args)]
struct TrainingTimeArgs {
    /// Dataset folders relative to datasets-root; repeat to combine ordinary datasets
    #[arg(long, required = true)]
    dataset: Vec<String>,
    /// Saved tokenizer path, absolute or relative to datasets-root (must match training)
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    datasets_root: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    dataset_format: DatasetFormat,
    /// Existing verified base token cache; this command never creates caches
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Cache validity limits must match those used when the cache was created
    #[command(flatten)]
    cache_limits: CacheLimitArgs,
    #[arg(long, conflicts_with = "validation_ratio")]
    validation_count: Option<usize>,
    #[arg(long, conflicts_with = "validation_count")]
    validation_ratio: Option<f64>,
    #[arg(long, default_value_t = 42)]
    split_seed: u64,
    #[arg(long, default_value_t = 64)]
    context_length: usize,
    #[arg(long, default_value_t = 32)]
    d_model: usize,
    #[arg(long, default_value_t = 4)]
    heads: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
    #[arg(long, default_value_t = 128)]
    d_ff: usize,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 65536)]
    max_batch_tokens: usize,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 0.003)]
    learning_rate: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    gradient_clip_norm: Option<f64>,
    /// Optimizer learning-rate warmup (distinct from benchmark warmup)
    #[arg(long, default_value_t = 0)]
    warmup_updates: usize,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: Backend,
    /// Discrete Vulkan adapter index; nonzero values require --backend vulkan
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Untimed disposable updates before sampling (1..100)
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Timed batches spread across the full dataset (1..1000, capped to one epoch)
    #[arg(long, default_value_t = 12)]
    samples: usize,
    /// Cooperative warmup + timing budget; excludes setup, checked between updates
    #[arg(long, default_value_t = 60.0)]
    max_seconds: f64,
    /// Print machine-readable JSON instead of a human summary
    #[arg(long)]
    json: bool,
    /// Save JSON to a NEW file; existing files are never overwritten
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct CacheLimitArgs {
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    max_source_bytes: u64,
    #[arg(long, default_value_t = 1_000_000)]
    max_document_tokens: usize,
    #[arg(long, default_value_t = 100_000)]
    max_documents: usize,
    #[arg(long, default_value_t = 200_000)]
    max_directory_entries: usize,
    #[arg(long, default_value_t = 32 * 1024 * 1024)]
    max_manifest_bytes: usize,
}

impl From<CacheLimitArgs> for omega_training::cache::CacheLimits {
    fn from(value: CacheLimitArgs) -> Self {
        Self {
            max_source_bytes: value.max_source_bytes,
            max_document_tokens: value.max_document_tokens,
            max_documents: value.max_documents,
            max_directory_entries: value.max_directory_entries,
            max_manifest_bytes: value.max_manifest_bytes,
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(status) => status,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    // Configure pools on a child before any tokenizer/backend initialization.
    // Never mutate this process's environment after threads may have started.
    const MARKER: &str = "OMEGA_BENCHMARK_CPU_REEXEC_V1";
    cli.cpu.validate()?;
    let requested = cli.cpu.cpu_threads.is_some() || cli.cpu.matmul_threads.is_some();
    let expected = format!("{:?}:{:?}", cli.cpu.cpu_threads, cli.cpu.matmul_threads);
    if let Some(marker) = std::env::var_os(MARKER) {
        if !requested || marker != std::ffi::OsString::from(expected) {
            return Err("Invalid or stale benchmark CPU startup marker".into());
        }
        for (key, count) in [
            ("RAYON_NUM_THREADS", cli.cpu.cpu_threads),
            ("RAYON_RS_NUM_CPUS", cli.cpu.cpu_threads),
            ("MATMUL_NUM_THREADS", cli.cpu.matmul_threads),
        ] {
            if let Some(count) = count
                && std::env::var(key).ok().as_deref() != Some(count.to_string().as_str())
            {
                return Err(format!("CPU startup environment does not match {key}"));
            }
        }
    } else if requested {
        let mut child =
            std::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        child
            .args(std::env::args_os().skip(1))
            .env(MARKER, expected);
        cli.cpu.apply_to_command(&mut child)?;
        let status = child
            .status()
            .map_err(|e| format!("Cannot start configured CPU process: {e}"))?;
        return Ok(if status.success() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        });
    }
    let Benchmark::TrainingTime(args) = cli.command;
    if args.output.as_ref().is_some_and(|p| p.exists()) {
        return Err("Output already exists; choose a new --output path".into());
    }
    let config = TrainingTimeConfig {
        dataset: DatasetOptions {
            root: args
                .datasets_root
                .unwrap_or_else(|| omega_training::project_root().join("datasets")),
            selections: args.dataset,
            tokenizer: args.tokenizer,
            format: args.dataset_format,
            validation: args
                .validation_count
                .map(ValidationSplit::Count)
                .or_else(|| args.validation_ratio.map(ValidationSplit::Ratio))
                .unwrap_or(ValidationSplit::None),
            split_seed: args.split_seed,
            cache: args.cache,
            cache_limits: args.cache_limits.into(),
        },
        model: ModelDimensions {
            context_length: args.context_length,
            d_model: args.d_model,
            heads: args.heads,
            layers: args.layers,
            d_ff: args.d_ff,
        },
        backend: args.backend,
        device: args.device,
        batching: BatchConfig {
            batch_size: args.batch_size,
            max_batch_tokens: args.max_batch_tokens,
        },
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        seed: args.seed,
        optimization: OptimizationOptions {
            gradient_clip_norm: args.gradient_clip_norm,
            warmup_updates: args.warmup_updates,
        },
        warmup: args.warmup,
        samples: args.samples,
        max_seconds: args.max_seconds,
    };
    config.validate()?;
    eprintln!("Preparing the selected dataset and a disposable model; no weights will be saved...");
    let report = training_time(&config)?;
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(path) = args.output {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("Cannot create report {}: {e}", path.display()))?;
        file.write_all(format!("{json}\n").as_bytes())
            .map_err(|e| format!("Cannot finish report {}: {e}", path.display()))?;
    }
    if args.json {
        println!("{json}");
    } else {
        println!(
            "Estimated training updates: {} for {} epoch(s)",
            duration(report.estimated_training_seconds),
            report.epochs
        );
        println!(
            "Per epoch: {} | {} batches | {} examples | {} supervised targets",
            duration(report.estimated_epoch_seconds),
            report.batches_per_epoch,
            report.examples_per_epoch,
            report.supervised_targets_per_epoch
        );
        println!(
            "Measured {} batches at {:.1} supervised targets/s on {:?}",
            report.samples.len(),
            report.measured_targets_per_second,
            report.backend
        );
        println!(
            "Observed setup: {} | benchmark warmup: {}",
            duration(report.setup_seconds),
            duration(report.warmup_seconds)
        );
        println!(
            "Setup + estimated updates: {}",
            duration(report.estimated_setup_plus_training_seconds)
        );
        for note in report.notes {
            println!("- {note}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.2} s")
    } else if seconds < 3600.0 {
        format!("{:.2} min", seconds / 60.0)
    } else {
        format!("{:.2} h", seconds / 3600.0)
    }
}
