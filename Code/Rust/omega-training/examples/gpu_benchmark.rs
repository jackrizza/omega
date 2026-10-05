//! One synthetic checked-training case per fresh process. Build --release and
//! run under an external timeout; the deadline here is cooperative between steps.
use std::{fs, time::Instant};

use burn::{
    module::{Module, ModuleVisitor, ParamId},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use clap::{Parser, ValueEnum};
use omega_nn::{Cpu, Gpt, GptConfig, TrainingBackend};
use omega_training::{
    SessionOptions, TrainingSession, TrainingSet,
    batching::BatchConfig,
    checkpoint::{BuildIdentity, ModelConfig},
    cpu::execution_profile,
    gpu::{GpuTraining, initialize_vulkan},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendChoice {
    Cpu,
    Vulkan,
}

#[derive(Debug, Parser)]
#[command(about = "One bounded synthetic release CPU/Vulkan training benchmark")]
struct Args {
    #[arg(long, value_enum)]
    backend: BackendChoice,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 151665)]
    vocab_size: usize,
    #[arg(long, default_value_t = 64)]
    context_length: usize,
    #[arg(long, default_value_t = 32)]
    width: usize,
    #[arg(long, default_value_t = 4)]
    heads: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
    #[arg(long, default_value_t = 128)]
    feed_forward: usize,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    samples: usize,
    #[arg(long, default_value_t = 60)]
    max_seconds: u64,
    /// Attribute host wall time to trainer stages (GPU work may finish in later stages).
    #[arg(long)]
    profile: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if ![1, 2, 4].contains(&args.batch_size)
        || !(1..=8).contains(&args.warmup)
        || !(1..=20).contains(&args.samples)
        || !(1..=600).contains(&args.max_seconds)
    {
        return Err(
            "Require batch-size 1/2/4, warmup 1..8, samples 1..20, max-seconds 1..600".into(),
        );
    }
    let start = Instant::now();
    let cpu_profile = execution_profile()?;
    let config = GptConfig {
        vocab_size: args.vocab_size,
        context_length: args.context_length,
        d_model: args.width,
        num_heads: args.heads,
        num_layers: args.layers,
        d_ff: args.feed_forward,
    };
    config.validate()?;
    Cpu::seed(42);
    let model = config.init::<Cpu>(&Default::default())?;
    // Force every CPU parameter before serializing. Hash values only: fresh ParamIds
    // are random, while the parameter values must match between backend cases.
    struct Materialize {
        hash: Sha256,
        valid: bool,
    }
    impl ModuleVisitor<Cpu> for Materialize {
        fn visit_float<const D: usize>(&mut self, _: ParamId, tensor: &Tensor<Cpu, D>) {
            for value in tensor.to_data().iter::<f32>() {
                self.valid &= value.is_finite();
                self.hash.update(value.to_le_bytes());
            }
        }
    }
    let mut materialize = Materialize {
        hash: Sha256::new(),
        valid: true,
    };
    model.visit(&mut materialize);
    if !materialize.valid {
        return Err("Non-finite initial parameters".into());
    }
    let parameter_hash = format!("{:x}", materialize.hash.finalize());
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = <_ as Recorder<Cpu>>::record(&recorder, model.into_record(), ())
        .map_err(|e| format!("Cannot serialize initial weights: {e}"))?;
    let initial_weight_bytes = bytes.len();
    let (result, runtime) = match args.backend {
        BackendChoice::Cpu => (
            run::<TrainingBackend>(&args, &config, bytes, &Default::default(), start)?,
            json!({"backend":"cpu", "cpu":cpu_profile}),
        ),
        BackendChoice::Vulkan => {
            let selected = initialize_vulkan(args.device)?;
            selected
                .adapter()
                .validate_model(&config, args.batch_size)?;
            let result = run::<GpuTraining>(&args, &config, bytes, selected.device(), start)?;
            (
                result,
                json!({"backend":"vulkan", "gpu":selected.profile(), "adapter":selected.adapter(), "host_cpu":cpu_profile}),
            )
        }
    };
    println!(
        "{}",
        json!({
            "schema_version":1, "synthetic":true, "model":ModelConfig::from(&config),
            "batch_size":args.batch_size, "seed":42, "learning_rate":0.001,
            "initial_parameter_sha256":parameter_hash, "initial_weight_record_bytes":initial_weight_bytes,
            "build":BuildIdentity::current(), "runtime":runtime, "result":result,
            "whole_process_seconds":start.elapsed().as_secs_f64(), "peak_rss_kib":peak_rss_kib(),
            "memory_note":"Linux VmHWM is whole-process host peak. DRM fdinfo samples are raw per-FD snapshots, may alias, and are not peak GPU allocation or total VRAM."
        })
    );
    Ok(())
}

fn run<B: AutodiffBackend<FloatElem = f32>>(
    args: &Args,
    config: &GptConfig,
    bytes: Vec<u8>,
    device: &B::Device,
    start: Instant,
) -> Result<Value, String> {
    let count = args.batch_size * (args.warmup + args.samples);
    let examples: Vec<Vec<u32>> = (0..count)
        .map(|row| {
            (0..=config.context_length)
                .map(|position| ((row * 17 + position * 7 + 3) % config.vocab_size) as u32)
                .collect()
        })
        .collect();
    let set = TrainingSet {
        token_count: count * (config.context_length + 1),
        examples,
        files: Vec::new(),
    };
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let record = <_ as Recorder<B>>::load(&recorder, bytes, device)
        .map_err(|e| format!("Cannot load initial weights: {e}"))?;
    let model: Gpt<B> = config.init(device)?.load_record(record);
    let options = SessionOptions {
        batching: BatchConfig {
            batch_size: args.batch_size,
            max_batch_tokens: args.batch_size * config.context_length,
        },
        ..Default::default()
    };
    let mut session =
        TrainingSession::from_model_on_device(config, set, model, 0.001, 42, options, device)?;
    B::sync(device);
    let setup_seconds = start.elapsed().as_secs_f64();
    let warmup_start = Instant::now();
    for _ in 0..args.warmup {
        deadline(start, args.max_seconds)?;
        session.step()?;
        B::sync(device);
    }
    let warmup_seconds = warmup_start.elapsed().as_secs_f64();
    let mut durations = Vec::new();
    let mut observations = Vec::new();
    let mut targets = 0;
    let mut memory_samples = vec![json!({"phase":"warmup", "drm":drm_memory()})];
    for _ in 0..args.samples {
        deadline(start, args.max_seconds)?;
        B::sync(device);
        let update_start = Instant::now();
        let (event, stages) = if args.profile {
            let (event, t) = session.step_profiled()?;
            (
                event,
                json!({
                    "preparation":t.preparation.as_secs_f64(),
                    "forward_loss":t.forward_loss.as_secs_f64(),
                    "backward":t.backward.as_secs_f64(),
                    "gradient_validation":t.gradient_validation.as_secs_f64(),
                    "optimizer":t.optimizer.as_secs_f64(),
                    "candidate_validation":t.candidate_validation.as_secs_f64()
                }),
            )
        } else {
            (session.step()?, Value::Null)
        };
        B::sync(device);
        let seconds = update_start.elapsed().as_secs_f64();
        if !event.pre_update_loss.is_finite() || !event.gradient_norm.is_finite() {
            return Err("Non-finite benchmark loss or gradient norm".into());
        }
        targets += event.target_count;
        durations.push(seconds);
        observations.push(json!({"seconds":seconds, "real_targets":event.target_count,
            "loss":event.pre_update_loss, "gradient_norm":event.gradient_norm,
            "stage_seconds":stages}));
        memory_samples.push(json!({"phase":"update", "drm":drm_memory()}));
    }
    let total_seconds: f64 = durations.iter().sum();
    durations.sort_by(f64::total_cmp);
    let median = if durations.len() % 2 == 0 {
        (durations[durations.len() / 2 - 1] + durations[durations.len() / 2]) / 2.0
    } else {
        durations[durations.len() / 2]
    };
    Ok(
        json!({"setup_seconds":setup_seconds, "warmup_seconds":warmup_seconds,
        "warmup_updates":args.warmup, "timed_updates":args.samples, "observations":observations,
        "median_update_seconds":median, "min_update_seconds":durations[0],
        "max_update_seconds":durations[durations.len()-1], "real_targets":targets,
        "timed_seconds":total_seconds, "real_targets_per_second":targets as f64/total_seconds,
        "memory_samples":memory_samples}),
    )
}

fn deadline(start: Instant, max_seconds: u64) -> Result<(), String> {
    if start.elapsed().as_secs_f64() >= max_seconds as f64 {
        return Err("Cooperative benchmark deadline exceeded; use an external timeout to bound tensor operations".into());
    }
    Ok(())
}

fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

fn drm_memory() -> Vec<Value> {
    let Ok(entries) = fs::read_dir("/proc/self/fdinfo") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let text = fs::read_to_string(entry.path()).ok()?;
            let lines: Vec<_> = text
                .lines()
                .filter(|line| line.starts_with("drm-"))
                .collect();
            if lines.is_empty() {
                None
            } else {
                Some(json!({"fd":entry.file_name().to_string_lossy(), "fields":lines}))
            }
        })
        .collect()
}
