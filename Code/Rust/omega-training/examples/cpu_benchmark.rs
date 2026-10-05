//! One bounded CPU benchmark case per process. Build with --release; set pool
//! environment in the parent before launching. No corpus/checkpoint I/O occurs.
//! Deadlines are cooperative between operations: a supervisor must enforce a
//! hard wall limit because a tensor operation cannot be interrupted safely.
#![recursion_limit = "256"]

use std::{collections::BTreeMap, hint::black_box, process::ExitCode, time::Instant};

use burn::tensor::{Bool, Int, Tensor, backend::Backend};
use clap::{Parser, ValueEnum};
use omega_nn::{Cpu, Gpt, GptConfig, generate, token_tensor};
use omega_training::{
    SessionOptions, TrainingSession, TrainingSet,
    batching::{BatchConfig, collate},
    checkpoint::{BuildIdentity, ModelConfig, sha256_bytes},
};
use serde::Serialize;
use serde_json::json;

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Train,
    Forward,
    Generate,
}

fn bounded<const MIN: usize, const MAX: usize>(value: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| "Expected an unsigned integer")?;
    if !(MIN..=MAX).contains(&value) {
        return Err(format!("Expected {MIN}..={MAX}"));
    }
    Ok(value)
}
fn batch_size(value: &str) -> Result<usize, String> {
    let value = bounded::<1, 4>(value)?;
    if ![1, 4].contains(&value) {
        return Err("batch-size must be 1 or 4".into());
    }
    Ok(value)
}
fn context_length(value: &str) -> Result<usize, String> {
    let value = bounded::<32, 128>(value)?;
    if ![32, 128].contains(&value) {
        return Err("context-length must be 32 or 128".into());
    }
    Ok(value)
}

#[derive(Debug, Parser)]
#[command(about = "Bounded release CPU baseline; one case per fresh process")]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    /// Opt-in checked trainer stage timings; normal runs call step() without clocks.
    #[arg(long)]
    profile: bool,
    #[arg(long, default_value_t=1, value_parser=batch_size)]
    batch_size: usize,
    #[arg(long, default_value_t=32, value_parser=context_length)]
    context_length: usize,
    #[arg(long, default_value_t=5, value_parser=bounded::<1,20>)]
    samples: usize,
    #[arg(long, default_value_t=1, value_parser=bounded::<1,8>)]
    iterations: usize,
    #[arg(long, default_value_t=1, value_parser=bounded::<1,3>)]
    warmup: usize,
    /// Cooperative whole-case limit including setup/warmup; supervisor enforces hard limit.
    #[arg(long, default_value_t=5, value_parser=bounded::<1,10>)]
    max_seconds: usize,
    /// New tokens per full-prefix request (generation only).
    #[arg(long, default_value_t=8, value_parser=bounded::<1,16>)]
    new_tokens: usize,
}

fn synthetic(context: usize) -> TrainingSet {
    let examples: Vec<_> = (0..16)
        .map(|row| {
            (0..=context)
                .map(|position| ((row * 17 + position * 7 + 3) % 64) as u32)
                .collect()
        })
        .collect();
    TrainingSet {
        examples,
        files: Vec::new(),
        token_count: 16 * (context + 1),
    }
}

fn data(tensor: Tensor<Cpu, 3>) -> Result<Vec<f32>, String> {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("Cannot read benchmark logits: {e:?}"))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite benchmark logits".into());
    }
    Ok(values)
}
fn logits_hash(values: &[f32]) -> String {
    sha256_bytes(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

enum Workload {
    Train(Box<TrainingSession>),
    Forward {
        model: Box<Gpt<Cpu>>,
        inputs: Tensor<Cpu, 2, Int>,
        valid: Tensor<Cpu, 2, Bool>,
        rows: usize,
        positions: usize,
    },
    Generate {
        model: Box<Gpt<Cpu>>,
        config: GptConfig,
        prompts: Vec<Vec<u32>>,
        new_tokens: usize,
    },
}

#[derive(Default)]
struct Observation {
    targets: usize,
    generated: usize,
    positions: usize,
    loss: Option<f32>,
    gradient_norm: Option<f64>,
    logits: Option<Vec<f32>>,
    tokens: Option<Vec<Vec<u32>>>,
    stages: Option<StageSeconds>,
}

#[derive(Default, Serialize)]
struct StageSeconds {
    preparation: f64,
    forward_loss: f64,
    backward: f64,
    gradient_validation: f64,
    optimizer: f64,
    candidate_validation: f64,
}
impl StageSeconds {
    fn from_timings(value: omega_training::trainer::StepTimings) -> Self {
        Self {
            preparation: value.preparation.as_secs_f64(),
            forward_loss: value.forward_loss.as_secs_f64(),
            backward: value.backward.as_secs_f64(),
            gradient_validation: value.gradient_validation.as_secs_f64(),
            optimizer: value.optimizer.as_secs_f64(),
            candidate_validation: value.candidate_validation.as_secs_f64(),
        }
    }
    fn add(&mut self, other: &Self) {
        self.preparation += other.preparation;
        self.forward_loss += other.forward_loss;
        self.backward += other.backward;
        self.gradient_validation += other.gradient_validation;
        self.optimizer += other.optimizer;
        self.candidate_validation += other.candidate_validation;
    }
}

impl Workload {
    fn setup(args: &Args, config: &GptConfig) -> Result<(Self, Vec<f32>), String> {
        let set = synthetic(args.context_length);
        let probe = token_tensor::<Cpu>(
            &set.examples[0][..args.context_length],
            64,
            args.context_length,
            &Default::default(),
        )?;
        let batching = BatchConfig {
            batch_size: args.batch_size,
            max_batch_tokens: args.batch_size * args.context_length,
        };
        if matches!(args.mode, Mode::Train) {
            let session = TrainingSession::new_with_options(
                config,
                set,
                0.001,
                42,
                SessionOptions {
                    batching,
                    ..Default::default()
                },
            )?;
            // Force all lazy model parameters before timing any update. First
            // Adam-state creation is exercised by the mandatory warmup update.
            let initial = data(session.inference_model().forward(probe))?;
            return Ok((Self::Train(Box::new(session)), initial));
        }
        Cpu::seed(42);
        let model = config.init::<Cpu>(&Default::default())?;
        let initial = data(model.forward(probe))?;
        if matches!(args.mode, Mode::Forward) {
            let indices: Vec<_> = (0..args.batch_size).collect();
            let tensors = collate(&set, &indices, config, &batching, 0)?
                .into_tensors::<Cpu>(&Default::default());
            Ok((
                Self::Forward {
                    model: Box::new(model),
                    inputs: tensors.inputs,
                    valid: tensors.valid,
                    rows: args.batch_size,
                    positions: args.batch_size * args.context_length,
                },
                initial,
            ))
        } else {
            let length = args.context_length - args.new_tokens;
            let prompts = set.examples[..args.batch_size]
                .iter()
                .map(|ids| ids[..length].to_vec())
                .collect();
            Ok((
                Self::Generate {
                    model: Box::new(model),
                    config: config.clone(),
                    prompts,
                    new_tokens: args.new_tokens,
                },
                initial,
            ))
        }
    }

    fn step(&mut self, profile: bool) -> Result<Observation, String> {
        match self {
            Self::Train(session) => {
                let (event, stages) = if profile {
                    let (event, timings) = session.step_profiled()?;
                    (event, Some(StageSeconds::from_timings(timings)))
                } else {
                    (session.step()?, None)
                };
                Ok(Observation {
                    targets: event.target_count,
                    loss: Some(event.pre_update_loss),
                    gradient_norm: Some(event.gradient_norm),
                    stages,
                    ..Default::default()
                })
            }
            Self::Forward {
                model,
                inputs,
                valid,
                rows,
                positions,
            } => {
                let logits = if *rows == 1 {
                    model.forward(inputs.clone())
                } else {
                    model.forward_masked(inputs.clone(), valid.clone())?
                };
                // Host materialization/finiteness is deliberately timed so a lazy
                // backend cannot make a forward appear faster than its execution.
                Ok(Observation {
                    positions: *positions,
                    logits: Some(data(logits)?),
                    ..Default::default()
                })
            }
            Self::Generate {
                model,
                config,
                prompts,
                new_tokens,
            } => {
                // The public generator is single-prompt. Multiple rows here are
                // serial requests, never described as batched inference.
                let tokens: Vec<_> = prompts
                    .iter()
                    .map(|prompt| generate(model, config, prompt, *new_tokens, None))
                    .collect::<Result<_, _>>()?;
                let generated = tokens
                    .iter()
                    .zip(prompts.iter())
                    .map(|(ids, prompt)| ids.len() - prompt.len())
                    .sum();
                Ok(Observation {
                    generated,
                    tokens: Some(tokens),
                    ..Default::default()
                })
            }
        }
    }
}

#[derive(Serialize)]
struct Sample {
    seconds: f64,
    iterations: usize,
    real_targets: usize,
    generated_tokens: usize,
    input_positions: usize,
    units_per_second: f64,
    stages: Option<StageSeconds>,
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

fn source_identity() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("cpu_benchmark.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../omega-nn/Cargo.toml"),
        include_bytes!("../src/trainer.rs"),
        include_bytes!("../src/optimization.rs"),
        include_bytes!("../src/batching.rs"),
        include_bytes!("../src/sampling.rs"),
        include_bytes!("../src/dataset.rs"),
        include_bytes!("../../omega-nn/src/lib.rs"),
        include_bytes!("../../omega-nn/src/model.rs"),
        include_bytes!("../../omega-nn/src/loss.rs"),
        include_bytes!("../../omega-nn/src/generation.rs"),
    ];
    let mut bytes = b"omega-cpu-benchmark-source-v1\0".to_vec();
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    sha256_bytes(&bytes)
}

fn run(args: Args) -> Result<serde_json::Value, String> {
    if cfg!(debug_assertions) {
        return Err("Benchmark requires a release build: cargo build -p omega-training --example cpu_benchmark --release --locked".into());
    }
    if args.profile && !matches!(args.mode, Mode::Train) {
        return Err("--profile is supported only for --mode train".into());
    }
    let started = Instant::now();
    let config = GptConfig {
        vocab_size: 64,
        context_length: args.context_length,
        d_model: 32,
        num_heads: 4,
        num_layers: 1,
        d_ff: 64,
    };
    let (mut workload, initial_logits) = Workload::setup(&args, &config)?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let exhausted = || started.elapsed().as_secs_f64() >= args.max_seconds as f64;
    let warmup_started = Instant::now();
    let mut warmups = 0;
    let mut last = Observation::default();
    while warmups < args.warmup && !exhausted() {
        last = black_box(workload.step(args.profile)?);
        warmups += 1;
    }
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();
    let mut samples = Vec::new();
    if warmups == args.warmup {
        for _ in 0..args.samples {
            if exhausted() {
                break;
            }
            let clock = Instant::now();
            let mut sample = Sample {
                seconds: 0.0,
                iterations: 0,
                real_targets: 0,
                generated_tokens: 0,
                input_positions: 0,
                units_per_second: 0.0,
                stages: args.profile.then(StageSeconds::default),
            };
            for _ in 0..args.iterations {
                if exhausted() {
                    break;
                }
                let observed = black_box(workload.step(args.profile)?);
                sample.iterations += 1;
                sample.real_targets += observed.targets;
                sample.generated_tokens += observed.generated;
                sample.input_positions += observed.positions;
                if let (Some(total), Some(stages)) = (&mut sample.stages, &observed.stages) {
                    total.add(stages);
                }
                last = observed;
            }
            sample.seconds = clock.elapsed().as_secs_f64();
            if sample.iterations == 0 {
                break;
            }
            if sample.seconds <= 0.0 {
                return Err("Benchmark clock did not advance".into());
            }
            sample.units_per_second =
                (sample.real_targets + sample.generated_tokens + sample.input_positions) as f64
                    / sample.seconds;
            samples.push(sample);
        }
    }
    let total_seconds = started.elapsed().as_secs_f64();
    let complete = warmups == args.warmup
        && samples.len() == args.samples
        && samples.iter().all(|s| s.iterations == args.iterations)
        && total_seconds <= args.max_seconds as f64;
    let times: Vec<_> = samples.iter().map(|s| s.seconds).collect();
    let rates: Vec<_> = samples.iter().map(|s| s.units_per_second).collect();
    // Correctness probing is outside the timed case; the parent still applies
    // its hard wall deadline to the entire child including this final forward.
    let probe_started = Instant::now();
    let final_logits = if let Workload::Train(session) = &workload {
        let ids: Vec<_> = (0..args.context_length)
            .map(|position| ((position * 7 + 3) % 64) as u32)
            .collect();
        let input = token_tensor::<Cpu>(&ids, 64, args.context_length, &Default::default())?;
        Some(data(session.inference_model().forward(input))?)
    } else {
        None
    };
    let final_probe_seconds = probe_started.elapsed().as_secs_f64();
    let environment: BTreeMap<_, _> = [
        "RAYON_NUM_THREADS",
        "RAYON_RS_NUM_CPUS",
        "MATMUL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    .into_iter()
    .map(|key| (key, std::env::var(key).ok()))
    .collect();
    let unit = match args.mode {
        Mode::Train => "real_targets_per_second",
        Mode::Forward => "input_positions_per_second",
        Mode::Generate => "generated_tokens_per_second",
    };
    let progress = match &workload {
        Workload::Train(session) => Some(session.progress()),
        _ => None,
    };
    Ok(json!({
        "schema_version":1,"status":if complete {"complete"} else {"budget_exhausted"},
        "mode":args.mode,"profiled":args.profile,"batch_size":args.batch_size,"context_length":args.context_length,"model":ModelConfig::from(&config),
        "seed":42,"learning_rate":0.001,"synthetic_examples":16,"synthetic_token_rule":"(row*17+position*7+3)%64",
        "new_tokens":args.new_tokens,"generation_requests":"serial_single_prompt","forward_path":"batch1:unmasked;batch4:masked;host_read_and_finiteness_included",
        "requested_samples":args.samples,"iterations_per_sample":args.iterations,"requested_warmup":args.warmup,"warmup_iterations_completed":warmups,
        "max_seconds":args.max_seconds,"deadline_policy":"cooperative_between_operations;supervisor_required_for_hard_limit",
        "setup_seconds":setup_seconds,"warmup_seconds":warmup_seconds,"total_seconds":total_seconds,"final_probe_seconds":final_probe_seconds,
        "summary":{"sample_count":samples.len(),"median_seconds":median(&times),"min_seconds":times.iter().copied().reduce(f64::min),"max_seconds":times.iter().copied().reduce(f64::max),"median_units_per_second":median(&rates),"total_timed_seconds":times.iter().sum::<f64>(),"unit":unit},
        "samples":samples,"progress_including_warmup":progress,
        "fingerprint":{"initial_logits_sha256":logits_hash(&initial_logits),"initial_logits_prefix":&initial_logits[..8],"last_pre_update_loss":last.loss,"last_gradient_norm":last.gradient_norm,"final_trained_logits_sha256":final_logits.as_ref().map(|v|logits_hash(v)),"final_trained_logits_prefix":final_logits.as_ref().map(|v|v[..8].to_vec()),"forward_logits_sha256":last.logits.as_ref().map(|v|logits_hash(v)),"forward_logits_prefix":last.logits.as_ref().map(|v|v[..8].to_vec()),"generated_ids":last.tokens},
        "build":BuildIdentity::current(),"source_sha256":source_identity(),"debug_assertions":cfg!(debug_assertions),"profile":"release_required","target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH,
        "declared_burn_features":["std","ndarray","autodiff"],"effective_feature_graph":"capture separately with build invocation;Cargo feature unification may add features",
        "thread_environment":environment,"effective_threads":null,"available_parallelism":std::thread::available_parallelism().ok().map(|v|v.get()),"process_memory_bytes":null,"memory_measurement":"supervisor_required",
        "timing_scope":"setup and mandatory warmup excluded;train uses actual checked persistent session;forward is a proxy,not subtractable stage instrumentation;report serialization/fingerprints excluded",
        "stage_scope":"optional train-only stages;seconds summed per sample;final commit/accounting and timer overhead not attributed to stages"
    }))
}

fn main() -> ExitCode {
    match run(Args::parse())
        .and_then(|report| serde_json::to_string(&report).map_err(|e| e.to_string()))
    {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("CPU benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cli_caps_and_fixture_are_checked_without_training() {
        for args in [
            vec!["bench", "--mode", "train", "--samples", "21"],
            vec!["bench", "--mode", "train", "--max-seconds", "11"],
            vec!["bench", "--mode", "forward", "--batch-size", "2"],
            vec!["bench", "--mode", "generate", "--context-length", "64"],
            vec!["bench", "--mode", "train", "--warmup", "0"],
        ] {
            assert!(Args::try_parse_from(args).is_err());
        }
        let args = Args::try_parse_from([
            "bench",
            "--mode",
            "train",
            "--batch-size",
            "4",
            "--context-length",
            "128",
        ])
        .unwrap();
        assert_eq!(args.samples, 5);
        assert_eq!(args.warmup, 1);
        let set = synthetic(32);
        assert_eq!(set.examples.len(), 16);
        assert!(
            set.examples
                .iter()
                .all(|ids| ids.len() == 33 && ids.iter().all(|id| *id < 64))
        );
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }
}
