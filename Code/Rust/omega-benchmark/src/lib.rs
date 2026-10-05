//! Measure disposable checked updates on real dataset batches and extrapolate
//! a fixed-order run. Never saves weights, changes a tokenizer, or downloads data.
use std::{path::PathBuf, time::Instant};

use burn::tensor::backend::AutodiffBackend;
use omega_nn::{GptConfig, TrainingBackend};
use omega_tokenizer::Tokens;
use omega_training::{
    ExampleSource, SessionOptions, TrainingSession, ValidationSplit,
    assistant::prepare_conversations,
    batching::{BatchConfig, collate},
    cache::{CacheLimits, CacheOptions, CachePartition, open_token_cache},
    checkpoint::{BuildIdentity, ModelConfig, tokenizer_identity},
    cpu::execution_profile,
    dataset::load_document_corpus_with_format,
    optimization::OptimizationOptions,
    selection::{Format, resolve_dataset_format},
    split_document_corpus,
};
use serde::{Deserialize, Serialize};

pub use omega_training::selection::Format as DatasetFormat;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Cpu,
    Vulkan,
    Cuda,
}

#[derive(Clone, Debug)]
pub struct DatasetOptions {
    pub root: PathBuf,
    pub selections: Vec<String>,
    /// Absolute path or relative to `root`; IDs come from this saved tokenizer.
    pub tokenizer: PathBuf,
    pub format: DatasetFormat,
    pub validation: ValidationSplit,
    pub split_seed: u64,
    /// An existing, verified base token cache; never created or overwritten here.
    pub cache: Option<PathBuf>,
    pub cache_limits: CacheLimits,
}
impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            root: omega_training::project_root().join("datasets"),
            selections: Vec::new(),
            tokenizer: "test.json".into(),
            format: DatasetFormat::Auto,
            validation: ValidationSplit::None,
            split_seed: 42,
            cache: None,
            cache_limits: CacheLimits::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelDimensions {
    pub context_length: usize,
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub d_ff: usize,
}
impl Default for ModelDimensions {
    fn default() -> Self {
        Self {
            context_length: 64,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 128,
        }
    }
}
impl ModelDimensions {
    fn config(&self, vocab_size: usize) -> GptConfig {
        GptConfig {
            vocab_size,
            context_length: self.context_length,
            d_model: self.d_model,
            num_heads: self.heads,
            num_layers: self.layers,
            d_ff: self.d_ff,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingTimeConfig {
    pub dataset: DatasetOptions,
    pub model: ModelDimensions,
    pub backend: Backend,
    pub device: usize,
    pub batching: BatchConfig,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub optimization: OptimizationOptions,
    /// Disposable untimed updates before sampling (not LR schedule warmup).
    pub warmup: usize,
    /// Maximum timed batches; capped to the number of batches in one epoch.
    pub samples: usize,
    /// Cooperative budget for warmup + timed updates. Checked between updates;
    /// dataset preparation/model initialization and a stuck backend are not bounded.
    pub max_seconds: f64,
}
impl Default for TrainingTimeConfig {
    fn default() -> Self {
        Self {
            dataset: DatasetOptions::default(),
            model: ModelDimensions::default(),
            backend: Backend::Cpu,
            device: 0,
            batching: BatchConfig::default(),
            epochs: 1,
            learning_rate: 0.003,
            seed: 42,
            optimization: OptimizationOptions::default(),
            warmup: 2,
            samples: 12,
            max_seconds: 60.0,
        }
    }
}
impl TrainingTimeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0
            || !(1..=100).contains(&self.warmup)
            || !(1..=1000).contains(&self.samples)
            || !self.max_seconds.is_finite()
            || !(0.0..=3600.0).contains(&self.max_seconds)
            || self.max_seconds == 0.0
        {
            return Err(
                "Require epochs > 0, warmup 1..100, samples 1..1000 and max-seconds in (0, 3600]"
                    .into(),
            );
        }
        if self.backend == Backend::Cpu && self.device != 0 {
            return Err("GPU device index requires Vulkan".into());
        }
        self.batching.validate()?;
        self.model.config(1).validate()?;
        self.optimization.learning_rate(self.learning_rate, 0)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SamplePoint {
    /// Zero-based original epoch batch, retaining its actual example boundaries.
    pub batch_index: usize,
    /// Number of epoch batches represented by this sample's contiguous interval.
    pub represented_batches: usize,
}

/// Midpoint of each evenly spaced interval; intervals exactly partition an epoch.
/// This is deterministic coverage sampling, not a statistical confidence model.
pub fn sampling_plan(batches: usize, requested: usize) -> Result<Vec<SamplePoint>, String> {
    if batches == 0 || !(1..=1000).contains(&requested) {
        return Err("Sampling requires nonempty data and 1..1000 samples".into());
    }
    let count = requested.min(batches);
    Ok((0..count)
        .map(|i| {
            let lo = (i as u128 * batches as u128 / count as u128) as usize;
            let hi = ((i + 1) as u128 * batches as u128 / count as u128) as usize;
            SamplePoint {
                batch_index: lo + (hi - lo) / 2,
                represented_batches: hi - lo,
            }
        })
        .collect())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimedSample {
    pub point: SamplePoint,
    pub examples: usize,
    pub supervised_targets: usize,
    pub seconds: f64,
}

/// Extrapolate only a complete plan. Partial timed prefixes must not produce an ETA.
pub fn project_seconds(
    batches: usize,
    epochs: usize,
    samples: &[TimedSample],
) -> Result<f64, String> {
    if epochs == 0 || samples.is_empty() {
        return Err("An estimate requires epochs and measured batches".into());
    }
    let expected = sampling_plan(batches, samples.len())?;
    if samples.len() != expected.len()
        || samples
            .iter()
            .zip(&expected)
            .any(|(s, p)| s.point != *p || !s.seconds.is_finite() || s.seconds <= 0.0)
    {
        return Err("Incomplete/invalid sampling plan or nonpositive/nonfinite timing".into());
    }
    let seconds = samples
        .iter()
        .map(|s| s.seconds * s.point.represented_batches as f64)
        .sum::<f64>()
        * epochs as f64;
    if !seconds.is_finite() {
        return Err("Estimated duration overflows".into());
    }
    Ok(seconds)
}

#[derive(Debug, Serialize)]
pub struct TrainingTimeReport {
    pub schema_version: u32,
    pub backend: Backend,
    pub runtime: serde_json::Value,
    pub build: BuildIdentity,
    pub model: ModelConfig,
    pub dataset_selections: Vec<String>,
    pub dataset_format: String,
    pub source_identity: String,
    pub tokenizer_identity: serde_json::Value,
    pub objective: String,
    pub cache_used: bool,
    pub cache_limits: Option<CacheLimits>,
    pub validation_split: ValidationSplit,
    pub split_seed: u64,
    pub examples_per_epoch: usize,
    pub supervised_targets_per_epoch: usize,
    pub input_positions_per_epoch: usize,
    pub padded_positions_per_epoch: usize,
    pub batches_per_epoch: usize,
    pub epochs: usize,
    pub total_updates: usize,
    pub batch_config: BatchConfig,
    pub seed: u64,
    pub learning_rate: f64,
    pub optimization: OptimizationOptions,
    pub warmup_updates: usize,
    pub warmup_seconds: f64,
    pub setup_seconds: f64,
    pub samples: Vec<TimedSample>,
    pub measured_targets_per_second: f64,
    pub estimated_epoch_seconds: f64,
    pub estimated_training_seconds: f64,
    /// Observed preparation/model setup + projected steady-state updates.
    /// Excludes compilation, cold-kernel warmup overhead, evaluation and saves.
    pub estimated_setup_plus_training_seconds: f64,
    pub notes: Vec<String>,
}

struct IndexedSource {
    source: Box<dyn ExampleSource>,
    indices: Vec<usize>,
    identity: String,
}
impl ExampleSource for IndexedSource {
    fn example_count(&self) -> usize {
        self.indices.len()
    }
    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        self.source
            .example(*self.indices.get(index).ok_or("Sample index out of range")?)
    }
    fn target_mask(&self, index: usize) -> Result<Option<Vec<bool>>, String> {
        self.source
            .target_mask(*self.indices.get(index).ok_or("Sample index out of range")?)
    }
    fn objective(&self) -> &str {
        self.source.objective()
    }
    fn target_count(&self) -> Result<usize, String> {
        let mut total = 0usize;
        for i in 0..self.indices.len() {
            let count = self.target_mask(i)?.map_or_else(
                || self.example(i).map(|x| x.len() - 1),
                |m| Ok(m.into_iter().filter(|v| *v).count()),
            )?;
            total = total
                .checked_add(count)
                .ok_or("Sample target count overflows")?;
        }
        Ok(total)
    }
    fn identity(&self) -> Result<String, String> {
        Ok(format!(
            "benchmark-only/{}/{:?}",
            self.identity, self.indices
        ))
    }
}

fn load_source(
    options: &DatasetOptions,
    tokenizer: &Tokens,
    context: usize,
    format: Format,
) -> Result<Box<dyn ExampleSource>, String> {
    if format == Format::Chat {
        if options.cache.is_some() {
            return Err("Chat token caches are unsupported".into());
        }
        return Ok(Box::new(
            prepare_conversations(
                &options.root,
                &options.selections,
                tokenizer,
                context,
                options.validation,
                options.split_seed,
            )?
            .training,
        ));
    }
    let cache_options = CacheOptions {
        format: format.into(),
        context_length: context,
        validation: options.validation,
        split_seed: options.split_seed,
        limits: options.cache_limits.clone(),
    };
    if let Some(path) = &options.cache {
        let cache = open_token_cache(
            path,
            &options.root,
            &options.selections,
            tokenizer,
            &cache_options,
        )?;
        return Ok(Box::new(cache.partition(CachePartition::Training)));
    }
    let corpus = load_document_corpus_with_format(
        &options.root,
        &options.selections,
        tokenizer,
        format.into(),
    )?;
    Ok(Box::new(
        split_document_corpus(&corpus, context, options.validation, options.split_seed)?
            .training
            .set,
    ))
}

fn deadline(start: Instant, seconds: f64, measured: usize, required: usize) -> Result<(), String> {
    if start.elapsed().as_secs_f64() >= seconds {
        return Err(format!(
            "Benchmark budget exceeded ({measured}/{required} timed batches completed); no full-run estimate produced. Increase --max-seconds or reduce --samples/model size. Preparation is separate; this deadline is checked between updates."
        ));
    }
    Ok(())
}

/// Measure on this machine. CPU pools must be configured before calling; the CLI
/// uses a fresh child for explicit thread flags. GPU requests never fall back.
pub fn training_time(options: &TrainingTimeConfig) -> Result<TrainingTimeReport, String> {
    options.validate()?;
    let start = Instant::now();
    let cpu = execution_profile()?;
    match options.backend {
        Backend::Cpu => run::<TrainingBackend>(
            options,
            &Default::default(),
            serde_json::json!({"cpu":cpu}),
            start,
        ),
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let selected = omega_training::cuda::initialize(options.device)?;
                run::<omega_training::cuda::CudaTraining>(
                    options,
                    &selected.device,
                    serde_json::json!({"cuda":selected.profile,"host_cpu":cpu}),
                    start,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA requires --features cuda; it is experimental".into())
            }
        }
        Backend::Vulkan => {
            #[cfg(feature = "gpu")]
            {
                let selected = omega_training::gpu::initialize_vulkan(options.device)?;
                let tokenizer = Tokens::new(options.dataset.root.join(&options.dataset.tokenizer))?;
                selected.adapter().validate_model(
                    &options.model.config(tokenizer.vocab_size()),
                    options.batching.batch_size,
                )?;
                run::<omega_training::gpu::GpuTraining>(
                    options,
                    selected.device(),
                    serde_json::json!({"gpu":selected.profile(),"host_cpu":cpu}),
                    start,
                )
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err("Vulkan requires building omega-benchmark with --features gpu".into())
            }
        }
    }
}

fn run<B: AutodiffBackend<FloatElem = f32>>(
    options: &TrainingTimeConfig,
    device: &B::Device,
    runtime: serde_json::Value,
    started: Instant,
) -> Result<TrainingTimeReport, String> {
    let tokenizer = Tokens::new(options.dataset.root.join(&options.dataset.tokenizer))?;
    omega_nn::validate_tokenizer(&tokenizer)?;
    let model = options.model.config(tokenizer.vocab_size());
    model.validate()?;
    let format = resolve_dataset_format(
        &options.dataset.root,
        &options.dataset.selections,
        options.dataset.format,
        true,
        options.dataset.validation,
    )?;
    let source = load_source(&options.dataset, &tokenizer, model.context_length, format)?;
    let examples = source.example_count();
    if examples == 0 {
        return Err("Selected training partition is empty".into());
    }
    let batches = examples.div_ceil(options.batching.batch_size);
    let total_updates = batches
        .checked_mul(options.epochs)
        .ok_or("Total update count overflows")?;
    let mut targets = 0usize;
    let mut inputs = 0usize;
    let mut padded = 0usize;
    // Validate every actual batch, including IDs/masks/padding limits. No prefix
    // sampling of the corpus or silently omitted final partial batch.
    for first in (0..examples).step_by(options.batching.batch_size) {
        let end = first
            .saturating_add(options.batching.batch_size)
            .min(examples);
        let indices: Vec<_> = (first..end).collect();
        let batch = collate(source.as_ref(), &indices, &model, &options.batching, 0)?;
        targets = targets
            .checked_add(batch.target_count())
            .ok_or("Target count overflows")?;
        padded = padded
            .checked_add(
                batch
                    .batch_size()
                    .checked_mul(batch.sequence_length())
                    .ok_or("Padded count overflows")?,
            )
            .ok_or("Padded count overflows")?;
        for index in indices {
            inputs = inputs
                .checked_add(source.example(index)?.len() - 1)
                .ok_or("Input count overflows")?;
        }
    }
    if targets != source.target_count()? {
        return Err("Source target count disagrees with checked batches".into());
    }
    let identity = source.identity()?;
    let objective = source.objective().to_owned();
    let plan = sampling_plan(batches, options.samples)?;
    let mut indices = Vec::new();
    if examples >= options.batching.batch_size {
        for _ in 0..options.warmup {
            indices.extend(0..options.batching.batch_size);
        }
    }
    for point in &plan {
        let first = point.batch_index * options.batching.batch_size;
        indices.extend(
            first
                ..first
                    .saturating_add(options.batching.batch_size)
                    .min(examples),
        );
    }
    let mapped = IndexedSource {
        source,
        indices,
        identity: identity.clone(),
    };
    let mut session = TrainingSession::<_, B>::from_source_with_options_on_device(
        &model,
        mapped,
        options.learning_rate,
        options.seed,
        SessionOptions {
            batching: options.batching,
            optimization: options.optimization.clone(),
            ..Default::default()
        },
        device,
    )?;
    B::sync(device);
    let setup_seconds = started.elapsed().as_secs_f64();
    let clock = Instant::now();
    for _ in 0..options.warmup {
        deadline(clock, options.max_seconds, 0, plan.len())?;
        session.step()?;
        B::sync(device);
    }
    let warmup_seconds = clock.elapsed().as_secs_f64();
    let count = plan.len();
    let mut samples = Vec::with_capacity(count);
    for point in plan {
        deadline(clock, options.max_seconds, samples.len(), count)?;
        B::sync(device);
        let step = Instant::now();
        let event = session.step()?;
        B::sync(device);
        samples.push(TimedSample {
            point,
            examples: event.example_count,
            supervised_targets: event.target_count,
            seconds: step.elapsed().as_secs_f64(),
        });
        deadline(clock, options.max_seconds, samples.len(), count)?;
    }
    let estimated = project_seconds(batches, options.epochs, &samples)?;
    let elapsed = samples.iter().map(|s| s.seconds).sum::<f64>();
    let measured_targets = samples
        .iter()
        .map(|s| s.supervised_targets as f64)
        .sum::<f64>();
    Ok(TrainingTimeReport {
        schema_version: 1, backend: options.backend, runtime, build: BuildIdentity::current(),
        model: ModelConfig::from(&model), dataset_selections: options.dataset.selections.clone(),
        dataset_format: format!("{format:?}").to_lowercase(), source_identity: identity,
        tokenizer_identity: serde_json::to_value(tokenizer_identity(&tokenizer)?).map_err(|e| e.to_string())?,
        objective, cache_used: options.dataset.cache.is_some(), validation_split: options.dataset.validation,
        cache_limits: options.dataset.cache.as_ref().map(|_| options.dataset.cache_limits.clone()),
        split_seed: options.dataset.split_seed, examples_per_epoch: examples,
        supervised_targets_per_epoch: targets, input_positions_per_epoch: inputs, padded_positions_per_epoch: padded,
        batches_per_epoch: batches, epochs: options.epochs, total_updates, batch_config: options.batching,
        seed: options.seed, learning_rate: options.learning_rate, optimization: options.optimization.clone(),
        warmup_updates: options.warmup, warmup_seconds, setup_seconds, samples,
        measured_targets_per_second: measured_targets / elapsed,
        estimated_epoch_seconds: estimated / options.epochs as f64,
        estimated_training_seconds: estimated, estimated_setup_plus_training_seconds: setup_seconds + estimated,
        notes: vec![
            "Fixed dataset order; midpoint batch from each contiguous interval. Weighted interval times cover the full epoch; no shuffle or dataset-weight projection.".into(),
            "Fresh disposable model/Adam, matching supplied dimensions/tokenizer/batching. Estimates time for the requested epochs, not time to reach a quality target.".into(),
            "Excludes build time, initial warmup, logging, checkpoint saves and held-out evaluation. Warmup and observed full-dataset setup are reported separately; unseen GPU shapes may still compile during sampling.".into(),
            format!("Measured {count} of {batches} epoch batches; short samples and varying shapes/background load limit accuracy. No confidence interval is claimed."),
        ],
    })
}
