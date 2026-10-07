//! Typed command operations shared by the standalone CLI and Omega workers.
//! Library callers configure CPU pools before calling and own cooperative signals.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OperationEvent {
    pub kind: String,
    pub data: serde_json::Value,
}

pub struct OperationControl {
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub observer: Box<dyn Fn(OperationEvent) -> Result<(), String> + Send + Sync>,
}
impl OperationControl {
    pub fn emit(&self, kind: &str, data: serde_json::Value) -> Result<(), String> {
        (self.observer)(OperationEvent {
            kind: kind.into(),
            data,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStopReason {
    Completed,
    SegmentLimit,
    UserStop,
    TimeBudget,
}

/// A successful bounded invocation always names a verified completed checkpoint.
/// Counts are cumulative within the stage. Time includes loading and final save;
/// budget overruns are reported because committed updates cannot be interrupted.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OperationOutcome {
    pub reason: OperationStopReason,
    pub checkpoint: std::path::PathBuf,
    pub completed_updates: usize,
    pub total_updates: usize,
    pub elapsed_seconds: f64,
    pub budget_overrun_seconds: f64,
}

struct SegmentExecution {
    budget: SegmentBudget,
    output_root: Option<PathBuf>,
    outcome: std::cell::RefCell<Option<OperationOutcome>>,
}

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::OpenOptions,
    io::{Write, stdout},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::atomic::Ordering,
};

use crate::batching::BatchConfig;
use crate::cache::{
    CacheLimits, CacheOptions, CachePartition, CachedPartition, create_token_cache,
    open_token_cache,
};
use crate::checkpoint::load_checkpoint_on_device;
use crate::checkpoint::{
    BuildIdentity, CheckpointMetadata, DatasetProvenance, DocumentFingerprint, TrainingProvenance,
    read_checkpoint_manifest, sha256_bytes,
};
use crate::checkpoint_catalog::{CatalogLimits, CatalogMode, discover_checkpoints};
use crate::dataset::{
    DatasetFormat, ExampleSource, load_document_corpus_with_format, load_text_documents,
};
use crate::evaluation::{evaluate_source_controlled_on_device, evaluate_source_on_device};
use crate::optimization::OptimizationOptions;
use crate::resume::{
    ParentCheckpoint, load_training_checkpoint, save_training_checkpoint_with_parent,
};
use crate::run_control::{SaveSchedule, SegmentBudget, advance_bounded};
use crate::sampling::{SamplingGroup, SamplingPolicy};
use crate::trainer::SessionOptions;
use crate::{
    DatasetSplit, DocumentCorpus, EvaluationMetrics, JsonlMetrics, TrainingSession, TrainingSet,
    ValidationSplit, project_root, split_document_corpus,
};
use burn::tensor::backend::AutodiffBackend;
use clap::{Parser, Subcommand, ValueEnum};
use omega_nn::{
    GenerationOptions, GptConfig, encode_text, generate_with_options_on_device, validate_tokenizer,
};
use omega_tokenizer::{ByteBpeConfig, Tokens, train_byte_bpe};

use crate::selection::{Format, resolve_dataset_format};

#[derive(Clone, Debug, clap::Args)]
#[group(required = true, multiple = false)]
pub struct CheckpointInput {
    /// Exact checkpoint directory name under the weights root
    #[arg(long, value_parser = checkpoint_name)]
    pub checkpoint: Option<String>,
    /// Select the latest completed checkpoint in this run; no fallback on load failure
    #[arg(long, value_parser = run_name)]
    pub latest_run: Option<String>,
}

impl CheckpointInput {
    fn resolve(&self, root: &Path, mode: CatalogMode) -> Result<PathBuf, String> {
        match (&self.checkpoint, &self.latest_run) {
            (Some(name), None) => Ok(root.join(name)),
            (None, Some(run)) => {
                let catalog = discover_checkpoints(root, run, &CatalogLimits::default())?;
                let selected = catalog.latest(mode)?;
                for skipped in &selected.skipped {
                    eprintln!("Skipped {}: {}", skipped.name, skipped.reason);
                }
                eprintln!(
                    "Selected checkpoint: {} ({})",
                    selected.path.display(),
                    selected.inspection
                );
                Ok(selected.path)
            }
            _ => Err("Specify exactly one of --checkpoint or --latest-run".into()),
        }
    }
}

#[derive(Clone, Debug, Default, clap::Args)]
pub struct GenerationFlags {
    /// Sample from probabilities instead of greedy argmax
    #[arg(long)]
    pub sample: bool,
    /// Positive finite sampling temperature (default with --sample: 1)
    #[arg(long, requires = "sample")]
    pub temperature: Option<f64>,
    /// Retain at most this many highest-probability tokens
    #[arg(long, requires = "sample")]
    pub top_k: Option<usize>,
    /// Nucleus probability in (0, 1], measured after top-k filtering
    #[arg(long, requires = "sample")]
    pub top_p: Option<f64>,
    /// Request-local sampling seed (default with --sample: 42)
    #[arg(long, requires = "sample")]
    pub sampling_seed: Option<u64>,
}

impl GenerationFlags {
    fn options(self) -> Result<GenerationOptions, String> {
        GenerationOptions::from_sampling_flags(
            self.sample,
            self.temperature,
            self.top_k,
            self.top_p,
            self.sampling_seed,
        )
    }
}

#[derive(Clone, Debug, clap::Args)]
pub struct CatalogArgs {
    #[arg(long)]
    pub weights_root: Option<PathBuf>,
    #[arg(long, value_parser = run_name)]
    pub run: String,
    /// Emit machine-readable JSON instead of text
    #[arg(long)]
    pub json: bool,
    /// Maximum directory entries inspected, including unrelated names
    #[arg(long, default_value_t = 100_000)]
    pub max_entries: usize,
    /// Maximum aggregate metadata bytes read for each candidate
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    pub max_metadata_bytes: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SelectionMode {
    Inference,
    Resume,
}

impl From<SelectionMode> for CatalogMode {
    fn from(mode: SelectionMode) -> Self {
        match mode {
            SelectionMode::Inference => Self::Inference,
            SelectionMode::Resume => Self::Resume,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum CheckpointCommand {
    /// Inspect checkpoint headers without loading tensors or verifying model payloads
    List {
        #[command(flatten)]
        catalog: CatalogArgs,
    },
    /// Select the highest completed numeric suffix for one operation
    Latest {
        #[command(flatten)]
        catalog: CatalogArgs,
        #[arg(long, value_enum, default_value_t = SelectionMode::Inference)]
        mode: SelectionMode,
    },
}

#[derive(Clone, Debug, clap::Args)]
pub struct Limits {
    #[arg(long, default_value_t = CacheLimits::default().max_source_bytes)]
    pub max_source_bytes: u64,
    #[arg(long, default_value_t = CacheLimits::default().max_document_tokens)]
    pub max_document_tokens: usize,
    #[arg(long, default_value_t = CacheLimits::default().max_documents)]
    pub max_documents: usize,
    #[arg(long, default_value_t = CacheLimits::default().max_directory_entries)]
    pub max_directory_entries: usize,
    #[arg(long, default_value_t = CacheLimits::default().max_manifest_bytes)]
    pub max_manifest_bytes: usize,
}

impl From<Limits> for CacheLimits {
    fn from(value: Limits) -> Self {
        Self {
            max_source_bytes: value.max_source_bytes,
            max_document_tokens: value.max_document_tokens,
            max_documents: value.max_documents,
            max_directory_entries: value.max_directory_entries,
            max_manifest_bytes: value.max_manifest_bytes,
        }
    }
}

#[derive(Clone, Debug, Default, clap::Args)]
pub struct SaveOptions {
    /// Save after each N cumulative optimizer updates (positive)
    #[arg(long)]
    pub save_every_updates: Option<usize>,
    /// Save after each N completed epochs (positive)
    #[arg(long)]
    pub save_every_epochs: Option<usize>,
}

impl From<SaveOptions> for SaveSchedule {
    fn from(value: SaveOptions) -> Self {
        Self {
            every_updates: value.save_every_updates,
            every_epochs: value.save_every_epochs,
        }
    }
}

#[derive(Clone, Debug, clap::Args)]
pub struct SessionFlags {
    /// Shuffle all training examples once per epoch, using --seed
    #[arg(long, conflicts_with = "dataset_weights")]
    pub shuffle: bool,
    /// Sample dataset groups with replacement; specify every selected FOLDER=WEIGHT
    #[arg(long = "dataset-weight", value_parser = dataset_weight)]
    pub dataset_weights: Vec<(String, u64)>,
    /// Draws in a weighted epoch (default: training example count)
    #[arg(long, requires = "dataset_weights")]
    pub samples_per_epoch: Option<usize>,
    /// Maximum examples per optimizer update; retain the final partial batch
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
    /// Maximum padded input positions per batch (not a process memory cap)
    #[arg(long, default_value_t = 65_536)]
    pub max_batch_tokens: usize,
    /// Clip the global gradient L2 norm to this positive value
    #[arg(long)]
    pub gradient_clip_norm: Option<f64>,
    /// Linearly increase learning rate over this many optimizer updates
    #[arg(long, default_value_t = 0)]
    pub warmup_updates: usize,
}

fn dataset_weight(value: &str) -> Result<(String, u64), String> {
    let (name, weight) = value.rsplit_once('=').ok_or("Expected FOLDER=WEIGHT")?;
    let weight = weight
        .parse::<u64>()
        .map_err(|_| "Dataset weight must be a positive integer")?;
    if name.is_empty() || weight == 0 {
        return Err("Expected nonempty FOLDER and positive WEIGHT".into());
    }
    Ok((name.to_owned(), weight))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum BackendChoice {
    #[default]
    Cpu,
    Vulkan,
    Cuda,
}

/// Prepare tokenizers, train a GPT, evaluate and generate from checkpoints.
#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Args {
    #[command(flatten)]
    pub cpu: crate::cpu::CpuThreadSettings,
    /// Compute backend; Vulkan requires a build with --features gpu
    #[arg(long, global = true, value_enum, default_value_t = BackendChoice::Cpu)]
    pub backend: BackendChoice,
    /// Index among discrete Vulkan adapters shown by devices
    #[arg(long, global = true)]
    pub device: Option<usize>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List discrete Vulkan adapters without initializing model tensors
    Devices,
    /// Begin fresh assistant optimization from our verified base weights (not exact resume)
    TrainStage {
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        #[arg(long)]
        weights_root: Option<PathBuf>,
        #[command(flatten)]
        input: CheckpointInput,
        #[arg(long, value_parser = run_name)]
        name: String,
        #[arg(short = 'd', long = "dataset", required = true)]
        datasets: Vec<String>,
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 0.0001)]
        learning_rate: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        max_updates: Option<usize>,
        #[arg(long)]
        metrics_jsonl: Option<PathBuf>,
        #[arg(long)]
        quiet: bool,
        #[command(flatten)]
        saves: SaveOptions,
        #[command(flatten)]
        session_options: Box<SessionFlags>,
    },
    /// Local bounded-history chat with the checkpoint's explicit Omega protocol
    Chat {
        /// Library callers can continue a previously accepted conversation.
        #[arg(skip)]
        history: Vec<omega_tokenizer::chat::ChatMessage>,
        #[arg(long)]
        weights_root: Option<PathBuf>,
        #[command(flatten)]
        input: CheckpointInput,
        /// One reply and exit; omit for interactive stdin (/reset and /exit)
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[command(flatten)]
        generation: GenerationFlags,
    },
    /// Train a fresh model on one or more folders under the dataset root
    Train {
        /// Checkpoint prefix, e.g. foo creates weights/foo-1, foo-2, ...
        #[arg(long, alias = "run-name", value_parser = run_name)]
        name: String,
        /// Dataset root; relative overrides use the working directory (default: checkout datasets/)
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        /// Checkpoint root; relative overrides use the working directory (default: checkout weights/)
        #[arg(long)]
        weights_root: Option<PathBuf>,
        /// Folder under the dataset root; published releases require an exact train partition
        #[arg(short = 'd', long = "dataset", required = true, action = clap::ArgAction::Append)]
        datasets: Vec<String>,
        /// Infer published base/chat partitions, or select an explicit format (ordinary default: text)
        #[arg(long, value_enum, default_value_t = Format::Auto)]
        dataset_format: Format,
        /// Tokenizer JSON relative to the dataset root or an absolute path
        #[arg(short = 'f', long, default_value = "test.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 0.003)]
        learning_rate: f64,
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
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Fraction of whole documents reserved for validation before chunking
        #[arg(long, conflicts_with = "validation_count")]
        validation_ratio: Option<f64>,
        /// Exact number of whole documents reserved for validation
        #[arg(long, conflicts_with = "validation_ratio")]
        validation_count: Option<usize>,
        #[arg(long, default_value_t = 42)]
        split_seed: u64,
        /// Write versioned JSON Lines metrics to a new file (never overwrite)
        #[arg(long)]
        metrics_jsonl: Option<PathBuf>,
        /// Suppress live progress and loss output
        #[arg(long)]
        quiet: bool,
        /// Use an existing token cache after verifying sources and settings
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Stop after at most this many updates and save a resumable boundary
        #[arg(long)]
        max_updates: Option<usize>,
        #[command(flatten)]
        saves: SaveOptions,
        #[command(flatten)]
        limits: Limits,
        #[command(flatten)]
        session_options: Box<SessionFlags>,
    },
    /// Continue saved Adam/model state toward a total epoch count
    Resume {
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        #[arg(long)]
        weights_root: Option<PathBuf>,
        #[command(flatten)]
        input: CheckpointInput,
        /// Prefix for a new checkpoint; the input checkpoint is never modified
        #[arg(long, value_parser = run_name)]
        name: String,
        /// Total target epochs, including already completed epochs
        #[arg(long)]
        epochs: usize,
        #[arg(long)]
        max_updates: Option<usize>,
        #[command(flatten)]
        saves: SaveOptions,
        #[arg(long)]
        cache: Option<PathBuf>,
        #[command(flatten)]
        limits: Limits,
        #[arg(long)]
        metrics_jsonl: Option<PathBuf>,
        #[arg(long)]
        quiet: bool,
    },
    /// Build a new bounded token cache; never replace an existing path
    PrepareCache {
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        #[arg(short = 'd', long = "dataset", required = true, action = clap::ArgAction::Append)]
        datasets: Vec<String>,
        #[arg(short = 'f', long, default_value = "test.json")]
        tokenizer: PathBuf,
        #[arg(long, value_enum, default_value_t = Format::Auto)]
        dataset_format: Format,
        #[arg(long, default_value_t = 64)]
        context_length: usize,
        #[arg(long, conflicts_with = "validation_count")]
        validation_ratio: Option<f64>,
        #[arg(long, conflicts_with = "validation_ratio")]
        validation_count: Option<usize>,
        #[arg(long, default_value_t = 42)]
        split_seed: u64,
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        limits: Limits,
    },
    /// Load saved weights and their tokenizer for greedy or sampled generation
    Generate {
        /// Checkpoint root; relative overrides use the working directory (default: checkout weights/)
        #[arg(long)]
        weights_root: Option<PathBuf>,
        #[command(flatten)]
        input: CheckpointInput,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,
        #[arg(long)]
        eos_token_id: Option<u32>,
        #[command(flatten)]
        generation: GenerationFlags,
    },
    /// Evaluate fixed checkpoint weights on selected text documents
    Evaluate {
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        #[arg(long)]
        weights_root: Option<PathBuf>,
        #[command(flatten)]
        input: CheckpointInput,
        #[arg(short = 'd', long = "dataset", required = true, action = clap::ArgAction::Append)]
        datasets: Vec<String>,
        #[arg(long, value_enum, default_value_t = Format::Auto)]
        dataset_format: Format,
        /// Evaluate the seeded validation partition; otherwise evaluate all selected documents
        #[arg(long, conflicts_with = "validation_count")]
        validation_ratio: Option<f64>,
        #[arg(long, conflicts_with = "validation_ratio")]
        validation_count: Option<usize>,
        #[arg(long, default_value_t = 42)]
        split_seed: u64,
    },
    /// Read-only checkpoint inventory and latest selection
    Checkpoints {
        #[command(subcommand)]
        command: CheckpointCommand,
    },
    /// Train case-preserving byte BPE with no automatic special tokens
    TrainTokenizer {
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        #[arg(short = 'd', long = "dataset", required = true, action = clap::ArgAction::Append)]
        datasets: Vec<String>,
        #[arg(long, value_enum, default_value_t = Format::Auto)]
        dataset_format: Format,
        /// New tokenizer path, relative to the working directory or absolute
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
        #[arg(long, default_value_t = 2)]
        min_frequency: u64,
        /// Register the frozen Omega v1 role/end-turn protocol before model training
        #[arg(long)]
        chat_protocol: bool,
    },
    /// Count encoded tokens and optionally an explicitly identified unknown token
    Coverage {
        #[arg(long)]
        datasets_root: Option<PathBuf>,
        /// Tokenizer JSON relative to the dataset root or an absolute path
        #[arg(short = 'f', long)]
        tokenizer: PathBuf,
        #[arg(long)]
        text: String,
        /// Caller-supplied unknown ID; omitted means unknown metrics are unavailable
        #[arg(long)]
        unknown_token_id: Option<u32>,
    },
}

fn run_name(value: &str) -> Result<String, String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err("Use only ASCII letters, digits, '-' or '_' for the run name".into());
    }
    Ok(value.into())
}

fn checkpoint_name(value: &str) -> Result<String, String> {
    run_name(value)
}

fn resolve_root(root: Option<PathBuf>, default_directory: &str) -> PathBuf {
    root.unwrap_or_else(|| project_root().join(default_directory))
}

fn split_policy(ratio: Option<f64>, count: Option<usize>) -> ValidationSplit {
    match (ratio, count) {
        (Some(ratio), _) => ValidationSplit::Ratio(ratio),
        (_, Some(count)) => ValidationSplit::Count(count),
        _ => ValidationSplit::None,
    }
}

fn report_evaluation(label: &str, metrics: &EvaluationMetrics) {
    println!(
        "{label}: {} targets, mean cross entropy {:.6}, perplexity {:.6}",
        metrics.target_count, metrics.mean_cross_entropy, metrics.perplexity
    );
}

fn dataset_provenance(
    corpus: &DocumentCorpus,
    split: &DatasetSplit,
    selections: &[String],
    seed: u64,
    policy: ValidationSplit,
    format: Format,
) -> Result<DatasetProvenance, String> {
    let validation: BTreeSet<_> = split.validation.documents.iter().map(|d| &d.id).collect();
    let mut documents = Vec::new();
    for document in &corpus.documents {
        let id = document
            .id
            .components()
            .map(|part| {
                part.as_os_str().to_str().map(str::to_owned).ok_or_else(|| {
                    "Checkpoint provenance requires UTF-8 document paths".to_string()
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("/");
        let bytes: Vec<_> = document
            .tokens
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        documents.push(DocumentFingerprint {
            id,
            partition: if validation.contains(&document.id) {
                "validation"
            } else {
                "training"
            }
            .into(),
            sha256: sha256_bytes(&bytes),
        });
    }
    documents.sort_by(|a, b| a.id.cmp(&b.id));
    let mut selections = selections.to_vec();
    selections.sort();
    selections.dedup();
    Ok(DatasetProvenance {
        selections,
        format: match format {
            Format::Auto => return Err("Unresolved dataset format".into()),
            Format::Text => "text",
            Format::Jsonl => "jsonl",
            Format::Chat => "chat",
        }
        .into(),
        fingerprint_kind: "token-ids-le-u32-v1".into(),
        documents,
        split_seed: seed,
        split_policy: match policy {
            ValidationSplit::None => "none".into(),
            ValidationSplit::Count(count) => format!("count:{count}"),
            ValidationSplit::Ratio(ratio) => format!("ratio:{ratio}"),
        },
    })
}

enum DataSource {
    Eager(TrainingSet),
    Cached(CachedPartition),
    Chat(crate::assistant::ConversationSet),
}

impl ExampleSource for DataSource {
    fn objective(&self) -> &str {
        match self {
            Self::Eager(s) => s.objective(),
            Self::Cached(s) => s.objective(),
            Self::Chat(s) => s.objective(),
        }
    }
    fn target_mask(&self, index: usize) -> Result<Option<Vec<bool>>, String> {
        match self {
            Self::Eager(s) => s.target_mask(index),
            Self::Cached(s) => s.target_mask(index),
            Self::Chat(s) => s.target_mask(index),
        }
    }
    fn example_count(&self) -> usize {
        match self {
            Self::Eager(s) => s.example_count(),
            Self::Cached(s) => s.example_count(),
            Self::Chat(s) => s.example_count(),
        }
    }
    fn target_count(&self) -> Result<usize, String> {
        match self {
            Self::Eager(s) => s.target_count(),
            Self::Cached(s) => s.target_count(),
            Self::Chat(s) => s.target_count(),
        }
    }
    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        match self {
            Self::Eager(s) => s.example(index),
            Self::Cached(s) => s.example(index),
            Self::Chat(s) => s.example(index),
        }
    }
    fn identity(&self) -> Result<String, String> {
        match self {
            Self::Eager(s) => s.identity(),
            Self::Cached(s) => s.identity(),
            Self::Chat(s) => s.identity(),
        }
    }
}

struct PreparedData {
    training: DataSource,
    validation: DataSource,
    provenance: DatasetProvenance,
    training_files: usize,
    training_tokens: usize,
    training_documents: Vec<crate::DocumentInfo>,
}

fn prepare_selected_data(
    root: &Path,
    selections: &[String],
    tokenizer: &Tokens,
    options: &CacheOptions,
    cache: Option<&Path>,
    format: Format,
) -> Result<PreparedData, String> {
    if format != Format::Chat {
        return prepare_data(root, selections, tokenizer, options, cache);
    }
    if cache.is_some() {
        return Err("Conversation caches are unsupported; use bounded eager chat data, never token-only text caches".into());
    }
    let data = crate::assistant::prepare_conversations(
        root,
        selections,
        tokenizer,
        options.context_length,
        options.validation,
        options.split_seed,
    )?;
    Ok(PreparedData {
        training: DataSource::Chat(data.training),
        validation: DataSource::Chat(data.validation),
        provenance: data.provenance,
        training_documents: data.training_documents,
        training_files: data.training_files,
        training_tokens: data.training_tokens,
    })
}

fn prepare_data(
    root: &Path,
    selections: &[String],
    tokenizer: &Tokens,
    options: &CacheOptions,
    cache_path: Option<&Path>,
) -> Result<PreparedData, String> {
    if let Some(path) = cache_path {
        let cache = open_token_cache(path, root, selections, tokenizer, options)?;
        let training = cache.partition(CachePartition::Training);
        let validation = cache.partition(CachePartition::Validation);
        return Ok(PreparedData {
            training_files: training.files().len(),
            training_tokens: training.token_count(),
            training_documents: training.documents().to_vec(),
            provenance: cache.dataset_provenance(),
            training: DataSource::Cached(training),
            validation: DataSource::Cached(validation),
        });
    }
    let corpus = load_document_corpus_with_format(root, selections, tokenizer, options.format)?;
    let split = split_document_corpus(
        &corpus,
        options.context_length,
        options.validation,
        options.split_seed,
    )?;
    let format = match options.format {
        DatasetFormat::Text => Format::Text,
        DatasetFormat::Jsonl => Format::Jsonl,
    };
    let provenance = dataset_provenance(
        &corpus,
        &split,
        selections,
        options.split_seed,
        options.validation,
        format,
    )?;
    Ok(PreparedData {
        training_files: split.training.set.files.len(),
        training_tokens: split.training.set.token_count,
        training_documents: split.training.documents.clone(),
        training: DataSource::Eager(split.training.set),
        validation: DataSource::Eager(split.validation.set),
        provenance,
    })
}

fn session_options(
    flags: SessionFlags,
    data: &PreparedData,
    root: &Path,
) -> Result<SessionOptions, String> {
    let sampling = if flags.dataset_weights.is_empty() {
        if flags.shuffle {
            SamplingPolicy::Shuffle
        } else {
            SamplingPolicy::Fixed
        }
    } else {
        let mut weights = BTreeMap::new();
        for (name, weight) in flags.dataset_weights {
            if weights.insert(name, weight).is_some() {
                return Err("Duplicate dataset weight".into());
            }
        }
        if weights.keys().cloned().collect::<Vec<_>>() != data.provenance.selections {
            return Err(
                "Specify exactly one --dataset-weight for every selected dataset folder".into(),
            );
        }
        let mut groups = Vec::new();
        let mut roots = Vec::new();
        for (name, weight) in weights {
            roots.push(
                std::fs::canonicalize(root.join(&name))
                    .map_err(|e| format!("Cannot resolve weighted dataset {name}: {e}"))?,
            );
            groups.push(SamplingGroup {
                name,
                weight,
                indices: Vec::new(),
            });
        }
        for document in &data.training_documents {
            let matched: Vec<_> = roots
                .iter()
                .enumerate()
                .filter(|(_, root)| document.path.starts_with(root))
                .map(|(i, _)| i)
                .collect();
            if matched.len() != 1 {
                return Err("Weighted datasets must be nonoverlapping and uniquely contain every training document".into());
            }
            groups[matched[0]]
                .indices
                .extend(document.example_range.clone());
        }
        SamplingPolicy::Weighted {
            groups,
            samples_per_epoch: flags
                .samples_per_epoch
                .unwrap_or(data.training.example_count()),
        }
    };
    sampling.validate(data.training.example_count())?;
    Ok(SessionOptions {
        sampling,
        batching: BatchConfig {
            batch_size: flags.batch_size,
            max_batch_tokens: flags.max_batch_tokens,
        },
        optimization: OptimizationOptions {
            gradient_clip_norm: flags.gradient_clip_norm,
            warmup_updates: flags.warmup_updates,
        },
    })
}

struct RunOutput<'a> {
    control: Option<&'a OperationControl>,
    segment: Option<&'a SegmentExecution>,
    parent: Option<&'a ParentCheckpoint>,
    weights_root: &'a Path,
    name: &'a str,
    tokenizer: &'a Tokens,
    metadata: &'a CheckpointMetadata,
    epochs: usize,
    max_updates: Option<usize>,
    metrics_jsonl: Option<PathBuf>,
    quiet: bool,
    resumed: bool,
    saves: SaveOptions,
}

trait Runtime {
    type Training: AutodiffBackend<FloatElem = f32>;
    fn device(&self) -> &<Self::Training as burn::tensor::backend::Backend>::Device;
    fn execution(&self) -> Option<(&str, &str)> {
        None
    }
    fn validate_model(&self, config: &GptConfig, _batch: usize) -> Result<(), String> {
        config.validate()
    }
    fn save(
        &self,
        session: &TrainingSession<DataSource, Self::Training>,
        output: &RunOutput<'_>,
    ) -> Result<PathBuf, String>;
    fn load(
        &self,
        path: &Path,
        source: DataSource,
        config: &GptConfig,
        tokenizer: &Tokens,
    ) -> Result<TrainingSession<DataSource, Self::Training>, String>;
}

struct CpuRuntime(burn::backend::ndarray::NdArrayDevice);
impl Runtime for CpuRuntime {
    type Training = omega_nn::TrainingBackend;
    fn device(&self) -> &burn::backend::ndarray::NdArrayDevice {
        &self.0
    }
    fn save(
        &self,
        session: &TrainingSession<DataSource>,
        output: &RunOutput<'_>,
    ) -> Result<PathBuf, String> {
        save_training_checkpoint_with_parent(
            output.weights_root,
            output.name,
            session,
            output.tokenizer,
            output.metadata,
            output.parent,
        )
    }
    fn load(
        &self,
        path: &Path,
        source: DataSource,
        config: &GptConfig,
        tokenizer: &Tokens,
    ) -> Result<TrainingSession<DataSource>, String> {
        load_training_checkpoint(path, source, config, tokenizer)
    }
}

#[cfg(feature = "gpu")]
impl Runtime for crate::gpu::VulkanDevice {
    type Training = crate::gpu::GpuTraining;
    fn device(&self) -> &burn::backend::wgpu::WgpuDevice {
        self.device()
    }
    fn execution(&self) -> Option<(&str, &str)> {
        Some(("vulkan", &self.adapter().name))
    }
    fn validate_model(&self, config: &GptConfig, batch: usize) -> Result<(), String> {
        self.adapter().validate_model(config, batch)
    }
    fn save(
        &self,
        session: &TrainingSession<DataSource, Self::Training>,
        output: &RunOutput<'_>,
    ) -> Result<PathBuf, String> {
        crate::resume::save_gpu_training_checkpoint_with_parent(
            output.weights_root,
            output.name,
            session,
            output.tokenizer,
            output.metadata,
            self,
            output.parent,
        )
    }
    fn load(
        &self,
        path: &Path,
        source: DataSource,
        config: &GptConfig,
        tokenizer: &Tokens,
    ) -> Result<TrainingSession<DataSource, Self::Training>, String> {
        crate::resume::load_gpu_training_checkpoint(path, source, config, tokenizer, self)
    }
}

#[cfg(feature = "cuda")]
impl Runtime for crate::cuda::CudaDevice {
    type Training = crate::cuda::CudaTraining;
    fn device(&self) -> &burn::backend::cuda::CudaDevice {
        &self.device
    }
    fn execution(&self) -> Option<(&str, &str)> {
        Some(("cuda-experimental", &self.profile.name))
    }
    fn save(
        &self,
        session: &TrainingSession<DataSource, Self::Training>,
        output: &RunOutput<'_>,
    ) -> Result<PathBuf, String> {
        crate::resume::save_cuda_training_checkpoint(
            output.weights_root,
            output.name,
            session,
            output.tokenizer,
            output.metadata,
            self,
            output.parent,
        )
    }
    fn load(
        &self,
        path: &Path,
        source: DataSource,
        config: &GptConfig,
        tokenizer: &Tokens,
    ) -> Result<TrainingSession<DataSource, Self::Training>, String> {
        crate::resume::load_cuda_training_checkpoint(path, source, config, tokenizer, self)
    }
}

fn run_session<R: Runtime>(
    mut session: TrainingSession<DataSource, R::Training>,
    validation: DataSource,
    output: RunOutput<'_>,
    runtime: &R,
) -> Result<(), String> {
    let mut output = output;
    if let Some(root) = output
        .segment
        .and_then(|segment| segment.output_root.as_deref())
    {
        output.weights_root = root;
    }
    let schedule = SaveSchedule::from(output.saves.clone());
    schedule.validate()?;
    let total = output
        .epochs
        .checked_mul(session.updates_per_epoch())
        .ok_or("Training update count overflows usize")?;
    let remaining = total
        .checked_sub(session.progress().completed_updates)
        .filter(|&n| n > 0)
        .ok_or("Target epochs must exceed already completed training updates")?;
    if output.max_updates == Some(0) {
        return Err("max-updates must be greater than zero".into());
    }
    let updates = output
        .max_updates
        .map_or(remaining, |limit| limit.min(remaining));
    let config = session.config().clone();
    let mut metrics = output
        .metrics_jsonl
        .as_ref()
        .map(|path| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map(JsonlMetrics::new)
                .map_err(|error| {
                    format!("Cannot create new metrics file {}: {error}", path.display())
                })
        })
        .transpose()?;
    if let Some(writer) = &mut metrics {
        if let Some((backend, device)) = runtime.execution() {
            writer.execution(backend, device)?;
        }
        if output.resumed {
            writer.resumed(output.epochs, session.progress())?;
        } else {
            writer.started(
                output.epochs,
                session.samples_per_epoch(),
                session.targets_per_epoch(),
            )?;
        }
    }
    let stop = output.control.map(|c| c.stop.clone()).unwrap_or_default();
    if output.control.is_none() {
        let signal_stop = stop.clone();
        ctrlc::set_handler(move || signal_stop.store(true, Ordering::SeqCst))
            .map_err(|error| format!("Cannot install graceful interruption handler: {error}"))?;
    }
    if let Some(control) = output.control {
        control.emit("training_started", serde_json::json!({"total_updates":total,"completed_updates":session.progress().completed_updates,"epochs":output.epochs}))?;
    }
    let timer = std::time::Instant::now();
    let starting_updates = session.progress().completed_updates;
    let starting_targets = session.progress().completed_targets;
    let bounded = advance_bounded(
        &mut session,
        updates,
        &stop,
        schedule,
        output.segment.map(|segment| &segment.budget),
        |current, event| {
            if let Some(control) = output.control {
                control.emit("update", serde_json::json!({"epoch":event.epoch,"completed_updates":event.completed_updates,"completed_targets":event.completed_targets,"loss":event.pre_update_loss,"gradient_norm":event.gradient_norm,"learning_rate":event.effective_learning_rate,"elapsed_seconds":timer.elapsed().as_secs_f64(),"run_updates":event.completed_updates.saturating_sub(starting_updates),"run_targets":event.completed_targets.saturating_sub(starting_targets)}))?;
            }
            if let Some(writer) = &mut metrics {
                writer.update(event)?;
            }
            if !output.quiet {
                println!(
                    "Epoch {} update {} targets {} pre-update loss: {:.6}",
                    event.epoch,
                    event.completed_updates,
                    event.completed_targets,
                    event.pre_update_loss
                );
                if let Some(summary) = event.epoch_summary {
                    println!(
                        "Epoch {} mean pre-update loss: {:.6}",
                        summary.epoch, summary.mean_pre_update_loss
                    );
                }
                stdout()
                    .flush()
                    .map_err(|e| format!("Cannot flush training progress: {e}"))?;
            }
            if event.epoch_summary.is_some() && validation.example_count() > 0 {
                let held_out = evaluate_source_on_device(
                    &current.inference_model(),
                    &config,
                    &validation,
                    runtime.device(),
                )?;
                if let Some(writer) = &mut metrics {
                    writer.validation(Some(event.epoch), event.completed_updates, &held_out)?;
                }
                if !output.quiet {
                    report_evaluation("Validation", &held_out);
                }
            }
            Ok(())
        },
        |current| {
            let checkpoint = runtime.save(current, &output)?;
            if output.segment.is_some() {
                // Do not report a worker checkpoint until its saved state and
                // payload hashes agree with the committed in-memory boundary.
                let saved = crate::resume::read_resume_manifest(&checkpoint)?;
                crate::checkpoint::load_checkpoint_header(&checkpoint)?;
                if saved.progress != current.progress()
                    || saved.parent.as_ref() != output.parent
                    || saved.source_identity != current.training_set().identity()?
                    || saved.objective() != current.training_set().objective()
                {
                    return Err(
                        "Saved segment checkpoint state disagrees with committed training state"
                            .into(),
                    );
                }
                for (name, digest) in [
                    ("model.mpk", &saved.model_sha256),
                    ("optimizer.mpk", &saved.optimizer_sha256),
                ] {
                    let bytes = std::fs::read(checkpoint.join(name))
                        .map_err(|e| format!("Cannot verify saved {name}: {e}"))?;
                    if sha256_bytes(&bytes) != *digest {
                        return Err(format!("Saved segment {name} checksum mismatch"));
                    }
                }
            }
            if let Some(control) = output.control {
                control.emit("checkpoint", serde_json::json!({"path":checkpoint}))?;
            }
            println!("Saved checkpoint: {}", checkpoint.display());
            Ok(checkpoint)
        },
    )?;
    let outcome = bounded.run;
    if let Some(writer) = &mut metrics {
        if outcome.interrupted && session.progress().completed_updates < total {
            writer.interrupted(session.progress())?;
        } else if session.progress().completed_updates == total {
            writer.training_complete(session.progress())?;
        } else {
            writer.segment_complete(session.progress())?;
        }
    }
    if let Some(control) = output.control {
        control.emit("training_finished",serde_json::json!({"completed_updates":session.progress().completed_updates,"total_updates":total,"segment_only":session.progress().completed_updates < total}))?;
    }
    if let Some(segment) = output.segment {
        let completed_updates = session.progress().completed_updates;
        let reason = if completed_updates == total {
            OperationStopReason::Completed
        } else if outcome.interrupted {
            OperationStopReason::UserStop
        } else if bounded.time_budget_reached {
            OperationStopReason::TimeBudget
        } else {
            OperationStopReason::SegmentLimit
        };
        let elapsed_seconds = segment.budget.started.elapsed().as_secs_f64();
        *segment.outcome.borrow_mut() = Some(OperationOutcome {
            reason,
            checkpoint: outcome.checkpoint,
            completed_updates,
            total_updates: total,
            elapsed_seconds,
            budget_overrun_seconds: (elapsed_seconds - segment.budget.max_seconds).max(0.0),
        });
        return Ok(());
    }
    if outcome.interrupted && session.progress().completed_updates < total {
        return Err(format!(
            "Training interrupted after {} updates; resume from {}",
            session.progress().completed_updates,
            outcome.checkpoint.display()
        ));
    }
    Ok(())
}

fn saved_policy(value: &str) -> Result<ValidationSplit, String> {
    if value == "none" {
        return Ok(ValidationSplit::None);
    }
    if let Some(count) = value.strip_prefix("count:") {
        return count
            .parse()
            .map(ValidationSplit::Count)
            .map_err(|_| "Invalid saved split count".into());
    }
    if let Some(ratio) = value.strip_prefix("ratio:") {
        return ratio
            .parse()
            .map(ValidationSplit::Ratio)
            .map_err(|_| "Invalid saved split ratio".into());
    }
    Err("Unsupported saved split policy".into())
}

fn run_checkpoint_command(command: CheckpointCommand) -> Result<(), String> {
    let (args, mode) = match command {
        CheckpointCommand::List { catalog } => (catalog, None),
        CheckpointCommand::Latest { catalog, mode } => (catalog, Some(CatalogMode::from(mode))),
    };
    let root = resolve_root(args.weights_root, "weights");
    let catalog = discover_checkpoints(
        &root,
        &args.run,
        &CatalogLimits {
            max_entries: args.max_entries,
            max_metadata_bytes: args.max_metadata_bytes,
        },
    )?;
    let rendered = if let Some(mode) = mode {
        let selected = catalog.latest(mode)?;
        if args.json {
            serde_json::to_string_pretty(&selected).map_err(|error| error.to_string())?
        } else {
            let mut lines = vec![
                format!(
                    "Selected: {} ({:?})",
                    selected.path.display(),
                    selected.status
                ),
                format!("Inspection: {}", selected.inspection),
            ];
            lines.extend(
                selected
                    .skipped
                    .iter()
                    .map(|entry| format!("Skipped {}: {}", entry.name, entry.reason)),
            );
            lines.join("\n")
        }
    } else if args.json {
        serde_json::to_string_pretty(&catalog).map_err(|error| error.to_string())?
    } else {
        let mut lines = vec![
            format!(
                "Checkpoints for {} in {}",
                catalog.run_name,
                catalog.root.display()
            ),
            format!("Inspection: {}", catalog.inspection),
        ];
        lines.extend(
            catalog
                .entries
                .iter()
                .map(|entry| format!("{} {:?}: {}", entry.name, entry.status, entry.reason)),
        );
        if catalog.entries.is_empty() {
            lines.push("No matching checkpoints".into());
        }
        lines.join("\n")
    };
    writeln!(stdout().lock(), "{rendered}")
        .map_err(|error| format!("Cannot write checkpoint inventory: {error}"))
}

pub fn execute(args: Args, control: Option<&OperationControl>) -> Result<(), String> {
    execute_internal(args, control, None)
}

/// Execute one bounded assistant-stage or exact-resume segment. TrainStage
/// explicitly transfers supported Omega weights across builds; Resume retains
/// all exact runtime/backend/source checks. The caller's stop flag is user-only.
/// CPU pools must already be configured by the caller, as with `execute`.
pub fn execute_segment(
    args: Args,
    control: &OperationControl,
    max_seconds: f64,
) -> Result<OperationOutcome, String> {
    execute_segment_inner(args, control, max_seconds, None)
}

/// As `execute_segment`, with a separate destination for newly saved checkpoints.
/// Parent/resume selection still uses `Args`' weights root. Relative destinations
/// resolve from the caller's working directory; existing saves are never replaced.
pub fn execute_segment_to(
    args: Args,
    control: &OperationControl,
    max_seconds: f64,
    output_root: &Path,
) -> Result<OperationOutcome, String> {
    if output_root.as_os_str().is_empty() {
        return Err("Segment output directory must not be empty".into());
    }
    execute_segment_inner(args, control, max_seconds, Some(output_root.to_owned()))
}

fn execute_segment_inner(
    args: Args,
    control: &OperationControl,
    max_seconds: f64,
    output_root: Option<PathBuf>,
) -> Result<OperationOutcome, String> {
    if !matches!(
        args.command,
        Command::TrainStage { .. } | Command::Resume { .. }
    ) {
        return Err("Segment execution requires train-stage or resume".into());
    }
    let segment = SegmentExecution {
        budget: SegmentBudget::new(max_seconds)?,
        output_root,
        outcome: Default::default(),
    };
    execute_internal(args, Some(control), Some(&segment))?;
    segment
        .outcome
        .into_inner()
        .ok_or_else(|| "Segment completed without a verified training outcome".into())
}

fn check_evaluation_control(
    control: Option<&OperationControl>,
    budget: Option<&SegmentBudget>,
) -> Result<(), String> {
    if control.is_some_and(|c| c.stop.load(Ordering::SeqCst)) {
        return Err("Evaluation stopped by user; no completed evaluation result".into());
    }
    if let Some(budget) = budget
        && budget.started.elapsed().as_secs_f64() >= budget.max_seconds
    {
        return Err(format!(
            "Evaluation time budget exhausted after {:.3}s (budget {:.3}s); no completed evaluation result",
            budget.started.elapsed().as_secs_f64(),
            budget.max_seconds
        ));
    }
    Ok(())
}

/// Evaluate fixed weights with cooperative checks before/after every example.
/// Loading, serialization and observer dispatch count toward the budget. No
/// partial aggregate is emitted; a dispatch that itself overruns is an error,
/// so callers must accept results only when this operation returns success.
pub fn execute_evaluation(
    args: Args,
    control: &OperationControl,
    max_seconds: f64,
) -> Result<(), String> {
    if !matches!(args.command, Command::Evaluate { .. }) {
        return Err("Bounded evaluation requires evaluate".into());
    }
    let execution = SegmentExecution {
        budget: SegmentBudget::new(max_seconds)?,
        output_root: None,
        outcome: Default::default(),
    };
    check_evaluation_control(Some(control), Some(&execution.budget))?;
    execute_internal(args, Some(control), Some(&execution))?;
    check_evaluation_control(Some(control), Some(&execution.budget))
}

fn execute_internal(
    args: Args,
    control: Option<&OperationControl>,
    segment: Option<&SegmentExecution>,
) -> Result<(), String> {
    if matches!(args.command, Command::Devices) && args.backend == BackendChoice::Cuda {
        if args.device.is_some()
            || args.cpu.cpu_threads.is_some()
            || args.cpu.matmul_threads.is_some()
        {
            return Err("devices does not accept device or CPU thread overrides".into());
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&crate::cuda::list_devices()?)
                .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if matches!(args.command, Command::Devices) {
        if args.device.is_some()
            || args.cpu.cpu_threads.is_some()
            || args.cpu.matmul_threads.is_some()
        {
            return Err("devices does not accept device or CPU thread overrides".into());
        }
        #[cfg(feature = "gpu")]
        {
            let adapters = crate::gpu::list_vulkan_devices();
            println!(
                "{}",
                serde_json::to_string_pretty(&adapters).map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        #[cfg(not(feature = "gpu"))]
        return Err(
            "Vulkan support is not enabled; rebuild omega-training with --features gpu".into(),
        );
    }
    match args.backend {
        BackendChoice::Cpu => {
            if args.device.is_some() {
                return Err("--device requires --backend vulkan or cuda".into());
            }
            run_on_backend(args, &CpuRuntime(Default::default()), control, segment)
        }
        BackendChoice::Cuda => {
            if !matches!(
                args.command,
                Command::Train { .. }
                    | Command::TrainStage { .. }
                    | Command::Resume { .. }
                    | Command::Generate { .. }
                    | Command::Chat { .. }
                    | Command::Evaluate { .. }
            ) {
                return Err(
                    "--backend cuda applies to training, evaluation and inference only".into(),
                );
            }
            if args.cpu.cpu_threads.is_some() || args.cpu.matmul_threads.is_some() {
                return Err("CPU thread overrides cannot be combined with --backend cuda".into());
            }
            #[cfg(feature = "cuda")]
            {
                let runtime = crate::cuda::initialize(args.device.unwrap_or(0))?;
                run_on_backend(args, &runtime, control, segment)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA requires --features cuda; it is experimental".into())
            }
        }
        BackendChoice::Vulkan => {
            if !matches!(
                args.command,
                Command::Train { .. }
                    | Command::TrainStage { .. }
                    | Command::Resume { .. }
                    | Command::Generate { .. }
                    | Command::Chat { .. }
                    | Command::Evaluate { .. }
            ) {
                return Err(
                    "--backend vulkan applies to train, train-stage, resume, generate, chat and evaluate".into(),
                );
            }
            if args.cpu.cpu_threads.is_some() || args.cpu.matmul_threads.is_some() {
                return Err("CPU thread overrides cannot be combined with --backend vulkan".into());
            }
            #[cfg(feature = "gpu")]
            {
                let runtime = crate::gpu::initialize_vulkan(args.device.unwrap_or(0))?;
                eprintln!(
                    "Backend: Vulkan/SPIR-V; discrete GPU {}: {} ({})",
                    runtime.adapter().index,
                    runtime.adapter().name,
                    runtime.adapter().driver_info
                );
                run_on_backend(args, &runtime, control, segment)
            }
            #[cfg(not(feature = "gpu"))]
            Err("Vulkan support is not enabled; rebuild omega-training with --features gpu".into())
        }
    }
}

fn run_on_backend<R: Runtime>(
    args: Args,
    runtime: &R,
    control: Option<&OperationControl>,
    segment: Option<&SegmentExecution>,
) -> Result<(), String> {
    match args.command {
        Command::Devices => unreachable!("device discovery is handled before model dispatch"),
        Command::TrainStage {
            datasets_root,
            weights_root,
            input,
            name,
            datasets,
            epochs,
            learning_rate,
            seed,
            max_updates,
            metrics_jsonl,
            quiet,
            saves,
            session_options: flags,
        } => {
            let weights_root = resolve_root(weights_root, "weights");
            let root = resolve_root(datasets_root, "datasets");
            resolve_dataset_format(&root, &datasets, Format::Chat, true, ValidationSplit::None)?;
            let path = input.resolve(&weights_root, CatalogMode::Resume)?;
            let config = crate::checkpoint::read_checkpoint_config(&path)?;
            runtime.validate_model(&config, flags.batch_size)?;
            let tokenizer = Tokens::new(path.join("tokenizer.json"))?;
            let data = prepare_selected_data(
                &root,
                &datasets,
                &tokenizer,
                &CacheOptions {
                    format: DatasetFormat::Jsonl,
                    context_length: config.context_length,
                    validation: ValidationSplit::None,
                    split_seed: 42,
                    limits: CacheLimits::default(),
                },
                None,
                Format::Chat,
            )?;
            let options = session_options(*flags, &data, &root)?;
            let metadata = CheckpointMetadata {
                dataset: Some(data.provenance),
                training: Some(TrainingProvenance {
                    epochs,
                    learning_rate,
                    seed,
                    optimizer: "adam-default-burn-0.18".into(),
                }),
                build: Some(BuildIdentity::current()),
            };
            if epochs == 0 {
                return Err("epochs must be greater than zero".into());
            }
            let initialize = if segment.is_some() {
                crate::resume::initialize_assistant_stage_transfer_on_device::<_, R::Training>
            } else {
                crate::resume::initialize_assistant_stage_on_device::<_, R::Training>
            };
            let (session, tokenizer, parent) = initialize(
                &path,
                data.training,
                learning_rate,
                seed,
                options,
                runtime.device(),
            )?;
            run_session(
                session,
                data.validation,
                RunOutput {
                    control,
                    segment,
                    parent: Some(&parent),
                    weights_root: &weights_root,
                    name: &name,
                    tokenizer: &tokenizer,
                    metadata: &metadata,
                    epochs,
                    max_updates,
                    metrics_jsonl,
                    quiet,
                    resumed: false,
                    saves,
                },
                runtime,
            )?;
        }
        Command::Chat {
            history: supplied_history,
            weights_root,
            input,
            prompt,
            system,
            max_new_tokens,
            generation,
        } => {
            use omega_tokenizer::chat::{ChatMessage, ChatRole};
            let path = input.resolve(
                &resolve_root(weights_root, "weights"),
                CatalogMode::Inference,
            )?;
            let artifact = read_checkpoint_manifest(&path)?
                .and_then(|m| m.chat)
                .ok_or("Checkpoint has no explicit chat protocol; use a chat-trained model")?;
            runtime.validate_model(&crate::checkpoint::read_checkpoint_config(&path)?, 1)?;
            let (model, config, tokenizer) = load_checkpoint_on_device::<
                <R::Training as AutodiffBackend>::InnerBackend,
            >(&path, runtime.device())?;
            let protocol = artifact.validate_tokenizer(&tokenizer)?;
            let options = generation.options()?;
            if max_new_tokens == 0 {
                return Err("Chat max-new-tokens must be positive".into());
            }
            let initial: Vec<_> = system
                .into_iter()
                .map(|content| ChatMessage {
                    role: ChatRole::System,
                    content,
                })
                .collect();
            let mut history = if supplied_history.is_empty() {
                initial.clone()
            } else {
                supplied_history
            };
            let one_shot = prompt.is_some();
            let mut supplied = prompt;
            loop {
                let content = if let Some(prompt) = supplied.take() {
                    prompt
                } else {
                    print!("You: ");
                    stdout().flush().map_err(|e| e.to_string())?;
                    let mut line = String::new();
                    if std::io::stdin()
                        .read_line(&mut line)
                        .map_err(|e| e.to_string())?
                        == 0
                    {
                        break;
                    }
                    line.trim_end_matches(['\r', '\n']).to_owned()
                };
                if !one_shot && content == "/exit" {
                    break;
                }
                if !one_shot && content == "/reset" {
                    history = initial.clone();
                    continue;
                }
                if content.trim().is_empty() {
                    if one_shot {
                        return Err("Chat prompt must not be blank".into());
                    }
                    continue;
                }
                let mut candidate = history.clone();
                candidate.push(ChatMessage {
                    role: ChatRole::User,
                    content,
                });
                let ids = protocol.encode_prompt(&tokenizer, &candidate)?;
                if ids
                    .len()
                    .checked_add(max_new_tokens)
                    .is_none_or(|n| n > config.context_length)
                {
                    let message = "Conversation plus reply budget exceeds the trained context; shorten the prompt/budget or use /reset. History was not changed.";
                    if one_shot {
                        return Err(message.into());
                    }
                    eprintln!("{message}");
                    continue;
                }
                let output = generate_with_options_on_device(
                    &model,
                    &config,
                    &ids,
                    max_new_tokens,
                    Some(protocol.token_ids().end_turn),
                    &options,
                    runtime.device(),
                )?;
                let reply_ids = &output[ids.len()..];
                let controls = protocol.token_ids();
                if reply_ids
                    .iter()
                    .any(|id| [controls.system, controls.user, controls.assistant].contains(id))
                {
                    return Err("Model generated an unexpected role token; reply rejected instead of leaking control delimiters".into());
                }
                let reply = tokenizer.decode(reply_ids, true)?;
                if one_shot {
                    println!("{reply}");
                } else {
                    println!("Assistant: {reply}");
                }
                if reply_ids.last() != Some(&controls.end_turn) {
                    eprintln!("Reply reached token budget before end-turn.");
                }
                candidate.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: reply.clone(),
                });
                if let Some(control) = control {
                    control.emit("chat",serde_json::json!({"text":reply,"history":candidate.iter().map(|m|serde_json::json!({"role":match m.role {ChatRole::System=>"system",ChatRole::User=>"user",ChatRole::Assistant=>"assistant"},"content":m.content})).collect::<Vec<_>>()}))?;
                }
                history = candidate;
                if one_shot {
                    break;
                }
            }
        }
        Command::Train {
            name,
            datasets_root,
            weights_root,
            datasets,
            dataset_format,
            tokenizer,
            epochs,
            learning_rate,
            context_length,
            d_model,
            heads,
            layers,
            d_ff,
            seed,
            validation_ratio,
            validation_count,
            split_seed,
            metrics_jsonl,
            quiet,
            cache,
            max_updates,
            saves,
            limits,
            session_options: flags,
        } => {
            let datasets_root = resolve_root(datasets_root, "datasets");
            let dataset_format = resolve_dataset_format(
                &datasets_root,
                &datasets,
                dataset_format,
                true,
                split_policy(validation_ratio, validation_count),
            )?;
            let weights_root = resolve_root(weights_root, "weights");
            let tokenizer = Tokens::new(datasets_root.join(tokenizer))?;
            let config = GptConfig {
                vocab_size: validate_tokenizer(&tokenizer)?,
                context_length,
                d_model,
                num_heads: heads,
                num_layers: layers,
                d_ff,
            };
            runtime.validate_model(&config, flags.batch_size)?;
            if epochs == 0 {
                return Err("epochs must be greater than zero".into());
            }
            let options = CacheOptions {
                format: dataset_format.into(),
                context_length,
                validation: split_policy(validation_ratio, validation_count),
                split_seed,
                limits: limits.into(),
            };
            let data = prepare_selected_data(
                &datasets_root,
                &datasets,
                &tokenizer,
                &options,
                cache.as_deref(),
                dataset_format,
            )?;
            let session_options = session_options(*flags, &data, &datasets_root)?;
            let metadata = CheckpointMetadata {
                dataset: Some(data.provenance),
                training: Some(TrainingProvenance {
                    epochs,
                    learning_rate,
                    seed,
                    optimizer: "adam-default-burn-0.18".into(),
                }),
                build: Some(BuildIdentity::current()),
            };
            if !quiet {
                println!(
                    "Training on {} files, {} source tokens, {} examples for {epochs} epochs",
                    data.training_files,
                    data.training_tokens,
                    data.training.example_count()
                );
            }
            let session = TrainingSession::<_, R::Training>::from_source_with_options_on_device(
                &config,
                data.training,
                learning_rate,
                seed,
                session_options,
                runtime.device(),
            )?;
            run_session(
                session,
                data.validation,
                RunOutput {
                    control,
                    segment,
                    parent: None,
                    weights_root: &weights_root,
                    name: &name,
                    tokenizer: &tokenizer,
                    metadata: &metadata,
                    epochs,
                    max_updates,
                    metrics_jsonl,
                    quiet,
                    resumed: false,
                    saves,
                },
                runtime,
            )?;
        }
        Command::Resume {
            datasets_root,
            weights_root,
            input,
            name,
            epochs,
            max_updates,
            saves,
            cache,
            limits,
            metrics_jsonl,
            quiet,
        } => {
            let weights_root = resolve_root(weights_root, "weights");
            let path = input.resolve(&weights_root, CatalogMode::Resume)?;
            let manifest = read_checkpoint_manifest(&path)?
                .ok_or("Legacy inference checkpoint has no resumable training state")?;
            let parent = crate::resume::read_resume_manifest(&path)?.parent;
            let mut metadata = manifest.metadata;
            let recipe = metadata
                .dataset
                .as_ref()
                .ok_or("Resume CLI requires saved dataset provenance")?;
            let config = GptConfig::from(manifest.model);
            let tokenizer = Tokens::new(path.join("tokenizer.json"))?;
            let format = match recipe.format.as_str() {
                "text" => Format::Text,
                "jsonl" => Format::Jsonl,
                "chat" => Format::Chat,
                _ => return Err("Unsupported saved dataset format".into()),
            };
            let options = CacheOptions {
                format: format.into(),
                context_length: config.context_length,
                validation: saved_policy(&recipe.split_policy)?,
                split_seed: recipe.split_seed,
                limits: limits.into(),
            };
            let root = resolve_root(datasets_root, "datasets");
            resolve_dataset_format(&root, &recipe.selections, format, true, options.validation)?;
            let data = prepare_selected_data(
                &root,
                &recipe.selections,
                &tokenizer,
                &options,
                cache.as_deref(),
                format,
            )?;
            if &data.provenance != recipe {
                return Err("Resume dataset provenance mismatch: source documents or partition recipe changed".into());
            }
            let session = runtime.load(&path, data.training, &config, &tokenizer)?;
            let training = metadata
                .training
                .as_mut()
                .ok_or("Resume CLI requires saved training provenance")?;
            training.epochs = epochs;
            metadata.build = Some(BuildIdentity::current());
            run_session(
                session,
                data.validation,
                RunOutput {
                    control,
                    segment,
                    parent: parent.as_ref(),
                    weights_root: &weights_root,
                    name: &name,
                    tokenizer: &tokenizer,
                    metadata: &metadata,
                    epochs,
                    max_updates,
                    metrics_jsonl,
                    quiet,
                    resumed: true,
                    saves,
                },
                runtime,
            )?;
        }
        Command::PrepareCache {
            datasets_root,
            datasets,
            tokenizer,
            dataset_format,
            context_length,
            validation_ratio,
            validation_count,
            split_seed,
            output,
            limits,
        } => {
            let root = resolve_root(datasets_root, "datasets");
            let dataset_format = resolve_dataset_format(
                &root,
                &datasets,
                dataset_format,
                false,
                split_policy(validation_ratio, validation_count),
            )?;
            if dataset_format == Format::Chat {
                return Err(
                    "Conversation caches are unsupported; use bounded eager chat data".into(),
                );
            }
            let tokenizer = Tokens::new(root.join(tokenizer))?;
            let options = CacheOptions {
                format: dataset_format.into(),
                context_length,
                validation: split_policy(validation_ratio, validation_count),
                split_seed,
                limits: limits.into(),
            };
            let cache = create_token_cache(&output, &root, &datasets, &tokenizer, &options)?;
            println!(
                "Saved token cache: {} ({} training examples, {} validation examples)",
                output.display(),
                cache.partition(CachePartition::Training).example_count(),
                cache.partition(CachePartition::Validation).example_count()
            );
        }
        Command::Generate {
            weights_root,
            input,
            prompt,
            max_new_tokens,
            eos_token_id,
            generation,
        } => {
            let options = generation.options()?;
            let weights_root = resolve_root(weights_root, "weights");
            let path = input.resolve(&weights_root, CatalogMode::Inference)?;
            runtime.validate_model(&crate::checkpoint::read_checkpoint_config(&path)?, 1)?;
            let (model, config, tokenizer) = load_checkpoint_on_device::<
                <R::Training as AutodiffBackend>::InnerBackend,
            >(&path, runtime.device())?;
            let prompt = encode_text(&tokenizer, &prompt)?;
            let ids = generate_with_options_on_device(
                &model,
                &config,
                &prompt,
                max_new_tokens,
                eos_token_id,
                &options,
                runtime.device(),
            )?;
            println!("Token IDs: {ids:?}");
            let decoded = tokenizer.decode(&ids, false)?;
            if let Some(control) = control {
                control.emit("generation", serde_json::json!({"text":decoded,"ids":ids}))?;
            }
            println!("Decoded: {decoded}");
        }
        Command::Evaluate {
            datasets_root,
            weights_root,
            input,
            datasets,
            dataset_format,
            validation_ratio,
            validation_count,
            split_seed,
        } => {
            check_evaluation_control(control, segment.map(|s| &s.budget))?;
            let path = input.resolve(
                &resolve_root(weights_root, "weights"),
                CatalogMode::Inference,
            )?;
            runtime.validate_model(&crate::checkpoint::read_checkpoint_config(&path)?, 1)?;
            let (model, config, tokenizer) = load_checkpoint_on_device::<
                <R::Training as AutodiffBackend>::InnerBackend,
            >(&path, runtime.device())?;
            let root = resolve_root(datasets_root, "datasets");
            let dataset_format = resolve_dataset_format(
                &root,
                &datasets,
                dataset_format,
                false,
                split_policy(validation_ratio, validation_count),
            )?;
            let data = prepare_selected_data(
                &root,
                &datasets,
                &tokenizer,
                &CacheOptions {
                    format: dataset_format.into(),
                    context_length: config.context_length,
                    validation: split_policy(validation_ratio, validation_count),
                    split_seed,
                    limits: CacheLimits::default(),
                },
                None,
                dataset_format,
            )?;
            let set = if validation_ratio.is_some() || validation_count.is_some() {
                &data.validation
            } else {
                &data.training
            };
            let metrics = evaluate_source_controlled_on_device(
                &model,
                &config,
                set,
                runtime.device(),
                || check_evaluation_control(control, segment.map(|s| &s.budget)),
            )?;
            if let Some(control) = control {
                let data = serde_json::json!({"targets":metrics.target_count,"cross_entropy":metrics.mean_cross_entropy,"perplexity":metrics.perplexity});
                check_evaluation_control(Some(control), segment.map(|s| &s.budget))?;
                control.emit("evaluation", data)?;
            }
            check_evaluation_control(control, segment.map(|s| &s.budget))?;
            report_evaluation("Evaluation", &metrics);
            check_evaluation_control(control, segment.map(|s| &s.budget))?;
        }
        Command::Checkpoints { command } => run_checkpoint_command(command)?,
        Command::TrainTokenizer {
            datasets_root,
            datasets,
            dataset_format,
            output,
            vocab_size,
            min_frequency,
            chat_protocol,
        } => {
            let root = resolve_root(datasets_root, "datasets");
            let dataset_format = resolve_dataset_format(
                &root,
                &datasets,
                dataset_format,
                true,
                ValidationSplit::None,
            )?;
            if dataset_format == Format::Chat {
                return Err("Train the chat tokenizer on the prepared base/train text partition; conversation JSONL is not plain text".into());
            }
            let config = ByteBpeConfig {
                vocab_size,
                min_frequency,
            };
            config.validate()?;
            let documents = load_text_documents(&root, &datasets, dataset_format.into())?;
            let texts: Vec<_> = documents
                .iter()
                .map(|document| document.text.as_str())
                .collect();
            let tokenizer = if chat_protocol {
                omega_tokenizer::train_chat_byte_bpe(&texts, &config)?
            } else {
                train_byte_bpe(&texts, &config)?
            };
            tokenizer.save_new(&output)?;
            println!(
                "Saved byte BPE tokenizer: {} ({} entries, {} documents)",
                output.display(),
                tokenizer.vocab_size(),
                documents.len()
            );
        }
        Command::Coverage {
            datasets_root,
            tokenizer,
            text,
            unknown_token_id,
        } => {
            let tokenizer = Tokens::new(resolve_root(datasets_root, "datasets").join(tokenizer))?;
            let report = tokenizer.coverage(&text, unknown_token_id)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": 1, "token_count": report.token_count,
                    "unknown_count": report.unknown_count, "unknown_rate": report.unknown_rate
                })
            );
        }
    }
    Ok(())
}

pub fn main_entry() -> ExitCode {
    main_with_args(std::env::args_os())
}

pub fn main_with_args(arguments: impl IntoIterator<Item = OsString>) -> ExitCode {
    match run_configured(Args::parse_from(arguments)) {
        Ok(status) => status,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

// Pool configuration belongs at a fresh-process boundary: Rust libraries and
// the CLI test harness may already have threads, so never call set_var here.
fn run_configured(args: Args) -> Result<ExitCode, String> {
    const CHILD: &str = "OMEGA_CPU_REEXEC_V1";
    args.cpu.validate()?;
    let requested = args.cpu.cpu_threads.is_some() || args.cpu.matmul_threads.is_some();
    let expected = format!("{:?}:{:?}", args.cpu.cpu_threads, args.cpu.matmul_threads);
    if let Some(marker) = std::env::var_os(CHILD) {
        if !requested || marker != OsString::from(&expected) {
            return Err("Invalid or stale CPU startup marker".into());
        }
        for (key, count) in [
            ("RAYON_NUM_THREADS", args.cpu.cpu_threads),
            ("RAYON_RS_NUM_CPUS", args.cpu.cpu_threads),
            ("MATMUL_NUM_THREADS", args.cpu.matmul_threads),
        ] {
            if let Some(count) = count
                && std::env::var(key).ok().as_deref() != Some(count.to_string().as_str())
            {
                return Err(format!("CPU startup environment does not match {key}"));
            }
        }
    } else if requested {
        let executable = std::env::current_exe()
            .map_err(|e| format!("Cannot locate CPU startup executable: {e}"))?;
        let mut child = std::process::Command::new(executable);
        // Preserve actual invocation, including the same-source test harness.
        child.args(std::env::args_os().skip(1)).env(CHILD, expected);
        args.cpu.apply_to_command(&mut child)?;
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            return Err(format!(
                "Cannot start configured CPU process: {}",
                child.exec()
            ));
        }
        #[cfg(not(unix))]
        {
            // The child shares the console and installs the real training stop
            // handler. Keep this waiting parent alive while the child saves.
            ctrlc::set_handler(|| {})
                .map_err(|e| format!("Cannot install CPU supervisor signal handler: {e}"))?;
            let status = child
                .status()
                .map_err(|e| format!("Cannot start configured CPU process: {e}"))?;
            return Ok(status
                .code()
                .and_then(|code| u8::try_from(code).ok())
                .map(ExitCode::from)
                .unwrap_or(ExitCode::FAILURE));
        }
    }
    crate::cpu::execution_profile()?;
    execute(args, None)?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_default_to_checkout_and_explicit_paths_are_preserved() {
        for directory in ["datasets", "weights"] {
            assert_eq!(
                resolve_root(None, directory),
                project_root().join(directory)
            );
            let relative = PathBuf::from("custom-root");
            assert_eq!(resolve_root(Some(relative.clone()), directory), relative);
            let absolute = project_root().join("custom-root");
            assert_eq!(resolve_root(Some(absolute.clone()), directory), absolute);
        }
    }

    #[test]
    fn parses_multiple_datasets_and_run_name() {
        let args = Args::try_parse_from([
            "main",
            "train",
            "--name",
            "foo",
            "--dataset",
            "one",
            "--dataset",
            "two",
        ])
        .unwrap();
        match args.command {
            Command::Train { name, datasets, .. } => {
                assert_eq!(name, "foo");
                assert_eq!(datasets, ["one", "two"]);
            }
            _ => panic!("expected train"),
        }
    }

    #[test]
    fn rejects_missing_and_unsafe_arguments() {
        assert!(Args::try_parse_from(["main", "train", "--name", "foo"]).is_err());
        assert!(Args::try_parse_from(["main", "train", "--dataset", "one"]).is_err());
        assert!(
            Args::try_parse_from(["main", "train", "--name", "../foo", "--dataset", "one"])
                .is_err()
        );
        assert!(
            Args::try_parse_from([
                "main",
                "generate",
                "--checkpoint",
                "../foo",
                "--prompt",
                "hello"
            ])
            .is_err()
        );
    }
}
