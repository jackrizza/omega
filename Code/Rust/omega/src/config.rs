use anyhow::{Context, Result, ensure};
use omega_benchmark::{Backend, ModelDimensions};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectConfig {
    pub omega_schema_version: u32,
    pub name: String,
    pub paths: Paths,
    pub dataset: DatasetSettings,
    pub tokenizer: TokenizerSettings,
    pub model: ModelDimensions,
    pub training: TrainingSettings,
    pub pipeline: PipelineSettings,
    pub assistant: Option<AssistantSettings>,
    pub benchmark: BenchmarkSettings,
    pub evaluation: EvaluationSettings,
    pub inference: InferenceSettings,
}
impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            omega_schema_version: 1,
            name: "my-model".into(),
            paths: Paths::default(),
            dataset: DatasetSettings::default(),
            tokenizer: TokenizerSettings::default(),
            model: ModelDimensions::default(),
            training: TrainingSettings::default(),
            pipeline: PipelineSettings::default(),
            assistant: None,
            benchmark: BenchmarkSettings::default(),
            evaluation: EvaluationSettings::default(),
            inference: InferenceSettings::default(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Paths {
    pub datasets: PathBuf,
    pub weights: PathBuf,
}
impl Default for Paths {
    fn default() -> Self {
        Self {
            datasets: "datasets".into(),
            weights: "weights".into(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatasetSettings {
    pub selections: Vec<String>,
    pub format: String,
    pub recipe: Option<omega_datasets::config::Config>,
    pub recipe_folder: String,
    pub validation_ratio: Option<f64>,
    pub validation_count: Option<usize>,
    pub split_seed: u64,
    pub cache: Option<PathBuf>,
}
impl Default for DatasetSettings {
    fn default() -> Self {
        Self {
            selections: vec![],
            format: "auto".into(),
            recipe: None,
            recipe_folder: "corpus".into(),
            validation_ratio: None,
            validation_count: None,
            split_seed: 42,
            cache: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokenizerSettings {
    pub path: PathBuf,
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub chat_protocol: bool,
}
impl Default for TokenizerSettings {
    fn default() -> Self {
        Self {
            path: "tokenizers/tokenizer.json".into(),
            vocab_size: 8192,
            min_frequency: 2,
            chat_protocol: true,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingSettings {
    pub backend: Backend,
    pub device: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub batch_size: usize,
    pub max_batch_tokens: usize,
    pub gradient_clip_norm: Option<f64>,
    pub warmup_updates: usize,
    pub shuffle: bool,
    pub cpu_threads: Option<usize>,
    pub matmul_threads: Option<usize>,
    pub save_every_updates: Option<usize>,
    pub save_every_epochs: Option<usize>,
    pub max_updates: Option<usize>,
}
impl Default for TrainingSettings {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            device: 0,
            epochs: 10,
            learning_rate: 0.003,
            seed: 42,
            batch_size: 1,
            max_batch_tokens: 65536,
            gradient_clip_norm: None,
            warmup_updates: 0,
            shuffle: false,
            cpu_threads: None,
            matmul_threads: None,
            save_every_updates: Some(100),
            save_every_epochs: Some(1),
            max_updates: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineSettings {
    pub prepare_dataset: bool,
    pub prepare_tokenizer: bool,
    pub prepare_cache: bool,
    pub benchmark: bool,
    pub train: bool,
    pub evaluate: bool,
}
impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            prepare_dataset: false,
            prepare_tokenizer: false,
            prepare_cache: false,
            benchmark: false,
            train: true,
            evaluate: false,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssistantSettings {
    pub selections: Vec<String>,
    pub epochs: usize,
    pub learning_rate: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BenchmarkSettings {
    pub samples: usize,
    pub warmup: usize,
    pub max_seconds: f64,
}
impl Default for BenchmarkSettings {
    fn default() -> Self {
        Self {
            samples: 12,
            warmup: 2,
            max_seconds: 60.0,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationSettings {
    pub selections: Vec<String>,
    pub format: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceSettings {
    pub max_new_tokens: usize,
    pub system: Option<String>,
    pub history: Vec<ChatTurn>,
}
impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            max_new_tokens: 8,
            system: None,
            history: vec![],
        }
    }
}

pub fn absolute(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.into()
    } else {
        root.join(path)
    }
}
pub fn format(value: &str) -> Result<omega_training::selection::Format> {
    use omega_training::selection::Format;
    Ok(match value {
        "auto" => Format::Auto,
        "text" => Format::Text,
        "jsonl" => Format::Jsonl,
        "chat" => Format::Chat,
        _ => anyhow::bail!("Format must be auto, text, jsonl or chat"),
    })
}
impl ProjectConfig {
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(
            text.len() <= 1024 * 1024,
            "Project configuration exceeds 1 MiB"
        );
        let document: toml::Value = toml::from_str(text).context("Invalid TOML")?;
        ensure!(
            document.get("omega_schema_version").is_some(),
            "This is a dataset recipe, not an Omega project. Use Import recipe to create a new project."
        );
        let config: Self = document.try_into()?;
        config.validate()?;
        Ok(config)
    }
    pub fn load(path: &Path) -> Result<Self> {
        Self::parse(&fs::read_to_string(path).with_context(|| format!("Read {}", path.display()))?)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=4096).contains(&self.inference.max_new_tokens),
            "Inference token budget must be 1..4096"
        );
        ensure!(
            self.inference.history.len() <= 4096
                && self
                    .inference
                    .history
                    .iter()
                    .all(|m| matches!(m.role.as_str(), "system" | "user" | "assistant")),
            "Invalid conversation history"
        );
        ensure!(
            self.omega_schema_version == 1,
            "Unsupported Omega project schema"
        );
        ensure!(
            !self.name.is_empty()
                && self.name.len() <= 80
                && self
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "Model name must be 1..80 ASCII letters, digits, '-' or '_'"
        );
        format(&self.dataset.format)?;
        if let Some(f) = &self.evaluation.format {
            format(f)?;
        }
        for selection in self
            .dataset
            .selections
            .iter()
            .chain(&self.evaluation.selections)
            .chain(self.assistant.iter().flat_map(|a| &a.selections))
        {
            omega_datasets::config::safe_relative(selection)?;
        }
        omega_datasets::config::safe_relative(&self.dataset.recipe_folder)?;
        if let Some(recipe) = &self.dataset.recipe {
            recipe.validate()?;
        }
        ensure!(
            self.dataset.validation_ratio.is_none() || self.dataset.validation_count.is_none(),
            "Choose validation_ratio or validation_count, not both"
        );
        if let Some(r) = self.dataset.validation_ratio {
            ensure!(
                r.is_finite() && (0.0..1.0).contains(&r),
                "Validation ratio must be in [0,1)"
            );
        }
        ensure!(
            self.tokenizer.min_frequency > 0
                && self.tokenizer.vocab_size
                    >= if self.tokenizer.chat_protocol {
                        260
                    } else {
                        256
                    },
            "Byte BPE requires at least 256 entries (260 with chat) and positive minimum frequency"
        );
        let t = &self.training;
        ensure!(
            t.epochs > 0 && t.max_updates != Some(0),
            "Training epochs/update limit must be positive"
        );
        omega_nn::GptConfig {
            vocab_size: 1,
            context_length: self.model.context_length,
            d_model: self.model.d_model,
            num_heads: self.model.heads,
            num_layers: self.model.layers,
            d_ff: self.model.d_ff,
        }
        .validate()
        .map_err(anyhow::Error::msg)?;
        self.session()
            .optimization
            .learning_rate(t.learning_rate, 0)
            .map_err(anyhow::Error::msg)?;
        self.session()
            .batching
            .validate()
            .map_err(anyhow::Error::msg)?;
        self.cpu().validate().map_err(anyhow::Error::msg)?;
        omega_training::run_control::SaveSchedule {
            every_updates: t.save_every_updates,
            every_epochs: t.save_every_epochs,
        }
        .validate()
        .map_err(anyhow::Error::msg)?;
        ensure!(
            t.backend != Backend::Cpu || t.device == 0,
            "CPU device must be zero"
        );
        ensure!(
            t.backend == Backend::Cpu || (t.cpu_threads.is_none() && t.matmul_threads.is_none()),
            "CPU thread overrides require the CPU backend"
        );
        ensure!(
            self.benchmark.samples > 0
                && self.benchmark.samples <= 1000
                && (1..=100).contains(&self.benchmark.warmup)
                && self.benchmark.max_seconds.is_finite()
                && self.benchmark.max_seconds > 0.0
                && self.benchmark.max_seconds <= 3600.0,
            "Invalid benchmark sample/warmup/budget settings"
        );
        if let Some(a) = &self.assistant {
            ensure!(
                a.epochs > 0
                    && a.learning_rate.is_finite()
                    && a.learning_rate > 0.0
                    && !a.selections.is_empty(),
                "Assistant stage requires data, positive epochs and learning rate"
            );
        }
        ensure!(
            !self.paths.datasets.as_os_str().is_empty()
                && !self.paths.weights.as_os_str().is_empty()
                && !self.tokenizer.path.as_os_str().is_empty(),
            "Project paths cannot be empty"
        );
        Ok(())
    }
    pub fn selections(&self) -> Vec<String> {
        if self.dataset.selections.is_empty() {
            self.dataset
                .recipe
                .as_ref()
                .map(|r| {
                    vec![format!(
                        "{}/{}/base/train",
                        self.dataset.recipe_folder, r.output
                    )]
                })
                .unwrap_or_default()
        } else {
            self.dataset.selections.clone()
        }
    }
    pub fn cpu(&self) -> omega_training::cpu::CpuThreadSettings {
        omega_training::cpu::CpuThreadSettings {
            cpu_threads: self.training.cpu_threads,
            matmul_threads: self.training.matmul_threads,
        }
    }
    pub fn session(&self) -> omega_training::SessionOptions {
        omega_training::SessionOptions {
            batching: omega_training::batching::BatchConfig {
                batch_size: self.training.batch_size,
                max_batch_tokens: self.training.max_batch_tokens,
            },
            optimization: omega_training::OptimizationOptions {
                gradient_clip_norm: self.training.gradient_clip_norm,
                warmup_updates: self.training.warmup_updates,
            },
            ..Default::default()
        }
    }
    pub fn create(&self, path: &Path) -> Result<()> {
        use std::io::Write;
        self.validate()?;
        let text = toml::to_string_pretty(self)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .context("Create project (existing files are never replaced)")?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        Ok(())
    }
    pub fn import_recipe(source: &Path, target: &Path) -> Result<Self> {
        let recipe = omega_datasets::config::Config::load(source)?;
        let config = Self {
            dataset: DatasetSettings {
                recipe: Some(recipe),
                ..Default::default()
            },
            pipeline: PipelineSettings {
                prepare_dataset: true,
                prepare_tokenizer: true,
                ..Default::default()
            },
            ..Default::default()
        };
        config.create(target)?;
        Ok(config)
    }
}

/// Update one existing/new scalar without reserializing unrelated TOML/comments.
pub fn edit_scalar(text: &str, section: &str, key: &str, value: &str) -> Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    let defaults: toml_edit::DocumentMut =
        toml::to_string(&ProjectConfig::parse(text)?)?.parse()?;
    let old = if section.is_empty() {
        &defaults[key]
    } else {
        &defaults[section][key]
    };
    let new = if old.as_array().is_some() {
        let parsed: toml_edit::DocumentMut = format!("value = {value}")
            .parse()
            .context("Enter an array such as [\"corpus\"]")?;
        parsed["value"].clone()
    } else if old.as_integer().is_some() {
        toml_edit::value(value.parse::<i64>()?)
    } else if old.as_float().is_some() {
        toml_edit::value(value.parse::<f64>()?)
    } else if old.as_bool().is_some() {
        toml_edit::value(value.parse::<bool>()?)
    } else {
        toml_edit::value(value)
    };
    let dest = if section.is_empty() {
        &mut doc[key]
    } else {
        &mut doc[section][key]
    };
    let decor = dest.as_value().map(|v| v.decor().clone());
    *dest = new;
    if let (Some(d), Some(v)) = (decor, dest.as_value_mut()) {
        *v.decor_mut() = d;
    }
    let result = doc.to_string();
    ProjectConfig::parse(&result)?;
    Ok(result)
}

pub fn save_edited(path: &Path, original: &str, new: &str) -> Result<()> {
    ProjectConfig::parse(new)?;
    ensure!(
        fs::read_to_string(path)? == original,
        "Configuration changed on disk; reload before saving"
    );
    let temp = path.with_extension(format!("toml.{}.new", std::process::id()));
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(new.as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temp, path).context("Publish edited configuration")?;
    Ok(())
}
