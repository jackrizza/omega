use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub output: String,
    pub seed: u64,
    pub split: Split,
    #[serde(default)]
    pub limits: Limits,
    pub sources: Vec<Source>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub train: f64,
    pub validation: f64,
    pub test: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_download_bytes: u64,
    pub max_file_bytes: u64,
    pub max_record_bytes: usize,
    pub max_records: usize,
    pub max_normalized_bytes: u64,
    pub max_files: usize,
    pub timeout_seconds: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_download_bytes: 1_073_741_824,
            max_file_bytes: 268_435_456,
            max_record_bytes: 1_048_576,
            max_records: 100_000,
            max_normalized_bytes: 536_870_912,
            max_files: 100,
            timeout_seconds: 300,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Jsonl,
    Json,
    Parquet,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Partition {
    Train,
    Validation,
    Test,
}

impl Partition {
    pub fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub repo: String,
    #[serde(default = "main_revision")]
    pub revision: String,
    pub files: Vec<String>,
    pub format: Format,
    pub language: String,
    pub permitted_use: String,
    pub mapping: Mapping,
    /// Fixed upstream partition, otherwise seeded group assignment.
    pub partition: Option<Partition>,
    /// One group for this entire source, e.g. a family of related books.
    pub group: Option<String>,
    /// Top-level column containing an original document/conversation family ID.
    pub group_column: Option<String>,
    /// Defaults to repo; share across repos to identify related source families.
    pub group_namespace: Option<String>,
}

fn main_revision() -> String {
    "main".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Mapping {
    Text {
        column: String,
    },
    Messages {
        column: String,
        #[serde(default = "role_field")]
        role_field: String,
        #[serde(default = "content_field")]
        content_field: String,
        #[serde(default = "user_role")]
        user_role: String,
        #[serde(default = "assistant_role")]
        assistant_role: String,
        #[serde(default = "system_role")]
        system_role: String,
    },
    Instruction {
        prompt_column: String,
        response_column: String,
        input_column: Option<String>,
        system: Option<String>,
    },
}

fn role_field() -> String {
    "role".into()
}
fn content_field() -> String {
    "content".into()
}
fn user_role() -> String {
    "user".into()
}
fn assistant_role() -> String {
    "assistant".into()
}
fn system_role() -> String {
    "system".into()
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        crate::reject_links(path)?;
        let text = crate::read_bounded(
            fs::File::open(path).with_context(|| format!("Open {}", path.display()))?,
            1_048_576,
        )?;
        let config: Self = toml::from_str(std::str::from_utf8(&text)?)
            .with_context(|| format!("Parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "schema_version must be 1");
        safe_relative(&self.output)?;
        ensure!(
            !self.output.contains('/'),
            "output must be a single new directory name"
        );
        let ratios = [self.split.train, self.split.validation, self.split.test];
        ensure!(
            ratios
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0)
                && (ratios.iter().sum::<f64>() - 1.0).abs() < 1e-10,
            "split ratios must be finite, nonnegative, and sum to 1"
        );
        ensure!(self.split.train > 0.0, "split.train must be positive");
        let limits = &self.limits;
        ensure!(
            limits.max_download_bytes > 0
                && limits.max_file_bytes > 0
                && limits.max_record_bytes > 0
                && limits.max_records > 0
                && limits.max_normalized_bytes > 0
                && limits.max_files > 0
                && limits.timeout_seconds > 0,
            "all limits must be positive"
        );
        ensure!(!self.sources.is_empty(), "sources must not be empty");
        let mut ids = BTreeSet::new();
        for source in &self.sources {
            ensure!(
                !source.id.is_empty()
                    && source
                        .id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
                "source id must contain only ASCII letters, digits, '-' or '_'"
            );
            ensure!(
                ids.insert(source.id.to_ascii_lowercase()),
                "duplicate source id (case insensitive): {}",
                source.id
            );
            safe_relative(&source.id)?;
            safe_relative(&source.repo)?;
            ensure!(
                source.repo.split('/').count() <= 2,
                "repo must be name or owner/name"
            );
            ensure!(
                !source.revision.trim().is_empty(),
                "revision must not be empty"
            );
            ensure!(
                !source.language.trim().is_empty() && !source.permitted_use.trim().is_empty(),
                "source {} requires language and permitted_use",
                source.id
            );
            ensure!(
                !source.files.is_empty(),
                "source {} requires files",
                source.id
            );
            for pattern in &source.files {
                ensure!(
                    !pattern.starts_with('/')
                        && !pattern.contains('\\')
                        && !pattern.contains(':')
                        && !pattern
                            .split('/')
                            .any(|p| p == ".." || p == "." || p.is_empty()),
                    "unsafe file glob: {pattern}"
                );
                globset::GlobBuilder::new(pattern)
                    .literal_separator(true)
                    .build()?;
            }
            ensure!(
                source.group.is_none() || source.group_column.is_none(),
                "choose group or group_column, not both"
            );
            for value in [&source.group, &source.group_column, &source.group_namespace]
                .into_iter()
                .flatten()
            {
                ensure!(!value.trim().is_empty(), "group settings must not be blank");
            }
            match &source.mapping {
                Mapping::Text { column } => {
                    ensure!(!column.is_empty(), "text column must not be empty")
                }
                Mapping::Messages {
                    column,
                    role_field,
                    content_field,
                    user_role,
                    assistant_role,
                    system_role,
                } => {
                    ensure!(
                        [
                            column,
                            role_field,
                            content_field,
                            user_role,
                            assistant_role,
                            system_role
                        ]
                        .iter()
                        .all(|s| !s.is_empty()),
                        "message mapping fields must not be empty"
                    );
                    ensure!(
                        BTreeSet::from([user_role, assistant_role, system_role]).len() == 3,
                        "message role aliases must be distinct"
                    );
                }
                Mapping::Instruction {
                    prompt_column,
                    response_column,
                    input_column,
                    system,
                } => {
                    ensure!(
                        !prompt_column.is_empty()
                            && !response_column.is_empty()
                            && input_column.as_ref().is_none_or(|s| !s.is_empty()),
                        "instruction columns must not be empty"
                    );
                    ensure!(
                        system.as_ref().is_none_or(|s| !s.trim().is_empty()),
                        "system must not be blank"
                    );
                }
            }
        }
        Ok(())
    }
}

/// Portable relative paths only: reject traversal, Windows aliases and devices.
pub fn safe_relative(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && !value.contains(['\\', ':', '*', '?', '"', '<', '>', '|'])
            && !value.chars().any(char::is_control),
        "unsafe relative path: {value:?}"
    );
    for part in value.split('/') {
        let stem = part.split('.').next().unwrap_or("").to_ascii_uppercase();
        ensure!(
            !part.is_empty()
                && part != "."
                && part != ".."
                && !part.ends_with(['.', ' '])
                && ![
                    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6",
                    "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7",
                    "LPT8", "LPT9"
                ]
                .contains(&stem.as_str()),
            "unsafe relative path: {value:?}"
        );
    }
    Ok(())
}
