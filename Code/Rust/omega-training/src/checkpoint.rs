//! Inference checkpoints: full-precision weights, model dimensions, and tokenizer.
//!
//! A failed save intentionally leaves its numbered directory reserved and visibly
//! incomplete (no `COMPLETE` file). Numbers are never reused while entries remain.
//! The weights root must be trusted: atomic reservation coordinates cooperating
//! writers, not processes that modify another writer's reserved directory.
//! Completion is a logical commit, not a power-loss durability guarantee.

use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use omega_nn::{Cpu, Gpt, GptConfig};
use omega_tokenizer::Tokens;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CONFIG: &str = "config.json";
const TOKENIZER: &str = "tokenizer.json";
const WEIGHTS: &str = "model";
const COMPLETE: &str = "COMPLETE";
const MANIFEST: &str = "manifest.json";
const VERSIONED_COMPLETE: &[u8] = b"omega-checkpoint-v1\n";
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;
const TOKENIZER_ALGORITHM: &str = "sha256-canonical-json-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
}

/// Full effective tokenizer pipeline identity, including vocabulary IDs,
/// normalization, pre/post-processing, decoder, special tokens and truncation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerIdentity {
    pub algorithm: String,
    pub sha256: String,
}

/// Caller-supplied provenance. A fingerprint's interpretation is specified by
/// the containing dataset's `fingerprint_kind`; it is not necessarily file bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentFingerprint {
    pub id: String,
    /// Either `training` or `validation`; document IDs must be disjoint.
    pub partition: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProvenance {
    pub selections: Vec<String>,
    pub format: String,
    /// For example `token-ids-le-u32-v1`: SHA256 of unchunked IDs as LE u32 bytes.
    /// The manifest's tokenizer identity separately pins the encoding pipeline.
    pub fingerprint_kind: String,
    pub documents: Vec<DocumentFingerprint>,
    pub split_seed: u64,
    pub split_policy: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingProvenance {
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub optimizer: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub package: String,
    pub version: String,
    #[serde(deserialize_with = "required_option")]
    pub revision: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub rustc: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub target: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub cargo_lock_sha256: Option<String>,
}

impl BuildIdentity {
    /// Known compile-time identity. Existing readers retain nullable legacy
    /// fields, but exact continuation requires this build's resolved identity.
    pub fn current() -> Self {
        Self {
            package: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            revision: option_env!("OMEGA_BUILD_REVISION")
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| Some(format!("source-sha256:{}", env!("OMEGA_SOURCE_SHA256")))),
            rustc: Some(env!("OMEGA_RUSTC_IDENTITY").into()),
            target: Some(env!("OMEGA_BUILD_TARGET").into()),
            cargo_lock_sha256: Some(sha256_bytes(include_bytes!("../../Cargo.lock"))),
        }
    }
}

/// Unknown provenance is serialized explicitly as null, never inferred from
/// filenames or invented. Metadata describes the caller's run, not resume state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointMetadata {
    #[serde(deserialize_with = "required_option")]
    pub dataset: Option<DatasetProvenance>,
    #[serde(deserialize_with = "required_option")]
    pub training: Option<TrainingProvenance>,
    #[serde(deserialize_with = "required_option")]
    pub build: Option<BuildIdentity>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    pub schema_version: u32,
    /// Authoritative architecture; config.json is retained for legacy tooling
    /// and must agree exactly with this configuration.
    pub model: ModelConfig,
    pub tokenizer: TokenizerIdentity,
    pub metadata: CheckpointMetadata,
    /// Required in schema 2, absent in legacy/plain schema 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<crate::assistant::ChatArtifact>,
}

// Unlike normal Option deserialization, this requires the field to be present;
// explicit null means unavailable, a missing field means an invalid manifest.
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => {
            let sorted: std::collections::BTreeMap<_, _> = object.into_iter().collect();
            serde_json::Value::Object(
                sorted
                    .into_iter()
                    .map(|(key, value)| (key, canonical_json(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonical_json).collect())
        }
        value => value,
    }
}

/// SHA256 of compact JSON with recursively sorted object keys, preserving array
/// order, from the tokenizer's complete effective serialized pipeline. This
/// detects saved-pipeline changes, not an independently authenticated historical
/// relationship between arbitrary caller-supplied weights and tokenizer IDs.
pub fn tokenizer_identity(tokenizer: &Tokens) -> Result<TokenizerIdentity, String> {
    let value: serde_json::Value = serde_json::from_str(&tokenizer.to_json()?)
        .map_err(|err| format!("Cannot parse serialized tokenizer: {err}"))?;
    let canonical = serde_json::to_vec(&canonical_json(value))
        .map_err(|err| format!("Cannot canonicalize tokenizer: {err}"))?;
    Ok(TokenizerIdentity {
        algorithm: TOKENIZER_ALGORITHM.into(),
        sha256: sha256_bytes(&canonical),
    })
}

fn validate_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{name} must be a lowercase 64-digit SHA256 digest"));
    }
    Ok(())
}

fn validate_metadata(metadata: &CheckpointMetadata) -> Result<(), String> {
    if let Some(dataset) = &metadata.dataset {
        if dataset.format.is_empty()
            || dataset.fingerprint_kind.is_empty()
            || dataset.split_policy.is_empty()
            || dataset.documents.is_empty()
            || dataset.selections.iter().any(String::is_empty)
        {
            return Err(
                "Dataset provenance requires format, fingerprint kind, split policy and documents"
                    .into(),
            );
        }
        let mut ids = BTreeSet::new();
        for document in &dataset.documents {
            if document.id.is_empty() || !ids.insert(&document.id) {
                return Err(
                    "Dataset provenance document IDs must be nonempty and unique across partitions"
                        .into(),
                );
            }
            if !matches!(document.partition.as_str(), "training" | "validation") {
                return Err("Dataset provenance partition must be training or validation".into());
            }
            validate_digest(&document.sha256, "Document fingerprint")?;
        }
    }
    if let Some(training) = &metadata.training
        && (training.epochs == 0
            || !training.learning_rate.is_finite()
            || training.learning_rate <= 0.0
            || training.optimizer.is_empty())
    {
        return Err(
            "Training provenance requires positive epochs/learning rate and an optimizer".into(),
        );
    }
    if let Some(build) = &metadata.build {
        if build.package.is_empty() || build.version.is_empty() {
            return Err("Build identity requires package and version".into());
        }
        if let Some(digest) = &build.cargo_lock_sha256 {
            validate_digest(digest, "Cargo.lock fingerprint")?;
        }
    }
    Ok(())
}

fn validate_manifest(manifest: &CheckpointManifest) -> Result<(), String> {
    if !matches!(
        (manifest.schema_version, &manifest.chat),
        (1, None) | (2, Some(_))
    ) {
        return Err(format!(
            "Unsupported checkpoint schema version {}",
            manifest.schema_version
        ));
    }
    GptConfig::from(manifest.model.clone()).validate()?;
    if let Some(chat) = &manifest.chat {
        chat.validate(manifest.model.vocab_size)?;
    }
    if manifest.tokenizer.algorithm != TOKENIZER_ALGORITHM {
        return Err(format!(
            "Unsupported tokenizer identity algorithm {}",
            manifest.tokenizer.algorithm
        ));
    }
    validate_digest(&manifest.tokenizer.sha256, "Tokenizer identity")?;
    validate_metadata(&manifest.metadata)
}

/// Read a completed checkpoint's manifest without loading tensors. Returns None
/// only for an unversioned checkpoint with an empty legacy COMPLETE marker and
/// no manifest. Versioned markers require manifests, preventing accidental
/// downgrade when a manifest is missing. These checks are not a signature.
pub fn read_checkpoint_manifest(path: &Path) -> Result<Option<CheckpointManifest>, String> {
    if !path.is_dir() || !path.join(COMPLETE).is_file() {
        return Err(format!(
            "Incomplete checkpoint (missing {COMPLETE} file): {}",
            path.display()
        ));
    }
    let marker = fs::read(path.join(COMPLETE))
        .map_err(|err| format!("Cannot read checkpoint completion marker: {err}"))?;
    let manifest_path = path.join(MANIFEST);
    if marker.is_empty() && !manifest_path.exists() {
        return Ok(None);
    }
    if marker != VERSIONED_COMPLETE {
        return Err("Unsupported or invalid checkpoint completion marker".into());
    }
    let bytes = fs::read(&manifest_path).map_err(|err| {
        format!(
            "Cannot read required manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    parse_checkpoint_manifest(&bytes).map(Some).map_err(|err| {
        format!(
            "Invalid checkpoint manifest {}: {err}",
            manifest_path.display()
        )
    })
}

/// Shared by explicit loading and bounded, metadata-only discovery.
pub(crate) fn parse_checkpoint_manifest(bytes: &[u8]) -> Result<CheckpointManifest, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|err| format!("Invalid JSON: {err}"))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or("Checkpoint manifest requires an integer schema_version")?;
    if !matches!(version, 1 | 2) {
        return Err(format!("Unsupported checkpoint schema version {version}"));
    }
    if version == 1 && value.get("chat").is_some() {
        return Err("Schema 1 cannot contain chat metadata".into());
    }
    let manifest: CheckpointManifest =
        serde_json::from_value(value).map_err(|err| format!("Invalid manifest fields: {err}"))?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

impl From<&GptConfig> for ModelConfig {
    fn from(config: &GptConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            context_length: config.context_length,
            d_model: config.d_model,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            d_ff: config.d_ff,
        }
    }
}

impl From<ModelConfig> for GptConfig {
    fn from(config: ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            context_length: config.context_length,
            d_model: config.d_model,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            d_ff: config.d_ff,
        }
    }
}

fn validate(config: &GptConfig, tokenizer: &Tokens) -> Result<(), String> {
    config.validate()?;
    let size = omega_nn::validate_tokenizer(tokenizer)?;
    if size != config.vocab_size {
        return Err(format!(
            "Tokenizer vocabulary size {size} does not match model vocab_size {}",
            config.vocab_size
        ));
    }
    Ok(())
}

/// Atomically reserve `lowercase_run_name-N`, above every matching entry.
/// ASCII case variants share a canonical reservation path, including on
/// case-sensitive filesystems. Existing mixed-case names remain discoverable.
/// All concurrent writers must use this policy; older binaries can still race
/// by creating differently cased names on a case-sensitive filesystem.
/// Files and directories both occupy numbers; gaps are not filled. A suffix is
/// numeric only if it is nonempty ASCII decimal digits (leading zeros count).
/// Numbers are bounded by u64; an oversized numeric suffix is an error, not an
/// irrelevant entry. Missing parent directories are created.
pub fn reserve_run_directory(weights_root: &Path, run_name: &str) -> Result<PathBuf, String> {
    let canonical_name = canonical_run_name(run_name)?;
    fs::create_dir_all(weights_root).map_err(|err| {
        format!(
            "Cannot create weights root {}: {err}",
            weights_root.display()
        )
    })?;
    if !weights_root.is_dir() {
        return Err(format!(
            "Weights root is not a directory: {}",
            weights_root.display()
        ));
    }
    loop {
        let mut maximum = 0_u64;
        let entries = fs::read_dir(weights_root)
            .map_err(|err| format!("Cannot scan weights root {}: {err}", weights_root.display()))?;
        for entry in entries {
            let entry = entry.map_err(|err| format!("Cannot read weights root entry: {err}"))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some(number) = checkpoint_number(name, &canonical_name)? {
                maximum = maximum.max(number);
            }
        }
        let next = maximum
            .checked_add(1)
            .ok_or_else(|| format!("Checkpoint numbering exhausted for {run_name}"))?;
        let path = weights_root.join(format!("{canonical_name}-{next}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(format!(
                    "Cannot reserve checkpoint {}: {err}",
                    path.display()
                ));
            }
        }
    }
}

pub(crate) fn canonical_run_name(run_name: &str) -> Result<String, String> {
    if run_name.is_empty()
        || !run_name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err(
            "Run name must contain only ASCII letters, digits, '-' or '_' and must not be empty"
                .into(),
        );
    }
    Ok(run_name.to_ascii_lowercase())
}

/// The numbering contract is identical for reservation and discovery: ASCII case
/// variants and leading zeros count, and oversized numeric suffixes are errors.
pub(crate) fn checkpoint_number(name: &str, run_name: &str) -> Result<Option<u64>, String> {
    let prefix = format!("{run_name}-");
    let Some(start) = name.get(..prefix.len()) else {
        return Ok(None);
    };
    if !start.eq_ignore_ascii_case(&prefix) {
        return Ok(None);
    }
    let suffix = &name[prefix.len()..];
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Ok(None);
    }
    suffix
        .parse::<u64>()
        .map(Some)
        .map_err(|_| format!("Checkpoint number exceeds u64 in {name}"))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("Cannot create {}: {err}", path.display()))?;
    file.write_all(bytes)
        .map_err(|err| format!("Cannot write {}: {err}", path.display()))?;
    file.flush()
        .map_err(|err| format!("Cannot flush {}: {err}", path.display()))
}

/// Save to a new numbered directory without replacing an existing checkpoint.
/// Model architecture, vocabulary size, and dense tokenizer IDs are validated
/// before reserving a name. Tokenizer semantic identity remains the caller's duty.
/// Failures after reservation leave an incomplete directory for inspection.
/// Writes schema version 1 with explicitly unavailable run/build provenance;
/// use `save_checkpoint_with_metadata` when that information is known.
pub fn save_checkpoint<B: Backend>(
    weights_root: &Path,
    run_name: &str,
    model: Gpt<B>,
    config: &GptConfig,
    tokenizer: &Tokens,
) -> Result<PathBuf, String> {
    save_checkpoint_with_metadata(
        weights_root,
        run_name,
        model,
        config,
        tokenizer,
        &CheckpointMetadata::default(),
    )
}

/// Save an inference checkpoint with caller-supplied, validated provenance.
/// The complete effective tokenizer pipeline is fingerprinted automatically.
/// Metadata is persisted before the versioned COMPLETE marker, which is written
/// last. This is not training-state persistence or a cryptographic signature.
pub fn save_checkpoint_with_metadata<B: Backend>(
    weights_root: &Path,
    run_name: &str,
    model: Gpt<B>,
    config: &GptConfig,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
) -> Result<PathBuf, String> {
    save_checkpoint_with_artifacts(
        weights_root,
        run_name,
        model,
        config,
        tokenizer,
        metadata,
        |_| Ok(()),
    )
}

/// Internal extension point: all additional artifacts must succeed before the
/// completion marker is written. Never modify an already completed checkpoint.
pub(crate) fn save_checkpoint_with_artifacts<B: Backend>(
    weights_root: &Path,
    run_name: &str,
    model: Gpt<B>,
    config: &GptConfig,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
    extra: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<PathBuf, String> {
    model.validate_config(config)?;
    validate(config, tokenizer)?;
    validate_metadata(metadata)?;
    let chat = crate::assistant::ChatArtifact::from_tokenizer(tokenizer)?;
    let manifest = CheckpointManifest {
        schema_version: if chat.is_some() {
            2
        } else {
            CHECKPOINT_SCHEMA_VERSION
        },
        model: ModelConfig::from(config),
        tokenizer: tokenizer_identity(tokenizer)?,
        metadata: metadata.clone(),
        chat,
    };
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|err| format!("Cannot serialize checkpoint manifest: {err}"))?;
    let json = serde_json::to_vec_pretty(&ModelConfig::from(config))
        .map_err(|err| format!("Cannot serialize model config: {err}"))?;
    let path = reserve_run_directory(weights_root, run_name)?;
    write_new(&path.join(CONFIG), &json)?;
    write_new(&path.join(MANIFEST), &manifest_json)?;
    tokenizer.save(path.join(TOKENIZER))?;
    model
        .save_file(
            path.join(WEIGHTS),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|err| format!("Cannot save model in {}: {err}", path.display()))?;

    extra(&path)?;

    // A partial marker is invalid on load; successful contents distinguish this
    // schema from legacy checkpoints even if manifest.json is later lost.
    write_new(&path.join(COMPLETE), VERSIONED_COMPLETE)?;
    Ok(path)
}

/// Load a completed checkpoint directory onto the default CPU device.
/// Checkpoints must be trusted and compatible with this model/Burn version;
/// Version 1 checks tokenizer identity and metadata/config agreement before
/// initializing tensors. Legacy checkpoints (empty COMPLETE, no manifest) stay
/// loadable but cannot authenticate tokenizer identity or provenance. There is no
/// model checksum, signature, resource limit, or optimizer/training state.
pub fn load_checkpoint(path: &Path) -> Result<(Gpt<Cpu>, GptConfig, Tokens), String> {
    load_checkpoint_on_device::<Cpu>(path, &Default::default())
}

/// Load inference weights onto an explicit backend/device; this does not restore training state.
pub fn load_checkpoint_on_device<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(Gpt<B>, GptConfig, Tokens), String> {
    let (config, tokenizer, _) = load_checkpoint_header(path)?;
    let model = config
        .init::<B>(device)?
        .load_file(
            path.join(WEIGHTS),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .map_err(|err| format!("Cannot load model from {}: {err}", path.display()))?;
    model.validate_config(&config)?;
    Ok((model, config, tokenizer))
}

/// Read a validated configuration/tokenizer header without allocating model tensors.
/// Includes supported legacy checkpoints, for device buffer preflight before load.
pub fn read_checkpoint_config(path: &Path) -> Result<GptConfig, String> {
    load_checkpoint_header(path).map(|(config, _, _)| config)
}

/// Validate completed metadata and the full tokenizer before tensor loading.
pub(crate) fn load_checkpoint_header(
    path: &Path,
) -> Result<(GptConfig, Tokens, Option<CheckpointManifest>), String> {
    let manifest = read_checkpoint_manifest(path)?;
    let config_path = path.join(CONFIG);
    let bytes = fs::read(&config_path)
        .map_err(|err| format!("Cannot read {}: {err}", config_path.display()))?;
    let config = parse_checkpoint_config(&bytes, manifest.as_ref())
        .map_err(|err| format!("Invalid model config {}: {err}", config_path.display()))?;
    let tokenizer = Tokens::new(path.join(TOKENIZER))?;
    validate(&config, &tokenizer)?;
    if let Some(manifest) = &manifest
        && tokenizer_identity(&tokenizer)? != manifest.tokenizer
    {
        return Err(
            "Checkpoint tokenizer identity mismatch: saved pipeline differs from manifest".into(),
        );
    }
    if let Some(chat) = manifest.as_ref().and_then(|m| m.chat.as_ref()) {
        chat.validate_tokenizer(&tokenizer)?;
    }
    Ok((config, tokenizer, manifest))
}

pub(crate) fn parse_checkpoint_config(
    bytes: &[u8],
    manifest: Option<&CheckpointManifest>,
) -> Result<GptConfig, String> {
    let dto: ModelConfig =
        serde_json::from_slice(bytes).map_err(|err| format!("Invalid config JSON: {err}"))?;
    let config = if let Some(manifest) = manifest {
        if dto != manifest.model {
            return Err(
                "Checkpoint config.json does not match authoritative manifest model configuration"
                    .into(),
            );
        }
        GptConfig::from(manifest.model.clone())
    } else {
        GptConfig::from(dto)
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor};
    use tempfile::tempdir;

    fn tokenizer(root: &Path) -> Tokens {
        let path = root.join("source-tokenizer.json");
        fs::write(
            &path,
            r#"{
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}
        }"#,
        )
        .unwrap();
        Tokens::new(path).unwrap()
    }

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 3,
            context_length: 4,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a real discrete Vulkan GPU; run explicitly under a timeout"]
    fn gpu_artifact_failure_never_publishes_complete_and_preserves_numbering() {
        let selected = crate::gpu::initialize_vulkan(0).unwrap();
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let config = config();
        let model = config.init::<crate::gpu::Gpu>(selected.device()).unwrap();
        let existing = save_checkpoint(
            temp.path(),
            "gpu-failure",
            model.clone(),
            &config,
            &tokenizer,
        )
        .unwrap();
        let original = fs::read(existing.join(WEIGHTS).with_extension("mpk")).unwrap();
        let error = save_checkpoint_with_artifacts(
            temp.path(),
            "gpu-failure",
            model.clone(),
            &config,
            &tokenizer,
            &CheckpointMetadata::default(),
            |path| {
                assert!(path.join("model.mpk").is_file());
                assert!(!path.join(COMPLETE).exists());
                write_new(&path.join("partial-state"), b"injected failure")?;
                Err("injected GPU artifact failure".into())
            },
        )
        .unwrap_err();
        assert_eq!(error, "injected GPU artifact failure");
        let incomplete = temp.path().join("gpu-failure-2");
        assert!(incomplete.join("model.mpk").is_file());
        assert!(incomplete.join("partial-state").is_file());
        assert!(!incomplete.join(COMPLETE).exists());
        assert!(
            load_checkpoint_on_device::<crate::gpu::Gpu>(&incomplete, selected.device()).is_err()
        );
        let subsequent =
            save_checkpoint(temp.path(), "gpu-failure", model, &config, &tokenizer).unwrap();
        assert_eq!(subsequent.file_name().unwrap(), "gpu-failure-3");
        assert_eq!(fs::read(existing.join("model.mpk")).unwrap(), original);
        assert!(load_checkpoint_on_device::<crate::gpu::Gpu>(&existing, selected.device()).is_ok());
    }

    #[test]
    fn extra_artifacts_fail_before_completion_without_touching_existing_saves() {
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let config = config();
        let root = temp.path().join("weights");
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let existing = save_checkpoint(&root, "extra", model.clone(), &config, &tokenizer).unwrap();
        let marker = fs::read(existing.join(COMPLETE)).unwrap();
        let error = save_checkpoint_with_artifacts(
            &root,
            "extra",
            model,
            &config,
            &tokenizer,
            &CheckpointMetadata::default(),
            |path| {
                assert!(path.join("model.mpk").is_file());
                assert!(!path.join(COMPLETE).exists());
                write_new(&path.join("partial-state"), b"partial")?;
                Err("injected additional artifact failure".into())
            },
        )
        .unwrap_err();
        assert!(error.contains("additional artifact failure"));
        assert!(root.join("extra-2/partial-state").is_file());
        assert!(!root.join("extra-2/COMPLETE").exists());
        assert_eq!(fs::read(existing.join(COMPLETE)).unwrap(), marker);
        assert!(load_checkpoint(&root.join("extra-2")).is_err());
    }

    #[test]
    fn starts_at_one_and_creates_parents() {
        let temp = tempdir().unwrap();
        let root = temp.path().join("nested/weights");
        assert_eq!(
            reserve_run_directory(&root, "Run_1-a").unwrap(),
            root.join("run_1-a-1")
        );
        assert_eq!(
            reserve_run_directory(&root, "Run_1-a").unwrap(),
            root.join("run_1-a-2")
        );
    }

    #[test]
    fn uses_maximum_including_files_and_ignores_unrelated_names() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        fs::create_dir(root.join("run-1")).unwrap();
        fs::create_dir(root.join("run-4")).unwrap();
        fs::write(root.join("run-0009"), "keep").unwrap();
        for name in [
            "other-999",
            "run-extra-999",
            "run-",
            "run--100",
            "run-+100",
            "run-100.bak",
            "run-100x",
            "run-１２",
        ] {
            fs::create_dir(root.join(name)).unwrap();
        }
        assert_eq!(
            reserve_run_directory(root, "run").unwrap(),
            root.join("run-10")
        );
        assert_eq!(fs::read_to_string(root.join("run-0009")).unwrap(), "keep");
        assert!(!root.join("run-2").exists());
    }

    #[test]
    fn run_names_share_numbers_across_case_variants() {
        let temp = tempdir().unwrap();
        fs::create_dir(temp.path().join("Foo-2")).unwrap();
        assert_eq!(
            reserve_run_directory(temp.path(), "foo").unwrap(),
            temp.path().join("foo-3")
        );
    }

    #[test]
    fn high_numbers_and_overflow_are_checked() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        fs::write(root.join(format!("run-{}", u64::MAX - 1)), "keep").unwrap();
        assert_eq!(
            reserve_run_directory(root, "run").unwrap(),
            root.join(format!("run-{}", u64::MAX))
        );
        assert!(reserve_run_directory(root, "run").is_err());
        fs::create_dir(root.join("huge-18446744073709551616")).unwrap();
        assert!(reserve_run_directory(root, "huge").is_err());
    }

    #[test]
    fn rejects_invalid_names_without_creating_root_and_rejects_file_parent() {
        let temp = tempdir().unwrap();
        let root = temp.path().join("weights");
        for name in [
            "", ".", "..", "../run", "a/b", "a\\b", "a b", "é", "a:b", "a\n", "a\0",
        ] {
            assert!(
                reserve_run_directory(&root, name).is_err(),
                "accepted {name:?}"
            );
        }
        assert!(!root.exists());
        fs::write(&root, "keep").unwrap();
        assert!(reserve_run_directory(&root, "run").is_err());
        assert_eq!(fs::read_to_string(root).unwrap(), "keep");
    }

    #[test]
    fn concurrent_reservations_are_unique() {
        let temp = tempdir().unwrap();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let root = temp.path().to_path_buf();
                std::thread::spawn(move || reserve_run_directory(&root, "run").unwrap())
            })
            .collect();
        let mut paths: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), 8);
    }

    #[test]
    fn concurrent_case_variants_share_one_numbering_series() {
        use std::sync::{Arc, Barrier};
        let temp = tempdir().unwrap();
        let root = temp.path();
        fs::create_dir(root.join("FoO-2")).unwrap();
        fs::write(root.join("FOO-0004"), "legacy file").unwrap();
        fs::write(root.join("FoO-2/keep"), "legacy checkpoint").unwrap();
        let barrier = Arc::new(Barrier::new(16));
        let handles: Vec<_> = (0..16)
            .map(|index| {
                let root = root.to_path_buf();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let name = ["Foo", "foo", "FOO", "fOo"][index % 4];
                    reserve_run_directory(&root, name).unwrap()
                })
            })
            .collect();
        let paths: std::collections::BTreeSet<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let expected = (5..=20).map(|n| root.join(format!("foo-{n}"))).collect();
        assert_eq!(paths, expected);
        assert_eq!(
            fs::read_to_string(root.join("FOO-0004")).unwrap(),
            "legacy file"
        );
        assert_eq!(
            fs::read_to_string(root.join("FoO-2/keep")).unwrap(),
            "legacy checkpoint"
        );
        for path in paths {
            assert!(path.is_dir());
            assert!(load_checkpoint(&path).is_err());
        }
    }

    #[test]
    fn model_round_trip_preserves_logits_config_and_tokenizer() {
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let config = config();
        let device = Default::default();
        let model = config.init::<Cpu>(&device).unwrap();
        let input = Tensor::<Cpu, 2, Int>::from_data([[0, 1, 2, 1]], &device);
        let expected = model
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let root = temp.path().join("weights");
        let path = save_checkpoint(&root, "run", model, &config, &tokenizer).unwrap();
        assert!(path.join("model.mpk").is_file());
        assert!(path.join(COMPLETE).is_file());
        let (loaded, loaded_config, loaded_tokenizer) = load_checkpoint(&path).unwrap();
        assert_eq!(
            ModelConfig::from(&config),
            ModelConfig::from(&loaded_config)
        );
        assert_eq!(tokenizer.vocab_size(), loaded_tokenizer.vocab_size());
        for id in 0..config.vocab_size as u32 {
            assert_eq!(tokenizer.id_to_token(id), loaded_tokenizer.id_to_token(id));
        }
        assert_eq!(
            loaded_tokenizer.encode("hello", false).unwrap().get_ids(),
            &[1]
        );
        assert_eq!(
            expected,
            loaded.forward(input).into_data().to_vec::<f32>().unwrap()
        );
        let original = fs::read(path.join("model.mpk")).unwrap();
        let second =
            save_checkpoint(&root, "run", loaded, &loaded_config, &loaded_tokenizer).unwrap();
        assert_eq!(second, root.join("run-2"));
        assert_eq!(fs::read(path.join("model.mpk")).unwrap(), original);
        fs::remove_file(path.join(COMPLETE)).unwrap();
        assert!(load_checkpoint(&path).is_err());
        fs::create_dir(path.join(COMPLETE)).unwrap();
        assert!(load_checkpoint(&path).is_err());
    }

    #[test]
    fn rejects_every_architecture_mismatch_before_reserving() {
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let original = config();
        let model = original.init::<Cpu>(&Default::default()).unwrap();
        let root = temp.path().join("weights");
        for field in 0..6 {
            let mut supplied = original.clone();
            match field {
                0 => supplied.vocab_size += 1,
                1 => supplied.context_length += 1,
                2 => supplied.d_model *= 2,
                3 => supplied.num_heads = 1,
                4 => supplied.num_layers += 1,
                _ => supplied.d_ff += 1,
            }
            let error =
                save_checkpoint(&root, "run", model.clone(), &supplied, &tokenizer).unwrap_err();
            assert!(error.contains("mismatch"), "{error}");
            assert!(!root.exists());
        }
    }

    #[test]
    fn rejects_incomplete_and_mismatched_checkpoints() {
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let mut config = config();
        let root = temp.path().join("weights");
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        config.vocab_size += 1;
        assert!(save_checkpoint(&root, "run", model, &config, &tokenizer).is_err());
        assert!(!root.exists());
        let reservation = reserve_run_directory(&root, "run").unwrap();
        assert!(load_checkpoint(&reservation).is_err());
        config.vocab_size -= 1;
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let path = save_checkpoint(&root, "run", model, &config, &tokenizer).unwrap();
        assert_eq!(path, root.join("run-2"));
        config.vocab_size += 1;
        fs::write(
            path.join(CONFIG),
            serde_json::to_vec(&ModelConfig::from(&config)).unwrap(),
        )
        .unwrap();
        assert!(load_checkpoint(&path).is_err());
    }

    fn saved_checkpoint(root: &Path) -> PathBuf {
        let tokenizer = tokenizer(root);
        let config = config();
        save_checkpoint(
            &root.join("weights"),
            "run",
            config.init::<Cpu>(&Default::default()).unwrap(),
            &config,
            &tokenizer,
        )
        .unwrap()
    }

    #[test]
    fn versioned_manifest_round_trip_preserves_known_and_unknown_provenance() {
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let config = config();
        let metadata = CheckpointMetadata {
            dataset: Some(DatasetProvenance {
                selections: vec!["examples/greetings".into()],
                format: "txt".into(),
                fingerprint_kind: "token-ids-le-u32-v1".into(),
                documents: vec![DocumentFingerprint {
                    id: "examples/greetings/a.txt".into(),
                    partition: "training".into(),
                    sha256: sha256_bytes(&[0, 0, 0, 0, 1, 0, 0, 0]),
                }],
                split_seed: 42,
                split_policy: "none".into(),
            }),
            training: Some(TrainingProvenance {
                epochs: 2,
                learning_rate: 0.003,
                seed: 42,
                optimizer: "adam".into(),
            }),
            build: Some(BuildIdentity::current()),
        };
        let path = save_checkpoint_with_metadata(
            temp.path(),
            "known",
            config.init::<Cpu>(&Default::default()).unwrap(),
            &config,
            &tokenizer,
            &metadata,
        )
        .unwrap();
        let manifest = read_checkpoint_manifest(&path).unwrap().unwrap();
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.model, ModelConfig::from(&config));
        assert_eq!(manifest.tokenizer, tokenizer_identity(&tokenizer).unwrap());
        assert_eq!(manifest.metadata, metadata);
        assert_eq!(fs::read(path.join(COMPLETE)).unwrap(), VERSIONED_COMPLETE);
        assert!(load_checkpoint(&path).is_ok());
        let unknown = save_checkpoint(
            temp.path(),
            "unknown",
            config.init::<Cpu>(&Default::default()).unwrap(),
            &config,
            &tokenizer,
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(unknown.join(MANIFEST)).unwrap()).unwrap();
        for field in ["dataset", "training", "build"] {
            assert!(value["metadata"].as_object().unwrap().contains_key(field));
            assert!(value["metadata"][field].is_null());
        }
    }

    #[test]
    fn tokenizer_digest_is_canonical_and_covers_ids_and_processing() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let left: serde_json::Value =
            serde_json::from_str(r#"{"b":{"y":1,"x":2},"a":[2,1]}"#).unwrap();
        let right: serde_json::Value =
            serde_json::from_str(r#"{"a":[2,1],"b":{"x":2,"y":1}}"#).unwrap();
        assert_eq!(
            serde_json::to_vec(&canonical_json(left)).unwrap(),
            serde_json::to_vec(&canonical_json(right)).unwrap()
        );
        let temp = tempdir().unwrap();
        let original = tokenizer(temp.path());
        let original_identity = tokenizer_identity(&original).unwrap();
        let saved = temp.path().join("copy.json");
        original.save(&saved).unwrap();
        assert_eq!(
            original_identity,
            tokenizer_identity(&Tokens::new(&saved).unwrap()).unwrap()
        );
        let value: serde_json::Value = serde_json::from_str(&original.to_json().unwrap()).unwrap();
        for change in ["ids", "normalizer", "decoder"] {
            let mut changed = value.clone();
            match change {
                "ids" => {
                    changed["model"]["vocab"]["hello"] = 2.into();
                    changed["model"]["vocab"]["world"] = 1.into();
                }
                "normalizer" => changed["normalizer"] = serde_json::json!({"type":"Lowercase"}),
                _ => {
                    changed["decoder"] =
                        serde_json::json!({"type":"WordPiece", "prefix":"##", "cleanup":true})
                }
            }
            fs::write(&saved, serde_json::to_vec(&changed).unwrap()).unwrap();
            let changed = Tokens::new(&saved).unwrap();
            assert_eq!(original.vocab_size(), changed.vocab_size());
            assert_ne!(
                original_identity,
                tokenizer_identity(&changed).unwrap(),
                "{change}"
            );
        }
    }

    #[test]
    fn same_size_tokenizer_tampering_is_rejected_before_loading_weights() {
        let temp = tempdir().unwrap();
        let path = saved_checkpoint(temp.path());
        let original = fs::read(path.join(TOKENIZER)).unwrap();
        // Invalid model bytes make ordering observable: tokenizer errors must be first.
        fs::write(path.join("model.mpk"), b"not a model").unwrap();
        for processing_only in [false, true] {
            let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
            if processing_only {
                value["normalizer"] = serde_json::json!({"type":"Lowercase"});
            } else {
                value["model"]["vocab"]["hello"] = 2.into();
                value["model"]["vocab"]["world"] = 1.into();
            }
            fs::write(path.join(TOKENIZER), serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(
                load_checkpoint(&path)
                    .err()
                    .unwrap()
                    .contains("tokenizer identity mismatch")
            );
        }
    }

    #[test]
    fn manifest_is_authoritative_for_every_configuration_field() {
        let temp = tempdir().unwrap();
        let path = saved_checkpoint(temp.path());
        let original = fs::read(path.join(CONFIG)).unwrap();
        fs::write(path.join("model.mpk"), b"not a model").unwrap();
        for field in [
            "vocab_size",
            "context_length",
            "d_model",
            "num_heads",
            "num_layers",
            "d_ff",
        ] {
            let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
            value[field] = (value[field].as_u64().unwrap() + 1).into();
            fs::write(path.join(CONFIG), serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(
                load_checkpoint(&path)
                    .err()
                    .unwrap()
                    .contains("authoritative manifest"),
                "{field}"
            );
        }
    }

    #[test]
    fn missing_manifest_fields_and_unknown_schema_are_rejected() {
        let temp = tempdir().unwrap();
        let path = saved_checkpoint(temp.path());
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(MANIFEST)).unwrap()).unwrap();
        for field in ["schema_version", "model", "tokenizer", "metadata"] {
            let mut value = original.clone();
            value.as_object_mut().unwrap().remove(field);
            fs::write(path.join(MANIFEST), serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(load_checkpoint(&path).is_err(), "missing {field}");
        }
        for field in ["dataset", "training", "build"] {
            let mut value = original.clone();
            value["metadata"].as_object_mut().unwrap().remove(field);
            fs::write(path.join(MANIFEST), serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(
                read_checkpoint_manifest(&path).is_err(),
                "missing nullable {field}"
            );
        }
        fs::write(path.join(MANIFEST), br#"{"schema_version":999}"#).unwrap();
        assert!(
            load_checkpoint(&path)
                .err()
                .unwrap()
                .contains("Unsupported checkpoint schema version 999")
        );
        fs::write(path.join(MANIFEST), b"{").unwrap();
        assert!(
            load_checkpoint(&path)
                .err()
                .unwrap()
                .contains("Invalid checkpoint manifest")
        );
        let mut unknown = original;
        unknown["unknown_field"] = true.into();
        fs::write(path.join(MANIFEST), serde_json::to_vec(&unknown).unwrap()).unwrap();
        assert!(load_checkpoint(&path).is_err());
    }

    #[test]
    fn versioned_checkpoints_reject_missing_files_and_partial_markers() {
        for file in [MANIFEST, CONFIG, TOKENIZER, "model.mpk", COMPLETE] {
            let temp = tempdir().unwrap();
            let path = saved_checkpoint(temp.path());
            fs::remove_file(path.join(file)).unwrap();
            assert!(load_checkpoint(&path).is_err(), "accepted missing {file}");
            if file == MANIFEST {
                assert!(
                    read_checkpoint_manifest(&path)
                        .unwrap_err()
                        .contains("required manifest")
                );
            }
        }
        let temp = tempdir().unwrap();
        let path = saved_checkpoint(temp.path());
        for marker in [
            &b"omega-checkpoint-"[..],
            &b"omega-checkpoint-v999\n"[..],
            &b""[..],
        ] {
            fs::write(path.join(COMPLETE), marker).unwrap();
            assert!(
                load_checkpoint(&path)
                    .err()
                    .unwrap()
                    .contains("completion marker")
            );
        }
    }

    #[test]
    fn legacy_empty_marker_checkpoints_remain_loadable() {
        let temp = tempdir().unwrap();
        let path = saved_checkpoint(temp.path());
        let (before, config, _) = load_checkpoint(&path).unwrap();
        let input = Tensor::<Cpu, 2, Int>::from_data([[0, 1, 2]], &Default::default());
        let expected = before
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // The model record/config/tokenizer format is unchanged. These two edits
        // reproduce the original unversioned checkpoint layout in temporary data.
        fs::remove_file(path.join(MANIFEST)).unwrap();
        fs::write(path.join(COMPLETE), b"").unwrap();
        assert!(read_checkpoint_manifest(&path).unwrap().is_none());
        let (loaded, loaded_config, _) = load_checkpoint(&path).unwrap();
        assert_eq!(loaded_config, config);
        assert_eq!(
            loaded.forward(input).into_data().to_vec::<f32>().unwrap(),
            expected
        );
    }

    #[test]
    fn invalid_provenance_is_rejected_before_reserving() {
        let temp = tempdir().unwrap();
        let tokenizer = tokenizer(temp.path());
        let root = temp.path().join("weights");
        let config = config();
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let invalid_training = CheckpointMetadata {
            training: Some(TrainingProvenance {
                epochs: 1,
                learning_rate: f64::NAN,
                seed: 42,
                optimizer: "adam".into(),
            }),
            ..Default::default()
        };
        let invalid_build = CheckpointMetadata {
            build: Some(BuildIdentity {
                cargo_lock_sha256: Some("invalid".into()),
                ..BuildIdentity::current()
            }),
            ..Default::default()
        };
        for metadata in [invalid_training, invalid_build] {
            assert!(
                save_checkpoint_with_metadata(
                    &root,
                    "run",
                    model.clone(),
                    &config,
                    &tokenizer,
                    &metadata
                )
                .is_err()
            );
            assert!(!root.exists());
        }
    }
}
