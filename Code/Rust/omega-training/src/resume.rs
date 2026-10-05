//! Verified disk continuation for CPU and explicitly identified Vulkan training. Epoch ordering uses
//! a versioned portable sampler; no backend RNG state is restored or needed.
//! Exact continuation is supported on the same runtime/build, not across arbitrary
//! platforms, toolchains or future stochastic model operations such as dropout.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use omega_nn::GptConfig;
use omega_tokenizer::Tokens;
use serde::{Deserialize, Serialize};

use crate::dataset::{ALL_TARGETS_OBJECTIVE, ASSISTANT_TARGETS_OBJECTIVE};
use crate::{
    TrainingSession,
    checkpoint::{
        BuildIdentity, CheckpointMetadata, ModelConfig, TokenizerIdentity, load_checkpoint_header,
        read_checkpoint_manifest, save_checkpoint_with_artifacts, sha256_bytes, tokenizer_identity,
    },
    cpu::{CpuExecutionProfile, execution_profile},
    dataset::ExampleSource,
    gpu::GpuExecutionProfile,
    sampling::SAMPLING_ALGORITHM,
    trainer::{
        AdamRecord, RestoredTrainingState, SessionOptions, TrainingProgress, validate_progress,
    },
};

const RESUME: &str = "resume.json";
const RESUME_HASH: &str = "resume.sha256";
const OPTIMIZER: &str = "optimizer.mpk";
const MODEL: &str = "model.mpk";
const MODE: &str = "epoch-order-cpu-ndarray-f32-v2";
pub const GPU_RESUME_SCHEMA_VERSION: u32 = 4;
pub const GPU_EXECUTION_MODE: &str = "epoch-order-vulkan-wgpu-f32-v1";
// Reviewed change adds only optional GPU dependency edges/packages; existing CPU
// dependency versions, checksums and kernels are unchanged. Never applies to GPU.
const PRE_GPU_LOCK: &str = "04991746127d1994da4354c1d969ed436a799b8ef6ac9e549b473a43b4782696";
const GPU_COMPATIBLE_LOCK: &str =
    "55fc999ee31d21418f04f8f4006d69a849c27cb75da94e9c51adee98bc159201";
const RNG: &str = "unused-after-initialization";
pub const RESUME_SCHEMA_VERSION: u32 = 3;
pub const ASSISTANT_CPU_SCHEMA_VERSION: u32 = 5;
pub const ASSISTANT_GPU_SCHEMA_VERSION: u32 = 6;
pub const CUDA_EXECUTION_MODE: &str = "epoch-order-cuda-f32-v1";
const LEGACY_LOCK: &str = "ce271dcacbe04b5f92fb0336b17fea24c663a4008b7a4522caf76efcc344ac1e";

/// All optimizer settings are explicit and compared before loading state. This
/// schema supports only the trainer's current defaults; no override is allowed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentCheckpoint {
    pub name: String,
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub resume_sha256: String,
}

impl ParentCheckpoint {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err("Invalid parent checkpoint name".into());
        }
        for digest in [
            &self.manifest_sha256,
            &self.model_sha256,
            &self.resume_sha256,
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                return Err("Invalid parent checkpoint SHA256".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdamSettings {
    pub beta_1_bits: u32,
    pub beta_2_bits: u32,
    pub epsilon_bits: u32,
    pub weight_decay: bool,
    pub gradient_clipping: bool,
}

impl Default for AdamSettings {
    fn default() -> Self {
        Self {
            beta_1_bits: 0.9_f32.to_bits(),
            beta_2_bits: 0.999_f32.to_bits(),
            epsilon_bits: 1e-5_f32.to_bits(),
            weight_decay: false,
            gradient_clipping: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeRuntime {
    pub build: BuildIdentity,
    pub operating_system: String,
    pub architecture: String,
}

impl ResumeRuntime {
    fn current() -> Self {
        Self {
            build: BuildIdentity::current(),
            operating_system: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
        }
    }
}

/// Required state at a committed optimizer boundary. Floats affecting continuation
/// are stored as IEEE bits, avoiding JSON decimal-parser round-trip differences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeManifest {
    /// Present only in masked-source schemas 5/6. Old schemas retain all targets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<ParentCheckpoint>,
    pub schema_version: u32,
    pub execution_mode: String,
    pub rng_policy: String,
    pub runtime: ResumeRuntime,
    /// Frozen startup configuration, not an observation of active worker counts.
    pub cpu_execution: CpuExecutionProfile,
    /// Required for schema 4 GPU continuation; absent in legacy CPU schemas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_execution: Option<GpuExecutionProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_execution: Option<crate::cuda::CudaExecutionProfile>,
    pub model: ModelConfig,
    pub tokenizer: TokenizerIdentity,
    /// SHA256 over the exact ordered example lengths and IDs; paths and source
    /// metadata do not affect updates and are recorded separately as provenance.
    pub source_identity: String,
    pub example_count: usize,
    pub targets_per_epoch: usize,
    pub learning_rate_bits: u64,
    pub initialization_seed: u64,
    pub adam: AdamSettings,
    pub options: SessionOptions,
    pub sampling_algorithm: String,
    pub progress: TrainingProgress,
    pub epoch_weighted_loss_bits: u64,
    pub model_sha256: String,
    pub optimizer_sha256: String,
}

impl ResumeManifest {
    pub fn objective(&self) -> &str {
        self.objective.as_deref().unwrap_or(ALL_TARGETS_OBJECTIVE)
    }
    pub(crate) fn validate_objective(&self) -> Result<(), String> {
        match self.schema_version {
            1..=4 | 7 if self.objective.is_none() && self.parent.is_none() => Ok(()),
            5 | 6 | 8 if self.objective.as_deref() == Some(ASSISTANT_TARGETS_OBJECTIVE) => {
                if let Some(parent) = &self.parent {
                    parent.validate()?;
                }
                Ok(())
            }
            _ => Err("Unsupported resume objective/schema or stage lineage".into()),
        }
    }
    pub fn config(&self) -> GptConfig {
        self.model.clone().into()
    }
    pub fn learning_rate(&self) -> f64 {
        f64::from_bits(self.learning_rate_bits)
    }

    fn validate(&self) -> Result<(), String> {
        self.validate_objective()?;
        if matches!(self.schema_version, 7 | 8) {
            if self.execution_mode != CUDA_EXECUTION_MODE || self.gpu_execution.is_some() {
                return Err("Invalid CUDA resume backend identity".into());
            }
            self.cuda_execution
                .as_ref()
                .ok_or("CUDA resume requires execution profile")?
                .validate()?;
        } else {
            if self.cuda_execution.is_some() {
                return Err("Legacy resume cannot contain CUDA execution identity".into());
            }
            match (
                self.schema_version,
                self.execution_mode.as_str(),
                &self.gpu_execution,
            ) {
                (RESUME_SCHEMA_VERSION, MODE, None) => {}
                (GPU_RESUME_SCHEMA_VERSION, GPU_EXECUTION_MODE, Some(profile)) => {
                    profile.validate()?
                }
                (ASSISTANT_CPU_SCHEMA_VERSION, MODE, None) => {}
                (ASSISTANT_GPU_SCHEMA_VERSION, GPU_EXECUTION_MODE, Some(profile)) => {
                    profile.validate()?
                }
                _ => return Err("Unsupported resume schema or backend execution profile".into()),
            }
        }
        if self.rng_policy != RNG
            || self.adam != AdamSettings::default()
            || self.sampling_algorithm != SAMPLING_ALGORITHM
        {
            return Err("Unsupported resume execution, RNG or optimizer configuration".into());
        }
        if self.runtime != ResumeRuntime::current() {
            return Err("Resume runtime/build identity differs; continuation requires the same supported build and platform".into());
        }
        self.cpu_execution.validate()?;
        if self.gpu_execution.is_none()
            && self.cuda_execution.is_none()
            && self.cpu_execution != execution_profile()?
        {
            return Err("Resume CPU execution profile differs; use the original thread settings and supported kernel/build on the same host".into());
        }
        self.config().validate()?;
        self.options.validate(self.example_count)?;
        if !self.learning_rate().is_finite() || self.learning_rate() <= 0.0 {
            return Err("Resume learning rate must be finite and positive".into());
        }
        if self.example_count == 0 || self.targets_per_epoch == 0 {
            return Err("Resume source counts must be positive".into());
        }
        for (name, digest) in [
            ("source", &self.source_identity),
            ("model", &self.model_sha256),
            ("optimizer", &self.optimizer_sha256),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(format!(
                    "Resume {name} SHA256 must be 64 lowercase hexadecimal digits"
                ));
            }
        }
        Ok(())
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("Cannot create resume artifact {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|e| format!("Cannot write resume artifact {}: {e}", path.display()))
}

/// Save inference artifacts and optimizer/session state in one numbered directory.
/// All artifacts precede COMPLETE; existing checkpoints are never modified.
/// Source errors or invalid state reject before reserving a name. A later I/O
/// failure leaves an incomplete reservation, preserving numbering semantics.
pub fn save_training_checkpoint<S: ExampleSource>(
    weights_root: &Path,
    run_name: &str,
    session: &TrainingSession<S>,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
) -> Result<PathBuf, String> {
    save_training_checkpoint_with_parent(weights_root, run_name, session, tokenizer, metadata, None)
}

pub fn save_training_checkpoint_with_parent<S: ExampleSource>(
    weights_root: &Path,
    run_name: &str,
    session: &TrainingSession<S>,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
    parent: Option<&ParentCheckpoint>,
) -> Result<PathBuf, String> {
    save_training_checkpoint_impl(
        weights_root,
        run_name,
        session,
        tokenizer,
        metadata,
        Execution::Cpu,
        parent.cloned(),
    )
}

/// Save a Vulkan session with the exact selected adapter/runtime identity.
#[cfg(feature = "gpu")]
pub fn save_gpu_training_checkpoint<S: ExampleSource>(
    weights_root: &Path,
    run_name: &str,
    session: &TrainingSession<S, crate::gpu::GpuTraining>,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
    device: &crate::gpu::VulkanDevice,
) -> Result<PathBuf, String> {
    save_gpu_training_checkpoint_with_parent(
        weights_root,
        run_name,
        session,
        tokenizer,
        metadata,
        device,
        None,
    )
}

#[cfg(feature = "gpu")]
pub fn save_gpu_training_checkpoint_with_parent<S: ExampleSource>(
    weights_root: &Path,
    run_name: &str,
    session: &TrainingSession<S, crate::gpu::GpuTraining>,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
    device: &crate::gpu::VulkanDevice,
    parent: Option<&ParentCheckpoint>,
) -> Result<PathBuf, String> {
    if session.device() != &device.device {
        return Err("GPU checkpoint device differs from training session device".into());
    }
    device.profile.validate()?;
    save_training_checkpoint_impl(
        weights_root,
        run_name,
        session,
        tokenizer,
        metadata,
        Execution::Vulkan(device.profile.clone()),
        parent.cloned(),
    )
}

enum Execution {
    Cpu,
    #[cfg(feature = "gpu")]
    Vulkan(GpuExecutionProfile),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaExecutionProfile),
}

#[cfg(feature = "cuda")]
pub fn save_cuda_training_checkpoint<S: ExampleSource>(
    root: &Path,
    name: &str,
    session: &TrainingSession<S, crate::cuda::CudaTraining>,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
    device: &crate::cuda::CudaDevice,
    parent: Option<&ParentCheckpoint>,
) -> Result<PathBuf, String> {
    if session.device() != &device.device {
        return Err("CUDA session/device mismatch".into());
    }
    device.profile.validate()?;
    save_training_checkpoint_impl(
        root,
        name,
        session,
        tokenizer,
        metadata,
        Execution::Cuda(device.profile.clone()),
        parent.cloned(),
    )
}

fn save_training_checkpoint_impl<S: ExampleSource, B: AutodiffBackend<FloatElem = f32>>(
    weights_root: &Path,
    run_name: &str,
    session: &TrainingSession<S, B>,
    tokenizer: &Tokens,
    metadata: &CheckpointMetadata,
    execution: Execution,
    parent: Option<ParentCheckpoint>,
) -> Result<PathBuf, String> {
    let (gpu_execution, cuda_execution) = match execution {
        Execution::Cpu => (None, None),
        #[cfg(feature = "gpu")]
        Execution::Vulkan(profile) => (Some(profile), None),
        #[cfg(feature = "cuda")]
        Execution::Cuda(profile) => (None, Some(profile)),
    };
    session.validate_state()?;
    if let Some(training) = &metadata.training
        && (training.seed != session.seed()
            || training.learning_rate.to_bits() != session.learning_rate().to_bits())
    {
        return Err("Resume training provenance does not describe the session settings".into());
    }
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let optimizer = <_ as Recorder<B>>::record(&recorder, session.optimizer_record(), ())
        .map_err(|e| format!("Cannot encode resume optimizer: {e}"))?;
    let masked = session.training_set().objective() == ASSISTANT_TARGETS_OBJECTIVE;
    let mut manifest = ResumeManifest {
        objective: masked.then(|| ASSISTANT_TARGETS_OBJECTIVE.into()),
        parent,
        schema_version: if cuda_execution.is_some() {
            if masked { 8 } else { 7 }
        } else {
            match (masked, gpu_execution.is_some()) {
                (false, false) => RESUME_SCHEMA_VERSION,
                (false, true) => GPU_RESUME_SCHEMA_VERSION,
                (true, false) => ASSISTANT_CPU_SCHEMA_VERSION,
                (true, true) => ASSISTANT_GPU_SCHEMA_VERSION,
            }
        },
        execution_mode: if cuda_execution.is_some() {
            CUDA_EXECUTION_MODE
        } else if gpu_execution.is_some() {
            GPU_EXECUTION_MODE
        } else {
            MODE
        }
        .into(),
        rng_policy: RNG.into(),
        runtime: ResumeRuntime::current(),
        cpu_execution: execution_profile()?,
        gpu_execution,
        cuda_execution,
        model: ModelConfig::from(session.config()),
        tokenizer: tokenizer_identity(tokenizer)?,
        source_identity: session.training_set().identity()?,
        example_count: session.training_set().example_count(),
        targets_per_epoch: session.training_set().target_count()?,
        learning_rate_bits: session.learning_rate().to_bits(),
        initialization_seed: session.seed(),
        adam: AdamSettings::default(),
        options: session.options().clone(),
        sampling_algorithm: SAMPLING_ALGORITHM.into(),
        progress: session.progress(),
        epoch_weighted_loss_bits: session.epoch_weighted_loss().to_bits(),
        model_sha256: "0".repeat(64),
        optimizer_sha256: sha256_bytes(&optimizer),
    };
    manifest.validate()?;
    save_checkpoint_with_artifacts(
        weights_root,
        run_name,
        session.inference_model(),
        session.config(),
        tokenizer,
        metadata,
        |path| {
            let model = fs::read(path.join(MODEL))
                .map_err(|e| format!("Cannot fingerprint saved resume model: {e}"))?;
            manifest.model_sha256 = sha256_bytes(&model);
            let json = serde_json::to_vec_pretty(&manifest)
                .map_err(|e| format!("Cannot serialize resume manifest: {e}"))?;
            write_new(&path.join(OPTIMIZER), &optimizer)?;
            write_new(&path.join(RESUME), &json)?;
            write_new(&path.join(RESUME_HASH), sha256_bytes(&json).as_bytes())
        },
    )
}

/// Read and validate resume metadata on a completed checkpoint. Inference-only
/// and legacy checkpoints are explicitly rejected; weights alone are not resume.
pub fn read_resume_manifest(path: &Path) -> Result<ResumeManifest, String> {
    if read_checkpoint_manifest(path)?.is_none() {
        return Err("Legacy inference checkpoint has no training state and cannot resume".into());
    }
    let bytes = fs::read(path.join(RESUME)).map_err(|e| {
        format!("Checkpoint is inference-only or missing resume.json; cannot resume: {e}")
    })?;
    let digest = fs::read(path.join(RESUME_HASH))
        .map_err(|e| format!("Cannot read required resume manifest checksum: {e}"))?;
    if digest != sha256_bytes(&bytes).as_bytes() {
        return Err("Resume manifest SHA256 mismatch; checkpoint is corrupt".into());
    }
    let mut value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid resume manifest: {e}"))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or("Resume manifest requires integer schema_version")?;
    if version <= 6 && value.get("cuda_execution").is_some() {
        return Err("Legacy resume cannot contain CUDA execution settings".into());
    }
    if matches!(version, 1..=3 | 5 | 7 | 8) && value.get("gpu_execution").is_some() {
        return Err("CPU resume schemas cannot contain GPU execution settings".into());
    }
    if matches!(version, 1..=3)
        && value["runtime"]["build"]["cargo_lock_sha256"] == PRE_GPU_LOCK
        && BuildIdentity::current().cargo_lock_sha256.as_deref() == Some(GPU_COMPATIBLE_LOCK)
    {
        value["runtime"]["build"]["cargo_lock_sha256"] =
            serde_json::to_value(BuildIdentity::current().cargo_lock_sha256)
                .map_err(|e| e.to_string())?;
    }
    if matches!(version, 1 | 2) && value.get("cpu_execution").is_some() {
        return Err("Legacy resume schemas cannot contain schema 3 CPU execution settings".into());
    }
    if version == 1 {
        // Exact schema-1 migration: only defaults, and only the explicitly
        // reviewed predecessor lockfile on an otherwise identical runtime.
        if value.get("options").is_some() || value.get("sampling_algorithm").is_some() {
            return Err("Schema 1 cannot contain schema 2 session options".into());
        }
        if value["execution_mode"] != "fixed-order-cpu-ndarray-f32-v1" {
            return Err("Unsupported legacy resume execution mode".into());
        }
        let runtime: ResumeRuntime = serde_json::from_value(value["runtime"].clone())
            .map_err(|e| format!("Invalid legacy resume runtime: {e}"))?;
        let mut migrated_runtime = runtime.clone();
        let current = ResumeRuntime::current();
        if runtime.build.cargo_lock_sha256.as_deref() == Some(LEGACY_LOCK) {
            migrated_runtime.build.cargo_lock_sha256 = current.build.cargo_lock_sha256.clone();
        }
        if migrated_runtime != current {
            return Err(
                "Legacy resume runtime/build is not an allowlisted compatible predecessor".into(),
            );
        }
        value["runtime"] = serde_json::to_value(current).map_err(|e| e.to_string())?;
        value["schema_version"] = RESUME_SCHEMA_VERSION.into();
        value["execution_mode"] = MODE.into();
        let mut options = SessionOptions::default();
        if let Some(context) = value["model"]["context_length"]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
        {
            options.batching.max_batch_tokens = options.batching.max_batch_tokens.max(context);
        }
        value["options"] = serde_json::to_value(options).map_err(|e| e.to_string())?;
        value["sampling_algorithm"] = SAMPLING_ALGORITHM.into();
    } else if !matches!(version, 2..=8) {
        return Err(format!("Unsupported resume schema version {version}"));
    }
    if matches!(version, 1 | 2) {
        let profile = execution_profile()?;
        // Historical schemas never saved thread settings. Migration is supported
        // only under their documented default-environment assumption, never by
        // inventing the historical counts or accepting new explicit overrides.
        if !profile.is_legacy_default() {
            return Err("Legacy resume has no CPU execution profile; migration requires unset RAYON_NUM_THREADS, RAYON_RS_NUM_CPUS and MATMUL_NUM_THREADS with default kernels".into());
        }
        value["cpu_execution"] = serde_json::to_value(profile).map_err(|e| e.to_string())?;
        value["schema_version"] = RESUME_SCHEMA_VERSION.into();
    }
    let manifest: ResumeManifest =
        serde_json::from_value(value).map_err(|e| format!("Invalid resume manifest: {e}"))?;
    manifest.validate()?;
    Ok(manifest)
}

fn read_verified(path: &Path, name: &str, expected: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path.join(name)).map_err(|e| format!("Cannot read resume {name}: {e}"))?;
    if sha256_bytes(&bytes) != expected {
        return Err(format!(
            "Resume {name} SHA256 mismatch; checkpoint is corrupt"
        ));
    }
    Ok(bytes)
}

/// Restore the next update using an externally supplied exact example sequence.
/// Configuration/tokenizer/source mismatches are rejected before model loading;
/// no runtime learning-rate, seed, ordering or optimizer override is supported.
/// The caller may choose how many additional updates to execute. Cache sources
/// must validate stored data on each read; moving identical data is permitted.
/// Checksums detect corruption, not malicious edits with recomputed hashes.
pub fn load_training_checkpoint<S: ExampleSource>(
    path: &Path,
    source: S,
    expected_config: &GptConfig,
    expected_tokenizer: &Tokens,
) -> Result<TrainingSession<S>, String> {
    let manifest = read_resume_manifest(path)?;
    if manifest.gpu_execution.is_some() || manifest.cuda_execution.is_some() {
        return Err(
            "GPU checkpoint cannot resume on CPU; select its original Vulkan backend/device".into(),
        );
    }
    load_training_checkpoint_impl(
        path,
        source,
        expected_config,
        expected_tokenizer,
        &Default::default(),
        manifest,
    )
}

/// Continue a GPU checkpoint only on its original supported backend/runtime.
/// CPU weights can be loaded for inference separately; they are never a GPU resume.
#[cfg(feature = "gpu")]
pub fn load_gpu_training_checkpoint<S: ExampleSource>(
    path: &Path,
    source: S,
    expected_config: &GptConfig,
    expected_tokenizer: &Tokens,
    device: &crate::gpu::VulkanDevice,
) -> Result<TrainingSession<S, crate::gpu::GpuTraining>, String> {
    let manifest = read_resume_manifest(path)?;
    device.profile.validate()?;
    if manifest.gpu_execution.as_ref() != Some(&device.profile) {
        return Err("Resume GPU execution profile differs; CPU-to-GPU or cross-device/runtime continuation is unsupported".into());
    }
    device
        .adapter
        .validate_model(expected_config, manifest.options.batching.batch_size)?;
    load_training_checkpoint_impl(
        path,
        source,
        expected_config,
        expected_tokenizer,
        &device.device,
        manifest,
    )
}

#[cfg(feature = "cuda")]
pub fn load_cuda_training_checkpoint<S: ExampleSource>(
    path: &Path,
    source: S,
    config: &GptConfig,
    tokenizer: &Tokens,
    device: &crate::cuda::CudaDevice,
) -> Result<TrainingSession<S, crate::cuda::CudaTraining>, String> {
    let manifest = read_resume_manifest(path)?;
    if manifest.cuda_execution.as_ref() != Some(&device.profile) || manifest.gpu_execution.is_some()
    {
        return Err("CUDA resume execution profile differs; cross-backend/device/build continuation rejected".into());
    }
    load_training_checkpoint_impl(path, source, config, tokenizer, &device.device, manifest)
}

fn load_training_checkpoint_impl<S: ExampleSource, B: AutodiffBackend<FloatElem = f32>>(
    path: &Path,
    source: S,
    expected_config: &GptConfig,
    expected_tokenizer: &Tokens,
    device: &B::Device,
    manifest: ResumeManifest,
) -> Result<TrainingSession<S, B>, String> {
    let (config, saved_tokenizer, inference) = load_checkpoint_header(path)?;
    if &config != expected_config || config != manifest.config() {
        return Err("Resume model/configuration mismatch".into());
    }
    let expected_identity = tokenizer_identity(expected_tokenizer)?;
    if manifest.tokenizer != expected_identity
        || tokenizer_identity(&saved_tokenizer)? != expected_identity
    {
        return Err("Resume tokenizer identity mismatch".into());
    }
    if let Some(training) = inference.and_then(|m| m.metadata.training)
        && (training.seed != manifest.initialization_seed
            || training.learning_rate.to_bits() != manifest.learning_rate_bits)
    {
        return Err("Resume training settings disagree with checkpoint provenance".into());
    }
    if source.objective() != manifest.objective()
        || source.example_count() != manifest.example_count
        || source.target_count()? != manifest.targets_per_epoch
        || source.identity()? != manifest.source_identity
    {
        return Err("Resume example source identity/order mismatch".into());
    }
    let weighted_loss = f64::from_bits(manifest.epoch_weighted_loss_bits);
    validate_progress(
        &source,
        manifest.progress,
        weighted_loss,
        &manifest.options,
        manifest.initialization_seed,
    )?;
    let model_bytes = read_verified(path, MODEL, &manifest.model_sha256)?;
    let optimizer_bytes = read_verified(path, OPTIMIZER, &manifest.optimizer_sha256)?;
    // Burn can panic while decoding invalid parameter IDs or tensor records.
    // Convert such malformed-record failures to library errors, never an update.
    let decoded = catch_unwind(AssertUnwindSafe(|| -> Result<_, String> {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let model_record = <_ as Recorder<B>>::load(&recorder, model_bytes, device)
            .map_err(|e| format!("Cannot decode resume model: {e}"))?;
        let model = config.init::<B>(device)?.load_record(model_record);
        let optimizer: AdamRecord<B> = <_ as Recorder<B>>::load(&recorder, optimizer_bytes, device)
            .map_err(|e| format!("Cannot decode resume optimizer: {e}"))?;
        Ok((model, optimizer))
    }))
    .map_err(|_| {
        "Malformed resume model/optimizer record caused a backend decode failure".to_string()
    })??;
    TrainingSession::restore_on_device(
        source,
        RestoredTrainingState {
            model: decoded.0,
            optimizer: decoded.1,
            config,
            learning_rate: manifest.learning_rate(),
            seed: manifest.initialization_seed,
            options: manifest.options.clone(),
            progress: manifest.progress,
            epoch_weighted_loss: weighted_loss,
        },
        device,
    )
}

/// Start a new assistant stage from verified Omega training weights with fresh Adam.
/// The initial implementation requires a parent accepted by this runtime's normal
/// metadata reader; it never relaxes exact-resume checks or imports external weights.
/// Library callers must encode `source` with this parent's exact tokenizer and
/// protocol; generic ExampleSource implementations cannot attest their encoding.
/// The CLI constructs its conversation source from the validated parent tokenizer.
pub fn initialize_assistant_stage_on_device<
    S: ExampleSource,
    B: AutodiffBackend<FloatElem = f32>,
>(
    path: &Path,
    source: S,
    learning_rate: f64,
    seed: u64,
    options: SessionOptions,
    device: &B::Device,
) -> Result<(TrainingSession<S, B>, Tokens, ParentCheckpoint), String> {
    if source.objective() != ASSISTANT_TARGETS_OBJECTIVE {
        return Err("New assistant stage requires assistant-target examples".into());
    }
    let resume = read_resume_manifest(path)?;
    let (config, tokenizer, manifest) = load_checkpoint_header(path)?;
    let manifest = manifest.ok_or("New stage requires a versioned parent checkpoint")?;
    let chat = manifest
        .chat
        .as_ref()
        .ok_or("Parent has no frozen chat protocol; train a new base with a chat tokenizer")?;
    chat.validate_tokenizer(&tokenizer)?;
    if resume.config() != config || resume.tokenizer != manifest.tokenizer {
        return Err("Parent inference/training headers disagree".into());
    }
    let model_bytes = read_verified(path, MODEL, &resume.model_sha256)?;
    let parent = ParentCheckpoint {
        name: path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("Parent name must be UTF-8")?
            .into(),
        manifest_sha256: sha256_bytes(
            &fs::read(path.join("manifest.json")).map_err(|e| e.to_string())?,
        ),
        model_sha256: resume.model_sha256.clone(),
        resume_sha256: sha256_bytes(&fs::read(path.join(RESUME)).map_err(|e| e.to_string())?),
    };
    parent.validate()?;
    let model = catch_unwind(AssertUnwindSafe(|| -> Result<_, String> {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record = <_ as Recorder<B>>::load(&recorder, model_bytes, device)
            .map_err(|e| format!("Cannot decode parent weights: {e}"))?;
        Ok(config.init::<B>(device)?.load_record(record))
    }))
    .map_err(|_| "Parent weights caused a backend decoding failure".to_string())??;
    let session = TrainingSession::from_model_on_device(
        &config,
        source,
        model,
        learning_rate,
        seed,
        options,
        device,
    )?;
    Ok((session, tokenizer, parent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TrainingSet,
        checkpoint::{TrainingProvenance, save_checkpoint},
    };
    use burn::{
        optim::record::{AdaptorRecord, AdaptorRecordV1},
        record::Record,
        tensor::Tensor,
    };
    use omega_nn::{Cpu, TrainingBackend, token_tensor};

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 13,
            context_length: 3,
            d_model: 4,
            num_heads: 1,
            num_layers: 1,
            d_ff: 8,
        }
    }

    fn source() -> TrainingSet {
        TrainingSet {
            examples: vec![vec![1, 2], vec![2, 3, 4], vec![4, 5, 6, 7]],
            files: vec![],
            token_count: 9,
        }
    }

    fn tokenizer() -> Tokens {
        Tokens::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"))
            .unwrap()
    }

    fn session() -> TrainingSession {
        TrainingSession::new(&config(), source(), 0.003, 42).unwrap()
    }

    fn logits(session: &TrainingSession) -> Vec<f32> {
        session
            .inference_model()
            .forward(token_tensor::<Cpu>(&[1, 2, 3], 13, 3, &Default::default()).unwrap())
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    fn optimizer_value(session: &TrainingSession) -> serde_json::Value {
        serde_json::to_value(<AdamRecord as Record<TrainingBackend>>::into_item::<
            FullPrecisionSettings,
        >(session.optimizer_record()))
        .unwrap()
    }

    fn save(root: &Path, session: &TrainingSession) -> PathBuf {
        save_training_checkpoint(
            root,
            "run",
            session,
            &tokenizer(),
            &CheckpointMetadata::default(),
        )
        .unwrap()
    }

    fn rewrite_manifest(path: &Path, value: &serde_json::Value) {
        let bytes = serde_json::to_vec_pretty(value).unwrap();
        fs::write(path.join(RESUME), &bytes).unwrap();
        fs::write(path.join(RESUME_HASH), sha256_bytes(&bytes)).unwrap();
    }

    #[test]
    fn enabled_sampling_batching_clipping_and_warmup_resume_exactly() {
        use crate::{
            batching::BatchConfig,
            optimization::OptimizationOptions,
            sampling::{SamplingGroup, SamplingPolicy},
        };
        for sampling in [
            SamplingPolicy::Shuffle,
            SamplingPolicy::Weighted {
                groups: vec![
                    SamplingGroup {
                        name: "short".into(),
                        indices: vec![0],
                        weight: 2,
                    },
                    SamplingGroup {
                        name: "long".into(),
                        indices: vec![1, 2],
                        weight: 1,
                    },
                ],
                samples_per_epoch: 5,
            },
        ] {
            let options = SessionOptions {
                sampling,
                batching: BatchConfig {
                    batch_size: 2,
                    max_batch_tokens: 16,
                },
                optimization: OptimizationOptions {
                    gradient_clip_norm: Some(0.01),
                    warmup_updates: 5,
                },
            };
            let initial =
                TrainingSession::new_with_options(&config(), source(), 0.003, 42, options).unwrap();
            for boundary in [
                1,
                initial.updates_per_epoch(),
                initial.updates_per_epoch() + 1,
            ] {
                let temp = tempfile::tempdir().unwrap();
                let mut control = initial.clone();
                control.advance_updates(boundary, |_, _| Ok(())).unwrap();
                let path = save(temp.path(), &control);
                let mut restored =
                    load_training_checkpoint(&path, source(), &config(), &tokenizer()).unwrap();
                assert_eq!(control.options(), restored.options());
                for _ in 0..7 {
                    assert_eq!(control.step().unwrap(), restored.step().unwrap());
                }
                assert_eq!(control.progress(), restored.progress());
                assert_eq!(logits(&control), logits(&restored));
                assert_eq!(optimizer_value(&control), optimizer_value(&restored));
                let mut value: serde_json::Value =
                    serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
                value.as_object_mut().unwrap().remove("options");
                rewrite_manifest(&path, &value);
                assert!(read_resume_manifest(&path).is_err());
            }
        }
    }

    #[test]
    fn schema_one_defaults_migrate_only_from_compatible_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let mut control = session();
        control.step().unwrap();
        let path = save(temp.path(), &control);
        let mut legacy: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        legacy["schema_version"] = 1.into();
        legacy["execution_mode"] = "fixed-order-cpu-ndarray-f32-v1".into();
        legacy.as_object_mut().unwrap().remove("options");
        legacy.as_object_mut().unwrap().remove("sampling_algorithm");
        legacy.as_object_mut().unwrap().remove("cpu_execution");
        legacy["runtime"]["build"]["cargo_lock_sha256"] = LEGACY_LOCK.into();
        rewrite_manifest(&path, &legacy);
        let mut restored =
            load_training_checkpoint(&path, source(), &config(), &tokenizer()).unwrap();
        assert_eq!(restored.options(), &SessionOptions::default());
        for _ in 0..4 {
            assert_eq!(control.step().unwrap(), restored.step().unwrap());
        }
        assert_eq!(logits(&control), logits(&restored));
        assert_eq!(optimizer_value(&control), optimizer_value(&restored));
        legacy["runtime"]["build"]["cargo_lock_sha256"] = "a".repeat(64).into();
        rewrite_manifest(&path, &legacy);
        assert!(read_resume_manifest(&path).is_err());
        legacy["runtime"]["build"]["cargo_lock_sha256"] = LEGACY_LOCK.into();
        legacy["runtime"]["architecture"] = "other".into();
        rewrite_manifest(&path, &legacy);
        assert!(read_resume_manifest(&path).is_err());
    }

    #[test]
    fn pre_gpu_cpu_lock_migration_is_narrow_and_keeps_platform_checks() {
        let temp = tempfile::tempdir().unwrap();
        let path = save(temp.path(), &session());
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        for schema in [1, 2, 3] {
            let mut legacy = original.clone();
            legacy["schema_version"] = schema.into();
            legacy["runtime"]["build"]["cargo_lock_sha256"] = PRE_GPU_LOCK.into();
            if schema < 3 {
                legacy.as_object_mut().unwrap().remove("cpu_execution");
            }
            if schema == 1 {
                legacy["execution_mode"] = "fixed-order-cpu-ndarray-f32-v1".into();
                legacy.as_object_mut().unwrap().remove("options");
                legacy.as_object_mut().unwrap().remove("sampling_algorithm");
            }
            rewrite_manifest(&path, &legacy);
            let restored = load_training_checkpoint(&path, source(), &config(), &tokenizer());
            let current_lock = BuildIdentity::current().cargo_lock_sha256;
            if matches!(
                current_lock.as_deref(),
                Some(GPU_COMPATIBLE_LOCK | PRE_GPU_LOCK)
            ) {
                restored.unwrap();
            } else {
                // Unrelated workspace additions must not widen the reviewed migration.
                assert!(restored.is_err());
            }
            // Exercise the platform guard independently of lockfile rejection.
            legacy["runtime"]["build"]["cargo_lock_sha256"] =
                original["runtime"]["build"]["cargo_lock_sha256"].clone();
            legacy["runtime"]["operating_system"] = "other-os".into();
            rewrite_manifest(&path, &legacy);
            assert!(read_resume_manifest(&path).is_err());
            legacy["runtime"]["operating_system"] = original["runtime"]["operating_system"].clone();
            legacy["runtime"]["build"]["cargo_lock_sha256"] = "b".repeat(64).into();
            rewrite_manifest(&path, &legacy);
            assert!(read_resume_manifest(&path).is_err());
        }
        let mut invalid = original;
        invalid["gpu_execution"] = serde_json::Value::Null;
        rewrite_manifest(&path, &invalid);
        assert!(
            read_resume_manifest(&path)
                .unwrap_err()
                .contains("CPU resume schemas")
        );
    }

    #[test]
    fn cpu_profile_is_required_and_schema_two_defaults_migrate() {
        let temp = tempfile::tempdir().unwrap();
        let mut control = session();
        control.step().unwrap();
        let path = save(temp.path(), &control);
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        assert_eq!(original["schema_version"], 3);
        let mut changed = original.clone();
        changed["cpu_execution"]["rayon_num_threads"] = "99".into();
        rewrite_manifest(&path, &changed);
        assert!(
            read_resume_manifest(&path)
                .unwrap_err()
                .contains("CPU execution profile differs")
        );
        changed = original.clone();
        changed.as_object_mut().unwrap().remove("cpu_execution");
        rewrite_manifest(&path, &changed);
        assert!(read_resume_manifest(&path).is_err());
        changed = original.clone();
        changed["cpu_execution"]
            .as_object_mut()
            .unwrap()
            .remove("matmul_num_threads");
        rewrite_manifest(&path, &changed);
        assert!(read_resume_manifest(&path).is_err());
        changed = original;
        changed["schema_version"] = 2.into();
        rewrite_manifest(&path, &changed);
        assert!(
            read_resume_manifest(&path)
                .unwrap_err()
                .contains("Legacy resume schemas")
        );
        changed.as_object_mut().unwrap().remove("cpu_execution");
        rewrite_manifest(&path, &changed);
        let mut restored =
            load_training_checkpoint(&path, source(), &config(), &tokenizer()).unwrap();
        for _ in 0..4 {
            assert_eq!(control.step().unwrap(), restored.step().unwrap());
        }
        assert_eq!(logits(&control), logits(&restored));
        assert_eq!(optimizer_value(&control), optimizer_value(&restored));
    }

    #[test]
    fn disk_resume_preserves_parameters_adam_cursors_and_partial_epoch_losses() {
        let initial = session();
        for boundary in [0, 1, 3, 4] {
            let temp = tempfile::tempdir().unwrap();
            let mut continuous = initial.clone();
            continuous.advance_updates(boundary, |_, _| Ok(())).unwrap();
            let path = save(temp.path(), &continuous);
            let mut restored =
                load_training_checkpoint(&path, source(), &config(), &tokenizer()).unwrap();
            assert_eq!(continuous.progress(), restored.progress());
            assert_eq!(
                continuous.epoch_weighted_loss().to_bits(),
                restored.epoch_weighted_loss().to_bits()
            );
            assert_eq!(logits(&continuous), logits(&restored));
            assert_eq!(optimizer_value(&continuous), optimizer_value(&restored));
            // Counts cross another partial and full epoch after every save point.
            for _ in 0..7 {
                assert_eq!(continuous.step().unwrap(), restored.step().unwrap());
            }
            assert_eq!(continuous.progress(), restored.progress());
            assert_eq!(optimizer_value(&continuous), optimizer_value(&restored));
            assert_eq!(logits(&continuous), logits(&restored));
        }
    }

    #[test]
    fn learning_rate_bits_survive_difficult_decimal_provenance_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        let rate = f64::from_bits(0x3f68_9374_bc6a_7efa);
        let session = TrainingSession::new(&config(), source(), rate, 17).unwrap();
        let metadata = CheckpointMetadata {
            training: Some(TrainingProvenance {
                epochs: 1,
                learning_rate: rate,
                seed: 17,
                optimizer: "adam-default-burn-0.18".into(),
            }),
            ..Default::default()
        };
        let path = save_training_checkpoint(temp.path(), "rate", &session, &tokenizer(), &metadata)
            .unwrap();
        let restored = load_training_checkpoint(&path, source(), &config(), &tokenizer()).unwrap();
        assert_eq!(restored.learning_rate().to_bits(), rate.to_bits());
    }

    #[test]
    fn incompatible_examples_config_tokenizer_and_runtime_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = save(temp.path(), &session());
        let mut changed = source();
        changed.examples.swap(0, 1);
        assert!(
            load_training_checkpoint(&path, changed, &config(), &tokenizer())
                .err()
                .unwrap()
                .contains("identity/order")
        );
        let mut changed = source();
        changed.examples[0][1] = 12;
        assert!(load_training_checkpoint(&path, changed, &config(), &tokenizer()).is_err());
        let changed = GptConfig {
            context_length: 4,
            ..config()
        };
        assert!(
            load_training_checkpoint(&path, source(), &changed, &tokenizer())
                .err()
                .unwrap()
                .contains("configuration mismatch")
        );
        let mut value: serde_json::Value =
            serde_json::from_str(&tokenizer().to_json().unwrap()).unwrap();
        value["normalizer"] = serde_json::Value::Null;
        let token_path = temp.path().join("changed.json");
        fs::write(&token_path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            load_training_checkpoint(
                &path,
                source(),
                &config(),
                &Tokens::new(token_path).unwrap()
            )
            .err()
            .unwrap()
            .contains("tokenizer identity")
        );
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        value["runtime"]["architecture"] = "different-platform".into();
        rewrite_manifest(&path, &value);
        assert!(
            load_training_checkpoint(&path, source(), &config(), &tokenizer())
                .err()
                .unwrap()
                .contains("runtime/build")
        );
    }

    #[test]
    fn exact_example_inputs_allow_source_metadata_and_path_relocation() {
        let temp = tempfile::tempdir().unwrap();
        let path = save(temp.path(), &session());
        let mut relocated = source();
        relocated.files = vec![PathBuf::from("another-root/file.txt")];
        relocated.token_count = 999;
        assert!(load_training_checkpoint(&path, relocated, &config(), &tokenizer()).is_ok());
    }

    #[test]
    fn corrupt_or_missing_artifacts_and_inference_only_checkpoints_are_rejected() {
        for name in [RESUME, RESUME_HASH, MODEL, OPTIMIZER, "COMPLETE"] {
            let temp = tempfile::tempdir().unwrap();
            let path = save(temp.path(), &session());
            fs::remove_file(path.join(name)).unwrap();
            assert!(
                load_training_checkpoint(&path, source(), &config(), &tokenizer()).is_err(),
                "{name}"
            );
        }
        for name in [RESUME, RESUME_HASH, MODEL, OPTIMIZER] {
            let temp = tempfile::tempdir().unwrap();
            let path = save(temp.path(), &session());
            fs::write(path.join(name), b"corrupt").unwrap();
            assert!(
                load_training_checkpoint(&path, source(), &config(), &tokenizer()).is_err(),
                "{name}"
            );
            assert_eq!(fs::read(path.join(name)).unwrap(), b"corrupt");
        }
        let temp = tempfile::tempdir().unwrap();
        let path = save_checkpoint(
            temp.path(),
            "inference",
            session().inference_model(),
            &config(),
            &tokenizer(),
        )
        .unwrap();
        assert!(
            load_training_checkpoint(&path, source(), &config(), &tokenizer())
                .err()
                .unwrap()
                .contains("inference-only")
        );
        fs::remove_file(path.join("manifest.json")).unwrap();
        fs::write(path.join("COMPLETE"), b"").unwrap();
        assert!(
            load_training_checkpoint(&path, source(), &config(), &tokenizer())
                .err()
                .unwrap()
                .contains("Legacy inference")
        );
    }

    #[test]
    fn manifest_hash_and_structural_state_checks_reject_corruption() {
        let temp = tempfile::tempdir().unwrap();
        let mut session = session();
        session.step().unwrap();
        let path = save(temp.path(), &session);
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        let mut finite_change = original.clone();
        finite_change["epoch_weighted_loss_bits"] = 1.0_f64.to_bits().into();
        fs::write(
            path.join(RESUME),
            serde_json::to_vec(&finite_change).unwrap(),
        )
        .unwrap();
        assert!(
            load_training_checkpoint(&path, source(), &config(), &tokenizer())
                .err()
                .unwrap()
                .contains("manifest SHA256 mismatch")
        );
        for field in [
            "schema_version",
            "execution_mode",
            "rng_policy",
            "progress",
            "loss",
            "adam",
            "missing",
        ] {
            let mut value = original.clone();
            match field {
                "schema_version" => value[field] = 999.into(),
                "execution_mode" | "rng_policy" => value[field] = "unsupported".into(),
                "progress" => value[field]["completed_updates"] = 100.into(),
                "loss" => value["epoch_weighted_loss_bits"] = f64::NAN.to_bits().into(),
                "adam" => value[field]["weight_decay"] = true.into(),
                _ => {
                    value.as_object_mut().unwrap().remove("initialization_seed");
                }
            }
            rewrite_manifest(&path, &value);
            assert!(
                load_training_checkpoint(&path, source(), &config(), &tokenizer()).is_err(),
                "{field}"
            );
        }
        for artifact in [MODEL, OPTIMIZER] {
            let mut value = original.clone();
            fs::write(path.join(artifact), b"not messagepack").unwrap();
            value[if artifact == MODEL {
                "model_sha256"
            } else {
                "optimizer_sha256"
            }] = sha256_bytes(b"not messagepack").into();
            rewrite_manifest(&path, &value);
            assert!(load_training_checkpoint(&path, source(), &config(), &tokenizer()).is_err());
            // Next iteration needs the original model bytes.
            if artifact == MODEL {
                let restored_path = save(temp.path(), &session);
                fs::copy(restored_path.join(MODEL), path.join(MODEL)).unwrap();
            }
        }
    }

    #[test]
    fn optimizer_ids_shapes_ranks_moments_and_time_must_match_model() {
        let temp = tempfile::tempdir().unwrap();
        let mut session = session();
        session.step().unwrap();
        let path = save(temp.path(), &session);
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        for corruption in 0..7 {
            let mut record = session.optimizer_record();
            let id = *record
                .iter()
                .find(|(_, state)| matches!(state, AdaptorRecord::V1(AdaptorRecordV1::Rank1(_))))
                .unwrap()
                .0;
            match corruption {
                0 => {
                    record.remove(&id);
                }
                1 => {
                    record.insert(burn::module::ParamId::new(), record[&id].clone());
                }
                2 => {
                    let rank2 = record
                        .values()
                        .find(|state| matches!(state, AdaptorRecord::V1(AdaptorRecordV1::Rank2(_))))
                        .unwrap()
                        .clone();
                    record.insert(id, rank2);
                }
                _ => {
                    let AdaptorRecord::V1(AdaptorRecordV1::Rank1(state)) =
                        record.get_mut(&id).unwrap()
                    else {
                        unreachable!()
                    };
                    match corruption {
                        3 => state.momentum.time += 1,
                        4 => state.momentum.moment_1 = Tensor::zeros([99], &Default::default()),
                        5 => {
                            state.momentum.moment_1 =
                                state.momentum.moment_1.clone().mul_scalar(f32::NAN)
                        }
                        _ => {
                            state.momentum.moment_2 =
                                Tensor::ones(state.momentum.moment_2.dims(), &Default::default())
                                    .mul_scalar(-1.0)
                        }
                    }
                }
            }
            let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
            let bytes = <_ as Recorder<TrainingBackend>>::record(&recorder, record, ()).unwrap();
            fs::write(path.join(OPTIMIZER), &bytes).unwrap();
            let mut value = original.clone();
            value["optimizer_sha256"] = sha256_bytes(&bytes).into();
            rewrite_manifest(&path, &value);
            let error = load_training_checkpoint(&path, source(), &config(), &tokenizer())
                .err()
                .unwrap();
            assert!(error.contains("optimizer"), "{corruption}: {error}");
            assert_eq!(fs::read(path.join(OPTIMIZER)).unwrap(), bytes);
        }
    }

    #[test]
    fn malformed_optimizer_parameter_id_is_an_error_instead_of_unwinding() {
        let temp = tempfile::tempdir().unwrap();
        let mut session = session();
        session.step().unwrap();
        let path = save(temp.path(), &session);
        let mut item = <AdamRecord as Record<TrainingBackend>>::into_item::<FullPrecisionSettings>(
            session.optimizer_record(),
        );
        let key = item.keys().next().unwrap().clone();
        let state = item.remove(&key).unwrap();
        item.insert("!invalid!".into(), state);
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let container = burn::record::BurnRecord::<_, TrainingBackend>::new::<
            NamedMpkBytesRecorder<FullPrecisionSettings>,
        >(item);
        let bytes = <_ as Recorder<TrainingBackend>>::save_item(&recorder, container, ()).unwrap();
        fs::write(path.join(OPTIMIZER), &bytes).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(RESUME)).unwrap()).unwrap();
        value["optimizer_sha256"] = sha256_bytes(&bytes).into();
        rewrite_manifest(&path, &value);
        let error = load_training_checkpoint(&path, source(), &config(), &tokenizer())
            .err()
            .unwrap();
        assert!(error.contains("decode failure"), "{error}");
    }
}
