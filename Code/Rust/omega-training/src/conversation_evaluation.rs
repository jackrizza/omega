//! Versioned deterministic conversation checks, not a semantic quality judge.
//!
//! Generation settings and prompts are fixed by the suite. Stop/time limits are
//! cooperative at turn boundaries: model loading and one in-flight reply can
//! overrun the deadline. Their time and actual outputs remain in the report.
//! Reports never promote checkpoints, write model state, or read sealed data
//! implicitly. The caller decides which suite is appropriate for development.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    time::Instant,
};

use burn::tensor::backend::Backend;
use omega_nn::{GenerationOptions, SamplingOptions, generate_with_options_on_device};
use omega_tokenizer::chat::{ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    checkpoint::{
        BuildIdentity, CheckpointManifest, load_checkpoint_on_device, read_checkpoint_manifest,
        sha256_bytes,
    },
    operations::{BackendChoice, OperationControl},
};

pub const EVALUATION_SCHEMA_VERSION: u32 = 1;
const SUITE_BYTES: usize = 1024 * 1024;
const REPORT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationSuite {
    pub schema_version: u32,
    pub name: String,
    pub generation: GenerationSettings,
    pub cases: Vec<EvaluationCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSettings {
    pub max_new_tokens: usize,
    pub strategy: GenerationStrategy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GenerationStrategy {
    Greedy,
    Sample {
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: u64,
    },
}

impl GenerationSettings {
    fn options(&self) -> GenerationOptions {
        match self.strategy {
            GenerationStrategy::Greedy => GenerationOptions::Greedy,
            GenerationStrategy::Sample {
                temperature,
                top_k,
                top_p,
                seed,
            } => GenerationOptions::Sample(SamplingOptions {
                temperature,
                top_k,
                top_p,
                seed,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCase {
    pub id: String,
    pub system: Option<String>,
    pub turns: Vec<EvaluationTurn>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationTurn {
    pub prompt: String,
    pub checks: Vec<EvaluationCheck>,
}

/// Exact and contains checks are case-sensitive and never trim model output.
/// JSON fields match top-level object values exactly; additional fields are allowed.
/// Repetition counts overlapping whitespace-separated n-grams, case-sensitively.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvaluationCheck {
    Exact {
        expected: String,
    },
    Contains {
        text: String,
    },
    JsonFields {
        fields: BTreeMap<String, Value>,
    },
    EndTurn,
    NoRoleLeakage,
    NonEmpty,
    NoRepetition {
        ngram_size: usize,
        max_occurrences: usize,
    },
    ContextOverflow,
}

impl EvaluationSuite {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = read_bounded(path.as_ref(), SUITE_BYTES)?;
        let suite: Self =
            serde_json::from_slice(&bytes).map_err(|e| format!("Invalid evaluation suite: {e}"))?;
        suite.validate()?;
        Ok(suite)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EVALUATION_SCHEMA_VERSION {
            return Err("Unsupported evaluation suite schema_version".into());
        }
        if self.name.trim().is_empty() || self.name.len() > 256 {
            return Err("Evaluation suite name must contain 1..=256 bytes".into());
        }
        if self.cases.is_empty() || self.cases.len() > 128 {
            return Err("Evaluation suite requires 1..=128 cases".into());
        }
        if !(1..=4096).contains(&self.generation.max_new_tokens) {
            return Err("Evaluation max_new_tokens must be in 1..=4096".into());
        }
        self.generation.options().validate(usize::MAX)?;
        let mut ids = BTreeSet::new();
        let mut turns = 0usize;
        for case in &self.cases {
            if case.id.trim().is_empty() || case.id.len() > 256 || !ids.insert(&case.id) {
                return Err("Evaluation case IDs must be unique, nonblank and <=256 bytes".into());
            }
            if case.turns.is_empty() || case.turns.len() > 16 {
                return Err(format!("Case {} requires 1..=16 turns", case.id));
            }
            turns += case.turns.len();
            for (index, turn) in case.turns.iter().enumerate() {
                if turn.prompt.trim().is_empty() || turn.checks.is_empty() || turn.checks.len() > 32
                {
                    return Err(format!(
                        "Case {} requires nonblank prompts and 1..=32 checks per turn",
                        case.id
                    ));
                }
                if turn.checks.contains(&EvaluationCheck::ContextOverflow)
                    && (turn.checks.len() != 1 || index + 1 != case.turns.len())
                {
                    return Err(
                        "ContextOverflow must be the only check in the final turn of its case"
                            .into(),
                    );
                }
                for check in &turn.checks {
                    match check {
                        EvaluationCheck::Contains { text } if text.is_empty() => {
                            return Err("Contains requires nonempty text".into());
                        }
                        EvaluationCheck::JsonFields { fields } if fields.is_empty() => {
                            return Err("JsonFields requires at least one field".into());
                        }
                        EvaluationCheck::NoRepetition {
                            ngram_size,
                            max_occurrences,
                        } if !(1..=32).contains(ngram_size) || *max_occurrences == 0 => {
                            return Err("NoRepetition requires ngram_size in 1..=32 and positive max_occurrences".into());
                        }
                        _ => {}
                    }
                }
            }
        }
        if turns
            .checked_mul(self.generation.max_new_tokens)
            .is_none_or(|n| n > 32768)
        {
            return Err("Suite requests more than 32768 generated tokens".into());
        }
        if serde_json::to_vec(self).map_err(|e| e.to_string())?.len() > SUITE_BYTES {
            return Err("Evaluation suite exceeds 1 MiB serialized limit".into());
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        self.validate()?;
        Ok(sha256_bytes(
            &serde_json::to_vec(self).map_err(|e| e.to_string())?,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointIdentity {
    pub path: PathBuf,
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub manifest: CheckpointManifest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRuntime {
    pub build: BuildIdentity,
    pub operating_system: String,
    pub architecture: String,
    pub backend: String,
    pub device: usize,
    /// CPU execution profile or the selected Vulkan/CUDA execution profile.
    pub execution: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStatus {
    Complete,
    Interrupted,
    TimedOut,
    GenerationError,
    ObserverError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Passed,
    Failed,
    Incomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionReason {
    EndTurn,
    TokenLimit,
    ContextOverflow,
    RoleLeakage,
    GenerationError,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckResult {
    pub check: EvaluationCheck,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnResult {
    pub prompt: String,
    /// Actual history supplied to the formatter, including this turn's user prompt.
    pub history: Vec<RecordedMessage>,
    pub prompt_ids: Vec<u32>,
    pub reply: String,
    /// Decoded generated IDs with controls retained, for failure inspection.
    pub raw_reply: String,
    pub reply_ids: Vec<u32>,
    pub completion_reason: CompletionReason,
    pub ended_turn: bool,
    pub role_leakage: bool,
    pub elapsed_seconds: f64,
    pub checks: Vec<CheckResult>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub id: String,
    pub status: CaseStatus,
    pub turns: Vec<TurnResult>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub suite: EvaluationSuite,
    pub suite_hash: String,
    pub checkpoint: CheckpointIdentity,
    pub runtime: EvaluationRuntime,
    pub status: EvaluationStatus,
    pub cases: Vec<CaseResult>,
    pub elapsed_seconds: f64,
    pub max_seconds: f64,
    pub generated_tokens: usize,
    pub budget_overrun_seconds: f64,
    pub error: Option<String>,
}

impl EvaluationReport {
    pub fn is_complete(&self) -> bool {
        self.status == EvaluationStatus::Complete
            && self
                .cases
                .iter()
                .all(|c| c.status != CaseStatus::Incomplete)
    }
    pub fn passed_checks(&self) -> usize {
        self.cases
            .iter()
            .flat_map(|c| &c.turns)
            .flat_map(|t| &t.checks)
            .filter(|c| c.passed)
            .count()
    }
    /// Includes unexecuted checks, so partial reports cannot inflate pass rate.
    pub fn total_checks(&self) -> usize {
        self.suite
            .cases
            .iter()
            .flat_map(|c| &c.turns)
            .map(|t| t.checks.len())
            .sum()
    }
    pub fn pass_rate(&self) -> Option<f64> {
        self.is_complete()
            .then(|| self.passed_checks() as f64 / self.total_checks() as f64)
    }
    pub fn all_passed(&self) -> bool {
        self.is_complete() && self.cases.iter().all(|c| c.status == CaseStatus::Passed)
    }

    pub fn compare_compatible(&self, other: &Self) -> Result<(), String> {
        self.validate()?;
        other.validate()?;
        if !self.is_complete() || !other.is_complete() {
            return Err("Cannot compare incomplete conversation reports".into());
        }
        if self.suite_hash != other.suite_hash
            || self.checkpoint.manifest.model != other.checkpoint.manifest.model
            || self.checkpoint.manifest.tokenizer != other.checkpoint.manifest.tokenizer
            || self.checkpoint.manifest.chat != other.checkpoint.manifest.chat
            || self.runtime != other.runtime
        {
            return Err("Conversation comparison requires matching suite, tokenizer, model configuration, protocol and evaluation runtime".into());
        }
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let report: Self = serde_json::from_slice(&read_bounded(path.as_ref(), REPORT_BYTES)?)
            .map_err(|e| format!("Invalid evaluation report: {e}"))?;
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.suite.validate()?;
        if self.schema_version != EVALUATION_SCHEMA_VERSION
            || self.suite_hash != self.suite.fingerprint()?
        {
            return Err("Unsupported evaluation report schema or suite hash mismatch".into());
        }
        for digest in [
            &self.checkpoint.manifest_sha256,
            &self.checkpoint.model_sha256,
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("Invalid checkpoint digest in evaluation report".into());
            }
        }
        if !self.max_seconds.is_finite()
            || self.max_seconds <= 0.0
            || !self.elapsed_seconds.is_finite()
            || self.elapsed_seconds < 0.0
            || !self.budget_overrun_seconds.is_finite()
            || self.budget_overrun_seconds < 0.0
            || self.cases.len() != self.suite.cases.len()
        {
            return Err("Invalid evaluation report limits/counts/timings".into());
        }
        let mut tokens = 0usize;
        for (case, expected) in self.cases.iter().zip(&self.suite.cases) {
            if case.id != expected.id || case.turns.len() > expected.turns.len() {
                return Err("Evaluation case identity/count mismatch".into());
            }
            for (turn, expected) in case.turns.iter().zip(&expected.turns) {
                if turn.prompt != expected.prompt
                    || turn.checks.len() != expected.checks.len()
                    || turn
                        .checks
                        .iter()
                        .zip(&expected.checks)
                        .any(|(a, b)| &a.check != b)
                    || turn.checks
                        != check_response(
                            &expected.checks,
                            &turn.reply,
                            turn.completion_reason,
                            turn.ended_turn,
                            turn.role_leakage,
                        )
                    || !turn.elapsed_seconds.is_finite()
                    || turn.elapsed_seconds < 0.0
                    || turn.reply_ids.len() > self.suite.generation.max_new_tokens
                {
                    return Err("Evaluation turn/check result mismatch".into());
                }
                tokens = tokens
                    .checked_add(turn.reply_ids.len())
                    .ok_or("Report token count overflow")?;
            }
            if case.status == CaseStatus::Passed
                && (case.turns.len() != expected.turns.len()
                    || case.turns.iter().any(|t| {
                        t.role_leakage
                            || t.checks.iter().any(|c| !c.passed)
                            || t.completion_reason == CompletionReason::GenerationError
                    }))
            {
                return Err("Evaluation case incorrectly marked passed".into());
            }
            if self.status == EvaluationStatus::Complete && case.status == CaseStatus::Incomplete {
                return Err("Complete report contains incomplete case".into());
            }
        }
        if tokens != self.generated_tokens {
            return Err("Evaluation generated token count mismatch".into());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if json.len() > REPORT_BYTES {
            return Err(
                "Evaluation report exceeds 64 MiB export limit; use a smaller suite".into(),
            );
        }
        Ok(json)
    }

    pub fn to_markdown(&self) -> Result<String, String> {
        self.validate()?;
        // Indented verbatim blocks keep arbitrary model text from breaking fences.
        let mut text = format!(
            "# Conversation evaluation\n\nStatus: {:?}. Checks passed: {}/{}. Elapsed: {:.3} s; budget: {:.3} s; overrun: {:.3} s.\n\nSuite SHA256: `{}`\n\nModel SHA256: `{}`\n\nThese deterministic checks do not establish semantic correctness or human acceptance.\n",
            self.status,
            self.passed_checks(),
            self.total_checks(),
            self.elapsed_seconds,
            self.max_seconds,
            self.budget_overrun_seconds,
            self.suite_hash,
            self.checkpoint.model_sha256
        );
        text.push_str("\nCheckpoint, generation and execution identity:\n\n");
        text.push_str(&indented(
            &serde_json::to_string_pretty(&serde_json::json!({
                "checkpoint": self.checkpoint,
                "generation": self.suite.generation,
                "runtime": self.runtime,
            }))
            .map_err(|e| e.to_string())?,
        ));
        for case in &self.cases {
            text.push_str(&format!("\n## Case {:?} — {:?}\n", case.id, case.status));
            for turn in &case.turns {
                text.push_str("\nActual message history:\n\n");
                text.push_str(&indented(
                    &serde_json::to_string_pretty(&turn.history).map_err(|e| e.to_string())?,
                ));
                text.push_str(&format!("\nCompletion: {:?}; {:.3} s.\n\nPrompt:\n\n{}\nReply (controls retained):\n\n{}", turn.completion_reason, turn.elapsed_seconds, indented(&turn.prompt), indented(&turn.raw_reply)));
                for check in &turn.checks {
                    text.push_str(&format!(
                        "\n- {}: {:?}\n",
                        if check.passed { "PASS" } else { "FAIL" },
                        check.check
                    ));
                }
                if let Some(error) = &turn.error {
                    text.push_str(&format!("\nError:\n\n{}", indented(error)));
                }
            }
        }
        if let Some(error) = &self.error {
            text.push_str(&format!("\nRun error:\n\n{}", indented(error)));
        }
        Ok(text)
    }
}

fn indented(text: &str) -> String {
    text.split('\n')
        .map(|line| format!("    {line}\n"))
        .collect()
}

/// Pure checks over actual decoded output and generation observations. This can
/// also score retained fixture outputs without claiming they came from a model.
pub fn check_response(
    checks: &[EvaluationCheck],
    reply: &str,
    reason: CompletionReason,
    ended_turn: bool,
    role_leakage: bool,
) -> Vec<CheckResult> {
    checks
        .iter()
        .map(|check| {
            let generated = !matches!(
                reason,
                CompletionReason::ContextOverflow | CompletionReason::GenerationError
            );
            let passed = match check {
                EvaluationCheck::ContextOverflow => reason == CompletionReason::ContextOverflow,
                _ if !generated => false,
                EvaluationCheck::Exact { expected } => reply == expected,
                EvaluationCheck::Contains { text } => reply.contains(text),
                EvaluationCheck::JsonFields { fields } => serde_json::from_str::<Value>(reply)
                    .ok()
                    .and_then(|v| v.as_object().cloned())
                    .is_some_and(|object| {
                        fields
                            .iter()
                            .all(|(key, value)| object.get(key) == Some(value))
                    }),
                EvaluationCheck::EndTurn => ended_turn,
                EvaluationCheck::NoRoleLeakage => !role_leakage,
                EvaluationCheck::NonEmpty => !reply.trim().is_empty(),
                EvaluationCheck::NoRepetition {
                    ngram_size,
                    max_occurrences,
                } => {
                    if *ngram_size == 0 || *max_occurrences == 0 {
                        false
                    } else {
                        let words: Vec<_> = reply.split_whitespace().collect();
                        let mut counts = BTreeMap::new();
                        words.windows(*ngram_size).all(|ngram| {
                            let count = counts.entry(ngram).or_insert(0usize);
                            *count += 1;
                            *count <= *max_occurrences
                        })
                    }
                }
            };
            CheckResult {
                check: check.clone(),
                passed,
                detail: if passed {
                    "Check satisfied"
                } else {
                    "Check failed against retained output/completion"
                }
                .into(),
            }
        })
        .collect()
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(limit as u64 + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!("{} exceeds {limit} byte limit", path.display()));
    }
    Ok(bytes)
}

fn file_hash(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| format!("Cannot hash {}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn runtime(backend: &str, device: usize, execution: Value) -> EvaluationRuntime {
    EvaluationRuntime {
        build: BuildIdentity::current(),
        operating_system: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        backend: backend.into(),
        device,
        execution,
    }
}

/// Load one frozen checkpoint once and evaluate the fixed suite in-process.
/// Errors before a valid checkpoint/runtime report can be constructed return Err;
/// later generation/observer/stop failures return a retained partial report.
pub fn run_suite(
    checkpoint: &Path,
    suite: &EvaluationSuite,
    backend: BackendChoice,
    device: usize,
    max_seconds: f64,
    control: &OperationControl,
) -> Result<EvaluationReport, String> {
    suite.validate()?;
    if !max_seconds.is_finite() || max_seconds <= 0.0 {
        return Err("Evaluation max_seconds must be positive and finite".into());
    }
    let started = Instant::now();
    let checkpoint = checkpoint
        .canonicalize()
        .map_err(|e| format!("Cannot resolve evaluation checkpoint: {e}"))?;
    let manifest = read_checkpoint_manifest(&checkpoint)?
        .ok_or("Conversation evaluation requires a versioned checkpoint")?;
    if manifest.chat.is_none() {
        return Err("Conversation evaluation requires explicit chat protocol metadata".into());
    }
    let identity = CheckpointIdentity {
        manifest_sha256: file_hash(&checkpoint.join("manifest.json"))?,
        model_sha256: file_hash(&checkpoint.join("model.mpk"))?,
        path: checkpoint,
        manifest,
    };
    match backend {
        BackendChoice::Cpu => {
            if device != 0 {
                return Err("CPU evaluation device must be zero".into());
            }
            let profile = serde_json::to_value(crate::cpu::execution_profile()?)
                .map_err(|e| e.to_string())?;
            run_on_device::<omega_nn::Cpu>(
                identity,
                suite,
                runtime("cpu", 0, profile),
                &Default::default(),
                max_seconds,
                control,
                started,
            )
        }
        BackendChoice::Vulkan => {
            #[cfg(feature = "gpu")]
            {
                let selected = crate::gpu::initialize_vulkan(device)?;
                selected
                    .adapter()
                    .validate_model(&identity.manifest.model.clone().into(), 1)?;
                let profile =
                    serde_json::to_value(selected.profile()).map_err(|e| e.to_string())?;
                run_on_device::<crate::gpu::Gpu>(
                    identity,
                    suite,
                    runtime("vulkan", device, profile),
                    selected.device(),
                    max_seconds,
                    control,
                    started,
                )
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err("Vulkan conversation evaluation requires --features gpu".into())
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let selected = crate::cuda::initialize(device)?;
                let profile = serde_json::to_value(&selected.profile).map_err(|e| e.to_string())?;
                run_on_device::<burn::backend::Cuda<f32, i32>>(
                    identity,
                    suite,
                    runtime("cuda", device, profile),
                    &selected.device,
                    max_seconds,
                    control,
                    started,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA conversation evaluation requires --features cuda".into())
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_on_device<B: Backend<FloatElem = f32>>(
    identity: CheckpointIdentity,
    suite: &EvaluationSuite,
    runtime: EvaluationRuntime,
    device: &B::Device,
    max_seconds: f64,
    control: &OperationControl,
    started: Instant,
) -> Result<EvaluationReport, String> {
    let mut report = EvaluationReport {
        schema_version: EVALUATION_SCHEMA_VERSION,
        suite: suite.clone(),
        suite_hash: suite.fingerprint()?,
        checkpoint: identity,
        runtime,
        status: EvaluationStatus::Complete,
        cases: suite
            .cases
            .iter()
            .map(|case| CaseResult {
                id: case.id.clone(),
                status: CaseStatus::Incomplete,
                turns: Vec::new(),
                error: None,
            })
            .collect(),
        elapsed_seconds: 0.0,
        max_seconds,
        generated_tokens: 0,
        budget_overrun_seconds: 0.0,
        error: None,
    };
    if stopped(&mut report, control, started) {
        return finish(report, started);
    }
    let (model, config, tokenizer) =
        load_checkpoint_on_device::<B>(&report.checkpoint.path, device)?;
    let protocol = report
        .checkpoint
        .manifest
        .chat
        .as_ref()
        .expect("validated chat metadata")
        .validate_tokenizer(&tokenizer)?;
    let options = suite.generation.options();
    options.validate(config.vocab_size)?;
    if stopped(&mut report, control, started) {
        return finish(report, started);
    }
    for (case_index, case) in suite.cases.iter().enumerate() {
        if stopped(&mut report, control, started) {
            break;
        }
        let mut history: Vec<ChatMessage> = case
            .system
            .iter()
            .map(|content| ChatMessage {
                role: ChatRole::System,
                content: content.clone(),
            })
            .collect();
        let mut case_failed = false;
        for turn in &case.turns {
            if stopped(&mut report, control, started) {
                break;
            }
            let turn_started = Instant::now();
            history.push(ChatMessage {
                role: ChatRole::User,
                content: turn.prompt.clone(),
            });
            let mut result = TurnResult {
                prompt: turn.prompt.clone(),
                history: history
                    .iter()
                    .map(|m| RecordedMessage {
                        role: match m.role {
                            ChatRole::System => "system",
                            ChatRole::User => "user",
                            ChatRole::Assistant => "assistant",
                        }
                        .into(),
                        content: m.content.clone(),
                    })
                    .collect(),
                prompt_ids: Vec::new(),
                reply: String::new(),
                raw_reply: String::new(),
                reply_ids: Vec::new(),
                completion_reason: CompletionReason::GenerationError,
                ended_turn: false,
                role_leakage: false,
                elapsed_seconds: 0.0,
                checks: Vec::new(),
                error: None,
            };
            let attempt = (|| -> Result<(), String> {
                result.prompt_ids = protocol.encode_prompt(&tokenizer, &history)?;
                if result
                    .prompt_ids
                    .len()
                    .checked_add(suite.generation.max_new_tokens)
                    .is_none_or(|n| n > config.context_length)
                {
                    result.completion_reason = CompletionReason::ContextOverflow;
                    return Ok(());
                }
                let output = generate_with_options_on_device(
                    &model,
                    &config,
                    &result.prompt_ids,
                    suite.generation.max_new_tokens,
                    Some(protocol.token_ids().end_turn),
                    &options,
                    device,
                )?;
                result.reply_ids = output[result.prompt_ids.len()..].to_vec();
                let ids = protocol.token_ids();
                result.ended_turn = result.reply_ids.last() == Some(&ids.end_turn);
                result.role_leakage = result
                    .reply_ids
                    .iter()
                    .any(|id| [ids.system, ids.user, ids.assistant].contains(id));
                result.reply = tokenizer.decode(&result.reply_ids, true)?;
                result.raw_reply = tokenizer.decode(&result.reply_ids, false)?;
                result.completion_reason = if result.role_leakage {
                    CompletionReason::RoleLeakage
                } else if result.ended_turn {
                    CompletionReason::EndTurn
                } else {
                    CompletionReason::TokenLimit
                };
                Ok(())
            })();
            if let Err(error) = attempt {
                result.error = Some(error.clone());
                result.completion_reason = CompletionReason::GenerationError;
                report.status = EvaluationStatus::GenerationError;
                report.error = Some(error);
            }
            result.elapsed_seconds = turn_started.elapsed().as_secs_f64();
            result.checks = check_response(
                &turn.checks,
                &result.reply,
                result.completion_reason,
                result.ended_turn,
                result.role_leakage,
            );
            case_failed |= result.checks.iter().any(|check| !check.passed) || result.role_leakage;
            let terminal_case = matches!(
                result.completion_reason,
                CompletionReason::RoleLeakage
                    | CompletionReason::ContextOverflow
                    | CompletionReason::GenerationError
            );
            history.push(ChatMessage {
                role: ChatRole::Assistant,
                content: result.reply.clone(),
            });
            report.generated_tokens += result.reply_ids.len();
            report.cases[case_index].turns.push(result);
            if let Err(error) = control.emit("conversation_case_turn", serde_json::json!({"case_id": case.id,"completed_turns":report.cases[case_index].turns.len(),"generated_tokens":report.generated_tokens})) {
                report.status = EvaluationStatus::ObserverError;
                report.error = Some(error);
            }
            if report.status != EvaluationStatus::Complete || stopped(&mut report, control, started)
            {
                break;
            }
            if terminal_case {
                break;
            }
        }
        if report.status != EvaluationStatus::Complete {
            break;
        }
        let result = &mut report.cases[case_index];
        if result.turns.len() != case.turns.len() {
            case_failed = true;
            result.error = Some(
                "Remaining turns were not run after a rejected response or context overflow".into(),
            );
        }
        result.status = if case_failed {
            CaseStatus::Failed
        } else {
            CaseStatus::Passed
        };
    }
    finish(report, started)
}

fn stopped(report: &mut EvaluationReport, control: &OperationControl, started: Instant) -> bool {
    if control.stop.load(Ordering::Relaxed) {
        report.status = EvaluationStatus::Interrupted;
        report.error = Some("Evaluation interrupted at a cooperative turn boundary".into());
        return true;
    }
    if started.elapsed().as_secs_f64() >= report.max_seconds {
        report.status = EvaluationStatus::TimedOut;
        report.error =
            Some("Evaluation time budget exhausted at a cooperative turn boundary".into());
        return true;
    }
    false
}

fn finish(mut report: EvaluationReport, started: Instant) -> Result<EvaluationReport, String> {
    report.elapsed_seconds = started.elapsed().as_secs_f64();
    report.budget_overrun_seconds = (report.elapsed_seconds - report.max_seconds).max(0.0);
    report.validate()?;
    Ok(report)
}
