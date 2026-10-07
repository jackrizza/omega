//! Sequential, budgeted orchestration. Recovery uses frozen inputs and explicit
//! checkpoint receipts, never directory ordering or an implicit latest checkpoint.
use super::{CandidateReport, FrozenInputs, PostTrainingConfig};
use crate::{
    ProjectConfig,
    config::absolute,
    jobs::{self, Action, JobSpec, Reporter},
    pipeline,
};
use anyhow::{Context, Result, ensure};
use omega_training::{
    conversation_evaluation::{EvaluationSuite, run_suite},
    operations as op,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Ready,
    Baseline,
    Training,
    Evaluation,
    Completed,
    Stopped,
    BudgetExhausted,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowState {
    pub schema_version: u32,
    pub spec: JobSpec,
    pub phase: Phase,
    pub directory: PathBuf,
    pub checkpoint: Option<PathBuf>,
    pub checkpoint_sha256: Option<String>,
    pub completed_updates: usize,
    /// Reserved before training; interrupted operations keep their reservation.
    pub charged_updates: usize,
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub active_since_ms: Option<u64>,
    pub pending_evaluation: bool,
    pub training_complete: bool,
    pub reports: Vec<PathBuf>,
    pub incomplete_reports: Vec<PathBuf>,
    pub report_hashes: BTreeMap<PathBuf, String>,
    pub message: String,
}
impl WorkflowState {
    pub fn load(directory: &Path) -> Result<Self> {
        let state: Self = jobs::read_json(&directory.join("workflow.json"))?;
        ensure!(
            state.schema_version == 1,
            "Unsupported post-training workflow schema"
        );
        ensure!(
            fs::canonicalize(directory)? == fs::canonicalize(&state.directory)?,
            "Workflow directory moved; restore its original explicit path before recovery"
        );
        state.spec.config.validate()?;
        let approved: JobSpec = jobs::read_json(&directory.join("approval.json"))?;
        ensure!(
            serde_json::to_value(&approved)? == serde_json::to_value(&state.spec)?,
            "Workflow configuration differs from its immutable approval snapshot"
        );
        ensure!(
            state.settings()?.frozen.is_some(),
            "Workflow has no approved frozen input identities"
        );
        let readiness: omega_training::readiness::ReadinessReport =
            jobs::read_json(&directory.join("readiness.json"))?;
        ensure!(
            omega_training::checkpoint::sha256_bytes(&serde_json::to_vec(&readiness)?)
                == state.settings()?.frozen.as_ref().unwrap().readiness_sha256,
            "Frozen readiness evidence changed"
        );
        ensure!(
            state.elapsed_seconds.is_finite() && state.elapsed_seconds >= 0.0,
            "Invalid elapsed budget record"
        );
        for path in state.reports.iter().chain(&state.incomplete_reports) {
            ensure!(
                state.report_hashes.get(path) == Some(&hash_file(path)?),
                "Saved evaluation report changed: {}",
                path.display()
            );
        }
        if let Some(path) = &state.checkpoint {
            omega_training::resume::read_stage_parent(path).map_err(anyhow::Error::msg)?;
            ensure!(
                state.checkpoint_sha256.as_ref() == Some(&hash_file(&path.join("resume.json"))?),
                "Saved checkpoint identity changed"
            );
        }
        Ok(state)
    }
    pub fn settings(&self) -> Result<&PostTrainingConfig> {
        self.spec
            .config
            .post_training
            .as_ref()
            .context("Missing post-training settings")
    }
    fn save(&self) -> Result<()> {
        jobs::atomic_json(&self.directory.join("workflow.json"), self)
    }
    fn begin(&mut self, phase: Phase) -> Result<()> {
        self.phase = phase;
        self.active_since_ms = Some(jobs::now_ms());
        self.save()
    }
    fn finish_operation(&mut self, elapsed: f64) {
        self.elapsed_seconds += elapsed;
        self.active_since_ms = None;
    }
}

pub fn readiness(
    c: &ProjectConfig,
    project: &Path,
) -> Result<omega_training::readiness::ReadinessReport> {
    c.validate()?;
    let p = c
        .post_training
        .as_ref()
        .context("Configure [post_training] in a schema-2 model.toml")?;
    omega_training::readiness::inspect(
        &absolute(project, &p.parent),
        &absolute(project, &c.paths.datasets),
        &p.training,
        &p.validation,
        &p.base_validation,
        &p.sealed,
    )
    .map_err(anyhow::Error::msg)
}
fn identities(
    c: &ProjectConfig,
    project: &Path,
) -> Result<(
    FrozenInputs,
    EvaluationSuite,
    omega_training::readiness::ReadinessReport,
)> {
    let p = c
        .post_training
        .as_ref()
        .context("Missing post-training settings")?;
    let r = readiness(c, project)?;
    ensure!(
        r.passed,
        "Post-training readiness failed: {}",
        r.errors.join("; ")
    );
    let suite = EvaluationSuite::load(absolute(project, &p.suite)).map_err(anyhow::Error::msg)?;
    // Suite files are development inputs. Sealed partition contents never enter
    // training or this evaluator, and are only structurally audited by readiness.
    let frozen = FrozenInputs {
        readiness_sha256: omega_training::checkpoint::sha256_bytes(&serde_json::to_vec(&r)?),
        suite_sha256: suite.fingerprint().map_err(anyhow::Error::msg)?,
    };
    Ok((frozen, suite, r))
}
pub fn prepare(c: &ProjectConfig, project: &Path) -> Result<(ProjectConfig, String)> {
    let (frozen, _, _) = identities(c, project)?;
    let mut approved = c.clone();
    let p = approved
        .post_training
        .as_mut()
        .context("Missing post-training settings")?;
    p.parent = fs::canonicalize(absolute(project, &p.parent))?;
    p.suite = fs::canonicalize(absolute(project, &p.suite))?;
    p.output = absolute(project, &p.output);
    ensure!(
        !p.output.exists(),
        "Output exists; use explicit workflow recovery or choose a new output directory"
    );
    // Compute identities again with canonical paths so the immutable snapshot
    // reproduces the reviewed input identity after launch.
    p.frozen = Some(frozen);
    let (frozen, _, _) = identities(&approved, project)?;
    approved.post_training.as_mut().unwrap().frozen = Some(frozen);
    Ok((
        approved.clone(),
        format!(
            "Post-training: baseline → bounded SFT segments → development evaluation → human review\nParent and tokenizer are preserved. No sealed-test execution or automatic promotion.\nResolved settings:\n{}",
            toml::to_string_pretty(&approved)?
        ),
    ))
}
pub fn recover_spec(directory: &Path) -> Result<JobSpec> {
    let s = WorkflowState::load(directory)?;
    ensure!(
        !matches!(s.phase, Phase::Completed | Phase::BudgetExhausted),
        "Workflow has finished or exhausted its approved budget; create a separately reviewed experiment"
    );
    ensure!(
        hash_file(&s.spec.executable)? == s.spec.executable_sha256,
        "Retained worker executable changed or is unavailable"
    );
    let (frozen, _, _) = identities(&s.spec.config, &s.spec.project)?;
    ensure!(
        serde_json::to_value(&frozen)?
            == serde_json::to_value(s.settings()?.frozen.as_ref().unwrap())?,
        "Frozen parent, data or suite changed; recovery rejected"
    );
    Ok(s.spec)
}
pub(crate) fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(crate) fn write_new_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    Ok(())
}
fn storage(path: &Path) -> Result<u64> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        !meta.file_type().is_symlink(),
        "Workflow storage cannot contain symlinks: {}",
        path.display()
    );
    if meta.is_file() {
        return Ok(meta.len());
    }
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        total = total
            .checked_add(storage(&entry?.path())?)
            .context("Storage count overflow")?;
    }
    Ok(total)
}
fn control(reporter: Arc<Reporter>, stop: Arc<AtomicBool>) -> op::OperationControl {
    op::OperationControl {
        stop,
        observer: Box::new(move |e| reporter.emit(&e.kind, e.data).map_err(|e| e.to_string())),
    }
}

fn verify_active(s: &WorkflowState) -> Result<()> {
    let report: omega_training::readiness::ReadinessReport =
        jobs::read_json(&s.directory.join("readiness.json"))?;
    ensure!(
        omega_training::checkpoint::sha256_bytes(&serde_json::to_vec(&report)?)
            == s.settings()?
                .frozen
                .as_ref()
                .context("Missing frozen inputs")?
                .readiness_sha256,
        "Frozen readiness evidence changed"
    );
    omega_training::readiness::verify_active_inputs(
        &report,
        &absolute(&s.spec.project, &s.spec.config.paths.datasets),
    )
    .map_err(anyhow::Error::msg)
}
fn metric(
    s: &WorkflowState,
    checkpoint: &Path,
    selections: Vec<String>,
    chat: bool,
    seconds: f64,
    reporter: Arc<Reporter>,
    stop: Arc<AtomicBool>,
) -> Result<(f64, usize)> {
    ensure!(
        !stop.load(Ordering::SeqCst) && seconds > 0.0,
        "Evaluation interrupted or time budget exhausted"
    );
    let (root, input) = pipeline::input(checkpoint)?;
    let result = Arc::new(Mutex::new(None));
    let capture = result.clone();
    let ctl = op::OperationControl {
        stop,
        observer: Box::new(move |e| {
            if e.kind == "evaluation" {
                *capture.lock().map_err(|e| e.to_string())? = e.data["cross_entropy"].as_f64().zip(
                    e.data["targets"]
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok()),
                );
            }
            reporter.emit(&e.kind, e.data).map_err(|e| e.to_string())
        }),
    };
    let cmd = op::Command::Evaluate {
        datasets_root: Some(absolute(&s.spec.project, &s.spec.config.paths.datasets)),
        weights_root: Some(root),
        input,
        datasets: selections,
        dataset_format: if chat {
            omega_training::selection::Format::Chat
        } else {
            omega_training::selection::Format::Auto
        },
        validation_ratio: None,
        validation_count: None,
        split_seed: s.spec.config.dataset.split_seed,
    };
    op::execute_evaluation(pipeline::args(&s.spec.config, cmd, true), &ctl, seconds)
        .map_err(anyhow::Error::msg)?;
    let value = result
        .lock()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .context("Evaluation did not produce a complete metric")?;
    ensure!(
        value.0.is_finite() && value.0 >= 0.0 && value.1 > 0,
        "Invalid evaluation loss or empty targets"
    );
    Ok(value)
}
fn evaluate(
    s: &mut WorkflowState,
    checkpoint: &Path,
    baseline: bool,
    suite: &EvaluationSuite,
    reporter: Arc<Reporter>,
    stop: Arc<AtomicBool>,
) -> Result<bool> {
    let p = s.settings()?.clone();
    let seconds = p
        .evaluation_max_seconds
        .min(p.max_seconds - s.elapsed_seconds);
    ensure!(seconds > 0.0, "No evaluation time remains");
    s.evaluations += 1;
    s.begin(if baseline {
        Phase::Baseline
    } else {
        Phase::Evaluation
    })?;
    let start = Instant::now();
    reporter.emit(
        "stage",
        serde_json::json!({"name":if baseline {"SFT baseline"}else{"SFT evaluation"}}),
    )?;
    let backend = match s.spec.config.training.backend {
        omega_benchmark::Backend::Cpu => op::BackendChoice::Cpu,
        omega_benchmark::Backend::Vulkan => op::BackendChoice::Vulkan,
        omega_benchmark::Backend::Cuda => op::BackendChoice::Cuda,
    };
    let conversation = run_suite(
        checkpoint,
        suite,
        backend,
        s.spec.config.training.device,
        seconds,
        &control(reporter.clone(), stop.clone()),
    )
    .map_err(anyhow::Error::msg)?;
    let mut report = CandidateReport {
        schema_version: 1,
        baseline,
        completed_updates: s.completed_updates,
        input_identity: p.frozen.as_ref().unwrap().readiness_sha256.clone(),
        conversation,
        assistant_loss: None,
        base_loss: None,
        assistant_targets: None,
        base_targets: None,
        evaluation_elapsed_seconds: 0.0,
        complete: false,
        error: None,
    };
    let metrics = (|| -> Result<()> {
        ensure!(
            report.conversation.is_complete(),
            "Conversational evaluation is incomplete: {:?}",
            report.conversation.status
        );
        let (loss, targets) = metric(
            s,
            checkpoint,
            p.validation.clone(),
            true,
            seconds - start.elapsed().as_secs_f64(),
            reporter.clone(),
            stop.clone(),
        )?;
        report.assistant_loss = Some(loss);
        report.assistant_targets = Some(targets);
        let (loss, targets) = metric(
            s,
            checkpoint,
            p.base_validation.clone(),
            false,
            seconds - start.elapsed().as_secs_f64(),
            reporter,
            stop.clone(),
        )?;
        report.base_loss = Some(loss);
        report.base_targets = Some(targets);
        verify_active(s)?;
        ensure!(
            !stop.load(Ordering::SeqCst) && start.elapsed().as_secs_f64() <= seconds,
            "Evaluation stopped or exceeded its safe-boundary time budget"
        );
        Ok(())
    })();
    report.complete = metrics.is_ok();
    report.error = metrics.err().map(|e| e.to_string());
    report.evaluation_elapsed_seconds = start.elapsed().as_secs_f64();
    let path = s
        .directory
        .join("reports")
        .join(format!("evaluation-{}.json", s.evaluations));
    let bytes = serde_json::to_vec_pretty(&report)?;
    ensure!(
        bytes.len() <= 64 * 1024 * 1024,
        "Evaluation report exceeds the 64 MiB report bound; reduce the development suite"
    );
    let mut markdown = report.markdown();
    ensure!(
        storage(&s.directory)?
            .saturating_add(bytes.len() as u64)
            .saturating_add(markdown.len() as u64)
            .saturating_add(65536)
            <= p.max_output_bytes,
        "Output storage budget cannot hold the evaluation report; evaluation remains incomplete"
    );
    // Recheck at the publication boundary after potentially expensive rendering
    // and storage scans. Completing the durable save is then a safe operation.
    if report.complete && (stop.load(Ordering::SeqCst) || start.elapsed().as_secs_f64() > seconds) {
        report.complete = false;
        report.error =
            Some("Evaluation stopped or exceeded its budget before report commit".into());
        markdown = report.markdown();
    }
    write_new_json(&path, &report)?;
    let mut md = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path.with_extension("md"))?;
    md.write_all(markdown.as_bytes())?;
    md.sync_all()?;
    s.report_hashes.insert(path.clone(), hash_file(&path)?);
    if report.complete {
        s.reports.push(path);
    } else {
        s.incomplete_reports.push(path);
    }
    s.finish_operation(start.elapsed().as_secs_f64());
    s.pending_evaluation = !report.complete;
    if !report.complete {
        s.message = report.error.clone().unwrap_or_default();
    }
    s.save()?;
    Ok(report.complete)
}

pub fn execute(spec: &JobSpec, reporter: Arc<Reporter>, stop: Arc<AtomicBool>) -> Result<()> {
    let setup = Instant::now();
    pipeline::preflight(spec.config.training.backend, spec.config.training.device)?;
    let (frozen, suite, readiness) = identities(&spec.config, &spec.project)?;
    let p = spec
        .config
        .post_training
        .as_ref()
        .context("Missing post-training settings")?
        .clone();
    ensure!(
        serde_json::to_value(&frozen)?
            == serde_json::to_value(
                p.frozen
                    .as_ref()
                    .context("Review and freeze inputs before launch")?
            )?,
        "Approved inputs changed before worker launch"
    );
    let mut s = if spec.action == Action::PostTrainRecover {
        let directory = spec.checkpoint.as_ref().context("Missing workflow path")?;
        let original = recover_spec(directory)?;
        ensure!(
            original.executable_sha256 == spec.executable_sha256
                && serde_json::to_value(&original.config)? == serde_json::to_value(&spec.config)?,
            "Recovery must use the retained executable and unchanged approved configuration"
        );
        let mut s = WorkflowState::load(directory)?;
        // Include downtime conservatively after a crash. Never reset a budget.
        if let Some(start) = s.active_since_ms.take() {
            s.elapsed_seconds += jobs::now_ms().saturating_sub(start) as f64 / 1000.0;
        }
        s.message = "Explicit recovery from verified artifacts".into();
        s.save()?;
        s
    } else {
        let directory = absolute(&spec.project, &p.output);
        fs::create_dir(&directory)
            .context("Create new workflow output (existing directories are never replaced)")?;
        fs::create_dir(directory.join("reports"))?;
        fs::create_dir(directory.join("checkpoints"))?;
        ensure!(
            serde_json::to_vec_pretty(&readiness)?.len() <= 8 * 1024 * 1024,
            "Readiness metadata exceeds 8 MiB; reduce the number of source files"
        );
        write_new_json(&directory.join("readiness.json"), &readiness)?;
        write_new_json(&directory.join("approval.json"), spec)?;
        let s = WorkflowState {
            schema_version: 1,
            spec: spec.clone(),
            phase: Phase::Ready,
            directory,
            checkpoint: None,
            checkpoint_sha256: None,
            completed_updates: 0,
            charged_updates: 0,
            evaluations: 0,
            elapsed_seconds: 0.0,
            active_since_ms: None,
            pending_evaluation: false,
            training_complete: false,
            reports: vec![],
            incomplete_reports: vec![],
            report_hashes: BTreeMap::new(),
            message: String::new(),
        };
        s.save()?;
        s
    };
    s.elapsed_seconds += setup.elapsed().as_secs_f64();
    let result = run_loop(&mut s, &p, &suite, reporter.clone(), stop.clone());
    if let Err(e) = &result {
        s.phase = Phase::Failed;
        s.message = e.to_string();
        s.save()?;
    }
    let output_bytes = storage(&s.directory)?;
    reporter.emit("result",serde_json::json!({"workflow":s.directory,"phase":s.phase,"updates":s.completed_updates,"charged_updates":s.charged_updates,"evaluations":s.evaluations,"elapsed_seconds":s.elapsed_seconds,"time_overrun_seconds":(s.elapsed_seconds-p.max_seconds).max(0.0),"output_bytes":output_bytes,"storage_overrun_bytes":output_bytes.saturating_sub(p.max_output_bytes),"message":s.message,"promotion":"Requires human review and explicit selection"}))?;
    result
}
fn run_loop(
    s: &mut WorkflowState,
    p: &PostTrainingConfig,
    suite: &EvaluationSuite,
    reporter: Arc<Reporter>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let invocation = Instant::now();
    let initial_elapsed = s.elapsed_seconds;
    loop {
        s.elapsed_seconds = initial_elapsed + invocation.elapsed().as_secs_f64();
        if stop.load(Ordering::SeqCst) {
            s.phase = Phase::Stopped;
            s.message = "User stopped the remaining workflow".into();
            break;
        }
        if s.elapsed_seconds >= p.max_seconds || storage(&s.directory)? >= p.max_output_bytes {
            s.phase = Phase::BudgetExhausted;
            s.message = "Elapsed-time or storage budget exhausted".into();
            break;
        }
        verify_active(s)?;
        s.elapsed_seconds = initial_elapsed + invocation.elapsed().as_secs_f64();
        if stop.load(Ordering::SeqCst) || s.elapsed_seconds >= p.max_seconds {
            continue;
        }
        if s.reports.is_empty() || s.pending_evaluation {
            if s.evaluations >= p.max_evaluations {
                s.phase = Phase::BudgetExhausted;
                s.message = "Evaluation count budget exhausted".into();
                break;
            }
            let baseline = s.reports.is_empty();
            let checkpoint = if baseline {
                p.parent.clone()
            } else {
                s.checkpoint
                    .clone()
                    .context("No verified candidate checkpoint")?
            };
            if !evaluate(
                s,
                &checkpoint,
                baseline,
                suite,
                reporter.clone(),
                stop.clone(),
            )? {
                s.phase = if stop.load(Ordering::SeqCst) {
                    Phase::Stopped
                } else {
                    Phase::Failed
                };
                break;
            }
            continue;
        }
        ensure!(
            s.reports
                .iter()
                .all(|path| super::read_report(path).is_ok_and(|r| r.complete)),
            "An incomplete evaluation needs review; create a new experiment instead of treating it as passing"
        );
        if s.training_complete {
            s.phase = Phase::Completed;
            s.message =
                "Training epoch target and evaluation completed; human acceptance is pending"
                    .into();
            break;
        }
        if s.charged_updates >= p.max_updates || s.evaluations >= p.max_evaluations {
            s.phase = Phase::BudgetExhausted;
            s.message = "Approved update or evaluation count limit reached".into();
            break;
        }
        // Reserve enough space for model+optimizer output and metadata. This is
        // a conservative admission estimate, followed by measured storage checks;
        // filesystem failures still fail the job and never delete older files.
        let source = s.checkpoint.clone().unwrap_or_else(|| p.parent.clone());
        let headroom = storage(&source)?
            .saturating_mul(3)
            .saturating_add(1024 * 1024);
        if storage(&s.directory)?.saturating_add(headroom) > p.max_output_bytes {
            s.phase = Phase::BudgetExhausted;
            s.message =
                "Insufficient approved storage headroom for another verified checkpoint".into();
            break;
        }
        let updates = p.segment_updates.min(p.max_updates - s.charged_updates);
        let before = s.completed_updates;
        let charged_before = s.charged_updates;
        s.charged_updates += updates;
        s.begin(Phase::Training)?;
        reporter.emit("stage", serde_json::json!({"name":"Fine-tuning segment"}))?;
        let (weights_root, input) = pipeline::input(&source)?;
        let saves = op::SaveOptions {
            save_every_updates: None,
            save_every_epochs: None,
        };
        let command = if s.checkpoint.is_some() {
            op::Command::Resume {
                datasets_root: Some(absolute(&s.spec.project, &s.spec.config.paths.datasets)),
                weights_root: Some(weights_root),
                input,
                name: format!("{}-sft", s.spec.config.name),
                epochs: p.epochs,
                max_updates: Some(updates),
                saves,
                cache: None,
                limits: pipeline::limits(),
                metrics_jsonl: None,
                quiet: true,
            }
        } else {
            op::Command::TrainStage {
                datasets_root: Some(absolute(&s.spec.project, &s.spec.config.paths.datasets)),
                weights_root: Some(weights_root),
                input,
                name: format!("{}-sft", s.spec.config.name),
                datasets: p.training.clone(),
                epochs: p.epochs,
                learning_rate: p.learning_rate,
                seed: p.seed,
                max_updates: Some(updates),
                metrics_jsonl: None,
                quiet: true,
                saves,
                session_options: Box::new(op::SessionFlags {
                    shuffle: false,
                    dataset_weights: vec![],
                    samples_per_epoch: None,
                    batch_size: p.batch_size,
                    max_batch_tokens: p.max_batch_tokens,
                    gradient_clip_norm: None,
                    warmup_updates: 0,
                }),
            }
        };
        // Persist the verified save receipt immediately, even if the worker dies
        // before execute_segment_to returns to the sequencer.
        let receipt = Arc::new(Mutex::new(s.clone()));
        let capture = receipt.clone();
        let observe = reporter.clone();
        let ctl = op::OperationControl {
            stop: stop.clone(),
            observer: Box::new(move |event| {
                if event.kind == "checkpoint" {
                    let path = event.data["path"]
                        .as_str()
                        .ok_or("Checkpoint event lacks path")?;
                    let meta = omega_training::resume::read_stage_parent(Path::new(path))?;
                    let mut state = capture.lock().map_err(|e| e.to_string())?;
                    state.checkpoint = Some(path.into());
                    state.checkpoint_sha256 = Some(
                        hash_file(&Path::new(path).join("resume.json"))
                            .map_err(|e| e.to_string())?,
                    );
                    state.completed_updates = meta.progress.completed_updates;
                    state.pending_evaluation = true;
                    state.training_complete = meta.progress.completed_epochs
                        >= state.settings().map_err(|e| e.to_string())?.epochs;
                    state.save().map_err(|e| e.to_string())?;
                }
                observe
                    .emit(&event.kind, event.data)
                    .map_err(|e| e.to_string())
            }),
        };
        let start = Instant::now();
        let result = op::execute_segment_to(
            pipeline::args(&s.spec.config, command, true),
            &ctl,
            p.max_seconds - s.elapsed_seconds,
            &s.directory.join("checkpoints"),
        );
        *s = receipt.lock().map_err(|e| anyhow::anyhow!("{e}"))?.clone();
        s.finish_operation(start.elapsed().as_secs_f64());
        let outcome = result.map_err(anyhow::Error::msg)?;
        verify_active(s)?;
        s.checkpoint = Some(outcome.checkpoint.clone());
        s.checkpoint_sha256 = Some(hash_file(&outcome.checkpoint.join("resume.json"))?);
        s.completed_updates = outcome.completed_updates;
        s.charged_updates = charged_before + outcome.completed_updates.saturating_sub(before);
        s.pending_evaluation = true;
        s.training_complete = outcome.reason == op::OperationStopReason::Completed;
        s.save()?;
        if outcome.reason == op::OperationStopReason::UserStop {
            s.phase = Phase::Stopped;
            s.message = "User stopped after a verified checkpoint".into();
            break;
        }
        if outcome.reason == op::OperationStopReason::TimeBudget {
            s.phase = Phase::BudgetExhausted;
            s.message = format!(
                "Time budget reached; save overrun {:.3}s",
                outcome.budget_overrun_seconds
            );
            break;
        }
    }
    s.elapsed_seconds = initial_elapsed + invocation.elapsed().as_secs_f64();
    s.active_since_ms = None;
    s.save()?;
    ensure!(
        s.phase != Phase::Failed,
        "Workflow evaluation failed: {}",
        s.message
    );
    Ok(())
}
