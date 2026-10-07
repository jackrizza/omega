use super::{PostTrainingConfig, WorkflowState};
use crate::jobs;
use anyhow::{Context, Result, ensure};
use omega_training::conversation_evaluation::EvaluationReport;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateReport {
    pub schema_version: u32,
    pub baseline: bool,
    pub completed_updates: usize,
    pub input_identity: String,
    pub conversation: EvaluationReport,
    pub assistant_loss: Option<f64>,
    pub base_loss: Option<f64>,
    pub assistant_targets: Option<usize>,
    pub base_targets: Option<usize>,
    pub evaluation_elapsed_seconds: f64,
    pub complete: bool,
    pub error: Option<String>,
}
/// Reports contain saved responses and may exceed the smaller job-control limit.
pub fn read_report(path: &Path) -> Result<CandidateReport> {
    let file = fs::File::open(path)?;
    ensure!(
        file.metadata()?.len() <= 64 * 1024 * 1024,
        "Evaluation report exceeds 64 MiB"
    );
    Ok(serde_json::from_reader(file)?)
}
impl CandidateReport {
    pub fn markdown(&self) -> String {
        format!(
            "# Post-training evaluation\n\nBaseline: {}\n\nUpdates: {}\n\nComplete: {}\n\nAssistant loss: {:?} (targets {:?})\n\nBase loss: {:?} (targets {:?})\n\nEvaluation elapsed: {:.3}s\n\nError: {}\n\n{}",
            self.baseline,
            self.completed_updates,
            self.complete,
            self.assistant_loss,
            self.assistant_targets,
            self.base_loss,
            self.base_targets,
            self.evaluation_elapsed_seconds,
            self.error.as_deref().unwrap_or("none"),
            self.conversation
                .to_markdown()
                .unwrap_or_else(|e| format!("Report export failed: {e}"))
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseScore {
    pub case_id: String,
    pub instruction_following: u8,
    pub correctness: u8,
    pub relevance: u8,
    pub coherence: u8,
    pub notes: String,
}
impl CaseScore {
    fn scores(&self) -> [u8; 4] {
        [
            self.instruction_following,
            self.correctness,
            self.relevance,
            self.coherence,
        ]
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HumanReview {
    pub schema_version: u32,
    pub report_sha256: String,
    pub rubric_version: String,
    pub reviewer: String,
    pub cases: Vec<CaseScore>,
}
impl HumanReview {
    fn validate(&self, report: &CandidateReport, hash: &str, rubric: &str) -> Result<()> {
        ensure!(
            self.schema_version == 1 && self.report_sha256 == hash && self.rubric_version == rubric,
            "Review must match the exact report hash and frozen rubric version"
        );
        ensure!(
            !self.reviewer.trim().is_empty() && self.reviewer.len() <= 200,
            "Name the human reviewer (1..200 bytes)"
        );
        let ids: BTreeSet<_> = report
            .conversation
            .suite
            .cases
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        let mut seen = BTreeSet::new();
        for c in &self.cases {
            ensure!(
                ids.contains(c.case_id.as_str()) && seen.insert(&c.case_id),
                "Unknown or duplicate reviewed case {}",
                c.case_id
            );
            ensure!(
                c.scores().into_iter().all(|v| v <= 4) && c.notes.len() <= 8192,
                "Human scores must be 0..4; notes at most 8192 bytes"
            );
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub report: PathBuf,
    pub checkpoint: PathBuf,
    pub assistant_loss: Option<f64>,
    pub completed_updates: usize,
    pub eligible: bool,
    pub reasons: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct SelectionManifest {
    pub schema_version: u32,
    pub checkpoint: PathBuf,
    pub model_sha256: String,
    pub checkpoint_manifest_sha256: String,
    pub report: PathBuf,
    pub report_sha256: String,
    pub review_sha256: String,
    pub baseline_report_sha256: String,
    pub approval_sha256: String,
    pub readiness_sha256: String,
    pub suite_sha256: String,
    pub rubric_version: String,
    pub approved_by: String,
    pub approved_ms: u64,
    pub final_acceptance: String,
}
fn review_path(report: &Path) -> PathBuf {
    report.with_extension("review.json")
}
pub fn save_review(workflow: &Path, report: &Path, review: &HumanReview) -> Result<PathBuf> {
    let state = WorkflowState::load(workflow)?;
    let report = included_report(&state, report)?;
    let data = read_report(&report)?;
    review.validate(
        &data,
        &super::workflow::hash_file(&report)?,
        &state.settings()?.rubric_version,
    )?;
    ensure!(
        review.cases.len() == data.conversation.suite.cases.len(),
        "Review is still a draft: score every case before submitting the immutable review"
    );
    let target = review_path(&report);
    // A submitted review is immutable; partial drafts should be edited before submission.
    ensure!(
        !target.exists(),
        "Review already exists; preserve it and create a new evaluation/report to revise scoring"
    );
    super::workflow::write_new_json(&target, review)?;
    Ok(target)
}
fn included_report(state: &WorkflowState, report: &Path) -> Result<PathBuf> {
    let report = fs::canonicalize(report)?;
    ensure!(
        state
            .reports
            .iter()
            .any(|p| fs::canonicalize(p).ok().as_ref() == Some(&report)),
        "Report is not a completed entry in this workflow"
    );
    Ok(report)
}
fn gates(
    report: &CandidateReport,
    baseline: &CandidateReport,
    p: &PostTrainingConfig,
) -> Vec<String> {
    let mut reasons = vec![];
    if report.baseline {
        reasons.push("Baseline is a comparator, not an SFT candidate".into());
    }
    if report.schema_version != 1
        || baseline.schema_version != 1
        || report.error.is_some()
        || baseline.error.is_some()
        || !report.complete
        || !report.conversation.is_complete()
        || !baseline.complete
        || !baseline.conversation.is_complete()
    {
        reasons.push("Evaluation incomplete or inconsistent".into());
    }
    if report.input_identity != baseline.input_identity
        || report
            .conversation
            .compare_compatible(&baseline.conversation)
            .is_err()
    {
        reasons.push("Incompatible evaluation inputs/settings/runtime".into());
    }
    if !report
        .assistant_loss
        .is_some_and(|x| x.is_finite() && x >= 0.0 && x <= p.max_assistant_loss)
    {
        reasons.push("Assistant loss gate failed or unavailable".into());
    }
    if !report
        .base_loss
        .zip(baseline.base_loss)
        .is_some_and(|(a, b)| {
            a.is_finite()
                && b.is_finite()
                && a >= 0.0
                && b >= 0.0
                && a <= b * (1.0 + p.max_base_loss_increase)
        })
    {
        reasons.push("Base-language regression gate failed or unavailable".into());
    }
    if !report
        .conversation
        .pass_rate()
        .is_some_and(|r| r >= p.min_test_pass_rate)
    {
        reasons.push("Conversation check gate failed or unavailable".into());
    }
    if report
        .conversation
        .cases
        .iter()
        .flat_map(|c| &c.turns)
        .any(|t| t.role_leakage || t.error.is_some())
    {
        reasons.push("Required chat-protocol gate failed".into());
    }
    reasons
}
pub fn rank(workflow: &Path) -> Result<Vec<RankedCandidate>> {
    let state = WorkflowState::load(workflow)?;
    let p = state.settings()?;
    let baseline_path = state
        .reports
        .first()
        .context("Baseline evaluation has not completed")?;
    let baseline = read_report(baseline_path)?;
    ensure!(
        baseline.baseline,
        "First workflow report is not the frozen baseline"
    );
    let mut candidates = vec![];
    for path in &state.reports {
        let r = read_report(path)?;
        let mut reasons = gates(&r, &baseline, p);
        let verified = omega_training::resume::read_stage_parent(&r.conversation.checkpoint.path)
            .map_err(anyhow::Error::msg)
            .and_then(|meta| {
                ensure!(
                    meta.model_sha256 == r.conversation.checkpoint.model_sha256
                        && super::workflow::hash_file(
                            &r.conversation.checkpoint.path.join("manifest.json")
                        )? == r.conversation.checkpoint.manifest_sha256,
                    "Candidate checkpoint changed since evaluation"
                );
                Ok(())
            });
        if let Err(e) = verified {
            reasons.push(format!("Checkpoint unavailable/incompatible: {e}"));
        }
        match jobs::read_json::<HumanReview>(&review_path(path)) {
            Ok(review) => {
                if let Err(e) =
                    review.validate(&r, &super::workflow::hash_file(path)?, &p.rubric_version)
                {
                    reasons.push(format!("Invalid human review: {e}"));
                }
                if review.cases.len() != r.conversation.suite.cases.len() {
                    reasons.push("Human review pending for some cases".into());
                }
                if review
                    .cases
                    .iter()
                    .any(|c| c.scores().into_iter().any(|s| s < p.min_human_score))
                {
                    reasons.push("Human quality gate failed".into());
                }
            }
            Err(e) => reasons.push(format!("Human review pending/unavailable: {e}")),
        }
        candidates.push(RankedCandidate {
            report: path.clone(),
            checkpoint: r.conversation.checkpoint.path.clone(),
            assistant_loss: r.assistant_loss,
            completed_updates: r.completed_updates,
            eligible: reasons.is_empty(),
            reasons,
        });
    }
    candidates.sort_by(|a, b| {
        b.eligible
            .cmp(&a.eligible)
            .then_with(|| {
                a.assistant_loss
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&b.assistant_loss.unwrap_or(f64::INFINITY))
            })
            .then(a.completed_updates.cmp(&b.completed_updates))
            .then(a.report.cmp(&b.report))
    });
    Ok(candidates)
}
pub fn promote(
    workflow: &Path,
    report: &Path,
    approved_by: &str,
    output: &Path,
) -> Result<SelectionManifest> {
    ensure!(
        !approved_by.trim().is_empty() && approved_by.len() <= 200,
        "Explicit human approver is required"
    );
    let state = WorkflowState::load(workflow)?;
    let report = included_report(&state, report)?;
    let candidates = rank(workflow)?;
    ensure!(
        candidates
            .iter()
            .any(|c| fs::canonicalize(&c.report).ok().as_ref() == Some(&report) && c.eligible),
        "Selected candidate is failed/pending; inspect comparison and human review first"
    );
    let data = read_report(&report)?;
    let parent = omega_training::resume::read_stage_parent(&data.conversation.checkpoint.path)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        parent.model_sha256 == data.conversation.checkpoint.model_sha256
            && super::workflow::hash_file(
                &data.conversation.checkpoint.path.join("manifest.json")
            )? == data.conversation.checkpoint.manifest_sha256,
        "Selected checkpoint changed since evaluation"
    );
    let selection = SelectionManifest {
        schema_version: 1,
        checkpoint: data.conversation.checkpoint.path,
        model_sha256: data.conversation.checkpoint.model_sha256,
        checkpoint_manifest_sha256: data.conversation.checkpoint.manifest_sha256,
        report: report.clone(),
        report_sha256: super::workflow::hash_file(&report)?,
        review_sha256: super::workflow::hash_file(&review_path(&report))?,
        baseline_report_sha256: super::workflow::hash_file(
            state.reports.first().context("Missing baseline evidence")?,
        )?,
        approval_sha256: super::workflow::hash_file(&state.directory.join("approval.json"))?,
        readiness_sha256: state
            .settings()?
            .frozen
            .as_ref()
            .context("Missing frozen readiness")?
            .readiness_sha256
            .clone(),
        suite_sha256: data.conversation.suite_hash,
        rubric_version: state.settings()?.rubric_version.clone(),
        approved_by: approved_by.into(),
        approved_ms: jobs::now_ms(),
        final_acceptance: "Pending separate sealed-test evaluation".into(),
    };
    super::workflow::write_new_json(output, &selection)?;
    Ok(selection)
}
