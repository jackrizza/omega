//! Terminal state and rendering never own a training session.
use crate::{
    ProjectConfig, config,
    jobs::{self, Action, JobRecord, JobSpec},
    pipeline, post_training,
};
use anyhow::{Context, Result, ensure};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
mod dashboard;
#[cfg(test)]
mod tests;
mod view;
use std::{
    fs,
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};

const ACTIONS: &[&str] = &[
    "Start reviewed pipeline",
    "Dataset plan / prepare",
    "Train tokenizer",
    "Prepare token cache",
    "Estimate training time",
    "Train base model",
    "Train assistant stage",
    "Resume selected checkpoint",
    "Evaluate selected checkpoint",
    "Generate from selected checkpoint",
    "Chat from selected checkpoint",
    "Checkpoint inventory",
    "Tokenizer inspect / encode / decode",
    "Configuration forms",
    "Edit model.toml",
    "Jobs / reconnect",
    "Doctor",
    "Inspect selected dataset releases",
    "Post-training: migrate project to schema 2",
    "Post-training: readiness",
    "Post-training: review and start",
    "Post-training: ranked reports and results",
    "Post-training: submit human review",
    "Post-training: promote candidate",
    "Post-training: recover workflow",
];
const FIELDS: &[(&str, &str)] = &[
    ("", "name"),
    ("paths", "datasets"),
    ("paths", "weights"),
    ("dataset", "selections"),
    ("dataset", "format"),
    ("tokenizer", "path"),
    ("tokenizer", "vocab_size"),
    ("tokenizer", "min_frequency"),
    ("tokenizer", "chat_protocol"),
    ("model", "context_length"),
    ("model", "d_model"),
    ("model", "heads"),
    ("model", "layers"),
    ("model", "d_ff"),
    ("training", "backend"),
    ("training", "device"),
    ("training", "epochs"),
    ("training", "learning_rate"),
    ("training", "batch_size"),
    ("training", "max_batch_tokens"),
    ("training", "seed"),
    ("training", "shuffle"),
    ("pipeline", "prepare_dataset"),
    ("pipeline", "prepare_tokenizer"),
    ("pipeline", "prepare_cache"),
    ("pipeline", "benchmark"),
    ("pipeline", "train"),
    ("pipeline", "evaluate"),
    ("inference", "max_new_tokens"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Welcome,
    Project,
    Forms,
    Editor,
    Review,
    Confirm,
    Jobs,
    Job,
    Checkpoints,
    Tokenizer,
    Result,
}
enum Input {
    Directory,
    Open,
    Import,
    Form(usize),
    Prompt(Action),
    Encode,
    Decode,
    Reports,
    HumanReview,
    Promotion,
    Recover,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewRequest {
    workflow: PathBuf,
    report: PathBuf,
    scores: PathBuf,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PromotionRequest {
    workflow: PathBuf,
    report: PathBuf,
    approved_by: String,
    output: PathBuf,
}
enum Confirmation {
    Migrate {
        path: PathBuf,
        original: String,
    },
    HumanReview {
        workflow: PathBuf,
        report: PathBuf,
        review: post_training::HumanReview,
    },
    Promote(PromotionRequest),
    Recover {
        workflow: PathBuf,
        spec: Box<JobSpec>,
    },
}
enum Reply {
    Review(ProjectConfig, String),
    Launch(JobSpec),
    Text(String),
    Confirm(Confirmation, String),
}
pub struct App {
    pub screen: Screen,
    pub cwd: PathBuf,
    pub config_path: Option<PathBuf>,
    pub text: String,
    original: String,
    pub message: String,
    pub cursor: usize,
    pub scroll: u16,
    state: PathBuf,
    entries: Vec<PathBuf>,
    recent: Vec<PathBuf>,
    records: Vec<JobRecord>,
    checkpoints: Vec<PathBuf>,
    checkpoint: Option<PathBuf>,
    selected_job: Option<String>,
    reviewed: Option<ProjectConfig>,
    review_path: Option<PathBuf>,
    confirmation: Option<Confirmation>,
    action: Action,
    prompt: Option<String>,
    conversation: Vec<config::ChatTurn>,
    input: Option<Input>,
    buffer: String,
    editor_cursor: usize,
    receiver: Option<mpsc::Receiver<Result<Reply, String>>>,
    pub busy: bool,
    job_tab: usize,
    dashboard: dashboard::Dashboard,
}
impl App {
    pub fn new(cwd: PathBuf, state: PathBuf) -> Result<Self> {
        let mut app = Self {
            screen: Screen::Welcome,
            cwd,
            config_path: None,
            text: String::new(),
            original: String::new(),
            message: String::new(),
            cursor: 0,
            scroll: 0,
            state,
            entries: vec![],
            recent: vec![],
            records: vec![],
            checkpoints: vec![],
            checkpoint: None,
            selected_job: None,
            reviewed: None,
            review_path: None,
            confirmation: None,
            action: Action::Pipeline,
            prompt: None,
            conversation: vec![],
            input: None,
            buffer: String::new(),
            editor_cursor: 0,
            receiver: None,
            busy: false,
            job_tab: 0,
            dashboard: dashboard::Dashboard::default(),
        };
        app.refresh()?;
        Ok(app)
    }
    fn refresh(&mut self) -> Result<()> {
        self.entries = fs::read_dir(&self.cwd)?
            .take(1000)
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir() || p.extension().is_some_and(|x| x == "toml"))
            .collect();
        self.entries.sort();
        self.recent = jobs::read_json(&self.state.join("recent.json")).unwrap_or_default();
        self.records = jobs::list(&self.state)?;
        Ok(())
    }
    pub fn open(&mut self, path: PathBuf) -> Result<()> {
        let path = fs::canonicalize(path)?;
        let text = fs::read_to_string(&path)?;
        ProjectConfig::parse(&text)?;
        self.cwd = path.parent().context("No project directory")?.into();
        self.config_path = Some(path.clone());
        self.text = text.clone();
        self.original = text;
        self.screen = Screen::Project;
        self.cursor = 0;
        self.conversation.clear();
        self.checkpoint = None;
        jobs::private_dir(&self.state)?;
        self.recent.retain(|p| p != &path);
        self.recent.insert(0, path);
        self.recent.truncate(20);
        jobs::atomic_json(&self.state.join("recent.json"), &self.recent)?;
        Ok(())
    }
    fn config(&self) -> Result<ProjectConfig> {
        ProjectConfig::parse(&self.text)
    }
    fn save(&mut self) -> Result<()> {
        config::save_edited(
            self.config_path.as_ref().context("No project")?,
            &self.original,
            &self.text,
        )?;
        self.original = self.text.clone();
        self.message = "Configuration saved".into();
        Ok(())
    }
    fn task(&mut self, f: impl FnOnce() -> Result<Reply> + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.busy = true;
        self.message = "Working… You can detach; active workers remain independent.".into();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                .map_err(|_| {
                    anyhow::anyhow!("Operation panicked; inspect driver/runtime requirements")
                })
                .and_then(|r| r)
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(result);
        });
    }
    pub fn poll(&mut self) {
        if let Some(rx) = &self.receiver
            && let Ok(reply) = rx.try_recv()
        {
            self.busy = false;
            self.receiver = None;
            match reply {
                Err(e) => self.message = e,
                Ok(Reply::Review(c, text)) => {
                    self.reviewed = Some(c);
                    self.buffer = text;
                    self.screen = Screen::Review;
                    self.scroll = 0;
                    self.message =
                        "Review the plan. Enter starts it; Esc returns without starting.".into();
                }
                Ok(Reply::Launch(spec)) => {
                    self.selected_job = Some(spec.id);
                    self.screen = Screen::Job;
                    self.job_tab = 0;
                    self.scroll = 0;
                    self.message =
                        "Worker started. q / Ctrl+C detaches without stopping it.".into();
                }
                Ok(Reply::Text(text)) => {
                    self.buffer = text;
                    self.screen = Screen::Result;
                    self.scroll = 0;
                    self.message.clear();
                }
                Ok(Reply::Confirm(confirmation, text)) => self.confirmation(confirmation, text),
            }
        }
        match jobs::list(&self.state) {
            Ok(records) => self.records = records,
            Err(e) => self.message = format!("Cannot read job history: {e:#}"),
        }
        if let Some(id) = &self.selected_job {
            self.dashboard.refresh(&self.state, id);
        }
    }
    fn review(&mut self, action: Action) -> Result<()> {
        ensure!(
            self.text == self.original,
            "Save your configuration with Ctrl+S before starting an operation"
        );
        let mut config = self.config()?;
        if action == Action::Chat && !self.conversation.is_empty() {
            config.inference.history = self.conversation.clone();
        }
        if action != Action::PostTrain {
            pipeline::validate_launch(&config, &action, &self.checkpoint)?;
        }
        self.action = action.clone();
        self.review_path = self.config_path.clone();
        let cwd = self.cwd.clone();
        self.task(move || {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "doctor",
                    "--backend",
                    &format!("{:?}", config.training.backend).to_lowercase(),
                    "--device",
                    &config.training.device.to_string(),
                ])
                .output()?;
            ensure!(
                output.status.success(),
                "Backend preflight failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let (c, text) = if action == Action::PostTrain {
                post_training::prepare(&config, &cwd)?
            } else {
                pipeline::review(&config, &cwd, &action)?
            };
            Ok(Reply::Review(c, text))
        });
        Ok(())
    }
    fn launch(&mut self) -> Result<()> {
        let c = self.reviewed.take().context("No reviewed plan")?;
        let root = self.state.clone();
        let path = self.review_path.take().context("No reviewed project")?;
        let action = self.action.clone();
        let checkpoint = if action == Action::PostTrain {
            None
        } else {
            self.checkpoint.clone()
        };
        let prompt = if action == Action::PostTrain {
            None
        } else {
            self.prompt.take()
        };
        // Exact resumes use the executable retained by the selected job when known.
        let retained = if action == Action::Resume {
            checkpoint
                .as_ref()
                .map(|p| jobs::retained_for_checkpoint(&root, p))
                .transpose()?
                .flatten()
        } else {
            None
        };
        self.task(move || {
            Ok(Reply::Launch(jobs::launch(
                &root,
                &path,
                action,
                checkpoint,
                prompt,
                retained.as_deref(),
                Some(c),
            )?))
        });
        Ok(())
    }
    fn input(&mut self, kind: Input, value: String) {
        self.input = Some(kind);
        self.buffer = value;
    }
    fn confirmation(&mut self, confirmation: Confirmation, text: String) {
        self.confirmation = Some(confirmation);
        self.buffer = text;
        self.screen = Screen::Confirm;
        self.scroll = 0;
        self.message = "Enter confirms this exact action. Esc cancels.".into();
    }
    fn confirm(&mut self) -> Result<()> {
        let confirmation = self
            .confirmation
            .take()
            .context("No pending confirmation")?;
        let root = self.state.clone();
        match confirmation {
            Confirmation::Migrate { path, original } => {
                ensure!(
                    fs::read_to_string(&path)? == original,
                    "Configuration changed; reopen and review migration again"
                );
                config::migrate_post_training(&path)?;
                self.open(path)?;
                self.screen = Screen::Editor;
                self.message = "Migrated to schema 2. Add explicit [post_training] settings in this editor; Ctrl+S validates and saves.".into();
            }
            Confirmation::HumanReview {
                workflow,
                report,
                review,
            } => self.task(move || {
                let saved = post_training::save_review(&workflow, &report, &review)?;
                Ok(Reply::Text(format!(
                    "Human review saved: {}\nPromotion remains a separate explicit action.",
                    saved.display()
                )))
            }),
            Confirmation::Promote(request) => self.task(move || {
                let selection = post_training::promote(
                    &request.workflow,
                    &request.report,
                    &request.approved_by,
                    &request.output,
                )?;
                Ok(Reply::Text(format!(
                    "Selection saved: {}\n\n{}",
                    request.output.display(),
                    serde_json::to_string_pretty(&selection)?
                )))
            }),
            Confirmation::Recover { workflow, spec } => self.task(move || {
                // Validate again at launch, retaining exactly the reviewed snapshot and binary.
                let current = post_training::recover_spec(&workflow)?;
                ensure!(
                    serde_json::to_value(&current)? == serde_json::to_value(&*spec)?,
                    "Workflow recovery settings changed; review recovery again"
                );
                Ok(Reply::Launch(jobs::launch(
                    &root,
                    &spec.config_path,
                    Action::PostTrainRecover,
                    Some(workflow),
                    None,
                    Some(&spec.executable),
                    Some(spec.config),
                )?))
            }),
        }
        Ok(())
    }
    fn submit(&mut self) -> Result<()> {
        let kind = self.input.take().context("No input")?;
        let value = self.buffer.clone();
        match kind {
            Input::Directory => {
                let path = fs::canonicalize(value)?;
                ensure!(path.is_dir(), "Select a directory");
                self.cwd = path;
                self.cursor = 0;
                self.refresh()?;
            }
            Input::Open => self.open(config::absolute(&self.cwd, &PathBuf::from(value)))?,
            Input::Import => {
                let path = config::absolute(&self.cwd, &PathBuf::from(value));
                let target = self.cwd.join("model.toml");
                ProjectConfig::import_recipe(&path, &target)?;
                self.open(target)?;
            }
            Input::Form(i) => {
                let (section, key) = FIELDS[i];
                self.text = config::edit_scalar(&self.text, section, key, &value)?;
                self.message = "Changed in memory. Ctrl+S saves; Esc returns.".into();
            }
            Input::Prompt(action) => {
                self.prompt = Some(value);
                self.review(action)?;
            }
            Input::Reports => {
                let workflow = config::absolute(&self.cwd, &PathBuf::from(value));
                self.task(move || {
                    let state = post_training::WorkflowState::load(&workflow)?;
                    let mut text = format!(
                        "Workflow: {}\nPhase: {:?}\n{}\n\n",
                        workflow.display(),
                        state.phase,
                        state.message
                    );
                    if state.reports.is_empty() {
                        text.push_str(
                            "No completed reports yet. Reconnect through Jobs to monitor progress.",
                        );
                    } else {
                        text.push_str(&serde_json::to_string_pretty(&post_training::rank(
                            &workflow,
                        )?)?);
                        for path in state.reports {
                            let report = post_training::read_report(&path)?;
                            text.push_str(&format!(
                                "\n\nReport: {}\n{}",
                                path.display(),
                                report.markdown()
                            ));
                        }
                    }
                    Ok(Reply::Text(text))
                });
            }
            Input::HumanReview => {
                let path = config::absolute(&self.cwd, &PathBuf::from(value));
                self.task(move || {
                    let request: ReviewRequest = jobs::read_json(&path)?;
                    let base = path.parent().context("Request file has no directory")?;
                    let workflow = config::absolute(base, &request.workflow);
                    let report = config::absolute(base, &request.report);
                    let review: post_training::HumanReview = jobs::read_json(&config::absolute(base, &request.scores))?;
                    post_training::WorkflowState::load(&workflow)?;
                    let text = format!("Submit immutable human review\nWorkflow: {}\nReport: {}\n\n{}\n\nEnter submits these human-provided scores. All report/rubric and completeness checks must pass.", workflow.display(),report.display(),serde_json::to_string_pretty(&review)?);
                    Ok(Reply::Confirm(Confirmation::HumanReview { workflow, report, review }, text))
                });
            }
            Input::Promotion => {
                let path = config::absolute(&self.cwd, &PathBuf::from(value));
                self.task(move || {
                    let mut request: PromotionRequest = jobs::read_json(&path)?;
                    let base = path.parent().context("Request file has no directory")?;
                    request.workflow = config::absolute(base, &request.workflow);
                    request.report = fs::canonicalize(config::absolute(base, &request.report))?;
                    request.output = config::absolute(base, &request.output);
                    ensure!(!request.approved_by.trim().is_empty(), "Explicit human approver is required");
                    let ranked = post_training::rank(&request.workflow)?;
                    ensure!(ranked.iter().any(|c| c.eligible && fs::canonicalize(&c.report).ok().as_ref() == Some(&request.report)), "Candidate has failed or pending gates; inspect ranked reports first");
                    let text = format!("Approve candidate selection\nWorkflow: {}\nReport: {}\nApprover: {}\nNew selection manifest: {}\n\n{}\n\nEnter explicitly promotes this candidate. Sealed-test acceptance remains separate.",request.workflow.display(),request.report.display(),request.approved_by,request.output.display(),serde_json::to_string_pretty(&ranked)?);
                    Ok(Reply::Confirm(Confirmation::Promote(request),text))
                });
            }
            Input::Recover => {
                let workflow =
                    fs::canonicalize(config::absolute(&self.cwd, &PathBuf::from(value)))?;
                self.task(move || {
                    let spec = post_training::recover_spec(&workflow)?;
                    let text = format!("Recover workflow: {}\nRetained executable: {}\nSHA256: {}\n\nFrozen configuration:\n{}\nEnter launches the retained worker with this frozen configuration.",workflow.display(),spec.executable.display(),spec.executable_sha256,toml::to_string_pretty(&spec.config)?);
                    Ok(Reply::Confirm(Confirmation::Recover { workflow, spec: Box::new(spec) },text))
                });
            }
            Input::Encode | Input::Decode => {
                let c = self.config()?;
                let t =
                    omega_tokenizer::Tokens::new(config::absolute(&self.cwd, &c.tokenizer.path))
                        .map_err(anyhow::Error::msg)?;
                self.buffer = if matches!(kind, Input::Encode) {
                    format!(
                        "{:?}",
                        t.encode(&value, false)
                            .map_err(anyhow::Error::msg)?
                            .get_ids()
                    )
                } else {
                    let ids: Vec<u32> = value
                        .split(|c: char| c.is_whitespace() || c == ',')
                        .filter(|x| !x.is_empty())
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?;
                    t.decode(&ids, false).map_err(anyhow::Error::msg)?
                };
                self.screen = Screen::Result;
                self.scroll = 0;
            }
        }
        Ok(())
    }
    fn inspect_checkpoints(&mut self) -> Result<()> {
        let c = self.config()?;
        let root = config::absolute(&self.cwd, &c.paths.weights);
        self.checkpoints.clear();
        if root.exists() {
            for name in [&c.name, &format!("{}-assistant", c.name)] {
                let catalog = omega_training::checkpoint_catalog::discover_checkpoints(
                    &root,
                    name,
                    &Default::default(),
                )
                .map_err(anyhow::Error::msg)?;
                for e in catalog.entries {
                    let p = root.join(&e.name);
                    if omega_training::checkpoint::read_checkpoint_manifest(&p).is_ok() {
                        self.checkpoints.push(p)
                    }
                }
            }
        }
        self.checkpoints.sort();
        self.screen = Screen::Checkpoints;
        self.cursor = 0;
        Ok(())
    }
    fn enter(&mut self) -> Result<()> {
        match self.screen {
            Screen::Welcome => {
                let path = self
                    .entries
                    .iter()
                    .chain(&self.recent)
                    .nth(self.cursor)
                    .cloned()
                    .context("No selected entry")?;
                if path.is_dir() {
                    self.cwd = path;
                    self.cursor = 0;
                    self.refresh()?;
                } else {
                    self.open(path)?;
                }
            }
            Screen::Project => match self.cursor {
                0 => self.review(Action::Pipeline)?,
                1 => self.review(Action::PrepareDataset)?,
                2 => self.review(Action::TrainTokenizer)?,
                3 => self.review(Action::PrepareCache)?,
                4 => self.review(Action::Benchmark)?,
                5 => self.review(Action::Train)?,
                6 => self.review(Action::Assistant)?,
                7 => self.review(Action::Resume)?,
                8 => self.review(Action::Evaluate)?,
                9 => self.input(Input::Prompt(Action::Generate), String::new()),
                10 => self.input(Input::Prompt(Action::Chat), String::new()),
                11 => self.inspect_checkpoints()?,
                12 => {
                    let c = self.config()?;
                    let t = omega_tokenizer::Tokens::new(config::absolute(
                        &self.cwd,
                        &c.tokenizer.path,
                    ))
                    .map_err(anyhow::Error::msg)?;
                    self.buffer = format!(
                        "Vocabulary entries: {}\n{}\n\n{}\n\ne: encode text | d: decode IDs",
                        t.vocab_size(),
                        serde_json::to_string_pretty(
                            &omega_training::checkpoint::tokenizer_identity(&t)
                                .map_err(anyhow::Error::msg)?
                        )?,
                        (0..t.vocab_size().min(10000) as u32)
                            .filter_map(|id| t
                                .id_to_token(id)
                                .map(|token| format!("{id}: {token:?}")))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                    self.screen = Screen::Tokenizer;
                }
                13 => {
                    self.screen = Screen::Forms;
                    self.cursor = 0;
                }
                14 => {
                    self.screen = Screen::Editor;
                    self.editor_cursor = 0;
                }
                15 => {
                    self.screen = Screen::Jobs;
                    self.cursor = 0;
                }
                16 => {
                    let backend = self.config()?.training.backend;
                    self.task(move || {
                        // Probe in a child: driver panics/stdout never corrupt the terminal.
                        let output = std::process::Command::new(std::env::current_exe()?)
                            .args([
                                "doctor",
                                "--backend",
                                &format!("{backend:?}").to_lowercase(),
                            ])
                            .output()?;
                        Ok(Reply::Text(format!(
                            "{}\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        )))
                    });
                }
                17 => {
                    let c = self.config()?;
                    let root = config::absolute(&self.cwd, &c.paths.datasets);
                    let mut lines = vec![];
                    for selected in c.selections() {
                        let path = fs::canonicalize(root.join(&selected))?;
                        let release = omega_training::release::inspect_partition(&path)
                            .map_err(anyhow::Error::msg)?;
                        lines.push(format!("{selected}: {release:#?}"));
                    }
                    self.buffer = lines.join("\n");
                    self.screen = Screen::Result;
                }
                18 => {
                    ensure!(
                        self.text == self.original,
                        "Save or discard unsaved edits before migration"
                    );
                    ensure!(
                        self.config()?.omega_schema_version == 1,
                        "Project already uses schema 2"
                    );
                    let path = self.config_path.clone().context("No project")?;
                    self.confirmation(Confirmation::Migrate { path: path.clone(), original: self.original.clone() },
                        format!("Migrate {} from schema 1 to schema 2.\nExisting configuration and job snapshots are preserved.\nAfter migration, configure explicit [post_training] inputs, budgets and gates in the TOML editor.\n\nEnter writes the schema change; Esc cancels.",path.display()));
                }
                19 => {
                    let c = self.config()?;
                    let cwd = self.cwd.clone();
                    self.task(move || {
                        Ok(Reply::Text(serde_json::to_string_pretty(
                            &post_training::readiness(&c, &cwd)?,
                        )?))
                    });
                }
                20 => self.review(Action::PostTrain)?,
                21 => self.input(Input::Reports, self.workflow_default()),
                22 => self.input(Input::HumanReview, String::new()),
                23 => self.input(Input::Promotion, String::new()),
                24 => self.input(Input::Recover, self.workflow_default()),
                _ => {}
            },
            Screen::Forms => {
                let (section, key) = FIELDS[self.cursor.min(FIELDS.len() - 1)];
                let value = toml::Value::try_from(self.config()?)?;
                let value = if section.is_empty() {
                    &value[key]
                } else {
                    &value[section][key]
                };
                let text = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                self.input(Input::Form(self.cursor), text);
            }
            Screen::Review => self.launch()?,
            Screen::Confirm => self.confirm()?,
            Screen::Jobs => {
                if let Some(r) = self.records.get(self.cursor) {
                    self.selected_job = Some(r.id.clone());
                    self.dashboard.refresh(&self.state, &r.id);
                    self.screen = Screen::Job;
                    self.job_tab = 0;
                    self.scroll = 0;
                }
            }
            Screen::Checkpoints => {
                self.checkpoint = self.checkpoints.get(self.cursor).cloned();
                self.screen = Screen::Project;
                self.cursor = 7;
                self.message =
                    "Checkpoint selected. Choose resume, evaluate, generate or chat.".into();
            }
            _ => {}
        }
        Ok(())
    }
    fn workflow_default(&self) -> String {
        self.config()
            .ok()
            .and_then(|c| c.post_training)
            .map(|p| config::absolute(&self.cwd, &p.output).display().to_string())
            .unwrap_or_default()
    }
    /// Returns true only to detach the UI, never to stop a worker.
    pub fn key(&mut self, key: KeyEvent) -> Result<bool> {
        if key.kind != KeyEventKind::Press {
            return Ok(false);
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Ok(true);
        }
        if self.busy {
            // Keep the request's project/action stable until its reviewed reply arrives.
            return Ok(key.code == KeyCode::Char('q'));
        }
        if self.input.is_some() {
            match key.code {
                KeyCode::Esc => self.input = None,
                KeyCode::Enter => self.submit()?,
                KeyCode::Backspace => {
                    self.buffer.pop();
                }
                KeyCode::Char(c) => self.buffer.push(c),
                _ => {}
            }
            return Ok(false);
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s') {
            self.save()?;
            return Ok(false);
        }
        if self.screen == Screen::Editor {
            let mut chars: Vec<char> = self.text.chars().collect();
            let cursor = self.editor_cursor.min(chars.len());
            match key.code {
                KeyCode::Esc => self.screen = Screen::Project,
                KeyCode::Left => self.editor_cursor = cursor.saturating_sub(1),
                KeyCode::Right => self.editor_cursor = (cursor + 1).min(chars.len()),
                KeyCode::Home => {
                    self.editor_cursor = chars[..cursor]
                        .iter()
                        .rposition(|c| *c == '\n')
                        .map_or(0, |p| p + 1)
                }
                KeyCode::End => {
                    self.editor_cursor = chars[cursor..]
                        .iter()
                        .position(|c| *c == '\n')
                        .map_or(chars.len(), |p| cursor + p)
                }
                KeyCode::Up | KeyCode::Down => {
                    let start = chars[..cursor]
                        .iter()
                        .rposition(|c| *c == '\n')
                        .map_or(0, |p| p + 1);
                    let col = cursor - start;
                    if key.code == KeyCode::Up && start > 0 {
                        let previous = chars[..start - 1]
                            .iter()
                            .rposition(|c| *c == '\n')
                            .map_or(0, |p| p + 1);
                        self.editor_cursor = (previous + col).min(start - 1);
                    } else if key.code == KeyCode::Down
                        && let Some(end) = chars[cursor..]
                            .iter()
                            .position(|c| *c == '\n')
                            .map(|p| cursor + p)
                    {
                        let next = end + 1;
                        let next_end = chars[next..]
                            .iter()
                            .position(|c| *c == '\n')
                            .map_or(chars.len(), |p| next + p);
                        self.editor_cursor = (next + col).min(next_end);
                    }
                }
                KeyCode::Backspace if cursor > 0 => {
                    chars.remove(cursor - 1);
                    self.editor_cursor = cursor - 1;
                }
                KeyCode::Delete if cursor < chars.len() => {
                    chars.remove(cursor);
                }
                KeyCode::Enter => {
                    chars.insert(cursor, '\n');
                    self.editor_cursor = cursor + 1;
                }
                KeyCode::Tab => {
                    for _ in 0..4 {
                        chars.insert(cursor, ' ');
                    }
                    self.editor_cursor = cursor + 4;
                }
                KeyCode::Char(c) => {
                    chars.insert(cursor, c);
                    self.editor_cursor = cursor + 1;
                }
                _ => {}
            }
            self.text = chars.into_iter().collect();
            return Ok(false);
        }
        match key.code {
            KeyCode::Char('q') => return Ok(true),
            KeyCode::F(2) if self.config_path.is_some() => {
                self.screen = Screen::Project;
                self.cursor = 0;
                self.scroll = 0;
            }
            KeyCode::F(3) => {
                self.screen = Screen::Jobs;
                self.cursor = 0;
                self.scroll = 0;
            }
            KeyCode::F(4) => {
                self.screen = Screen::Welcome;
                self.cursor = 0;
                self.refresh()?;
            }
            KeyCode::Tab | KeyCode::BackTab if self.screen == Screen::Job => {
                self.job_tab = (self.job_tab + if key.code == KeyCode::Tab { 1 } else { 3 }) % 4;
                self.scroll = 0;
            }
            KeyCode::Char(c @ '1'..='4') if self.screen == Screen::Job => {
                self.job_tab = (c as u8 - b'1') as usize;
                self.scroll = 0;
            }
            KeyCode::Up | KeyCode::PageUp if self.screen == Screen::Job && self.job_tab == 1 => {
                self.scroll =
                    self.scroll
                        .saturating_add(if key.code == KeyCode::Up { 1 } else { 10 });
            }
            KeyCode::Down | KeyCode::PageDown
                if self.screen == Screen::Job && self.job_tab == 1 =>
            {
                self.scroll =
                    self.scroll
                        .saturating_sub(if key.code == KeyCode::Down { 1 } else { 10 });
            }
            KeyCode::End if self.screen == Screen::Job => self.scroll = 0,
            KeyCode::Esc => {
                self.confirmation = None;
                self.reviewed = None;
                self.review_path = None;
                self.screen = if self.config_path.is_some() {
                    Screen::Project
                } else {
                    Screen::Welcome
                };
                self.cursor = 0;
                self.scroll = 0;
            }
            KeyCode::Up => {
                self.cursor = self.cursor.saturating_sub(1);
                self.scroll = self.scroll.saturating_sub(1);
            }
            KeyCode::Down => {
                let n = match self.screen {
                    Screen::Welcome => self.entries.len() + self.recent.len(),
                    Screen::Project => ACTIONS.len(),
                    Screen::Forms => FIELDS.len(),
                    Screen::Jobs => self.records.len(),
                    Screen::Checkpoints => self.checkpoints.len(),
                    _ => 1,
                };
                self.cursor = (self.cursor + 1).min(n.saturating_sub(1));
                self.scroll = self.scroll.saturating_add(1);
            }
            KeyCode::PageDown => self.scroll = self.scroll.saturating_add(10),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::Enter if !self.busy => self.enter()?,
            KeyCode::Char('w') if self.screen == Screen::Welcome => {
                self.input(Input::Directory, self.cwd.display().to_string())
            }
            KeyCode::Char('o') if self.screen == Screen::Welcome => {
                self.input(Input::Open, "model.toml".into())
            }
            KeyCode::Char('c') if self.screen == Screen::Welcome => {
                let path = self.cwd.join("model.toml");
                ProjectConfig::default().create(&path)?;
                self.open(path)?;
            }
            KeyCode::Char('i') if self.screen == Screen::Welcome => {
                self.input(Input::Import, String::new())
            }
            KeyCode::Backspace if self.screen == Screen::Welcome => {
                if let Some(parent) = self.cwd.parent() {
                    self.cwd = parent.into();
                    self.refresh()?;
                    self.cursor = 0;
                }
            }
            KeyCode::Char('j') => {
                self.screen = Screen::Jobs;
                self.cursor = 0;
            }
            KeyCode::Char('w') => {
                self.screen = Screen::Welcome;
                self.cursor = 0;
                self.refresh()?;
            }
            KeyCode::Char('s') if self.screen == Screen::Job => {
                if let Some(id) = &self.selected_job {
                    jobs::request(&self.state, id, "stop")?;
                    self.message="Checkpoint-and-stop requested. The worker finishes its current update before saving.".into();
                }
            }
            KeyCode::Char('r') if self.screen == Screen::Job => {
                if let Some(r) = self
                    .records
                    .iter()
                    .find(|r| Some(&r.id) == self.selected_job.as_ref())
                {
                    self.checkpoint = r.checkpoint.clone();
                    let spec: JobSpec =
                        jobs::read_json(&jobs::job_dir(&self.state, &r.id)?.join("job.json"))?;
                    self.open(spec.config_path.clone())?;
                    self.checkpoint = r_checkpoint(&spec, &self.records);
                    self.review(Action::Resume)?;
                }
            }
            KeyCode::Char('c' | 'n') if self.screen == Screen::Job => {
                let record = self
                    .records
                    .iter()
                    .find(|r| Some(&r.id) == self.selected_job.as_ref())
                    .context("Select a completed chat job")?;
                let spec: JobSpec =
                    jobs::read_json(&jobs::job_dir(&self.state, &record.id)?.join("job.json"))?;
                ensure!(
                    spec.action == Action::Chat && record.status == jobs::JobStatus::Completed,
                    "Select a successfully completed chat reply first"
                );
                let history =
                    record.result.as_ref().context("Chat result missing")?["history"].clone();
                self.open(spec.config_path)?;
                self.checkpoint = spec.checkpoint;
                if key.code == KeyCode::Char('c') {
                    self.conversation = serde_json::from_value(history)?;
                }
                self.input(Input::Prompt(Action::Chat), String::new());
            }
            KeyCode::Char('e') if self.screen == Screen::Tokenizer => {
                self.input(Input::Encode, String::new())
            }
            KeyCode::Char('d') if self.screen == Screen::Tokenizer => {
                self.input(Input::Decode, String::new())
            }
            _ => {}
        }
        Ok(false)
    }
    pub fn render(&self, frame: &mut Frame) {
        view::render(self, frame);
    }
}
fn r_checkpoint(spec: &JobSpec, records: &[JobRecord]) -> Option<PathBuf> {
    records
        .iter()
        .find(|r| r.id == spec.id)
        .and_then(|r| r.checkpoint.clone())
}

pub fn run(project: Option<PathBuf>) -> Result<()> {
    use std::io::IsTerminal;
    ensure!(
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
        "The TUI needs a terminal; use --help for headless commands"
    );
    let mut app = App::new(std::env::current_dir()?, jobs::state_root()?)?;
    if let Some(p) = project {
        app.open(p)?;
    }
    let mut terminal = ratatui::init();
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            ratatui::restore();
        }
    }
    let _restore = Restore;
    let mut last = Instant::now();
    loop {
        if last.elapsed() >= Duration::from_millis(500) {
            app.poll();
            last = Instant::now();
        }
        terminal.draw(|frame| app.render(frame))?;
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            match app.key(key) {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => app.message = format!("{e:#}"),
            }
        }
    }
    Ok(())
}
