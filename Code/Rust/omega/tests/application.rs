use omega::{
    ProjectConfig, config,
    jobs::{self, Action, JobRecord, JobSpec, JobStatus, Reporter},
    tui::{App, Screen},
};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

fn config_fixture(root: &std::path::Path) -> ProjectConfig {
    fs::create_dir_all(root.join("datasets/base")).unwrap();
    fs::create_dir_all(root.join("datasets/heldout")).unwrap();
    fs::create_dir_all(root.join("datasets/chat")).unwrap();
    fs::write(
        root.join("datasets/base/a.txt"),
        "Hello world this is a tiny training corpus. Hello world!",
    )
    .unwrap();
    fs::write(root.join("datasets/heldout/a.txt"), "Hello tiny world!").unwrap();
    fs::write(root.join("datasets/chat/a.jsonl"),r#"{"schema_version":1,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"ok"}]}"#).unwrap();
    let mut c = ProjectConfig {
        name: "tiny".into(),
        ..Default::default()
    };
    c.dataset.selections = vec!["base".into()];
    c.model = omega_benchmark::ModelDimensions {
        context_length: 32,
        d_model: 4,
        heads: 1,
        layers: 1,
        d_ff: 8,
    };
    c.tokenizer.vocab_size = 260;
    c.tokenizer.min_frequency = 1;
    c.training.epochs = 1;
    c.training.batch_size = 2;
    c.pipeline.prepare_tokenizer = true;
    c.pipeline.benchmark = true;
    c.pipeline.evaluate = true;
    c.benchmark.samples = 1;
    c.benchmark.warmup = 1;
    c.evaluation.selections = vec!["heldout".into()];
    c
}
fn spec(
    root: &std::path::Path,
    c: ProjectConfig,
    action: Action,
    checkpoint: Option<PathBuf>,
) -> JobSpec {
    JobSpec {
        schema_version: 1,
        id: "abc-123".into(),
        project: root.into(),
        config_path: root.join("model.toml"),
        config: c,
        action,
        checkpoint,
        prompt: Some("Hi".into()),
        executable: "test".into(),
        executable_sha256: "test".into(),
        created_ms: jobs::now_ms(),
    }
}
fn reporter(root: &std::path::Path) -> Arc<Reporter> {
    fs::create_dir_all(root).unwrap();
    Arc::new(Reporter::new(
        root.into(),
        JobRecord {
            id: "abc-123".into(),
            status: JobStatus::Running,
            process: None,
            project: root.into(),
            stage: "test".into(),
            started_ms: jobs::now_ms(),
            updated_ms: jobs::now_ms(),
            progress: serde_json::Value::Null,
            total_updates: None,
            checkpoint: None,
            error: None,
            result: None,
        },
    ))
}

#[test]
fn forms_preserve_comments_and_validate_raw_edits_and_imports() {
    let minimal = "omega_schema_version = 1 # preserve\n";
    let edited = config::edit_scalar(minimal, "training", "epochs", "3").unwrap();
    assert!(edited.contains("# preserve"));
    assert_eq!(ProjectConfig::parse(&edited).unwrap().training.epochs, 3);
    let temp = tempfile::tempdir().unwrap();
    let c = config_fixture(temp.path());
    let text = toml::to_string_pretty(&c)
        .unwrap()
        .replace("epochs = 1", "epochs = 1 # keep this comment");
    let edited = config::edit_scalar(&text, "training", "epochs", "3").unwrap();
    assert!(edited.contains("# keep this comment"));
    assert_eq!(ProjectConfig::parse(&edited).unwrap().training.epochs, 3);
    let edited =
        config::edit_scalar(&edited, "dataset", "selections", "[\"base\", \"heldout\"]").unwrap();
    assert_eq!(
        ProjectConfig::parse(&edited)
            .unwrap()
            .dataset
            .selections
            .len(),
        2
    );
    assert!(config::edit_scalar(&edited, "training", "batch_size", "0").is_err());
    assert!(
        ProjectConfig::parse(
            &edited.replace("omega_schema_version = 1", "omega_schema_version = 2")
        )
        .is_err()
    );
    let recipe = temp.path().join("recipe.toml");
    fs::write(&recipe, omega_datasets::EXAMPLE_CONFIG).unwrap();
    let imported = temp.path().join("model.toml");
    ProjectConfig::import_recipe(&recipe, &imported).unwrap();
    assert!(
        ProjectConfig::load(&imported)
            .unwrap()
            .dataset
            .recipe
            .is_some()
    );
    assert_eq!(
        fs::read_to_string(&recipe).unwrap(),
        omega_datasets::EXAMPLE_CONFIG
    );
    assert!(ProjectConfig::import_recipe(&recipe, &imported).is_err());
    assert!(ProjectConfig::load(&recipe).is_err());
    assert_eq!(
        config::absolute(temp.path(), std::path::Path::new("datasets")),
        temp.path().join("datasets")
    );
    assert!(config::save_edited(&imported, "stale", &text).is_err());
}

#[test]
fn offline_pipeline_tokenizes_benchmarks_trains_evaluates_and_resumes() {
    let temp = tempfile::tempdir().unwrap();
    let c = config_fixture(temp.path());
    let report = reporter(&temp.path().join("run"));
    omega::pipeline::execute(
        &spec(temp.path(), c.clone(), Action::Pipeline, None),
        report.clone(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let checkpoint = report.checkpoint().unwrap();
    assert!(checkpoint.join("COMPLETE").exists());
    assert!(
        report.record.lock().unwrap().result.as_ref().unwrap()["cross_entropy"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
    let mut continued = c.clone();
    continued.training.epochs = 2;
    omega::pipeline::execute(
        &spec(
            temp.path(),
            continued,
            Action::Resume,
            Some(checkpoint.clone()),
        ),
        report.clone(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_ne!(report.checkpoint().unwrap(), checkpoint);
    omega::pipeline::execute(
        &spec(
            temp.path(),
            c.clone(),
            Action::Generate,
            report.checkpoint(),
        ),
        report.clone(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(report.record.lock().unwrap().result.as_ref().unwrap()["text"].is_string());
    // Existing tokenizer is reusable only while receipt, data and IDs still agree.
    omega::pipeline::execute(
        &spec(temp.path(), c.clone(), Action::TrainTokenizer, None),
        reporter(&temp.path().join("reuse")),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    fs::write(temp.path().join("datasets/base/a.txt"), "Changed data").unwrap();
    assert!(
        omega::pipeline::execute(
            &spec(temp.path(), c, Action::TrainTokenizer, None),
            reporter(&temp.path().join("changed")),
            Arc::new(AtomicBool::new(false))
        )
        .is_err()
    );
}

#[test]
fn assistant_stage_uses_parent_tokenizer_and_produces_chat_checkpoint() {
    let temp = tempfile::tempdir().unwrap();
    let mut c = config_fixture(temp.path());
    c.pipeline.benchmark = false;
    c.pipeline.evaluate = false;
    c.assistant = Some(config::AssistantSettings {
        selections: vec!["chat".into()],
        epochs: 2,
        learning_rate: 0.01,
    });
    let report = reporter(&temp.path().join("run"));
    omega::pipeline::execute(
        &spec(temp.path(), c.clone(), Action::Pipeline, None),
        report.clone(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let checkpoint = report.checkpoint().unwrap();
    let resume = omega_training::read_resume_manifest(&checkpoint).unwrap();
    assert_eq!(resume.schema_version, 5);
    assert!(resume.parent.is_some());
    let reply = omega::pipeline::execute(
        &spec(temp.path(), c, Action::Chat, Some(checkpoint)),
        report.clone(),
        Arc::new(AtomicBool::new(false)),
    );
    match reply {
        Ok(()) => {
            assert!(report.record.lock().unwrap().result.as_ref().unwrap()["text"].is_string())
        }
        Err(e) => assert!(e.to_string().contains("unexpected role token"), "{e}"),
    }
}

#[test]
fn tui_renders_onboarding_editor_dashboard_and_small_terminal_without_training() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let temp = tempfile::tempdir().unwrap();
    let c = config_fixture(temp.path());
    let path = temp.path().join("model.toml");
    c.create(&path).unwrap();
    let mut app = App::new(temp.path().into(), temp.path().join("state")).unwrap();
    for screen in [
        Screen::Welcome,
        Screen::Project,
        Screen::Forms,
        Screen::Editor,
        Screen::Review,
        Screen::Jobs,
        Screen::Job,
        Screen::Checkpoints,
        Screen::Tokenizer,
        Screen::Result,
    ] {
        app.screen = screen;
        for (width, height) in [(100, 30), (24, 8), (8, 4)] {
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| app.render(frame)).unwrap();
        }
    }
    app.open(path).unwrap();
    app.screen = Screen::Editor;
    app.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .unwrap();
    app.key(KeyEvent::new(KeyCode::Char('#'), KeyModifiers::NONE))
        .unwrap();
    assert!(
        app.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .unwrap()
    );
    assert!(!temp.path().join("weights").exists());
}

#[test]
fn journals_ignore_partial_final_records_and_invalid_job_ids() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("events");
    fs::write(
        &path,
        "{\"time_ms\":1,\"kind\":\"update\",\"data\":{}}\n{\"time_ms\":2",
    )
    .unwrap();
    assert_eq!(jobs::events(&path).unwrap().len(), 1);
    assert!(jobs::job_dir(temp.path(), "../../escape").is_err());
    let file = temp.path().join("file");
    fs::write(&file, "existing").unwrap();
    assert!(jobs::private_dir(&file).is_err());
}

#[test]
fn completed_dataset_release_is_verified_before_pipeline_reuse() {
    use omega_datasets::hub::{Hub, RemoteFile, ResolvedSource};
    struct Offline;
    impl Hub for Offline {
        fn resolve(
            &self,
            source: &omega_datasets::config::Source,
        ) -> anyhow::Result<ResolvedSource> {
            Ok(ResolvedSource {
                source_id: source.id.clone(),
                repo: source.repo.clone(),
                revision: "a".repeat(40),
                files: vec![RemoteFile {
                    path: "rows.jsonl".into(),
                    size: None,
                }],
            })
        }
        fn download(
            &self,
            _: &ResolvedSource,
            _: &RemoteFile,
            output: &mut dyn std::io::Write,
            _: u64,
        ) -> anyhow::Result<u64> {
            let bytes = b"{\"text\":\"a bounded offline training document\"}\n";
            output.write_all(bytes)?;
            Ok(bytes.len() as u64)
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let mut c = config_fixture(temp.path());
    let mut recipe: omega_datasets::config::Config =
        toml::from_str(omega_datasets::EXAMPLE_CONFIG).unwrap();
    recipe.sources[0].repo = "test/offline".into();
    recipe.sources[0].revision = "a".repeat(40);
    recipe.sources[0].files = vec!["rows.jsonl".into()];
    recipe.sources[0].format = omega_datasets::config::Format::Jsonl;
    recipe.split.train = 1.0;
    recipe.split.validation = 0.0;
    recipe.split.test = 0.0;
    let folder = temp.path().join("datasets/corpus");
    fs::create_dir_all(&folder).unwrap();
    omega_datasets::build(&folder, &recipe, &Offline).unwrap();
    c.dataset.recipe = Some(recipe.clone());
    let invoke = |c| {
        omega::pipeline::execute(
            &spec(temp.path(), c, Action::PrepareDataset, None),
            reporter(&temp.path().join("release-job")),
            Arc::new(AtomicBool::new(false)),
        )
    };
    invoke(c.clone()).unwrap();
    fs::write(
        folder.join(recipe.output).join("base/train/data.jsonl"),
        "corrupt",
    )
    .unwrap();
    assert!(invoke(c).is_err());
}
