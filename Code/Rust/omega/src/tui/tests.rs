use super::*;
use crate::jobs::JobStatus;
use ratatui::{Terminal, backend::TestBackend};
use serde_json::json;

fn record() -> JobRecord {
    JobRecord {
        id: "abc-123".into(),
        status: JobStatus::Running,
        process: None,
        project: "/workspace/omega_tui_test".into(),
        stage: "Train".into(),
        started_ms: 1,
        updated_ms: 2,
        total_updates: Some(180_000),
        checkpoint: Some("/workspace/weights/my-model-31".into()),
        progress: json!({"completed_updates":31_340,"run_updates":3134,"run_targets":174707,
            "elapsed_seconds":250.9,"epoch":1,"loss":5.8089,"gradient_norm":1.1179,"learning_rate":0.0003}),
        error: None,
        result: None,
    }
}
fn draw(app: &App, width: u16, height: u16) -> (String, Terminal<TestBackend>) {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| app.render(f)).unwrap();
    let text = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|c| c.symbol())
        .collect::<String>();
    (text, terminal)
}
fn press(app: &mut App, key: KeyCode) -> bool {
    app.key(KeyEvent::new(key, KeyModifiers::NONE)).unwrap()
}

fn project_app(temp: &tempfile::TempDir) -> App {
    let path = temp.path().join("model.toml");
    ProjectConfig::default().create(&path).unwrap();
    let mut app = App::new(temp.path().into(), temp.path().join("state")).unwrap();
    app.open(path).unwrap();
    app
}

#[test]
fn post_training_actions_preserve_indices_and_require_explicit_migration() {
    let temp = tempfile::tempdir().unwrap();
    let mut app = project_app(&temp);
    assert_eq!(ACTIONS[0], "Start reviewed pipeline");
    assert_eq!(ACTIONS[17], "Inspect selected dataset releases");
    assert_eq!(ACTIONS.len(), 25);
    for _ in 0..30 {
        press(&mut app, KeyCode::Down);
    }
    assert_eq!(app.cursor, 24);
    let original = fs::read_to_string(app.config_path.as_ref().unwrap()).unwrap();
    app.cursor = 18;
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.screen, Screen::Confirm);
    assert_eq!(
        fs::read_to_string(app.config_path.as_ref().unwrap()).unwrap(),
        original
    );
    assert!(draw(&app, 120, 30).0.contains("schema 1 to schema 2"));
    press(&mut app, KeyCode::Esc);
    assert!(app.confirmation.is_none());
    assert_eq!(app.config().unwrap().omega_schema_version, 1);
    app.cursor = 18;
    press(&mut app, KeyCode::Enter);
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.screen, Screen::Editor);
    assert_eq!(app.config().unwrap().omega_schema_version, 2);
    assert!(app.config().unwrap().post_training.is_none());
    assert_eq!(app.text, app.original);
    assert!(!temp.path().join("weights").exists());
}

#[test]
fn post_training_inputs_confirmations_and_menu_resize_without_side_effects() {
    let temp = tempfile::tempdir().unwrap();
    let mut app = project_app(&temp);
    for index in 18..25 {
        app.screen = Screen::Project;
        app.cursor = index;
        for (w, h) in [(140, 44), (80, 24), (24, 8), (8, 4), (1, 1)] {
            draw(&app, w, h);
        }
        if index >= 21 {
            press(&mut app, KeyCode::Enter);
            assert!(app.input.is_some());
            for (w, h) in [(140, 44), (80, 24), (24, 8), (8, 4), (1, 1)] {
                draw(&app, w, h);
            }
            press(&mut app, KeyCode::Esc);
            assert!(app.input.is_none());
        }
    }
    app.screen = Screen::Confirm;
    app.buffer = "Explicit action details".into();
    for (w, h) in [(140, 44), (80, 24), (24, 8), (8, 4), (1, 1)] {
        draw(&app, w, h);
    }
    press(&mut app, KeyCode::F(3));
    assert_eq!(app.screen, Screen::Jobs);
    press(&mut app, KeyCode::F(2));
    assert_eq!(app.screen, Screen::Project);
    assert!(!temp.path().join("weights").exists());
}

#[test]
fn post_training_review_blocks_unsaved_edits_and_freezes_reply_until_launch() {
    let temp = tempfile::tempdir().unwrap();
    let mut app = project_app(&temp);
    app.text.push_str("\n# unsaved\n");
    app.cursor = 20;
    assert!(
        app.enter()
            .unwrap_err()
            .to_string()
            .contains("Save your configuration")
    );
    assert!(!app.busy);
    app.text = app.original.clone();
    let frozen = app.config().unwrap();
    let (tx, rx) = mpsc::channel();
    app.receiver = Some(rx);
    app.busy = true;
    let cwd = app.cwd.clone();
    press(&mut app, KeyCode::F(4));
    assert_eq!(app.screen, Screen::Project);
    assert_eq!(app.cwd, cwd);
    tx.send(Ok(Reply::Review(
        frozen.clone(),
        "Reviewed exact settings".into(),
    )))
    .unwrap();
    app.poll();
    assert_eq!(app.screen, Screen::Review);
    app.text = app.text.replace(&frozen.name, "changed-after-review");
    assert_eq!(app.reviewed.as_ref().unwrap().name, frozen.name);
    press(&mut app, KeyCode::Esc);
    assert!(app.reviewed.is_none());
}

#[test]
fn human_scores_are_loaded_without_submission_until_confirmation() {
    let temp = tempfile::tempdir().unwrap();
    let mut app = project_app(&temp);
    let review = post_training::HumanReview {
        schema_version: 1,
        report_sha256: "a".repeat(64),
        rubric_version: "rubric-v1".into(),
        reviewer: "Named reviewer".into(),
        cases: vec![],
    };
    let target = temp.path().join("missing-report.json");
    app.confirmation(
        Confirmation::HumanReview {
            workflow: temp.path().join("workflow"),
            report: target.clone(),
            review,
        },
        "Named reviewer supplied scores; Enter submits".into(),
    );
    assert!(draw(&app, 120, 30).0.contains("Named reviewer"));
    assert!(!app.busy);
    press(&mut app, KeyCode::Esc);
    assert!(app.confirmation.is_none());
    assert!(!target.with_extension("review.json").exists());
    let invalid = temp.path().join("invalid.json");
    fs::write(
        &invalid,
        r#"{"workflow":"run","report":"report.json","scores":"scores.json","auto_accept":true}"#,
    )
    .unwrap();
    assert!(jobs::read_json::<ReviewRequest>(&invalid).is_err());
}
#[test]
fn dashboard_metrics_tabs_tail_and_detach_use_existing_records() {
    let temp = tempfile::tempdir().unwrap();
    let mut app = App::new(temp.path().into(), temp.path().join("state")).unwrap();
    app.records = vec![record()];
    app.selected_job = Some("abc-123".into());
    app.screen = Screen::Job;
    app.dashboard.log = (0..50).map(|i| format!("worker line {i}\n")).collect();
    app.dashboard.losses = (0..100)
        .map(|i| {
            (
                31240.0 + i as f64,
                6.4 - i as f64 * 0.006 + (i as f64 * 0.2).sin() * 0.05,
            )
        })
        .collect();
    app.dashboard.checkpoints = (24..=31)
        .rev()
        .map(|i| PathBuf::from(format!("/workspace/weights/my-model-{i}")))
        .collect();
    let (text, terminal) = draw(&app, 140, 44);
    for expected in [
        "RUNNING",
        "31,340 / 180,000",
        "5.8089",
        "4m 10s",
        "696.3 targets/s",
        "Loss · recent",
        "my-model-31",
        "worker line 49",
    ] {
        assert!(text.contains(expected), "Missing {expected}");
    }
    assert!(!text.contains("Some(") && !text.contains("completed_targets"));
    // Optional visual QA export of the actual Ratatui buffer, never a mock renderer.
    if let Ok(path) = std::env::var("OMEGA_TUI_SNAPSHOT") {
        let cells: Vec<_> = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| json!({"text":c.symbol(),"fg":format!("{:?}",c.fg),"bg":format!("{:?}",c.bg)}))
            .collect();
        fs::write(
            path,
            serde_json::to_vec(&json!({"width":140,"height":44,"cells":cells})).unwrap(),
        )
        .unwrap();
    }
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.job_tab, 1);
    assert!(draw(&app, 120, 30).0.contains("worker line 49"));
    press(&mut app, KeyCode::PageUp);
    let older = draw(&app, 120, 30).0;
    assert!(!older.contains("worker line 49"));
    assert!(older.contains("history / End"));
    press(&mut app, KeyCode::End);
    assert!(draw(&app, 120, 30).0.contains("worker line 49"));
    press(&mut app, KeyCode::Char('3'));
    assert!(
        draw(&app, 120, 30)
            .0
            .contains("/workspace/weights/my-model-31")
    );
    app.records[0].error = Some("Storage unavailable".into());
    app.records[0].result = Some(json!({"text":"An actual generated reply"}));
    press(&mut app, KeyCode::Char('4'));
    let result = draw(&app, 120, 30).0;
    assert!(result.contains("Storage unavailable") && result.contains("An actual generated reply"));
    assert!(press(&mut app, KeyCode::Char('q')));
    assert_eq!(app.records[0].status, JobStatus::Running);
    assert!(!temp.path().join("weights").exists());
}
#[test]
fn views_resize_with_unknown_metrics_errors_and_input_dialogs() {
    let temp = tempfile::tempdir().unwrap();
    let mut app = App::new(temp.path().into(), temp.path().join("state")).unwrap();
    let path = temp.path().join("model.toml");
    ProjectConfig::default().create(&path).unwrap();
    app.open(path).unwrap();
    app.records = vec![record()];
    app.records[0].progress = json!({});
    app.records[0].total_updates = None;
    app.records[0].status = JobStatus::Failed;
    app.records[0].error = Some("Storage unavailable".into());
    app.selected_job = Some("abc-123".into());
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
        for tab in 0..4 {
            app.job_tab = tab;
            for (w, h) in [(140, 44), (80, 24), (24, 8), (8, 4), (1, 1)] {
                draw(&app, w, h);
                app.input = Some(Input::Directory);
                draw(&app, w, h);
                app.input = None;
            }
        }
    }
    app.screen = Screen::Job;
    app.job_tab = 0;
    let text = draw(&app, 140, 44).0;
    assert!(text.contains("Storage unavailable") && text.contains("— / —"));
    assert!(!text.contains("NaN") && !text.contains("Some("));
    press(&mut app, KeyCode::F(2));
    assert_eq!(app.screen, Screen::Project);
    press(&mut app, KeyCode::F(3));
    assert_eq!(app.screen, Screen::Jobs);
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.screen, Screen::Job);
    assert_eq!(app.selected_job.as_deref(), Some("abc-123"));
    press(&mut app, KeyCode::F(4));
    assert_eq!(app.screen, Screen::Welcome);
}
#[test]
fn history_is_bounded_tolerates_partial_events_and_clears_on_job_change() {
    let temp = tempfile::tempdir().unwrap();
    let dir = jobs::job_dir(temp.path(), "abc-1").unwrap();
    fs::create_dir_all(&dir).unwrap();
    let mut events = String::new();
    for i in 0..500 {
        events.push_str(&format!(
            "{}\n",
            json!({"time_ms":i,"kind":"update","data":{"completed_updates":i,"loss":5.0}})
        ));
    }
    events.push_str("{\"time_ms\":501");
    fs::write(dir.join("events.jsonl"), events).unwrap();
    fs::write(dir.join("worker.log"), "x".repeat(40_000)).unwrap();
    let paths: Vec<_> = (0..150)
        .map(|i| PathBuf::from(format!("weights/m-{i}")))
        .collect();
    jobs::atomic_json(&dir.join("checkpoints.json"), &paths).unwrap();
    let mut dashboard = dashboard::Dashboard::default();
    dashboard.refresh(temp.path(), "abc-1");
    assert_eq!(dashboard.losses.len(), 240);
    assert_eq!(dashboard.losses.last().unwrap().0, 499.0);
    assert_eq!(dashboard.log.len(), 32 * 1024);
    assert_eq!(dashboard.checkpoints.len(), 100);
    assert!(dashboard.checkpoints[0].ends_with("m-149"));
    fs::write(
        dir.join("events.jsonl"),
        "{\"time_ms\":502,\"kind\":\"stage\",\"data\":{}}\n",
    )
    .unwrap();
    dashboard.refresh(temp.path(), "abc-1");
    assert!(dashboard.losses.is_empty());
    dashboard.refresh(temp.path(), "abc-2");
    assert!(
        dashboard.losses.is_empty() && dashboard.checkpoints.is_empty() && dashboard.log.is_empty()
    );
}
#[test]
fn numbers_preserve_unknowns_and_format_long_runs() {
    assert_eq!(dashboard::duration(None), "—");
    assert_eq!(dashboard::duration(Some(f64::INFINITY)), "—");
    assert_eq!(dashboard::duration(Some(-1.0)), "—");
    assert_eq!(dashboard::duration(Some(1436318.0)), "16d 14h 58m");
    assert_eq!(dashboard::count(17_947_550), "17,947,550");
    assert_eq!(dashboard::number(&json!({"loss":-1.0}), "loss"), None);
}
