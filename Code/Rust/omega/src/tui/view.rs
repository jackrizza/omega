use super::{
    ACTIONS, App, FIELDS, Input, Screen,
    dashboard::{count, duration, number},
};
use crate::jobs::{JobRecord, JobStatus};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Axis, Block, BorderType, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem,
        ListState, Paragraph, Tabs, Wrap,
    },
};
use std::path::Path;

const BG: Color = Color::Rgb(24, 25, 38);
const PANEL: Color = Color::Rgb(30, 32, 48);
const TEXT: Color = Color::Rgb(219, 224, 239);
const MUTED: Color = Color::Rgb(147, 157, 181);
const EDGE: Color = Color::Rgb(69, 76, 100);
const ACCENT: Color = Color::Rgb(129, 213, 199);
const BLUE: Color = Color::Rgb(142, 180, 250);
const WARN: Color = Color::Rgb(239, 194, 126);
const RED: Color = Color::Rgb(244, 143, 159);

fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(format!(" {title} "))
        .border_style(Style::default().fg(EDGE))
        .title_style(Style::default().fg(BLUE))
        .style(Style::default().bg(PANEL).fg(TEXT))
}
fn paragraph(frame: &mut Frame, area: Rect, title: &str, body: impl Into<String>, scroll: u16) {
    frame.render_widget(
        Paragraph::new(body.into())
            .block(panel(title))
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
}
fn list(frame: &mut Frame, area: Rect, title: &str, items: Vec<String>, selected: usize) {
    if items.is_empty() {
        paragraph(frame, area, title, "No entries yet.", 0);
        return;
    }
    let mut state = ListState::default().with_selected(Some(selected.min(items.len() - 1)));
    frame.render_stateful_widget(
        List::new(items.into_iter().map(ListItem::new))
            .block(panel(title).border_style(Style::default().fg(ACCENT)))
            .highlight_symbol("› ")
            .highlight_style(
                Style::default()
                    .bg(Color::Rgb(49, 66, 77))
                    .fg(ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
        area,
        &mut state,
    );
}
fn name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}
fn label(screen: Screen) -> &'static str {
    match screen {
        Screen::Welcome => "Workspace",
        Screen::Project => "Project",
        Screen::Forms => "Configuration",
        Screen::Editor => "TOML editor",
        Screen::Review => "Operation review",
        Screen::Confirm => "Explicit confirmation",
        Screen::Jobs => "Jobs",
        Screen::Job => "Training monitor",
        Screen::Checkpoints => "Checkpoints",
        Screen::Tokenizer => "Tokenizer",
        Screen::Result => "Results",
    }
}
fn status(status: JobStatus) -> (&'static str, Color) {
    match status {
        JobStatus::Starting => ("STARTING", WARN),
        JobStatus::Running => ("RUNNING", ACCENT),
        JobStatus::Stopped => ("STOPPED", WARN),
        JobStatus::Completed => ("COMPLETED", BLUE),
        JobStatus::Failed => ("FAILED", RED),
    }
}

pub(super) fn render(app: &App, frame: &mut Frame) {
    frame.render_widget(
        Block::default().style(Style::default().bg(BG).fg(TEXT)),
        frame.area(),
    );
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(if frame.area().width < 100 { 5 } else { 4 }),
    ])
    .split(frame.area());
    let dirty = if app.text != app.original {
        "  • unsaved"
    } else {
        ""
    };
    let project = app
        .records
        .iter()
        .find(|r| app.screen == Screen::Job && Some(&r.id) == app.selected_job.as_ref())
        .map(|r| &r.project)
        .unwrap_or(&app.cwd);
    let header = Line::from(vec![
        Span::styled(" OMEGA ", Style::default().fg(ACCENT).bold()),
        Span::styled(
            format!(" / {}  ", label(app.screen)),
            Style::default().fg(TEXT),
        ),
        Span::styled(
            format!("{}{dirty}", project.display()),
            Style::default().fg(MUTED),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(header).block(panel("Training workspace")),
        areas[0],
    );
    let body = if areas[1].width >= 100 {
        let cols = Layout::horizontal([Constraint::Length(23), Constraint::Min(0)]).split(areas[1]);
        sidebar(app, frame, cols[0]);
        cols[1]
    } else {
        areas[1]
    };
    match app.screen {
        Screen::Job => job(app, frame, body),
        Screen::Project | Screen::Forms | Screen::Welcome | Screen::Jobs | Screen::Checkpoints => {
            browser(app, frame, body)
        }
        Screen::Editor => editor(app, frame, body),
        _ => paragraph(frame, body, label(app.screen), &app.buffer, app.scroll),
    }
    let help = match app.screen {
        Screen::Welcome => {
            "Enter open  w path  o file  c create  i import  Backspace parent  j jobs  q Detach"
        }
        Screen::Editor => "Ctrl+S validate/save  Esc return  Ctrl+C Detach",
        Screen::Job => {
            "Tab/1–4 view  ↑↓ scroll  End follow  s Save checkpoint and stop  r Resume  q Detach"
        }
        Screen::Review => {
            "Enter START reviewed operation  Esc cancel  ↑↓/PgUp/PgDn scroll  q Detach"
        }
        Screen::Confirm => {
            "Enter CONFIRM displayed action  Esc cancel  ↑↓/PgUp/PgDn scroll  q Detach"
        }
        Screen::Tokenizer => "e encode  d decode  ↑↓ scroll  Esc back  q Detach",
        _ => {
            "↑↓ select  Enter open  Ctrl+S save  Esc back  F2 project  F3 jobs  F4 workspace  q Detach"
        }
    };
    let message = if app.message.is_empty() {
        "Workers run independently. Detaching leaves training running."
    } else {
        &app.message
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(help, Style::default().fg(ACCENT)),
            Line::styled(
                message,
                Style::default().fg(if app.busy { WARN } else { TEXT }),
            ),
        ])
        .block(panel(if app.busy { "Working…" } else { "Controls" }))
        .wrap(Wrap { trim: false }),
        areas[2],
    );
    if let Some(input) = &app.input {
        input_dialog(app, frame, input);
    }
}

fn sidebar(app: &App, frame: &mut Frame, area: Rect) {
    let sections = Layout::vertical([Constraint::Min(8), Constraint::Length(8)]).split(area);
    let current = match app.screen {
        Screen::Welcome => 2,
        Screen::Jobs | Screen::Job => 1,
        _ => 0,
    };
    let nav = ["F2  Project", "F3  Jobs", "F4  Workspace"];
    let mut lines = vec![
        Line::styled("  NAVIGATION", Style::default().fg(MUTED)),
        Line::default(),
    ];
    for (i, item) in nav.iter().enumerate() {
        lines.push(Line::styled(
            format!("{} {item}", if i == current { "›" } else { " " }),
            Style::default().fg(if i == current { ACCENT } else { TEXT }),
        ));
        lines.push(Line::default());
    }
    lines.extend([
        Line::styled("  WORKFLOW", Style::default().fg(MUTED)),
        Line::default(),
        Line::from("  Data & tokenizer"),
        Line::from("  Model & training"),
        Line::from("  Evaluate & chat"),
    ]);
    frame.render_widget(Paragraph::new(lines).block(panel("Omega")), sections[0]);
    let active = app
        .records
        .iter()
        .filter(|r| matches!(r.status, JobStatus::Starting | JobStatus::Running))
        .count();
    paragraph(
        frame,
        sections[1],
        "Host",
        format!(
            "{active} active pipeline\n{} recorded jobs\n\nq  Detach safely\nCtrl+C  Detach",
            app.records.len()
        ),
        0,
    );
}

fn browser(app: &App, frame: &mut Frame, area: Rect) {
    let (items, details) = match app.screen {
        Screen::Welcome => (app.entries.iter().map(|p| format!("{} {}", if p.is_dir() { "▸" } else { "·" }, name(p)))
            .chain(app.recent.iter().map(|p| format!("recent  {}", p.display()))).collect(),
            "Choose your workspace\n\nBrowse with arrows and Enter, or press w to enter a path.\n\no  Open model.toml\nc  Create a project here\ni  Import a dataset recipe\n\nRecent projects appear after directory entries.\n\nF3  Reconnect to a running job".into()),
        Screen::Project => (ACTIONS.iter().map(|s| s.to_string()).collect(), project_details(app)),
        Screen::Forms => {
            let values = app.config().ok().and_then(|c| toml::Value::try_from(c).ok());
            let rows = FIELDS.iter().map(|(section, key)| {
                let v = values.as_ref().and_then(|v| if section.is_empty() { v.get(*key) } else { v.get(*section)?.get(*key) });
                format!("{}{}{}  =  {}", section, if section.is_empty() { "" } else { "." }, key,
                    v.map(ToString::to_string).unwrap_or_else(|| "—".into()))
            }).collect();
            let (section, key) = FIELDS[app.cursor.min(FIELDS.len() - 1)];
            (rows, format!("Edit {section}.{key}\n\nEnter opens the selected value.\nCtrl+S validates and saves all changes.\nEsc returns to the project.\n\nStrings: enter plain text.\nLists: use TOML, e.g. [\"corpus\"].\nBooleans: true or false.\n\nComments are preserved. Unsaved changes must be saved before launch."))
        }
        Screen::Jobs => (app.records.iter().map(|r| format!("{}  {}  {}", status(r.status).0, r.stage, r.id)).collect(),
            app.records.get(app.cursor).map(|r| format!("{}\n\nJob {}\nStage: {}\nProject: {}\n\nEnter reconnects to this job.\nClosing the interface leaves workers running.", status(r.status).0, r.id, r.stage, r.project.display()))
                .unwrap_or_else(|| "No jobs yet.\n\nOpen a project and review a pipeline to start training.".into())),
        Screen::Checkpoints => (app.checkpoints.iter().map(|p| name(p)).collect(),
            app.checkpoints.get(app.cursor).map(|p| format!("{}\n\n{}\n\nEnter selects this checkpoint. Then choose resume, evaluate, generate or chat in the project menu.", name(p), p.display()))
                .unwrap_or_else(|| "No complete checkpoints found for this model.".into())),
        _ => unreachable!(),
    };
    if area.width >= 76 {
        let cols = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);
        list(frame, cols[0], label(app.screen), items, app.cursor);
        paragraph(frame, cols[1], "Details", details, 0);
    } else {
        list(frame, area, label(app.screen), items, app.cursor);
    }
}
fn project_details(app: &App) -> String {
    if app.cursor >= 18 {
        let details = match app.cursor {
            18 => {
                "Explicitly migrate schema 1 to schema 2. Then use Edit model.toml to configure [post_training]. Existing snapshots are preserved."
            }
            19 => {
                "Check parent checkpoint, training and development partitions, and sealed partition separation using the current configuration."
            }
            20 => {
                "Review explicit inputs, limits and gates, then Enter starts a detached worker with the frozen configuration. Human review and promotion remain separate."
            }
            21 => {
                "Enter a workflow directory to inspect its saved state, ranked candidates, gate failures and complete reports with actual prompts and replies."
            }
            22 => {
                "Enter a JSON request file path:\n{\"workflow\":\"run\",\"report\":\"run/report.json\",\"scores\":\"scores.json\"}\n\nPaths resolve beside the request file. Scores use the HumanReview schema. Inspect the displayed scores, then explicitly confirm submission."
            }
            23 => {
                "Enter a JSON request file path:\n{\"workflow\":\"run\",\"report\":\"run/report.json\",\"approved_by\":\"Your name\",\"output\":\"selection.json\"}\n\nPaths resolve beside the request file. Only eligible candidates can be confirmed. Output must be a new file."
            }
            _ => {
                "Enter an existing workflow directory. Inspect its frozen configuration and retained executable before explicitly confirming recovery. Exhausted budgets cannot be renewed here."
            }
        };
        return format!(
            "{}\n\n{details}\n\nConfigure post-training settings in Edit model.toml; Ctrl+S validates and saves.",
            ACTIONS[app.cursor.min(ACTIONS.len() - 1)]
        );
    }
    match app.config() {
        Ok(c) => {
            let stages = [
                (c.pipeline.prepare_dataset, "Dataset"),
                (c.pipeline.prepare_tokenizer, "Tokenizer"),
                (c.pipeline.prepare_cache, "Token cache"),
                (c.pipeline.benchmark, "Benchmark"),
                (c.pipeline.train, "Base training"),
                (c.assistant.is_some(), "Assistant training"),
                (c.pipeline.evaluate, "Evaluation"),
            ]
            .iter()
            .filter(|(on, _)| *on)
            .map(|(_, s)| *s)
            .collect::<Vec<_>>()
            .join(" → ");
            format!(
                "{}\n\nBACKEND\n{:?} · device {}\n\nMODEL\n{} layers · {} heads · width {}\nContext {} · batch {}\n{} epochs · learning rate {}\n\nSELECTED PIPELINE\n{}\n\nDATASETS\n{}\n\nCHECKPOINT\n{}\n\nEnter reviews the selected action before any worker starts.",
                c.name,
                c.training.backend,
                c.training.device,
                c.model.layers,
                c.model.heads,
                c.model.d_model,
                c.model.context_length,
                c.training.batch_size,
                c.training.epochs,
                c.training.learning_rate,
                if stages.is_empty() {
                    "No stages selected"
                } else {
                    &stages
                },
                c.selections().join(", "),
                app.checkpoint
                    .as_deref()
                    .map(name)
                    .unwrap_or_else(|| "None selected".into())
            )
        }
        Err(e) => {
            format!("Configuration needs attention\n\n{e:#}\n\nOpen the TOML editor to correct it.")
        }
    }
}

fn job(app: &App, frame: &mut Frame, area: Rect) {
    let Some(r) = app
        .records
        .iter()
        .find(|r| Some(&r.id) == app.selected_job.as_ref())
    else {
        paragraph(
            frame,
            area,
            "Training monitor",
            "Waiting for worker status…\nF3 opens recorded jobs.",
            0,
        );
        return;
    };
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(0),
    ])
    .split(area);
    let (state, color) = status(r.status);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!("● {state}  "), Style::default().fg(color).bold()),
            Span::raw(format!("{}  ·  {}", r.stage, r.id)),
        ]))
        .block(panel("Worker")),
        rows[0],
    );
    frame.render_widget(
        Tabs::new(["1 Overview", "2 Logs", "3 Checkpoints", "4 Results"])
            .select(app.job_tab)
            .highlight_style(Style::default().fg(ACCENT).bold())
            .style(Style::default().fg(MUTED))
            .block(panel("Views · Tab to switch")),
        rows[1],
    );
    let body = rows[2];
    match app.job_tab {
        1 => log(app, frame, body),
        2 => {
            let mut text = format!(
                "Latest complete checkpoint\n{}\n\nRecent saves (newest first; up to 100)\n",
                r.checkpoint
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "No checkpoint saved yet".into())
            );
            for p in &app.dashboard.checkpoints {
                text.push_str(&format!("\n{}\n  {}\n", name(p), p.display()));
            }
            if !app.dashboard.notice.is_empty() {
                text.push_str(&format!("\n{}", app.dashboard.notice));
            }
            paragraph(frame, body, "Checkpoint history", text, app.scroll);
        }
        3 => {
            let result = r
                .result
                .as_ref()
                .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
                .unwrap_or_else(|| "No result reported yet.".into());
            paragraph(
                frame,
                body,
                "Results · c continue chat / n new chat",
                format!(
                    "{}\n\n{result}",
                    r.error
                        .as_deref()
                        .unwrap_or("Worker results and evaluation output")
                ),
                app.scroll,
            );
        }
        _ => overview(app, r, frame, body),
    }
}
fn overview(app: &App, r: &JobRecord, frame: &mut Frame, area: Rect) {
    let elapsed = number(&r.progress, "elapsed_seconds");
    let run = number(&r.progress, "run_updates").filter(|n| *n > 0.0);
    let updates = r.progress["completed_updates"].as_u64();
    let eta = if matches!(r.status, JobStatus::Running) {
        elapsed
            .zip(run)
            .zip(r.total_updates.zip(updates))
            .map(|((s, run), (total, done))| s * total.saturating_sub(done) as f64 / run)
    } else {
        None
    };
    let throughput = elapsed
        .filter(|s| *s > 0.0)
        .zip(number(&r.progress, "run_targets"))
        .map(|(s, n)| n / s);
    let done = updates.map(count).unwrap_or_else(|| "—".into());
    let total = r.total_updates.map(count).unwrap_or_else(|| "—".into());
    let loss = number(&r.progress, "loss")
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "—".into());
    let speed = throughput
        .map(|v| format!("{v:.1} targets/s"))
        .unwrap_or_else(|| "—".into());
    let epoch = r.progress["epoch"]
        .as_u64()
        .map(count)
        .unwrap_or_else(|| "—".into());
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(6),
        Constraint::Min(0),
    ])
    .split(area);
    let ratio = updates
        .zip(r.total_updates.filter(|n| *n > 0))
        .map(|(done, total)| (done as f64 / total as f64).clamp(0.0, 1.0));
    frame.render_widget(
        Gauge::default()
            .block(panel("Committed updates"))
            .gauge_style(Style::default().fg(ACCENT).bg(PANEL))
            .ratio(ratio.unwrap_or(0.0))
            .label(format!(
                "{done} / {total}{}",
                ratio
                    .map(|r| format!("  ·  {:.2}%", r * 100.0))
                    .unwrap_or_default()
            )),
        rows[0],
    );
    let cards = Layout::horizontal([
        Constraint::Percentage(34),
        Constraint::Percentage(33),
        Constraint::Percentage(33),
    ])
    .split(rows[1]);
    paragraph(
        frame,
        cards[0],
        "Training",
        format!(
            "Loss    {loss}\nEpoch   {epoch}\nLR      {}",
            number(&r.progress, "learning_rate")
                .map(|v| format!("{v:.2e}"))
                .unwrap_or_else(|| "—".into())
        ),
        0,
    );
    paragraph(
        frame,
        cards[1],
        "Throughput",
        format!(
            "{speed}\nRun average\nGrad norm  {}",
            number(&r.progress, "gradient_norm")
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into())
        ),
        0,
    );
    paragraph(
        frame,
        cards[2],
        "Timing",
        format!(
            "Elapsed  {}\nETA      {}\nUpdate-based estimate",
            duration(elapsed),
            duration(eta)
        ),
        0,
    );
    let rest =
        Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).split(rows[2]);
    loss_chart(app, frame, rest[0]);
    if rest[1].width >= 70 {
        let cols = Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(rest[1]);
        if let Some(error) = &r.error {
            paragraph(frame, cols[0], "Worker failed", error, 0);
        } else if !app.dashboard.notice.is_empty() {
            paragraph(
                frame,
                cols[0],
                "History unavailable",
                &app.dashboard.notice,
                0,
            );
        } else {
            log(app, frame, cols[0]);
        }
        let paths = if app.dashboard.checkpoints.is_empty() {
            r.checkpoint.iter().collect::<Vec<_>>()
        } else {
            app.dashboard.checkpoints.iter().take(8).collect()
        };
        paragraph(
            frame,
            cols[1],
            "Recent checkpoints",
            if paths.is_empty() {
                "No saves yet".into()
            } else {
                paths.iter().map(|p| name(p)).collect::<Vec<_>>().join("\n")
            },
            0,
        );
    } else if let Some(error) = &r.error {
        paragraph(frame, rest[1], "Worker failed", error, 0);
    } else {
        log(app, frame, rest[1]);
    }
}
fn log(app: &App, frame: &mut Frame, area: Rect) {
    let lines: Vec<_> = app.dashboard.log.lines().collect();
    let height = area.height.saturating_sub(2) as usize;
    let max_back = lines.len().saturating_sub(height);
    let start = max_back.saturating_sub(app.scroll as usize);
    let title = if app.scroll == 0 {
        "Worker log · following"
    } else {
        "Worker log · history / End to follow"
    };
    // Each log entry occupies one terminal row, keeping tail scrolling predictable.
    frame.render_widget(
        Paragraph::new(if lines.is_empty() {
            "No worker output yet.".into()
        } else {
            lines
                .into_iter()
                .skip(start)
                .take(height)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .block(panel(title))
        .style(Style::default().fg(MUTED)),
        area,
    );
}
fn loss_chart(app: &App, frame: &mut Frame, area: Rect) {
    let data = &app.dashboard.losses;
    if data.len() < 2 || area.height < 6 || area.width < 25 {
        paragraph(
            frame,
            area,
            "Loss · recent committed updates",
            "Waiting for loss history…",
            0,
        );
        return;
    }
    let min = data.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max = data.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let pad = ((max - min) * 0.1).max(0.01);
    let y0 = (min - pad).max(0.0);
    let y1 = max + pad;
    let x0 = data[0].0;
    let x1 = data[data.len() - 1].0.max(x0 + 1.0);
    let set = Dataset::default()
        .data(data)
        .graph_type(GraphType::Line)
        .marker(Marker::Braille)
        .style(Style::default().fg(ACCENT));
    frame.render_widget(
        Chart::new(vec![set])
            .block(panel("Loss · recent committed updates"))
            .x_axis(
                Axis::default()
                    .bounds([x0, x1])
                    .style(Style::default().fg(EDGE))
                    .labels([format!("{x0:.0}"), format!("{x1:.0}")]),
            )
            .y_axis(
                Axis::default()
                    .bounds([y0, y1])
                    .style(Style::default().fg(EDGE))
                    .labels([format!("{y0:.2}"), format!("{y1:.2}")]),
            ),
        area,
    );
}

fn editor(app: &App, frame: &mut Frame, area: Rect) {
    let prefix: String = app.text.chars().take(app.editor_cursor).collect();
    let line = prefix
        .chars()
        .filter(|c| *c == '\n')
        .count()
        .min(u16::MAX as usize) as u16;
    let col = Line::from(prefix.rsplit('\n').next().unwrap_or(""))
        .width()
        .min(u16::MAX as usize) as u16;
    let scroll = line.saturating_sub(area.height.saturating_sub(4));
    let horizontal = col.saturating_sub(area.width.saturating_sub(4));
    // Do not wrap source lines: the editor cursor must match the displayed text.
    frame.render_widget(
        Paragraph::new(app.text.as_str())
            .block(panel("model.toml · Ctrl+S save"))
            .scroll((scroll, horizontal)),
        area,
    );
    if area.width > 2 && area.height > 2 {
        frame.set_cursor_position((
            area.x + 1 + col.saturating_sub(horizontal).min(area.width - 3),
            area.y + 1 + line.saturating_sub(scroll).min(area.height - 3),
        ));
    }
}
fn input_dialog(app: &App, frame: &mut Frame, input: &Input) {
    let title = match input {
        Input::Directory => "Working directory",
        Input::Open => "Open model.toml",
        Input::Import => "Import dataset recipe",
        Input::Form(_) => "Edit setting",
        Input::Prompt(_) => "Prompt",
        Input::Encode => "Encode text",
        Input::Decode => "Decode token IDs",
        Input::Reports => "Workflow directory · ranked reports",
        Input::HumanReview => "Review request JSON path · workflow / report / scores",
        Input::Promotion => {
            "Promotion request JSON path · workflow / report / approved_by / output"
        }
        Input::Recover => "Workflow directory · review recovery",
    };
    let bounds = frame.area();
    let width = bounds.width.saturating_sub(4).min(90);
    let height = bounds.height.min(7);
    let area = Rect::new(
        bounds.x + (bounds.width - width) / 2,
        bounds.y + (bounds.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, area);
    paragraph(
        frame,
        area,
        title,
        format!("{}\n\nEnter submit   Esc cancel", app.buffer),
        0,
    );
}
