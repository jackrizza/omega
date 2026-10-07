//! Bounded, read-only presentation data. No worker control or training state lives here.
use crate::jobs;
use std::{
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Default)]
pub(super) struct Dashboard {
    pub id: String,
    pub losses: Vec<(f64, f64)>,
    pub checkpoints: Vec<PathBuf>,
    pub log: String,
    pub notice: String,
    checkpoint_stamp: Option<(u64, SystemTime)>,
}
impl Dashboard {
    pub fn refresh(&mut self, root: &Path, id: &str) {
        if self.id != id {
            *self = Self {
                id: id.into(),
                ..Self::default()
            };
        }
        self.notice.clear();
        let Ok(dir) = jobs::job_dir(root, id) else {
            return;
        };
        match jobs::events(&dir.join("events.jsonl")) {
            Ok(events) => {
                self.losses.clear();
                for event in events {
                    if event.kind == "stage" {
                        self.losses.clear();
                    }
                    if event.kind == "update"
                        && let (Some(x), Some(y)) = (
                            number(&event.data, "completed_updates"),
                            number(&event.data, "loss"),
                        )
                    {
                        self.losses.push((x, y));
                    }
                }
                let excess = self.losses.len().saturating_sub(240);
                self.losses.drain(..excess);
            }
            Err(e) => self.read_error("Loss history", &dir.join("events.jsonl"), &e),
        }
        match jobs::tail(&dir.join("worker.log"), 32 * 1024) {
            Ok(log) => self.log = log,
            Err(e) => self.read_error("Worker log", &dir.join("worker.log"), &e),
        }
        let path = dir.join("checkpoints.json");
        let stamp = fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok().map(|t| (m.len(), t)));
        if stamp != self.checkpoint_stamp || stamp.is_none() {
            match jobs::read_json::<Vec<PathBuf>>(&path) {
                Ok(mut paths) => {
                    paths.reverse();
                    paths.truncate(100);
                    self.checkpoints = paths;
                    self.checkpoint_stamp = stamp;
                }
                Err(e) => self.read_error("Checkpoint history", &path, &e),
            }
        }
    }
    fn read_error(&mut self, label: &str, path: &Path, error: &anyhow::Error) {
        if error
            .downcast_ref::<std::io::Error>()
            .is_none_or(|e| e.kind() != std::io::ErrorKind::NotFound)
        {
            self.notice = format!("{label} unavailable ({}): {error}", path.display());
        }
    }
}
pub(super) fn number(value: &serde_json::Value, key: &str) -> Option<f64> {
    value[key].as_f64().filter(|x| x.is_finite() && *x >= 0.0)
}
pub(super) fn duration(seconds: Option<f64>) -> String {
    let Some(s) = seconds.filter(|s| s.is_finite() && *s >= 0.0) else {
        return "—".into();
    };
    let s = s as u64;
    if s >= 86400 {
        format!("{}d {:02}h {:02}m", s / 86400, s / 3600 % 24, s / 60 % 60)
    } else if s >= 3600 {
        format!("{}h {:02}m {:02}s", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}m {:02}s", s / 60, s % 60)
    }
}
pub(super) fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
