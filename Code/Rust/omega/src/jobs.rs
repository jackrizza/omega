//! Same-binary workers; local durable records and private Unix control sockets.
use crate::ProjectConfig;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
#[cfg(target_os = "linux")]
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Pipeline,
    PrepareDataset,
    TrainTokenizer,
    PrepareCache,
    Benchmark,
    Train,
    Assistant,
    Resume,
    Evaluate,
    Generate,
    Chat,
    PostTrain,
    PostTrainRecover,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobSpec {
    pub schema_version: u32,
    pub id: String,
    pub project: PathBuf,
    pub config_path: PathBuf,
    pub config: ProjectConfig,
    pub action: Action,
    pub checkpoint: Option<PathBuf>,
    pub prompt: Option<String>,
    pub executable: PathBuf,
    pub executable_sha256: String,
    pub created_ms: u64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Starting,
    Running,
    Stopped,
    Completed,
    Failed,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub boot: String,
    pub start: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub status: JobStatus,
    pub process: Option<ProcessIdentity>,
    pub project: PathBuf,
    pub stage: String,
    pub started_ms: u64,
    pub updated_ms: u64,
    pub progress: serde_json::Value,
    pub total_updates: Option<u64>,
    pub checkpoint: Option<PathBuf>,
    pub error: Option<String>,
    pub result: Option<serde_json::Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobEvent {
    pub time_ms: u64,
    pub kind: String,
    pub data: serde_json::Value,
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
pub fn state_root() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("OMEGA_STATE_DIR") {
        ensure!(
            Path::new(&path).is_absolute(),
            "OMEGA_STATE_DIR must be absolute and host-local"
        );
        return Ok(path.into());
    }
    if let Some(path) = std::env::var_os("XDG_STATE_HOME") {
        ensure!(
            Path::new(&path).is_absolute(),
            "XDG_STATE_HOME must be absolute and host-local"
        );
        return Ok(PathBuf::from(path).join("omega"));
    }
    Ok(PathBuf::from(
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .context("Cannot determine home directory")?,
    )
    .join(".local/state/omega"))
}
pub fn private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    ensure!(
        !fs::symlink_metadata(path)?.file_type().is_symlink(),
        "State directory cannot be a symlink"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        ensure!(
            fs::metadata(path)?.uid() == unsafe { libc::geteuid() },
            "State directory belongs to another user"
        );
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
pub fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let temp = path.with_extension(format!("{}.new", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    if let Err(e) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temp);
        return Err(e.into());
    }
    drop(file);
    if let Err(e) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(e.into());
    }
    Ok(())
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= 8 * 1024 * 1024,
        "Job metadata exceeds 8 MiB"
    );
    Ok(serde_json::from_reader(file)?)
}
pub fn job_dir(root: &Path, id: &str) -> Result<PathBuf> {
    ensure!(
        !id.is_empty() && id.len() < 80 && id.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-'),
        "Invalid job ID"
    );
    Ok(root.join("jobs").join(id))
}
pub fn process_identity(pid: u32) -> Result<ProcessIdentity> {
    #[cfg(target_os = "linux")]
    {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
        let fields: Vec<_> = stat
            .rsplit_once(") ")
            .context("Invalid process stat")?
            .1
            .split_whitespace()
            .collect();
        ensure!(fields.first() != Some(&"Z"), "Worker is a zombie");
        Ok(ProcessIdentity {
            pid,
            boot: fs::read_to_string("/proc/sys/kernel/random/boot_id")?
                .trim()
                .into(),
            start: fields
                .get(19)
                .context("Missing process start time")?
                .to_string(),
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        anyhow::bail!("Persistent workers require Linux")
    }
}
pub fn list(root: &Path) -> Result<Vec<JobRecord>> {
    let jobs = root.join("jobs");
    if !jobs.exists() {
        return Ok(vec![]);
    }
    let mut records = vec![];
    for entry in fs::read_dir(jobs)?.take(10_000) {
        let path = entry?.path().join("status.json");
        if let Ok(mut r) = read_json::<JobRecord>(&path) {
            if matches!(r.status, JobStatus::Starting | JobStatus::Running)
                && (r
                    .process
                    .as_ref()
                    .is_some_and(|p| process_identity(p.pid).ok().as_ref() != Some(p))
                    || (r.process.is_none() && now_ms().saturating_sub(r.started_ms) > 300_000))
            {
                r.status = JobStatus::Failed;
                r.error=Some("Worker exited without a completed boundary; resume the last complete checkpoint manually".into());
                atomic_json(&path, &r)?;
            }
            records.push(r);
        }
    }
    records.sort_by_key(|r| std::cmp::Reverse(r.started_ms));
    Ok(records)
}
pub fn tail(path: &Path, limit: usize) -> Result<String> {
    use std::io::{Seek, SeekFrom};
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let start = length.saturating_sub(limit as u64);
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.take(limit as u64).read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
pub fn events(path: &Path) -> Result<Vec<JobEvent>> {
    let text = tail(path, 256 * 1024)?;
    Ok(text
        .split_inclusive('\n')
        .filter(|line| line.ends_with('\n'))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect())
}

pub struct Reporter {
    pub record: Mutex<JobRecord>,
    directory: PathBuf,
}
impl Reporter {
    pub fn new(directory: PathBuf, record: JobRecord) -> Self {
        Self {
            directory,
            record: Mutex::new(record),
        }
    }
    pub fn emit(&self, kind: &str, data: serde_json::Value) -> Result<()> {
        let mut r = self
            .record
            .lock()
            .map_err(|_| anyhow::anyhow!("Job record lock poisoned"))?;
        r.updated_ms = now_ms();
        match kind {
            "stage" => {
                r.stage = data["name"].as_str().unwrap_or("operation").into();
                r.progress = serde_json::Value::Null;
                r.total_updates = None;
            }
            "training_started" => r.total_updates = data["total_updates"].as_u64(),
            "update" => r.progress = data.clone(),
            "checkpoint" => {
                let checkpoint =
                    PathBuf::from(data["path"].as_str().context("Missing checkpoint path")?);
                omega_training::checkpoint::read_checkpoint_manifest(&checkpoint)
                    .map_err(anyhow::Error::msg)?;
                let index = self.directory.join("checkpoints.json");
                let mut history: Vec<PathBuf> = if index.exists() {
                    read_json(&index)?
                } else {
                    vec![]
                };
                if !history.contains(&checkpoint) {
                    history.push(checkpoint.clone());
                    atomic_json(&index, &history)?;
                }
                r.checkpoint = Some(checkpoint);
            }
            "result" | "generation" | "evaluation" | "chat" => r.result = Some(data.clone()),
            _ => {}
        }
        let event = JobEvent {
            time_ms: r.updated_ms,
            kind: kind.into(),
            data,
        };
        let path = self.directory.join("events.jsonl");
        if fs::metadata(&path).is_ok_and(|m| m.len() > 16 * 1024 * 1024) {
            fs::rename(&path, self.directory.join("events.previous.jsonl"))?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, &event)?;
        file.write_all(b"\n")?;
        file.flush()?;
        atomic_json(&self.directory.join("status.json"), &*r)
    }
    pub fn checkpoint(&self) -> Option<PathBuf> {
        self.record.lock().ok().and_then(|r| r.checkpoint.clone())
    }
}

/// Resolve every checkpoint produced by a job, including periodic saves that are
/// no longer its latest checkpoint. Never depend on a rotated event-log tail.
pub fn retained_for_checkpoint(root: &Path, checkpoint: &Path) -> Result<Option<PathBuf>> {
    let requested = fs::canonicalize(checkpoint)?;
    for record in list(root)? {
        let directory = job_dir(root, &record.id)?;
        let history: Vec<PathBuf> =
            read_json(&directory.join("checkpoints.json")).unwrap_or_default();
        if history
            .iter()
            .any(|p| fs::canonicalize(p).ok().as_ref() == Some(&requested))
        {
            let spec: JobSpec = read_json(&directory.join("job.json"))?;
            ensure!(
                spec.executable.is_file(),
                "Retained worker executable is missing: {}",
                spec.executable.display()
            );
            return Ok(Some(spec.executable));
        }
    }
    Ok(None)
}

#[cfg(target_os = "linux")]
fn socket_path(id: &str) -> Result<PathBuf> {
    let base = PathBuf::from("/tmp");
    let dir = base.join(format!("omega-{}", unsafe { libc::geteuid() }));
    private_dir(&dir)?;
    let path = dir.join(format!("{id}.sock"));
    ensure!(
        path.as_os_str().len() < 104,
        "Runtime socket path is too long"
    );
    Ok(path)
}

pub fn request(root: &Path, id: &str, command: &str) -> Result<JobRecord> {
    let _ = job_dir(root, id)?;
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::net::UnixStream;
        let mut stream = UnixStream::connect(socket_path(id)?)
            .context("Worker is not accepting connections; inspect its persistent status/log")?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        stream.set_write_timeout(Some(Duration::from_secs(2)))?;
        stream.write_all(format!("{command}\n").as_bytes())?;
        let mut response = String::new();
        stream.take(128 * 1024).read_to_string(&mut response)?;
        Ok(serde_json::from_str(&response)?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = command;
        anyhow::bail!("Workers require Linux")
    }
}

pub fn launch(
    root: &Path,
    config_path: &Path,
    action: Action,
    checkpoint: Option<PathBuf>,
    prompt: Option<String>,
    retained: Option<&Path>,
    approved: Option<ProjectConfig>,
) -> Result<JobSpec> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            root,
            config_path,
            action,
            checkpoint,
            prompt,
            retained,
            approved,
        );
        anyhow::bail!("Persistent workers are supported on Linux x64 only")
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::{fs::PermissionsExt, process::CommandExt};
        private_dir(root)?;
        private_dir(&root.join("jobs"))?;
        private_dir(&root.join("executables"))?;
        let launch_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(socket_path("launch")?.with_extension("lock"))?;
        launch_lock
            .try_lock()
            .context("Another launch is in progress")?;
        let compute = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(socket_path("compute")?.with_extension("lock"))?;
        compute.try_lock().context("An Omega compute pipeline is already running; detach/reconnect or save and stop it first")?;
        drop(compute);
        let config_path = fs::canonicalize(config_path)?;
        let project = config_path
            .parent()
            .context("Project has no parent")?
            .to_path_buf();
        let config = match approved {
            Some(c) => c,
            None => ProjectConfig::load(&config_path)?,
        };
        crate::pipeline::validate_launch(&config, &action, &checkpoint)?;
        let source = retained
            .map(PathBuf::from)
            .unwrap_or(std::env::current_exe()?);
        let hash = hash_file(&source)?;
        let executable = root.join("executables").join(&hash);
        if !executable.exists() {
            let mut f = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&executable)?;
            std::io::copy(&mut File::open(&source)?, &mut f)?;
            f.sync_all()?;
            fs::set_permissions(&executable, fs::Permissions::from_mode(0o500))?;
        }
        ensure!(
            hash_file(&executable)? == hash,
            "Retained executable checksum differs"
        );
        let id = format!("{:x}-{:x}", now_ms(), std::process::id());
        let directory = job_dir(root, &id)?;
        fs::create_dir(&directory)?;
        let spec = JobSpec {
            schema_version: 1,
            id: id.clone(),
            project: project.clone(),
            config_path,
            config: config.clone(),
            action,
            checkpoint,
            prompt,
            executable: executable.clone(),
            executable_sha256: hash,
            created_ms: now_ms(),
        };
        atomic_json(&directory.join("job.json"), &spec)?;
        let record = JobRecord {
            id: id.clone(),
            status: JobStatus::Starting,
            process: None,
            project,
            stage: "starting".into(),
            started_ms: now_ms(),
            updated_ms: now_ms(),
            progress: serde_json::Value::Null,
            total_updates: None,
            checkpoint: None,
            error: None,
            result: None,
        };
        atomic_json(&directory.join("status.json"), &record)?;
        let log = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.join("worker.log"))?;
        let mut cmd = std::process::Command::new(executable);
        cmd.arg("__worker")
            .arg("--state")
            .arg(root)
            .arg("--id")
            .arg(&id)
            .current_dir(&spec.project)
            .stdin(std::process::Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log);
        config
            .cpu()
            .apply_to_command(&mut cmd)
            .map_err(anyhow::Error::msg)?;
        // The child performs only the async-signal-safe setsid call before exec.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        let mut child = cmd.spawn().context("Start detached worker")?;
        let identity = process_identity(child.id()).ok();
        // Wait for the worker to own its record/lock before another launch.
        for _ in 0..600 {
            if let Ok(r) = read_json::<JobRecord>(&directory.join("status.json"))
                && r.process.is_some()
            {
                break;
            }
            if let Some(status) = child.try_wait()? {
                anyhow::bail!(
                    "Worker failed to start ({status}): {}",
                    tail(&directory.join("worker.log"), 4096).unwrap_or_default()
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let r: JobRecord = read_json(&directory.join("status.json"))?;
        ensure!(
            r.process == identity && r.process.is_some(),
            "Worker startup is unconfirmed; inspect job {} before retrying",
            id
        );
        std::thread::spawn(move || {
            let _ = child.wait();
        });
        Ok(spec)
    }
}

pub fn worker(root: &Path, id: &str) -> Result<()> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (root, id);
        anyhow::bail!("Workers require Linux")
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::net::UnixListener;
        let directory = job_dir(root, id)?;
        let spec: JobSpec = read_json(&directory.join("job.json"))?;
        ensure!(
            spec.schema_version == 1 && spec.id == id,
            "Unsupported/mismatched job snapshot"
        );
        spec.config.validate()?;
        ensure!(
            hash_file(&std::env::current_exe()?)? == spec.executable_sha256,
            "Worker executable differs from frozen job"
        );
        let compute = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(socket_path("compute")?.with_extension("lock"))?;
        compute
            .try_lock()
            .context("Another compute pipeline holds the worker lock")?;
        let mut record: JobRecord = read_json(&directory.join("status.json"))?;
        ensure!(
            record.status == JobStatus::Starting,
            "Job already started; launch a new explicit resume instead"
        );
        record.process = Some(process_identity(std::process::id())?);
        record.status = JobStatus::Running;
        let reporter = Arc::new(Reporter {
            record: Mutex::new(record),
            directory: directory.clone(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        ctrlc::set_handler(move || signal.store(true, Ordering::SeqCst))?;
        let socket = socket_path(id)?;
        let listener = UnixListener::bind(&socket)?;
        listener.set_nonblocking(true)?;
        let done = Arc::new(AtomicBool::new(false));
        let thread_done = done.clone();
        let report = reporter.clone();
        let stop_request = stop.clone();
        let server = std::thread::spawn(move || {
            while !thread_done.load(Ordering::SeqCst) {
                if let Ok((mut stream, _)) = listener.accept() {
                    let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
                    let _ = stream.set_write_timeout(Some(Duration::from_millis(250)));
                    let mut bytes = [0u8; 64];
                    if let Ok(count) = stream.read(&mut bytes) {
                        match std::str::from_utf8(&bytes[..count]).unwrap_or("").trim() {
                            "stop" => stop_request.store(true, Ordering::SeqCst),
                            "status" => {}
                            _ => continue,
                        }
                        let bytes = report
                            .record
                            .lock()
                            .ok()
                            .and_then(|r| serde_json::to_vec(&*r).ok());
                        if let Some(bytes) = bytes {
                            let _ = stream.write_all(&bytes);
                        }
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(30));
                }
            }
        });
        reporter.emit("started", serde_json::json!({"action":spec.action}))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::pipeline::execute(&spec, reporter.clone(), stop.clone())
        }))
        .unwrap_or_else(|_| {
            Err(anyhow::anyhow!(
                "Worker operation panicked; last complete checkpoint retained"
            ))
        });
        {
            let mut r = reporter
                .record
                .lock()
                .map_err(|_| anyhow::anyhow!("Job record poisoned"))?;
            r.status = match &result {
                Ok(()) if stop.load(Ordering::SeqCst) => JobStatus::Stopped,
                Ok(()) => JobStatus::Completed,
                Err(e)
                    if stop.load(Ordering::SeqCst)
                        && e.to_string().starts_with("Training interrupted after")
                        && r.checkpoint.is_some() =>
                {
                    JobStatus::Stopped
                }
                Err(_) => JobStatus::Failed,
            };
            r.error = result.as_ref().err().map(|e| format!("{e:#}"));
        }
        let final_result = reporter.emit(
            "finished",
            serde_json::json!({"stop_requested":stop.load(Ordering::SeqCst)}),
        );
        done.store(true, Ordering::SeqCst);
        let _ = server.join();
        let _ = fs::remove_file(socket);
        final_result?;
        result
    }
}

#[cfg(target_os = "linux")]
fn hash_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
