//! Real GPU CLI subprocess qualification, deliberately opt-in and Linux-only.
//! Run with `--features gpu --test gpu_subprocess -- --ignored --nocapture`.
#![cfg(all(feature = "gpu", target_os = "linux"))]

use std::{
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, ExitStatus, Stdio},
    time::{Duration, Instant},
};

use burn::{
    module::Module,
    optim::{Adam, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Record, Recorder},
};
use omega_nn::{Cpu, Gpt, TrainingBackend};
use omega_training::{
    checkpoint::{load_checkpoint, read_checkpoint_manifest},
    checkpoint_catalog::{CatalogLimits, CatalogMode, discover_checkpoints},
    resume::read_resume_manifest,
};

mod program {
    pub fn entry(arguments: Vec<std::ffi::OsString>) -> std::process::ExitCode {
        let _normal_entry: fn() -> std::process::ExitCode = main;
        main_with_args(arguments)
    }
    include!("../src/bin/main.rs");
}

#[test]
fn gpu_cli_child() {
    let Some(count) = std::env::var_os("OMEGA_GPU_CLI_ARG_COUNT") else {
        return;
    };
    let count: usize = count.to_str().unwrap().parse().unwrap();
    let mut arguments = vec![OsString::from("omega-training")];
    arguments.extend(
        (0..count).map(|index| std::env::var_os(format!("OMEGA_GPU_CLI_ARG_{index}")).unwrap()),
    );
    let exit = program::entry(arguments);
    std::process::exit(if exit == ExitCode::SUCCESS { 0 } else { 1 });
}

/// Reap even when an assertion fails while observing a running child.
struct Running {
    child: Option<Child>,
    started: Instant,
    stdout: PathBuf,
    stderr: PathBuf,
}
impl Drop for Running {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
impl Running {
    fn diagnostic(&self) -> String {
        format!(
            "stdout={}\nstderr={}",
            fs::read_to_string(&self.stdout).unwrap_or_default(),
            fs::read_to_string(&self.stderr).unwrap_or_default()
        )
    }
    fn wait(mut self) -> (ExitStatus, String) {
        loop {
            if let Some(status) = self.child.as_mut().unwrap().try_wait().unwrap() {
                // wait() explicitly reaps the process even after try_wait observed its exit.
                self.child.take().unwrap().wait().unwrap();
                return (status, self.diagnostic());
            }
            assert!(
                self.started.elapsed() < Duration::from_secs(30),
                "GPU child exceeded 30 seconds: {}",
                self.diagnostic()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
fn spawn(root: &Path, label: &str, arguments: &[&str]) -> Running {
    // Include the actual CLI source, avoiding shared `main` binary collisions
    // between concurrently built packages or CPU/GPU feature configurations.
    let stdout = root.join(format!("{label}.stdout"));
    let stderr = root.join(format!("{label}.stderr"));
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .current_dir(root)
        .args(["--exact", "gpu_cli_child", "--nocapture"])
        .env("OMEGA_GPU_CLI_ARG_COUNT", arguments.len().to_string())
        .stdout(Stdio::from(File::create(&stdout).unwrap()))
        .stderr(Stdio::from(File::create(&stderr).unwrap()));
    for (index, argument) in arguments.iter().enumerate() {
        command.env(format!("OMEGA_GPU_CLI_ARG_{index}"), argument);
    }
    Running {
        child: Some(command.spawn().unwrap()),
        started: Instant::now(),
        stdout,
        stderr,
    }
}
fn succeeds(root: &Path, label: &str, arguments: &[&str]) {
    let (status, output) = spawn(root, label, arguments).wait();
    assert!(status.success(), "{label}: {status}: {output}");
}
fn prepare(root: &Path) {
    fs::create_dir_all(root.join("data/text")).unwrap();
    fs::write(root.join("data/text/a.txt"), "hello world omega").unwrap();
    fs::write(root.join("data/text/b.txt"), "this is a test").unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"),
        root.join("data/test.json"),
    )
    .unwrap();
}
fn train<'a>(name: &'a str, metrics: &'a str, epochs: &'a str) -> Vec<&'a str> {
    vec![
        "train",
        "--backend",
        "vulkan",
        "--device",
        "0",
        "--name",
        name,
        "--datasets-root",
        "data",
        "--weights-root",
        "weights",
        "--dataset",
        "text",
        "--tokenizer",
        "test.json",
        "--epochs",
        epochs,
        "--context-length",
        "4",
        "--d-model",
        "4",
        "--heads",
        "1",
        "--layers",
        "1",
        "--d-ff",
        "8",
        "--metrics-jsonl",
        metrics,
        "--quiet",
        "--save-every-updates",
        "1",
    ]
}
fn resume<'a>(
    checkpoint: &'a str,
    name: &'a str,
    metrics: &'a str,
    updates: &'a str,
) -> Vec<&'a str> {
    vec![
        "resume",
        "--backend",
        "vulkan",
        "--device",
        "0",
        "--checkpoint",
        checkpoint,
        "--name",
        name,
        "--datasets-root",
        "data",
        "--weights-root",
        "weights",
        "--epochs",
        "10000",
        "--max-updates",
        updates,
        "--metrics-jsonl",
        metrics,
        "--quiet",
        "--save-every-updates",
        "1",
    ]
}
fn rows(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn updates(path: &Path) -> Vec<serde_json::Value> {
    rows(path)
        .into_iter()
        .filter(|row| row["event"] == "update")
        .collect()
}
fn latest(root: &Path, name: &str) -> PathBuf {
    discover_checkpoints(&root.join("weights"), name, &CatalogLimits::default())
        .unwrap()
        .latest(CatalogMode::Resume)
        .unwrap()
        .path
}
fn checkpoint_name(path: &Path) -> &str {
    path.file_name().unwrap().to_str().unwrap()
}
fn model_value(path: &Path) -> serde_json::Value {
    let (model, _, _) = load_checkpoint(path).unwrap();
    serde_json::to_value(<_ as Record<Cpu>>::into_item::<FullPrecisionSettings>(
        model.into_record(),
    ))
    .unwrap()
}
fn optimizer_value(path: &Path) -> serde_json::Value {
    type Optim = OptimizerAdaptor<Adam, Gpt<TrainingBackend>, TrainingBackend>;
    type State = <Optim as Optimizer<Gpt<TrainingBackend>, TrainingBackend>>::Record;
    let record: State = <_ as Recorder<TrainingBackend>>::load(
        &NamedMpkBytesRecorder::<FullPrecisionSettings>::default(),
        fs::read(path.join("optimizer.mpk")).unwrap(),
        &Default::default(),
    )
    .unwrap();
    // Full tensor bytes and ParamIds are compared; HashMap iteration order is irrelevant.
    serde_json::to_value(<State as Record<TrainingBackend>>::into_item::<
        FullPrecisionSettings,
    >(record))
    .unwrap()
}
fn assert_periodic_deduplicated(root: &Path, name: &str) {
    let catalog =
        discover_checkpoints(&root.join("weights"), name, &CatalogLimits::default()).unwrap();
    let mut previous = None;
    for entry in catalog.entries {
        assert!(entry.path.join("COMPLETE").is_file());
        read_checkpoint_manifest(&entry.path).unwrap().unwrap();
        let progress = read_resume_manifest(&entry.path).unwrap().progress;
        assert!(
            previous.is_none_or(|value| progress.completed_updates > value),
            "duplicate final/periodic save"
        );
        previous = Some(progress.completed_updates);
    }
}

#[test]
#[ignore = "requires a real discrete Vulkan GPU and real Linux process signals"]
fn gpu_cli_cross_process_continuation_and_sigint_sigterm() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    prepare(root);
    // All branches inherit identical serialized parameters and optimizer IDs.
    let mut base = train("base", "base.jsonl", "5");
    base.extend(["--max-updates", "1"]);
    succeeds(root, "base", &base);
    let start = latest(root, "base");
    succeeds(
        root,
        "whole",
        &resume(checkpoint_name(&start), "whole", "whole.jsonl", "3"),
    );
    succeeds(
        root,
        "split",
        &resume(checkpoint_name(&start), "split", "split.jsonl", "1"),
    );
    let split = latest(root, "split");
    succeeds(
        root,
        "continued",
        &resume(checkpoint_name(&split), "continued", "continued.jsonl", "2"),
    );
    let expected = updates(&root.join("whole.jsonl"));
    let mut actual = updates(&root.join("split.jsonl"));
    actual.extend(updates(&root.join("continued.jsonl")));
    assert_eq!(
        expected, actual,
        "GPU update events changed across a fresh process"
    );
    let whole = latest(root, "whole");
    let continued = latest(root, "continued");
    assert_eq!(
        model_value(&whole),
        model_value(&continued),
        "GPU model tensor bytes changed across a fresh process"
    );
    assert_eq!(
        optimizer_value(&whole),
        optimizer_value(&continued),
        "GPU Adam state changed across a fresh process"
    );
    let a = read_resume_manifest(&whole).unwrap();
    let b = read_resume_manifest(&continued).unwrap();
    assert_eq!(a.progress, b.progress);
    assert_eq!(a.epoch_weighted_loss_bits, b.epoch_weighted_loss_bits);
    for name in ["base", "whole", "split", "continued"] {
        assert_periodic_deduplicated(root, name);
    }

    for (name, signal) in [("sigint", "-INT"), ("sigterm", "-TERM")] {
        let log = format!("{name}.jsonl");
        let mut running = spawn(root, name, &train(name, &log, "10000"));
        // Signal only after observing an actual committed update, not just adapter setup.
        loop {
            let contents = fs::read_to_string(root.join(&log)).unwrap_or_default();
            if contents
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .any(|row| row["event"] == "update")
            {
                break;
            }
            assert!(
                running
                    .child
                    .as_mut()
                    .unwrap()
                    .try_wait()
                    .unwrap()
                    .is_none(),
                "GPU child exited before signal: {}",
                running.diagnostic()
            );
            assert!(
                running.started.elapsed() < Duration::from_secs(30),
                "GPU child never committed: {}",
                running.diagnostic()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let pid = running.child.as_ref().unwrap().id();
        let result = Command::new("kill")
            .args([signal, &pid.to_string()])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "signal delivery failed: {result:?}"
        );
        let (status, output) = running.wait();
        assert!(
            !status.success(),
            "interrupted GPU CLI returned success: {output}"
        );
        let records = rows(&root.join(&log));
        let terminal = records.last().unwrap();
        assert_eq!(terminal["event"], "training_interrupted", "{output}");
        let saved = latest(root, name);
        let before = read_resume_manifest(&saved).unwrap();
        assert_eq!(
            terminal["completed_updates"],
            before.progress.completed_updates
        );
        assert_periodic_deduplicated(root, name);
        let next_name = format!("{name}-continued");
        let next_log = format!("{next_name}.jsonl");
        succeeds(
            root,
            &next_name,
            &resume(checkpoint_name(&saved), &next_name, &next_log, "1"),
        );
        let after = read_resume_manifest(&latest(root, &next_name)).unwrap();
        assert_eq!(
            after.progress.completed_updates,
            before.progress.completed_updates + 1
        );
        assert_eq!(updates(&root.join(next_log)).len(), 1);
    }
}
