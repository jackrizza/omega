//! Hardware tests are explicit: --features gpu --test gpu_cli -- --ignored.
use std::{
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};
use tempfile::TempDir;

mod program {
    pub fn entry(arguments: Vec<std::ffi::OsString>) -> std::process::ExitCode {
        let _entry: fn() -> std::process::ExitCode = main;
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
    let mut args = vec![std::ffi::OsString::from("omega-training")];
    args.extend(
        (0..count).map(|index| std::env::var_os(format!("OMEGA_GPU_CLI_ARG_{index}")).unwrap()),
    );
    let status = program::entry(args);
    std::process::exit(if status == std::process::ExitCode::SUCCESS {
        0
    } else {
        1
    });
}

fn invoke(args: &[&str]) -> Output {
    let scratch = TempDir::new().unwrap();
    let stdout = std::fs::File::create(scratch.path().join("stdout")).unwrap();
    let stderr = std::fs::File::create(scratch.path().join("stderr")).unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "gpu_cli_child", "--nocapture"])
        .env("OMEGA_GPU_CLI_ARG_COUNT", args.len().to_string());
    for (index, arg) in args.iter().enumerate() {
        command.env(format!("OMEGA_GPU_CLI_ARG_{index}"), arg);
    }
    let mut child = command
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("CLI exceeded 30 second deadline: {args:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Output {
        status,
        stdout: std::fs::read(scratch.path().join("stdout")).unwrap(),
        stderr: std::fs::read(scratch.path().join("stderr")).unwrap(),
    }
}

fn success(args: &[&str]) -> Output {
    let output = invoke(args);
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn backend_help_and_invalid_combinations() {
    let output = success(&["train", "--help"]);
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("--backend"));
    assert!(text.contains("--device"));
    let output = invoke(&[
        "--device",
        "0",
        "coverage",
        "--tokenizer",
        "test.json",
        "--text",
        "hello",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--device requires --backend vulkan"));
    #[cfg(not(feature = "gpu"))]
    {
        let output = invoke(&["devices"]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("--features gpu"));
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a discrete Vulkan GPU; run explicitly on qualified hardware"]
fn vulkan_train_resume_generate_evaluate_and_rejections() {
    let root = TempDir::new().unwrap();
    let datasets = root.path().join("datasets");
    let weights = root.path().join("weights");
    std::fs::create_dir_all(datasets.join("tiny")).unwrap();
    std::fs::write(
        datasets.join("test.json"),
        include_bytes!("../../../../datasets/test.json"),
    )
    .unwrap();
    std::fs::write(datasets.join("tiny/a.txt"), "hello world ! hello world !").unwrap();
    std::fs::write(datasets.join("tiny/b.txt"), "this is a test !").unwrap();
    let data = datasets.to_str().unwrap();
    let saves = weights.to_str().unwrap();
    let metrics = root.path().join("metrics.jsonl");
    success(&["devices"]);
    success(&[
        "train",
        "--backend",
        "vulkan",
        "--device",
        "0",
        "--datasets-root",
        data,
        "--weights-root",
        saves,
        "--name",
        "gpu",
        "--dataset",
        "tiny",
        "--epochs",
        "2",
        "--context-length",
        "3",
        "--d-model",
        "4",
        "--heads",
        "1",
        "--layers",
        "1",
        "--d-ff",
        "8",
        "--batch-size",
        "2",
        "--shuffle",
        "--gradient-clip-norm",
        "1",
        "--warmup-updates",
        "2",
        "--max-updates",
        "1",
        "--metrics-jsonl",
        metrics.to_str().unwrap(),
    ]);
    assert!(weights.join("gpu-1/COMPLETE").is_file());
    let resume: serde_json::Value =
        serde_json::from_slice(&std::fs::read(weights.join("gpu-1/resume.json")).unwrap()).unwrap();
    assert_eq!(resume["schema_version"], 4);
    assert_eq!(resume["gpu_execution"]["backend"], "Vulkan");
    assert!(
        std::fs::read_to_string(metrics)
            .unwrap()
            .contains("\"event\":\"execution\"")
    );
    success(&[
        "resume",
        "--backend",
        "vulkan",
        "--datasets-root",
        data,
        "--weights-root",
        saves,
        "--checkpoint",
        "gpu-1",
        "--name",
        "continued",
        "--epochs",
        "2",
        "--max-updates",
        "1",
    ]);
    for backend in ["cpu", "vulkan"] {
        success(&[
            "generate",
            "--backend",
            backend,
            "--weights-root",
            saves,
            "--checkpoint",
            "continued-1",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ]);
        success(&[
            "evaluate",
            "--backend",
            backend,
            "--datasets-root",
            data,
            "--weights-root",
            saves,
            "--checkpoint",
            "continued-1",
            "--dataset",
            "tiny",
        ]);
    }
    // Legacy headers also receive buffer checks before decoding model records.
    let legacy = weights.join("legacy-1");
    std::fs::create_dir(&legacy).unwrap();
    for name in ["config.json", "tokenizer.json", "model.mpk"] {
        std::fs::copy(weights.join("gpu-1").join(name), legacy.join(name)).unwrap();
    }
    std::fs::write(legacy.join("COMPLETE"), "").unwrap();
    success(&[
        "generate",
        "--backend",
        "vulkan",
        "--weights-root",
        saves,
        "--checkpoint",
        "legacy-1",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
    ]);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(legacy.join("config.json")).unwrap()).unwrap();
    config["d_model"] = 1_000_000usize.into();
    std::fs::write(
        legacy.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(
        legacy.join("model.mpk"),
        "invalid model must not be decoded",
    )
    .unwrap();
    let output = invoke(&[
        "generate",
        "--backend",
        "vulkan",
        "--weights-root",
        saves,
        "--checkpoint",
        "legacy-1",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("adapter limit"));
    let output = invoke(&[
        "resume",
        "--backend",
        "cpu",
        "--datasets-root",
        data,
        "--weights-root",
        saves,
        "--checkpoint",
        "gpu-1",
        "--name",
        "invalid",
        "--epochs",
        "2",
    ]);
    assert!(!output.status.success());
    assert!(!weights.join("invalid-1").exists());
    let output = invoke(&[
        "generate",
        "--backend",
        "vulkan",
        "--device",
        "999999",
        "--weights-root",
        saves,
        "--checkpoint",
        "gpu-1",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unavailable"));
}
