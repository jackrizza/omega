use std::process::{Command, Output};

mod program {
    include!("../src/bin/main.rs");
    pub fn entry(arguments: Vec<std::ffi::OsString>) -> std::process::ExitCode {
        let _normal_entry: fn() -> std::process::ExitCode = main;
        main_with_args(arguments)
    }
}

#[test]
fn cli_child() {
    let Some(count) = std::env::var_os("OMEGA_NN_GENERATION_ARG_COUNT") else {
        return;
    };
    let count: usize = count.to_str().unwrap().parse().unwrap();
    let mut args = vec![std::ffi::OsString::from("omega-nn")];
    args.extend(
        (0..count).map(|i| std::env::var_os(format!("OMEGA_NN_GENERATION_ARG_{i}")).unwrap()),
    );
    let exit = program::entry(args);
    std::process::exit(if exit == std::process::ExitCode::SUCCESS {
        0
    } else {
        1
    });
}

fn cli(args: &[&str]) -> Output {
    // Cargo's shared main binary can belong to another package. This subprocess
    // runs the actual NN CLI entry point compiled into this integration test.
    let mut child = Command::new(std::env::current_exe().unwrap());
    child
        .args(["--exact", "cli_child", "--nocapture"])
        .env("OMEGA_NN_GENERATION_ARG_COUNT", args.len().to_string());
    for (i, arg) in args.iter().enumerate() {
        child.env(format!("OMEGA_NN_GENERATION_ARG_{i}"), arg);
    }
    child.output().unwrap()
}

fn train(extra: &[&str]) -> Output {
    let mut args = vec![
        "-f",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../test-fixtures/wordlevel.json"
        ),
        "--context-length",
        "8",
        "--d-model",
        "4",
        "--heads",
        "1",
        "--layers",
        "1",
        "--d-ff",
        "8",
        "train",
        "--text",
        "hello world !",
        "--prompt",
        "hello",
        "--steps",
        "1",
        "--max-new-tokens",
        "3",
    ];
    args.extend_from_slice(extra);
    cli(&args)
}

fn success(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn help_repeatable_sampling_and_greedy_compatibility() {
    let help = success(cli(&["train", "--help"]));
    for flag in [
        "--sample",
        "--temperature",
        "--top-k",
        "--top-p",
        "--sampling-seed",
    ] {
        assert!(help.contains(flag), "missing {flag}");
    }
    let sampled = success(train(&[
        "--sample",
        "--temperature",
        "0.7",
        "--top-k",
        "3",
        "--top-p",
        "0.9",
        "--sampling-seed",
        "9",
    ]));
    assert!(sampled.contains("Generated IDs (including prompt)"));
    assert_eq!(
        sampled,
        success(train(&[
            "--sample",
            "--temperature",
            "0.7",
            "--top-k",
            "3",
            "--top-p",
            "0.9",
            "--sampling-seed",
            "9"
        ]))
    );
    assert_eq!(
        success(train(&[])),
        success(train(&["--sample", "--top-k", "1"]))
    );
    assert_eq!(
        success(train(&["--sample"])),
        success(train(&[
            "--sample",
            "--temperature",
            "1",
            "--sampling-seed",
            "42"
        ]))
    );
}

#[test]
fn sampling_flag_conflicts_and_invalid_values_fail_before_training() {
    for flags in [
        vec!["--temperature", "1"],
        vec!["--top-k", "1"],
        vec!["--top-p", "1"],
        vec!["--sampling-seed", "42"],
        vec!["--sample", "--temperature", "0"],
        vec!["--sample", "--temperature", "NaN"],
        vec!["--sample", "--temperature", "inf"],
        vec!["--sample", "--top-k", "0"],
        vec!["--sample", "--top-k", "14"],
        vec!["--sample", "--top-p", "0"],
        vec!["--sample", "--top-p", "1.1"],
        vec!["--sample", "--top-p", "NaN"],
    ] {
        let output = train(&flags);
        assert!(!output.status.success(), "{flags:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("Sampling"),
            "{:?}",
            output
        );
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Training on"));
    }
}
