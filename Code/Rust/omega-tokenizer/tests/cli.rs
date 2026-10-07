mod support;

use omega_tokenizer::Tokens;
use std::process::{Command, Output};
use support::{TempDir, save_pipeline};

mod program {
    include!("../src/bin/main.rs");

    pub fn entry(arguments: Vec<std::ffi::OsString>) -> std::process::ExitCode {
        // Keep the normal entry point checked while supplying child-process args.
        let _normal_entry: fn() -> std::process::ExitCode = main;
        main_with_args(arguments)
    }
}

#[test]
fn cli_child() {
    let Some(count) = std::env::var_os("OMEGA_TOKENIZER_CLI_ARG_COUNT") else {
        return;
    };
    let count: usize = count.to_str().unwrap().parse().unwrap();
    let mut arguments = vec![std::ffi::OsString::from("omega-tokenizer")];
    arguments.extend(
        (0..count)
            .map(|index| std::env::var_os(format!("OMEGA_TOKENIZER_CLI_ARG_{index}")).unwrap()),
    );
    let exit = program::entry(arguments);
    std::process::exit(if exit == std::process::ExitCode::SUCCESS {
        0
    } else {
        1
    });
}

fn cli(directory: &std::path::Path, arguments: &[&str]) -> Output {
    // All three packages call their binary `main`; Cargo's shared main.exe can
    // be replaced by a concurrent package build. Run the same CLI entry point
    // in this test executable instead, with isolated cwd and process exit.
    let mut child = Command::new(std::env::current_exe().unwrap());
    child
        .env("RUST_LOG", "info")
        .env("OMEGA_TOKENIZER_CLI_ARG_COUNT", arguments.len().to_string())
        .current_dir(directory)
        .args(["--exact", "cli_child", "--nocapture"]);
    for (index, argument) in arguments.iter().enumerate() {
        child.env(format!("OMEGA_TOKENIZER_CLI_ARG_{index}"), argument);
    }
    child.output().unwrap()
}

fn succeeds(output: &Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn cli_actions_special_tokens_and_help() {
    let temp = TempDir::new();
    let path = temp.path().join("pipeline.json");
    save_pipeline(&path);
    let path = path.to_str().unwrap();
    let output = cli(
        temp.path(),
        &[
            "-f",
            path,
            "--test-string",
            "Hello",
            "--add-special-tokens",
            "--skip-special-tokens",
        ],
    );
    let stdout = succeeds(&output);
    assert!(stdout.contains("Input IDs: [13, 2, 14]"), "{stdout}");
    assert!(stdout.contains("Decoded: \"hello\""), "{stdout}");
    let output = cli(temp.path(), &["-f", path, "--decode-ids", "13,2,14"]);
    assert!(succeeds(&output).contains("Decoded: \"<start> hello <end>\""));
    let inspection = cli(temp.path(), &["-f", path]);
    succeeds(&inspection);
    assert!(String::from_utf8_lossy(&inspection.stderr).contains("15 vocabulary entries"));
    let help = succeeds(&cli(temp.path(), &["--help"]));
    for flag in [
        "--file-name",
        "--test-string",
        "--decode-ids",
        "--add-special-tokens",
        "--skip-special-tokens",
    ] {
        assert!(help.contains(flag), "missing {flag} from {help}");
    }
}

#[test]
fn conflicting_actions_invalid_ids_and_load_failures_exit_nonzero() {
    let temp = TempDir::new();
    let path = temp.path().join("pipeline.json");
    save_pipeline(&path);
    let path = path.to_str().unwrap();
    for arguments in [
        vec!["-f", path, "--test-string", "hello", "--decode-ids", "2"],
        vec!["-f", path, "--decode-ids", "2", "--test-string", "hello"],
        vec!["-f", path, "--decode-ids", "not-an-id"],
        vec!["-f", path, "--decode-ids", "4294967296"],
        vec!["--test-string", "hello"],
    ] {
        let output = cli(temp.path(), &arguments);
        assert!(
            !output.status.success(),
            "unexpected success: {arguments:?}"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("error:"));
    }
    for (name, contents) in [("missing.json", None), ("malformed.json", Some("{broken"))] {
        let path = temp.path().join(name);
        if let Some(contents) = contents {
            std::fs::write(&path, contents).unwrap();
        }
        let output = cli(
            temp.path(),
            &["-f", path.to_str().unwrap(), "--test-string", "hello"],
        );
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Failed to load tokenizer"), "{stderr}");
        assert!(stderr.contains(name), "{stderr}");
    }
}

#[test]
fn library_paths_follow_working_directory_but_cli_paths_follow_datasets() {
    const CHILD_FLAG: &str = "OMEGA_TOKENIZER_RELATIVE_PATH_CHILD";
    if std::env::var_os(CHILD_FLAG).is_some() {
        let tokenizer = Tokens::new("test.json").unwrap();
        assert_eq!(
            tokenizer.encode("hello", true).unwrap().get_ids(),
            &[13, 2, 14]
        );
        tokenizer.save("saved-relative.json").unwrap();
        assert_eq!(Tokens::new("saved-relative.json").unwrap().vocab_size(), 15);
        return;
    }
    let temp = TempDir::new();
    save_pipeline(&temp.path().join("test.json"));
    // A subprocess isolates cwd from the parallel Rust test harness.
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "library_paths_follow_working_directory_but_cli_paths_follow_datasets",
        ])
        .env(CHILD_FLAG, "1")
        .current_dir(temp.path())
        .output()
        .unwrap();
    succeeds(&child);
    assert!(temp.path().join("saved-relative.json").is_file());

    let stdout = succeeds(&cli(
        temp.path(),
        &[
            "-f",
            "../Code/Rust/test-fixtures/wordlevel.json",
            "--test-string",
            "hello",
            "--add-special-tokens",
        ],
    ));
    assert!(stdout.contains("Input IDs: [2]"), "{stdout}");
}
