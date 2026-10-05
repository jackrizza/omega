use std::{
    ffi::OsString,
    fs,
    path::Path,
    process::{Command, ExitCode, Output, Stdio},
    time::{Duration, Instant},
};

mod program {
    pub fn entry(arguments: Vec<std::ffi::OsString>) -> std::process::ExitCode {
        let _normal_entry: fn() -> std::process::ExitCode = main;
        main_with_args(arguments)
    }

    include!("../src/bin/main.rs");
}

#[test]
fn cli_child() {
    let Some(count) = std::env::var_os("OMEGA_TRAINING_CLI_ARG_COUNT") else {
        return;
    };
    let count: usize = count.to_str().unwrap().parse().unwrap();
    let mut arguments = vec![OsString::from("omega-training")];
    arguments.extend(
        (0..count)
            .map(|index| std::env::var_os(format!("OMEGA_TRAINING_CLI_ARG_{index}")).unwrap()),
    );
    let exit = program::entry(arguments);
    std::process::exit(if exit == ExitCode::SUCCESS { 0 } else { 1 });
}

fn cli(directory: &Path, arguments: &[&str]) -> Output {
    cli_with_env(directory, arguments, &[])
}

fn cli_with_env(
    directory: &Path,
    arguments: &[&str],
    environment: &[(&str, Option<&str>)],
) -> Output {
    // Every package names its binary main. Run the actual CLI source in this
    // test executable, avoiding a shared target/debug/main.exe build collision.
    // Subprocesses isolate working directories and Burn RNG between tests.
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .current_dir(directory)
        .args(["--exact", "cli_child", "--nocapture"])
        .env("OMEGA_TRAINING_CLI_ARG_COUNT", arguments.len().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (index, argument) in arguments.iter().enumerate() {
        command.env(format!("OMEGA_TRAINING_CLI_ARG_{index}"), argument);
    }
    for (name, value) in environment {
        match value {
            Some(value) => {
                command.env(name, value);
            }
            None => {
                command.env_remove(name);
            }
        }
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!("CLI exceeded 30 seconds: {arguments:?}\n{output:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn succeeds(output: Output) -> String {
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn fails(output: Output, expected: &str) {
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(expected), "{stderr}");
}

fn prepare(root: &Path) {
    fs::create_dir_all(root.join("data/first")).unwrap();
    fs::create_dir_all(root.join("data/second")).unwrap();
    fs::write(root.join("data/first/a.txt"), "hello world omega").unwrap();
    fs::write(root.join("data/second/b.txt"), "this is a test").unwrap();
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"),
        root.join("data/test.json"),
    )
    .unwrap();
}

fn train_arguments<'a>(datasets: &'a str, weights: &'a str, tokenizer: &'a str) -> Vec<&'a str> {
    vec![
        "train",
        "--name",
        "Tiny",
        "--datasets-root",
        datasets,
        "--weights-root",
        weights,
        "--dataset",
        "first",
        "--dataset",
        "second",
        "--tokenizer",
        tokenizer,
        "--epochs",
        "1",
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
    ]
}

#[test]
fn tiny_train_save_generate_and_repeated_saves_across_working_directories() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let elsewhere = temp.path().join("other-cwd");
    fs::create_dir(&elsewhere).unwrap();
    let datasets = temp.path().join("data");
    let weights = temp.path().join("saved-weights");
    let tokenizer = datasets.join("test.json");
    // Absolute roots work from an unrelated cwd; absolute tokenizer remains valid.
    let first = succeeds(cli(
        &elsewhere,
        &train_arguments(
            datasets.to_str().unwrap(),
            weights.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
        ),
    ));
    assert!(
        first.contains("Training on 2 files, 7 source tokens, 2 examples"),
        "{first}"
    );
    assert!(first.contains("Saved checkpoint:"), "{first}");
    assert!(first.contains("tiny-1"), "{first}");
    // Relative roots and tokenizer are interpreted from this second cwd.
    let second = succeeds(cli(
        temp.path(),
        &train_arguments("data", "saved-weights", "test.json"),
    ));
    assert!(second.contains("tiny-2"), "{second}");
    for name in ["tiny-1", "tiny-2"] {
        for file in [
            "config.json",
            "tokenizer.json",
            "model.mpk",
            "optimizer.mpk",
            "resume.json",
            "COMPLETE",
        ] {
            assert!(weights.join(name).join(file).is_file(), "{name}/{file}");
        }
    }
    assert_eq!(fs::read_dir(&weights).unwrap().count(), 2);
    let generated = succeeds(cli(
        &elsewhere,
        &[
            "generate",
            "--weights-root",
            weights.to_str().unwrap(),
            "--checkpoint",
            "tiny-1",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ],
    ));
    assert!(generated.contains("Token IDs: [2, "), "{generated}");
    assert!(generated.contains("Decoded:"), "{generated}");
    let relative = succeeds(cli(
        temp.path(),
        &[
            "generate",
            "--weights-root",
            "saved-weights",
            "--checkpoint",
            "tiny-1",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ],
    ));
    assert_eq!(generated, relative);
}

#[test]
fn help_exposes_roots_and_argument_errors_exit_nonzero() {
    let temp = tempfile::tempdir().unwrap();
    let train_help = succeeds(cli(temp.path(), &["train", "--help"]));
    for flag in [
        "--datasets-root",
        "--weights-root",
        "--dataset",
        "--tokenizer",
    ] {
        assert!(train_help.contains(flag), "{train_help}");
    }
    assert!(train_help.contains("working directory"));
    assert!(train_help.contains("checkout datasets/"));
    let generate_help = succeeds(cli(temp.path(), &["generate", "--help"]));
    assert!(generate_help.contains("--weights-root"));
    for arguments in [
        vec!["train", "--name", "x"],
        vec!["train", "--name", "../x", "--dataset", "first"],
        vec![
            "train",
            "--name",
            "x",
            "--dataset",
            "first",
            "--weights-root",
        ],
        vec!["generate", "--checkpoint", "../x", "--prompt", "hello"],
        vec![
            "generate",
            "--checkpoint",
            "x",
            "--prompt",
            "hello",
            "--unknown",
        ],
    ] {
        fails(cli(temp.path(), &arguments), "error:");
    }
}

#[test]
fn dataset_tokenizer_training_and_checkpoint_failures_exit_nonzero() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let base = train_arguments("data", "weights", "test.json");
    for folder in ["../first", "first/../second", "missing"] {
        let mut arguments = base.clone();
        let value = arguments.iter().position(|&arg| arg == "first").unwrap();
        arguments[value] = folder;
        fails(cli(temp.path(), &arguments), "Error:");
    }
    for tokenizer in ["missing.json", "malformed.json"] {
        fs::write(temp.path().join("data/malformed.json"), "{bad").unwrap();
        fails(
            cli(temp.path(), &train_arguments("data", "weights", tokenizer)),
            "Error:",
        );
    }
    let mut zero_epochs = base;
    let epochs = zero_epochs
        .iter()
        .position(|&arg| arg == "--epochs")
        .unwrap();
    zero_epochs[epochs + 1] = "0";
    fails(
        cli(temp.path(), &zero_epochs),
        "epochs must be greater than zero",
    );
    assert!(!temp.path().join("weights").exists());
    fs::create_dir_all(temp.path().join("weights/incomplete-1")).unwrap();
    for checkpoint in ["missing-1", "incomplete-1"] {
        fails(
            cli(
                temp.path(),
                &[
                    "generate",
                    "--weights-root",
                    "weights",
                    "--checkpoint",
                    checkpoint,
                    "--prompt",
                    "hello",
                    "--max-new-tokens",
                    "1",
                ],
            ),
            "Error:",
        );
    }
}

fn json_lines(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn validation_metrics_evaluation_and_manifest_agree() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let mut args = train_arguments("data", "weights", "test.json");
    let epochs = args.iter().position(|&a| a == "--epochs").unwrap();
    args[epochs + 1] = "2";
    args.extend([
        "--validation-count",
        "1",
        "--split-seed",
        "7",
        "--metrics-jsonl",
        "metrics.jsonl",
    ]);
    let output = succeeds(cli(temp.path(), &args));
    assert!(output.contains("Epoch 1 update 1 targets"), "{output}");
    assert_eq!(output.matches("Validation:").count(), 2);
    assert!(output.contains("mean pre-update loss"), "{output}");
    let records = json_lines(&temp.path().join("metrics.jsonl"));
    let events: Vec<_> = records
        .iter()
        .map(|r| r["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        events,
        [
            "training_started",
            "update",
            "validation",
            "update",
            "validation",
            "training_complete"
        ]
    );
    assert!(records.iter().all(|r| r["schema_version"] == 1));
    assert_eq!(records[3]["completed_updates"], 2);
    assert_eq!(records[5]["completed_epochs"], 2);
    let manifest =
        omega_training::checkpoint::read_checkpoint_manifest(&temp.path().join("weights/tiny-1"))
            .unwrap()
            .unwrap();
    let data = manifest.metadata.dataset.unwrap();
    assert_eq!(data.format, "text");
    assert_eq!(data.fingerprint_kind, "token-ids-le-u32-v1");
    assert_eq!(data.split_seed, 7);
    assert_eq!(data.split_policy, "count:1");
    assert_eq!(data.selections, ["first", "second"]);
    assert_eq!(data.documents.len(), 2);
    let tokenizer = omega_tokenizer::Tokens::new(temp.path().join("data/test.json")).unwrap();
    let corpus = omega_training::load_document_corpus(
        &temp.path().join("data"),
        &data.selections,
        &tokenizer,
    )
    .unwrap();
    let split = omega_training::split_document_corpus(
        &corpus,
        4,
        omega_training::ValidationSplit::Count(1),
        7,
    )
    .unwrap();
    let mut held_out_count = 0;
    for doc in &data.documents {
        let source = corpus
            .documents
            .iter()
            .find(|d| d.id.to_string_lossy().replace('\\', "/") == doc.id)
            .unwrap();
        let bytes: Vec<_> = source
            .tokens
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        assert_eq!(doc.sha256, omega_training::checkpoint::sha256_bytes(&bytes));
        let is_validation = split.validation.documents.iter().any(|d| d.id == source.id);
        assert_eq!(doc.partition == "validation", is_validation);
        if is_validation {
            held_out_count += source.tokens.len() - 1;
        }
    }
    assert_eq!(records[4]["target_count"], held_out_count);
    let training = manifest.metadata.training.unwrap();
    assert_eq!(training.epochs, 2);
    assert_eq!(training.seed, 42);
    let build = manifest.metadata.build.unwrap();
    assert_eq!(build.cargo_lock_sha256.unwrap().len(), 64);
    let (model, config, _) =
        omega_training::load_checkpoint(&temp.path().join("weights/tiny-1")).unwrap();
    let measured = omega_training::evaluate(&model, &config, &split.validation.set).unwrap();
    assert!(
        (measured.mean_cross_entropy - records[4]["mean_cross_entropy"].as_f64().unwrap()).abs()
            < 1e-6
    );
    let eval = succeeds(cli(
        temp.path(),
        &[
            "evaluate",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "tiny-1",
            "--dataset",
            "first",
            "--dataset",
            "second",
            "--validation-count",
            "1",
            "--split-seed",
            "7",
        ],
    ));
    assert!(
        eval.contains(&format!("Evaluation: {held_out_count} targets")),
        "{eval}"
    );
    assert!(
        eval.contains(&format!("{:.6}", measured.mean_cross_entropy)),
        "{eval}"
    );
    fails(
        cli(
            temp.path(),
            &[
                "evaluate",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--checkpoint",
                "tiny-1",
                "--dataset",
                "first",
                "--validation-count",
                "0",
            ],
        ),
        "Evaluation set must contain",
    );
}

#[test]
fn quiet_metrics_failures_and_split_argument_validation() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let base = train_arguments("data", "weights", "test.json");
    let mut args = base.clone();
    args.extend(["--quiet", "--metrics-jsonl", "quiet.jsonl"]);
    let output = succeeds(cli(temp.path(), &args));
    assert!(output.contains("Saved checkpoint"));
    for forbidden in ["Training on", "Epoch", "loss", "Validation:"] {
        assert!(!output.contains(forbidden), "{output}");
    }
    let original = fs::read(temp.path().join("quiet.jsonl")).unwrap();
    fails(cli(temp.path(), &args), "Cannot create new metrics file");
    assert_eq!(fs::read(temp.path().join("quiet.jsonl")).unwrap(), original);
    let mut bad_parent = base.clone();
    bad_parent.extend(["--metrics-jsonl", "absent/log.jsonl"]);
    fails(
        cli(temp.path(), &bad_parent),
        "Cannot create new metrics file",
    );
    for flags in [
        vec!["--validation-count", "1", "--validation-ratio", "0.5"],
        vec!["--validation-count", "2"],
        vec!["--validation-ratio", "NaN"],
        vec!["--dataset-format", "csv"],
    ] {
        let mut args = base.clone();
        args.extend(flags);
        assert!(!cli(temp.path(), &args).status.success());
    }
    assert_eq!(
        fs::read_dir(temp.path().join("weights")).unwrap().count(),
        1
    );
    let records = json_lines(&temp.path().join("quiet.jsonl"));
    assert_eq!(records.last().unwrap()["event"], "training_complete");
    assert!(!records.iter().any(|r| r["event"] == "validation"));
}

#[test]
fn tokenizer_commands_and_explicit_jsonl_training() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    fs::create_dir(temp.path().join("data/records")).unwrap();
    fs::write(
        temp.path().join("data/records/docs.jsonl"),
        "{\"text\":\"hello world omega\",\"source\":\"one\"}\n\n{\"text\":\"this is a test\"}\n",
    )
    .unwrap();
    let tok_args = [
        "train-tokenizer",
        "--datasets-root",
        "data",
        "--dataset",
        "records",
        "--dataset-format",
        "jsonl",
        "--output",
        "byte.json",
        "--vocab-size",
        "256",
        "--min-frequency",
        "1",
    ];
    let output = succeeds(cli(temp.path(), &tok_args));
    assert!(output.contains("256 entries, 2 documents"), "{output}");
    let tokenizer = omega_tokenizer::Tokens::new(temp.path().join("byte.json")).unwrap();
    let text = "Hello Ω! 🦀\n";
    assert_eq!(
        tokenizer
            .decode(tokenizer.encode(text, false).unwrap().get_ids(), false)
            .unwrap(),
        text
    );
    let saved = fs::read(temp.path().join("byte.json")).unwrap();
    fails(
        cli(temp.path(), &tok_args),
        "existing paths are never overwritten",
    );
    assert_eq!(fs::read(temp.path().join("byte.json")).unwrap(), saved);
    let coverage = succeeds(cli(
        temp.path(),
        &[
            "coverage",
            "--datasets-root",
            ".",
            "--tokenizer",
            "byte.json",
            "--text",
            "Hello Ω!",
        ],
    ));
    let report: serde_json::Value =
        serde_json::from_str(coverage.lines().find(|line| line.starts_with('{')).unwrap()).unwrap();
    assert!(report["token_count"].as_u64().unwrap() > 0);
    assert!(report["unknown_rate"].is_null());
    let known = succeeds(cli(
        temp.path(),
        &[
            "coverage",
            "--datasets-root",
            "data",
            "--tokenizer",
            "test.json",
            "--text",
            "hello unseen",
            "--unknown-token-id",
            "0",
        ],
    ));
    let report: serde_json::Value =
        serde_json::from_str(known.lines().find(|line| line.starts_with('{')).unwrap()).unwrap();
    assert_eq!(report["token_count"], 2);
    assert_eq!(report["unknown_count"], 1);
    assert_eq!(report["unknown_rate"], 0.5);
    let mut args = train_arguments("data", "weights", "test.json");
    let first = args.iter().position(|&a| a == "first").unwrap();
    args[first] = "records";
    let second = args.iter().position(|&a| a == "second").unwrap();
    args[second] = "records";
    args.extend(["--dataset-format", "jsonl", "--validation-count", "1"]);
    succeeds(cli(temp.path(), &args));
    let manifest =
        omega_training::checkpoint::read_checkpoint_manifest(&temp.path().join("weights/tiny-1"))
            .unwrap()
            .unwrap();
    let data = manifest.metadata.dataset.unwrap();
    assert_eq!(data.format, "jsonl");
    assert_eq!(data.documents.len(), 2);
    assert_eq!(data.documents[0].id, "records/docs.jsonl/@record-1");
    assert_eq!(data.documents[1].id, "records/docs.jsonl/@record-3");
    let eval = succeeds(cli(
        temp.path(),
        &[
            "evaluate",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "tiny-1",
            "--dataset",
            "records",
            "--dataset-format",
            "jsonl",
        ],
    ));
    assert!(eval.contains("Evaluation: 5 targets"), "{eval}");
    fs::write(
        temp.path().join("data/records/docs.jsonl"),
        "{\"text\":\"hello\"}\n{\"text\":7}\n",
    )
    .unwrap();
    fails(cli(temp.path(), &tok_args), "docs.jsonl:2");
    for command in ["evaluate", "train-tokenizer", "coverage"] {
        assert!(succeeds(cli(temp.path(), &[command, "--help"])).contains("Usage:"));
    }
}

fn replace_argument<'a>(arguments: &mut [&'a str], flag: &str, value: &'a str) {
    let index = arguments
        .iter()
        .position(|&argument| argument == flag)
        .unwrap();
    arguments[index + 1] = value;
}

fn resume_arguments<'a>(checkpoint: &'a str, name: &'a str, metrics: &'a str) -> Vec<&'a str> {
    vec![
        "resume",
        "--checkpoint",
        checkpoint,
        "--name",
        name,
        "--epochs",
        "2",
        "--datasets-root",
        "data",
        "--weights-root",
        "weights",
        "--metrics-jsonl",
        metrics,
        "--quiet",
    ]
}

fn cache_arguments(output: &str) -> Vec<&str> {
    vec![
        "prepare-cache",
        "--datasets-root",
        "data",
        "--dataset",
        "first",
        "--dataset",
        "second",
        "--tokenizer",
        "test.json",
        "--context-length",
        "4",
        "--output",
        output,
    ]
}

fn snapshot_tree(root: &Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    fn visit(
        root: &Path,
        path: &Path,
        output: &mut std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    ) {
        for entry in fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                visit(root, &entry.path(), output);
            } else {
                assert!(entry.file_type().unwrap().is_file());
                output.insert(
                    entry.path().strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(entry.path()).unwrap(),
                );
            }
        }
    }
    let mut output = std::collections::BTreeMap::new();
    visit(root, root, &mut output);
    output
}

fn checkpoint_logits(path: &Path) -> Vec<f32> {
    let (model, config, _) = omega_training::load_checkpoint(path).unwrap();
    let input = omega_nn::token_tensor::<omega_nn::Cpu>(
        &[2, 3, 4],
        config.vocab_size,
        config.context_length,
        &Default::default(),
    )
    .unwrap();
    model.forward(input).into_data().to_vec::<f32>().unwrap()
}

fn updates(records: &[serde_json::Value]) -> Vec<serde_json::Value> {
    records
        .iter()
        .filter(|record| record["event"] == "update")
        .cloned()
        .collect()
}

#[test]
fn mid_epoch_and_epoch_boundary_resume_match_uninterrupted_training() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let mut full = train_arguments("data", "weights", "test.json");
    replace_argument(&mut full, "--epochs", "2");
    full.extend(["--metrics-jsonl", "full.jsonl", "--quiet"]);
    succeeds(cli(temp.path(), &full));
    let expected_logits = checkpoint_logits(&temp.path().join("weights/tiny-1"));
    let expected_metrics = json_lines(&temp.path().join("full.jsonl"));
    let expected_updates = updates(&expected_metrics);
    assert_eq!(expected_updates.len(), 4);

    for (name, cap, completed, checkpoint, resumed, partial_log, resume_log) in [
        (
            "mid",
            "1",
            1usize,
            "mid-1",
            "mid-done",
            "mid.jsonl",
            "mid-resume.jsonl",
        ),
        (
            "boundary",
            "2",
            2usize,
            "boundary-1",
            "boundary-done",
            "boundary.jsonl",
            "boundary-resume.jsonl",
        ),
    ] {
        let mut partial = train_arguments("data", "weights", "test.json");
        replace_argument(&mut partial, "--name", name);
        replace_argument(&mut partial, "--epochs", "2");
        partial.extend([
            "--max-updates",
            cap,
            "--metrics-jsonl",
            partial_log,
            "--quiet",
        ]);
        succeeds(cli(temp.path(), &partial));
        let original_path = temp.path().join("weights").join(checkpoint);
        let original = snapshot_tree(&original_path);
        assert!(original.contains_key(Path::new("optimizer.mpk")));
        assert!(original.contains_key(Path::new("resume.json")));
        let partial_metrics = json_lines(&temp.path().join(partial_log));
        assert_eq!(updates(&partial_metrics), expected_updates[..completed]);
        let boundary = partial_metrics.last().unwrap();
        assert_eq!(boundary["event"], "segment_complete");
        assert_eq!(boundary["completed_updates"], completed);
        assert_eq!(boundary["completed_epochs"], completed / 2);
        assert_eq!(boundary["next_example_index"], completed % 2);

        succeeds(cli(
            temp.path(),
            &resume_arguments(checkpoint, resumed, resume_log),
        ));
        assert_eq!(snapshot_tree(&original_path), original);
        assert_eq!(
            checkpoint_logits(&temp.path().join("weights").join(format!("{resumed}-1"))),
            expected_logits
        );
        let resumed_metrics = json_lines(&temp.path().join(resume_log));
        assert_eq!(resumed_metrics[0]["event"], "training_resumed");
        assert_eq!(resumed_metrics[0]["completed_updates"], completed);
        assert_eq!(resumed_metrics[0]["next_example_index"], completed % 2);
        assert_eq!(resumed_metrics[0]["target_epochs"], 2);
        assert_eq!(updates(&resumed_metrics), expected_updates[completed..]);
        assert_eq!(resumed_metrics.last(), expected_metrics.last());
    }
}

#[test]
fn cached_training_and_capped_resume_match_eager_updates() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli(temp.path(), &cache_arguments("cache")));
    let original_cache = snapshot_tree(&temp.path().join("cache"));
    let mut eager = train_arguments("data", "weights", "test.json");
    replace_argument(&mut eager, "--epochs", "2");
    eager.extend(["--metrics-jsonl", "eager.jsonl", "--quiet"]);
    succeeds(cli(temp.path(), &eager));
    let expected = json_lines(&temp.path().join("eager.jsonl"));
    let expected_updates = updates(&expected);

    let mut cached = train_arguments("data", "weights", "test.json");
    replace_argument(&mut cached, "--name", "cached");
    replace_argument(&mut cached, "--epochs", "2");
    cached.extend([
        "--cache",
        "cache",
        "--max-updates",
        "1",
        "--metrics-jsonl",
        "cached.jsonl",
        "--quiet",
    ]);
    succeeds(cli(temp.path(), &cached));
    assert_eq!(
        updates(&json_lines(&temp.path().join("cached.jsonl"))),
        expected_updates[..1]
    );
    let first = snapshot_tree(&temp.path().join("weights/cached-1"));

    let mut partial_resume = resume_arguments("cached-1", "cached-middle", "cached-middle.jsonl");
    partial_resume.extend(["--cache", "cache", "--max-updates", "1"]);
    succeeds(cli(temp.path(), &partial_resume));
    let middle = json_lines(&temp.path().join("cached-middle.jsonl"));
    assert_eq!(updates(&middle), expected_updates[1..2]);
    assert_eq!(middle.last().unwrap()["event"], "segment_complete");
    assert_eq!(middle.last().unwrap()["completed_epochs"], 1);
    assert_eq!(middle.last().unwrap()["next_example_index"], 0);
    assert_eq!(snapshot_tree(&temp.path().join("weights/cached-1")), first);

    let second = snapshot_tree(&temp.path().join("weights/cached-middle-1"));
    let mut finish = resume_arguments("cached-middle-1", "cached-final", "cached-final.jsonl");
    finish.extend(["--cache", "cache"]);
    succeeds(cli(temp.path(), &finish));
    let final_metrics = json_lines(&temp.path().join("cached-final.jsonl"));
    assert_eq!(updates(&final_metrics), expected_updates[2..]);
    assert_eq!(final_metrics.last(), expected.last());
    assert_eq!(
        checkpoint_logits(&temp.path().join("weights/cached-final-1")),
        checkpoint_logits(&temp.path().join("weights/tiny-1"))
    );
    assert_eq!(
        snapshot_tree(&temp.path().join("weights/cached-middle-1")),
        second
    );
    assert_eq!(snapshot_tree(&temp.path().join("cache")), original_cache);
}

#[test]
fn cached_jsonl_split_resume_preserves_validation_recipe_and_results() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    fs::write(
        temp.path().join("data/first/a.jsonl"),
        "{\"text\":\"hello world omega\"}\n{\"text\":\"this is a test\"}\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("data/second/b.jsonl"),
        "{\"text\":\"omega world hello\"}\n{\"text\":\"a test is this\"}\n",
    )
    .unwrap();
    let recipe = [
        "--dataset-format",
        "jsonl",
        "--validation-count",
        "1",
        "--split-seed",
        "7",
    ];
    let mut prepare = cache_arguments("cache");
    prepare.extend(recipe);
    succeeds(cli(temp.path(), &prepare));
    let mut eager = train_arguments("data", "weights", "test.json");
    replace_argument(&mut eager, "--epochs", "2");
    eager.extend(recipe);
    eager.extend(["--metrics-jsonl", "eager.jsonl", "--quiet"]);
    succeeds(cli(temp.path(), &eager));
    let expected = json_lines(&temp.path().join("eager.jsonl"));
    let expected_updates = updates(&expected);
    assert_eq!(expected_updates.len(), 6);

    let mut partial = train_arguments("data", "weights", "test.json");
    replace_argument(&mut partial, "--epochs", "2");
    replace_argument(&mut partial, "--name", "partial");
    partial.extend(recipe);
    partial.extend([
        "--cache",
        "cache",
        "--max-updates",
        "2",
        "--metrics-jsonl",
        "partial.jsonl",
        "--quiet",
    ]);
    succeeds(cli(temp.path(), &partial));
    assert_eq!(
        updates(&json_lines(&temp.path().join("partial.jsonl"))),
        expected_updates[..2]
    );
    // Resume has no format/split overrides: it must infer the saved recipe.
    let mut resume = resume_arguments("partial-1", "resumed", "resumed.jsonl");
    resume.extend(["--cache", "cache"]);
    succeeds(cli(temp.path(), &resume));
    let resumed = json_lines(&temp.path().join("resumed.jsonl"));
    assert_eq!(updates(&resumed), expected_updates[2..]);
    let validations = |records: &[serde_json::Value]| {
        records
            .iter()
            .filter(|record| record["event"] == "validation")
            .cloned()
            .collect::<Vec<_>>()
    };
    assert_eq!(validations(&expected).len(), 2);
    assert_eq!(validations(&resumed), validations(&expected));
    assert_eq!(resumed.last(), expected.last());
    assert_eq!(
        checkpoint_logits(&temp.path().join("weights/resumed-1")),
        checkpoint_logits(&temp.path().join("weights/tiny-1"))
    );
}

#[test]
fn stale_cache_options_tokenizer_sources_and_limits_fail_before_saving() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli(temp.path(), &cache_arguments("cache")));
    let original_cache = snapshot_tree(&temp.path().join("cache"));
    fails(cli(temp.path(), &cache_arguments("cache")), "Error:");
    assert_eq!(snapshot_tree(&temp.path().join("cache")), original_cache);
    fs::write(temp.path().join("occupied-cache"), "keep this file").unwrap();
    fails(
        cli(temp.path(), &cache_arguments("occupied-cache")),
        "Error:",
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("occupied-cache")).unwrap(),
        "keep this file"
    );

    let mut base = train_arguments("data", "rejected-weights", "test.json");
    base.extend(["--cache", "cache", "--quiet"]);
    let mut wrong_context = base.clone();
    replace_argument(&mut wrong_context, "--context-length", "3");
    fails(cli(temp.path(), &wrong_context), "Error:");
    for options in [
        vec!["--validation-count", "1"],
        vec!["--split-seed", "7"],
        vec!["--dataset-format", "jsonl"],
    ] {
        let mut arguments = base.clone();
        arguments.extend(options);
        fails(cli(temp.path(), &arguments), "Error:");
    }
    let source = temp.path().join("data/first/a.txt");
    let original_source = fs::read(&source).unwrap();
    // Same token IDs but changed raw bytes must still invalidate provenance.
    fs::write(&source, "hello world omega\n").unwrap();
    fails(cli(temp.path(), &base), "Error:");
    fs::write(&source, original_source).unwrap();

    let tokenizer = temp.path().join("data/test.json");
    let original_tokenizer = fs::read(&tokenizer).unwrap();
    let mut changed: serde_json::Value = serde_json::from_slice(&original_tokenizer).unwrap();
    changed["model"]["vocab"]["hello"] = 3.into();
    changed["model"]["vocab"]["world"] = 2.into();
    fs::write(&tokenizer, serde_json::to_vec(&changed).unwrap()).unwrap();
    fails(cli(temp.path(), &base), "Error:");
    fs::write(&tokenizer, original_tokenizer).unwrap();

    for flag in [
        "--max-source-bytes",
        "--max-document-tokens",
        "--max-documents",
        "--max-directory-entries",
        "--max-manifest-bytes",
    ] {
        let mut opening = base.clone();
        opening.extend([flag, "1"]);
        fails(cli(temp.path(), &opening), "Error:");
        let output = format!("small{}", flag);
        let mut creating = cache_arguments(&output);
        creating.extend([flag, "1"]);
        fails(cli(temp.path(), &creating), "Error:");
    }
    assert!(!temp.path().join("rejected-weights").exists());
    assert_eq!(snapshot_tree(&temp.path().join("cache")), original_cache);
}

#[test]
fn resume_rejects_stale_sources_completed_targets_and_inference_only_checkpoints() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let mut args = train_arguments("data", "weights", "test.json");
    args.extend(["--quiet"]);
    succeeds(cli(temp.path(), &args));
    let original = snapshot_tree(&temp.path().join("weights/tiny-1"));
    let mut completed = resume_arguments("tiny-1", "rejected", "completed.jsonl");
    replace_argument(&mut completed, "--epochs", "1");
    fails(cli(temp.path(), &completed), "Error:");
    let mut zero_cap = resume_arguments("tiny-1", "rejected", "zero.jsonl");
    zero_cap.extend(["--max-updates", "0"]);
    fails(cli(temp.path(), &zero_cap), "Error:");
    let source = temp.path().join("data/first/a.txt");
    let original_source = fs::read(&source).unwrap();
    fs::write(&source, "omega world hello").unwrap();
    fails(
        cli(
            temp.path(),
            &resume_arguments("tiny-1", "rejected", "stale.jsonl"),
        ),
        "Error:",
    );
    fs::write(&source, original_source).unwrap();
    assert_eq!(snapshot_tree(&temp.path().join("weights/tiny-1")), original);

    let (model, config, tokenizer) =
        omega_training::load_checkpoint(&temp.path().join("weights/tiny-1")).unwrap();
    let inference = omega_training::save_checkpoint(
        &temp.path().join("weights"),
        "inference",
        model,
        &config,
        &tokenizer,
    )
    .unwrap();
    // Construct the original unversioned inference layout from this test's new
    // checkpoint; legacy generation must still work, while resume must fail.
    fs::remove_file(inference.join("manifest.json")).unwrap();
    fs::write(inference.join("COMPLETE"), []).unwrap();
    let original_inference = snapshot_tree(&inference);
    assert!(!inference.join("resume.json").exists());
    fails(
        cli(
            temp.path(),
            &resume_arguments("inference-1", "rejected", "inference.jsonl"),
        ),
        "Error:",
    );
    succeeds(cli(
        temp.path(),
        &[
            "generate",
            "--weights-root",
            "weights",
            "--checkpoint",
            "inference-1",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ],
    ));
    assert_eq!(snapshot_tree(&inference), original_inference);
    assert_eq!(
        fs::read_dir(temp.path().join("weights")).unwrap().count(),
        2
    );
}

#[test]
fn resume_cache_help_and_argument_contracts_are_explicit() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    for command in ["train", "resume", "prepare-cache"] {
        let help = succeeds(cli(temp.path(), &[command, "--help"]));
        for flag in [
            "--max-source-bytes",
            "--max-document-tokens",
            "--max-documents",
            "--max-directory-entries",
            "--max-manifest-bytes",
        ] {
            assert!(help.contains(flag), "{command}: {help}");
        }
        if command != "prepare-cache" {
            assert!(help.contains("--max-updates"), "{help}");
            assert!(help.contains("--cache"), "{help}");
        }
    }
    let resume = succeeds(cli(temp.path(), &["resume", "--help"]));
    assert!(resume.contains("Total target epochs"), "{resume}");
    assert!(!resume.contains("--learning-rate"), "{resume}");
    for arguments in [
        vec!["resume", "--checkpoint", "tiny-1", "--name", "next"],
        vec![
            "resume",
            "--checkpoint",
            "../tiny-1",
            "--name",
            "next",
            "--epochs",
            "2",
        ],
        vec![
            "resume",
            "--checkpoint",
            "tiny-1",
            "--name",
            "next",
            "--epochs",
            "2",
            "--learning-rate",
            "0.1",
        ],
        vec![
            "resume",
            "--checkpoint",
            "tiny-1",
            "--name",
            "next",
            "--epochs",
            "2",
            "--d-model",
            "8",
        ],
        vec!["prepare-cache", "--dataset", "first"],
    ] {
        fails(cli(temp.path(), &arguments), "error:");
    }
    let mut zero = train_arguments("data", "weights", "test.json");
    zero.extend(["--max-updates", "0"]);
    fails(cli(temp.path(), &zero), "Error:");
    assert!(!temp.path().join("weights").exists());
}

fn resume_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path.join("resume.json")).unwrap()).unwrap()
}

#[test]
fn periodic_update_epoch_and_final_saves_share_boundaries_and_resume_numbering() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let mut train = train_arguments("data", "weights", "test.json");
    replace_argument(&mut train, "--epochs", "2");
    train.extend([
        "--save-every-updates",
        "1",
        "--save-every-epochs",
        "1",
        "--metrics-jsonl",
        "periodic.jsonl",
        "--quiet",
    ]);
    let output = succeeds(cli(temp.path(), &train));
    assert_eq!(output.matches("Saved checkpoint:").count(), 4, "{output}");
    let weights = temp.path().join("weights");
    assert_eq!(fs::read_dir(&weights).unwrap().count(), 4);
    for number in 1..=4 {
        let path = weights.join(format!("tiny-{number}"));
        let saved = resume_json(&path);
        assert!(path.join("COMPLETE").is_file());
        assert_eq!(saved["progress"]["completed_updates"], number);
        assert_eq!(saved["progress"]["completed_epochs"], number / 2);
        assert_eq!(saved["progress"]["next_example_index"], number % 2);
    }
    let first = snapshot_tree(&weights.join("tiny-1"));
    let mut resume = resume_arguments("tiny-1", "continued", "continued.jsonl");
    resume.extend(["--save-every-updates", "2", "--save-every-epochs", "1"]);
    let output = succeeds(cli(temp.path(), &resume));
    assert_eq!(output.matches("Saved checkpoint:").count(), 2, "{output}");
    assert_eq!(fs::read_dir(&weights).unwrap().count(), 6);
    assert_eq!(
        resume_json(&weights.join("continued-1"))["progress"]["completed_updates"],
        2
    );
    assert_eq!(
        resume_json(&weights.join("continued-2"))["progress"]["completed_updates"],
        4
    );
    assert_eq!(
        checkpoint_logits(&weights.join("continued-2")),
        checkpoint_logits(&weights.join("tiny-4"))
    );
    assert_eq!(snapshot_tree(&weights.join("tiny-1")), first);
    assert_eq!(
        updates(&json_lines(&temp.path().join("continued.jsonl"))),
        updates(&json_lines(&temp.path().join("periodic.jsonl")))[1..]
    );
    for (name, flag, interval, boundaries) in [
        ("epochs", "--save-every-epochs", "1", [2, 4]),
        ("remainder", "--save-every-updates", "3", [3, 4]),
    ] {
        let mut arguments = train_arguments("data", "weights", "test.json");
        replace_argument(&mut arguments, "--epochs", "2");
        replace_argument(&mut arguments, "--name", name);
        arguments.extend([flag, interval, "--quiet"]);
        let output = succeeds(cli(temp.path(), &arguments));
        assert_eq!(output.matches("Saved checkpoint:").count(), 2, "{output}");
        for (index, boundary) in boundaries.into_iter().enumerate() {
            let path = weights.join(format!("{name}-{}", index + 1));
            assert_eq!(
                resume_json(&path)["progress"]["completed_updates"],
                boundary
            );
        }
        assert_eq!(
            checkpoint_logits(&weights.join(format!("{name}-2"))),
            checkpoint_logits(&weights.join("tiny-4"))
        );
    }
}

#[test]
fn periodic_save_failure_stops_after_its_committed_update() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    fs::write(temp.path().join("occupied-weights"), "preserve user file").unwrap();
    let mut train = train_arguments("data", "occupied-weights", "test.json");
    replace_argument(&mut train, "--epochs", "2");
    train.extend([
        "--save-every-updates",
        "1",
        "--metrics-jsonl",
        "failed.jsonl",
        "--quiet",
    ]);
    fails(cli(temp.path(), &train), "Error:");
    let records = json_lines(&temp.path().join("failed.jsonl"));
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["event"], "training_started");
    assert_eq!(records[1]["event"], "update");
    assert_eq!(records[1]["completed_updates"], 1);
    assert_eq!(
        fs::read_to_string(temp.path().join("occupied-weights")).unwrap(),
        "preserve user file"
    );
}

#[test]
fn shuffled_and_weighted_batches_clip_warmup_and_resume_exactly_with_cache() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    fs::write(temp.path().join("data/first/c.txt"), "hello test").unwrap();
    fs::write(temp.path().join("data/second/d.txt"), "world this is a").unwrap();
    let split = ["--validation-count", "1", "--split-seed", "7"];
    let mut cache = cache_arguments("cache");
    cache.extend(split);
    succeeds(cli(temp.path(), &cache));

    for (name, sampling, per_epoch, counts) in [
        ("shuffle", vec!["--shuffle"], 2usize, vec![2, 1]),
        (
            "weighted",
            vec![
                "--dataset-weight",
                "first=3",
                "--dataset-weight",
                "second=1",
                "--samples-per-epoch",
                "5",
            ],
            3,
            vec![2, 2, 1],
        ),
    ] {
        let full_log = format!("{name}-full.jsonl");
        let partial_name = format!("{name}-partial");
        let partial_log = format!("{name}-partial.jsonl");
        let final_name = format!("{name}-final");
        let final_log = format!("{name}-final.jsonl");
        let mut full = train_arguments("data", "weights", "test.json");
        replace_argument(&mut full, "--epochs", "2");
        replace_argument(&mut full, "--name", name);
        full.extend(split);
        full.extend(sampling);
        full.extend([
            "--batch-size",
            "2",
            "--max-batch-tokens",
            "8",
            "--gradient-clip-norm",
            "0.000001",
            "--warmup-updates",
            "3",
            "--metrics-jsonl",
            &full_log,
            "--quiet",
        ]);
        succeeds(cli(temp.path(), &full));
        let expected = json_lines(&temp.path().join(&full_log));
        let expected_updates = updates(&expected);
        assert_eq!(expected_updates.len(), 2 * per_epoch);
        for (index, record) in expected_updates.iter().enumerate() {
            assert_eq!(record["example_count"], counts[index % per_epoch]);
            let expected_rate = 0.003 * ((index + 1).min(3) as f64 / 3.0);
            assert!(
                (record["effective_learning_rate"].as_f64().unwrap() - expected_rate).abs() < 1e-15
            );
            assert!(record["gradient_norm"].as_f64().unwrap().is_finite());
            assert!(record["target_count"].as_u64().unwrap() >= counts[index % per_epoch]);
        }
        assert!(
            expected_updates
                .iter()
                .any(|record| record["clipped"] == true)
        );
        let expected_validations: Vec<_> = expected
            .iter()
            .filter(|r| r["event"] == "validation")
            .cloned()
            .collect();
        assert_eq!(expected_validations.len(), 2);

        let mut partial = full.clone();
        replace_argument(&mut partial, "--name", &partial_name);
        replace_argument(&mut partial, "--metrics-jsonl", &partial_log);
        partial.extend(["--cache", "cache", "--max-updates", "1"]);
        succeeds(cli(temp.path(), &partial));
        assert_eq!(
            updates(&json_lines(&temp.path().join(&partial_log))),
            expected_updates[..1]
        );
        let partial_checkpoint = format!("{partial_name}-1");
        let partial_path = temp.path().join("weights").join(&partial_checkpoint);
        let unchanged = snapshot_tree(&partial_path);
        let saved = resume_json(&partial_path);
        assert_eq!(saved["schema_version"], 3);
        assert_eq!(saved["progress"]["completed_updates"], 1);
        assert_eq!(saved["progress"]["next_example_index"], 2);
        assert_eq!(saved["options"]["batching"]["batch_size"], 2);
        assert_eq!(
            saved["options"]["sampling"]["kind"],
            if name == "shuffle" {
                "shuffle"
            } else {
                "weighted"
            }
        );
        let mut resume = resume_arguments(&partial_checkpoint, &final_name, &final_log);
        resume.extend(["--cache", "cache"]);
        succeeds(cli(temp.path(), &resume));
        let actual = json_lines(&temp.path().join(&final_log));
        assert_eq!(updates(&actual), expected_updates[1..]);
        assert_eq!(
            actual
                .iter()
                .filter(|r| r["event"] == "validation")
                .cloned()
                .collect::<Vec<_>>(),
            expected_validations
        );
        assert_eq!(actual.last(), expected.last());
        assert_eq!(
            checkpoint_logits(&temp.path().join("weights").join(format!("{final_name}-1"))),
            checkpoint_logits(&temp.path().join("weights").join(format!("{name}-1")))
        );
        assert_eq!(snapshot_tree(&partial_path), unchanged);
    }
}

#[test]
fn new_training_controls_reject_invalid_values_and_resume_overrides() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let train_help = succeeds(cli(temp.path(), &["train", "--help"]));
    let resume_help = succeeds(cli(temp.path(), &["resume", "--help"]));
    for flag in [
        "--shuffle",
        "--dataset-weight",
        "--samples-per-epoch",
        "--batch-size",
        "--max-batch-tokens",
        "--gradient-clip-norm",
        "--warmup-updates",
    ] {
        assert!(train_help.contains(flag), "{train_help}");
        assert!(!resume_help.contains(flag), "{resume_help}");
    }
    for flag in ["--save-every-updates", "--save-every-epochs"] {
        assert!(train_help.contains(flag) && resume_help.contains(flag));
    }
    for flags in [
        vec!["--shuffle", "--dataset-weight", "first=1"],
        vec!["--samples-per-epoch", "2"],
        vec!["--dataset-weight", "first=0"],
        vec!["--dataset-weight", "first=1.5"],
        vec!["--dataset-weight", "first"],
    ] {
        let mut arguments = train_arguments("data", "rejected", "test.json");
        arguments.extend(flags);
        fails(cli(temp.path(), &arguments), "error:");
    }
    for flags in [
        vec!["--batch-size", "0"],
        vec!["--max-batch-tokens", "0"],
        vec!["--batch-size", "2", "--max-batch-tokens", "1"],
        vec!["--gradient-clip-norm", "0"],
        vec!["--gradient-clip-norm", "NaN"],
        vec!["--gradient-clip-norm", "inf"],
        vec!["--save-every-updates", "0"],
        vec!["--save-every-epochs", "0"],
        vec!["--dataset-weight", "first=1"],
        vec![
            "--dataset-weight",
            "first=1",
            "--dataset-weight",
            "first=2",
            "--dataset-weight",
            "second=1",
        ],
        vec![
            "--dataset-weight",
            "first=1",
            "--dataset-weight",
            "second=1",
            "--samples-per-epoch",
            "0",
        ],
    ] {
        let mut arguments = train_arguments("data", "rejected", "test.json");
        arguments.extend(flags);
        fails(cli(temp.path(), &arguments), "Error:");
    }
    for flags in [
        vec!["--shuffle"],
        vec!["--batch-size", "2"],
        vec!["--dataset-weight", "first=1"],
        vec!["--samples-per-epoch", "3"],
        vec!["--max-batch-tokens", "4"],
        vec!["--gradient-clip-norm", "1"],
        vec!["--warmup-updates", "3"],
    ] {
        let mut arguments = resume_arguments("absent-1", "rejected", "rejected.jsonl");
        arguments.extend(flags);
        fails(cli(temp.path(), &arguments), "error:");
    }
    assert!(!temp.path().join("rejected").exists());
    assert!(!temp.path().join("rejected.jsonl").exists());

    let mut weighted = train_arguments("data", "weights", "test.json");
    weighted.extend([
        "--dataset-weight",
        "first=1",
        "--dataset-weight",
        "second=2",
        "--metrics-jsonl",
        "weighted-default.jsonl",
        "--quiet",
    ]);
    succeeds(cli(temp.path(), &weighted));
    let records = json_lines(&temp.path().join("weighted-default.jsonl"));
    assert_eq!(records[0]["examples_per_epoch"], 2);
    assert_eq!(updates(&records).len(), 2);
    assert_eq!(
        resume_json(&temp.path().join("weights/tiny-1"))["options"]["sampling"]["samples_per_epoch"],
        2
    );
}

fn generated_ids(output: &str) -> Vec<u32> {
    let ids = output
        .lines()
        .find_map(|line| line.strip_prefix("Token IDs: "))
        .unwrap();
    serde_json::from_str(ids).unwrap()
}

#[test]
fn generation_sampling_is_repeatable_and_top_k_one_preserves_unique_greedy_choices() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
    ));
    let base = [
        "generate",
        "--weights-root",
        "weights",
        "--checkpoint",
        "tiny-1",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "3",
    ];
    let greedy = generated_ids(&succeeds(cli(temp.path(), &base)));
    let path = temp.path().join("weights/tiny-1");
    let unchanged = snapshot_tree(&path);
    let (model, config, tokenizer) = omega_training::load_checkpoint(&path).unwrap();
    let mut reference = omega_nn::encode_text(&tokenizer, "hello").unwrap();
    for _ in 0..3 {
        let input = omega_nn::token_tensor::<omega_nn::Cpu>(
            &reference,
            config.vocab_size,
            config.context_length,
            &Default::default(),
        )
        .unwrap();
        let logits = model.forward(input).into_data().to_vec::<f32>().unwrap();
        let mut choices: Vec<_> = logits[logits.len() - config.vocab_size..]
            .iter()
            .copied()
            .enumerate()
            .collect();
        choices.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert!(
            choices[0].1 > choices[1].1,
            "fixture must have a unique best logit"
        );
        reference.push(choices[0].0 as u32);
    }
    assert_eq!(greedy, reference);
    let mut top_one = base.to_vec();
    top_one.extend(["--sample", "--top-k", "1", "--sampling-seed", "99"]);
    assert_eq!(generated_ids(&succeeds(cli(temp.path(), &top_one))), greedy);

    let mut sampled = base.to_vec();
    sampled.extend([
        "--sample",
        "--temperature",
        "0.7",
        "--top-k",
        "3",
        "--top-p",
        "0.9",
        "--sampling-seed",
        "17",
    ]);
    let first = generated_ids(&succeeds(cli(temp.path(), &sampled)));
    assert_eq!(first, generated_ids(&succeeds(cli(temp.path(), &sampled))));
    assert_eq!(first.len(), 4);
    assert_eq!(first[0], 2);
    assert!(first.iter().all(|id| (*id as usize) < config.vocab_size));
    let mut defaults = base.to_vec();
    defaults.push("--sample");
    let expected = generated_ids(&succeeds(cli(temp.path(), &defaults)));
    defaults.extend(["--temperature", "1", "--sampling-seed", "42"]);
    assert_eq!(
        generated_ids(&succeeds(cli(temp.path(), &defaults))),
        expected
    );
    replace_argument(&mut defaults, "--max-new-tokens", "0");
    assert_eq!(generated_ids(&succeeds(cli(temp.path(), &defaults))), [2]);
    assert_eq!(snapshot_tree(&path), unchanged);
}

#[test]
fn sampling_flags_and_checkpoint_selectors_have_checked_help_and_errors() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
    ));
    let base = [
        "generate",
        "--weights-root",
        "weights",
        "--checkpoint",
        "tiny-1",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
    ];
    let help = succeeds(cli(temp.path(), &["generate", "--help"]));
    for flag in [
        "--sample",
        "--temperature",
        "--top-k",
        "--top-p",
        "--sampling-seed",
        "--latest-run",
    ] {
        assert!(help.contains(flag), "{help}");
    }
    for options in [
        vec!["--temperature", "1"],
        vec!["--top-k", "1"],
        vec!["--top-p", "1"],
        vec!["--sampling-seed", "42"],
    ] {
        let mut arguments = base.to_vec();
        arguments.extend(options);
        fails(cli(temp.path(), &arguments), "error:");
    }
    for options in [
        vec!["--temperature", "0"],
        vec!["--temperature", "NaN"],
        vec!["--temperature", "inf"],
        vec!["--top-k", "0"],
        vec!["--top-k", "14"],
        vec!["--top-p", "0"],
        vec!["--top-p", "1.1"],
        vec!["--top-p", "NaN"],
        vec!["--top-p", "inf"],
    ] {
        let mut arguments = base.to_vec();
        arguments.push("--sample");
        arguments.extend(options);
        fails(cli(temp.path(), &arguments), "Error:");
    }
    for command in ["generate", "evaluate", "resume"] {
        let help = succeeds(cli(temp.path(), &[command, "--help"]));
        assert!(
            help.contains("--checkpoint") && help.contains("--latest-run"),
            "{help}"
        );
        let mandatory = match command {
            "generate" => vec![command, "--prompt", "hello"],
            "evaluate" => vec![command, "--dataset", "first"],
            _ => vec![command, "--name", "next", "--epochs", "2"],
        };
        fails(cli(temp.path(), &mandatory), "error:");
        let mut conflicting = mandatory.clone();
        conflicting.extend(["--checkpoint", "tiny-1", "--latest-run", "tiny"]);
        fails(cli(temp.path(), &conflicting), "error:");
        let mut unsafe_name = mandatory;
        unsafe_name.extend(["--latest-run", "../tiny"]);
        fails(cli(temp.path(), &unsafe_name), "error:");
    }
    for subcommand in ["list", "latest"] {
        let help = succeeds(cli(temp.path(), &["checkpoints", subcommand, "--help"]));
        for flag in [
            "--weights-root",
            "--run",
            "--json",
            "--max-entries",
            "--max-metadata-bytes",
        ] {
            assert!(help.contains(flag), "{help}");
        }
        fails(
            cli(
                temp.path(),
                &["checkpoints", subcommand, "--run", "../tiny"],
            ),
            "error:",
        );
        for flag in ["--max-entries", "--max-metadata-bytes"] {
            fails(
                cli(
                    temp.path(),
                    &[
                        "checkpoints",
                        subcommand,
                        "--weights-root",
                        "weights",
                        "--run",
                        "tiny",
                        flag,
                        "0",
                    ],
                ),
                "Error:",
            );
        }
    }
}

fn cli_json(output: Output) -> serde_json::Value {
    let output = succeeds(output);
    // The subprocess harness prints its test-start prefix before the CLI JSON.
    let start = output.find('{').expect("CLI must emit a JSON object");
    serde_json::from_str(&output[start..]).unwrap()
}

fn copy_test_checkpoint(from: &Path, to: &Path) {
    // Only this test's freshly generated, flat checkpoint directories are copied.
    fs::create_dir(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

fn catalog_arguments<'a>(command: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut arguments = vec![
        "checkpoints",
        command,
        "--weights-root",
        "weights",
        "--run",
        "TiNy",
        "--json",
    ];
    arguments.extend_from_slice(extra);
    arguments
}

#[test]
fn checkpoint_catalog_and_latest_commands_choose_numeric_candidates_by_mode() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let mut train = train_arguments("data", "weights", "test.json");
    train.push("--quiet");
    succeeds(cli(temp.path(), &train));
    train.extend(["--seed", "99"]);
    succeeds(cli(temp.path(), &train));
    let weights = temp.path().join("weights");
    let (model, config, tokenizer) =
        omega_training::load_checkpoint(&weights.join("tiny-1")).unwrap();
    let inference =
        omega_training::save_checkpoint(&weights, "tiny", model, &config, &tokenizer).unwrap();
    assert!(inference.ends_with("tiny-3"));
    copy_test_checkpoint(&inference, &weights.join("TiNy-10"));
    // This smaller number is created later: modification/creation order is irrelevant.
    copy_test_checkpoint(&weights.join("tiny-2"), &weights.join("tiny-4"));
    copy_test_checkpoint(&inference, &weights.join("tiny-5"));
    fs::remove_file(weights.join("tiny-5/manifest.json")).unwrap();
    fs::write(weights.join("tiny-5/COMPLETE"), []).unwrap();
    fs::create_dir(weights.join("tiny-11")).unwrap();
    fs::write(weights.join("tiny-12"), "occupied number").unwrap();
    let original = snapshot_tree(&weights);

    let list = cli_json(cli(temp.path(), &catalog_arguments("list", &[])));
    assert!(Path::new(list["root"].as_str().unwrap()).is_absolute());
    assert!(
        list["run_name"]
            .as_str()
            .unwrap()
            .eq_ignore_ascii_case("tiny")
    );
    assert_eq!(list["inspection"], "metadata_only_payloads_unverified");
    let entries = list["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 8);
    for (number, status) in [
        (1, "resumable"),
        (2, "resumable"),
        (3, "complete_inference"),
        (4, "resumable"),
        (5, "legacy_unverified"),
        (10, "complete_inference"),
        (11, "incomplete"),
        (12, "incomplete"),
    ] {
        let entry = entries
            .iter()
            .find(|entry| entry["number"] == number)
            .unwrap();
        assert_eq!(entry["status"], status);
        assert!(Path::new(entry["path"].as_str().unwrap()).is_absolute());
        if number >= 11 {
            assert!(!entry["reason"].as_str().unwrap().is_empty());
        }
    }
    let latest = cli_json(cli(temp.path(), &catalog_arguments("latest", &[])));
    assert_eq!(latest["number"], 10);
    assert_eq!(latest["status"], "complete_inference");
    assert!(Path::new(latest["path"].as_str().unwrap()).ends_with("TiNy-10"));
    assert_eq!(latest["inspection"], "metadata_only_payloads_unverified");
    let skipped: Vec<_> = latest["skipped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["number"].as_u64().unwrap())
        .collect();
    assert_eq!(skipped, [12, 11]);
    let resumable = cli_json(cli(
        temp.path(),
        &catalog_arguments("latest", &["--mode", "resume"]),
    ));
    assert_eq!(resumable["number"], 4);
    assert_eq!(resumable["status"], "resumable");
    assert_eq!(
        resumable["skipped"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["number"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [12, 11, 10, 5]
    );
    let human = succeeds(cli(
        temp.path(),
        &[
            "checkpoints",
            "list",
            "--weights-root",
            "weights",
            "--run",
            "tiny",
        ],
    ));
    assert!(
        human.contains("tiny-4") && human.contains("TiNy-10"),
        "{human}"
    );
    fails(
        cli(
            temp.path(),
            &catalog_arguments("list", &["--max-entries", "1"]),
        ),
        "Error:",
    );
    fails(
        cli(
            temp.path(),
            &catalog_arguments("latest", &["--max-metadata-bytes", "1"]),
        ),
        "Error:",
    );

    let mut generation = vec![
        "generate",
        "--weights-root",
        "weights",
        "--checkpoint",
        "TiNy-10",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
    ];
    let expected = succeeds(cli(temp.path(), &generation));
    let selector = generation
        .iter()
        .position(|item| *item == "--checkpoint")
        .unwrap();
    generation[selector] = "--latest-run";
    generation[selector + 1] = "tiny";
    assert_eq!(succeeds(cli(temp.path(), &generation)), expected);
    let mut evaluation = vec![
        "evaluate",
        "--weights-root",
        "weights",
        "--datasets-root",
        "data",
        "--checkpoint",
        "TiNy-10",
        "--dataset",
        "first",
    ];
    let expected = succeeds(cli(temp.path(), &evaluation));
    let selector = evaluation
        .iter()
        .position(|item| *item == "--checkpoint")
        .unwrap();
    evaluation[selector] = "--latest-run";
    evaluation[selector + 1] = "tiny";
    assert_eq!(succeeds(cli(temp.path(), &evaluation)), expected);
    assert_eq!(snapshot_tree(&weights), original);

    let explicit = resume_arguments("tiny-4", "explicit", "explicit.jsonl");
    succeeds(cli(temp.path(), &explicit));
    let mut automatic = resume_arguments("tiny", "automatic", "automatic.jsonl");
    let selector = automatic
        .iter()
        .position(|item| *item == "--checkpoint")
        .unwrap();
    automatic[selector] = "--latest-run";
    let output = cli(temp.path(), &automatic);
    assert!(String::from_utf8_lossy(&output.stderr).contains("tiny-4"));
    succeeds(output);
    assert_eq!(
        checkpoint_logits(&weights.join("automatic-1")),
        checkpoint_logits(&weights.join("explicit-1"))
    );
    assert_eq!(
        json_lines(&temp.path().join("automatic.jsonl")),
        json_lines(&temp.path().join("explicit.jsonl"))
    );
    for (name, bytes) in original {
        assert_eq!(fs::read(weights.join(name)).unwrap(), bytes);
    }
}

#[test]
fn latest_never_falls_back_from_corrupt_completed_payloads_or_metadata() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
    ));
    let weights = temp.path().join("weights");
    copy_test_checkpoint(&weights.join("tiny-1"), &weights.join("tiny-5"));
    fs::write(weights.join("tiny-5/model.mpk"), "corrupt payload").unwrap();
    let original = snapshot_tree(&weights);
    // Catalog selection is explicitly header-only; payload failure belongs to loading.
    let selected = cli_json(cli(temp.path(), &catalog_arguments("latest", &[])));
    assert_eq!(selected["number"], 5);
    assert_eq!(selected["inspection"], "metadata_only_payloads_unverified");
    let generation = [
        "generate",
        "--weights-root",
        "weights",
        "--latest-run",
        "tiny",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "1",
    ];
    let evaluation = [
        "evaluate",
        "--weights-root",
        "weights",
        "--datasets-root",
        "data",
        "--latest-run",
        "tiny",
        "--dataset",
        "first",
    ];
    let mut resume = resume_arguments("tiny", "never-created", "never-created.jsonl");
    let selector = resume
        .iter()
        .position(|item| *item == "--checkpoint")
        .unwrap();
    resume[selector] = "--latest-run";
    for arguments in [&generation[..], &evaluation[..], &resume[..]] {
        let output = cli(temp.path(), arguments);
        assert!(String::from_utf8_lossy(&output.stderr).contains("tiny-5"));
        fails(output, "Error:");
    }
    succeeds(cli(
        temp.path(),
        &[
            "generate",
            "--weights-root",
            "weights",
            "--checkpoint",
            "tiny-1",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ],
    ));
    assert!(!temp.path().join("never-created.jsonl").exists());
    assert_eq!(snapshot_tree(&weights), original);

    fs::write(weights.join("tiny-5/manifest.json"), "{").unwrap();
    let corrupt = snapshot_tree(&weights);
    let list = cli_json(cli(temp.path(), &catalog_arguments("list", &[])));
    let newest = list["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["number"] == 5)
        .unwrap();
    assert_eq!(newest["status"], "invalid");
    assert!(!newest["reason"].as_str().unwrap().is_empty());
    fails(
        cli(temp.path(), &catalog_arguments("latest", &[])),
        "Error:",
    );
    fails(cli(temp.path(), &generation), "Error:");
    assert_eq!(snapshot_tree(&weights), corrupt);
}

#[test]
fn latest_resume_rejects_selected_source_or_runtime_mismatch_without_fallback() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
    ));
    let source = temp.path().join("data/first/a.txt");
    let original_source = fs::read(&source).unwrap();
    fs::write(&source, "omega world hello").unwrap();
    succeeds(cli(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
    ));
    fs::write(&source, original_source).unwrap();
    let weights = temp.path().join("weights");
    let original = snapshot_tree(&weights);
    let mut latest = resume_arguments("tiny", "rejected", "rejected.jsonl");
    let selector = latest
        .iter()
        .position(|item| *item == "--checkpoint")
        .unwrap();
    latest[selector] = "--latest-run";
    let output = cli(temp.path(), &latest);
    assert!(String::from_utf8_lossy(&output.stderr).contains("tiny-2"));
    fails(output, "provenance");
    assert_eq!(snapshot_tree(&weights), original);
    assert!(!temp.path().join("rejected.jsonl").exists());

    // A valid older snapshot exists, but an unsupported newest runtime is fatal.
    copy_test_checkpoint(&weights.join("tiny-1"), &weights.join("tiny-3"));
    let mut unsupported = resume_json(&weights.join("tiny-3"));
    unsupported["runtime"]["operating_system"] = "unsupported-test-platform".into();
    let bytes = serde_json::to_vec(&unsupported).unwrap();
    fs::write(weights.join("tiny-3/resume.json"), &bytes).unwrap();
    fs::write(
        weights.join("tiny-3/resume.sha256"),
        omega_training::checkpoint::sha256_bytes(&bytes),
    )
    .unwrap();
    let original = snapshot_tree(&weights);
    fails(cli(temp.path(), &latest), "Error:");
    assert_eq!(snapshot_tree(&weights), original);
    assert!(!temp.path().join("rejected.jsonl").exists());
    succeeds(cli(
        temp.path(),
        &resume_arguments("tiny-1", "explicit-older", "explicit-older.jsonl"),
    ));
}

#[test]
fn latest_rejects_duplicate_numeric_aliases_and_no_eligible_candidates() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    fs::create_dir(temp.path().join("weights")).unwrap();
    fs::write(
        temp.path().join("weights/unrelated.txt"),
        "count this root entry",
    )
    .unwrap();
    let empty = cli_json(cli(temp.path(), &catalog_arguments("list", &[])));
    assert!(empty["entries"].as_array().unwrap().is_empty());
    fails(
        cli(temp.path(), &catalog_arguments("latest", &[])),
        "Error:",
    );
    succeeds(cli(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
    ));
    fails(
        cli(
            temp.path(),
            &catalog_arguments("list", &["--max-entries", "1"]),
        ),
        "Error:",
    );
    let weights = temp.path().join("weights");
    copy_test_checkpoint(&weights.join("tiny-1"), &weights.join("TiNy-01"));
    let original = snapshot_tree(&weights);
    let list = cli_json(cli(temp.path(), &catalog_arguments("list", &[])));
    assert_eq!(list["entries"].as_array().unwrap().len(), 2);
    assert!(
        list["entries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|entry| entry["number"] == 1)
    );
    for mode in ["inference", "resume"] {
        fails(
            cli(temp.path(), &catalog_arguments("latest", &["--mode", mode])),
            "Error:",
        );
    }
    fails(
        cli(
            temp.path(),
            &[
                "generate",
                "--weights-root",
                "weights",
                "--latest-run",
                "tiny",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "1",
            ],
        ),
        "Error:",
    );
    succeeds(cli(
        temp.path(),
        &[
            "generate",
            "--weights-root",
            "weights",
            "--checkpoint",
            "tiny-1",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
        ],
    ));
    assert_eq!(snapshot_tree(&weights), original);
}

const POOL_KEYS: [&str; 3] = [
    "RAYON_NUM_THREADS",
    "RAYON_RS_NUM_CPUS",
    "MATMUL_NUM_THREADS",
];
const UNSET_POOLS: [(&str, Option<&str>); 3] = [
    ("RAYON_NUM_THREADS", None),
    ("RAYON_RS_NUM_CPUS", None),
    ("MATMUL_NUM_THREADS", None),
];

#[test]
fn cpu_flags_override_child_environment_and_omission_preserves_inheritance() {
    let original: Vec<_> = POOL_KEYS.iter().map(std::env::var_os).collect();
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    let inherited = [
        ("RAYON_NUM_THREADS", Some("2")),
        ("RAYON_RS_NUM_CPUS", Some("20")),
        ("MATMUL_NUM_THREADS", Some("2")),
    ];
    let mut arguments = train_arguments("data", "weights", "test.json");
    replace_argument(&mut arguments, "--name", "inherited");
    arguments.push("--quiet");
    succeeds(cli_with_env(temp.path(), &arguments, &inherited));
    let saved = resume_json(&temp.path().join("weights/inherited-1"));
    assert_eq!(saved["schema_version"], 3);
    let profile = &saved["cpu_execution"];
    assert_eq!(profile["kernel"], "ndarray-f32-checked-v1");
    assert_eq!(profile["rayon_num_threads"], "2");
    assert_eq!(profile["rayon_rs_num_cpus"], "20");
    assert_eq!(profile["matmul_num_threads"], "2");
    assert!(
        profile
            .as_object()
            .unwrap()
            .contains_key("available_parallelism")
    );
    assert!(
        profile["available_parallelism"].is_null()
            || profile["available_parallelism"].as_u64().unwrap() > 0
    );

    replace_argument(&mut arguments, "--name", "override");
    arguments.extend(["--cpu-threads", "4", "--matmul-threads", "1"]);
    succeeds(cli_with_env(temp.path(), &arguments, &inherited));
    let overridden = resume_json(&temp.path().join("weights/override-1"));
    assert_eq!(overridden["cpu_execution"]["rayon_num_threads"], "4");
    assert_eq!(overridden["cpu_execution"]["rayon_rs_num_cpus"], "4");
    assert_eq!(overridden["cpu_execution"]["matmul_num_threads"], "1");
    assert_eq!(
        POOL_KEYS.iter().map(std::env::var_os).collect::<Vec<_>>(),
        original
    );
}

#[test]
fn cpu_thread_counts_preserve_tiny_training_logits_and_greedy_generation() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    for (name, rayon, matmul) in [("one", "1", "1"), ("several", "4", "2")] {
        let mut arguments = train_arguments("data", "weights", "test.json");
        replace_argument(&mut arguments, "--name", name);
        arguments.extend([
            "--cpu-threads",
            rayon,
            "--matmul-threads",
            matmul,
            "--quiet",
        ]);
        succeeds(cli_with_env(temp.path(), &arguments, &UNSET_POOLS));
    }
    let one = checkpoint_logits(&temp.path().join("weights/one-1"));
    let several = checkpoint_logits(&temp.path().join("weights/several-1"));
    assert_eq!(one.len(), several.len());
    for (left, right) in one.iter().zip(&several) {
        // Same-host CPU f32 parity; changing pool configuration is not an exact-resume promise.
        assert!(
            left.is_finite() && right.is_finite() && (left - right).abs() <= 1e-5,
            "{left} versus {right}"
        );
    }
    let mut expected = None;
    for (name, rayon, matmul) in [
        ("one-1", "1", "1"),
        ("one-1", "4", "2"),
        ("several-1", "4", "2"),
    ] {
        let arguments = [
            "--cpu-threads",
            rayon,
            "--matmul-threads",
            matmul,
            "generate",
            "--weights-root",
            "weights",
            "--checkpoint",
            name,
            "--prompt",
            "hello",
            "--max-new-tokens",
            "3",
        ];
        let ids = generated_ids(&succeeds(cli_with_env(
            temp.path(),
            &arguments,
            &UNSET_POOLS,
        )));
        if let Some(expected) = &expected {
            assert_eq!(&ids, expected);
        } else {
            expected = Some(ids);
        }
    }
}

#[test]
fn controlled_cpu_resume_matches_full_run_and_rejects_profile_mismatches_before_updates() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    fs::write(temp.path().join("data/first/c.txt"), "hello test").unwrap();
    let inherited = [
        ("RAYON_NUM_THREADS", Some("8")),
        ("RAYON_RS_NUM_CPUS", Some("20")),
        ("MATMUL_NUM_THREADS", Some("4")),
    ];
    let mut full = train_arguments("data", "weights", "test.json");
    replace_argument(&mut full, "--epochs", "2");
    full.extend([
        "--cpu-threads",
        "2",
        "--matmul-threads",
        "1",
        "--shuffle",
        "--batch-size",
        "2",
        "--gradient-clip-norm",
        "0.000001",
        "--warmup-updates",
        "3",
        "--metrics-jsonl",
        "full.jsonl",
        "--quiet",
    ]);
    succeeds(cli_with_env(temp.path(), &full, &inherited));
    let expected = json_lines(&temp.path().join("full.jsonl"));
    assert_eq!(updates(&expected).len(), 4);
    let mut partial = full.clone();
    replace_argument(&mut partial, "--name", "partial");
    replace_argument(&mut partial, "--metrics-jsonl", "partial.jsonl");
    partial.extend(["--max-updates", "1"]);
    succeeds(cli_with_env(temp.path(), &partial, &inherited));
    assert_eq!(
        updates(&json_lines(&temp.path().join("partial.jsonl"))),
        updates(&expected)[..1]
    );
    let weights = temp.path().join("weights");
    let original = snapshot_tree(&weights);
    for (rayon, matmul) in [("1", "1"), ("2", "2")] {
        let mut rejected = resume_arguments("partial-1", "rejected", "rejected.jsonl");
        rejected.extend(["--cpu-threads", rayon, "--matmul-threads", matmul]);
        fails(
            cli_with_env(temp.path(), &rejected, &inherited),
            "Resume CPU execution profile differs",
        );
        assert!(!temp.path().join("rejected.jsonl").exists());
        assert_eq!(snapshot_tree(&weights), original);
    }
    let raw_alias_mismatch = [
        ("RAYON_NUM_THREADS", Some("2")),
        ("RAYON_RS_NUM_CPUS", Some("3")),
        ("MATMUL_NUM_THREADS", Some("1")),
    ];
    fails(
        cli_with_env(
            temp.path(),
            &resume_arguments("partial-1", "rejected", "rejected.jsonl"),
            &raw_alias_mismatch,
        ),
        "Resume CPU execution profile differs",
    );
    assert!(!temp.path().join("rejected.jsonl").exists());
    assert_eq!(snapshot_tree(&weights), original);

    let mut matching = resume_arguments("partial-1", "restored", "restored.jsonl");
    matching.extend(["--cpu-threads", "2", "--matmul-threads", "1"]);
    succeeds(cli_with_env(temp.path(), &matching, &raw_alias_mismatch));
    let actual = json_lines(&temp.path().join("restored.jsonl"));
    assert_eq!(updates(&actual), updates(&expected)[1..]);
    assert_eq!(actual.last(), expected.last());
    assert_eq!(
        checkpoint_logits(&weights.join("restored-1")),
        checkpoint_logits(&weights.join("tiny-1"))
    );
    assert_eq!(
        resume_json(&weights.join("restored-1"))["cpu_execution"],
        resume_json(&weights.join("partial-1"))["cpu_execution"]
    );
    for (path, bytes) in original {
        assert_eq!(fs::read(weights.join(path)).unwrap(), bytes);
    }
}

#[test]
fn legacy_schema_two_cpu_resume_requires_unset_pool_environment() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    succeeds(cli_with_env(
        temp.path(),
        &train_arguments("data", "weights", "test.json"),
        &UNSET_POOLS,
    ));
    let weights = temp.path().join("weights");
    copy_test_checkpoint(&weights.join("tiny-1"), &weights.join("tiny-2"));
    let mut legacy = resume_json(&weights.join("tiny-2"));
    assert_eq!(legacy["schema_version"], 3);
    legacy["schema_version"] = 2.into();
    assert!(
        legacy
            .as_object_mut()
            .unwrap()
            .remove("cpu_execution")
            .is_some()
    );
    let bytes = serde_json::to_vec(&legacy).unwrap();
    fs::write(weights.join("tiny-2/resume.json"), &bytes).unwrap();
    fs::write(
        weights.join("tiny-2/resume.sha256"),
        omega_training::checkpoint::sha256_bytes(&bytes),
    )
    .unwrap();
    let original = snapshot_tree(&weights);
    let mut rejected = resume_arguments("tiny-2", "rejected", "rejected.jsonl");
    rejected.extend(["--cpu-threads", "1", "--matmul-threads", "1"]);
    fails(
        cli_with_env(temp.path(), &rejected, &UNSET_POOLS),
        "Legacy resume has no CPU execution profile",
    );
    assert!(!temp.path().join("rejected.jsonl").exists());
    assert_eq!(snapshot_tree(&weights), original);
    let inherited = [
        ("RAYON_NUM_THREADS", Some("1")),
        ("RAYON_RS_NUM_CPUS", None),
        ("MATMUL_NUM_THREADS", None),
    ];
    fails(
        cli_with_env(
            temp.path(),
            &resume_arguments("tiny-2", "rejected", "rejected.jsonl"),
            &inherited,
        ),
        "Legacy resume has no CPU execution profile",
    );
    assert!(!temp.path().join("rejected.jsonl").exists());

    succeeds(cli_with_env(
        temp.path(),
        &resume_arguments("tiny-2", "migrated", "migrated.jsonl"),
        &UNSET_POOLS,
    ));
    let saved = resume_json(&weights.join("migrated-1"));
    assert_eq!(saved["schema_version"], 3);
    for key in [
        "rayon_num_threads",
        "rayon_rs_num_cpus",
        "matmul_num_threads",
    ] {
        assert!(saved["cpu_execution"][key].is_null());
    }
    for (path, bytes) in original {
        assert_eq!(fs::read(weights.join(path)).unwrap(), bytes);
    }
}

#[test]
fn global_cpu_controls_have_help_and_reject_unsupported_thread_counts() {
    let temp = tempfile::tempdir().unwrap();
    prepare(temp.path());
    for command in [
        None,
        Some("train"),
        Some("generate"),
        Some("resume"),
        Some("evaluate"),
    ] {
        let mut arguments = Vec::new();
        if let Some(command) = command {
            arguments.push(command);
        }
        arguments.push("--help");
        let help = succeeds(cli(temp.path(), &arguments));
        assert!(
            help.contains("--cpu-threads") && help.contains("--matmul-threads"),
            "{help}"
        );
    }
    for (flag, value) in [
        ("--cpu-threads", "0"),
        ("--cpu-threads", "257"),
        ("--matmul-threads", "0"),
        ("--matmul-threads", "3"),
        ("--matmul-threads", "5"),
    ] {
        let mut arguments = train_arguments("data", "rejected", "test.json");
        arguments.extend([flag, value]);
        fails(
            cli_with_env(temp.path(), &arguments, &UNSET_POOLS),
            "thread",
        );
    }
    assert!(!temp.path().join("rejected").exists());
}
