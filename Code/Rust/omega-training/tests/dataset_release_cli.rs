//! Real omega-datasets producer -> training CLI integration, using an offline Hub
//! and tiny owned fixtures. This validates plumbing, not training quality.
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use omega_datasets::{
    BuildReport,
    config::{Config, Source},
    hub::{Hub, RemoteFile, ResolvedSource},
};
use serde_json::{Value, json};

const REVISION: &str = "0123456789012345678901234567890123456789";
const BASE_TRAIN: &str = "prepared/release-v1/base/train";
const CHAT_TRAIN: &str = "prepared/release-v1/chat/train";

struct MemoryHub(BTreeMap<String, Vec<u8>>);
impl Hub for MemoryHub {
    fn resolve(&self, source: &Source) -> Result<ResolvedSource> {
        Ok(ResolvedSource {
            source_id: source.id.clone(),
            repo: source.repo.clone(),
            revision: REVISION.into(),
            files: source
                .files
                .iter()
                .map(|path| RemoteFile {
                    path: path.clone(),
                    size: Some(self.0[path].len() as u64),
                })
                .collect(),
        })
    }

    fn download(
        &self,
        source: &ResolvedSource,
        file: &RemoteFile,
        output: &mut dyn Write,
        limit: u64,
    ) -> Result<u64> {
        ensure!(
            source.revision == REVISION,
            "fixture revision must be pinned"
        );
        let bytes = &self.0[&file.path];
        ensure!(
            bytes.len() as u64 <= limit,
            "fixture download exceeds limit"
        );
        output.write_all(bytes)?;
        Ok(bytes.len() as u64)
    }
}

fn publish(root: &Path, include_base: bool) -> (PathBuf, BuildReport) {
    let folder = root.join("data/prepared");
    fs::create_dir_all(&folder).unwrap();
    let mut text = String::from(
        "schema_version=1\noutput='release-v1'\nseed=42\n[split]\ntrain=0.5\nvalidation=0.25\ntest=0.25\n",
    );
    let mut files = BTreeMap::new();
    for stage in ["base", "chat"] {
        if stage == "base" && !include_base {
            continue;
        }
        for partition in ["train", "validation", "test"] {
            let id = format!("{stage}-{partition}");
            let mapping = if stage == "base" {
                "{kind='text',column='text'}"
            } else {
                "{kind='instruction',prompt_column='ask',response_column='answer'}"
            };
            text.push_str(&format!("\n[[sources]]\nid='{id}'\nrepo='offline/fixture'\nfiles=['{id}.jsonl']\nformat='jsonl'\nlanguage='en'\npermitted_use='Synthetic integration fixture'\npartition='{partition}'\nmapping={mapping}\n"));
            let count = if partition == "train" { 2 } else { 1 };
            let rows = (0..count).map(|i| {
                if stage == "base" {
                    json!({"text":format!("A {partition} base sentence number {i}.")})
                } else {
                    json!({"ask":format!("{partition} question {i}?"),"answer":format!("{partition} reply {i}.")})
                }.to_string()
            }).collect::<Vec<_>>().join("\n");
            files.insert(format!("{id}.jsonl"), rows.into_bytes());
        }
    }
    let config_path = folder.join("model.toml");
    fs::write(&config_path, text).unwrap();
    let config = Config::load(&config_path).unwrap();
    let report = omega_datasets::build(&folder, &config, &MemoryHub(files)).unwrap();
    let release = folder.join(&config.output);
    assert!(release.join("COMPLETE").is_file());
    (release, report)
}

fn cli(root: &Path, arguments: &[&str]) -> Output {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let index = NEXT.fetch_add(1, Ordering::Relaxed);
    let stdout = root.join(format!("release-child-{index}.stdout"));
    let stderr = root.join(format!("release-child-{index}.stderr"));
    let mut child = Command::new(env!("CARGO_BIN_EXE_omega-training"))
        .current_dir(root)
        .args(arguments)
        .env("RAYON_NUM_THREADS", "1")
        .env("RAYON_RS_NUM_CPUS", "1")
        .env("MATMUL_NUM_THREADS", "1")
        .env_remove("OMEGA_CPU_REEXEC_V1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "CLI exceeded deadline: {arguments:?}\nstdout:{}\nstderr:{}",
                fs::read_to_string(stdout).unwrap(),
                fs::read_to_string(stderr).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: fs::read(stdout).unwrap(),
        stderr: fs::read(stderr).unwrap(),
    }
}

fn succeeds(output: Output) -> String {
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}
fn fails(output: Output) {
    assert!(!output.status.success(), "Unexpected success: {output:?}");
    assert!(
        !output.stderr.is_empty(),
        "Failure must have an actionable diagnostic"
    );
}
fn read_json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn tokenizer_attempt(root: &Path, selection: &str, output: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "train-tokenizer",
        "--datasets-root",
        "data",
        "--dataset",
        selection,
        "--output",
        output,
        "--vocab-size",
        "260",
        "--chat-protocol",
    ];
    args.extend_from_slice(extra);
    cli(root, &args)
}

#[test]
fn produced_partitions_train_cache_evaluate_initialize_stage_and_resume() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let (release, report) = publish(root, true);
    assert_eq!(report.counts["base/train"], 2);
    assert_eq!(report.counts["chat/train"], 2);
    // No format flag: the consumer must resolve the producer's base/chat schemas.
    succeeds(tokenizer_attempt(
        root,
        BASE_TRAIN,
        "data/tokenizer.json",
        &[],
    ));
    succeeds(cli(
        root,
        &[
            "prepare-cache",
            "--datasets-root",
            "data",
            "--dataset",
            BASE_TRAIN,
            "--tokenizer",
            "tokenizer.json",
            "--context-length",
            "128",
            "--output",
            "base-cache",
        ],
    ));
    succeeds(cli(
        root,
        &[
            "train",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--dataset",
            BASE_TRAIN,
            "--tokenizer",
            "tokenizer.json",
            "--cache",
            "base-cache",
            "--name",
            "base",
            "--context-length",
            "128",
            "--d-model",
            "4",
            "--heads",
            "1",
            "--layers",
            "1",
            "--d-ff",
            "8",
            "--epochs",
            "2",
            "--max-updates",
            "1",
            "--quiet",
        ],
    ));
    let base = read_json(root.join("weights/base-1/manifest.json"));
    assert_eq!(base["metadata"]["dataset"]["format"], "jsonl");
    assert_eq!(
        base["metadata"]["dataset"]["selections"],
        json!([BASE_TRAIN])
    );
    assert!(base["chat"].is_object());
    let parent_hash = omega_training::checkpoint::sha256_bytes(
        &fs::read(root.join("weights/base-1/model.mpk")).unwrap(),
    );
    for selection in [
        "prepared/release-v1/base/validation",
        "prepared/release-v1/base/test",
    ] {
        assert!(
            succeeds(cli(
                root,
                &[
                    "evaluate",
                    "--datasets-root",
                    "data",
                    "--weights-root",
                    "weights",
                    "--checkpoint",
                    "base-1",
                    "--dataset",
                    selection
                ]
            ))
            .contains("mean cross entropy")
        );
    }
    succeeds(cli(
        root,
        &[
            "train-stage",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "base-1",
            "--dataset",
            CHAT_TRAIN,
            "--name",
            "assistant",
            "--epochs",
            "2",
            "--max-updates",
            "1",
            "--quiet",
        ],
    ));
    let stage = read_json(root.join("weights/assistant-1/resume.json"));
    assert_eq!(stage["schema_version"], 5);
    assert_eq!(stage["parent"]["model_sha256"], parent_hash);
    assert_eq!(stage["progress"]["completed_updates"], 1);
    let manifest = read_json(root.join("weights/assistant-1/manifest.json"));
    assert_eq!(manifest["metadata"]["dataset"]["format"], "chat");
    for selection in [
        "prepared/release-v1/chat/validation",
        "prepared/release-v1/chat/test",
    ] {
        assert!(
            succeeds(cli(
                root,
                &[
                    "evaluate",
                    "--datasets-root",
                    "data",
                    "--weights-root",
                    "weights",
                    "--checkpoint",
                    "assistant-1",
                    "--dataset",
                    selection
                ]
            ))
            .contains("mean cross entropy")
        );
    }
    succeeds(cli(
        root,
        &[
            "resume",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "assistant-1",
            "--name",
            "continued",
            "--epochs",
            "2",
            "--max-updates",
            "1",
            "--quiet",
        ],
    ));
    let resumed = read_json(root.join("weights/continued-1/resume.json"));
    assert_eq!(resumed["parent"], stage["parent"]);
    assert_eq!(resumed["progress"]["completed_updates"], 2);

    for selection in [
        "prepared/release-v1/base/validation",
        "prepared/release-v1/base/test",
    ] {
        fails(tokenizer_attempt(
            root,
            selection,
            "data/forbidden.json",
            &[],
        ));
        fails(cli(
            root,
            &[
                "train",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--dataset",
                selection,
                "--tokenizer",
                "tokenizer.json",
                "--name",
                "forbidden",
                "--max-updates",
                "1",
            ],
        ));
    }
    for selection in [
        "prepared/release-v1/chat/validation",
        "prepared/release-v1/chat/test",
        BASE_TRAIN,
    ] {
        fails(cli(
            root,
            &[
                "train-stage",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--checkpoint",
                "base-1",
                "--dataset",
                selection,
                "--name",
                "forbidden",
                "--max-updates",
                "1",
            ],
        ));
    }
    for selection in [
        "prepared",
        "prepared/release-v1",
        "prepared/release-v1/base",
        "prepared/release-v1/chat",
    ] {
        fails(tokenizer_attempt(
            root,
            selection,
            "data/forbidden.json",
            &[],
        ));
    }
    for format in ["text", "chat"] {
        fails(tokenizer_attempt(
            root,
            BASE_TRAIN,
            "data/forbidden.json",
            &["--dataset-format", format],
        ));
    }
    fails(cli(
        root,
        &[
            "evaluate",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "assistant-1",
            "--dataset",
            CHAT_TRAIN,
            "--dataset-format",
            "jsonl",
        ],
    ));
    fails(cli(
        root,
        &[
            "prepare-cache",
            "--datasets-root",
            "data",
            "--dataset",
            BASE_TRAIN,
            "--tokenizer",
            "tokenizer.json",
            "--context-length",
            "128",
            "--validation-count",
            "1",
            "--output",
            "forbidden-cache",
        ],
    ));
    fails(cli(
        root,
        &[
            "evaluate",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "base-1",
            "--dataset",
            BASE_TRAIN,
            "--validation-ratio",
            "0.5",
        ],
    ));
    fails(cli(
        root,
        &[
            "train",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--dataset",
            BASE_TRAIN,
            "--tokenizer",
            "tokenizer.json",
            "--name",
            "forbidden",
            "--max-updates",
            "1",
            "--validation-count",
            "1",
        ],
    ));
    // A resume recipe cannot opt a saved run into a published held-out partition.
    // Copy only this test's checkpoint files; the original base remains immutable.
    let heldout = root.join("weights/heldout-1");
    fs::create_dir(&heldout).unwrap();
    for entry in fs::read_dir(root.join("weights/base-1")).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), heldout.join(entry.file_name())).unwrap();
    }
    let mut changed = read_json(heldout.join("manifest.json"));
    changed["metadata"]["dataset"]["selections"] = json!(["prepared/release-v1/base/validation"]);
    fs::write(
        heldout.join("manifest.json"),
        serde_json::to_vec(&changed).unwrap(),
    )
    .unwrap();
    fails(cli(
        root,
        &[
            "resume",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "heldout-1",
            "--name",
            "forbidden",
            "--epochs",
            "2",
            "--max-updates",
            "1",
        ],
    ));
    assert!(!root.join("data/forbidden.json").exists());
    assert!(!root.join("weights/forbidden-1").exists());
    assert!(!root.join("forbidden-cache").exists());
    for (path, expected) in report.output_hashes {
        assert_eq!(
            omega_datasets::hash_file(&release.join(path)).unwrap(),
            expected
        );
    }
    assert_eq!(
        omega_datasets::hash_file(&release.join("membership.jsonl")).unwrap(),
        report.membership_sha256
    );
}

#[test]
fn incomplete_malformed_changed_and_empty_producer_releases_are_rejected() {
    for corruption in ["incomplete", "manifest", "payload", "empty"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let (release, _) = publish(root, corruption != "empty");
        match corruption {
            "incomplete" => fs::remove_file(release.join("COMPLETE")).unwrap(),
            "manifest" => fs::write(release.join("manifest.json"), b"{broken").unwrap(),
            "payload" => fs::write(
                release.join("base/train/data.jsonl"),
                b"{\"text\":\"changed\"}\n",
            )
            .unwrap(),
            "empty" => assert_eq!(
                read_json(release.join("manifest.json"))["counts"]["base/train"],
                0
            ),
            _ => unreachable!(),
        }
        fails(tokenizer_attempt(
            root,
            BASE_TRAIN,
            "data/should-not-exist.json",
            &[],
        ));
        assert!(!root.join("data/should-not-exist.json").exists());
    }
}
