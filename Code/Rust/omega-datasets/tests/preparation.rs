use std::{collections::BTreeMap, fs, io::Write, process::Command, sync::Arc};

use anyhow::{Result, ensure};
use omega_datasets::{
    build,
    config::{Config, Format, Mapping, Partition, Source},
    hub::{Hub, RemoteFile, ResolvedSource},
};
use serde_json::{Value, json};

const SHA: &str = "0123456789012345678901234567890123456789";

struct MemoryHub(BTreeMap<String, Vec<u8>>);
impl Hub for MemoryHub {
    fn resolve(&self, source: &Source) -> Result<ResolvedSource> {
        Ok(ResolvedSource {
            source_id: source.id.clone(),
            repo: source.repo.clone(),
            revision: SHA.into(),
            files: source
                .files
                .iter()
                .map(|name| RemoteFile {
                    path: name.clone(),
                    size: Some(self.0[name].len() as u64),
                })
                .collect(),
        })
    }
    fn download(
        &self,
        _: &ResolvedSource,
        file: &RemoteFile,
        output: &mut dyn Write,
        limit: u64,
    ) -> Result<u64> {
        let bytes = &self.0[&file.path];
        ensure!(bytes.len() as u64 <= limit, "limit exceeded");
        output.write_all(bytes)?;
        Ok(bytes.len() as u64)
    }
}

fn config() -> Config {
    let mut config: Config = toml::from_str(omega_datasets::EXAMPLE_CONFIG).unwrap();
    config.sources[0].repo = "test/corpus".into();
    config.sources[0].format = Format::Jsonl;
    config.sources[0].files = vec!["rows.jsonl".into()];
    config.split.train = 1.0;
    config.split.validation = 0.0;
    config.split.test = 0.0;
    config
}

fn hub(rows: &[Value]) -> MemoryHub {
    MemoryHub(BTreeMap::from([("rows.jsonl".into(), jsonl(rows))]))
}

fn jsonl(rows: &[Value]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, row).unwrap();
        bytes.push(b'\n');
    }
    bytes
}

fn read_lines(path: impl AsRef<std::path::Path>) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn preserves_payload_deduplicates_unicode_and_writes_replay_manifest() {
    let root = tempfile::tempdir().unwrap();
    let config = config();
    let report = build(
        root.path(),
        &config,
        &hub(&[
            json!({"text":"Hello  world"}),
            json!({"text":"Ｈello world"}),
            json!({"text":" \n "}),
            json!({"text":"different text"}),
        ]),
    )
    .unwrap();
    assert_eq!(report.records_seen, 4);
    assert_eq!(report.blank_records, 1);
    assert_eq!(report.duplicates_removed, 1);
    assert_eq!(report.counts["base/train"], 2);
    let output = root.path().join(&config.output);
    assert!(output.join("COMPLETE").is_file());
    let records = read_lines(output.join("base/train/data.jsonl"));
    assert_eq!(records[0]["text"], "Hello  world");
    let members = read_lines(output.join("membership.jsonl"));
    assert_eq!(members[1]["duplicate_of"], members[0]["id"]);
    assert!(members[1]["output_row"].is_null());
    assert_eq!(
        report.membership_sha256,
        omega_datasets::hash_file(&output.join("membership.jsonl")).unwrap()
    );
    let locked = Config::load(&output.join("model.lock.toml")).unwrap();
    assert_eq!(locked.sources[0].revision, SHA);
    assert!(
        build(root.path(), &config, &hub(&[]))
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
}

#[test]
fn deterministic_group_splits_keep_related_and_duplicates_together() {
    let root_a = tempfile::tempdir().unwrap();
    let root_b = tempfile::tempdir().unwrap();
    let mut config = config();
    config.split.train = 0.70;
    config.split.validation = 0.15;
    config.split.test = 0.15;
    config.sources[0].group_column = Some("family".into());
    let rows: Vec<_> = (0..400)
        .map(|i| json!({"text":format!("document number {i}"), "family":i / 2}))
        .collect();
    let hub = hub(&rows);
    let a = build(root_a.path(), &config, &hub).unwrap();
    let b = build(root_b.path(), &config, &hub).unwrap();
    assert_eq!(a.counts, b.counts);
    assert_eq!(a.membership_sha256, b.membership_sha256);
    let members = read_lines(root_a.path().join(&config.output).join("membership.jsonl"));
    for pair in members.chunks(2) {
        assert_eq!(pair[0]["partition"], pair[1]["partition"]);
    }
    config.output = "other-seed".into();
    config.seed += 1;
    let changed = build(root_b.path(), &config, &hub).unwrap();
    assert_ne!(a.membership_sha256, changed.membership_sha256);
}

#[test]
fn message_and_instruction_mapping_match_training_schema_and_overlap_partition() {
    let root = tempfile::tempdir().unwrap();
    let mut config = config();
    let mut chat = config.sources[0].clone();
    chat.id = "chat".into();
    chat.files = vec!["chat.jsonl".into()];
    chat.partition = Some(Partition::Test);
    chat.mapping = Mapping::Instruction {
        prompt_column: "ask".into(),
        response_column: "answer".into(),
        input_column: Some("input".into()),
        system: Some("Be precise".into()),
    };
    config.sources.push(chat);
    let hub = MemoryHub(BTreeMap::from([
        (
            "rows.jsonl".into(),
            jsonl(&[
                json!({"text":"An answer with exact overlap"}),
                json!({"text":"independent base document"}),
            ]),
        ),
        (
            "chat.jsonl".into(),
            jsonl(&[
                json!({"ask":"Question?", "input":"Context", "answer":"An answer with exact overlap"}),
            ]),
        ),
    ]));
    // Fixed chat test membership would leave no chat training data, so add an
    // independent source explicitly pinned to training.
    let mut training = config.sources[1].clone();
    training.id = "chat-train".into();
    training.files = vec!["train.jsonl".into()];
    training.partition = Some(Partition::Train);
    config.sources.push(training);
    let mut hub = hub;
    hub.0.insert(
        "train.jsonl".into(),
        jsonl(&[json!({"ask":"Another question?", "input":"", "answer":"Another answer"})]),
    );
    let report = build(root.path(), &config, &hub).unwrap();
    assert_eq!(report.counts["base/test"], 1);
    assert_eq!(report.counts["chat/test"], 1);
    let chat = read_lines(
        root.path()
            .join(&config.output)
            .join("chat/test/data.jsonl"),
    );
    assert_eq!(
        chat[0],
        json!({"schema_version":1,"messages":[{"role":"system","content":"Be precise"},{"role":"user","content":"Question?\n\nContext"},{"role":"assistant","content":"An answer with exact overlap"}]})
    );
}

#[test]
fn fixed_partition_conflict_from_transitive_overlap_fails_without_complete() {
    let root = tempfile::tempdir().unwrap();
    let mut config = config();
    config.sources[0].partition = Some(Partition::Train);
    config.sources[0].group_column = Some("family".into());
    let mut other = config.sources[0].clone();
    other.id = "heldout".into();
    other.files = vec!["test.jsonl".into()];
    other.partition = Some(Partition::Test);
    config.sources.push(other);
    let data = MemoryHub(BTreeMap::from([
        (
            "rows.jsonl".into(),
            jsonl(&[
                json!({"text":"first", "family":"a"}),
                json!({"text":"bridge", "family":"a"}),
            ]),
        ),
        (
            "test.jsonl".into(),
            jsonl(&[json!({"text":"bridge", "family":"b"})]),
        ),
    ]));
    let error = format!("{:#}", build(root.path(), &config, &data).unwrap_err());
    assert!(error.contains("fixed partition conflict"), "{error}");
    assert!(!root.path().join(&config.output).join("COMPLETE").exists());
}

#[test]
fn maps_sharegpt_roles_and_rejects_invalid_conversations() {
    for (messages, valid) in [
        (
            json!([{"from":"human", "value":"Hi"},{"from":"gpt", "value":"Hello"}]),
            true,
        ),
        (json!([{"from":"gpt", "value":"Hello"}]), false),
        (json!([{"from":"human", "value":"Hi"}]), false),
        (
            json!([{"from":"human", "value":"Hi"},{"from":"gpt", "value":" "}]),
            false,
        ),
        (json!([{"from":"tool", "value":"x"}]), false),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut config = config();
        config.sources[0].mapping = Mapping::Messages {
            column: "conversations".into(),
            role_field: "from".into(),
            content_field: "value".into(),
            user_role: "human".into(),
            assistant_role: "gpt".into(),
            system_role: "system".into(),
        };
        let result = build(
            root.path(),
            &config,
            &hub(&[json!({"conversations":messages})]),
        );
        assert_eq!(result.is_ok(), valid, "{result:?}");
    }
}

#[test]
fn json_arrays_gzip_and_parquet_are_read() {
    for format in [Format::Json, Format::Jsonl, Format::Parquet] {
        let root = tempfile::tempdir().unwrap();
        let mut config = config();
        config.sources[0].format = format;
        let (name, bytes) = match format {
            Format::Json => (
                "rows.json",
                serde_json::to_vec(&vec![json!({"text":"array text"})]).unwrap(),
            ),
            Format::Jsonl => {
                let mut gzip =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                gzip.write_all(&jsonl(&[json!({"text":"gzip text"})]))
                    .unwrap();
                ("rows.jsonl.gz", gzip.finish().unwrap())
            }
            Format::Parquet => {
                use parquet::{
                    data_type::{ByteArray, ByteArrayType},
                    file::writer::SerializedFileWriter,
                    schema::parser::parse_message_type,
                };
                let mut bytes = Vec::new();
                let schema = Arc::new(
                    parse_message_type("message schema { REQUIRED BYTE_ARRAY text (UTF8); }")
                        .unwrap(),
                );
                let mut writer =
                    SerializedFileWriter::new(&mut bytes, schema, Default::default()).unwrap();
                let mut row_group = writer.next_row_group().unwrap();
                let mut column = row_group.next_column().unwrap().unwrap();
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&[ByteArray::from("parquet text")], None, None)
                    .unwrap();
                column.close().unwrap();
                row_group.close().unwrap();
                writer.close().unwrap();
                ("rows.parquet", bytes)
            }
        };
        config.sources[0].files = vec![name.into()];
        let report = build(
            root.path(),
            &config,
            &MemoryHub(BTreeMap::from([(name.into(), bytes)])),
        )
        .unwrap();
        assert_eq!(report.counts["base/train"], 1);
    }
}

#[test]
fn rejects_bad_config_unknown_fields_and_unsafe_paths() {
    let config = config();
    config.validate().unwrap();
    let text = toml::to_string(&config).unwrap();
    assert!(toml::from_str::<Config>(&format!("typo = 4\n{text}")).is_err());
    for output in [
        "../escape",
        "CON",
        "bad.",
        "a/b",
        "x\\y",
        "C:drive",
        "NUL.txt",
    ] {
        let mut bad = config.clone();
        bad.output = output.into();
        assert!(bad.validate().is_err(), "{output}");
    }
    for ratios in [[0.8, 0.1, 0.2], [1.1, -0.1, 0.0], [f64::NAN, 0.0, 0.0]] {
        let mut bad = config.clone();
        [bad.split.train, bad.split.validation, bad.split.test] = ratios;
        assert!(bad.validate().is_err());
    }
    let root = tempfile::tempdir().unwrap();
    assert!(omega_datasets::dataset_folder(root.path(), "../elsewhere").is_err());
    let path = omega_datasets::init(root.path(), "nested/corpus").unwrap();
    Config::load(&path).unwrap();
    assert!(omega_datasets::init(root.path(), "nested/corpus").is_err());
}

#[test]
fn limits_and_empty_partitions_fail_without_silent_sampling() {
    for case in 0..5 {
        let root = tempfile::tempdir().unwrap();
        let mut config = config();
        match case {
            0 => config.limits.max_records = 1,
            1 => config.limits.max_record_bytes = 5,
            2 => config.limits.max_normalized_bytes = 5,
            3 => config.limits.max_file_bytes = 5,
            _ => {
                config.split.train = 0.9;
                config.split.test = 0.1;
                config.sources[0].group = Some("one-family".into());
            }
        }
        assert!(
            build(
                root.path(),
                &config,
                &hub(&[json!({"text":"hello"}), json!({"text":"world"})])
            )
            .is_err()
        );
        assert!(!root.path().join(&config.output).join("COMPLETE").exists());
    }
}

#[test]
fn malformed_json_missing_fields_and_missing_groups_are_actionable() {
    for bytes in [
        b"{broken\n".to_vec(),
        jsonl(&[json!({"text":3})]),
        jsonl(&[json!({"no_text":"bad"})]),
    ] {
        let root = tempfile::tempdir().unwrap();
        let error = format!(
            "{:#}",
            build(
                root.path(),
                &config(),
                &MemoryHub(BTreeMap::from([("rows.jsonl".into(), bytes)]))
            )
            .unwrap_err()
        );
        assert!(error.contains("rows.jsonl"), "{error}");
    }
    let root = tempfile::tempdir().unwrap();
    let mut config = config();
    config.sources[0].group_column = Some("family".into());
    let error = format!(
        "{:#}",
        build(root.path(), &config, &hub(&[json!({"text":"hello"})])).unwrap_err()
    );
    assert!(error.contains("group_column"));
}

#[test]
fn cli_help_check_init_and_error_exit_are_offline() {
    let root = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_omega-datasets");
    for subcommand in ["init", "check", "plan", "build"] {
        assert!(
            Command::new(binary)
                .args([subcommand, "--help"])
                .output()
                .unwrap()
                .status
                .success()
        );
    }
    let run = |args: &[&str]| {
        Command::new(binary)
            .arg("--datasets-root")
            .arg(root.path())
            .args(args)
            .output()
            .unwrap()
    };
    assert!(run(&["init", "demo"]).status.success());
    assert!(run(&["check", "demo"]).status.success());
    assert!(!run(&["init", "demo"]).status.success());
    assert!(!run(&["check", "missing"]).status.success());
    assert!(!run(&["check", "../unsafe"]).status.success());
    assert!(!run(&["build"]).status.success());
    assert!(!root.path().join("demo/release-v1").exists());
}

#[cfg(unix)]
#[test]
fn rejects_symlink_roots() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), root.path().join("linked")).unwrap();
    assert!(omega_datasets::dataset_folder(root.path(), "linked/new").is_err());
}

#[cfg(windows)]
#[test]
fn rejects_windows_junction_roots() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let link = root.path().join("linked");
    // Junction creation needs no symlink privilege. Command args are separate;
    // no deletion/move is performed by cmd.exe.
    let result = Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(&link)
        .arg(outside.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(omega_datasets::dataset_folder(root.path(), "linked/new").is_err());
    fs::remove_dir(&link).unwrap();
}
