use omega_training::release::{ReleaseStage, inspect_partition, reject_release_parent};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn publish(root: &Path, manifest: &Value) {
    let bytes = serde_json::to_vec_pretty(manifest).unwrap();
    fs::write(root.join("manifest.json"), &bytes).unwrap();
    fs::write(
        root.join("COMPLETE"),
        format!("omega-datasets-v1\n{}\n", hash(&bytes)),
    )
    .unwrap();
}

fn fixture(root: &Path) -> Value {
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("model.lock.toml"),
        "# immutable producer recipe\n",
    )
    .unwrap();
    let mut counts = serde_json::Map::new();
    let mut hashes = serde_json::Map::new();
    for stage in ["base", "chat"] {
        for partition in ["train", "validation", "test"] {
            let key = format!("{stage}/{partition}");
            let path = root.join(&key);
            fs::create_dir_all(&path).unwrap();
            let payload = if stage == "base" {
                b"{\"text\":\"hello world\"}\n".as_slice()
            } else {
                b"{\"schema_version\":1,\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"},{\"role\":\"assistant\",\"content\":\"Hi\"}]}\n".as_slice()
            };
            fs::write(path.join("data.jsonl"), payload).unwrap();
            counts.insert(key.clone(), json!(1));
            hashes.insert(format!("{key}/data.jsonl"), json!(hash(payload)));
        }
    }
    let manifest = json!({"schema_version": 1, "counts": counts,
        "membership_sha256": hash(b"membership not needed for consumption"),
        "output_hashes": hashes, "tool_version":"fixture", "config":{}});
    publish(root, &manifest);
    manifest
}

fn inspect(
    root: &Path,
    selection: &str,
) -> Result<Option<omega_training::release::ReleasePartition>, String> {
    inspect_partition(&fs::canonicalize(root.join(selection)).unwrap())
}

#[test]
fn all_exact_partitions_validate_without_raw_or_membership_files() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    for (stage, expected) in [("base", ReleaseStage::Base), ("chat", ReleaseStage::Chat)] {
        for partition in ["train", "validation", "test"] {
            let result = inspect(temporary.path(), &format!("{stage}/{partition}"))
                .unwrap()
                .unwrap();
            assert_eq!(result.stage, expected);
            assert_eq!(result.partition, partition);
        }
    }
    reject_release_parent(&temporary.path().join("base/train")).unwrap();
}

#[test]
fn plain_and_other_app_manifests_are_untouched() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    fs::create_dir(root.join("plain")).unwrap();
    fs::write(root.join("plain/data.txt"), "hello world").unwrap();
    for manifest in [
        json!({"schema_version":1,"counts":{"base":{"train":1}},"records":[],"audit":{}}),
        json!({"schema_version":1,"sources":[],"training_identity":"cache"}),
        json!({"output_hashes":{},"other_application":true}),
    ] {
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        fs::write(root.join("COMPLETE"), "another application's marker").unwrap();
        assert_eq!(inspect(root, "plain").unwrap(), None);
        reject_release_parent(root).unwrap();
    }
}

#[test]
fn recognizable_manifest_or_marker_detects_release_without_lock() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    fs::remove_file(temporary.path().join("model.lock.toml")).unwrap();
    assert!(inspect(temporary.path(), "base/train").unwrap().is_some());
    fs::remove_file(temporary.path().join("COMPLETE")).unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("COMPLETE")
    );
    fs::write(
        temporary.path().join("COMPLETE"),
        "omega-datasets-v1\nwrong\n",
    )
    .unwrap();
    fs::write(temporary.path().join("manifest.json"), "broken JSON").unwrap();
    assert!(inspect(temporary.path(), "base/train").is_err());
}

#[test]
fn root_stage_nested_and_wrong_stage_selection_are_rejected() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    fs::create_dir_all(temporary.path().join("base/train/nested")).unwrap();
    fs::create_dir_all(temporary.path().join("other/train")).unwrap();
    fs::create_dir_all(temporary.path().join("base/extra")).unwrap();
    for selection in [
        "",
        "base",
        "chat",
        "base/train/nested",
        "other/train",
        "base/extra",
    ] {
        assert!(inspect(temporary.path(), selection).is_err(), "{selection}");
    }
    assert!(
        reject_release_parent(temporary.path())
            .unwrap_err()
            .contains("held-out")
    );
}

#[test]
fn incomplete_lock_only_release_is_not_ordinary_text() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    fs::create_dir_all(root.join("base/train")).unwrap();
    fs::write(root.join("model.lock.toml"), "incomplete").unwrap();
    fs::write(root.join("base/train/data.jsonl"), "{\"text\":\"data\"}\n").unwrap();
    assert!(reject_release_parent(root).is_err());
    assert!(
        inspect(root, "base/train")
            .unwrap_err()
            .contains("manifest.json")
    );
}

#[test]
fn completion_marker_requires_exact_version_hash_and_shape() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    let raw = fs::read(temporary.path().join("manifest.json")).unwrap();
    for marker in [
        format!("omega-datasets-v2\n{}\n", hash(&raw)),
        format!("omega-datasets-v1\n{}\n", "0".repeat(64)),
        format!("omega-datasets-v1\n{}\nextra", hash(&raw)),
        String::new(),
    ] {
        fs::write(temporary.path().join("COMPLETE"), marker).unwrap();
        assert!(
            inspect(temporary.path(), "base/train")
                .unwrap_err()
                .contains("COMPLETE")
        );
    }
    fs::remove_file(temporary.path().join("COMPLETE")).unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("COMPLETE")
    );
}

#[test]
fn manifest_schema_required_fields_and_sha_forms_are_checked() {
    let temporary = tempfile::tempdir().unwrap();
    let original = fixture(temporary.path());
    let mut bad = original.clone();
    bad["schema_version"] = json!(2);
    publish(temporary.path(), &bad);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("schema_version")
    );
    for field in [
        "schema_version",
        "counts",
        "membership_sha256",
        "output_hashes",
    ] {
        let mut bad = original.clone();
        bad.as_object_mut().unwrap().remove(field);
        publish(temporary.path(), &bad);
        assert!(inspect(temporary.path(), "base/train").is_err(), "{field}");
    }
    let mut bad = original.clone();
    bad["membership_sha256"] = json!("unknown");
    publish(temporary.path(), &bad);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("membership_sha256")
    );
    let mut bad = original;
    bad["output_hashes"]["base/train/data.jsonl"] = json!("bad");
    publish(temporary.path(), &bad);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("SHA256")
    );
}

#[test]
fn selected_integrity_checks_ignore_unselected_payload_corruption() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    fs::write(
        temporary.path().join("chat/test/data.jsonl"),
        "not the selected corpus",
    )
    .unwrap();
    assert!(inspect(temporary.path(), "base/train").unwrap().is_some());
    assert!(
        inspect(temporary.path(), "chat/test")
            .unwrap_err()
            .contains("SHA256 mismatch")
    );
    fs::write(temporary.path().join("base/train/data.jsonl"), "changed").unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("SHA256 mismatch")
    );
}

#[test]
fn empty_missing_or_inconsistent_record_counts_fail() {
    let temporary = tempfile::tempdir().unwrap();
    let original = fixture(temporary.path());
    let mut bad = original.clone();
    bad["counts"]["base/train"] = json!(0);
    publish(temporary.path(), &bad);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("empty")
    );
    bad["counts"]["base/train"] = json!(2);
    publish(temporary.path(), &bad);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("record count mismatch")
    );
    bad["counts"].as_object_mut().unwrap().remove("base/train");
    publish(temporary.path(), &bad);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("missing")
    );
    let mut empty = original;
    empty["output_hashes"]["base/train/data.jsonl"] = json!(hash(b""));
    fs::write(temporary.path().join("base/train/data.jsonl"), "").unwrap();
    publish(temporary.path(), &empty);
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("record count mismatch")
    );
}

#[test]
fn extra_files_or_directories_and_missing_payload_are_rejected() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    let selected = temporary.path().join("base/train");
    fs::write(selected.join("unhashed.jsonl"), "{}").unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("unexpected entry")
    );
    fs::remove_file(selected.join("unhashed.jsonl")).unwrap();
    fs::create_dir(selected.join("nested")).unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("unexpected entry")
    );
    fs::remove_dir(selected.join("nested")).unwrap();
    fs::remove_file(selected.join("data.jsonl")).unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("Missing")
    );
}

#[test]
fn metadata_reads_are_bounded() {
    let temporary = tempfile::tempdir().unwrap();
    fixture(temporary.path());
    fs::write(temporary.path().join("COMPLETE"), vec![b'x'; 1025]).unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("byte limit")
    );
    fs::File::create(temporary.path().join("manifest.json"))
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    assert!(
        inspect(temporary.path(), "base/train")
            .unwrap_err()
            .contains("byte limit")
    );
}

#[test]
fn payload_line_count_and_hash_stream_across_buffers() {
    let temporary = tempfile::tempdir().unwrap();
    let mut manifest = fixture(temporary.path());
    let payload = format!(
        "{{\"text\":\"{}\"}}\n{{\"text\":\"final no newline\"}}",
        "x".repeat(128 * 1024)
    );
    fs::write(temporary.path().join("base/train/data.jsonl"), &payload).unwrap();
    manifest["counts"]["base/train"] = json!(2);
    manifest["output_hashes"]["base/train/data.jsonl"] = json!(hash(payload.as_bytes()));
    publish(temporary.path(), &manifest);
    assert!(inspect(temporary.path(), "base/train").unwrap().is_some());
}

#[cfg(unix)]
#[test]
fn rejects_metadata_payload_and_selected_directory_symlinks() {
    use std::os::unix::fs::symlink;
    let temporary = tempfile::tempdir().unwrap();
    let release = temporary.path().join("release");
    fixture(&release);
    for relative in [
        "model.lock.toml",
        "manifest.json",
        "COMPLETE",
        "base/train/data.jsonl",
    ] {
        let path = release.join(relative);
        let saved = temporary.path().join("saved");
        fs::rename(&path, &saved).unwrap();
        symlink(&saved, &path).unwrap();
        assert!(
            inspect(&release, "base/train")
                .unwrap_err()
                .contains("symlink/reparse")
        );
        fs::remove_file(&path).unwrap();
        fs::rename(&saved, &path).unwrap();
    }
    let selected_alias = temporary.path().join("alias");
    symlink(release.join("base/train"), &selected_alias).unwrap();
    assert!(
        inspect_partition(&selected_alias)
            .unwrap_err()
            .contains("symlink/reparse")
    );
}

#[cfg(windows)]
#[test]
fn rejects_windows_partition_junction() {
    use std::process::Command;
    let temporary = tempfile::tempdir().unwrap();
    let release = temporary.path().join("release");
    fixture(&release);
    let alias = temporary.path().join("alias");
    let output = Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(&alias)
        .arg(release.join("base").join("train"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = inspect_partition(&alias);
    fs::remove_dir(&alias).unwrap();
    assert!(result.unwrap_err().contains("symlink/reparse"));
}
