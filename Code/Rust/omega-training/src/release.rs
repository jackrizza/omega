//! Integrity checks for explicitly selected `omega-datasets` release partitions.
//!
//! Only the selected partition is streamed. Raw downloads and membership payloads
//! are not needed to consume a release. Hashes establish consistency, not source
//! permission, dataset quality, or authenticity. Keep paths stable while reading;
//! these checks are not a security boundary against concurrent path replacement.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, Metadata},
    io::Read,
    path::Path,
};

const MANIFEST_LIMIT: u64 = 16 * 1024 * 1024;
const MARKER_LIMIT: u64 = 1024;
const MAGIC: &str = "omega-datasets-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseStage {
    Base,
    Chat,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleasePartition {
    pub stage: ReleaseStage,
    pub partition: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    counts: BTreeMap<String, u64>,
    membership_sha256: String,
    output_hashes: BTreeMap<String, String>,
}

fn linked(metadata: &Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return true;
        }
    }
    false
}

fn metadata(path: &Path) -> Result<Option<Metadata>, String> {
    match fs::symlink_metadata(path) {
        Ok(value) => {
            if linked(&value) {
                return Err(format!(
                    "Release path must not be a symlink/reparse point: {}",
                    path.display()
                ));
            }
            Ok(Some(value))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "Cannot inspect release path {}: {error}",
            path.display()
        )),
    }
}

fn directory_chain(path: &Path) -> Result<(), String> {
    for ancestor in path.ancestors().filter(|part| !part.as_os_str().is_empty()) {
        if !metadata(ancestor)?.is_some_and(|entry| entry.is_dir()) {
            return Err(format!(
                "Expected release directory: {}",
                ancestor.display()
            ));
        }
    }
    Ok(())
}

fn read_metadata(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let entry =
        metadata(path)?.ok_or_else(|| format!("Missing release metadata: {}", path.display()))?;
    if !entry.is_file() {
        return Err(format!(
            "Expected regular release metadata file: {}",
            path.display()
        ));
    }
    if entry.len() > limit {
        return Err(format!(
            "Release metadata exceeds {limit} byte limit: {}",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(limit + 1).read_to_end(&mut bytes))
        .map_err(|error| format!("Cannot read release metadata {}: {error}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!(
            "Release metadata exceeds {limit} byte limit: {}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn is_release(directory: &Path) -> Result<bool, String> {
    let lock = directory.join("model.lock.toml");
    if metadata(&lock)?.is_some() {
        return Ok(true);
    }
    // The marker remains recognizable even if the lock was removed and the
    // manifest became malformed. Other applications' COMPLETE markers are ignored.
    let marker = directory.join("COMPLETE");
    if let Some(info) = metadata(&marker)?
        && info.is_file()
        && info.len() <= MARKER_LIMIT
    {
        let bytes = read_metadata(&marker, MARKER_LIMIT)?;
        if bytes.starts_with(b"omega-datasets-") {
            return Ok(true);
        }
    }
    let manifest = directory.join("manifest.json");
    if let Some(info) = metadata(&manifest)?
        && info.is_file()
        && info.len() <= MANIFEST_LIMIT
    {
        let bytes = read_metadata(&manifest, MANIFEST_LIMIT)?;
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            return Ok(
                value.get("output_hashes").is_some() && value.get("membership_sha256").is_some()
            );
        }
    }
    Ok(false)
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn partition_payload(selected: &Path) -> Result<(String, u64), String> {
    let mut found = false;
    for entry in fs::read_dir(selected).map_err(|error| {
        format!(
            "Cannot read release partition {}: {error}",
            selected.display()
        )
    })? {
        let path = entry
            .map_err(|error| format!("Cannot inspect release partition entry: {error}"))?
            .path();
        let info = metadata(&path)?
            .ok_or_else(|| format!("Release partition entry disappeared: {}", path.display()))?;
        if path.file_name().is_none_or(|name| name != "data.jsonl") || !info.is_file() {
            return Err(format!(
                "Release partition may contain only its hashed data.jsonl; unexpected entry: {}",
                path.display()
            ));
        }
        found = true;
    }
    if !found {
        return Err(format!(
            "Missing release partition data.jsonl: {}",
            selected.display()
        ));
    }
    let path = selected.join("data.jsonl");
    let mut file = File::open(&path)
        .map_err(|error| format!("Cannot read release payload {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut records = 0u64;
    let mut nonblank = false;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("Cannot read release payload {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
        for byte in &buffer[..read] {
            if *byte == b'\n' {
                if nonblank {
                    records = records
                        .checked_add(1)
                        .ok_or("Release record count overflow")?;
                }
                nonblank = false;
            } else if !byte.is_ascii_whitespace() {
                nonblank = true;
            }
        }
    }
    if nonblank {
        records = records
            .checked_add(1)
            .ok_or("Release record count overflow")?;
    }
    Ok((format!("{:x}", hash.finalize()), records))
}

/// Inspect a canonical existing selected directory after caller path validation.
/// Plain folders return `None`. Recognized releases require the exact published
/// `release/{base|chat}/{train|validation|test}` directory, a valid completion
/// marker/manifest, and the selected payload's hash and positive record count.
/// This does not decide whether training on a partition is permitted.
pub fn inspect_partition(selected: &Path) -> Result<Option<ReleasePartition>, String> {
    directory_chain(selected)?;
    for root in selected
        .ancestors()
        .filter(|part| !part.as_os_str().is_empty())
    {
        if !is_release(root)? {
            continue;
        }
        let relative = selected
            .strip_prefix(root)
            .map_err(|error| format!("Cannot resolve release selection: {error}"))?;
        let parts: Vec<_> = relative.iter().collect();
        if parts.len() != 2 {
            return Err(format!(
                "Select exactly a published release partition below {}: base/train, base/validation, base/test, chat/train, chat/validation or chat/test",
                root.display()
            ));
        }
        let stage = match parts[0].to_str() {
            Some("base") => ReleaseStage::Base,
            Some("chat") => ReleaseStage::Chat,
            _ => return Err("Release selection must use the base or chat stage".into()),
        };
        let partition = match parts[1].to_str() {
            Some(value @ ("train" | "validation" | "test")) => value,
            _ => return Err("Release partition must be train, validation or test".into()),
        };
        let raw = read_metadata(&root.join("manifest.json"), MANIFEST_LIMIT)?;
        let marker = read_metadata(&root.join("COMPLETE"), MARKER_LIMIT)?;
        let hash = format!("{:x}", Sha256::digest(&raw));
        if marker != format!("{MAGIC}\n{hash}\n").as_bytes() {
            return Err(format!(
                "Release COMPLETE must contain {MAGIC} and the SHA256 of its unchanged manifest: {}",
                root.display()
            ));
        }
        let manifest: Manifest = serde_json::from_slice(&raw)
            .map_err(|error| format!("Invalid release manifest {}: {error}", root.display()))?;
        if manifest.schema_version != 1 {
            return Err(format!(
                "Unsupported release manifest schema_version {}",
                manifest.schema_version
            ));
        }
        if !valid_hash(&manifest.membership_sha256) {
            return Err("Release manifest membership_sha256 is not a lowercase SHA256".into());
        }
        let stage_name = if stage == ReleaseStage::Base {
            "base"
        } else {
            "chat"
        };
        let key = format!("{stage_name}/{partition}");
        let expected_count = *manifest
            .counts
            .get(&key)
            .ok_or_else(|| format!("Release manifest is missing the {key} count"))?;
        if expected_count == 0 {
            return Err(format!(
                "Release partition {key} is empty; select a populated partition"
            ));
        }
        let output_key = format!("{key}/data.jsonl");
        let expected_hash = manifest
            .output_hashes
            .get(&output_key)
            .ok_or_else(|| format!("Release manifest is missing the {output_key} hash"))?;
        if !valid_hash(expected_hash) {
            return Err(format!(
                "Release payload hash for {output_key} is not a lowercase SHA256"
            ));
        }
        let (actual_hash, actual_count) = partition_payload(selected)?;
        if actual_hash != *expected_hash {
            return Err(format!("Release payload SHA256 mismatch: {output_key}"));
        }
        if actual_count != expected_count {
            return Err(format!(
                "Release partition {key} record count mismatch: expected {expected_count}, found {actual_count}"
            ));
        }
        return Ok(Some(ReleasePartition {
            stage,
            partition: partition.into(),
        }));
    }
    Ok(None)
}

/// Refuse recursively ingesting a recognized release root, including an
/// incomplete build with only `model.lock.toml`. Discovery calls this for every
/// visited directory; explicitly selected partition folders are not roots.
pub fn reject_release_parent(directory: &Path) -> Result<(), String> {
    if is_release(directory)? {
        return Err(format!(
            "Cannot recursively select omega-datasets release {}; choose one explicit base/chat train/validation/test partition to avoid held-out leakage",
            directory.display()
        ));
    }
    Ok(())
}
