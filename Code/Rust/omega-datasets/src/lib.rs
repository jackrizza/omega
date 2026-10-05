//! Acquire and prepare bounded, immutable Omega corpus releases from Hugging Face.
//!
//! Library roots and transports are explicit. `build` never trains a tokenizer or
//! model, runs dataset scripts, overwrites a release, or silently truncates rows.
pub mod config;
pub mod hub;
mod records;

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use config::{Config, Partition, safe_relative};
use hub::{Hub, ResolvedSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const EXAMPLE_CONFIG: &str = include_str!("../examples/model.toml");

pub(crate) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn read_bounded(reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "input exceeds {limit} byte limit"
    );
    Ok(bytes)
}

/// Reject symlinks and Windows junctions, including existing path ancestors.
/// The caller must keep the filesystem stable during acquisition/publication.
pub fn reject_links(path: &Path) -> Result<()> {
    let absolute = std::path::absolute(path)?;
    for ancestor in absolute.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) => {
                let link = metadata.file_type().is_symlink();
                #[cfg(windows)]
                let link = {
                    use std::os::windows::fs::MetadataExt;
                    link || metadata.file_attributes() & 0x400 != 0
                };
                ensure!(
                    !link,
                    "symlink/reparse path is not allowed: {}",
                    ancestor.display()
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("Inspect {}", ancestor.display()));
            }
        }
    }
    Ok(())
}

/// Select a folder using '/'-separated names relative to an explicit dataset root.
pub fn dataset_folder(root: &Path, name: &str) -> Result<PathBuf> {
    safe_relative(name)?;
    let path = root.join(name);
    reject_links(&path)?;
    Ok(path)
}

/// Create a starter configuration without replacing existing model.toml files.
pub fn init(root: &Path, name: &str) -> Result<PathBuf> {
    let folder = dataset_folder(root, name)?;
    fs::create_dir_all(&folder)?;
    let path = folder.join("model.toml");
    write_new(&path, EXAMPLE_CONFIG.as_bytes())?;
    Ok(path)
}

fn create_new(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "Create new file {}; existing files are never replaced",
                path.display()
            )
        })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_new(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BuildReport {
    pub schema_version: u32,
    pub tool_version: String,
    pub config: Config,
    pub resolved: Vec<ResolvedSource>,
    pub files: Vec<FileReport>,
    pub records_seen: usize,
    pub blank_records: usize,
    pub duplicates_removed: usize,
    pub counts: BTreeMap<String, usize>,
    pub membership_sha256: String,
    pub output_hashes: BTreeMap<String, String>,
    pub split_algorithm: String,
    pub audit: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileReport {
    pub source_id: String,
    pub remote_path: String,
    pub local_path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Verify all published payloads, including intentionally empty partitions.
/// This does not make empty partitions valid training selections.
pub fn verify_release(output: &Path) -> Result<BuildReport> {
    use std::io::{BufRead, BufReader};
    reject_links(output)?;
    let manifest_path = output.join("manifest.json");
    reject_links(&manifest_path)?;
    let manifest = read_bounded(File::open(manifest_path)?, 8 * 1024 * 1024)?;
    let marker_path = output.join("COMPLETE");
    reject_links(&marker_path)?;
    let marker = read_bounded(File::open(marker_path)?, 256)?;
    ensure!(
        marker == format!("omega-datasets-v1\n{}\n", hash(&manifest)).as_bytes(),
        "Release completion marker/hash mismatch"
    );
    let report: BuildReport = serde_json::from_slice(&manifest)?;
    ensure!(
        report.schema_version == 1,
        "Unsupported dataset release schema"
    );
    for stage in ["base", "chat"] {
        for split in ["train", "validation", "test"] {
            let key = format!("{stage}/{split}");
            let payload = format!("{key}/data.jsonl");
            let path = output.join(&payload);
            reject_links(&path)?;
            ensure!(
                report.output_hashes.get(&payload) == Some(&hash_file(&path)?),
                "Release payload checksum mismatch: {payload}"
            );
            let mut reader = BufReader::new(File::open(&path)?);
            let mut count = 0usize;
            // Count delimiters without allocating an unbounded JSONL record.
            loop {
                let bytes = reader.fill_buf()?;
                if bytes.is_empty() {
                    break;
                }
                count += bytes.iter().filter(|b| **b == b'\n').count();
                let n = bytes.len();
                reader.consume(n);
            }
            ensure!(
                report.counts.get(&key) == Some(&count),
                "Release payload count mismatch: {payload}"
            );
        }
    }
    Ok(report)
}

struct Record {
    id: String,
    source_index: usize,
    stage: &'static str,
    value: Value,
    digest: String,
    overlap_hashes: Vec<String>,
    group: Option<String>,
}

struct Groups(Vec<usize>);
impl Groups {
    fn root(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }
    fn join(&mut self, a: usize, b: usize) {
        let a = self.root(a);
        let b = self.root(b);
        self.0[a.max(b)] = a.min(b);
    }
}

/// Build `folder/config.output`. A failed build retains its incomplete directory
/// for inspection; reruns must use a fresh output name. COMPLETE is written last.
pub fn build(folder: &Path, config: &Config, hub: &impl Hub) -> Result<BuildReport> {
    config.validate()?;
    reject_links(folder)?;
    ensure!(
        folder.is_dir(),
        "dataset folder must already exist: {}",
        folder.display()
    );
    let output = folder.join(&config.output);
    reject_links(&output)?;
    ensure!(
        !output.try_exists()?,
        "output already exists: {}; choose a new output name",
        output.display()
    );
    let resolved = hub::plan(config, hub)?;
    fs::create_dir(&output).with_context(|| format!("Reserve new release {}", output.display()))?;
    let result = build_reserved(&output, config, resolved, hub);
    result.with_context(|| {
        format!(
            "Build {} (failed outputs have no COMPLETE marker; choose a new output name to retry)",
            output.display()
        )
    })
}

fn build_reserved(
    output: &Path,
    config: &Config,
    resolved: Vec<ResolvedSource>,
    hub: &impl Hub,
) -> Result<BuildReport> {
    // Save a replayable config with immutable SHAs and explicit selected files.
    let mut locked = config.clone();
    for (source, remote) in locked.sources.iter_mut().zip(&resolved) {
        source.revision = remote.revision.clone();
        source.files = remote.files.iter().map(|f| f.path.clone()).collect();
    }
    write_new(
        &output.join("model.lock.toml"),
        toml::to_string_pretty(&locked)?.as_bytes(),
    )?;
    fs::create_dir(output.join("raw"))?;
    let mut report = BuildReport {
        schema_version: 1, tool_version: env!("CARGO_PKG_VERSION").into(), config: config.clone(),
        resolved, files: Vec::new(), records_seen: 0, blank_records: 0, duplicates_removed: 0,
        counts: BTreeMap::new(), membership_sha256: String::new(), output_hashes: BTreeMap::new(),
        split_algorithm: "sha256-group-v1: seed LE-u64 + minimum connected normalized record hash; first 53 hash bits; ratio thresholds".into(),
        audit: "Exact NFKC/whitespace-normalized records, explicit groups, and exact full user/assistant messages or joined dialogue are connected globally before splitting. System-only, substring, near-duplicate, semantic, license, privacy and human quality reviews are not certified. COMPLETE means files were written, not corpus acceptance.".into(),
    };
    let mut records = Vec::new();
    let mut downloaded = 0u64;
    let mut normalized_bytes = 0u64;
    for (source_index, remote) in report.resolved.iter().enumerate() {
        let source = &config.sources[source_index];
        fs::create_dir(output.join("raw").join(&source.id))?;
        for (file_index, file) in remote.files.iter().enumerate() {
            // Numeric local names avoid upstream path and case aliases. Originals
            // are byte-for-byte preserved and mapped to Hub names in the manifest.
            let local = format!("raw/{}/{file_index:06}.download", source.id);
            let path = output.join(&local);
            let mut raw = create_new(&path)?;
            let limit = config
                .limits
                .max_file_bytes
                .min(config.limits.max_download_bytes - downloaded);
            let bytes = hub
                .download(remote, file, &mut raw, limit)
                .with_context(|| format!("Download {} / {}", source.id, file.path))?;
            raw.flush()?;
            ensure!(
                bytes <= limit && raw.metadata()?.len() == bytes,
                "download transport violated byte limit/count"
            );
            downloaded += bytes;
            drop(raw);
            report.files.push(FileReport {
                source_id: source.id.clone(),
                remote_path: file.path.clone(),
                local_path: local,
                bytes,
                sha256: hash_file(&path)?,
            });
            records::visit_file(&path, &file.path, source.format, &config.limits, |row, value| {
                let id = format!("{}/{}#row-{row}", source.id, file.path);
                report.records_seen += 1;
                ensure!(report.records_seen <= config.limits.max_records, "{id}: max_records exceeded; narrow source files or raise the limit (rows are never truncated)");
                ensure!(serde_json::to_vec(&value)?.len() <= config.limits.max_record_bytes, "{id}: max_record_bytes exceeded");
                let Some(mapped) = records::map_record(&value, source).with_context(|| format!("Invalid record {id}"))? else {
                    report.blank_records += 1;
                    return Ok(());
                };
                let size = serde_json::to_vec(&mapped.value)?.len();
                ensure!(size <= config.limits.max_record_bytes, "{id}: normalized output exceeds max_record_bytes");
                normalized_bytes = normalized_bytes.checked_add(size as u64).context("normalized size overflow")?;
                ensure!(normalized_bytes <= config.limits.max_normalized_bytes, "max_normalized_bytes exceeded");
                records.push(Record { id, source_index, stage: mapped.stage, value: mapped.value, digest: mapped.digest, overlap_hashes: mapped.overlap_hashes, group: mapped.group });
                Ok(())
            }).with_context(|| format!("Read source {} / {}", source.id, file.path))?;
        }
    }
    ensure!(!records.is_empty(), "no nonblank records remain");
    let mut groups = Groups((0..records.len()).collect());
    let mut keys = BTreeMap::new();
    let mut first = BTreeMap::new();
    let mut duplicates = vec![None; records.len()];
    for (index, record) in records.iter().enumerate() {
        let mut record_keys = vec![format!("record:{}", record.digest)];
        if let Some(group) = &record.group {
            record_keys.push(format!("group:{group}"));
        }
        record_keys.extend(record.overlap_hashes.iter().map(|h| format!("text:{h}")));
        for key in record_keys {
            if let Some(previous) = keys.insert(key, index) {
                groups.join(previous, index);
            }
        }
        let duplicate_key = (record.stage, &record.digest);
        if let Some(&previous) = first.get(&duplicate_key) {
            duplicates[index] = Some(previous);
            report.duplicates_removed += 1;
        } else {
            first.insert(duplicate_key, index);
        }
    }
    let mut anchors: BTreeMap<usize, String> = BTreeMap::new();
    let mut fixed = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        let root = groups.root(index);
        anchors
            .entry(root)
            .and_modify(|s| {
                if record.digest < *s {
                    *s = record.digest.clone();
                }
            })
            .or_insert_with(|| record.digest.clone());
        if let Some(partition) = config.sources[record.source_index].partition
            && let Some(previous) = fixed.insert(root, partition)
        {
            ensure!(
                previous == partition,
                "fixed partition conflict at {}: a source group, duplicate or exact text overlap connects {} and {}; correct the source partition/group recipe",
                record.id,
                previous.name(),
                partition.name()
            );
        }
    }
    let partitions: BTreeMap<_, _> = anchors
        .iter()
        .map(|(&root, anchor)| {
            (
                root,
                fixed
                    .get(&root)
                    .copied()
                    .unwrap_or_else(|| split_group(anchor, config)),
            )
        })
        .collect();
    let mut writers = BTreeMap::new();
    for stage in ["base", "chat"] {
        for partition in ["train", "validation", "test"] {
            let key = format!("{stage}/{partition}");
            fs::create_dir_all(output.join(&key))?;
            writers.insert(
                key.clone(),
                std::io::BufWriter::new(create_new(&output.join(&key).join("data.jsonl"))?),
            );
            report.counts.insert(key, 0);
        }
    }
    let membership_path = output.join("membership.jsonl");
    let mut membership = std::io::BufWriter::new(create_new(&membership_path)?);
    for (index, record) in records.iter().enumerate() {
        let root = groups.root(index);
        let partition = partitions[&root];
        let key = format!("{}/{}", record.stage, partition.name());
        let duplicate_of = duplicates[index].map(|i| records[i].id.as_str());
        let row = if duplicate_of.is_none() {
            let writer = writers.get_mut(&key).unwrap();
            serde_json::to_writer(&mut *writer, &record.value)?;
            writer.write_all(b"\n")?;
            let count = report.counts.get_mut(&key).unwrap();
            *count += 1;
            Some(*count)
        } else {
            None
        };
        serde_json::to_writer(
            &mut membership,
            &serde_json::json!({
                "id": record.id, "stage": record.stage, "partition": partition,
                "source_group": record.group, "connected_group": anchors[&root],
                "normalized_sha256": record.digest, "payload_sha256": hash(&serde_json::to_vec(&record.value)?),
                "duplicate_of": duplicate_of, "output": format!("{key}/data.jsonl"), "output_row": row,
            }),
        )?;
        membership.write_all(b"\n")?;
    }
    membership.flush()?;
    drop(membership);
    report.membership_sha256 = hash_file(&membership_path)?;
    for (key, mut writer) in writers {
        writer.flush()?;
        drop(writer);
        let path = format!("{key}/data.jsonl");
        report
            .output_hashes
            .insert(path.clone(), hash_file(&output.join(path))?);
    }
    // A large connected family can defeat requested held-out splits. Never
    // break the family to manufacture the requested counts.
    for stage in ["base", "chat"] {
        if !records.iter().any(|r| r.stage == stage) {
            continue;
        }
        for (partition, ratio) in [
            ("train", config.split.train),
            ("validation", config.split.validation),
            ("test", config.split.test),
        ] {
            ensure!(
                ratio == 0.0 || report.counts[&format!("{stage}/{partition}")] > 0,
                "{stage}/{partition} is empty: add more independent groups, set explicit source partitions, or adjust split ratios/seed; no records were moved across group boundaries"
            );
        }
    }
    let mut manifest = serde_json::to_vec_pretty(&report)?;
    manifest.push(b'\n');
    write_new(&output.join("manifest.json"), &manifest)?;
    write_new(
        &output.join("COMPLETE"),
        format!("omega-datasets-v1\n{}\n", hash(&manifest)).as_bytes(),
    )?;
    Ok(report)
}

fn split_group(anchor: &str, config: &Config) -> Partition {
    let mut digest = Sha256::new();
    digest.update(b"omega-datasets-group-v1\0");
    digest.update(config.seed.to_le_bytes());
    digest.update(anchor.as_bytes());
    let hash = digest.finalize();
    let rank =
        (u64::from_be_bytes(hash[..8].try_into().unwrap()) >> 11) as f64 / ((1u64 << 53) as f64);
    if rank < config.split.train {
        Partition::Train
    } else if rank < config.split.train + config.split.validation {
        Partition::Validation
    } else {
        Partition::Test
    }
}

pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
