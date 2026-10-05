//! Bounded, read-only checkpoint discovery. Headers are inspected without loading
//! tokenizers or tensors; payload integrity and resume compatibility are NOT
//! certified. Load the selected path normally, propagating failure without retrying
//! an older checkpoint. The trusted root may contain cooperating concurrent saves,
//! but discovery is not an atomic snapshot or a defense against hostile path swaps.

use std::{
    collections::BTreeSet,
    fs::{self, File, Metadata},
    io::Read,
    path::{Path, PathBuf},
    time::SystemTime,
};

use serde::Serialize;

use crate::{
    checkpoint::{
        CheckpointManifest, canonical_run_name, checkpoint_number, parse_checkpoint_config,
        parse_checkpoint_manifest, sha256_bytes,
    },
    resume::{AdamSettings, GPU_EXECUTION_MODE, ResumeManifest},
    sampling::{SAMPLING_ALGORITHM, SamplingPolicy},
    trainer::SessionOptions,
};

const INSPECTION: &str = "metadata_only_payloads_unverified";
const COMPLETE: &[u8] = b"omega-checkpoint-v1\n";

#[derive(Clone, Debug)]
pub struct CatalogLimits {
    /// All visited root entries count, including unrelated names.
    pub max_entries: usize,
    /// Aggregate header bytes read per candidate, including both marker reads.
    /// JSON allocation overhead is additional; tokenizer and tensor files are not read.
    pub max_metadata_bytes: usize,
}

impl Default for CatalogLimits {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            max_metadata_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogMode {
    Inference,
    Resume,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogStatus {
    CompleteInference,
    Resumable,
    LegacyUnverified,
    Incomplete,
    Invalid,
}

#[derive(Clone, Debug, Serialize)]
pub struct CatalogEntry {
    pub name: String,
    pub path: PathBuf,
    pub number: u64,
    pub status: CatalogStatus,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CheckpointCatalog {
    pub root: PathBuf,
    pub run_name: String,
    /// Ascending numeric suffix, then filename. Equal-number aliases remain visible.
    pub entries: Vec<CatalogEntry>,
    pub inspection: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Selection {
    pub path: PathBuf,
    pub number: u64,
    pub status: CatalogStatus,
    /// Higher-numbered reservations excluded for this operation, in descending order.
    pub skipped: Vec<CatalogEntry>,
    pub inspection: String,
}

impl CheckpointCatalog {
    /// Choose once. Invalid newer entries are errors, never silent fallbacks.
    /// This does not re-open files; the caller must use the normal checked loader.
    pub fn latest(&self, mode: CatalogMode) -> Result<Selection, String> {
        // Do not rely on public entries remaining sorted if a caller constructs a catalog.
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by_key(|entry| entry.number);
        for pair in entries.windows(2) {
            if pair[0].number == pair[1].number {
                return Err(format!(
                    "Ambiguous checkpoint number {}: {} and {}",
                    pair[0].number, pair[0].name, pair[1].name
                ));
            }
        }
        let mut skipped = Vec::new();
        for entry in entries.into_iter().rev() {
            match entry.status {
                CatalogStatus::Invalid => {
                    return Err(format!(
                        "Invalid checkpoint {}: {}; refusing fallback to an older save",
                        entry.path.display(),
                        entry.reason
                    ));
                }
                CatalogStatus::Incomplete => skipped.push(entry.clone()),
                CatalogStatus::CompleteInference | CatalogStatus::LegacyUnverified
                    if mode == CatalogMode::Resume =>
                {
                    let mut diagnostic = entry.clone();
                    diagnostic.reason = format!("Not resumable: {}", diagnostic.reason);
                    skipped.push(diagnostic);
                }
                _ => {
                    return Ok(Selection {
                        path: entry.path.clone(),
                        number: entry.number,
                        status: entry.status,
                        skipped,
                        inspection: INSPECTION.into(),
                    });
                }
            }
        }
        Err(format!(
            "No eligible {mode:?} checkpoint for run {} ({} entries skipped)",
            self.run_name,
            skipped.len()
        ))
    }
}

/// Inspect an existing root without creating it. Symlink/reparse roots and
/// candidates are rejected; occupied regular files remain incomplete reservations.
pub fn discover_checkpoints(
    root: &Path,
    run_name: &str,
    limits: &CatalogLimits,
) -> Result<CheckpointCatalog, String> {
    let run_name = canonical_run_name(run_name)?;
    if limits.max_entries == 0 || limits.max_metadata_bytes == 0 {
        return Err("Catalog limits must be positive".into());
    }
    let metadata = fs::symlink_metadata(root)
        .map_err(|e| format!("Cannot inspect weights root {}: {e}", root.display()))?;
    if is_link(&metadata) || !metadata.is_dir() {
        return Err(
            "Weights root must be an existing directory, not a symlink/reparse point".into(),
        );
    }
    let root = root
        .canonicalize()
        .map_err(|e| format!("Cannot resolve weights root: {e}"))?;
    let mut entries = Vec::new();
    for (index, entry) in fs::read_dir(&root)
        .map_err(|e| format!("Cannot scan weights root: {e}"))?
        .enumerate()
    {
        if index >= limits.max_entries {
            return Err(format!(
                "Catalog exceeds max_entries {}",
                limits.max_entries
            ));
        }
        let entry = entry.map_err(|e| format!("Cannot inspect weights root entry: {e}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(number) = checkpoint_number(name, &run_name)? else {
            continue;
        };
        let path = entry.path();
        let (status, reason) = inspect_entry(&root, &path, limits, || {});
        entries
            .try_reserve(1)
            .map_err(|e| format!("Cannot allocate checkpoint catalog: {e}"))?;
        entries.push(CatalogEntry {
            name: name.into(),
            path,
            number,
            status,
            reason,
        });
    }
    entries.sort_by(|a, b| a.number.cmp(&b.number).then(a.name.cmp(&b.name)));
    Ok(CheckpointCatalog {
        root,
        run_name,
        entries,
        inspection: INSPECTION.into(),
    })
}

fn is_link(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0 // FILE_ATTRIBUTE_REPARSE_POINT
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

fn regular_file(path: &Path) -> Result<Option<Metadata>, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if is_link(&metadata) || !metadata.is_file() => Err(format!(
            "Artifact {} must be a regular file, not a symlink/reparse point",
            path.display()
        )),
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("Cannot inspect {}: {error}", path.display())),
    }
}

struct HeaderReader {
    remaining: usize,
}

impl HeaderReader {
    fn read(&mut self, path: &Path) -> Result<Vec<u8>, String> {
        let metadata = regular_file(path)?
            .ok_or_else(|| format!("Missing required artifact {}", path.display()))?;
        let limit = u64::try_from(self.remaining).map_err(|_| "Metadata limit exceeds u64")?;
        if metadata.len() > limit {
            return Err(format!(
                "Metadata byte limit exceeded at {}",
                path.display()
            ));
        }
        let file = File::open(path)
            .map_err(|e| format!("Cannot open metadata {}: {e}", path.display()))?;
        let mut bytes = Vec::new();
        // The read is bounded even if a cooperating writer grows the file after stat.
        file.take(limit.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|e| format!("Cannot read metadata {}: {e}", path.display()))?;
        if bytes.len() > self.remaining {
            return Err(format!(
                "Metadata byte limit exceeded at {}",
                path.display()
            ));
        }
        regular_file(path)?
            .ok_or_else(|| format!("Metadata disappeared during inspection: {}", path.display()))?;
        self.remaining -= bytes.len();
        Ok(bytes)
    }
}

#[derive(PartialEq, Eq)]
struct Marker {
    bytes: Vec<u8>,
    modified: Option<SystemTime>,
}

fn marker(path: &Path, reader: &mut HeaderReader) -> Result<Option<Marker>, String> {
    let Some(metadata) = regular_file(&path.join("COMPLETE"))? else {
        return Ok(None);
    };
    Ok(Some(Marker {
        bytes: reader.read(&path.join("COMPLETE"))?,
        modified: metadata.modified().ok(),
    }))
}

// The hook gives tests a deterministic seam for marker publication/removal.
fn inspect_entry(
    root: &Path,
    path: &Path,
    limits: &CatalogLimits,
    after_headers: impl FnOnce(),
) -> (CatalogStatus, String) {
    let inspect = || -> Result<_, String> {
        let metadata =
            fs::symlink_metadata(path).map_err(|e| format!("Cannot inspect reservation: {e}"))?;
        if is_link(&metadata) {
            return Err("Checkpoint candidate is a symlink/reparse point".into());
        }
        if !metadata.is_dir() {
            return Ok((
                CatalogStatus::Incomplete,
                "Occupied non-directory reservation".into(),
            ));
        }
        if path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root) {
            return Err("Checkpoint candidate escapes its weights root".into());
        }
        let mut reader = HeaderReader {
            remaining: limits.max_metadata_bytes,
        };
        let before = marker(path, &mut reader)?;
        let result = match &before {
            None => Ok((CatalogStatus::Incomplete, "Missing COMPLETE marker".into())),
            Some(marker) => inspect_headers(path, &marker.bytes, &mut reader),
        };
        after_headers();
        let after = marker(path, &mut reader)?;
        if before != after {
            return Ok((
                CatalogStatus::Incomplete,
                "COMPLETE changed during discovery; rediscover before selection".into(),
            ));
        }
        let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if is_link(&metadata)
            || path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root)
        {
            return Err("Checkpoint path changed or escaped during inspection".into());
        }
        result
    };
    inspect().unwrap_or_else(|reason| (CatalogStatus::Invalid, reason))
}

fn inspect_headers(
    path: &Path,
    marker: &[u8],
    reader: &mut HeaderReader,
) -> Result<(CatalogStatus, String), String> {
    let has_manifest = regular_file(&path.join("manifest.json"))?.is_some();
    let manifest = if marker.is_empty() && !has_manifest {
        None
    } else if marker == COMPLETE {
        Some(parse_checkpoint_manifest(
            &reader.read(&path.join("manifest.json"))?,
        )?)
    } else {
        return Err("Unsupported or invalid checkpoint completion marker".into());
    };
    parse_checkpoint_config(&reader.read(&path.join("config.json"))?, manifest.as_ref())?;
    for name in ["tokenizer.json", "model.mpk"] {
        regular_file(&path.join(name))?
            .ok_or_else(|| format!("Missing required artifact {name}"))?;
    }
    let mut resume_parts = 0;
    for name in ["resume.json", "resume.sha256", "optimizer.mpk"] {
        resume_parts += usize::from(regular_file(&path.join(name))?.is_some());
    }
    if resume_parts != 0 && resume_parts != 3 {
        return Err("Incomplete resume artifact set in completed checkpoint".into());
    }
    let Some(manifest) = manifest else {
        if resume_parts != 0 {
            return Err("Legacy inference checkpoint cannot contain resumable state".into());
        }
        return Ok((
            CatalogStatus::LegacyUnverified,
            "Legacy header; tokenizer identity and payload integrity unverified".into(),
        ));
    };
    if resume_parts == 3 {
        let bytes = reader.read(&path.join("resume.json"))?;
        let digest = reader.read(&path.join("resume.sha256"))?;
        if digest != sha256_bytes(&bytes).as_bytes() {
            return Err("Resume manifest SHA256 mismatch".into());
        }
        inspect_resume_header(&bytes, &manifest)?;
        Ok((CatalogStatus::Resumable, "Resume metadata present; payload integrity, source-dependent counters/order and runtime compatibility unverified".into()))
    } else {
        Ok((
            CatalogStatus::CompleteInference,
            "Inference header; tokenizer and model payloads unverified".into(),
        ))
    }
}

fn inspect_resume_header(bytes: &[u8], checkpoint: &CheckpointManifest) -> Result<(), String> {
    let mut value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("Invalid resume JSON: {e}"))?;
    let version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if !matches!(version, Some(7 | 8)) && value.get("cuda_execution").is_some() {
        return Err("Legacy resume cannot contain CUDA execution settings".into());
    }
    if matches!(version, Some(1..=3 | 5 | 7 | 8)) && value.get("gpu_execution").is_some() {
        return Err("CPU resume schemas cannot contain GPU execution settings".into());
    }
    if matches!(version, Some(1 | 2)) && value.get("cpu_execution").is_some() {
        return Err("Legacy resume schemas cannot contain schema 3 CPU execution settings".into());
    }
    match version {
        Some(1) => {
            if value.get("options").is_some() || value.get("sampling_algorithm").is_some() {
                return Err("Schema 1 cannot contain schema 2 session options".into());
            }
            if value["execution_mode"] != "fixed-order-cpu-ndarray-f32-v1" {
                return Err("Unsupported legacy resume execution mode".into());
            }
            // Only normalize the field shape for inspection. Do not migrate runtime
            // identity or claim that this predecessor can be restored on this build.
            value["options"] =
                serde_json::to_value(SessionOptions::default()).map_err(|e| e.to_string())?;
            value["sampling_algorithm"] = SAMPLING_ALGORITHM.into();
        }
        Some(2 | 3) if value["execution_mode"] == "epoch-order-cpu-ndarray-f32-v2" => {}
        Some(4) if value["execution_mode"] == GPU_EXECUTION_MODE => {}
        Some(5) if value["execution_mode"] == "epoch-order-cpu-ndarray-f32-v2" => {}
        Some(6) if value["execution_mode"] == GPU_EXECUTION_MODE => {}
        Some(7 | 8) if value["execution_mode"] == crate::resume::CUDA_EXECUTION_MODE => {}
        _ => return Err("Unsupported resume schema or execution mode".into()),
    }
    if matches!(version, Some(1 | 2)) {
        // Shape normalization for metadata inspection only. Historical settings
        // are unknown; never read or initialize the current process's CPU pools.
        value["cpu_execution"] = serde_json::json!({
            "kernel": "ndarray-f32-checked-v1",
            "rayon_num_threads": null, "rayon_rs_num_cpus": null,
            "matmul_num_threads": null, "available_parallelism": null
        });
    }
    let header: ResumeManifest = serde_json::from_value(value)
        .map_err(|e| format!("Invalid resume metadata fields: {e}"))?;
    header.cpu_execution.validate()?;
    header.validate_objective()?;
    if matches!(version, Some(4 | 6)) {
        header
            .gpu_execution
            .as_ref()
            .ok_or("GPU resume requires execution profile")?
            .validate()?;
    }
    if matches!(version, Some(7 | 8)) {
        header
            .cuda_execution
            .as_ref()
            .ok_or("CUDA resume requires execution profile")?
            .validate()?;
    }
    if header.model != checkpoint.model || header.tokenizer != checkpoint.tokenizer {
        return Err("Resume configuration/tokenizer metadata disagrees with checkpoint".into());
    }
    if let Some(training) = &checkpoint.metadata.training
        && (training.seed != header.initialization_seed
            || training.learning_rate.to_bits() != header.learning_rate_bits)
    {
        return Err("Resume training settings disagree with checkpoint provenance".into());
    }
    if header.rng_policy != "unused-after-initialization"
        || header.adam != AdamSettings::default()
        || header.sampling_algorithm != SAMPLING_ALGORITHM
        || header.example_count == 0
        || header.targets_per_epoch == 0
        || !header.learning_rate().is_finite()
        || header.learning_rate() <= 0.0
    {
        return Err("Invalid or unsupported resume header settings".into());
    }
    for digest in [
        &header.source_identity,
        &header.model_sha256,
        &header.optimizer_sha256,
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("Invalid resume SHA256 field".into());
        }
    }
    header.options.batching.validate()?;
    header
        .options
        .optimization
        .learning_rate(header.learning_rate(), header.progress.completed_updates)?;
    let draws = match &header.options.sampling {
        SamplingPolicy::Fixed | SamplingPolicy::Shuffle => header.example_count,
        SamplingPolicy::Weighted {
            groups,
            samples_per_epoch,
        } => {
            if groups.is_empty() || *samples_per_epoch == 0 {
                return Err(
                    "Resume weighted sampling needs groups and positive epoch draws".into(),
                );
            }
            // Bound work/storage by indices actually present in bounded metadata,
            // never by the untrusted declared source size or epoch draw count.
            let mut names = BTreeSet::new();
            let mut indices = BTreeSet::new();
            let mut weight_sum = 0_u64;
            for group in groups {
                if group.name.trim().is_empty()
                    || !names.insert(&group.name)
                    || group.weight == 0
                    || group.indices.is_empty()
                {
                    return Err("Invalid resume sampling group name, weight or membership".into());
                }
                weight_sum = weight_sum
                    .checked_add(group.weight)
                    .ok_or("Resume sampling weights overflow u64")?;
                for index in &group.indices {
                    if *index >= header.example_count || !indices.insert(index) {
                        return Err(
                            "Resume sampling groups overlap or contain out-of-range indices".into(),
                        );
                    }
                }
            }
            if indices.len() != header.example_count {
                return Err("Resume sampling groups do not cover the declared source".into());
            }
            *samples_per_epoch
        }
    };
    let progress = header.progress;
    let batch = header.options.batching.batch_size;
    let loss = f64::from_bits(header.epoch_weighted_loss_bits);
    let updates = progress
        .completed_epochs
        .checked_mul(draws.div_ceil(batch))
        .and_then(|n| n.checked_add(progress.next_example_index / batch))
        .ok_or("Resume update count overflows usize")?;
    if progress.next_example_index >= draws
        || !progress.next_example_index.is_multiple_of(batch)
        || updates != progress.completed_updates
        || header.targets_per_epoch < header.example_count
        || progress.targets_in_epoch > progress.completed_targets
        || progress.targets_in_epoch < progress.next_example_index
        || progress.completed_targets < progress.completed_updates
        || !loss.is_finite()
        || loss < 0.0
        || (progress.next_example_index == 0 && (loss != 0.0 || progress.targets_in_epoch != 0))
    {
        return Err("Invalid resume cursor, counters or partial-epoch loss".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_during_inspection_is_not_a_stable_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let path = root.join("run-1");
        fs::create_dir(&path).unwrap();
        let result = inspect_entry(&root, &path, &CatalogLimits::default(), || {
            fs::write(path.join("COMPLETE"), COMPLETE).unwrap();
        });
        assert_eq!(result.0, CatalogStatus::Incomplete);
        assert!(result.1.contains("changed"));
        // A partial marker changing while read must not be misclassified as legacy.
        fs::write(path.join("COMPLETE"), b"").unwrap();
        let result = inspect_entry(&root, &path, &CatalogLimits::default(), || {
            fs::write(path.join("COMPLETE"), COMPLETE).unwrap();
        });
        assert_eq!(result.0, CatalogStatus::Incomplete);
    }
}
