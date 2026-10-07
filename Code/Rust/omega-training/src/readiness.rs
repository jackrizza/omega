//! Read-only technical readiness for a new assistant training stage.
//!
//! No model is executed and no quality or permission approval is inferred.
//! Chat preparation uses the existing eager bounds. Additional overlap auditing
//! caps each partition at 64 MiB raw input, 100,000 records and 200,000 units.
//! Exact auditing collapses Unicode whitespace but does not normalize Unicode
//! composition, inspect substrings, or establish semantic/source-family isolation.

use crate::{
    ValidationSplit,
    assistant::{ChatArtifact, parse_messages, prepare_conversations},
    checkpoint::{ModelConfig, TokenizerIdentity, load_checkpoint_header, sha256_bytes},
    dataset::{
        DatasetFormat, ExampleSource, discover_dataset_files, parse_text_documents,
        resolve_dataset_folders,
    },
    selection::{Format, resolve_dataset_format},
};
use omega_tokenizer::{Tokens, chat::ChatRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

const RAW_LIMIT: u64 = 64 * 1024 * 1024;
const RECORD_LIMIT: usize = 100_000;
const UNIT_LIMIT: usize = 200_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadinessCheck {
    pub category: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactIdentity {
    pub role: String,
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParentReadiness {
    pub path: PathBuf,
    pub model: ModelConfig,
    pub tokenizer: TokenizerIdentity,
    pub chat: ChatArtifact,
    pub manifest_sha256: String,
    pub model_sha256: String,
    pub protocol_sha256: String,
    pub resume_schema: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionReadiness {
    pub role: String,
    pub selections: Vec<String>,
    pub source_identity: String,
    pub examples: usize,
    pub input_tokens: Option<usize>,
    pub eligible_targets: Option<usize>,
    pub min_input_positions: Option<usize>,
    pub max_input_positions: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadinessReport {
    pub schema_version: u32,
    /// Technical checks only; never permission, human review or quality approval.
    pub passed: bool,
    pub production_acceptance: String,
    pub parent: Option<ParentReadiness>,
    pub partitions: Vec<PartitionReadiness>,
    pub checks: Vec<ReadinessCheck>,
    pub findings: Vec<String>,
    pub errors: Vec<String>,
    pub identities: Vec<ArtifactIdentity>,
}

impl ReadinessReport {
    fn check(&mut self, category: &str, result: Result<String, String>) {
        let passed = result.is_ok();
        let detail = result.unwrap_or_else(|error| error);
        if !passed {
            self.errors.push(format!("{category}: {detail}"));
        }
        self.checks.push(ReadinessCheck {
            category: category.into(),
            passed,
            detail,
        });
    }
}

fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("Cannot read identity {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Recheck the frozen inputs used by automatic training/development phases.
/// This does not tokenize, repeat overlap auditing, or open sealed payloads.
/// Callers must authenticate the saved report itself and retain its original
/// selections. Checks before and after an operation detect mutations; they are
/// not an atomic filesystem snapshot or protection against hostile replacement.
pub fn verify_active_inputs(report: &ReadinessReport, datasets_root: &Path) -> Result<(), String> {
    if report.schema_version != 1 || !report.passed || !report.errors.is_empty() {
        return Err("Active inputs require a successful frozen schema-1 readiness report".into());
    }
    if report.identities.len() > UNIT_LIMIT {
        return Err(format!(
            "Frozen readiness exceeds {UNIT_LIMIT} artifact identities"
        ));
    }
    let parent = report
        .parent
        .as_ref()
        .ok_or("Frozen readiness has no verified parent")?;
    for name in [
        "COMPLETE",
        "config.json",
        "manifest.json",
        "tokenizer.json",
        "resume.json",
        "resume.sha256",
        "model.mpk",
        "optimizer.mpk",
    ] {
        let role = format!("parent/{name}");
        let entries: Vec<_> = report
            .identities
            .iter()
            .filter(|item| item.role == role)
            .collect();
        if entries.len() != 1 || entries[0].path != parent.path.join(name) {
            return Err(format!(
                "Frozen readiness requires one exact {role} identity"
            ));
        }
    }
    // Reject growth of already frozen corpus files before release discovery,
    // which streams published payloads to verify their producer hashes.
    let mut sizes = BTreeMap::<&str, u64>::new();
    for item in report.identities.iter().filter(|item| {
        matches!(
            item.role.as_str(),
            "train" | "validation" | "base_validation"
        )
    }) {
        plain_path(&item.path, false)?;
        for ancestor in item
            .path
            .ancestors()
            .skip(1)
            .filter(|path| !path.as_os_str().is_empty())
        {
            plain_path(ancestor, true)?;
        }
        let size = fs::metadata(&item.path)
            .map_err(|error| error.to_string())?
            .len();
        let total = sizes.entry(&item.role).or_default();
        *total = total
            .checked_add(size)
            .ok_or("Active input size overflow")?;
        if *total > RAW_LIMIT {
            return Err(format!(
                "Active {} partition exceeds {RAW_LIMIT} raw bytes before release verification",
                item.role
            ));
        }
    }
    // Rediscover with the same format and release checks used for initial
    // readiness. Only the three active selections are passed to discovery.
    for role in ["train", "validation", "base_validation"] {
        let partitions: Vec<_> = report
            .partitions
            .iter()
            .filter(|part| part.role == role)
            .collect();
        if partitions.len() != 1 {
            return Err(format!("Frozen readiness requires one {role} partition"));
        }
        let requested = if role == "base_validation" {
            Format::Auto
        } else {
            Format::Chat
        };
        let format = resolve_dataset_format(
            datasets_root,
            &partitions[0].selections,
            requested,
            role == "train",
            ValidationSplit::None,
        )?;
        if role == "base_validation" && format == Format::Chat {
            return Err("Frozen base-validation selection changed to chat".into());
        }
        let (_, paths) = raw_sources(datasets_root, &partitions[0].selections, format.into())?;
        let current: BTreeSet<_> = paths.into_iter().collect();
        let expected: BTreeSet<_> = report
            .identities
            .iter()
            .filter(|item| item.role == role)
            .map(|item| item.path.clone())
            .collect();
        if expected.is_empty() || current != expected {
            return Err(format!(
                "Frozen {role} file inventory changed; restore the reviewed inputs or prepare a new experiment"
            ));
        }
    }
    let mut seen = BTreeSet::new();
    let mut consumed = BTreeMap::<&str, u64>::new();
    for item in &report.identities {
        if item.role == "sealed" {
            continue;
        }
        if !seen.insert((&item.role, &item.path)) {
            return Err(format!(
                "Duplicate frozen artifact identity: {}",
                item.path.display()
            ));
        }
        if !item.path.is_absolute()
            || item.sha256.len() != 64
            || !item.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!(
                "Invalid frozen identity for {}",
                item.path.display()
            ));
        }
        plain_path(&item.path, false)?;
        for ancestor in item
            .path
            .ancestors()
            .skip(1)
            .filter(|path| !path.as_os_str().is_empty())
        {
            plain_path(ancestor, true)?;
        }
        let corpus = matches!(
            item.role.as_str(),
            "train" | "validation" | "base_validation"
        );
        let limit = if corpus {
            RAW_LIMIT
                .checked_sub(*consumed.get(item.role.as_str()).unwrap_or(&0))
                .ok_or("Active input byte limit exceeded")?
        } else if item.role == "release_metadata" {
            16 * 1024 * 1024
        } else if item.role.starts_with("parent/") {
            u64::MAX - 1
        } else {
            return Err(format!(
                "Unknown frozen readiness artifact role: {}",
                item.role
            ));
        };
        let (hash, bytes) = hash_active_file(&item.path, limit)?;
        if hash != item.sha256 {
            return Err(format!(
                "Frozen {} input changed: {}; restore it or prepare a new experiment",
                item.role,
                item.path.display()
            ));
        }
        if corpus {
            *consumed.entry(&item.role).or_default() += bytes;
        }
    }
    Ok(())
}

fn hash_active_file(path: &Path, limit: u64) -> Result<(String, u64), String> {
    let mut file = File::open(path)
        .map_err(|error| format!("Cannot verify frozen input {}: {error}", path.display()))?
        .take(limit + 1);
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > limit {
            return Err(format!(
                "Active input exceeds its bounded byte allowance: {}",
                path.display()
            ));
        }
        hash.update(&buffer[..count]);
    }
    Ok((format!("{:x}", hash.finalize()), total))
}

fn check_parent(path: &Path, report: &mut ReadinessReport) -> Result<(Tokens, usize), String> {
    for entry in path
        .ancestors()
        .filter(|entry| !entry.as_os_str().is_empty())
    {
        plain_path(entry, true)?;
    }
    for name in [
        "COMPLETE",
        "config.json",
        "manifest.json",
        "tokenizer.json",
        "resume.json",
        "resume.sha256",
        "model.mpk",
        "optimizer.mpk",
    ] {
        plain_path(&path.join(name), false)?;
    }
    let saved = crate::resume::read_stage_parent(path)?;
    let (config, tokenizer, manifest) = load_checkpoint_header(path)?;
    let manifest = manifest.ok_or("Parent needs a versioned checkpoint manifest")?;
    let chat = manifest.chat.ok_or(
        "Parent has no frozen omega-chat-v1 controls; tokenizer resizing is not supported",
    )?;
    chat.validate_tokenizer(&tokenizer)?;
    let canonical = fs::canonicalize(path).map_err(|error| error.to_string())?;
    for name in [
        "COMPLETE",
        "config.json",
        "manifest.json",
        "tokenizer.json",
        "resume.json",
        "resume.sha256",
    ] {
        report.identities.push(ArtifactIdentity {
            role: format!("parent/{name}"),
            path: canonical.join(name),
            sha256: hash_file(&canonical.join(name))?,
        });
    }
    for (name, sha256) in [
        ("model.mpk", saved.model_sha256.clone()),
        ("optimizer.mpk", saved.optimizer_sha256.clone()),
    ] {
        report.identities.push(ArtifactIdentity {
            role: format!("parent/{name}"),
            path: canonical.join(name),
            sha256,
        });
    }
    let manifest_sha256 = report
        .identities
        .iter()
        .find(|item| item.role == "parent/manifest.json")
        .ok_or("Parent manifest identity was not recorded")?
        .sha256
        .clone();
    report.parent = Some(ParentReadiness {
        path: canonical,
        model: ModelConfig::from(&config),
        tokenizer: saved.tokenizer,
        protocol_sha256: sha256_bytes(
            &serde_json::to_vec(&chat).map_err(|error| error.to_string())?,
        ),
        chat,
        manifest_sha256,
        model_sha256: saved.model_sha256,
        resume_schema: saved.schema_version,
    });
    if manifest.metadata.dataset.is_none() {
        report.findings.push(
            "Parent has no recorded dataset provenance; prior training overlap cannot be certified"
                .into(),
        );
    }
    Ok((tokenizer, config.context_length))
}

fn plain_path(path: &Path, directory: bool) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("Cannot inspect parent {}: {error}", path.display()))?;
    let linked = metadata.file_type().is_symlink();
    #[cfg(windows)]
    let linked = {
        use std::os::windows::fs::MetadataExt;
        linked || metadata.file_attributes() & 0x400 != 0
    };
    if linked || (directory && !metadata.is_dir()) || (!directory && !metadata.is_file()) {
        return Err(format!(
            "Parent path must be a regular {} without symlinks/reparse points: {}",
            if directory { "directory" } else { "file" },
            path.display()
        ));
    }
    Ok(())
}

#[derive(Default)]
struct Audit {
    units: BTreeMap<String, String>,
    files: BTreeSet<PathBuf>,
}

fn normalized(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Audit {
    fn add(&mut self, kind: &str, content: &str, location: &str) -> Result<(), String> {
        if content.is_empty() {
            return Ok(());
        }
        let key = format!("{kind}:{}", sha256_bytes(content.as_bytes()));
        self.units.entry(key).or_insert_with(|| location.into());
        if self.units.len() > UNIT_LIMIT {
            return Err(format!(
                "Overlap audit exceeds {UNIT_LIMIT} distinct units; prepare a smaller reviewed partition"
            ));
        }
        Ok(())
    }
}

fn raw_sources(
    root: &Path,
    selections: &[String],
    format: DatasetFormat,
) -> Result<(PathBuf, Vec<PathBuf>), String> {
    let (root, paths) = discover_dataset_files(root, selections, format, 200_000)?;
    let mut total = 0u64;
    for path in &paths {
        let size = fs::metadata(path).map_err(|error| error.to_string())?.len();
        total = total
            .checked_add(size)
            .ok_or("Readiness input size overflow")?;
        if total > RAW_LIMIT {
            return Err(format!(
                "Readiness partition exceeds {RAW_LIMIT} raw bytes; select a bounded reviewed partition"
            ));
        }
    }
    Ok((root, paths.into_iter().collect()))
}

fn source_text(path: &Path, consumed: &mut u64) -> Result<String, String> {
    let remaining = RAW_LIMIT
        .checked_sub(*consumed)
        .ok_or("Readiness input byte limit exceeded")?;
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| error.to_string())?
        .take(remaining + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    *consumed += bytes.len() as u64;
    if *consumed > RAW_LIMIT {
        return Err("Readiness input changed or exceeds the raw byte limit".into());
    }
    String::from_utf8(bytes)
        .map_err(|error| format!("{} must contain UTF-8: {error}", path.display()))
}

fn note_file(report: &mut ReadinessReport, role: &str, path: &Path, raw: &str) {
    report.identities.push(ArtifactIdentity {
        role: role.into(),
        path: path.to_path_buf(),
        sha256: sha256_bytes(raw.as_bytes()),
    });
}

fn partition_roles(
    root: &Path,
    selections: &[String],
    role: &str,
    report: &mut ReadinessReport,
) -> Result<(), String> {
    let (_, directories) = resolve_dataset_folders(root, selections)?;
    for directory in directories {
        if let Some(published) = crate::release::inspect_partition(&directory)? {
            let expected = match role {
                "train" => "train",
                "sealed" => "test",
                _ => "validation",
            };
            if published.partition != expected {
                return Err(format!(
                    "{role} requires published {expected}, not {}",
                    published.partition
                ));
            }
            let release = directory
                .parent()
                .and_then(Path::parent)
                .ok_or("Missing published release root")?;
            for name in ["manifest.json", "COMPLETE"] {
                let path = release.join(name);
                if !report.identities.iter().any(|item| item.path == path) {
                    report.identities.push(ArtifactIdentity {
                        role: "release_metadata".into(),
                        sha256: hash_file(&path)?,
                        path,
                    });
                }
            }
            report.findings.push(format!("{role}: published membership and payload hashes are available at {}; source permission and near/semantic review are not approved by readiness", release.display()));
        } else {
            report.findings.push(format!("{role}: ordinary folder {} has no verified published source-group membership; explicit human provenance review is required", directory.display()));
        }
    }
    Ok(())
}

fn chat_partition(
    root: &Path,
    selections: &[String],
    role: &str,
    tokenizer: &Tokens,
    context: usize,
    report: &mut ReadinessReport,
) -> Result<Audit, String> {
    resolve_dataset_format(
        root,
        selections,
        Format::Chat,
        role == "train",
        ValidationSplit::None,
    )?;
    partition_roles(root, selections, role, report)?;
    let prepared = prepare_conversations(
        root,
        selections,
        tokenizer,
        context,
        ValidationSplit::None,
        42,
    )?;
    let source = &prepared.training;
    let count = source.example_count();
    let mut minimum = usize::MAX;
    let mut maximum = 0;
    let mut inputs = 0usize;
    for index in 0..count {
        let positions = source.example(index)?.len() - 1;
        minimum = minimum.min(positions);
        maximum = maximum.max(positions);
        inputs = inputs
            .checked_add(positions)
            .ok_or("Conversation input-token count overflow")?;
    }
    let summary = PartitionReadiness {
        role: role.into(),
        selections: selections.to_vec(),
        source_identity: source.identity()?,
        examples: count,
        input_tokens: Some(inputs),
        eligible_targets: Some(source.target_count()?),
        min_input_positions: Some(minimum),
        max_input_positions: Some(maximum),
    };
    let (canonical, paths) = raw_sources(root, selections, DatasetFormat::Jsonl)?;
    let mut audit = Audit::default();
    let mut consumed = 0u64;
    let mut records = 0usize;
    for path in paths {
        let raw = source_text(&path, &mut consumed)?;
        note_file(report, role, &path, &raw);
        audit.files.insert(path.clone());
        let relative = path
            .strip_prefix(&canonical)
            .map_err(|error| error.to_string())?;
        for (line, json) in raw
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
        {
            records += 1;
            if records > RECORD_LIMIT {
                return Err("Conversation overlap audit record limit exceeded".into());
            }
            let location = format!("{}/@record-{}", relative.display(), line + 1);
            let messages = parse_messages(json).map_err(|error| format!("{location}: {error}"))?;
            let canonical_record: Vec<_> = messages
                .iter()
                .map(|message| {
                    let role = match message.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    (role, normalized(&message.content))
                })
                .collect();
            audit.add(
                "record",
                &serde_json::to_string(&canonical_record).map_err(|error| error.to_string())?,
                &location,
            )?;
            let mut joined = Vec::new();
            for (index, message) in messages.iter().enumerate() {
                if message.role == ChatRole::System {
                    continue;
                }
                let text = normalized(&message.content);
                audit.add(
                    "content",
                    &text,
                    &format!("{location}/@message-{}", index + 1),
                )?;
                joined.push(text);
            }
            audit.add("content", &joined.join(" "), &location)?;
        }
    }
    if records != count {
        return Err(
            "Conversation sources changed between preparation and overlap inspection".into(),
        );
    }
    report.partitions.push(summary);
    Ok(audit)
}

fn base_partition(
    root: &Path,
    selections: &[String],
    report: &mut ReadinessReport,
) -> Result<Audit, String> {
    let format =
        resolve_dataset_format(root, selections, Format::Auto, false, ValidationSplit::None)?;
    if format == Format::Chat {
        return Err("Base-language regression requires a base text partition, not chat".into());
    }
    partition_roles(root, selections, "base_validation", report)?;
    let format: DatasetFormat = format.into();
    let (canonical, paths) = raw_sources(root, selections, format)?;
    let mut audit = Audit::default();
    let mut consumed = 0;
    let mut count = 0usize;
    let mut identity = Sha256::new();
    identity.update(b"omega-readiness-base-source-v1\0");
    for path in paths {
        let raw = source_text(&path, &mut consumed)?;
        note_file(report, "base_validation", &path, &raw);
        audit.files.insert(path.clone());
        let documents = parse_text_documents(&canonical, &path, &raw, format)?;
        for document in documents {
            count += 1;
            if count > RECORD_LIMIT {
                return Err("Base-language readiness record limit exceeded".into());
            }
            let id = document.id.to_string_lossy().replace('\\', "/");
            let content = normalized(&document.text);
            audit.add("content", &content, &id)?;
            identity.update((id.len() as u64).to_le_bytes());
            identity.update(id.as_bytes());
            identity.update((document.text.len() as u64).to_le_bytes());
            identity.update(document.text.as_bytes());
        }
    }
    if count == 0 {
        return Err("Base-language regression partition has no nonblank documents".into());
    }
    report.partitions.push(PartitionReadiness {
        role: "base_validation".into(),
        selections: selections.to_vec(),
        source_identity: format!("{:x}", identity.finalize()),
        examples: count,
        input_tokens: None,
        eligible_targets: None,
        min_input_positions: None,
        max_input_positions: None,
    });
    Ok(audit)
}

fn overlap(left: &Audit, right: &Audit) -> Result<String, String> {
    let paths: Vec<_> = left
        .files
        .intersection(&right.files)
        .take(4)
        .map(|path| path.display().to_string())
        .collect();
    if !paths.is_empty() {
        return Err(format!(
            "Shared physical source paths across partitions: {}",
            paths.join(", ")
        ));
    }
    let matches: Vec<_> = left
        .units
        .iter()
        .filter_map(|(key, location)| {
            right
                .units
                .get(key)
                .map(|other| format!("{location} overlaps {other}"))
        })
        .take(8)
        .collect();
    if !matches.is_empty() {
        return Err(format!(
            "Exact whole-record/message content overlap (up to 8 shown): {}",
            matches.join("; ")
        ));
    }
    Ok("No exact whole-record, non-system message or joined-dialogue overlap detected; semantic/group isolation remains unverified".into())
}

/// Inspect inputs without fitting, inference, checkpoint mutation or runtime
/// equality requirements. Input failures are retained as failed report checks.
/// All four partitions are required. Sealed data is read for identity/overlap and
/// structural/context readiness only; it is never evaluated or used for tuning.
pub fn inspect(
    parent: &Path,
    datasets_root: &Path,
    train: &[String],
    validation: &[String],
    base_validation: &[String],
    sealed: &[String],
) -> Result<ReadinessReport, String> {
    let mut report = ReadinessReport {
        schema_version: 1, passed: false, production_acceptance: "not_evaluated".into(),
        parent: None, partitions: Vec::new(), checks: Vec::new(), identities: Vec::new(), errors: Vec::new(),
        findings: vec![
            "Technical readiness does not approve permissions, source-family assignment, privacy, human response quality, budgets or promotion".into(),
            "Overlap checks collapse whitespace only; system-only boilerplate, substrings, Unicode composition, near duplicates and semantic leakage are not certified".into(),
            "Parent tensors are integrity checked but not decoded/executed here; supported weights transfer validates tensor loading separately".into(),
        ],
    };
    let prepared = check_parent(parent, &mut report);
    let (tokenizer, context) = match prepared {
        Ok(value) => {
            report.check("parent", Ok("Completed stage-parent headers, tokenizer/chat controls and payload hashes verified without requiring runtime equality".into()));
            value
        }
        Err(error) => {
            report.check("parent", Err(error));
            report.check(
                "data",
                Err("Data checks cannot run without a verified parent tokenizer/context".into()),
            );
            return Ok(report);
        }
    };
    let mut audits = BTreeMap::new();
    for (role, selections) in [
        ("train", train),
        ("validation", validation),
        ("sealed", sealed),
    ] {
        let result = chat_partition(
            datasets_root,
            selections,
            role,
            &tokenizer,
            context,
            &mut report,
        );
        match result {
            Ok(audit) => {
                audits.insert(role, audit);
                report.check(role, Ok("Whole conversation roles, context, assistant targets and source identity checked".into()));
            }
            Err(error) => report.check(role, Err(error)),
        }
    }
    match base_partition(datasets_root, base_validation, &mut report) {
        Ok(audit) => {
            audits.insert("base_validation", audit);
            report.check(
                "base_validation",
                Ok("Explicit base regression text and identities checked without fitting".into()),
            );
        }
        Err(error) => report.check("base_validation", Err(error)),
    }
    for (left, right) in [
        ("train", "validation"),
        ("train", "sealed"),
        ("train", "base_validation"),
        ("validation", "sealed"),
        ("base_validation", "sealed"),
    ] {
        if let (Some(a), Some(b)) = (audits.get(left), audits.get(right)) {
            report.check(&format!("overlap/{left}/{right}"), overlap(a, b));
        }
    }
    report.passed = report.errors.is_empty();
    Ok(report)
}
