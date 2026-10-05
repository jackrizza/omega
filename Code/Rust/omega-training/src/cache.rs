//! Immutable, bounded-source token caches for indexed training and continuation.
//!
//! Creation holds one capped source file (including its parsed JSONL text), one
//! capped encoded document, and bounded file/document metadata. Reads hold one
//! capped token document plus one output chunk; no corpus-sized token/example
//! vectors are retained. Token limits are checked after encoding, so tokenizer
//! scratch/output allocations can exceed them before rejection. Serialized
//! manifest input/output is byte-capped; deserialization adds per-entry/string
//! allocation overhead. These are input/count limits, not a process RSS cap;
//! model/backend memory is separate.
//! JSONL parsing retains all raw records from the current capped source file;
//! the retained corpus document-count limit is checked as those records encode.
//! This is bounded-source processing, not line-at-a-time JSONL streaming.
//! Directory discovery counts all visited entries, including ignored files and
//! repeated overlapping selections. This is not an authenticated storage format
//! or a power-loss durability guarantee; caches must not be concurrently edited.

use crate::{
    checkpoint::{
        DatasetProvenance, DocumentFingerprint, TokenizerIdentity, sha256_bytes, tokenizer_identity,
    },
    dataset::{
        DatasetFormat, DocumentInfo, ExampleIdentity, ExampleSource, ValidationSplit,
        discover_dataset_files, document_rank, parse_text_documents, validate_context_length,
        validation_document_count,
    },
};
use omega_tokenizer::Tokens;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
};

const MANIFEST: &str = "manifest.json";
const COMPLETE: &str = "COMPLETE";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheLimits {
    pub max_source_bytes: u64,
    pub max_document_tokens: usize,
    pub max_documents: usize,
    pub max_directory_entries: usize,
    pub max_manifest_bytes: usize,
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 16 * 1024 * 1024,
            max_document_tokens: 1_000_000,
            max_documents: 100_000,
            max_directory_entries: 200_000,
            max_manifest_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheOptions {
    pub format: DatasetFormat,
    pub context_length: usize,
    pub validation: ValidationSplit,
    pub split_seed: u64,
    pub limits: CacheLimits,
}

impl CacheOptions {
    fn validate(&self) -> Result<(), String> {
        validate_context_length(self.context_length)?;
        let limits = &self.limits;
        if limits.max_source_bytes == 0
            || limits.max_source_bytes >= usize::MAX as u64
            || limits.max_document_tokens < 2
            || limits.max_document_tokens > usize::MAX / 4
            || limits.max_documents == 0
            || limits.max_directory_entries == 0
            || limits.max_manifest_bytes == 0
            || limits.max_manifest_bytes == usize::MAX
        {
            return Err("Cache limits must be positive, tokens at least two, and byte/token bounds must fit in memory indices".into());
        }
        if let ValidationSplit::Ratio(ratio) = self.validation
            && (!ratio.is_finite() || !(0.0..1.0).contains(&ratio))
        {
            return Err("Validation ratio must be finite and in [0, 1)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePartition {
    Training,
    Validation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    id: PathBuf,
    source: usize,
    file_index: usize,
    token_count: usize,
    token_sha256: String,
    partition: CachePartition,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    preprocessing: String,
    options: CacheOptions,
    selections: Vec<String>,
    tokenizer: TokenizerIdentity,
    sources: Vec<Source>,
    documents: Vec<Document>,
    training_identity: String,
    validation_identity: String,
}

#[derive(Debug)]
struct CacheData {
    directory: PathBuf,
    source_root: PathBuf,
    manifest: Manifest,
}

#[derive(Debug)]
struct PartitionData {
    indices: Vec<usize>,
    documents: Vec<DocumentInfo>,
    files: Vec<PathBuf>,
    example_count: usize,
    target_count: usize,
    token_count: usize,
    identity: String,
}

#[derive(Clone, Debug)]
pub struct TokenCache {
    data: Arc<CacheData>,
    training: Arc<PartitionData>,
    validation: Arc<PartitionData>,
}

#[derive(Clone, Debug)]
pub struct CachedPartition {
    data: Arc<CacheData>,
    partition: Arc<PartitionData>,
}

impl TokenCache {
    pub fn partition(&self, partition: CachePartition) -> CachedPartition {
        CachedPartition {
            data: self.data.clone(),
            partition: match partition {
                CachePartition::Training => self.training.clone(),
                CachePartition::Validation => self.validation.clone(),
            },
        }
    }

    pub fn dataset_provenance(&self) -> DatasetProvenance {
        let manifest = &self.data.manifest;
        DatasetProvenance {
            selections: manifest.selections.clone(),
            format: match manifest.options.format {
                DatasetFormat::Text => "text",
                DatasetFormat::Jsonl => "jsonl",
            }
            .into(),
            fingerprint_kind: "token-ids-le-u32-v1".into(),
            documents: manifest
                .documents
                .iter()
                .map(|doc| DocumentFingerprint {
                    id: portable_id(&doc.id).expect("validated UTF-8 cache identity"),
                    partition: match doc.partition {
                        CachePartition::Training => "training",
                        CachePartition::Validation => "validation",
                    }
                    .into(),
                    sha256: doc.token_sha256.clone(),
                })
                .collect(),
            split_seed: manifest.options.split_seed,
            split_policy: match manifest.options.validation {
                ValidationSplit::None => "none".into(),
                ValidationSplit::Count(count) => format!("count:{count}"),
                ValidationSplit::Ratio(ratio) => format!("ratio:{ratio}"),
            },
        }
    }
}

impl CachedPartition {
    pub fn files(&self) -> &[PathBuf] {
        &self.partition.files
    }
    pub fn documents(&self) -> &[DocumentInfo] {
        &self.partition.documents
    }
    pub fn token_count(&self) -> usize {
        self.partition.token_count
    }
    pub fn example_count(&self) -> usize {
        self.partition.example_count
    }
    pub fn target_count(&self) -> Result<usize, String> {
        Ok(self.partition.target_count)
    }
    pub fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        if index >= self.example_count() {
            return Err(format!("Cached example index {index} is out of bounds"));
        }
        let position = self
            .partition
            .documents
            .partition_point(|doc| doc.example_range.end <= index);
        let info = &self.partition.documents[position];
        let doc = &self.data.manifest.documents[self.partition.indices[position]];
        let tokens = read_tokens(
            &self.data.directory,
            doc,
            &self.data.manifest.options.limits,
        )?;
        let start = (index - info.example_range.start)
            .checked_mul(self.data.manifest.options.context_length)
            .ok_or("Cached chunk offset overflows usize")?;
        let end = tokens
            .len()
            .min(start.saturating_add(self.data.manifest.options.context_length + 1));
        Ok(tokens[start..end].to_vec())
    }

    /// Identity of ordered update inputs, equivalent to eager TrainingSet. The
    /// cache was fully verified on open; each subsequent read rechecks its file.
    pub fn identity(&self) -> Result<String, String> {
        Ok(self.partition.identity.clone())
    }

    /// A cursor is the next example index. `example_count()` is a valid exhausted
    /// cursor. Errors stop iteration; callers must not advance a training cursor
    /// until the corresponding example/update succeeded.
    pub fn iter_from(&self, cursor: usize) -> Result<CachedExamples, String> {
        if cursor > self.example_count() {
            return Err("Cached example cursor is out of bounds".into());
        }
        Ok(CachedExamples {
            partition: self.clone(),
            cursor,
            failed: false,
        })
    }
}

impl ExampleSource for CachedPartition {
    fn example_count(&self) -> usize {
        self.example_count()
    }
    fn target_count(&self) -> Result<usize, String> {
        self.target_count()
    }
    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        self.example(index)
    }
    fn identity(&self) -> Result<String, String> {
        self.identity()
    }
}

pub struct CachedExamples {
    partition: CachedPartition,
    cursor: usize,
    failed: bool,
}
impl Iterator for CachedExamples {
    type Item = Result<Vec<u32>, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.cursor == self.partition.example_count() {
            return None;
        }
        let result = self.partition.example(self.cursor);
        match &result {
            Ok(_) => self.cursor += 1,
            Err(_) => self.failed = true,
        }
        Some(result)
    }
}

fn portable_id(path: &Path) -> Result<String, String> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!(
            "Cache identities must be nonempty relative paths without traversal: {path:?}"
        ));
    }
    path.components()
        .map(|part| {
            part.as_os_str()
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| "Cache identities require UTF-8 paths".to_string())
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join("/"))
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("Cannot inspect {}: {e}", path.display()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "Expected a regular non-symlink file: {}",
            path.display()
        ));
    }
    if metadata.len() > maximum {
        return Err(format!("{} exceeds byte limit {maximum}", path.display()));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?
        .take(maximum.checked_add(1).ok_or("Read limit overflows u64")?)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    if bytes.len() as u64 > maximum {
        return Err(format!("{} exceeds byte limit {maximum}", path.display()));
    }
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("Cannot create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|e| format!("Cannot write {}: {e}", path.display()))
}

struct BoundedManifest {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedManifest {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("Cache manifest exceeds byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn token_path(root: &Path, index: usize) -> PathBuf {
    root.join(format!("tokens-{index}.bin"))
}

fn validate_cache_tokenizer(tokenizer: &Tokens) -> Result<usize, String> {
    let pipeline: serde_json::Value = serde_json::from_str(&tokenizer.to_json()?)
        .map_err(|e| format!("Cannot inspect cache tokenizer pipeline: {e}"))?;
    if pipeline
        .get("truncation")
        .is_some_and(|value| !value.is_null())
    {
        return Err("Token caches require tokenizer truncation to be disabled; upstream overflow reporting cannot prove complete documents".into());
    }
    omega_nn::validate_tokenizer(tokenizer)
}

fn read_tokens(root: &Path, document: &Document, limits: &CacheLimits) -> Result<Vec<u32>, String> {
    if document.token_count < 2 || document.token_count > limits.max_document_tokens {
        return Err("Cached document violates token count limit".into());
    }
    let expected = document
        .token_count
        .checked_mul(4)
        .ok_or("Cached token size overflows usize")?;
    let bytes = read_bounded(&token_path(root, document.file_index), expected as u64)?;
    if bytes.len() != expected || sha256_bytes(&bytes) != document.token_sha256 {
        return Err(format!(
            "Cached token file checksum/length mismatch for {:?}",
            document.id
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn source_records(
    root: &Path,
    paths: &[PathBuf],
    limits: &CacheLimits,
) -> Result<Vec<Source>, String> {
    paths
        .iter()
        .map(|path| {
            let relative = path
                .strip_prefix(root)
                .map_err(|e| format!("Cannot identify source: {e}"))?
                .to_path_buf();
            portable_id(&relative)?;
            let bytes = read_bounded(path, limits.max_source_bytes)?;
            Ok(Source {
                path: relative,
                bytes: bytes.len() as u64,
                sha256: sha256_bytes(&bytes),
            })
        })
        .collect()
}

fn assign_partitions(documents: &mut [Document], options: &CacheOptions) -> Result<(), String> {
    let count = validation_document_count(documents.len(), options.validation)?;
    let mut ranks: Vec<_> = documents
        .iter()
        .enumerate()
        .map(|(index, doc)| (document_rank(&doc.id, options.split_seed), &doc.id, index))
        .collect();
    ranks.sort();
    let validation: BTreeSet<_> = ranks[..count].iter().map(|(_, _, index)| *index).collect();
    for (index, doc) in documents.iter_mut().enumerate() {
        doc.partition = if validation.contains(&index) {
            CachePartition::Validation
        } else {
            CachePartition::Training
        };
    }
    Ok(())
}

fn partition_data(data: &CacheData, partition: CachePartition) -> Result<PartitionData, String> {
    let mut result = PartitionData {
        indices: Vec::new(),
        documents: Vec::new(),
        files: Vec::new(),
        example_count: 0,
        target_count: 0,
        token_count: 0,
        identity: String::new(),
    };
    for (index, doc) in data
        .manifest
        .documents
        .iter()
        .enumerate()
        .filter(|(_, doc)| doc.partition == partition)
    {
        let chunks = (doc.token_count - 1).div_ceil(data.manifest.options.context_length);
        let end = result
            .example_count
            .checked_add(chunks)
            .ok_or("Cache example count overflows usize")?;
        result.target_count = result
            .target_count
            .checked_add(doc.token_count - 1)
            .ok_or("Cache target count overflows usize")?;
        result.token_count = result
            .token_count
            .checked_add(doc.token_count)
            .ok_or("Cache source token count overflows usize")?;
        let path = data
            .source_root
            .join(&data.manifest.sources[doc.source].path);
        result.files.push(path.clone());
        result.documents.push(DocumentInfo {
            id: doc.id.clone(),
            path,
            example_range: result.example_count..end,
        });
        result.indices.push(index);
        result.example_count = end;
    }
    result.files.sort();
    result.files.dedup();
    let mut identity = ExampleIdentity::new(result.example_count);
    for &index in &result.indices {
        let tokens = read_tokens(
            &data.directory,
            &data.manifest.documents[index],
            &data.manifest.options.limits,
        )?;
        for start in (0..tokens.len() - 1).step_by(data.manifest.options.context_length) {
            identity.add(
                &tokens[start
                    ..tokens
                        .len()
                        .min(start.saturating_add(data.manifest.options.context_length + 1))],
            );
        }
    }
    result.identity = identity.finish();
    Ok(result)
}

/// Create a new cache directory; an existing path is never reused or overwritten.
/// Parents must exist. Failures leave only this newly-created incomplete cache.
/// Sources are rechecked before COMPLETE is written last. Cache options, including
/// limits, are part of validity; opening with different options rejects the cache.
pub fn create_token_cache(
    directory: &Path,
    datasets_root: &Path,
    folders: &[String],
    tokenizer: &Tokens,
    options: &CacheOptions,
) -> Result<TokenCache, String> {
    options.validate()?;
    validate_cache_tokenizer(tokenizer)?;
    let (root, paths) = discover_dataset_files(
        datasets_root,
        folders,
        options.format,
        options.limits.max_directory_entries,
    )?;
    fs::create_dir(directory)
        .map_err(|e| format!("Cannot create new token cache {}: {e}", directory.display()))?;
    let mut sources = Vec::new();
    let mut documents = Vec::new();
    for path in &paths {
        let bytes = read_bounded(path, options.limits.max_source_bytes)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| format!("Cannot read UTF-8 source {}: {e}", path.display()))?;
        let relative = path
            .strip_prefix(&root)
            .map_err(|e| format!("Cannot identify source: {e}"))?
            .to_path_buf();
        portable_id(&relative)?;
        let source = sources.len();
        sources.push(Source {
            path: relative,
            bytes: bytes.len() as u64,
            sha256: sha256_bytes(&bytes),
        });
        for document in parse_text_documents(&root, path, text, options.format)? {
            if documents.len() >= options.limits.max_documents {
                return Err("Source documents exceed cache document limit".into());
            }
            portable_id(&document.id)?;
            let tokens = omega_nn::encode_text(tokenizer, &document.text)
                .map_err(|e| format!("Cannot tokenize {:?}: {e}", document.id))?;
            if tokens.len() < 2 || tokens.len() > options.limits.max_document_tokens {
                return Err(format!(
                    "Document {:?} must contain 2..={} tokens",
                    document.id, options.limits.max_document_tokens
                ));
            }
            let bytes: Vec<_> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
            let file_index = documents.len();
            write_new(&token_path(directory, file_index), &bytes)?;
            documents.push(Document {
                id: document.id,
                source,
                file_index,
                token_count: tokens.len(),
                token_sha256: sha256_bytes(&bytes),
                partition: CachePartition::Training,
            });
        }
    }
    documents.sort_by(|left, right| left.id.cmp(&right.id));
    assign_partitions(&mut documents, options)?;
    let mut selections = folders.to_vec();
    selections.sort();
    selections.dedup();
    let mut data = CacheData {
        directory: fs::canonicalize(directory)
            .map_err(|e| format!("Cannot resolve cache directory: {e}"))?,
        source_root: root.clone(),
        manifest: Manifest {
            schema_version: VERSION,
            preprocessing: "omega-text-jsonl-records-v1".into(),
            options: options.clone(),
            selections,
            tokenizer: tokenizer_identity(tokenizer)?,
            sources,
            documents,
            training_identity: String::new(),
            validation_identity: String::new(),
        },
    };
    let training = partition_data(&data, CachePartition::Training)?;
    let validation = partition_data(&data, CachePartition::Validation)?;
    data.manifest.training_identity = training.identity.clone();
    data.manifest.validation_identity = validation.identity.clone();
    let mut output = BoundedManifest {
        bytes: Vec::new(),
        limit: options.limits.max_manifest_bytes,
    };
    serde_json::to_writer_pretty(&mut output, &data.manifest)
        .map_err(|e| format!("Cannot encode cache manifest: {e}"))?;
    let manifest = output.bytes;
    let (_, current_paths) = discover_dataset_files(
        datasets_root,
        folders,
        options.format,
        options.limits.max_directory_entries,
    )?;
    if source_records(&root, &current_paths, &options.limits)? != data.manifest.sources {
        return Err("Sources changed while creating token cache".into());
    }
    write_new(&directory.join(MANIFEST), &manifest)?;
    write_new(
        &directory.join(COMPLETE),
        format!("omega-token-cache-v1\n{}\n", sha256_bytes(&manifest)).as_bytes(),
    )?;
    Ok(TokenCache {
        data: Arc::new(data),
        training: Arc::new(training),
        validation: Arc::new(validation),
    })
}

/// Open only a complete cache matching source bytes/paths, tokenizer pipeline,
/// format, chunking/split settings and limits. Rehashes every source and token
/// file with bounded reads, validates indices and recomputes example identities.
/// Source-root relocation is supported when relative paths/content are unchanged.
pub fn open_token_cache(
    directory: &Path,
    datasets_root: &Path,
    folders: &[String],
    tokenizer: &Tokens,
    options: &CacheOptions,
) -> Result<TokenCache, String> {
    options.validate()?;
    let marker = read_bounded(&directory.join(COMPLETE), 128)
        .map_err(|e| format!("Incomplete token cache: {e}"))?;
    let bytes = read_bounded(
        &directory.join(MANIFEST),
        options.limits.max_manifest_bytes as u64,
    )?;
    if marker != format!("omega-token-cache-v1\n{}\n", sha256_bytes(&bytes)).as_bytes() {
        return Err("Token cache completion marker/manifest checksum mismatch".into());
    }
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid token cache manifest: {e}"))?;
    if manifest.schema_version != VERSION || manifest.preprocessing != "omega-text-jsonl-records-v1"
    {
        return Err("Unsupported token cache schema/preprocessing version".into());
    }
    if &manifest.options != options {
        return Err("Token cache settings/limits mismatch".into());
    }
    let mut selections = folders.to_vec();
    selections.sort();
    selections.dedup();
    if manifest.selections != selections {
        return Err("Token cache dataset selections mismatch".into());
    }
    if manifest.tokenizer != tokenizer_identity(tokenizer)? {
        return Err("Token cache tokenizer identity mismatch".into());
    }
    let vocab_size = validate_cache_tokenizer(tokenizer)?;
    let (root, paths) = discover_dataset_files(
        datasets_root,
        folders,
        options.format,
        options.limits.max_directory_entries,
    )?;
    if manifest.sources.len() > options.limits.max_directory_entries
        || manifest.documents.len() > options.limits.max_documents
    {
        return Err("Cache metadata exceeds count limits".into());
    }
    if source_records(&root, &paths, &options.limits)? != manifest.sources {
        return Err("Stale token cache: source identity/content changed".into());
    }
    let mut ids = BTreeSet::new();
    let mut file_indices = BTreeSet::new();
    for doc in &manifest.documents {
        portable_id(&doc.id)?;
        if doc.source >= manifest.sources.len()
            || doc.file_index >= manifest.documents.len()
            || !file_indices.insert(doc.file_index)
            || !ids.insert(&doc.id)
        {
            return Err("Invalid or duplicate cache document/source indices".into());
        }
        let source = &manifest.sources[doc.source].path;
        let valid_id = match options.format {
            DatasetFormat::Text => &doc.id == source,
            DatasetFormat::Jsonl => {
                doc.id.parent() == Some(source.as_path())
                    && doc
                        .id
                        .file_name()
                        .and_then(|n| n.to_str())
                        .and_then(|n| n.strip_prefix("@record-"))
                        .is_some_and(|n| {
                            n.parse::<usize>()
                                .is_ok_and(|line| line > 0 && line.to_string() == n)
                        })
            }
        };
        if !valid_id {
            return Err("Cached document identity does not match its source".into());
        }
        let tokens = read_tokens(directory, doc, &options.limits)?;
        if tokens.iter().any(|&id| u64::from(id) >= vocab_size as u64) {
            return Err("Cached tokens are outside the tokenizer vocabulary".into());
        }
    }
    if !manifest
        .documents
        .windows(2)
        .all(|pair| pair[0].id < pair[1].id)
    {
        return Err("Cache document ordering is invalid".into());
    }
    let mut expected_documents = manifest.documents.clone();
    assign_partitions(&mut expected_documents, options)?;
    if expected_documents
        .iter()
        .zip(&manifest.documents)
        .any(|(expected, actual)| expected.partition != actual.partition)
    {
        return Err("Cache split membership mismatch".into());
    }
    let data = CacheData {
        directory: fs::canonicalize(directory)
            .map_err(|e| format!("Cannot resolve cache directory: {e}"))?,
        source_root: root,
        manifest,
    };
    let training = partition_data(&data, CachePartition::Training)?;
    let validation = partition_data(&data, CachePartition::Validation)?;
    if training.identity != data.manifest.training_identity
        || validation.identity != data.manifest.validation_identity
    {
        return Err("Cache ordered-example identity mismatch".into());
    }
    Ok(TokenCache {
        data: Arc::new(data),
        training: Arc::new(training),
        validation: Arc::new(validation),
    })
}
