//! Deterministic, document-local training sequences from explicit corpus formats.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    ops::Range,
    path::{Component, Path, PathBuf},
};

use sha2::{Digest, Sha256};

/// Original next-token objective used by text and existing token caches.
pub const ALL_TARGETS_OBJECTIVE: &str = "all-next-token-v1";
/// Assistant content and reply terminators under the version-one chat protocol.
pub const ASSISTANT_TARGETS_OBJECTIVE: &str = "assistant-next-token-omega-chat-v1";

/// Indexed, deterministic unpadded examples. Read failures must be returned before
/// an optimizer update. Implementations must keep the same order and identity
/// throughout a session. Identity must include order, IDs, target masks and the
/// objective/protocol for masked sources. Legacy all-target identities stay stable.
pub trait ExampleSource {
    fn example_count(&self) -> usize;
    /// Total eligible next-token labels, excluding masked context and padding.
    fn target_count(&self) -> Result<usize, String>;
    fn example(&self, index: usize) -> Result<Vec<u32>, String>;
    fn identity(&self) -> Result<String, String>;
    /// `None` preserves legacy all-target behavior. Explicit masks align with
    /// `example(index)?[1..]`: these labels are already shifted once.
    fn target_mask(&self, _index: usize) -> Result<Option<Vec<bool>>, String> {
        Ok(None)
    }
    fn objective(&self) -> &str {
        ALL_TARGETS_OBJECTIVE
    }
}

/// Validate eligibility independently from input validity. Every example must
/// contribute a target; user/system context still remains in the input sequence.
pub(crate) fn checked_target_mask<S: ExampleSource + ?Sized>(
    source: &S,
    index: usize,
    token_count: usize,
) -> Result<Option<Vec<bool>>, String> {
    if token_count < 2 {
        return Err(format!("Example {index} must contain at least two tokens"));
    }
    let mask = source.target_mask(index)?;
    match (source.objective(), &mask) {
        (ALL_TARGETS_OBJECTIVE, None) => {}
        (ASSISTANT_TARGETS_OBJECTIVE, Some(mask)) => {
            if mask.len() != token_count - 1 {
                return Err(format!(
                    "Example {index} target mask must match the already shifted targets"
                ));
            }
            if !mask.iter().any(|included| *included) {
                return Err(format!("Example {index} has no supervised targets"));
            }
        }
        (ALL_TARGETS_OBJECTIVE, Some(_)) => {
            return Err("All-target objective must not supply an explicit target mask".into());
        }
        (ASSISTANT_TARGETS_OBJECTIVE, None) => {
            return Err(format!(
                "Assistant objective requires a target mask for example {index}"
            ));
        }
        (objective, _) => return Err(format!("Unsupported source objective {objective:?}")),
    }
    Ok(mask)
}

pub(crate) struct ExampleIdentity(Sha256);

impl ExampleIdentity {
    pub(crate) fn new(count: usize) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"omega-example-sequences-v1\0");
        digest.update((count as u64).to_le_bytes());
        Self(digest)
    }

    pub(crate) fn add(&mut self, ids: &[u32]) {
        self.0.update((ids.len() as u64).to_le_bytes());
        for id in ids {
            self.0.update(id.to_le_bytes());
        }
    }

    pub(crate) fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}

/// Text documents split into next-token training sequences.
#[derive(Debug)]
pub struct TrainingSet {
    /// Sequences of 2..=context_length + 1 tokens, in file/chunk order.
    /// Consecutive sequences from a document share exactly one boundary token.
    pub examples: Vec<Vec<u32>>,
    /// Sorted, absolute paths of unique source files contributing examples.
    /// Multiple JSONL documents in one file share one entry. Blank documents
    /// are omitted; source files may contribute records to both partitions.
    pub files: Vec<PathBuf>,
    /// Source tokens in contributing documents, counting overlap only once.
    /// This is not the number of prediction targets (one fewer per document).
    pub token_count: usize,
}

impl ExampleSource for TrainingSet {
    fn example_count(&self) -> usize {
        self.examples.len()
    }

    fn target_count(&self) -> Result<usize, String> {
        self.examples
            .iter()
            .enumerate()
            .try_fold(0usize, |total, (index, ids)| {
                if ids.len() < 2 {
                    return Err(format!("Example {index} must contain at least two tokens"));
                }
                total
                    .checked_add(ids.len() - 1)
                    .ok_or_else(|| "Example target count overflows usize".into())
            })
    }

    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        self.examples
            .get(index)
            .cloned()
            .ok_or_else(|| format!("Example index {index} is out of bounds"))
    }

    fn identity(&self) -> Result<String, String> {
        let mut digest = ExampleIdentity::new(self.examples.len());
        for ids in &self.examples {
            digest.add(ids);
        }
        Ok(digest.finish())
    }
}

/// One complete, unchunked document. Text IDs are paths relative to the dataset root,
/// so moving a corpus without renaming its files preserves split membership.
/// IDs identify paths, not content: copies under different names are documents.
/// JSONL IDs append `/@record-N` for the 1-based physical line. These are logical
/// document IDs, not filesystem paths; `path` is always the actual source file.
#[derive(Clone, Debug)]
pub struct TokenizedDocument {
    pub id: PathBuf,
    pub path: PathBuf,
    pub tokens: Vec<u32>,
}

/// Selected nonblank documents, before any split or chunking.
#[derive(Clone, Debug)]
pub struct DocumentCorpus {
    pub documents: Vec<TokenizedDocument>,
}

/// Select exactly one corpus format; extensions and schema are never guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DatasetFormat {
    /// UTF-8 `.txt`: each nonblank file is one document.
    Text,
    /// UTF-8 `.jsonl`: each nonblank line is an object with a required string
    /// `text` field. Extra fields are allowed. Blank text values are skipped.
    Jsonl,
}

impl DatasetFormat {
    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Jsonl => "jsonl",
        }
    }
}

/// Nonblank corpus text before tokenizer processing. IDs follow
/// [`TokenizedDocument`]; `path` is the absolute physical source file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextDocument {
    pub id: PathBuf,
    pub path: PathBuf,
    pub text: String,
}

/// Requested number of validation documents, not a fraction of tokens/chunks.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ValidationSplit {
    None,
    /// Finite ratio in [0, 1). Uses floor(document_count * ratio).
    /// A positive ratio rounding to zero is rejected, never silently clamped.
    Ratio(f64),
    /// Must leave at least one training document. Zero disables validation.
    Count(usize),
}

/// Identity and chunk provenance for a document in a partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentInfo {
    pub id: PathBuf,
    pub path: PathBuf,
    /// Indices into this partition's `set.examples`.
    pub example_range: Range<usize>,
}

#[derive(Debug)]
pub struct DocumentPartition {
    pub set: TrainingSet,
    pub documents: Vec<DocumentInfo>,
}

#[derive(Debug)]
pub struct DatasetSplit {
    pub training: DocumentPartition,
    /// Empty only when validation was explicitly disabled (None/zero).
    pub validation: DocumentPartition,
}

/// Load one or more relative folder paths below `datasets_root`.
///
/// Recursively reads regular files with the exact extension `.txt` as UTF-8.
/// Other formats, including JSON, are not parsed. Every selected folder must
/// contain at least one `.txt` file, even when selections overlap. Blank files
/// are skipped; nonblank files encoding to fewer than two tokens are errors.
/// Each adjacent token pair appears in exactly one example, never across files.
///
/// Rejects empty selections, absolute paths, traversal, symlinks (including
/// root/selection ancestors and all recursively encountered entries), zero or
/// overflowing context lengths, I/O/encoding errors, and an empty result.
/// Filesystem validation assumes the tree is not concurrently modified; it is
/// not a security boundary against symlink replacement races.
pub fn build_training_set(
    datasets_root: &Path,
    folders: &[String],
    tokenizer: &omega_tokenizer::Tokens,
    context_length: usize,
) -> Result<TrainingSet, String> {
    validate_context_length(context_length)?;
    let corpus = load_document_corpus(datasets_root, folders, tokenizer)?;
    Ok(
        split_document_corpus(&corpus, context_length, ValidationSplit::None, 0)?
            .training
            .set,
    )
}

pub(crate) fn validate_context_length(context_length: usize) -> Result<usize, String> {
    if context_length == 0 {
        return Err("Context length must be greater than zero".into());
    }
    context_length
        .checked_add(1)
        .ok_or_else(|| "Context length + 1 overflows usize".to_string())
}

/// Read complete documents using the same selection, safety, encoding and blank
/// file rules as [`build_training_set`]. Overlapping selections are deduplicated.
/// Selected directories are canonicalized after symlink validation, so case
/// aliases share identity on case-insensitive filesystems without lowercasing
/// distinct paths on case-sensitive filesystems.
/// Returned documents are sorted by path and retain root-relative identities.
pub fn load_document_corpus(
    datasets_root: &Path,
    folders: &[String],
    tokenizer: &omega_tokenizer::Tokens,
) -> Result<DocumentCorpus, String> {
    load_document_corpus_with_format(datasets_root, folders, tokenizer, DatasetFormat::Text)
}

/// Tokenize the documents from [`load_text_documents`] without changing their
/// identities or merging record boundaries. Nonblank documents must encode to
/// at least two tokens; tokenizer padding/overflow errors are preserved.
pub fn load_document_corpus_with_format(
    datasets_root: &Path,
    folders: &[String],
    tokenizer: &omega_tokenizer::Tokens,
    format: DatasetFormat,
) -> Result<DocumentCorpus, String> {
    let documents = load_text_documents(datasets_root, folders, format)?;
    let documents = documents.into_iter().map(|document| {
        let location = match document_record_number(&document.id, &document.path)? {
            Some(line) => format!("{}:{line}", document.path.display()),
            None => document.path.display().to_string(),
        };
        let tokens = omega_nn::encode_text(tokenizer, &document.text)
            .map_err(|e| format!("Cannot tokenize {location}: {e}"))?;
        if tokens.len() < 2 {
            return Err(format!(
                "Nonblank document {location} must encode to at least two tokens for next-token training"
            ));
        }
        Ok(TokenizedDocument { id: document.id, path: document.path, tokens })
    }).collect::<Result<Vec<_>, String>>()?;
    Ok(DocumentCorpus { documents })
}

/// Read raw documents with the same path/symlink safety and selection deduplication
/// as [`load_document_corpus`]. Each folder must contain a file with the selected
/// exact lowercase extension; other formats are ignored. Text is not normalized.
///
/// JSONL is one object per physical line with a required string `text` field;
/// extra fields are ignored, while malformed JSON/missing/non-string text fails
/// with the source path and 1-based physical line. Blank lines/text are skipped.
/// JSONL IDs are `relative/file.jsonl/@record-N`, retaining physical line numbers
/// even across blank records. Every record is a separate document, not a chunk.
/// Reordering/inserting lines changes identity. IDs never collide with text file
/// IDs, and repeated source files are read only once. Empty results are errors.
pub fn load_text_documents(
    datasets_root: &Path,
    folders: &[String],
    format: DatasetFormat,
) -> Result<Vec<TextDocument>, String> {
    let (root, paths) = discover_dataset_files(datasets_root, folders, format, usize::MAX)?;
    let mut documents = Vec::new();
    for path in paths {
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read UTF-8 text file {}: {e}", path.display()))?;
        documents.extend(parse_text_documents(&root, &path, &text, format)?);
    }
    if documents.is_empty() {
        return Err(format!(
            "No training examples: all selected .{} documents are blank",
            format.extension()
        ));
    }
    Ok(documents)
}

/// Resolve explicit dataset folders without hiding symlinks or traversal.
pub fn resolve_dataset_folders(
    datasets_root: &Path,
    folders: &[String],
) -> Result<(PathBuf, Vec<PathBuf>), String> {
    if folders.is_empty() {
        return Err("Select at least one dataset folder".into());
    }
    // Validate before canonicalizing, which would otherwise hide symlinks.
    let root_path = if datasets_root.is_absolute() {
        datasets_root.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("Cannot resolve current directory: {e}"))?
            .join(datasets_root)
    };
    require_directory_chain(&root_path)?;
    let root = fs::canonicalize(&root_path)
        .map_err(|e| format!("Cannot resolve dataset root {}: {e}", root_path.display()))?;
    let mut directories = Vec::new();
    for folder in folders {
        let relative = Path::new(folder);
        // Also reject Windows path syntax on Unix, where backslashes and drive
        // prefixes would otherwise be ordinary filename characters.
        if folder.is_empty()
            || folder.contains(':')
            || folder.contains('\\')
            || relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(format!(
                "Invalid dataset folder {folder:?}: expected a relative path without traversal"
            ));
        }
        let selected = root.join(relative);
        require_directory_chain(&selected)?;
        // Resolve filesystem spelling before discovering paths. Caller spelling
        // alone is not document identity on case-insensitive filesystems.
        let selected = fs::canonicalize(&selected)
            .map_err(|e| format!("Cannot resolve dataset folder {}: {e}", selected.display()))?;
        directories.push(selected);
    }
    Ok((root, directories))
}

pub(crate) fn discover_dataset_files(
    datasets_root: &Path,
    folders: &[String],
    format: DatasetFormat,
    max_entries: usize,
) -> Result<(PathBuf, Vec<PathBuf>), String> {
    if folders.is_empty() {
        return Err("Select at least one dataset folder".into());
    }
    if folders.len() > max_entries {
        return Err("Dataset folder selections exceed directory entry limit".into());
    }
    let mut remaining = max_entries;

    let (root, directories) = resolve_dataset_folders(datasets_root, folders)?;
    let mut paths = BTreeSet::new();
    for selected in directories {
        if crate::release::inspect_partition(&selected)?.is_some() && format != DatasetFormat::Jsonl
        {
            return Err(
                "omega-datasets partitions contain JSONL; use the matching jsonl/chat format"
                    .into(),
            );
        }
        let mut selected_files = BTreeSet::new();
        collect_dataset_files(
            &selected,
            &mut selected_files,
            format.extension(),
            &mut remaining,
        )?;
        if selected_files.is_empty() {
            return Err(match format {
                DatasetFormat::Text => format!(
                    "Dataset folder {} contains no .txt files; only UTF-8 .txt datasets are supported in text mode (JSON is not supported)",
                    selected.display()
                ),
                DatasetFormat::Jsonl => format!(
                    "Dataset folder {} contains no .jsonl files; select JSONL files with one text-string object per line",
                    selected.display()
                ),
            });
        }
        paths.extend(selected_files);
    }
    Ok((root, paths.into_iter().collect()))
}

pub(crate) fn parse_text_documents(
    root: &Path,
    path: &Path,
    text: &str,
    format: DatasetFormat,
) -> Result<Vec<TextDocument>, String> {
    let mut documents = Vec::new();
    let id = path
        .strip_prefix(root)
        .map_err(|e| format!("Cannot identify document {}: {e}", path.display()))?
        .to_path_buf();
    match format {
        DatasetFormat::Text => {
            if !text.trim().is_empty() {
                documents.push(TextDocument {
                    id,
                    path: path.to_path_buf(),
                    text: text.to_string(),
                });
            }
        }
        DatasetFormat::Jsonl => {
            for (index, line) in text.lines().enumerate() {
                let line_number = index + 1;
                if line.trim().is_empty() {
                    continue;
                }
                let record: serde_json::Value = serde_json::from_str(line).map_err(|e| format!(
                        "Invalid JSONL record {}:{line_number}: expected an object with a string text field: {e}", path.display()
                    ))?;
                let text = record.as_object().and_then(|object| object.get("text"))
                        .and_then(serde_json::Value::as_str).ok_or_else(|| format!(
                            "Invalid JSONL record {}:{line_number}: expected an object with a string text field", path.display()
                        ))?;
                if !text.trim().is_empty() {
                    documents.push(TextDocument {
                        id: id.join(format!("@record-{line_number}")),
                        path: path.to_path_buf(),
                        text: text.to_string(),
                    });
                }
            }
        }
    }
    Ok(documents)
}

fn document_record_number(id: &Path, path: &Path) -> Result<Option<usize>, String> {
    if path
        .extension()
        .is_none_or(|extension| extension != "jsonl")
    {
        return Ok(None);
    }
    let number = id
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("@record-"))
        .and_then(|value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|&number| number > 0 && value == number.to_string())
        });
    if let Some(number) = number
        && id.parent().is_some_and(|parent| {
            parent.extension().is_some_and(|ext| ext == "jsonl") && path.ends_with(parent)
        })
    {
        return Ok(Some(number));
    }
    Err(format!(
        "Invalid JSONL document identity {id:?} for {}: expected its relative source path followed by /@record-N with a positive physical line",
        path.display()
    ))
}

/// Split whole documents, then chunk each partition without crossing files.
///
/// Membership is determined by ascending (seeded identity rank, identity), using
/// the fixed FNV-1a/SplitMix64 rank below. Input order, folder selection order,
/// context length and root location do not affect membership. Adding/renaming
/// documents can change membership; these IDs are not content fingerprints.
/// Output documents/chunks are sorted by identity, not randomized for training.
/// Non-UTF-8 path identity bytes are stable on the same platform only.
///
/// Rejects empty corpora, duplicate IDs/source records, invalid relative IDs/nonabsolute
/// source paths, documents with fewer than two tokens, invalid context lengths,
/// non-finite/out-of-range ratios, and counts that leave an empty partition.
/// For positive validation, at least two contributing documents are required.
/// Zero ratio/count and `None` explicitly produce an empty validation partition.
/// Distinct valid JSONL record IDs can share a source path and land in separate
/// partitions. Other duplicate source paths are rejected. `set.files` lists each
/// contributing physical file once; `documents` retains every logical document.
pub fn split_document_corpus(
    corpus: &DocumentCorpus,
    context_length: usize,
    policy: ValidationSplit,
    seed: u64,
) -> Result<DatasetSplit, String> {
    let sequence_length = validate_context_length(context_length)?;
    let count = corpus.documents.len();
    if count == 0 {
        return Err("Cannot split an empty document corpus".into());
    }
    let validation_count = validation_document_count(count, policy)?;

    let mut identities = BTreeSet::new();
    let mut paths: BTreeMap<&PathBuf, BTreeSet<Option<usize>>> = BTreeMap::new();
    for document in &corpus.documents {
        if document.id.as_os_str().is_empty()
            || document
                .id
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            || !document.path.is_absolute()
        {
            return Err(format!(
                "Document {:?} requires a nonempty relative identity without traversal and an absolute source path",
                document.id
            ));
        }
        let record = document_record_number(&document.id, &document.path)?;
        let records = paths.entry(&document.path).or_default();
        if !identities.insert(&document.id)
            || (!records.is_empty() && (record.is_none() || records.contains(&None)))
            || !records.insert(record)
        {
            return Err(format!(
                "Duplicate document identity or path: {:?}",
                document.id
            ));
        }
        if document.tokens.len() < 2 {
            return Err(format!(
                "Document {:?} must contain at least two tokens",
                document.id
            ));
        }
    }
    let mut ranked: Vec<_> = corpus.documents.iter().collect();
    ranked.sort_by_cached_key(|doc| (document_rank(&doc.id, seed), &doc.id));
    let validation_ids: BTreeSet<_> = ranked[..validation_count]
        .iter()
        .map(|doc| &doc.id)
        .collect();
    let mut documents: Vec<_> = corpus.documents.iter().collect();
    documents.sort_by_key(|doc| &doc.id);
    let mut result = DatasetSplit {
        training: empty_partition(),
        validation: empty_partition(),
    };
    for document in documents {
        let partition = if validation_ids.contains(&document.id) {
            &mut result.validation
        } else {
            &mut result.training
        };
        let tokens = &document.tokens;
        partition.set.token_count = partition
            .set
            .token_count
            .checked_add(tokens.len())
            .ok_or_else(|| "Dataset token count overflows usize".to_string())?;
        let first_example = partition.set.examples.len();
        let mut start = 0;
        while tokens.len() - start >= 2 {
            let length = sequence_length.min(tokens.len() - start);
            let end = start
                .checked_add(length)
                .ok_or_else(|| "Training sequence boundary overflows usize".to_string())?;
            partition.set.examples.push(tokens[start..end].to_vec());
            if end == tokens.len() {
                break;
            }
            start = start
                .checked_add(context_length)
                .ok_or_else(|| "Training sequence step overflows usize".to_string())?;
        }
        partition.set.files.push(document.path.clone());
        partition.documents.push(DocumentInfo {
            id: document.id.clone(),
            path: document.path.clone(),
            example_range: first_example..partition.set.examples.len(),
        });
    }
    for partition in [&mut result.training, &mut result.validation] {
        partition.set.files.sort();
        partition.set.files.dedup();
    }
    Ok(result)
}

pub(crate) fn validation_document_count(
    count: usize,
    policy: ValidationSplit,
) -> Result<usize, String> {
    if count == 0 {
        return Err("Cannot split an empty document corpus".into());
    }
    let validation_count = match policy {
        ValidationSplit::None => 0,
        ValidationSplit::Count(count) => count,
        ValidationSplit::Ratio(ratio) => {
            if !ratio.is_finite() || !(0.0..1.0).contains(&ratio) {
                return Err("Validation ratio must be finite and in [0, 1)".into());
            }
            let requested = (count as f64 * ratio).floor() as usize;
            if ratio > 0.0 && requested == 0 {
                return Err(format!(
                    "Validation ratio {ratio} yields zero validation documents from {count}; increase the ratio/count or explicitly disable validation"
                ));
            }
            requested
        }
    };
    if validation_count >= count {
        return Err(format!(
            "Validation count {validation_count} must be smaller than the {count} contributing documents to leave training nonempty"
        ));
    }

    Ok(validation_count)
}

fn empty_partition() -> DocumentPartition {
    DocumentPartition {
        set: TrainingSet {
            examples: Vec::new(),
            files: Vec::new(),
            token_count: 0,
        },
        documents: Vec::new(),
    }
}

pub(crate) fn document_rank(id: &Path, seed: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    // Component separators are fixed, so Windows and Unix UTF-8 paths agree.
    for component in id.components() {
        for byte in component.as_os_str().as_encoded_bytes().iter().chain(b"/") {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }
    let mut rank = (hash ^ seed).wrapping_add(0x9e3779b97f4a7c15);
    rank = (rank ^ (rank >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    rank = (rank ^ (rank >> 27)).wrapping_mul(0x94d049bb133111eb);
    rank ^ (rank >> 31)
}

fn dataset_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_type().is_symlink() || metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

fn require_directory_chain(path: &Path) -> Result<(), String> {
    for ancestor in path.ancestors().filter(|p| !p.as_os_str().is_empty()) {
        let metadata = fs::symlink_metadata(ancestor)
            .map_err(|e| format!("Cannot inspect directory {}: {e}", ancestor.display()))?;
        if dataset_link(&metadata) {
            return Err(format!("Symlinks are not allowed: {}", ancestor.display()));
        }
        if !metadata.is_dir() {
            return Err(format!(
                "Expected a dataset directory: {}",
                ancestor.display()
            ));
        }
    }
    Ok(())
}

fn collect_dataset_files(
    path: &Path,
    files: &mut BTreeSet<PathBuf>,
    extension: &str,
    remaining: &mut usize,
) -> Result<(), String> {
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        crate::release::reject_release_parent(&directory)?;
        let entries = fs::read_dir(&directory)
            .map_err(|e| format!("Cannot read directory {}: {e}", directory.display()))?;
        for entry in entries {
            *remaining = remaining
                .checked_sub(1)
                .ok_or("Dataset discovery exceeds directory entry limit")?;
            let entry = entry
                .map_err(|e| format!("Cannot read directory entry: {e}"))?
                .path();
            let metadata = fs::symlink_metadata(&entry)
                .map_err(|e| format!("Cannot inspect {}: {e}", entry.display()))?;
            if dataset_link(&metadata) {
                return Err(format!("Symlinks are not allowed: {}", entry.display()));
            }
            if metadata.is_dir() {
                pending.push(entry);
            } else if metadata.is_file() && entry.extension().is_some_and(|ext| ext == extension) {
                files.insert(entry);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tokenizer() -> omega_tokenizer::Tokens {
        omega_tokenizer::Tokens::new(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"),
        )
        .expect("load fixture tokenizer")
    }

    fn root() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    fn put(root: &Path, name: &str, text: &str) {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn build(root: &Path, folders: &[&str], context: usize) -> Result<TrainingSet, String> {
        build_training_set(
            root,
            &folders.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &tokenizer(),
            context,
        )
    }

    #[test]
    fn multiple_folders_are_sorted_and_overlaps_deduplicated() {
        let temp = root();
        put(temp.path(), "b/z.txt", "omega tokenizer");
        put(temp.path(), "a/nested/y.txt", "this is a test");
        put(temp.path(), "a/x.txt", "hello world");
        put(temp.path(), "a/blank.txt", " \n\t");
        put(temp.path(), "a/ignored.json", "not parsed as text");
        let first = build(temp.path(), &["b", "a/nested", "a", "a"], 8).unwrap();
        let second = build(temp.path(), &["a", "b"], 8).unwrap();
        assert_eq!(first.files, second.files);
        assert_eq!(first.examples, second.examples);
        assert_eq!(
            first.examples,
            vec![vec![6, 7, 8, 1], vec![2, 3], vec![4, 5]]
        );
        assert_eq!(first.token_count, 8);
        assert_eq!(first.files.len(), 3);
        assert!(first.files.windows(2).all(|p| p[0] < p[1]));
        assert!(first.files.iter().all(|p| p.is_absolute()));
    }

    #[test]
    fn case_alias_selections_follow_filesystem_identity_without_split_leakage() {
        let temp = root();
        put(temp.path(), "a/file.txt", "hello world");
        // Probe this directory rather than assuming every Windows volume is
        // insensitive or every Unix volume is sensitive.
        let case_sensitive = match fs::create_dir(temp.path().join("A")) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(error) => panic!("Cannot probe filesystem case behavior: {error}"),
        };
        if case_sensitive {
            put(temp.path(), "A/file.txt", "omega tokenizer");
        }
        let selected = corpus(temp.path(), &["a", "A", "a"]);
        let reversed = corpus(temp.path(), &["A", "a"]);
        let ids: Vec<_> = selected.documents.iter().map(|doc| &doc.id).collect();
        assert_eq!(
            ids,
            reversed
                .documents
                .iter()
                .map(|doc| &doc.id)
                .collect::<Vec<_>>()
        );
        assert_eq!(selected.documents.len(), if case_sensitive { 2 } else { 1 });
        let single = corpus(temp.path(), &["a"]);
        assert!(split_document_corpus(&single, 2, ValidationSplit::Count(1), 42).is_err());
        if case_sensitive {
            assert_eq!(ids, vec![Path::new("A/file.txt"), Path::new("a/file.txt")]);
            let split = split_document_corpus(&selected, 2, ValidationSplit::Count(1), 42).unwrap();
            assert_ne!(
                split.training.documents[0].id,
                split.validation.documents[0].id
            );
            assert_ne!(split.training.set.examples, split.validation.set.examples);
        } else {
            assert_eq!(selected.documents[0].id, single.documents[0].id);
            assert_eq!(selected.documents[0].path, single.documents[0].path);
            assert!(
                split_document_corpus(&selected, 2, ValidationSplit::Count(1), 42)
                    .unwrap_err()
                    .contains("leave training nonempty")
            );
            let all_training = build(temp.path(), &["a", "A"], 2).unwrap();
            assert_eq!(all_training.files.len(), 1);
            assert_eq!(all_training.examples, vec![vec![2, 3]]);
            assert_eq!(all_training.token_count, 2);
        }
    }

    #[test]
    fn chunks_cover_each_adjacent_pair_once_including_short_remainders() {
        let temp = root();
        let words = [
            "hello",
            "world",
            "omega",
            "tokenizer",
            "this",
            "is",
            "a",
            "test",
        ];
        let tokenizer = tokenizer();
        for count in 2..=words.len() {
            let text = words[..count].join(" ");
            put(temp.path(), "a/document.txt", &text);
            let tokens = omega_nn::encode_text(&tokenizer, &text).unwrap();
            for context in [1, 2, 3, 7, 8, usize::MAX - 1] {
                let set = build(temp.path(), &["a"], context).unwrap();
                let pairs: Vec<_> = set
                    .examples
                    .iter()
                    .flat_map(|chunk| chunk.windows(2).map(|p| (p[0], p[1])))
                    .collect();
                let expected: Vec<_> = tokens.windows(2).map(|p| (p[0], p[1])).collect();
                assert_eq!(pairs, expected);
                assert_eq!(set.token_count, count);
                assert!(
                    set.examples
                        .iter()
                        .all(|c| c.len() >= 2 && c.len() <= context + 1)
                );
            }
        }
        put(
            temp.path(),
            "a/document.txt",
            "hello world omega tokenizer this is",
        );
        assert_eq!(
            build(temp.path(), &["a"], 4).unwrap().examples,
            vec![vec![2, 3, 4, 5, 6], vec![6, 7]]
        );
    }

    #[test]
    fn documents_never_share_targets() {
        let temp = root();
        put(temp.path(), "a/first.txt", "hello world");
        put(temp.path(), "a/second.txt", "omega tokenizer");
        let set = build(temp.path(), &["a"], 100).unwrap();
        assert_eq!(set.examples, vec![vec![2, 3], vec![4, 5]]);
        assert_eq!(set.token_count, 4);
    }

    fn corpus(root: &Path, folders: &[&str]) -> DocumentCorpus {
        load_document_corpus(
            root,
            &folders.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &tokenizer(),
        )
        .unwrap()
    }

    fn validation_ids(split: &DatasetSplit) -> Vec<PathBuf> {
        split
            .validation
            .documents
            .iter()
            .map(|d| d.id.clone())
            .collect()
    }

    #[test]
    fn split_is_seeded_order_independent_and_stable_when_relocated() {
        let first_root = root();
        let second_root = root();
        for index in 0..12 {
            let name = format!("a/nested/doc{index:02}.txt");
            for root in [first_root.path(), second_root.path()] {
                put(root, &name, "hello world omega tokenizer this is");
            }
        }
        put(first_root.path(), "a/blank.txt", " \n");
        let first = corpus(first_root.path(), &["a/nested", "a", "a"]);
        let mut second = corpus(first_root.path(), &["a", "a/nested"]);
        second.documents.reverse();
        let relocated = corpus(second_root.path(), &["a"]);
        let split = split_document_corpus(&first, 4, ValidationSplit::Count(4), 42).unwrap();
        let ids = validation_ids(&split);
        for corpus in [&second, &relocated] {
            let other =
                split_document_corpus(corpus, 1, ValidationSplit::Ratio(1.0 / 3.0), 42).unwrap();
            assert_eq!(validation_ids(&other), ids);
        }
        assert_eq!(split.training.documents.len(), 8);
        assert_eq!(split.validation.documents.len(), 4);
        assert!(
            split
                .training
                .documents
                .iter()
                .all(|doc| !ids.contains(&doc.id))
        );
        assert!(ids.iter().all(|id| id.is_relative()));
        assert!((0..10).any(|seed| validation_ids(
            &split_document_corpus(&first, 4, ValidationSplit::Count(4), seed).unwrap()
        ) != ids));
    }

    #[test]
    fn split_preserves_every_document_pair_and_chunk_provenance() {
        let temp = root();
        put(
            temp.path(),
            "a/first.txt",
            "hello world omega tokenizer this is",
        );
        put(temp.path(), "a/second.txt", "this is a test");
        put(temp.path(), "a/third.txt", "hello omega");
        let corpus = corpus(temp.path(), &["a"]);
        for context in [1, 2, 4, 100] {
            let split =
                split_document_corpus(&corpus, context, ValidationSplit::Count(1), 9).unwrap();
            let mut seen = BTreeSet::new();
            let mut total_targets = 0;
            let mut total_tokens = 0;
            for partition in [&split.training, &split.validation] {
                assert_eq!(partition.documents.len(), partition.set.files.len());
                let mut next_example = 0;
                let mut partition_tokens = 0;
                for (index, doc) in partition.documents.iter().enumerate() {
                    assert!(seen.insert(&doc.id));
                    assert_eq!(doc.path, partition.set.files[index]);
                    assert_eq!(doc.example_range.start, next_example);
                    let source = corpus
                        .documents
                        .iter()
                        .find(|source| source.id == doc.id)
                        .unwrap();
                    let actual: Vec<_> = partition.set.examples[doc.example_range.clone()]
                        .iter()
                        .flat_map(|chunk| chunk.windows(2).map(|p| (p[0], p[1])))
                        .collect();
                    let expected: Vec<_> = source.tokens.windows(2).map(|p| (p[0], p[1])).collect();
                    assert_eq!(actual, expected);
                    total_targets += actual.len();
                    partition_tokens += source.tokens.len();
                    next_example = doc.example_range.end;
                }
                assert_eq!(next_example, partition.set.examples.len());
                assert_eq!(partition_tokens, partition.set.token_count);
                total_tokens += partition_tokens;
            }
            assert_eq!(seen.len(), corpus.documents.len());
            assert_eq!(total_tokens, 12);
            assert_eq!(total_targets, 9);
        }
    }

    #[test]
    fn zero_validation_preserves_all_training_wrapper() {
        let temp = root();
        put(
            temp.path(),
            "a/document.txt",
            "hello world omega tokenizer this is",
        );
        let corpus = corpus(temp.path(), &["a"]);
        let previous = build(temp.path(), &["a"], 4).unwrap();
        for policy in [
            ValidationSplit::None,
            ValidationSplit::Count(0),
            ValidationSplit::Ratio(0.0),
        ] {
            let split = split_document_corpus(&corpus, 4, policy, 17).unwrap();
            assert_eq!(split.training.set.examples, previous.examples);
            assert_eq!(split.training.set.files, previous.files);
            assert_eq!(split.training.set.token_count, previous.token_count);
            assert!(split.validation.set.examples.is_empty());
            assert!(split.validation.set.files.is_empty());
            assert!(split.validation.documents.is_empty());
            assert_eq!(split.validation.set.token_count, 0);
        }
    }

    #[test]
    fn rejects_invalid_split_policies_and_empty_partitions() {
        let temp = root();
        put(temp.path(), "a/first.txt", "hello world");
        let one = corpus(temp.path(), &["a"]);
        assert!(
            split_document_corpus(&one, 2, ValidationSplit::Count(1), 0)
                .unwrap_err()
                .contains("leave training nonempty")
        );
        assert!(
            split_document_corpus(&one, 2, ValidationSplit::Ratio(0.5), 0)
                .unwrap_err()
                .contains("zero validation")
        );
        put(temp.path(), "a/second.txt", "omega tokenizer");
        let two = corpus(temp.path(), &["a"]);
        for ratio in [-0.1, 1.0, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                split_document_corpus(&two, 2, ValidationSplit::Ratio(ratio), 0)
                    .unwrap_err()
                    .contains("Validation ratio")
            );
        }
        for count in [2, 3, usize::MAX] {
            assert!(split_document_corpus(&two, 2, ValidationSplit::Count(count), 0).is_err());
        }
        assert!(split_document_corpus(&two, 2, ValidationSplit::Ratio(0.1), 0).is_err());
        assert!(split_document_corpus(&two, 2, ValidationSplit::Ratio(0.5), 0).is_ok());
        for context in [0, usize::MAX] {
            assert!(split_document_corpus(&two, context, ValidationSplit::None, 0).is_err());
        }
        assert!(
            split_document_corpus(
                &DocumentCorpus { documents: vec![] },
                2,
                ValidationSplit::None,
                0
            )
            .unwrap_err()
            .contains("empty document corpus")
        );
    }

    #[test]
    fn rejects_invalid_corpus_document_identity_and_tokens() {
        let temp = root();
        put(temp.path(), "a/first.txt", "hello world");
        let original = corpus(temp.path(), &["a"]);
        let check = |corpus: &DocumentCorpus| {
            split_document_corpus(corpus, 2, ValidationSplit::None, 0).unwrap_err()
        };
        let mut duplicate = original.clone();
        duplicate.documents.push(duplicate.documents[0].clone());
        assert!(check(&duplicate).contains("Duplicate"));
        duplicate.documents[1].id = PathBuf::from("another.txt");
        assert!(check(&duplicate).contains("Duplicate"));
        for id in [
            PathBuf::new(),
            PathBuf::from("../a"),
            temp.path().join("absolute"),
        ] {
            let mut invalid = original.clone();
            invalid.documents[0].id = id;
            assert!(check(&invalid).contains("relative identity"));
        }
        let mut invalid = original.clone();
        invalid.documents[0].path = PathBuf::from("relative.txt");
        assert!(check(&invalid).contains("absolute source path"));
        for tokens in [vec![], vec![1]] {
            let mut invalid = original.clone();
            invalid.documents[0].tokens = tokens;
            assert!(check(&invalid).contains("at least two tokens"));
        }
    }

    #[test]
    fn document_ranking_has_a_fixed_algorithm() {
        // Freeze the ranking implementation so dependency/RNG changes cannot
        // silently alter existing seeded split membership.
        assert_eq!(
            document_rank(Path::new("a/first.txt"), 42),
            12_343_880_375_399_922_854
        );
    }

    #[test]
    fn rejects_invalid_selections_and_contexts() {
        let temp = root();
        put(temp.path(), "a/file.txt", "hello world");
        for folder in [
            "",
            ".",
            "..",
            "../a",
            "a/../a",
            "/a",
            "C:\\a",
            "a\\..\\b",
            "missing",
            "a/file.txt",
        ] {
            assert!(build(temp.path(), &[folder], 2).is_err(), "{folder:?}");
        }
        let absolute = temp.path().join("a").to_string_lossy().into_owned();
        assert!(build(temp.path(), &[&absolute], 2).is_err());
        assert!(build(temp.path(), &[], 2).is_err());
        assert!(build(temp.path(), &["a"], 0).is_err());
        assert!(build(temp.path(), &["a"], usize::MAX).is_err());
    }

    #[test]
    fn rejects_empty_json_only_blank_and_single_token_datasets() {
        let temp = root();
        fs::create_dir(temp.path().join("empty")).unwrap();
        put(temp.path(), "json/data.json", "[\"hello world\"]");
        put(temp.path(), "blank/file.txt", "\n \t");
        put(temp.path(), "single/file.txt", "hello");
        put(temp.path(), "valid/file.txt", "hello world");
        for folder in ["empty", "json"] {
            let error = build(temp.path(), &["valid", folder], 2).unwrap_err();
            assert!(error.contains("no .txt files"));
            assert!(error.contains("JSON is not supported"));
        }
        assert!(
            build(temp.path(), &["blank"], 2)
                .unwrap_err()
                .contains("No training examples")
        );
        assert!(
            build(temp.path(), &["single"], 2)
                .unwrap_err()
                .contains("at least two tokens")
        );
    }

    #[test]
    fn rejects_invalid_utf8() {
        let temp = root();
        put(temp.path(), "a/file.txt", "hello world");
        fs::write(temp.path().join("a/file.txt"), [0xff, 0xfe]).unwrap();
        assert!(build(temp.path(), &["a"], 2).unwrap_err().contains("UTF-8"));
    }

    fn load_jsonl(root: &Path, folders: &[&str]) -> Result<DocumentCorpus, String> {
        load_document_corpus_with_format(
            root,
            &folders.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &tokenizer(),
            DatasetFormat::Jsonl,
        )
    }

    #[test]
    fn raw_documents_preserve_text_and_jsonl_record_identity() {
        let temp = root();
        put(temp.path(), "a/document.txt", " Hello world\n");
        put(temp.path(), "a/blank.txt", " \n");
        put(
            temp.path(),
            "a/records.jsonl",
            "\n{\"text\":\" Hello world\\n\",\"metadata\":7}\n{\"text\":\" \"}\n\n{\"text\":\"omega\"}\n",
        );
        let text = load_text_documents(temp.path(), &["a".into()], DatasetFormat::Text).unwrap();
        assert_eq!(text.len(), 1);
        assert_eq!(text[0].text, " Hello world\n");
        assert_eq!(text[0].id, Path::new("a/document.txt"));
        let jsonl =
            load_text_documents(temp.path(), &["a".into(), "a".into()], DatasetFormat::Jsonl)
                .unwrap();
        assert_eq!(jsonl.len(), 2);
        assert_eq!(jsonl[0].id, Path::new("a/records.jsonl/@record-2"));
        assert_eq!(jsonl[1].id, Path::new("a/records.jsonl/@record-5"));
        assert_eq!(jsonl[0].text, " Hello world\n");
        assert_eq!(jsonl[1].text, "omega");
        assert_eq!(jsonl[0].path, jsonl[1].path);
        assert_eq!(
            jsonl[0].path,
            fs::canonicalize(temp.path().join("a/records.jsonl")).unwrap()
        );
        assert!(
            load_jsonl(temp.path(), &["a"])
                .unwrap_err()
                .contains("records.jsonl:5")
        );
        // Old APIs continue to load only text, with exactly the previous output.
        let old = build(temp.path(), &["a"], 2).unwrap();
        assert_eq!(old.examples, vec![vec![2, 3]]);
    }

    #[test]
    fn jsonl_records_split_before_chunking_preserving_all_local_pairs() {
        let temp = root();
        put(
            temp.path(),
            "a/nested/records.jsonl",
            "{\"text\":\"hello world omega tokenizer this is\"}\n{\"text\":\"omega tokenizer\"}\n{\"text\":\"this is a test\"}\n",
        );
        let corpus = load_jsonl(temp.path(), &["a", "a/nested", "a"]).unwrap();
        let reversed = load_jsonl(temp.path(), &["a/nested", "a"]).unwrap();
        assert_eq!(corpus.documents.len(), 3);
        for context in [1, 2, 4, 10] {
            let split =
                split_document_corpus(&corpus, context, ValidationSplit::Count(1), 42).unwrap();
            let other =
                split_document_corpus(&reversed, context, ValidationSplit::Count(1), 42).unwrap();
            assert_eq!(validation_ids(&split), validation_ids(&other));
            let mut seen = BTreeSet::new();
            let mut targets = 0;
            for partition in [&split.training, &split.validation] {
                assert_eq!(partition.set.files.len(), 1);
                assert_eq!(partition.set.files[0], corpus.documents[0].path);
                for doc in &partition.documents {
                    assert!(seen.insert(&doc.id));
                    let original = corpus
                        .documents
                        .iter()
                        .find(|source| source.id == doc.id)
                        .unwrap();
                    let actual: Vec<_> = partition.set.examples[doc.example_range.clone()]
                        .iter()
                        .flat_map(|chunk| chunk.windows(2).map(|pair| (pair[0], pair[1])))
                        .collect();
                    let expected: Vec<_> = original
                        .tokens
                        .windows(2)
                        .map(|pair| (pair[0], pair[1]))
                        .collect();
                    assert_eq!(actual, expected);
                    targets += actual.len();
                }
            }
            assert_eq!(targets, 9);
            assert_eq!(seen.len(), 3);
            let all = split_document_corpus(&corpus, context, ValidationSplit::None, 0).unwrap();
            assert_eq!(all.training.set.files.len(), 1);
            assert_eq!(all.training.documents.len(), 3);
            assert_eq!(all.training.set.token_count, 12);
        }
    }

    #[test]
    fn jsonl_errors_identify_physical_line_and_validate_schema() {
        let temp = root();
        for record in [
            "{broken",
            "{}",
            "{\"text\":null}",
            "{\"text\":4}",
            "[\"hello world\"]",
            "\"hello world\"",
            "true",
            "{\"text\":\"hello world\"} trailing",
        ] {
            put(
                temp.path(),
                "a/records.jsonl",
                &format!("\n{{\"text\":\" \"}}\n{record}\n"),
            );
            let error = load_jsonl(temp.path(), &["a"]).unwrap_err();
            assert!(error.contains("records.jsonl:3"), "{error}");
            assert!(error.contains("string text field"), "{error}");
        }
        put(temp.path(), "a/records.jsonl", "\n{\"text\":\" \"}\n");
        assert!(
            load_jsonl(temp.path(), &["a"])
                .unwrap_err()
                .contains("documents are blank")
        );
        fs::write(temp.path().join("a/records.jsonl"), [0xff]).unwrap();
        let error = load_jsonl(temp.path(), &["a"]).unwrap_err();
        assert!(error.contains("UTF-8") && error.contains("records.jsonl"));
        put(temp.path(), "text/only.txt", "hello world");
        assert!(
            load_jsonl(temp.path(), &["text"])
                .unwrap_err()
                .contains("no .jsonl files")
        );
        put(
            temp.path(),
            "upper/only.JSONL",
            "{\"text\":\"hello world\"}",
        );
        assert!(
            load_jsonl(temp.path(), &["upper"])
                .unwrap_err()
                .contains("no .jsonl files")
        );
    }

    #[test]
    fn jsonl_selection_safety_and_case_alias_deduplication_are_preserved() {
        let temp = root();
        put(temp.path(), "a/records.jsonl", "{\"text\":\"hello world\"}");
        for folder in ["../a", "a/../a", "/a", "a\\..\\b", "a/records.jsonl"] {
            assert!(load_jsonl(temp.path(), &[folder]).is_err());
        }
        if temp.path().join("A").is_dir() {
            let corpus = load_jsonl(temp.path(), &["a", "A"]).unwrap();
            assert_eq!(corpus.documents.len(), 1);
            assert!(split_document_corpus(&corpus, 2, ValidationSplit::Count(1), 0).is_err());
        }
    }

    #[test]
    fn jsonl_duplicate_source_records_and_forged_ids_are_rejected() {
        let temp = root();
        put(
            temp.path(),
            "a/records.jsonl",
            "{\"text\":\"hello world\"}\n{\"text\":\"omega tokenizer\"}\n",
        );
        let original = load_jsonl(temp.path(), &["a"]).unwrap();
        for id in [
            "a/records.jsonl/@record-0",
            "a/records.jsonl/@record-01",
            "a/other.jsonl/@record-1",
            "a/records.jsonl/first",
            "a/records.jsonl",
        ] {
            let mut invalid = original.clone();
            invalid.documents[0].id = PathBuf::from(id);
            assert!(
                split_document_corpus(&invalid, 2, ValidationSplit::None, 0)
                    .unwrap_err()
                    .contains("Invalid JSONL document identity")
            );
        }
        let mut duplicate = original;
        // A suffix alias remains the same physical record, even with another ID.
        duplicate.documents[1].id = PathBuf::from("records.jsonl/@record-1");
        assert!(
            split_document_corpus(&duplicate, 2, ValidationSplit::None, 0)
                .unwrap_err()
                .contains("Duplicate")
        );
    }

    #[cfg(unix)]
    #[test]
    fn jsonl_rejects_symlinks_before_canonicalization() {
        use std::os::unix::fs::symlink;
        let temp = root();
        put(temp.path(), "a/records.jsonl", "{\"text\":\"hello world\"}");
        symlink(temp.path().join("a"), temp.path().join("alias")).unwrap();
        assert!(
            load_jsonl(temp.path(), &["alias"])
                .unwrap_err()
                .contains("Symlinks")
        );
        symlink(
            temp.path().join("a/records.jsonl"),
            temp.path().join("a/copy.jsonl"),
        )
        .unwrap();
        assert!(
            load_jsonl(temp.path(), &["a"])
                .unwrap_err()
                .contains("Symlinks")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_selected_recursive_and_root_symlinks() {
        use std::os::unix::fs::symlink;
        let temp = root();
        put(temp.path(), "a/file.txt", "hello world");
        symlink(temp.path().join("a"), temp.path().join("selected")).unwrap();
        assert!(
            build(temp.path(), &["selected"], 2)
                .unwrap_err()
                .contains("Symlinks")
        );
        assert!(
            build(&temp.path().join("selected"), &["child"], 2)
                .unwrap_err()
                .contains("Symlinks")
        );
        symlink(
            temp.path().join("missing"),
            temp.path().join("a/ignored.json"),
        )
        .unwrap();
        assert!(
            build(temp.path(), &["a"], 2)
                .unwrap_err()
                .contains("Symlinks")
        );
        fs::remove_file(temp.path().join("a/ignored.json")).unwrap();
        symlink(temp.path().join("a"), temp.path().join("a/cycle")).unwrap();
        assert!(
            build(temp.path(), &["a"], 2)
                .unwrap_err()
                .contains("Symlinks")
        );
    }
}
