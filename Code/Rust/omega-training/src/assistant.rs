//! Explicit conversation data and the persisted Omega chat protocol.
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use omega_tokenizer::{
    Tokens,
    chat::{CHAT_PROTOCOL_VERSION, ChatMessage, ChatProtocol, ChatRole},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    checkpoint::{DatasetProvenance, DocumentFingerprint},
    dataset::{
        ASSISTANT_TARGETS_OBJECTIVE, DatasetFormat, DocumentInfo, ExampleSource, ValidationSplit,
        discover_dataset_files, document_rank, validation_document_count,
    },
};

/// Schema 2 checkpoint metadata. Full tokenizer identity is separately required.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatArtifact {
    pub protocol: String,
    pub token_ids: [u32; 4],
}

impl ChatArtifact {
    pub fn from_tokenizer(tokenizer: &Tokens) -> Result<Option<Self>, String> {
        if !ChatProtocol::has_registered_controls(tokenizer) {
            return Ok(None);
        }
        let protocol = ChatProtocol::from_tokenizer(tokenizer)?;
        Ok(Some(Self {
            protocol: protocol.version().into(),
            token_ids: protocol.token_ids().as_array(),
        }))
    }

    pub fn validate(&self, vocabulary: usize) -> Result<(), String> {
        if self.protocol != CHAT_PROTOCOL_VERSION
            || self.token_ids.iter().any(|&id| id as usize >= vocabulary)
            || self.token_ids.iter().collect::<BTreeSet<_>>().len() != 4
        {
            return Err("Invalid or unsupported checkpoint chat protocol/IDs".into());
        }
        Ok(())
    }

    pub fn validate_tokenizer(&self, tokenizer: &Tokens) -> Result<ChatProtocol, String> {
        if Self::from_tokenizer(tokenizer)?.as_ref() != Some(self) {
            return Err("Checkpoint chat protocol does not match its tokenizer".into());
        }
        ChatProtocol::from_tokenizer(tokenizer)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageRecord {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversationRecord {
    schema_version: u32,
    messages: Vec<MessageRecord>,
}

/// Parse the explicit conversation schema; text-field JSONL remains separate.
pub fn parse_messages(json: &str) -> Result<Vec<ChatMessage>, String> {
    let record: ConversationRecord =
        serde_json::from_str(json).map_err(|e| format!("Invalid conversation JSON: {e}"))?;
    if record.schema_version != 1 {
        return Err("Unsupported conversation schema_version".into());
    }
    record
        .messages
        .into_iter()
        .map(|m| {
            Ok(ChatMessage {
                role: match m.role.as_str() {
                    "system" => ChatRole::System,
                    "user" => ChatRole::User,
                    "assistant" => ChatRole::Assistant,
                    _ => return Err(format!("Unsupported conversation role {}", m.role)),
                },
                content: m.content,
            })
        })
        .collect()
}

#[derive(Clone, Debug)]
struct ConversationExample {
    id: String,
    ids: Vec<u32>,
    mask: Vec<bool>,
}

/// Eager, bounded conversation examples; no token-only cache can represent these masks.
#[derive(Clone, Debug, Default)]
pub struct ConversationSet {
    examples: Vec<ConversationExample>,
}

impl ExampleSource for ConversationSet {
    fn objective(&self) -> &str {
        ASSISTANT_TARGETS_OBJECTIVE
    }
    fn example_count(&self) -> usize {
        self.examples.len()
    }
    fn target_count(&self) -> Result<usize, String> {
        self.examples.iter().try_fold(0usize, |n, e| {
            n.checked_add(e.mask.iter().filter(|&&v| v).count())
                .ok_or_else(|| "Conversation target count overflow".into())
        })
    }
    fn example(&self, index: usize) -> Result<Vec<u32>, String> {
        self.examples
            .get(index)
            .map(|e| e.ids.clone())
            .ok_or_else(|| format!("Missing conversation {index}"))
    }
    fn target_mask(&self, index: usize) -> Result<Option<Vec<bool>>, String> {
        self.examples
            .get(index)
            .map(|e| Some(e.mask.clone()))
            .ok_or_else(|| format!("Missing conversation {index}"))
    }
    fn identity(&self) -> Result<String, String> {
        let mut hash = Sha256::new();
        hash.update(b"omega-conversation-source-v1\0");
        hash.update(CHAT_PROTOCOL_VERSION.as_bytes());
        hash.update(ASSISTANT_TARGETS_OBJECTIVE.as_bytes());
        hash.update((self.examples.len() as u64).to_le_bytes());
        for example in &self.examples {
            hash.update((example.id.len() as u64).to_le_bytes());
            hash.update(example.id.as_bytes());
            hash.update((example.ids.len() as u64).to_le_bytes());
            for id in &example.ids {
                hash.update(id.to_le_bytes());
            }
            hash.update((example.mask.len() as u64).to_le_bytes());
            for value in &example.mask {
                hash.update([u8::from(*value)]);
            }
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

pub struct PreparedConversations {
    pub training: ConversationSet,
    pub validation: ConversationSet,
    pub provenance: DatasetProvenance,
    pub training_documents: Vec<DocumentInfo>,
    pub training_files: usize,
    pub training_tokens: usize,
}

/// Load prepartitioned whole records; reject automatic splits and over-context input.
/// Source files <=16 MiB, aggregate raw bytes <=64 MiB, <=100,000 records.
/// These are eager input caps, not an aggregate process-memory guarantee.
pub fn prepare_conversations(
    root: &Path,
    selections: &[String],
    tokenizer: &Tokens,
    context: usize,
    policy: ValidationSplit,
    seed: u64,
) -> Result<PreparedConversations, String> {
    if !matches!(policy, ValidationSplit::None | ValidationSplit::Count(0))
        && !matches!(policy, ValidationSplit::Ratio(ratio) if ratio == 0.0)
    {
        return Err("Chat data requires reviewed source-group partitions; select explicit training/validation folders and evaluate separately, without automatic record splitting".into());
    }
    if context == 0 {
        return Err("Conversation context must be positive".into());
    }
    let protocol = ChatProtocol::from_tokenizer(tokenizer)?;
    let (root, paths) = discover_dataset_files(root, selections, DatasetFormat::Jsonl, 200_000)?;
    let mut records = Vec::new();
    let mut source_bytes = 0u64;
    for path in paths {
        let size = fs::metadata(&path).map_err(|e| e.to_string())?.len();
        source_bytes = source_bytes
            .checked_add(size)
            .ok_or("Conversation source size overflow")?;
        if size > 16 * 1024 * 1024 || source_bytes > 64 * 1024 * 1024 {
            return Err("Conversation input exceeds 16 MiB/file or 64 MiB aggregate eager limit; prepare a smaller release".into());
        }
        let mut raw = String::new();
        fs::File::open(&path)
            .map_err(|e| format!("Cannot open {}: {e}", path.display()))?
            .take(16 * 1024 * 1024 + 1)
            .read_to_string(&mut raw)
            .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
        if raw.len() as u64 != size {
            return Err("Conversation source changed while reading".into());
        }
        let relative = path.strip_prefix(&root).map_err(|e| e.to_string())?;
        let relative = relative
            .to_str()
            .ok_or("Conversation provenance requires UTF-8 paths")?
            .replace('\\', "/");
        for (line, json) in raw.lines().enumerate() {
            if json.trim().is_empty() {
                continue;
            }
            if records.len() >= 100_000 {
                return Err("Conversation record limit exceeded".into());
            }
            let id = format!("{relative}/@record-{}", line + 1);
            let messages = parse_messages(json).map_err(|e| format!("{id}: {e}"))?;
            let encoded = protocol
                .encode_conversation(tokenizer, &messages)
                .map_err(|e| format!("{id}: {e}"))?;
            if encoded.ids.len() < 2 || encoded.ids.len() - 1 > context {
                return Err(format!(
                    "{id}: conversation needs {} input positions; context is {context}; shorten the conversation explicitly",
                    encoded.ids.len().saturating_sub(1)
                ));
            }
            let example = ConversationExample {
                id,
                ids: encoded.ids,
                mask: encoded.target_mask,
            };
            records.push((path.clone(), example));
        }
    }
    if records.is_empty() {
        return Err("No conversation records found".into());
    }
    let validation_count = validation_document_count(records.len(), policy)?;
    let mut ranking: Vec<_> = records.iter().map(|(_, r)| &r.id).collect();
    ranking.sort_by_key(|id| (document_rank(Path::new(id), seed), *id));
    let validation_ids: BTreeSet<_> = ranking[..validation_count]
        .iter()
        .map(|id| (*id).clone())
        .collect();
    records.sort_by(|a, b| a.1.id.cmp(&b.1.id));
    let mut result = PreparedConversations {
        training: ConversationSet::default(),
        validation: ConversationSet::default(),
        provenance: DatasetProvenance {
            selections: selections.to_vec(),
            format: "chat".into(),
            fingerprint_kind: "conversation-ids-and-target-mask-v1".into(),
            documents: Vec::new(),
            split_seed: seed,
            split_policy: match policy {
                ValidationSplit::None => "none".into(),
                ValidationSplit::Count(n) => format!("count:{n}"),
                ValidationSplit::Ratio(r) => format!("ratio:{r}"),
            },
        },
        training_documents: Vec::new(),
        training_files: 0,
        training_tokens: 0,
    };
    result.provenance.selections.sort();
    result.provenance.selections.dedup();
    let mut training_files = BTreeSet::new();
    for (path, example) in records {
        let validation = validation_ids.contains(&example.id);
        let single = ConversationSet {
            examples: vec![example.clone()],
        };
        result.provenance.documents.push(DocumentFingerprint {
            id: example.id.clone(),
            partition: if validation { "validation" } else { "training" }.into(),
            sha256: single.identity()?,
        });
        if validation {
            result.validation.examples.push(example);
        } else {
            let index = result.training.examples.len();
            result.training_documents.push(DocumentInfo {
                id: PathBuf::from(&example.id),
                path: path.clone(),
                example_range: index..index + 1,
            });
            training_files.insert(path);
            result.training_tokens += example.ids.len();
            result.training.examples.push(example);
        }
    }
    result.training_files = training_files.len();
    Ok(result)
}
