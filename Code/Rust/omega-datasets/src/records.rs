use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;

use crate::{
    config::{Format, Limits, Mapping, Source},
    hash, read_bounded,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub(crate) struct Normalized {
    pub stage: &'static str,
    pub value: Value,
    pub digest: String,
    pub overlap_hashes: Vec<String>,
    pub group: Option<String>,
}

/// NFKC + collapsed whitespace for auditing only; output text stays unchanged.
fn normalized(text: &str) -> String {
    text.nfkc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("required string field {field:?} is missing or not a string"))
}

pub(crate) fn map_record(value: &Value, source: &Source) -> Result<Option<Normalized>> {
    ensure!(value.is_object(), "record must be a JSON object");
    let (stage, output, identity, overlaps) = match &source.mapping {
        Mapping::Text { column } => {
            let text = string(value, column)?;
            let norm = normalized(text);
            if norm.is_empty() {
                return Ok(None);
            }
            (
                "base",
                json!({"text": text}),
                json!({"text": norm}),
                vec![hash(norm.as_bytes())],
            )
        }
        mapping => {
            let messages = match mapping {
                Mapping::Messages {
                    column,
                    role_field,
                    content_field,
                    user_role,
                    assistant_role,
                    system_role,
                } => value
                    .get(column)
                    .and_then(Value::as_array)
                    .with_context(|| format!("required array field {column:?} is missing"))?
                    .iter()
                    .map(|m| {
                        let raw_role = string(m, role_field)?;
                        let role = if raw_role == user_role {
                            "user"
                        } else if raw_role == assistant_role {
                            "assistant"
                        } else if raw_role == system_role {
                            "system"
                        } else {
                            bail!("unsupported message role {raw_role:?}")
                        };
                        Ok(Message {
                            role: role.into(),
                            content: string(m, content_field)?.into(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                Mapping::Instruction {
                    prompt_column,
                    response_column,
                    input_column,
                    system,
                } => {
                    let mut messages = Vec::new();
                    if let Some(system) = system {
                        messages.push(Message {
                            role: "system".into(),
                            content: system.clone(),
                        });
                    }
                    let mut prompt = string(value, prompt_column)?.to_owned();
                    if let Some(column) = input_column {
                        let input = string(value, column)?;
                        if !input.trim().is_empty() {
                            prompt.push_str("\n\n");
                            prompt.push_str(input);
                        }
                    }
                    messages.push(Message {
                        role: "user".into(),
                        content: prompt,
                    });
                    messages.push(Message {
                        role: "assistant".into(),
                        content: string(value, response_column)?.into(),
                    });
                    messages
                }
                Mapping::Text { .. } => unreachable!(),
            };
            validate_messages(&messages)?;
            let canonical: Vec<_> = messages
                .iter()
                .map(|m| json!({"role": m.role, "content": normalized(&m.content)}))
                .collect();
            // Generic system instructions must not join an entire corpus. User and
            // assistant text, plus joined dialogue, catch exact cross-stage overlap.
            let mut overlaps: Vec<_> = messages
                .iter()
                .filter(|m| m.role != "system")
                .map(|m| hash(normalized(&m.content).as_bytes()))
                .collect();
            let joined = messages
                .iter()
                .filter(|m| m.role != "system")
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            overlaps.push(hash(normalized(&joined).as_bytes()));
            (
                "chat",
                json!({"schema_version": 1, "messages": messages}),
                json!({"messages": canonical}),
                overlaps,
            )
        }
    };
    let group = if let Some(group) = &source.group {
        Some(format!("string:{group}"))
    } else if let Some(column) = &source.group_column {
        let field = value
            .get(column)
            .with_context(|| format!("missing group_column {column:?}"))?;
        let group = match field {
            Value::String(s) if !s.trim().is_empty() => format!("string:{s}"),
            Value::Number(n) => format!("number:{n}"),
            _ => bail!("group_column {column:?} must be a nonblank string or number"),
        };
        Some(group)
    } else {
        None
    };
    let group = group.map(|g| {
        serde_json::to_string(&(source.group_namespace.as_ref().unwrap_or(&source.repo), g))
            .unwrap()
    });
    Ok(Some(Normalized {
        stage,
        value: output,
        digest: hash(&serde_json::to_vec(&identity)?),
        overlap_hashes: overlaps,
        group,
    }))
}

fn validate_messages(messages: &[Message]) -> Result<()> {
    let mut expected = "user";
    let mut turns = 0;
    for (index, message) in messages.iter().enumerate() {
        ensure!(
            !message.content.trim().is_empty(),
            "message {} is blank",
            index + 1
        );
        if index == 0 && message.role == "system" {
            continue;
        }
        ensure!(
            message.role == expected,
            "message {}: expected {expected}, got {}",
            index + 1,
            message.role
        );
        turns += 1;
        expected = if expected == "user" {
            "assistant"
        } else {
            "user"
        };
    }
    ensure!(
        turns >= 2 && expected == "user",
        "conversation must have alternating user/assistant turns and end with assistant"
    );
    Ok(())
}

/// Stream rows; JSON arrays are bounded by max_file_bytes and loaded as a unit.
pub(crate) fn visit_file(
    path: &Path,
    remote_name: &str,
    format: Format,
    limits: &Limits,
    mut visit: impl FnMut(usize, Value) -> Result<()>,
) -> Result<()> {
    let file = File::open(path)?;
    if format == Format::Parquet {
        ensure!(
            !remote_name.ends_with(".gz"),
            "gzip-wrapped Parquet is unsupported"
        );
        let reader = SerializedFileReader::new(file).context("Open Parquet")?;
        ensure!(
            reader.metadata().file_metadata().num_rows() >= 0,
            "invalid Parquet row count"
        );
        for (index, row) in reader.get_row_iter(None)?.enumerate() {
            visit(
                index + 1,
                row.context("Decode Parquet row")?.to_json_value(),
            )?;
        }
        return Ok(());
    }
    let reader: Box<dyn Read> = if remote_name.ends_with(".gz") {
        Box::new(flate2::read::MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    if format == Format::Json {
        let bytes = read_bounded(reader, limits.max_file_bytes)?;
        let records: Vec<Value> =
            serde_json::from_slice(&bytes).context("JSON source must be an array of objects")?;
        for (index, value) in records.into_iter().enumerate() {
            visit(index + 1, value)?;
        }
    } else {
        let mut reader = BufReader::new(reader);
        let mut line = Vec::new();
        let mut line_number = 0;
        let mut total = 0u64;
        loop {
            line.clear();
            // Take caps allocation even for a single unbroken line.
            let n = reader
                .by_ref()
                .take((limits.max_record_bytes as u64).saturating_add(1))
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            line_number += 1;
            total = total
                .checked_add(n as u64)
                .context("expanded input size overflow")?;
            ensure!(
                n <= limits.max_record_bytes,
                "line {line_number} exceeds max_record_bytes"
            );
            ensure!(
                total <= limits.max_file_bytes,
                "expanded file exceeds max_file_bytes"
            );
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            visit(
                line_number,
                serde_json::from_slice(&line)
                    .with_context(|| format!("Invalid JSONL line {line_number}"))?,
            )?;
        }
    }
    Ok(())
}
