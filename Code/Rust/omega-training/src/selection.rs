//! Shared explicit dataset-format selection and published-partition guards.
use crate::{ValidationSplit, dataset::DatasetFormat};
use clap::ValueEnum;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Infer published base/chat partitions; ordinary folders retain text mode
    Auto,
    Text,
    Jsonl,
    Chat,
}

impl From<Format> for DatasetFormat {
    fn from(value: Format) -> Self {
        match value {
            Format::Auto => unreachable!("Resolve dataset format before loading"),
            Format::Text => Self::Text,
            Format::Jsonl => Self::Jsonl,
            Format::Chat => Self::Jsonl,
        }
    }
}

/// Resolve only explicit published partitions, never recursively infer a release root.
pub fn resolve_dataset_format(
    root: &Path,
    selections: &[String],
    requested: Format,
    training: bool,
    policy: ValidationSplit,
) -> Result<Format, String> {
    use crate::release::{ReleaseStage, inspect_partition};
    let (_, directories) = crate::dataset::resolve_dataset_folders(root, selections)?;
    let mut expected = None;
    let mut selected_partition = None;
    let mut ordinary = false;
    for directory in directories {
        if let Some(release) = inspect_partition(&directory)? {
            let format = match release.stage {
                ReleaseStage::Base => Format::Jsonl,
                ReleaseStage::Chat => Format::Chat,
            };
            if expected.is_some_and(|previous| previous != format)
                || selected_partition
                    .as_ref()
                    .is_some_and(|previous| previous != &release.partition)
            {
                return Err("Select one omega-datasets stage and partition per operation; do not mix base/chat or train/validation/test".into());
            }
            if training && release.partition != "train" {
                return Err("Training and tokenizer fitting require the published train partition; held-out validation/test partitions are evaluation-only".into());
            }
            if !matches!(policy, ValidationSplit::None | ValidationSplit::Count(0))
                && !matches!(policy, ValidationSplit::Ratio(r) if r == 0.0)
            {
                return Err("omega-datasets partitions already preserve source groups; do not re-split them. Evaluate the external validation partition separately".into());
            }
            expected = Some(format);
            selected_partition = Some(release.partition);
        } else {
            ordinary = true;
        }
    }
    if let Some(format) = expected {
        if ordinary {
            return Err(
                "Select published partitions separately from ordinary dataset folders".into(),
            );
        }
        if requested != Format::Auto && requested != format {
            return Err(format!(
                "Dataset format mismatch: published partition requires --dataset-format {} (or auto)",
                match format {
                    Format::Chat => "chat",
                    _ => "jsonl",
                }
            ));
        }
        Ok(format)
    } else {
        Ok(if requested == Format::Auto {
            Format::Text
        } else {
            requested
        })
    }
}
