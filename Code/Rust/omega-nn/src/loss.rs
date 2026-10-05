//! Differentiable target masking, independent of input-attention padding masks.

use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};

/// Mean cross entropy over exactly `target_count` valid targets.
/// The loss remains attached to its autodiff graph.
#[derive(Debug)]
pub struct MaskedLoss<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub target_count: usize,
}

/// Compute mean next-token cross entropy for already aligned logits and targets.
/// This function does not shift labels. Shapes are `[batch, sequence, vocabulary]`
/// and `[batch, sequence]` for targets/mask; true means included. Unlike the input
/// attention mask, this mask may have gaps or entirely ignored rows.
///
/// Only selected targets must be in vocabulary; ignored IDs/logits may contain
/// arbitrary values. Rows are gathered before cross entropy, so ignored targets
/// contribute neither loss nor logits gradients. An entirely ignored batch is an
/// error. Shapes, devices, selected IDs/logit finiteness and finite final loss are
/// checked using host synchronization. Backend failures may still panic.
pub fn masked_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    valid: Tensor<B, 2, Bool>,
) -> Result<MaskedLoss<B>, String> {
    let [batch, sequence, vocabulary] = logits.dims();
    if batch == 0 || sequence == 0 || vocabulary == 0 {
        return Err(
            "Masked loss requires positive batch, sequence and vocabulary dimensions".into(),
        );
    }
    if targets.dims() != [batch, sequence] || valid.dims() != [batch, sequence] {
        return Err("Masked loss target/mask shapes must match logits [batch, sequence]".into());
    }
    let device = logits.device();
    if targets.device() != device || valid.device() != device {
        return Err("Masked loss logits, targets and mask must be on the same device".into());
    }
    let rows = batch
        .checked_mul(sequence)
        .ok_or("Masked loss row count overflows usize")?;
    let mask = valid.to_data();
    let target_data = targets.to_data();
    let mut selected = Vec::new();
    for (index, (included, target)) in mask
        .iter::<bool>()
        .zip(target_data.iter::<i64>())
        .enumerate()
    {
        if included {
            if target < 0 || target as u64 >= vocabulary as u64 {
                return Err(format!(
                    "Valid target ID {target} at flattened position {index} is outside vocabulary 0..{vocabulary}"
                ));
            }
            selected.push(i64::try_from(index).map_err(|_| "Masked loss row index exceeds i64")?);
        }
    }
    let target_count = selected.len();
    if target_count == 0 {
        return Err("Masked loss requires at least one valid target".into());
    }
    let indices =
        Tensor::<B, 1, Int>::from_data(TensorData::new(selected, [target_count]), &device);
    let selected_logits = logits
        .reshape([rows, vocabulary])
        .select(0, indices.clone());
    let selected_targets = targets.reshape([rows]).select(0, indices);
    if selected_logits
        .to_data()
        .iter::<f64>()
        .any(|value| !value.is_finite())
    {
        return Err("Masked loss has non-finite logits at a valid target".into());
    }
    let loss = CrossEntropyLossConfig::new()
        .init::<B>(&device)
        .forward(selected_logits, selected_targets);
    if loss.to_data().iter::<f64>().any(|value| !value.is_finite()) {
        return Err("Masked cross entropy is non-finite".into());
    }
    Ok(MaskedLoss { loss, target_count })
}
