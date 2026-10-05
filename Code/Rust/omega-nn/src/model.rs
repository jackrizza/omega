//! Tiny pre-norm decoder-only GPT for Burn 0.18.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, activation, backend::Backend},
};

/// Model dimensions. All dimensions must be positive, and `d_model` must be
/// divisible by `num_heads`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            context_length: 128,
            d_model: 64,
            num_heads: 4,
            num_layers: 2,
            d_ff: 256,
        }
    }
}

impl GptConfig {
    /// Check architectural constraints, not device memory availability.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("context_length", self.context_length),
            ("d_model", self.d_model),
            ("num_heads", self.num_heads),
            ("num_layers", self.num_layers),
            ("d_ff", self.d_ff),
        ] {
            if value == 0 {
                return Err(format!("{name} must be greater than zero"));
            }
        }
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err("d_model must be divisible by num_heads".into());
        }
        if i64::try_from(self.context_length).is_err() {
            return Err("context_length must fit in i64 for position indices".into());
        }
        Ok(())
    }

    /// Validate dimensions and initialize trainable parameters on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<Gpt<B>, String> {
        self.validate()?;
        Ok(Gpt {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.context_length, self.d_model)
                .init(device),
            blocks: (0..self.num_layers)
                .map(|_| DecoderBlock::new(self, device))
                .collect(),
            final_norm: LayerNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
            context_length: self.context_length,
        })
    }
}

/// Decoder with learned absolute positions, no dropout, and an untied LM head.
#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<DecoderBlock<B>>,
    final_norm: LayerNorm<B>,
    lm_head: Linear<B>,
    context_length: usize,
}

impl<B: Backend> Gpt<B> {
    /// Inspect the actual parameter shapes and existing attention metadata.
    /// No forward pass or parameter values are needed. Inconsistent loaded
    /// records return an error instead of reporting a plausible configuration.
    ///
    /// Burn records do not store primitive metadata such as the head count;
    /// when loading legacy weights, that metadata comes from initialization.
    pub fn architecture(&self) -> Result<GptConfig, String> {
        let [vocab_size, d_model] = self.token_embedding.weight.shape().dims();
        let first = self
            .blocks
            .first()
            .ok_or_else(|| "Model architecture has no decoder blocks".to_string())?;
        let config = GptConfig {
            vocab_size,
            context_length: self.context_length,
            d_model,
            num_heads: first.num_heads,
            num_layers: self.blocks.len(),
            d_ff: first.ffn_in.weight.shape().dims::<2>()[1],
        };
        config.validate()?;
        if self.position_embedding.weight.shape().dims::<2>() != [config.context_length, d_model] {
            return Err("Model architecture has inconsistent position embedding dimensions".into());
        }
        validate_linear(&self.lm_head, [d_model, vocab_size], false, "lm_head")?;
        validate_norm(&self.final_norm, d_model, "final_norm")?;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.num_heads != config.num_heads || block.head_dim != d_model / config.num_heads {
                return Err(format!(
                    "Model architecture has inconsistent attention metadata in block {index}"
                ));
            }
            for (name, linear, shape) in [
                ("query", &block.query, [d_model, d_model]),
                ("key", &block.key, [d_model, d_model]),
                ("value", &block.value, [d_model, d_model]),
                (
                    "attention_output",
                    &block.attention_output,
                    [d_model, d_model],
                ),
                ("ffn_in", &block.ffn_in, [d_model, config.d_ff]),
                ("ffn_out", &block.ffn_out, [config.d_ff, d_model]),
            ] {
                validate_linear(linear, shape, true, &format!("block {index} {name}"))?;
            }
            validate_norm(
                &block.attention_norm,
                d_model,
                &format!("block {index} attention_norm"),
            )?;
            validate_norm(&block.ffn_norm, d_model, &format!("block {index} ffn_norm"))?;
        }
        Ok(config)
    }

    /// Reject a configuration that does not describe this model, before tensor
    /// computation or checkpoint writes. This does not verify tokenizer identity.
    pub fn validate_config(&self, config: &GptConfig) -> Result<(), String> {
        config.validate()?;
        let actual = self.architecture()?;
        for (name, supplied, expected) in [
            ("vocab_size", config.vocab_size, actual.vocab_size),
            (
                "context_length",
                config.context_length,
                actual.context_length,
            ),
            ("d_model", config.d_model, actual.d_model),
            ("num_heads", config.num_heads, actual.num_heads),
            ("num_layers", config.num_layers, actual.num_layers),
            ("d_ff", config.d_ff, actual.d_ff),
        ] {
            if supplied != expected {
                return Err(format!(
                    "Model/config mismatch for {name}: supplied {supplied}, model requires {expected}"
                ));
            }
        }
        Ok(())
    }

    /// Map token IDs `[batch, sequence]` to raw logits `[batch, sequence, vocab]`.
    /// Each position can attend to itself and earlier positions only.
    ///
    /// Inputs must be on the model's device, with IDs in `0..vocab_size`.
    /// Token bounds are delegated to the backend embedding operation.
    ///
    /// # Panics
    /// Panics for an empty batch/sequence or a sequence longer than the context.
    /// Invalid IDs, incompatible devices, or malformed loaded parameters may
    /// panic in the backend. Use `token_tensor` and `validate_config` for checked
    /// host-side validation; this low-level method does not copy IDs to the host.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, sequence] = tokens.dims();
        assert!(batch > 0, "GPT requires a nonempty batch");
        assert!(sequence > 0, "GPT requires a nonempty sequence");
        assert!(
            sequence <= self.context_length,
            "GPT sequence length exceeds context_length"
        );

        // Burn's tril_mask is true ABOVE the diagonal: true means masked out.
        // Singleton batch/head dimensions broadcast across all attention heads.
        let mask = Tensor::<B, 2, Bool>::tril_mask([sequence, sequence], 0, &tokens.device())
            .reshape([1, 1, sequence, sequence]);
        self.forward_with_attention_mask(tokens, mask)
    }

    /// Checked right-padded input: `valid[batch, sequence]` is true for real
    /// tokens and false for padding. Each row must have a nonempty contiguous
    /// valid prefix. All input IDs, including padding IDs, must be in vocabulary.
    ///
    /// Attention excludes padding keys and future keys. Padded query logits are
    /// zero; pass a separate target mask to `masked_cross_entropy` for training.
    /// Positions start at zero in every row. Invalid shapes, devices, IDs, masks,
    /// model architecture, or non-finite output return errors. Validation copies
    /// IDs/masks/output to the host and synchronizes the backend; backend failures
    /// may still panic. The unpadded low-level `forward` contract is unchanged.
    pub fn forward_masked(
        &self,
        tokens: Tensor<B, 2, Int>,
        valid: Tensor<B, 2, Bool>,
    ) -> Result<Tensor<B, 3>, String> {
        let [batch, sequence] = tokens.dims();
        if batch == 0 || sequence == 0 {
            return Err("Masked forward requires nonempty batch and sequence dimensions".into());
        }
        if valid.dims() != [batch, sequence] {
            return Err("Input and validity mask shapes must match [batch, sequence]".into());
        }
        let device = tokens.device();
        if valid.device() != device || self.devices().iter().any(|item| *item != device) {
            return Err("Model, input IDs and validity mask must be on the same device".into());
        }
        let config = self.architecture()?;
        if sequence > config.context_length {
            return Err("Masked forward sequence length exceeds context_length".into());
        }
        for (index, id) in tokens.to_data().iter::<i64>().enumerate() {
            if id < 0 || id as u64 >= config.vocab_size as u64 {
                return Err(format!(
                    "Input token ID {id} at flattened position {index} is outside the model vocabulary"
                ));
            }
        }
        let validity: Vec<_> = valid.to_data().iter::<bool>().collect();
        for (row, mask) in validity.chunks_exact(sequence).enumerate() {
            let length = mask.iter().take_while(|&&value| value).count();
            if length == 0 || mask[length..].iter().any(|&value| value) {
                return Err(format!(
                    "Validity mask row {row} must be a nonempty valid prefix followed only by right padding"
                ));
            }
        }
        let padding = valid.clone().bool_not();
        let causal = Tensor::<B, 2, Bool>::tril_mask([sequence, sequence], 0, &device)
            .reshape([1, 1, sequence, sequence])
            .expand([batch, 1, sequence, sequence]);
        // NdArray's boolean OR requires equal shapes, unlike mask_fill.
        let padding_keys = padding
            .clone()
            .reshape([batch, 1, 1, sequence])
            .expand([batch, 1, sequence, sequence]);
        let attention_mask = causal.bool_or(padding_keys);
        let logits = self
            .forward_with_attention_mask(tokens, attention_mask)
            .mask_fill(padding.reshape([batch, sequence, 1]), 0.0);
        if logits
            .to_data()
            .iter::<f64>()
            .any(|value| !value.is_finite())
        {
            return Err("Masked forward produced non-finite logits".into());
        }
        Ok(logits)
    }

    fn forward_with_attention_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        mask: Tensor<B, 4, Bool>,
    ) -> Tensor<B, 3> {
        let [_, sequence] = tokens.dims();
        let device = tokens.device();
        let positions =
            Tensor::<B, 1, Int>::arange(0..sequence as i64, &device).reshape([1, sequence]);
        let mut hidden =
            self.token_embedding.forward(tokens) + self.position_embedding.forward(positions);
        for block in &self.blocks {
            hidden = block.forward(hidden, mask.clone());
        }
        self.lm_head.forward(self.final_norm.forward(hidden))
    }
}

fn validate_linear<B: Backend>(
    linear: &Linear<B>,
    shape: [usize; 2],
    bias: bool,
    name: &str,
) -> Result<(), String> {
    let valid_bias = match &linear.bias {
        Some(value) => bias && value.shape().dims::<1>() == [shape[1]],
        None => !bias,
    };
    if linear.weight.shape().dims::<2>() != shape || !valid_bias {
        return Err(format!(
            "Model architecture has inconsistent {name} dimensions"
        ));
    }
    Ok(())
}

fn validate_norm<B: Backend>(norm: &LayerNorm<B>, width: usize, name: &str) -> Result<(), String> {
    if norm.gamma.shape().dims::<1>() != [width] || norm.beta.shape().dims::<1>() != [width] {
        return Err(format!(
            "Model architecture has inconsistent {name} dimensions"
        ));
    }
    Ok(())
}

#[derive(Module, Debug)]
struct DecoderBlock<B: Backend> {
    attention_norm: LayerNorm<B>,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    attention_output: Linear<B>,
    ffn_norm: LayerNorm<B>,
    ffn_in: Linear<B>,
    ffn_out: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> DecoderBlock<B> {
    fn new(config: &GptConfig, device: &B::Device) -> Self {
        Self {
            attention_norm: LayerNormConfig::new(config.d_model).init(device),
            query: LinearConfig::new(config.d_model, config.d_model).init(device),
            key: LinearConfig::new(config.d_model, config.d_model).init(device),
            value: LinearConfig::new(config.d_model, config.d_model).init(device),
            attention_output: LinearConfig::new(config.d_model, config.d_model).init(device),
            ffn_norm: LayerNormConfig::new(config.d_model).init(device),
            ffn_in: LinearConfig::new(config.d_model, config.d_ff).init(device),
            ffn_out: LinearConfig::new(config.d_ff, config.d_model).init(device),
            num_heads: config.num_heads,
            head_dim: config.d_model / config.num_heads,
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let [batch, sequence, d_model] = input.dims();
        let normalized = self.attention_norm.forward(input.clone());
        let split_heads = |tensor: Tensor<B, 3>| {
            tensor
                .reshape([batch, sequence, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let query = split_heads(self.query.forward(normalized.clone()));
        let key = split_heads(self.key.forward(normalized.clone()));
        let value = split_heads(self.value.forward(normalized));

        // Scores are [batch, heads, query position, key position]. Every row
        // retains at least one valid prefix key, so softmax never sees an
        // entirely masked row, including padded queries in forward_masked.
        let scores = query
            .matmul(key.swap_dims(2, 3))
            .div_scalar((self.head_dim as f64).sqrt())
            .mask_fill(mask, f32::NEG_INFINITY);
        let attended = activation::softmax(scores, 3)
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch, sequence, d_model]);
        let hidden = input + self.attention_output.forward(attended);
        let feed_forward = self.ffn_out.forward(activation::gelu(
            self.ffn_in.forward(self.ffn_norm.forward(hidden.clone())),
        ));
        hidden + feed_forward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Cpu = NdArray<f32>;

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 16,
            context_length: 6,
            d_model: 12,
            num_heads: 3,
            num_layers: 2,
            d_ff: 24,
        }
    }

    #[test]
    fn logits_have_expected_shape_and_are_finite() {
        let device = Default::default();
        let model = config().init::<Cpu>(&device).unwrap();
        for sequence in [1, 4, 6] {
            let tokens = Tensor::<Cpu, 2, Int>::zeros([2, sequence], &device);
            let logits = model.forward(tokens);
            assert_eq!(logits.dims(), [2, sequence, 16]);
            assert!(
                logits
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn future_tokens_cannot_change_prefix_logits() {
        let device = Default::default();
        let model = config().init::<Cpu>(&device).unwrap();
        let tokens =
            Tensor::<Cpu, 2, Int>::from_data([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], &device);
        let full = model.forward(tokens.clone());
        for prefix in 1..6 {
            let changed = tokens.clone().slice_assign(
                [0..2, prefix..6],
                Tensor::<Cpu, 2, Int>::ones([2, 6 - prefix], &device).mul_scalar(15),
            );
            let expected = full.clone().slice([0..2, 0..prefix, 0..16]);
            let changed_prefix = model.forward(changed).slice([0..2, 0..prefix, 0..16]);
            let truncated = model.forward(tokens.clone().slice([0..2, 0..prefix]));
            for actual in [changed_prefix, truncated] {
                let expected = expected.clone().into_data().to_vec::<f32>().unwrap();
                let actual = actual.into_data().to_vec::<f32>().unwrap();
                assert_eq!(expected.len(), actual.len());
                for (expected, actual) in expected.iter().zip(actual.iter()) {
                    assert!(
                        expected.is_finite()
                            && actual.is_finite()
                            && (expected - actual).abs() <= 1e-5,
                        "noncausal logits for prefix {prefix}: {expected} vs {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        assert!(GptConfig::default().validate().is_ok());
        for field in 0..6 {
            let mut invalid = config();
            match field {
                0 => invalid.vocab_size = 0,
                1 => invalid.context_length = 0,
                2 => invalid.d_model = 0,
                3 => invalid.num_heads = 0,
                4 => invalid.num_layers = 0,
                _ => invalid.d_ff = 0,
            }
            assert!(invalid.validate().is_err());
            assert!(invalid.init::<Cpu>(&Default::default()).is_err());
        }
        let invalid = GptConfig {
            num_heads: 5,
            ..config()
        };
        assert!(invalid.validate().is_err());
        assert!(invalid.init::<Cpu>(&Default::default()).is_err());
    }

    #[test]
    fn architecture_survives_compatible_record_loading() {
        let config = config();
        let device = Default::default();
        let original = config.init::<Cpu>(&device).unwrap();
        assert_eq!(original.architecture().unwrap(), config);
        let loaded = config
            .init::<Cpu>(&device)
            .unwrap()
            .load_record(original.into_record());
        assert_eq!(loaded.architecture().unwrap(), config);
        loaded.validate_config(&config).unwrap();
    }

    #[test]
    fn inconsistent_model_structure_is_rejected_without_forward() {
        let device = Default::default();
        let original = config().init::<Cpu>(&device).unwrap();
        let mut model = original.clone();
        model.context_length += 1;
        assert!(
            model
                .architecture()
                .unwrap_err()
                .contains("position embedding")
        );
        let mut model = original.clone();
        model.blocks[1].head_dim += 1;
        assert!(model.architecture().unwrap_err().contains("block 1"));
        let mut model = original.clone();
        model.lm_head = LinearConfig::new(12, 15).with_bias(false).init(&device);
        assert!(model.architecture().unwrap_err().contains("lm_head"));
        let mut model = original.clone();
        model.blocks[1].ffn_in = LinearConfig::new(12, 25).init(&device);
        assert!(model.architecture().unwrap_err().contains("ffn_in"));
        let mut model = original.clone();
        model.final_norm = LayerNormConfig::new(11).init(&device);
        assert!(model.architecture().unwrap_err().contains("final_norm"));
        let mut model = original;
        model.blocks.clear();
        assert!(
            model
                .architecture()
                .unwrap_err()
                .contains("no decoder blocks")
        );
    }

    #[test]
    #[should_panic(expected = "GPT requires a nonempty batch")]
    fn low_level_forward_rejects_empty_batch() {
        let device = Default::default();
        config()
            .init::<Cpu>(&device)
            .unwrap()
            .forward(Tensor::zeros([0, 1], &device));
    }

    #[test]
    #[should_panic(expected = "GPT requires a nonempty sequence")]
    fn low_level_forward_rejects_empty_sequence() {
        let device = Default::default();
        config()
            .init::<Cpu>(&device)
            .unwrap()
            .forward(Tensor::zeros([1, 0], &device));
    }

    #[test]
    #[should_panic]
    fn low_level_forward_delegates_negative_ids_to_backend() {
        let device = Default::default();
        config()
            .init::<Cpu>(&device)
            .unwrap()
            .forward(Tensor::from_data([[-1]], &device));
    }

    #[test]
    #[should_panic]
    fn low_level_forward_delegates_out_of_bounds_ids_to_backend() {
        let device = Default::default();
        config()
            .init::<Cpu>(&device)
            .unwrap()
            .forward(Tensor::from_data([[16]], &device));
    }

    #[test]
    #[should_panic(expected = "GPT sequence length exceeds context_length")]
    fn rejects_sequences_beyond_context_length() {
        let device = Default::default();
        let model = config().init::<Cpu>(&device).unwrap();
        model.forward(Tensor::<Cpu, 2, Int>::zeros([1, 7], &device));
    }
}
