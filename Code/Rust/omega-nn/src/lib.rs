//! CPU training and configurable generation for a tiny decoder-only GPT.

pub mod generation;
pub mod loss;
pub mod model;

pub use generation::{
    GENERATION_SAMPLING_ALGORITHM, GenerationOptions, SamplingOptions, generate_with_options,
    generate_with_options_on_device,
};
pub use loss::{MaskedLoss, masked_cross_entropy};
pub use model::{Gpt, GptConfig};

use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use omega_tokenizer::Tokens;

pub type Cpu = burn::backend::NdArray<f32>;
pub type TrainingBackend = burn::backend::Autodiff<Cpu>;

/// Encode without adding special tokens or silently discarding overflow/padding.
pub fn encode_text(tokenizer: &Tokens, text: &str) -> Result<Vec<u32>, String> {
    let encoding = tokenizer.encode(text, false)?;
    if !encoding.get_overflowing().is_empty() {
        return Err("Tokenizer truncated the text; disable truncation before encoding".into());
    }
    if encoding.get_attention_mask().contains(&0) {
        return Err("Tokenizer padded the encoding; disable padding before encoding".into());
    }
    if encoding.get_ids().is_empty() {
        return Err("Text must encode to at least one token".into());
    }
    Ok(encoding.get_ids().to_vec())
}

/// Require a nonempty vocabulary whose IDs are exactly `0..vocab_size`.
pub fn validate_tokenizer(tokenizer: &Tokens) -> Result<usize, String> {
    let size = tokenizer.vocab_size();
    if size == 0 {
        return Err("Tokenizer vocabulary must not be empty".into());
    }
    u32::try_from(size - 1).map_err(|_| "Tokenizer vocabulary IDs exceed u32".to_string())?;
    for id in 0..size {
        if tokenizer.id_to_token(id as u32).is_none() {
            return Err(format!(
                "Tokenizer vocabulary must have dense IDs; missing ID {id}"
            ));
        }
    }
    Ok(size)
}

fn validate_ids(ids: &[u32], vocab_size: usize) -> Result<(), String> {
    if vocab_size == 0 {
        return Err("vocab_size must be greater than zero".into());
    }
    if ids.is_empty() {
        return Err("Token sequence must not be empty".into());
    }
    for (position, &id) in ids.iter().enumerate() {
        if u64::from(id) >= vocab_size as u64 {
            return Err(format!(
                "Token ID {id} at position {position} is outside vocabulary 0..{vocab_size}"
            ));
        }
    }
    Ok(())
}

/// Construct a single-sequence integer tensor of shape `[1, ids.len()]`.
pub fn token_tensor<B: Backend>(
    ids: &[u32],
    vocab_size: usize,
    context_length: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>, String> {
    validate_ids(ids, vocab_size)?;
    if context_length == 0 || ids.len() > context_length {
        return Err(format!(
            "Token sequence length {} exceeds context_length {context_length} (which must be positive)",
            ids.len()
        ));
    }
    let data = TensorData::new(
        ids.iter().map(|&id| i64::from(id)).collect(),
        [1, ids.len()],
    );
    Ok(Tensor::from_data(data, device))
}

type NextTokenTensors<B> = (Tensor<B, 2, Int>, Tensor<B, 1, Int>);

// Shared by the single-sequence training path and its alignment regression test.
// The source includes the final target; shift exactly once here.
fn next_token_tensors<B: Backend>(
    config: &GptConfig,
    ids: &[u32],
    device: &B::Device,
) -> Result<NextTokenTensors<B>, String> {
    validate_ids(ids, config.vocab_size)?;
    if ids.len() < 2 || ids.len() - 1 > config.context_length {
        return Err(
            "Training requires 2..=context_length+1 tokens; truncation is not performed".into(),
        );
    }
    let sequence = ids.len() - 1;
    let inputs = token_tensor(
        &ids[..sequence],
        config.vocab_size,
        config.context_length,
        device,
    )?;
    let targets = token_tensor::<B>(&ids[1..], config.vocab_size, config.context_length, device)?
        .reshape([sequence]);
    Ok((inputs, targets))
}

/// Fit one sequence using Adam and next-token cross entropy, without truncation.
/// Returns the inference model and one pre-update loss per optimization step.
/// Seeding affects Burn's shared CPU random-number generator.
pub fn train_on_tokens(
    config: &GptConfig,
    ids: &[u32],
    steps: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<(Gpt<Cpu>, Vec<f32>), String> {
    config.validate()?;
    validate_ids(ids, config.vocab_size)?;
    if ids.len() < 2 || ids.len() - 1 > config.context_length {
        return Err(
            "Training requires 2..=context_length+1 tokens; truncation is not performed".into(),
        );
    }
    if steps == 0 {
        return Err("Training steps must be greater than zero".into());
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning_rate must be finite and greater than zero".into());
    }

    let device = Default::default();
    TrainingBackend::seed(seed);
    let mut model = config.init::<TrainingBackend>(&device)?;
    let sequence = ids.len() - 1;
    let (inputs, targets) = next_token_tensors::<TrainingBackend>(config, ids, &device)?;
    let criterion = CrossEntropyLossConfig::new().init::<TrainingBackend>(&device);
    let mut optimizer = AdamConfig::new().init::<TrainingBackend, Gpt<TrainingBackend>>();
    let mut losses = Vec::with_capacity(steps);
    for step in 0..steps {
        let logits = model
            .forward(inputs.clone())
            .reshape([sequence, config.vocab_size]);
        let loss = criterion.forward(logits, targets.clone());
        let value: f32 = loss.clone().into_scalar();
        if !value.is_finite() {
            return Err(format!(
                "Non-finite training loss at step {step}; try a smaller learning_rate"
            ));
        }
        losses.push(value);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, gradients);
    }
    Ok((model.valid(), losses))
}

/// Greedily append tokens using the entire prefix and fixed absolute positions.
/// A mismatched `config` returns an error before inference, even for a zero-token
/// budget. Newly generated EOS is included in the result.
/// An EOS already present in the prompt does not stop generation.
pub fn generate(
    model: &Gpt<Cpu>,
    config: &GptConfig,
    prompt: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
) -> Result<Vec<u32>, String> {
    generate_with_options(
        model,
        config,
        prompt,
        max_new_tokens,
        eos_token_id,
        &GenerationOptions::Greedy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 4,
            context_length: 6,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        }
    }

    #[test]
    fn training_loss_is_finite_and_improves() {
        let (_, losses) = train_on_tokens(&config(), &[0, 1, 2, 3, 0, 1, 2], 40, 0.01, 42).unwrap();
        assert_eq!(losses.len(), 40);
        assert!(losses.iter().all(|loss| loss.is_finite()));
        assert!(losses.last().unwrap() < &losses[0]);
    }

    #[test]
    fn tensors_preserve_ids_and_reject_invalid_inputs() {
        let device = Default::default();
        let tensor = token_tensor::<Cpu>(&[3, 0, 2], 4, 3, &device).unwrap();
        assert_eq!(tensor.dims(), [1, 3]);
        assert_eq!(tensor.into_data().to_vec::<i64>().unwrap(), [3, 0, 2]);
        for (ids, vocab, context) in [
            (&[][..], 4, 3),
            (&[4][..], 4, 3),
            (&[u32::MAX][..], 4, 3),
            (&[0][..], 0, 3),
            (&[0][..], 4, 0),
            (&[0, 1][..], 4, 1),
        ] {
            assert!(token_tensor::<Cpu>(ids, vocab, context, &device).is_err());
        }
    }

    #[test]
    fn invalid_training_arguments_are_rejected() {
        for ids in [&[][..], &[0][..], &[0, 4][..], &[0; 8][..]] {
            assert!(train_on_tokens(&config(), ids, 1, 0.01, 42).is_err());
        }
        assert!(train_on_tokens(&config(), &[0, 1], 0, 0.01, 42).is_err());
        for rate in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(train_on_tokens(&config(), &[0, 1], 1, rate, 42).is_err());
        }
    }

    #[test]
    fn training_inputs_and_targets_are_shifted_exactly_once() {
        let device = Default::default();
        for ids in [&[3, 0][..], &[3, 0, 2, 1, 3, 2, 0][..]] {
            let (inputs, targets) = next_token_tensors::<Cpu>(&config(), ids, &device).unwrap();
            assert_eq!(inputs.dims(), [1, ids.len() - 1]);
            assert_eq!(targets.dims(), [ids.len() - 1]);
            assert_eq!(
                inputs.into_data().to_vec::<i64>().unwrap(),
                ids[..ids.len() - 1]
                    .iter()
                    .map(|&id| i64::from(id))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                targets.into_data().to_vec::<i64>().unwrap(),
                ids[1..].iter().map(|&id| i64::from(id)).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn generation_rejects_every_configuration_mismatch_before_inference() {
        let original = config();
        let model = original.init::<Cpu>(&Default::default()).unwrap();
        for field in [
            "vocab_size",
            "context_length",
            "d_model",
            "num_heads",
            "num_layers",
            "d_ff",
        ] {
            let mut supplied = original.clone();
            match field {
                "vocab_size" => supplied.vocab_size += 1,
                "context_length" => supplied.context_length += 1,
                "d_model" => supplied.d_model *= 2,
                "num_heads" => supplied.num_heads *= 2,
                "num_layers" => supplied.num_layers += 1,
                _ => supplied.d_ff += 1,
            }
            supplied.validate().unwrap();
            for budget in [0, 1] {
                let error = generate(&model, &supplied, &[0], budget, None).unwrap_err();
                assert!(
                    error.contains("Model/config mismatch") && error.contains(field),
                    "{error}"
                );
            }
        }
        assert_eq!(
            generate(&model, &original, &[0; 6], 0, None).unwrap(),
            [0; 6]
        );
        assert!(generate(&model, &original, &[0; 7], 0, None).is_err());
    }

    #[test]
    fn generation_preserves_prefix_and_respects_length_and_eos() {
        let config = config();
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let prompt = [0, 1];
        let output = generate(&model, &config, &prompt, 4, None).unwrap();
        assert_eq!(&output[..2], &prompt);
        assert_eq!(output.len(), 6);
        assert!(output.iter().all(|&id| id < 4));
        assert_eq!(generate(&model, &config, &prompt, 0, None).unwrap(), prompt);
        let stopped = generate(&model, &config, &prompt, 4, Some(output[2])).unwrap();
        assert_eq!(stopped, output[..3]);
        assert!(generate(&model, &config, &prompt, 5, None).is_err());
        assert!(generate(&model, &config, &prompt, usize::MAX, None).is_err());
        assert!(generate(&model, &config, &prompt, 0, Some(4)).is_err());
        assert!(generate(&model, &config, &[], 1, None).is_err());
        assert!(generate(&model, &config, &[4], 1, None).is_err());
    }

    #[test]
    fn tokenizer_fixture_integrates_with_model() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json");
        let tokenizer = Tokens::new(path).unwrap();
        let vocab_size = validate_tokenizer(&tokenizer).unwrap();
        assert_eq!(vocab_size, 13);
        let ids = encode_text(&tokenizer, "Hello world!").unwrap();
        assert_eq!(ids, [2, 3, 11]);
        assert!(encode_text(&tokenizer, "").is_err());
        let config = GptConfig {
            vocab_size,
            ..config()
        };
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let output = generate(&model, &config, &ids, 1, None).unwrap();
        assert_eq!(&output[..ids.len()], ids.as_slice());
        assert!(tokenizer.decode(&output, false).is_ok());
    }
}
