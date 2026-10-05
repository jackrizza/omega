//! CPU generation with greedy defaults and request-local seeded sampling.
//!
//! Sampling v1 uses SplitMix64 (wrapping u64 arithmetic), then its upper 53 bits
//! divided by 2^53 for one [0,1) draw per generated token, even singleton support.
//! It never seeds or consumes Burn's global RNG. Floating-point softmax can differ
//! across platforms; the portable PRNG alone is not a cross-platform token promise.
//! A fresh Burn model can still initialize its parameters lazily on first forward;
//! that existing model-initialization RNG use is separate from token selection.

use crate::{Cpu, Gpt, GptConfig, token_tensor, validate_ids};
use burn::{
    module::Module,
    tensor::{ElementConversion, backend::Backend},
};

pub const GENERATION_SAMPLING_ALGORITHM: &str = "splitmix64-softmax-topk-topp-v1";

#[derive(Clone, Debug, Default, PartialEq)]
pub enum GenerationOptions {
    #[default]
    Greedy,
    Sample(SamplingOptions),
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingOptions {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: u64,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: 42,
        }
    }
}

impl GenerationOptions {
    /// Sampling-only flags require explicit sampling, even when equal to defaults.
    /// Vocabulary-dependent checks are performed by `validate` and generation.
    pub fn from_sampling_flags(
        sample: bool,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if !sample {
            if temperature.is_some() || top_k.is_some() || top_p.is_some() || seed.is_some() {
                return Err("Sampling options require --sample".into());
            }
            return Ok(Self::Greedy);
        }
        let options = Self::Sample(SamplingOptions {
            temperature: temperature.unwrap_or(1.0),
            top_k,
            top_p,
            seed: seed.unwrap_or(42),
        });
        options.validate(usize::MAX)?;
        Ok(options)
    }

    pub fn validate(&self, vocab_size: usize) -> Result<(), String> {
        if vocab_size == 0 {
            return Err("Generation vocabulary must not be empty".into());
        }
        if let Self::Sample(options) = self {
            if !options.temperature.is_finite() || options.temperature <= 0.0 {
                return Err("Sampling temperature must be finite and greater than zero".into());
            }
            if options.top_k.is_some_and(|k| k == 0 || k > vocab_size) {
                return Err(format!("Sampling top_k must be in 1..={vocab_size}"));
            }
            if options
                .top_p
                .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
            {
                return Err("Sampling top_p must be finite and in (0, 1]".into());
            }
        }
        Ok(())
    }
}

struct RequestRng(u64);
impl RequestRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    fn draw(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn finite_logits(logits: &[f32]) -> Result<(), String> {
    if logits.is_empty() {
        return Err("Generation logits must not be empty".into());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("Generation encountered non-finite logits".into());
    }
    Ok(())
}

/// Softmax in f64 after subtracting the largest finite f32 logit. Sort descending
/// probability, ascending ID on ties; retain top-k, then the smallest nonempty
/// prefix reaching top-p of that retained mass. Finally normalize the prefix.
fn distribution(logits: &[f32], options: &SamplingOptions) -> Result<Vec<(usize, f64)>, String> {
    GenerationOptions::Sample(options.clone()).validate(logits.len())?;
    finite_logits(logits)?;
    let max = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .expect("nonempty logits") as f64;
    // Subtract before division: very small temperatures can produce -infinity,
    // whose exponential is safely zero, while a maximum always has weight one.
    let mut probabilities: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(id, &value)| (id, ((f64::from(value) - max) / options.temperature).exp()))
        .collect();
    let total: f64 = probabilities.iter().map(|entry| entry.1).sum();
    for entry in &mut probabilities {
        entry.1 /= total;
    }
    probabilities.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    probabilities.truncate(options.top_k.unwrap_or(probabilities.len()));
    if let Some(p) = options.top_p {
        let retained: f64 = probabilities.iter().map(|entry| entry.1).sum();
        let threshold = p * retained;
        let mut mass = 0.0;
        let mut count = 0;
        for entry in &probabilities {
            mass += entry.1;
            count += 1;
            if mass >= threshold {
                break;
            }
        }
        probabilities.truncate(count);
    }
    let retained: f64 = probabilities.iter().map(|entry| entry.1).sum();
    for entry in &mut probabilities {
        entry.1 /= retained;
    }
    Ok(probabilities)
}

fn draw_token(probabilities: &[(usize, f64)], draw: f64) -> usize {
    let mut cumulative = 0.0;
    let mut last_positive = probabilities[0].0;
    for &(id, probability) in probabilities {
        if probability > 0.0 {
            last_positive = id;
        }
        cumulative += probability;
        if draw < cumulative {
            return id;
        }
    }
    // Rounding can leave the cumulative mass just below one; zero-mass tails
    // must never be selected as a fallback.
    last_positive
}

/// Append tokens using full-prefix forward passes and fixed absolute positions.
/// Returns prompt plus output, including newly generated EOS. Prompt EOS does
/// not stop generation. Validate the entire requested context budget up front,
/// even if EOS might stop early. Greedy uses Burn's existing NdArray argmax,
/// choosing the lowest token ID for equal finite logits. No cache, sliding
/// window or tokenizer changes.
pub fn generate_with_options(
    model: &Gpt<Cpu>,
    config: &GptConfig,
    prompt: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
    options: &GenerationOptions,
) -> Result<Vec<u32>, String> {
    generate_with_options_on_device(
        model,
        config,
        prompt,
        max_new_tokens,
        eos_token_id,
        options,
        &Default::default(),
    )
}

/// Generate on the explicitly selected device using the shared token-selection loop.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_options_on_device<B: Backend<FloatElem = f32>>(
    model: &Gpt<B>,
    config: &GptConfig,
    prompt: &[u32],
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
    options: &GenerationOptions,
    device: &B::Device,
) -> Result<Vec<u32>, String> {
    model.validate_config(config)?;
    if model.devices().iter().any(|actual| actual != device) {
        return Err("Generation model is not on the requested device".into());
    }
    validate_ids(prompt, config.vocab_size)?;
    if let Some(eos) = eos_token_id {
        validate_ids(&[eos], config.vocab_size)?;
    }
    options.validate(config.vocab_size)?;
    let total = prompt
        .len()
        .checked_add(max_new_tokens)
        .ok_or("Requested generation length overflows usize")?;
    if total > config.context_length {
        return Err(
            "Prompt plus max_new_tokens exceeds context_length; sliding context is not supported"
                .into(),
        );
    }
    let mut rng = RequestRng(match options {
        GenerationOptions::Greedy => 0,
        GenerationOptions::Sample(options) => options.seed,
    });
    let mut output = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let sequence = output.len();
        let tokens = token_tensor::<B>(&output, config.vocab_size, config.context_length, device)?;
        let logits =
            model
                .forward(tokens)
                .slice([0..1, sequence - 1..sequence, 0..config.vocab_size]);
        let values = logits
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| format!("Cannot read generation logits: {e:?}"))?;
        finite_logits(&values)?;
        let next = match options {
            GenerationOptions::Greedy => {
                u32::try_from(logits.argmax(2).into_scalar().elem::<i64>())
                    .map_err(|_| "Generated token ID exceeds u32")?
            }
            GenerationOptions::Sample(options) => {
                u32::try_from(draw_token(&distribution(&values, options)?, rng.draw()))
                    .map_err(|_| "Generated token ID exceeds u32")?
            }
        };
        output.push(next);
        if Some(next) == eos_token_id {
            break;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_has_fixed_words_and_token_vectors() {
        let mut rng = RequestRng(42);
        assert_eq!(
            (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>(),
            vec![
                0xbdd732262feb6e95,
                0x28efe333b266f103,
                0x47526757130f9f52,
                0x581ce1ff0e4ae394,
                0x09bc585a244823f2
            ]
        );
        let probabilities = distribution(&[0.0; 4], &SamplingOptions::default()).unwrap();
        let mut rng = RequestRng(42);
        assert_eq!(
            (0..5)
                .map(|_| draw_token(&probabilities, rng.draw()))
                .collect::<Vec<_>>(),
            vec![2, 0, 1, 1, 0]
        );
        let mut rng = RequestRng(u64::MAX);
        for _ in 0..100 {
            let draw = rng.draw();
            assert!((0.0..1.0).contains(&draw));
        }
    }

    #[test]
    fn probabilities_filter_in_order_break_ties_and_normalize() {
        let logits = [4.0_f32.ln(), 2.0_f32.ln(), 0.0, 0.0];
        let probabilities = distribution(&logits, &SamplingOptions::default()).unwrap();
        for (entry, expected) in probabilities.iter().zip([0.5, 0.25, 0.125, 0.125]) {
            assert!((entry.1 - expected).abs() < 1e-7);
        }
        let mut options = SamplingOptions {
            top_k: Some(2),
            top_p: Some(0.6),
            ..Default::default()
        };
        assert_eq!(distribution(&logits, &options).unwrap(), vec![(0, 1.0)]);
        options.top_p = Some(0.7);
        let retained = distribution(&logits, &options).unwrap();
        assert_eq!(retained.len(), 2);
        assert!((retained[0].1 - 2.0 / 3.0).abs() < 1e-7);
        assert!((retained.iter().map(|v| v.1).sum::<f64>() - 1.0).abs() < 1e-15);
        options.top_k = Some(3);
        options.top_p = Some(2.0 / 3.0);
        assert_eq!(
            distribution(&[0.0; 4], &options).unwrap(),
            vec![(0, 0.5), (1, 0.5)]
        );
        assert_eq!(draw_token(&[(0, 0.5), (1, 0.5)], 0.0), 0);
        assert_eq!(draw_token(&[(0, 0.5), (1, 0.5)], 0.5), 1);
        assert_eq!(
            draw_token(
                &[(0, 0.5), (1, 0.4999999999999998), (2, 0.0)],
                1.0 - f64::EPSILON / 2.0
            ),
            1
        );
        options.top_p = Some(f64::from_bits(1));
        assert_eq!(distribution(&[0.0; 4], &options).unwrap(), vec![(0, 1.0)]);
    }

    #[test]
    fn extreme_finite_logits_and_temperatures_remain_valid() {
        let mut options = SamplingOptions {
            temperature: f64::from_bits(1),
            ..Default::default()
        };
        assert_eq!(
            distribution(&[f32::MIN, f32::MAX, 0.0], &options).unwrap(),
            vec![(1, 1.0), (0, 0.0), (2, 0.0)]
        );
        options.temperature = f64::MAX;
        assert_eq!(
            distribution(&[f32::MIN, f32::MAX], &options).unwrap(),
            vec![(0, 0.5), (1, 0.5)]
        );
        options.temperature = 0.5;
        let probabilities = distribution(&[0.0, 2.0_f32.ln()], &options).unwrap();
        assert!((probabilities[0].1 - 0.8).abs() < 1e-7);
    }

    #[test]
    fn invalid_options_logits_and_greedy_flags_are_rejected() {
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                GenerationOptions::Sample(SamplingOptions {
                    temperature,
                    ..Default::default()
                })
                .validate(4)
                .is_err()
            );
        }
        for top_k in [0, 5] {
            assert!(
                GenerationOptions::Sample(SamplingOptions {
                    top_k: Some(top_k),
                    ..Default::default()
                })
                .validate(4)
                .is_err()
            );
        }
        for top_p in [0.0, -1.0, 1.1, f64::NAN, f64::INFINITY] {
            assert!(
                GenerationOptions::Sample(SamplingOptions {
                    top_p: Some(top_p),
                    ..Default::default()
                })
                .validate(4)
                .is_err()
            );
        }
        for logits in [
            vec![],
            vec![f32::NAN],
            vec![0.0, f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            assert!(distribution(&logits, &SamplingOptions::default()).is_err());
        }
        assert!(GenerationOptions::Greedy.validate(0).is_err());
        for flags in [
            (Some(1.0), None, None, None),
            (None, Some(1), None, None),
            (None, None, Some(1.0), None),
            (None, None, None, Some(42)),
        ] {
            assert!(
                GenerationOptions::from_sampling_flags(false, flags.0, flags.1, flags.2, flags.3)
                    .unwrap_err()
                    .contains("--sample")
            );
        }
        assert_eq!(
            GenerationOptions::from_sampling_flags(true, None, None, None, None).unwrap(),
            GenerationOptions::Sample(SamplingOptions::default())
        );
    }
}
