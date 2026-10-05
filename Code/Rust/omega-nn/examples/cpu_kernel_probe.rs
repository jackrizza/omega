//! Numerical experiment, not a performance benchmark or supported resume profile.
//! Compare default and `--features burn/simd` builds using the same inputs/seed.
use burn::tensor::{Tensor, TensorData, backend::Backend};
use omega_nn::{Cpu, GptConfig, token_tensor};

fn main() -> Result<(), String> {
    let device = Default::default();
    // 64 elements exceed Burn's optional SIMD eligibility threshold. Report exact
    // bits, since approximate reciprocal kernels may differ from scalar division.
    let input = vec![3.0_f32; 64];
    let reciprocal = Tensor::<Cpu, 1>::from_data(TensorData::new(input, [64]), &device)
        .recip()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("Cannot read reciprocal: {e:?}"))?;
    println!("scalar_reciprocal_bits={:08x}", 3.0_f32.recip().to_bits());
    println!("tensor_reciprocal_bits={:08x}", reciprocal[0].to_bits());
    Cpu::seed(42);
    let config = GptConfig {
        vocab_size: 16,
        context_length: 8,
        d_model: 8,
        num_heads: 2,
        num_layers: 1,
        d_ff: 16,
    };
    let model = config.init::<Cpu>(&device)?;
    let input = token_tensor::<Cpu>(&[0, 1, 2, 3, 4, 5, 6, 7], 16, 8, &device)?;
    let logits = model
        .forward(input)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("Cannot read logits: {e:?}"))?;
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite kernel-probe logits".into());
    }
    println!(
        "logit_bits={:?}",
        logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}
