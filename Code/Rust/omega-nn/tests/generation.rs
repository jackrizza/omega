use burn::{
    module::{Module, ModuleMapper, ParamId},
    tensor::{Distribution, Tensor, backend::Backend},
};
use omega_nn::{
    Cpu, GenerationOptions, Gpt, GptConfig, SamplingOptions, generate, generate_with_options,
    token_tensor,
};

fn config() -> GptConfig {
    GptConfig {
        vocab_size: 4,
        context_length: 8,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    }
}
fn model() -> Gpt<Cpu> {
    config().init(&Default::default()).unwrap()
}
struct Fill(f32);
impl ModuleMapper<Cpu> for Fill {
    fn map_float<const D: usize>(&mut self, _: ParamId, tensor: Tensor<Cpu, D>) -> Tensor<Cpu, D> {
        Tensor::full(tensor.dims(), self.0, &Default::default())
    }
}

#[test]
fn default_and_greedy_match_original_argmax_including_ties() {
    for model in [model(), model().map(&mut Fill(0.0))] {
        let mut expected = vec![0, 1];
        for _ in 0..4 {
            let sequence = expected.len();
            let logits =
                model.forward(token_tensor::<Cpu>(&expected, 4, 8, &Default::default()).unwrap());
            expected.push(
                logits
                    .slice([0..1, sequence - 1..sequence, 0..4])
                    .argmax(2)
                    .into_scalar() as u32,
            );
        }
        assert_eq!(
            generate(&model, &config(), &[0, 1], 4, None).unwrap(),
            expected
        );
        assert_eq!(
            generate_with_options(
                &model,
                &config(),
                &[0, 1],
                4,
                None,
                &GenerationOptions::default()
            )
            .unwrap(),
            expected
        );
    }
    assert_eq!(
        generate(&model().map(&mut Fill(0.0)), &config(), &[2], 3, None).unwrap(),
        vec![2, 0, 0, 0]
    );
}

#[test]
fn sampled_generation_repeats_preserves_prompt_and_stopping_contracts() {
    let model = model().map(&mut Fill(0.0));
    let options = GenerationOptions::Sample(SamplingOptions::default());
    let expected = vec![3, 2, 0, 1, 1, 0];
    for _ in 0..2 {
        assert_eq!(
            generate_with_options(&model, &config(), &[3], 5, None, &options).unwrap(),
            expected
        );
    }
    let alternate = GenerationOptions::Sample(SamplingOptions {
        seed: 9,
        ..Default::default()
    });
    assert_eq!(
        generate_with_options(&model, &config(), &[3], 5, None, &alternate).unwrap(),
        vec![3, 2, 3, 1, 3, 1]
    );
    assert_eq!(
        generate_with_options(&model, &config(), &[3], 5, Some(2), &options).unwrap(),
        vec![3, 2]
    );
    assert_eq!(
        generate_with_options(&model, &config(), &[3], 5, Some(3), &options).unwrap(),
        expected
    );
    assert_eq!(
        generate_with_options(&model, &config(), &[3], 0, None, &options).unwrap(),
        vec![3]
    );
    assert_eq!(
        generate_with_options(&model, &config(), &[0; 8], 0, None, &options).unwrap(),
        vec![0; 8]
    );
    for (prompt, budget, eos) in [
        (&[][..], 1, None),
        (&[4][..], 1, None),
        (&[0][..], 8, None),
        (&[0][..], usize::MAX, None),
        (&[0][..], 0, Some(4)),
    ] {
        assert!(generate_with_options(&model, &config(), prompt, budget, eos, &options).is_err());
    }
    let changed = GptConfig {
        d_ff: 9,
        ..config()
    };
    assert!(generate_with_options(&model, &changed, &[0], 0, None, &options).is_err());
    let invalid = GenerationOptions::Sample(SamplingOptions {
        top_k: Some(5),
        ..Default::default()
    });
    assert!(generate_with_options(&model, &config(), &[0], 0, None, &invalid).is_err());
}

#[test]
fn nonfinite_model_logits_fail_in_both_modes() {
    let model = model().map(&mut Fill(f32::NAN));
    for options in [
        GenerationOptions::Greedy,
        GenerationOptions::Sample(SamplingOptions::default()),
    ] {
        assert!(
            generate_with_options(&model, &config(), &[0], 1, None, &options)
                .unwrap_err()
                .contains("non-finite logits")
        );
    }
}

#[test]
fn sampling_does_not_touch_backend_rng() {
    // Isolate shared Burn seeding from all other parallel model initialization.
    if std::env::var_os("OMEGA_GENERATION_RNG_PROBE").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "sampling_does_not_touch_backend_rng",
                "--nocapture",
            ])
            .env("OMEGA_GENERATION_RNG_PROBE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let model = model();
    // Burn initializes parameter tensors lazily. Materialize the fresh model
    // before probing sampling, just as loaded/trained models already are.
    let _ = model
        .forward(token_tensor::<Cpu>(&[0], 4, 8, &Default::default()).unwrap())
        .to_data();
    Cpu::seed(123);
    let expected =
        Tensor::<Cpu, 1>::random([8], Distribution::Default, &Default::default()).to_data();
    Cpu::seed(123);
    generate_with_options(
        &model,
        &config(),
        &[0],
        3,
        None,
        &GenerationOptions::Sample(SamplingOptions::default()),
    )
    .unwrap();
    let actual =
        Tensor::<Cpu, 1>::random([8], Distribution::Default, &Default::default()).to_data();
    assert_eq!(actual, expected);
}
