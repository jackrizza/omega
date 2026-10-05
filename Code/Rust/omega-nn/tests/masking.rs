use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, ParamId},
    nn::loss::CrossEntropyLossConfig,
    optim::GradientsParams,
    tensor::{Bool, Int, Tensor, TensorData},
};
use omega_nn::{Cpu, GptConfig, TrainingBackend, masked_cross_entropy};

fn config() -> GptConfig {
    GptConfig {
        vocab_size: 8,
        context_length: 5,
        d_model: 8,
        num_heads: 2,
        num_layers: 2,
        d_ff: 16,
    }
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-5,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn padded_batch_matches_independent_unpadded_sequences_and_zeros_padding() {
    let device = Default::default();
    let model = config().init::<Cpu>(&device).unwrap();
    let tokens = Tensor::<Cpu, 2, Int>::from_data([[1, 2, 7, 7], [3, 4, 5, 6]], &device);
    let mask = Tensor::<Cpu, 2, Bool>::from_data([[true, true, false, false], [true; 4]], &device);
    let padded = model.forward_masked(tokens.clone(), mask.clone()).unwrap();
    assert_eq!(padded.dims(), [2, 4, 8]);
    for (row, length) in [(0, 2), (1, 4)] {
        let unpadded = model.forward(tokens.clone().slice([row..row + 1, 0..length]));
        let actual = padded.clone().slice([row..row + 1, 0..length, 0..8]);
        close(
            &actual.into_data().to_vec::<f32>().unwrap(),
            &unpadded.into_data().to_vec::<f32>().unwrap(),
        );
    }
    let padding = padded
        .clone()
        .slice([0..1, 2..4, 0..8])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(padding, vec![0.0; 16]);
    let changed = tokens
        .clone()
        .slice_assign([0..1, 2..4], Tensor::zeros([1, 2], &device));
    close(
        &padded.into_data().to_vec::<f32>().unwrap(),
        &model
            .forward_masked(changed, mask)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );

    let all_valid = Tensor::<Cpu, 2, Bool>::from_data([[true; 4]; 2], &device);
    close(
        &model
            .forward(tokens.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        &model
            .forward_masked(tokens, all_valid)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
}

#[test]
fn padded_forward_remains_causal() {
    let device = Default::default();
    let model = config().init::<Cpu>(&device).unwrap();
    let tokens = Tensor::<Cpu, 2, Int>::from_data([[1, 2, 3, 4, 7]], &device);
    let valid = Tensor::<Cpu, 2, Bool>::from_data([[true, true, true, true, false]], &device);
    let expected = model.forward_masked(tokens.clone(), valid.clone()).unwrap();
    for prefix in 1..4 {
        let changed = tokens
            .clone()
            .slice_assign([0..1, prefix..4], Tensor::zeros([1, 4 - prefix], &device));
        let actual = model.forward_masked(changed, valid.clone()).unwrap();
        close(
            &actual
                .slice([0..1, 0..prefix, 0..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            &expected
                .clone()
                .slice([0..1, 0..prefix, 0..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
        );
    }
}

#[test]
fn invalid_forward_masks_shapes_ids_and_nonfinite_models_return_errors() {
    let device = Default::default();
    let model = config().init::<Cpu>(&device).unwrap();
    let tokens = Tensor::<Cpu, 2, Int>::from_data([[1, 2, 3]], &device);
    for mask in [[false; 3], [false, true, true], [true, false, true]] {
        let error = model
            .forward_masked(tokens.clone(), Tensor::from_data([mask], &device))
            .unwrap_err();
        assert!(error.contains("nonempty valid prefix"), "{error}");
    }
    assert!(
        model
            .forward_masked(tokens, Tensor::from_data([[true; 2]], &device))
            .unwrap_err()
            .contains("shapes")
    );
    for shape in [[0, 1], [1, 0], [1, 6]] {
        assert!(
            model
                .forward_masked(
                    Tensor::zeros(shape, &device),
                    Tensor::from_data(
                        TensorData::new(vec![true; shape[0] * shape[1]], shape),
                        &device
                    )
                )
                .is_err()
        );
    }
    for ids in [[-1, 0], [8, 0], [0, -1], [0, 8]] {
        assert!(
            model
                .forward_masked(
                    Tensor::from_data([ids], &device),
                    Tensor::from_data([[true, false]], &device)
                )
                .unwrap_err()
                .contains("outside the model vocabulary")
        );
    }
    let tokens = Tensor::from_data([[0, 1], [0, 1]], &device);
    assert!(
        model
            .forward_masked(tokens, Tensor::from_data([[true; 2], [false; 2]], &device))
            .is_err()
    );

    struct Nonfinite;
    impl ModuleMapper<Cpu> for Nonfinite {
        fn map_float<const D: usize>(
            &mut self,
            _id: ParamId,
            tensor: Tensor<Cpu, D>,
        ) -> Tensor<Cpu, D> {
            tensor.mul_scalar(f32::NAN)
        }
    }
    let broken = model.map(&mut Nonfinite);
    assert!(
        broken
            .forward_masked(
                Tensor::from_data([[0]], &device),
                Tensor::from_data([[true]], &device)
            )
            .unwrap_err()
            .contains("non-finite logits")
    );
}

#[test]
fn ignored_targets_and_nonfinite_logits_have_zero_gradient_and_no_loss_effect() {
    let device = Default::default();
    let log_three = 3_f32.ln();
    let logits = Tensor::<TrainingBackend, 3>::from_data(
        [
            [
                [0.0, log_three],
                [f32::NAN, f32::INFINITY],
                [log_three, 0.0],
            ],
            [[f32::NEG_INFINITY, f32::NAN], [f32::NAN; 2], [f32::NAN; 2]],
        ],
        &device,
    )
    .require_grad();
    let result = masked_cross_entropy(
        logits.clone(),
        Tensor::from_data([[1, -999, 0], [i64::MAX, -1, 200]], &device),
        Tensor::from_data([[true, false, true], [false; 3]], &device),
    )
    .unwrap();
    assert_eq!(result.target_count, 2);
    assert!((result.loss.clone().into_scalar() - (4_f32 / 3.0).ln()).abs() < 1e-6);
    let gradients = result.loss.backward();
    let values = logits
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    close(
        &values,
        &[
            0.125, -0.125, 0.0, 0.0, -0.125, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    assert_eq!(&values[2..4], &[0.0, 0.0]);
    assert_eq!(&values[6..], &[0.0; 6]);
}

#[test]
fn all_valid_loss_and_gradients_match_existing_cross_entropy() {
    let device = Default::default();
    let logits = Tensor::<TrainingBackend, 3>::from_data(
        [
            [[0.0, 1.0, 2.0], [3.0, 0.0, -1.0]],
            [[1.0, 2.0, 0.0], [2.0, 0.0, 1.0]],
        ],
        &device,
    )
    .require_grad();
    let targets = Tensor::from_data([[2, 0], [1, 2]], &device);
    let masked = masked_cross_entropy(
        logits.clone(),
        targets.clone(),
        Tensor::from_data([[true; 2]; 2], &device),
    )
    .unwrap();
    let expected = CrossEntropyLossConfig::new()
        .init::<TrainingBackend>(&device)
        .forward(logits.clone().reshape([4, 3]), targets.reshape([4]));
    assert_eq!(masked.target_count, 4);
    assert_eq!(
        masked.loss.clone().into_scalar(),
        expected.clone().into_scalar()
    );
    let masked_grads = masked.loss.backward();
    let expected_grads = expected.backward();
    close(
        &logits
            .grad(&masked_grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        &logits
            .grad(&expected_grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
}

#[test]
fn masked_forward_and_loss_remain_differentiable_together() {
    let device = Default::default();
    let model = config().init::<TrainingBackend>(&device).unwrap();
    let logits = model
        .forward_masked(
            Tensor::from_data([[1, 2, 0], [3, 4, 5]], &device),
            Tensor::from_data([[true, true, false], [true; 3]], &device),
        )
        .unwrap();
    let loss = masked_cross_entropy(
        logits,
        Tensor::from_data([[2, 3, -1], [4, 5, 6]], &device),
        Tensor::from_data([[true, true, false], [true; 3]], &device),
    )
    .unwrap();
    assert_eq!(loss.target_count, 5);
    let gradients = burn::optim::GradientsParams::from_grads(loss.loss.backward(), &model);
    assert!(!gradients.is_empty());
}

#[test]
fn padded_parameter_gradients_match_target_weighted_unpadded_gradients() {
    let device = Default::default();
    // A distinct vocabulary width makes the token embedding identifiable by
    // shape. ID 8 occurs only as padding, never as a real input or target.
    let config = GptConfig {
        vocab_size: 9,
        ..config()
    };
    let padded_model = config.init::<TrainingBackend>(&device).unwrap();
    let unpadded_model = padded_model.clone();
    let padded_logits = padded_model
        .forward_masked(
            Tensor::from_data([[1, 2, 8, 8], [3, 4, 5, 8]], &device),
            Tensor::from_data(
                [[true, true, false, false], [true, true, true, false]],
                &device,
            ),
        )
        .unwrap();
    let padded_loss = masked_cross_entropy(
        padded_logits,
        Tensor::from_data([[2, 3, -1, -1], [4, 5, 6, -1]], &device),
        Tensor::from_data(
            [[true, true, false, false], [true, true, true, false]],
            &device,
        ),
    )
    .unwrap();
    assert_eq!(padded_loss.target_count, 5);

    let criterion = CrossEntropyLossConfig::new().init::<TrainingBackend>(&device);
    let first = criterion.forward(
        unpadded_model
            .forward(Tensor::from_data([[1, 2]], &device))
            .reshape([2, 9]),
        Tensor::from_data([2, 3], &device),
    );
    let second = criterion.forward(
        unpadded_model
            .forward(Tensor::from_data([[3, 4, 5]], &device))
            .reshape([3, 9]),
        Tensor::from_data([4, 5, 6], &device),
    );
    // Each separate CE is a mean: combine by target count, not batch count.
    let expected_loss = (first.mul_scalar(2.0) + second.mul_scalar(3.0)).div_scalar(5.0);
    close(
        &[padded_loss.loss.clone().into_scalar()],
        &[expected_loss.clone().into_scalar()],
    );
    let actual = GradientsParams::from_grads(padded_loss.loss.backward(), &padded_model);
    let expected = GradientsParams::from_grads(expected_loss.backward(), &unpadded_model);

    struct CompareGradients<'a> {
        actual: &'a GradientsParams,
        expected: &'a GradientsParams,
        checked: usize,
        checked_padding_embedding: bool,
    }
    impl ModuleVisitor<TrainingBackend> for CompareGradients<'_> {
        fn visit_float<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: &Tensor<TrainingBackend, D>,
        ) {
            let actual = self
                .actual
                .get::<Cpu, D>(id)
                .expect("padded parameter gradient");
            let expected = self
                .expected
                .get::<Cpu, D>(id)
                .expect("unpadded parameter gradient");
            let actual = actual.into_data().to_vec::<f32>().unwrap();
            let expected = expected.into_data().to_vec::<f32>().unwrap();
            close(&actual, &expected);
            if tensor.dims().as_slice() == [9, 8] {
                assert_eq!(&actual[8 * 8..9 * 8], &[0.0; 8]);
                assert_eq!(&expected[8 * 8..9 * 8], &[0.0; 8]);
                self.checked_padding_embedding = true;
            }
            self.checked += 1;
        }
    }
    let mut comparison = CompareGradients {
        actual: &actual,
        expected: &expected,
        checked: 0,
        checked_padding_embedding: false,
    };
    padded_model.visit(&mut comparison);
    assert_eq!(comparison.checked, actual.len());
    assert_eq!(comparison.checked, expected.len());
    assert!(comparison.checked > 0);
    assert!(comparison.checked_padding_embedding);
}

#[test]
fn invalid_loss_shapes_targets_masks_and_finite_values_are_rejected() {
    let device = Default::default();
    for shape in [[0, 1, 2], [1, 0, 2], [1, 1, 0]] {
        let [batch, sequence, _] = shape;
        assert!(
            masked_cross_entropy(
                Tensor::<Cpu, 3>::zeros(shape, &device),
                Tensor::zeros([batch, sequence], &device),
                Tensor::from_data(
                    TensorData::new(vec![true; batch * sequence], [batch, sequence]),
                    &device
                )
            )
            .is_err()
        );
    }
    let logits = Tensor::<Cpu, 3>::zeros([1, 2, 3], &device);
    assert!(
        masked_cross_entropy(
            logits.clone(),
            Tensor::from_data([[0]], &device),
            Tensor::from_data([[true; 2]], &device)
        )
        .unwrap_err()
        .contains("shapes")
    );
    assert!(
        masked_cross_entropy(
            logits.clone(),
            Tensor::from_data([[0, 1]], &device),
            Tensor::from_data([[true]], &device)
        )
        .unwrap_err()
        .contains("shapes")
    );
    assert!(
        masked_cross_entropy(
            logits.clone(),
            Tensor::from_data([[-99, 99]], &device),
            Tensor::from_data([[false; 2]], &device)
        )
        .unwrap_err()
        .contains("at least one valid target")
    );
    for target in [-1, 3] {
        assert!(
            masked_cross_entropy(
                logits.clone(),
                Tensor::from_data([[target, 0]], &device),
                Tensor::from_data([[true; 2]], &device)
            )
            .unwrap_err()
            .contains("outside vocabulary")
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            masked_cross_entropy(
                Tensor::<Cpu, 3>::from_data([[[value, 0.0]]], &device),
                Tensor::from_data([[0]], &device),
                Tensor::from_data([[true]], &device)
            )
            .unwrap_err()
            .contains("non-finite logits")
        );
    }
    assert!(
        masked_cross_entropy(
            Tensor::<Cpu, 3>::from_data([[[f32::MAX, -f32::MAX]]], &device),
            Tensor::from_data([[1]], &device),
            Tensor::from_data([[true]], &device)
        )
        .unwrap_err()
        .contains("non-finite")
    );
}
