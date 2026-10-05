//! Global gradient clipping and update-count based linear warmup.
use burn::{
    module::{Module, ModuleVisitor, ParamId},
    optim::GradientsParams,
    tensor::{Tensor, backend::AutodiffBackend},
};
use omega_nn::Gpt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizationOptions {
    pub gradient_clip_norm: Option<f64>,
    pub warmup_updates: usize,
}

impl OptimizationOptions {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(norm) = self.gradient_clip_norm
            && (!norm.is_finite() || norm <= 0.0)
        {
            return Err("gradient_clip_norm must be finite and positive".into());
        }
        Ok(())
    }

    /// First warmup update uses base/warmup; the last uses base. No accumulation:
    /// each committed minibatch is one microstep and one optimizer update.
    pub fn learning_rate(&self, base: f64, completed_updates: usize) -> Result<f64, String> {
        self.validate()?;
        if !base.is_finite() || base <= 0.0 {
            return Err("Base learning rate must be finite and positive".into());
        }
        let next = completed_updates
            .checked_add(1)
            .ok_or("Training update count overflows usize")?;
        let rate = if self.warmup_updates == 0 || next >= self.warmup_updates {
            base
        } else {
            base * (next as f64 / self.warmup_updates as f64)
        };
        if !rate.is_finite() || rate <= 0.0 {
            return Err("Scheduled learning rate is not finite and positive".into());
        }
        Ok(rate)
    }
}

/// Validate every model gradient before optionally scaling by one global factor.
/// CPU host f64 accumulation and scaled GPU reductions avoid squared-norm
/// overflow. Disabled clipping never changes the optimizer's gradient inputs.
pub(crate) fn prepare_gradients<B: AutodiffBackend<FloatElem = f32>>(
    model: &Gpt<B>,
    gradients: &mut GradientsParams,
    options: &OptimizationOptions,
) -> Result<(f64, bool), String> {
    options.validate()?;
    struct Inspect<'a, B: AutodiffBackend<FloatElem = f32>> {
        gradients: &'a GradientsParams,
        sum: f64,
        count: usize,
        error: Option<String>,
        device_norm: Option<crate::numerical::DeviceNorm<B::InnerBackend>>,
    }
    impl<B: AutodiffBackend<FloatElem = f32>> ModuleVisitor<B> for Inspect<'_, B> {
        fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
            self.count += 1;
            let Some(gradient) = self.gradients.get::<B::InnerBackend, D>(id) else {
                self.error = Some("Missing model parameter gradient".into());
                return;
            };
            if gradient.dims() != tensor.dims() {
                self.error = Some("Gradient shape mismatch".into());
                return;
            }
            if let Some(norm) = &mut self.device_norm {
                norm.add(gradient);
                return;
            }
            for value in gradient.to_data().iter::<f32>() {
                if !value.is_finite() {
                    self.error = Some("Non-finite parameter gradient; update rejected".into());
                    return;
                }
                self.sum += f64::from(value) * f64::from(value);
            }
        }
    }
    let mut inspect = Inspect::<B> {
        gradients,
        sum: 0.0,
        count: 0,
        error: None,
        device_norm: crate::numerical::on_device::<B::InnerBackend>()
            .then(crate::numerical::DeviceNorm::new),
    };
    model.visit(&mut inspect);
    if let Some(error) = inspect.error {
        return Err(error);
    }
    if inspect.count != gradients.len() {
        return Err("Gradient parameter IDs do not match model".into());
    }
    let norm = match inspect.device_norm {
        Some(device) => device.finish()?,
        None => inspect.sum.sqrt(),
    };
    if !norm.is_finite() {
        return Err("Non-finite global gradient norm".into());
    }
    let clipped = options.gradient_clip_norm.is_some_and(|limit| norm > limit);
    if clipped {
        struct Scale<'a> {
            gradients: &'a mut GradientsParams,
            factor: f64,
        }
        impl<B: AutodiffBackend<FloatElem = f32>> ModuleVisitor<B> for Scale<'_> {
            fn visit_float<const D: usize>(&mut self, id: ParamId, _: &Tensor<B, D>) {
                let gradient = self
                    .gradients
                    .remove::<B::InnerBackend, D>(id)
                    .expect("validated gradient");
                let scaled = if crate::numerical::on_device::<B::InnerBackend>()
                    && self.factor > 0.0
                    && self.factor < f64::from(f32::MIN_POSITIVE)
                {
                    // Keep a tiny clip multiplier out of the GPU's denormal
                    // range. The two powers cancel without changing the norm
                    // target; neither intermediate overflows for f32 inputs.
                    gradient
                        .mul_scalar(2.0_f32.powi(-64))
                        .mul_scalar(self.factor * 2.0_f64.powi(64))
                } else {
                    gradient.mul_scalar(self.factor)
                };
                self.gradients.register(id, scaled);
            }
        }
        model.visit(&mut Scale {
            gradients,
            factor: options.gradient_clip_norm.unwrap() / norm,
        });
    }
    Ok((norm, clipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_nn::{Cpu, GptConfig, TrainingBackend};

    fn model() -> Gpt<TrainingBackend> {
        GptConfig {
            vocab_size: 4,
            context_length: 3,
            d_model: 4,
            num_heads: 1,
            num_layers: 1,
            d_ff: 8,
        }
        .init(&Default::default())
        .unwrap()
    }
    pub(crate) fn gradients(model: &Gpt<TrainingBackend>, value: f32) -> GradientsParams {
        struct Fill {
            result: GradientsParams,
            value: f32,
        }
        impl ModuleVisitor<TrainingBackend> for Fill {
            fn visit_float<const D: usize>(
                &mut self,
                id: ParamId,
                tensor: &Tensor<TrainingBackend, D>,
            ) {
                self.result.register(
                    id,
                    Tensor::<Cpu, D>::full(tensor.dims(), self.value, &Default::default()),
                );
            }
        }
        let mut fill = Fill {
            result: GradientsParams::new(),
            value,
        };
        model.visit(&mut fill);
        fill.result
    }
    #[test]
    fn global_clip_and_nonfinite_gradient_checks() {
        let model = model();
        let mut raw = gradients(&model, 2.0);
        let (norm, clipped) =
            prepare_gradients(&model, &mut raw, &OptimizationOptions::default()).unwrap();
        assert!(norm > 2.0);
        assert!(!clipped);
        let options = OptimizationOptions {
            gradient_clip_norm: Some(1.0),
            warmup_updates: 0,
        };
        let (before, clipped) = prepare_gradients(&model, &mut raw, &options).unwrap();
        assert_eq!(before, norm);
        assert!(clipped);
        let (after, _) =
            prepare_gradients(&model, &mut raw, &OptimizationOptions::default()).unwrap();
        assert!((after - 1.0).abs() < 1e-6);
        let mut large = gradients(&model, 1e30);
        assert!(
            prepare_gradients(&model, &mut large, &options)
                .unwrap()
                .0
                .is_finite()
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                prepare_gradients(&model, &mut gradients(&model, value), &options)
                    .unwrap_err()
                    .contains("Non-finite")
            );
        }
        assert!(prepare_gradients(&model, &mut GradientsParams::new(), &options).is_err());
    }
    #[test]
    fn linear_warmup_and_invalid_settings() {
        let options = OptimizationOptions {
            gradient_clip_norm: None,
            warmup_updates: 4,
        };
        assert_eq!(
            (0..6)
                .map(|i| options.learning_rate(0.04, i).unwrap())
                .collect::<Vec<_>>(),
            vec![0.01, 0.02, 0.03, 0.04, 0.04, 0.04]
        );
        assert_eq!(
            OptimizationOptions::default()
                .learning_rate(0.04, 0)
                .unwrap(),
            0.04
        );
        for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                OptimizationOptions {
                    gradient_clip_norm: Some(limit),
                    warmup_updates: 0
                }
                .validate()
                .is_err()
            );
        }
        assert!(options.learning_rate(0.1, usize::MAX).is_err());
        assert!(options.learning_rate(f64::from_bits(1), 0).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires qualified Vulkan GPU"]
    fn gpu_gradient_clipping_and_rejection() {
        use crate::gpu::{Gpu, GpuTraining, initialize_vulkan};
        let selected = initialize_vulkan(0).unwrap();
        let config = GptConfig {
            vocab_size: 4,
            context_length: 3,
            d_model: 4,
            num_heads: 1,
            num_layers: 1,
            d_ff: 8,
        };
        let model = config.init::<GpuTraining>(selected.device()).unwrap();
        struct Fill {
            result: GradientsParams,
            value: f32,
        }
        impl ModuleVisitor<GpuTraining> for Fill {
            fn visit_float<const D: usize>(
                &mut self,
                id: ParamId,
                tensor: &Tensor<GpuTraining, D>,
            ) {
                self.result.register(
                    id,
                    Tensor::<Gpu, D>::full(tensor.dims(), self.value, &tensor.device()),
                );
            }
        }
        let make = |value| {
            let mut fill = Fill {
                result: GradientsParams::new(),
                value,
            };
            model.visit(&mut fill);
            fill.result
        };
        let options = OptimizationOptions {
            gradient_clip_norm: Some(1.0),
            warmup_updates: 0,
        };
        for value in [2.0, 1e30, f32::MAX] {
            let mut gradients = make(value);
            let (before, clipped) = prepare_gradients(&model, &mut gradients, &options).unwrap();
            assert!(before > 1.0 && before.is_finite() && clipped);
            let (after, _) =
                prepare_gradients(&model, &mut gradients, &OptimizationOptions::default()).unwrap();
            assert!((after - 1.0).abs() < 2e-5, "{after}");
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(prepare_gradients(&model, &mut make(bad), &options).is_err());
        }
        assert!(prepare_gradients(&model, &mut GradientsParams::new(), &options).is_err());
        let (norm, clipped) = prepare_gradients(&model, &mut make(0.0), &options).unwrap();
        assert_eq!(norm, 0.0);
        assert!(!clipped);
    }
}
