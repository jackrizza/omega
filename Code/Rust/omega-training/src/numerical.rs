//! Compact device reductions for GPU validation. CPU training keeps its original
//! host f64 norm and elementwise checks. These helpers use non-autodiff tensors.
use burn::tensor::{Bool, ElementConversion, Int, Tensor, backend::Backend};

/// The qualified Wgpu backend stores f32/i32 in the same four-byte CubeTensor
/// buffer layout. Change only the view's dtype, retaining its handle, strides,
/// shape and device. Integer kernels inspect IEEE bits without GPU denormal
/// flushing. No unsafe cast, host read, or mutation of the source allocation.
fn float_bits<B: Backend<FloatElem = f32>, const D: usize>(
    tensor: Tensor<B, D>,
) -> Option<Tensor<B, D, Int>> {
    #[cfg(feature = "gpu")]
    if on_device::<B>() {
        use burn::tensor::DType;
        let gpu: Tensor<crate::gpu::Gpu, D> = same_type(tensor);
        let mut raw = gpu.into_primitive().tensor();
        assert_eq!(raw.dtype, DType::F32);
        raw.dtype = DType::I32;
        return Some(same_type(
            Tensor::<crate::gpu::Gpu, D, Int>::from_primitive(raw),
        ));
    }
    let _ = tensor;
    None
}

#[cfg(feature = "gpu")]
fn same_type<T: 'static, U: 'static>(value: T) -> U {
    let value: Box<dyn std::any::Any> = Box::new(value);
    *value
        .downcast::<U>()
        .unwrap_or_else(|_| unreachable!("exact GPU TypeId was checked"))
}

fn bits_float<B: Backend<FloatElem = f32>, const D: usize>(
    tensor: Tensor<B, D, Int>,
) -> Tensor<B, D> {
    #[cfg(feature = "gpu")]
    if on_device::<B>() {
        use burn::tensor::{DType, TensorPrimitive};
        let gpu: Tensor<crate::gpu::Gpu, D, Int> = same_type(tensor);
        let mut raw = gpu.into_primitive();
        assert_eq!(raw.dtype, DType::I32);
        raw.dtype = DType::F32;
        return same_type(Tensor::<crate::gpu::Gpu, D>::from_primitive(
            TensorPrimitive::Float(raw),
        ));
    }
    let _ = tensor;
    unreachable!("bit views are only created for the exact GPU backend")
}

pub(crate) fn on_device<B: Backend>() -> bool {
    let backend = std::any::TypeId::of::<B>();
    #[cfg(feature = "gpu")]
    {
        backend == std::any::TypeId::of::<crate::gpu::Gpu>()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = backend;
        false
    }
}

/// Queue checks without synchronizing each parameter; read one boolean at finish.
pub(crate) struct DeviceChecks<B: Backend> {
    flags: Vec<Tensor<B, 1, Bool>>,
}

impl<B: Backend<FloatElem = f32>> DeviceChecks<B> {
    pub(crate) fn new() -> Self {
        Self { flags: Vec::new() }
    }

    pub(crate) fn add<const D: usize>(&mut self, tensor: Tensor<B, D>, nonnegative: bool) {
        if let Some(bits) = float_bits(tensor.clone()) {
            let magnitude = bits.clone().bitwise_and_scalar(i32::MAX.elem());
            let finite = magnitude.clone().lower_elem(0x7f80_0000_i32);
            let valid = if nonnegative {
                finite.bool_and(bits.greater_equal_elem(0).bool_or(magnitude.equal_elem(0)))
            } else {
                finite
            };
            self.flags.push(valid.all());
            return;
        }
        // Ordered comparison rejects NaN as well as either infinity. Unlike a
        // sum-based check, finite large values cannot overflow into a false error.
        let finite = tensor.clone().abs().lower_equal_elem(f32::MAX);
        let valid = if nonnegative {
            finite.bool_and(tensor.greater_equal_elem(0.0))
        } else {
            finite
        };
        self.flags.push(valid.all());
    }

    pub(crate) fn finish(self) -> bool {
        self.flags.is_empty()
            || Tensor::cat(self.flags, 0)
                .all()
                .into_scalar()
                .elem::<bool>()
    }
}

/// Each tensor contributes three f32 scalars: finite flag, absolute maximum, and
/// scaled sum of squares. Scaling avoids f32 square overflow for large finite
/// gradients. Host f64 accumulation combines only these small summaries; it is
/// intentionally not bitwise identical to the legacy host elementwise norm.
pub(crate) struct DeviceNorm<B: Backend> {
    summaries: Vec<Tensor<B, 1>>,
}

impl<B: Backend<FloatElem = f32>> DeviceNorm<B> {
    pub(crate) fn new() -> Self {
        Self {
            summaries: Vec::new(),
        }
    }

    pub(crate) fn add<const D: usize>(&mut self, tensor: Tensor<B, D>) {
        let flat = tensor.flatten::<1>(0, D - 1);
        if let Some(bits) = float_bits(flat.clone()) {
            let magnitude = bits.bitwise_and_scalar(i32::MAX.elem());
            let finite = magnitude.clone().lower_elem(0x7f80_0000_i32).all().float();
            let max_bits = magnitude.clone().max();
            let scale = bits_float(max_bits.clone());
            let high = 2.0_f32.powi(64);
            let low = 2.0_f32.powi(-64);
            let factor = scale
                .ones_like()
                .mask_fill(max_bits.clone().greater_elem((high.to_bits()) as i32), low)
                .mask_fill(max_bits.clone().lower_elem((low.to_bits()) as i32), high);
            // Reconstruct subnormal magnitudes from their integer mantissa,
            // already multiplied by 2^64. Arithmetic never consumes a denormal.
            let tiny = magnitude.clone().lower_elem(0x0080_0000_i32);
            let tiny_values = magnitude.float() * 2.0_f32.powi(-85);
            let scaled = (flat.abs() * factor.clone()).mask_where(tiny, tiny_values);
            let tiny_scale = max_bits.clone().float() * 2.0_f32.powi(-85);
            let divisor = (scale.clone() * factor)
                .mask_where(max_bits.clone().lower_elem(0x0080_0000_i32), tiny_scale)
                .mask_fill(max_bits.equal_elem(0), 1.0);
            let normalized = scaled / divisor;
            // Burn's global sum defaults to cross-workgroup atomic addition;
            // its order can change after reload. A single-axis reduction uses
            // a fixed tree for this shape instead, preserving exact resume.
            let squares = (normalized.clone() * normalized).sum_dim(0);
            self.summaries
                .push(Tensor::cat(vec![finite, scale, squares], 0));
            return;
        }
        let finite = flat.clone().abs().lower_equal_elem(f32::MAX).all().float();
        let scale = flat.clone().abs().max();
        // Some GPU division kernels use a reciprocal. Keep that reciprocal in
        // the normal f32 range even for f32::MAX or very small gradients.
        // Powers of two avoid introducing an extra mantissa rounding step.
        let high = 2.0_f32.powi(64);
        let low = 2.0_f32.powi(-64);
        let factor = scale
            .ones_like()
            .mask_fill(scale.clone().greater_elem(high), low)
            .mask_fill(scale.clone().lower_elem(low), high);
        let divisor =
            (scale.clone() * factor.clone()).mask_fill(scale.clone().equal_elem(0.0), 1.0);
        let normalized = (flat * factor) / divisor;
        let squares = (normalized.clone() * normalized).sum_dim(0);
        self.summaries
            .push(Tensor::cat(vec![finite, scale, squares], 0));
    }

    pub(crate) fn finish(self) -> Result<f64, String> {
        if self.summaries.is_empty() {
            return Ok(0.0);
        }
        let summaries = Tensor::cat(self.summaries, 0).into_data();
        let values = summaries
            .to_vec::<f32>()
            .map_err(|e| format!("Cannot read gradient summaries: {e:?}"))?;
        let mut sum = 0.0_f64;
        for part in values.chunks_exact(3) {
            let [finite, scale, squares] = [part[0], part[1], part[2]];
            if finite != 1.0 {
                return Err("Non-finite parameter gradient; update rejected".into());
            }
            if !scale.is_finite()
                || !squares.is_finite()
                || scale < 0.0
                || squares < 0.0
                || (scale > 0.0 && squares == 0.0)
            {
                return Err("Invalid device gradient norm reduction; update rejected".into());
            }
            sum += f64::from(scale).powi(2) * f64::from(squares);
        }
        let norm = sum.sqrt();
        if !norm.is_finite() {
            return Err("Non-finite global gradient norm".into());
        }
        Ok(norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_nn::Cpu;

    fn reductions<B: Backend<FloatElem = f32>>(device: &B::Device) {
        for values in [
            vec![0.0, -0.0, 0.0],
            vec![3.0, -4.0, 12.0],
            vec![1e30, -1e30, f32::MAX],
            vec![1e-30, -2e-30, 3e-30],
            vec![f32::from_bits(1), -f32::from_bits(2), 1e-40],
        ] {
            let expected = values
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            let mut norm = DeviceNorm::<B>::new();
            norm.add(Tensor::<B, 1>::from_data(values.as_slice(), device));
            let actual = norm.finish().unwrap();
            assert!(
                (actual - expected).abs() <= expected * 2e-5,
                "{actual} != {expected}"
            );
        }
        // A bad value must be found regardless of its position in a reduction.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, 1, 256, 1024] {
                let mut values = vec![1.0_f32; 1025];
                values[index] = bad;
                let tensor = Tensor::<B, 1>::from_data(values.as_slice(), device);
                let mut checks = DeviceChecks::new();
                checks.add(tensor.clone(), false);
                assert!(!checks.finish());
                let mut norm = DeviceNorm::new();
                norm.add(tensor);
                assert!(
                    norm.finish()
                        .unwrap_err()
                        .contains("Non-finite parameter gradient")
                );
            }
        }
        for negative in [-1.0_f32, -f32::MIN_POSITIVE, -f32::from_bits(1)] {
            let mut checks = DeviceChecks::<B>::new();
            checks.add(Tensor::<B, 1>::from_data([0.0, negative], device), true);
            assert!(!checks.finish(), "accepted negative moment {negative}");
        }
        let mut checks = DeviceChecks::<B>::new();
        checks.add(
            Tensor::<B, 1>::from_data([f32::MAX, -f32::MAX], device),
            false,
        );
        checks.add(
            Tensor::<B, 1>::from_data([0.0, -0.0, f32::MAX], device),
            true,
        );
        assert!(checks.finish());

        let values: Vec<f32> = (0..8193)
            .map(|i| ((i * 29 % 193) as f32 - 96.0) / 31.0)
            .collect();
        let tensor = Tensor::<B, 1>::from_data(values.as_slice(), device);
        let expected = (values.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() + 25.0).sqrt();
        let mut norm = DeviceNorm::new();
        norm.add(tensor.clone());
        // A transposed tensor also verifies that bit views preserve strides.
        norm.add(Tensor::<B, 2>::from_data([[3.0, 0.0], [0.0, -4.0]], device).transpose());
        assert!((norm.finish().unwrap() - expected).abs() < expected * 2e-5);
        assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), values);
    }

    #[test]
    fn compact_reductions_cpu_reference() {
        reductions::<Cpu>(&Default::default());
        assert!(!on_device::<Cpu>());
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires qualified Vulkan GPU"]
    fn compact_reductions_gpu() {
        let selected = crate::gpu::initialize_vulkan(0).unwrap();
        assert!(on_device::<crate::gpu::Gpu>());
        reductions::<crate::gpu::Gpu>(selected.device());
    }
}
