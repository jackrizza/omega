//! Bounded hardware qualification: run the built binary under an external timeout.
use burn::{
    module::{AutodiffModule, Module, ModuleVisitor, ParamId},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use omega_nn::{Cpu, Gpt, GptConfig, TrainingBackend, masked_cross_entropy};
use omega_training::gpu::{Gpu, GpuTraining, initialize_vulkan, list_vulkan_devices};

fn main() -> Result<(), String> {
    let index = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0".into())
        .parse::<usize>()
        .map_err(|e| e.to_string())?;
    println!("adapters={:?}", list_vulkan_devices());
    let selected = initialize_vulkan(index)?;
    println!(
        "selected={:?}; tolerances: logits/loss/gradients/updates abs=0.003 rel=0.003",
        selected.adapter()
    );
    selected.profile().validate()?;
    if initialize_vulkan(usize::MAX).is_ok() {
        return Err("Invalid GPU index accepted".into());
    }
    if initialize_vulkan(index)?.device() != selected.device() {
        return Err("GPU initialization is not stable".into());
    }
    let device = selected.device();
    let config = GptConfig {
        vocab_size: 16,
        context_length: 4,
        d_model: 8,
        num_heads: 2,
        num_layers: 1,
        d_ff: 16,
    };
    let cpu = config.init::<Cpu>(&Default::default())?;
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = <_ as Recorder<Cpu>>::record(&recorder, cpu.clone().into_record(), ())
        .map_err(|e| e.to_string())?;
    let cpu_record =
        <_ as Recorder<TrainingBackend>>::load(&recorder, bytes.clone(), &Default::default())
            .map_err(|e| e.to_string())?;
    let mut cpu_model: Gpt<TrainingBackend> =
        config.init(&Default::default())?.load_record(cpu_record);
    let mut cpu_optimizer = AdamConfig::new().init();
    let record =
        <_ as Recorder<GpuTraining>>::load(&recorder, bytes, device).map_err(|e| e.to_string())?;
    let mut model: Gpt<GpuTraining> = config.init(device)?.load_record(record);
    let ids = TensorData::new(vec![0i32, 1, 2, 3, 4, 5, 0, 0], [2, 4]);
    let mask = TensorData::new(
        vec![true, true, true, true, true, true, false, false],
        [2, 4],
    );
    let cpu_logits = cpu.forward_masked(
        Tensor::from_data(ids.clone(), &Default::default()),
        Tensor::from_data(mask.clone(), &Default::default()),
    )?;
    let input = Tensor::<GpuTraining, 2, Int>::from_data(ids, device);
    let valid = Tensor::<GpuTraining, 2, Bool>::from_data(mask, device);
    let before = model.forward_masked(input.clone(), valid.clone())?;
    compare(
        &cpu_logits.to_data().to_vec::<f32>().unwrap(),
        &before.to_data().to_vec::<f32>().unwrap(),
    )?;
    // Perturb a future token and padding only: earlier valid logits must be invariant.
    let altered = Tensor::<GpuTraining, 2, Int>::from_data(
        TensorData::new(vec![0i32, 1, 2, 9, 4, 5, 10, 11], [2, 4]),
        device,
    );
    let altered_logits = model
        .forward_masked(altered, valid.clone())?
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let original_logits = before.to_data().to_vec::<f32>().unwrap();
    compare(&original_logits[..3 * 16], &altered_logits[..3 * 16])?;
    compare(
        &original_logits[4 * 16..6 * 16],
        &altered_logits[4 * 16..6 * 16],
    )?;
    let mut optimizer = AdamConfig::new().init();
    let mut gradient_tensors = 0;
    for step in 0..2 {
        let logits = model.forward_masked(input.clone(), valid.clone())?;
        let targets = Tensor::<GpuTraining, 2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 0, 0], [2, 4]),
            device,
        );
        let loss = masked_cross_entropy(logits, targets, valid.clone())?.loss;
        let cpu_logits = cpu_model.forward_masked(
            Tensor::from_data(input.to_data(), &Default::default()),
            Tensor::from_data(valid.to_data(), &Default::default()),
        )?;
        let cpu_loss = masked_cross_entropy(
            cpu_logits,
            Tensor::from_data(
                TensorData::new(vec![1i32, 2, 3, 4, 5, 6, 0, 0], [2, 4]),
                &Default::default(),
            ),
            Tensor::from_data(valid.to_data(), &Default::default()),
        )?
        .loss;
        compare(
            &[cpu_loss.clone().into_scalar()],
            &[loss.clone().into_scalar()],
        )?;
        println!("step={step} loss={}", loss.clone().into_scalar());
        let cpu_gradients = GradientsParams::from_grads(cpu_loss.backward(), &cpu_model);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        struct Inspect<'a> {
            gradients: &'a GradientsParams,
            cpu_gradients: &'a GradientsParams,
            count: usize,
            valid: bool,
        }
        impl ModuleVisitor<GpuTraining> for Inspect<'_> {
            fn visit_float<const D: usize>(
                &mut self,
                id: ParamId,
                _tensor: &Tensor<GpuTraining, D>,
            ) {
                match self.gradients.get::<Gpu, D>(id) {
                    Some(gradient) => {
                        self.count += 1;
                        self.valid &= gradient.to_data().iter::<f32>().all(f32::is_finite);
                        match self.cpu_gradients.get::<Cpu, D>(id) {
                            Some(cpu_gradient) => {
                                self.valid &= compare(
                                    &cpu_gradient.to_data().to_vec::<f32>().unwrap(),
                                    &gradient.to_data().to_vec::<f32>().unwrap(),
                                )
                                .is_ok()
                            }
                            None => self.valid = false,
                        }
                    }
                    None => self.valid = false,
                }
            }
        }
        let mut inspect = Inspect {
            gradients: &gradients,
            cpu_gradients: &cpu_gradients,
            count: 0,
            valid: true,
        };
        model.visit(&mut inspect);
        if !inspect.valid || inspect.count == 0 {
            return Err("Missing/non-finite GPU gradients or CPU/GPU gradient mismatch".into());
        }
        gradient_tensors = inspect.count;
        model = optimizer.step(0.001, model, gradients);
        cpu_model = cpu_optimizer.step(0.001, cpu_model, cpu_gradients);
        let bytes = <_ as Recorder<GpuTraining>>::record(&recorder, optimizer.to_record(), ())
            .map_err(|e| e.to_string())?;
        optimizer = optimizer.load_record(
            <_ as Recorder<GpuTraining>>::load(&recorder, bytes, device)
                .map_err(|e| e.to_string())?,
        );
    }
    let gpu_logits = model
        .forward_masked(input.clone(), valid.clone())?
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    let cpu_updated = cpu_model
        .forward_masked(
            Tensor::from_data(input.to_data(), &Default::default()),
            Tensor::from_data(valid.to_data(), &Default::default()),
        )?
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    compare(&cpu_updated, &gpu_logits)?;
    let bytes = <_ as Recorder<Gpu>>::record(&recorder, model.valid().into_record(), ())
        .map_err(|e| e.to_string())?;
    let record = <_ as Recorder<Cpu>>::load(&recorder, bytes, &Default::default())
        .map_err(|e| e.to_string())?;
    let restored = config.init::<Cpu>(&Default::default())?.load_record(record);
    let cpu_logits = restored
        .forward_masked(
            Tensor::from_data(input.to_data(), &Default::default()),
            Tensor::from_data(valid.to_data(), &Default::default()),
        )?
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    compare(&cpu_logits, &gpu_logits)?;
    let before = before.to_data().to_vec::<f32>().unwrap();
    if before
        .iter()
        .zip(&gpu_logits)
        .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        return Err("Adam did not change model output".into());
    }
    println!("finite gradient tensors per step={gradient_tensors}");
    Gpu::sync(device);
    println!(
        "PASS: CPU/GPU record transfer, masked forward/loss, backward, Adam record round trip and GPU synchronization"
    );
    Ok(())
}

fn compare(expected: &[f32], actual: &[f32]) -> Result<(), String> {
    if expected.len() != actual.len()
        || expected.iter().zip(actual).any(|(a, b)| {
            !a.is_finite() || !b.is_finite() || (a - b).abs() > 0.003 + 0.003 * a.abs()
        })
    {
        return Err("CPU/GPU values exceed declared tolerance".into());
    }
    Ok(())
}
