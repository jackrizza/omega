//! Explicit discrete Vulkan GPU selection; CPU builds retain only identity types.

#[cfg(feature = "gpu")]
use burn::backend::{
    Autodiff, Wgpu,
    wgpu::{WgpuDevice, graphics::Vulkan, init_setup},
};
use serde::{Deserialize, Serialize};

/// f32 GPU tensors, i32 token IDs and u32 masks; fusion is intentionally disabled.
#[cfg(feature = "gpu")]
pub type Gpu = Wgpu<f32, i32, u32>;
#[cfg(feature = "gpu")]
pub type GpuTraining = Autodiff<Gpu>;

/// Adapter identity and per-buffer limits (not free/total VRAM measurements).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanAdapter {
    pub index: usize,
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub driver_info: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct VulkanDevice {
    pub(crate) device: WgpuDevice,
    pub(crate) adapter: VulkanAdapter,
    pub(crate) profile: GpuExecutionProfile,
}

#[cfg(feature = "gpu")]
impl VulkanDevice {
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
    pub fn adapter(&self) -> &VulkanAdapter {
        &self.adapter
    }
    pub fn profile(&self) -> &GpuExecutionProfile {
        &self.profile
    }
}

impl VulkanAdapter {
    /// Check known f32 weights and full-context training intermediates against
    /// the adapter's individual storage-buffer limit before allocating a model.
    /// This is not an aggregate/peak VRAM estimate or a guarantee that all backend
    /// temporary allocations fit. Shorter actual sequences use no more than this
    /// full-context bound; optimizer moments have the corresponding weight shape.
    pub fn validate_model(&self, config: &omega_nn::GptConfig, batch: usize) -> Result<(), String> {
        config.validate()?;
        if batch == 0 {
            return Err("GPU batch size must be positive".into());
        }
        let limit = self
            .max_buffer_size
            .min(u64::from(self.max_storage_buffer_binding_size));
        let shapes: &[(&str, &[usize])] = &[
            (
                "token embedding/output weights",
                &[config.vocab_size, config.d_model],
            ),
            (
                "position embedding",
                &[config.context_length, config.d_model],
            ),
            (
                "attention projection weights",
                &[config.d_model, config.d_model],
            ),
            ("feed-forward weights", &[config.d_model, config.d_ff]),
            ("output bias", &[config.vocab_size]),
            ("logits", &[batch, config.context_length, config.vocab_size]),
            (
                "attention scores/masks",
                &[
                    batch,
                    config.num_heads,
                    config.context_length,
                    config.context_length,
                ],
            ),
            (
                "hidden states",
                &[batch, config.context_length, config.d_model],
            ),
            (
                "feed-forward activations",
                &[batch, config.context_length, config.d_ff],
            ),
        ];
        for (name, shape) in shapes {
            let bytes = shape
                .iter()
                .try_fold(4usize, |bytes, dimension| bytes.checked_mul(*dimension))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| format!("GPU {name} buffer dimensions overflow addressable size"))?;
            if bytes > limit {
                return Err(format!(
                    "GPU {name} requires {bytes} bytes in one buffer; adapter limit is {limit} bytes. Reduce model dimensions, context or batch size."
                ));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
fn describe(index: usize, adapter: &wgpu::Adapter) -> VulkanAdapter {
    let info = adapter.get_info();
    let limits = adapter.limits();
    VulkanAdapter {
        index,
        name: info.name,
        vendor: info.vendor,
        device: info.device,
        driver: info.driver,
        driver_info: info.driver_info,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
    }
}

/// Indices enumerate discrete Vulkan adapters only, in the order used by Burn.
/// Integrated, virtual and software adapters are deliberately excluded.
#[cfg(feature = "gpu")]
pub fn list_vulkan_devices() -> Vec<VulkanAdapter> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .iter()
        .filter(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .enumerate()
        .map(|(index, adapter)| describe(index, adapter))
        .collect()
}

/// Initialize exactly the requested discrete adapter, rejecting any fallback.
#[cfg(feature = "gpu")]
pub fn initialize_vulkan(index: usize) -> Result<VulkanDevice, String> {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };
    static DEVICES: OnceLock<Mutex<HashMap<usize, VulkanDevice>>> = OnceLock::new();
    let mut devices = DEVICES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "Vulkan initialization state is poisoned".to_string())?;
    if let Some(device) = devices.get(&index) {
        return Ok(device.clone());
    }
    let selected = initialize_vulkan_once(index)?;
    devices.insert(index, selected.clone());
    Ok(selected)
}

#[cfg(feature = "gpu")]
fn initialize_vulkan_once(index: usize) -> Result<VulkanDevice, String> {
    let adapters = list_vulkan_devices();
    let expected = adapters.get(index).ok_or_else(|| {
        format!(
            "Vulkan discrete GPU index {index} is unavailable ({} discrete adapters found)",
            adapters.len()
        )
    })?;
    let device = WgpuDevice::DiscreteGpu(index);
    let setup = std::panic::catch_unwind(|| init_setup::<Vulkan>(&device, Default::default()))
        .map_err(|_| {
            "Burn could not initialize the requested Vulkan GPU; inspect the driver error above"
                .to_string()
        })?;
    let adapter = describe(index, &setup.adapter);
    if setup.backend != wgpu::Backend::Vulkan
        || setup.adapter.get_info().device_type != wgpu::DeviceType::DiscreteGpu
        || &adapter != expected
    {
        return Err(
            "Vulkan selected a different adapter; CPU/software fallback is forbidden".into(),
        );
    }
    let profile = GpuExecutionProfile {
        schema_version: 1,
        kernel: "wgpu-vulkan-f32-device-checks-v2".into(),
        precision: "f32".into(),
        adapter_index: adapter.index,
        adapter_name: adapter.name.clone(),
        vendor: adapter.vendor,
        device: adapter.device,
        device_type: "DiscreteGpu".into(),
        backend: "Vulkan".into(),
        driver: adapter.driver.clone(),
        driver_info: adapter.driver_info.clone(),
        host_kernel: crate::host_kernel()?,
        rustflags: env!("OMEGA_BUILD_FLAGS").into(),
        rustc: env!("OMEGA_RUSTC_IDENTITY").into(),
        target: env!("OMEGA_BUILD_TARGET").into(),
        build_profile: env!("OMEGA_BUILD_PROFILE").into(),
        features: concat!(
            "gpu;spirv;no-fusion;",
            env!("OMEGA_BUILD_PROFILE"),
            ";opt=",
            env!("OMEGA_OPT_LEVEL"),
            ";debug=",
            env!("OMEGA_DEBUG_INFO")
        )
        .into(),
    };
    profile.validate()?;
    Ok(VulkanDevice {
        device,
        adapter,
        profile,
    })
}

/// Conservative execution identity for same-runtime GPU continuation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuExecutionProfile {
    pub schema_version: u32,
    pub kernel: String,
    pub precision: String,
    /// Enumeration index distinguishes identical adapters, but is not a stable PCI UUID.
    pub adapter_index: usize,
    pub adapter_name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub host_kernel: String,
    pub rustflags: String,
    pub rustc: String,
    pub target: String,
    pub features: String,
    pub build_profile: String,
}

impl GpuExecutionProfile {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || !matches!(
                self.kernel.as_str(),
                "wgpu-vulkan-f32-checked-v1" | "wgpu-vulkan-f32-device-checks-v2"
            )
            || self.precision != "f32"
            || self.backend != "Vulkan"
            || self.device_type != "DiscreteGpu"
            || [
                &self.adapter_name,
                &self.driver,
                &self.driver_info,
                &self.host_kernel,
                &self.rustc,
                &self.target,
                &self.features,
                &self.build_profile,
            ]
            .iter()
            .any(|field| field.trim().is_empty())
        {
            return Err("Unsupported or incomplete GPU execution profile".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuExecutionProfile, VulkanAdapter};

    #[test]
    fn execution_profile_rejects_missing_identity_and_wrong_backend() {
        let mut profile = GpuExecutionProfile {
            schema_version: 1,
            kernel: "wgpu-vulkan-f32-checked-v1".into(),
            precision: "f32".into(),
            adapter_index: 0,
            adapter_name: "test adapter".into(),
            vendor: 0x8086,
            device: 1,
            device_type: "DiscreteGpu".into(),
            backend: "Vulkan".into(),
            driver: "test driver".into(),
            driver_info: "1.0".into(),
            host_kernel: "Linux test".into(),
            rustflags: String::new(),
            rustc: "test compiler".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            features: "gpu;spirv;no-fusion;debug".into(),
            build_profile: "debug".into(),
        };
        assert!(profile.validate().is_ok());
        profile.backend = "Cpu".into();
        assert!(profile.validate().is_err());
        profile.backend = "Vulkan".into();
        profile.rustc.clear();
        assert!(profile.validate().is_err());
    }
    #[test]
    fn known_model_buffers_reject_limits_zero_batch_and_overflow() {
        let mut adapter = VulkanAdapter {
            index: 0,
            name: "test".into(),
            vendor: 0,
            device: 0,
            driver: "test".into(),
            driver_info: "test".into(),
            max_buffer_size: 512,
            max_storage_buffer_binding_size: 512,
        };
        let mut config = omega_nn::GptConfig {
            vocab_size: 16,
            context_length: 4,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        };
        assert!(adapter.validate_model(&config, 1).is_ok());
        adapter.max_storage_buffer_binding_size = 511;
        assert!(
            adapter
                .validate_model(&config, 1)
                .unwrap_err()
                .contains("adapter limit")
        );
        adapter.max_storage_buffer_binding_size = 512;
        assert!(adapter.validate_model(&config, 0).is_err());
        config.vocab_size = usize::MAX;
        assert!(
            adapter
                .validate_model(&config, 1)
                .unwrap_err()
                .contains("overflow")
        );
    }
}
