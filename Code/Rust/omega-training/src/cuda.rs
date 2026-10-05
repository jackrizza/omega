//! Experimental CUDA f32 backend. Host numerical checks, no fusion or fallback.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaExecutionProfile {
    pub kernel: String,
    pub index: usize,
    pub name: String,
    pub uuid: String,
    pub driver: i32,
    pub nvrtc: (i32, i32),
    pub host_kernel: String,
    pub rustc: String,
    pub target: String,
    pub features: String,
    pub rustflags: String,
    pub build_profile: String,
    pub source_sha256: String,
    pub headers_sha256: String,
}
impl CudaExecutionProfile {
    pub fn validate(&self) -> Result<(), String> {
        if self.kernel != "cuda-f32-host-checks-v1"
            || self.name.is_empty()
            || (self.uuid.len() != 32 || !self.uuid.bytes().all(|b| b.is_ascii_hexdigit()))
            || self.driver <= 0
            || (self.nvrtc < (12, 5) || self.nvrtc.0 != 12)
            || self.host_kernel.is_empty()
            || self.rustc.is_empty()
            || self.target.is_empty()
            || self.features.is_empty()
            || self.build_profile.is_empty()
            || self.source_sha256.len() != 64
            || !self.source_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || self.headers_sha256.len() != 64
            || !self.headers_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("Unsupported/incomplete experimental CUDA execution identity".into());
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub type CudaTraining = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
#[cfg(feature = "cuda")]
pub struct CudaDevice {
    pub device: burn::backend::cuda::CudaDevice,
    pub profile: CudaExecutionProfile,
}

pub fn list_devices() -> Result<Vec<CudaExecutionProfile>, String> {
    #[cfg(feature = "cuda")]
    {
        std::panic::catch_unwind(|| {
            let count = cudarc::driver::CudaContext::device_count().map_err(|e| format!("CUDA device discovery: {e}"))?;
            (0..count as usize).map(profile).collect()
        }).map_err(|_| "CUDA driver/NVRTC 12.5–12.9 unavailable or incompatible. CUDA 13 NVRTC is unsupported by Burn 0.18; Blackwell requires NVRTC 12.8+. Select compatible libraries through LD_LIBRARY_PATH (CUDA is experimental)".to_string())?
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("CUDA is not compiled in; build with --features cuda".into())
    }
}

#[cfg(feature = "cuda")]
fn profile(index: usize) -> Result<CudaExecutionProfile, String> {
    let headers_sha256 = headers_identity()?;
    let context = cudarc::driver::CudaContext::new(index)
        .map_err(|e| format!("Cannot open CUDA device {index}: {e}"))?;
    let uuid = context
        .uuid()
        .map_err(|e| e.to_string())?
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect();
    let mut driver = 0;
    let (mut major, mut minor) = (0, 0);
    // CUDA writes only the provided live integer outputs. Loading is deferred
    // until an explicit CUDA request; callers contain missing-library panics.
    unsafe {
        cudarc::driver::sys::cuDriverGetVersion(&mut driver)
            .result()
            .map_err(|e| e.to_string())?;
        cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor)
            .result()
            .map_err(|e| e.to_string())?;
    }
    let result = CudaExecutionProfile {
        kernel: "cuda-f32-host-checks-v1".into(),
        index,
        name: context.name().map_err(|e| e.to_string())?,
        uuid,
        driver,
        nvrtc: (major, minor),
        host_kernel: crate::host_kernel()?,
        rustc: env!("OMEGA_RUSTC_IDENTITY").into(),
        target: env!("OMEGA_BUILD_TARGET").into(),
        features: format!("cuda=true;vulkan={};fusion=false", cfg!(feature = "gpu")),
        rustflags: env!("OMEGA_BUILD_FLAGS").into(),
        build_profile: format!(
            "{}:{}:{}",
            env!("OMEGA_BUILD_PROFILE"),
            env!("OMEGA_OPT_LEVEL"),
            env!("OMEGA_DEBUG_INFO")
        ),
        source_sha256: env!("OMEGA_SOURCE_SHA256").into(),
        headers_sha256,
    };
    result.validate()?;
    Ok(result)
}

/// Match CubeCL 0.6's runtime include lookup. It compiles CUDA C++ through
/// NVRTC, so driver/NVRTC libraries alone are insufficient. Hash the installed
/// include tree, following header symlinks but rejecting cycles and huge trees.
#[cfg(feature = "cuda")]
fn headers_identity() -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeSet,
        io::Read,
        path::{Path, PathBuf},
    };
    let root=std::env::var_os("CUDA_PATH").map(PathBuf::from)
        .or_else(||["/usr/local/cuda","/opt/cuda"].into_iter().map(PathBuf::from).find(|p|p.exists()))
        .or_else(||Path::new("/usr/bin/nvcc").exists().then(||PathBuf::from("/usr")))
        .ok_or("CUDA runtime headers are missing: set CUDA_PATH to a toolkit containing include/cuda_runtime.h")?.join("include");
    if !root.join("cuda_runtime.h").is_file() {
        return Err(format!(
            "CUDA runtime headers missing at {}; set CUDA_PATH to a complete CUDA toolkit",
            root.display()
        ));
    }
    fn visit(
        root: &Path,
        path: &Path,
        hash: &mut Sha256,
        ancestors: &mut BTreeSet<PathBuf>,
        budget: &mut (u64, usize),
    ) -> Result<(), String> {
        let canonical = path.canonicalize().map_err(|e| e.to_string())?;
        if path.is_dir() {
            if !ancestors.insert(canonical.clone()) {
                return Err("CUDA header directory has a symlink cycle".into());
            }
            let mut entries = std::fs::read_dir(path)
                .map_err(|e| e.to_string())?
                .map(|e| e.map(|e| e.path()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            entries.sort();
            for entry in entries {
                visit(root, &entry, hash, ancestors, budget)?;
            }
            ancestors.remove(&canonical);
        } else if path.is_file() {
            budget.1 += 1;
            if budget.1 > 100_000 {
                return Err("CUDA header tree exceeds 100,000 files".into());
            }
            hash.update(
                path.strip_prefix(root)
                    .map_err(|e| e.to_string())?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
            hash.update([0]);
            let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
            let mut bytes = [0u8; 65536];
            loop {
                let count = file.read(&mut bytes).map_err(|e| e.to_string())?;
                if count == 0 {
                    break;
                }
                budget.0 += count as u64;
                if budget.0 > 512 * 1024 * 1024 {
                    return Err("CUDA headers exceed 512 MiB".into());
                }
                hash.update(&bytes[..count]);
            }
            hash.update([0]);
        } else {
            return Err("CUDA header tree contains a non-regular file".into());
        }
        Ok(())
    }
    let mut hash = Sha256::new();
    visit(&root, &root, &mut hash, &mut BTreeSet::new(), &mut (0, 0))?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(feature = "cuda")]
pub fn initialize(index: usize) -> Result<CudaDevice, String> {
    let profile = list_devices()?
        .into_iter()
        .find(|p| p.index == index)
        .ok_or_else(|| format!("CUDA device {index} is unavailable; no fallback"))?;
    Ok(CudaDevice {
        device: burn::backend::cuda::CudaDevice::new(index),
        profile,
    })
}
