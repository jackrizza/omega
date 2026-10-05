//! Explicit CPU settings belong on a fresh child process before backend or
//! tokenizer pools initialize. This module never changes the current environment
//! or creates/reconfigures thread pools. Library callers must capture the execution
//! profile before using backend, tokenizer, or custom pools; capture cannot certify
//! the settings of pools initialized externally before that call.

use std::{ffi::OsString, process::Command, sync::OnceLock};

use clap::Args;
use serde::{Deserialize, Deserializer, Serialize};

/// Current checked NdArray f32 kernel contract; no optional SIMD kernel is enabled.
pub const CPU_KERNEL: &str = "ndarray-f32-checked-v1";
const MAX_ENV_BYTES: usize = 64;
const RAYON_THREADS: &str = "RAYON_NUM_THREADS";
const RAYON_CPUS: &str = "RAYON_RS_NUM_CPUS";
const MATMUL_THREADS: &str = "MATMUL_NUM_THREADS";

fn parse_cpu_threads(value: &str) -> Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|_| "cpu-threads must be an integer in 1..=256")?;
    if !(1..=256).contains(&count) {
        return Err("cpu-threads must be in 1..=256".into());
    }
    Ok(count)
}

fn parse_matmul_threads(value: &str) -> Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|_| "matmul-threads must be 1, 2, or 4")?;
    if ![1, 2, 4].contains(&count) {
        return Err("matmul-threads must be 1, 2, or 4".into());
    }
    Ok(count)
}

/// Explicit overrides for a new process. Omitted settings inherit its environment.
/// These are separate compute-pool controls, not a cap on every process thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Args)]
pub struct CpuThreadSettings {
    /// Override the child Rayon pool size (1..=256); omitted inherits environment
    #[arg(long, global = true, value_parser = parse_cpu_threads)]
    pub cpu_threads: Option<usize>,
    /// Override child matrixmultiply threads (1, 2, or 4); omitted inherits environment
    #[arg(long, global = true, value_parser = parse_matmul_threads)]
    pub matmul_threads: Option<usize>,
}

impl CpuThreadSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .cpu_threads
            .is_some_and(|count| !(1..=256).contains(&count))
        {
            return Err("cpu-threads must be in 1..=256".into());
        }
        if self
            .matmul_threads
            .is_some_and(|count| ![1, 2, 4].contains(&count))
        {
            return Err("matmul-threads must be 1, 2, or 4".into());
        }
        Ok(())
    }

    /// Set overrides on a not-yet-spawned command only. Both Rayon variables are
    /// set when requested, so an inherited fallback cannot override the choice.
    /// All settings are checked before modifying the command.
    pub fn apply_to_command(&self, command: &mut Command) -> Result<(), String> {
        self.validate()?;
        if let Some(count) = self.cpu_threads {
            command.env(RAYON_THREADS, count.to_string());
            command.env(RAYON_CPUS, count.to_string());
        }
        if let Some(count) = self.matmul_threads {
            command.env(MATMUL_THREADS, count.to_string());
        }
        Ok(())
    }
}

fn required_option<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

/// Raw startup settings, not measured effective pool sizes or physical core count.
/// All fields are required in serialized metadata; unavailable values use null.
/// Raw inherited strings (including zero/invalid backend fallback requests) are
/// preserved exactly, since normalizing them would erase execution provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuExecutionProfile {
    pub kernel: String,
    #[serde(deserialize_with = "required_option")]
    pub rayon_num_threads: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub rayon_rs_num_cpus: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub matmul_num_threads: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub available_parallelism: Option<usize>,
}

impl CpuExecutionProfile {
    /// Check metadata shape and supported kernel, without comparing this process.
    pub fn validate(&self) -> Result<(), String> {
        if self.kernel != CPU_KERNEL {
            return Err(format!("Unsupported CPU kernel profile: {}", self.kernel));
        }
        for (name, value) in [
            (RAYON_THREADS, &self.rayon_num_threads),
            (RAYON_CPUS, &self.rayon_rs_num_cpus),
            (MATMUL_THREADS, &self.matmul_num_threads),
        ] {
            if value
                .as_ref()
                .is_some_and(|value| value.len() > MAX_ENV_BYTES)
            {
                return Err(format!("CPU profile {name} exceeds {MAX_ENV_BYTES} bytes"));
            }
        }
        if self.available_parallelism == Some(0) {
            return Err("CPU profile available_parallelism must be positive or null".into());
        }
        Ok(())
    }

    /// Legacy snapshots carried no thread settings. Only absent variables and
    /// the original kernel qualify; this does not assert legacy pool identity.
    pub fn is_legacy_default(&self) -> bool {
        self.kernel == CPU_KERNEL
            && self.rayon_num_threads.is_none()
            && self.rayon_rs_num_cpus.is_none()
            && self.matmul_num_threads.is_none()
    }
}

fn capture_profile(
    mut environment: impl FnMut(&str) -> Option<OsString>,
    available_parallelism: Option<usize>,
) -> Result<CpuExecutionProfile, String> {
    let mut read = |name| {
        environment(name)
            .map(|value| {
                value
                    .into_string()
                    .map_err(|_| format!("CPU profile {name} is not valid Unicode"))
            })
            .transpose()
    };
    let profile = CpuExecutionProfile {
        kernel: CPU_KERNEL.into(),
        rayon_num_threads: read(RAYON_THREADS)?,
        rayon_rs_num_cpus: read(RAYON_CPUS)?,
        matmul_num_threads: read(MATMUL_THREADS)?,
        available_parallelism,
    };
    profile.validate()?;
    Ok(profile)
}

fn freeze_profile(
    captured: CpuExecutionProfile,
    frozen: &OnceLock<CpuExecutionProfile>,
) -> Result<CpuExecutionProfile, String> {
    captured.validate()?;
    let original = frozen.get_or_init(|| captured.clone());
    if original != &captured {
        return Err("CPU execution profile changed after initial capture; start a fresh process to change CPU settings".into());
    }
    Ok(original.clone())
}

/// Capture before backend/tokenizer/custom pool initialization. The first valid
/// profile is frozen process-wide; later changed environment or available CPU
/// parallelism is rejected. This neither initializes pools nor certifies external
/// pools, and never mutates environment. Use a fresh process for different settings.
pub fn execution_profile() -> Result<CpuExecutionProfile, String> {
    static PROFILE: OnceLock<CpuExecutionProfile> = OnceLock::new();
    let current = capture_profile(
        |name| std::env::var_os(name),
        std::thread::available_parallelism().ok().map(usize::from),
    )?;
    freeze_profile(current, &PROFILE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Parser, Subcommand};
    use serde_json::json;
    use std::collections::BTreeMap;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        cpu: CpuThreadSettings,
        #[command(subcommand)]
        command: Operation,
    }
    #[derive(Subcommand)]
    enum Operation {
        Train,
    }

    fn profile() -> CpuExecutionProfile {
        capture_profile(|_| None, Some(8)).unwrap()
    }

    fn command_environment(command: &Command) -> BTreeMap<OsString, Option<OsString>> {
        command
            .get_envs()
            .map(|(name, value)| (name.into(), value.map(OsString::from)))
            .collect()
    }

    #[test]
    fn flags_are_global_optional_and_bounded() {
        assert_eq!(
            Cli::try_parse_from(["test", "train"]).unwrap().cpu,
            CpuThreadSettings::default()
        );
        for args in [
            [
                "test",
                "--cpu-threads",
                "1",
                "train",
                "--matmul-threads",
                "4",
            ],
            [
                "test",
                "train",
                "--cpu-threads",
                "256",
                "--matmul-threads",
                "2",
            ],
        ] {
            assert!(Cli::try_parse_from(args).is_ok());
        }
        for (flag, value) in [
            ("--cpu-threads", "0"),
            ("--cpu-threads", "257"),
            ("--cpu-threads", "abc"),
            ("--cpu-threads", "-1"),
            ("--matmul-threads", "0"),
            ("--matmul-threads", "3"),
            ("--matmul-threads", "5"),
            ("--matmul-threads", "1.5"),
        ] {
            assert!(Cli::try_parse_from(["test", "train", flag, value]).is_err());
        }
    }

    #[test]
    fn command_overrides_are_explicit_and_invalid_settings_are_atomic() {
        let mut command = Command::new("not-spawned");
        assert!(command_environment(&command).is_empty());
        CpuThreadSettings::default()
            .apply_to_command(&mut command)
            .unwrap();
        assert!(command_environment(&command).is_empty());
        command
            .env(RAYON_THREADS, "20")
            .env(RAYON_CPUS, "10")
            .env(MATMUL_THREADS, "2")
            .env("OTHER", "keep");
        let initial = command_environment(&command);
        assert!(
            CpuThreadSettings {
                cpu_threads: Some(1),
                matmul_threads: Some(3)
            }
            .apply_to_command(&mut command)
            .is_err()
        );
        assert_eq!(command_environment(&command), initial);
        CpuThreadSettings {
            cpu_threads: Some(1),
            matmul_threads: None,
        }
        .apply_to_command(&mut command)
        .unwrap();
        let environment = command_environment(&command);
        assert_eq!(
            environment[&OsString::from(RAYON_THREADS)],
            Some("1".into())
        );
        assert_eq!(environment[&OsString::from(RAYON_CPUS)], Some("1".into()));
        assert_eq!(
            environment[&OsString::from(MATMUL_THREADS)],
            Some("2".into())
        );
        assert_eq!(environment[&OsString::from("OTHER")], Some("keep".into()));
        CpuThreadSettings {
            cpu_threads: None,
            matmul_threads: Some(4),
        }
        .apply_to_command(&mut command)
        .unwrap();
        assert_eq!(
            command_environment(&command)[&OsString::from(MATMUL_THREADS)],
            Some("4".into())
        );
    }

    #[test]
    fn metadata_requires_nullable_fields_and_rejects_unknown_fields() {
        let valid = serde_json::to_value(profile()).unwrap();
        assert_eq!(
            serde_json::from_value::<CpuExecutionProfile>(valid.clone()).unwrap(),
            profile()
        );
        for field in [
            "kernel",
            "rayon_num_threads",
            "rayon_rs_num_cpus",
            "matmul_num_threads",
            "available_parallelism",
        ] {
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<CpuExecutionProfile>(missing).is_err(),
                "{field}"
            );
        }
        let mut unknown = valid;
        unknown["extra"] = json!(true);
        assert!(serde_json::from_value::<CpuExecutionProfile>(unknown).is_err());
        let mut unavailable = profile();
        unavailable.available_parallelism = None;
        assert_eq!(
            serde_json::from_str::<CpuExecutionProfile>(
                &serde_json::to_string(&unavailable).unwrap()
            )
            .unwrap(),
            unavailable
        );
    }

    #[test]
    fn inherited_values_remain_raw_and_metadata_validation_is_bounded() {
        let captured = capture_profile(
            |name| {
                Some(
                    match name {
                        RAYON_THREADS => "0",
                        RAYON_CPUS => "invalid",
                        MATMUL_THREADS => " 2 ",
                        _ => unreachable!(),
                    }
                    .into(),
                )
            },
            None,
        )
        .unwrap();
        assert_eq!(captured.rayon_num_threads.as_deref(), Some("0"));
        assert_eq!(captured.rayon_rs_num_cpus.as_deref(), Some("invalid"));
        assert_eq!(captured.matmul_num_threads.as_deref(), Some(" 2 "));
        assert!(!captured.is_legacy_default());
        assert!(profile().is_legacy_default());
        assert!(capture_profile(|_| Some("x".repeat(65).into()), None).is_err());
        assert!(capture_profile(|_| Some("x".repeat(64).into()), None).is_ok());
        let mut invalid = profile();
        invalid.available_parallelism = Some(0);
        assert!(invalid.validate().is_err());
        invalid.available_parallelism = None;
        invalid.kernel = "other".into();
        assert!(invalid.validate().is_err());
        assert!(!invalid.is_legacy_default());
        for field in [RAYON_THREADS, RAYON_CPUS, MATMUL_THREADS] {
            assert!(
                !capture_profile(|name| (name == field).then(OsString::new), None)
                    .unwrap()
                    .is_legacy_default()
            );
        }
    }

    #[test]
    fn local_freeze_rejects_each_changed_execution_field_without_mutation() {
        let frozen = OnceLock::new();
        let initial = profile();
        assert_eq!(freeze_profile(initial.clone(), &frozen).unwrap(), initial);
        assert_eq!(freeze_profile(initial.clone(), &frozen).unwrap(), initial);
        for index in 0..4 {
            let mut changed = initial.clone();
            match index {
                0 => changed.rayon_num_threads = Some("1".into()),
                1 => changed.rayon_rs_num_cpus = Some("1".into()),
                2 => changed.matmul_num_threads = Some("1".into()),
                _ => changed.available_parallelism = Some(4),
            }
            assert!(
                freeze_profile(changed, &frozen)
                    .unwrap_err()
                    .contains("fresh process")
            );
            assert_eq!(frozen.get(), Some(&initial));
        }
        let empty = OnceLock::new();
        let mut invalid = initial;
        invalid.kernel.clear();
        assert!(freeze_profile(invalid, &empty).is_err());
        assert!(empty.get().is_none());
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn non_unicode_environment_is_an_actionable_error() {
        #[cfg(windows)]
        let invalid = {
            use std::os::windows::ffi::OsStringExt;
            OsString::from_wide(&[0xd800])
        };
        #[cfg(unix)]
        let invalid = {
            use std::os::unix::ffi::OsStringExt;
            OsString::from_vec(vec![0xff])
        };
        let error = capture_profile(|_| Some(invalid.clone()), None).unwrap_err();
        assert!(error.contains(RAYON_THREADS) && error.contains("Unicode"));
    }
}
