use sha2::{Digest, Sha256};
use std::{env, process::Command};

fn sources(path: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(path).expect("Read training sources") {
        let path = entry.expect("Read source entry").path();
        if path.is_dir() {
            sources(&path, files)
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path)
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=OMEGA_BUILD_REVISION");
    let mut hash = Sha256::new();
    for package in [".", "../omega-nn", "../omega-tokenizer"] {
        let root = std::path::Path::new(package);
        let mut files = vec![root.join("Cargo.toml")];
        sources(&root.join("src"), &mut files);
        files.sort();
        println!("cargo:rerun-if-changed={}", root.join("src").display());
        for path in files {
            println!("cargo:rerun-if-changed={}", path.display());
            hash.update(path.to_string_lossy().replace('\\', "/"));
            hash.update([0]);
            hash.update(std::fs::read(path).expect("Read source identity"));
            hash.update([0]);
        }
    }
    hash.update(std::fs::read("build.rs").expect("Read build script"));
    println!("cargo:rustc-env=OMEGA_SOURCE_SHA256={:x}", hash.finalize());
    let flags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let flags = flags.replace('\u{1f}', " ");
    println!("cargo:rustc-env=OMEGA_BUILD_FLAGS={flags}");
    println!(
        "cargo:rustc-env=OMEGA_OPT_LEVEL={}",
        env::var("OPT_LEVEL").unwrap()
    );
    println!(
        "cargo:rustc-env=OMEGA_DEBUG_INFO={}",
        env::var("DEBUG").unwrap()
    );
    let rustc = env::var_os("RUSTC").expect("Cargo supplies RUSTC");
    let output = Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("Cannot inspect Rust compiler identity");
    assert!(
        output.status.success(),
        "Cannot inspect Rust compiler identity"
    );
    let identity = String::from_utf8(output.stdout)
        .expect("Compiler identity must be UTF-8")
        .lines()
        .collect::<Vec<_>>()
        .join("; ");
    println!("cargo:rustc-env=OMEGA_RUSTC_IDENTITY={identity}");
    println!(
        "cargo:rustc-env=OMEGA_BUILD_TARGET={}",
        env::var("TARGET").unwrap()
    );
    println!(
        "cargo:rustc-env=OMEGA_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap()
    );
}
