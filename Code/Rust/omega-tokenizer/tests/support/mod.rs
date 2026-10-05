use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use tokenizers::{AddedToken, Tokenizer, processors::template::TemplateProcessing};

pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "omega-tokenizer-tests-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create temporary directory: {error}"),
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove this test's temporary directory");
    }
}

pub fn save_pipeline(path: &Path) {
    let mut tokenizer =
        Tokenizer::from_bytes(include_bytes!("../../../../../datasets/test.json")).unwrap();
    tokenizer
        .add_special_tokens([
            AddedToken::from("<start>", true),
            AddedToken::from("<end>", true),
        ])
        .unwrap();
    // These IDs belong only to this generated fixture and are saved explicitly.
    assert_eq!(tokenizer.token_to_id("<start>"), Some(13));
    assert_eq!(tokenizer.token_to_id("<end>"), Some(14));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("<start> $A <end>")
            .unwrap()
            .special_tokens(vec![("<start>", 13), ("<end>", 14)])
            .build()
            .unwrap(),
    ));
    tokenizer.save(path, true).unwrap();
}
