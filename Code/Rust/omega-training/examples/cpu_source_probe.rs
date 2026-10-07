//! Bounded warmed source-read probe, not a tensor or whole-training benchmark.
//! All writes are inside one owned TempDir. The saved fixture tokenizer is read-only.
use std::{fs, hint::black_box, path::Path, process::ExitCode, time::Instant};

use omega_tokenizer::Tokens;
use omega_training::{
    TrainingSet,
    cache::{
        CacheLimits, CacheOptions, CachePartition, CachedPartition, create_token_cache,
        open_token_cache,
    },
    checkpoint::{BuildIdentity, TokenizerIdentity, sha256_bytes, tokenizer_identity},
    dataset::{
        DatasetFormat, ExampleSource, ValidationSplit, load_document_corpus_with_format,
        split_document_corpus,
    },
};
use serde_json::{Value, json};

const DOCUMENTS: usize = 16;
const DOCUMENT_TOKENS: usize = 128;
const CONTEXT: usize = 32;
const SAMPLES: usize = 5;
const MAX_SECONDS: f64 = 5.0;

struct Fixture {
    directory: tempfile::TempDir,
    eager: TrainingSet,
    cached: CachedPartition,
    options: CacheOptions,
    tokenizer: TokenizerIdentity,
    source_bytes: usize,
}

impl Fixture {
    fn new() -> Result<Self, String> {
        let directory = tempfile::Builder::new()
            .prefix("omega-cpu-source-probe-")
            .tempdir()
            .map_err(|e| e.to_string())?;
        let texts = directory.path().join("texts");
        fs::create_dir(&texts).map_err(|e| e.to_string())?;
        let words = [
            "hello",
            "world",
            "omega",
            "this",
            "is",
            "a",
            "test",
            "tokenizer",
        ];
        let mut source_bytes = 0;
        for index in 0..DOCUMENTS {
            let text = (0..DOCUMENT_TOKENS)
                .map(|position| words[(position + index) % words.len()])
                .collect::<Vec<_>>()
                .join(" ");
            source_bytes += text.len();
            fs::write(texts.join(format!("{index:02}.txt")), text).map_err(|e| e.to_string())?;
        }
        let tokenizer = Tokens::new(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/wordlevel.json"),
        )?;
        let options = CacheOptions {
            format: DatasetFormat::Text,
            context_length: CONTEXT,
            validation: ValidationSplit::None,
            split_seed: 42,
            limits: CacheLimits {
                max_source_bytes: 1024,
                max_document_tokens: DOCUMENT_TOKENS,
                max_documents: DOCUMENTS,
                max_directory_entries: 64,
                max_manifest_bytes: 64 * 1024,
            },
        };
        let folders = vec!["texts".to_string()];
        let corpus = load_document_corpus_with_format(
            directory.path(),
            &folders,
            &tokenizer,
            options.format,
        )?;
        if corpus.documents.len() != DOCUMENTS
            || corpus
                .documents
                .iter()
                .any(|document| document.tokens.len() != DOCUMENT_TOKENS)
        {
            return Err("Fixture must encode exactly 16 documents of 128 tokens".into());
        }
        let eager =
            split_document_corpus(&corpus, CONTEXT, options.validation, options.split_seed)?
                .training
                .set;
        let cache_path = directory.path().join("cache");
        create_token_cache(
            &cache_path,
            directory.path(),
            &folders,
            &tokenizer,
            &options,
        )?;
        let cached = open_token_cache(
            &cache_path,
            directory.path(),
            &folders,
            &tokenizer,
            &options,
        )?
        .partition(CachePartition::Training);
        if eager.identity()? != cached.identity()?
            || eager.example_count() != cached.example_count()
            || eager.target_count()? != cached.target_count()?
        {
            return Err("Eager/cache ordered identity or counts differ".into());
        }
        let fixture = Self {
            directory,
            eager,
            cached,
            options,
            tokenizer: tokenizer_identity(&tokenizer)?,
            source_bytes,
        };
        fixture.verify(&read_pass(&fixture.cached)?)?;
        Ok(fixture)
    }

    fn verify(&self, examples: &[Vec<u32>]) -> Result<(), String> {
        if examples != self.eager.examples {
            return Err("Source-read ordered examples differ from the eager reference".into());
        }
        Ok(())
    }

    fn close(self) -> Result<(), String> {
        self.directory
            .close()
            .map_err(|e| format!("Cannot clean owned source-probe temporary directory: {e}"))
    }
}

fn check_deadline(started: Instant) -> Result<(), String> {
    if started.elapsed().as_secs_f64() > MAX_SECONDS {
        return Err("Source probe exceeded its cooperative 5-second limit".into());
    }
    Ok(())
}

fn read_pass(source: &impl ExampleSource) -> Result<Vec<Vec<u32>>, String> {
    (0..source.example_count())
        .map(|index| source.example(index))
        .collect()
}

fn timed_pass(source: &impl ExampleSource) -> Result<(f64, Vec<Vec<u32>>), String> {
    let started = Instant::now();
    let examples = black_box(read_pass(source)?);
    let seconds = started.elapsed().as_secs_f64();
    if seconds <= 0.0 {
        return Err("Source probe clock did not advance".into());
    }
    Ok((seconds, examples))
}

fn summary(samples: &[f64], example_count: usize, targets: usize) -> Value {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let median = ordered[ordered.len() / 2];
    json!({"samples_seconds":samples,"min_seconds":ordered[0],"median_seconds":median,"max_seconds":ordered[ordered.len()-1],"median_examples_per_second":example_count as f64/median,"median_targets_supplied_per_second":targets as f64/median})
}

fn source_identity() -> String {
    let mut bytes = b"omega-cpu-source-probe-v1\0".to_vec();
    for source in [
        include_bytes!("cpu_source_probe.rs").as_slice(),
        include_bytes!("../src/cache.rs").as_slice(),
        include_bytes!("../src/dataset.rs").as_slice(),
        include_bytes!("../Cargo.toml").as_slice(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    sha256_bytes(&bytes)
}

fn run() -> Result<Value, String> {
    if cfg!(debug_assertions) {
        return Err("Source probe requires --release".into());
    }
    let started = Instant::now();
    let fixture = Fixture::new()?;
    let setup_seconds = started.elapsed().as_secs_f64();
    check_deadline(started)?;
    let warmup = Instant::now();
    fixture.verify(&black_box(read_pass(&fixture.eager)?))?;
    fixture.verify(&black_box(read_pass(&fixture.cached)?))?;
    let warmup_seconds = warmup.elapsed().as_secs_f64();
    check_deadline(started)?;
    let mut eager_samples = Vec::with_capacity(SAMPLES);
    let mut cached_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        // Alternate which source goes first while retaining identical ascending
        // example order. Output collection is timed for both; comparison/drop is not.
        for cached_first in [sample % 2 == 1, sample % 2 != 1] {
            check_deadline(started)?;
            let (seconds, examples) = if cached_first {
                timed_pass(&fixture.cached)?
            } else {
                timed_pass(&fixture.eager)?
            };
            check_deadline(started)?;
            fixture.verify(&examples)?;
            black_box(&examples);
            if cached_first {
                cached_samples.push(seconds);
            } else {
                eager_samples.push(seconds);
            }
        }
    }
    let example_count = fixture.eager.example_count();
    let targets = fixture.eager.target_count()?;
    let mut report = json!({
        "schema_version":1,"status":"complete","profile":"release_required",
        "measurement":"warmed_source_reads_only_no_tensors_or_training",
        "documents":DOCUMENTS,"tokens_per_document":DOCUMENT_TOKENS,"context_length":CONTEXT,
        "source_text_bytes":fixture.source_bytes,"source_tokens":fixture.eager.token_count,
        "example_count":example_count,"real_targets_per_pass":targets,"samples_per_source":SAMPLES,
        "warmup_passes_per_source":1,"sample_order":"alternate_eager_first_and_cache_first;each_pass_ascending_example_index",
        "tokenizer":fixture.tokenizer,"cache_options":fixture.options,
        "ordered_source_identity":fixture.eager.identity()?,"parity":"identity_counts_and_every_ordered_example_verified_before_and_after_timed_reads",
        "eager":{"cost":"example_vec_clone_and_common_output_collection;comparison_drop_excluded","timings":summary(&eager_samples,example_count,targets)},
        "cached":{"cost":"indexed_lookup_whole_document_file_read_length_sha256_check_le_decode_chunk_clone_and_common_output_collection;comparison_drop_excluded","timings":summary(&cached_samples,example_count,targets)},
        "setup_seconds":setup_seconds,"warmup_seconds":warmup_seconds,"max_seconds":MAX_SECONDS,
        "deadline":"cooperative_between_bounded_setup_and_passes;nonzero_error_on_overrun;no_io_preemption",
        "cache_state":"OS_filesystem_cache_warmed_not_flushed;not_cold_storage_latency",
        "build":BuildIdentity::current(),"source_sha256":source_identity(),"target_os":std::env::consts::OS,"target_arch":std::env::consts::ARCH
    });
    let cleanup = Instant::now();
    fixture.close()?;
    report["cleanup_seconds"] = cleanup.elapsed().as_secs_f64().into();
    check_deadline(started)?;
    report["total_seconds"] = started.elapsed().as_secs_f64().into();
    report["owned_temp_cleanup"] = true.into();
    Ok(report)
}

fn main() -> ExitCode {
    match run().and_then(|report| serde_json::to_string(&report).map_err(|e| e.to_string())) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("CPU source probe failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_fixture_has_identical_ordered_sources_and_owned_cleanup() {
        let fixture = Fixture::new().unwrap();
        assert_eq!(fixture.eager.example_count(), 64);
        assert_eq!(fixture.eager.target_count().unwrap(), 16 * 127);
        let (_, eager) = timed_pass(&fixture.eager).unwrap();
        let (_, cached) = timed_pass(&fixture.cached).unwrap();
        fixture.verify(&eager).unwrap();
        fixture.verify(&cached).unwrap();
        assert_eq!(eager, cached);
        let path = fixture.directory.path().to_path_buf();
        fixture.close().unwrap();
        assert!(!path.exists());
    }
}
