use omega_tokenizer::Tokens;
use omega_training::{
    cache::{CacheLimits, CacheOptions, CachePartition, create_token_cache, open_token_cache},
    checkpoint::sha256_bytes,
    dataset::{
        DatasetFormat, ExampleSource, ValidationSplit, load_document_corpus_with_format,
        split_document_corpus,
    },
};
use std::{fs, path::Path};

fn tokenizer() -> Tokens {
    Tokens::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json")).unwrap()
}

fn options(format: DatasetFormat) -> CacheOptions {
    CacheOptions {
        format,
        context_length: 2,
        validation: ValidationSplit::Count(1),
        split_seed: 42,
        limits: CacheLimits::default(),
    }
}

fn corpus(root: &Path, format: DatasetFormat) {
    fs::create_dir_all(root.join("texts/nested")).unwrap();
    match format {
        DatasetFormat::Text => {
            fs::write(
                root.join("texts/a.txt"),
                "hello world omega tokenizer this is",
            )
            .unwrap();
            fs::write(root.join("texts/nested/b.txt"), "this is a test").unwrap();
            fs::write(root.join("texts/blank.txt"), " \n").unwrap();
        }
        DatasetFormat::Jsonl => {
            fs::write(root.join("texts/nested/a.jsonl"), "\n{\"text\":\"hello world omega tokenizer this is\"}\n{\"text\":\" \"}\n{\"text\":\"this is a test\"}\n").unwrap();
        }
    }
}

#[test]
fn cache_matches_eager_splits_chunks_provenance_and_resume_cursors() {
    for format in [DatasetFormat::Text, DatasetFormat::Jsonl] {
        let temp = tempfile::tempdir().unwrap();
        corpus(temp.path(), format);
        let tokenizer = tokenizer();
        let options = options(format);
        let folders = vec!["texts".into(), "texts/nested".into(), "texts".into()];
        let eager =
            load_document_corpus_with_format(temp.path(), &folders, &tokenizer, format).unwrap();
        let split = split_document_corpus(&eager, 2, options.validation, 42).unwrap();
        let path = temp.path().join("cache");
        let created =
            create_token_cache(&path, temp.path(), &folders, &tokenizer, &options).unwrap();
        let opened = open_token_cache(&path, temp.path(), &folders, &tokenizer, &options).unwrap();
        assert_eq!(opened.dataset_provenance(), created.dataset_provenance());
        for (kind, eager) in [
            (CachePartition::Training, split.training),
            (CachePartition::Validation, split.validation),
        ] {
            let cached = opened.partition(kind);
            assert_eq!(cached.files(), &eager.set.files);
            assert_eq!(cached.documents(), &eager.documents);
            assert_eq!(cached.token_count(), eager.set.token_count);
            assert_eq!(
                cached.target_count().unwrap(),
                eager.set.target_count().unwrap()
            );
            assert_eq!(cached.example_count(), eager.set.example_count());
            assert_eq!(cached.identity().unwrap(), eager.set.identity().unwrap());
            for cursor in 0..=cached.example_count() {
                assert_eq!(
                    cached
                        .iter_from(cursor)
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap(),
                    eager.set.examples[cursor..]
                );
            }
            assert!(cached.iter_from(cached.example_count() + 1).is_err());
            assert!(cached.example(cached.example_count()).is_err());
        }
        // Source roots may move without changing relative identities or tokens.
        let relocated = tempfile::tempdir().unwrap();
        corpus(relocated.path(), format);
        let relocated =
            open_token_cache(&path, relocated.path(), &folders, &tokenizer, &options).unwrap();
        assert_eq!(
            relocated
                .partition(CachePartition::Training)
                .identity()
                .unwrap(),
            opened
                .partition(CachePartition::Training)
                .identity()
                .unwrap()
        );
    }
}

#[test]
fn changed_sources_pipeline_selections_settings_and_limits_invalidate_cache() {
    let temp = tempfile::tempdir().unwrap();
    corpus(temp.path(), DatasetFormat::Text);
    let tokenizer = tokenizer();
    let options = options(DatasetFormat::Text);
    let folders = vec!["texts".into()];
    let path = temp.path().join("cache");
    create_token_cache(&path, temp.path(), &folders, &tokenizer, &options).unwrap();
    let source = temp.path().join("texts/a.txt");
    let original = fs::read(&source).unwrap();
    fs::write(&source, [original.as_slice(), b" "].concat()).unwrap();
    assert!(
        open_token_cache(&path, temp.path(), &folders, &tokenizer, &options)
            .unwrap_err()
            .contains("Stale")
    );
    fs::write(&source, &original).unwrap();
    fs::write(temp.path().join("texts/new.txt"), "hello world").unwrap();
    assert!(open_token_cache(&path, temp.path(), &folders, &tokenizer, &options).is_err());
    fs::remove_file(temp.path().join("texts/new.txt")).unwrap();
    for changed in [
        CacheOptions {
            context_length: 3,
            ..options.clone()
        },
        CacheOptions {
            split_seed: 4,
            ..options.clone()
        },
        CacheOptions {
            validation: ValidationSplit::None,
            ..options.clone()
        },
        CacheOptions {
            limits: CacheLimits {
                max_source_bytes: 1024,
                ..options.limits.clone()
            },
            ..options.clone()
        },
    ] {
        assert!(
            open_token_cache(&path, temp.path(), &folders, &tokenizer, &changed)
                .unwrap_err()
                .contains("settings/limits")
        );
    }
    assert!(
        open_token_cache(
            &path,
            temp.path(),
            &["texts/nested".into()],
            &tokenizer,
            &options
        )
        .unwrap_err()
        .contains("selections")
    );
    let mut changed: serde_json::Value =
        serde_json::from_str(&tokenizer.to_json().unwrap()).unwrap();
    changed["normalizer"] = serde_json::Value::Null;
    let changed_path = temp.path().join("changed.json");
    fs::write(&changed_path, serde_json::to_vec(&changed).unwrap()).unwrap();
    let changed = Tokens::new(changed_path).unwrap();
    assert!(
        open_token_cache(&path, temp.path(), &folders, &changed, &options)
            .unwrap_err()
            .contains("tokenizer identity")
    );
}

fn rewrite_manifest(path: &Path, update: impl FnOnce(&mut serde_json::Value)) {
    let mut value = serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap();
    update(&mut value);
    let bytes = serde_json::to_vec(&value).unwrap();
    fs::write(path.join("manifest.json"), &bytes).unwrap();
    fs::write(
        path.join("COMPLETE"),
        format!("omega-token-cache-v1\n{}\n", sha256_bytes(&bytes)),
    )
    .unwrap();
}

#[test]
fn incomplete_corrupt_and_structurally_invalid_caches_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    corpus(temp.path(), DatasetFormat::Text);
    let tokenizer = tokenizer();
    let options = CacheOptions {
        validation: ValidationSplit::None,
        ..options(DatasetFormat::Text)
    };
    let folders = vec!["texts".into()];
    let path = temp.path().join("cache");
    let cache = create_token_cache(&path, temp.path(), &folders, &tokenizer, &options).unwrap();
    let manifest = fs::read(path.join("manifest.json")).unwrap();
    let marker = fs::read(path.join("COMPLETE")).unwrap();
    assert!(create_token_cache(&path, temp.path(), &folders, &tokenizer, &options).is_err());
    assert_eq!(fs::read(path.join("manifest.json")).unwrap(), manifest);
    fs::remove_file(path.join("COMPLETE")).unwrap();
    assert!(
        open_token_cache(&path, temp.path(), &folders, &tokenizer, &options)
            .unwrap_err()
            .contains("Incomplete")
    );
    fs::write(path.join("COMPLETE"), &marker).unwrap();
    fs::write(path.join("manifest.json"), b"{}").unwrap();
    assert!(
        open_token_cache(&path, temp.path(), &folders, &tokenizer, &options)
            .unwrap_err()
            .contains("checksum")
    );
    for case in 0..5 {
        fs::write(path.join("manifest.json"), &manifest).unwrap();
        rewrite_manifest(&path, |value| match case {
            0 => value["schema_version"] = 99.into(),
            1 => value["documents"][0]["source"] = 99999.into(),
            2 => value["documents"][0]["token_count"] = usize::MAX.into(),
            3 => value["documents"][0]["id"] = "../outside.txt".into(),
            _ => value["training_identity"] = "bad-digest".into(),
        });
        assert!(
            open_token_cache(&path, temp.path(), &folders, &tokenizer, &options).is_err(),
            "case {case}"
        );
    }
    fs::write(path.join("manifest.json"), &manifest).unwrap();
    fs::write(path.join("COMPLETE"), &marker).unwrap();
    let token_file = path.join("tokens-0.bin");
    let original_tokens = fs::read(&token_file).unwrap();
    fs::write(&token_file, vec![0; original_tokens.len()]).unwrap();
    let mut iterator = cache
        .partition(CachePartition::Training)
        .iter_from(0)
        .unwrap();
    assert!(iterator.next().unwrap().is_err());
    assert!(iterator.next().is_none());
    assert!(open_token_cache(&path, temp.path(), &folders, &tokenizer, &options).is_err());
    fs::write(&token_file, &original_tokens[..original_tokens.len() - 1]).unwrap();
    assert!(
        cache
            .partition(CachePartition::Training)
            .example(0)
            .unwrap_err()
            .contains("checksum/length")
    );
}

#[test]
fn explicit_memory_limits_fail_without_completion_and_never_truncate() {
    let temp = tempfile::tempdir().unwrap();
    corpus(temp.path(), DatasetFormat::Text);
    let tokenizer = tokenizer();
    let base = options(DatasetFormat::Text);
    for (index, limits) in [
        CacheLimits {
            max_source_bytes: 5,
            ..CacheLimits::default()
        },
        CacheLimits {
            max_document_tokens: 2,
            ..CacheLimits::default()
        },
        CacheLimits {
            max_documents: 1,
            ..CacheLimits::default()
        },
        CacheLimits {
            max_directory_entries: 1,
            ..CacheLimits::default()
        },
        CacheLimits {
            max_manifest_bytes: 10,
            ..CacheLimits::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        let path = temp.path().join(format!("limited-{index}"));
        let limited = CacheOptions {
            limits,
            ..base.clone()
        };
        assert!(
            create_token_cache(&path, temp.path(), &["texts".into()], &tokenizer, &limited)
                .is_err()
        );
        assert!(!path.join("COMPLETE").exists());
    }
    let mut invalid = base;
    invalid.limits.max_source_bytes = 0;
    let path = temp.path().join("invalid");
    assert!(
        create_token_cache(&path, temp.path(), &["texts".into()], &tokenizer, &invalid).is_err()
    );
    assert!(!path.exists());
}

#[test]
fn zero_validation_empty_cursor_and_eager_identity_ignore_metadata() {
    let temp = tempfile::tempdir().unwrap();
    corpus(temp.path(), DatasetFormat::Text);
    let options = CacheOptions {
        validation: ValidationSplit::None,
        ..options(DatasetFormat::Text)
    };
    let cache = create_token_cache(
        &temp.path().join("cache"),
        temp.path(),
        &["texts".into()],
        &tokenizer(),
        &options,
    )
    .unwrap();
    let validation = cache.partition(CachePartition::Validation);
    assert_eq!(validation.example_count(), 0);
    assert_eq!(validation.target_count().unwrap(), 0);
    assert!(validation.iter_from(0).unwrap().next().is_none());
    let mut set = omega_training::TrainingSet {
        examples: vec![vec![0, 1], vec![1, 2]],
        files: vec![],
        token_count: 0,
    };
    let before = set.identity().unwrap();
    set.files.push(temp.path().to_path_buf());
    set.token_count = 50;
    assert_eq!(before, set.identity().unwrap());
    set.examples.reverse();
    assert_ne!(before, set.identity().unwrap());
}

#[test]
fn configured_truncation_is_rejected_before_creating_a_cache() {
    let temp = tempfile::tempdir().unwrap();
    corpus(temp.path(), DatasetFormat::Text);
    let original = tokenizer();
    let mut value: serde_json::Value = serde_json::from_str(&original.to_json().unwrap()).unwrap();
    value["truncation"] = serde_json::json!({"direction":"Right", "max_length":2, "strategy":"LongestFirst", "stride":0});
    let path = temp.path().join("truncated.json");
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let truncated = Tokens::new(path).unwrap();
    let output = temp.path().join("cache");
    let error = create_token_cache(
        &output,
        temp.path(),
        &["texts".into()],
        &truncated,
        &options(DatasetFormat::Text),
    )
    .unwrap_err();
    assert!(error.contains("truncation to be disabled"), "{error}");
    assert!(!output.exists());
}
