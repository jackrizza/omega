use crate::{
    ProjectConfig,
    config::{absolute, format},
    jobs::{Action, JobSpec, Reporter},
};
use anyhow::{Context, Result, ensure};
use omega_training::operations as op;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub fn validate_launch(
    c: &ProjectConfig,
    action: &Action,
    checkpoint: &Option<PathBuf>,
) -> Result<()> {
    c.validate()?;
    if matches!(action, Action::PostTrain | Action::PostTrainRecover) {
        c.post_training
            .as_ref()
            .context("Configure [post_training] in a schema-2 project")?
            .validate()?;
        if action == &Action::PostTrainRecover {
            ensure!(checkpoint.is_some(), "Select a saved workflow directory");
        }
        return Ok(());
    }
    if !matches!(action, Action::Generate | Action::Chat | Action::Resume) {
        ensure!(
            !c.selections().is_empty(),
            "Select dataset folders or configure a dataset recipe first"
        );
    }
    if matches!(
        action,
        Action::Resume | Action::Generate | Action::Chat | Action::Evaluate | Action::Assistant
    ) {
        ensure!(checkpoint.is_some(), "Select a complete checkpoint first");
    }
    if matches!(action, Action::PrepareDataset)
        || (action == &Action::Pipeline && c.pipeline.prepare_dataset)
    {
        ensure!(
            c.dataset.recipe.is_some(),
            "Dataset preparation requires a recipe"
        );
    }
    if c.pipeline.prepare_cache && action == &Action::Pipeline || action == &Action::PrepareCache {
        ensure!(
            c.dataset.cache.is_some(),
            "Set dataset.cache to a new cache directory first"
        );
    }
    if c.pipeline.benchmark && c.training.shuffle {
        anyhow::bail!(
            "Benchmark projection supports fixed order only; disable shuffle or the benchmark stage"
        );
    }
    if action == &Action::Pipeline && c.assistant.is_some() {
        ensure!(
            c.pipeline.train,
            "An assistant pipeline needs a newly trained base; use the separate Assistant action for an existing parent"
        );
    }
    if action == &Action::Pipeline && c.pipeline.evaluate {
        ensure!(
            !c.evaluation.selections.is_empty(),
            "Select evaluation data"
        );
    }
    Ok(())
}

pub fn stages(c: &ProjectConfig, action: &Action) -> Vec<Action> {
    if action != &Action::Pipeline {
        return vec![action.clone()];
    }
    let p = &c.pipeline;
    let mut v = vec![];
    for (enabled, stage) in [
        (p.prepare_dataset, Action::PrepareDataset),
        (p.prepare_tokenizer, Action::TrainTokenizer),
        (p.prepare_cache, Action::PrepareCache),
        (p.benchmark, Action::Benchmark),
        (p.train, Action::Train),
        (c.assistant.is_some(), Action::Assistant),
        (p.evaluate, Action::Evaluate),
    ] {
        if enabled {
            v.push(stage)
        }
    }
    v
}

/// Resolve remote metadata only, then freeze commits and exact files for review.
pub fn review(
    c: &ProjectConfig,
    project: &Path,
    action: &Action,
) -> Result<(ProjectConfig, String)> {
    if matches!(action, Action::PostTrain | Action::PostTrainRecover) {
        return crate::post_training::prepare(c, project);
    }
    let mut pinned = c.clone();
    let mut details = vec![];
    let plan = stages(c, action);
    ensure!(!plan.is_empty(), "Select at least one pipeline stage");
    if plan.contains(&Action::PrepareDataset) {
        let recipe = pinned
            .dataset
            .recipe
            .as_mut()
            .context("No dataset recipe")?;
        let hub = omega_datasets::hub::HuggingFace::new(
            std::env::var("HF_TOKEN").ok(),
            recipe.limits.timeout_seconds,
        )?;
        let resolved = omega_datasets::hub::plan(recipe, &hub)?;
        for (source, remote) in recipe.sources.iter_mut().zip(resolved) {
            source.revision = remote.revision;
            source.files = remote.files.iter().map(|f| f.path.clone()).collect();
            details.push(format!(
                "{} @ {}\n{}",
                source.repo,
                source.revision,
                remote
                    .files
                    .iter()
                    .map(|f| format!(
                        "  {}: {}",
                        f.path,
                        f.size
                            .map(|n| format!("{n} bytes"))
                            .unwrap_or("unknown size".into())
                    ))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }
    details.push(format!(
        "Resolved model.toml:\n{}",
        toml::to_string_pretty(&pinned)?
    ));
    Ok((
        pinned,
        format!(
            "Project: {}\nStages: {:?}\nBackend: {:?}, device {}\nDataset root: {}\nData: {:?}\nTokenizer: {}\nCheckpoints: {}\nModel: context {}, width {}, heads {}, layers {}, FF {}\nTraining: {} epochs, batch {}, learning rate {}\n{}\n\nStart launches an independent worker. Detach does not stop training. Preparation may read the entire dataset. CUDA is experimental.",
            project.display(),
            plan,
            c.training.backend,
            c.training.device,
            absolute(project, &c.paths.datasets).display(),
            c.selections(),
            absolute(project, &c.tokenizer.path).display(),
            absolute(project, &c.paths.weights).display(),
            c.model.context_length,
            c.model.d_model,
            c.model.heads,
            c.model.layers,
            c.model.d_ff,
            c.training.epochs,
            c.training.batch_size,
            c.training.learning_rate,
            details.join("\n")
        ),
    ))
}

pub fn doctor(backend: omega_benchmark::Backend) -> Result<serde_json::Value> {
    use omega_benchmark::Backend;
    match backend {
        Backend::Cpu => Ok(
            serde_json::json!({"backend":"cpu","status":"available","parallelism":std::thread::available_parallelism().map(|n|n.get()).ok()}),
        ),
        Backend::Vulkan => {
            #[cfg(feature = "gpu")]
            {
                let adapters = omega_training::gpu::list_vulkan_devices();
                ensure!(
                    !adapters.is_empty(),
                    "No discrete Vulkan adapter found; install compatible drivers"
                );
                Ok(serde_json::json!({"backend":"vulkan","devices":adapters}))
            }
            #[cfg(not(feature = "gpu"))]
            {
                anyhow::bail!("Vulkan is not compiled into this executable")
            }
        }
        Backend::Cuda => {
            let devices = omega_training::cuda::list_devices().map_err(anyhow::Error::msg)?;
            ensure!(!devices.is_empty(), "No NVIDIA CUDA devices found");
            Ok(serde_json::json!({"backend":"cuda","experimental":true,"devices":devices}))
        }
    }
}

pub fn preflight(backend: omega_benchmark::Backend, index: usize) -> Result<serde_json::Value> {
    let report = doctor(backend)?;
    if backend == omega_benchmark::Backend::Cpu {
        ensure!(index == 0, "CPU device must be zero");
    } else {
        ensure!(
            report["devices"].as_array().is_some_and(|devices| devices
                .iter()
                .any(|d| d["index"].as_u64() == Some(index as u64))),
            "Selected device {index} is unavailable; run omega doctor and select a listed index"
        );
    }
    Ok(report)
}

pub(crate) fn input(path: &Path) -> Result<(PathBuf, op::CheckpointInput)> {
    Ok((
        path.parent()
            .context("Checkpoint requires parent directory")?
            .into(),
        op::CheckpointInput {
            checkpoint: Some(
                path.file_name()
                    .context("Checkpoint requires a name")?
                    .to_string_lossy()
                    .into_owned(),
            ),
            latest_run: None,
        },
    ))
}
pub(crate) fn limits() -> op::Limits {
    let l = omega_training::cache::CacheLimits::default();
    op::Limits {
        max_source_bytes: l.max_source_bytes,
        max_document_tokens: l.max_document_tokens,
        max_documents: l.max_documents,
        max_directory_entries: l.max_directory_entries,
        max_manifest_bytes: l.max_manifest_bytes,
    }
}
fn flags(c: &ProjectConfig) -> Box<op::SessionFlags> {
    Box::new(op::SessionFlags {
        shuffle: c.training.shuffle,
        dataset_weights: vec![],
        samples_per_epoch: None,
        batch_size: c.training.batch_size,
        max_batch_tokens: c.training.max_batch_tokens,
        gradient_clip_norm: c.training.gradient_clip_norm,
        warmup_updates: c.training.warmup_updates,
    })
}
fn saves(c: &ProjectConfig) -> op::SaveOptions {
    op::SaveOptions {
        save_every_updates: c.training.save_every_updates,
        save_every_epochs: c.training.save_every_epochs,
    }
}
pub(crate) fn args(c: &ProjectConfig, command: op::Command, compute: bool) -> op::Args {
    op::Args {
        cpu: Default::default(),
        backend: if !compute {
            op::BackendChoice::Cpu
        } else {
            match c.training.backend {
                omega_benchmark::Backend::Cpu => op::BackendChoice::Cpu,
                omega_benchmark::Backend::Vulkan => op::BackendChoice::Vulkan,
                omega_benchmark::Backend::Cuda => op::BackendChoice::Cuda,
            }
        },
        device: if compute && c.training.backend != omega_benchmark::Backend::Cpu {
            Some(c.training.device)
        } else {
            None
        },
        command,
    }
}

pub fn execute(spec: &JobSpec, reporter: Arc<Reporter>, stop: Arc<AtomicBool>) -> Result<()> {
    if matches!(spec.action, Action::PostTrain | Action::PostTrainRecover) {
        return crate::post_training::execute(spec, reporter, stop);
    }
    let c = &spec.config;
    let root = &spec.project;
    let datasets = absolute(root, &c.paths.datasets);
    let weights = absolute(root, &c.paths.weights);
    let tokenizer = absolute(root, &c.tokenizer.path);
    let selected = c.selections();
    let cache = c.dataset.cache.as_ref().map(|p| absolute(root, p));
    let observe = reporter.clone();
    let segment_stop = stop.clone();
    let control = op::OperationControl {
        stop: stop.clone(),
        observer: Box::new(move |event| {
            if event.kind == "training_finished"
                && event.data["segment_only"].as_bool() == Some(true)
            {
                segment_stop.store(true, Ordering::SeqCst);
            }
            observe
                .emit(&event.kind, event.data)
                .map_err(|e| e.to_string())
        }),
    };
    let plan = stages(c, &spec.action);
    if plan.iter().any(|a| {
        matches!(
            a,
            Action::Train
                | Action::Assistant
                | Action::Resume
                | Action::Benchmark
                | Action::Evaluate
                | Action::Generate
                | Action::Chat
        )
    }) {
        reporter.emit(
            "preflight",
            preflight(c.training.backend, c.training.device)?,
        )?;
    }
    omega_training::cpu::execution_profile().map_err(anyhow::Error::msg)?;
    for stage in plan {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        reporter.emit("stage", serde_json::json!({"name":format!("{stage:?}")}))?;
        let checkpoint = reporter
            .checkpoint()
            .or_else(|| spec.checkpoint.as_ref().map(|p| absolute(root, p)));
        let command = match stage {
            Action::PrepareDataset => {
                let recipe = c.dataset.recipe.as_ref().context("Missing recipe")?;
                let folder = omega_datasets::dataset_folder(&datasets, &c.dataset.recipe_folder)?;
                fs::create_dir_all(&folder)?;
                let output = folder.join(&recipe.output);
                if output.exists() {
                    let saved =
                        omega_datasets::config::Config::load(&output.join("model.lock.toml"))?;
                    ensure!(
                        toml::to_string(&saved)? == toml::to_string(recipe)?,
                        "Existing release recipe differs from the reviewed recipe; choose a new output or select existing data explicitly"
                    );
                    omega_datasets::verify_release(&output)?;
                    reporter.emit("result", serde_json::json!({"reused_release":output}))?;
                } else {
                    let hub = omega_datasets::hub::HuggingFace::new(
                        std::env::var("HF_TOKEN").ok(),
                        recipe.limits.timeout_seconds,
                    )?;
                    let result = omega_datasets::build(&folder, recipe, &hub)?;
                    reporter.emit("result", serde_json::to_value(result)?)?;
                }
                continue;
            }
            Action::TrainTokenizer => {
                if tokenizer.exists() {
                    // Reusing a tokenizer requires a receipt from the same frozen
                    // inputs/settings, not merely an equal vocabulary length.
                    verify_tokenizer_receipt(c, root, &tokenizer)?;
                    reporter.emit("result", serde_json::json!({"reused_tokenizer":tokenizer}))?;
                    continue;
                }
                fs::create_dir_all(tokenizer.parent().context("Tokenizer parent missing")?)?;
                op::Command::TrainTokenizer {
                    datasets_root: Some(datasets.clone()),
                    datasets: selected.clone(),
                    dataset_format: format(&c.dataset.format)?,
                    output: tokenizer.clone(),
                    vocab_size: c.tokenizer.vocab_size,
                    min_frequency: c.tokenizer.min_frequency,
                    chat_protocol: c.tokenizer.chat_protocol,
                }
            }
            Action::PrepareCache => {
                let output = cache.clone().context("Set dataset.cache")?;
                if output.exists() {
                    let tokens =
                        omega_tokenizer::Tokens::new(&tokenizer).map_err(anyhow::Error::msg)?;
                    let fmt = omega_training::selection::resolve_dataset_format(
                        &datasets,
                        &selected,
                        format(&c.dataset.format)?,
                        true,
                        validation(c),
                    )
                    .map_err(anyhow::Error::msg)?;
                    ensure!(
                        fmt != omega_training::selection::Format::Chat,
                        "Chat caches are unsupported"
                    );
                    omega_training::cache::open_token_cache(
                        &output,
                        &datasets,
                        &selected,
                        &tokens,
                        &omega_training::cache::CacheOptions {
                            format: fmt.into(),
                            context_length: c.model.context_length,
                            validation: validation(c),
                            split_seed: c.dataset.split_seed,
                            limits: Default::default(),
                        },
                    )
                    .map_err(anyhow::Error::msg)?;
                    continue;
                }
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)?;
                }
                op::Command::PrepareCache {
                    datasets_root: Some(datasets.clone()),
                    datasets: selected.clone(),
                    tokenizer: tokenizer.clone(),
                    dataset_format: format(&c.dataset.format)?,
                    context_length: c.model.context_length,
                    validation_ratio: c.dataset.validation_ratio,
                    validation_count: c.dataset.validation_count,
                    split_seed: c.dataset.split_seed,
                    output,
                    limits: limits(),
                }
            }
            Action::Benchmark => {
                ensure!(
                    !c.training.shuffle,
                    "Fixed-order benchmark cannot project shuffled training"
                );
                let report = omega_benchmark::training_time(&omega_benchmark::TrainingTimeConfig {
                    dataset: omega_benchmark::DatasetOptions {
                        root: datasets.clone(),
                        selections: selected.clone(),
                        tokenizer: tokenizer.clone(),
                        format: format(&c.dataset.format)?,
                        validation: validation(c),
                        split_seed: c.dataset.split_seed,
                        cache: cache.clone(),
                        ..Default::default()
                    },
                    model: c.model.clone(),
                    backend: c.training.backend,
                    device: c.training.device,
                    batching: c.session().batching,
                    epochs: c.training.epochs,
                    learning_rate: c.training.learning_rate,
                    seed: c.training.seed,
                    optimization: c.session().optimization,
                    warmup: c.benchmark.warmup,
                    samples: c.benchmark.samples,
                    max_seconds: c.benchmark.max_seconds,
                })
                .map_err(anyhow::Error::msg)?;
                reporter.emit("result", serde_json::to_value(report)?)?;
                continue;
            }
            Action::Train => op::Command::Train {
                name: c.name.clone(),
                datasets_root: Some(datasets.clone()),
                weights_root: Some(weights.clone()),
                datasets: selected.clone(),
                dataset_format: format(&c.dataset.format)?,
                tokenizer: tokenizer.clone(),
                epochs: c.training.epochs,
                learning_rate: c.training.learning_rate,
                context_length: c.model.context_length,
                d_model: c.model.d_model,
                heads: c.model.heads,
                layers: c.model.layers,
                d_ff: c.model.d_ff,
                seed: c.training.seed,
                validation_ratio: c.dataset.validation_ratio,
                validation_count: c.dataset.validation_count,
                split_seed: c.dataset.split_seed,
                metrics_jsonl: None,
                quiet: true,
                cache: cache.clone(),
                max_updates: c.training.max_updates,
                saves: saves(c),
                limits: limits(),
                session_options: flags(c),
            },
            Action::Assistant => {
                let a = c
                    .assistant
                    .as_ref()
                    .context("Configure an assistant stage first")?;
                let (weights_root, input) = input(&checkpoint.context("No base checkpoint")?)?;
                op::Command::TrainStage {
                    datasets_root: Some(datasets.clone()),
                    weights_root: Some(weights_root),
                    input,
                    name: format!("{}-assistant", c.name),
                    datasets: a.selections.clone(),
                    epochs: a.epochs,
                    learning_rate: a.learning_rate,
                    seed: c.training.seed,
                    max_updates: c.training.max_updates,
                    metrics_jsonl: None,
                    quiet: true,
                    saves: saves(c),
                    session_options: flags(c),
                }
            }
            Action::Resume => {
                let (weights_root, input) = input(&checkpoint.context("No resume checkpoint")?)?;
                op::Command::Resume {
                    datasets_root: Some(datasets.clone()),
                    weights_root: Some(weights_root),
                    input,
                    name: c.name.clone(),
                    epochs: c.training.epochs,
                    max_updates: c.training.max_updates,
                    saves: saves(c),
                    cache: cache.clone(),
                    limits: limits(),
                    metrics_jsonl: None,
                    quiet: true,
                }
            }
            Action::Evaluate => {
                let (weights_root, input) =
                    input(&checkpoint.context("No evaluation checkpoint")?)?;
                ensure!(
                    !c.evaluation.selections.is_empty(),
                    "Configure held-out evaluation selections"
                );
                op::Command::Evaluate {
                    datasets_root: Some(datasets.clone()),
                    weights_root: Some(weights_root),
                    input,
                    datasets: c.evaluation.selections.clone(),
                    dataset_format: format(
                        c.evaluation.format.as_deref().unwrap_or(&c.dataset.format),
                    )?,
                    validation_ratio: None,
                    validation_count: None,
                    split_seed: c.dataset.split_seed,
                }
            }
            Action::Generate => {
                let (weights_root, input) =
                    input(&checkpoint.context("No generation checkpoint")?)?;
                op::Command::Generate {
                    weights_root: Some(weights_root),
                    input,
                    prompt: spec.prompt.clone().context("Enter a generation prompt")?,
                    max_new_tokens: c.inference.max_new_tokens,
                    eos_token_id: None,
                    generation: Default::default(),
                }
            }
            Action::Chat => {
                let (weights_root, input) = input(&checkpoint.context("No chat checkpoint")?)?;
                op::Command::Chat {
                    history: c
                        .inference
                        .history
                        .iter()
                        .map(|m| {
                            Ok(omega_tokenizer::chat::ChatMessage {
                                role: match m.role.as_str() {
                                    "system" => omega_tokenizer::chat::ChatRole::System,
                                    "user" => omega_tokenizer::chat::ChatRole::User,
                                    "assistant" => omega_tokenizer::chat::ChatRole::Assistant,
                                    _ => anyhow::bail!("Invalid chat role"),
                                },
                                content: m.content.clone(),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                    weights_root: Some(weights_root),
                    input,
                    prompt: Some(spec.prompt.clone().context("Enter a chat prompt")?),
                    system: c.inference.system.clone(),
                    max_new_tokens: c.inference.max_new_tokens,
                    generation: Default::default(),
                }
            }
            Action::Pipeline | Action::PostTrain | Action::PostTrainRecover => {
                unreachable!("workflow expanded above")
            }
        };
        let compute = !matches!(stage, Action::TrainTokenizer | Action::PrepareCache);
        if compute {
            fs::create_dir_all(&weights)?;
        }
        op::execute(args(c, command, compute), Some(&control)).map_err(anyhow::Error::msg)?;
        if stage == Action::TrainTokenizer {
            write_tokenizer_receipt(c, root, &tokenizer)?;
        }
    }
    Ok(())
}
fn validation(c: &ProjectConfig) -> omega_training::ValidationSplit {
    c.dataset
        .validation_count
        .map(omega_training::ValidationSplit::Count)
        .or_else(|| {
            c.dataset
                .validation_ratio
                .map(omega_training::ValidationSplit::Ratio)
        })
        .unwrap_or(omega_training::ValidationSplit::None)
}
fn tokenizer_receipt(c: &ProjectConfig, root: &Path, path: &Path) -> Result<serde_json::Value> {
    let datasets = absolute(root, &c.paths.datasets);
    let selected = c.selections();
    let fmt = omega_training::selection::resolve_dataset_format(
        &datasets,
        &selected,
        format(&c.dataset.format)?,
        true,
        omega_training::ValidationSplit::None,
    )
    .map_err(anyhow::Error::msg)?;
    ensure!(
        fmt != omega_training::selection::Format::Chat,
        "Tokenizer fitting requires base text"
    );
    let docs = omega_training::dataset::load_text_documents(&datasets, &selected, fmt.into())
        .map_err(anyhow::Error::msg)?;
    let tokens = omega_tokenizer::Tokens::new(path).map_err(anyhow::Error::msg)?;
    Ok(
        serde_json::json!({"schema_version":1,"settings":c.tokenizer,"documents":docs.iter().map(|d|omega_training::checkpoint::sha256_bytes(d.text.as_bytes())).collect::<Vec<_>>(),"identity":omega_training::checkpoint::tokenizer_identity(&tokens).map_err(anyhow::Error::msg)?}),
    )
}
fn write_tokenizer_receipt(c: &ProjectConfig, root: &Path, path: &Path) -> Result<()> {
    crate::jobs::atomic_json(
        &path.with_extension("omega-receipt.json"),
        &tokenizer_receipt(c, root, path)?,
    )
}
fn verify_tokenizer_receipt(c: &ProjectConfig, root: &Path, path: &Path) -> Result<()> {
    let saved:serde_json::Value=crate::jobs::read_json(&path.with_extension("omega-receipt.json")).context("Existing tokenizer lacks an Omega preparation receipt; disable tokenizer preparation to explicitly use it, or choose a new tokenizer path")?;
    ensure!(
        saved == tokenizer_receipt(c, root, path)?,
        "Tokenizer or preparation inputs changed; choose a new tokenizer path"
    );
    Ok(())
}
