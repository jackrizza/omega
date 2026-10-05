//! Persistent device-aware training with explicit, transactional optimizer boundaries.
use crate::{
    TrainingSet,
    batching::{BatchConfig, collate},
    dataset::{ALL_TARGETS_OBJECTIVE, ExampleSource, checked_target_mask},
    optimization::{OptimizationOptions, prepare_gradients},
    sampling::SamplingPolicy,
};
use burn::{
    module::{AutodiffModule, Module, ModuleVisitor, ParamId},
    nn::loss::CrossEntropyLossConfig,
    optim::{
        Adam, AdamConfig, GradientsParams, Optimizer,
        adaptor::OptimizerAdaptor,
        record::{AdaptorRecord, AdaptorRecordV1},
    },
    tensor::{Tensor, backend::AutodiffBackend},
};
use omega_nn::{Gpt, GptConfig, TrainingBackend, masked_cross_entropy};
use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

type AdamOptimizer<B = TrainingBackend> = OptimizerAdaptor<Adam, Gpt<B>, B>;
pub(crate) type AdamRecord<B = TrainingBackend> =
    <AdamOptimizer<B> as Optimizer<Gpt<B>, B>>::Record;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionOptions {
    pub sampling: SamplingPolicy,
    pub batching: BatchConfig,
    pub optimization: OptimizationOptions,
}
impl SessionOptions {
    pub fn validate(&self, example_count: usize) -> Result<(), String> {
        self.sampling.validate(example_count)?;
        self.batching.validate()?;
        self.optimization.validate()
    }
    fn legacy(config: &GptConfig) -> Self {
        let mut options = Self::default();
        options.batching.max_batch_tokens =
            options.batching.max_batch_tokens.max(config.context_length);
        options
    }
}

/// The cursor is the next draw position in the epoch order, not a source index.
/// One committed minibatch is one update; there is no gradient accumulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingProgress {
    pub completed_updates: usize,
    pub completed_epochs: usize,
    pub completed_targets: usize,
    pub next_example_index: usize,
    pub targets_in_epoch: usize,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpochSummary {
    pub epoch: usize,
    pub target_count: usize,
    pub mean_pre_update_loss: f32,
}
/// Emitted only after model, Adam and counters commit together. The example index
/// is the starting draw position; example_count includes the final partial batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdateEvent {
    pub epoch: usize,
    pub example_index: usize,
    pub example_count: usize,
    pub completed_updates: usize,
    pub completed_targets: usize,
    pub target_count: usize,
    pub pre_update_loss: f32,
    pub effective_learning_rate: f64,
    pub gradient_norm: f64,
    pub clipped: bool,
    pub epoch_summary: Option<EpochSummary>,
}

/// Opt-in wall-clock measurements for a successful checked update. Durations
/// exclude the final model/optimizer assignment and progress/event bookkeeping.
/// They include instrumentation overhead and are not returned for failed updates.
/// Asynchronous backends may defer work into later host-read stages. Synchronize
/// the device around whole-update timing when comparing throughput.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepTimings {
    /// Source reads, collation, counter/order planning and input tensor creation.
    pub preparation: Duration,
    /// Forward pass, loss calculation/host read and accumulated-loss validation.
    pub forward_loss: Duration,
    /// Backward pass and extraction of parameter gradients.
    pub backward: Duration,
    /// Gradient shape/finiteness checks, global norm and optional clipping.
    pub gradient_validation: Duration,
    /// Candidate model/optimizer clones and the Adam step.
    pub optimizer: Duration,
    /// Candidate optimizer record conversion and model/Adam state validation.
    pub candidate_validation: Duration,
}

struct StepClock<const PROFILE: bool>(Option<Instant>);

impl<const PROFILE: bool> StepClock<PROFILE> {
    fn new() -> Self {
        Self(if PROFILE { Some(Instant::now()) } else { None })
    }

    fn split(&mut self) -> Duration {
        if PROFILE {
            let now = Instant::now();
            let elapsed = now.duration_since(self.0.expect("profiling clock initialized"));
            self.0 = Some(now);
            elapsed
        } else {
            Duration::ZERO
        }
    }
}

pub struct TrainingSession<
    S: ExampleSource = TrainingSet,
    B: AutodiffBackend<FloatElem = f32> = TrainingBackend,
> {
    device: B::Device,
    model: Gpt<B>,
    optimizer: AdamOptimizer<B>,
    config: GptConfig,
    set: Arc<S>,
    learning_rate: f64,
    seed: u64,
    options: SessionOptions,
    order: Vec<usize>,
    target_lengths: Vec<usize>,
    targets_per_epoch: usize,
    progress: TrainingProgress,
    epoch_weighted_loss: f64,
}
impl<S: ExampleSource, B: AutodiffBackend<FloatElem = f32>> Clone for TrainingSession<S, B> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            model: self.model.clone(),
            optimizer: self.optimizer.clone(),
            config: self.config.clone(),
            set: Arc::clone(&self.set),
            learning_rate: self.learning_rate,
            seed: self.seed,
            options: self.options.clone(),
            order: self.order.clone(),
            target_lengths: self.target_lengths.clone(),
            targets_per_epoch: self.targets_per_epoch,
            progress: self.progress,
            epoch_weighted_loss: self.epoch_weighted_loss,
        }
    }
}
pub(crate) struct RestoredTrainingState<B: AutodiffBackend<FloatElem = f32> = TrainingBackend> {
    pub model: Gpt<B>,
    pub optimizer: AdamRecord<B>,
    pub config: GptConfig,
    pub learning_rate: f64,
    pub seed: u64,
    pub options: SessionOptions,
    pub progress: TrainingProgress,
    pub epoch_weighted_loss: f64,
}
impl TrainingSession {
    pub fn new(
        config: &GptConfig,
        set: TrainingSet,
        learning_rate: f64,
        seed: u64,
    ) -> Result<Self, String> {
        Self::from_source(config, set, learning_rate, seed)
    }
    pub fn new_with_options(
        config: &GptConfig,
        set: TrainingSet,
        learning_rate: f64,
        seed: u64,
        options: SessionOptions,
    ) -> Result<Self, String> {
        Self::from_source_with_options(config, set, learning_rate, seed, options)
    }
}
fn source_lengths<S: ExampleSource>(set: &S, config: &GptConfig) -> Result<Vec<usize>, String> {
    if set.example_count() == 0 {
        return Err("Training set must contain at least one example".into());
    }
    let mut lengths = Vec::new();
    lengths
        .try_reserve_exact(set.example_count())
        .map_err(|e| format!("Cannot allocate example target lengths: {e}"))?;
    let mut total = 0usize;
    for index in 0..set.example_count() {
        let ids = set.example(index)?;
        if ids.len() < 2 || ids.len() - 1 > config.context_length {
            return Err(format!(
                "Example {index} must contain 2..=context_length+1 tokens"
            ));
        }
        if ids
            .iter()
            .any(|&id| u64::from(id) >= config.vocab_size as u64)
        {
            return Err(format!(
                "Example {index} contains an ID outside the model vocabulary"
            ));
        }
        let mask = checked_target_mask(set, index, ids.len())?;
        let length = mask.as_ref().map_or(ids.len() - 1, |mask| {
            mask.iter().filter(|included| **included).count()
        });
        total = total
            .checked_add(length)
            .ok_or("Training target count overflows usize")?;
        lengths.push(length);
    }
    if total != set.target_count()? {
        return Err("Example source target count does not match its examples".into());
    }
    Ok(lengths)
}
fn order_targets(order: &[usize], lengths: &[usize]) -> Result<usize, String> {
    order.iter().try_fold(0usize, |sum, &index| {
        sum.checked_add(lengths[index])
            .ok_or_else(|| "Epoch target count overflows usize".into())
    })
}
impl<S: ExampleSource> TrainingSession<S> {
    pub fn from_source(
        config: &GptConfig,
        set: S,
        learning_rate: f64,
        seed: u64,
    ) -> Result<Self, String> {
        Self::from_source_with_options(
            config,
            set,
            learning_rate,
            seed,
            SessionOptions::legacy(config),
        )
    }
    pub fn from_source_with_options(
        config: &GptConfig,
        set: S,
        learning_rate: f64,
        seed: u64,
        options: SessionOptions,
    ) -> Result<Self, String> {
        crate::cpu::execution_profile()?;
        Self::from_source_with_options_on_device(
            config,
            set,
            learning_rate,
            seed,
            options,
            &Default::default(),
        )
    }
}
impl<S: ExampleSource, B: AutodiffBackend<FloatElem = f32>> TrainingSession<S, B> {
    pub fn from_source_on_device(
        config: &GptConfig,
        set: S,
        learning_rate: f64,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, String> {
        Self::from_source_with_options_on_device(
            config,
            set,
            learning_rate,
            seed,
            SessionOptions::legacy(config),
            device,
        )
    }
    pub fn from_source_with_options_on_device(
        config: &GptConfig,
        set: S,
        learning_rate: f64,
        seed: u64,
        options: SessionOptions,
        device: &B::Device,
    ) -> Result<Self, String> {
        Self::from_initializer(config, set, learning_rate, seed, options, device, || {
            B::seed(seed);
            config.init::<B>(device)
        })
    }

    /// Start fresh optimization from supplied weights. Adam and progress start at
    /// zero; this is not training resume and does not restore backend RNG state.
    pub fn from_model_on_device(
        config: &GptConfig,
        set: S,
        model: Gpt<B>,
        learning_rate: f64,
        seed: u64,
        options: SessionOptions,
        device: &B::Device,
    ) -> Result<Self, String> {
        Self::from_initializer(config, set, learning_rate, seed, options, device, || {
            model.validate_config(config)?;
            if model.devices().iter().any(|current| current != device) {
                return Err("Initial model does not match the selected device".into());
            }
            let optimizer: AdamOptimizer<B> = AdamConfig::new().init();
            validate_optimizer(&model, &optimizer.to_record(), 0)?;
            Ok(model)
        })
    }

    fn from_initializer(
        config: &GptConfig,
        set: S,
        learning_rate: f64,
        seed: u64,
        options: SessionOptions,
        device: &B::Device,
        initialize: impl FnOnce() -> Result<Gpt<B>, String>,
    ) -> Result<Self, String> {
        config.validate()?;
        options.validate(set.example_count())?;
        options.optimization.learning_rate(learning_rate, 0)?;
        let target_lengths = source_lengths(&set, config)?;
        let order = options
            .sampling
            .epoch_indices(set.example_count(), 0, seed)?;
        let targets_per_epoch = order_targets(&order, &target_lengths)?;
        Ok(Self {
            device: device.clone(),
            model: initialize()?,
            optimizer: AdamConfig::new().init(),
            config: config.clone(),
            set: Arc::new(set),
            learning_rate,
            seed,
            options,
            order,
            target_lengths,
            targets_per_epoch,
            progress: TrainingProgress::default(),
            epoch_weighted_loss: 0.0,
        })
    }
    pub fn device(&self) -> &B::Device {
        &self.device
    }
    pub fn config(&self) -> &GptConfig {
        &self.config
    }
    pub fn training_set(&self) -> &S {
        &self.set
    }
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn options(&self) -> &SessionOptions {
        &self.options
    }
    pub fn progress(&self) -> TrainingProgress {
        self.progress
    }
    /// Planned supervised targets for this epoch; weighted sampling may vary.
    pub fn targets_per_epoch(&self) -> usize {
        self.targets_per_epoch
    }
    pub fn samples_per_epoch(&self) -> usize {
        self.order.len()
    }
    pub fn updates_per_epoch(&self) -> usize {
        self.order.len().div_ceil(self.options.batching.batch_size)
    }
    pub fn epoch_target_count(&self, epoch: usize) -> Result<usize, String> {
        order_targets(
            &self
                .options
                .sampling
                .epoch_indices(self.set.example_count(), epoch, self.seed)?,
            &self.target_lengths,
        )
    }
    pub fn model(&self) -> &Gpt<B> {
        &self.model
    }
    pub fn inference_model(&self) -> Gpt<B::InnerBackend> {
        self.model.valid()
    }

    /// Train one bounded minibatch. Checked data/numerical errors commit nothing.
    /// Tensor backend failures can still panic. Callbacks run after commit.
    pub fn step(&mut self) -> Result<UpdateEvent, String> {
        self.step_inner::<false>().map(|(event, _)| event)
    }

    /// Run exactly the same checked update as `step`, with opt-in stage timings.
    /// Ordinary `step` uses a separate const-generic instantiation with no clock calls.
    pub fn step_profiled(&mut self) -> Result<(UpdateEvent, StepTimings), String> {
        self.step_inner::<true>()
    }

    fn step_inner<const PROFILE: bool>(&mut self) -> Result<(UpdateEvent, StepTimings), String> {
        let mut clock = StepClock::<PROFILE>::new();
        let mut timings = StepTimings::default();
        let index = self.progress.next_example_index;
        let end = index
            .saturating_add(self.options.batching.batch_size)
            .min(self.order.len());
        let count = end - index;
        let batch = collate(
            &*self.set,
            &self.order[index..end],
            &self.config,
            &self.options.batching,
            0,
        )?;
        let length = batch.target_count();
        let epoch = self
            .progress
            .completed_epochs
            .checked_add(1)
            .ok_or("Training epoch count overflows usize")?;
        let completed_updates = self
            .progress
            .completed_updates
            .checked_add(1)
            .ok_or("Training update count overflows usize")?;
        let completed_targets = self
            .progress
            .completed_targets
            .checked_add(length)
            .ok_or("Training target count overflows usize")?;
        let targets_in_epoch = self
            .progress
            .targets_in_epoch
            .checked_add(length)
            .ok_or("Epoch target count overflows usize")?;
        let effective_learning_rate = self
            .options
            .optimization
            .learning_rate(self.learning_rate, self.progress.completed_updates)?;
        let is_epoch_end = end == self.order.len();
        // Prepare the next order before tensor work so planning errors also commit nothing.
        let next_order = if is_epoch_end {
            Some(
                self.options
                    .sampling
                    .epoch_indices(self.set.example_count(), epoch, self.seed)?,
            )
        } else {
            None
        };
        let next_targets = next_order
            .as_ref()
            .map(|order| order_targets(order, &self.target_lengths))
            .transpose()?;
        let tensors = batch.into_tensors::<B>(&self.device);
        timings.preparation = clock.split();
        let loss = if count == 1 && self.set.objective() == ALL_TARGETS_OBJECTIVE {
            // Preserve the original unpadded arithmetic for batch size one.
            let logits = self
                .model
                .forward(tensors.inputs)
                .reshape([length, self.config.vocab_size]);
            CrossEntropyLossConfig::new()
                .init::<B>(&self.device)
                .forward(logits, tensors.targets.reshape([length]))
        } else {
            let logits = self
                .model
                .forward_masked(tensors.inputs, tensors.valid.clone())?;
            masked_cross_entropy(logits, tensors.targets, tensors.loss_valid)?.loss
        };
        let value: f32 = loss.clone().into_scalar();
        if !value.is_finite() {
            return Err(format!(
                "Non-finite loss at epoch {epoch}, example {index}; update rejected"
            ));
        }
        let weighted_loss = self.epoch_weighted_loss + f64::from(value) * length as f64;
        if !weighted_loss.is_finite() {
            return Err("Non-finite accumulated training loss".into());
        }
        timings.forward_loss = clock.split();
        let mut gradients = GradientsParams::from_grads(loss.backward(), &self.model);
        timings.backward = clock.split();
        let (gradient_norm, clipped) =
            prepare_gradients(&self.model, &mut gradients, &self.options.optimization)?;
        timings.gradient_validation = clock.split();
        self.commit_candidate_inner::<PROFILE>(
            gradients,
            effective_learning_rate,
            completed_updates,
            &mut timings,
        )?;
        let epoch_summary = is_epoch_end.then_some(EpochSummary {
            epoch,
            target_count: targets_in_epoch,
            mean_pre_update_loss: (weighted_loss / targets_in_epoch as f64) as f32,
        });
        self.progress = TrainingProgress {
            completed_updates,
            completed_epochs: if is_epoch_end {
                epoch
            } else {
                self.progress.completed_epochs
            },
            completed_targets,
            next_example_index: if is_epoch_end { 0 } else { end },
            targets_in_epoch: if is_epoch_end { 0 } else { targets_in_epoch },
        };
        self.epoch_weighted_loss = if is_epoch_end { 0.0 } else { weighted_loss };
        if let Some(order) = next_order {
            self.order = order;
            self.targets_per_epoch = next_targets.expect("prepared next epoch");
        }
        let event = UpdateEvent {
            epoch,
            example_index: index,
            example_count: count,
            completed_updates,
            completed_targets,
            target_count: length,
            pre_update_loss: value,
            effective_learning_rate,
            gradient_norm,
            clipped,
            epoch_summary,
        };
        Ok((event, timings))
    }
    #[cfg(test)]
    fn commit_candidate(
        &mut self,
        gradients: GradientsParams,
        rate: f64,
        updates: usize,
    ) -> Result<(), String> {
        self.commit_candidate_inner::<false>(gradients, rate, updates, &mut StepTimings::default())
    }
    fn commit_candidate_inner<const PROFILE: bool>(
        &mut self,
        gradients: GradientsParams,
        rate: f64,
        updates: usize,
        timings: &mut StepTimings,
    ) -> Result<(), String> {
        let mut clock = StepClock::<PROFILE>::new();
        let mut optimizer = self.optimizer.clone();
        let model = optimizer.step(rate, self.model.clone(), gradients);
        timings.optimizer = clock.split();
        validate_optimizer(&model, &optimizer.to_record(), updates)
            .map_err(|e| format!("Candidate update rejected: {e}"))?;
        timings.candidate_validation = clock.split();
        self.model = model;
        self.optimizer = optimizer;
        Ok(())
    }
    pub fn advance_updates(
        &mut self,
        updates: usize,
        mut observer: impl FnMut(&Self, &UpdateEvent) -> Result<(), String>,
    ) -> Result<(), String> {
        for _ in 0..updates {
            let event = self.step()?;
            observer(self, &event)?;
        }
        Ok(())
    }
    pub(crate) fn optimizer_record(&self) -> AdamRecord<B> {
        self.optimizer.to_record()
    }
    pub(crate) fn epoch_weighted_loss(&self) -> f64 {
        self.epoch_weighted_loss
    }
    pub(crate) fn validate_state(&self) -> Result<(), String> {
        validate_progress(
            &*self.set,
            self.progress,
            self.epoch_weighted_loss,
            &self.options,
            self.seed,
        )?;
        validate_optimizer(
            &self.model,
            &self.optimizer.to_record(),
            self.progress.completed_updates,
        )
    }
    pub(crate) fn restore_on_device(
        set: S,
        state: RestoredTrainingState<B>,
        device: &B::Device,
    ) -> Result<Self, String> {
        state.model.validate_config(&state.config)?;
        if state
            .model
            .devices()
            .iter()
            .any(|current| current != device)
        {
            return Err("Restored model does not match the selected device".into());
        }
        state.options.validate(set.example_count())?;
        state
            .options
            .optimization
            .learning_rate(state.learning_rate, state.progress.completed_updates)?;
        let target_lengths = source_lengths(&set, &state.config)?;
        let targets_per_epoch = validate_progress(
            &set,
            state.progress,
            state.epoch_weighted_loss,
            &state.options,
            state.seed,
        )?;
        let order = state.options.sampling.epoch_indices(
            set.example_count(),
            state.progress.completed_epochs,
            state.seed,
        )?;
        validate_optimizer(
            &state.model,
            &state.optimizer,
            state.progress.completed_updates,
        )?;
        Ok(Self {
            device: device.clone(),
            model: state.model,
            optimizer: AdamConfig::new().init().load_record(state.optimizer),
            config: state.config,
            set: Arc::new(set),
            learning_rate: state.learning_rate,
            seed: state.seed,
            options: state.options,
            order,
            target_lengths,
            targets_per_epoch,
            progress: state.progress,
            epoch_weighted_loss: state.epoch_weighted_loss,
        })
    }
}
/// Validate draw cursor and cumulative targets. Weighted epochs are replayed from
/// portable sampler state; restore cost grows with elapsed epoch draws.
pub(crate) fn validate_progress<S: ExampleSource>(
    set: &S,
    progress: TrainingProgress,
    weighted_loss: f64,
    options: &SessionOptions,
    seed: u64,
) -> Result<usize, String> {
    options.validate(set.example_count())?;
    let count = options.sampling.samples_per_epoch(set.example_count())?;
    let batch = options.batching.batch_size;
    if progress.next_example_index >= count || !progress.next_example_index.is_multiple_of(batch) {
        return Err("Resume cursor is outside the epoch order or batch boundary".into());
    }
    let mut lengths = Vec::new();
    lengths
        .try_reserve_exact(set.example_count())
        .map_err(|e| format!("Cannot allocate resume example target lengths: {e}"))?;
    for index in 0..set.example_count() {
        let length = set.example(index)?.len();
        if length < 2 {
            return Err(format!("Resume example {index} has fewer than two tokens"));
        }
        let mask = checked_target_mask(set, index, length)?;
        lengths.push(mask.as_ref().map_or(length - 1, |mask| {
            mask.iter().filter(|included| **included).count()
        }));
    }
    let total = lengths.iter().try_fold(0usize, |sum, &length| {
        sum.checked_add(length)
            .ok_or("Resume target count overflows usize")
    })?;
    if total != set.target_count()? {
        return Err("Resume source target count mismatch".into());
    }
    let order =
        options
            .sampling
            .epoch_indices(set.example_count(), progress.completed_epochs, seed)?;
    let current_total = order_targets(&order, &lengths)?;
    let prefix = order_targets(&order[..progress.next_example_index], &lengths)?;
    let mut previous = 0usize;
    if matches!(options.sampling, SamplingPolicy::Weighted { .. }) {
        for epoch in 0..progress.completed_epochs {
            previous = previous
                .checked_add(order_targets(
                    &options
                        .sampling
                        .epoch_indices(set.example_count(), epoch, seed)?,
                    &lengths,
                )?)
                .ok_or("Resume target count overflows usize")?;
        }
    } else {
        previous = progress
            .completed_epochs
            .checked_mul(total)
            .ok_or("Resume target count overflows usize")?;
    }
    let expected_updates = progress
        .completed_epochs
        .checked_mul(count.div_ceil(batch))
        .and_then(|v| v.checked_add(progress.next_example_index / batch))
        .ok_or("Resume update count overflows usize")?;
    let expected_targets = previous
        .checked_add(prefix)
        .ok_or("Resume target count overflows usize")?;
    if progress.completed_updates != expected_updates
        || progress.completed_targets != expected_targets
        || progress.targets_in_epoch != prefix
    {
        return Err("Resume counters do not agree with the example order and cursor".into());
    }
    if !weighted_loss.is_finite()
        || weighted_loss < 0.0
        || (progress.next_example_index == 0 && weighted_loss != 0.0)
    {
        return Err("Resume partial-epoch loss is invalid for its cursor".into());
    }
    Ok(current_total)
}

fn validate_optimizer<B: AutodiffBackend<FloatElem = f32>>(
    model: &Gpt<B>,
    record: &AdamRecord<B>,
    updates: usize,
) -> Result<(), String> {
    struct Check<'a, B: AutodiffBackend<FloatElem = f32>> {
        record: &'a AdamRecord<B>,
        updates: usize,
        seen: HashSet<ParamId>,
        error: Option<String>,
        device_checks: Option<crate::numerical::DeviceChecks<B::InnerBackend>>,
    }
    impl<B: AutodiffBackend<FloatElem = f32>> ModuleVisitor<B> for Check<'_, B> {
        fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
            if self.error.is_some() {
                return;
            }
            if !self.seen.insert(id) {
                self.error = Some("Resume model has duplicate parameter IDs".into());
                return;
            }
            if let Some(checks) = &mut self.device_checks {
                checks.add(tensor.clone().inner(), false);
            } else if tensor
                .to_data()
                .iter::<f32>()
                .any(|value| !value.is_finite())
            {
                self.error = Some("Resume model has non-finite parameters".into());
                return;
            }
            if self.updates == 0 {
                return;
            }
            let Some(record) = self.record.get(&id) else {
                self.error = Some("Resume optimizer is missing a model parameter ID".into());
                return;
            };
            let rank = match record {
                AdaptorRecord::V1(AdaptorRecordV1::Rank1(_)) => 1,
                AdaptorRecord::V1(AdaptorRecordV1::Rank2(_)) => 2,
                _ => {
                    self.error = Some("Resume optimizer has an unsupported state rank".into());
                    return;
                }
            };
            if rank != D {
                self.error =
                    Some("Resume optimizer state rank does not match its parameter".into());
                return;
            }
            let state = record.clone().into_state::<D>();
            if state.momentum.time != self.updates
                || state.momentum.moment_1.dims() != tensor.dims()
                || state.momentum.moment_2.dims() != tensor.dims()
            {
                self.error =
                    Some("Resume optimizer moments, shapes or update counters are invalid".into());
                return;
            }
            if let Some(checks) = &mut self.device_checks {
                checks.add(state.momentum.moment_1, false);
                checks.add(state.momentum.moment_2, true);
            } else if state
                .momentum
                .moment_1
                .to_data()
                .iter::<f32>()
                .any(|v| !v.is_finite())
                || state
                    .momentum
                    .moment_2
                    .to_data()
                    .iter::<f32>()
                    .any(|v| !v.is_finite() || v < 0.0)
            {
                self.error =
                    Some("Resume optimizer moments, shapes or update counters are invalid".into());
            }
        }
    }
    let mut check = Check {
        record,
        updates,
        seen: HashSet::new(),
        error: None,
        device_checks: crate::numerical::on_device::<B::InnerBackend>()
            .then(crate::numerical::DeviceChecks::new),
    };
    model.visit(&mut check);
    if let Some(error) = check.error {
        return Err(error);
    }
    if (updates == 0 && !record.is_empty()) || (updates > 0 && record.len() != check.seen.len()) {
        return Err("Resume optimizer parameter IDs do not match the model/update count".into());
    }
    if check.device_checks.is_some_and(|checks| !checks.finish()) {
        return Err(
            "Model/optimizer contains non-finite parameters or moments, or negative second moments"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        module::{Module, ModuleMapper, ParamId},
        tensor::Tensor,
    };

    use omega_nn::{Cpu, token_tensor};

    fn config() -> GptConfig {
        GptConfig {
            vocab_size: 4,
            context_length: 4,
            d_model: 8,
            num_heads: 2,
            num_layers: 1,
            d_ff: 16,
        }
    }

    fn set(examples: Vec<Vec<u32>>) -> TrainingSet {
        TrainingSet {
            examples,
            files: Vec::new(),
            token_count: 0,
        }
    }

    fn session() -> TrainingSession {
        TrainingSession::new(&config(), set(vec![vec![0, 1], vec![2, 3, 0, 1]]), 0.01, 42).unwrap()
    }

    fn logits<S: ExampleSource>(session: &TrainingSession<S>) -> Vec<f32> {
        let input = token_tensor::<Cpu>(&[0, 1, 2], 4, 4, &Default::default()).unwrap();
        session
            .inference_model()
            .forward(input)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn explicit_device_session_restores_and_evaluates_through_shared_loop() {
        let device = Default::default();
        let options = SessionOptions {
            sampling: SamplingPolicy::Shuffle,
            batching: BatchConfig {
                batch_size: 2,
                max_batch_tokens: 16,
            },
            optimization: OptimizationOptions {
                gradient_clip_norm: Some(0.1),
                warmup_updates: 4,
            },
        };
        let mut run =
            TrainingSession::<TrainingSet, TrainingBackend>::from_source_with_options_on_device(
                &config(),
                set(vec![vec![0, 1], vec![1, 2, 3], vec![3, 0, 1, 2]]),
                0.01,
                42,
                options,
                &device,
            )
            .unwrap();
        let mut fresh = TrainingSession::from_model_on_device(
            run.config(),
            set(run.training_set().examples.clone()),
            run.model().clone(),
            run.learning_rate(),
            run.seed(),
            run.options().clone(),
            &device,
        )
        .unwrap();
        assert_eq!(fresh.progress(), TrainingProgress::default());
        assert_eq!(run.step().unwrap(), fresh.step().unwrap());
        let state = RestoredTrainingState {
            model: run.model().clone(),
            optimizer: run.optimizer_record(),
            config: run.config().clone(),
            learning_rate: run.learning_rate(),
            seed: run.seed(),
            options: run.options().clone(),
            progress: run.progress(),
            epoch_weighted_loss: run.epoch_weighted_loss(),
        };
        let mut restored = TrainingSession::<TrainingSet, TrainingBackend>::restore_on_device(
            set(run.training_set().examples.clone()),
            state,
            &device,
        )
        .unwrap();
        assert_eq!(restored.device(), &device);
        for _ in 0..4 {
            assert_eq!(run.step().unwrap(), restored.step().unwrap());
            assert_eq!(logits(&run), logits(&restored));
        }
        let model = run.inference_model();
        assert_eq!(
            crate::evaluation::evaluate(&model, run.config(), run.training_set()).unwrap(),
            crate::evaluation::evaluate_on_device(
                &model,
                run.config(),
                run.training_set(),
                run.device()
            )
            .unwrap(),
        );
    }

    #[test]
    fn minibatches_preserve_all_targets_and_final_partial_batch() {
        let options = SessionOptions {
            batching: BatchConfig {
                batch_size: 2,
                max_batch_tokens: 16,
            },
            ..Default::default()
        };
        let mut run = TrainingSession::new_with_options(
            &config(),
            set(vec![vec![0, 1], vec![1, 2, 3], vec![3, 0, 1, 2]]),
            0.01,
            4,
            options,
        )
        .unwrap();
        assert_eq!(run.samples_per_epoch(), 3);
        assert_eq!(run.updates_per_epoch(), 2);
        let first = run.step().unwrap();
        let last = run.step().unwrap();
        assert_eq!((first.example_count, first.target_count), (2, 3));
        assert_eq!((last.example_count, last.target_count), (1, 3));
        assert_eq!(last.example_index, 2);
        assert_eq!(last.epoch_summary.unwrap().target_count, 6);
        assert_eq!(run.progress().next_example_index, 0);
        run.validate_state().unwrap();
        let mean =
            (f64::from(first.pre_update_loss) * 3.0 + f64::from(last.pre_update_loss) * 3.0) / 6.0;
        assert_eq!(
            last.epoch_summary.unwrap().mean_pre_update_loss,
            mean as f32
        );
    }

    #[test]
    fn defaults_match_original_single_example_update_and_final_candidate_is_transactional() {
        let mut run = session();
        let mut expected_model = run.model.clone();
        let mut expected_optimizer = run.optimizer.clone();
        let ids = &run.training_set().examples[0];
        let length = ids.len() - 1;
        let input =
            token_tensor::<TrainingBackend>(&ids[..length], 4, 4, &Default::default()).unwrap();
        let target = token_tensor::<TrainingBackend>(&ids[1..], 4, 4, &Default::default())
            .unwrap()
            .reshape([length]);
        let loss = CrossEntropyLossConfig::new()
            .init::<TrainingBackend>(&Default::default())
            .forward(expected_model.forward(input).reshape([length, 4]), target);
        let gradients = GradientsParams::from_grads(loss.backward(), &expected_model);
        expected_model = expected_optimizer.step(0.01, expected_model, gradients);
        run.step().unwrap();
        let input = token_tensor::<Cpu>(&[0, 1, 2], 4, 4, &Default::default()).unwrap();
        assert_eq!(
            logits(&run),
            expected_model
                .valid()
                .forward(input)
                .to_data()
                .to_vec::<f32>()
                .unwrap()
        );
        let before = logits(&run);
        let progress = run.progress();
        fn optimizer_value(record: AdamRecord) -> serde_json::Value {
            use burn::record::{FullPrecisionSettings, Record};
            serde_json::to_value(<AdamRecord as Record<TrainingBackend>>::into_item::<
                FullPrecisionSettings,
            >(record))
            .unwrap()
        }
        let before_optimizer = optimizer_value(run.optimizer_record());
        struct Fill {
            gradients: GradientsParams,
        }
        impl ModuleVisitor<TrainingBackend> for Fill {
            fn visit_float<const D: usize>(
                &mut self,
                id: ParamId,
                tensor: &Tensor<TrainingBackend, D>,
            ) {
                self.gradients.register(
                    id,
                    Tensor::<Cpu, D>::ones(tensor.dims(), &Default::default()),
                );
            }
        }
        let mut fill = Fill {
            gradients: GradientsParams::new(),
        };
        run.model.visit(&mut fill);
        assert!(
            run.commit_candidate(fill.gradients, f64::MAX, 2)
                .unwrap_err()
                .contains("Candidate update rejected")
        );
        assert_eq!(logits(&run), before);
        assert_eq!(run.progress(), progress);
        assert_eq!(optimizer_value(run.optimizer_record()), before_optimizer);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires qualified Vulkan GPU"]
    fn gpu_checks_preserve_unclipped_updates_and_reject_candidates_transactionally() {
        use crate::gpu::{Gpu, GpuTraining, initialize_vulkan};
        use burn::record::{FullPrecisionSettings, Record};
        let selected = initialize_vulkan(0).unwrap();
        let device = selected.device();
        let mut run = TrainingSession::<_, GpuTraining>::from_source_with_options_on_device(
            &config(),
            set(vec![vec![0, 1, 2, 3]]),
            0.01,
            42,
            SessionOptions::default(),
            device,
        )
        .unwrap();
        let expected_optimizer = run.optimizer.clone();
        let model = run.model.clone();
        let input = token_tensor::<GpuTraining>(&[0, 1, 2], 4, 4, device).unwrap();
        let target = token_tensor::<GpuTraining>(&[1, 2, 3], 4, 4, device)
            .unwrap()
            .reshape([3]);
        let loss = CrossEntropyLossConfig::new()
            .init::<GpuTraining>(device)
            .forward(model.forward(input).reshape([3, 4]), target);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        let mut expected_optimizer = expected_optimizer;
        let expected_model = expected_optimizer.step(0.01, model, gradients);
        fn model_value(model: &Gpt<GpuTraining>) -> serde_json::Value {
            serde_json::to_value(
                model
                    .clone()
                    .into_record()
                    .into_item::<FullPrecisionSettings>(),
            )
            .unwrap()
        }
        fn optimizer_value(optimizer: &AdamOptimizer<GpuTraining>) -> serde_json::Value {
            serde_json::to_value(optimizer.to_record().into_item::<FullPrecisionSettings>())
                .unwrap()
        }
        run.step().unwrap();
        assert_eq!(model_value(&run.model), model_value(&expected_model));
        assert_eq!(
            optimizer_value(&run.optimizer),
            optimizer_value(&expected_optimizer)
        );
        let before = model_value(&run.model);
        let before_optimizer = optimizer_value(&run.optimizer);
        let before_progress = run.progress();
        let before_loss = run.epoch_weighted_loss;
        struct Fill {
            gradients: GradientsParams,
        }
        impl ModuleVisitor<GpuTraining> for Fill {
            fn visit_float<const D: usize>(
                &mut self,
                id: ParamId,
                tensor: &Tensor<GpuTraining, D>,
            ) {
                self.gradients
                    .register(id, Tensor::<Gpu, D>::ones(tensor.dims(), &tensor.device()));
            }
        }
        let mut fill = Fill {
            gradients: GradientsParams::new(),
        };
        run.model.visit(&mut fill);
        assert!(
            run.commit_candidate(fill.gradients, f64::MAX, 2)
                .unwrap_err()
                .contains("Candidate update rejected")
        );
        assert_eq!(model_value(&run.model), before);
        assert_eq!(optimizer_value(&run.optimizer), before_optimizer);
        assert_eq!(run.progress(), before_progress);
        assert_eq!(run.epoch_weighted_loss, before_loss);
    }

    #[test]
    fn profiled_steps_preserve_events_progress_parameters_and_optimizer() {
        use burn::record::{FullPrecisionSettings, Record};

        fn optimizer_value(run: &TrainingSession) -> serde_json::Value {
            serde_json::to_value(<AdamRecord as Record<TrainingBackend>>::into_item::<
                FullPrecisionSettings,
            >(run.optimizer_record()))
            .unwrap()
        }

        for batch_size in [1, 2] {
            let options = SessionOptions {
                sampling: SamplingPolicy::Shuffle,
                batching: BatchConfig {
                    batch_size,
                    max_batch_tokens: 16,
                },
                optimization: OptimizationOptions {
                    gradient_clip_norm: Some(0.1),
                    warmup_updates: 4,
                },
            };
            let mut normal = TrainingSession::new_with_options(
                &config(),
                set(vec![vec![0, 1], vec![1, 2, 3], vec![2, 3, 0, 1]]),
                0.01,
                42,
                options,
            )
            .unwrap();
            // Initialize lazy parameters and Adam once before forking; comparisons
            // must not depend on reseeding the process-shared backend RNG.
            normal.step().unwrap();
            let mut profiled = normal.clone();
            for _ in 0..4 {
                let expected = normal.step().unwrap();
                let (actual, _timings) = profiled.step_profiled().unwrap();
                assert_eq!(actual, expected);
                assert_eq!(profiled.progress(), normal.progress());
                assert_eq!(profiled.epoch_weighted_loss(), normal.epoch_weighted_loss());
                assert_eq!(logits(&profiled), logits(&normal));
                assert_eq!(optimizer_value(&profiled), optimizer_value(&normal));
            }
            let (_, untimed) = normal.step_inner::<false>().unwrap();
            assert_eq!(untimed, StepTimings::default());
        }
    }

    #[test]
    fn chunked_calls_preserve_model_optimizer_cursor_and_losses() {
        let mut uninterrupted = session();
        // Fork initialized parameters, rather than reseeding a process-shared RNG.
        let mut chunked = uninterrupted.clone();
        let mut expected = Vec::new();
        uninterrupted
            .advance_updates(7, |_, event| {
                expected.push(*event);
                Ok(())
            })
            .unwrap();
        let mut actual = Vec::new();
        for updates in [1, 2, 0, 4] {
            chunked
                .advance_updates(updates, |_, event| {
                    actual.push(*event);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(expected, actual);
        assert_eq!(uninterrupted.progress(), chunked.progress());
        assert_eq!(logits(&uninterrupted), logits(&chunked));
        assert!(!chunked.optimizer.to_record().is_empty());

        // A reset optimizer is invalid after committed updates; reject rather
        // than silently discarding accumulated moments.
        let mut reset = chunked.clone();
        reset.optimizer = AdamConfig::new().init();
        let before = logits(&reset);
        let progress = reset.progress();
        assert!(
            reset
                .step()
                .unwrap_err()
                .contains("Candidate update rejected")
        );
        assert_eq!(logits(&reset), before);
        assert_eq!(reset.progress(), progress);
    }

    #[test]
    fn events_mark_update_and_weighted_epoch_boundaries() {
        let mut session = session();
        assert_eq!(session.progress(), TrainingProgress::default());
        assert_eq!(session.targets_per_epoch(), 4);
        assert_eq!(session.learning_rate(), 0.01);
        assert_eq!(session.seed(), 42);
        assert_eq!(session.config(), &config());
        assert_eq!(session.training_set().examples.len(), 2);
        session.model().validate_config(session.config()).unwrap();
        let mut events = Vec::new();
        session
            .advance_updates(4, |current, event| {
                assert_eq!(
                    current.progress().completed_updates,
                    event.completed_updates
                );
                assert_eq!(
                    current.progress().completed_targets,
                    event.completed_targets
                );
                events.push(*event);
                Ok(())
            })
            .unwrap();
        for (index, pair) in events.chunks_exact(2).enumerate() {
            assert_eq!((pair[0].epoch, pair[0].example_index), (index + 1, 0));
            assert_eq!((pair[1].epoch, pair[1].example_index), (index + 1, 1));
            assert_eq!((pair[0].target_count, pair[1].target_count), (1, 3));
            assert!(pair[0].epoch_summary.is_none());
            let summary = pair[1].epoch_summary.unwrap();
            assert_eq!(summary.epoch, index + 1);
            assert_eq!(summary.target_count, 4);
            let expected = (f64::from(pair[0].pre_update_loss)
                + 3.0 * f64::from(pair[1].pre_update_loss))
                / 4.0;
            assert_eq!(summary.mean_pre_update_loss, expected as f32);
        }
        assert_eq!(
            session.progress(),
            TrainingProgress {
                completed_updates: 4,
                completed_epochs: 2,
                completed_targets: 8,
                next_example_index: 0,
                targets_in_epoch: 0
            }
        );
    }

    #[test]
    fn observer_failure_leaves_update_committed_and_can_continue() {
        let mut interrupted = session();
        let mut uninterrupted = interrupted.clone();
        uninterrupted.advance_updates(3, |_, _| Ok(())).unwrap();
        assert_eq!(
            interrupted
                .advance_updates(3, |_, _| Err("observer failed".into()))
                .unwrap_err(),
            "observer failed"
        );
        assert_eq!(interrupted.progress().completed_updates, 1);
        assert_eq!(interrupted.progress().next_example_index, 1);
        let mut updates = Vec::new();
        interrupted
            .advance_updates(2, |_, event| {
                updates.push(event.completed_updates);
                Ok(())
            })
            .unwrap();
        assert_eq!(updates, [2, 3]);
        assert_eq!(interrupted.progress(), uninterrupted.progress());
        assert_eq!(logits(&interrupted), logits(&uninterrupted));
    }

    #[test]
    fn nonfinite_loss_does_not_commit_an_update() {
        struct NonfiniteParameters;
        impl ModuleMapper<TrainingBackend> for NonfiniteParameters {
            fn map_float<const D: usize>(
                &mut self,
                _id: ParamId,
                tensor: Tensor<TrainingBackend, D>,
            ) -> Tensor<TrainingBackend, D> {
                tensor.mul_scalar(f32::NAN)
            }
        }
        let mut session = session();
        session.model = session.model.clone().map(&mut NonfiniteParameters);
        assert!(
            TrainingSession::from_model_on_device(
                session.config(),
                set(vec![vec![0, 1]]),
                session.model().clone(),
                session.learning_rate(),
                session.seed(),
                session.options().clone(),
                session.device(),
            )
            .is_err()
        );
        assert!(
            session
                .step()
                .unwrap_err()
                .contains("Non-finite loss at epoch 1, example 0")
        );
        assert_eq!(session.progress(), TrainingProgress::default());
        assert_eq!(session.epoch_weighted_loss, 0.0);
        assert!(session.optimizer.to_record().is_empty());
    }

    #[test]
    fn source_read_failure_does_not_advance_or_reset_session() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct FallibleSource {
            set: TrainingSet,
            fail: Arc<AtomicBool>,
        }
        impl ExampleSource for FallibleSource {
            fn example_count(&self) -> usize {
                self.set.example_count()
            }
            fn target_count(&self) -> Result<usize, String> {
                self.set.target_count()
            }
            fn identity(&self) -> Result<String, String> {
                self.set.identity()
            }
            fn example(&self, index: usize) -> Result<Vec<u32>, String> {
                if self.fail.load(Ordering::Relaxed) {
                    Err("source data changed".into())
                } else {
                    self.set.example(index)
                }
            }
        }
        let fail = Arc::new(AtomicBool::new(false));
        let source = FallibleSource {
            set: set(vec![vec![0, 1, 2]]),
            fail: Arc::clone(&fail),
        };
        let mut session = TrainingSession::from_source(&config(), source, 0.01, 42).unwrap();
        // Source itself is not Clone; cloning shares its immutable handle.
        let mut control = session.clone();
        let before = logits(&session);
        fail.store(true, Ordering::Relaxed);
        assert!(session.step().unwrap_err().contains("source data changed"));
        assert_eq!(session.progress(), TrainingProgress::default());
        assert_eq!(logits(&session), before);
        fail.store(false, Ordering::Relaxed);
        assert_eq!(session.step().unwrap(), control.step().unwrap());
        assert_eq!(logits(&session), logits(&control));
    }

    #[test]
    fn invalid_inputs_and_counter_overflow_are_rejected() {
        for examples in [
            vec![],
            vec![vec![]],
            vec![vec![0]],
            vec![vec![0, 4]],
            vec![vec![0; 6]],
        ] {
            assert!(TrainingSession::new(&config(), set(examples), 0.01, 42).is_err());
        }
        for rate in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            assert!(TrainingSession::new(&config(), set(vec![vec![0, 1]]), rate, 42).is_err());
        }
        let mut invalid = config();
        invalid.num_heads = 3;
        assert!(TrainingSession::new(&invalid, set(vec![vec![0, 1]]), 0.01, 42).is_err());
        let mut session = session();
        session
            .advance_updates(0, |_, _| panic!("zero updates must not call observer"))
            .unwrap();
        let before = logits(&session);
        session.progress.completed_updates = usize::MAX;
        assert!(
            session
                .step()
                .unwrap_err()
                .contains("update count overflows")
        );
        assert_eq!(session.progress().next_example_index, 0);
        assert!(session.optimizer.to_record().is_empty());
        assert_eq!(logits(&session), before);
    }
}
