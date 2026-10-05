//! Deterministic epoch ordering over a caller-supplied training partition.
//!
//! No tokenizer, model, validation partition, backend RNG or mutable RNG state is
//! used. Persist the algorithm identifier, policy (including ordered groups and
//! member indices), seed, zero-based epoch and next draw cursor to reproduce an
//! epoch exactly. The underlying example-source identity must also match.
//! An ordering vector holds one usize per epoch draw; group validation retains
//! one byte per source example. This is additional memory even for token caches.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Changing any PRNG, shuffle, seed-mixing or draw rule requires a new version.
pub const SAMPLING_ALGORITHM: &str = "splitmix64-fisher-yates-weighted-v1";

/// A named, nonempty subset of TRAINING example indices. Groups must be disjoint
/// and exhaustive over `0..example_count`; their order and member order are part
/// of the reproducible policy. A weight of 3 gives this group three times the
/// selection probability of a group with weight 1, regardless of group size.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingGroup {
    pub name: String,
    pub indices: Vec<usize>,
    pub weight: u64,
}

/// Fixed remains the default and covers every example once in source order.
/// Shuffle covers each once in a seeded Fisher-Yates permutation. Weighted
/// sampling draws groups by positive integer weights, then chooses uniformly
/// within the selected group, with replacement at both stages. A weighted epoch
/// is exactly `samples_per_epoch` draws and may repeat or omit source examples;
/// target totals can vary across epochs when examples have different lengths.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SamplingPolicy {
    #[default]
    Fixed,
    Shuffle,
    Weighted {
        groups: Vec<SamplingGroup>,
        samples_per_epoch: usize,
    },
}

impl<'de> Deserialize<'de> for SamplingPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Serde ignores extra fields on internally tagged unit variants even
        // with deny_unknown_fields. Empty struct variants enforce the schema.
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum Repr {
            Fixed {},
            Shuffle {},
            Weighted {
                groups: Vec<SamplingGroup>,
                samples_per_epoch: usize,
            },
        }
        Ok(match Repr::deserialize(deserializer)? {
            Repr::Fixed {} => Self::Fixed,
            Repr::Shuffle {} => Self::Shuffle,
            Repr::Weighted {
                groups,
                samples_per_epoch,
            } => Self::Weighted {
                groups,
                samples_per_epoch,
            },
        })
    }
}

impl SamplingPolicy {
    /// Validate against the training partition only. Bounds checks prevent this
    /// policy from selecting indices outside that partition; callers are still
    /// responsible for supplying a training-only source and correct group labels.
    pub fn validate(&self, example_count: usize) -> Result<(), String> {
        if example_count == 0 {
            return Err("Sampling requires a nonempty training example source".into());
        }
        u64::try_from(example_count)
            .map_err(|_| "Training example count exceeds portable u64 sampling range")?;
        if let Self::Weighted {
            groups,
            samples_per_epoch,
        } = self
        {
            if groups.is_empty() || *samples_per_epoch == 0 {
                return Err(
                    "Weighted sampling requires nonempty groups and a positive epoch draw count"
                        .into(),
                );
            }
            let mut seen = Vec::<u8>::new();
            seen.try_reserve_exact(example_count)
                .map_err(|e| format!("Cannot allocate sampling membership validation: {e}"))?;
            seen.resize(example_count, 0);
            let mut names = BTreeSet::new();
            let mut weight_sum = 0_u64;
            for group in groups {
                if group.name.trim().is_empty() || !names.insert(&group.name) {
                    return Err("Sampling group names must be nonblank and unique".into());
                }
                if group.indices.is_empty() || group.weight == 0 {
                    return Err(format!(
                        "Sampling group {:?} must have members and a positive integer weight",
                        group.name
                    ));
                }
                weight_sum = weight_sum
                    .checked_add(group.weight)
                    .ok_or("Sampling group weight sum overflows u64")?;
                for &index in &group.indices {
                    let member = seen.get_mut(index).ok_or_else(|| {
                        format!(
                            "Sampling group {:?} index {index} is outside the training source",
                            group.name
                        )
                    })?;
                    if *member != 0 {
                        return Err(format!(
                            "Training example {index} appears more than once in sampling groups"
                        ));
                    }
                    *member = 1;
                }
            }
            if seen.contains(&0) {
                return Err(
                    "Sampling groups must cover every training example exactly once".into(),
                );
            }
        }
        Ok(())
    }

    pub fn samples_per_epoch(&self, example_count: usize) -> Result<usize, String> {
        self.validate(example_count)?;
        Ok(match self {
            Self::Fixed | Self::Shuffle => example_count,
            Self::Weighted {
                samples_per_epoch, ..
            } => *samples_per_epoch,
        })
    }

    /// Reconstruct the full epoch without touching global RNG state. Epoch zero
    /// is the first epoch. Failed allocations return errors; no partial ordering
    /// is returned. Weighted selection intentionally does not balance token counts.
    pub fn epoch_indices(
        &self,
        example_count: usize,
        epoch: usize,
        seed: u64,
    ) -> Result<Vec<usize>, String> {
        let count = self.samples_per_epoch(example_count)?;
        let epoch =
            u64::try_from(epoch).map_err(|_| "Epoch exceeds portable u64 sampling range")?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|e| format!("Cannot allocate epoch sampling indices: {e}"))?;
        let mut rng = SplitMix64::for_epoch(seed, epoch);
        match self {
            Self::Fixed => result.extend(0..example_count),
            Self::Shuffle => {
                result.extend(0..example_count);
                for index in (1..result.len()).rev() {
                    let other = rng.below((index + 1) as u64) as usize;
                    result.swap(index, other);
                }
            }
            Self::Weighted { groups, .. } => {
                // Validation checked overflow and strictly positive weights.
                let total: u64 = groups.iter().map(|group| group.weight).sum();
                for _ in 0..count {
                    let mut choice = rng.below(total);
                    for group in groups {
                        if choice < group.weight {
                            let member = rng.below(group.indices.len() as u64) as usize;
                            result.push(group.indices[member]);
                            break;
                        }
                        choice -= group.weight;
                    }
                }
            }
        }
        Ok(result)
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn for_epoch(seed: u64, epoch: u64) -> Self {
        // Two domain constants separate initialization from draw stepping.
        Self {
            state: mix(seed ^ 0xd1b54a32d192ed03) ^ mix(epoch ^ 0x94d049bb133111eb),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        mix(self.state)
    }

    fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        // Reject the short leading interval rather than biasing modulo draws.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % bound;
            }
        }
    }
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weighted(draws: usize) -> SamplingPolicy {
        SamplingPolicy::Weighted {
            groups: vec![
                SamplingGroup {
                    name: "first".into(),
                    indices: vec![0, 1],
                    weight: 1,
                },
                SamplingGroup {
                    name: "second".into(),
                    indices: vec![2, 3, 4],
                    weight: 3,
                },
            ],
            samples_per_epoch: draws,
        }
    }

    #[test]
    fn fixed_default_preserves_order_and_shuffle_is_repeatable_per_epoch() {
        assert_eq!(SamplingPolicy::default(), SamplingPolicy::Fixed);
        for (epoch, seed) in [(0, 0), (10, 42), (usize::MAX, u64::MAX)] {
            assert_eq!(
                SamplingPolicy::Fixed.epoch_indices(5, epoch, seed).unwrap(),
                [0, 1, 2, 3, 4]
            );
        }
        let first = SamplingPolicy::Shuffle.epoch_indices(20, 0, 42).unwrap();
        let mut sorted = first.clone();
        sorted.sort();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        assert_eq!(
            first,
            SamplingPolicy::Shuffle.epoch_indices(20, 0, 42).unwrap()
        );
        assert_ne!(
            first,
            SamplingPolicy::Shuffle.epoch_indices(20, 1, 42).unwrap()
        );
        assert_ne!(
            first,
            SamplingPolicy::Shuffle.epoch_indices(20, 0, 43).unwrap()
        );
        assert_eq!(
            SamplingPolicy::Shuffle.epoch_indices(1, 0, 42).unwrap(),
            [0]
        );
    }

    #[test]
    fn weighted_draws_repeat_with_replacement_and_follow_group_weights() {
        let policy = weighted(20_000);
        let indices = policy.epoch_indices(5, 0, 42).unwrap();
        assert_eq!(indices.len(), 20_000);
        assert!(indices.iter().all(|&index| index < 5));
        assert_eq!(indices, policy.epoch_indices(5, 0, 42).unwrap());
        assert_ne!(indices, policy.epoch_indices(5, 1, 42).unwrap());
        let first = indices.iter().filter(|&&index| index < 2).count();
        assert!((4500..5500).contains(&first), "first group draws: {first}");
        let counts: Vec<_> = (0..5)
            .map(|index| indices.iter().filter(|&&item| item == index).count())
            .collect();
        assert!(counts[0].abs_diff(counts[1]) < 500);
        assert!(counts[2].abs_diff(counts[3]) < 500 && counts[3].abs_diff(counts[4]) < 500);
        assert_eq!(weighted(1).samples_per_epoch(5).unwrap(), 1);
    }

    #[test]
    fn policy_serialization_and_epoch_cursor_reconstruction_are_exact() {
        for policy in [SamplingPolicy::Fixed, SamplingPolicy::Shuffle, weighted(17)] {
            let serialized = serde_json::to_vec(&policy).unwrap();
            let restored: SamplingPolicy = serde_json::from_slice(&serialized).unwrap();
            assert_eq!(policy, restored);
            let order = policy.epoch_indices(5, 3, 72).unwrap();
            for cursor in 0..=order.len() {
                assert_eq!(
                    order[cursor..],
                    restored.epoch_indices(5, 3, 72).unwrap()[cursor..]
                );
            }
        }
        for json in [
            r#"{"kind":"unknown"}"#,
            r#"{"kind":"fixed","surprise":1}"#,
            r#"{"kind":"weighted","groups":[],"samples_per_epoch":-1}"#,
        ] {
            assert!(
                serde_json::from_str::<SamplingPolicy>(json).is_err(),
                "{json}"
            );
        }
    }

    #[test]
    fn invalid_groups_counts_weights_and_training_bounds_are_rejected() {
        assert!(SamplingPolicy::Fixed.validate(0).is_err());
        assert!(SamplingPolicy::Shuffle.validate(0).is_err());
        assert!(weighted(0).validate(5).is_err());
        for mutation in 0..9 {
            let SamplingPolicy::Weighted {
                mut groups,
                samples_per_epoch,
            } = weighted(5)
            else {
                unreachable!()
            };
            match mutation {
                0 => groups.clear(),
                1 => groups[0].name = " ".into(),
                2 => groups[1].name = groups[0].name.clone(),
                3 => groups[0].indices.clear(),
                4 => groups[0].weight = 0,
                5 => {
                    groups[0].weight = u64::MAX;
                    groups[1].weight = 1;
                }
                6 => groups[1].indices.push(0),
                7 => groups[1].indices.push(5),
                _ => {
                    groups[1].indices.pop();
                }
            }
            let policy = SamplingPolicy::Weighted {
                groups,
                samples_per_epoch,
            };
            assert!(
                policy.epoch_indices(5, 0, 42).is_err(),
                "mutation {mutation}"
            );
        }
        assert!(weighted(usize::MAX).epoch_indices(5, 0, 42).is_err());
    }

    #[test]
    fn portable_prng_and_ordering_have_fixed_vectors() {
        let mut rng = SplitMix64 { state: 0 };
        assert_eq!(
            [rng.next(), rng.next(), rng.next()],
            [0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f]
        );
        // Golden epoch outputs freeze seed mixing, draw consumption and policies.
        assert_eq!(
            SamplingPolicy::Shuffle.epoch_indices(8, 0, 42).unwrap(),
            vec![7, 2, 1, 0, 4, 3, 6, 5]
        );
        assert_eq!(
            weighted(12).epoch_indices(5, 2, 42).unwrap(),
            vec![3, 3, 1, 2, 0, 4, 2, 1, 4, 2, 1, 2]
        );
    }
}
