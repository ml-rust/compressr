//! The types a ladder allocation is stated in, and the pruning that makes a
//! tensor's options into a ladder.
//!
//! A menu arrives in any order and may hold options no allocation should ever
//! choose. What the greedy in [`super::ladder`] needs is a LADDER: bytes
//! strictly increasing, damage strictly decreasing, so that every step has a
//! defined and positive marginal ratio. [`prune`] is what turns one into the
//! other, and the count it returns is the number of steps refused for buying
//! nothing.

use anyhow::{bail, Result};

/// One measured rung: what an encoding costs this tensor, and what it does to
/// the metric.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderOption {
    /// The encoding, as the report and the tie-break name it
    pub encoding: String,
    /// Payload bytes this tensor takes at that encoding
    pub bytes: u64,
    /// Measured task-metric damage at that encoding
    pub delta: f64,
}

/// One tensor's menu.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderMenu {
    /// The tensor's position in the caller's plan list
    pub index: usize,
    /// The tensor's name, as the source checkpoint spells it
    pub name: String,
    /// Every option measured for it, in any order
    pub options: Vec<LadderOption>,
    /// The option the two-way allocation would have forced on it
    ///
    /// Its position in [`LadderMenu::options`]. The allocation is reported
    /// against it: an option below it is a demotion, above it a promotion.
    pub base: usize,
}

/// What the allocation chose for one tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderChoice {
    /// The tensor's position in the caller's plan list
    pub index: usize,
    /// The encoding chosen
    pub encoding: String,
    /// Payload bytes at that encoding
    pub bytes: u64,
    /// Measured damage at that encoding
    pub delta: f64,
    /// Bytes the tensor's base option would have cost
    pub base_bytes: u64,
    /// Measured damage the tensor's base option would have done
    pub base_delta: f64,
}

impl LadderChoice {
    /// True when the chosen encoding is cheaper than the base option.
    pub fn is_demoted(&self) -> bool {
        self.bytes < self.base_bytes
    }

    /// True when the chosen encoding costs more than the base option.
    pub fn is_promoted(&self) -> bool {
        self.bytes > self.base_bytes
    }
}

/// What one ladder allocation did.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderOutcome {
    /// One choice per menu, in the caller's order
    pub chosen: Vec<LadderChoice>,
    /// Bytes the all-cheapest allocation the greedy starts from would write
    pub start_bytes: u64,
    /// Measured damage of that starting allocation
    pub start_damage: f64,
    /// Bytes the chosen allocation writes
    pub chosen_bytes: u64,
    /// Measured damage of the chosen allocation
    pub chosen_damage: f64,
    /// Bytes an all-base allocation would write
    pub base_bytes: u64,
    /// Measured damage an all-base allocation would do
    pub base_damage: f64,
    /// Bytes the greedy was allowed to spend above the starting allocation
    pub available_bytes: u64,
    /// Bytes it actually spent
    pub spent_bytes: u64,
    /// Upgrade steps taken
    pub upgrades: usize,
    /// Steps refused because their damage reduction is not positive
    ///
    /// Every dominated option is one such step: it costs bytes and buys no
    /// measured improvement. They are counted where they are dropped, before
    /// the ranking, so none of them can occupy a step.
    pub nonpositive_steps: usize,
    /// Steps the remaining budget could not pay for
    pub unaffordable_steps: usize,
}

impl LadderOutcome {
    /// Budget bytes the greedy could not place.
    pub fn unspent_bytes(&self) -> u64 {
        self.available_bytes.saturating_sub(self.spent_bytes)
    }
}

/// One tensor's pruned ladder: bytes strictly increasing, damage strictly
/// decreasing.
pub(super) struct Rungs {
    /// The tensor's position in the caller's plan list
    pub(super) index: usize,
    /// The rungs, cheapest first
    pub(super) options: Vec<LadderOption>,
    /// The option a two-way allocation would have written, which the report
    /// measures every choice against
    ///
    /// It is not a rung. A base option another option dominates is dropped
    /// from the ladder like any other dominated option — keeping it would put
    /// a step buying nothing in front of every step above it, and the tensor
    /// would be stuck below its own ladder.
    pub(super) base: LadderOption,
}

/// One tensor's options with the dominated ones removed, the option the
/// two-way allocation would have written, and how many were dropped.
///
/// The order is by bytes ascending, and an option is kept only when it does
/// strictly less damage than every cheaper option kept before it. That is
/// exactly the dominance rule: an option costing at least as many bytes for at
/// least as much damage is never worth choosing. What remains is a strictly
/// increasing byte ladder with strictly decreasing damage, so every marginal
/// ratio is defined and positive.
///
/// A dropped option is a step whose damage reduction would not have been
/// positive, which is why the count is returned: the caller reports those
/// steps, and pruning is where they are refused.
///
/// # Errors
/// The menu is empty, or its base index is not one of its options.
pub(super) fn prune(menu: &LadderMenu) -> Result<(Vec<LadderOption>, LadderOption, usize)> {
    if menu.options.is_empty() {
        bail!(
            "Tensor '{}' reached the ladder with no measured option, so nothing names an \
             encoding for it.",
            menu.name
        );
    }
    let base = menu
        .options
        .get(menu.base)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Tensor '{}' names option {} as its base, and it holds {} option(s).",
                menu.name,
                menu.base,
                menu.options.len()
            )
        })?
        .clone();

    let mut sorted: Vec<LadderOption> = menu.options.clone();
    sorted.sort_by(|left, right| {
        left.bytes
            .cmp(&right.bytes)
            .then_with(|| left.delta.total_cmp(&right.delta))
            .then_with(|| left.encoding.cmp(&right.encoding))
    });

    let mut kept: Vec<LadderOption> = Vec::with_capacity(sorted.len());
    let mut dropped = 0usize;
    let mut best = f64::INFINITY;
    for option in sorted {
        if option.delta < best {
            best = option.delta;
            kept.push(option);
        } else {
            dropped += 1;
        }
    }
    Ok((kept, base, dropped))
}

/// One rung, or an error naming the ladder that lost it.
pub(super) fn option_at(rungs: &Rungs, rung: usize) -> Result<&LadderOption> {
    rungs.options.get(rung).ok_or_else(|| {
        anyhow::anyhow!(
            "Ladder for plan index {} has no rung {rung}, and it holds {} rung(s).",
            rungs.index,
            rungs.options.len()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option(encoding: &str, bytes: u64, delta: f64) -> LadderOption {
        LadderOption {
            encoding: encoding.to_string(),
            bytes,
            delta,
        }
    }

    /// What survives pruning is a ladder: strictly more bytes for strictly
    /// less damage, whatever order the options arrived in.
    #[test]
    fn pruning_leaves_a_ladder() {
        let menu = LadderMenu {
            index: 0,
            name: "a.weight".into(),
            options: vec![
                option("dear", 300, 0.1),
                option("wasteful", 200, 0.6),
                option("cheap", 100, 0.5),
            ],
            base: 2,
        };
        let (kept, base, dropped) = prune(&menu).expect("prunes");
        assert_eq!(dropped, 1);
        assert_eq!(base.encoding, "cheap");
        assert_eq!(kept.len(), 2);
        assert!(kept[0].bytes < kept[1].bytes);
        assert!(kept[0].delta > kept[1].delta);
    }

    /// Equal bytes are settled by damage, then by name, so one file prunes
    /// the same way twice.
    #[test]
    fn equal_bytes_are_settled_deterministically() {
        let menu = LadderMenu {
            index: 0,
            name: "a.weight".into(),
            options: vec![option("b", 100, 0.5), option("a", 100, 0.5)],
            base: 0,
        };
        let (kept, _, dropped) = prune(&menu).expect("prunes");
        assert_eq!(dropped, 1);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].encoding, "a");
    }

    /// A base option another option dominates is off the ladder: keeping it
    /// would put a step buying nothing in front of every step above it.
    #[test]
    fn a_dominated_base_is_still_reported_against() {
        let menu = LadderMenu {
            index: 0,
            name: "a.weight".into(),
            options: vec![option("cheap", 100, 0.2), option("base", 200, 0.2)],
            base: 1,
        };
        let (kept, base, _) = prune(&menu).expect("prunes");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].encoding, "cheap");
        assert_eq!(base.encoding, "base");
        assert_eq!(base.bytes, 200);
    }
}
