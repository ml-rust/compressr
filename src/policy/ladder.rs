//! A per-tensor ladder allocation: one encoding per tensor, under a byte
//! budget, chosen from measured options.
//!
//! # What this replaces
//!
//! A two-way allocation starts every tensor at a base encoding and raises a
//! ranked subset. The base is a FLOOR, so a tensor the measurement shows no
//! damage on still pays the base price. It cannot be demoted on that
//! evidence: the delta says what the BASE did, and a tensor free at four bits
//! can be ruined at three. Extrapolating downward from it is guessing.
//!
//! A file measuring several encodings per tensor removes the guess: each
//! tensor arrives with a menu of (encoding, bytes, delta), every entry
//! measured, and the allocation is a multiple-choice knapsack — one option per
//! tensor, minimizing measured damage subject to total bytes.
//!
//! # The greedy, and where it is not optimal
//!
//! Every tensor starts on its cheapest option. The budget is then spent one
//! upgrade step at a time, always on the step with the best damage reduction
//! per additional byte, until no step fits or no step improves.
//!
//! This is the standard greedy, and it is optimal for the LINEAR relaxation
//! only when each tensor's steps are taken in convex-hull order — a step whose
//! ratio is better than the step in front of it is unreachable until that
//! worse step is paid for, and the greedy will pass over both. It is also
//! subject to the usual integrality gap: the last few bytes of budget can go
//! unspent while a large step that would have used them is refused. Neither
//! costs much here, because the menus are short (a handful of rungs) and the
//! tensors are many, so the allocation is dominated by cheap steps that fit.
//!
//! # What is never chosen
//!
//! A DOMINATED option — one costing at least as many bytes as another option
//! of the same tensor for at least as much damage — is dropped by
//! [`super::ladder_rungs::prune`] before any ranking, so it can never occupy a
//! step. That is also where a step whose damage reduction is not positive is
//! counted and refused: spending bytes to make the metric worse is wrong with
//! any budget left. What remains per tensor is a strictly increasing byte
//! ladder with strictly decreasing damage.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::Result;

use super::ladder_rungs::{option_at, prune, Rungs};
pub use super::ladder_rungs::{LadderChoice, LadderMenu, LadderOption, LadderOutcome};

/// One upgrade step, ordered by damage reduction per additional byte.
struct Step {
    /// Damage reduction per additional byte
    ratio: f64,
    /// Additional bytes the step costs
    bytes: u64,
    /// Damage the step removes
    reduction: f64,
    /// The ladder's position in this allocation
    tensor: usize,
    /// The tensor's position in the caller's plan list, which breaks a tie
    plan: usize,
    /// The rung the step lands on
    rung: usize,
    /// The encoding that rung carries, which breaks a tie the plan index does
    /// not
    encoding: String,
}

impl PartialEq for Step {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Step {}

impl PartialOrd for Step {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Step {
    /// Greater is taken first: the better ratio, then the lower plan index,
    /// then the earlier encoding name.
    ///
    /// The heap pops the greatest, so the two tie-breaks are reversed here —
    /// a LOWER index and an EARLIER name have to compare as greater. Nothing
    /// is left to arrival order, so two runs over one file take the same
    /// steps in the same order.
    fn cmp(&self, other: &Self) -> Ordering {
        self.ratio
            .total_cmp(&other.ratio)
            .then_with(|| other.plan.cmp(&self.plan))
            .then_with(|| other.encoding.cmp(&self.encoding))
    }
}

/// Choose one option per tensor, spending at most `available_bytes` above the
/// all-cheapest allocation.
///
/// # Errors
/// A menu holds no option, or names a base option that is not in it. Both are
/// a caller that resolved a tensor to nothing, which must fail loudly rather
/// than allocate that tensor by default.
pub fn allocate_ladder(menus: &[LadderMenu], available_bytes: u64) -> Result<LadderOutcome> {
    let mut ladders: Vec<Rungs> = Vec::with_capacity(menus.len());
    let mut nonpositive_steps = 0usize;
    for menu in menus {
        let (options, base, dropped) = prune(menu)?;
        nonpositive_steps += dropped;
        ladders.push(Rungs {
            index: menu.index,
            options,
            base,
        });
    }

    let mut at: Vec<usize> = vec![0; ladders.len()];
    let mut heap: BinaryHeap<Step> = BinaryHeap::new();
    for (tensor, rungs) in ladders.iter().enumerate() {
        if let Some(step) = step_of(tensor, rungs, 0) {
            heap.push(step);
        }
    }

    let mut spent_bytes = 0u64;
    let mut upgrades = 0usize;
    let mut unaffordable_steps = 0usize;
    while let Some(step) = heap.pop() {
        if step.reduction <= 0.0 {
            // Pruning leaves no such step; this is the guard that keeps one
            // from being taken if a menu ever reaches here unpruned.
            nonpositive_steps += 1;
            continue;
        }
        if spent_bytes.saturating_add(step.bytes) > available_bytes {
            // Every later step of this tensor sits behind this one, so the
            // tensor is finished. A cheaper step of another tensor may still
            // fit, so the spend continues.
            unaffordable_steps += 1;
            continue;
        }
        spent_bytes += step.bytes;
        upgrades += 1;
        let tensor = step.tensor;
        at[tensor] = step.rung;
        let next = ladders
            .get(tensor)
            .and_then(|rungs| step_of(tensor, rungs, step.rung));
        if let Some(next) = next {
            heap.push(next);
        }
    }

    let mut chosen = Vec::with_capacity(ladders.len());
    let mut start_bytes = 0u64;
    let mut start_damage = 0.0f64;
    let mut chosen_bytes = 0u64;
    let mut chosen_damage = 0.0f64;
    let mut base_bytes = 0u64;
    let mut base_damage = 0.0f64;
    for (tensor, rungs) in ladders.iter().enumerate() {
        let first = option_at(rungs, 0)?;
        let base = &rungs.base;
        let taken = option_at(rungs, at.get(tensor).copied().unwrap_or(0))?;
        start_bytes = start_bytes.saturating_add(first.bytes);
        start_damage += first.delta;
        chosen_bytes = chosen_bytes.saturating_add(taken.bytes);
        chosen_damage += taken.delta;
        base_bytes = base_bytes.saturating_add(base.bytes);
        base_damage += base.delta;
        chosen.push(LadderChoice {
            index: rungs.index,
            encoding: taken.encoding.clone(),
            bytes: taken.bytes,
            delta: taken.delta,
            base_bytes: base.bytes,
            base_delta: base.delta,
        });
    }

    Ok(LadderOutcome {
        chosen,
        start_bytes,
        start_damage,
        chosen_bytes,
        chosen_damage,
        base_bytes,
        base_damage,
        available_bytes,
        spent_bytes,
        upgrades,
        nonpositive_steps,
        unaffordable_steps,
    })
}

/// The step from rung `from` to the next rung, `None` at the top of a ladder.
fn step_of(tensor: usize, rungs: &Rungs, from: usize) -> Option<Step> {
    let current = rungs.options.get(from)?;
    let next = rungs.options.get(from + 1)?;
    let bytes = next.bytes.saturating_sub(current.bytes);
    let reduction = current.delta - next.delta;
    Some(Step {
        // Bytes never fall along a pruned ladder, and a step of zero bytes
        // would be a free improvement; the floor keeps the ratio finite so
        // one such step cannot outrank every other by being infinite.
        ratio: reduction / bytes.max(1) as f64,
        bytes,
        reduction,
        tensor,
        plan: rungs.index,
        rung: from + 1,
        encoding: next.encoding.clone(),
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

    fn menu(index: usize, options: Vec<LadderOption>, base: usize) -> LadderMenu {
        LadderMenu {
            index,
            name: format!("tensor.{index}.weight"),
            options,
            base,
        }
    }

    /// A menu with one option leaves the allocator no choice, and the tensor
    /// lands on it whatever the budget is. This is the two-way file's shape,
    /// one tensor at a time.
    #[test]
    fn a_single_option_menu_is_the_option() {
        let menus = vec![menu(0, vec![option("Q4", 100, 0.5)], 0)];
        let outcome = allocate_ladder(&menus, 10_000).expect("allocates");
        assert_eq!(outcome.chosen.len(), 1);
        assert_eq!(outcome.chosen[0].encoding, "Q4");
        assert_eq!(outcome.spent_bytes, 0);
        assert_eq!(outcome.chosen_bytes, 100);
        assert_eq!(outcome.base_bytes, 100);
        assert!(!outcome.chosen[0].is_demoted());
        assert!(!outcome.chosen[0].is_promoted());
    }

    /// An option costing more bytes for no less damage is never chosen, even
    /// with the budget wide open.
    #[test]
    fn a_dominated_option_is_never_chosen() {
        let menus = vec![menu(
            0,
            vec![
                option("cheap", 100, 0.4),
                // More bytes, more damage: dominated by both others.
                option("wasteful", 150, 0.9),
                option("dear", 200, 0.1),
            ],
            0,
        )];
        let outcome = allocate_ladder(&menus, 10_000).expect("allocates");
        assert_eq!(outcome.chosen[0].encoding, "dear");
        assert_eq!(outcome.spent_bytes, 100);
        assert_eq!(outcome.nonpositive_steps, 1);
    }

    /// A step that buys no measured improvement is refused with the whole
    /// budget free.
    #[test]
    fn a_non_positive_step_is_never_taken() {
        let menus = vec![menu(
            0,
            vec![option("cheap", 100, 0.4), option("dear", 200, 0.4)],
            0,
        )];
        let outcome = allocate_ladder(&menus, 10_000).expect("allocates");
        assert_eq!(outcome.chosen[0].encoding, "cheap");
        assert_eq!(outcome.spent_bytes, 0);
        assert_eq!(outcome.upgrades, 0);
        assert_eq!(outcome.nonpositive_steps, 1);
    }

    /// The budget is a ceiling, and a step that does not fit is counted
    /// rather than taken.
    #[test]
    fn the_budget_is_respected() {
        let menus = vec![
            menu(
                0,
                vec![option("cheap", 100, 1.0), option("dear", 300, 0.0)],
                0,
            ),
            menu(
                1,
                vec![option("cheap", 100, 0.6), option("dear", 150, 0.1)],
                0,
            ),
        ];
        // 40 bytes buys neither step.
        let outcome = allocate_ladder(&menus, 40).expect("allocates");
        assert_eq!(outcome.spent_bytes, 0);
        assert_eq!(outcome.unaffordable_steps, 2);
        assert_eq!(outcome.chosen_bytes, 200);

        // 50 bytes buys the second tensor's step, which has the better ratio
        // (0.5 damage for 50 bytes against 1.0 for 200).
        let outcome = allocate_ladder(&menus, 50).expect("allocates");
        assert_eq!(outcome.spent_bytes, 50);
        assert_eq!(outcome.chosen[1].encoding, "dear");
        assert_eq!(outcome.chosen[0].encoding, "cheap");
        assert!(outcome.chosen_bytes <= outcome.start_bytes + 50);
    }

    /// Two steps of identical ratio are settled by plan index, then by
    /// encoding name, so the same file allocates the same way twice.
    #[test]
    fn a_tie_is_broken_by_plan_index_then_encoding_name() {
        let menus = vec![
            menu(
                7,
                vec![option("cheap", 100, 1.0), option("b_dear", 200, 0.5)],
                0,
            ),
            menu(
                3,
                vec![option("cheap", 100, 1.0), option("a_dear", 200, 0.5)],
                0,
            ),
        ];
        // One step's worth of budget, and the two steps score identically.
        let outcome = allocate_ladder(&menus, 100).expect("allocates");
        // The lower PLAN index wins, which is the second menu.
        assert_eq!(outcome.chosen[1].encoding, "a_dear");
        assert_eq!(outcome.chosen[0].encoding, "cheap");

        // Reversing the menu order changes nothing.
        let reversed = vec![menus[1].clone(), menus[0].clone()];
        let again = allocate_ladder(&reversed, 100).expect("allocates");
        assert_eq!(again.chosen[0].encoding, "a_dear");
        assert_eq!(again.chosen[1].encoding, "cheap");
        assert_eq!(again.spent_bytes, outcome.spent_bytes);
        assert_eq!(again.chosen_damage, outcome.chosen_damage);
    }

    /// A tensor the cheapest option does no measured damage to is demoted,
    /// and the bytes that frees pay for another tensor's upgrade.
    #[test]
    fn a_cheap_option_with_no_damage_demotes_the_tensor() {
        let menus = vec![
            // Undamaged at the cheap rung: the base price buys nothing.
            menu(
                0,
                vec![option("cheap", 100, 0.0), option("base", 200, 0.0)],
                1,
            ),
            menu(
                1,
                vec![option("base", 200, 0.9), option("dear", 260, 0.1)],
                0,
            ),
        ];
        // A two-way allocation would have spent 400 bytes on the two base
        // options; this allocation may spend the same total.
        let outcome = allocate_ladder(&menus, 0).expect("allocates");
        assert!(outcome.chosen[0].is_demoted());
        assert_eq!(outcome.chosen[0].encoding, "cheap");
        assert_eq!(outcome.base_bytes, 400);
        assert_eq!(outcome.start_bytes, 300);

        // The 100 bytes the demotion freed buy the other tensor's upgrade,
        // and the artifact is no larger than the all-base one.
        let outcome = allocate_ladder(&menus, 100).expect("allocates");
        assert!(outcome.chosen[0].is_demoted());
        assert!(outcome.chosen[1].is_promoted());
        assert_eq!(outcome.chosen_bytes, 360);
        assert!(outcome.chosen_bytes < outcome.base_bytes);
        assert!(outcome.chosen_damage < outcome.base_damage);
    }

    /// A tensor that resolved to nothing is an allocation with no encoding
    /// for it, which fails rather than defaulting.
    #[test]
    fn an_empty_menu_is_refused() {
        let menus = vec![menu(0, Vec::new(), 0)];
        let error = allocate_ladder(&menus, 100)
            .expect_err("no option")
            .to_string();
        assert!(error.contains("no measured option"), "{error}");
    }
}
