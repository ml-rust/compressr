//! The bump set a measured sensitivity file selects.
//!
//! # Why the match is strict
//!
//! The measurement's names come from the model's parameter traversal. The
//! plan's names come from the source checkpoint. Those are two namespaces, and
//! nothing guarantees they agree. A candidate with no record is therefore an
//! error, never a fall back to the weight-error screen and never a zero: a set
//! ranked half by task delta and half by weight error is a ranking of nothing,
//! and a missing entry read as zero sensitivity would push an unmeasured
//! tensor to the bottom of the order on no evidence at all. Refusing the run
//! is the honest outcome, and the error names both namespaces so the mismatch
//! is visible.
//!
//! # Why the rank is delta per element
//!
//! The ideal rank is delta per unit of effective cost — bytes plus weighted
//! decode instructions. compressr has no decode-instruction model for a
//! consumer's backend, so the instruction term is not approximated; it is
//! left out, and the ranking says so. The remaining cost is the budget
//! compressr actually spends, which is elements: a mix's rungs are bpw
//! numbers and only elements track bytes. So the order is `delta / elements`,
//! descending, ties broken by plan index.
//!
//! A tensor whose delta is not positive was not damaged by the base encoding —
//! quantizing it changed nothing measurable, or the metric improved. Bumping
//! it buys nothing and costs budget, so it is never bumped, even when budget
//! remains.

use anyhow::{bail, Result};

use super::name_examples::examples;
use super::SensitivityMeasurement;

/// One bump candidate offered to the measured ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BumpCandidate<'a> {
    /// The tensor's position in the conversion's plan list
    pub index: usize,
    /// The tensor's name, as the source checkpoint spells it
    pub name: &'a str,
    /// Elements in the tensor, which is what the bump budget spends
    pub elements: u64,
}

/// One candidate's place in the measured order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedCandidate {
    /// The candidate's position in the caller's plan list
    pub index: usize,
    /// Elements in the tensor
    pub elements: u64,
    /// The measured task delta, always positive
    pub delta: f64,
}

/// The measured order over one candidate set.
///
/// The order is the whole ranking, not a selection: what a budget can afford
/// depends on what the budget is spent in, and that differs between a TCF
/// profile (elements) and a GGUF mix (bytes). Both walk this same order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeasuredRanking {
    /// Candidates with a positive delta, most damaged per element first
    pub order: Vec<RankedCandidate>,
    /// Candidates the measurement showed no damage on, so never promoted
    pub excluded_nonpositive: usize,
}

/// What the measured ranking selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredBump {
    /// Plan indices raised to the bump encoding, ascending
    pub bumped: Vec<usize>,
    /// Candidates the measurement showed no damage on, so never bumped
    pub excluded_nonpositive: usize,
}

/// Rank the candidates by measured damage per element and spend the budget.
///
/// The budget is `fraction` of every candidate's elements, including the ones
/// the measurement excludes, which is how [`super::allocation::select_bumped`]
/// counts it. The tensor that crosses the budget is taken, for the same
/// reason: a single tensor larger than the whole budget must still be bumped
/// rather than rounded away.
///
/// # Errors
/// Any candidate has no record in the measurement.
pub fn select_bumped_measured(
    measurement: &SensitivityMeasurement,
    candidates: &[BumpCandidate<'_>],
    fraction: f64,
) -> Result<MeasuredBump> {
    let ranking = rank_candidates(measurement, candidates)?;
    if fraction <= 0.0 || candidates.is_empty() {
        return Ok(MeasuredBump {
            bumped: Vec::new(),
            excluded_nonpositive: ranking.excluded_nonpositive,
        });
    }

    let total: u64 = candidates.iter().map(|candidate| candidate.elements).sum();
    let budget = total as f64 * fraction;

    let mut spent = 0.0f64;
    let mut bumped = Vec::new();
    for ranked in ranking.order {
        if spent >= budget {
            break;
        }
        bumped.push(ranked.index);
        spent += ranked.elements as f64;
    }
    bumped.sort_unstable();
    Ok(MeasuredBump {
        bumped,
        excluded_nonpositive: ranking.excluded_nonpositive,
    })
}

/// Put the candidates in measured order, most damaged per element first.
///
/// This is the ranking itself, with no budget: the caller decides what the
/// budget is and what it is spent in. A candidate with a delta that is not
/// positive is left out entirely rather than placed at the bottom, for the
/// reason this module's header gives.
///
/// # Errors
/// Any candidate has no record in the measurement.
pub fn rank_candidates(
    measurement: &SensitivityMeasurement,
    candidates: &[BumpCandidate<'_>],
) -> Result<MeasuredRanking> {
    let mut order: Vec<RankedCandidate> = Vec::with_capacity(candidates.len());
    let mut unmatched: Vec<&str> = Vec::new();
    let mut excluded_nonpositive = 0usize;
    for candidate in candidates {
        let Some(delta) = measurement.delta_for(candidate.name) else {
            unmatched.push(candidate.name);
            continue;
        };
        if delta > 0.0 {
            order.push(RankedCandidate {
                index: candidate.index,
                elements: candidate.elements,
                delta,
            });
        } else {
            excluded_nonpositive += 1;
        }
    }
    if !unmatched.is_empty() {
        bail!(
            "{}",
            unmatched_error(measurement, &unmatched, candidates.len())
        );
    }
    order.sort_by(|left, right| {
        per_element(right.delta, right.elements)
            .total_cmp(&per_element(left.delta, left.elements))
            .then(left.index.cmp(&right.index))
    });
    Ok(MeasuredRanking {
        order,
        excluded_nonpositive,
    })
}

/// Measured damage per unit of the budget this tensor spends.
///
/// A planned tensor never holds zero elements; the floor keeps the division
/// defined rather than resting on that.
fn per_element(delta: f64, elements: u64) -> f64 {
    delta / elements.max(1) as f64
}

/// The message for candidates the measurement does not name.
fn unmatched_error(
    measurement: &SensitivityMeasurement,
    unmatched: &[&str],
    candidates: usize,
) -> String {
    format!(
        "{} of {candidates} bump candidates have no record in the sensitivity file. The \
         measurement's names come from the model's parameter traversal and the plan's names come \
         from the source checkpoint; if those are different namespaces the whole ranking is \
         wrong, so no candidate is ranked from a partial match and none falls back to the \
         weight-error screen. Unmatched plan names: {}. Names the file carries: {}.",
        unmatched.len(),
        examples(unmatched),
        examples(&measurement.measured_names())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::parse_sensitivity;

    /// A measurement over four tensors, with one tensor the base encoding did
    /// not damage.
    fn measurement() -> SensitivityMeasurement {
        let mut lines = vec![
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4_K","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#
                .to_string(),
        ];
        for (name, delta) in [
            ("small.weight", 0.3f64),
            ("large.weight", 1.0),
            ("mid.weight", 0.4),
            ("helped.weight", -0.5),
        ] {
            lines.push(format!(
                r#"{{"record":"sensitivity","name":"{name}","delta":{delta}}}"#
            ));
        }
        parse_sensitivity(&lines.join("\n")).expect("parses")
    }

    fn candidate(index: usize, name: &str, elements: u64) -> BumpCandidate<'_> {
        BumpCandidate {
            index,
            name,
            elements,
        }
    }

    /// `small.weight` has the smallest delta and the fewest elements, which
    /// buys it the best damage-per-element and the first place in the order.
    /// Ranking by delta alone would have taken `large.weight` first.
    #[test]
    fn the_order_is_damage_per_element() {
        let candidates = [
            candidate(0, "large.weight", 1000),
            candidate(1, "mid.weight", 200),
            candidate(2, "small.weight", 100),
        ];
        // 0.05 of 1300 elements is 65, which `small.weight` alone fills.
        let selected = select_bumped_measured(&measurement(), &candidates, 0.05).expect("ranks");
        assert_eq!(selected.bumped, vec![2]);
        // 0.10 of 1300 is 130: `small.weight` then `mid.weight`, both ahead
        // of the tensor with the largest raw delta.
        let selected = select_bumped_measured(&measurement(), &candidates, 0.10).expect("ranks");
        assert_eq!(selected.bumped, vec![1, 2]);
    }

    /// Equal damage per element falls back to plan index, as the screen does.
    #[test]
    fn a_tie_is_broken_by_plan_index() {
        let candidates = [
            candidate(3, "mid.weight", 400),
            candidate(1, "small.weight", 300),
        ];
        // Both score 0.001 per element; the lower plan index goes first, and
        // 0.40 of 700 elements is 280, which one tensor fills.
        let selected = select_bumped_measured(&measurement(), &candidates, 0.40).expect("ranks");
        assert_eq!(selected.bumped, vec![1]);
    }

    /// A tensor the base encoding did not damage is never bumped, even with
    /// the whole budget free.
    #[test]
    fn a_non_positive_delta_is_never_bumped() {
        let candidates = [
            candidate(0, "helped.weight", 100),
            candidate(1, "small.weight", 100),
        ];
        let selected = select_bumped_measured(&measurement(), &candidates, 1.0).expect("ranks");
        assert_eq!(selected.bumped, vec![1]);
        assert_eq!(selected.excluded_nonpositive, 1);
    }

    /// Every candidate must resolve, and the error shows both namespaces.
    #[test]
    fn an_unmatched_candidate_is_refused() {
        let candidates = [
            candidate(0, "small.weight", 100),
            candidate(1, "model.layers.0.attn.q_proj.weight", 100),
        ];
        let error = select_bumped_measured(&measurement(), &candidates, 0.50)
            .expect_err("an unmatched candidate")
            .to_string();
        assert!(error.contains("1 of 2 bump candidates"), "{error}");
        assert!(
            error.contains("'model.layers.0.attn.q_proj.weight'"),
            "{error}"
        );
        assert!(error.contains("'small.weight'"), "{error}");
        assert!(error.contains("no candidate is ranked"), "{error}");
    }

    /// A flat profile bumps nothing, and still reports what it excluded.
    #[test]
    fn a_zero_fraction_bumps_nothing() {
        let candidates = [candidate(0, "small.weight", 100)];
        let selected = select_bumped_measured(&measurement(), &candidates, 0.0).expect("ranks");
        assert!(selected.bumped.is_empty());
    }

    /// One tensor larger than the whole budget is still bumped, for the same
    /// reason the screen takes it: rounding it down leaves a profile with a
    /// bump encoding it never uses.
    #[test]
    fn a_tensor_larger_than_the_budget_is_still_taken() {
        let candidates = [
            candidate(0, "large.weight", 4096),
            candidate(1, "helped.weight", 64),
        ];
        let selected = select_bumped_measured(&measurement(), &candidates, 0.01).expect("ranks");
        assert_eq!(selected.bumped, vec![0]);
    }
}
