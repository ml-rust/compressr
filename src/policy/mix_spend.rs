//! Spending a GGUF mix's byte budget: two ways, or over a ladder.
//!
//! [`super::mix_measured`] settles what every tensor costs and what the budget
//! is. This module spends it, one way or the other, and reports the same
//! counters whichever ran:
//!
//! - A file measured at ONE encoding drives the two-way promotion in
//!   [`super::mix_promote`], unchanged.
//! - A file measuring SEVERAL encodings per tensor drives the ladder in
//!   [`super::mix_ladder`], where a tensor can land below the base format as
//!   well as above it.

use anyhow::Result;

use crate::quantization::QuantType;

use super::gguf_mix::GgufMix;
use super::ladder_report::LadderReport;
use super::mix_bytes::{element_count, plan_bytes};
use super::mix_ladder::{allocate_mix_ladder, can_ladder};
use super::mix_measured::MeasuredTensor;
use super::mix_promote::{promote_in_order, PromotionFormats};
use super::mix_rules::plan_mixed_tensor;
use super::sensitivity::SensitivityMeasurement;
use super::sensitivity_rank::BumpCandidate;

/// What the allocation is spending, and what it is spending it against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpendInputs {
    /// The formats a two-way allocation moves a tensor between
    pub formats: PromotionFormats,
    /// Bytes an all-base allocation of this checkpoint would write
    pub floor_bytes: u64,
    /// Bytes the positional mix spends above that
    pub budget_bytes: u64,
}

/// What one spend did, whichever allocation made it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spend {
    /// Bytes spent above the all-base allocation
    pub spent_bytes: u64,
    /// Bytes the resulting allocation writes
    pub measured_bytes: u64,
    /// Candidates moved off the base format
    pub promoted: usize,
    /// Candidates llama.cpp's positional rules moved as well
    pub overlap: usize,
    /// Blocked candidates the budget reached
    pub wanted_blocked_tensors: usize,
    /// Parameters in [`Spend::wanted_blocked_tensors`]
    pub wanted_blocked_parameters: u64,
    /// Steps or candidates refused for a delta that is not positive
    pub excluded_nonpositive: usize,
    /// What the ladder did, when one ran
    pub ladder: Option<LadderReport>,
}

/// Spend the budget over `candidates`, writing the result into `assigned`.
///
/// # Errors
/// A candidate the measurement does not resolve, which either allocation
/// refuses rather than allocating from a partial match.
pub fn spend_budget(
    mix: GgufMix,
    tensors: &[MeasuredTensor<'_>],
    candidates: &[BumpCandidate<'_>],
    assigned: &mut [Option<QuantType>],
    promoted_by_heuristic: &[bool],
    inputs: SpendInputs,
    measurement: &SensitivityMeasurement,
) -> Result<Spend> {
    if !can_ladder(mix, measurement) {
        let promotion = promote_in_order(
            tensors,
            candidates,
            assigned,
            promoted_by_heuristic,
            inputs.formats,
            inputs.budget_bytes,
            measurement,
        )?;
        return Ok(Spend {
            spent_bytes: promotion.spent_bytes,
            measured_bytes: inputs.floor_bytes + promotion.spent_bytes,
            promoted: promotion.promoted,
            overlap: promotion.overlap,
            wanted_blocked_tensors: promotion.wanted_blocked_tensors,
            wanted_blocked_parameters: promotion.wanted_blocked_parameters,
            excluded_nonpositive: promotion.excluded_nonpositive,
            ladder: None,
        });
    }

    let indices: Vec<usize> = candidates.iter().map(|candidate| candidate.index).collect();
    let report = allocate_mix_ladder(
        mix,
        tensors,
        &indices,
        assigned,
        inputs.budget_bytes,
        measurement,
    )?;

    let mut measured_bytes = 0u64;
    for (index, (_, shape, dtype)) in tensors.iter().enumerate() {
        measured_bytes += plan_bytes(
            plan_mixed_tensor(shape, *dtype, assigned.get(index).copied().flatten()),
            element_count(shape),
            *dtype,
        );
    }
    // Only a candidate can be promoted: every other tensor holds whatever the
    // positional allocation left it at, which is no promotion of this one's.
    let mut promoted = 0usize;
    let mut overlap = 0usize;
    for index in &indices {
        if assigned
            .get(*index)
            .copied()
            .flatten()
            .is_some_and(|format| format != inputs.formats.base)
        {
            promoted += 1;
            if promoted_by_heuristic.get(*index).copied().unwrap_or(false) {
                overlap += 1;
            }
        }
    }

    Ok(Spend {
        spent_bytes: measured_bytes.saturating_sub(inputs.floor_bytes),
        measured_bytes,
        promoted,
        overlap,
        wanted_blocked_tensors: 0,
        wanted_blocked_parameters: 0,
        excluded_nonpositive: report.nonpositive_steps,
        ladder: Some(report),
    })
}
