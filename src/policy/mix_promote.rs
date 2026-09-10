//! The two-way GGUF promotion: base format for every candidate, a ranked
//! subset raised to the promoted one.
//!
//! This is the allocation a sensitivity file measured at ONE encoding drives,
//! and it is unchanged by the ladder that sits beside it in
//! [`super::mix_ladder`]. It is here rather than inline in
//! [`super::mix_measured`] so that adding the ladder moved none of it.
//!
//! The budget is spent IN ORDER, and the first tensor that does not fit ends
//! the spend. What is left over is reported rather than filled by a
//! lower-ranked tensor: filling it would no longer be the measured order.

use anyhow::Result;

use crate::quantization::QuantType;

use super::mix_bytes::{holds_format, promotion_cost};
use super::mix_measured::MeasuredTensor;
use super::mix_rules::step_down_unquantizable;
use super::sensitivity::SensitivityMeasurement;
use super::sensitivity_rank::{rank_candidates, BumpCandidate};

/// The two formats a two-way allocation moves a tensor between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotionFormats {
    /// The format every candidate takes unless the ranking raises it
    pub base: QuantType,
    /// The format the ranking raises a candidate to
    pub promoted: QuantType,
}

/// What the two-way promotion did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Promotion {
    /// Bytes spent above the all-base allocation
    pub spent_bytes: u64,
    /// Candidates raised to the promoted format
    pub promoted: usize,
    /// Candidates llama.cpp's positional rules raised as well
    pub overlap: usize,
    /// Blocked candidates the budget reached, so the ranking wanted them
    pub wanted_blocked_tensors: usize,
    /// Parameters in [`Promotion::wanted_blocked_tensors`]
    pub wanted_blocked_parameters: u64,
    /// Candidates the measurement showed no damage on, so never promoted
    pub excluded_nonpositive: usize,
}

/// Spend the byte budget on the measured order, most damaged per element
/// first.
///
/// `assigned` arrives as the all-base allocation and leaves carrying the
/// promotions.
///
/// # Errors
/// Any candidate has no record in the measurement.
pub fn promote_in_order(
    tensors: &[MeasuredTensor<'_>],
    candidates: &[BumpCandidate<'_>],
    assigned: &mut [Option<QuantType>],
    promoted_by_heuristic: &[bool],
    formats: PromotionFormats,
    budget_bytes: u64,
    measurement: &SensitivityMeasurement,
) -> Result<Promotion> {
    let ranking = rank_candidates(measurement, candidates)?;
    let mut promotion = Promotion {
        excluded_nonpositive: ranking.excluded_nonpositive,
        ..Promotion::default()
    };
    if budget_bytes == 0 {
        return Ok(promotion);
    }

    for ranked in &ranking.order {
        let Some((_, shape, dtype)) = tensors.get(ranked.index).copied() else {
            continue;
        };
        if !holds_format(shape, formats.promoted) {
            promotion.wanted_blocked_tensors += 1;
            promotion.wanted_blocked_parameters += ranked.elements;
            continue;
        }
        let cost = promotion_cost(
            shape,
            dtype,
            ranked.elements,
            formats.base,
            formats.promoted,
        );
        if promotion.spent_bytes + cost > budget_bytes {
            break;
        }
        if let Some(slot) = assigned.get_mut(ranked.index) {
            *slot = Some(step_down_unquantizable(formats.promoted, shape));
        }
        promotion.spent_bytes += cost;
        promotion.promoted += 1;
        if promoted_by_heuristic
            .get(ranked.index)
            .copied()
            .unwrap_or(false)
        {
            promotion.overlap += 1;
        }
    }
    Ok(promotion)
}
