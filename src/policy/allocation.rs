//! Which tensors a TCF profile raises to its bump encoding.
//!
//! # The screen compressr can honestly run
//!
//! `PROFILES.md` Section 4 screens tensors (step 1), measures a task-metric
//! delta per candidate (step 2), then allocates against the profile's byte
//! budget (step 3). Step 2 needs a model and a 128-prompt calibration set.
//! compressr has neither, so it never runs, and no value here is ever written
//! to `sensitivity_delta`, `sensitivity_ci95`, `sensitivity_samples` or
//! `calibration_id`.
//!
//! What is left is a weight-only screen: quantize a tensor at the base
//! encoding, dequantize it again, and take the RMS of the error over the RMS
//! of the source. That number ranks tensors by how badly the base encoding
//! reconstructs their weights. It is not a task delta and never stands in for
//! one.
//!
//! # The budget is counted in elements
//!
//! `PROFILES.md` Section 2 spends "top N%" by parameter count, and the profile
//! rows are bpw numbers, so the budget that matters is bytes. Elements track
//! bytes; tensor count does not — 15% of tensors can be 2% or 60% of a model's
//! parameters, and either way the achieved bpw would be unrelated to the rung.
//! So the bump set is chosen by element count.
//!
//! This module ranks and selects. It holds no kernel: the quantize and the
//! dequantize both come from `tcf-core`.

use anyhow::{bail, Result};

/// One quantizable tensor's screen result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenedTensor {
    /// The tensor's position in the conversion's plan list
    pub index: usize,
    /// Relative RMS reconstruction error at the base encoding
    pub relative_rms: f64,
    /// Elements in the tensor, which is what the bump budget spends
    pub elements: u64,
}

/// Relative RMS reconstruction error: RMS of the error over RMS of the source.
///
/// The result is dimensionless, so tensors of different scale compare
/// directly. A source of all zeros reconstructs exactly and scores 0.0 rather
/// than dividing by zero.
///
/// # Errors
/// The two slices differ in length, which means the caller dequantized a
/// different element space than it quantized.
pub fn relative_rms_error(source: &[f32], reconstructed: &[f32]) -> Result<f64> {
    if source.len() != reconstructed.len() {
        bail!(
            "Screen compared {} source values against {} reconstructed values",
            source.len(),
            reconstructed.len()
        );
    }
    let mut error_energy = 0.0f64;
    let mut source_energy = 0.0f64;
    for (value, reconstruction) in source.iter().zip(reconstructed) {
        let value = f64::from(*value);
        let difference = value - f64::from(*reconstruction);
        error_energy += difference * difference;
        source_energy += value * value;
    }
    if source_energy == 0.0 {
        return Ok(0.0);
    }
    Ok((error_energy / source_energy).sqrt())
}

/// The tensors a profile raises to its bump encoding, as plan indices.
///
/// `candidates` holds the quantizable tensors only. A tensor stored raw — rank
/// below 2, a row that is not a whole number of tiles, a non-finite value — has
/// no base encoding to be bumped from, so it is absent from the ranking and
/// from the element total the fraction is taken of.
///
/// Selection walks the candidates by descending relative RMS, breaking ties by
/// plan index, and takes each one whose predecessors have not already filled
/// the budget. The tensor that crosses the budget is taken, so the set
/// overshoots by at most one tensor rather than rounding down to nothing when
/// one tensor is larger than the whole budget.
pub fn select_bumped(candidates: &[ScreenedTensor], fraction: f64) -> Vec<usize> {
    if fraction <= 0.0 || candidates.is_empty() {
        return Vec::new();
    }
    let total: u64 = candidates.iter().map(|c| c.elements).sum();
    let budget = total as f64 * fraction;

    let mut ranked: Vec<&ScreenedTensor> = candidates.iter().collect();
    ranked.sort_by(|a, b| {
        b.relative_rms
            .total_cmp(&a.relative_rms)
            .then(a.index.cmp(&b.index))
    });

    let mut spent = 0.0f64;
    let mut bumped = Vec::new();
    for candidate in ranked {
        if spent >= budget {
            break;
        }
        bumped.push(candidate.index);
        spent += candidate.elements as f64;
    }
    bumped.sort_unstable();
    bumped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screened(index: usize, relative_rms: f64, elements: u64) -> ScreenedTensor {
        ScreenedTensor {
            index,
            relative_rms,
            elements,
        }
    }

    #[test]
    fn an_exact_reconstruction_scores_zero() {
        let values = [1.0f32, -2.0, 3.5];
        assert_eq!(relative_rms_error(&values, &values).unwrap(), 0.0);
    }

    /// A source of all zeros has no energy to divide by, and reconstructs
    /// exactly, so it scores 0.0 instead of a NaN.
    #[test]
    fn an_all_zero_source_scores_zero() {
        assert_eq!(relative_rms_error(&[0.0, 0.0], &[0.0, 0.0]).unwrap(), 0.0);
    }

    #[test]
    fn the_error_is_relative_to_the_source_energy() {
        // Error energy 4 over source energy 16: RMS ratio 0.5.
        let error = relative_rms_error(&[2.0, 2.0, 2.0, 2.0], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!((error - 0.5).abs() < 1e-12, "{error}");
    }

    #[test]
    fn a_length_mismatch_is_an_error() {
        assert!(relative_rms_error(&[1.0, 2.0], &[1.0]).is_err());
    }

    /// Ten equal candidates and a 0.10 fraction spend one tensor's worth.
    #[test]
    fn the_bump_set_is_the_expected_size() {
        let candidates: Vec<ScreenedTensor> =
            (0..10).map(|i| screened(i, i as f64 * 0.01, 100)).collect();
        assert_eq!(select_bumped(&candidates, 0.10), vec![9]);
        // 0.15 of 1000 elements is 150: the second tensor is taken because
        // the first alone had not filled the budget.
        assert_eq!(select_bumped(&candidates, 0.15), vec![8, 9]);
    }

    /// The ranking is by error, not by position or by size.
    #[test]
    fn the_worst_reconstructed_tensors_are_bumped_first() {
        let candidates = vec![
            screened(0, 0.9, 64),
            screened(1, 0.1, 64),
            screened(2, 0.5, 64),
        ];
        assert_eq!(select_bumped(&candidates, 0.34), vec![0, 2]);
    }

    /// A flat profile bumps nothing.
    #[test]
    fn a_zero_fraction_bumps_nothing() {
        let candidates = vec![screened(0, 0.9, 64)];
        assert!(select_bumped(&candidates, 0.0).is_empty());
    }

    /// One tensor larger than the whole budget is still bumped: rounding it
    /// down would leave a profile with a bump encoding it never uses.
    #[test]
    fn a_tensor_larger_than_the_budget_is_still_taken() {
        let candidates = vec![screened(0, 0.9, 4096), screened(1, 0.8, 64)];
        assert_eq!(select_bumped(&candidates, 0.10), vec![0]);
    }

    /// A raw tensor is not a candidate, so it neither ranks nor enlarges the
    /// budget. The caller excludes it; this fixes what that exclusion buys.
    #[test]
    fn only_candidates_set_the_budget() {
        let candidates = vec![screened(0, 0.9, 64), screened(1, 0.8, 64)];
        // 0.10 of 128 elements is 12.8, which the first tensor fills.
        assert_eq!(select_bumped(&candidates, 0.10), vec![0]);
        // Had a 4096-element raw tensor counted, the budget would be 422.4
        // and every tensor would fit inside it.
        let with_raw = vec![
            screened(0, 0.9, 64),
            screened(1, 0.8, 64),
            screened(2, 0.7, 4096),
        ];
        assert_eq!(select_bumped(&with_raw, 0.10), vec![0, 1, 2]);
    }
}
