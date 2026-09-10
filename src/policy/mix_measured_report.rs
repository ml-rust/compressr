//! What a measured GGUF mixed-precision allocation did, against the
//! positional one it replaced.
//!
//! The whole point of the measured path is a comparison, so every number a
//! comparison needs is here and is MEASURED from the bytes each allocation
//! would write: how many tensors each policy promoted, how many tensors both
//! promoted, what the measurement wanted to promote and GGUF's 256-element
//! super-block could not hold, and the bits per weight each policy achieved.
//!
//! The blocked counts are the payload. A measurement can rank a tensor first
//! and still be unable to place a single extra bit in it, because a K-quant
//! row is a whole number of 256-element super-blocks or it is nothing. That
//! number is the direct evidence for or against a finer-grained container.

use super::ladder_report::LadderReport;

/// A measured GGUF allocation, and the positional one it is matched against.
///
/// `Eq` is not derived: the ladder report it can carry holds measured damage,
/// which is a float.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredMixAllocation {
    /// The mix, as llama.cpp names it
    pub mix_name: String,
    /// The format every quantizable tensor takes unless the ranking raises it
    pub base_format: String,
    /// The format the ranking raises a tensor to
    pub promoted_format: String,
    /// The encoding the measurement was taken at, as the file spells it
    pub baseline_encoding: String,
    /// Tensors the mix quantizes, which is the set the ranking ranks
    pub candidates: usize,
    /// Candidates the measurement showed no damage on, so never promoted
    pub excluded_nonpositive: usize,
    /// Tensors llama.cpp's positional rules moved off the base format
    pub heuristic_promoted: usize,
    /// Tensors the measured ranking moved off the base format
    pub measured_promoted: usize,
    /// Tensors both policies promoted
    pub overlap: usize,
    /// Candidates whose row is not a whole number of super-blocks
    pub blocked_tensors: usize,
    /// Parameters in [`MeasuredMixAllocation::blocked_tensors`]
    pub blocked_parameters: u64,
    /// Blocked candidates the budget reached, so the ranking wanted them
    pub wanted_blocked_tensors: usize,
    /// Parameters in [`MeasuredMixAllocation::wanted_blocked_tensors`]
    pub wanted_blocked_parameters: u64,
    /// Bytes the positional mix spends above an all-base allocation
    pub budget_bytes: u64,
    /// Bytes the ranking actually spent of that budget
    pub spent_bytes: u64,
    /// Bytes an all-base allocation of this checkpoint would write
    pub floor_bytes: u64,
    /// Bytes the positional mix would write
    pub heuristic_bytes: u64,
    /// Bytes the measured allocation writes
    pub measured_bytes: u64,
    /// Elements over every tensor, quantized or not
    pub total_elements: u64,
    /// What the ladder allocation did, when one ran
    ///
    /// `Some` when the sensitivity file measured several encodings per tensor,
    /// which is what lets a tensor land BELOW the base format instead of only
    /// above it. `None` for a file measured at one encoding, which the two-way
    /// promotion above still allocates exactly as it always has.
    pub ladder: Option<LadderReport>,
}

impl MeasuredMixAllocation {
    /// Bits per weight the positional mix achieves, over every tensor.
    pub fn heuristic_bpw(&self) -> f64 {
        bits_per_weight(self.heuristic_bytes, self.total_elements)
    }

    /// Bits per weight the measured allocation achieves, over every tensor.
    pub fn measured_bpw(&self) -> f64 {
        bits_per_weight(self.measured_bytes, self.total_elements)
    }

    /// Budget bytes the ranking could not place.
    ///
    /// A ranking is spent in order, so the first tensor that does not fit
    /// ends the spend. What is left over is reported rather than filled by a
    /// lower-ranked tensor, which would no longer be the measured order.
    pub fn unspent_bytes(&self) -> u64 {
        self.budget_bytes.saturating_sub(self.spent_bytes)
    }
}

/// Bits per weight, or 0 rather than a NaN for an empty checkpoint.
fn bits_per_weight(bytes: u64, elements: u64) -> f64 {
    if elements == 0 {
        return 0.0;
    }
    bytes as f64 * 8.0 / elements as f64
}

impl std::fmt::Display for MeasuredMixAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Measured mixed precision {}: base {}, promoted {}, ranked by a task-metric delta \
             per element measured at {}",
            self.mix_name, self.base_format, self.promoted_format, self.baseline_encoding
        )?;
        writeln!(
            f,
            "  promoted {} of {} candidate tensor(s) by measurement, against {} by llama.cpp's \
             positional rules; {} tensor(s) in both sets",
            self.measured_promoted, self.candidates, self.heuristic_promoted, self.overlap
        )?;
        writeln!(
            f,
            "  {} candidate(s) excluded for a delta that is not positive",
            self.excluded_nonpositive
        )?;
        writeln!(
            f,
            "  {} candidate(s) holding {} parameter(s) cannot take {} at all: their row is not a \
             whole number of 256-element super-blocks. The ranking reached {} of them, holding \
             {} parameter(s), and could place no extra bit in any of them.",
            self.blocked_tensors,
            self.blocked_parameters,
            self.promoted_format,
            self.wanted_blocked_tensors,
            self.wanted_blocked_parameters
        )?;
        writeln!(
            f,
            "  byte budget {} (what the positional mix spends above an all-base allocation of \
             {} bytes), spent {}, unspent {}",
            self.budget_bytes,
            self.floor_bytes,
            self.spent_bytes,
            self.unspent_bytes()
        )?;
        writeln!(
            f,
            "  bpw over {} weights: measured {:.3} ({} bytes) against positional {:.3} ({} \
             bytes), weight payload only",
            self.total_elements,
            self.measured_bpw(),
            self.measured_bytes,
            self.heuristic_bpw(),
            self.heuristic_bytes
        )?;
        match &self.ladder {
            Some(ladder) => write!(f, "{ladder}"),
            None => write!(
                f,
                "  the sensitivity file measures one encoding, so this is the two-way \
                 allocation: every candidate at the base format, a ranked subset promoted, and \
                 none demoted"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocation() -> MeasuredMixAllocation {
        MeasuredMixAllocation {
            mix_name: "Q4_K_M".into(),
            base_format: "Q4_K".into(),
            promoted_format: "Q6_K".into(),
            baseline_encoding: "Q4AS32D_T64".into(),
            candidates: 4,
            excluded_nonpositive: 1,
            heuristic_promoted: 2,
            measured_promoted: 2,
            overlap: 1,
            blocked_tensors: 1,
            blocked_parameters: 96,
            wanted_blocked_tensors: 1,
            wanted_blocked_parameters: 96,
            budget_bytes: 132,
            spent_bytes: 66,
            floor_bytes: 576,
            heuristic_bytes: 708,
            measured_bytes: 642,
            total_elements: 1024,
            ladder: None,
        }
    }

    /// The bpw is measured from the bytes, and the leftover budget is stated
    /// rather than filled by a lower-ranked tensor.
    #[test]
    fn the_report_states_both_allocations() {
        let allocation = allocation();
        assert_eq!(allocation.unspent_bytes(), 66);
        assert!((allocation.measured_bpw() - 642.0 * 8.0 / 1024.0).abs() < 1e-9);
        assert!((allocation.heuristic_bpw() - 708.0 * 8.0 / 1024.0).abs() < 1e-9);

        let text = allocation.to_string();
        assert!(text.contains("256-element super-blocks"), "{text}");
        assert!(text.contains("1 tensor(s) in both sets"), "{text}");
        // A two-way allocation says so, rather than leaving a reader to infer
        // that nothing was demoted.
        assert!(text.contains("none demoted"), "{text}");
    }

    #[test]
    fn an_empty_checkpoint_has_no_bpw_rather_than_a_nan() {
        let allocation = MeasuredMixAllocation {
            total_elements: 0,
            ..allocation()
        };
        assert_eq!(allocation.measured_bpw(), 0.0);
        assert_eq!(allocation.heuristic_bpw(), 0.0);
    }
}
