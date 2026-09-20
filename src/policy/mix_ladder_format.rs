//! Which block format a measured encoding names, and what a mix says when a
//! candidate resolves to nothing.
//!
//! A sensitivity file names the GGML block formats boostr's sweep measured
//! at, spelled as [`quant_type_label`] spells them, so a measured encoding IS
//! the format the mix writes. A name outside the writable block formats is
//! not an option at all.

use crate::quantization::QuantType;

use super::mix_report::quant_type_label;
use super::name_examples::examples;
use super::sensitivity::SensitivityMeasurement;
use super::sensitivity_options::normalized_encoding_name;

/// The block formats a measurement can name: every one compressr has a
/// writer for. A roadmap format is not an option: a rung no writer exists
/// for cannot be allocated to.
const MEASURABLE: [QuantType; 10] = [
    QuantType::Q4_0,
    QuantType::Q4_1,
    QuantType::Q8_0,
    QuantType::Q2K,
    QuantType::Q3K,
    QuantType::Q4K,
    QuantType::Q5K,
    QuantType::Q6K,
    QuantType::IQ4NL,
    QuantType::IQ4XS,
];

/// The block format one measured encoding names, or `None` when the name is
/// not a format compressr writes. Case, underscores and hyphens are ignored,
/// so `Q4_K` and `q4k` are the same rung.
pub(super) fn ladder_format(encoding: &str) -> Option<QuantType> {
    let key = normalized_encoding_name(encoding);
    MEASURABLE
        .into_iter()
        .find(|format| normalized_encoding_name(quant_type_label(*format)) == key)
}

/// Nominal bits per weight of a block format's codes, the width a
/// measurement is compared against a mix's base at.
pub(super) fn nominal_bits(format: QuantType) -> u8 {
    match format {
        QuantType::Q2K => 2,
        QuantType::Q3K => 3,
        QuantType::Q4_0
        | QuantType::Q4_1
        | QuantType::Q4K
        | QuantType::IQ4NL
        | QuantType::IQ4XS => 4,
        QuantType::Q5_0 | QuantType::Q5_1 | QuantType::Q5K => 5,
        QuantType::Q6K => 6,
        QuantType::Q8_0 | QuantType::Q8_1 => 8,
        // A trit occupies a 2-bit code slot.
        QuantType::PQ2_0 | QuantType::PTQ1_0 => 2,
        QuantType::NF4 | QuantType::FP4 => 4,
        QuantType::FP8 => 8,
        QuantType::GPTQ | QuantType::AWQ => 4,
    }
}

/// The message for candidates the ladder could not resolve.
pub(super) fn unresolved_error(
    measurement: &SensitivityMeasurement,
    unmatched: &[&str],
    baseless: &[&str],
    candidates: usize,
    base: QuantType,
) -> String {
    format!(
        "{} of {candidates} ladder candidates have no record in the sensitivity file, and {} \
         more have no usable option at the base format {}'s bit width — no record at that \
         width, or a row that cannot hold it. Every candidate is allocated \
         from measured options or none is: a set chosen half by measurement and half by a \
         positional rule is an allocation of nothing, and a missing option read as no damage \
         would demote an unmeasured tensor on no evidence at all. Unmatched plan names: {}. \
         Candidates with no base option: {}. Names the file carries: {}. Encodings it \
         measures: {}.",
        unmatched.len(),
        baseless.len(),
        quant_type_label(base),
        examples(unmatched),
        examples(baseless),
        examples(&measurement.measured_names()),
        examples(&measurement.encodings()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A measured name is the block format written at it, in any spelling,
    /// and a name compressr does not write becomes nothing.
    #[test]
    fn a_measured_name_is_the_format_written_at_it() {
        assert_eq!(ladder_format("Q4_K"), Some(QuantType::Q4K));
        assert_eq!(ladder_format("q4k"), Some(QuantType::Q4K));
        assert_eq!(ladder_format("Q6_K"), Some(QuantType::Q6K));
        assert_eq!(ladder_format("Q8_0"), Some(QuantType::Q8_0));
        assert_eq!(ladder_format("Q2_K"), Some(QuantType::Q2K));
        assert_eq!(ladder_format("Q5_0"), None);
        assert_eq!(ladder_format("NF4"), None);
    }

    #[test]
    fn nominal_bits_follow_the_code_width() {
        assert_eq!(nominal_bits(QuantType::Q4K), 4);
        assert_eq!(nominal_bits(QuantType::Q5K), 5);
        assert_eq!(nominal_bits(QuantType::Q6K), 6);
        assert_eq!(nominal_bits(QuantType::Q8_0), 8);
    }
}
