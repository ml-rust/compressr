//! What a measured encoding becomes in GGUF, and what a mix says when a
//! candidate resolves to nothing.
//!
//! A sensitivity file names TCF encodings and GGUF writes GGML blocks, so
//! something has to carry between them. That something is the bit width — the
//! same property [`super::mix_measured_encoding`] checks a file's baseline
//! against — and this module is where the map and the refusal live.

use crate::formats::parse_tcf_encoding;
use crate::quantization::QuantType;

use super::mix_report::quant_type_label;
use super::name_examples::examples;
use super::sensitivity::SensitivityMeasurement;

/// The block format compressr writes at one measured encoding's bit width.
///
/// `None` when the encoding is not one compressr recognizes, or when nothing
/// it can write sits at that width. A roadmap format is not an option: a rung
/// no writer exists for cannot be allocated to.
pub(super) fn ladder_format(encoding: &str) -> Option<QuantType> {
    let bits = parse_tcf_encoding(encoding).ok()?.geometry().bits;
    match bits {
        4 => Some(QuantType::Q4K),
        5 => Some(QuantType::Q5K),
        6 => Some(QuantType::Q6K),
        8 => Some(QuantType::Q8_0),
        _ => None,
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

    /// Each measured width becomes the block format compressr writes at it,
    /// and a name compressr does not know becomes nothing.
    #[test]
    fn a_width_becomes_the_format_written_at_it() {
        assert_eq!(ladder_format("Q4AS32D_T64"), Some(QuantType::Q4K));
        assert_eq!(ladder_format("Q4S32_T64"), Some(QuantType::Q4K));
        assert_eq!(ladder_format("Q6S16D_T64"), Some(QuantType::Q6K));
        assert_eq!(ladder_format("Q8S32_T64"), Some(QuantType::Q8_0));
        assert_eq!(ladder_format("Q3S16_T32"), None);
    }
}
