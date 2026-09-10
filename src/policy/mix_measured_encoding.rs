//! Whether a measurement is the right input for ranking a GGUF mix.
//!
//! The TCF path checks the measurement was taken at exactly the encoding the
//! profile bumps from — see
//! [`super::sensitivity::SensitivityMeasurement::check_encoding`].
//! That check cannot apply here: the sensitivity file's baseline encoding is a
//! TCF encoding name, and a TCF encoding name never equals a GGML format name,
//! so equality would refuse every file. The property the equality check
//! protects is what carries over, and that property is the bit width.

use anyhow::{bail, Context, Result};

use crate::formats::parse_tcf_encoding;

use super::gguf_mix::GgufMix;
use super::sensitivity::SensitivityMeasurement;

/// Check a measurement is the right input for ranking this mix.
///
/// The measurement's baseline encoding is a TCF encoding name, so it can
/// never equal a GGML format name and an equality check is meaningless here.
/// What actually has to hold is the property the equality check protects on
/// the TCF path: a delta is the damage ONE encoding did, and an encoding that
/// damages almost nothing produces an order that is mostly noise. So the rule
/// is the bit width.
///
/// - A measurement taken at MORE bits than the base format is refused. At
///   8 bits almost nothing is damaged, and ranking a 4-bit allocation by that
///   order is ranking by noise.
/// - A measurement taken at one bit FEWER is accepted. It overstates the
///   damage uniformly and preserves the order, which is the conservative
///   direction; it is also the only way to rank a `q5_k_*` mix, since TCF
///   defines no 5-bit encoding.
/// - Anything further below is refused: two bits of difference is a different
///   experiment. TCF defines no encoding below four bits, so no file reaches
///   that branch today; it is a guard, not a path.
///
/// # Errors
/// The baseline names an encoding compressr does not recognize, or one whose
/// bit width fails the rule above.
pub fn check_measurement_encoding(
    mix: GgufMix,
    measurement: &SensitivityMeasurement,
) -> Result<()> {
    let named = &measurement.baseline().encoding;
    let measured = parse_tcf_encoding(named).with_context(|| {
        format!(
            "Sensitivity file's baseline names encoding '{named}', which compressr does not \
             recognize"
        )
    })?;
    let measured_bits = measured.geometry().bits;
    let base_bits = base_bits(mix);
    if measured_bits > base_bits {
        bail!(
            "Sensitivity file was measured at {measured_bits} bits per weight, and mix {} \
             quantizes at {base_bits}. A delta is the damage one encoding did: measured above \
             the format being allocated, almost nothing is damaged and the order is noise. \
             Re-run the measurement at {base_bits} bits.",
            mix.name()
        );
    }
    if base_bits - measured_bits > 1 {
        bail!(
            "Sensitivity file was measured at {measured_bits} bits per weight, and mix {} \
             quantizes at {base_bits}. One bit below is accepted, because it overstates the \
             damage uniformly and keeps the order; {} bits below is a different experiment. \
             Re-run the measurement at {base_bits} bits.",
            mix.name(),
            base_bits - measured_bits
        );
    }
    Ok(())
}

/// Bits per weight the mix's base format quantizes at.
fn base_bits(mix: GgufMix) -> u8 {
    match mix {
        GgufMix::Q4KS | GgufMix::Q4KM => 4,
        GgufMix::Q5KS | GgufMix::Q5KM => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::parse_sensitivity;

    /// A one-tensor file measured at the named encoding.
    fn measured_at(encoding: &str) -> SensitivityMeasurement {
        let lines = [
            format!(
                r#"{{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"{encoding}","eval_rows":12,"eval_seed":7,"baseline_total":1.17}}"#
            ),
            r#"{"record":"sensitivity","name":"a.weight","delta":0.5}"#.to_string(),
        ];
        parse_sensitivity(&lines.join("\n")).expect("parses")
    }

    /// A 4-bit measurement is the right input for a 4-bit base mix, and for a
    /// 5-bit one, which is the only measurement a `q5_k_*` mix can have.
    #[test]
    fn a_measurement_at_or_one_bit_below_the_base_is_accepted() {
        assert!(check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q4AS32D_T64")).is_ok());
        assert!(check_measurement_encoding(GgufMix::Q5KM, &measured_at("Q4AS32D_T64")).is_ok());
    }

    /// Measured above the format being allocated, almost nothing is damaged
    /// and the order is noise.
    #[test]
    fn a_measurement_above_the_base_is_refused() {
        let error = check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q8S32T64"))
            .expect_err("8 bits over a 4-bit base")
            .to_string();
        assert!(error.contains("measured at 8 bits"), "{error}");
        assert!(error.contains("Q4_K_M"), "{error}");
    }

    /// Six bits over a four-bit base is refused for the same reason eight is.
    #[test]
    fn a_six_bit_measurement_is_refused_for_a_four_bit_base() {
        let error = check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q6S32T64"))
            .expect_err("6 bits over a 4-bit base")
            .to_string();
        assert!(error.contains("measured at 6 bits"), "{error}");
        assert!(error.contains("the order is noise"), "{error}");
    }

    #[test]
    fn an_unrecognized_encoding_is_refused() {
        let error = check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q3S16T32"))
            .expect_err("not a TCF encoding")
            .to_string();
        assert!(error.contains("does not recognize"), "{error}");
    }
}
