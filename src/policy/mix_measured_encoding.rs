//! Whether a measurement is the right input for ranking a GGUF mix.
//!
//! A delta is the damage ONE format did, and a format that damages almost
//! nothing produces an order that is mostly noise. The rule that protects the
//! ranking is the bit width of the format the file was measured at against
//! the mix's base.

use anyhow::{bail, Context, Result};

use super::gguf_mix::GgufMix;
use super::mix_ladder_format::{ladder_format, nominal_bits};
use super::sensitivity::SensitivityMeasurement;

/// Check a measurement is the right input for ranking this mix.
///
/// - A measurement taken at MORE bits than the base format is refused. At
///   8 bits almost nothing is damaged, and ranking a 4-bit allocation by that
///   order is ranking by noise.
/// - A measurement taken at one bit FEWER is accepted. It overstates the
///   damage uniformly and preserves the order, which is the conservative
///   direction.
/// - Anything further below is refused: two bits of difference is a different
///   experiment.
///
/// # Errors
/// The baseline names a format compressr does not write, or one whose bit
/// width fails the rule above.
pub fn check_measurement_encoding(
    mix: GgufMix,
    measurement: &SensitivityMeasurement,
) -> Result<()> {
    let named = &measurement.baseline().encoding;
    let measured = ladder_format(named).with_context(|| {
        format!(
            "Sensitivity file's baseline names encoding '{named}', which compressr does not \
             write"
        )
    })?;
    let measured_bits = nominal_bits(measured);
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
    /// 5-bit one.
    #[test]
    fn a_measurement_at_or_one_bit_below_the_base_is_accepted() {
        assert!(check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q4_K")).is_ok());
        assert!(check_measurement_encoding(GgufMix::Q5KM, &measured_at("Q4_K")).is_ok());
        assert!(check_measurement_encoding(GgufMix::Q5KM, &measured_at("Q5_K")).is_ok());
    }

    /// Measured above the format being allocated, almost nothing is damaged
    /// and the order is noise.
    #[test]
    fn a_measurement_above_the_base_is_refused() {
        let error = check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q8_0"))
            .expect_err("8 bits over a 4-bit base")
            .to_string();
        assert!(error.contains("measured at 8 bits"), "{error}");
        assert!(error.contains("Q4_K_M"), "{error}");
    }

    /// Six bits over a four-bit base is refused for the same reason eight is.
    #[test]
    fn a_six_bit_measurement_is_refused_for_a_four_bit_base() {
        let error = check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q6_K"))
            .expect_err("6 bits over a 4-bit base")
            .to_string();
        assert!(error.contains("measured at 6 bits"), "{error}");
        assert!(error.contains("the order is noise"), "{error}");
    }

    #[test]
    fn an_unrecognized_encoding_is_refused() {
        let error = check_measurement_encoding(GgufMix::Q4KM, &measured_at("Q4S32_T64"))
            .expect_err("not a block format compressr writes")
            .to_string();
        assert!(error.contains("does not write"), "{error}");
    }
}
