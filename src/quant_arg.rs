//! Resolving the `--quantization` value.
//!
//! One value names either a flat GGUF block format (`q4_k`) or one of
//! llama.cpp's mixed-precision allocations (`q4_k_m`). A mix carries a base
//! format, so both cases produce a [`QuantizationConfig`]; a mix additionally
//! produces the [`GgufMix`] the planning layer allocates from.

use anyhow::Result;
use compressr::{parse_gguf_mix, GgufMix, QuantizationConfig, QuantizationMethod, GGUF_MIX_NAMES};

/// What one `--quantization` value resolves to.
#[derive(Debug, Clone)]
pub struct QuantizationChoice {
    /// The base format, and the format a flat run uses everywhere
    pub config: QuantizationConfig,
    /// The mixed-precision allocation, when the value named one
    pub mix: Option<GgufMix>,
}

/// Resolve a `--quantization` value.
///
/// The `_S` / `_M` / `_L` suffix is matched EXACTLY rather than trimmed as a
/// character set, so `q4_k_mmm` is refused rather than read as `q4_k`.
///
/// # Errors
/// The value names no format, or it names a mix whose base format has no
/// writer in boostr — flat `q2_k` and `q3_k` are writable, but the
/// `q2_k*` and `q3_k*` MIXES need llama.cpp's mix rules for them, which do
/// not exist here, so those mixes are refused by name.
pub fn parse_quantization(s: &str) -> Result<QuantizationChoice> {
    let lowered = s.to_lowercase();

    // A mix name is a whole name, tested before the flat names: `q4_k_m` is
    // not `q4_k` with a suffix to discard, it is a different allocation.
    for suffix in ["_s", "_m", "_l"] {
        if lowered.ends_with(suffix) {
            let mix = parse_gguf_mix(&lowered)?;
            return Ok(QuantizationChoice {
                config: method_config(mix_base_method(mix)),
                mix: Some(mix),
            });
        }
    }

    let method = match lowered.as_str() {
        "q4_0" | "q4.0" => QuantizationMethod::Q4_0,
        "q4_1" | "q4.1" => QuantizationMethod::Q4_1,
        "q8_0" | "q8.0" => QuantizationMethod::Q8_0,
        "q2_k" | "q2k" => QuantizationMethod::Q2K,
        "q3_k" | "q3k" => QuantizationMethod::Q3K,
        "q4_k" | "q4k" => QuantizationMethod::Q4K,
        "q5_k" | "q5k" => QuantizationMethod::Q5K,
        "q6_k" | "q6k" => QuantizationMethod::Q6K,
        "iq4_nl" | "iq4nl" => QuantizationMethod::IQ4NL,
        "iq4_xs" | "iq4xs" => QuantizationMethod::IQ4XS,
        "pq2_0" | "pq2.0" => QuantizationMethod::PQ2_0,
        "ptq1_0" | "ptq1.0" => QuantizationMethod::PTQ1_0,
        "nf4" => QuantizationMethod::Nf4,
        "fp8" => QuantizationMethod::Fp8,
        _ => anyhow::bail!(
            "Unsupported quantization: {}. Flat formats: q4_0, q4_1, q8_0, q2_k, q3_k, \
             q4_k, q5_k, q6_k, iq4_nl, iq4_xs, nf4, fp8. Repack of a lowbit GGUF: pq2_0 \
             (from PTQ1_0), ptq1_0 (from PQ2_0). Mixed precision: {}.",
            s,
            GGUF_MIX_NAMES.join(", ")
        ),
    };

    Ok(QuantizationChoice {
        config: method_config(method),
        mix: None,
    })
}

/// The base format a mix falls back to for every tensor no rule moves.
fn mix_base_method(mix: GgufMix) -> QuantizationMethod {
    match mix {
        GgufMix::Q4KS | GgufMix::Q4KM => QuantizationMethod::Q4K,
        GgufMix::Q5KS | GgufMix::Q5KM => QuantizationMethod::Q5K,
    }
}

/// The bit width and block size that go with a method.
fn method_config(method: QuantizationMethod) -> QuantizationConfig {
    let (bits, group_size) = match method {
        QuantizationMethod::Q4_0 | QuantizationMethod::Q4_1 | QuantizationMethod::Nf4 => (4, 32),
        QuantizationMethod::Q8_0 => (8, 32),
        QuantizationMethod::Q2K => (2, 256),
        QuantizationMethod::Q3K => (3, 256),
        QuantizationMethod::Q4K => (4, 256),
        QuantizationMethod::Q5K => (5, 256),
        QuantizationMethod::Q6K => (6, 256),
        QuantizationMethod::IQ4NL => (4, 32),
        QuantizationMethod::IQ4XS => (4, 256),
        // A trit occupies a 2-bit code slot in both layouts.
        QuantizationMethod::PQ2_0 | QuantizationMethod::PTQ1_0 => (2, 128),
        QuantizationMethod::Fp8 => (8, 1),
        QuantizationMethod::Gptq | QuantizationMethod::Awq => (4, 128),
    };

    QuantizationConfig {
        method,
        bits,
        group_size,
        calibration_dataset: None,
        calibration_samples: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_name_carries_no_mix() {
        let choice = parse_quantization("q4_k").unwrap();
        assert_eq!(choice.config.method, QuantizationMethod::Q4K);
        assert!(choice.mix.is_none());
        assert_eq!(choice.config.group_size, 256);
    }

    #[test]
    fn a_mix_name_carries_its_base_format() {
        let choice = parse_quantization("q4_k_m").unwrap();
        assert_eq!(choice.mix, Some(GgufMix::Q4KM));
        assert_eq!(choice.config.method, QuantizationMethod::Q4K);

        let choice = parse_quantization("Q5_K_S").unwrap();
        assert_eq!(choice.mix, Some(GgufMix::Q5KS));
        assert_eq!(choice.config.method, QuantizationMethod::Q5K);
    }

    /// A repeated suffix is not a mix name. Trimming `_s`/`_m`/`_l` as a
    /// character set used to accept this and quantize it as `q4_k`.
    #[test]
    fn a_repeated_suffix_is_refused() {
        assert!(parse_quantization("q4_k_mmm").is_err());
        assert!(parse_quantization("q4_k_m_m").is_err());
    }

    /// A real llama.cpp mix compressr has no writer for is refused by name,
    /// naming the missing kernel rather than substituting a nearer format.
    #[test]
    fn a_mix_needing_a_missing_writer_is_refused() {
        let err = parse_quantization("q3_k_l").unwrap_err().to_string();
        assert!(err.contains("Q2_K or Q3_K"), "{err}");
    }

    #[test]
    fn an_unknown_name_lists_what_is_accepted() {
        let err = parse_quantization("q4_9").unwrap_err().to_string();
        assert!(err.contains("q4_k_m"), "{err}");
        assert!(err.contains("q4_0"), "{err}");
        assert!(err.contains("iq4_nl"), "{err}");
        assert!(err.contains("iq4_xs"), "{err}");
    }

    /// Both spellings resolve to the same method, at IQ4_NL's real bpw.
    #[test]
    fn iq4_nl_parses_under_both_spellings() {
        for name in ["iq4_nl", "iq4nl"] {
            let choice = parse_quantization(name).unwrap();
            assert_eq!(choice.config.method, QuantizationMethod::IQ4NL);
            assert!(choice.mix.is_none());
            assert_eq!(choice.config.bits, 4);
            assert_eq!(choice.config.group_size, 32);
        }
    }

    /// The lowbit names resolve to the repack-only methods at group 128.
    #[test]
    fn lowbit_names_parse_as_repack_methods() {
        let choice = parse_quantization("PQ2_0").unwrap();
        assert_eq!(choice.config.method, QuantizationMethod::PQ2_0);
        assert!(choice.config.method.is_repack_only());
        assert_eq!(choice.config.group_size, 128);
        let choice = parse_quantization("ptq1_0").unwrap();
        assert_eq!(choice.config.method, QuantizationMethod::PTQ1_0);
        assert!(choice.mix.is_none());
    }

    /// Both spellings resolve to the same method, at IQ4_XS's super-block
    /// geometry — 256 elements per block, unlike IQ4_NL's 32.
    #[test]
    fn iq4_xs_parses_under_both_spellings() {
        for name in ["iq4_xs", "iq4xs"] {
            let choice = parse_quantization(name).unwrap();
            assert_eq!(choice.config.method, QuantizationMethod::IQ4XS);
            assert!(choice.mix.is_none());
            assert_eq!(choice.config.bits, 4);
            assert_eq!(choice.config.group_size, 256);
        }
    }
}
