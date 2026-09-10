//! llama.cpp's mixed-precision GGUF allocations, reproduced.
//!
//! `Q4_K_M` is not "Q4_K everywhere". It is a base format plus a table of
//! per-role exceptions: `Q6_K` on the output weight, `Q6_K` on the attention-V
//! and FFN-down tensors of selected layers, `Q5_K` on a fused QKV projection,
//! `Q8_0` on the attention tensors of an 8-expert model. The rules live in
//! `llama_tensor_get_type` and `llama_model_quantize_impl` in llama.cpp's
//! `src/llama-quant.cpp`; [`super::mix_rules`] holds the table and names every
//! place compressr's version approximates it.
//!
//! Q4_K_S, Q4_K_M, Q5_K_S and Q5_K_M draw only on Q4_K, Q5_K, Q6_K and Q8_0,
//! which boostr writes. Q2_K, Q2_K_S, Q3_K_S, Q3_K_M and Q3_K_L need
//! llama.cpp's Q2_K/Q3_K mix rules, which this module does not reproduce, even
//! though boostr now writes flat Q2_K and Q3_K; the IQ mixes need the I-quant
//! grids, which boostr neither reads nor writes. [`parse_gguf_mix`] refuses
//! all of these by name rather than substituting a nearer format.
//!
//! Upstream matches GGUF tensor names; compressr writes the names its input
//! carried, so [`super::tensor_role`] maps both onto one role set.
//!
//! The per-tensor walk that applies the rules lives in [`super::mix_assign`].

use crate::quantization::QuantType;
use anyhow::{bail, Result};

/// A mixed-precision GGUF allocation, named as llama.cpp names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMix {
    /// Q4_K base, small variant
    Q4KS,
    /// Q4_K base, medium variant
    Q4KM,
    /// Q5_K base, small variant
    Q5KS,
    /// Q5_K base, medium variant
    Q5KM,
}

/// The mix names [`parse_gguf_mix`] accepts.
pub const GGUF_MIX_NAMES: &[&str] = &["q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m"];

impl GgufMix {
    /// The format every quantizable tensor takes unless a rule moves it.
    pub fn base(self) -> QuantType {
        match self {
            GgufMix::Q4KS | GgufMix::Q4KM => QuantType::Q4K,
            GgufMix::Q5KS | GgufMix::Q5KM => QuantType::Q5K,
        }
    }

    /// The format a promoted tensor takes.
    ///
    /// All four mixes raise to `Q6_K`: it is what llama.cpp's rules put on
    /// the output weight and on the selected attention-V and FFN-down
    /// tensors. A measured allocation
    /// ([`super::mix_measured::assign_mix_measured`]) draws on the same two
    /// formats as the positional one, so the two differ in WHERE the bits go
    /// and in nothing else.
    pub fn promoted(self) -> QuantType {
        QuantType::Q6K
    }

    /// The name llama.cpp's `llama-quantize` accepts for this mix.
    pub fn name(self) -> &'static str {
        match self {
            GgufMix::Q4KS => "Q4_K_S",
            GgufMix::Q4KM => "Q4_K_M",
            GgufMix::Q5KS => "Q5_K_S",
            GgufMix::Q5KM => "Q5_K_M",
        }
    }

    /// The `llama_ftype` this mix writes to `general.file_type`.
    ///
    /// The values are llama.cpp's own enum, from `include/llama.h`. They are
    /// NOT GGML type ids: a reader takes the type of each tensor from the
    /// tensor directory, and reads this key only to name the file.
    pub fn llama_ftype(self) -> u32 {
        match self {
            GgufMix::Q4KS => 14,
            GgufMix::Q4KM => 15,
            GgufMix::Q5KS => 16,
            GgufMix::Q5KM => 17,
        }
    }
}

/// Resolve a `--quantization` value that names a mix.
///
/// One spelling per mix, the one `llama-quantize` accepts.
///
/// # Errors
/// The name is a real llama.cpp mix this module has no rule table for
/// (Q2_K/Q3_K, or an I-quant grid), or it is not a mix name at all.
pub fn parse_gguf_mix(name: &str) -> Result<GgufMix> {
    match name.to_ascii_lowercase().as_str() {
        "q4_k_s" => Ok(GgufMix::Q4KS),
        "q4_k_m" => Ok(GgufMix::Q4KM),
        "q5_k_s" => Ok(GgufMix::Q5KS),
        "q5_k_m" => Ok(GgufMix::Q5KM),
        "q2_k" | "q2_k_s" | "q3_k_s" | "q3_k_m" | "q3_k_l" => bail!(
            "Mix `{name}` is not implemented: its base format is Q2_K or Q3_K, which \
             boostr now writes, but this module has no llama.cpp mix rule table for \
             either, so compressr cannot allocate the per-tensor exceptions. \
             Implemented mixes: {}.",
            GGUF_MIX_NAMES.join(", ")
        ),
        _ => bail!(
            "Unsupported mixed-precision name: {name}. Implemented mixes: {}.",
            GGUF_MIX_NAMES.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real llama.cpp mix with no writer here is refused by name, and the
    /// ftype values are llama.cpp's own enum rather than GGML type ids.
    #[test]
    fn names_and_ftypes_match_llama_cpp() {
        let err = parse_gguf_mix("q3_k_m").unwrap_err().to_string();
        assert!(err.contains("Q2_K or Q3_K"), "{err}");
        assert!(err.contains("q4_k_m"), "{err}");
        assert!(parse_gguf_mix("q4_k_mmm").is_err());
        assert_eq!(parse_gguf_mix("Q4_K_M").unwrap(), GgufMix::Q4KM);

        assert_eq!(GgufMix::Q4KS.llama_ftype(), 14);
        assert_eq!(GgufMix::Q4KM.llama_ftype(), 15);
        assert_eq!(GgufMix::Q5KS.llama_ftype(), 16);
        assert_eq!(GgufMix::Q5KM.llama_ftype(), 17);
    }
}
