//! Quantization configuration
//!
//! User-friendly configuration for quantization methods.

use super::types::QuantType;

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// GPTQ (weight-only quantization with Hessian)
    Gptq,
    /// AWQ (activation-aware weight quantization)
    Awq,
    /// NF4 (Normal Float 4)
    Nf4,
    /// FP8 (E4M3 or E5M2)
    Fp8,
    /// GGUF Q4_0 (llama.cpp compatible)
    Q4_0,
    /// GGUF Q4_1 (llama.cpp compatible)
    Q4_1,
    /// GGUF Q8_0 (llama.cpp compatible)
    Q8_0,
    /// GGUF Q2_K (llama.cpp K-quant)
    Q2K,
    /// GGUF Q3_K (llama.cpp K-quant)
    Q3K,
    /// GGUF Q4_K (llama.cpp K-quant)
    Q4K,
    /// GGUF Q5_K (llama.cpp K-quant)
    Q5K,
    /// GGUF Q6_K (llama.cpp K-quant)
    Q6K,
    /// GGUF IQ4_NL (llama.cpp I-quant, flat only — no mix rules exist here)
    IQ4NL,
    /// GGUF IQ4_XS (llama.cpp I-quant super-block, flat only — no mix rules
    /// exist here)
    IQ4XS,
}

impl QuantizationMethod {
    /// Check if this method requires calibration data
    pub fn requires_calibration(&self) -> bool {
        matches!(self, Self::Gptq | Self::Awq)
    }

    /// Get the approximate bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Gptq | Self::Awq | Self::Nf4 | Self::Q4_0 | Self::Q4_1 | Self::Q4K => 4.5,
            Self::IQ4NL => 4.5,
            // 136 bytes per 256-element super-block: 4.25 bpw, not 4.5.
            Self::IQ4XS => 4.25,
            Self::Fp8 | Self::Q8_0 => 8.5,
            Self::Q2K => 2.5,
            Self::Q3K => 3.5,
            Self::Q5K => 5.5,
            Self::Q6K => 6.5,
        }
    }

    /// Whether compressr can write a tensor packed by this method.
    ///
    /// GGUF is the only container compressr writes quantized tensors into, and
    /// only these ten have a GGML block type to write them as. The other
    /// four each need an on-disk layout that does not exist yet: NF4 and FP8
    /// keep their scales OUTSIDE the packed payload, and GPTQ and AWQ split
    /// one layer into several tensors (`qweight` packed as I32, plus
    /// `scales`, `qzeros`, `g_idx`). Writing their payload alone produces a
    /// file no loader can dequantize, so [`crate::ConversionConfig::validate`]
    /// refuses them before any output file is created.
    pub fn is_writable(&self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q8_0
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::IQ4NL
                | Self::IQ4XS
        )
    }

    /// The [`QuantType`] this method packs a tensor as.
    ///
    /// The two enums are separate because a method is a user-facing choice
    /// while a type identifies a packed payload; a mixed-precision run picks a
    /// type per tensor and never goes through a method at all.
    pub fn quant_type(&self) -> QuantType {
        match self {
            Self::Q4_0 => QuantType::Q4_0,
            Self::Q4_1 => QuantType::Q4_1,
            Self::Q8_0 => QuantType::Q8_0,
            Self::Q2K => QuantType::Q2K,
            Self::Q3K => QuantType::Q3K,
            Self::Q4K => QuantType::Q4K,
            Self::Q5K => QuantType::Q5K,
            Self::Q6K => QuantType::Q6K,
            Self::IQ4NL => QuantType::IQ4NL,
            Self::IQ4XS => QuantType::IQ4XS,
            Self::Nf4 => QuantType::NF4,
            Self::Fp8 => QuantType::FP8,
            Self::Gptq => QuantType::GPTQ,
            Self::Awq => QuantType::AWQ,
        }
    }

    /// Get the block size for this method
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::IQ4XS => 256,
            Self::IQ4NL => 32,
            Self::Nf4 => 64,
            Self::Gptq | Self::Awq => 128, // Typical group size
            Self::Fp8 => 1,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization method
    pub method: QuantizationMethod,
    /// Number of bits (2, 3, 4, 8)
    pub bits: u8,
    /// Group size for quantization
    pub group_size: usize,
    /// Calibration dataset name
    pub calibration_dataset: Option<String>,
    /// Number of calibration samples
    pub calibration_samples: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self::q4_0()
    }
}

impl QuantizationConfig {
    /// Create a GPTQ configuration
    pub fn gptq(bits: u8, group_size: usize) -> Self {
        Self {
            method: QuantizationMethod::Gptq,
            bits,
            group_size,
            calibration_dataset: None,
            calibration_samples: 128,
        }
    }

    /// Create an AWQ configuration
    pub fn awq(bits: u8, group_size: usize) -> Self {
        Self {
            method: QuantizationMethod::Awq,
            bits,
            group_size,
            calibration_dataset: Some("wikitext2".to_string()),
            calibration_samples: 128,
        }
    }

    /// Create an NF4 configuration
    pub fn nf4() -> Self {
        Self {
            method: QuantizationMethod::Nf4,
            bits: 4,
            group_size: 64,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create an FP8 configuration
    pub fn fp8() -> Self {
        Self {
            method: QuantizationMethod::Fp8,
            bits: 8,
            group_size: 1,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q4_0 configuration (llama.cpp compatible)
    pub fn q4_0() -> Self {
        Self {
            method: QuantizationMethod::Q4_0,
            bits: 4,
            group_size: 32,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q8_0 configuration (llama.cpp compatible)
    pub fn q8_0() -> Self {
        Self {
            method: QuantizationMethod::Q8_0,
            bits: 8,
            group_size: 32,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q2_K configuration (llama.cpp K-quant)
    pub fn q2_k() -> Self {
        Self {
            method: QuantizationMethod::Q2K,
            bits: 2,
            group_size: 256,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q3_K configuration (llama.cpp K-quant)
    pub fn q3_k() -> Self {
        Self {
            method: QuantizationMethod::Q3K,
            bits: 3,
            group_size: 256,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q4_K configuration (llama.cpp K-quant)
    pub fn q4_k() -> Self {
        Self {
            method: QuantizationMethod::Q4K,
            bits: 4,
            group_size: 256,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q5_K configuration
    pub fn q5_k() -> Self {
        Self {
            method: QuantizationMethod::Q5K,
            bits: 5,
            group_size: 256,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create a Q6_K configuration
    pub fn q6_k() -> Self {
        Self {
            method: QuantizationMethod::Q6K,
            bits: 6,
            group_size: 256,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create an IQ4_NL configuration (llama.cpp I-quant, flat only)
    pub fn iq4_nl() -> Self {
        Self {
            method: QuantizationMethod::IQ4NL,
            bits: 4,
            group_size: 32,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }

    /// Create an IQ4_XS configuration (llama.cpp I-quant super-block, flat only)
    pub fn iq4_xs() -> Self {
        Self {
            method: QuantizationMethod::IQ4XS,
            bits: 4,
            group_size: 256,
            calibration_dataset: None,
            calibration_samples: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_only_gguf_block_formats_are_writable() {
        for method in [
            QuantizationMethod::Q4_0,
            QuantizationMethod::Q4_1,
            QuantizationMethod::Q8_0,
            QuantizationMethod::Q2K,
            QuantizationMethod::Q3K,
            QuantizationMethod::Q4K,
            QuantizationMethod::Q5K,
            QuantizationMethod::Q6K,
            QuantizationMethod::IQ4NL,
            QuantizationMethod::IQ4XS,
        ] {
            assert!(method.is_writable(), "{method:?} has a GGML block type");
        }

        for method in [
            QuantizationMethod::Nf4,
            QuantizationMethod::Fp8,
            QuantizationMethod::Gptq,
            QuantizationMethod::Awq,
        ] {
            assert!(
                !method.is_writable(),
                "{method:?} has no on-disk layout in compressr"
            );
        }
    }

    /// IQ4_NL packs as `QuantType::IQ4NL`, at 4.5 bits per weight and a
    /// 32-element block — a simple block, not a K-quant super-block.
    #[test]
    fn test_iq4_nl_maps_to_its_own_quant_type() {
        let method = QuantizationMethod::IQ4NL;
        assert_eq!(method.quant_type(), QuantType::IQ4NL);
        assert_eq!(method.bits_per_weight(), 4.5);
        assert_eq!(method.block_size(), 32);
    }

    /// IQ4_XS packs as `QuantType::IQ4XS`, at 4.25 bits per weight (136
    /// bytes per 256-element super-block) — a super-block, not IQ4_NL's
    /// simple block, and lighter than IQ4_NL's 4.5 bpw.
    #[test]
    fn test_iq4_xs_maps_to_its_own_quant_type() {
        let method = QuantizationMethod::IQ4XS;
        assert_eq!(method.quant_type(), QuantType::IQ4XS);
        assert_eq!(method.bits_per_weight(), 4.25);
        assert_eq!(method.block_size(), 256);
    }
}
