//! Quantization type identifiers
//!
//! Identifies which format a [`crate::quantization::QuantizedTensor`] was
//! packed with. The GGML type ID and block-byte-size mapping live on
//! `GgmlType` in `crate::formats::gguf`, which is the writer's own source of
//! truth — this enum does not duplicate it.

use boostr::quant::QuantFormat;

/// Quantization type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    // Simple block formats (32 elements per block)
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    // K-Quant super-block formats (256 elements per block)
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    // I-quant, grid-based (32 elements per block, flat format only)
    IQ4NL,
    // I-quant, grid-based super-block (256 elements per block, flat format
    // only, 4.25 bits per weight — NOT the 4.5 of IQ4_NL)
    IQ4XS,
    // Lowbit 2-bit codes, 128 elements per 34-byte block. Reached by
    // repacking PTQ1_0 only: no float-to-PQ2_0 quantizer exists here.
    PQ2_0,
    // Lowbit base-3 trits, 128 elements per 28-byte block. Reached by
    // repacking PQ2_0 only: no float-to-ternary quantizer exists here.
    PTQ1_0,
    // Advanced formats
    NF4,
    FP4,
    FP8,
    GPTQ,
    AWQ,
}

impl QuantType {
    /// Whether boostr has an IMPORTANCE-WEIGHTED writer for this format.
    ///
    /// `ggml-quants.c` carries a second writer per K-quant, plus one each for
    /// IQ4_NL and IQ4_XS, taken the moment an importance matrix is supplied,
    /// and boostr reproduces all seven. Every other block format has one
    /// writer, which reads no importance, so a matrix cannot change a single
    /// byte it produces.
    pub fn has_importance_writer(&self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::IQ4NL | Self::IQ4XS
        )
    }

    /// Whether this type is written by repacking a lowbit source tensor's
    /// codes, never by quantizing float weights.
    ///
    /// PTQ1_0 and PQ2_0 store the same codes in two layouts. Producing those
    /// codes from float weights is the training-aware pipeline that made the
    /// checkpoint, which compressr does not carry.
    pub fn is_repack_only(&self) -> bool {
        matches!(self, Self::PQ2_0 | Self::PTQ1_0)
    }

    /// The boostr block format behind a GGML type; `None` for the
    /// SafeTensors-side schemes (NF4, FP4, FP8, GPTQ, AWQ). Block geometry
    /// is read from it, never restated here.
    pub fn ggml_format(&self) -> Option<QuantFormat> {
        Some(match self {
            Self::Q4_0 => QuantFormat::Q4_0,
            Self::Q4_1 => QuantFormat::Q4_1,
            Self::Q5_0 => QuantFormat::Q5_0,
            Self::Q5_1 => QuantFormat::Q5_1,
            Self::Q8_0 => QuantFormat::Q8_0,
            Self::Q8_1 => QuantFormat::Q8_1,
            Self::Q2K => QuantFormat::Q2K,
            Self::Q3K => QuantFormat::Q3K,
            Self::Q4K => QuantFormat::Q4K,
            Self::Q5K => QuantFormat::Q5K,
            Self::Q6K => QuantFormat::Q6K,
            Self::IQ4NL => QuantFormat::IQ4NL,
            Self::IQ4XS => QuantFormat::IQ4XS,
            Self::PQ2_0 => QuantFormat::PQ2_0,
            Self::PTQ1_0 => QuantFormat::PTQ1_0,
            Self::NF4 | Self::FP4 | Self::FP8 | Self::GPTQ | Self::AWQ => return None,
        })
    }

    /// Get elements per block for this type
    pub fn block_size(&self) -> usize {
        match self.ggml_format() {
            Some(format) => format.block_size(),
            None => match self {
                Self::NF4 | Self::FP4 => 64,
                Self::FP8 => 1,
                _ => 128,
            },
        }
    }
}
