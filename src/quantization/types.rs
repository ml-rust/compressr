//! Quantization type identifiers
//!
//! Identifies which format a [`crate::quantization::QuantizedTensor`] was
//! packed with. The GGML type ID and block-byte-size mapping live on
//! `GgmlType` in `crate::formats::gguf`, which is the writer's own source of
//! truth — this enum does not duplicate it.

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
    /// `ggml-quants.c` carries a second writer per K-quant, plus one for
    /// IQ4_NL, taken the moment an importance matrix is supplied, and boostr
    /// reproduces all six. Every other block format has one writer, which
    /// reads no importance, so a matrix cannot change a single byte it
    /// produces.
    pub fn has_importance_writer(&self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::IQ4NL
        )
    }

    /// Get elements per block for this type
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            Self::IQ4NL => 32,
            Self::NF4 | Self::FP4 => 64,
            Self::FP8 => 1,
            Self::GPTQ | Self::AWQ => 128,
        }
    }
}
