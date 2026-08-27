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
    // Advanced formats
    NF4,
    FP4,
    FP8,
    GPTQ,
    AWQ,
}

impl QuantType {
    /// Get elements per block for this type
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            Self::NF4 | Self::FP4 => 64,
            Self::FP8 => 1,
            Self::GPTQ | Self::AWQ => 128,
        }
    }
}
