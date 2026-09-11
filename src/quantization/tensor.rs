//! Quantized tensor representation
//!
//! Thin wrapper around quantized data with metadata for format writing.

use super::types::QuantType;
use half::f16;

// Use boostr's quantization tables and utilities
use boostr::quant::cpu::kernels::dequant;
use boostr::quant::cpu::kernels::dequant_iq4;
use boostr::quant::tables::dequantize_nf4 as nf4_codebook_dequant;

/// Quantized tensor data
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized data (format-specific layout)
    pub packed_data: Vec<u8>,
    /// Side-band scales, for formats whose `packed_data` does not carry them
    ///
    /// EMPTY for every GGUF block format (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K):
    /// those are written by `boostr::QuantizeOps`, which returns blocks with
    /// the scale — and the min, where the format has one — packed inside each
    /// block, exactly where a GGUF reader looks for it. Nothing reads this
    /// field for those formats, and a writer must not consult it.
    ///
    /// Populated only by the two formats that have nowhere else to put a
    /// scale: NF4 (one f16 absmax per 64-element block) and FP8 (a single
    /// tensor-wide dequant scale).
    pub scales: Vec<f16>,
    /// Original tensor shape
    pub original_shape: Vec<usize>,
    /// Original number of elements
    pub original_numel: usize,
    /// Quantization type
    pub quant_type: QuantType,
    /// Block size
    pub block_size: usize,
}

impl QuantizedTensor {
    /// Get the compressed size in bytes
    pub fn size_bytes(&self) -> usize {
        self.packed_data.len()
    }

    /// Get the original size in bytes (assuming F16)
    pub fn original_size_bytes(&self) -> usize {
        self.original_numel * 2 // F16 = 2 bytes
    }

    /// Get the compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.original_size_bytes() as f32 / self.size_bytes() as f32
    }

    /// Dequantize back to f32 (for validation)
    ///
    /// Dispatches to boostr's dequantization functions where available.
    pub fn dequantize(&self) -> Vec<f32> {
        match self.quant_type {
            QuantType::NF4 => self.dequantize_nf4_boostr(),
            QuantType::Q4_0 => self.dequantize_ggml(32, 18, dequant::dequant_q4_0),
            QuantType::Q8_0 => self.dequantize_ggml(32, 34, dequant::dequant_q8_0),
            QuantType::Q4_1 => self.dequantize_ggml(32, 20, dequant::dequant_q4_1),
            QuantType::Q2K => self.dequantize_ggml(256, 84, dequant::dequant_q2k),
            QuantType::Q3K => self.dequantize_ggml(256, 110, dequant::dequant_q3k),
            QuantType::Q4K => self.dequantize_ggml(256, 144, dequant::dequant_q4k),
            QuantType::Q5K => self.dequantize_ggml(256, 176, dequant::dequant_q5k),
            QuantType::Q6K => self.dequantize_ggml(256, 210, dequant::dequant_q6k),
            QuantType::IQ4NL => self.dequantize_ggml(32, 18, dequant_iq4::dequant_iq4_nl),
            QuantType::IQ4XS => self.dequantize_ggml(256, 136, dequant_iq4::dequant_iq4_xs),
            _ => {
                log::warn!("Dequantization not implemented for {:?}", self.quant_type);
                vec![0.0; self.original_numel]
            }
        }
    }

    /// Compute quantization error (RMSE)
    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        let dequantized = self.dequantize();
        if dequantized.len() != original.len() {
            return f32::MAX;
        }

        let mse: f32 = original
            .iter()
            .zip(dequantized.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / original.len() as f32;

        mse.sqrt()
    }

    // ========================================================================
    // NF4 Dequantization (using boostr codebook)
    // ========================================================================

    fn dequantize_nf4_boostr(&self) -> Vec<f32> {
        const NF4_BLOCK_SIZE: usize = 64;
        let n_blocks = self.original_numel.div_ceil(NF4_BLOCK_SIZE);
        let mut output = Vec::with_capacity(self.original_numel);

        for block_idx in 0..n_blocks {
            let block_offset = block_idx * 32; // 32 bytes per NF4 block (64 4-bit indices)
                                               // A truncated `scales` means the block has no absmax to apply; 0
                                               // keeps the length contract instead of panicking mid-dequant.
            let absmax = self
                .scales
                .get(block_idx)
                .map_or(0.0, |scale| scale.to_f32());

            for i in 0..NF4_BLOCK_SIZE {
                if output.len() >= self.original_numel {
                    break;
                }

                // Unpack the 4-bit index. A truncated `packed_data` reads as 0
                // rather than panicking, matching how a short `scales` above
                // keeps the output length contract.
                let byte_idx = block_offset + i / 2;
                let byte = self.packed_data.get(byte_idx).copied().unwrap_or(0);
                let idx = if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };

                // Dequantize using boostr's NF4 codebook
                let normalized = nf4_codebook_dequant(idx);
                output.push(normalized * absmax);
            }
        }

        output.truncate(self.original_numel);
        output
    }

    /// Dequantize a GGML block format with boostr's reader kernel.
    ///
    /// Every GGML layout is decoded by boostr, never re-derived here: a second
    /// copy of a block layout in compressr can agree with compressr's writer and
    /// still disagree with every real reader, which is exactly how Q6_K shipped
    /// broken. `block_elems` and `block_bytes` describe the format; `kernel` is
    /// the matching `boostr::quant::cpu::kernels::dequant` function.
    ///
    /// Trailing elements of a partial final super-block are dropped, and a
    /// truncated `packed_data` is zero-filled up to `original_numel`.
    fn dequantize_ggml(
        &self,
        block_elems: usize,
        block_bytes: usize,
        kernel: fn(&[u8], &mut [f32]),
    ) -> Vec<f32> {
        let n_blocks = self.packed_data.len() / block_bytes;
        let mut output = vec![0.0f32; n_blocks * block_elems];
        kernel(&self.packed_data[..n_blocks * block_bytes], &mut output);
        output.resize(self.original_numel, 0.0);
        output
    }
}
