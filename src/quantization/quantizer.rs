//! Quantization orchestration
//!
//! Thin wrapper that dispatches to boostr's quantization kernels.
//! compressr handles user-facing config; boostr does the heavy computation.
//!
//! Every GGUF block format is written by `boostr::QuantizeOps`, never here,
//! through the single call in [`super::gguf_kernel`]. boostr's writers are
//! also the accurate ones: their K-quant scales come from llama.cpp's
//! iterative per-sub-block search, worth roughly a tenth of the quantization
//! error against a plain absmax pick.
//!
//! An OPTIONAL importance matrix reaches those writers through
//! [`Quantizer::quantize_tensor_weighted`]. It is off unless a run names one,
//! and it selects a different writer for the five K-quants only.
//!
//! NF4 and FP8 stay here: neither is a GGUF block format, and boostr has no
//! writer for either.

use anyhow::{bail, Result};
use half::f16;

use super::config::QuantizationConfig;
use super::gguf_kernel::quantize_gguf;
use super::tensor::QuantizedTensor;
use super::types::QuantType;

use boostr::quant::QuantFormat;

// Use boostr's quantization tables and codebooks
use boostr::quant::tables::f32_to_fp8_e4m3;
use boostr::quant::tables::quantize_nf4 as nf4_codebook_quant;

/// Quantization orchestrator
///
/// Dispatches to appropriate boostr quantization functions based on config.
pub struct Quantizer {
    config: QuantizationConfig,
}

impl Quantizer {
    /// Create a new quantizer with the given configuration
    pub fn new(config: QuantizationConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Quantize a tensor from raw bytes
    ///
    /// # Arguments
    /// * `data` - Raw tensor bytes (F32, F16, or BF16)
    /// * `shape` - Tensor shape [rows, cols] or [numel]
    /// * `dtype` - Original data type
    pub fn quantize_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        dtype: crate::loader::UnifiedDType,
    ) -> Result<QuantizedTensor> {
        self.quantize_tensor_as(data, shape, dtype, self.config.method.quant_type())
    }

    /// Quantize a tensor into a type the caller chose for THIS tensor.
    ///
    /// A mixed-precision GGUF run assigns a format per tensor, so the type
    /// cannot come from the configuration. The planning layer and the write
    /// loop read the same assignment, because a byte count reserved for one
    /// type and filled with another shifts every later tensor offset in the
    /// output file with no error raised.
    ///
    /// # Arguments
    /// * `data` - Raw tensor bytes (F32, F16, or BF16)
    /// * `shape` - Tensor shape [rows, cols] or [numel]
    /// * `dtype` - Original data type
    /// * `quant_type` - The format to pack into
    pub fn quantize_tensor_as(
        &self,
        data: &[u8],
        shape: &[usize],
        dtype: crate::loader::UnifiedDType,
        quant_type: QuantType,
    ) -> Result<QuantizedTensor> {
        self.quantize_tensor_weighted(data, shape, dtype, quant_type, None)
    }

    /// Quantize a tensor, weighting each COLUMN by `importance`.
    ///
    /// `importance` holds one non-negative entry per element of the LAST axis
    /// — the axis quantization blocks along — so a `[out_features,
    /// in_features]` weight takes `in_features` entries and every row indexes
    /// the same vector. It comes from an importance matrix a calibration run
    /// accumulated, and it is OPTIONAL: `None` is the plain
    /// [`Quantizer::quantize_tensor_as`] path, which calls boostr's
    /// unweighted entry point and produces the bytes every earlier run
    /// produced.
    ///
    /// A tensor the matrix never measured is passed `None` by its caller and
    /// quantized unweighted. A zero vector is never substituted: zero
    /// importance is legal and says something else.
    ///
    /// # Errors
    /// - `quant_type` has no importance-weighted writer and `importance` is
    ///   `Some`. Silently dropping the vector would produce a file
    ///   indistinguishable from an unweighted one.
    /// - Whatever [`Quantizer::quantize_tensor_as`] raises, plus boostr's
    ///   refusal of a vector whose length is not the last axis.
    pub fn quantize_tensor_weighted(
        &self,
        data: &[u8],
        shape: &[usize],
        dtype: crate::loader::UnifiedDType,
        quant_type: QuantType,
        importance: Option<&[f32]>,
    ) -> Result<QuantizedTensor> {
        if importance.is_some() && !quant_type.has_importance_writer() {
            bail!(
                "{quant_type:?} has no importance-weighted quantize kernel in boostr, so an \
                 importance matrix cannot change a byte of it. Weighted: Q2_K, Q3_K, Q4_K, \
                 Q5_K, Q6_K."
            );
        }
        log::debug!(
            "Quantizing tensor: shape={:?}, dtype={}, type={:?}",
            shape,
            dtype,
            quant_type
        );

        // Convert to f32 first for uniform processing
        let f32_data = self.to_f32(data, dtype)?;

        match quant_type {
            QuantType::NF4 => self.quantize_nf4(&f32_data, shape),
            QuantType::FP8 => self.quantize_fp8(&f32_data, shape),
            // Unreachable through `convert`: `ConversionConfig::validate`
            // refuses every method with no on-disk layout before a file is
            // opened. Kept so a direct `Quantizer` caller gets the same answer.
            QuantType::GPTQ | QuantType::AWQ => {
                anyhow::bail!(
                    "{:?} is not implemented: it needs calibration statistics and a \
                     multi-tensor on-disk layout (qweight, scales, qzeros, g_idx), \
                     neither of which compressr has.",
                    quant_type
                )
            }
            QuantType::Q4_0 => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q4_0, QuantType::Q4_0, None)
            }
            QuantType::Q4_1 => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q4_1, QuantType::Q4_1, None)
            }
            QuantType::Q8_0 => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q8_0, QuantType::Q8_0, None)
            }
            QuantType::Q2K => quantize_gguf(
                &f32_data,
                shape,
                QuantFormat::Q2K,
                QuantType::Q2K,
                importance,
            ),
            QuantType::Q3K => quantize_gguf(
                &f32_data,
                shape,
                QuantFormat::Q3K,
                QuantType::Q3K,
                importance,
            ),
            QuantType::Q4K => quantize_gguf(
                &f32_data,
                shape,
                QuantFormat::Q4K,
                QuantType::Q4K,
                importance,
            ),
            QuantType::Q5K => quantize_gguf(
                &f32_data,
                shape,
                QuantFormat::Q5K,
                QuantType::Q5K,
                importance,
            ),
            QuantType::Q6K => quantize_gguf(
                &f32_data,
                shape,
                QuantFormat::Q6K,
                QuantType::Q6K,
                importance,
            ),
            // Q5_0, Q5_1, Q8_1 and FP4 have a GGML block layout and no writer
            // in boostr, so a mix never assigns one and a direct caller is
            // told rather than handed a nearby format.
            other => anyhow::bail!(
                "{other:?} has no quantize kernel in boostr, so compressr cannot pack it. \
                 Writable: Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K."
            ),
        }
    }

    /// Convert raw bytes to f32
    fn to_f32(&self, data: &[u8], dtype: crate::loader::UnifiedDType) -> Result<Vec<f32>> {
        use crate::loader::UnifiedDType;

        match dtype {
            UnifiedDType::F32 => {
                if !data.len().is_multiple_of(4) {
                    anyhow::bail!("F32 data length must be multiple of 4");
                }
                Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            UnifiedDType::F16 => {
                if !data.len().is_multiple_of(2) {
                    anyhow::bail!("F16 data length must be multiple of 2");
                }
                Ok(data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect())
            }
            UnifiedDType::BF16 => {
                if !data.len().is_multiple_of(2) {
                    anyhow::bail!("BF16 data length must be multiple of 2");
                }
                Ok(data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect())
            }
            _ => anyhow::bail!("Unsupported dtype for quantization: {:?}", dtype),
        }
    }

    // ========================================================================
    // Non-GGUF formats — no boostr writer exists for these
    // ========================================================================

    /// Quantize to NF4 using boostr's codebook API
    ///
    /// The packed block is 32 bytes of 4-bit codebook indices for 64 elements
    /// and carries no scale, so the per-block absmax goes in
    /// [`QuantizedTensor::scales`] and the dequantizer reads it from there.
    fn quantize_nf4(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        const NF4_BLOCK_SIZE: usize = 64;
        let n = data.len();
        let n_blocks = n.div_ceil(NF4_BLOCK_SIZE);

        let mut packed = Vec::with_capacity(n_blocks * 32); // 32 bytes per NF4 block
        let mut scales = Vec::with_capacity(n_blocks);

        for block_idx in 0..n_blocks {
            let start = block_idx * NF4_BLOCK_SIZE;
            let end = (start + NF4_BLOCK_SIZE).min(n);

            // Find absmax for this block
            let absmax = data[start..end]
                .iter()
                .fold(0.0f32, |acc, &x| acc.max(x.abs()));

            let absmax = if absmax > 0.0 { absmax } else { 1.0 };
            scales.push(f16::from_f32(absmax));

            // Quantize values using boostr's codebook
            let inv_absmax = 1.0 / absmax;
            let mut block_quants = [0u8; NF4_BLOCK_SIZE];

            for i in 0..(end - start) {
                let normalized = data[start + i] * inv_absmax;
                block_quants[i] = nf4_codebook_quant(normalized);
            }

            // Pack 4-bit indices (2 per byte)
            for i in (0..NF4_BLOCK_SIZE).step_by(2) {
                let lo = block_quants[i];
                let hi = if i + 1 < NF4_BLOCK_SIZE {
                    block_quants[i + 1]
                } else {
                    0
                };
                packed.push(lo | (hi << 4));
            }
        }

        Ok(QuantizedTensor {
            packed_data: packed,
            scales,
            original_shape: shape.to_vec(),
            original_numel: data.len(),
            quant_type: QuantType::NF4,
            block_size: NF4_BLOCK_SIZE,
        })
    }

    /// Quantize to FP8 (E4M3 format)
    ///
    /// One byte per element and one tensor-wide scale, which has nowhere to
    /// live in the payload and so goes in [`QuantizedTensor::scales`].
    fn quantize_fp8(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        let amax = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = if amax > 0.0 { 448.0 / amax } else { 1.0 };

        let mut packed = Vec::with_capacity(data.len());

        for &x in data {
            let scaled = x * scale;
            let fp8 = f32_to_fp8_e4m3(scaled);
            packed.push(fp8);
        }

        Ok(QuantizedTensor {
            packed_data: packed,
            scales: vec![f16::from_f32(1.0 / scale)],
            original_shape: shape.to_vec(),
            original_numel: data.len(),
            quant_type: QuantType::FP8,
            block_size: 1,
        })
    }
}
