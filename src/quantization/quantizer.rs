//! Quantization orchestration
//!
//! Thin wrapper that dispatches to boostr's quantization kernels.
//! compressr handles user-facing config; boostr does the heavy computation.
//!
//! Every GGUF block format is written by `boostr::QuantizeOps`, never here. A
//! second writer in compressr can agree with compressr's own reader and still
//! disagree with every real reader — that is exactly how Q6_K once shipped with
//! a wrong field order. boostr's writers are also the accurate ones: their
//! K-quant scales come from llama.cpp's iterative per-sub-block search, worth
//! roughly a tenth of the quantization error against a plain absmax pick.
//!
//! NF4 and FP8 stay here: neither is a GGUF block format, and boostr has no
//! writer for either.

use anyhow::{anyhow, bail, Result};
use half::f16;

use super::config::{QuantizationConfig, QuantizationMethod};
use super::tensor::QuantizedTensor;
use super::types::QuantType;

use boostr::quant::{QuantFormat, QuantizeOps};
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};

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
        log::debug!(
            "Quantizing tensor: shape={:?}, dtype={}, method={:?}",
            shape,
            dtype,
            self.config.method
        );

        // Convert to f32 first for uniform processing
        let f32_data = self.to_f32(data, dtype)?;

        match self.config.method {
            QuantizationMethod::Nf4 => self.quantize_nf4(&f32_data, shape),
            QuantizationMethod::Fp8 => self.quantize_fp8(&f32_data, shape),
            // Unreachable through `convert`: `ConversionConfig::validate`
            // refuses every method with no on-disk layout before a file is
            // opened. Kept so a direct `Quantizer` caller gets the same answer.
            QuantizationMethod::Gptq | QuantizationMethod::Awq => {
                anyhow::bail!(
                    "{:?} is not implemented: it needs calibration statistics and a \
                     multi-tensor on-disk layout (qweight, scales, qzeros, g_idx), \
                     neither of which compressr has.",
                    self.config.method
                )
            }
            QuantizationMethod::Q4_0 => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q4_0, QuantType::Q4_0)
            }
            QuantizationMethod::Q4_1 => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q4_1, QuantType::Q4_1)
            }
            QuantizationMethod::Q8_0 => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q8_0, QuantType::Q8_0)
            }
            QuantizationMethod::Q4K => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q4K, QuantType::Q4K)
            }
            QuantizationMethod::Q5K => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q5K, QuantType::Q5K)
            }
            QuantizationMethod::Q6K => {
                quantize_gguf(&f32_data, shape, QuantFormat::Q6K, QuantType::Q6K)
            }
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

/// Quantize into a GGUF block format with boostr's writer kernels.
///
/// boostr packs blocks along the LAST axis of `shape`, so a
/// `[out_features, in_features]` weight blocks along `in_features` — the same
/// axis GGUF calls `ne[0]`. It requires that axis to be a whole number of
/// blocks, which [`crate::policy::plan_tensor`] already checks before a tensor
/// gets here; its error is surfaced rather than swallowed, because the two
/// rules drifting apart is precisely the failure worth seeing.
///
/// The returned blocks carry their own scales inside them, so
/// [`QuantizedTensor::scales`] stays empty for every format written this way.
fn quantize_gguf(
    data: &[f32],
    shape: &[usize],
    format: QuantFormat,
    quant_type: QuantType,
) -> Result<QuantizedTensor> {
    let numel: usize = shape.iter().product();
    if shape.is_empty() || numel != data.len() {
        bail!(
            "shape {:?} describes {} elements but the tensor has {}",
            shape,
            numel,
            data.len()
        );
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(data, shape, &device).map_err(|e| {
        anyhow!(
            "failed to stage tensor for {} quantization: {e}",
            format.name()
        )
    })?;

    let quantized = client
        .quantize(&input, format)
        .map_err(|e| anyhow!("boostr {} quantization failed: {e}", format.name()))?;

    let packed_data = quantized
        .to_bytes()
        .map_err(|e| anyhow!("failed to read back {} blocks: {e}", format.name()))?;

    Ok(QuantizedTensor {
        packed_data,
        scales: Vec::new(),
        original_shape: shape.to_vec(),
        original_numel: numel,
        quant_type,
        block_size: format.block_size(),
    })
}
