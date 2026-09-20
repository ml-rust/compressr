//! Decoding block-quantized tensors to F32 for a `--dtype` cast.
//!
//! A cast names the element type every weight is stored as, so a tensor that
//! arrives as GGML blocks is decoded first. The decode is boostr's reader
//! kernel, the one blazr and the round-trip tests use; compressr holds no
//! block layout of its own.

use crate::loader::UnifiedDType;
use anyhow::{anyhow, bail, Result};
use boostr::quant::cpu::kernels::quant_matmul::dequant_row_f32;
use boostr::quant::QuantFormat;

/// The boostr block format a stored dtype names, or `None` for an element type.
pub fn quant_format_of(dtype: UnifiedDType) -> Option<QuantFormat> {
    Some(match dtype {
        UnifiedDType::Q4_0 => QuantFormat::Q4_0,
        UnifiedDType::Q4_1 => QuantFormat::Q4_1,
        UnifiedDType::Q5_0 => QuantFormat::Q5_0,
        UnifiedDType::Q5_1 => QuantFormat::Q5_1,
        UnifiedDType::Q8_0 => QuantFormat::Q8_0,
        UnifiedDType::Q8_1 => QuantFormat::Q8_1,
        UnifiedDType::Q2K => QuantFormat::Q2K,
        UnifiedDType::Q3K => QuantFormat::Q3K,
        UnifiedDType::Q4K => QuantFormat::Q4K,
        UnifiedDType::Q5K => QuantFormat::Q5K,
        UnifiedDType::Q6K => QuantFormat::Q6K,
        UnifiedDType::Q8K => QuantFormat::Q8K,
        UnifiedDType::IQ2XXS => QuantFormat::IQ2XXS,
        UnifiedDType::IQ2XS => QuantFormat::IQ2XS,
        UnifiedDType::IQ3XXS => QuantFormat::IQ3XXS,
        UnifiedDType::IQ1S => QuantFormat::IQ1S,
        UnifiedDType::IQ4NL => QuantFormat::IQ4NL,
        UnifiedDType::IQ3S => QuantFormat::IQ3S,
        UnifiedDType::IQ2S => QuantFormat::IQ2S,
        UnifiedDType::IQ4XS => QuantFormat::IQ4XS,
        UnifiedDType::IQ1M => QuantFormat::IQ1M,
        UnifiedDType::Q1_0 => QuantFormat::Q1_0,
        UnifiedDType::Q2_0 => QuantFormat::Q2_0,
        UnifiedDType::PQ2_0 => QuantFormat::PQ2_0,
        UnifiedDType::PTQ1_0 => QuantFormat::PTQ1_0,
        _ => return None,
    })
}

/// Decode a block-quantized tensor to F32 through boostr's reader kernel.
///
/// `numel` is the tensor's element count; the blocks run contiguously over
/// it, row after row, exactly as the file stores them.
///
/// # Errors
/// `dtype` is not a block format, `numel` is not a whole number of blocks,
/// or `data` is not the byte count the format stores for it.
pub fn dequant_to_f32(data: &[u8], dtype: UnifiedDType, numel: usize) -> Result<Vec<f32>> {
    let Some(format) = quant_format_of(dtype) else {
        bail!("{dtype} is not a block-quantized type, so there is nothing to decode");
    };
    let expected = format
        .storage_bytes(numel)
        .map_err(|e| anyhow!("{dtype}: {e}"))?;
    if data.len() != expected {
        bail!(
            "{dtype}: {} bytes for {numel} elements, expected {expected}",
            data.len()
        );
    }
    let mut out = vec![0.0f32; numel];
    dequant_row_f32(data, &mut out, format);
    Ok(out)
}

/// [`dequant_to_f32`], as little-endian F32 bytes ready for [`super::cast_bytes`].
pub fn dequant_to_f32_bytes(data: &[u8], dtype: UnifiedDType, numel: usize) -> Result<Vec<u8>> {
    let values = dequant_to_f32(data, dtype, numel)?;
    Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::quant::cpu::kernels::lowbit_codec::{encode_pq2_0_block, LOWBIT_BLOCK_SIZE};
    use half::f16;

    /// Every quantized dtype names a boostr format, and no element type does.
    #[test]
    fn every_block_dtype_maps_and_no_element_type_does() {
        for dtype in [
            UnifiedDType::Q4_0,
            UnifiedDType::Q6K,
            UnifiedDType::IQ4XS,
            UnifiedDType::PQ2_0,
            UnifiedDType::PTQ1_0,
        ] {
            assert!(quant_format_of(dtype).is_some(), "{dtype}");
        }
        for dtype in [UnifiedDType::F32, UnifiedDType::BF16, UnifiedDType::I64] {
            assert!(quant_format_of(dtype).is_none(), "{dtype}");
            assert!(dequant_to_f32(&[], dtype, 0).is_err());
        }
    }

    /// One PQ2_0 block decodes to `(code - 1) * d` through boostr's kernel.
    #[test]
    fn a_pq2_0_block_decodes_through_boostr() {
        let codes: [u8; LOWBIT_BLOCK_SIZE] = std::array::from_fn(|i| (i % 4) as u8);
        let block = encode_pq2_0_block(&codes, f16::from_f32(0.5).to_le_bytes());
        let values = dequant_to_f32(&block, UnifiedDType::PQ2_0, LOWBIT_BLOCK_SIZE).unwrap();
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(v, ((i % 4) as f32 - 1.0) * 0.5, "element {i}");
        }
        let bytes = dequant_to_f32_bytes(&block, UnifiedDType::PQ2_0, LOWBIT_BLOCK_SIZE).unwrap();
        assert_eq!(bytes.len(), LOWBIT_BLOCK_SIZE * 4);
    }

    #[test]
    fn a_wrong_byte_count_is_refused() {
        let block = [0u8; 34];
        let err = dequant_to_f32(&block[..33], UnifiedDType::PQ2_0, 128)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 34"), "{err}");
        assert!(dequant_to_f32(&block, UnifiedDType::PQ2_0, 100).is_err());
    }
}
