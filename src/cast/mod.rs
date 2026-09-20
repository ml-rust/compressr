//! Element-type casting for float tensors
//!
//! A dtype cast changes the *element type* of a tensor (f32 -> bf16). It is not
//! quantization: there are no scales, no blocks, and no packed representation.
//! For that reason the cast target is modelled as [`UnifiedDType`], never as a
//! `QuantizationMethod` variant.
//!
//! All conversions to and from 16-bit floats use round-to-nearest-even, supplied
//! by the `half` crate. Truncating the low 16 bits of an f32 is *not* a bf16
//! cast: it loses up to a full ulp and biases every result toward zero.

pub mod dequant;

pub use dequant::{dequant_to_f32, dequant_to_f32_bytes, quant_format_of};

use crate::loader::UnifiedDType;
use anyhow::{bail, Result};
use half::{bf16, f16};

/// Accepted spellings for the `--dtype` flag, in the order they are reported.
pub const ACCEPTED_DTYPE_NAMES: &[&str] = &[
    "f32", "fp32", "float32", "float", "f16", "fp16", "float16", "half", "bf16", "bfloat16",
    "brain16",
];

/// Parse a user-supplied cast target, case-insensitively.
///
/// # Errors
///
/// Returns an error listing [`ACCEPTED_DTYPE_NAMES`] if `s` is not a recognised
/// spelling of a supported float type.
pub fn parse_target_dtype(s: &str) -> Result<UnifiedDType> {
    match s.trim().to_lowercase().as_str() {
        "f32" | "fp32" | "float32" | "float" => Ok(UnifiedDType::F32),
        "f16" | "fp16" | "float16" | "half" => Ok(UnifiedDType::F16),
        "bf16" | "bfloat16" | "brain16" => Ok(UnifiedDType::BF16),
        _ => bail!(
            "Unsupported dtype: {}. Accepted: {}",
            s,
            ACCEPTED_DTYPE_NAMES.join(", ")
        ),
    }
}

/// Name a dtype the way a HuggingFace `config.json` `torch_dtype` field does.
pub fn torch_dtype_name(dtype: UnifiedDType) -> Option<&'static str> {
    match dtype {
        UnifiedDType::F32 => Some("float32"),
        UnifiedDType::F16 => Some("float16"),
        UnifiedDType::BF16 => Some("bfloat16"),
        UnifiedDType::F64 => Some("float64"),
        _ => None,
    }
}

/// Decode one float element of `src` at byte offset `off` into an `f32`.
///
/// `src` is little-endian, which is what both SafeTensors and GGUF mandate.
fn read_f32(src: &[u8], off: usize, dtype: UnifiedDType) -> f32 {
    match dtype {
        UnifiedDType::F32 => {
            f32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
        }
        UnifiedDType::F16 => f16::from_le_bytes([src[off], src[off + 1]]).to_f32(),
        UnifiedDType::BF16 => bf16::from_le_bytes([src[off], src[off + 1]]).to_f32(),
        UnifiedDType::F64 => f64::from_le_bytes([
            src[off],
            src[off + 1],
            src[off + 2],
            src[off + 3],
            src[off + 4],
            src[off + 5],
            src[off + 6],
            src[off + 7],
        ]) as f32,
        // `cast_bytes` rejects every other source dtype before calling this.
        _ => f32::NAN,
    }
}

/// Cast raw little-endian float bytes from one element type to another.
///
/// Returns `data` unchanged when `src` and `dst` are the same type.
///
/// # Errors
///
/// - `src` is not a float type. Non-float tensors must pass through untouched,
///   so the caller is expected to filter them with [`UnifiedDType::is_float`].
/// - `dst` is not one of F32, F16 or BF16.
/// - `data.len()` is not a whole number of `src` elements.
pub fn cast_bytes(data: &[u8], src: UnifiedDType, dst: UnifiedDType) -> Result<Vec<u8>> {
    if !src.is_float() {
        bail!("Cannot cast non-float tensor data of type {}", src);
    }
    if !matches!(
        dst,
        UnifiedDType::F32 | UnifiedDType::F16 | UnifiedDType::BF16
    ) {
        bail!(
            "Unsupported cast target: {}. Accepted: {}",
            dst,
            ACCEPTED_DTYPE_NAMES.join(", ")
        );
    }
    if src == dst {
        return Ok(data.to_vec());
    }

    let src_size = src
        .element_size()
        .ok_or_else(|| anyhow::anyhow!("Source dtype {} has no fixed element size", src))?;
    if !data.len().is_multiple_of(src_size) {
        bail!(
            "Tensor data of {} bytes is not a whole number of {} elements ({} bytes each)",
            data.len(),
            src,
            src_size
        );
    }

    let numel = data.len() / src_size;
    let dst_size = dst
        .element_size()
        .ok_or_else(|| anyhow::anyhow!("Target dtype {} has no fixed element size", dst))?;
    let mut out = Vec::with_capacity(numel * dst_size);

    for i in 0..numel {
        let v = read_f32(data, i * src_size, src);
        match dst {
            UnifiedDType::F32 => out.extend_from_slice(&v.to_le_bytes()),
            // `half` rounds to nearest, ties to even.
            UnifiedDType::F16 => out.extend_from_slice(&f16::from_f32(v).to_le_bytes()),
            UnifiedDType::BF16 => out.extend_from_slice(&bf16::from_f32(v).to_le_bytes()),
            _ => bail!("Unsupported cast target: {}", dst),
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the little-endian byte buffer for a slice of f32 values.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Read a byte buffer back as little-endian 16-bit patterns.
    fn u16le(bytes: &[u8]) -> Vec<u16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn test_parse_target_dtype_aliases() {
        assert_eq!(parse_target_dtype("f32").unwrap(), UnifiedDType::F32);
        assert_eq!(parse_target_dtype("FP32").unwrap(), UnifiedDType::F32);
        assert_eq!(parse_target_dtype("Float32").unwrap(), UnifiedDType::F32);
        assert_eq!(parse_target_dtype("f16").unwrap(), UnifiedDType::F16);
        assert_eq!(parse_target_dtype("float16").unwrap(), UnifiedDType::F16);
        assert_eq!(parse_target_dtype("BF16").unwrap(), UnifiedDType::BF16);
        assert_eq!(parse_target_dtype("bfloat16").unwrap(), UnifiedDType::BF16);
    }

    #[test]
    fn test_parse_target_dtype_unknown_lists_accepted_names() {
        let err = parse_target_dtype("q4_k").unwrap_err().to_string();
        assert_eq!(
            err,
            "Unsupported dtype: q4_k. Accepted: f32, fp32, float32, float, f16, fp16, float16, half, bf16, bfloat16, brain16"
        );
    }

    #[test]
    fn test_parse_target_dtype_rejects_int_targets() {
        assert!(parse_target_dtype("i8").is_err());
        assert!(parse_target_dtype("u8").is_err());
    }

    /// The load-bearing test: proves round-to-nearest-even, not truncation.
    ///
    /// `0x3F80_C000` = 1.005859375. Its low 16 bits are `0xC000`, above the
    /// halfway point, so round-to-nearest-even carries into the kept mantissa
    /// and yields `0x3F81` (1.0078125). Truncating the low 16 bits would yield
    /// `0x3F80` (1.0) instead. The two answers differ, so this value alone
    /// identifies which algorithm is implemented.
    #[test]
    fn test_f32_to_bf16_rounds_to_nearest_not_truncated() {
        let v = f32::from_bits(0x3F80_C000);
        assert_eq!(v, 1.005_859_4_f32);

        let out = cast_bytes(&f32_bytes(&[v]), UnifiedDType::F32, UnifiedDType::BF16).unwrap();
        let bits = u16le(&out);

        assert_eq!(bits, vec![0x3F81]);
        assert_ne!(bits[0], 0x3F80, "truncation would have produced 0x3F80");
        assert_eq!(bf16::from_bits(bits[0]).to_f32(), 1.0078125_f32);
    }

    /// Exact tie: `0x3F81_8000` sits halfway between bf16 `0x3F81` and `0x3F82`.
    /// Ties-to-even picks the even mantissa, `0x3F82`. Truncation gives `0x3F81`.
    #[test]
    fn test_f32_to_bf16_ties_to_even() {
        let v = f32::from_bits(0x3F81_8000);
        assert_eq!(v, 1.011_718_8_f32);

        let out = cast_bytes(&f32_bytes(&[v]), UnifiedDType::F32, UnifiedDType::BF16).unwrap();
        assert_eq!(u16le(&out), vec![0x3F82]);
    }

    /// A known vector with exact expected bf16 bit patterns.
    #[test]
    fn test_f32_to_bf16_known_vector() {
        let values = [
            0.0_f32,
            -0.0_f32,
            1.0_f32,
            -2.0_f32,
            0.5_f32,
            f32::from_bits(0x3F80_C000), // rounds up
            f32::from_bits(0x3F81_8000), // ties to even
            f32::from_bits(0x3F80_4000), // below halfway, rounds down
        ];

        let out = cast_bytes(&f32_bytes(&values), UnifiedDType::F32, UnifiedDType::BF16).unwrap();

        assert_eq!(out.len(), values.len() * 2);
        assert_eq!(
            u16le(&out),
            vec![0x0000, 0x8000, 0x3F80, 0xC000, 0x3F00, 0x3F81, 0x3F82, 0x3F80]
        );
    }

    /// bf16 -> f32 is exact: every bf16 bit pattern is an f32 with zeroed low bits.
    #[test]
    fn test_bf16_to_f32_is_lossless() {
        let originals: Vec<u16> = vec![0x0000, 0x3F80, 0xBF80, 0x3F81, 0x4049, 0x7F7F];
        let bytes: Vec<u8> = originals.iter().flat_map(|b| b.to_le_bytes()).collect();

        let widened = cast_bytes(&bytes, UnifiedDType::BF16, UnifiedDType::F32).unwrap();
        assert_eq!(widened.len(), originals.len() * 4);

        // Each f32 is the bf16 pattern in the high half, zeros in the low half.
        for (i, &b) in originals.iter().enumerate() {
            let bits = u32::from_le_bytes([
                widened[i * 4],
                widened[i * 4 + 1],
                widened[i * 4 + 2],
                widened[i * 4 + 3],
            ]);
            assert_eq!(bits, (b as u32) << 16);
        }

        // Narrowing straight back recovers the original bit patterns exactly.
        let narrowed = cast_bytes(&widened, UnifiedDType::F32, UnifiedDType::BF16).unwrap();
        assert_eq!(u16le(&narrowed), originals);
    }

    #[test]
    fn test_f32_to_f16_rounds_to_nearest_even() {
        // 0x3C01 is the f16 above 1.0; 1.0 + 2^-11 sits exactly halfway, so
        // ties-to-even keeps 0x3C00.
        let out = cast_bytes(
            &f32_bytes(&[1.0 + 2f32.powi(-11)]),
            UnifiedDType::F32,
            UnifiedDType::F16,
        )
        .unwrap();
        assert_eq!(u16le(&out), vec![0x3C00]);

        // Just above halfway rounds up to 0x3C01.
        let out = cast_bytes(
            &f32_bytes(&[1.0 + 2f32.powi(-11) + 2f32.powi(-20)]),
            UnifiedDType::F32,
            UnifiedDType::F16,
        )
        .unwrap();
        assert_eq!(u16le(&out), vec![0x3C01]);
    }

    #[test]
    fn test_cast_halves_the_byte_count() {
        let data = f32_bytes(&[1.0; 1024]);
        assert_eq!(data.len(), 4096);

        let out = cast_bytes(&data, UnifiedDType::F32, UnifiedDType::BF16).unwrap();
        assert_eq!(out.len(), 2048);
    }

    #[test]
    fn test_cast_same_dtype_is_identity() {
        let data = f32_bytes(&[1.5, -2.25, 3.75]);
        let out = cast_bytes(&data, UnifiedDType::F32, UnifiedDType::F32).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_cast_rejects_non_float_source() {
        let err = cast_bytes(&[1, 2, 3, 4], UnifiedDType::I32, UnifiedDType::BF16)
            .unwrap_err()
            .to_string();
        assert_eq!(err, "Cannot cast non-float tensor data of type I32");
    }

    #[test]
    fn test_cast_rejects_ragged_input() {
        let err = cast_bytes(&[0u8; 6], UnifiedDType::F32, UnifiedDType::BF16)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "Tensor data of 6 bytes is not a whole number of F32 elements (4 bytes each)"
        );
    }

    #[test]
    fn test_torch_dtype_name() {
        assert_eq!(torch_dtype_name(UnifiedDType::BF16), Some("bfloat16"));
        assert_eq!(torch_dtype_name(UnifiedDType::F16), Some("float16"));
        assert_eq!(torch_dtype_name(UnifiedDType::F32), Some("float32"));
        assert_eq!(torch_dtype_name(UnifiedDType::Q4K), None);
    }
}
