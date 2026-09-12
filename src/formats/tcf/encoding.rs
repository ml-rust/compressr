//! How a tensor's bytes and shape reach `boostr::tcf`.

use anyhow::{bail, Context, Result};
use boostr::tcf::RawEncoding;

use crate::loader::UnifiedDType;

/// The raw encoding a tensor stored unquantized keeps.
///
/// The source element type is preserved rather than widened to F32: TCF names
/// F16 and BF16 directly, so a fallback costs no size. GGUF has no such
/// choice, which is why the GGUF path stores every fallback as F32.
///
/// # Errors
/// TCF v1 defines no raw encoding for this element type.
pub fn raw_encoding_for(dtype: UnifiedDType, name: &str) -> Result<RawEncoding> {
    match dtype {
        UnifiedDType::F32 => Ok(RawEncoding::F32),
        UnifiedDType::F16 => Ok(RawEncoding::F16),
        UnifiedDType::BF16 => Ok(RawEncoding::Bf16),
        UnifiedDType::I8 => Ok(RawEncoding::I8),
        UnifiedDType::I16 => Ok(RawEncoding::I16),
        UnifiedDType::I32 => Ok(RawEncoding::I32),
        UnifiedDType::U8 => Ok(RawEncoding::U8),
        UnifiedDType::U32 => Ok(RawEncoding::U32),
        other => bail!(
            "Tensor '{name}' has dtype {other} which TCF v1 has no raw encoding for. \
             Accepted: F32, F16, BF16, I8, I16, I32, U8, U32."
        ),
    }
}

/// A shape as TCF's eight dimensions, trailing entries zero.
///
/// # Errors
/// The rank is 0 or above 8, or a dimension does not fit `u64`.
pub fn dims_of(shape: &[usize], name: &str) -> Result<[u64; 8]> {
    if shape.is_empty() || shape.len() > 8 {
        bail!(
            "Tensor '{name}' has rank {}, and TCF v1 stores rank 1 through 8.",
            shape.len()
        );
    }
    let mut dims = [0u64; 8];
    for (slot, dim) in dims.iter_mut().zip(shape) {
        *slot = u64::try_from(*dim)
            .with_context(|| format!("Tensor '{name}' has a dimension larger than u64"))?;
    }
    Ok(dims)
}

/// One tensor's shape in the two forms TCF asks for, derived once.
///
/// The record stores the padded eight dimensions; the codec takes exactly
/// `rank` of them and answers `E_INVALID_RANK` for any other count. Both come
/// from one [`dims_of`] call, so the registration pass and the payload pass
/// cannot disagree about a shape and cannot pay for it twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    dims: [u64; 8],
    rank: u32,
}

impl TensorShape {
    /// Derive both forms from a standard-order shape.
    ///
    /// # Errors
    /// Whatever [`dims_of`] rejects, plus a rank that does not fit `u32`.
    pub fn of(shape: &[usize], name: &str) -> Result<Self> {
        let dims = dims_of(shape, name)?;
        let rank = u32::try_from(shape.len())
            .with_context(|| format!("Tensor '{name}' has an unstorable rank"))?;
        Ok(Self { dims, rank })
    }

    /// The padded eight dimensions the record's wire shape carries.
    pub fn dims(&self) -> [u64; 8] {
        self.dims
    }

    /// The rank the record carries.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// EXACTLY `rank` dimensions, which is what the codec accepts. Handing it
    /// the padded eight is `E_INVALID_RANK`.
    pub fn extent(&self) -> &[u64] {
        let extent = usize::try_from(self.rank).unwrap_or(self.dims.len());
        let extent = extent.min(self.dims.len());
        &self.dims[..extent]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_source_element_type_for_a_raw_tensor() {
        assert_eq!(
            raw_encoding_for(UnifiedDType::BF16, "w").unwrap(),
            RawEncoding::Bf16
        );
        assert_eq!(
            raw_encoding_for(UnifiedDType::I32, "ids").unwrap(),
            RawEncoding::I32
        );
        let err = raw_encoding_for(UnifiedDType::I64, "ids")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no raw encoding"), "{err}");
    }

    #[test]
    fn dims_pad_to_eight_and_reject_an_unstorable_rank() {
        assert_eq!(dims_of(&[2, 64], "w").unwrap(), [2, 64, 0, 0, 0, 0, 0, 0]);
        let shape = TensorShape::of(&[2, 64], "w").unwrap();
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.dims(), [2, 64, 0, 0, 0, 0, 0, 0]);
        assert_eq!(shape.extent(), &[2, 64]);
        assert!(dims_of(&[], "w").is_err());
        assert!(dims_of(&[1, 1, 1, 1, 1, 1, 1, 1, 1], "w").is_err());
        assert!(TensorShape::of(&[], "w").is_err());
        assert!(TensorShape::of(&[1, 1, 1, 1, 1, 1, 1, 1, 1], "w").is_err());
    }
}
