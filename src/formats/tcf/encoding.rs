//! Which TCF encoding a conversion writes, and how a tensor's bytes and shape
//! reach `tcf-core`.

use anyhow::{bail, Context, Result};
use tcf_core::{NativeEncoding, RawEncoding};

use super::profile::TcfProfile;
use crate::loader::UnifiedDType;

/// The `--tcf-encoding` values, in the order the CLI help lists them.
pub const TCF_ENCODING_NAMES: [&str; 7] = [
    "q4s32t64",
    "q4as32t64",
    "q4as64t64",
    "q6s32t64",
    "q6s16dt64",
    "q4as32dt64",
    "q8s32t64",
];

/// One encoding name with its case, underscores and hyphens removed.
///
/// Two spellings of one encoding are one encoding. Every place that compares
/// encoding names — the parser below, and the sensitivity file's per-encoding
/// records — compares this form, so a file spelling `Q4AS32D_T64` and one
/// spelling `q4as32dt64` name the same rung rather than two.
pub fn normalized_encoding_name(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// The spelling `PROFILES.md` and a sensitivity file use for one encoding.
///
/// A report names encodings, and a ladder breaks a tie between two of them by
/// name, so the name has to be one fixed string rather than whatever a caller
/// happened to read out of a file.
pub fn tcf_encoding_name(encoding: NativeEncoding) -> &'static str {
    match encoding {
        NativeEncoding::Q4S32T64 => "Q4S32_T64",
        NativeEncoding::Q4AS32T64 => "Q4AS32_T64",
        NativeEncoding::Q4AS32DT64 => "Q4AS32D_T64",
        NativeEncoding::Q4AS64T64 => "Q4AS64_T64",
        NativeEncoding::Q6S32T64 => "Q6S32_T64",
        NativeEncoding::Q6S16DT64 => "Q6S16D_T64",
        NativeEncoding::Q8S32T64 => "Q8S32_T64",
        // `NativeEncoding` is `#[non_exhaustive]`: a rung tcf-core adds and
        // compressr has no name for is reported by its identifier rather than
        // by a name invented here.
        _ => "unnamed TCF encoding",
    }
}

/// Parse a `NativeEncoding` name.
///
/// Bit width alone does not name a TCF encoding: symmetry and group size are
/// part of the identifier, and the two 4-bit encodings differ in both. Case,
/// underscores and hyphens are ignored, so `Q4AS64_T64` and `q4as64t64` are
/// the same value.
///
/// # Errors
/// The name is not one of [`TCF_ENCODING_NAMES`].
pub fn parse_tcf_encoding(s: &str) -> Result<NativeEncoding> {
    let normalized = normalized_encoding_name(s);
    match normalized.as_str() {
        "q4s32t64" => Ok(NativeEncoding::Q4S32T64),
        "q4as32t64" => Ok(NativeEncoding::Q4AS32T64),
        "q4as64t64" => Ok(NativeEncoding::Q4AS64T64),
        "q6s32t64" => Ok(NativeEncoding::Q6S32T64),
        "q6s16dt64" => Ok(NativeEncoding::Q6S16DT64),
        "q4as32dt64" => Ok(NativeEncoding::Q4AS32DT64),
        "q8s32t64" => Ok(NativeEncoding::Q8S32T64),
        _ => bail!(
            "Unsupported TCF encoding: {}. Supported: {}",
            s,
            TCF_ENCODING_NAMES.join(", ")
        ),
    }
}

/// What a TCF conversion quantizes into, and what its module says it prefers.
///
/// Both are explicit. `preferred_encoding` has no source of truth in a
/// checkpoint, and TCF reads it to decide which tensors carry a
/// `fallback_reason`, so compressr requires it rather than inventing one.
///
/// `--tcf-encoding` sets one encoding for every tensor. `--tcf-profile` sets
/// `encoding` and `preferred_encoding` to the profile's base and carries the
/// profile, which raises the screened tensors to its bump encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcfOptions {
    /// The encoding a quantizable tensor is written in unless a profile bumps it
    pub encoding: NativeEncoding,
    /// The module's highest-ranked `preferred_encoding`
    pub preferred_encoding: NativeEncoding,
    /// The profile driving a mixed-precision allocation, if one was named
    pub profile: Option<TcfProfile>,
}

impl TcfOptions {
    /// One encoding for every quantizable tensor, as `--tcf-encoding` sets it.
    pub fn uniform(encoding: NativeEncoding, preferred_encoding: NativeEncoding) -> Self {
        Self {
            encoding,
            preferred_encoding,
            profile: None,
        }
    }

    /// A profile's allocation, as `--tcf-profile` sets it.
    ///
    /// The base encoding is also the module preference, so a tensor left at
    /// the base records no fallback reason and a bumped one records
    /// `TaskSensitivity`.
    pub fn from_profile(profile: TcfProfile) -> Self {
        Self {
            encoding: profile.base,
            preferred_encoding: profile.base,
            profile: Some(profile),
        }
    }
}

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

/// Read a float tensor's bytes as f32 values.
///
/// Already-F32 source bytes are read directly rather than run through
/// [`crate::cast::cast_bytes`], which for a same-type cast would only copy
/// `data` before this function copies it again into the `Vec<f32>` it
/// returns.
///
/// # Errors
/// The element type is not a float, or the byte count is not a whole number
/// of elements.
pub fn values_f32(data: &[u8], dtype: UnifiedDType, name: &str) -> Result<Vec<f32>> {
    let owned;
    let bytes: &[u8] = if dtype == UnifiedDType::F32 {
        // `cast_bytes` returns `data.to_vec()` unchanged for a same-type
        // cast, with no length check either side of it; reading `data`
        // directly here matches that behavior exactly, minus the copy.
        data
    } else {
        owned = crate::cast::cast_bytes(data, dtype, UnifiedDType::F32)
            .with_context(|| format!("Failed to read tensor '{name}' as f32"))?;
        &owned
    };
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
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

    /// Every encoding names itself, and that name parses back to it.
    #[test]
    fn a_name_round_trips_through_the_parser() {
        for encoding in [
            NativeEncoding::Q4S32T64,
            NativeEncoding::Q4AS32T64,
            NativeEncoding::Q4AS32DT64,
            NativeEncoding::Q4AS64T64,
            NativeEncoding::Q6S32T64,
            NativeEncoding::Q6S16DT64,
            NativeEncoding::Q8S32T64,
        ] {
            let name = tcf_encoding_name(encoding);
            assert_eq!(
                parse_tcf_encoding(name).expect("parses"),
                encoding,
                "{name}"
            );
        }
    }

    /// Case, underscores and hyphens never separate two spellings of one
    /// encoding.
    #[test]
    fn one_encoding_has_one_normalized_name() {
        assert_eq!(
            normalized_encoding_name("Q4AS32D_T64"),
            normalized_encoding_name("q4as32d-t64")
        );
        assert_ne!(
            normalized_encoding_name("Q4AS32D_T64"),
            normalized_encoding_name("Q4AS32_T64")
        );
    }

    #[test]
    fn parses_every_encoding_name() {
        assert_eq!(
            parse_tcf_encoding("q4s32t64").unwrap(),
            NativeEncoding::Q4S32T64
        );
        assert_eq!(
            parse_tcf_encoding("q4as32t64").unwrap(),
            NativeEncoding::Q4AS32T64
        );
        assert_eq!(
            parse_tcf_encoding("Q4AS64_T64").unwrap(),
            NativeEncoding::Q4AS64T64
        );
        assert_eq!(
            parse_tcf_encoding("q6s32t64").unwrap(),
            NativeEncoding::Q6S32T64
        );
        assert_eq!(
            parse_tcf_encoding("Q6S16D_T64").unwrap(),
            NativeEncoding::Q6S16DT64
        );
        assert_eq!(
            parse_tcf_encoding("Q8S32T64").unwrap(),
            NativeEncoding::Q8S32T64
        );
    }

    /// Bit width alone is not an encoding: "q4" names neither a symmetry nor
    /// a group size, so it is refused rather than resolved to one of the two
    /// 4-bit encodings.
    #[test]
    fn rejects_an_unknown_encoding_name() {
        let err = parse_tcf_encoding("q4").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF encoding: q4"), "{err}");
        assert!(err.contains("q4as64t64"), "{err}");
    }

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

    #[test]
    fn values_come_back_as_f32() {
        let bytes: Vec<u8> = [1.0f32, -2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            values_f32(&bytes, UnifiedDType::F32, "w").unwrap(),
            vec![1.0, -2.5]
        );
    }
}
