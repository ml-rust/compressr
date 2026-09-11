//! Per-tensor conversion policy: what may be quantized, and what the output
//! model is called.
//!
//! # Why a tensor may refuse quantization
//!
//! A block-quantized GGUF tensor packs whole blocks along a ROW. GGUF stores
//! dimensions reversed, so `ne[0]` — the row length — is the LAST dimension of
//! the standard-order shape a loader reports. A row that is not a whole number
//! of blocks cannot be packed at all, and a rank-1 buffer (norm weights,
//! biases) has no rows to pack. llama.cpp keeps both cases as F32; so does
//! compressr, rather than failing the whole conversion.
//!
//! The decision is computed ONCE per tensor and read by both the registration
//! loop (which reserves the byte count in the header) and the write loop (which
//! writes the bytes). If those two disagreed by a single tensor, every
//! subsequent tensor offset in the file would be wrong, with no error raised.

pub mod allocation;
pub mod config_facts;
pub mod gguf_mix;
pub mod ladder;
pub mod ladder_report;
pub mod ladder_rungs;
pub mod mix_allocation;
pub mod mix_assign;
pub mod mix_bytes;
pub mod mix_components;
pub mod mix_ladder;
pub mod mix_ladder_format;
pub mod mix_measured;
pub mod mix_measured_encoding;
pub mod mix_measured_report;
pub mod mix_model;
pub mod mix_promote;
pub mod mix_report;
pub mod mix_rules;
pub mod mix_spend;
pub mod name_examples;
pub mod sensitivity;
pub mod sensitivity_options;
pub mod sensitivity_rank;
pub mod sensitivity_record;
pub mod tensor_role;

pub use allocation::{relative_rms_error, select_bumped, ScreenedTensor};
pub use config_facts::{find_config_usize, LayerCountSource, LAYER_COUNT_KEYS};
pub use gguf_mix::{parse_gguf_mix, GgufMix, GGUF_MIX_NAMES};
pub use ladder::{allocate_ladder, LadderChoice, LadderMenu, LadderOption, LadderOutcome};
pub use ladder_report::LadderReport;
pub use mix_allocation::{ComponentAllocation, MixAllocation};
pub use mix_assign::{assign_mix, assign_mix_reported};
pub use mix_components::{
    partition_components, principal_component, split_component, Component, UNLAYERED_COMPONENT,
};
pub use mix_measured::{assign_mix_measured, MeasuredTensor};
pub use mix_measured_encoding::check_measurement_encoding;
pub use mix_measured_report::MeasuredMixAllocation;
pub use mix_model::{mix_model_facts, MixModel};
pub use mix_report::{plan_format_label, quant_type_label, FormatUsage, MixEntry, MixReport};
pub use mix_rules::{plan_mixed_tensor, step_down_unquantizable, use_more_bits, MixCounters};
pub use name_examples::examples;
pub use sensitivity::{parse_sensitivity, SensitivityBaseline, SensitivityMeasurement};
pub use sensitivity_options::{MeasuredOption, MeasuredOptions};
pub use sensitivity_rank::{
    rank_candidates, select_bumped_measured, BumpCandidate, MeasuredBump, MeasuredRanking,
    RankedCandidate,
};
pub use tensor_role::{classify, is_quantizable_role, TensorRole, NEVER_QUANTIZED_SUBSTRINGS};

use crate::loader::UnifiedDType;
use crate::quantization::QuantType;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// What a conversion does with one tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPlan {
    /// Quantize into this type
    Quantize(QuantType),
    /// Quantization was requested but is not possible: store as F32
    StoreF32(FallbackReason),
    /// Store as the conversion's own dtype, unquantized
    ///
    /// This covers a run with no quantizer, and the integer or boolean buffers
    /// that have no meaningful quantized form (token ids, masks). Those pass
    /// through byte-for-byte rather than being cast up to F32.
    StoreAsIs,
}

/// Why a tensor falls back to F32 instead of being quantized
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// Rank below 2: there is no row to pack blocks along
    Rank,
    /// Row length (the last dimension) is not a multiple of the block size
    RowLength,
    /// The source element type has no quantized form: token ids, masks
    SourceDtype,
    /// The source values contain a NaN or an infinity
    ///
    /// Only the quantizer can report this, so a tensor carrying it was
    /// re-planned while the conversion loop ran. See [`TcfPlan`].
    SourceNonfinite,
    /// A group's scale is not representable at the requested geometry
    ///
    /// Value-dependent like [`FallbackReason::SourceNonfinite`], and decided
    /// at the same point.
    NumericRange,
    /// A mixed-precision allocation keeps this tensor unquantized
    ///
    /// llama.cpp's mixes never round a normalization weight, a
    /// mixture-of-experts router, or the handful of small 2D weights listed in
    /// [`NEVER_QUANTIZED_SUBSTRINGS`], whatever their shape. Only the GGUF mix
    /// path produces this reason; a flat quantization has no role rules.
    MixRole,
}

/// The TCF `fallback_reason` a compressr reason maps to.
///
/// TCF requires a reason on every tensor whose encoding differs from its
/// module's highest-ranked `preferred_encoding` (SPECIFICATION.md Section 8.6),
/// so the mapping is total. The names differ because compressr's enum predates
/// TCF and describes GGUF blocks.
pub fn tcf_fallback_reason(reason: FallbackReason) -> tcf_core::FallbackReason {
    match reason {
        FallbackReason::Rank => tcf_core::FallbackReason::RankLt2,
        FallbackReason::RowLength => tcf_core::FallbackReason::ShapeIncompatibleWithEncoding,
        FallbackReason::SourceDtype => tcf_core::FallbackReason::UnsupportedEncoding,
        FallbackReason::SourceNonfinite => tcf_core::FallbackReason::SourceNonfinite,
        FallbackReason::NumericRange => tcf_core::FallbackReason::NumericRange,
        // A mix keeps norms, routers and the listed small weights unrounded
        // because of what they are, which is the role reason TCF names.
        FallbackReason::MixRole => tcf_core::FallbackReason::RoleForbidsQuant,
    }
}

/// What a TCF conversion does with one tensor, as far as its shape and its
/// element type decide.
///
/// Two of the five [`FallbackReason`] values are invisible here:
/// `SourceNonfinite` and `NumericRange` depend on the values themselves, which
/// no shape check can see. The TCF writer applies this plan first and re-plans
/// a tensor as [`TcfPlan::Raw`] if the quantizer then rejects its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcfPlan {
    /// Quantize into the requested native encoding
    Quantize,
    /// Store verbatim in a raw encoding, for this reason
    Raw(FallbackReason),
}

/// Decide what a TCF conversion does with one tensor, from its shape alone.
///
/// `shape` is in standard order, so the row length is its LAST dimension.
/// `tile` is the execution tile width of the requested encoding — the writer
/// reads it from `tcf-core`, and no tile number is written here.
///
/// A native quantized TCF tensor MUST have rank 2 or more and a row length
/// that is a whole number of tiles (SPECIFICATION.md Section 12). Anything
/// else is stored raw rather than failing the conversion.
pub fn plan_tcf_tensor(shape: &[usize], dtype: UnifiedDType, tile: u16) -> TcfPlan {
    if !dtype.is_float() {
        return TcfPlan::Raw(FallbackReason::SourceDtype);
    }
    if shape.len() < 2 {
        return TcfPlan::Raw(FallbackReason::Rank);
    }
    let row_len = shape[shape.len() - 1];
    if tile == 0 || !row_len.is_multiple_of(usize::from(tile)) {
        return TcfPlan::Raw(FallbackReason::RowLength);
    }
    TcfPlan::Quantize
}

/// Decide what to do with one tensor.
///
/// `shape` is in standard order, so the row length is its LAST dimension.
pub fn plan_tensor(
    shape: &[usize],
    dtype: UnifiedDType,
    quant_type: Option<QuantType>,
) -> TensorPlan {
    let Some(quant_type) = quant_type else {
        return TensorPlan::StoreAsIs;
    };

    if !dtype.is_float() {
        return TensorPlan::StoreAsIs;
    }
    if shape.len() < 2 {
        return TensorPlan::StoreF32(FallbackReason::Rank);
    }

    let row_len = shape[shape.len() - 1];
    if !row_len.is_multiple_of(quant_type.block_size()) {
        return TensorPlan::StoreF32(FallbackReason::RowLength);
    }

    TensorPlan::Quantize(quant_type)
}

/// How many tensors took each plan, for the one-line conversion summary
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanCounts {
    /// Tensors actually quantized
    pub quantized: usize,
    /// Float tensors stored as F32 because their rank is below 2
    pub fallback_rank: usize,
    /// Float tensors stored as F32 because their row length is not block-aligned
    pub fallback_row_length: usize,
    /// Tensors stored unquantized because of their element type or their values
    ///
    /// Only the TCF path reaches this: the GGUF path filters non-float tensors
    /// into `passthrough` and its kernels never reject a value.
    pub fallback_source: usize,
    /// Non-float tensors passed through untouched
    pub passthrough: usize,
}

impl PlanCounts {
    /// Record one tensor's plan
    pub fn record(&mut self, plan: TensorPlan) {
        match plan {
            TensorPlan::Quantize(_) => self.quantized += 1,
            TensorPlan::StoreF32(FallbackReason::Rank) => self.fallback_rank += 1,
            TensorPlan::StoreF32(FallbackReason::RowLength) => self.fallback_row_length += 1,
            TensorPlan::StoreF32(FallbackReason::SourceDtype)
            | TensorPlan::StoreF32(FallbackReason::SourceNonfinite)
            | TensorPlan::StoreF32(FallbackReason::NumericRange)
            | TensorPlan::StoreF32(FallbackReason::MixRole) => self.fallback_source += 1,
            TensorPlan::StoreAsIs => self.passthrough += 1,
        }
    }

    /// Record one TCF plan, once its value-dependent form is known.
    pub fn record_tcf(&mut self, plan: TcfPlan) {
        match plan {
            TcfPlan::Quantize => self.quantized += 1,
            TcfPlan::Raw(FallbackReason::Rank) => self.fallback_rank += 1,
            TcfPlan::Raw(FallbackReason::RowLength) => self.fallback_row_length += 1,
            TcfPlan::Raw(FallbackReason::SourceDtype)
            | TcfPlan::Raw(FallbackReason::SourceNonfinite)
            | TcfPlan::Raw(FallbackReason::NumericRange)
            | TcfPlan::Raw(FallbackReason::MixRole) => self.fallback_source += 1,
        }
    }
}

impl std::fmt::Display for PlanCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tensors quantized, {} stored unquantized ({} rank < 2, {} row length not \
             block-aligned, {} source dtype or values), {} non-float passed through",
            self.quantized,
            self.fallback_rank + self.fallback_row_length + self.fallback_source,
            self.fallback_rank,
            self.fallback_row_length,
            self.fallback_source,
            self.passthrough
        )
    }
}

/// Resolve the value written to `general.architecture`.
///
/// GGUF is self-describing: every consumer keys its tensor-name mapping off
/// this string, so a guess here silently mislabels the model. Precedence:
///
/// 1. `arch_override` — what the user passed as `--arch`.
/// 2. The architecture field of the checkpoint's `config.json`. Three spellings
///    are read, in order: `model_type` (the HuggingFace convention), then
///    `architecture` (a bare string, as VoxCPM2 writes it), then the first entry
///    of `architectures` (HuggingFace's class-name list). They are alternatives,
///    not a merge — the first one present wins.
/// 3. `metadata_arch` — set only when the input already carried it (GGUF in).
///
/// A safetensors input has no architecture in its header, so when none of the
/// three is available this errors rather than inventing one.
pub fn resolve_architecture(
    arch_override: Option<&str>,
    metadata_arch: Option<&str>,
    input_path: &Path,
) -> Result<String> {
    if let Some(arch) = arch_override.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(arch.to_string());
    }
    if let Some(arch) = read_config_architecture(input_path)? {
        return Ok(arch);
    }
    if let Some(arch) = metadata_arch.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(arch.to_string());
    }

    bail!(
        "Cannot determine the model architecture for '{}': the input carries none and no \
         config.json naming it (\"model_type\", \"architecture\" or \"architectures\") was found \
         beside it. Pass --arch <NAME> to set \
         GGUF's general.architecture explicitly.",
        input_path.display()
    )
}

/// Read the architecture from the `config.json` of the input checkpoint.
///
/// Returns `Ok(None)` when there is no `config.json` or it names no
/// architecture under any of the three spellings; a `config.json` that is
/// present but unreadable or malformed is an error, because silently ignoring
/// it would fabricate an architecture again.
fn read_config_architecture(input_path: &Path) -> Result<Option<String>> {
    let Some(dir) = checkpoint_dir(input_path) else {
        return Ok(None);
    };
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", config_path.display()))?;

    let from_key = |key: &str| value.get(key).and_then(serde_json::Value::as_str);
    let from_list = || {
        value
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .and_then(|list| list.first())
            .and_then(serde_json::Value::as_str)
    };

    Ok(from_key("model_type")
        .or_else(|| from_key("architecture"))
        .or_else(from_list)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string))
}

/// The checkpoint directory an input path belongs to.
///
/// An input names either the directory itself or one weights file inside it.
pub(crate) fn checkpoint_dir(input_path: &Path) -> Option<PathBuf> {
    if input_path.is_dir() {
        return Some(input_path.to_path_buf());
    }
    match input_path.parent() {
        // A bare file name sits in the current directory.
        Some(parent) if parent.as_os_str().is_empty() => Some(PathBuf::from(".")),
        Some(parent) => Some(parent.to_path_buf()),
        None => None,
    }
}

/// The model name for `general.name`: the checkpoint directory's own name.
///
/// The architecture is not a name — `llama` names a family, not this model — so
/// when the directory name is unavailable no name is written at all.
pub fn model_name(input_path: &Path) -> Option<String> {
    let dir = checkpoint_dir(input_path)?;
    // A relative path such as "." or "model/." carries no usable file name of
    // its own, so resolve it first; an unresolvable path keeps its literal form.
    let dir = match dir.canonicalize() {
        Ok(resolved) => resolved,
        Err(_) => dir,
    };
    dir.file_name().and_then(|n| n.to_str()).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_rank_1_tensor_falls_back() {
        assert_eq!(
            plan_tensor(&[64], UnifiedDType::F32, Some(QuantType::Q4K)),
            TensorPlan::StoreF32(FallbackReason::Rank)
        );
    }

    #[test]
    fn test_row_length_is_the_last_dimension() {
        // [1024, 64]: 1024 is divisible by 256, but the ROW is 64 elements.
        assert_eq!(
            plan_tensor(&[1024, 64], UnifiedDType::F32, Some(QuantType::Q4K)),
            TensorPlan::StoreF32(FallbackReason::RowLength)
        );
        // [73448, 2048]: the row is 2048 elements, a whole number of blocks.
        assert_eq!(
            plan_tensor(&[73448, 2048], UnifiedDType::BF16, Some(QuantType::Q4K)),
            TensorPlan::Quantize(QuantType::Q4K)
        );
    }

    #[test]
    fn test_no_quantizer_stores_as_is() {
        assert_eq!(
            plan_tensor(&[73448, 2048], UnifiedDType::F32, None),
            TensorPlan::StoreAsIs
        );
    }

    #[test]
    fn test_non_float_tensor_is_never_quantized() {
        assert_eq!(
            plan_tensor(&[8, 256], UnifiedDType::I64, Some(QuantType::Q4K)),
            TensorPlan::StoreAsIs
        );
    }

    #[test]
    fn test_plan_counts_summary() {
        let mut counts = PlanCounts::default();
        counts.record(TensorPlan::Quantize(QuantType::Q4K));
        counts.record(TensorPlan::StoreF32(FallbackReason::Rank));
        counts.record(TensorPlan::StoreF32(FallbackReason::RowLength));
        counts.record(TensorPlan::StoreAsIs);

        assert_eq!(
            counts,
            PlanCounts {
                quantized: 1,
                fallback_rank: 1,
                fallback_row_length: 1,
                fallback_source: 0,
                passthrough: 1,
            }
        );
        assert!(counts.to_string().contains("1 tensors quantized"));
    }

    /// A native quantized TCF tensor needs a row to tile along, so rank 1
    /// falls back with the reason TCF spells `RANK_LT_2`.
    #[test]
    fn test_tcf_rank_1_tensor_is_raw() {
        let plan = plan_tcf_tensor(&[64], UnifiedDType::F32, 64);
        assert_eq!(plan, TcfPlan::Raw(FallbackReason::Rank));
        assert_eq!(
            tcf_fallback_reason(FallbackReason::Rank),
            tcf_core::FallbackReason::RankLt2
        );
    }

    /// The row length is the LAST dimension. 96 is not a whole number of
    /// 64-element tiles, so the tensor cannot be packed at all.
    #[test]
    fn test_tcf_row_not_tile_aligned_is_raw() {
        let plan = plan_tcf_tensor(&[1024, 96], UnifiedDType::BF16, 64);
        assert_eq!(plan, TcfPlan::Raw(FallbackReason::RowLength));
        assert_eq!(
            tcf_fallback_reason(FallbackReason::RowLength),
            tcf_core::FallbackReason::ShapeIncompatibleWithEncoding
        );
        // The leading dimension being tile-aligned decides nothing.
        assert_eq!(
            plan_tcf_tensor(&[64, 128], UnifiedDType::BF16, 64),
            TcfPlan::Quantize
        );
    }

    #[test]
    fn test_tcf_non_float_tensor_is_raw() {
        assert_eq!(
            plan_tcf_tensor(&[8, 256], UnifiedDType::I32, 64),
            TcfPlan::Raw(FallbackReason::SourceDtype)
        );
    }

    /// Every reason maps to a TCF value: Section 8.6 makes the field
    /// mandatory whenever the encoding differs from the module's preference.
    #[test]
    fn test_every_fallback_reason_maps_to_tcf() {
        for reason in [
            FallbackReason::Rank,
            FallbackReason::RowLength,
            FallbackReason::SourceDtype,
            FallbackReason::SourceNonfinite,
            FallbackReason::NumericRange,
            FallbackReason::MixRole,
        ] {
            assert_ne!(
                tcf_fallback_reason(reason),
                tcf_core::FallbackReason::None,
                "{reason:?} must name a reason"
            );
        }
    }

    #[test]
    fn test_architecture_comes_from_config_model_type() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "voxcpm2", "hidden_size": 2048}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "voxcpm2");
    }

    /// VoxCPM2's own `config.json` spells it `architecture`, a bare string,
    /// with no `model_type` anywhere. Reading only the HuggingFace spelling
    /// makes that checkpoint unconvertible without `--arch`.
    #[test]
    fn test_architecture_comes_from_config_architecture_key() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"architecture": "voxcpm2", "patch_size": 4}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "voxcpm2");
    }

    /// HuggingFace's `architectures` list holds class names; its first entry is
    /// the last resort before erroring.
    #[test]
    fn test_architecture_falls_back_to_the_architectures_list() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["LlamaForCausalLM"]}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "LlamaForCausalLM");
    }

    /// The three spellings are alternatives in a fixed order, not a merge.
    #[test]
    fn test_model_type_wins_over_the_other_two_spellings() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "first", "architecture": "second", "architectures": ["third"]}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "first");
    }

    #[test]
    fn test_arch_override_wins_over_config() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "voxcpm2"}"#,
        )
        .unwrap();

        let arch = resolve_architecture(Some("qwen2"), None, dir.path()).unwrap();
        assert_eq!(arch, "qwen2");
    }

    #[test]
    fn test_missing_architecture_is_an_error() {
        let dir = TempDir::new().unwrap();
        let err = resolve_architecture(None, None, dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("--arch"), "{}", err);
    }

    #[test]
    fn test_input_metadata_architecture_is_used_when_present() {
        let dir = TempDir::new().unwrap();
        let arch = resolve_architecture(None, Some("llama"), dir.path()).unwrap();
        assert_eq!(arch, "llama");
    }

    #[test]
    fn test_config_json_beside_a_weights_file_is_found() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "mamba2"}"#,
        )
        .unwrap();
        let weights = dir.path().join("model.safetensors");
        fs::write(&weights, b"not a real checkpoint").unwrap();

        assert_eq!(
            resolve_architecture(None, None, &weights).unwrap(),
            "mamba2"
        );
    }

    #[test]
    fn test_model_name_is_the_checkpoint_directory_name() {
        let dir = TempDir::new().unwrap();
        let checkpoint = dir.path().join("voxcpm2-0.5b");
        fs::create_dir(&checkpoint).unwrap();

        assert_eq!(model_name(&checkpoint).as_deref(), Some("voxcpm2-0.5b"));
    }
}
