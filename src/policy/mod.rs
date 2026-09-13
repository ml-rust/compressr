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

pub mod architecture;
pub mod audio_vae;
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

pub(crate) use architecture::checkpoint_dir;
pub use architecture::{model_name, resolve_architecture};
pub use audio_vae::{
    hf_tokenizer_json, is_audio_vae_tensor, is_voxcpm2, plan_named_mixed_tensor, plan_named_tensor,
    voxcpm2_config_json, AUDIO_VAE_PREFIX, HF_TOKENIZER_JSON_KEY, VOXCPM2_CONFIG_JSON_KEY,
};
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
    /// A mixed-precision allocation keeps this tensor unquantized
    ///
    /// llama.cpp's mixes never round a normalization weight, a
    /// mixture-of-experts router, or the handful of small 2D weights listed in
    /// [`NEVER_QUANTIZED_SUBSTRINGS`], whatever their shape. Only the GGUF mix
    /// path produces this reason; a flat quantization has no role rules.
    MixRole,
    /// An embedded AudioVAE tensor (`vae.*`): the codec is stored dense
    ///
    /// A VoxCPM2 conversion folds the companion `audiovae.pth` into the same
    /// file. Its decoder is verified against F32 fixtures and boostr refuses
    /// to load a block-quantized copy, so every float tensor under the prefix
    /// stays F32 in a flat run and a mix alike — see [`plan_named_tensor`].
    AudioVae,
}

/// The TCF `fallback_reason` a compressr reason maps to.
///
/// TCF requires a reason on every tensor whose encoding differs from its
/// module's highest-ranked `preferred_encoding` (`FORMAT.md` Section 8.6),
/// so the mapping is total. The names differ because compressr's enum predates
/// TCF and describes GGUF blocks.
pub fn tcf_fallback_reason(reason: FallbackReason) -> boostr::tcf::FallbackReason {
    match reason {
        FallbackReason::Rank => boostr::tcf::FallbackReason::RankLt2,
        FallbackReason::RowLength => boostr::tcf::FallbackReason::ShapeIncompatibleWithEncoding,
        FallbackReason::SourceDtype => boostr::tcf::FallbackReason::UnsupportedEncoding,
        // A mix keeps norms, routers and the listed small weights unrounded
        // because of what they are, which is the role reason TCF names.
        FallbackReason::MixRole | FallbackReason::AudioVae => {
            boostr::tcf::FallbackReason::RoleForbidsQuant
        }
    }
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
            | TensorPlan::StoreF32(FallbackReason::MixRole)
            | TensorPlan::StoreF32(FallbackReason::AudioVae) => self.fallback_source += 1,
            TensorPlan::StoreAsIs => self.passthrough += 1,
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Every reason maps to a TCF value: Section 8.6 makes the field
    /// mandatory whenever the encoding differs from the module's preference.
    #[test]
    fn test_every_fallback_reason_maps_to_tcf() {
        for reason in [
            FallbackReason::Rank,
            FallbackReason::RowLength,
            FallbackReason::SourceDtype,
            FallbackReason::MixRole,
            FallbackReason::AudioVae,
        ] {
            assert_ne!(
                tcf_fallback_reason(reason),
                boostr::tcf::FallbackReason::None,
                "{reason:?} must name a reason"
            );
        }
    }
}
