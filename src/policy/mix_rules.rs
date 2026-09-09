//! The per-tensor format table of llama.cpp's `_S` and `_M` mixes.
//!
//! This is `llama_tensor_get_type` from llama.cpp's `src/llama-quant.cpp`,
//! restricted to the four mixes compressr has every kernel for. Every upstream
//! branch guarded by an ftype none of those four equals leaves the base format
//! in place, so it is absent here rather than present and dead.
//!
//! # Where this is an approximation, precisely
//!
//! - **The 70B attention-V bump** reproduces the llama row of upstream's
//!   hparams table and no other architecture's. See
//!   [`super::mix_model::MixModel`].
//! - **The unquantizable-row step-down** implements upstream's `Q6_K` to
//!   `Q8_0` step, which boostr writes. Its `Q4_K` to `Q5_0` and `Q5_K` to
//!   `Q5_1` steps have no writer here, so those tensors fall back to F32 —
//!   what compressr does for an unquantizable row everywhere else.
//! - **The imatrix.** Every upstream rule guarded by `qs.has_imatrix` takes
//!   its no-imatrix branch, because compressr computes no importance matrix.
//!   None of the four mixes here reads that flag.
//! - **A model with more than one transformer stack.** `i_layer` and
//!   `n_layer` below describe ONE stack, which is all upstream supports.
//!   [`super::mix_assign`] applies these rules once per stack, and
//!   [`super::mix_allocation`] states in the conversion report that the result
//!   is compressr's approximation rather than upstream's allocation.

use crate::loader::{extract_layer_index, UnifiedDType};
use crate::quantization::QuantType;

use super::gguf_mix::GgufMix;
use super::mix_model::MixModel;
use super::tensor_role::TensorRole;
use super::{FallbackReason, TensorPlan};

/// The running counters upstream's rules read instead of a layer number.
///
/// llama.cpp holds its weights in a map ordered by transformer block and then
/// by name, and advances these as it walks that order. The caller reproduces
/// the walk; this holds its state.
#[derive(Debug, Clone, Copy, Default)]
pub struct MixCounters {
    /// Attention-V tensors seen so far
    pub i_attention_wv: i64,
    /// Attention-V-like tensors in the model, the denominator of its rule
    pub n_attention_wv: i64,
    /// FFN-down tensors seen so far
    pub i_ffn_down: i64,
    /// Blocks in the stack being walked, the denominator of the FFN-down rules
    ///
    /// Upstream has one stack and reads its block count here. compressr walks
    /// one stack at a time, so this is that stack's own block count.
    pub n_ffn_down: i64,
    /// Whether the model has an output weight distinct from its embedding
    pub has_output: bool,
}

/// llama.cpp's `use_more_bits`: which layers of a run get the richer format.
///
/// The first eighth, the last eighth, and every third layer of the middle.
/// The arithmetic is signed because `i_layer - n_layers/8` is negative for the
/// leading layers, and a negative remainder is never 2.
pub fn use_more_bits(i_layer: i64, n_layers: i64) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

/// The format one tensor takes, before the unquantizable-row step-down.
///
/// `counters` is advanced by the rules that read it, so this must be called
/// once per tensor and in upstream's walk order.
pub fn tensor_type(
    mix: GgufMix,
    model: &MixModel,
    name: &str,
    role: TensorRole,
    shape: &[usize],
    counters: &mut MixCounters,
) -> QuantType {
    let base = mix.base();
    // GGUF's `ne[0]` is the row length, which is the LAST dimension of the
    // standard-order shape a loader reports.
    let row_len = shape.last().copied().unwrap_or(0);

    // A model that shares one tensor between the embedding table and the
    // un-embedding quantizes it with the OUTPUT rule.
    if role == TensorRole::Output || (!counters.has_output && role == TensorRole::TokenEmbd) {
        if model.is_falcon() || !row_len.is_multiple_of(base.block_size()) {
            return QuantType::Q8_0;
        }
        return QuantType::Q6K;
    }

    match role {
        TensorRole::AttnV => attention_v_type(mix, model, counters),
        TensorRole::AttnK => {
            if model.n_expert == 8 {
                QuantType::Q8_0
            } else {
                base
            }
        }
        TensorRole::FfnDown => ffn_down_type(mix, model, name, counters),
        TensorRole::AttnOutput => {
            if !model.is_falcon()
                && model.n_expert == 8
                && matches!(mix, GgufMix::Q4KS | GgufMix::Q4KM)
            {
                QuantType::Q5K
            } else {
                base
            }
        }
        TensorRole::AttnQkv => match mix {
            GgufMix::Q4KM => QuantType::Q5K,
            GgufMix::Q5KM => QuantType::Q6K,
            _ => base,
        },
        _ => base,
    }
}

/// The attention-V rule, which advances its own counter.
fn attention_v_type(mix: GgufMix, model: &MixModel, counters: &mut MixCounters) -> QuantType {
    let mut new_type = mix.base();
    if matches!(mix, GgufMix::Q4KM | GgufMix::Q5KM)
        && use_more_bits(counters.i_attention_wv, counters.n_attention_wv)
    {
        new_type = QuantType::Q6K;
    } else if mix == GgufMix::Q4KS && counters.i_attention_wv < 4 {
        new_type = QuantType::Q5K;
    }
    // Eight heads share one attention-V tensor in the 70B shape, so it is an
    // eighth the size of attention-Q and cheap to keep richer.
    if model.is_70b && new_type == QuantType::Q4K {
        new_type = QuantType::Q5K;
    }
    if model.n_expert == 8 {
        new_type = QuantType::Q8_0;
    }
    counters.i_attention_wv += 1;
    new_type
}

/// The FFN-down rule, which advances its own counter.
fn ffn_down_type(
    mix: GgufMix,
    model: &MixModel,
    name: &str,
    counters: &mut MixCounters,
) -> QuantType {
    // With more than one expert the FFN tensors are not one per layer, so
    // upstream reads the layer out of the name instead of counting.
    let i_layer = if model.n_expert > 1 {
        extract_layer_index(name).map_or(counters.i_ffn_down, |v| v as i64)
    } else {
        counters.i_ffn_down
    };
    let n_layer = counters.n_ffn_down;

    let new_type = match mix {
        GgufMix::Q4KM if model.is_falcon() => {
            if i_layer < n_layer / 16 {
                QuantType::Q6K
            } else if use_more_bits(i_layer, n_layer) {
                QuantType::Q5K
            } else {
                QuantType::Q4K
            }
        }
        GgufMix::Q4KM | GgufMix::Q5KM if use_more_bits(i_layer, n_layer) => QuantType::Q6K,
        GgufMix::Q4KS if !model.is_falcon() && i_layer < n_layer / 8 => QuantType::Q5K,
        _ => mix.base(),
    };
    counters.i_ffn_down += 1;
    new_type
}

/// Upstream's `convert_incompatible_tensor`, restricted to writable formats.
///
/// A row that is not a whole number of super-blocks cannot be packed as a
/// K-quant at all. `Q6_K` steps down to `Q8_0`, which boostr writes; the other
/// two steps have no writer, so the type is returned unchanged and the
/// caller's block-alignment check stores the tensor as F32.
pub fn step_down_unquantizable(new_type: QuantType, shape: &[usize]) -> QuantType {
    let row_len = shape.last().copied().unwrap_or(0);
    if row_len.is_multiple_of(new_type.block_size()) {
        return new_type;
    }
    match new_type {
        QuantType::Q6K if row_len.is_multiple_of(QuantType::Q8_0.block_size()) => QuantType::Q8_0,
        other => other,
    }
}

/// Decide what to do with one tensor of a mixed-precision GGUF conversion.
///
/// `assigned` is the format [`super::mix_assign::assign_mix`] gave this tensor,
/// or `None` when the mix's role rules keep it unquantized. The shape and
/// dtype checks that follow are the same ones [`super::plan_tensor`] applies,
/// so a flat and a mixed run fall back for the same reasons.
pub fn plan_mixed_tensor(
    shape: &[usize],
    dtype: UnifiedDType,
    assigned: Option<QuantType>,
) -> TensorPlan {
    if !dtype.is_float() {
        return TensorPlan::StoreAsIs;
    }
    let Some(quant_type) = assigned else {
        // Rank names the more specific reason where it applies, so a bias and
        // a router are not reported as the same thing.
        return TensorPlan::StoreF32(if shape.len() < 2 {
            FallbackReason::Rank
        } else {
            FallbackReason::MixRole
        });
    };
    if shape.len() < 2 {
        return TensorPlan::StoreF32(FallbackReason::Rank);
    }
    let row_len = shape[shape.len() - 1];
    if !row_len.is_multiple_of(quant_type.block_size()) {
        return TensorPlan::StoreF32(FallbackReason::RowLength);
    }
    TensorPlan::Quantize(quant_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tensor the mix keeps unquantized reports WHY, so the conversion
    /// summary does not report a router and a bias as the same thing.
    #[test]
    fn an_unassigned_tensor_names_its_reason() {
        assert_eq!(
            plan_mixed_tensor(&[8, 4096], UnifiedDType::F32, None),
            TensorPlan::StoreF32(FallbackReason::MixRole)
        );
        assert_eq!(
            plan_mixed_tensor(&[4096], UnifiedDType::F32, None),
            TensorPlan::StoreF32(FallbackReason::Rank)
        );
        assert_eq!(
            plan_mixed_tensor(&[8, 4096], UnifiedDType::I64, None),
            TensorPlan::StoreAsIs
        );
    }

    /// An assigned format still has to fit the row, and a row that is not a
    /// whole number of blocks falls back exactly as a flat run's does.
    #[test]
    fn an_unaligned_row_falls_back_to_f32() {
        assert_eq!(
            plan_mixed_tensor(&[4096, 96], UnifiedDType::F32, Some(QuantType::Q4K)),
            TensorPlan::StoreF32(FallbackReason::RowLength)
        );
        assert_eq!(
            plan_mixed_tensor(&[4096, 512], UnifiedDType::BF16, Some(QuantType::Q6K)),
            TensorPlan::Quantize(QuantType::Q6K)
        );
    }

    /// Layer 0 is in the first eighth, layer 31 in the last, and layer 6 is
    /// the third of the middle run.
    #[test]
    fn use_more_bits_matches_the_upstream_formula() {
        assert!(use_more_bits(0, 32));
        assert!(use_more_bits(3, 32));
        assert!(use_more_bits(31, 32));
        assert!(!use_more_bits(5, 32));
        assert!(use_more_bits(6, 32));
        assert!(!use_more_bits(7, 32));
    }

    /// A Q6_K row that is not a whole number of super-blocks steps down to
    /// Q8_0, which packs in blocks of 32.
    #[test]
    fn an_unquantizable_q6k_row_steps_down_to_q8_0() {
        assert_eq!(
            step_down_unquantizable(QuantType::Q6K, &[4096, 96]),
            QuantType::Q8_0
        );
        // A row that is already a whole number of super-blocks is untouched.
        assert_eq!(
            step_down_unquantizable(QuantType::Q6K, &[4096, 512]),
            QuantType::Q6K
        );
        // Q4_K has no writable step-down, so it is returned unchanged and the
        // caller's alignment check stores the tensor as F32.
        assert_eq!(
            step_down_unquantizable(QuantType::Q4K, &[4096, 96]),
            QuantType::Q4K
        );
    }
}
