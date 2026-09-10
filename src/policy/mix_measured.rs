//! A GGUF mixed-precision allocation driven by a measured task metric.
//!
//! `Q4_K_M` is already a per-tensor allocation. What decides it is
//! llama.cpp's positional heuristic: a tensor's role, its layer index, and a
//! running counter — see [`super::mix_rules`]. Nothing in that heuristic ever
//! measured the model. `--sensitivity` supplies a measurement that did, and
//! this module spends the same bytes by that measurement instead.
//!
//! # What is held fixed
//!
//! The comparison is only about WHERE the bits go, so everything else is
//! pinned to the positional mix:
//!
//! - The same two formats. Every quantizable tensor takes the mix's base
//!   format, and a promoted one takes [`GgufMix::promoted`].
//! - The same byte budget. The positional allocation is computed first, its
//!   bytes are measured, and the ranking may spend exactly what that
//!   allocation spends ABOVE an all-base allocation of the same checkpoint —
//!   see [`assign_mix_measured`]. The two artifacts are therefore size
//!   matched, and the difference between them is the allocation policy alone.
//! - The same fallbacks. A tensor no mix quantizes is untouched, and
//!   [`step_down_unquantizable`] still runs.
//!
//! # Why the blocked count is the point
//!
//! A K-quant row is a whole number of 256-element super-blocks or it cannot
//! be packed at all. So a tensor the measurement ranks first can be one GGUF
//! cannot place a single extra bit in. Those tensors are counted, with their
//! parameters, and reported — never silently skipped. That count is the
//! direct evidence for or against a finer-grained container.
//!
//! Without `--sensitivity` none of this runs and [`super::mix_assign`]'s
//! allocation is byte-identical to what it has always been.

use anyhow::Result;

use crate::formats::GgmlType;
use crate::loader::UnifiedDType;
use crate::quantization::QuantType;

use super::gguf_mix::GgufMix;
use super::mix_allocation::MixAllocation;
use super::mix_assign::assign_mix_reported;
use super::mix_measured_report::MeasuredMixAllocation;
use super::mix_model::MixModel;
use super::mix_report::quant_type_label;
use super::mix_rules::{plan_mixed_tensor, step_down_unquantizable};
use super::sensitivity::SensitivityMeasurement;
use super::sensitivity_rank::{rank_candidates, BumpCandidate};
use super::TensorPlan;

/// One tensor as the measured allocation reads it: name, shape, element type.
pub type MeasuredTensor<'a> = (&'a str, &'a [usize], UnifiedDType);

/// Assign one GGUF block format per tensor from a measured task metric.
///
/// The returned vector is in the caller's order, exactly as
/// [`super::mix_assign::assign_mix`] returns it. The [`MixAllocation`] beside
/// it is the POSITIONAL allocation this run is matched against, unmodified,
/// so a report can state both.
///
/// # How the budget is defined
///
/// 1. The positional allocation is computed first, untouched, and its bytes
///    are measured over every tensor.
/// 2. An ALL-BASE allocation of the same checkpoint is measured the same way:
///    every tensor the mix quantizes at the base format, every other tensor
///    exactly as the positional allocation left it.
/// 3. The budget is the difference. It is what the positional mix spends to
///    promote anything at all, and it is what the ranking may spend.
///
/// The budget is spent IN ORDER. The first tensor that does not fit ends the
/// spend, and what is left over is reported rather than filled by a
/// lower-ranked tensor — filling it would no longer be the measured order,
/// and it would make the measured artifact the larger of the two.
///
/// # Errors
/// Any tensor the mix quantizes has no record in the measurement. The match
/// is strict for the reason [`super::sensitivity_rank`] gives: a set ranked
/// half by measurement and half by anything else is a ranking of nothing.
pub fn assign_mix_measured(
    mix: GgufMix,
    tensors: &[MeasuredTensor<'_>],
    model: &MixModel,
    measurement: &SensitivityMeasurement,
) -> Result<(Vec<Option<QuantType>>, MixAllocation, MeasuredMixAllocation)> {
    let base = mix.base();
    let promoted = mix.promoted();
    let (heuristic, positional) = assign_mix_reported(
        mix,
        tensors.iter().map(|(name, shape, _)| (*name, *shape)),
        model,
    );

    let mut assigned: Vec<Option<QuantType>> = Vec::with_capacity(tensors.len());
    let mut candidates: Vec<BumpCandidate<'_>> = Vec::new();
    let mut promoted_by_heuristic = vec![false; tensors.len()];
    let mut heuristic_promoted = 0usize;
    let mut blocked_tensors = 0usize;
    let mut blocked_parameters = 0u64;
    let mut floor_bytes = 0u64;
    let mut heuristic_bytes = 0u64;
    let mut total_elements = 0u64;

    for (index, (name, shape, dtype)) in tensors.iter().enumerate() {
        let elements = element_count(shape);
        total_elements += elements;
        heuristic_bytes += plan_bytes(
            plan_mixed_tensor(shape, *dtype, heuristic[index]),
            elements,
            *dtype,
        );
        if heuristic[index].is_some_and(|format| format != base) {
            promoted_by_heuristic[index] = true;
            heuristic_promoted += 1;
        }

        // A tensor the mix assigns a format to and whose values are floats is
        // exactly the set the mix quantizes, so the role rules are read from
        // that assignment rather than re-derived here.
        let is_candidate = heuristic[index].is_some() && dtype.is_float();
        let base_assigned = if is_candidate {
            Some(step_down_unquantizable(base, shape))
        } else {
            heuristic[index]
        };
        floor_bytes += plan_bytes(
            plan_mixed_tensor(shape, *dtype, base_assigned),
            elements,
            *dtype,
        );
        assigned.push(base_assigned);

        if is_candidate {
            if !holds_format(shape, promoted) {
                blocked_tensors += 1;
                blocked_parameters += elements;
            }
            candidates.push(BumpCandidate {
                index,
                name,
                elements,
            });
        }
    }

    let budget_bytes = heuristic_bytes.saturating_sub(floor_bytes);
    let ranking = rank_candidates(measurement, &candidates)?;

    let mut spent_bytes = 0u64;
    let mut measured_promoted = 0usize;
    let mut overlap = 0usize;
    let mut wanted_blocked_tensors = 0usize;
    let mut wanted_blocked_parameters = 0u64;

    if budget_bytes > 0 {
        for ranked in &ranking.order {
            let (_, shape, dtype) = tensors[ranked.index];
            if !holds_format(shape, promoted) {
                wanted_blocked_tensors += 1;
                wanted_blocked_parameters += ranked.elements;
                continue;
            }
            let cost = promotion_cost(shape, dtype, ranked.elements, base, promoted);
            if spent_bytes + cost > budget_bytes {
                break;
            }
            assigned[ranked.index] = Some(step_down_unquantizable(promoted, shape));
            spent_bytes += cost;
            measured_promoted += 1;
            if promoted_by_heuristic[ranked.index] {
                overlap += 1;
            }
        }
    }

    let allocation = MeasuredMixAllocation {
        mix_name: mix.name().to_string(),
        base_format: quant_type_label(base).to_string(),
        promoted_format: quant_type_label(promoted).to_string(),
        baseline_encoding: measurement.baseline().encoding.clone(),
        candidates: candidates.len(),
        excluded_nonpositive: ranking.excluded_nonpositive,
        heuristic_promoted,
        measured_promoted,
        overlap,
        blocked_tensors,
        blocked_parameters,
        wanted_blocked_tensors,
        wanted_blocked_parameters,
        budget_bytes,
        spent_bytes,
        floor_bytes,
        heuristic_bytes,
        measured_bytes: floor_bytes + spent_bytes,
        total_elements,
    };

    Ok((assigned, positional, allocation))
}

/// Elements in a tensor of this shape.
fn element_count(shape: &[usize]) -> u64 {
    shape.iter().product::<usize>() as u64
}

/// Whether a row of this shape is a whole number of the format's blocks.
///
/// GGUF packs blocks along a ROW, which is the LAST dimension of the standard
/// order shape. For `Q4_K` and `Q6_K` that block is 256 elements, so a row
/// that is not a multiple of 256 can hold neither.
fn holds_format(shape: &[usize], format: QuantType) -> bool {
    shape.len() >= 2 && shape[shape.len() - 1].is_multiple_of(format.block_size())
}

/// Extra bytes promoting one tensor costs, over leaving it at the base format.
fn promotion_cost(
    shape: &[usize],
    dtype: UnifiedDType,
    elements: u64,
    base: QuantType,
    promoted: QuantType,
) -> u64 {
    let at = |format: QuantType| {
        plan_bytes(
            plan_mixed_tensor(shape, dtype, Some(format)),
            elements,
            dtype,
        )
    };
    at(promoted).saturating_sub(at(base))
}

/// Bytes one plan writes for a tensor of this many elements, read from the
/// GGUF writer's own block table rather than from a nominal bit width.
fn plan_bytes(plan: TensorPlan, elements: u64, dtype: UnifiedDType) -> u64 {
    match plan {
        TensorPlan::Quantize(format) => {
            GgmlType::from_quant_type(format).tensor_bytes(elements as usize) as u64
        }
        // An unquantizable tensor is stored as F32, which is what the
        // registration loop reserves for it.
        TensorPlan::StoreF32(_) => elements * 4,
        TensorPlan::StoreAsIs => elements * dtype.element_size().unwrap_or(4) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::config_facts::LayerCountSource;
    use crate::policy::parse_sensitivity;

    /// A model whose rules read one stack of `n_layer` blocks.
    fn dense(n_layer: usize) -> MixModel {
        MixModel {
            arch: "llama".into(),
            n_layer,
            layer_count_source: LayerCountSource::Config("num_hidden_layers".into()),
            n_expert: 0,
            is_70b: false,
        }
    }

    /// A measurement at the encoding a `q4_k_*` base is ranked from.
    fn measurement(deltas: &[(&str, f64)]) -> SensitivityMeasurement {
        let mut lines = vec![
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4AS32D_T64","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#
                .to_string(),
        ];
        for (name, delta) in deltas {
            lines.push(format!(
                r#"{{"record":"sensitivity","name":"{name}","delta":{delta}}}"#
            ));
        }
        parse_sensitivity(&lines.join("\n")).expect("parses")
    }

    /// One transformer block, plus an output weight the positional rules
    /// always promote.
    fn block(shape: &[usize]) -> Vec<MeasuredTensor<'_>> {
        [
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_up.weight",
        ]
        .into_iter()
        .map(|name| (name, shape, UnifiedDType::F32))
        .collect()
    }

    /// The two artifacts are size matched: the ranking never spends a byte
    /// the positional mix did not spend, and the bpw it achieves is at most
    /// the positional mix's.
    #[test]
    fn the_budget_is_what_the_positional_mix_spent() {
        let shape = [512usize, 512];
        let tensors = block(&shape);
        let measurement = measurement(&[
            ("output.weight", 0.1),
            ("blk.0.attn_q.weight", 0.9),
            ("blk.0.attn_v.weight", 0.8),
            ("blk.0.ffn_down.weight", 0.7),
            ("blk.0.ffn_up.weight", 0.6),
        ]);

        let (_, _, report) =
            assign_mix_measured(GgufMix::Q4KM, &tensors, &dense(1), &measurement).expect("ranks");

        assert!(report.budget_bytes > 0);
        assert!(report.spent_bytes <= report.budget_bytes);
        assert!(report.measured_bytes <= report.heuristic_bytes);
        assert!(report.measured_bpw() <= report.heuristic_bpw());
        // Every tensor here holds the same number of elements, so the budget
        // buys exactly as many promotions as the positional mix made.
        assert_eq!(report.measured_promoted, report.heuristic_promoted);
        // The ranking took the three most damaged, not the positional set.
        assert!(report.overlap < report.measured_promoted);
    }

    /// A tensor the base format did not damage is never promoted, whatever
    /// budget is left.
    #[test]
    fn a_non_positive_delta_is_never_promoted() {
        let shape = [512usize, 512];
        let tensors = block(&shape);
        let measurement = measurement(&[
            ("output.weight", -0.5),
            ("blk.0.attn_q.weight", 0.0),
            ("blk.0.attn_v.weight", 0.8),
            ("blk.0.ffn_down.weight", -0.1),
            ("blk.0.ffn_up.weight", 0.6),
        ]);

        let (assigned, _, report) =
            assign_mix_measured(GgufMix::Q4KM, &tensors, &dense(1), &measurement).expect("ranks");

        assert_eq!(report.excluded_nonpositive, 3);
        for (index, (name, _, _)) in tensors.iter().enumerate() {
            if matches!(
                *name,
                "output.weight" | "blk.0.attn_q.weight" | "blk.0.ffn_down.weight"
            ) {
                assert_eq!(assigned[index], Some(QuantType::Q4K), "{name}");
            }
        }
    }

    /// A row that is not a whole number of super-blocks cannot take Q6_K, or
    /// even Q4_K. The measurement ranks it first and it is counted, with its
    /// parameters, rather than skipped.
    #[test]
    fn an_alignment_blocked_tensor_is_counted() {
        let wide = [512usize, 512];
        let narrow = [512usize, 96];
        let tensors: Vec<MeasuredTensor<'_>> = vec![
            ("output.weight", &wide[..], UnifiedDType::F32),
            ("blk.0.attn_v.weight", &wide[..], UnifiedDType::F32),
            ("blk.0.ffn_up.weight", &narrow[..], UnifiedDType::F32),
        ];
        // The narrow tensor has the largest damage per element by far, so the
        // ranking reaches it first.
        let measurement = measurement(&[
            ("output.weight", 0.1),
            ("blk.0.attn_v.weight", 0.2),
            ("blk.0.ffn_up.weight", 0.9),
        ]);

        let (assigned, _, report) =
            assign_mix_measured(GgufMix::Q4KM, &tensors, &dense(1), &measurement).expect("ranks");

        assert_eq!(report.blocked_tensors, 1);
        assert_eq!(report.blocked_parameters, 512 * 96);
        assert_eq!(report.wanted_blocked_tensors, 1);
        assert_eq!(report.wanted_blocked_parameters, 512 * 96);
        // It stays at the base format, which its row cannot hold either, so
        // the writer stores it as F32.
        assert_eq!(assigned[2], Some(QuantType::Q4K));
    }

    /// Every tensor the mix quantizes must resolve, and the error names both
    /// namespaces rather than falling back to the positional rules.
    #[test]
    fn an_unmatched_candidate_is_refused() {
        let shape = [512usize, 512];
        let tensors = block(&shape);
        let measurement = measurement(&[("output.weight", 0.1)]);

        let error = assign_mix_measured(GgufMix::Q4KM, &tensors, &dense(1), &measurement)
            .expect_err("four unmatched candidates")
            .to_string();
        assert!(error.contains("of 5 bump candidates"), "{error}");
        assert!(error.contains("'blk.0.attn_q.weight'"), "{error}");
        assert!(error.contains("'output.weight'"), "{error}");
    }
}
