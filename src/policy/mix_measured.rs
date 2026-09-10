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
//! pinned to the positional mix: the same formats, the same fallbacks, and
//! the same byte budget — what the positional allocation spends ABOVE an
//! all-base allocation of the same checkpoint. The two artifacts are size
//! matched, and the difference between them is the allocation policy alone.
//!
//! # Two ways, or a ladder
//!
//! A file measured at ONE encoding says only what the base format did, so the
//! allocation it drives is two-way: base, or promoted. A file measuring
//! SEVERAL encodings per tensor is a menu, and [`super::mix_ladder`] spends
//! the same budget over it — a tensor can then land BELOW the base as well as
//! above it. The two-way path is unchanged and still runs for every file that
//! carries one encoding.
//!
//! # Why the blocked count is the point
//!
//! A K-quant row is a whole number of 256-element super-blocks or it cannot
//! be packed at all. So a tensor the measurement ranks first can be one GGUF
//! cannot place a single extra bit in. Those tensors are counted, with their
//! parameters, and reported — never silently skipped.
//!
//! Without `--sensitivity` none of this runs and [`super::mix_assign`]'s
//! allocation is byte-identical to what it has always been.

use anyhow::Result;

use crate::loader::UnifiedDType;
use crate::quantization::QuantType;

use super::gguf_mix::GgufMix;
use super::mix_allocation::MixAllocation;
use super::mix_assign::assign_mix_reported;
use super::mix_bytes::{element_count, holds_format, plan_bytes};
use super::mix_measured_report::MeasuredMixAllocation;
use super::mix_model::MixModel;
use super::mix_promote::PromotionFormats;
use super::mix_report::quant_type_label;
use super::mix_rules::{plan_mixed_tensor, step_down_unquantizable};
use super::mix_spend::{spend_budget, SpendInputs};
use super::sensitivity::SensitivityMeasurement;
use super::sensitivity_rank::BumpCandidate;

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

    let spend = spend_budget(
        mix,
        tensors,
        &candidates,
        &mut assigned,
        &promoted_by_heuristic,
        SpendInputs {
            formats: PromotionFormats { base, promoted },
            floor_bytes,
            budget_bytes,
        },
        measurement,
    )?;

    let allocation = MeasuredMixAllocation {
        mix_name: mix.name().to_string(),
        base_format: quant_type_label(base).to_string(),
        promoted_format: quant_type_label(promoted).to_string(),
        baseline_encoding: measurement.baseline().encoding.clone(),
        candidates: candidates.len(),
        excluded_nonpositive: spend.excluded_nonpositive,
        heuristic_promoted,
        measured_promoted: spend.promoted,
        overlap: spend.overlap,
        blocked_tensors,
        blocked_parameters,
        wanted_blocked_tensors: spend.wanted_blocked_tensors,
        wanted_blocked_parameters: spend.wanted_blocked_parameters,
        budget_bytes,
        spent_bytes: spend.spent_bytes,
        floor_bytes,
        heuristic_bytes,
        measured_bytes: spend.measured_bytes,
        total_elements,
        ladder: spend.ladder,
    };

    Ok((assigned, positional, allocation))
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

    /// A file measured at the encoding a `q4_k_*` base is ranked from.
    fn file(deltas: &[(&str, f64)]) -> String {
        let mut lines = vec![
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4AS32D_T64","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#
                .to_string(),
        ];
        for (name, delta) in deltas {
            lines.push(format!(
                r#"{{"record":"sensitivity","name":"{name}","delta":{delta}}}"#
            ));
        }
        lines.join("\n")
    }

    fn measurement(deltas: &[(&str, f64)]) -> SensitivityMeasurement {
        parse_sensitivity(&file(deltas)).expect("parses")
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

    /// A file measured at one encoding takes the two-way path, and a file
    /// measuring several takes the ladder. This is the compatibility
    /// guarantee: a single-encoding menu allocates exactly as it always has.
    #[test]
    fn one_encoding_allocates_two_ways_and_a_menu_ladders() {
        let shape = [512usize, 512];
        let tensors = block(&shape);
        let deltas = [
            ("output.weight", 0.1),
            ("blk.0.attn_q.weight", 0.9),
            ("blk.0.attn_v.weight", 0.8),
            ("blk.0.ffn_down.weight", 0.7),
            ("blk.0.ffn_up.weight", 0.6),
        ];
        let two_way = measurement(&deltas);
        let (_, _, report) =
            assign_mix_measured(GgufMix::Q4KM, &tensors, &dense(1), &two_way).expect("ranks");
        assert!(report.ladder.is_none());
        assert!(report.to_string().contains("none demoted"));

        // The same file, plus a rung at another encoding, is a menu.
        let mut text = file(&deltas);
        for (name, delta) in &deltas {
            text.push_str(&format!(
                "\n{{\"record\":\"sensitivity\",\"name\":\"{name}\",\"encoding\":\"Q6S16D_T64\",\"delta\":{}}}",
                delta / 8.0
            ));
        }
        let laddered = parse_sensitivity(&text).expect("parses");
        let (_, _, report) =
            assign_mix_measured(GgufMix::Q4KM, &tensors, &dense(1), &laddered).expect("allocates");
        let ladder = report.ladder.expect("a ladder ran");
        assert_eq!(ladder.tensors, 5);
        assert!(ladder.to_string().contains("DEMOTED below the base"));
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
