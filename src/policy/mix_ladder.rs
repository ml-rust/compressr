//! A GGUF mixed-precision allocation chosen from a per-encoding measurement.
//!
//! [`super::mix_measured`] spends the mix's byte budget two ways: every
//! quantizable tensor at the base format, a ranked subset raised to the
//! promoted one. This module spends the same budget over a MENU — one measured
//! option per (tensor, encoding) pair — and lets a tensor land below the base
//! as well as above it. The greedy that picks the rungs is
//! [`super::ladder`]; everything here is the translation between a
//! measurement's encodings and the block formats GGUF can actually write.
//!
//! # Which formats a measured encoding can become
//!
//! A sensitivity file names the block formats it measured at, so a measured
//! encoding IS the rung. A name compressr has no writer for is not an option
//! at all. A format measured twice for one tensor collapses to one rung
//! holding the LARGER delta: an estimate that understates the damage would
//! buy a demotion the measurement never supported.
//!
//! # When the ladder does not run
//!
//! A mix whose base format no record in the file measures has no rung to
//! stand on. The two-way allocation runs instead, unchanged, and says so.

use anyhow::{bail, Result};

use crate::quantization::QuantType;

use super::gguf_mix::GgufMix;
use super::ladder::{allocate_ladder, LadderMenu, LadderOption};
use super::ladder_report::LadderReport;
use super::mix_bytes::{element_count, format_bytes, holds_format};
use super::mix_ladder_format::{ladder_format, unresolved_error};
use super::mix_measured::MeasuredTensor;
use super::mix_report::quant_type_label;
use super::sensitivity::SensitivityMeasurement;

/// Whether a ladder allocation can run for this mix and this measurement.
///
/// It needs a menu — more than one encoding — and it needs one of those
/// encodings to be at the base format's bit width, which is the rung every
/// tensor is reported against and the one a two-way allocation would have
/// written. Without that rung the base is unmeasured, and a demotion would be
/// a move away from a number nothing measured.
pub fn can_ladder(mix: GgufMix, measurement: &SensitivityMeasurement) -> bool {
    let base = mix.base();
    measurement.is_ladder()
        && measurement
            .encodings()
            .iter()
            .any(|encoding| ladder_format(encoding) == Some(base))
}

/// Allocate one block format per candidate tensor from the measured menu.
///
/// `assigned` arrives as the all-base allocation and leaves carrying the
/// ladder's choice for every candidate. `budget_bytes` is what the positional
/// mix spends above that all-base allocation — the same budget the two-way
/// path spends, so the two artifacts stay comparable.
///
/// # Errors
/// A candidate has no record in the measurement, or none at the base format's
/// bit width. The match is strict for the reason
/// [`super::sensitivity_rank`] gives: a set ranked half by measurement and
/// half by anything else is a ranking of nothing.
pub fn allocate_mix_ladder(
    mix: GgufMix,
    tensors: &[MeasuredTensor<'_>],
    candidates: &[usize],
    assigned: &mut [Option<QuantType>],
    budget_bytes: u64,
    measurement: &SensitivityMeasurement,
) -> Result<LadderReport> {
    let base = mix.base();
    let mut menus: Vec<LadderMenu> = Vec::with_capacity(candidates.len());
    // One format per rung, positionally aligned with each menu's options, so
    // a chosen encoding name resolves back to the format that produced it.
    let mut formats: Vec<Vec<QuantType>> = Vec::with_capacity(candidates.len());
    let mut unmatched: Vec<&str> = Vec::new();
    let mut baseless: Vec<&str> = Vec::new();
    let mut unusable_pairs = 0usize;
    let mut single_option = 0usize;
    let mut base_bytes = 0u64;
    let mut cheapest_bytes = 0u64;

    for index in candidates {
        let Some((name, shape, dtype)) = tensors.get(*index).copied() else {
            bail!("Ladder named tensor {index}, which the mix never planned");
        };
        let options = measurement.options_for(name);
        if options.is_empty() {
            unmatched.push(name);
            continue;
        }
        let elements = element_count(shape);

        // One rung per writable format, holding the largest delta any encoding
        // at that width measured.
        let mut rungs: Vec<(QuantType, f64)> = Vec::new();
        for option in options {
            let format = option
                .is_measured()
                .then(|| ladder_format(&option.encoding))
                .flatten()
                .filter(|format| holds_format(shape, *format));
            let Some(format) = format else {
                unusable_pairs += 1;
                continue;
            };
            match rungs.iter_mut().find(|(held, _)| *held == format) {
                Some((_, delta)) => *delta = delta.max(option.delta),
                None => rungs.push((format, option.delta)),
            }
        }

        let Some(base_rung) = rungs.iter().position(|(format, _)| *format == base) else {
            baseless.push(name);
            continue;
        };
        if rungs.len() == 1 {
            single_option += 1;
        }

        let mut menu_options = Vec::with_capacity(rungs.len());
        let mut menu_formats = Vec::with_capacity(rungs.len());
        for (format, delta) in &rungs {
            menu_options.push(LadderOption {
                encoding: quant_type_label(*format).to_string(),
                bytes: format_bytes(shape, dtype, elements, *format),
                delta: *delta,
            });
            menu_formats.push(*format);
        }
        base_bytes =
            base_bytes.saturating_add(menu_options.get(base_rung).map_or(0, |option| option.bytes));
        cheapest_bytes = cheapest_bytes.saturating_add(
            menu_options
                .iter()
                .map(|option| option.bytes)
                .min()
                .unwrap_or(0),
        );
        menus.push(LadderMenu {
            index: *index,
            name: name.to_string(),
            options: menu_options,
            base: base_rung,
        });
        formats.push(menu_formats);
    }

    if !unmatched.is_empty() || !baseless.is_empty() {
        bail!(
            "{}",
            unresolved_error(measurement, &unmatched, &baseless, candidates.len(), base)
        );
    }

    // The ceiling is the two-way allocation's: an all-base allocation plus
    // what the positional mix spends above it. Starting every tensor at its
    // cheapest measured option is below that floor, and the difference is
    // budget the ladder may spend as well.
    let available = base_bytes
        .saturating_add(budget_bytes)
        .saturating_sub(cheapest_bytes);
    let outcome = allocate_ladder(&menus, available)?;

    // `chosen` is in menu order, and `formats` was built in that same order,
    // so the rungs beside a choice are the ones its own menu offered.
    for (choice, menu_formats) in outcome.chosen.iter().zip(&formats) {
        let format = menu_formats
            .iter()
            .copied()
            .find(|format| quant_type_label(*format) == choice.encoding);
        let Some(format) = format else {
            bail!(
                "Ladder chose '{}' for tensor {}, which is no format this mix offered it.",
                choice.encoding,
                choice.index
            );
        };
        let Some(slot) = assigned.get_mut(choice.index) else {
            bail!(
                "Ladder named tensor {}, which was never planned",
                choice.index
            );
        };
        *slot = Some(format);
    }

    Ok(LadderReport::of(
        format!("GGUF mix {}", mix.name()),
        quant_type_label(base).to_string(),
        budget_bytes,
        unusable_pairs,
        &outcome,
    )
    .with_single_option_tensors(single_option))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::UnifiedDType;
    use crate::policy::parse_sensitivity;

    /// A file measuring one encoding per (tensor, encoding) pair.
    fn measurement(records: &[(&str, &str, f64)]) -> SensitivityMeasurement {
        let mut lines = vec![
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4_K","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#
                .to_string(),
        ];
        for (name, encoding, delta) in records {
            lines.push(format!(
                r#"{{"record":"sensitivity","name":"{name}","encoding":"{encoding}","delta":{delta}}}"#
            ));
        }
        parse_sensitivity(&lines.join("\n")).expect("parses")
    }

    /// A four-bit mix laddered against a file that measures four and six
    /// bits; a five-bit mix has no rung to stand on.
    #[test]
    fn a_ladder_needs_the_base_width_measured() {
        let measured = measurement(&[("a.weight", "Q4_K", 0.5), ("a.weight", "Q6_K", 0.1)]);
        assert!(can_ladder(GgufMix::Q4KM, &measured));
        assert!(!can_ladder(GgufMix::Q5KM, &measured));

        // One encoding is a two-way file, whatever its width.
        let two_way = measurement(&[("a.weight", "Q4_K", 0.5)]);
        assert!(!can_ladder(GgufMix::Q4KM, &two_way));
    }

    /// The budget the positional mix spends is spent over the menu, and a
    /// tensor the measurement damages most takes the wider format.
    #[test]
    fn the_budget_buys_the_most_damaged_tensor_the_wider_format() {
        let shape = [512usize, 512];
        let tensors: Vec<MeasuredTensor<'_>> = vec![
            ("a.weight", &shape[..], UnifiedDType::F32),
            ("b.weight", &shape[..], UnifiedDType::F32),
        ];
        let measured = measurement(&[
            ("a.weight", "Q4_K", 0.9),
            ("a.weight", "Q6_K", 0.1),
            ("b.weight", "Q4_K", 0.2),
            ("b.weight", "Q6_K", 0.1),
        ]);
        let mut assigned = vec![Some(QuantType::Q4K), Some(QuantType::Q4K)];
        let elements = element_count(&shape);
        let step = format_bytes(&shape, UnifiedDType::F32, elements, QuantType::Q6K)
            - format_bytes(&shape, UnifiedDType::F32, elements, QuantType::Q4K);

        let report = allocate_mix_ladder(
            GgufMix::Q4KM,
            &tensors,
            &[0, 1],
            &mut assigned,
            step,
            &measured,
        )
        .expect("allocates");

        assert_eq!(assigned[0], Some(QuantType::Q6K));
        assert_eq!(assigned[1], Some(QuantType::Q4K));
        assert_eq!(report.promoted, 1);
        assert_eq!(report.demoted, 0);
        assert_eq!(report.spent_bytes, step);
        assert!(report.chosen_bytes <= report.base_bytes + step);
    }

    /// A candidate the file does not name stops the allocation, and the error
    /// shows both namespaces.
    #[test]
    fn an_unresolved_candidate_is_refused() {
        let shape = [512usize, 512];
        let tensors: Vec<MeasuredTensor<'_>> = vec![
            ("a.weight", &shape[..], UnifiedDType::F32),
            (
                "model.layers.0.attn.q_proj.weight",
                &shape[..],
                UnifiedDType::F32,
            ),
        ];
        let measured = measurement(&[("a.weight", "Q4_K", 0.9), ("a.weight", "Q6_K", 0.1)]);
        let mut assigned = vec![Some(QuantType::Q4K), Some(QuantType::Q4K)];
        let error = allocate_mix_ladder(
            GgufMix::Q4KM,
            &tensors,
            &[0, 1],
            &mut assigned,
            0,
            &measured,
        )
        .expect_err("one unmatched candidate")
        .to_string();
        assert!(error.contains("1 of 2 ladder candidates"), "{error}");
        assert!(
            error.contains("'model.layers.0.attn.q_proj.weight'"),
            "{error}"
        );
    }

    /// A candidate measured only above the base format has no rung the
    /// allocation is reported against, and is refused rather than allocated
    /// from a rung nothing measured.
    #[test]
    fn a_candidate_with_no_base_option_is_refused() {
        let shape = [512usize, 512];
        let tensors: Vec<MeasuredTensor<'_>> = vec![("a.weight", &shape[..], UnifiedDType::F32)];
        let measured = measurement(&[("a.weight", "Q6_K", 0.1), ("b.weight", "Q4_K", 0.4)]);
        let mut assigned = vec![Some(QuantType::Q4K)];
        let error = allocate_mix_ladder(GgufMix::Q4KM, &tensors, &[0], &mut assigned, 0, &measured)
            .expect_err("no base option")
            .to_string();
        assert!(
            error.contains("no usable option at the base format"),
            "{error}"
        );
        assert!(error.contains("Q4_K"), "{error}");
    }

    /// A tensor whose row is not a whole number of super-blocks cannot take a
    /// K-quant at all: the pair is counted and never chosen.
    #[test]
    fn a_pair_the_row_cannot_hold_is_never_chosen() {
        let wide = [512usize, 512];
        let narrow = [512usize, 96];
        let tensors: Vec<MeasuredTensor<'_>> = vec![
            ("a.weight", &wide[..], UnifiedDType::F32),
            ("narrow.weight", &narrow[..], UnifiedDType::F32),
        ];
        // The narrow tensor is measured at both widths and can hold neither
        // K-quant, but its row is a whole number of Q8_0's 32-element blocks.
        let measured = measurement(&[
            ("a.weight", "Q4_K", 0.9),
            ("a.weight", "Q6_K", 0.1),
            ("narrow.weight", "Q4_K", 0.9),
            ("narrow.weight", "Q6_K", 0.1),
        ]);
        let mut assigned = vec![Some(QuantType::Q4K), Some(QuantType::Q4K)];
        let error = allocate_mix_ladder(
            GgufMix::Q4KM,
            &tensors,
            &[0, 1],
            &mut assigned,
            0,
            &measured,
        )
        .expect_err("the narrow tensor has no base rung")
        .to_string();
        // Its 4-bit rung is the base, and its row cannot hold it, so the
        // candidate has no base option rather than a silent fallback.
        assert!(
            error.contains("no usable option at the base format"),
            "{error}"
        );
    }
}
