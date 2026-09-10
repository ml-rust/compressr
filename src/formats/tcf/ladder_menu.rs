//! One tensor's menu of measured encodings, and the allocation it feeds.
//!
//! [`super::ladder`] decides WHETHER a ladder allocation runs. This module is
//! the allocation itself: it turns each quantizable tensor into a menu of
//! (encoding, bytes, delta), hands the menus to [`crate::policy::ladder`]'s
//! greedy, and writes the chosen encoding back into the plans.
//!
//! # Where the numbers come from
//!
//! - The DELTA of each option is the file's, at that (tensor, encoding) pair.
//!   Nothing is extrapolated from another encoding's delta.
//! - The BYTES of each option are `tcf-core`'s, from the same layout the
//!   record and the payload pass use. A budget spent in numbers the writer
//!   does not agree with is not a budget, so the file's own `bytes` field is
//!   compared against them and a disagreement is reported.
//! - Which encodings a tensor may take is [`super::screen`]'s answer, asked of
//!   the tensor's values before any record is fixed.
//!
//! # The budget
//!
//! A profile's bump fraction is the share of quantizable parameters its bump
//! encoding covers, and the two-way allocation spends it in elements. The same
//! quantity in bytes is what the ladder may spend above an all-base
//! allocation, so the artifact's ceiling is the one the two-way allocation
//! would have written and the two stay comparable.

use anyhow::{bail, Result};

use tcf_core::{Encoding, NativeEncoding};

use super::plan::PlannedTensor;
use super::profile::TcfProfile;

use crate::formats::{parse_tcf_encoding, tcf_encoding_name};
use crate::policy::{
    allocate_ladder, examples, LadderMenu, LadderOption, LadderReport, SensitivityMeasurement,
};

/// Choose one encoding per quantizable tensor and write it into the plans.
///
/// # Errors
/// A candidate has no record in the measurement, or none at the profile's base
/// encoding. The match is strict for the reason
/// [`crate::policy::sensitivity_rank`] gives: an allocation made half from a
/// measurement and half from a default is an allocation of nothing.
pub(super) fn allocate_profile_ladder(
    planned: &mut [PlannedTensor],
    profile: TcfProfile,
    measurement: &SensitivityMeasurement,
) -> Result<LadderReport> {
    let base = profile.base;
    let mut menus: Vec<LadderMenu> = Vec::new();
    // One encoding per rung, aligned with each menu's options.
    let mut rungs: Vec<Vec<NativeEncoding>> = Vec::new();
    let mut unmatched: Vec<&str> = Vec::new();
    let mut baseless: Vec<&str> = Vec::new();
    let mut candidates = 0usize;
    let mut unusable_pairs = 0usize;
    let mut single_option = 0usize;
    let mut byte_disagreements = 0usize;
    let mut base_bytes = 0u64;
    let mut cheapest_bytes = 0u64;
    let mut bump_budget = 0u64;

    for (index, plan) in planned.iter().enumerate() {
        if plan.fallback.is_some() {
            // A tensor stored raw has no encoding to choose between.
            continue;
        }
        candidates += 1;
        let options = measurement.options_for(&plan.name);
        if options.is_empty() {
            unmatched.push(plan.name.as_str());
            continue;
        }

        let mut menu_options: Vec<LadderOption> = Vec::new();
        let mut menu_rungs: Vec<NativeEncoding> = Vec::new();
        for option in options {
            // The base is the encoding this tensor was quantized at, so its
            // values already accepted it; every other rung had to be asked,
            // and `accepts` is that answer.
            let encoding = option
                .is_measured()
                .then(|| parse_tcf_encoding(&option.encoding).ok())
                .flatten()
                .filter(|encoding| *encoding == base || plan.accepts.contains(encoding));
            let Some(encoding) = encoding else {
                unusable_pairs += 1;
                continue;
            };
            let Some(bytes) = encoding_bytes(encoding, plan.elements) else {
                unusable_pairs += 1;
                continue;
            };
            if option.bytes.is_some_and(|measured| measured != bytes) {
                byte_disagreements += 1;
            }
            menu_options.push(LadderOption {
                encoding: tcf_encoding_name(encoding).to_string(),
                bytes,
                delta: option.delta,
            });
            menu_rungs.push(encoding);
        }

        let Some(base_rung) = menu_rungs.iter().position(|encoding| *encoding == base) else {
            baseless.push(plan.name.as_str());
            continue;
        };
        if menu_rungs.len() == 1 {
            single_option += 1;
        }
        let at_base = menu_options.get(base_rung).map_or(0, |option| option.bytes);
        base_bytes = base_bytes.saturating_add(at_base);
        cheapest_bytes = cheapest_bytes.saturating_add(
            menu_options
                .iter()
                .map(|option| option.bytes)
                .min()
                .unwrap_or(0),
        );
        // The profile's budget in bytes: raising this tensor to the bump
        // encoding, taken at the profile's bump fraction.
        if let Some(bump) = profile.bump.filter(|bump| plan.accepts.contains(bump)) {
            if let Some(at_bump) = encoding_bytes(bump, plan.elements) {
                let step = at_bump.saturating_sub(at_base) as f64 * profile.bump_fraction;
                bump_budget = bump_budget.saturating_add(step as u64);
            }
        }

        menus.push(LadderMenu {
            index,
            name: plan.name.clone(),
            options: menu_options,
            base: base_rung,
        });
        rungs.push(menu_rungs);
    }

    if !unmatched.is_empty() || !baseless.is_empty() {
        bail!(
            "{}",
            unresolved_error(measurement, &unmatched, &baseless, candidates, base)
        );
    }
    if byte_disagreements > 0 {
        log::warn!(
            "Sensitivity file names a payload byte count that differs from this encoding's own \
             layout for {byte_disagreements} (tensor, encoding) pair(s). tcf-core's count is \
             used: it is what the record and the payload pass write, and a budget spent in any \
             other number is not the budget the file spends."
        );
    }

    let available = base_bytes
        .saturating_add(bump_budget)
        .saturating_sub(cheapest_bytes);
    let outcome = allocate_ladder(&menus, available)?;

    for (choice, menu_rungs) in outcome.chosen.iter().zip(&rungs) {
        let encoding = menu_rungs
            .iter()
            .copied()
            .find(|encoding| tcf_encoding_name(*encoding) == choice.encoding);
        let Some(encoding) = encoding else {
            bail!(
                "Ladder chose '{}' for tensor {}, which is no encoding its menu offered.",
                choice.encoding,
                choice.index
            );
        };
        let Some(plan) = planned.get_mut(choice.index) else {
            bail!(
                "Ladder named tensor {}, which was never planned",
                choice.index
            );
        };
        plan.encoding = Encoding::Native(encoding);
        plan.bumped = choice.is_promoted();
        plan.demoted = choice.is_demoted();
        plan.sensitivity_delta = Some(choice.delta);
    }

    // A tensor no ladder allocated still carries what the file measured.
    for plan in planned.iter_mut() {
        if plan.sensitivity_delta.is_none() {
            plan.sensitivity_delta = measurement.delta_for(&plan.name);
        }
    }

    Ok(LadderReport::of(
        format!("TCF profile {}", profile.name),
        tcf_encoding_name(base).to_string(),
        bump_budget,
        unusable_pairs,
        &outcome,
    )
    .with_single_option_tensors(single_option))
}

/// Payload bytes one tensor takes at one encoding, from `tcf-core`'s own
/// layout arithmetic.
///
/// `None` when the element count is not a whole number of tiles, or when the
/// layout refuses it: neither is an encoding this tensor can be written at.
fn encoding_bytes(encoding: NativeEncoding, elements: u64) -> Option<u64> {
    let layout = encoding.layout();
    let tile = u64::from(layout.geometry.tile);
    if tile == 0 || !elements.is_multiple_of(tile) {
        return None;
    }
    layout.logical_payload_bytes(elements / tile).ok()
}

/// The message for candidates the ladder could not resolve.
fn unresolved_error(
    measurement: &SensitivityMeasurement,
    unmatched: &[&str],
    baseless: &[&str],
    candidates: usize,
    base: NativeEncoding,
) -> String {
    format!(
        "{} of {candidates} ladder candidates have no record in the sensitivity file, and {} \
         more have no usable option at the profile's base encoding {} — no record at it, or \
         values it refuses. Every candidate is allocated from measured options or none is: an \
         allocation made half from a measurement and half from a default is an allocation of \
         nothing, and a missing option read as no damage would demote an unmeasured tensor on \
         no evidence at all. Unmatched plan names: {}. Candidates with no base option: {}. \
         Names the file carries: {}. Encodings it measures: {}.",
        unmatched.len(),
        baseless.len(),
        tcf_encoding_name(base),
        examples(unmatched),
        examples(baseless),
        examples(&measurement.measured_names()),
        examples(&measurement.encodings()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::UnifiedDType;
    use crate::policy::{parse_sensitivity, FallbackReason};

    /// One quantized tensor, before any allocation.
    fn planned(name: &str, elements: u64, accepts: Vec<NativeEncoding>) -> PlannedTensor {
        PlannedTensor {
            name: name.to_string(),
            dtype: UnifiedDType::F32,
            fallback: None,
            encoding: Encoding::Native(NativeEncoding::Q4AS32DT64),
            screen_error: Some(0.1),
            elements,
            bumpable: accepts.contains(&NativeEncoding::Q8S32T64),
            accepts,
            bumped: false,
            demoted: false,
            sensitivity_delta: None,
        }
    }

    fn measurement(records: &[(&str, &str, f64)]) -> SensitivityMeasurement {
        let mut lines = vec![
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4AS32D_T64","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#
                .to_string(),
        ];
        for (name, encoding, delta) in records {
            lines.push(format!(
                r#"{{"record":"sensitivity","name":"{name}","encoding":"{encoding}","delta":{delta}}}"#
            ));
        }
        parse_sensitivity(&lines.join("\n")).expect("parses")
    }

    /// Every encoding compressr can write becomes a rung; the rest are named
    /// and left out.
    #[test]
    fn the_ladder_is_the_encodings_the_file_measures() {
        let measured = measurement(&[
            ("a.weight", "Q4AS32D_T64", 0.5),
            ("a.weight", "Q6S16D_T64", 0.1),
            ("a.weight", "Q9S32_T64", 0.0),
        ]);
        assert_eq!(
            super::super::ladder::ladder_encodings(&measured),
            vec![NativeEncoding::Q4AS32DT64, NativeEncoding::Q6S16DT64]
        );
    }

    /// A tensor the cheaper encoding does no measured damage to is DEMOTED,
    /// which no two-way allocation can do, and the bytes it frees buy the
    /// damaged tensor its bump. `TCF-HIGH` is the rung this shows on:
    /// `TCF-COMPACT`'s base is already the cheapest encoding TCF defines.
    #[test]
    fn an_undamaged_tensor_is_demoted() {
        let cheap = NativeEncoding::Q4AS32DT64;
        let bump = NativeEncoding::Q8S32T64;
        let mut plans = vec![
            planned("free.weight", 4096, vec![cheap, bump]),
            planned("damaged.weight", 4096, vec![cheap, bump]),
        ];
        let measured = measurement(&[
            ("free.weight", "Q4AS32D_T64", 0.0),
            ("free.weight", "Q6S16D_T64", 0.0),
            ("free.weight", "Q8S32_T64", 0.0),
            ("damaged.weight", "Q4AS32D_T64", 1.4),
            ("damaged.weight", "Q6S16D_T64", 0.9),
            ("damaged.weight", "Q8S32_T64", 0.1),
        ]);

        let report =
            allocate_profile_ladder(&mut plans, super::super::profile::TCF_HIGH, &measured)
                .expect("allocates");

        assert_eq!(plans[0].encoding, Encoding::Native(cheap));
        assert!(plans[0].demoted);
        assert!(!plans[0].bumped);
        assert_eq!(plans[1].encoding, Encoding::Native(bump));
        assert!(plans[1].bumped);
        assert_eq!(report.demoted, 1);
        assert_eq!(report.promoted, 1);
        assert!(report.damage_removed() > 0.0);
        // The artifact is no larger than the all-base one it replaces.
        assert!(report.chosen_bytes <= report.base_bytes + report.budget_bytes);
        // Each tensor records the delta of the encoding it landed on.
        assert_eq!(plans[1].sensitivity_delta, Some(0.1));
    }

    /// A tensor whose values refuse an encoding is never allocated to it.
    #[test]
    fn an_encoding_the_values_refuse_is_never_chosen() {
        let bump = NativeEncoding::Q6S16DT64;
        // `accepts` is empty, so nothing but the base is reachable.
        let mut plans = vec![planned("a.weight", 4096, Vec::new())];
        let measured = measurement(&[
            ("a.weight", "Q4AS32D_T64", 0.9),
            ("a.weight", "Q6S16D_T64", 0.0),
        ]);

        let report =
            allocate_profile_ladder(&mut plans, super::super::profile::TCF_COMPACT, &measured)
                .expect("allocates");

        assert_eq!(
            plans[0].encoding,
            Encoding::Native(NativeEncoding::Q4AS32DT64)
        );
        assert!(!plans[0].bumped);
        assert!(!plans[0].demoted);
        assert_eq!(report.unusable_pairs, 1);
        assert_eq!(report.single_option_tensors, 1);
        assert!(!plans[0].accepts.contains(&bump));
    }

    /// A candidate the file does not name stops the allocation.
    #[test]
    fn an_unresolved_candidate_is_refused() {
        let mut plans = vec![
            planned("a.weight", 4096, vec![NativeEncoding::Q6S16DT64]),
            planned("b.weight", 4096, vec![NativeEncoding::Q6S16DT64]),
        ];
        let measured = measurement(&[
            ("a.weight", "Q4AS32D_T64", 0.9),
            ("a.weight", "Q6S16D_T64", 0.1),
        ]);
        let error =
            allocate_profile_ladder(&mut plans, super::super::profile::TCF_COMPACT, &measured)
                .expect_err("one unmatched candidate")
                .to_string();
        assert!(error.contains("1 of 2 ladder candidates"), "{error}");
        assert!(error.contains("'b.weight'"), "{error}");
    }

    /// A tensor stored raw has no encoding to choose, so it is no candidate.
    #[test]
    fn a_raw_tensor_is_no_candidate() {
        let mut plans = vec![planned("a.weight", 4096, vec![NativeEncoding::Q6S16DT64])];
        plans[0].fallback = Some(FallbackReason::Rank);
        let measured = measurement(&[
            ("b.weight", "Q4AS32D_T64", 0.9),
            ("b.weight", "Q6S16D_T64", 0.1),
        ]);
        let report =
            allocate_profile_ladder(&mut plans, super::super::profile::TCF_COMPACT, &measured)
                .expect("allocates nothing");
        assert_eq!(report.tensors, 0);
        assert_eq!(report.demoted, 0);
    }
}
