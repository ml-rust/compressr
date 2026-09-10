//! Which tensors a TCF profile raises to its bump encoding, and why that one.
//!
//! [`super::plan`] settles one tensor at a time. This module settles them
//! against each other: a profile's bump set is a share of the whole model, so
//! nothing here can run until every tensor has been screened.
//!
//! Two rankings feed it, and they are never mixed. The default is
//! [`crate::policy::allocation`]'s weight-error screen, which is not a task
//! delta and leaves every sensitivity field unset. `--sensitivity` instead
//! supplies `PROFILES.md` Section 4 step 2's measurement, taken elsewhere on a
//! model compressr never runs, and [`crate::policy::sensitivity_rank`] ranks
//! the same budget by measured task damage per element.
//!
//! A measurement that scored SEVERAL encodings per tensor is not a ranking at
//! all: it is a menu, and [`super::ladder`] chooses one encoding per tensor
//! from it under the same byte ceiling. That path allocates the whole profile
//! and returns, so nothing below it runs.
//!
//! The selection is all this module does. Both rankings live in
//! [`crate::policy`], and the record fields the measured path fills are
//! written by [`super::record`].

use anyhow::{anyhow, Result};

use tcf_core::Encoding;

use super::encoding::TcfOptions;
use super::ladder::allocate_ladder_profile;
use super::plan::PlannedTensor;
use crate::policy::{
    select_bumped, select_bumped_measured, BumpCandidate, ScreenedTensor, SensitivityMeasurement,
};

/// Raise the profile's bump set from the base encoding.
///
/// Candidates are the quantized tensors the bump encoding also accepts. A
/// tensor stored raw has no base encoding to be bumped from, and one the bump
/// encoding refuses cannot carry it, so neither is ranked and neither counts
/// toward the element total the profile's fraction is taken of. The budget
/// is spent in elements, not in tensors: the ladder's rungs are bpw
/// numbers, and only elements track bytes.
///
/// The ranking has two forms. Without `sensitivity` it is the weight-error
/// screen: `PROFILES.md` Section 4 step 2 — the task-metric measurement that
/// separates a solved profile from a heuristic — did not run, so every
/// sensitivity field stays unset and the file is not profile-solved. With
/// `sensitivity` it is that measurement, ranked by delta per element, and the
/// deltas are recorded on the tensors for the writer to store.
///
/// # Errors
/// The measurement was taken at an encoding other than the profile's base, or
/// it does not name every bump candidate.
pub(super) fn allocate(
    planned: &mut [PlannedTensor],
    options: TcfOptions,
    sensitivity: Option<&SensitivityMeasurement>,
) -> Result<()> {
    let Some(profile) = options.profile else {
        return Ok(());
    };

    // A measurement that scored several encodings per tensor makes the base a
    // choice rather than a floor, so it allocates the whole profile — a flat
    // one included, which has no bump set to rank but can still demote.
    if let Some(measurement) = sensitivity {
        measurement.check_encoding(profile.base)?;
        if allocate_ladder_profile(planned, profile, measurement)? {
            return Ok(());
        }
    }

    let Some(bump) = profile.bump else {
        log::info!(
            "TCF profile {}: {:?} for every quantizable tensor, no bump set",
            profile.name,
            profile.base
        );
        return Ok(());
    };

    let candidates: Vec<ScreenedTensor> = planned
        .iter()
        .enumerate()
        .filter_map(|(index, plan)| {
            plan.screen_error
                .filter(|_| plan.bumpable)
                .map(|relative_rms| ScreenedTensor {
                    index,
                    relative_rms,
                    elements: plan.elements,
                })
        })
        .collect();
    let candidate_count = candidates.len();
    let candidate_elements: u64 = candidates.iter().map(|c| c.elements).sum();
    let (bumped, excluded_nonpositive) = match sensitivity {
        Some(measurement) => {
            measurement.check_encoding(profile.base)?;
            let named = name_candidates(planned, &candidates)?;
            let selected = select_bumped_measured(measurement, &named, profile.bump_fraction)?;
            (selected.bumped, Some(selected.excluded_nonpositive))
        }
        None => (select_bumped(&candidates, profile.bump_fraction), None),
    };

    let mut bumped_elements: u64 = 0;
    for index in &bumped {
        let plan = planned
            .get_mut(*index)
            .ok_or_else(|| anyhow!("Allocation named tensor {index}, which was never planned"))?;
        plan.encoding = Encoding::Native(bump);
        plan.bumped = true;
        bumped_elements = bumped_elements.saturating_add(plan.elements);
    }

    let share = if candidate_elements == 0 {
        0.0
    } else {
        bumped_elements as f64 * 100.0 / candidate_elements as f64
    };
    log::info!(
        "TCF profile {}: base {:?}, {} of {candidate_count} quantizable tensors bumped to \
         {bump:?}, {share:.1}% of quantizable parameters against a {:.0}% budget",
        profile.name,
        profile.base,
        bumped.len(),
        profile.bump_fraction * 100.0
    );
    match (sensitivity, excluded_nonpositive) {
        (Some(measurement), Some(excluded)) => {
            for plan in planned.iter_mut() {
                plan.sensitivity_delta = measurement.delta_for(&plan.name);
            }
            log::info!(
                "TCF profile {}: the bump set was ranked by a MEASURED task-metric delta per \
                 element, from a sensitivity file of {} tensors measured over {} evaluation \
                 rows at {:?}. {excluded} of {candidate_count} candidates were excluded for a \
                 delta that is not positive: the base encoding did them no measurable damage. \
                 sensitivity_delta, sensitivity_samples and calibration_id are written from \
                 that measurement; sensitivity_ci95 stays unset, because it carries no \
                 confidence interval.",
                profile.name,
                measurement.measured(),
                measurement.baseline().eval_rows,
                profile.base
            );
        }
        _ => log::info!(
            "TCF profile {}: the bump set was ranked by a weight-error screen — relative RMS of \
             the dequantized weights against the source. No task-metric calibration ran, so \
             sensitivity_delta, sensitivity_ci95, sensitivity_samples and calibration_id stay \
             unset and this file is NOT profile-solved per PROFILES.md Section 4.",
            profile.name
        ),
    }
    Ok(())
}

/// Pair each screened candidate with the tensor name the measurement is keyed
/// by.
///
/// # Errors
/// A candidate names a plan index that does not exist, which the screen and
/// the plan list disagreeing would be.
fn name_candidates<'a>(
    planned: &'a [PlannedTensor],
    candidates: &[ScreenedTensor],
) -> Result<Vec<BumpCandidate<'a>>> {
    candidates
        .iter()
        .map(|candidate| {
            let plan = planned.get(candidate.index).ok_or_else(|| {
                anyhow!(
                    "Allocation named tensor {}, which was never planned",
                    candidate.index
                )
            })?;
            Ok(BumpCandidate {
                index: candidate.index,
                name: plan.name.as_str(),
                elements: candidate.elements,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::profile::{TCF_COMPACT, TCF_HIGH, TCF_REF};
    use super::*;
    use crate::loader::UnifiedDType;
    use crate::policy::FallbackReason;
    use tcf_core::{NativeEncoding, RawEncoding};

    fn planned(
        encoding: Encoding,
        bumped: bool,
        fallback: Option<FallbackReason>,
    ) -> PlannedTensor {
        PlannedTensor {
            name: "w".to_string(),
            dtype: UnifiedDType::F32,
            fallback,
            encoding,
            screen_error: fallback.is_none().then_some(0.1),
            elements: 64,
            bumpable: fallback.is_none(),
            accepts: Vec::new(),
            bumped,
            demoted: false,
            sensitivity_delta: None,
        }
    }

    /// The budget counts the quantizable elements alone, and the raw tensor
    /// here is larger than both candidates together.
    #[test]
    fn allocation_excludes_a_raw_tensor_from_the_ranking_and_the_budget() {
        let mut tensors = vec![
            planned(Encoding::Native(NativeEncoding::Q4AS32DT64), false, None),
            planned(Encoding::Native(NativeEncoding::Q4AS32DT64), false, None),
            planned(
                Encoding::Raw(RawEncoding::F32),
                false,
                Some(FallbackReason::Rank),
            ),
        ];
        tensors[0].screen_error = Some(0.4);
        tensors[1].screen_error = Some(0.9);
        tensors[2].elements = 4096;

        allocate(&mut tensors, TcfOptions::from_profile(TCF_COMPACT), None).expect("allocates");

        assert!(!tensors[0].bumped);
        assert!(tensors[1].bumped, "the worst-reconstructed candidate");
        assert_eq!(
            tensors[1].encoding,
            Encoding::Native(NativeEncoding::Q6S16DT64)
        );
        assert!(!tensors[2].bumped, "a raw tensor is no candidate");
        assert_eq!(tensors[2].encoding, Encoding::Raw(RawEncoding::F32));
    }

    /// A tensor the bump encoding cannot represent stays at the base, and is
    /// no more a candidate than a raw one is.
    #[test]
    fn a_tensor_the_bump_encoding_refuses_is_no_candidate() {
        let base = Encoding::Native(NativeEncoding::Q4AS32DT64);
        let mut tensors = vec![planned(base, false, None), planned(base, false, None)];
        tensors[0].screen_error = Some(0.9);
        tensors[0].bumpable = false;
        tensors[1].screen_error = Some(0.1);

        allocate(&mut tensors, TcfOptions::from_profile(TCF_COMPACT), None).expect("allocates");

        assert!(!tensors[0].bumped, "the bump encoding refuses these values");
        assert!(tensors[1].bumped);
    }

    fn measurement(deltas: &[(&str, f64)]) -> crate::policy::SensitivityMeasurement {
        let mut lines = vec![
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4AS32D_T64","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#
                .to_string(),
        ];
        for (name, delta) in deltas {
            lines.push(format!(
                r#"{{"record":"sensitivity","name":"{name}","delta":{delta}}}"#
            ));
        }
        crate::policy::parse_sensitivity(&lines.join("\n")).expect("parses")
    }

    fn two_candidates() -> Vec<PlannedTensor> {
        let base = Encoding::Native(NativeEncoding::Q4AS32DT64);
        let mut tensors = vec![planned(base, false, None), planned(base, false, None)];
        tensors[0].name = "a.weight".to_string();
        tensors[1].name = "b.weight".to_string();
        // The weight-error screen would take `a.weight`, which reconstructs
        // worst. The measurement below disagrees, and it wins.
        tensors[0].screen_error = Some(0.9);
        tensors[1].screen_error = Some(0.1);
        tensors
    }

    /// A file measuring ONE encoding per tensor is the two-way allocation,
    /// unchanged: the ranked subset is raised to the bump encoding and nothing
    /// is demoted. This is the compatibility guarantee the ladder keeps.
    #[test]
    fn a_single_encoding_measurement_takes_the_two_way_path() {
        let mut tensors = two_candidates();
        let measured = measurement(&[("a.weight", 0.1), ("b.weight", 0.4)]);
        assert!(!measured.is_ladder());

        allocate(
            &mut tensors,
            TcfOptions::from_profile(TCF_COMPACT),
            Some(&measured),
        )
        .expect("allocates");

        // The file damages `b.weight` most per element, so it is the one
        // bumped, and no two-way allocation demotes anything.
        assert!(!tensors[0].bumped);
        assert!(tensors[1].bumped);
        assert!(!tensors[0].demoted);
        assert!(!tensors[1].demoted);
    }

    /// A measured run ranks by the file, not by the screen, and records every
    /// delta the file names.
    #[test]
    fn a_measurement_ranks_the_bump_set_and_is_recorded() {
        let mut tensors = two_candidates();
        let measured = measurement(&[("a.weight", -0.1), ("b.weight", 0.4)]);

        allocate(
            &mut tensors,
            TcfOptions::from_profile(TCF_COMPACT),
            Some(&measured),
        )
        .expect("allocates");

        assert!(!tensors[0].bumped, "the measurement shows no damage");
        assert!(tensors[1].bumped);
        assert_eq!(
            tensors[1].encoding,
            Encoding::Native(NativeEncoding::Q6S16DT64)
        );
        assert_eq!(tensors[0].sensitivity_delta, Some(-0.1));
        assert_eq!(tensors[1].sensitivity_delta, Some(0.4));
    }

    /// A candidate the measurement does not name stops the allocation, rather
    /// than falling back to the screen for that one tensor.
    #[test]
    fn an_unmeasured_candidate_fails_the_allocation() {
        let mut tensors = two_candidates();
        let measured = measurement(&[("a.weight", 0.4)]);

        let error = allocate(
            &mut tensors,
            TcfOptions::from_profile(TCF_COMPACT),
            Some(&measured),
        )
        .expect_err("an unmeasured candidate")
        .to_string();

        assert!(error.contains("1 of 2 bump candidates"), "{error}");
        assert!(error.contains("'b.weight'"), "{error}");
    }

    /// A measurement taken at another encoding is refused before anything is
    /// bumped.
    #[test]
    fn a_measurement_at_another_encoding_fails_the_allocation() {
        let mut tensors = two_candidates();
        let measured = measurement(&[("a.weight", 0.4), ("b.weight", 0.1)]);

        // `TCF-REF` is flat, so `TCF-HIGH` is the profile with a bump whose
        // base is not the encoding this file was measured at.
        let error = allocate(
            &mut tensors,
            TcfOptions::from_profile(TCF_HIGH),
            Some(&measured),
        )
        .expect_err("another encoding")
        .to_string();

        assert!(error.contains("does not transfer"), "{error}");
        assert!(!tensors[0].bumped);
        assert!(!tensors[1].bumped);
    }

    /// `TCF-REF` names no bump encoding, so every tensor keeps the base.
    #[test]
    fn a_flat_profile_bumps_nothing() {
        let mut tensors = vec![planned(
            Encoding::Native(NativeEncoding::Q8S32T64),
            false,
            None,
        )];
        allocate(&mut tensors, TcfOptions::from_profile(TCF_REF), None).expect("allocates");
        assert!(!tensors[0].bumped);
    }

    /// A `--tcf-encoding` run has no profile, so nothing is bumped and the
    /// encoding the flag named stands.
    #[test]
    fn a_run_without_a_profile_bumps_nothing() {
        let options = TcfOptions::uniform(NativeEncoding::Q4S32T64, NativeEncoding::Q4S32T64);
        let mut tensors = vec![planned(
            Encoding::Native(NativeEncoding::Q4S32T64),
            false,
            None,
        )];
        allocate(&mut tensors, options, None).expect("allocates");
        assert!(!tensors[0].bumped);
    }
}
