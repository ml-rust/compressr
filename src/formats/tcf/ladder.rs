//! A TCF profile allocated from a per-encoding measurement.
//!
//! [`super::allocate`] raises a ranked subset of the quantizable tensors from
//! the profile's base encoding to its bump encoding, and every other tensor
//! pays the base price whatever the measurement says about it. This module
//! spends the same budget over a MENU instead: every encoding the sensitivity
//! file scored for a tensor is an option, and one of them is chosen. A tensor
//! measured as undamaged at a cheaper encoding lands there, and the bytes that
//! frees pay for a tensor the measurement does show damage on.
//!
//! # Where the numbers come from
//!
//! - The DELTA of each option is the file's, at that (tensor, encoding) pair.
//!   Nothing is extrapolated from another encoding's delta.
//! - The BYTES of each option are `tcf-core`'s, from the same layout the
//!   record and the payload pass use. A budget spent in numbers the writer
//!   does not agree with is not a budget. The file's own `bytes` field is
//!   compared against them and a disagreement is reported.
//! - The set of encodings a tensor may take is [`super::screen`]'s, asked of
//!   the tensor's values before any record is fixed.
//!
//! # The budget
//!
//! `PROFILES.md` Section 2 gives a profile a bump fraction — the share of
//! quantizable parameters the bump encoding covers. The two-way allocation
//! spends that in elements. The same quantity in bytes is what the bump
//! fraction would cost if it were spent on the bump encoding, which is what
//! this module allows the ladder to spend above an all-base allocation. The
//! ceiling on the artifact is therefore the one the two-way allocation would
//! have written, and the two stay comparable.

use anyhow::Result;

use tcf_core::NativeEncoding;

use super::ladder_menu::allocate_profile_ladder;
use super::plan::PlannedTensor;
use super::profile::TcfProfile;
use crate::formats::parse_tcf_encoding;
use crate::policy::{examples, SensitivityMeasurement};

/// Every encoding a measurement scores, in the order `tcf-core` names them.
///
/// An encoding compressr does not recognize is left out: it cannot be written,
/// so it can be no rung. The list is what each tensor's values are screened
/// against, so it is derived once from the file and never per tensor.
pub(super) fn ladder_encodings(measurement: &SensitivityMeasurement) -> Vec<NativeEncoding> {
    let mut encodings: Vec<NativeEncoding> = Vec::new();
    let mut unknown: Vec<&str> = Vec::new();
    for name in measurement.encodings() {
        match parse_tcf_encoding(name) {
            Ok(encoding) if !encodings.contains(&encoding) => encodings.push(encoding),
            Ok(_) => {}
            Err(_) => unknown.push(name),
        }
    }
    if !unknown.is_empty() {
        log::warn!(
            "Sensitivity file measures {} encoding(s) compressr cannot write, so no tensor is \
             allocated to them: {}",
            unknown.len(),
            examples(&unknown)
        );
    }
    encodings.sort_by_key(|encoding| encoding.to_u16());
    encodings
}

/// Allocate the profile from the measurement's menu, when it carries one.
///
/// Returns `false` when the file measures one encoding, which is the two-way
/// allocation's input and not a menu at all; the caller then runs that
/// allocation unchanged.
///
/// # Errors
/// Whatever [`allocate_profile_ladder`] refuses.
pub(super) fn allocate_ladder_profile(
    planned: &mut [PlannedTensor],
    profile: TcfProfile,
    measurement: &SensitivityMeasurement,
) -> Result<bool> {
    if !can_ladder(profile, measurement) {
        return Ok(false);
    }
    let report = allocate_profile_ladder(planned, profile, measurement)?;
    log::info!("{report}");
    log::info!(
        "TCF profile {}: every tensor's encoding was chosen from MEASURED options — one \
         task-metric delta per (tensor, encoding) pair, from a file of {} tensors measured over \
         {} evaluation rows. sensitivity_delta is the delta of the encoding each tensor landed \
         on; sensitivity_samples and calibration_id come from the same file, and \
         sensitivity_ci95 stays unset, because it carries no confidence interval.",
        profile.name,
        measurement.measured(),
        measurement.baseline().eval_rows
    );
    Ok(true)
}

/// Whether a ladder allocation can run for this profile and this measurement.
///
/// It needs a menu — more than one encoding — and it needs the profile's base
/// among them: the base is the rung every choice is reported against, and the
/// one the two-way allocation would have written.
fn can_ladder(profile: TcfProfile, measurement: &SensitivityMeasurement) -> bool {
    measurement.is_ladder()
        && measurement
            .encodings()
            .iter()
            .any(|name| parse_tcf_encoding(name).is_ok_and(|encoding| encoding == profile.base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::parse_sensitivity;

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
            ladder_encodings(&measured),
            vec![NativeEncoding::Q4AS32DT64, NativeEncoding::Q6S16DT64]
        );
    }

    /// A file measured at one encoding is no menu, so the ladder stands down
    /// and the two-way allocation runs. This is the compatibility guarantee.
    #[test]
    fn a_file_measured_at_one_encoding_does_not_ladder() {
        let two_way = measurement(&[("a.weight", "Q4AS32D_T64", 0.5)]);
        assert!(!can_ladder(super::super::profile::TCF_COMPACT, &two_way));
        let mut plans: Vec<PlannedTensor> = Vec::new();
        assert!(
            !allocate_ladder_profile(&mut plans, super::super::profile::TCF_COMPACT, &two_way)
                .expect("stands down")
        );
    }

    /// A profile whose base the file never measured has no rung to stand on.
    #[test]
    fn a_profile_whose_base_is_unmeasured_does_not_ladder() {
        let measured = measurement(&[
            ("a.weight", "Q4AS32D_T64", 0.5),
            ("a.weight", "Q6S16D_T64", 0.1),
        ]);
        // `TCF-REF` is written at Q8S32_T64, which this file never measured.
        assert!(!can_ladder(super::super::profile::TCF_REF, &measured));
        assert!(can_ladder(super::super::profile::TCF_COMPACT, &measured));
    }
}
