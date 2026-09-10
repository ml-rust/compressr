//! What a run says about the objective it quantized under.
//!
//! Two lines, and they answer different questions. [`objective_policy`] is
//! written before the conversion and states the POLICY: whether the objective
//! came from the per-encoding defaults or from an explicit flag, which
//! encodings that policy covers, and — when a default moved — that an earlier
//! file no longer reproduces byte for byte. [`report_objectives`] is written
//! after the allocation and states the RESULT: the objective each encoding was
//! actually quantized under, with the tensor count behind it.
//!
//! A mixed-precision file must never be reported as though one objective was
//! used throughout, so neither line names a single objective when the file
//! holds more than one encoding.

use std::collections::BTreeMap;

use tcf_core::{Encoding, NativeEncoding};

use super::encoding::tcf_encoding_name;
use super::encoding_table::default_objective_for;
use super::objective::TcfObjective;
use super::options::TcfOptions;
use super::plan::PlannedTensor;

/// The one sentence every line adds about what an objective does not touch.
const CONFORMANCE: &str = "The objective decides which values the encoder chooses; it moves no \
                           bit layout, tile order, plane order, offset or digest, so this file \
                           is a normal conforming TCF v1 file and every reader decodes it the \
                           same way.";

/// The one sentence a run whose defaults changed the bytes has to state, so
/// nobody debugs the difference as a regression.
const REPRODUCIBILITY: &str =
    "Reproducibility: the 6-bit encodings default to a weighted objective, which a measurement \
     chose. A 6-bit TCF file written before that default does NOT reproduce byte for byte from \
     the same source; pass --tcf-objective uniform to write the earlier bytes. 4-bit and 8-bit \
     output is unchanged either way.";

/// How this run decided the objective, stated before it quantizes anything.
///
/// A profile is mixed precision, so the line names EVERY encoding the policy
/// covers with its own objective rather than a single answer.
pub fn objective_policy(options: TcfOptions) -> String {
    let encodings = planned_encodings(options);
    if let Some(objective) = options.objective_override {
        return format!(
            "TCF objective {} ({}), from an explicit --tcf-objective: it overrides the \
             per-encoding default on every tensor, whatever encoding that tensor is written in. \
             {CONFORMANCE}",
            objective.name(),
            objective.rule()
        );
    }
    let mut line = String::from(
        "TCF objective: none named, so every tensor takes the default its own encoding carries. ",
    );
    if encodings.len() > 1 {
        line.push_str("This file is MIXED precision, so no single objective was used throughout. ");
    }
    for encoding in &encodings {
        let objective = default_objective_for(*encoding);
        line.push_str(&format!(
            "{} quantizes under {} ({}). ",
            tcf_encoding_name(*encoding),
            objective.name(),
            objective.rule()
        ));
    }
    if options.profile.is_some() {
        line.push_str(
            "A --sensitivity ladder can place a tensor at another rung, which takes that rung's \
             own default in turn. ",
        );
    }
    line.push_str(CONFORMANCE);
    if encodings
        .iter()
        .any(|encoding| !default_objective_for(*encoding).is_uniform())
    {
        line.push(' ');
        line.push_str(REPRODUCIBILITY);
    }
    line
}

/// Every encoding this run's flags can write, base first.
///
/// A ladder allocation can reach a rung neither flag names, which is why the
/// policy line says so and [`report_objectives`] states what was written.
fn planned_encodings(options: TcfOptions) -> Vec<NativeEncoding> {
    let mut encodings = vec![options.encoding];
    if let Some(bump) = options.profile.and_then(|profile| profile.bump) {
        if !encodings.contains(&bump) {
            encodings.push(bump);
        }
    }
    encodings
}

/// The objective each encoding was actually quantized under, once the
/// allocation has fixed every tensor's encoding.
///
/// This is the line that proves the resolution ran per tensor: a mixed file
/// reports one row per encoding, and a row's objective is that encoding's,
/// not the base encoding's.
pub(super) fn report_objectives(planned: &[PlannedTensor], options: TcfOptions) {
    let counts = objective_counts(planned, options);
    if counts.is_empty() {
        // Every tensor fell back to a raw encoding, so nothing was quantized
        // and no objective was applied to anything.
        log::info!("TCF objective: no tensor was quantized, so no objective was applied");
        return;
    }
    let source = options.objective_choice(options.encoding).1;
    let rows: Vec<String> = counts
        .iter()
        .map(|((name, objective), tensors)| {
            format!("{name} under {objective} on {tensors} tensor(s)")
        })
        .collect();
    log::info!(
        "TCF objective resolved per tensor from {}: {}. {}",
        source.label(),
        rows.join(", "),
        if counts.len() > 1 {
            "This file is mixed precision, so no single objective was used throughout."
        } else {
            "Every quantized tensor took the same encoding, so one objective covers the file."
        }
    );
    if !source.is_explicit()
        && counts
            .keys()
            .any(|(_, objective)| *objective != TcfObjective::Uniform.name())
    {
        log::info!("{REPRODUCIBILITY}");
    }
}

/// One count per (encoding name, objective name) pair, in name order.
fn objective_counts(
    planned: &[PlannedTensor],
    options: TcfOptions,
) -> BTreeMap<(&'static str, &'static str), usize> {
    let mut counts: BTreeMap<(&'static str, &'static str), usize> = BTreeMap::new();
    for plan in planned {
        let Encoding::Native(encoding) = plan.encoding else {
            // A raw tensor is stored verbatim: no quantizer ran on it, so it
            // was minimized under no objective at all.
            continue;
        };
        let objective = options.objective_for(encoding);
        *counts
            .entry((tcf_encoding_name(encoding), objective.name()))
            .or_default() += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::super::profile::{TCF_BALANCED, TCF_COMPACT, TCF_HIGH};
    use super::*;
    use crate::loader::UnifiedDType;

    fn planned(encoding: Encoding) -> PlannedTensor {
        PlannedTensor {
            name: "w".to_string(),
            dtype: UnifiedDType::F32,
            fallback: None,
            encoding,
            screen_error: Some(0.1),
            elements: 64,
            bumpable: true,
            accepts: Vec::new(),
            bumped: false,
            demoted: false,
            sensitivity_delta: None,
        }
    }

    /// A mixed profile resolves the objective per tensor, so its 4-bit half
    /// and its 6-bit half are counted under different objectives.
    #[test]
    fn a_mixed_profile_counts_one_objective_per_encoding() {
        let options = TcfOptions::from_profile(TCF_COMPACT);
        let tensors = vec![
            planned(Encoding::Native(NativeEncoding::Q4AS32DT64)),
            planned(Encoding::Native(NativeEncoding::Q4AS32DT64)),
            planned(Encoding::Native(NativeEncoding::Q6S16DT64)),
            planned(Encoding::Raw(tcf_core::RawEncoding::F32)),
        ];
        let counts = objective_counts(&tensors, options);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.get(&("Q4AS32D_T64", "uniform")), Some(&2));
        assert_eq!(counts.get(&("Q6S16D_T64", "rms-plus-abs")), Some(&1));
    }

    /// An explicit flag collapses the rows to one objective, on every
    /// encoding of the same mixed file.
    #[test]
    fn an_explicit_flag_covers_every_encoding_of_a_mixed_file() {
        let options = TcfOptions::from_profile(TCF_COMPACT).with_objective(TcfObjective::Squared);
        let tensors = vec![
            planned(Encoding::Native(NativeEncoding::Q4AS32DT64)),
            planned(Encoding::Native(NativeEncoding::Q6S16DT64)),
        ];
        let counts = objective_counts(&tensors, options);
        assert_eq!(counts.get(&("Q4AS32D_T64", "squared")), Some(&1));
        assert_eq!(counts.get(&("Q6S16D_T64", "squared")), Some(&1));
    }

    /// The policy line of a mixed file names both encodings and refuses to
    /// imply one objective covered it.
    #[test]
    fn the_policy_line_of_a_mixed_file_names_every_encoding() {
        let line = objective_policy(TcfOptions::from_profile(TCF_COMPACT));
        assert!(line.contains("MIXED precision"), "{line}");
        assert!(
            line.contains("Q4AS32D_T64 quantizes under uniform"),
            "{line}"
        );
        assert!(
            line.contains("Q6S16D_T64 quantizes under rms-plus-abs"),
            "{line}"
        );
        assert!(line.contains("does NOT reproduce byte for byte"), "{line}");
    }

    /// A file whose encodings all default to uniform states no
    /// reproducibility warning, because its bytes did not move.
    #[test]
    fn a_four_bit_and_eight_bit_file_warns_about_no_moved_bytes() {
        let line = objective_policy(TcfOptions::from_profile(TCF_BALANCED));
        assert!(!line.contains("does NOT reproduce"), "{line}");
        let flat = objective_policy(TcfOptions::uniform(
            NativeEncoding::Q8S32T64,
            NativeEncoding::Q8S32T64,
        ));
        assert!(!flat.contains("does NOT reproduce"), "{flat}");
        assert!(!flat.contains("MIXED precision"), "{flat}");
        assert!(flat.contains("Q8S32_T64 quantizes under uniform"), "{flat}");
    }

    /// An explicit flag says so, and says it overrode the defaults.
    #[test]
    fn the_policy_line_names_an_explicit_flag() {
        let line = objective_policy(
            TcfOptions::from_profile(TCF_HIGH).with_objective(TcfObjective::Uniform),
        );
        assert!(line.contains("explicit --tcf-objective"), "{line}");
        assert!(
            line.contains("overrides the per-encoding default"),
            "{line}"
        );
        assert!(!line.contains("does NOT reproduce byte for byte"), "{line}");
    }
}
