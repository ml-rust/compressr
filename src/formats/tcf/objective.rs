//! The `--tcf-objective` choice: WHICH error the TCF quantizer minimizes.
//!
//! `tcf-core` scores every scale search, least-squares refit and sub-level
//! refinement by a weighted squared reconstruction error, and
//! `tcf_core::WeightSource` names how the per-element weight `w_i` is
//! derived. This module is the CLI's name for that choice and nothing else:
//! the weighting itself, like every bit position, lives in `tcf-core`.
//!
//! # What a non-uniform objective does and does not change
//!
//! It changes which values the encoder CHOOSES to store. It changes no bit
//! layout, no tile order, no plane order, no offset and no digest, so a file
//! written under any objective is a normal conforming TCF v1 file and every
//! reader decodes it the same way.
//!
//! # Where the default comes from
//!
//! There is no ONE default. Each encoding carries its own, in the single
//! table in [`super::encoding_table`], because a measurement found the best
//! objective differs by bit width. `--tcf-objective` OVERRIDES that default
//! on every tensor: a run that names an objective asked for it and gets it.
//! `TcfOptions::objective_choice` resolves the pair, per tensor, from that
//! tensor's own encoding.
//!
//! # Where an imatrix attaches
//!
//! Not here. [`TcfObjective`] is a POLICY SELECTOR: it names a rule and
//! carries no data, so it stays free of a lifetime and every type that stores
//! it — `TcfOptions`, `ConversionConfig`, the public API — stays free of one
//! too. An importance is data, per-COLUMN where `tcf-core` wants per-ELEMENT,
//! so it is expanded at the point a tensor is quantized into the owning
//! [`super::objective_weights::ResolvedWeights`], which
//! [`super::importance::weights_for`] returns for both call sites.
//!
//! # An importance takes one of two forms, and `--tcf-importance` picks it
//!
//! llama.cpp splits rather than settling it: Q3_K, Q4_K and Q5_K multiply the
//! importance by a data-derived term, and Q6_K passes the RAW importance.
//! `tcf-core` defines both rules beside the bit positions —
//! `WeightSource::Combined` and `WeightSource::Explicit` — so compressr
//! selects one rather than computing a codec semantic of its own.
//!
//! The product is the default, because the encodings compressr ships at are
//! the 4-bit and 6-bit tiers whose ggml counterparts are the K-quants.
//! [`super::importance_form`] owns the choice and the flag.
//!
//! A tensor the calibration run never exercised takes NEITHER form: it keeps
//! the objective its encoding carries, or the one `--tcf-objective` named.

use anyhow::{bail, Result};
use tcf_core::WeightSource;

/// The `--tcf-objective` values, in the order the CLI help lists them.
pub const TCF_OBJECTIVE_NAMES: [&str; 3] = ["uniform", "squared", "rms-plus-abs"];

/// Which error objective a TCF conversion's quantizer minimizes.
///
/// Every variant is VALUE-DERIVED: its weights come from the group's own
/// values, so nothing outside the group is read and no caller supplies data.
/// A group never spans two tiles and a chunk never splits a tile, so all
/// three are chunk-invariant: the threaded path scores what a whole-tensor
/// call scores.
///
/// It carries no data on purpose. An importance matrix is data, and it
/// reaches the quantizer through
/// [`super::objective_weights::ResolvedWeights`] instead — see the module
/// documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TcfObjective {
    /// `w_i = 1`: the plain squared reconstruction error.
    ///
    /// The `Default`, which is the objective an encoding with no table row
    /// falls back to; a run's actual objective comes from its encoding's row
    /// or from `--tcf-objective`, never from here.
    #[default]
    Uniform,
    /// `w_i = x_i^2`, the weighting ggml applies on its 6-bit path.
    Squared,
    /// `w_i = sigma + |x_i|`, with `sigma` the group's root mean square.
    RmsPlusAbs,
}

impl TcfObjective {
    /// The `tcf-core` source that names this rule.
    ///
    /// It carries nothing and allocates nothing, which is why it is
    /// `'static`: the weight is read from each group's own values inside
    /// `tcf-core`.
    pub fn source(self) -> WeightSource<'static> {
        match self {
            Self::Uniform => WeightSource::Uniform,
            Self::Squared => WeightSource::Squared,
            Self::RmsPlusAbs => WeightSource::RmsPlusAbs,
        }
    }

    /// Whether this is the unweighted objective, `w = 1`.
    pub fn is_uniform(self) -> bool {
        self == Self::Uniform
    }

    /// The one fixed spelling a report and a log line use for this objective.
    pub fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Squared => "squared",
            Self::RmsPlusAbs => "rms-plus-abs",
        }
    }

    /// The weight rule itself, for a line that has to say what was minimized
    /// rather than only name it.
    pub fn rule(self) -> &'static str {
        match self {
            Self::Uniform => "w = 1, the plain squared reconstruction error",
            Self::Squared => "w = x^2",
            Self::RmsPlusAbs => "w = sqrt(mean(x^2)) + |x| over each group",
        }
    }
}

/// Parse a `--tcf-objective` name.
///
/// Case, underscores and hyphens are ignored, exactly as they are for an
/// encoding name, so `rms-plus-abs`, `RMS_PLUS_ABS` and `rmsplusabs` are the
/// same objective.
///
/// An importance has no name here on purpose: it is a vector, it comes from
/// `--imatrix` and never from a flag value, and it is not an objective.
///
/// # Errors
/// The name is not one of [`TCF_OBJECTIVE_NAMES`].
pub fn parse_tcf_objective(s: &str) -> Result<TcfObjective> {
    let normalized = super::encoding::normalized_encoding_name(s);
    match normalized.as_str() {
        "uniform" => Ok(TcfObjective::Uniform),
        "squared" => Ok(TcfObjective::Squared),
        "rmsplusabs" => Ok(TcfObjective::RmsPlusAbs),
        _ => bail!(
            "Unsupported TCF objective: {}. Supported: {}",
            s,
            TCF_OBJECTIVE_NAMES.join(", ")
        ),
    }
}

/// Where one tensor's objective came from.
///
/// A per-encoding default and an explicit flag that name the same objective
/// are two different runs: the first moves when the table moves, the second
/// never does. A report that only named the objective could not tell them
/// apart, so every line that names one names this beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSource {
    /// The default the tensor's own encoding carries
    EncodingDefault,
    /// `--tcf-objective`, which overrides every per-encoding default
    ExplicitFlag,
}

impl ObjectiveSource {
    /// The one fixed phrase a report and a log line use for this source.
    pub fn label(self) -> &'static str {
        match self {
            Self::EncodingDefault => "the per-encoding default",
            Self::ExplicitFlag => "an explicit --tcf-objective",
        }
    }

    /// Whether an explicit flag chose this objective.
    pub fn is_explicit(self) -> bool {
        self == Self::ExplicitFlag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `Default` is the unweighted objective, which is what an encoding
    /// with no table row falls back to.
    #[test]
    fn the_fallback_objective_is_uniform() {
        assert_eq!(TcfObjective::default(), TcfObjective::Uniform);
        assert!(TcfObjective::default().is_uniform());
        assert!(TcfObjective::default().source().is_uniform());
        assert!(!TcfObjective::Squared.is_uniform());
        assert!(!TcfObjective::Squared.source().is_uniform());
        assert!(!TcfObjective::RmsPlusAbs.source().is_uniform());
    }

    /// Every objective names a rule `tcf-core` derives from the group's own
    /// values, so none of them carries data.
    #[test]
    fn every_objective_resolves_to_a_derived_source() {
        assert!(matches!(
            TcfObjective::Uniform.source(),
            WeightSource::Uniform
        ));
        assert!(matches!(
            TcfObjective::Squared.source(),
            WeightSource::Squared
        ));
        assert!(matches!(
            TcfObjective::RmsPlusAbs.source(),
            WeightSource::RmsPlusAbs
        ));
    }

    #[test]
    fn parses_every_objective_name() {
        assert_eq!(
            parse_tcf_objective("uniform").unwrap(),
            TcfObjective::Uniform
        );
        assert_eq!(
            parse_tcf_objective("squared").unwrap(),
            TcfObjective::Squared
        );
        assert_eq!(
            parse_tcf_objective("rms-plus-abs").unwrap(),
            TcfObjective::RmsPlusAbs
        );
    }

    /// Every name the CLI help lists parses, and names itself back.
    #[test]
    fn a_name_round_trips_through_the_parser() {
        for name in TCF_OBJECTIVE_NAMES {
            let objective = parse_tcf_objective(name).expect("parses");
            assert_eq!(objective.name(), name);
        }
    }

    /// Case, underscores and hyphens never separate two spellings of one
    /// objective.
    #[test]
    fn one_objective_has_one_normalized_name() {
        assert_eq!(
            parse_tcf_objective("RMS_PLUS_ABS").unwrap(),
            TcfObjective::RmsPlusAbs
        );
        assert_eq!(
            parse_tcf_objective("rmsplusabs").unwrap(),
            TcfObjective::RmsPlusAbs
        );
    }

    /// An unknown name is refused naming the whole accepted set, so the error
    /// answers the question it raises.
    #[test]
    fn rejects_an_unknown_objective_name() {
        let err = parse_tcf_objective("imatrix").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF objective: imatrix"), "{err}");
        for name in TCF_OBJECTIVE_NAMES {
            assert!(err.contains(name), "{err}");
        }
    }

    /// A source names itself one way, and says whether a flag chose it.
    #[test]
    fn a_source_names_itself() {
        assert!(ObjectiveSource::ExplicitFlag.is_explicit());
        assert!(!ObjectiveSource::EncodingDefault.is_explicit());
        assert!(ObjectiveSource::EncodingDefault
            .label()
            .contains("per-encoding default"));
        assert!(ObjectiveSource::ExplicitFlag
            .label()
            .contains("--tcf-objective"));
    }
}
