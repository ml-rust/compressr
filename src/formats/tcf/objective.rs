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
//! `tcf_core::WeightSource::Explicit` is a per-ELEMENT slice parallel to the
//! values of one call, and `tcf_core::column_weights` expands a per-COLUMN
//! importance vector onto it. That is a fourth variant of [`TcfObjective`]
//! carrying the vector, resolved to a `WeightSource` per call rather than
//! once — see [`TcfObjective::weights`], which is the single point every call
//! site already goes through.

use anyhow::{bail, Result};
use tcf_core::WeightSource;

/// The `--tcf-objective` values, in the order the CLI help lists them.
pub const TCF_OBJECTIVE_NAMES: [&str; 3] = ["uniform", "squared", "rms-plus-abs"];

/// Which error objective a TCF conversion's quantizer minimizes.
///
/// Every variant is VALUE-DERIVED: its weights come from the group's own
/// values, so nothing outside the group is read and no caller supplies data.
/// A group never spans two tiles and a chunk never splits a tile, so the
/// choice is chunk-invariant: the threaded path scores what a whole-tensor
/// call scores.
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
    /// The `tcf-core` weight source this objective names.
    ///
    /// Every quantizer call site resolves its source here, so an objective
    /// that carried data — an imatrix vector — would be expanded in this one
    /// function and reach every call site unchanged.
    pub fn weights(self) -> WeightSource<'static> {
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
        assert!(TcfObjective::default().weights().is_uniform());
        assert!(!TcfObjective::Squared.is_uniform());
        assert!(!TcfObjective::Squared.weights().is_uniform());
        assert!(!TcfObjective::RmsPlusAbs.weights().is_uniform());
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
