//! What one TCF conversion quantizes into, and the objective it minimizes.
//!
//! [`super::encoding`] owns encoding IDENTITY — the names, the parser, the
//! shape and value readers. This module owns the CHOICE a run makes out of
//! them, which is the pair `(encoding, objective)` every quantizer call needs.
//!
//! # The objective is per tensor, never per file
//!
//! A profile mixes encodings: `TCF-COMPACT` is a 4-bit base with a 6-bit
//! bump, `TCF-HIGH` a 6-bit base with an 8-bit bump, and a ladder allocation
//! can put a tensor at any measured rung. So [`TcfOptions::objective_for`]
//! takes the encoding of the TENSOR being quantized, and every call site
//! passes the encoding its record already fixed. Resolving once per file
//! would apply one half of a mixed file's answer to the other half.

use tcf_core::NativeEncoding;

use super::encoding_table::default_objective_for;
use super::objective::{ObjectiveSource, TcfObjective};
use super::profile::TcfProfile;

/// What a TCF conversion quantizes into, and what its module says it prefers.
///
/// Both are explicit. `preferred_encoding` has no source of truth in a
/// checkpoint, and TCF reads it to decide which tensors carry a
/// `fallback_reason`, so compressr requires it rather than inventing one.
///
/// `--tcf-encoding` sets one encoding for every tensor. `--tcf-profile` sets
/// `encoding` and `preferred_encoding` to the profile's base and carries the
/// profile, which raises the screened tensors to its bump encoding.
///
/// The objective is orthogonal to both: an encoding fixes the BITS a tensor is
/// stored in, and the objective fixes WHICH VALUES the encoder picks within
/// them. It is resolved per tensor from that tensor's own encoding — see
/// [`TcfOptions::objective_for`] — because a profile mixes encodings and one
/// objective applied to a mixed file is one of them applied to the wrong half.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcfOptions {
    /// The encoding a quantizable tensor is written in unless a profile bumps it
    pub encoding: NativeEncoding,
    /// The module's highest-ranked `preferred_encoding`
    pub preferred_encoding: NativeEncoding,
    /// The profile driving a mixed-precision allocation, if one was named
    pub profile: Option<TcfProfile>,
    /// The objective `--tcf-objective` named, which OVERRIDES every
    /// per-encoding default
    ///
    /// `None` is the flagless run: every tensor takes the default its own
    /// encoding carries in the `encoding_table` module's one table.
    pub objective_override: Option<TcfObjective>,
}

impl TcfOptions {
    /// One encoding for every quantizable tensor, as `--tcf-encoding` sets it.
    pub fn uniform(encoding: NativeEncoding, preferred_encoding: NativeEncoding) -> Self {
        Self {
            encoding,
            preferred_encoding,
            profile: None,
            objective_override: None,
        }
    }

    /// A profile's allocation, as `--tcf-profile` sets it.
    ///
    /// The base encoding is also the module preference, so a tensor left at
    /// the base records no fallback reason and a bumped one records
    /// `TaskSensitivity`.
    pub fn from_profile(profile: TcfProfile) -> Self {
        Self {
            encoding: profile.base,
            preferred_encoding: profile.base,
            profile: Some(profile),
            objective_override: None,
        }
    }

    /// Set the objective every tensor is quantized under, whatever its
    /// encoding's default is.
    ///
    /// An explicit `--tcf-objective` always wins: a run that names one asked
    /// for that objective and gets it on every tensor of a mixed file.
    pub fn with_objective(mut self, objective: TcfObjective) -> Self {
        self.objective_override = Some(objective);
        self
    }

    /// Set the objective only when a run named one.
    ///
    /// `None` leaves every tensor on its own encoding's default.
    pub fn with_objective_override(mut self, objective: Option<TcfObjective>) -> Self {
        self.objective_override = objective;
        self
    }

    /// The objective ONE tensor is quantized under, from the encoding that
    /// tensor is written in.
    ///
    /// Call this with the tensor's own encoding, never with the run's base:
    /// under `TCF-COMPACT` a bumped tensor is 6-bit inside a 4-bit file, and
    /// resolving from the base would quantize it under the 4-bit default.
    pub fn objective_for(self, encoding: NativeEncoding) -> TcfObjective {
        self.objective_choice(encoding).0
    }

    /// The objective one tensor is quantized under, and where it came from.
    ///
    /// A log line has to say which, because a per-encoding default and an
    /// explicit flag that happen to name the same objective are two different
    /// runs.
    pub fn objective_choice(self, encoding: NativeEncoding) -> (TcfObjective, ObjectiveSource) {
        match self.objective_override {
            Some(objective) => (objective, ObjectiveSource::ExplicitFlag),
            None => (
                default_objective_for(encoding),
                ObjectiveSource::EncodingDefault,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flagless run names no objective at all: every tensor takes the
    /// default its own encoding carries.
    #[test]
    fn options_start_with_no_objective_override() {
        let flat = TcfOptions::uniform(NativeEncoding::Q4S32T64, NativeEncoding::Q4S32T64);
        assert_eq!(flat.objective_override, None);
        assert_eq!(
            TcfOptions::from_profile(super::super::profile::TCF_COMPACT).objective_override,
            None
        );
        assert_eq!(
            flat.with_objective(TcfObjective::Squared)
                .objective_override,
            Some(TcfObjective::Squared)
        );
        assert_eq!(flat.with_objective_override(None).objective_override, None);
    }

    /// A flagless run resolves the objective from the TENSOR's encoding, so
    /// the 4-bit half and the 6-bit half of one mixed file differ.
    #[test]
    fn a_flagless_run_resolves_the_objective_per_encoding() {
        let options = TcfOptions::from_profile(super::super::profile::TCF_COMPACT);
        assert_eq!(
            options.objective_choice(NativeEncoding::Q4AS32DT64),
            (TcfObjective::Uniform, ObjectiveSource::EncodingDefault)
        );
        assert_eq!(
            options.objective_choice(NativeEncoding::Q6S16DT64),
            (TcfObjective::RmsPlusAbs, ObjectiveSource::EncodingDefault)
        );
        assert_eq!(
            options.objective_for(NativeEncoding::Q8S32T64),
            TcfObjective::Uniform
        );
    }

    /// An explicit flag wins over every per-encoding default, on every
    /// encoding of a mixed file.
    #[test]
    fn an_explicit_objective_overrides_every_encoding_default() {
        let options = TcfOptions::from_profile(super::super::profile::TCF_COMPACT)
            .with_objective(TcfObjective::Squared);
        for encoding in [
            NativeEncoding::Q4S32T64,
            NativeEncoding::Q4AS32T64,
            NativeEncoding::Q4AS64T64,
            NativeEncoding::Q4AS32DT64,
            NativeEncoding::Q6S32T64,
            NativeEncoding::Q6S16DT64,
            NativeEncoding::Q8S32T64,
        ] {
            assert_eq!(
                options.objective_choice(encoding),
                (TcfObjective::Squared, ObjectiveSource::ExplicitFlag),
                "{encoding:?}"
            );
        }
        let uniform = TcfOptions::uniform(NativeEncoding::Q6S16DT64, NativeEncoding::Q6S16DT64)
            .with_objective(TcfObjective::Uniform);
        assert_eq!(
            uniform.objective_choice(NativeEncoding::Q6S16DT64),
            (TcfObjective::Uniform, ObjectiveSource::ExplicitFlag)
        );
    }
}
