//! The `--tcf-profile` ladder: a named mixed-precision allocation.
//!
//! A profile names three things — a base encoding, a bump encoding, and the
//! fraction of quantizable parameters the bump covers. `PROFILES.md` Section 2
//! fixes all three. compressr solves the allocation with the screen in
//! [`crate::policy::allocation`], or with a measured sensitivity file when
//! `--sensitivity` supplies one, and writes the result.
//!
//! # What a profile here is not
//!
//! `PROFILES.md` Section 4 solves a profile in six steps. Step 2 measures a
//! task-metric delta on a calibration set, and that measurement is what
//! separates a profile from a heuristic. compressr runs no model, so it never
//! takes that measurement: without `--sensitivity` step 2 does not run and
//! every sensitivity field stays unset. `--sensitivity` supplies the
//! measurement from a harness that did run the model, and the allocation is
//! ranked by it — see [`crate::policy::sensitivity`]. compressr still writes
//! no profile name into the file either way: a name is a claim about how the
//! whole allocation was solved, and steps 4 and 6 have no compressr path.
//! `ladder_bpw` is Section 2's arithmetic under Section 4's own allocation, so
//! it is reported as a target and never as an achieved size.

use anyhow::{bail, Result};
use tcf_core::NativeEncoding;

/// The `--tcf-profile` values, in ladder order.
pub const TCF_PROFILE_NAMES: [&str; 4] = ["ref", "high", "balanced", "compact"];

/// One rung of the ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcfProfile {
    /// The profile's spelling in `PROFILES.md`, e.g. `TCF-BALANCED`
    pub name: &'static str,
    /// The encoding every quantizable tensor starts at
    ///
    /// This is also the module's `preferred_encoding`, so a tensor left at the
    /// base records no fallback reason and a bumped one records
    /// `TaskSensitivity`.
    pub base: NativeEncoding,
    /// The encoding the bumped tensors are raised to, `None` for a flat profile
    pub bump: Option<NativeEncoding>,
    /// Share of quantizable parameters the bump covers, 0.0 to 1.0
    pub bump_fraction: f64,
    /// `PROFILES.md` Section 2's bpw for this rung
    ///
    /// A target, computed from Section 4's own allocation. The achieved bpw is
    /// measured from the bytes written and will differ.
    pub ladder_bpw: f64,
}

/// `TCF-REF`: one encoding, no bump.
pub const TCF_REF: TcfProfile = TcfProfile {
    name: "TCF-REF",
    base: NativeEncoding::Q8S32T64,
    bump: None,
    bump_fraction: 0.0,
    ladder_bpw: 8.50,
};

/// `TCF-HIGH`: `Q6S16D_T64` with `Q8S32_T64` on the top 15%.
///
/// `0.85 * 6.5625 + 0.15 * 8.5000 = 6.853125`.
pub const TCF_HIGH: TcfProfile = TcfProfile {
    name: "TCF-HIGH",
    base: NativeEncoding::Q6S16DT64,
    bump: Some(NativeEncoding::Q8S32T64),
    bump_fraction: 0.15,
    ladder_bpw: 6.853_125,
};

/// `TCF-BALANCED`: `Q4AS32D_T64` with `Q8S32_T64` on the top 15%.
///
/// `0.85 * 4.5000 + 0.15 * 8.5000 = 5.100000`.
pub const TCF_BALANCED: TcfProfile = TcfProfile {
    name: "TCF-BALANCED",
    base: NativeEncoding::Q4AS32DT64,
    bump: Some(NativeEncoding::Q8S32T64),
    bump_fraction: 0.15,
    ladder_bpw: 5.100,
};

/// `TCF-COMPACT`: `Q4AS32D_T64` with `Q6S16D_T64` on the top 10%.
///
/// `0.90 * 4.5000 + 0.10 * 6.5625 = 4.706250`.
pub const TCF_COMPACT: TcfProfile = TcfProfile {
    name: "TCF-COMPACT",
    base: NativeEncoding::Q4AS32DT64,
    bump: Some(NativeEncoding::Q6S16DT64),
    bump_fraction: 0.10,
    ladder_bpw: 4.706_25,
};

/// Parse a profile name.
///
/// Case, underscores, hyphens and a leading `tcf` are ignored, so
/// `TCF-BALANCED`, `tcf_balanced` and `balanced` are the same rung.
///
/// # Errors
/// The name is not one of [`TCF_PROFILE_NAMES`].
pub fn parse_tcf_profile(s: &str) -> Result<TcfProfile> {
    let normalized: String = s
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect();
    let stripped = normalized.strip_prefix("tcf").unwrap_or(&normalized);
    match stripped {
        "ref" => Ok(TCF_REF),
        "high" => Ok(TCF_HIGH),
        "balanced" => Ok(TCF_BALANCED),
        "compact" => Ok(TCF_COMPACT),
        _ => bail!(
            "Unsupported TCF profile: {}. Supported: {}",
            s,
            TCF_PROFILE_NAMES.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_profile_name() {
        assert_eq!(parse_tcf_profile("ref").unwrap(), TCF_REF);
        assert_eq!(parse_tcf_profile("high").unwrap(), TCF_HIGH);
        assert_eq!(parse_tcf_profile("balanced").unwrap(), TCF_BALANCED);
        assert_eq!(parse_tcf_profile("compact").unwrap(), TCF_COMPACT);
    }

    /// The spelling in `PROFILES.md` Section 2 is accepted verbatim.
    #[test]
    fn parses_the_specification_spelling() {
        assert_eq!(parse_tcf_profile("TCF-COMPACT").unwrap(), TCF_COMPACT);
        assert_eq!(parse_tcf_profile("tcf_high").unwrap(), TCF_HIGH);
    }

    #[test]
    fn rejects_an_unknown_profile_name() {
        let err = parse_tcf_profile("tiny").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF profile: tiny"), "{err}");
        assert!(err.contains("compact"), "{err}");
    }

    /// The ladder is `PROFILES.md` Section 2, rung for rung.
    #[test]
    fn the_ladder_matches_the_specification() {
        assert_eq!(TCF_REF.base, NativeEncoding::Q8S32T64);
        assert_eq!(TCF_REF.bump, None);
        assert_eq!(TCF_HIGH.bump, Some(NativeEncoding::Q8S32T64));
        assert_eq!(TCF_HIGH.bump_fraction, 0.15);
        assert_eq!(TCF_HIGH.base, NativeEncoding::Q6S16DT64);
        assert_eq!(TCF_BALANCED.base, NativeEncoding::Q4AS32DT64);
        assert_eq!(TCF_BALANCED.bump_fraction, 0.15);
        assert_eq!(TCF_COMPACT.base, NativeEncoding::Q4AS32DT64);
        assert_eq!(TCF_COMPACT.bump, Some(NativeEncoding::Q6S16DT64));
        assert_eq!(TCF_COMPACT.bump_fraction, 0.10);
    }
}
