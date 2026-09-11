//! `--tcf-search`: HOW WIDE the TCF quantizer's candidate-scale search runs.
//!
//! `tcf_core::SearchEffort` is defined and owned by `tcf-core` — this module
//! adds no rule of its own, only the CLI's name for the choice: parsing,
//! self-naming and the one report line, exactly as [`super::objective`] and
//! [`super::importance_form`] do for their own flags.
//!
//! # Every wider level is a strict superset of a narrower one
//!
//! `SearchEffort::Wide` and `SearchEffort::Exhaustive` try every candidate
//! `SearchEffort::Standard` tries, plus more, so a wider search can only
//! match or beat a narrower one ON THE ENCODER'S OWN ERROR OBJECTIVE. The
//! cost is conversion time only: no bit layout, tile order, plane order,
//! offset or digest moves, so a file written under any effort is a normal
//! conforming TCF v1 file.
//!
//! # The guarantee is on the objective, NOT on the model
//!
//! The objective is a proxy. Minimizing it harder can cost model quality.
//! Measured on both evaluation splits: a wider search improved the task
//! metric with no importance matrix, and degraded it with one.
//!
//! An unweighted objective is loose, so extra candidates find better fits. An
//! importance-weighted objective is already sharp, so a wider search overfits
//! it. Never describe a wider search as free quality.
//!
//! # `Standard` is the byte-for-byte default
//!
//! `SearchEffort::Standard` is `tcf-core`'s [`Default`] and is what every
//! entry point ran before `SearchEffort` existed, so a run that names no
//! `--tcf-search` writes exactly the bytes it always wrote.

use anyhow::{bail, Result};
use tcf_core::SearchEffort;

/// The `--tcf-search` values, in the order the CLI help lists them.
pub const TCF_SEARCH_NAMES: [&str; 3] = ["standard", "wide", "exhaustive"];

/// The one fixed spelling a report and a log line use for this effort.
pub fn tcf_search_name(effort: SearchEffort) -> &'static str {
    match effort {
        SearchEffort::Standard => "standard",
        SearchEffort::Wide => "wide",
        SearchEffort::Exhaustive => "exhaustive",
    }
}

/// The one line a TCF run states about a non-default search effort.
///
/// Written only when `effort` is not [`SearchEffort::Standard`]: a default
/// run's log is unchanged from before this flag existed.
pub fn tcf_search_report(effort: SearchEffort) -> &'static str {
    match effort {
        SearchEffort::Standard => {
            "TCF search: standard, the default candidate-scale search. This line is not \
             normally reached: a default run states nothing about its search effort."
        }
        SearchEffort::Wide => {
            "TCF search: wide. Every scale search tries every candidate the standard search \
             tries, plus more, so it can only match or improve the fit ON THE ENCODER'S ERROR \
             OBJECTIVE. That is not a promise about model quality: the objective is a proxy, \
             and a wider search has been measured to improve the task metric without an \
             importance matrix and to degrade it with one. It costs conversion time only: no \
             bit layout, tile order, plane order, offset or digest moves."
        }
        SearchEffort::Exhaustive => {
            "TCF search: exhaustive. Every scale search tries every candidate the standard \
             search tries, plus more, at twice the density of the wide search, so it can only \
             match or improve the fit ON THE ENCODER'S ERROR OBJECTIVE. That is not a promise \
             about model quality: the objective is a proxy, and a wider search has been \
             measured to improve the task metric without an importance matrix and to degrade \
             it with one. It costs conversion time only: no bit layout, tile order, plane \
             order, offset or digest moves."
        }
    }
}

/// Parse a `--tcf-search` name.
///
/// Case, underscores and hyphens are ignored, exactly as they are for an
/// encoding, an objective or an importance form name.
///
/// # Errors
/// The name is not one of [`TCF_SEARCH_NAMES`].
pub fn parse_tcf_search(s: &str) -> Result<SearchEffort> {
    let normalized = super::encoding::normalized_encoding_name(s);
    match normalized.as_str() {
        "standard" => Ok(SearchEffort::Standard),
        "wide" => Ok(SearchEffort::Wide),
        "exhaustive" => Ok(SearchEffort::Exhaustive),
        _ => bail!(
            "Unsupported TCF search effort: {}. Supported: {}",
            s,
            TCF_SEARCH_NAMES.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default is the standard search, unchanged from before this flag
    /// existed.
    #[test]
    fn the_default_effort_is_standard() {
        assert_eq!(SearchEffort::default(), SearchEffort::Standard);
    }

    /// Every name the CLI help lists parses, and names itself back.
    #[test]
    fn a_name_round_trips_through_the_parser() {
        for name in TCF_SEARCH_NAMES {
            let effort = parse_tcf_search(name).expect("parses");
            assert_eq!(tcf_search_name(effort), name);
        }
        assert_eq!(
            parse_tcf_search("WIDE").expect("parses"),
            SearchEffort::Wide
        );
    }

    /// Case, underscores and hyphens never separate two spellings of one
    /// effort.
    #[test]
    fn one_effort_has_one_normalized_name() {
        assert_eq!(
            parse_tcf_search("EXHAUSTIVE").unwrap(),
            SearchEffort::Exhaustive
        );
        assert_eq!(
            parse_tcf_search("ex_haustive").unwrap(),
            SearchEffort::Exhaustive
        );
        assert_eq!(
            parse_tcf_search("ex-haustive").unwrap(),
            SearchEffort::Exhaustive
        );
    }

    /// An unknown name is refused naming the whole accepted set.
    #[test]
    fn rejects_an_unknown_effort_name() {
        let err = parse_tcf_search("thorough").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF search effort"), "{err}");
        for name in TCF_SEARCH_NAMES {
            assert!(err.contains(name), "{err}");
        }
    }

    /// A non-default effort's report line names itself, scopes the superset
    /// guarantee to the ERROR OBJECTIVE, and refuses to sell it as model
    /// quality — a wider search has been measured to cost task metric when an
    /// importance matrix is supplied, so a line claiming an unqualified win
    /// would be false.
    #[test]
    fn a_non_default_effort_scopes_its_guarantee_to_the_objective() {
        for effort in [SearchEffort::Wide, SearchEffort::Exhaustive] {
            let report = tcf_search_report(effort);
            assert!(report.contains(tcf_search_name(effort)), "{report}");
            assert!(report.contains("ERROR OBJECTIVE"), "{report}");
            assert!(
                report.contains("not a promise about model quality"),
                "{report}"
            );
            assert!(report.contains("importance matrix"), "{report}");
        }
    }
}
