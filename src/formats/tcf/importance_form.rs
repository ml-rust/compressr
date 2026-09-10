//! WHICH form an `--imatrix` importance takes when it meets the objective.
//!
//! An importance is a per-column mean square activation. There are two
//! established ways to score a weight by it, and ggml uses both:
//!
//! - The PRODUCT. `quantize_row_q4_K_impl` in `ggml-quants.c` minimizes
//!   `q_i * sqrt(sigma2 + x_i^2)`, the importance TIMES a data-derived term,
//!   with `sigma2 = 2 * mean(x^2)`. Q3_K, Q4_K and Q5_K all take it.
//! - The IMPORTANCE ALONE. Q6_K's imatrix path passes the raw importance and
//!   the data term never appears.
//!
//! The two are different objectives, not two spellings of one, so compressr
//! names them and lets a run choose. [`ImportanceForm::Combined`] is the
//! default because the encodings compressr ships at are 4-bit and 6-bit tiers
//! whose ggml counterparts are the K-quants, and
//! [`ImportanceForm::Replace`] is `--tcf-importance replace`.
//!
//! # Neither form is reachable without a matrix
//!
//! Both describe what to do with an importance vector, and a run with no
//! `--imatrix` has none. Such a run never reaches this module at all: it
//! takes the objective its encoding carries, and writes exactly the bytes it
//! always wrote.

use anyhow::{bail, Result};
use tcf_core::{WeightSource, K_QUANT_SIGMA2_FACTOR};

/// The `--tcf-importance` values, in the order the CLI help lists them.
pub const TCF_IMPORTANCE_NAMES: [&str; 2] = ["combined", "replace"];

/// How a measured tensor's importance is turned into the weight the TCF
/// quantizer minimizes under.
///
/// Like [`super::objective::TcfObjective`] this is a lifetime-free POLICY
/// SELECTOR: it names a rule and carries no vector, so `TcfOptions` and every
/// configuration that stores one stays free of a lifetime. The vector itself
/// lives in [`super::objective_weights::ResolvedWeights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceForm {
    /// `w_i = q_i * sqrt(sigma2 + x_i^2)`: the importance TIMES the
    /// data-derived term, which is ggml's K-quant imatrix weighting.
    ///
    /// Reproducing ggml here does NOT reproduce ggml's result, because the
    /// encoding underneath is not a K-quant. A measurement on one text model
    /// scored this form behind [`Self::Replace`] on both evaluation splits,
    /// which is why it is not the default. Selected by
    /// `--tcf-importance combined`.
    Combined,
    /// `w_i = q_i`: the importance ALONE, which REPLACES the objective the
    /// tensor's encoding carries. This is ggml's Q6_K imatrix path.
    ///
    /// The `Default`, chosen by measurement rather than by analogy: on one
    /// text model it beat [`Self::Combined`] on both evaluation splits. One
    /// model separates the two forms, so this row moves if a broader
    /// measurement disagrees.
    #[default]
    Replace,
}

impl ImportanceForm {
    /// The `tcf-core` source this form puts one quantizer call on, over the
    /// per-element `weights` expanded for that call.
    ///
    /// `sigma2`'s span is not chosen here: `tcf-core` widens it to the span
    /// the encoder resolves weights within, which is the super-block for a
    /// two-level layout and the tile for a flat one.
    pub fn source(self, weights: &[f32]) -> WeightSource<'_> {
        match self {
            Self::Combined => WeightSource::importance_times_energy(weights, K_QUANT_SIGMA2_FACTOR),
            Self::Replace => WeightSource::Explicit(weights),
        }
    }

    /// The one fixed spelling a report and a log line use for this form.
    pub fn name(self) -> &'static str {
        match self {
            Self::Combined => "combined",
            Self::Replace => "replace",
        }
    }

    /// The one line a TCF run states about how its importance met the
    /// objective.
    ///
    /// Written only on an `--imatrix` run, and it names the form that
    /// actually ran: the two weight different things, and a reader comparing
    /// two files has to be able to see which.
    pub fn report(self) -> &'static str {
        match self {
            Self::Combined => {
                "Importance: a measured tensor is scored by its importance TIMES \
                 sqrt(sigma2 + x^2), the product ggml's K-quant imatrix path minimizes, with \
                 sigma2 = 2 * mean(x^2) over the span the encoder resolves weights within; a \
                 tensor with no entry keeps the objective its encoding carries. Pass \
                 --tcf-importance replace for the importance alone."
            }
            Self::Replace => {
                "Importance: a measured tensor is scored by the importance ALONE, which \
                 replaces the objective its encoding carries, the form ggml's Q6_K imatrix \
                 path takes; a tensor with no entry keeps that objective. This form was \
                 selected by an explicit --tcf-importance replace."
            }
        }
    }
}

/// Parse a `--tcf-importance` name.
///
/// Case, underscores and hyphens are ignored, exactly as they are for an
/// encoding or an objective name.
///
/// # Errors
/// The name is not one of [`TCF_IMPORTANCE_NAMES`].
pub fn parse_tcf_importance(s: &str) -> Result<ImportanceForm> {
    let normalized = super::encoding::normalized_encoding_name(s);
    match normalized.as_str() {
        "combined" => Ok(ImportanceForm::Combined),
        "replace" => Ok(ImportanceForm::Replace),
        _ => bail!(
            "Unsupported TCF importance form: {}. Supported: {}",
            s,
            TCF_IMPORTANCE_NAMES.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default is the importance alone — the MEASURED winner, not the
    /// closer analogy to ggml's K-quant path.
    #[test]
    fn the_default_form_replaces_the_objective() {
        assert_eq!(ImportanceForm::default(), ImportanceForm::Replace);
        assert!(matches!(
            ImportanceForm::default().source(&[1.0, 2.0]),
            WeightSource::Explicit(_)
        ));
        // The product form stays reachable, and is a different source.
        assert!(matches!(
            ImportanceForm::Combined.source(&[1.0, 2.0]),
            WeightSource::Combined(_)
        ));
    }

    /// The replacement form puts the call on the raw importance, which is
    /// what `WeightSource::Explicit` means.
    #[test]
    fn the_replacement_form_is_the_raw_importance() {
        match ImportanceForm::Replace.source(&[1.0, 2.0]) {
            WeightSource::Explicit(weights) => assert_eq!(weights, [1.0f32, 2.0].as_slice()),
            other => panic!("the replacement form is the raw importance, got {other:?}"),
        }
    }

    /// Every name the CLI help lists parses, and names itself back.
    #[test]
    fn a_name_round_trips_through_the_parser() {
        for name in TCF_IMPORTANCE_NAMES {
            let form = parse_tcf_importance(name).expect("parses");
            assert_eq!(form.name(), name);
        }
        assert_eq!(
            parse_tcf_importance("COMBINED").expect("parses"),
            ImportanceForm::Combined
        );
    }

    /// An unknown name is refused naming the whole accepted set.
    #[test]
    fn rejects_an_unknown_form_name() {
        let err = parse_tcf_importance("explicit").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF importance form"), "{err}");
        for name in TCF_IMPORTANCE_NAMES {
            assert!(err.contains(name), "{err}");
        }
    }

    /// Each form's report line states the form that actually ran, and neither
    /// line claims the other's rule.
    #[test]
    fn each_form_reports_the_rule_it_applied() {
        let combined = ImportanceForm::Combined.report();
        assert!(combined.contains("TIMES sqrt(sigma2 + x^2)"), "{combined}");
        assert!(!combined.contains("ALONE"), "{combined}");
        let replace = ImportanceForm::Replace.report();
        assert!(replace.contains("importance ALONE"), "{replace}");
        assert!(replace.contains("--tcf-importance replace"), "{replace}");
    }
}
