//! One quantizer call's RESOLVED weights.
//!
//! [`super::importance::weights_for`] answers with this. A value-derived
//! objective describes a rule, so it resolves to a `tcf_core::WeightSource`
//! that carries nothing and allocates nothing. An importance is data, and the
//! data `tcf-core` wants is per ELEMENT while an importance matrix is per
//! COLUMN, so that one resolves to an expansion this type owns for the length
//! of the call — see [`ResolvedWeights::expand`].
//!
//! An expansion carries the [`super::importance_form::ImportanceForm`] it was
//! expanded under beside the vector, because the vector alone does not say
//! whether the call minimizes the importance TIMES a data term or the
//! importance alone.
//!
//! The owner has to be a value the call site holds, because
//! `WeightSource::Explicit` borrows its slice. Returning the `WeightSource`
//! alone would return a borrow of a temporary. It is also why the expansion
//! stops here and never reaches [`super::objective::TcfObjective`], which is
//! a policy selector every configuration stores and so carries no data and no
//! lifetime.

use anyhow::{anyhow, Context, Result};
use tcf_core::{column_weights, WeightSource};

use super::importance_form::ImportanceForm;

/// The weights one quantizer call is scored by.
///
/// Not comparable: `tcf_core::WeightSource` is not, and two derived sources
/// naming the same rule are the same run anyway. A test that has to inspect
/// these matches on the variant.
#[derive(Debug, Clone)]
pub enum ResolvedWeights {
    /// A rule read from each group's own values, carrying no data
    Derived(WeightSource<'static>),
    /// A per-COLUMN importance expanded onto this call's elements, and the
    /// form it is applied in
    Expanded {
        /// Whether the importance multiplies a data term or replaces it
        form: ImportanceForm,
        /// One weight per element of this call
        weights: Vec<f32>,
    },
}

impl ResolvedWeights {
    /// Wrap a value-derived source.
    pub(super) fn derived(source: WeightSource<'static>) -> Self {
        Self::Derived(source)
    }

    /// Wrap a per-element expansion of an importance vector, in the form the
    /// run applies it in.
    pub(super) fn expanded(form: ImportanceForm, weights: Vec<f32>) -> Self {
        Self::Expanded { form, weights }
    }

    /// Expand a per-COLUMN importance vector onto the elements of ONE
    /// quantizer call.
    ///
    /// Every call site that holds an importance expands it here, which is
    /// what keeps one tensor's importance from being expanded two different
    /// ways. `columns` holds one entry per column of the tensor, which is its
    /// last dimension; `row_width` is that dimension; `first_element` is the
    /// flat element index this call starts at — `0` for a whole tensor, and
    /// `chunk_start(tile, first)` for a chunk, which is what lets a chunk
    /// beginning mid-row pick up at the column it actually sits in; `count`
    /// is the elements the call covers. `form` is how the expanded vector
    /// meets the values, which is what decides the `tcf-core` source
    /// [`ResolvedWeights::source`] hands the quantizer.
    ///
    /// # Errors
    /// A vector that is not exactly `row_width` long, or a range that does
    /// not fit `usize`. A silently ignored importance vector produces a file
    /// indistinguishable from an unweighted one.
    pub(super) fn expand(
        columns: &[f32],
        form: ImportanceForm,
        row_width: usize,
        first_element: usize,
        count: usize,
    ) -> Result<Self> {
        let expanded = column_weights(columns, row_width, first_element, count)
            .map_err(|e| anyhow!(e))
            .with_context(|| {
                format!(
                    "Failed to expand a {}-column importance vector onto a row of {row_width}",
                    columns.len()
                )
            })?;
        Ok(Self::expanded(form, expanded))
    }

    /// The `tcf-core` source, borrowing the expansion when there is one.
    ///
    /// This is what a quantizer call takes. The parallel path narrows it per
    /// chunk with `WeightSource::slice`, which keeps an expanded slice
    /// parallel to the values that chunk holds.
    pub fn source(&self) -> WeightSource<'_> {
        match self {
            Self::Derived(source) => *source,
            Self::Expanded { form, weights } => form.source(weights),
        }
    }

    /// Whether these weights came from an importance matrix.
    pub fn is_expanded(&self) -> bool {
        matches!(self, Self::Expanded { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A derived source passes through untouched and allocates nothing.
    #[test]
    fn a_derived_source_passes_through() {
        let resolved = ResolvedWeights::derived(WeightSource::RmsPlusAbs);
        assert!(!resolved.is_expanded());
        assert!(!resolved.source().is_uniform());
        assert!(ResolvedWeights::derived(WeightSource::Uniform)
            .source()
            .is_uniform());
    }

    /// An expansion in the replacement form is handed to `tcf-core` as the
    /// explicit per-element slice, and narrows to a chunk the way the values
    /// do.
    #[test]
    fn an_expansion_is_borrowed_as_an_explicit_slice() {
        let resolved = ResolvedWeights::expanded(ImportanceForm::Replace, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(resolved.is_expanded());
        match resolved.source() {
            WeightSource::Explicit(weights) => {
                assert_eq!(weights, [1.0f32, 2.0, 3.0, 4.0].as_slice());
            }
            other => panic!("an expansion resolves to an explicit slice, got {other:?}"),
        }
        match resolved.source().slice(1, 3) {
            WeightSource::Explicit(weights) => assert_eq!(weights, [2.0f32, 3.0].as_slice()),
            other => panic!("an explicit slice narrows to a sub-slice, got {other:?}"),
        }
    }

    /// The same expansion in the combined form is handed over as the product
    /// source instead, and narrows the same way.
    #[test]
    fn a_combined_expansion_is_borrowed_as_the_product_source() {
        let resolved =
            ResolvedWeights::expanded(ImportanceForm::Combined, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(resolved.is_expanded());
        assert!(matches!(resolved.source(), WeightSource::Combined(_)));
        assert!(matches!(
            resolved.source().slice(1, 3),
            WeightSource::Combined(_)
        ));
    }

    /// An importance vector is expanded onto the call's elements by the
    /// column each one sits in, and a chunk that starts mid-row picks up at
    /// that column.
    #[test]
    fn an_importance_vector_expands_by_column() {
        let columns = [1.0f32, 2.0, 3.0, 4.0];
        match ResolvedWeights::expand(&columns, ImportanceForm::Combined, 4, 0, 8).expect("expands")
        {
            ResolvedWeights::Expanded { weights, .. } => {
                assert_eq!(weights, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("an importance expands per element, got {other:?}"),
        }

        // A chunk starting six elements in picks up at the column six sits
        // in, which is what keeps the threaded path scoring what a
        // whole-tensor call scores.
        match ResolvedWeights::expand(&columns, ImportanceForm::Combined, 4, 6, 2).expect("expands")
        {
            ResolvedWeights::Expanded { weights, .. } => assert_eq!(weights, vec![3.0, 4.0]),
            other => panic!("an importance expands per element, got {other:?}"),
        }
    }

    /// A vector that is not exactly as long as the row is refused. Ignoring
    /// it would write a file indistinguishable from an unweighted one.
    #[test]
    fn a_wrong_width_importance_vector_is_refused() {
        let columns = [1.0f32, 2.0];
        let err = ResolvedWeights::expand(&columns, ImportanceForm::Combined, 4, 0, 8)
            .expect_err("a width disagreement is an error")
            .to_string();
        assert!(err.contains("2-column importance vector"), "{err}");
        assert!(err.contains("row of 4"), "{err}");
    }
}
