//! Where an OPTIONAL importance matrix attaches to a TCF conversion.
//!
//! Two questions, answered here so pass 1 and pass 2 answer them the same
//! way: what does the matrix say about ONE tensor, and which weights does
//! that put that tensor's quantizer call on.
//!
//! # A measured tensor takes the run's importance FORM
//!
//! There are two, and [`super::importance_form`] owns the choice between
//! them. The default multiplies the importance by the data-derived term
//! ggml's K-quant imatrix path uses; `--tcf-importance replace` scores the
//! tensor by the importance ALONE, which replaces the objective its encoding
//! carries. Either way a tensor the matrix never exercised keeps the
//! objective its encoding carries, or the one `--tcf-objective` named: it
//! never reaches the importance path at all.
//!
//! # It costs one expansion per quantized tensor
//!
//! Both forms are per ELEMENT, so a measured tensor's per-column vector is
//! expanded to its element count for the length of the call. That is an
//! `f32` per weight on top of the values already resident, and it is paid
//! only on an `--imatrix` run.

use anyhow::Result;
use boostr::quant::{ImportanceEntry, ImportanceMatrix};
use tcf_core::NativeEncoding;

use super::objective_weights::ResolvedWeights;
use super::options::TcfOptions;

/// One tensor's per-column importance, or `None` when there is none to use.
///
/// `None` covers no matrix at all, a tensor with no entry, and an entry
/// accumulated over zero rows. None of the three is turned into a zero
/// vector: zero importance is legal and means something else.
pub(super) fn columns_for(matrix: Option<&ImportanceMatrix>, name: &str) -> Option<Vec<f32>> {
    matrix?.get(name).and_then(ImportanceEntry::mean_square)
}

/// The weights ONE quantizer call is scored by, for a tensor written at
/// `encoding`.
///
/// `columns` present is expanded onto this call's elements and applied in
/// `options.importance_form`; absent keeps the objective, which carries no
/// data. This is the single point both passes go through, so a tensor
/// requantized in pass 2 is scored exactly as pass 1 scored it.
///
/// `row_width`, `first_element` and `count` describe the call the weights are
/// for — see [`ResolvedWeights::expand`].
///
/// # Errors
/// An importance vector that is not exactly `row_width` long, or a range that
/// does not fit `usize`.
pub(super) fn weights_for(
    options: TcfOptions,
    encoding: NativeEncoding,
    columns: Option<&[f32]>,
    row_width: usize,
    first_element: usize,
    count: usize,
) -> Result<ResolvedWeights> {
    match columns {
        Some(columns) => ResolvedWeights::expand(
            columns,
            options.importance_form,
            row_width,
            first_element,
            count,
        ),
        None => Ok(ResolvedWeights::derived(
            options.objective_for(encoding).source(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tcf_core::WeightSource;

    use super::super::importance_form::ImportanceForm;
    use super::super::objective::TcfObjective;

    fn matrix() -> ImportanceMatrix {
        let mut matrix = ImportanceMatrix::new(1024);
        matrix
            .insert(ImportanceEntry {
                name: "measured".to_string(),
                rows: 4,
                sums: vec![8.0, 4.0],
            })
            .expect("one entry");
        matrix
    }

    fn options() -> TcfOptions {
        TcfOptions::uniform(NativeEncoding::Q6S16DT64, NativeEncoding::Q6S16DT64)
    }

    /// A measured tensor is scored by its mean square activation ALONE, which
    /// is what an `--imatrix` run takes unless it names the other form.
    #[test]
    fn a_measured_tensor_defaults_to_the_replacement_form() {
        let matrix = matrix();
        let columns = columns_for(Some(&matrix), "measured").expect("an entry");
        assert_eq!(columns, vec![2.0, 1.0]);
        let options = options();
        assert_eq!(options.importance_form, ImportanceForm::Replace);
        let weights = weights_for(options, NativeEncoding::Q6S16DT64, Some(&columns), 2, 0, 4)
            .expect("the importance expands");
        assert!(weights.is_expanded());
        assert!(matches!(weights.source(), WeightSource::Explicit(_)));
        match weights {
            ResolvedWeights::Expanded { form, weights } => {
                assert_eq!(form, ImportanceForm::Replace);
                assert_eq!(weights, vec![2.0, 1.0, 2.0, 1.0]);
            }
            other => panic!("a measured tensor is scored by an expansion, got {other:?}"),
        }
    }

    /// The product form is reachable only by naming it, and then the same
    /// tensor is scored by the importance TIMES the data-derived term.
    #[test]
    fn the_product_form_needs_an_explicit_selection() {
        let matrix = matrix();
        let columns = columns_for(Some(&matrix), "measured").expect("an entry");
        let options = options().with_importance_form(ImportanceForm::Combined);
        let weights = weights_for(options, NativeEncoding::Q6S16DT64, Some(&columns), 2, 0, 4)
            .expect("the importance expands");
        match weights.source() {
            WeightSource::Combined(_) => {}
            other => panic!("the product form multiplies a data term in, got {other:?}"),
        }
    }

    /// Neither form touches a tensor the matrix never measured: it keeps its
    /// encoding's own objective under both.
    #[test]
    fn an_unmeasured_tensor_is_unaffected_by_either_form() {
        for form in [ImportanceForm::Combined, ImportanceForm::Replace] {
            let options = options().with_importance_form(form);
            let weights = weights_for(options, NativeEncoding::Q6S16DT64, None, 2, 0, 4)
                .expect("a derived objective resolves");
            assert!(!weights.is_expanded());
            assert!(matches!(weights.source(), WeightSource::RmsPlusAbs));
        }
    }

    /// A tensor the run never exercised keeps its encoding's own objective,
    /// which for a 6-bit encoding is `rms-plus-abs` and not `uniform`.
    #[test]
    fn an_absent_tensor_keeps_its_encodings_objective() {
        let matrix = matrix();
        assert_eq!(columns_for(Some(&matrix), "never-measured"), None);
        assert_eq!(
            options().objective_for(NativeEncoding::Q6S16DT64),
            TcfObjective::RmsPlusAbs
        );
        let weights = weights_for(options(), NativeEncoding::Q6S16DT64, None, 2, 0, 4)
            .expect("a derived objective resolves");
        assert!(!weights.is_expanded());
        assert!(!weights.source().is_uniform());
    }

    /// A run with no matrix at all expands nothing anywhere, and every
    /// tensor resolves to the per-encoding default it always resolved to.
    #[test]
    fn no_matrix_leaves_every_tensor_on_its_encodings_objective() {
        assert_eq!(columns_for(None, "measured"), None);
        let weights = weights_for(options(), NativeEncoding::Q4S32T64, None, 2, 0, 4)
            .expect("a derived objective resolves");
        assert!(!weights.is_expanded());
        assert!(weights.source().is_uniform());
        let six = weights_for(options(), NativeEncoding::Q6S16DT64, None, 2, 0, 4)
            .expect("a derived objective resolves");
        assert!(matches!(six.source(), WeightSource::RmsPlusAbs));
    }
}

/// How one tensor is to be encoded: the run's TCF options, and the importance
/// matrix when the run has one.
///
/// The two always travel together — every quantizer call site needs the
/// objective policy AND the chance that this tensor was measured — so they
/// move as one value rather than as a pair of parameters threaded separately.
#[derive(Clone, Copy)]
pub struct EncoderPolicy<'a> {
    /// The run's encoding, objective override and profile settings.
    pub options: TcfOptions,
    /// The importance matrix, when `--imatrix` supplied one.
    pub importance: Option<&'a ImportanceMatrix>,
}
