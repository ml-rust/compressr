//! Reading an importance matrix, and checking it against the checkpoint.
//!
//! Every byte of the format — magic, version, entry layout, the reader that
//! parses it and [`ImportanceMatrix::check_against`] — lives in `boostr`,
//! beside the writer that produces the file. This module reads the file
//! through that reader, turns its errors into the `anyhow` ones the rest of
//! compressr raises, and answers the two questions a conversion asks: does
//! this matrix belong to this checkpoint, and what does it say about one
//! tensor.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

use super::counts::ImportancePath;
use boostr::quant::{ImportanceCheck, ImportanceEntry, ImportanceMatrix};

/// The flag combinations `--imatrix` is accepted with, named in every refusal.
///
/// One string, so a refusal from the config check and a refusal from the
/// quantizer cannot list two different sets.
pub const IMPORTANCE_ACCEPTED: &str = "--format gguf with a K-quant or IQ4_NL --quantization \
     (q2_k, q3_k, q4_k, q5_k, q6_k, iq4_nl, q4_k_s, q4_k_m, q5_k_s, q5_k_m), or --format tcf";

/// Read an importance matrix file.
///
/// # Errors
/// The file cannot be read, or `boostr`'s reader refuses its bytes.
pub fn read_importance_matrix(path: &Path) -> Result<ImportanceMatrix> {
    ImportanceMatrix::read_from_path(path)
        .map_err(|e| anyhow!(e))
        .with_context(|| format!("Failed to read the importance matrix '{}'", path.display()))
}

/// `(name, in_features)` for every tensor an importance vector could describe.
///
/// The width is the LAST dimension, because that is the axis both codecs
/// block along and the axis an importance vector has one entry per. A tensor
/// with no dimensions at all describes no column and is left out.
pub fn quantizable_widths<'a>(
    tensors: impl Iterator<Item = (&'a str, &'a [usize])>,
) -> Vec<(String, usize)> {
    tensors
        .filter_map(|(name, shape)| shape.last().map(|width| (name.to_string(), *width)))
        .collect()
}

/// Refuse a matrix whose widths disagree with this checkpoint.
///
/// A name the two share whose column count differs means the file was
/// accumulated over a DIFFERENT model, and importance read at the wrong width
/// is worse than no importance at all. Everything else is a finding rather
/// than an error: an absent name is a weight the run never exercised, and an
/// unknown name is an entry for a weight this checkpoint does not have.
///
/// # Errors
/// A name present in both at two different widths.
pub fn check_importance_matrix(
    matrix: &ImportanceMatrix,
    expected: &[(String, usize)],
) -> Result<ImportanceCheck> {
    matrix
        .check_against(expected)
        .map_err(|e| anyhow!(e))
        .context(
            "The importance matrix does not describe this checkpoint: a tensor it measures has a \
         different number of columns here. Accumulate the matrix over the model being \
         converted.",
        )
}

/// One tensor's per-column importance, or `None` when there is none to use.
///
/// `None` covers both a tensor with no entry and an entry accumulated over
/// zero rows, whose mean is undefined. Neither is turned into a zero vector:
/// see the module documentation of [`super`].
pub fn importance_columns(matrix: &ImportanceMatrix, name: &str) -> Option<Vec<f32>> {
    matrix.get(name).and_then(ImportanceEntry::mean_square)
}

/// One tensor's importance vector, and which path that choice puts it on.
///
/// `weightable` is `false` for a block format with no importance-weighted
/// writer. Such a tensor is quantized UNWEIGHTED rather than refused: a GGUF
/// mix assigns those formats itself, so refusing would refuse the mix.
pub fn resolve_importance(
    matrix: &ImportanceMatrix,
    name: &str,
    weightable: bool,
) -> (Option<Vec<f32>>, ImportancePath) {
    if !weightable {
        return (None, ImportancePath::Unweightable);
    }
    match importance_columns(matrix, name) {
        Some(columns) => (Some(columns), ImportancePath::Weighted),
        None => (None, ImportancePath::Absent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> ImportanceMatrix {
        let mut matrix = ImportanceMatrix::new(4096);
        matrix
            .insert(ImportanceEntry {
                name: "blk.0.attn_q.weight".to_string(),
                rows: 2,
                sums: vec![4.0, 8.0, 0.0, 2.0],
            })
            .expect("one entry");
        matrix
    }

    /// A tensor the file measures at another width belongs to another
    /// checkpoint, so the matrix is refused rather than read at the wrong
    /// stride.
    #[test]
    fn a_width_disagreement_is_refused() {
        let expected = vec![("blk.0.attn_q.weight".to_string(), 8)];
        let err = check_importance_matrix(&matrix(), &expected)
            .expect_err("a width disagreement is an error")
            .to_string();
        assert!(err.contains("does not describe this checkpoint"), "{err}");
    }

    /// Matching widths are accepted, and every name is sorted into measured,
    /// never exercised, or naming no tensor here.
    #[test]
    fn matching_widths_are_reported_rather_than_refused() {
        let expected = vec![
            ("blk.0.attn_q.weight".to_string(), 4),
            ("blk.0.attn_k.weight".to_string(), 4),
        ];
        let check = check_importance_matrix(&matrix(), &expected).expect("widths agree");
        assert_eq!(check.present, vec!["blk.0.attn_q.weight".to_string()]);
        assert_eq!(check.absent, vec!["blk.0.attn_k.weight".to_string()]);
        assert!(check.unknown.is_empty());
    }

    /// A measured tensor is quantized WEIGHTED, by the entry's per-column
    /// mean square.
    #[test]
    fn a_measured_tensor_goes_weighted() {
        let (columns, path) = resolve_importance(&matrix(), "blk.0.attn_q.weight", true);
        assert_eq!(path, ImportancePath::Weighted);
        assert_eq!(columns, Some(vec![2.0, 4.0, 0.0, 1.0]));
    }

    /// A tensor the calibration run never exercised is quantized UNWEIGHTED,
    /// and never against a synthesized zero vector.
    #[test]
    fn an_absent_tensor_goes_unweighted() {
        let (columns, path) = resolve_importance(&matrix(), "blk.9.ffn_down.weight", true);
        assert_eq!(path, ImportancePath::Absent);
        assert_eq!(columns, None);
    }

    /// A block format with no importance-weighted writer is quantized
    /// unweighted, and counted apart from a tensor that simply has no entry.
    #[test]
    fn a_format_with_no_weighted_writer_goes_unweighted() {
        let (columns, path) = resolve_importance(&matrix(), "blk.0.attn_q.weight", false);
        assert_eq!(path, ImportancePath::Unweightable);
        assert_eq!(columns, None);
    }

    /// The width list reads the LAST dimension, which is the axis both codecs
    /// block along.
    #[test]
    fn widths_come_from_the_last_dimension() {
        let shapes: Vec<(String, Vec<usize>)> = vec![
            ("a".to_string(), vec![8, 4]),
            ("b".to_string(), vec![16]),
            ("c".to_string(), Vec::new()),
        ];
        let widths = quantizable_widths(
            shapes
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
        );
        assert_eq!(
            widths,
            vec![("a".to_string(), 4), ("b".to_string(), 16)],
            "a tensor with no dimensions describes no column"
        );
    }
}
