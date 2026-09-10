//! What each tensor did with the importance matrix, and the lines a run
//! states about it.
//!
//! Three paths, counted apart because they are three different facts. A
//! tensor was weighted; a tensor had no entry, so the calibration run never
//! exercised it; or a tensor's block format has no importance-weighted writer
//! at all. Folding the last two together would report a calibration gap that
//! is not there.

use std::fmt;

use boostr::quant::{ImportanceCheck, ImportanceMatrix};

/// Which path one tensor's quantizer took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportancePath {
    /// Quantized against the matrix's per-column importance
    Weighted,
    /// Quantized UNWEIGHTED: the matrix holds no usable entry for it
    Absent,
    /// Quantized UNWEIGHTED: its block format has no weighted writer
    Unweightable,
}

/// How many tensors took each path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImportanceCounts {
    /// Tensors quantized against the matrix
    pub weighted: usize,
    /// Tensors quantized unweighted for want of an entry
    pub absent: usize,
    /// Tensors quantized unweighted because their format has no weighted writer
    pub unweightable: usize,
}

impl ImportanceCounts {
    /// Record one tensor.
    pub fn record(&mut self, path: ImportancePath) {
        match path {
            ImportancePath::Weighted => self.weighted += 1,
            ImportancePath::Absent => self.absent += 1,
            ImportancePath::Unweightable => self.unweightable += 1,
        }
    }

    /// Tensors counted on any path.
    pub fn total(&self) -> usize {
        self.weighted + self.absent + self.unweightable
    }
}

impl fmt::Display for ImportanceCounts {
    /// Every clause is stated on every run, zero included: a line whose shape
    /// changes with the numbers cannot be compared between two conversions.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} tensor(s) quantized importance-weighted, {} unweighted for want of an entry, \
             {} unweighted because their format has no importance-weighted writer",
            self.weighted, self.absent, self.unweightable
        )
    }
}

/// What the matrix itself carries, and how it lines up with the checkpoint.
///
/// The token count is the file's own: it is what the importance was
/// accumulated over, and it is the one number that says how much measurement
/// is behind every weight below.
pub fn importance_source_line(matrix: &ImportanceMatrix, check: &ImportanceCheck) -> String {
    format!(
        "Importance matrix: {} measured tensor(s) accumulated over {} token(s); {} of {} \
         checkpoint tensor(s) measured, {} never exercised, {} file entry(s) name no tensor in \
         this checkpoint",
        matrix.len(),
        matrix.token_count(),
        check.present.len(),
        check.present.len() + check.absent.len(),
        check.absent.len(),
        check.unknown.len()
    )
}

/// The one sentence a run states about what an importance matrix changed.
///
/// `format_name` names the output format, spelled as that format's own
/// section of `AGENTS.md` spells it.
pub fn importance_layout_line(format_name: &str) -> String {
    format!(
        "Importance: an importance matrix changes which values the encoder chooses and no bit \
         layout, so the output is a normal conforming {format_name} file."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each path is counted apart, and the total is their sum.
    #[test]
    fn every_path_is_counted_apart() {
        let mut counts = ImportanceCounts::default();
        counts.record(ImportancePath::Weighted);
        counts.record(ImportancePath::Weighted);
        counts.record(ImportancePath::Absent);
        counts.record(ImportancePath::Unweightable);
        assert_eq!(counts.weighted, 2);
        assert_eq!(counts.absent, 1);
        assert_eq!(counts.unweightable, 1);
        assert_eq!(counts.total(), 4);
    }

    /// A run that weighted nothing still states all three clauses.
    #[test]
    fn the_report_line_states_every_clause_at_zero() {
        let line = ImportanceCounts::default().to_string();
        assert!(
            line.contains("0 tensor(s) quantized importance-weighted"),
            "{line}"
        );
        assert!(line.contains("0 unweighted for want of an entry"), "{line}");
        assert!(
            line.contains("0 unweighted because their format has no importance-weighted writer"),
            "{line}"
        );
    }

    /// The layout line says the output is a normal file of its own format.
    #[test]
    fn the_layout_line_names_the_format() {
        let line = importance_layout_line("GGUF v3");
        assert!(line.contains("no bit layout"), "{line}");
        assert!(line.contains("normal conforming GGUF v3 file"), "{line}");
    }
}
