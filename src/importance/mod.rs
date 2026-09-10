//! The OPTIONAL importance matrix a conversion can weight its quantizer by.
//!
//! An importance matrix records, per weight, the mean squared activation each
//! input COLUMN saw during a calibration run. `boostr` owns the file format,
//! its writer, its reader and its checker; this module is compressr's use of
//! them, shared by both codecs.
//!
//! # It is a lever, never a default
//!
//! `--imatrix` is off unless a run names a file. A conversion without it is
//! byte for byte what it always was, for every format, and a file written
//! without one is a normal conforming file. Producing a matrix costs a
//! calibration run over a dataset, and the benefit concentrates at the very
//! low bit widths, so compressr neither defaults it nor reports its absence
//! as a deficiency.
//!
//! # Absent is a decision
//!
//! A tensor with no entry is one the calibration run never exercised. It is
//! quantized UNWEIGHTED. A zero vector is never synthesized in its place:
//! zero importance is legal and means "measured, and unimportant", which is a
//! different statement. [`ImportanceCounts`] keeps the two apart in the
//! report.

mod counts;
mod matrix;

pub use boostr::quant::{ImportanceCheck, ImportanceEntry, ImportanceMatrix};
pub use counts::{
    importance_layout_line, importance_source_line, ImportanceCounts, ImportancePath,
};
pub use matrix::{
    check_importance_matrix, importance_columns, quantizable_widths, read_importance_matrix,
    resolve_importance, IMPORTANCE_ACCEPTED,
};
