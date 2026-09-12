//! TCF writer: GGML block encodings in TCF's container.
//!
//! A `--format tcf` run plans and quantizes exactly as a GGUF run does — the
//! same per-tensor plan, the same `ggml-quants.c` writers in boostr, the same
//! bytes — and [`TcfBlockFormatWriter`] stores them in the container: a
//! record per tensor, the ggml-family activation contract, a proof vector
//! checked with boostr's own decoder, and, on a `--sensitivity` run, the
//! calibration record and per-tensor task-metric delta the allocation was
//! ranked by. A TCF file is never behind the GGUF of the same model.
//!
//! There is no GGUF-to-TCF path: a GGML block is already quantized, so
//! converting one would quantize twice.
//!
//! # What this module does not own
//!
//! Layout and packing. Every block layout lives in boostr's quantize and
//! dequant kernels, and `boostr::tcf` computes every offset, span and digest.
//! This module chooses policy — encoding, `fallback_reason`, residency — and
//! hands the bytes over.

mod block_spill;
mod block_writer;
mod encoding;
mod record;

pub use block_writer::{block_encoding_for, TcfBlockFormatWriter};
