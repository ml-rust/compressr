//! TCF v1 writer.
//!
//! compressr produces TCF from a full-precision checkpoint and quantizes it
//! once. There is no GGUF-to-TCF path: a GGML block is already quantized, so
//! converting one would quantize twice and would need a GGML dequantizer per
//! format. See `hats/tcf/MIGRATION.md` Section 3.
//!
//! # What this module does not own
//!
//! Layout and packing. Every bit position, nibble index, tile order and plane
//! order lives in `tcf-core`, which also computes every offset, span and
//! digest. This module chooses policy — encoding, `fallback_reason`,
//! residency — and calls [`tcf_core::encoding::quantize`] and
//! `TcfWriter::finish_streaming` for the rest. GGUF's block layouts stay in
//! boostr for the same reason: a writer checked against a reader beside it
//! proves nothing.

mod encoding;
mod record;
mod writer;

pub use encoding::{parse_tcf_encoding, TcfOptions, TCF_ENCODING_NAMES};
pub use writer::{PlannedTensor, TcfFormatWriter};
