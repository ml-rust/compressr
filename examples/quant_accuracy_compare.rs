//! Head-to-head quantization ACCURACY: TCF against GGUF at matched bits per weight.
//!
//! Quantizes the SAME source tensor with both codecs, dequantizes both, and
//! diffs each reconstruction against the f32 source. Nothing is rescaled and no
//! "closest bpw" fudge is applied: every pair below is a bit-for-bit bpw
//! match, so the only variable left is which codec reconstructs better.
//!
//! # Two modes
//!
//! `--mode super-block-256` (the DEFAULT, and the tool's original behaviour):
//!
//! | TCF          | GGUF | bytes / 256 elements | bpw    |
//! |--------------|------|----------------------|--------|
//! | `Q4AS32DT64` | Q4_K | 144                  | 4.50   |
//! | `Q6S16DT64`  | Q6_K | 210                  | 6.5625 |
//! | `Q8S32T64`   | Q8_0 | 272                  | 8.50   |
//!
//! The 8.50 row is not two formats. `Q8S32T64` and `Q8_0` are BIT-FOR-BIT the
//! same layout: 32-element blocks, one binary16 scale, 8-bit codes clamped to
//! [-127, 127], 34 bytes per 32 elements. That tier compares two ENCODERS of
//! one format, and the report labels it so wherever it appears.
//!
//! Those first two TCF encodings carry a 256-element super-block, the same
//! granularity the GGUF K-quants they face carry. So that mode structurally
//! CANNOT observe a row where the two sides differ in what they can encode at
//! all: whatever `Q4_K` refuses, `Q4AS32DT64`'s super-block is fitted over the
//! same 256 elements.
//!
//! `--mode tile-64` exists for exactly that regime. TCF's flat-scale
//! encodings tile at 64 and need only `row % 64 == 0`. GGUF's K-quants need
//! `row % 256 == 0`; only its legacy 32-element formats reach a row of 64.
//!
//! | TCF          | GGUF | bpw  | granularity        |
//! |--------------|------|------|--------------------|
//! | `Q4S32T64`   | Q4_0 | 4.50 | scale group 32 both, symmetric both |
//! | `Q4AS64T64`  | Q4_0 | 4.50 | TCF group 64 asymmetric vs GGUF block 32 symmetric |
//! | `Q4AS32T64`  | Q4_1 | 5.00 | scale+min group 32 both, asymmetric both |
//! | `Q6S32T64`   | NONE | 6.50 | no equal-budget GGUF — BRACKETED, see below |
//!
//! The 6.50 row has no equal-budget GGUF side. `Q6_K` is 6.5625 bpw but needs
//! `row % 256 == 0`, and GGUF's 32-element formats step 4.50, 5.00, 5.50,
//! 6.00, 8.50 with nothing at 6.50. Of those, boostr carries a CPU quantize
//! kernel for `Q4_0`, `Q4_1` and `Q8_0` only, so `Q5_0` and `Q5_1` cannot
//! serve as a comparison side at all even though their budgets sit between.
//!
//! # Bracketing: what a tier with no equal-budget counterpart reports
//!
//! Printing TCF's error with `n/a` in every ratio column and stopping there
//! discards the actual finding. For a row-64 tensor the real deployment choice
//! is not "6.50 versus nothing"; it is 5.00 (`Q4_1`) or 8.50 (`Q8_0`) from
//! GGUF against 6.50 from TCF. So such a tier is compared against the
//! ENCODABLE GGUF budgets that BRACKET it — the nearest below and the nearest
//! above, at the granularity the TCF tile allows.
//!
//! | side  | GGUF   | bpw  | relation to TCF's 6.50 |
//! |-------|--------|------|------------------------|
//! | below | `Q4_1` | 5.00 | 1.50 bpw cheaper       |
//! | above | `Q8_0` | 8.50 | 2.00 bpw dearer        |
//!
//! Those two are not asserted. `verify_brackets` re-derives the candidate set
//! from `QuantFormat`'s own block layouts, keeps only formats whose block
//! divides the TCF tile, PROBES each one with a real `QuantizeOps::quantize`
//! call to establish that boostr can actually write it, and then reports
//! whether the declared pair really is the nearest encodable budget on each
//! side. A format that sits between but has no quantize kernel is named as
//! excluded rather than silently dropped.
//!
//! A bracketed comparison DELIBERATELY differs in bytes, so the byte-match
//! verification that gates the equal-budget tiers does not apply to it and
//! does not reject it. Every bracketed row and every bracketed verdict line
//! therefore states both budgets, and the word `BRACKETED` appears on each, so
//! no reader takes a bracketed ratio for an equal-bpw one.
//!
//! The finding is computed, never asserted. Per objective the report states
//! whether TCF's error at its budget lands at or below the upper bracket's
//! (TCF buys the dearer format's quality at a cheaper size), between the two
//! (with the proportional position printed), or above the lower bracket's
//! (TCF loses to both). "Better than 5.00, worse than 8.50" is a real answer
//! and is printed as one.
//!
//! # What a "row" is
//!
//! The row is the LAST dimension of the standard-order shape, and every bpw,
//! byte and shape-support figure here follows that. This was checked against
//! both codecs rather than assumed:
//!
//! - `tcf-core`'s `row_major_tiles` (SPECIFICATION.md Section 12.3) sets
//!   `K = dims[rank - 1]` and `R = product(dims[0..rank - 1])`, and refuses
//!   the tensor when `K % tile != 0`. Tiles never cross a row boundary.
//! - boostr's `QuantizeOps::quantize` for `CpuRuntime` refuses the tensor when
//!   `shape.last() % format.block_size() != 0`.
//! - `compressr`'s own GGUF writer and its `policy` module apply the same
//!   last-dimension rule, because GGUF stores dimensions reversed and `ne[0]`
//!   is the last standard-order dimension.
//!
//! So a rank-3 tensor is NOT flattened to `[shape[0], product(shape[1..])]` by
//! either codec. A `[1024, 1024, 7]` conv weight has a row of 7, both codecs
//! genuinely refuse it, and `compressr convert` genuinely stores it as F32.
//! Reporting it as unencodable is the correct measurement, not an inflated
//! count.
//!
//! Every number is PER TENSOR. A pooled model-wide number averages away the
//! tensor where a gap actually shows, and attention output projections,
//! embeddings and FFN-down are exactly the tensors that carry the outlier
//! channels dominating LLM quality loss — which is why GGUF's own `_K_M` mixes
//! assign formats per tensor role.
//!
//! The byte count each codec produced is measured, not assumed, per tensor. A
//! tensor whose two sides differ in bytes is NOT a matched comparison; it is
//! flagged `UNMATCHED` and excluded from the verdict rather than averaged in.
//!
//! Bytes compared are PAYLOAD bytes: TCF's packed planes (SPECIFICATION.md
//! Section 12.2 logical payload, before the file writer's trailing alignment)
//! against GGUF's packed blocks. Those payload bytes are genuinely matched.
//! Container overhead is NOT counted, and it is NOT symmetric between the two
//! sides. Every quantized TCF tensor additionally carries a 128-byte proof
//! vector (`PROOF_COUNT` = 64 binary16 values, Section 15.3), a 16-byte
//! semantic digest, a 16-byte payload digest, and padding out to a 64-byte
//! physical span — roughly 160-190 bytes per tensor that this tool does not
//! charge TCF for. Read every byte figure here as payload only.
//!
//! # Why three error objectives, not one
//!
//! The two encoders do not minimize the same thing, so scoring both under one
//! side's objective decides the result before a tensor is read.
//!
//! - TCF's quantizer minimizes UNWEIGHTED squared error, in f64.
//! - GGUF's quantizers minimize a WEIGHTED squared error. Q6_K weights each
//!   element by `w = x*x`; Q4_K weights it by `w = sqrt(sum(x^2)/n) + |x|`.
//!   The weights exist to protect large-magnitude elements, which is a
//!   deliberate design choice, not an accident of implementation.
//!
//! So every tensor is scored under all three, plus max absolute error:
//!
//! | column           | weight `w_i`                | whose objective |
//! |------------------|-----------------------------|-----------------|
//! | `relRMS`         | 1                           | TCF's           |
//! | `relRMS w=x^2`   | `x_i^2`                     | GGUF Q6_K's     |
//! | `relRMS w=s+abs` | `sigma + abs(x_i)`          | GGUF Q4_K's     |
//! | `maxAbs`         | (worst single element)      | neither's       |
//!
//! Each weighted figure is `sqrt(sum(w_i * (src_i - rec_i)^2) / sum(w_i *
//! src_i^2))`, accumulated in f64, so it stays dimensionless and comparable
//! across tensors exactly as the unweighted one is. `sigma` is
//! `sqrt(sum(x^2)/n)` over the WHOLE tensor: the metric has to be identical
//! for both codecs, and a per-block sigma would have to borrow one codec's
//! block geometry. Both codecs go through the same metric function over the
//! same buffer in the same order.
//!
//! # The comparison is search-stage-asymmetric
//!
//! TCF has a search stage GGUF does not. After rounding the sub-scale, TCF
//! refines that ROUNDED value against the decoder's own reconstruction grid,
//! sweeping candidates around it (`refine_symmetric_sub_scale`,
//! `refine_asymmetric_sub_levels` in `tcf-core`). GGUF's second pass only
//! re-rounds codes against the scale it already chose. Neither figure below
//! is a like-for-like measurement of search effort.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example quant_accuracy_compare -- \
//!     /path/to/safetensors-checkpoint
//!
//! # only the attention output projections, first 8 of them
//! cargo run --release --example quant_accuracy_compare -- \
//!     /path/to/safetensors-checkpoint --filter o_proj --limit 8
//!
//! # include smaller tensors than the default floor
//! cargo run --release --example quant_accuracy_compare -- \
//!     /path/to/safetensors-checkpoint --min-elements 1024
//!
//! # the 64-element-tile regime, where the two sides differ in reach
//! cargo run --release --example quant_accuracy_compare -- \
//!     /path/to/safetensors-checkpoint --mode tile-64
//! ```
//!
//! # Which code actually runs
//!
//! No codec is reimplemented here. TCF goes through `tcf_core::encoding`'s
//! `quantize` / `pack` / `unpack` / `dequantize` — the CONFORMANCE.md reference
//! codec that `compressr`'s TCF writer calls. GGUF goes through `boostr`'s
//! `QuantizeOps` and `DequantOps` on `CpuClient` — the same writer and reader
//! kernels `compressr convert` and every GGUF consumer use.
//!
//! Both sides traverse their own BYTES. TCF quantizes to tiles, packs those
//! tiles to a payload, unpacks that payload back to tiles, and dequantizes
//! what came out of the unpack — never the tiles the quantizer returned. GGUF
//! quantizes to packed blocks and dequantizes those blocks. A packing bug on
//! either side therefore shows up in the error columns instead of hiding
//! behind an in-memory shortcut.
//!
//! The unweighted error metric is `compressr::relative_rms_error`. The
//! weighted ones are `weighted_relative_rms` below, which accumulates the same
//! way in f64 and is called with each codec's weight function.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;

use boostr::quant::{DequantOps, QuantFormat, QuantizeOps};
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor};
use compressr::loader::{ModelLoader, UnifiedDType};
use compressr::{cast_bytes, relative_rms_error};
use tcf_core::encoding::{dequantize, pack, quantize, unpack};
use tcf_core::NativeEncoding;

/// One bits-per-weight tier: a TCF encoding and, when GGUF has one at that
/// budget and granularity, the GGUF format occupying exactly the same bytes
/// for the same element count.
struct Tier {
    /// Nominal bits per weight, for the report header only. The per-tensor
    /// byte counts below are what actually proves the match, and
    /// `print_encoding_budgets` re-derives this figure from each side's own
    /// block layout rather than trusting either name.
    bpw: f64,
    tcf: NativeEncoding,
    /// `None` when GGUF has NO format at this tier's bits per weight and
    /// granularity. Such a tier still runs TCF and prints its reconstruction;
    /// what it cannot print is a ratio, and `gguf_gap` says why.
    gguf: Option<QuantFormat>,
    /// Set when the two sides are not two formats but ONE format written by
    /// two encoders. The text is printed with the tier so no reader takes
    /// that tier's numbers for a format result.
    same_format: Option<&'static str>,
    /// Set when `gguf` is `None`: the capability gap, stated in full so the
    /// report names it instead of silently skipping the tier.
    gguf_gap: Option<&'static str>,
    /// Set when the two sides are matched in bytes but NOT in granularity or
    /// symmetry. Printed with the tier so the reader is not told a
    /// like-for-like pairing that is not one.
    granularity_note: Option<&'static str>,
    /// The GGUF budgets that BRACKET this tier when `gguf` is `None`: the
    /// nearest encodable budget below and the nearest above, at the
    /// granularity this tier's TCF tile allows.
    ///
    /// Empty on every tier that has an equal-budget GGUF counterpart, so the
    /// equal-budget tiers print and score exactly as they did before.
    /// `verify_brackets` re-derives what belongs here from the formats' own
    /// layouts and reports whether this declaration matches.
    brackets: &'static [Bracket],
}

/// One GGUF format standing in for a budget TCF's tier cannot be matched at.
struct Bracket {
    side: BracketSide,
    format: QuantFormat,
}

/// Which way a bracket misses the TCF tier's budget.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BracketSide {
    /// Cheaper than TCF's budget: TCF must beat it on error to be worth the
    /// extra bits.
    Below,
    /// Dearer than TCF's budget: TCF matching it on error means equal quality
    /// at a smaller size.
    Above,
}

impl BracketSide {
    fn label(self) -> &'static str {
        match self {
            BracketSide::Below => "below",
            BracketSide::Above => "above",
        }
    }
}

impl Tier {
    /// The GGUF side's name, or a placeholder when the tier has no GGUF side.
    fn gguf_name(&self) -> &'static str {
        match self.gguf {
            Some(format) => format.name(),
            None => "(no GGUF format at this budget)",
        }
    }
}

/// Which set of encoding pairs the run compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// The original three pairs. Both sides carry a 256-element super-block,
    /// so both sides accept and refuse exactly the same shapes.
    SuperBlock256,
    /// TCF's 64-element-tile encodings against the best GGUF format at a
    /// matched byte budget. This is the regime where the two sides differ in
    /// what they can encode at all.
    Tile64,
}

// `static`, not `const`: `process_batch` below takes `&'static Tier`
// references into these arrays from inside a rayon closure, which a `const`
// cannot provide — each use of a `const` re-materializes a fresh temporary.
static SUPER_BLOCK_256_TIERS: [Tier; 3] = [
    Tier {
        bpw: 4.50,
        tcf: NativeEncoding::Q4AS32DT64,
        gguf: Some(QuantFormat::Q4K),
        same_format: None,
        gguf_gap: None,
        granularity_note: None,
        brackets: &[],
    },
    Tier {
        bpw: 6.5625,
        tcf: NativeEncoding::Q6S16DT64,
        gguf: Some(QuantFormat::Q6K),
        same_format: None,
        gguf_gap: None,
        granularity_note: None,
        brackets: &[],
    },
    Tier {
        bpw: 8.50,
        tcf: NativeEncoding::Q8S32T64,
        gguf: Some(QuantFormat::Q8_0),
        same_format: Some(
            "SAME FORMAT, TWO ENCODERS: Q8S32T64 and Q8_0 are bit-for-bit the \
             same layout — 32-element blocks, one binary16 scale, 8-bit codes \
             clamped to [-127, 127], 34 bytes per 32 elements. Any gap here is \
             encoder work, NOT a format result.",
        ),
        gguf_gap: None,
        granularity_note: None,
        brackets: &[],
    },
];

/// The 64-element-tile regime. Every TCF encoding here is flat-scale, so its
/// payload is exactly `tiles * bytes_per_tile` with no super-block to fit,
/// and it needs only `row % 64 == 0`.
static TILE_64_TIERS: [Tier; 4] = [
    Tier {
        bpw: 4.50,
        tcf: NativeEncoding::Q4S32T64,
        gguf: Some(QuantFormat::Q4_0),
        same_format: None,
        gguf_gap: None,
        granularity_note: None,
        brackets: &[],
    },
    Tier {
        bpw: 4.50,
        tcf: NativeEncoding::Q4AS64T64,
        gguf: Some(QuantFormat::Q4_0),
        same_format: None,
        gguf_gap: None,
        granularity_note: Some(
            "MATCHED IN BYTES, NOT IN GRANULARITY: Q4AS64T64 spends its 4.50 \
             bpw on an ASYMMETRIC scale-and-minimum pair per 64 weights; Q4_0 \
             spends the same budget on a SYMMETRIC scale per 32 weights. The \
             byte budget is the matched quantity here, nothing else.",
        ),
        brackets: &[],
    },
    Tier {
        bpw: 5.00,
        tcf: NativeEncoding::Q4AS32T64,
        gguf: Some(QuantFormat::Q4_1),
        same_format: None,
        gguf_gap: None,
        granularity_note: None,
        brackets: &[],
    },
    Tier {
        bpw: 6.50,
        tcf: NativeEncoding::Q6S32T64,
        gguf: None,
        same_format: None,
        gguf_gap: Some(
            "CAPABILITY GAP: GGUF has NO format at 6.50 bpw that works at \
             64-element granularity. Q6_K is 6.5625 bpw but needs \
             row % 256 == 0. GGUF's 32-element formats step 4.50 (Q4_0), \
             5.00 (Q4_1), 5.50 (Q5_0), 6.00 (Q5_1), 8.50 (Q8_0) — nothing at \
             6.50 — and of those boostr carries a CPU quantize kernel for \
             Q4_0, Q4_1 and Q8_0 only, so Q5_0 and Q5_1 cannot serve as a \
             comparison side even though their budgets sit between. There is \
             no equal-budget ratio to print. What IS a real deployment \
             choice is reported instead: TCF at 6.50 against the nearest \
             ENCODABLE GGUF budgets that BRACKET it, Q4_1 at 5.00 below and \
             Q8_0 at 8.50 above. Those rows differ in bytes ON PURPOSE and \
             carry both budgets everywhere they appear.",
        ),
        granularity_note: None,
        brackets: &[
            Bracket {
                side: BracketSide::Below,
                format: QuantFormat::Q4_1,
            },
            Bracket {
                side: BracketSide::Above,
                format: QuantFormat::Q8_0,
            },
        ],
    },
];

/// Every GGUF format this tool will consider putting on the other side of a
/// bracketed comparison.
///
/// This list is candidates only. `verify_brackets` filters it by each
/// format's OWN `block_size` against the TCF tile, and probes each survivor
/// with a real quantize call, so a format's membership here never decides
/// whether it can bracket anything.
static BRACKET_CANDIDATES: [QuantFormat; 7] = [
    QuantFormat::Q4_0,
    QuantFormat::Q4_1,
    QuantFormat::Q5_0,
    QuantFormat::Q5_1,
    QuantFormat::Q8_0,
    QuantFormat::Q8_1,
    QuantFormat::IQ4NL,
];

/// The tier table one mode compares.
fn tiers(mode: Mode) -> &'static [Tier] {
    match mode {
        Mode::SuperBlock256 => &SUPER_BLOCK_256_TIERS,
        Mode::Tile64 => &TILE_64_TIERS,
    }
}

/// TCF's bits per weight, from the layout's OWN byte arithmetic — never from
/// the encoding's name.
fn tcf_bits_per_weight(encoding: NativeEncoding) -> f64 {
    encoding.layout().bits_per_weight()
}

/// GGUF's bits per weight, from the format's OWN block layout — never from
/// the format's name.
fn gguf_bits_per_weight(format: QuantFormat) -> f64 {
    let block = format.block_size();
    if block == 0 {
        return 0.0;
    }
    (format.block_bytes() * 8) as f64 / block as f64
}

/// Whether boostr can actually WRITE this GGUF format on the CPU.
///
/// Established by calling the writer kernel on a one-block tensor, not by
/// listing names: a kernel added or removed in boostr changes this answer
/// without anyone editing a list here. boostr's CPU `QuantizeOps` covers a
/// strict subset of the formats its `DequantOps` reads, and a format it can
/// only READ cannot stand on the other side of a comparison at all.
fn gguf_quantize_supported(format: QuantFormat) -> bool {
    let block = format.block_size();
    if block == 0 {
        return false;
    }
    // A ramp with both signs and a non-zero mean, so no format's kernel takes
    // an all-zero or all-positive shortcut and reports success on a degenerate
    // input.
    let values: Vec<f32> = (0..block)
        .map(|index| (index as f32) * 0.125 - 2.0)
        .collect();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let Ok(input) = Tensor::<CpuRuntime>::from_slice(&values, &[1, block], &device) else {
        return false;
    };
    client.quantize(&input, format).is_ok()
}

/// One GGUF format's standing as a possible bracket for a TCF tier.
struct BracketCandidate {
    format: QuantFormat,
    bpw: f64,
    /// Whether boostr's CPU writer can produce it, PROBED, not assumed.
    encodable: bool,
}

/// The candidate GGUF formats a tier's TCF tile can actually be compared
/// against, each with its own layout-derived budget and a probed writer flag.
///
/// A format qualifies on granularity when its block divides the TCF tile: a
/// 32-element block tiles a 64-element row, a 256-element block does not.
/// That is derived from `block_size` here rather than stated in prose.
fn bracket_candidates(tile: usize) -> Vec<BracketCandidate> {
    let mut candidates: Vec<BracketCandidate> = BRACKET_CANDIDATES
        .iter()
        .filter_map(|format| {
            let block = format.block_size();
            if block == 0 || tile == 0 || !tile.is_multiple_of(block) {
                return None;
            }
            Some(BracketCandidate {
                format: *format,
                bpw: gguf_bits_per_weight(*format),
                encodable: gguf_quantize_supported(*format),
            })
        })
        .collect();
    // Deterministic order, by budget then by name, so two runs print the same
    // table and two formats at one budget never swap places.
    candidates.sort_by(|a, b| {
        a.bpw
            .partial_cmp(&b.bpw)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.format.name().cmp(b.format.name()))
    });
    candidates
}

/// Re-derive the nearest encodable budget on one side of a target, from the
/// candidates' own layouts.
fn nearest_encodable(
    candidates: &[BracketCandidate],
    target: f64,
    side: BracketSide,
) -> Option<&BracketCandidate> {
    match side {
        // Candidates are sorted ascending, so the nearest below is the last
        // encodable one under the target and the nearest above is the first
        // encodable one over it.
        BracketSide::Below => candidates.iter().rfind(|c| c.encodable && c.bpw < target),
        BracketSide::Above => candidates.iter().find(|c| c.encodable && c.bpw > target),
    }
}

/// Print the bracket check for one tier: every candidate at this granularity,
/// which of them boostr can write, which pair genuinely brackets the TCF
/// budget, and whether the tier's declared brackets are that pair.
///
/// The declaration in the tier table is what the run compares against, so it
/// is checked against the layouts rather than trusted. A format whose budget
/// sits between the brackets but which boostr cannot write is named, because
/// its absence is a boostr limitation and not a GGUF one.
fn verify_brackets(tier: &Tier) {
    if tier.brackets.is_empty() {
        return;
    }
    let tcf_bpw = tcf_bits_per_weight(tier.tcf);
    let tile = usize::from(tier.tcf.geometry().tile);
    let candidates = bracket_candidates(tile);

    println!(
        "    BRACKET CHECK: TCF {:?} at {:.4} bpw, tile {}. GGUF candidates whose block divides \
         that tile:",
        tier.tcf, tcf_bpw, tile
    );
    for candidate in &candidates {
        println!(
            "      {:<8} {:.4} bpw  block {:>3}  boostr CPU quantize kernel: {}",
            candidate.format.name(),
            candidate.bpw,
            candidate.format.block_size(),
            if candidate.encodable { "YES" } else { "NO" }
        );
    }

    for bracket in tier.brackets {
        let declared_bpw = gguf_bits_per_weight(bracket.format);
        let nearest = nearest_encodable(&candidates, tcf_bpw, bracket.side);
        let verdict = match nearest {
            Some(candidate) if candidate.format == bracket.format => {
                "IS the nearest encodable budget on that side".to_string()
            }
            Some(candidate) => format!(
                "NOT the nearest — {} at {:.4} bpw is nearer, so this bracket is wider than it \
                 needs to be",
                candidate.format.name(),
                candidate.bpw
            ),
            None => "no encodable GGUF budget exists on that side at this granularity".to_string(),
        };
        println!(
            "      declared bracket {:<5} : {} at {:.4} bpw ({:+.4} bpw against TCF) — {}",
            bracket.side.label(),
            bracket.format.name(),
            declared_bpw,
            declared_bpw - tcf_bpw,
            verdict
        );
    }

    // The budgets GGUF defines between the brackets but boostr cannot write.
    // Named, because "GGUF has nothing there" and "boostr cannot write what
    // GGUF has there" are different claims and only the second one is true.
    let below = tier
        .brackets
        .iter()
        .find(|b| b.side == BracketSide::Below)
        .map(|b| gguf_bits_per_weight(b.format));
    let above = tier
        .brackets
        .iter()
        .find(|b| b.side == BracketSide::Above)
        .map(|b| gguf_bits_per_weight(b.format));
    if let (Some(below), Some(above)) = (below, above) {
        let excluded: Vec<String> = candidates
            .iter()
            .filter(|c| !c.encodable && c.bpw > below && c.bpw < above)
            .map(|c| format!("{} at {:.4} bpw", c.format.name(), c.bpw))
            .collect();
        if excluded.is_empty() {
            println!(
                "      GGUF defines no further budget between {below:.4} and {above:.4} bpw at \
                 this granularity."
            );
        } else {
            println!(
                "      between {:.4} and {:.4} bpw GGUF DOES define {}, but boostr has no CPU \
                 quantize kernel for {}, so {} cannot serve as a bracket.",
                below,
                above,
                excluded.join(" and "),
                if excluded.len() == 1 { "it" } else { "them" },
                if excluded.len() == 1 { "it" } else { "they" }
            );
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "quant_accuracy_compare",
    about = "Per-tensor TCF vs GGUF reconstruction error at matched bits per weight"
)]
struct Args {
    /// Checkpoint to read: a SafeTensors directory, or any model compressr's
    /// loader opens.
    model: String,

    /// Keep only tensors whose name contains this substring.
    #[arg(long)]
    filter: Option<String>,

    /// Compare at most this many tensors, in the loader's deterministic order.
    #[arg(long)]
    limit: Option<usize>,

    /// Smallest tensor compared, in elements. A tiny tensor's error is
    /// dominated by a handful of groups and ranks nothing.
    #[arg(long, default_value_t = 4096)]
    min_elements: usize,

    /// Which encoding pairs to compare. Defaults to the original three
    /// 256-element-super-block pairs, so a prior run reproduces unchanged.
    #[arg(long, value_enum, default_value_t = Mode::SuperBlock256)]
    mode: Mode,
}

/// What one codec produced for one tensor at one tier: the same reconstruction
/// scored under every objective, so no single objective decides the report.
struct CodecResult {
    /// Relative RMS with no weighting — the objective TCF's quantizer
    /// minimizes.
    relative_rms: f64,
    /// Relative RMS weighted by `w = x*x` — the objective GGUF's Q6_K
    /// quantizer minimizes.
    relative_rms_w_squared: f64,
    /// Relative RMS weighted by `w = sigma + abs(x)`, `sigma` the RMS of the
    /// whole source tensor — the objective GGUF's Q4_K quantizer minimizes.
    relative_rms_w_rms_abs: f64,
    /// Worst single element, which no encoder here minimizes.
    max_abs: f64,
    bytes: usize,
    elements: usize,
}

impl CodecResult {
    /// This reconstruction's score under one objective. Lower is better under
    /// all four, so a ratio below 1.0 always means the numerator won.
    fn score(&self, objective: Objective) -> f64 {
        match objective {
            Objective::Unweighted => self.relative_rms,
            Objective::SquaredMagnitude => self.relative_rms_w_squared,
            Objective::RmsPlusMagnitude => self.relative_rms_w_rms_abs,
            Objective::MaxAbsolute => self.max_abs,
        }
    }
}

/// One error objective the report scores both codecs under.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Objective {
    /// `w = 1`. TCF's quantizer minimizes this.
    Unweighted,
    /// `w = x*x`. GGUF's Q6_K quantizer minimizes this.
    SquaredMagnitude,
    /// `w = sigma + abs(x)`. GGUF's Q4_K quantizer minimizes this.
    RmsPlusMagnitude,
    /// Worst single element. Neither encoder minimizes this.
    MaxAbsolute,
}

/// Every objective, in report order. Scoring under one of these alone is what
/// made the earlier version of this tool circular, so the report never prints
/// one without the others.
static OBJECTIVES: [Objective; 4] = [
    Objective::Unweighted,
    Objective::SquaredMagnitude,
    Objective::RmsPlusMagnitude,
    Objective::MaxAbsolute,
];

impl Objective {
    /// Short column label.
    fn label(self) -> &'static str {
        match self {
            Objective::Unweighted => "relRMS unweighted",
            Objective::SquaredMagnitude => "relRMS w=x^2",
            Objective::RmsPlusMagnitude => "relRMS w=sigma+abs(x)",
            Objective::MaxAbsolute => "max abs error",
        }
    }

    /// Whose objective this is, printed beside the label so the reader never
    /// has to remember which encoder minimizes which.
    fn owner(self) -> &'static str {
        match self {
            Objective::Unweighted => "TCF's own objective",
            Objective::SquaredMagnitude => "GGUF Q6_K's own objective",
            Objective::RmsPlusMagnitude => "GGUF Q4_K's own objective",
            Objective::MaxAbsolute => "neither encoder's objective",
        }
    }

    /// A relative RMS reads as a percentage; an absolute error does not.
    fn is_relative(self) -> bool {
        self != Objective::MaxAbsolute
    }

    /// One codec's score, formatted for a table cell.
    fn format(self, value: f64) -> String {
        if self.is_relative() {
            format!("{:>12.4}%", value * 100.0)
        } else {
            format!("{value:>13.3e}")
        }
    }
}

/// One tensor's outcome at one tier.
enum Outcome {
    /// Both codecs ran. They may still disagree on byte or element count,
    /// which `matched` records.
    Ran {
        tcf: CodecResult,
        gguf: CodecResult,
        matched: bool,
        mismatch: Option<String>,
    },
    /// The tier has no EQUAL-BUDGET GGUF side. TCF ran, and so did each GGUF
    /// format that brackets TCF's budget — the nearest encodable one below and
    /// the nearest above. There is still no equal-bpw ratio to print; the
    /// bracketed ones are, and they carry both budgets. The tier's `gguf_gap`
    /// says what GGUF is missing at the matched budget.
    NoGgufCounterpart {
        tcf: CodecResult,
        /// One entry per declared bracket, in the tier's declaration order.
        brackets: Vec<BracketOutcome>,
    },
    /// This tier cannot encode this tensor's shape, on one side or both.
    /// Each side's refusal is kept SEPARATELY: pooling them hides which codec
    /// is the limited one, which is the whole claim under test.
    Unsupported {
        /// TCF's refusal, or `None` when TCF could encode this shape.
        tcf: Option<String>,
        /// GGUF's refusal, or `None` when GGUF could encode this shape. Also
        /// `None` on a tier with no GGUF side, where GGUF has nothing to
        /// refuse with.
        gguf: Option<String>,
    },
    /// A codec refused the tensor's values, or a call failed.
    Failed(String),
}

impl Outcome {
    /// Both sides' refusals as one line, for the per-tensor status column.
    /// The verdict counts them separately; a table cell reads better joined.
    fn unsupported_reason(tcf: Option<&str>, gguf: Option<&str>) -> String {
        let mut parts = Vec::new();
        if let Some(reason) = tcf {
            parts.push(format!("TCF: {reason}"));
        }
        if let Some(reason) = gguf {
            parts.push(format!("GGUF: {reason}"));
        }
        parts.join("; ")
    }
}

/// One bracketing GGUF format's result for one tensor.
///
/// The byte count here DELIBERATELY differs from TCF's: a bracket is a
/// different budget, which is the whole point of printing it. So no
/// byte-match check applies, and `bpw` travels with every result so the
/// difference is stated wherever the numbers are.
struct BracketOutcome {
    side: BracketSide,
    format: QuantFormat,
    /// This format's bits per weight, from its own block layout.
    bpw: f64,
    state: BracketState,
}

/// What the bracketing format did on one tensor.
enum BracketState {
    Ran(CodecResult),
    /// The row is not a whole number of this format's blocks.
    Unsupported(String),
    /// The kernel refused the values, or a call failed.
    Failed(String),
}

impl BracketState {
    fn result(&self) -> Option<&CodecResult> {
        match self {
            BracketState::Ran(result) => Some(result),
            BracketState::Unsupported(_) | BracketState::Failed(_) => None,
        }
    }
}

/// One tensor's row in the report.
struct Row {
    name: String,
    shape: Vec<usize>,
    outcome: Outcome,
}

/// Upper bound, in f32 elements, held across all tensors of one batch at
/// once. Tensors are read and converted to f32 in bounded batches rather than
/// all at once: an embedding tensor's f32 buffer can be far larger than every
/// other tensor's in a checkpoint, and this bound keeps the largest batch's
/// memory well short of holding the whole checkpoint as f32.
const BATCH_ELEMENT_BUDGET: usize = 256 * 1024 * 1024; // 1 GiB of f32 data

fn main() -> Result<()> {
    let args = Args::parse();
    let mut model = ModelLoader::new(&args.model)
        .load()
        .with_context(|| format!("Failed to open {}", args.model))?;

    // An already-quantized source would compare RE-quantized weights against a
    // reference that is itself lossy, so both codecs would be scored on damage
    // neither of them caused. The result would look like a real comparison, so
    // refuse rather than print it.
    match model.format() {
        compressr::InputFormat::Safetensors | compressr::InputFormat::PyTorch => {}
        other => {
            anyhow::bail!(
                "{} is {other:?}, which is already quantized. This tool measures \
                 quantization error against a full-precision reference, so it needs \
                 a Safetensors or PyTorch checkpoint.",
                args.model
            );
        }
    }

    let tiers = tiers(args.mode);

    println!("model    : {}", args.model);
    println!("format   : {:?}", model.format());
    println!("tensors  : {}", model.tensor_count());
    println!("workers  : {} rayon threads", rayon::current_num_threads());
    println!("mode     : {:?}", args.mode);

    let (selected, skipped) = select_tensors(&model, &args);

    println!();
    println!(
        "== skipped {} tensors (not comparable weight matrices) ==",
        skipped.len()
    );
    for (name, reason) in &skipped {
        println!("  {name}: {reason}");
    }

    println!();
    println!("== comparing {} tensors ==", selected.len());
    print_encoding_budgets(tiers);

    // One row list per tier, in the same tensor order, so a reader can scan
    // one tier top to bottom without a per-tensor block of six numbers.
    let mut tier_rows: Vec<Vec<Row>> = tiers.iter().map(|_| Vec::new()).collect();

    // `ModelLoader::read_tensor_bytes` seeks a shared file handle and takes
    // `&mut self`, so it is not `Sync` and cannot be called from multiple
    // rayon workers at once. Reading and converting to f32 therefore stays
    // serial, one tensor at a time, exactly as before. The expensive part —
    // quantizing, dequantizing and scoring both codecs at every tier —
    // is what runs in parallel, in `process_batch` below, over batches
    // bounded by `BATCH_ELEMENT_BUDGET` so at most one batch's worth of f32
    // data is resident at once rather than the whole checkpoint's.
    let mut batch: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut batch_elements = 0usize;

    for (name, shape, dtype) in &selected {
        let bytes = model
            .read_tensor_bytes(name)
            .with_context(|| format!("Failed to read tensor '{name}'"))?;
        // Read once, converted once. Both codecs see this identical buffer, so
        // no dtype conversion difference can leak into the comparison.
        let values = to_f32(&bytes, *dtype)
            .with_context(|| format!("Failed to convert tensor '{name}' to f32"))?;
        let numel: usize = shape.iter().product();
        if values.len() != numel {
            bail!(
                "Tensor '{name}' has shape {shape:?} ({numel} elements) but its bytes decode to \
                 {} values",
                values.len()
            );
        }

        batch_elements += values.len();
        batch.push((name.clone(), shape.clone(), values));

        if batch_elements >= BATCH_ELEMENT_BUDGET {
            process_batch(&batch, tiers, &mut tier_rows);
            batch.clear();
            batch_elements = 0;
        }
    }
    if !batch.is_empty() {
        process_batch(&batch, tiers, &mut tier_rows);
    }

    for (tier, rows) in tiers.iter().zip(&tier_rows) {
        print_tier(tier, rows);
    }

    print_verdict(tiers, &tier_rows);
    Ok(())
}

/// Print, per tier, the bits per weight each side ACTUALLY spends, derived
/// from its own block layout, plus the granularity each side works at.
///
/// The names claim a budget; the layouts prove one. A tier whose two sides
/// disagree here is not a matched comparison, and this says so up front
/// rather than leaving it to the per-tensor byte counts to expose.
fn print_encoding_budgets(tiers: &[Tier]) {
    for tier in tiers {
        let tcf_bpw = tcf_bits_per_weight(tier.tcf);
        let geometry = tier.tcf.geometry();
        match tier.gguf {
            Some(format) => {
                let gguf_bpw = gguf_bits_per_weight(format);
                // A tolerance, not equality: both figures are computed in f64
                // from integer byte counts, so an exact match is expected and
                // a real difference is far larger than this.
                let verdict = if (tcf_bpw - gguf_bpw).abs() < 1e-9 {
                    "bpw MATCHED"
                } else {
                    "bpw MISMATCH — not a matched comparison"
                };
                println!(
                    "  TCF {:?} {:.4} bpw (scale group {}, tile {}, {}) vs GGUF {} {:.4} bpw \
                     (block {}) — {}",
                    tier.tcf,
                    tcf_bpw,
                    geometry.group,
                    geometry.tile,
                    if geometry.symmetric {
                        "symmetric"
                    } else {
                        "asymmetric"
                    },
                    format.name(),
                    gguf_bpw,
                    format.block_size(),
                    verdict
                );
            }
            None => {
                println!(
                    "  TCF {:?} {:.4} bpw (scale group {}, tile {}, {}) vs {}",
                    tier.tcf,
                    tcf_bpw,
                    geometry.group,
                    geometry.tile,
                    if geometry.symmetric {
                        "symmetric"
                    } else {
                        "asymmetric"
                    },
                    tier.gguf_name()
                );
            }
        }
        if let Some(note) = tier.same_format {
            println!("    {note}");
        }
        if let Some(note) = tier.granularity_note {
            println!("    {note}");
        }
        if let Some(note) = tier.gguf_gap {
            println!("    {note}");
        }
        verify_brackets(tier);
    }
}

/// Run every (tensor, tier) pair in one batch across rayon's pool, then append
/// each tensor's per-tier outcome to `tier_rows` in the batch's original
/// order — the axis is (tensor, tier), not tensor alone, so a checkpoint
/// dominated by one huge tensor still spreads its tiers across workers
/// instead of leaving them idle behind it.
///
/// `Vec::into_par_iter` is an indexed parallel iterator: `.collect::<Vec<_>>()`
/// re-assembles results at the same positions they were submitted at,
/// regardless of which worker finished first or in what order. That is what
/// keeps this report byte-identical in content and order to the serial
/// version — no explicit re-sort is needed after the collect.
fn process_batch(
    batch: &[(String, Vec<usize>, Vec<f32>)],
    tiers: &'static [Tier],
    tier_rows: &mut [Vec<Row>],
) {
    let tier_count = tiers.len();
    let jobs: Vec<(usize, usize)> = (0..batch.len())
        .flat_map(|tensor_index| (0..tier_count).map(move |tier_index| (tensor_index, tier_index)))
        .collect();

    let outcomes: Vec<Outcome> = jobs
        .into_par_iter()
        .map(|(tensor_index, tier_index)| {
            let (_, shape, values) = &batch[tensor_index];
            compare_tensor(values, shape, &tiers[tier_index])
        })
        .collect();

    let mut outcomes = outcomes.into_iter();
    for (name, shape, _values) in batch {
        for rows in tier_rows.iter_mut().take(tier_count) {
            let outcome = outcomes
                .next()
                .expect("exactly one outcome per (tensor, tier) pair in this batch");
            rows.push(Row {
                name: name.clone(),
                shape: shape.clone(),
                outcome,
            });
        }
    }
}

/// Split the model's tensors into those worth comparing and those that are not,
/// each skip carrying the reason it was skipped.
#[allow(clippy::type_complexity)]
fn select_tensors(
    model: &compressr::loader::LoadedModel,
    args: &Args,
) -> (
    Vec<(String, Vec<usize>, UnifiedDType)>,
    Vec<(String, String)>,
) {
    let mut selected = Vec::new();
    let mut skipped = Vec::new();

    for (name, info) in model.tensors() {
        if let Some(filter) = &args.filter {
            if !name.contains(filter.as_str()) {
                continue;
            }
        }
        if !info.dtype.is_float() {
            skipped.push((name, format!("dtype {} is not float", info.dtype)));
            continue;
        }
        if info.shape.len() < 2 {
            skipped.push((
                name,
                format!(
                    "rank {} has no row to pack blocks along (norm weight or bias)",
                    info.shape.len()
                ),
            ));
            continue;
        }
        if info.numel < args.min_elements {
            skipped.push((
                name,
                format!(
                    "{} elements is below the --min-elements floor of {}",
                    info.numel, args.min_elements
                ),
            ));
            continue;
        }
        if let Some(limit) = args.limit {
            if selected.len() >= limit {
                skipped.push((name, format!("past the --limit of {limit}")));
                continue;
            }
        }
        selected.push((name, info.shape, info.dtype));
    }

    (selected, skipped)
}

/// Decode a tensor's raw little-endian bytes to f32, ONCE, whatever float type
/// the checkpoint stored. `compressr::cast_bytes` does the conversion, so bf16
/// and f16 follow exactly the rule a conversion follows.
fn to_f32(bytes: &[u8], dtype: UnifiedDType) -> Result<Vec<f32>> {
    let owned;
    let f32_bytes: &[u8] = if dtype == UnifiedDType::F32 {
        bytes
    } else {
        owned = cast_bytes(bytes, dtype, UnifiedDType::F32)?;
        &owned
    };
    if !f32_bytes.len().is_multiple_of(4) {
        bail!(
            "{} bytes is not a whole number of f32 elements",
            f32_bytes.len()
        );
    }
    Ok(f32_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Run both codecs over one tensor at one tier, or TCF alone when the tier has
/// no GGUF counterpart.
fn compare_tensor(values: &[f32], shape: &[usize], tier: &Tier) -> Outcome {
    // The row is the LAST dimension, and that is what both codecs use, not a
    // convention adopted here: `tcf-core`'s `row_major_tiles` sets
    // `K = dims[rank - 1]` (SPECIFICATION.md Section 12.3) and boostr's
    // `QuantizeOps::quantize` gates on `shape.last() % block_size`. A rank-3
    // tensor is NOT flattened to `[shape[0], product(shape[1..])]` by either
    // side, so flattening it here would report a reach neither codec has.
    let row_len = match shape.last() {
        Some(len) => *len,
        None => {
            return Outcome::Unsupported {
                tcf: Some("rank 0 has no row".to_string()),
                gguf: Some("rank 0 has no row".to_string()),
            }
        }
    };

    // A row that is not a whole number of blocks cannot be encoded at all,
    // and the two sides have different block widths, so the check is per side
    // and each side's refusal is kept apart from the other's.
    let tile = usize::from(tier.tcf.geometry().tile);
    let tcf_refusal = (tile == 0 || !row_len.is_multiple_of(tile))
        .then(|| format!("row {row_len} is not a whole number of {tile}-element TCF tiles"));
    let gguf_refusal = tier.gguf.and_then(|format| {
        let block = format.block_size();
        (block == 0 || !row_len.is_multiple_of(block)).then(|| {
            format!(
                "row {row_len} is not a whole number of {block}-element {} blocks",
                format.name()
            )
        })
    });
    if tcf_refusal.is_some() || gguf_refusal.is_some() {
        return Outcome::Unsupported {
            tcf: tcf_refusal,
            gguf: gguf_refusal,
        };
    }

    let tcf = match tcf_result(values, shape, tier.tcf) {
        Ok(result) => result,
        Err(e) => return Outcome::Failed(format!("TCF {:?}: {e:#}", tier.tcf)),
    };
    let Some(gguf_format) = tier.gguf else {
        // No GGUF format occupies this budget at this granularity, so no
        // equal-bpw ratio exists and none is invented. The bracketing budgets
        // DO exist and are what a deployment actually chooses between, so each
        // one is run here and reported beside TCF with its own bits per
        // weight attached.
        let brackets = tier
            .brackets
            .iter()
            .map(|bracket| {
                let block = bracket.format.block_size();
                let state = if block == 0 || !row_len.is_multiple_of(block) {
                    BracketState::Unsupported(format!(
                        "row {row_len} is not a whole number of {block}-element {} blocks",
                        bracket.format.name()
                    ))
                } else {
                    match gguf_result(values, shape, bracket.format) {
                        Ok(result) => BracketState::Ran(result),
                        Err(e) => {
                            BracketState::Failed(format!("GGUF {}: {e:#}", bracket.format.name()))
                        }
                    }
                };
                BracketOutcome {
                    side: bracket.side,
                    format: bracket.format,
                    bpw: gguf_bits_per_weight(bracket.format),
                    state,
                }
            })
            .collect();
        return Outcome::NoGgufCounterpart { tcf, brackets };
    };
    let gguf = match gguf_result(values, shape, gguf_format) {
        Ok(result) => result,
        Err(e) => return Outcome::Failed(format!("GGUF {}: {e:#}", gguf_format.name())),
    };

    // A disagreement here is either a bug or a genuine shape-driven mismatch.
    // Either way the tensor is not a matched comparison and must not be
    // averaged into the verdict.
    let mut mismatch = Vec::new();
    if tcf.elements != gguf.elements {
        mismatch.push(format!(
            "element count TCF {} vs GGUF {}",
            tcf.elements, gguf.elements
        ));
    }
    if tcf.bytes != gguf.bytes {
        mismatch.push(format!(
            "byte count TCF {} vs GGUF {}",
            tcf.bytes, gguf.bytes
        ));
    }
    let matched = mismatch.is_empty();
    Outcome::Ran {
        tcf,
        gguf,
        matched,
        mismatch: (!matched).then(|| mismatch.join("; ")),
    }
}

/// TCF: `tcf-core`'s reference quantizer, packer and dequantizer.
fn tcf_result(values: &[f32], shape: &[usize], encoding: NativeEncoding) -> Result<CodecResult> {
    let layout = encoding.layout();
    let dims: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
    let rank = u32::try_from(dims.len()).map_err(|_| anyhow!("rank does not fit u32"))?;

    let tiles = quantize(values, &dims, rank, layout).map_err(|e| anyhow!("{e}"))?;
    // The real packed payload, measured. Not a bpw formula.
    let packed = pack(&tiles, layout).map_err(|e| anyhow!("{e}"))?;
    // Dequantize what came back OUT of the bytes, not the tiles the quantizer
    // returned. GGUF's side has always gone f32 -> packed blocks -> dequantize,
    // and reconstructing TCF from in-memory tiles would let a packing bug pass
    // unmeasured on one side only.
    let tile_count =
        u64::try_from(tiles.len()).map_err(|_| anyhow!("tile count does not fit u64"))?;
    let unpacked = unpack(&packed, tile_count, layout).map_err(|e| anyhow!("{e}"))?;
    let reconstructed = dequantize(&unpacked, layout).map_err(|e| anyhow!("{e}"))?;

    metrics(values, &reconstructed, packed.len())
}

/// GGUF: boostr's CPU writer kernel and the reader kernel it is built as the
/// inverse of — the same pair every GGUF consumer reads with.
fn gguf_result(values: &[f32], shape: &[usize], format: QuantFormat) -> Result<CodecResult> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(values, shape, &device)
        .map_err(|e| anyhow!("failed to stage tensor: {e}"))?;
    let quantized = client
        .quantize(&input, format)
        .map_err(|e| anyhow!("quantize failed: {e}"))?;
    let bytes = quantized.storage_bytes();
    let dequantized = client
        .dequantize(&quantized, DType::F32)
        .map_err(|e| anyhow!("dequantize failed: {e}"))?;
    let reconstructed = dequantized.to_vec::<f32>();

    metrics(values, &reconstructed, bytes)
}

/// Score one reconstruction under every objective at once.
///
/// BOTH codecs call this, on the same source buffer, in the same order, so no
/// part of the difference between them can come from the measurement. The
/// unweighted figure is `compressr::relative_rms_error`, unchanged: the same
/// number the TCF bump screen ranks tensors by. The weighted figures go
/// through `weighted_relative_rms` with the weight each GGUF quantizer
/// actually minimizes under.
fn metrics(source: &[f32], reconstructed: &[f32], bytes: usize) -> Result<CodecResult> {
    // Checked here so the message names the element spaces, not a metric.
    if source.len() != reconstructed.len() {
        bail!(
            "codec reconstructed {} values from a {}-element source",
            reconstructed.len(),
            source.len()
        );
    }
    let relative_rms = relative_rms_error(source, reconstructed)?;
    let relative_rms_w_squared =
        weighted_relative_rms(source, reconstructed, |value| value * value);
    // Q4_K's weight adds a floor term so a near-zero element is not scored at
    // near-zero weight. `sigma` is the source tensor's own RMS.
    let sigma = source_rms(source);
    let relative_rms_w_rms_abs =
        weighted_relative_rms(source, reconstructed, |value| sigma + value.abs());
    let mut max_abs = 0.0f64;
    for (value, reconstruction) in source.iter().zip(reconstructed) {
        let difference = (f64::from(*value) - f64::from(*reconstruction)).abs();
        if difference > max_abs {
            max_abs = difference;
        }
    }
    Ok(CodecResult {
        relative_rms,
        relative_rms_w_squared,
        relative_rms_w_rms_abs,
        max_abs,
        bytes,
        elements: reconstructed.len(),
    })
}

/// `sqrt(sum(w_i * (src_i - rec_i)^2) / sum(w_i * src_i^2))`, accumulated in
/// f64.
///
/// The weight is applied to the numerator and the denominator alike, so the
/// result stays dimensionless and comparable across tensors of any scale —
/// the same property `relative_rms_error` has, which is why the unweighted
/// path is that function and is not re-derived here.
///
/// `weight` receives the SOURCE value, never the reconstruction: a weight that
/// depended on what a codec produced would rank each codec on a different
/// element space.
///
/// A denominator of zero means every weighted source element is zero, which
/// only a codec that reconstructed them exactly can be scored against. It
/// returns 0.0 rather than a NaN, matching `relative_rms_error`.
///
/// The caller has already checked the two slices are the same length.
fn weighted_relative_rms<W: Fn(f64) -> f64>(
    source: &[f32],
    reconstructed: &[f32],
    weight: W,
) -> f64 {
    let mut error_energy = 0.0f64;
    let mut source_energy = 0.0f64;
    for (value, reconstruction) in source.iter().zip(reconstructed) {
        let value = f64::from(*value);
        let w = weight(value);
        let difference = value - f64::from(*reconstruction);
        error_energy += w * difference * difference;
        source_energy += w * value * value;
    }
    if source_energy == 0.0 {
        return 0.0;
    }
    (error_energy / source_energy).sqrt()
}

/// `sqrt(sum(x^2) / n)` over the WHOLE tensor, in f64.
///
/// GGUF's Q4_K kernel takes this over one block. Taken per block here, the
/// metric would inherit one codec's block geometry and stop being neutral
/// between the two, so it is taken over the tensor and labelled that way in
/// the report.
fn source_rms(source: &[f32]) -> f64 {
    if source.is_empty() {
        return 0.0;
    }
    let energy: f64 = source.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    (energy / source.len() as f64).sqrt()
}

/// `a / b`, or `None` when `b` is zero and the ratio would be meaningless.
fn ratio(a: f64, b: f64) -> Option<f64> {
    (b != 0.0).then(|| a / b)
}

fn show_ratio(value: Option<f64>) -> String {
    match value {
        Some(r) => format!("{r:>8.4}"),
        None => "     n/a".to_string(),
    }
}

fn print_tier(tier: &Tier, rows: &[Row]) {
    // From the layout, not from `tier.bpw`, because every bracketed line below
    // states TCF's budget against a different one and the two figures have to
    // come from the same kind of source.
    let tcf_bpw = tcf_bits_per_weight(tier.tcf);
    println!();
    println!(
        "================ {:.4} bpw : TCF {:?} vs GGUF {} ================",
        tier.bpw,
        tier.tcf,
        tier.gguf_name()
    );
    if let Some(note) = tier.same_format {
        println!("  {note}");
    }
    if let Some(note) = tier.granularity_note {
        println!("  {note}");
    }
    if let Some(note) = tier.gguf_gap {
        println!("  {note}");
    }

    // Bytes and status first, once per tensor. The metric tables below repeat
    // neither, so each of them stays narrow enough to read.
    println!();
    println!("-- encoded payload bytes and status --");
    println!(
        "{:<48} {:>18} {:>12} {:>12}  status",
        "tensor", "shape", "bytes TCF", "bytes GGUF"
    );
    for row in rows {
        let shape = format!("{:?}", row.shape);
        match &row.outcome {
            Outcome::Ran {
                tcf,
                gguf,
                matched,
                mismatch,
            } => {
                let status = if *matched {
                    "matched".to_string()
                } else {
                    format!(
                        "UNMATCHED — EXCLUDED: {}",
                        mismatch.as_deref().unwrap_or("shape disagreement")
                    )
                };
                println!(
                    "{:<48} {:>18} {:>12} {:>12}  {}",
                    row.name, shape, tcf.bytes, gguf.bytes, status
                );
            }
            Outcome::NoGgufCounterpart { tcf, brackets } => {
                println!(
                    "{:<48} {:>18} {:>12} {:>12}  TCF ONLY: no GGUF format at this budget",
                    row.name, shape, tcf.bytes, "n/a"
                );
                for bracket in brackets {
                    // Indented under the tensor it belongs to, and never in the
                    // "bytes GGUF" column of the row above: these bytes are a
                    // DIFFERENT budget's bytes, not a matched count.
                    match &bracket.state {
                        BracketState::Ran(result) => {
                            let size = ratio(tcf_bpw, bracket.bpw)
                                .map(|r| format!("TCF spends {:.1}% of these bytes", r * 100.0))
                                .unwrap_or_else(|| "size ratio n/a".to_string());
                            println!(
                                "  {:<46} {:>18} {:>12} {:>12}  BRACKETED {} — {} at {:.4} bpw vs \
                                 TCF {:.4} bpw, {}",
                                format!("^ bracket {}", bracket.side.label()),
                                "",
                                "",
                                result.bytes,
                                bracket.side.label(),
                                bracket.format.name(),
                                bracket.bpw,
                                tcf_bpw,
                                size
                            );
                        }
                        BracketState::Unsupported(reason) => {
                            println!(
                                "  {:<46} {:>18} {:>12} {:>12}  BRACKETED {} UNSUPPORTED: {}",
                                format!("^ bracket {}", bracket.side.label()),
                                "",
                                "",
                                "n/a",
                                bracket.side.label(),
                                reason
                            );
                        }
                        BracketState::Failed(reason) => {
                            println!(
                                "  {:<46} {:>18} {:>12} {:>12}  BRACKETED {} ERROR: {}",
                                format!("^ bracket {}", bracket.side.label()),
                                "",
                                "",
                                "n/a",
                                bracket.side.label(),
                                reason
                            );
                        }
                    }
                }
            }
            Outcome::Unsupported { tcf, gguf } => {
                println!(
                    "{:<48} {:>18}  UNSUPPORTED at this tier: {}",
                    row.name,
                    shape,
                    Outcome::unsupported_reason(tcf.as_deref(), gguf.as_deref())
                );
            }
            Outcome::Failed(reason) => {
                println!("{:<48} {:>18}  ERROR: {}", row.name, shape, reason);
            }
        }
    }

    // One table per objective, over the tensors both codecs actually encoded.
    // A tensor absent here appears with its reason in the status table above.
    for objective in &OBJECTIVES {
        println!();
        println!("-- {} ({}) --", objective.label(), objective.owner());
        println!(
            "{:<48} {:>18} {:>13} {:>13} {:>8}  status",
            "tensor", "shape", "TCF", "GGUF", "ratio"
        );
        for row in rows {
            let shape = format!("{:?}", row.shape);
            match &row.outcome {
                Outcome::Ran {
                    tcf,
                    gguf,
                    matched,
                    mismatch,
                } => {
                    let status = if *matched {
                        "matched".to_string()
                    } else {
                        format!(
                            "UNMATCHED — EXCLUDED: {}",
                            mismatch.as_deref().unwrap_or("shape disagreement")
                        )
                    };
                    let tcf_score = tcf.score(*objective);
                    let gguf_score = gguf.score(*objective);
                    println!(
                        "{:<48} {:>18} {} {} {}  {}",
                        row.name,
                        shape,
                        objective.format(tcf_score),
                        objective.format(gguf_score),
                        show_ratio(ratio(tcf_score, gguf_score)),
                        status
                    );
                }
                Outcome::NoGgufCounterpart { tcf, brackets } => {
                    // TCF's figure, and nothing in the GGUF column, because
                    // GGUF has no format at this budget to put there.
                    println!(
                        "{:<48} {:>18} {} {:>13} {:>8}  TCF ONLY: no GGUF format at this budget",
                        row.name,
                        shape,
                        objective.format(tcf.score(*objective)),
                        "n/a",
                        "n/a"
                    );
                    // The bracketing budgets DO have a figure. Their ratio is
                    // real and is printed, with both budgets on the same line
                    // so it is never read as an equal-bpw ratio.
                    let tcf_score = tcf.score(*objective);
                    for bracket in brackets {
                        match &bracket.state {
                            BracketState::Ran(result) => {
                                let bracket_score = result.score(*objective);
                                println!(
                                    "  {:<46} {:>18} {} {} {}  BRACKETED {}: TCF {:.4} bpw vs {} \
                                     {:.4} bpw — DIFFERENT BUDGETS, not an equal-bpw ratio",
                                    format!("^ bracket {}", bracket.side.label()),
                                    "",
                                    objective.format(tcf_score),
                                    objective.format(bracket_score),
                                    show_ratio(ratio(tcf_score, bracket_score)),
                                    bracket.side.label(),
                                    tcf_bpw,
                                    bracket.format.name(),
                                    bracket.bpw
                                );
                            }
                            BracketState::Unsupported(reason) => {
                                println!(
                                    "  {:<46} {:>18} {:>13} {:>13} {:>8}  BRACKETED {} \
                                     UNSUPPORTED: {}",
                                    format!("^ bracket {}", bracket.side.label()),
                                    "",
                                    "",
                                    "n/a",
                                    "n/a",
                                    bracket.side.label(),
                                    reason
                                );
                            }
                            BracketState::Failed(reason) => {
                                println!(
                                    "  {:<46} {:>18} {:>13} {:>13} {:>8}  BRACKETED {} ERROR: {}",
                                    format!("^ bracket {}", bracket.side.label()),
                                    "",
                                    "",
                                    "n/a",
                                    "n/a",
                                    bracket.side.label(),
                                    reason
                                );
                            }
                        }
                    }
                }
                Outcome::Unsupported { .. } | Outcome::Failed(_) => continue,
            }
        }
    }

    println!();
    println!("  ratio is TCF / GGUF. Below 1.0 means TCF reconstructs better.");
    println!(
        "  A codec winning the table that names its own objective is expected, not a finding."
    );
    if !tier.brackets.is_empty() {
        println!(
            "  A ^ bracket line is TCF at {tcf_bpw:.4} bpw against a DIFFERENT budget, so its \
             ratio is"
        );
        println!(
            "  quality per byte, never quality at equal bytes. Read it with both budgets on the \
             line."
        );
    }
}

/// One objective's win/loss count and ratio sample for one tier.
struct Tally {
    tcf: usize,
    gguf: usize,
    tied: usize,
    ratios: Vec<f64>,
}

impl Tally {
    fn new() -> Self {
        Tally {
            tcf: 0,
            gguf: 0,
            tied: 0,
            ratios: Vec::new(),
        }
    }

    /// Which codec won more tensors under this objective, or `None` when the
    /// count is even and there is nothing to call.
    fn winner(&self) -> Option<&'static str> {
        match self.tcf.cmp(&self.gguf) {
            std::cmp::Ordering::Greater => Some("TCF"),
            std::cmp::Ordering::Less => Some("GGUF"),
            std::cmp::Ordering::Equal => None,
        }
    }
}

/// The verdict: per tier, who wins how many tensors and by how much, reported
/// SEPARATELY under every objective.
///
/// There is no single headline number here on purpose. One codec winning
/// under its own objective and losing under the other's is the finding, and a
/// pooled score would bury it.
fn print_verdict(tiers: &[Tier], tier_rows: &[Vec<Row>]) {
    println!();
    println!("================================ VERDICT ================================");
    println!(
        "ratio is TCF / GGUF. Below 1.0 means TCF is more accurate at identical payload bytes."
    );
    println!(
        "Each objective is reported on its own. TCF's quantizer minimizes the unweighted one;"
    );
    println!("GGUF's minimize the two weighted ones. Read every row against who owns it.");
    println!(
        "The comparison is also search-stage-asymmetric: TCF refines its ROUNDED sub-scale against"
    );
    println!(
        "the decoder's reconstruction grid, a stage GGUF has no counterpart for — GGUF's second"
    );
    println!("pass only re-rounds codes against the scale it already chose.");
    println!("Payload bytes are matched; container overhead is not counted and is not symmetric —");
    println!("a quantized TCF tensor carries roughly 160-190 further bytes this tool ignores.");
    println!("Shape exclusions are reported per side, never pooled: a tensor TCF encodes and GGUF");
    println!("cannot is the capability claim under test, so it has its own line in every tier.");
    println!("A tier with no equal-budget GGUF format gets a BRACKETED COMPARISON instead: TCF");
    println!("against the nearest encodable GGUF budget below and above. Those bytes DIFFER from");
    println!("TCF's by design, so a bracketed ratio is quality per byte, never quality at equal");
    println!("bytes, and every bracketed line names both budgets.");

    for (tier, rows) in tiers.iter().zip(tier_rows) {
        let mut tallies: Vec<Tally> = OBJECTIVES.iter().map(|_| Tally::new()).collect();
        let mut unmatched = 0usize;
        // Three separate exclusion counts, never pooled. The middle one is the
        // capability claim under test: tensors TCF encodes and GGUF cannot.
        let mut unsupported_tcf_only = 0usize;
        let mut unsupported_gguf_only = 0usize;
        let mut unsupported_neither = 0usize;
        let mut failed = 0usize;
        let mut compared = 0usize;
        let mut tcf_only = 0usize;

        for row in rows {
            match &row.outcome {
                Outcome::Ran {
                    tcf,
                    gguf,
                    matched: true,
                    ..
                } => {
                    compared += 1;
                    for (objective, tally) in OBJECTIVES.iter().zip(&mut tallies) {
                        let tcf_score = tcf.score(*objective);
                        let gguf_score = gguf.score(*objective);
                        match tcf_score.partial_cmp(&gguf_score) {
                            Some(std::cmp::Ordering::Less) => tally.tcf += 1,
                            Some(std::cmp::Ordering::Greater) => tally.gguf += 1,
                            _ => tally.tied += 1,
                        }
                        if let Some(r) = ratio(tcf_score, gguf_score) {
                            tally.ratios.push(r);
                        }
                    }
                }
                Outcome::Ran { matched: false, .. } => unmatched += 1,
                Outcome::NoGgufCounterpart { .. } => tcf_only += 1,
                Outcome::Unsupported { tcf, gguf } => match (tcf.is_some(), gguf.is_some()) {
                    (true, true) => unsupported_neither += 1,
                    (true, false) => unsupported_tcf_only += 1,
                    (false, true) => unsupported_gguf_only += 1,
                    // `compare_tensor` only builds this variant when at least
                    // one side refused, so this arm is unreachable. Counting
                    // it as an exclusion of neither side keeps the three
                    // numbers summing to the rows they came from.
                    (false, false) => unsupported_neither += 1,
                },
                Outcome::Failed(_) => failed += 1,
            }
        }

        println!();
        println!(
            "{:.4} bpw  TCF {:?} vs GGUF {}  —  {} matched tensors compared",
            tier.bpw,
            tier.tcf,
            tier.gguf_name(),
            compared
        );
        if let Some(note) = tier.same_format {
            println!("  {note}");
        }
        if let Some(note) = tier.granularity_note {
            println!("  {note}");
        }
        if let Some(note) = tier.gguf_gap {
            println!("  {note}");
        }
        if tier.gguf.is_none() {
            println!("  TCF-only tensors             : {tcf_only} encoded by TCF, no GGUF side");
        }
        for (objective, tally) in OBJECTIVES.iter().zip(&mut tallies) {
            println!(
                "  {:<22} : TCF wins {:>4}, GGUF wins {:>4}, tied {:>4}  |  median {:>8}  ({})",
                objective.label(),
                tally.tcf,
                tally.gguf,
                tally.tied,
                show_median(&mut tally.ratios),
                objective.owner()
            );
        }
        // Three separate shape-exclusion counts, never pooled into one. The
        // GGUF-only line is the capability claim under test — tensors TCF
        // encodes at this budget and GGUF cannot — so it is on its own line
        // rather than summed with the others.
        println!("  excluded, TCF cannot encode  : {unsupported_tcf_only}");
        println!("  excluded, GGUF cannot encode : {unsupported_gguf_only}");
        println!("  excluded, NEITHER can encode : {unsupported_neither}");
        println!(
            "  excluded, other              : {unmatched} byte/element mismatches, \
             {failed} codec errors"
        );

        if compared > 0 {
            print_objective_split(&tallies);
        }
        print_bracket_verdict(tier, rows);
    }
}

/// One bracketing budget's outcome under one objective, over the tensors
/// where both it and TCF ran.
struct BracketObjectiveStats {
    /// Tensors where TCF's error is at or below this bracket's.
    tcf_at_or_below: usize,
    /// Tensors where TCF's error is above this bracket's.
    tcf_worse: usize,
    /// TCF / bracket, one per tensor. A DIFFERENT-budget ratio.
    ratios: Vec<f64>,
}

impl BracketObjectiveStats {
    fn new() -> Self {
        BracketObjectiveStats {
            tcf_at_or_below: 0,
            tcf_worse: 0,
            ratios: Vec::new(),
        }
    }

    fn compared(&self) -> usize {
        self.tcf_at_or_below + self.tcf_worse
    }
}

/// The bracketed finding for one tier: TCF's error at its own budget against
/// the nearest encodable GGUF budget below and the nearest above, plus the
/// bits per weight each of them actually spends.
///
/// This is the answer the tier used to refuse to give. The budgets differ, so
/// no ratio here is an equal-bpw ratio and every line says so. The conclusion
/// is derived from the medians printed beside it, never asserted: TCF at or
/// below the upper bracket, between the two, or above the lower one are three
/// different findings and each is stated as what it is.
fn print_bracket_verdict(tier: &Tier, rows: &[Row]) {
    if tier.brackets.is_empty() {
        return;
    }
    let tcf_bpw = tcf_bits_per_weight(tier.tcf);

    // stats[bracket index][objective index], index-aligned with
    // `tier.brackets` and `OBJECTIVES`, because `compare_tensor` builds each
    // row's bracket list in the tier's declaration order.
    let mut stats: Vec<Vec<BracketObjectiveStats>> = tier
        .brackets
        .iter()
        .map(|_| {
            OBJECTIVES
                .iter()
                .map(|_| BracketObjectiveStats::new())
                .collect()
        })
        .collect();
    let mut excluded: Vec<usize> = tier.brackets.iter().map(|_| 0).collect();
    // Where TCF's error falls between the two brackets, per objective: 0.0 is
    // the lower bracket's error, 1.0 is the upper bracket's.
    let mut positions: Vec<Vec<f64>> = OBJECTIVES.iter().map(|_| Vec::new()).collect();
    // Tensors where the DEARER bracket did not reconstruct better than the
    // cheaper one, per objective. The position figure is meaningless there,
    // so those tensors are counted out rather than folded in.
    let mut inverted: Vec<usize> = OBJECTIVES.iter().map(|_| 0).collect();

    let below_index = tier
        .brackets
        .iter()
        .position(|bracket| bracket.side == BracketSide::Below);
    let above_index = tier
        .brackets
        .iter()
        .position(|bracket| bracket.side == BracketSide::Above);

    for row in rows {
        let Outcome::NoGgufCounterpart { tcf, brackets } = &row.outcome else {
            continue;
        };
        for (bracket_index, bracket) in brackets.iter().enumerate() {
            let Some(entry) = stats.get_mut(bracket_index) else {
                continue;
            };
            let Some(result) = bracket.state.result() else {
                if let Some(count) = excluded.get_mut(bracket_index) {
                    *count += 1;
                }
                continue;
            };
            for (objective_index, objective) in OBJECTIVES.iter().enumerate() {
                let Some(objective_stats) = entry.get_mut(objective_index) else {
                    continue;
                };
                let tcf_score = tcf.score(*objective);
                let bracket_score = result.score(*objective);
                if tcf_score <= bracket_score {
                    objective_stats.tcf_at_or_below += 1;
                } else {
                    objective_stats.tcf_worse += 1;
                }
                if let Some(r) = ratio(tcf_score, bracket_score) {
                    objective_stats.ratios.push(r);
                }
            }
        }

        // The proportional position needs both brackets on the same tensor.
        if let (Some(below), Some(above)) = (below_index, above_index) {
            let below_result = brackets.get(below).and_then(|b| b.state.result());
            let above_result = brackets.get(above).and_then(|b| b.state.result());
            if let (Some(below_result), Some(above_result)) = (below_result, above_result) {
                for (objective_index, objective) in OBJECTIVES.iter().enumerate() {
                    let low_budget_error = below_result.score(*objective);
                    let high_budget_error = above_result.score(*objective);
                    let span = low_budget_error - high_budget_error;
                    if span <= 0.0 {
                        if let Some(count) = inverted.get_mut(objective_index) {
                            *count += 1;
                        }
                        continue;
                    }
                    if let Some(sample) = positions.get_mut(objective_index) {
                        sample.push((low_budget_error - tcf.score(*objective)) / span);
                    }
                }
            }
        }
    }

    println!(
        "  BRACKETED COMPARISON — TCF {:?} at {:.4} bpw has NO equal-budget GGUF counterpart, so",
        tier.tcf, tcf_bpw
    );
    println!(
        "  it is scored against the nearest encodable GGUF budget on each side. Those budgets"
    );
    println!(
        "  DIFFER from TCF's on purpose: every ratio below is quality per byte, NOT quality at"
    );
    println!("  equal bytes, and both budgets are named on every line that carries one.");
    for (bracket, skipped) in tier.brackets.iter().zip(&excluded) {
        let bracket_bpw = gguf_bits_per_weight(bracket.format);
        let size = ratio(tcf_bpw, bracket_bpw)
            .map(|r| format!("TCF spends {:.1}% of its bytes", r * 100.0))
            .unwrap_or_else(|| "size ratio n/a".to_string());
        println!(
            "    bracket {:<5} : {} at {:.4} bpw ({:+.4} bpw against TCF) — {} | {} tensors it \
             could not encode",
            bracket.side.label(),
            bracket.format.name(),
            bracket_bpw,
            bracket_bpw - tcf_bpw,
            size,
            skipped
        );
    }

    for (objective_index, objective) in OBJECTIVES.iter().enumerate() {
        println!("    {} ({})", objective.label(), objective.owner());
        for (bracket_index, bracket) in tier.brackets.iter().enumerate() {
            let Some(objective_stats) = stats
                .get_mut(bracket_index)
                .and_then(|entry| entry.get_mut(objective_index))
            else {
                continue;
            };
            println!(
                "      vs bracket {:<5} {} {:.4} bpw : TCF at or below on {:>4} of {:>4}, median \
                 ratio {:>8}",
                bracket.side.label(),
                bracket.format.name(),
                gguf_bits_per_weight(bracket.format),
                objective_stats.tcf_at_or_below,
                objective_stats.compared(),
                show_median(&mut objective_stats.ratios)
            );
        }
        print_bracket_finding(
            tier,
            tcf_bpw,
            &mut stats,
            objective_index,
            below_index,
            above_index,
            positions.get_mut(objective_index),
            inverted.get(objective_index).copied().unwrap_or(0),
        );
    }
}

/// State the bracketed finding for one objective in plain terms, computed from
/// the medians printed above it.
///
/// Three outcomes, and each is printed as itself. "Better than the cheap
/// bracket, worse than the dear one" is a real answer, so it is stated with
/// the proportional position rather than rounded up into a win.
#[allow(clippy::too_many_arguments)]
fn print_bracket_finding(
    tier: &Tier,
    tcf_bpw: f64,
    stats: &mut [Vec<BracketObjectiveStats>],
    objective_index: usize,
    below_index: Option<usize>,
    above_index: Option<usize>,
    positions: Option<&mut Vec<f64>>,
    inverted: usize,
) {
    let (Some(below_index), Some(above_index)) = (below_index, above_index) else {
        println!(
            "      FINDING          : only one bracket declared, so there is no span to place TCF \
             in."
        );
        return;
    };
    let below_format = tier.brackets[below_index].format;
    let above_format = tier.brackets[above_index].format;
    let below_bpw = gguf_bits_per_weight(below_format);
    let above_bpw = gguf_bits_per_weight(above_format);

    let below_median = stats
        .get_mut(below_index)
        .and_then(|entry| entry.get_mut(objective_index))
        .and_then(|entry| median(&mut entry.ratios));
    let above_median = stats
        .get_mut(above_index)
        .and_then(|entry| entry.get_mut(objective_index))
        .and_then(|entry| median(&mut entry.ratios));
    let (Some(below_median), Some(above_median)) = (below_median, above_median) else {
        println!(
            "      FINDING          : not enough tensors ran on both brackets to place TCF \
             between them."
        );
        return;
    };

    let below_size = ratio(tcf_bpw, below_bpw).map(|r| r * 100.0);
    let above_size = ratio(tcf_bpw, above_bpw).map(|r| r * 100.0);
    let show_size = |value: Option<f64>| match value {
        Some(percent) => format!("{percent:.1}%"),
        None => "n/a".to_string(),
    };

    if above_median <= 1.0 {
        println!(
            "      FINDING          : TCF at {:.4} bpw MATCHES OR BEATS the upper bracket {} at \
             {:.4} bpw",
            tcf_bpw,
            above_format.name(),
            above_bpw
        );
        println!(
            "                         (median ratio {:.4}) while spending {} of its bytes — the \
             upper bracket's",
            above_median,
            show_size(above_size)
        );
        println!("                         quality at the smaller size.");
    } else if below_median < 1.0 {
        let position = positions.and_then(|sample| median(sample));
        println!(
            "      FINDING          : TCF at {:.4} bpw is BETTER than the lower bracket {} at \
             {:.4} bpw",
            tcf_bpw,
            below_format.name(),
            below_bpw
        );
        println!(
            "                         (median ratio {:.4}) and WORSE than the upper bracket {} \
             at {:.4} bpw",
            below_median,
            above_format.name(),
            above_bpw
        );
        println!(
            "                         (median ratio {above_median:.4}). It does NOT buy the upper \
             bracket's quality."
        );
        match position {
            Some(position) => println!(
                "                         Its error sits {:.1}% of the way from the {:.4} bpw \
                 bracket to the {:.4} bpw one,",
                position * 100.0,
                below_bpw,
                above_bpw
            ),
            None => println!(
                "                         The two brackets never separated on error, so there is \
                 no position to report,"
            ),
        }
        println!(
            "                         for {} of the lower bracket's bytes and {} of the upper's.",
            show_size(below_size),
            show_size(above_size)
        );
    } else {
        println!(
            "      FINDING          : TCF at {:.4} bpw is WORSE than BOTH brackets — median ratio \
             {:.4} against",
            tcf_bpw, below_median
        );
        println!(
            "                         {} at {:.4} bpw and {:.4} against {} at {:.4} bpw. The \
             {:.4} bpw it spends",
            below_format.name(),
            below_bpw,
            above_median,
            above_format.name(),
            above_bpw,
            tcf_bpw - below_bpw
        );
        println!(
            "                         over the lower bracket buys nothing under this objective."
        );
    }
    if inverted > 0 {
        println!(
            "                         {inverted} tensors excluded from the position figure: the \
             dearer bracket did not"
        );
        println!("                         reconstruct better than the cheaper one there.");
    }
}

/// Say plainly where the objectives disagree.
///
/// A codec that wins the objective its own encoder minimizes and loses the
/// other side's has not been shown to be better; it has been shown to be
/// better at its own target. That sentence is the point of the whole report,
/// so it is printed rather than left for the reader to derive.
fn print_objective_split(tallies: &[Tally]) {
    let Some(unweighted) = tally_for(tallies, Objective::Unweighted) else {
        return;
    };

    for weighted_objective in [Objective::SquaredMagnitude, Objective::RmsPlusMagnitude] {
        let Some(weighted) = tally_for(tallies, weighted_objective) else {
            continue;
        };
        match (unweighted.winner(), weighted.winner()) {
            (Some(a), Some(b)) if a != b => {
                println!(
                    "  SPLIT                  : {a} wins {}, {b} wins {} — the ranking flips with \
                     the objective, so neither codec is more accurate outright.",
                    Objective::Unweighted.label(),
                    weighted_objective.label()
                );
            }
            (Some(a), Some(_)) => {
                println!(
                    "  AGREES                 : {a} wins both {} and {}.",
                    Objective::Unweighted.label(),
                    weighted_objective.label()
                );
            }
            _ => {
                println!(
                    "  NO CALL                : {} and {} did not both produce a winner.",
                    Objective::Unweighted.label(),
                    weighted_objective.label()
                );
            }
        }
    }
}

/// One objective's tally, found by its position in `OBJECTIVES`.
///
/// `tallies` is built by mapping over `OBJECTIVES`, so the two are index for
/// index. A `None` here means the caller built a shorter list, which no caller
/// does.
fn tally_for(tallies: &[Tally], wanted: Objective) -> Option<&Tally> {
    OBJECTIVES
        .iter()
        .position(|objective| *objective == wanted)
        .and_then(|index| tallies.get(index))
}

/// Median of a ratio sample, formatted. The median, not the mean: one tensor
/// with a near-zero denominator would otherwise carry the whole tier.
fn show_median(values: &mut [f64]) -> String {
    match median(values) {
        Some(median) => format!("{median:.4}"),
        None => "n/a".to_string(),
    }
}

/// The median itself, for the callers that compare against it rather than
/// print it. `show_median` formats what this returns, so the two can never
/// disagree about what the middle value is.
fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}
