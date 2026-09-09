//! Head-to-head quantization ACCURACY: TCF against GGUF at matched bits per weight.
//!
//! Quantizes the SAME source tensor with both codecs, dequantizes both, and
//! diffs each reconstruction against the f32 source. Nothing is rescaled and no
//! "closest bpw" fudge is applied: the three pairs below are bit-for-bit bpw
//! matches, so the only variable left is which codec reconstructs better.
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
use clap::Parser;
use rayon::prelude::*;

use boostr::quant::{DequantOps, QuantFormat, QuantizeOps};
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor};
use compressr::loader::{ModelLoader, UnifiedDType};
use compressr::{cast_bytes, relative_rms_error};
use tcf_core::encoding::{dequantize, pack, quantize, unpack};
use tcf_core::NativeEncoding;

/// One matched bits-per-weight tier: a TCF encoding and the GGUF format that
/// occupies exactly the same bytes for the same element count.
struct Tier {
    /// Nominal bits per weight, for the report header only. The per-tensor
    /// byte counts below are what actually proves the match.
    bpw: f64,
    tcf: NativeEncoding,
    gguf: QuantFormat,
    /// Set when the two sides are not two formats but ONE format written by
    /// two encoders. The text is printed with the tier so no reader takes
    /// that tier's numbers for a format result.
    same_format: Option<&'static str>,
}

// `static`, not `const`: `process_batch` below takes `&'static Tier`
// references into this array from inside a rayon closure, which a `const`
// cannot provide — each use of a `const` re-materializes a fresh temporary.
static TIERS: [Tier; 3] = [
    Tier {
        bpw: 4.50,
        tcf: NativeEncoding::Q4AS32DT64,
        gguf: QuantFormat::Q4K,
        same_format: None,
    },
    Tier {
        bpw: 6.5625,
        tcf: NativeEncoding::Q6S16DT64,
        gguf: QuantFormat::Q6K,
        same_format: None,
    },
    Tier {
        bpw: 8.50,
        tcf: NativeEncoding::Q8S32T64,
        gguf: QuantFormat::Q8_0,
        same_format: Some(
            "SAME FORMAT, TWO ENCODERS: Q8S32T64 and Q8_0 are bit-for-bit the \
             same layout — 32-element blocks, one binary16 scale, 8-bit codes \
             clamped to [-127, 127], 34 bytes per 32 elements. Any gap here is \
             encoder work, NOT a format result.",
        ),
    },
];

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
    /// This tier cannot encode this tensor's shape at all, on one side or
    /// both. Not an error, and not a comparison either.
    Unsupported(String),
    /// A codec refused the tensor's values, or a call failed.
    Failed(String),
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

    println!("model    : {}", args.model);
    println!("format   : {:?}", model.format());
    println!("tensors  : {}", model.tensor_count());
    println!("workers  : {} rayon threads", rayon::current_num_threads());

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
    for tier in &TIERS {
        println!(
            "  {:.4} bpw: TCF {:?} vs GGUF {}",
            tier.bpw,
            tier.tcf,
            tier.gguf.name()
        );
        if let Some(note) = tier.same_format {
            println!("    {note}");
        }
    }

    // One row list per tier, in the same tensor order, so a reader can scan
    // one tier top to bottom without a per-tensor block of six numbers.
    let mut tier_rows: Vec<Vec<Row>> = TIERS.iter().map(|_| Vec::new()).collect();

    // `ModelLoader::read_tensor_bytes` seeks a shared file handle and takes
    // `&mut self`, so it is not `Sync` and cannot be called from multiple
    // rayon workers at once. Reading and converting to f32 therefore stays
    // serial, one tensor at a time, exactly as before. The expensive part —
    // quantizing, dequantizing and scoring both codecs at all three tiers —
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
            process_batch(&batch, &mut tier_rows);
            batch.clear();
            batch_elements = 0;
        }
    }
    if !batch.is_empty() {
        process_batch(&batch, &mut tier_rows);
    }

    for (tier, rows) in TIERS.iter().zip(&tier_rows) {
        print_tier(tier, rows);
    }

    print_verdict(&tier_rows);
    Ok(())
}

/// Run every (tensor, tier) pair in one batch across rayon's pool, then append
/// each tensor's per-tier outcome to `tier_rows` in the batch's original
/// order — the axis is (tensor, tier), not tensor alone, so a checkpoint
/// dominated by one huge tensor still spreads its three tiers across workers
/// instead of leaving them idle behind it.
///
/// `Vec::into_par_iter` is an indexed parallel iterator: `.collect::<Vec<_>>()`
/// re-assembles results at the same positions they were submitted at,
/// regardless of which worker finished first or in what order. That is what
/// keeps this report byte-identical in content and order to the serial
/// version — no explicit re-sort is needed after the collect.
fn process_batch(batch: &[(String, Vec<usize>, Vec<f32>)], tier_rows: &mut [Vec<Row>]) {
    let tier_count = TIERS.len();
    let jobs: Vec<(usize, usize)> = (0..batch.len())
        .flat_map(|tensor_index| (0..tier_count).map(move |tier_index| (tensor_index, tier_index)))
        .collect();

    let outcomes: Vec<Outcome> = jobs
        .into_par_iter()
        .map(|(tensor_index, tier_index)| {
            let (_, shape, values) = &batch[tensor_index];
            compare_tensor(values, shape, &TIERS[tier_index])
        })
        .collect();

    let mut outcomes = outcomes.into_iter();
    for (name, shape, _values) in batch {
        for tier_index in 0..tier_count {
            let outcome = outcomes
                .next()
                .expect("exactly one outcome per (tensor, tier) pair in this batch");
            tier_rows[tier_index].push(Row {
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
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Run both codecs over one tensor at one tier.
fn compare_tensor(values: &[f32], shape: &[usize], tier: &Tier) -> Outcome {
    let row_len = match shape.last() {
        Some(len) => *len,
        None => return Outcome::Unsupported("rank 0 has no row".to_string()),
    };

    // Both codecs pack whole blocks along the LAST axis. A row that is not a
    // whole number of blocks cannot be encoded at all, and the two sides have
    // different block widths, so the check is per side.
    let tile = usize::from(tier.tcf.geometry().tile);
    let block = tier.gguf.block_size();
    let mut refusals = Vec::new();
    if tile == 0 || !row_len.is_multiple_of(tile) {
        refusals.push(format!("row {row_len} is not a whole number of {tile}-element TCF tiles"));
    }
    if block == 0 || !row_len.is_multiple_of(block) {
        refusals.push(format!(
            "row {row_len} is not a whole number of {block}-element {} blocks",
            tier.gguf.name()
        ));
    }
    if !refusals.is_empty() {
        return Outcome::Unsupported(refusals.join("; "));
    }

    let tcf = match tcf_result(values, shape, tier.tcf) {
        Ok(result) => result,
        Err(e) => return Outcome::Failed(format!("TCF {:?}: {e:#}", tier.tcf)),
    };
    let gguf = match gguf_result(values, shape, tier.gguf) {
        Ok(result) => result,
        Err(e) => return Outcome::Failed(format!("GGUF {}: {e:#}", tier.gguf.name())),
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
        mismatch.push(format!("byte count TCF {} vs GGUF {}", tcf.bytes, gguf.bytes));
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
    println!();
    println!(
        "================ {:.4} bpw : TCF {:?} vs GGUF {} ================",
        tier.bpw,
        tier.tcf,
        tier.gguf.name()
    );
    if let Some(note) = tier.same_format {
        println!("  {note}");
    }

    // Bytes and status first, once per tensor. The metric tables below repeat
    // neither, so each of them stays narrow enough to read.
    println!();
    println!("-- encoded payload bytes and status --");
    println!(
        "{:<48} {:>18} {:>12} {:>12}  {}",
        "tensor", "shape", "bytes TCF", "bytes GGUF", "status"
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
            Outcome::Unsupported(reason) => {
                println!(
                    "{:<48} {:>18}  UNSUPPORTED at this tier: {}",
                    row.name, shape, reason
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
            "{:<48} {:>18} {:>13} {:>13} {:>8}  {}",
            "tensor", "shape", "TCF", "GGUF", "ratio", "status"
        );
        for row in rows {
            let Outcome::Ran {
                tcf,
                gguf,
                matched,
                mismatch,
            } = &row.outcome
            else {
                continue;
            };
            let status = if *matched {
                "matched".to_string()
            } else {
                format!(
                    "UNMATCHED — EXCLUDED: {}",
                    mismatch.as_deref().unwrap_or("shape disagreement")
                )
            };
            let shape = format!("{:?}", row.shape);
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
    }

    println!();
    println!("  ratio is TCF / GGUF. Below 1.0 means TCF reconstructs better.");
    println!(
        "  A codec winning the table that names its own objective is expected, not a finding."
    );
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
fn print_verdict(tier_rows: &[Vec<Row>]) {
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
    println!(
        "Payload bytes are matched; container overhead is not counted and is not symmetric —"
    );
    println!("a quantized TCF tensor carries roughly 160-190 further bytes this tool ignores.");

    for (tier, rows) in TIERS.iter().zip(tier_rows) {
        let mut tallies: Vec<Tally> = OBJECTIVES.iter().map(|_| Tally::new()).collect();
        let mut unmatched = 0usize;
        let mut unsupported = 0usize;
        let mut failed = 0usize;
        let mut compared = 0usize;

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
                Outcome::Unsupported(_) => unsupported += 1,
                Outcome::Failed(_) => failed += 1,
            }
        }

        println!();
        println!(
            "{:.4} bpw  TCF {:?} vs GGUF {}  —  {} matched tensors compared",
            tier.bpw,
            tier.tcf,
            tier.gguf.name(),
            compared
        );
        if let Some(note) = tier.same_format {
            println!("  {note}");
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
        println!(
            "  excluded               : {unmatched} byte/element mismatches, {unsupported} shapes \
             this tier cannot encode, {failed} codec errors"
        );

        if compared > 0 {
            print_objective_split(&tallies);
        }
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
    if values.is_empty() {
        return "n/a".to_string();
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    };
    format!("{median:.4}")
}
