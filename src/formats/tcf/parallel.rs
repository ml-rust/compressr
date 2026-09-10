//! Quantize ONE tensor's tile chunks across rayon's thread pool.
//!
//! # Why the threading is here
//!
//! `tcf-core` permits exactly one dependency, blake3, and CONFORMANCE.md makes
//! its scalar implementation the definition of codec semantics. So it gains a
//! chunk-aware entry point — [`quantize_tile_chunk_weighted`] — and nothing
//! else; the scheduling that calls it lives in the consumer, which is this
//! module.
//!
//! # Why WITHIN a tensor, never across tensors
//!
//! compressr keeps one tensor in flight, so peak RAM tracks the largest
//! tensor. Quantizing several tensors at once would multiply that by the
//! thread count. Splitting one tensor's tiles adds only the tiles themselves,
//! which for a 64-element tile are far smaller than the 256 source bytes they
//! came from.
//!
//! # Why the bytes cannot move
//!
//! Determinism is a TCF format guarantee. Every scale fit, sub-scale refit and
//! code is computed inside one group or one super-block, and a chunk boundary
//! never splits either: [`quantize_tile_chunk_weighted`] refuses a chunk that
//! would. Nothing is accumulated across tiles, so no float is reassociated.
//! Chunks are concatenated in tile order, and the first error IN TILE ORDER is
//! the one returned, so even a rejected tensor takes the serial path's
//! fallback reason.
//!
//! # Why the objective does not move them either
//!
//! A weighted objective weights each element by a rule read from the GROUP's
//! own values — `w = x^2`, or `w = sqrt(mean(x^2)) + |x|` over that group.
//! A group never spans two tiles and a chunk never splits a tile, so a group's
//! weights are the same whichever chunk its tile lands in, and the chunked
//! path scores exactly what the whole-tensor path scores. The one source that
//! would need care is an importance source, whose slice is CHUNK-LOCAL: it
//! has to be narrowed to the chunk before the call, which is what
//! `WeightSource::slice` and `tcf_core::column_weights`'s `first_element` are
//! for. An `--imatrix` run does pass one —
//! [`super::objective_weights::ResolvedWeights`] expands the per-column
//! importance over the WHOLE tensor once, and the narrowing below cuts it
//! exactly where the values are cut. Every element keeps the weight of the
//! column it sits in, so the chunked path scores what a whole-tensor call
//! scores under `WeightSource::Explicit` too.
//!
//! `WeightSource::Combined` adds a `sigma2` taken over the super-block, which
//! is the one term that reads values outside the group. [`super::chunk`]
//! rounds every chunk to a whole number of super-blocks, so a super-block
//! never spans two chunks and its `sigma2` is the same value either path
//! computes.

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;

use tcf_core::encoding::{
    accepts, accepts_tile_chunk, quantize_tile_chunk_with, quantize_with, tile_count,
};
use tcf_core::{LogicalTile, QuantLayout, QuantizeParams, TcfError};

use super::chunk::{chunk_bounds, chunk_slice, chunk_start, chunk_tiles};

/// Tiles below which a tensor is quantized serially, with no rayon call at
/// all.
///
/// 512 tiles is 32768 weights. At roughly 10 microseconds of scale search per
/// tile that is about 5 ms of work — small enough that a serial run finishes
/// before a split would pay for itself, and large enough that every tensor
/// with real cost is still split. Norm and bias vectors sit far below it and
/// stay on the serial path entirely.
const MIN_PARALLEL_TILES: u64 = 512;

/// Quantize a whole tensor, splitting its tiles across rayon's pool.
///
/// Byte-for-byte what `tcf_core::encoding::quantize_with` returns for the
/// same arguments — this is a scheduling change and nothing else. `values`
/// holds the tensor row-major, `dims[0..rank]` is its shape, and `params`
/// names the error objective every group is scored by and how wide its
/// candidate-scale search runs.
///
/// `QuantizeParams::default()` is what every earlier run used, and
/// `tcf-core` reproduces those bytes exactly under it.
///
/// # Errors
/// Whatever `tcf_core::encoding::quantize_with` raises, and for the same
/// values. When several chunks fail, the failure of the LOWEST tile index is
/// returned, which is the one a serial run would have hit first.
pub(super) fn quantize_tensor(
    values: &[f32],
    dims: &[u64],
    rank: u32,
    layout: QuantLayout,
    params: QuantizeParams<'_>,
) -> Result<Vec<LogicalTile>, TcfError> {
    let tile = usize::from(layout.geometry.tile);
    let total = tile_count(dims, rank, layout.geometry.tile)?;
    if total < MIN_PARALLEL_TILES {
        // The serial path is `quantize_with` itself, not a one-chunk call: a
        // small tensor then touches no scheduling code at all, and the two
        // paths cannot drift.
        return quantize_with(values, dims, rank, layout, params);
    }

    let chunk = chunk_tiles(total, rayon::current_num_threads());
    let bounds = chunk_bounds(total, chunk);

    // Collected as a Vec of Results rather than folded into one Result: an
    // indexed rayon collect preserves order, so taking the first Err by index
    // below names the same failure a serial run names. Letting rayon pick the
    // surviving error would let two chunks that fail for different reasons
    // decide the tensor's fallback reason by thread timing.
    let parts: Vec<Result<Vec<LogicalTile>, TcfError>> = bounds
        .into_par_iter()
        .map(|(first, count)| {
            let slice = chunk_slice(values, tile, first, count)?;
            let final_chunk = first.saturating_add(count) == total;
            // The weights are narrowed the same way the values are, so a
            // source that carries data stays parallel to the chunk it scores.
            // Every value-derived source narrows to itself. The effort
            // travels unchanged: it names a candidate set per group, not a
            // span over the tensor, so no chunk boundary narrows it.
            let start = chunk_start(tile, first)?;
            let weights = params
                .weights
                .slice(start, start.saturating_add(slice.len()));
            let chunk_params = QuantizeParams {
                weights,
                effort: params.effort,
            };
            quantize_tile_chunk_with(slice, count, first, final_chunk, layout, chunk_params)
        })
        .collect();

    let capacity = usize::try_from(total).unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(capacity);
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

/// Whether `layout` can represent this tensor, splitting the scan across
/// rayon's pool.
///
/// Exactly what `tcf_core::encoding::accepts` answers for the same arguments,
/// which is exactly what a full quantize's success or `SourceNonfinite` /
/// `NumericRange` failure answers — without the scale search. The bump screen
/// asks this of every candidate and keeps a bool, so the search it used to run
/// was pure waste.
///
/// A chunk needs no super-block alignment: every verdict is one group's own.
/// The split is still `quantize_tensor`'s, so the two paths cut a tensor the
/// same way and one threshold governs both.
///
/// It takes no objective. The question is whether a group's values can be
/// represented at all — a NaN, an infinity, or a scale that underflows
/// binary16 — and a weight scores candidates rather than admitting values, so
/// every objective answers this identically.
///
/// # Errors
/// Whatever `tcf_core::encoding::accepts` raises: a geometry or tiling
/// rejection, never a value-dependent one.
pub(super) fn accepts_tensor(
    values: &[f32],
    dims: &[u64],
    rank: u32,
    layout: QuantLayout,
) -> Result<bool, TcfError> {
    let tile = usize::from(layout.geometry.tile);
    let total = tile_count(dims, rank, layout.geometry.tile)?;
    if total < MIN_PARALLEL_TILES {
        // The serial path is `accepts` itself, for the same reason the
        // quantize path calls `quantize_with`: the two cannot drift.
        return accepts(values, dims, rank, layout);
    }

    let chunk = chunk_tiles(total, rayon::current_num_threads());
    let bounds = chunk_bounds(total, chunk);

    let parts: Vec<Result<bool, TcfError>> = bounds
        .into_par_iter()
        .map(|(first, count)| {
            let slice = chunk_slice(values, tile, first, count)?;
            accepts_tile_chunk(slice, count, layout)
        })
        .collect();

    // Read in tile order and stop at the first refusal, so a serial run and a
    // split run report the same verdict on a tensor whose chunks disagree.
    for part in parts {
        if !part? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Quantize a whole tensor, reporting the tensor name on failure.
///
/// The `anyhow` wrapper the writer and the planner both want, kept in one
/// place so the two call sites cannot describe the same failure differently.
///
/// # Errors
/// Whatever [`quantize_tensor`] raises, tagged with `name`.
pub(super) fn quantize_named(
    values: &[f32],
    dims: &[u64],
    rank: u32,
    layout: QuantLayout,
    params: QuantizeParams<'_>,
    name: &str,
) -> Result<Vec<LogicalTile>> {
    quantize_tensor(values, dims, rank, layout, params)
        .map_err(|e| anyhow!(e))
        .with_context(|| format!("Failed to quantize tensor '{name}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::encoding::{pack, quantize, quantize_weighted};
    use tcf_core::{NativeEncoding, WeightSource};

    /// The objectives a `--tcf-objective` run can select.
    const OBJECTIVES: [WeightSource<'static>; 3] = [
        WeightSource::Uniform,
        WeightSource::Squared,
        WeightSource::RmsPlusAbs,
    ];

    const ENCODINGS: [NativeEncoding; 7] = [
        NativeEncoding::Q4S32T64,
        NativeEncoding::Q4AS32T64,
        NativeEncoding::Q4AS32DT64,
        NativeEncoding::Q4AS64T64,
        NativeEncoding::Q6S32T64,
        NativeEncoding::Q6S16DT64,
        NativeEncoding::Q8S32T64,
    ];

    fn values(count: usize) -> Vec<f32> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..count)
            .map(|i| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as f32) / 16_777_216.0 - 0.5;
                unit * (1.0 + (i % 7) as f32)
            })
            .collect()
    }

    /// The whole point: the parallel path is byte-identical to `quantize`,
    /// for every encoding, on a tensor whose tiles-per-row is not a multiple
    /// of four and whose tile count is not either.
    #[test]
    fn the_parallel_path_matches_the_serial_one() {
        // 65 rows of 640: 10 tiles per row, 650 tiles, neither a multiple of
        // four, so super-blocks cross rows and the run ends on a partial one.
        // Above MIN_PARALLEL_TILES, so this is the parallel path.
        let shape = [65u64, 640];
        let source = values(65 * 640);
        for encoding in ENCODINGS {
            let layout = encoding.layout();
            let serial = quantize(&source, &shape, 2, layout).expect("serial quantize");
            let parallel = quantize_tensor(&source, &shape, 2, layout, QuantizeParams::default())
                .expect("parallel quantize");
            assert_eq!(parallel.len(), serial.len(), "{encoding:?}");
            let left = pack(&parallel, layout).expect("parallel packs");
            let right = pack(&serial, layout).expect("serial packs");
            assert_eq!(left, right, "{encoding:?} packed bytes");
        }
    }

    /// A tensor under the threshold takes the serial path and still matches.
    #[test]
    fn a_small_tensor_skips_the_parallel_path() {
        let shape = [4u64, 640];
        let source = values(4 * 640);
        for encoding in ENCODINGS {
            let layout = encoding.layout();
            let serial = quantize(&source, &shape, 2, layout).expect("serial quantize");
            let parallel = quantize_tensor(&source, &shape, 2, layout, QuantizeParams::default())
                .expect("parallel quantize");
            let left = pack(&parallel, layout).expect("parallel packs");
            let right = pack(&serial, layout).expect("serial packs");
            assert_eq!(left, right, "{encoding:?}");
        }
    }

    /// The bump screen's question, answered without the scale search, must
    /// match what a full quantize answers — on both the split and the serial
    /// path, and for every encoding.
    #[test]
    fn accepts_matches_a_full_quantize() {
        // 650 tiles: the parallel path. 40 tiles: the serial one.
        for shape in [[65u64, 640], [4u64, 640]] {
            let count = usize::try_from(shape[0] * shape[1]).expect("a small fixture");
            let ordinary = values(count);
            let tiny: Vec<f32> = ordinary.iter().map(|v| v * 1e-9).collect();
            let mut nonfinite = values(count);
            nonfinite[count / 2] = f32::NAN;
            let mut one_tiny_group = values(count);
            for slot in one_tiny_group.iter_mut().skip(128).take(64) {
                *slot *= 1e-9;
            }
            let cases = [
                ("ordinary", ordinary),
                ("underflowing", tiny),
                ("one underflowing group", one_tiny_group),
                ("nonfinite", nonfinite),
            ];
            for encoding in ENCODINGS {
                let layout = encoding.layout();
                for (name, source) in &cases {
                    let quantized =
                        quantize_tensor(source, &shape, 2, layout, QuantizeParams::default())
                            .is_ok();
                    let accepted = accepts_tensor(source, &shape, 2, layout).expect("decides");
                    assert_eq!(accepted, quantized, "{encoding:?} {name} {shape:?}");
                }
            }
        }
    }

    /// A weighted objective is scored from each GROUP's own values, and a
    /// chunk never splits a group, so the split path must match the
    /// whole-tensor path under every objective — not only the uniform one.
    #[test]
    fn the_parallel_path_matches_the_serial_one_under_every_objective() {
        let shape = [65u64, 640];
        let source = values(65 * 640);
        for encoding in ENCODINGS {
            let layout = encoding.layout();
            for weights in OBJECTIVES {
                let serial = quantize_weighted(&source, &shape, 2, layout, weights)
                    .expect("serial quantize");
                let parallel = quantize_tensor(
                    &source,
                    &shape,
                    2,
                    layout,
                    QuantizeParams::with_weights(weights),
                )
                .expect("parallel quantize");
                let left = pack(&parallel, layout).expect("parallel packs");
                let right = pack(&serial, layout).expect("serial packs");
                assert_eq!(left, right, "{encoding:?} {weights:?}");
            }
        }
    }

    /// The uniform objective is the one every earlier run used, so it writes
    /// the bytes the unweighted entry point writes — exactly, not nearly.
    #[test]
    fn the_uniform_objective_is_the_unweighted_one() {
        let shape = [65u64, 640];
        let source = values(65 * 640);
        for encoding in ENCODINGS {
            let layout = encoding.layout();
            let unweighted = quantize(&source, &shape, 2, layout).expect("unweighted quantize");
            let uniform = quantize_tensor(&source, &shape, 2, layout, QuantizeParams::default())
                .expect("uniform quantize");
            let left = pack(&uniform, layout).expect("uniform packs");
            let right = pack(&unweighted, layout).expect("unweighted packs");
            assert_eq!(left, right, "{encoding:?}");
        }
    }

    /// The search effort travels to every chunk unchanged, so a wider search
    /// must match the whole-tensor path exactly, the same way a weighted
    /// objective does.
    #[test]
    fn the_parallel_path_matches_the_serial_one_under_every_effort() {
        let shape = [65u64, 640];
        let source = values(65 * 640);
        for encoding in ENCODINGS {
            let layout = encoding.layout();
            for effort in [
                tcf_core::SearchEffort::Standard,
                tcf_core::SearchEffort::Wide,
                tcf_core::SearchEffort::Exhaustive,
            ] {
                let params = QuantizeParams::default().with_effort(effort);
                let serial =
                    quantize_with(&source, &shape, 2, layout, params).expect("serial quantize");
                let parallel =
                    quantize_tensor(&source, &shape, 2, layout, params).expect("parallel quantize");
                let left = pack(&parallel, layout).expect("parallel packs");
                let right = pack(&serial, layout).expect("serial packs");
                assert_eq!(left, right, "{encoding:?} {effort:?}");
            }
        }
    }

    /// A NaN late in a large tensor reports the same tile a serial run
    /// reports, so the fallback reason cannot depend on thread timing.
    #[test]
    fn a_nonfinite_value_reports_the_serial_error() {
        let shape = [65u64, 640];
        let mut source = values(65 * 640);
        source[30_000] = f32::NAN;
        let layout = NativeEncoding::Q4S32T64.layout();
        let serial = quantize(&source, &shape, 2, layout).unwrap_err();
        let parallel =
            quantize_tensor(&source, &shape, 2, layout, QuantizeParams::default()).unwrap_err();
        assert_eq!(parallel, serial);
        assert!(matches!(parallel, TcfError::SourceNonfinite { .. }));
    }
}
