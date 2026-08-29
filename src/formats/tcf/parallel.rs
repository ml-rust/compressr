//! Quantize ONE tensor's tile chunks across rayon's thread pool.
//!
//! # Why the threading is here
//!
//! `tcf-core` permits exactly one dependency, blake3, and CONFORMANCE.md makes
//! its scalar implementation the definition of codec semantics. So it gains a
//! chunk-aware entry point — [`quantize_tile_chunk`] — and nothing else; the
//! scheduling that calls it lives in the consumer, which is this module.
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
//! never splits either: [`quantize_tile_chunk`] refuses a chunk that would.
//! Nothing is accumulated across tiles, so no float is reassociated. Chunks
//! are concatenated in tile order, and the first error IN TILE ORDER is the
//! one returned, so even a rejected tensor takes the serial path's fallback
//! reason.

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;

use tcf_core::encoding::{
    accepts, accepts_tile_chunk, quantize, quantize_tile_chunk, tile_count, SUPER_BLOCK_TILES,
};
use tcf_core::{LogicalTile, QuantLayout, TcfError};

/// Tiles below which a tensor is quantized serially, with no rayon call at
/// all.
///
/// 512 tiles is 32768 weights. At roughly 10 microseconds of scale search per
/// tile that is about 5 ms of work — small enough that a serial run finishes
/// before a split would pay for itself, and large enough that every tensor
/// with real cost is still split. Norm and bias vectors sit far below it and
/// stay on the serial path entirely.
const MIN_PARALLEL_TILES: u64 = 512;

/// Smallest chunk the split ever produces, in tiles.
///
/// 64 tiles is 4096 weights, hundreds of microseconds of work — orders above
/// rayon's per-task overhead, so the split never degenerates into scheduling.
const MIN_CHUNK_TILES: u64 = 64;

/// Chunks aimed at per thread. More than one so a thread that draws an
/// expensive chunk does not hold up the tensor; few enough that the queue
/// stays short.
const CHUNKS_PER_THREAD: u64 = 4;

/// Quantize a whole tensor, splitting its tiles across rayon's pool.
///
/// Byte-for-byte what `tcf_core::encoding::quantize` returns for the same
/// arguments — this is a scheduling change and nothing else. `values` holds
/// the tensor row-major, `dims[0..rank]` is its shape.
///
/// # Errors
/// Whatever `tcf_core::encoding::quantize` raises, and for the same values.
/// When several chunks fail, the failure of the LOWEST tile index is
/// returned, which is the one a serial run would have hit first.
pub(super) fn quantize_tensor(
    values: &[f32],
    dims: &[u64],
    rank: u32,
    layout: QuantLayout,
) -> Result<Vec<LogicalTile>, TcfError> {
    let tile = usize::from(layout.geometry.tile);
    let total = tile_count(dims, rank, layout.geometry.tile)?;
    if total < MIN_PARALLEL_TILES {
        // The serial path is `quantize` itself, not a one-chunk call: a small
        // tensor then touches no scheduling code at all, and the two paths
        // cannot drift.
        return quantize(values, dims, rank, layout);
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
            quantize_tile_chunk(slice, count, first, final_chunk, layout)
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
        // quantize path calls `quantize`: the two cannot drift.
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

/// The `count` tiles starting at tile `first`, as a sub-slice of a tensor's
/// row-major values.
///
/// Tiles never cross a row boundary and a row holds a whole number of tiles,
/// so tile `t` covers `t * tile .. (t + 1) * tile` and a chunk is a plain
/// sub-slice.
fn chunk_slice(values: &[f32], tile: usize, first: u64, count: u64) -> Result<&[f32], TcfError> {
    let offset = usize::try_from(first).map_err(|_| TcfError::TileArithmeticOverflow)?;
    let held = usize::try_from(count).map_err(|_| TcfError::TileArithmeticOverflow)?;
    let start = offset
        .checked_mul(tile)
        .ok_or(TcfError::TileArithmeticOverflow)?;
    let span = held
        .checked_mul(tile)
        .ok_or(TcfError::TileArithmeticOverflow)?;
    let end = start
        .checked_add(span)
        .ok_or(TcfError::TileArithmeticOverflow)?;
    values
        .get(start..end)
        .ok_or(TcfError::TileArithmeticOverflow)
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
    name: &str,
) -> Result<Vec<LogicalTile>> {
    quantize_tensor(values, dims, rank, layout)
        .map_err(|e| anyhow!(e))
        .with_context(|| format!("Failed to quantize tensor '{name}'"))
}

/// Tiles per chunk: `total / (threads * CHUNKS_PER_THREAD)`, rounded UP to a
/// multiple of [`SUPER_BLOCK_TILES`] and floored at [`MIN_CHUNK_TILES`].
///
/// The multiple-of-four rounding is what keeps a two-level super-block whole:
/// every chunk boundary then lands on a super-block boundary, so only the
/// final chunk can be short and its trailing partial block is the tensor's
/// own. [`MIN_CHUNK_TILES`] is already a multiple of four, so the floor
/// cannot undo the rounding.
fn chunk_tiles(total: u64, threads: usize) -> u64 {
    let threads = u64::try_from(threads).unwrap_or(1).max(1);
    let target = threads.saturating_mul(CHUNKS_PER_THREAD).max(1);
    let even = total.div_ceil(target).max(1);
    let span = u64::from(SUPER_BLOCK_TILES).max(1);
    even.div_ceil(span)
        .saturating_mul(span)
        .max(MIN_CHUNK_TILES)
}

/// `(first_tile_index, tile_count)` for each chunk, in tile order.
fn chunk_bounds(total: u64, chunk: u64) -> Vec<(u64, u64)> {
    let chunk = chunk.max(1);
    let mut bounds = Vec::new();
    let mut first = 0u64;
    while first < total {
        let count = chunk.min(total.saturating_sub(first));
        bounds.push((first, count));
        first = first.saturating_add(count);
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::encoding::pack;
    use tcf_core::NativeEncoding;

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

    /// Every chunk boundary lands on a super-block, whatever the thread
    /// count, so a two-level block is never split.
    #[test]
    fn every_chunk_size_is_a_multiple_of_the_super_block() {
        for threads in [1usize, 2, 8, 24, 128] {
            for total in [512u64, 513, 1000, 4096, 1_000_003] {
                let chunk = chunk_tiles(total, threads);
                assert_eq!(chunk % u64::from(SUPER_BLOCK_TILES), 0, "{total} {threads}");
                assert!(chunk >= MIN_CHUNK_TILES, "{total} {threads}");
            }
        }
    }

    /// Only the last chunk may be short, and the bounds cover every tile
    /// exactly once in order.
    #[test]
    fn chunk_bounds_cover_every_tile_in_order() {
        let bounds = chunk_bounds(1000, 64);
        assert_eq!(bounds.first(), Some(&(0, 64)));
        assert_eq!(bounds.last(), Some(&(960, 40)));
        let mut expected = 0u64;
        for (first, count) in &bounds {
            assert_eq!(*first, expected);
            expected += count;
        }
        assert_eq!(expected, 1000);
        assert!(chunk_bounds(0, 64).is_empty());
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
            let parallel = quantize_tensor(&source, &shape, 2, layout).expect("parallel quantize");
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
            let parallel = quantize_tensor(&source, &shape, 2, layout).expect("parallel quantize");
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
                    let quantized = quantize_tensor(source, &shape, 2, layout).is_ok();
                    let accepted = accepts_tensor(source, &shape, 2, layout).expect("decides");
                    assert_eq!(accepted, quantized, "{encoding:?} {name} {shape:?}");
                }
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
        let parallel = quantize_tensor(&source, &shape, 2, layout).unwrap_err();
        assert_eq!(parallel, serial);
        assert!(matches!(parallel, TcfError::SourceNonfinite { .. }));
    }
}
