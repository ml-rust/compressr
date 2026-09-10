//! How ONE tensor's flat tile order is cut into chunks, and where a chunk
//! sits in the tensor's elements.
//!
//! The scheduling that uses this is [`super::parallel`]. The cut is a
//! property of the tile geometry alone — it reads no value and knows no
//! encoding — so it is settled here once and both the quantize path and the
//! `accepts` path take it, which is what keeps the two from cutting a tensor
//! two different ways.

use tcf_core::encoding::SUPER_BLOCK_TILES;
use tcf_core::TcfError;

/// Smallest chunk the split ever produces, in tiles.
///
/// 64 tiles is 4096 weights, hundreds of microseconds of work — orders above
/// rayon's per-task overhead, so the split never degenerates into scheduling.
const MIN_CHUNK_TILES: u64 = 64;

/// Chunks aimed at per thread. More than one so a thread that draws an
/// expensive chunk does not hold up the tensor; few enough that the queue
/// stays short.
const CHUNKS_PER_THREAD: u64 = 4;

/// Tiles per chunk: `total / (threads * CHUNKS_PER_THREAD)`, rounded UP to a
/// multiple of [`SUPER_BLOCK_TILES`] and floored at [`MIN_CHUNK_TILES`].
///
/// The multiple-of-four rounding is what keeps a two-level super-block whole:
/// every chunk boundary then lands on a super-block boundary, so only the
/// final chunk can be short and its trailing partial block is the tensor's
/// own. [`MIN_CHUNK_TILES`] is already a multiple of four, so the floor
/// cannot undo the rounding.
pub(super) fn chunk_tiles(total: u64, threads: usize) -> u64 {
    let threads = u64::try_from(threads).unwrap_or(1).max(1);
    let target = threads.saturating_mul(CHUNKS_PER_THREAD).max(1);
    let even = total.div_ceil(target).max(1);
    let span = u64::from(SUPER_BLOCK_TILES).max(1);
    even.div_ceil(span)
        .saturating_mul(span)
        .max(MIN_CHUNK_TILES)
}

/// `(first_tile_index, tile_count)` for each chunk, in tile order.
pub(super) fn chunk_bounds(total: u64, chunk: u64) -> Vec<(u64, u64)> {
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

/// The `count` tiles starting at tile `first`, as a sub-slice of a tensor's
/// row-major values.
///
/// Tiles never cross a row boundary and a row holds a whole number of tiles,
/// so tile `t` covers `t * tile .. (t + 1) * tile` and a chunk is a plain
/// sub-slice.
pub(super) fn chunk_slice(
    values: &[f32],
    tile: usize,
    first: u64,
    count: u64,
) -> Result<&[f32], TcfError> {
    let start = chunk_start(tile, first)?;
    let held = usize::try_from(count).map_err(|_| TcfError::TileArithmeticOverflow)?;
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

/// The flat ELEMENT index tile `first` starts at.
///
/// This is also `first_element` in `tcf_core::column_weights`: a per-column
/// importance vector expanded for this chunk starts here.
pub(super) fn chunk_start(tile: usize, first: u64) -> Result<usize, TcfError> {
    let offset = usize::try_from(first).map_err(|_| TcfError::TileArithmeticOverflow)?;
    offset
        .checked_mul(tile)
        .ok_or(TcfError::TileArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
