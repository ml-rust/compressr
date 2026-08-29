//! A temporary home for the payloads pass 1 already computed.
//!
//! # Why a file and not a `Vec`
//!
//! TCF fixes every tensor's encoding and offset in a directory written before
//! any payload, and a profile's bump set ranks tensors against each other, so
//! no payload can be written to the output until every tensor has been
//! screened. Pass 1 nevertheless computes the winning tiles for every tensor
//! it settles at the base encoding, and pass 2 used to throw them away and
//! quantize the same values a second time — the same bytes, at roughly 150 ns
//! per element.
//!
//! Holding those tiles in memory would make peak RAM the whole quantized
//! model, which breaks compressr's one-tensor-in-flight bound. Holding them in
//! a temporary file does not: one tensor's payload is packed, appended, and
//! dropped, and pass 2 reads one back at a time. The bound stays the largest
//! tensor.
//!
//! # Why packed bytes and not tiles
//!
//! Packing is `tcf-core`'s own [`pack`], and reading back is its own
//! [`unpack`] — the reader's inverse, derived independently per CONFORMANCE.md
//! Section 0.1 and lossless over every logical field a tile carries. So the
//! spill holds no layout knowledge of its own, which MIGRATION.md Section
//! 4.5.3 forbids a producer from carrying. `unpack(pack(tiles))` restores the
//! same scales, minima, super-scales and codes, so the semantic digest
//! `tcf-core` computes from them is the digest the tiles would have produced.
//!
//! # Failure is never fatal
//!
//! Every entry point returns the spill to an unusable state rather than
//! aborting a conversion: quantizing again yields exactly the same bytes, so a
//! full temporary filesystem costs time and nothing else. The writer drops the
//! spill on the first error and falls back to the second quantize.
//!
//! The file is unnamed — [`tempfile::tempfile_in`] unlinks it at creation — so
//! it is reclaimed on success, on error, and on a crash alike, with no cleanup
//! path to get wrong.

use anyhow::{anyhow, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use tcf_core::{pack, unpack, LogicalTile, NativeEncoding};

/// Where one tensor's packed payload sits, and what it was packed as.
#[derive(Debug, Clone, Copy)]
struct SpillEntry {
    /// Byte offset in the spill file
    offset: u64,
    /// Packed length in bytes
    length: u64,
    /// Tiles the payload holds, which `unpack` needs and the bytes do not carry
    tiles: u64,
    /// The encoding it was quantized at
    ///
    /// A profile bumps a tensor AFTER pass 1, and a bumped tensor's spilled
    /// payload is at the base encoding — the wrong one for its record. The
    /// encoding is stored so pass 2 can tell, rather than trusting that the
    /// plan it holds is the plan pass 1 spilled.
    encoding: NativeEncoding,
}

/// Pass 1's packed payloads, appended in tensor order.
pub(super) struct PayloadSpill {
    /// Buffered append handle, taken by [`PayloadSpill::seal`]
    sink: Option<BufWriter<File>>,
    /// Read handle, set by [`PayloadSpill::seal`]
    source: Option<File>,
    /// One slot per tensor, `None` for a tensor with nothing spilled
    entries: Vec<Option<SpillEntry>>,
    /// Bytes appended so far, which is the next entry's offset
    end: u64,
}

impl PayloadSpill {
    /// Open a spill beside `output`, so it lands on the filesystem that is
    /// already sized for the file being written.
    ///
    /// # Errors
    /// The directory refuses a temporary file.
    pub(super) fn create(output: &Path) -> Result<Self> {
        let directory = match output.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        let file = tempfile::tempfile_in(directory).with_context(|| {
            format!(
                "Failed to open a TCF payload spill in {}",
                directory.display()
            )
        })?;
        Ok(Self {
            sink: Some(BufWriter::new(file)),
            source: None,
            entries: Vec::new(),
            end: 0,
        })
    }

    /// Append one tensor's payload, packed at `encoding`.
    ///
    /// # Errors
    /// The spill is already sealed, or the write failed.
    pub(super) fn push(&mut self, tiles: &[LogicalTile], encoding: NativeEncoding) -> Result<()> {
        let packed = pack(tiles, encoding.layout())
            .map_err(|e| anyhow!(e))
            .context("Failed to pack a payload for the TCF spill")?;
        let length = u64::try_from(packed.len())
            .context("A TCF payload is longer than the spill can address")?;
        let tiles = u64::try_from(tiles.len())
            .context("A TCF payload holds more tiles than the spill can address")?;
        let sink = self
            .sink
            .as_mut()
            .ok_or_else(|| anyhow!("The TCF payload spill was written after it was sealed"))?;
        sink.write_all(&packed)
            .context("Failed to write the TCF payload spill")?;
        let offset = self.end;
        self.end = self
            .end
            .checked_add(length)
            .context("The TCF payload spill grew past u64")?;
        self.entries.push(Some(SpillEntry {
            offset,
            length,
            tiles,
            encoding,
        }));
        Ok(())
    }

    /// Record that this tensor has no spilled payload, keeping the slots
    /// aligned with the plans.
    pub(super) fn push_absent(&mut self) {
        self.entries.push(None);
    }

    /// Stop appending and open the spill for reading.
    ///
    /// # Errors
    /// The buffered tail did not reach the file.
    pub(super) fn seal(&mut self) -> Result<()> {
        let Some(sink) = self.sink.take() else {
            return Ok(());
        };
        // `into_inner` flushes, and reports the buffered bytes it could not
        // write rather than dropping them silently the way a `BufWriter` drop
        // would.
        let file = sink
            .into_inner()
            .map_err(|e| anyhow!("Failed to flush the TCF payload spill: {e}"))?;
        self.source = Some(file);
        Ok(())
    }

    /// Whether the payload spilled for `index` is the one a record encoded at
    /// `encoding` needs.
    pub(super) fn holds(&self, index: usize, encoding: NativeEncoding) -> bool {
        self.entries
            .get(index)
            .and_then(|entry| *entry)
            .is_some_and(|entry| entry.encoding == encoding)
    }

    /// Read back one tensor's tiles, or `None` when nothing usable was
    /// spilled for it.
    ///
    /// The tiles are byte-for-byte what pass 1 produced: `unpack` is
    /// `tcf-core`'s inverse of the `pack` that wrote them.
    ///
    /// # Errors
    /// The spill is unsealed, the read failed, or the bytes do not decode —
    /// each of which the caller answers by quantizing again.
    pub(super) fn take(
        &mut self,
        index: usize,
        encoding: NativeEncoding,
    ) -> Result<Option<Vec<LogicalTile>>> {
        if !self.holds(index, encoding) {
            return Ok(None);
        }
        let Some(entry) = self.entries.get(index).and_then(|entry| *entry) else {
            return Ok(None);
        };
        let source = self
            .source
            .as_mut()
            .ok_or_else(|| anyhow!("The TCF payload spill was read before it was sealed"))?;
        let length = usize::try_from(entry.length)
            .context("A spilled TCF payload is longer than this machine addresses")?;
        source
            .seek(SeekFrom::Start(entry.offset))
            .context("Failed to seek the TCF payload spill")?;
        let mut packed = vec![0u8; length];
        source
            .read_exact(&mut packed)
            .context("Failed to read the TCF payload spill")?;
        let tiles = unpack(&packed, entry.tiles, encoding.layout())
            .map_err(|e| anyhow!(e))
            .context("Failed to decode a spilled TCF payload")?;
        Ok(Some(tiles))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::quantize;

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

    /// The whole point: a payload that goes through the spill comes back as
    /// the tiles that went in. If this ever fails, pass 2 writes bytes pass 1
    /// did not compute.
    #[test]
    fn a_spilled_payload_comes_back_unchanged() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let output = dir.path().join("model.tcf");
        let mut spill = PayloadSpill::create(&output).expect("opens a spill");

        let shape = [65u64, 640];
        let source = values(65 * 640);
        let mut expected = Vec::new();
        for encoding in ENCODINGS {
            let tiles = quantize(&source, &shape, 2, encoding.layout()).expect("quantizes");
            expected.push(pack(&tiles, encoding.layout()).expect("packs"));
            spill.push(&tiles, encoding).expect("spills");
        }
        spill.seal().expect("seals");

        for (index, encoding) in ENCODINGS.into_iter().enumerate() {
            let recovered = spill
                .take(index, encoding)
                .expect("reads back")
                .expect("a payload was spilled");
            let packed = pack(&recovered, encoding.layout()).expect("repacks");
            assert_eq!(packed, expected[index], "{encoding:?}");
        }
    }

    /// A bumped tensor's spilled payload is at the base encoding, so the
    /// spill refuses it rather than writing a payload the record misdescribes.
    #[test]
    fn a_payload_spilled_at_another_encoding_is_refused() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let mut spill = PayloadSpill::create(&dir.path().join("model.tcf")).expect("opens a spill");

        let shape = [1u64, 64];
        let source = values(64);
        let base = NativeEncoding::Q4AS32DT64;
        let tiles = quantize(&source, &shape, 2, base.layout()).expect("quantizes");
        spill.push(&tiles, base).expect("spills");
        spill.push_absent();
        spill.seal().expect("seals");

        assert!(spill.holds(0, base));
        assert!(!spill.holds(0, NativeEncoding::Q6S16DT64));
        assert!(spill
            .take(0, NativeEncoding::Q6S16DT64)
            .expect("refuses without an error")
            .is_none());
        assert!(!spill.holds(1, base));
        assert!(spill.take(1, base).expect("nothing was spilled").is_none());
        assert!(spill.take(7, base).expect("no such tensor").is_none());
    }
}
