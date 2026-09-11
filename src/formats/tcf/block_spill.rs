//! A temporary home for block payloads between registration and the write.
//!
//! The block writer receives each tensor's finished bytes from the
//! conversion loop, one tensor at a time, before TCF's directory can be
//! written. Holding every payload until then would make peak RAM the whole
//! quantized model; appending each to an unnamed temporary file beside the
//! output keeps the bound at one tensor, the same bound [`super::spill`]
//! gives the native writer.
//!
//! Unlike that spill, this one carries no layout: a block payload is the
//! bytes the GGUF writer would store, and comes back verbatim. An error here
//! is fatal — there is no quantizer to ask again.

use anyhow::{anyhow, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Where one tensor's payload sits in the spill file.
#[derive(Debug, Clone, Copy)]
struct Entry {
    offset: u64,
    length: u64,
}

/// Finished payloads, appended in tensor order.
pub(super) struct BlockSpill {
    /// Buffered append handle, taken by [`BlockSpill::seal`]
    sink: Option<BufWriter<File>>,
    /// Read handle, set by [`BlockSpill::seal`]
    source: Option<File>,
    entries: Vec<Entry>,
    /// Bytes appended so far, which is the next entry's offset
    end: u64,
}

impl BlockSpill {
    /// Open a spill beside `output`, on the filesystem already sized for the
    /// file being written. The file is unlinked at creation, so it is
    /// reclaimed on every exit path.
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

    /// Append one tensor's payload.
    pub(super) fn push(&mut self, bytes: &[u8]) -> Result<()> {
        let length = u64::try_from(bytes.len())
            .context("A TCF payload is longer than the spill can address")?;
        let sink = self
            .sink
            .as_mut()
            .ok_or_else(|| anyhow!("The TCF payload spill was written after it was sealed"))?;
        sink.write_all(bytes)
            .context("Failed to write the TCF payload spill")?;
        let offset = self.end;
        self.end = self
            .end
            .checked_add(length)
            .context("The TCF payload spill grew past u64")?;
        self.entries.push(Entry { offset, length });
        Ok(())
    }

    /// Payloads appended so far.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Stop appending and open the spill for reading.
    pub(super) fn seal(&mut self) -> Result<()> {
        let Some(sink) = self.sink.take() else {
            return Ok(());
        };
        let file = sink
            .into_inner()
            .map_err(|e| anyhow!("Failed to flush the TCF payload spill: {e}"))?;
        self.source = Some(file);
        Ok(())
    }

    /// Read back tensor `index`'s payload, verbatim.
    pub(super) fn take(&mut self, index: usize) -> Result<Vec<u8>> {
        let entry = *self
            .entries
            .get(index)
            .ok_or_else(|| anyhow!("No TCF payload was spilled for tensor {index}"))?;
        let source = self
            .source
            .as_mut()
            .ok_or_else(|| anyhow!("The TCF payload spill was read before it was sealed"))?;
        let length = usize::try_from(entry.length)
            .context("A spilled TCF payload is longer than this machine addresses")?;
        source
            .seek(SeekFrom::Start(entry.offset))
            .context("Failed to seek the TCF payload spill")?;
        let mut bytes = vec![0u8; length];
        source
            .read_exact(&mut bytes)
            .context("Failed to read the TCF payload spill")?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payloads_come_back_verbatim_in_any_order() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let mut spill = BlockSpill::create(&dir.path().join("model.tcf")).expect("opens");
        let payloads: Vec<Vec<u8>> = vec![vec![1, 2, 3], Vec::new(), vec![9; 4096]];
        for payload in &payloads {
            spill.push(payload).expect("spills");
        }
        assert_eq!(spill.len(), 3);
        spill.seal().expect("seals");
        assert_eq!(spill.take(2).expect("reads"), payloads[2]);
        assert_eq!(spill.take(0).expect("reads"), payloads[0]);
        assert_eq!(spill.take(1).expect("reads"), payloads[1]);
        assert!(spill.take(3).is_err());
    }

    #[test]
    fn a_sealed_spill_refuses_writes_and_an_unsealed_one_refuses_reads() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let mut spill = BlockSpill::create(&dir.path().join("model.tcf")).expect("opens");
        spill.push(&[1]).expect("spills");
        assert!(spill.take(0).is_err());
        spill.seal().expect("seals");
        assert!(spill.push(&[2]).is_err());
    }
}
