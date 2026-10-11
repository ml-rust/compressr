//! The GGUF [`FormatWriter`]: header keys, tensor names and row orders over a
//! [`StreamingGgufWriter`].

use super::gguf::{GgmlType, GgufValue, StreamingGgufWriter, GGUF_DEFAULT_ALIGNMENT};
use super::gguf_hf::{reorder_rows, GgufTensorLayout};
use super::{ggml_type_for, FormatWriter};
use crate::loader::UnifiedDType;
use crate::quantization::QuantType;
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// GGUF format writer implementing FormatWriter trait
pub struct GgufFormatWriter {
    /// Output path
    path: Option<PathBuf>,
    /// Inner streaming writer (created on initialize)
    inner: Option<StreamingGgufWriter>,
    /// Pending metadata (set before initialize)
    pending_metadata: Vec<(String, GgufValue)>,
    /// Alignment
    alignment: usize,
    /// Output names, row orders and added tensors
    layout: GgufTensorLayout,
    /// Per registered tensor, in order: its name and `(rows, heads)` when its
    /// rows are reordered on write
    row_plans: VecDeque<(String, Option<(usize, usize)>)>,
    /// Bytes of the layout's added tensors, written before the first
    /// checkpoint tensor
    extra_data: Vec<Vec<u8>>,
}

impl GgufFormatWriter {
    /// Create a new GGUF format writer
    pub fn new() -> Self {
        Self {
            path: None,
            inner: None,
            pending_metadata: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            layout: GgufTensorLayout::default(),
            row_plans: VecDeque::new(),
            extra_data: Vec::new(),
        }
    }

    /// Set model architecture
    pub fn set_architecture(&mut self, arch: &str) {
        self.pending_metadata.push((
            "general.architecture".into(),
            GgufValue::String(arch.to_string()),
        ));
    }

    /// Set model name
    pub fn set_name(&mut self, name: &str) {
        self.pending_metadata
            .push(("general.name".into(), GgufValue::String(name.to_string())));
    }

    /// Set file type (quantization indicator)
    pub fn set_file_type(&mut self, file_type: u32) {
        self.pending_metadata
            .push(("general.file_type".into(), GgufValue::Uint32(file_type)));
    }

    /// Set alignment
    pub fn set_alignment(&mut self, alignment: usize) {
        self.alignment = alignment;
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: &str, value: GgufValue) {
        self.pending_metadata.push((key.into(), value));
    }

    /// Whether `key` is already queued for the header.
    pub fn has_metadata(&self, key: &str) -> bool {
        self.pending_metadata.iter().any(|(k, _)| k == key)
    }

    /// Name, order and add tensors by `layout`. Set it before `initialize`.
    pub fn set_tensor_layout(&mut self, layout: GgufTensorLayout) {
        self.layout = layout;
    }

    fn inner(&mut self) -> Result<&mut StreamingGgufWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Writer not initialized"))
    }

    /// Write the added tensors' bytes, once, ahead of the checkpoint's own.
    fn flush_extra(&mut self) -> Result<()> {
        let pending = std::mem::take(&mut self.extra_data);
        for data in pending {
            self.inner()?.write_tensor_data(&data)?;
        }
        Ok(())
    }
}

impl Default for GgufFormatWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatWriter for GgufFormatWriter {
    fn initialize(&mut self, path: &Path) -> Result<()> {
        // Create the streaming writer
        let mut writer = StreamingGgufWriter::new(path);

        // Apply alignment if custom
        if self.alignment != GGUF_DEFAULT_ALIGNMENT {
            writer = writer.with_alignment(self.alignment);
        }

        // Add pending metadata
        for (key, value) in self.pending_metadata.drain(..) {
            writer.add_metadata(key, value)?;
        }

        // Add quantization version
        writer.add_metadata("general.quantization_version", GgufValue::Uint32(2))?;

        // The added tensors come first, so their bytes go out on the first
        // data write, before any checkpoint tensor.
        for extra in &self.layout.extra {
            let size = GgmlType::F32.tensor_bytes(extra.values.len());
            writer.register_tensor(
                extra.name.clone(),
                vec![extra.values.len()],
                GgmlType::F32,
                size,
            )?;
            self.extra_data
                .push(extra.values.iter().flat_map(|v| v.to_le_bytes()).collect());
        }

        self.path = Some(path.to_path_buf());
        self.inner = Some(writer);
        Ok(())
    }

    fn register_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: UnifiedDType,
        quant_type: Option<QuantType>,
    ) -> Result<()> {
        let out_name = self.layout.output_name(name).to_string();
        let ggml_type = match quant_type {
            Some(qt) => GgmlType::from_quant_type(qt),
            None => ggml_type_for(dtype, &out_name)?,
        };

        // Calculate size based on shape and type
        let n_elements: usize = shape.iter().product();
        let size = ggml_type.tensor_bytes(n_elements);

        let rows = shape.first().copied().unwrap_or(1);
        let plan = self.layout.row_heads(name).map(|heads| (rows, heads));
        self.inner()?
            .register_tensor(out_name.clone(), shape.to_vec(), ggml_type, size)?;
        self.row_plans.push_back((out_name, plan));
        Ok(())
    }

    fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        self.flush_extra()?;
        let (name, plan) = self
            .row_plans
            .pop_front()
            .context("Internal error: tensor data written with no tensor registered for it")?;
        match plan {
            Some((rows, heads)) => {
                let reordered = reorder_rows(&name, data, rows, heads)?;
                self.inner()?.write_tensor_data(&reordered)?;
            }
            None => self.inner()?.write_tensor_data(data)?,
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.inner.is_some() {
            self.flush_extra()?;
        }
        if let Some(ref mut writer) = self.inner {
            writer.finalize()?;
        }
        Ok(())
    }
}
