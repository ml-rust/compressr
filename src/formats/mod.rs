//! Format-specific writers for output formats
//!
//! This module provides writers for various model formats:
//! - GGUF (llama.cpp compatibility)
//! - SafeTensors (HuggingFace format)
//!
//! Each writer has two modes:
//! - **Batch mode**: Collects all tensors in memory, then writes them
//! - **Streaming mode**: O(1) memory usage for large models

pub mod checkpoint;
pub mod gguf;
pub mod safetensors;
pub mod shard;
pub mod tcf;

pub use checkpoint::{copy_sidecars, rewrite_torch_dtype, SIDECAR_FILES};

pub use shard::{
    index_file_name, parse_shard_size, plan_shards, read_index_shard_files, write_index, ShardPlan,
    DEFAULT_MAX_SHARD_SIZE, INDEX_SUFFIX,
};

// Re-export main types
pub use gguf::{
    GgmlType, GgufArray, GgufTensorInfo, GgufValue, GgufWriter as GgufBatchWriter,
    StreamingGgufWriter, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION,
};

pub use safetensors::{
    SafeTensorsDType, SafeTensorsWriter as SafeTensorsBatchWriter, StreamingSafeTensorsWriter,
    TensorEntry,
};

pub use tcf::{
    normalized_encoding_name, parse_tcf_encoding, parse_tcf_profile, tcf_encoding_name,
    PlannedTensor, TcfFormatWriter, TcfOptions, TcfProfile, TCF_ENCODING_NAMES, TCF_PROFILE_NAMES,
};

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::loader::UnifiedDType;
use crate::quantization::QuantType;

// ============================================================================
// FormatWriter Trait (Object-Safe)
// ============================================================================

/// Common interface for format writers
///
/// This trait provides a unified interface for writing quantized models
/// to different output formats. Uses `&Path` instead of generic to be object-safe.
pub trait FormatWriter: Send {
    /// Initialize the writer with output path
    fn initialize(&mut self, path: &Path) -> Result<()>;

    /// Register a tensor to be written (for streaming)
    ///
    /// `dtype` is the element type the tensor will be *stored* as. It decides
    /// the reserved byte count, so it must match the data handed to
    /// [`FormatWriter::write_tensor_data`]. It is ignored when `quant_type` is
    /// set, because the quantized representation then decides the size.
    fn register_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: UnifiedDType,
        quant_type: Option<QuantType>,
    ) -> Result<()>;

    /// Write tensor data (in registration order)
    fn write_tensor_data(&mut self, data: &[u8]) -> Result<()>;

    /// Finalize and close the file
    fn finalize(&mut self) -> Result<()>;
}

// ============================================================================
// GGUF FormatWriter Implementation
// ============================================================================

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
}

impl GgufFormatWriter {
    /// Create a new GGUF format writer
    pub fn new() -> Self {
        Self {
            path: None,
            inner: None,
            pending_metadata: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
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
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Writer not initialized"))?;

        let ggml_type = match quant_type {
            Some(qt) => GgmlType::from_quant_type(qt),
            None => ggml_type_for(dtype, name)?,
        };

        // Calculate size based on shape and type
        let n_elements: usize = shape.iter().product();
        let size = ggml_type.tensor_bytes(n_elements);

        writer.register_tensor(name, shape.to_vec(), ggml_type, size)?;
        Ok(())
    }

    fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Writer not initialized"))?;
        writer.write_tensor_data(data)?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.inner {
            writer.finalize()?;
        }
        Ok(())
    }
}

// ============================================================================
// SafeTensors FormatWriter Implementation
// ============================================================================

/// SafeTensors format writer implementing FormatWriter trait.
///
/// The safetensors header records every tensor's name, shape and byte offset,
/// so a shard cannot be opened until every tensor going into it is registered —
/// the inner writer's own `initialize` rejects an empty registration list. GGUF
/// is the opposite: its `register_tensor` needs an already-open writer.
///
/// The trait contract is GGUF's order — `initialize`, then `register_tensor`,
/// then data — so this wrapper only REMEMBERS the path in `initialize`, plans
/// the shard layout on the first data write, by which point every registration
/// has arrived, and opens each shard as the writes reach it.
///
/// Output that fits inside `max_shard_size` stays a single `model.safetensors`
/// with no index, exactly as an unsharded write. Output that does not becomes
/// `model-00001-of-000NN.safetensors` … plus `model.safetensors.index.json`.
pub struct SafeTensorsFormatWriter {
    /// Registered tensors in write order: name, shape, stored type
    registrations: Vec<(String, Vec<usize>, SafeTensorsDType)>,
    /// Destination, recorded by `initialize`
    path: Option<PathBuf>,
    /// Shard-size limit in bytes; `None` writes one file however large
    max_shard_size: Option<u64>,
    /// Header metadata, replayed into every shard
    metadata: Vec<(String, String)>,
    /// Shard layout, computed on the first data write
    plan: Option<ShardPlan>,
    /// The shard currently open, with its index in the plan
    current: Option<(usize, StreamingSafeTensorsWriter)>,
    /// Index of the tensor the next `write_tensor_data` call belongs to
    next_tensor: usize,
    /// Tensor bytes written so far, across every shard
    total_size: u64,
}

impl SafeTensorsFormatWriter {
    /// Create a new SafeTensors format writer, sharding at the default limit.
    pub fn new() -> Self {
        Self {
            registrations: Vec::new(),
            path: None,
            max_shard_size: Some(DEFAULT_MAX_SHARD_SIZE),
            metadata: Vec::new(),
            plan: None,
            current: None,
            next_tensor: 0,
            total_size: 0,
        }
    }

    /// Set the shard-size limit in bytes. `None` writes one file however large.
    pub fn with_max_shard_size(mut self, max_shard_size: Option<u64>) -> Self {
        self.max_shard_size = max_shard_size;
        self
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.push((key.to_string(), value.to_string()));
    }

    /// The directory the shards and the index are written into.
    fn output_dir(&self) -> Result<PathBuf> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SafeTensors writer used before initialize()"))?;
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => Ok(parent.to_path_buf()),
            _ => Ok(PathBuf::from(".")),
        }
    }

    /// The weights file stem the shard names and the index name are built from.
    ///
    /// `out/model.safetensors` gives `model`, so the shards are
    /// `model-00001-of-000NN.safetensors` and the index is
    /// `model.safetensors.index.json`.
    fn stem(&self) -> Result<String> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SafeTensors writer used before initialize()"))?;
        Ok(path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string()))
    }

    /// Compute the shard layout if it has not been computed yet.
    fn ensure_planned(&mut self) -> Result<()> {
        if self.plan.is_some() {
            return Ok(());
        }
        if self.registrations.is_empty() {
            bail!("No tensors registered");
        }

        let sizes: Vec<u64> = self
            .registrations
            .iter()
            .map(|(_, shape, dtype)| {
                (shape.iter().product::<usize>() * dtype.element_size()) as u64
            })
            .collect();

        let plan = plan_shards(&self.stem()?, &sizes, self.max_shard_size);
        if plan.is_sharded() {
            log::info!(
                "Output exceeds the shard limit: writing {} shards plus an index",
                plan.shard_files.len()
            );
        }
        self.plan = Some(plan);
        Ok(())
    }

    /// Make `shard` the open shard, finalizing whichever one preceded it.
    fn open_shard(&mut self, shard: usize) -> Result<()> {
        if let Some((open, _)) = &self.current {
            if *open == shard {
                return Ok(());
            }
        }

        if let Some((_, mut previous)) = self.current.take() {
            previous.finalize()?;
        }

        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Shard layout not planned"))?;
        let file_name = plan
            .shard_files
            .get(shard)
            .ok_or_else(|| anyhow::anyhow!("Shard {} is not in the shard layout", shard))?
            .clone();
        let members: Vec<usize> = plan
            .shard_of
            .iter()
            .enumerate()
            .filter(|(_, &s)| s == shard)
            .map(|(i, _)| i)
            .collect();

        let mut writer = StreamingSafeTensorsWriter::new();
        for (key, value) in &self.metadata {
            writer.set_metadata(key.as_str(), value.as_str());
        }
        for i in members {
            let (name, shape, dtype) = &self.registrations[i];
            writer.register_tensor(name.clone(), shape.clone(), *dtype)?;
        }

        let shard_path = self.output_dir()?.join(&file_name);
        writer.initialize(&shard_path)?;
        self.current = Some((shard, writer));
        Ok(())
    }

    /// Write the index for a sharded checkpoint, or clear a stale one.
    ///
    /// A single-shard write produces no index. It also REMOVES an index left
    /// behind by an earlier sharded conversion into the same directory: that
    /// file names shards that are no longer there, and the loader trusts it.
    fn write_index_file(&self) -> Result<()> {
        let Some(plan) = self.plan.as_ref() else {
            return Ok(());
        };
        let index_path = self.output_dir()?.join(index_file_name(&self.stem()?));

        if !plan.is_sharded() {
            if index_path.is_file() {
                log::warn!(
                    "Removing a stale shard index from an earlier conversion: {}",
                    index_path.display()
                );
                std::fs::remove_file(&index_path).with_context(|| {
                    format!("Failed to remove stale index: {}", index_path.display())
                })?;
            }
            return Ok(());
        }

        let names: Vec<String> = self
            .registrations
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect();
        write_index(&index_path, &plan.weight_map(&names), self.total_size)?;
        log::info!("Wrote shard index: {}", index_path.display());
        Ok(())
    }
}

impl Default for SafeTensorsFormatWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatWriter for SafeTensorsFormatWriter {
    fn initialize(&mut self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create output directory: {}", parent.display())
                })?;
            }
        }
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    fn register_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: UnifiedDType,
        quant_type: Option<QuantType>,
    ) -> Result<()> {
        // For SafeTensors, we store quantized data as raw bytes (U8)
        // The metadata should indicate the original quantization type
        let st_dtype = match quant_type {
            Some(_) => SafeTensorsDType::U8, // Quantized data stored as bytes
            None => safetensors_dtype_for(dtype, name)?,
        };

        // Names must be unique across the whole checkpoint, not only within one
        // shard: the index maps a name onto exactly one shard file.
        if self.registrations.iter().any(|(n, _, _)| n == name) {
            bail!("Duplicate tensor name: {}", name);
        }

        self.registrations
            .push((name.to_string(), shape.to_vec(), st_dtype));
        Ok(())
    }

    fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        self.ensure_planned()?;

        if self.next_tensor >= self.registrations.len() {
            bail!("All tensors already written");
        }

        let shard = match self.plan.as_ref() {
            Some(plan) => plan.shard_of[self.next_tensor],
            None => bail!("Shard layout not planned"),
        };
        self.open_shard(shard)?;

        let writer = match self.current.as_mut() {
            Some((_, writer)) => writer,
            None => bail!("Shard {} could not be opened", shard),
        };
        writer.write_complete_tensor(data)?;

        self.total_size += data.len() as u64;
        self.next_tensor += 1;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        // A checkpoint with no tensors still needs the error, not an empty
        // directory: planning surfaces "No tensors registered".
        self.ensure_planned()?;

        if self.next_tensor != self.registrations.len() {
            bail!(
                "Cannot finalize: {} tensors not yet written (current: {}/{})",
                self.registrations.len() - self.next_tensor,
                self.next_tensor,
                self.registrations.len()
            );
        }

        // Only the last shard is still open; the ones before it were finalized
        // as the writes moved past them.
        if let Some((_, mut writer)) = self.current.take() {
            writer.finalize()?;
        }

        self.write_index_file()
    }
}

// ============================================================================
// dtype Mapping
// ============================================================================

/// Map a unified dtype onto the GGML type used to store it.
///
/// Errors rather than guessing: silently storing an unmapped type as F16 would
/// reserve the wrong number of bytes and corrupt every tensor after it.
fn ggml_type_for(dtype: UnifiedDType, name: &str) -> Result<GgmlType> {
    match dtype {
        UnifiedDType::F32 => Ok(GgmlType::F32),
        UnifiedDType::F16 => Ok(GgmlType::F16),
        UnifiedDType::BF16 => Ok(GgmlType::Bf16),
        UnifiedDType::F64 => Ok(GgmlType::F64),
        UnifiedDType::I8 => Ok(GgmlType::I8),
        UnifiedDType::I16 => Ok(GgmlType::I16),
        UnifiedDType::I32 => Ok(GgmlType::I32),
        UnifiedDType::I64 => Ok(GgmlType::I64),
        UnifiedDType::Q4_0 => Ok(GgmlType::Q4_0),
        UnifiedDType::Q4_1 => Ok(GgmlType::Q4_1),
        UnifiedDType::Q5_0 => Ok(GgmlType::Q5_0),
        UnifiedDType::Q5_1 => Ok(GgmlType::Q5_1),
        UnifiedDType::Q8_0 => Ok(GgmlType::Q8_0),
        UnifiedDType::Q8_1 => Ok(GgmlType::Q8_1),
        UnifiedDType::Q2K => Ok(GgmlType::Q2K),
        UnifiedDType::Q3K => Ok(GgmlType::Q3K),
        UnifiedDType::Q4K => Ok(GgmlType::Q4K),
        UnifiedDType::Q5K => Ok(GgmlType::Q5K),
        UnifiedDType::Q6K => Ok(GgmlType::Q6K),
        UnifiedDType::Q8K => Ok(GgmlType::Q8K),
        other => anyhow::bail!(
            "Tensor '{}' has dtype {} which has no GGUF equivalent",
            name,
            other
        ),
    }
}

/// Map a unified dtype onto the SafeTensors type used to store it.
fn safetensors_dtype_for(dtype: UnifiedDType, name: &str) -> Result<SafeTensorsDType> {
    match dtype {
        UnifiedDType::F32 => Ok(SafeTensorsDType::F32),
        UnifiedDType::F16 => Ok(SafeTensorsDType::F16),
        UnifiedDType::BF16 => Ok(SafeTensorsDType::BF16),
        UnifiedDType::F64 => Ok(SafeTensorsDType::F64),
        UnifiedDType::I8 => Ok(SafeTensorsDType::I8),
        UnifiedDType::I16 => Ok(SafeTensorsDType::I16),
        UnifiedDType::I32 => Ok(SafeTensorsDType::I32),
        UnifiedDType::I64 => Ok(SafeTensorsDType::I64),
        UnifiedDType::U8 => Ok(SafeTensorsDType::U8),
        UnifiedDType::U32 => Ok(SafeTensorsDType::U32),
        UnifiedDType::Bool => Ok(SafeTensorsDType::Bool),
        other => anyhow::bail!(
            "Tensor '{}' has dtype {} which SafeTensors cannot store unquantized",
            name,
            other
        ),
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a format writer for the given output format
///
/// `max_shard_size` caps the bytes in one output shard. It applies to the
/// SafeTensors-based formats only: GGUF is one self-contained file with no
/// index to point at a second one. `None` writes a single file however large.
/// An unimplemented format is an ERROR, not a silent fallback. Writing plain
/// SafeTensors for `--format gptq` produced a file that claimed to be GPTQ and
/// was not: AWQ and GPTQ store a quantized layer as several tensors (`qweight`
/// packed as I32, plus `scales`, `qzeros`, `g_idx`), so nothing downstream can
/// read the fallback as the format it asked for. A `log::warn!` is not enough —
/// it scrolls past and the wrong file ships.
pub fn create_writer(
    format: crate::OutputFormat,
    max_shard_size: Option<u64>,
) -> Result<Box<dyn FormatWriter>> {
    match format {
        crate::OutputFormat::Gguf => Ok(Box::new(GgufFormatWriter::new())),
        crate::OutputFormat::Safetensors => Ok(Box::new(
            SafeTensorsFormatWriter::new().with_max_shard_size(max_shard_size),
        )),
        // Unconfigured, exactly as the GGUF arm is: `convert` replaces it with
        // a writer carrying the encoding and the input path, and an
        // unconfigured one errors in `initialize` rather than writing a file
        // with an invented encoding.
        crate::OutputFormat::Tcf => Ok(Box::new(TcfFormatWriter::new())),
        other => anyhow::bail!(
            "Output format {other:?} is not implemented. compressr writes gguf, safetensors \
             and tcf; {other:?} needs its own on-disk layout (separate qweight / \
             scales / qzeros tensors), which is not built yet. Earlier versions silently \
             wrote plain safetensors here, producing a file that claimed to be {other:?}."
        ),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_format_writer_creation() {
        let mut writer = GgufFormatWriter::new();
        writer.set_architecture("llama");
        writer.set_name("test-model");
        writer.set_file_type(12); // Q4_K
    }

    #[test]
    fn test_safetensors_format_writer_creation() {
        let mut writer = SafeTensorsFormatWriter::new();
        writer.set_metadata("format", "pt");
    }

    #[test]
    fn test_create_writer_gguf() {
        let writer = create_writer(crate::OutputFormat::Gguf, None).unwrap();
        // Just verify it doesn't panic
        drop(writer);
    }

    #[test]
    fn test_create_writer_safetensors() {
        let writer = create_writer(crate::OutputFormat::Safetensors, None).unwrap();
        // Just verify it doesn't panic
        drop(writer);
    }

    #[test]
    fn test_create_writer_tcf() {
        let writer = create_writer(crate::OutputFormat::Tcf, None).unwrap();
        drop(writer);
    }

    #[test]
    fn test_safetensors_dtype_mapping() {
        assert_eq!(
            safetensors_dtype_for(UnifiedDType::BF16, "w").unwrap(),
            SafeTensorsDType::BF16
        );
        assert_eq!(
            safetensors_dtype_for(UnifiedDType::F32, "w").unwrap(),
            SafeTensorsDType::F32
        );
        assert_eq!(
            safetensors_dtype_for(UnifiedDType::I64, "w").unwrap(),
            SafeTensorsDType::I64
        );

        let err = safetensors_dtype_for(UnifiedDType::Q4K, "w")
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "Tensor 'w' has dtype Q4_K which SafeTensors cannot store unquantized"
        );
    }

    #[test]
    fn test_ggml_type_mapping() {
        assert_eq!(
            ggml_type_for(UnifiedDType::BF16, "w").unwrap(),
            GgmlType::Bf16
        );
        assert_eq!(
            ggml_type_for(UnifiedDType::F32, "w").unwrap(),
            GgmlType::F32
        );
        assert_eq!(
            ggml_type_for(UnifiedDType::Q6K, "w").unwrap(),
            GgmlType::Q6K
        );

        let err = ggml_type_for(UnifiedDType::Bool, "w")
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "Tensor 'w' has dtype Bool which has no GGUF equivalent"
        );
    }

    /// A registered dtype must reserve the byte count the data actually has.
    /// Registering everything as F16 (the old default) truncated F32 inputs.
    #[test]
    fn test_safetensors_writer_reserves_source_dtype_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let mut writer = SafeTensorsFormatWriter::new();
        writer
            .register_tensor("w", &[2, 2], UnifiedDType::F32, None)
            .unwrap();
        writer.initialize(&path).unwrap();
        // 4 F32 elements = 16 bytes. An F16 registration would reject this.
        writer.write_tensor_data(&[0u8; 16]).unwrap();
        writer.finalize().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();

        assert!(
            header.contains("\"dtype\":\"F32\""),
            "header was: {}",
            header
        );
        assert_eq!(bytes.len(), 8 + header_len + 16);
    }

    /// Read a SafeTensors file into (parsed header, data block).
    fn read_st(path: &Path) -> (serde_json::Value, Vec<u8>) {
        let bytes = std::fs::read(path).unwrap();
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        (header, bytes[8 + header_len..].to_vec())
    }

    #[test]
    fn test_safetensors_writer_shards_and_writes_an_index() {
        let dir = tempfile::tempdir().unwrap();

        // Three 16-byte F32 tensors with a 20-byte limit: one per shard after
        // the first, because a second 16-byte tensor would overflow.
        let mut writer = SafeTensorsFormatWriter::new().with_max_shard_size(Some(20));
        for name in ["a", "b", "c"] {
            writer
                .register_tensor(name, &[2, 2], UnifiedDType::F32, None)
                .unwrap();
        }
        writer
            .initialize(&dir.path().join("model.safetensors"))
            .unwrap();
        writer.write_tensor_data(&[1u8; 16]).unwrap();
        writer.write_tensor_data(&[2u8; 16]).unwrap();
        writer.write_tensor_data(&[3u8; 16]).unwrap();
        writer.finalize().unwrap();

        assert!(!dir.path().join("model.safetensors").exists());
        for i in 1..=3 {
            assert!(dir
                .path()
                .join(format!("model-{:05}-of-00003.safetensors", i))
                .exists());
        }

        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(index["metadata"]["total_size"], 48);
        assert_eq!(index["weight_map"]["a"], "model-00001-of-00003.safetensors");
        assert_eq!(index["weight_map"]["b"], "model-00002-of-00003.safetensors");
        assert_eq!(index["weight_map"]["c"], "model-00003-of-00003.safetensors");

        let (header, data) = read_st(&dir.path().join("model-00002-of-00003.safetensors"));
        assert_eq!(header["b"]["dtype"], "F32");
        assert_eq!(data, vec![2u8; 16]);
        assert!(header.get("a").is_none());
    }

    #[test]
    fn test_safetensors_writer_single_shard_writes_no_index() {
        let dir = tempfile::tempdir().unwrap();

        let mut writer = SafeTensorsFormatWriter::new().with_max_shard_size(Some(1_000_000));
        writer
            .register_tensor("w", &[2, 2], UnifiedDType::F32, None)
            .unwrap();
        writer
            .initialize(&dir.path().join("model.safetensors"))
            .unwrap();
        writer.write_tensor_data(&[7u8; 16]).unwrap();
        writer.finalize().unwrap();

        assert!(dir.path().join("model.safetensors").is_file());
        assert!(!dir.path().join("model.safetensors.index.json").exists());

        let (header, data) = read_st(&dir.path().join("model.safetensors"));
        assert_eq!(header["w"]["data_offsets"][1], 16);
        assert_eq!(data, vec![7u8; 16]);
    }

    #[test]
    fn test_safetensors_writer_single_shard_removes_a_stale_index() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &index,
            r#"{"weight_map":{"gone":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let mut writer = SafeTensorsFormatWriter::new().with_max_shard_size(None);
        writer
            .register_tensor("w", &[2], UnifiedDType::F32, None)
            .unwrap();
        writer
            .initialize(&dir.path().join("model.safetensors"))
            .unwrap();
        writer.write_tensor_data(&[0u8; 8]).unwrap();
        writer.finalize().unwrap();

        assert!(!index.exists());
    }

    #[test]
    fn test_safetensors_writer_uses_the_weights_stem_for_shard_names() {
        let dir = tempfile::tempdir().unwrap();

        let mut writer = SafeTensorsFormatWriter::new().with_max_shard_size(Some(8));
        writer
            .register_tensor("a", &[2], UnifiedDType::F32, None)
            .unwrap();
        writer
            .register_tensor("b", &[2], UnifiedDType::F32, None)
            .unwrap();
        writer
            .initialize(&dir.path().join("weights.safetensors"))
            .unwrap();
        writer.write_tensor_data(&[0u8; 8]).unwrap();
        writer.write_tensor_data(&[0u8; 8]).unwrap();
        writer.finalize().unwrap();

        assert!(dir
            .path()
            .join("weights-00001-of-00002.safetensors")
            .is_file());
        assert!(dir.path().join("weights.safetensors.index.json").is_file());
    }

    #[test]
    fn test_safetensors_writer_rejects_a_duplicate_name_across_shards() {
        let mut writer = SafeTensorsFormatWriter::new().with_max_shard_size(Some(8));
        writer
            .register_tensor("w", &[2], UnifiedDType::F32, None)
            .unwrap();
        let err = writer
            .register_tensor("w", &[2], UnifiedDType::F32, None)
            .unwrap_err()
            .to_string();
        assert_eq!(err, "Duplicate tensor name: w");
    }

    #[test]
    fn test_safetensors_writer_finalize_rejects_missing_data() {
        let dir = tempfile::tempdir().unwrap();

        let mut writer = SafeTensorsFormatWriter::new().with_max_shard_size(Some(8));
        writer
            .register_tensor("a", &[2], UnifiedDType::F32, None)
            .unwrap();
        writer
            .register_tensor("b", &[2], UnifiedDType::F32, None)
            .unwrap();
        writer
            .initialize(&dir.path().join("model.safetensors"))
            .unwrap();
        writer.write_tensor_data(&[0u8; 8]).unwrap();

        let err = writer.finalize().unwrap_err().to_string();
        assert_eq!(
            err,
            "Cannot finalize: 1 tensors not yet written (current: 1/2)"
        );
    }

    #[test]
    fn test_safetensors_writer_finalize_without_tensors_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = SafeTensorsFormatWriter::new();
        writer
            .initialize(&dir.path().join("model.safetensors"))
            .unwrap();

        let err = writer.finalize().unwrap_err().to_string();
        assert_eq!(err, "No tensors registered");
    }

    #[test]
    /// An unimplemented format must ERROR. It used to fall back to plain
    /// safetensors behind a `log::warn!`, so `--format gptq` shipped a file that
    /// claimed to be GPTQ and no GPTQ loader could read.
    fn test_create_writer_refuses_unimplemented_formats() {
        for format in [
            crate::OutputFormat::Awq,
            crate::OutputFormat::Gptq,
            crate::OutputFormat::Exl2,
            crate::OutputFormat::Fp8,
        ] {
            let Err(err) = create_writer(format, None) else {
                panic!("{format:?} is not implemented and must not silently fall back");
            };
            let msg = err.to_string();
            assert!(msg.contains("not implemented"), "{msg}");
            assert!(
                msg.contains(&format!("{format:?}")),
                "the error must name the format: {msg}"
            );
        }
    }
}
