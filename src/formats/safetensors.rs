//! SafeTensors format writer
//!
//! Implements writing models in the HuggingFace SafeTensors format.
//!
//! # Format Specification
//!
//! ```text
//! [8 bytes] header_size (little-endian u64)
//! [header_size bytes] JSON header containing:
//!   - "__metadata__": optional dict of string key-value pairs
//!   - "<tensor_name>": { "dtype": str, "shape": [int], "data_offsets": [start, end] }
//! [remaining bytes] raw tensor data (contiguous)
//! ```
//!
//! # Supported Data Types
//!
//! - F32, F16, BF16 (primary for LLM weights)
//! - F64, I64, I32, U32, U16, U8, I8, BOOL
//!
//! # Usage
//!
//! ```ignore
//! use compressr::formats::safetensors::SafeTensorsWriter;
//!
//! let mut writer = SafeTensorsWriter::new();
//! writer.set_metadata("format", "pt");
//! writer.add_tensor("weight", &[1024, 768], DType::F16, &data)?;
//! writer.write("model.safetensors")?;
//! ```

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// ============================================================================
// Data Types
// ============================================================================

/// SafeTensors data type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SafeTensorsDType {
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE 754 half)
    F16,
    /// 16-bit bfloat (brain floating point)
    BF16,
    /// 64-bit float
    F64,
    /// 64-bit signed integer
    I64,
    /// 32-bit signed integer
    I32,
    /// 16-bit signed integer
    I16,
    /// 8-bit signed integer
    I8,
    /// 32-bit unsigned integer
    U32,
    /// 16-bit unsigned integer
    U16,
    /// 8-bit unsigned integer
    U8,
    /// Boolean (1 byte)
    Bool,
}

impl SafeTensorsDType {
    /// Get the string representation for JSON header
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U32 => "U32",
            Self::U16 => "U16",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
        }
    }

    /// Get bytes per element
    pub fn element_size(&self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Parse from string.
    ///
    /// Deliberately NOT `std::str::FromStr`: that returns `Result` and forces an
    /// error type, while an unrecognised dtype name here is an ordinary absence
    /// the callers branch on.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "F32" | "FLOAT32" => Some(Self::F32),
            "F16" | "FLOAT16" => Some(Self::F16),
            "BF16" | "BFLOAT16" => Some(Self::BF16),
            "F64" | "FLOAT64" => Some(Self::F64),
            "I64" | "INT64" => Some(Self::I64),
            "I32" | "INT32" => Some(Self::I32),
            "I16" | "INT16" => Some(Self::I16),
            "I8" | "INT8" => Some(Self::I8),
            "U32" | "UINT32" => Some(Self::U32),
            "U16" | "UINT16" => Some(Self::U16),
            "U8" | "UINT8" => Some(Self::U8),
            "BOOL" | "BOOLEAN" => Some(Self::Bool),
            _ => None,
        }
    }
}

// ============================================================================
// Tensor Info
// ============================================================================

/// Information about a tensor to be written
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Tensor name
    pub name: String,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: SafeTensorsDType,
    /// Raw tensor data bytes
    pub data: Vec<u8>,
}

impl TensorEntry {
    /// Create a new tensor entry
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: SafeTensorsDType,
        data: Vec<u8>,
    ) -> Result<Self> {
        let name = name.into();
        let expected_size = shape.iter().product::<usize>() * dtype.element_size();
        if data.len() != expected_size {
            bail!(
                "Tensor '{}' data size mismatch: expected {} bytes ({:?} shape, {} dtype), got {}",
                name,
                expected_size,
                shape,
                dtype.as_str(),
                data.len()
            );
        }
        Ok(Self {
            name,
            shape,
            dtype,
            data,
        })
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// SafeTensors Writer (Batch Mode)
// ============================================================================

/// SafeTensors writer for batch mode
///
/// Collects all tensors in memory, then writes them in one go.
/// Use `StreamingSafeTensorsWriter` for O(1) memory streaming.
pub struct SafeTensorsWriter {
    /// Tensors to write (in order)
    tensors: Vec<TensorEntry>,
    /// File-level metadata
    metadata: HashMap<String, String>,
}

impl SafeTensorsWriter {
    /// Create a new SafeTensors writer
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set a metadata key-value pair
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Add a tensor to be written
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: SafeTensorsDType,
        data: Vec<u8>,
    ) -> Result<()> {
        let entry = TensorEntry::new(name, shape, dtype, data)?;

        // Check for duplicate names
        if self.tensors.iter().any(|t| t.name == entry.name) {
            bail!("Duplicate tensor name: {}", entry.name);
        }

        self.tensors.push(entry);
        Ok(())
    }

    /// Get the number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get total data size in bytes
    pub fn total_data_size(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }

    /// Write all tensors to a file
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("Failed to create file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        // Build header JSON, padded so the data block starts 8-byte aligned
        let header_bytes = pad_header(self.build_header_json()?);
        let header_size = header_bytes.len() as u64;

        // Write header size (8 bytes, little-endian)
        writer.write_all(&header_size.to_le_bytes())?;

        // Write header JSON
        writer.write_all(&header_bytes)?;

        // Write tensor data in order
        for tensor in &self.tensors {
            writer.write_all(&tensor.data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Build the header JSON string
    fn build_header_json(&self) -> Result<String> {
        use std::fmt::Write;

        let mut json = String::with_capacity(4096);
        json.push('{');

        let mut first = true;

        // Write metadata if present
        if !self.metadata.is_empty() {
            json.push_str("\"__metadata__\":{");
            let mut meta_first = true;
            for (key, value) in &self.metadata {
                if !meta_first {
                    json.push(',');
                }
                write!(
                    json,
                    "\"{}\":\"{}\"",
                    escape_json_string(key),
                    escape_json_string(value)
                )?;
                meta_first = false;
            }
            json.push('}');
            first = false;
        }

        // Calculate offsets and write tensor entries
        let mut offset: usize = 0;
        for tensor in &self.tensors {
            if !first {
                json.push(',');
            }

            let start = offset;
            let end = offset + tensor.size_bytes();
            offset = end;

            // Write tensor entry
            write!(json, "\"{}\":{{", escape_json_string(&tensor.name))?;
            write!(json, "\"dtype\":\"{}\",", tensor.dtype.as_str())?;
            write!(
                json,
                "\"shape\":[{}],",
                tensor
                    .shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
            write!(json, "\"data_offsets\":[{},{}]", start, end)?;
            json.push('}');

            first = false;
        }

        json.push('}');
        Ok(json)
    }
}

impl Default for SafeTensorsWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Streaming SafeTensors Writer
// ============================================================================

/// Phase of the streaming writer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingPhase {
    /// Collecting tensor info (not yet writing)
    Collecting,
    /// Header has been written, waiting for data
    WritingData,
    /// All data written, finalized
    Finalized,
}

/// Streaming SafeTensors writer for O(1) memory usage
///
/// Use when you want to write tensors without loading all data into memory.
/// Requires registering all tensors first, then writing data in the same order.
///
/// # Usage
///
/// ```ignore
/// let mut writer = StreamingSafeTensorsWriter::new();
///
/// // Phase 1: Register all tensors
/// writer.register_tensor("weight1", vec![1024, 768], SafeTensorsDType::F16, 1024 * 768 * 2)?;
/// writer.register_tensor("weight2", vec![768, 768], SafeTensorsDType::F16, 768 * 768 * 2)?;
///
/// // Phase 2: Initialize file and write header
/// writer.initialize("model.safetensors")?;
///
/// // Phase 3: Write tensor data in order
/// writer.write_tensor_data(&weight1_bytes)?;
/// writer.write_tensor_data(&weight2_bytes)?;
///
/// // Phase 4: Finalize
/// writer.finalize()?;
/// ```
pub struct StreamingSafeTensorsWriter {
    /// Phase of the writer
    phase: StreamingPhase,
    /// File writer (None until initialized)
    file: Option<BufWriter<File>>,
    /// Registered tensor info (name, shape, dtype, expected_size)
    tensor_info: Vec<(String, Vec<usize>, SafeTensorsDType, usize)>,
    /// File-level metadata
    metadata: HashMap<String, String>,
    /// Current tensor index being written
    current_tensor: usize,
    /// Bytes written for current tensor
    bytes_written: usize,
}

impl StreamingSafeTensorsWriter {
    /// Create a new streaming writer
    pub fn new() -> Self {
        Self {
            phase: StreamingPhase::Collecting,
            file: None,
            tensor_info: Vec::new(),
            metadata: HashMap::new(),
            current_tensor: 0,
            bytes_written: 0,
        }
    }

    /// Set a metadata key-value pair
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        if self.phase != StreamingPhase::Collecting {
            log::warn!("Cannot set metadata after initialization");
            return;
        }
        self.metadata.insert(key.into(), value.into());
    }

    /// Register a tensor to be written
    ///
    /// Must be called before `initialize()`. Tensors must be written in the same order.
    pub fn register_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: SafeTensorsDType,
    ) -> Result<()> {
        if self.phase != StreamingPhase::Collecting {
            bail!("Cannot register tensors after initialization");
        }

        let name = name.into();

        // Check for duplicates
        if self.tensor_info.iter().any(|(n, _, _, _)| n == &name) {
            bail!("Duplicate tensor name: {}", name);
        }

        let size = shape.iter().product::<usize>() * dtype.element_size();
        self.tensor_info.push((name, shape, dtype, size));
        Ok(())
    }

    /// Initialize the file and write header
    ///
    /// Call this after registering all tensors, before writing any data.
    pub fn initialize<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        if self.phase != StreamingPhase::Collecting {
            bail!("Writer already initialized");
        }

        if self.tensor_info.is_empty() {
            bail!("No tensors registered");
        }

        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("Failed to create file: {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        // Build and write the header, padded so the data block starts 8-byte
        // aligned.
        let header_bytes = pad_header(self.build_header_json()?);
        let header_size = header_bytes.len() as u64;

        writer.write_all(&header_size.to_le_bytes())?;
        writer.write_all(&header_bytes)?;

        self.file = Some(writer);
        self.phase = StreamingPhase::WritingData;
        self.current_tensor = 0;
        self.bytes_written = 0;

        Ok(())
    }

    /// Write data for the current tensor
    ///
    /// Data must be written in the order tensors were registered.
    /// Can be called multiple times for partial writes.
    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        if self.phase != StreamingPhase::WritingData {
            bail!("Writer not in data writing phase");
        }

        if self.current_tensor >= self.tensor_info.len() {
            bail!("All tensors already written");
        }

        let (name, _, _, expected_size) = &self.tensor_info[self.current_tensor];
        let remaining = *expected_size - self.bytes_written;

        if data.len() > remaining {
            bail!(
                "Too much data for tensor '{}': expected {} more bytes, got {}",
                name,
                remaining,
                data.len()
            );
        }

        let file = self.file.as_mut().unwrap();
        file.write_all(data)?;

        self.bytes_written += data.len();

        // Check if this tensor is complete
        if self.bytes_written == *expected_size {
            self.current_tensor += 1;
            self.bytes_written = 0;
        }

        Ok(())
    }

    /// Write a complete tensor's data
    ///
    /// Convenience method that writes all data for one tensor.
    pub fn write_complete_tensor(&mut self, data: &[u8]) -> Result<()> {
        if self.phase != StreamingPhase::WritingData {
            bail!("Writer not in data writing phase");
        }

        if self.current_tensor >= self.tensor_info.len() {
            bail!("All tensors already written");
        }

        let (name, _, _, expected_size) = &self.tensor_info[self.current_tensor];

        if data.len() != *expected_size {
            bail!(
                "Data size mismatch for tensor '{}': expected {} bytes, got {}",
                name,
                expected_size,
                data.len()
            );
        }

        if self.bytes_written != 0 {
            bail!("Cannot write complete tensor when partial data already written");
        }

        let file = self.file.as_mut().unwrap();
        file.write_all(data)?;

        self.current_tensor += 1;
        self.bytes_written = 0;

        Ok(())
    }

    /// Finalize the file
    pub fn finalize(&mut self) -> Result<()> {
        if self.phase == StreamingPhase::Finalized {
            return Ok(());
        }

        if self.phase != StreamingPhase::WritingData {
            bail!("Cannot finalize: writer not initialized");
        }

        if self.current_tensor != self.tensor_info.len() {
            bail!(
                "Cannot finalize: {} tensors not yet written (current: {}/{})",
                self.tensor_info.len() - self.current_tensor,
                self.current_tensor,
                self.tensor_info.len()
            );
        }

        if let Some(mut file) = self.file.take() {
            file.flush()?;
        }

        self.phase = StreamingPhase::Finalized;
        Ok(())
    }

    /// Build the header JSON string
    fn build_header_json(&self) -> Result<String> {
        use std::fmt::Write;

        let mut json = String::with_capacity(4096);
        json.push('{');

        let mut first = true;

        // Write metadata if present
        if !self.metadata.is_empty() {
            json.push_str("\"__metadata__\":{");
            let mut meta_first = true;
            for (key, value) in &self.metadata {
                if !meta_first {
                    json.push(',');
                }
                write!(
                    json,
                    "\"{}\":\"{}\"",
                    escape_json_string(key),
                    escape_json_string(value)
                )?;
                meta_first = false;
            }
            json.push('}');
            first = false;
        }

        // Calculate offsets and write tensor entries
        let mut offset: usize = 0;
        for (name, shape, dtype, size) in &self.tensor_info {
            if !first {
                json.push(',');
            }

            let start = offset;
            let end = offset + size;
            offset = end;

            // Write tensor entry
            write!(json, "\"{}\":{{", escape_json_string(name))?;
            write!(json, "\"dtype\":\"{}\",", dtype.as_str())?;
            write!(
                json,
                "\"shape\":[{}],",
                shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
            write!(json, "\"data_offsets\":[{},{}]", start, end)?;
            json.push('}');

            first = false;
        }

        json.push('}');
        Ok(json)
    }

    /// Get number of registered tensors
    pub fn tensor_count(&self) -> usize {
        self.tensor_info.len()
    }

    /// Get number of tensors already written
    pub fn tensors_written(&self) -> usize {
        self.current_tensor
    }

    /// Check if all tensors have been written
    pub fn is_complete(&self) -> bool {
        self.current_tensor == self.tensor_info.len()
    }
}

impl Default for StreamingSafeTensorsWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StreamingSafeTensorsWriter {
    fn drop(&mut self) {
        if self.phase == StreamingPhase::WritingData {
            log::warn!("StreamingSafeTensorsWriter dropped without finalizing");
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Pad a header to a multiple of 8 bytes with trailing spaces.
///
/// The data block starts at `8 + header_len`, so an unpadded header leaves it
/// at an arbitrary offset. HuggingFace's own serializer pads to 8, and that is
/// what lets a reader memory-map the file and view an F32 or F64 tensor in
/// place instead of copying it to an aligned buffer. Trailing spaces are JSON
/// whitespace, so every parser still reads the header unchanged.
fn pad_header(json: String) -> Vec<u8> {
    let mut bytes = json.into_bytes();
    let padding = (8 - bytes.len() % 8) % 8;
    bytes.resize(bytes.len() + padding, b' ');
    bytes
}

/// Escape a string for JSON
fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_dtype_properties() {
        assert_eq!(SafeTensorsDType::F32.element_size(), 4);
        assert_eq!(SafeTensorsDType::F16.element_size(), 2);
        assert_eq!(SafeTensorsDType::BF16.element_size(), 2);
        assert_eq!(SafeTensorsDType::I64.element_size(), 8);
        assert_eq!(SafeTensorsDType::U8.element_size(), 1);

        assert_eq!(SafeTensorsDType::F32.as_str(), "F32");
        assert_eq!(SafeTensorsDType::BF16.as_str(), "BF16");
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(
            SafeTensorsDType::from_str("F32"),
            Some(SafeTensorsDType::F32)
        );
        assert_eq!(
            SafeTensorsDType::from_str("float32"),
            Some(SafeTensorsDType::F32)
        );
        assert_eq!(
            SafeTensorsDType::from_str("bf16"),
            Some(SafeTensorsDType::BF16)
        );
        assert_eq!(SafeTensorsDType::from_str("invalid"), None);
    }

    #[test]
    fn test_tensor_entry_validation() {
        // Valid tensor
        let data = vec![0u8; 4 * 4]; // 4 F32 elements
        let entry = TensorEntry::new("test", vec![2, 2], SafeTensorsDType::F32, data);
        assert!(entry.is_ok());

        // Invalid size
        let data = vec![0u8; 10]; // Wrong size
        let entry = TensorEntry::new("test", vec![2, 2], SafeTensorsDType::F32, data);
        assert!(entry.is_err());
    }

    #[test]
    fn test_writer_basic() {
        let mut writer = SafeTensorsWriter::new();
        writer.set_metadata("format", "pt");

        // Add a 2x2 F32 tensor
        let data = vec![0u8; 16]; // 4 elements * 4 bytes
        writer
            .add_tensor("weight", vec![2, 2], SafeTensorsDType::F32, data)
            .unwrap();

        assert_eq!(writer.tensor_count(), 1);
        assert_eq!(writer.total_data_size(), 16);

        // Write to temp file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_safetensors_basic.safetensors");
        writer.write(&temp_file).unwrap();

        // Verify file exists and has content
        let mut file = File::open(&temp_file).unwrap();
        let mut content = Vec::new();
        file.read_to_end(&mut content).unwrap();

        // Should have: 8 byte header size + header + 16 bytes data
        assert!(content.len() > 24);

        // First 8 bytes should be header size
        let header_size = u64::from_le_bytes(content[0..8].try_into().unwrap());
        assert!(header_size > 0);

        // Clean up
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_duplicate_tensor_rejection() {
        let mut writer = SafeTensorsWriter::new();

        let data = vec![0u8; 4];
        writer
            .add_tensor("weight", vec![1], SafeTensorsDType::F32, data.clone())
            .unwrap();

        // Adding duplicate should fail
        let result = writer.add_tensor("weight", vec![1], SafeTensorsDType::F32, data);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_writer() {
        let mut writer = StreamingSafeTensorsWriter::new();
        writer.set_metadata("format", "pt");

        // Register tensors
        writer
            .register_tensor("weight1", vec![2, 2], SafeTensorsDType::F32)
            .unwrap();
        writer
            .register_tensor("weight2", vec![3], SafeTensorsDType::F16)
            .unwrap();

        // Initialize file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_streaming_safetensors.safetensors");
        writer.initialize(&temp_file).unwrap();

        // Write tensor data
        let data1 = vec![0u8; 16]; // 4 * 4 bytes
        let data2 = vec![0u8; 6]; // 3 * 2 bytes

        writer.write_complete_tensor(&data1).unwrap();
        writer.write_complete_tensor(&data2).unwrap();

        assert!(writer.is_complete());
        writer.finalize().unwrap();

        // Verify file
        let mut file = File::open(&temp_file).unwrap();
        let mut content = Vec::new();
        file.read_to_end(&mut content).unwrap();

        // Should have header + 22 bytes of data
        let header_size = u64::from_le_bytes(content[0..8].try_into().unwrap()) as usize;
        assert_eq!(content.len(), 8 + header_size + 22);

        // Clean up
        std::fs::remove_file(&temp_file).ok();
    }

    /// The data block must start 8-byte aligned, or a reader cannot map an F32
    /// tensor in place. Both writers pad the header with spaces to get there.
    #[test]
    fn test_header_is_padded_to_eight_bytes() {
        let temp_dir = std::env::temp_dir();

        // Batch writer.
        let batch_file = temp_dir.join("test_st_align_batch.safetensors");
        let mut writer = SafeTensorsWriter::new();
        writer
            .add_tensor("w", vec![2], SafeTensorsDType::F32, vec![0u8; 8])
            .unwrap();
        writer.write(&batch_file).unwrap();

        // Streaming writer.
        let stream_file = temp_dir.join("test_st_align_stream.safetensors");
        let mut writer = StreamingSafeTensorsWriter::new();
        writer
            .register_tensor("w", vec![2], SafeTensorsDType::F32)
            .unwrap();
        writer.initialize(&stream_file).unwrap();
        writer.write_complete_tensor(&[0u8; 8]).unwrap();
        writer.finalize().unwrap();

        for path in [&batch_file, &stream_file] {
            let bytes = std::fs::read(path).unwrap();
            let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

            assert_eq!(
                header_size % 8,
                0,
                "{}: header of {} bytes is not 8-byte aligned",
                path.display(),
                header_size
            );

            // The padding is JSON whitespace, so the header still parses.
            let header: serde_json::Value =
                serde_json::from_slice(&bytes[8..8 + header_size]).unwrap();
            assert_eq!(header["w"]["dtype"], "F32");
            assert_eq!(header["w"]["data_offsets"][1], 8);

            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn test_escape_json_string() {
        assert_eq!(escape_json_string("hello"), "hello");
        assert_eq!(escape_json_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json_string("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json_string("path\\to\\file"), "path\\\\to\\\\file");
    }

    #[test]
    fn test_header_json_format() {
        let mut writer = SafeTensorsWriter::new();
        writer.set_metadata("format", "pt");

        let data = vec![0u8; 16];
        writer
            .add_tensor("test.weight", vec![2, 2], SafeTensorsDType::F32, data)
            .unwrap();

        let header = writer.build_header_json().unwrap();

        // Verify it's valid JSON
        assert!(header.starts_with('{'));
        assert!(header.ends_with('}'));
        assert!(header.contains("\"__metadata__\""));
        assert!(header.contains("\"test.weight\""));
        assert!(header.contains("\"dtype\":\"F32\""));
        assert!(header.contains("\"shape\":[2,2]"));
        assert!(header.contains("\"data_offsets\":[0,16]"));
    }
}
