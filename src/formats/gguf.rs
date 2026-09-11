//! GGUF Format Writer - llama.cpp compatible output
//!
//! Implements the GGUF v3 specification for writing quantized models that are
//! 100% compatible with llama.cpp and other GGUF-compatible inference engines.
//!
//! # GGUF File Structure
//! ```text
//! [MAGIC: 4 bytes] [VERSION: 4 bytes] [N_TENSORS: 8 bytes] [N_KV: 8 bytes]
//! [KV_PAIRS...]
//! [TENSOR_INFO...]
//! [PADDING to alignment]
//! [TENSOR_DATA...]
//! ```

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::quantization::QuantType;

// ============================================================================
// GGUF Constants
// ============================================================================

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32

/// Current GGUF version
pub const GGUF_VERSION: u32 = 3;

/// Default data alignment (power of 2)
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ============================================================================
// GGUF Value Types
// ============================================================================

/// GGUF metadata value types
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// ============================================================================
// GGML Tensor Types
// ============================================================================

/// GGML tensor data types (matches ggml_type enum)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // deprecated
    // Q4_3 = 5, // deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
}

impl GgmlType {
    /// Block size for this type (number of elements per block)
    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::Iq4Nl => 32,
            GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8K
            | GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq3Xxs
            | GgmlType::Iq1S
            | GgmlType::Iq3S
            | GgmlType::Iq2S
            | GgmlType::Iq4Xs
            | GgmlType::Iq1M => 256,
        }
    }

    /// Bytes per block for this type
    pub fn bytes_per_block(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Bf16 => 2,
            GgmlType::Q4_0 => 18, // 2 (scale) + 16 (nibbles)
            GgmlType::Q4_1 => 20, // 2 (scale) + 2 (min) + 16 (nibbles)
            GgmlType::Q5_0 => 22, // 2 (scale) + 4 (high bits) + 16 (low nibbles)
            GgmlType::Q5_1 => 24, // 2 (scale) + 2 (min) + 4 (high bits) + 16 (low nibbles)
            GgmlType::Q8_0 => 34, // 2 (scale) + 32 (int8s)
            GgmlType::Q8_1 => 40, // 4 (scale) + 4 (sum) + 32 (int8s)
            GgmlType::Q2K => 84,  // 16 (scales) + 64 (qs) + 2 (d) + 2 (dmin)
            GgmlType::Q3K => 110, // 32 (hmask) + 64 (qs) + 12 (scales) + 2 (d)
            GgmlType::Q4K => 144, // 2 + 2 + 12 + 128
            GgmlType::Q5K => 176, // 2 + 2 + 12 + 32 + 128
            GgmlType::Q6K => 210, // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d, last)
            GgmlType::Q8K => 292, // 4 + 256 + 32
            GgmlType::Iq2Xxs => 66,
            GgmlType::Iq2Xs => 74,
            GgmlType::Iq3Xxs => 98,
            GgmlType::Iq1S => 50,
            GgmlType::Iq4Nl => 18,
            GgmlType::Iq3S => 110,
            GgmlType::Iq2S => 82,
            GgmlType::Iq4Xs => 136,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::F64 => 8,
            GgmlType::Iq1M => 56,
        }
    }

    /// Calculate total bytes for n elements
    pub fn tensor_bytes(&self, n_elements: usize) -> usize {
        let block_size = self.block_size();
        let n_blocks = n_elements.div_ceil(block_size);
        n_blocks * self.bytes_per_block()
    }

    /// Convert from compressr QuantType
    pub fn from_quant_type(qt: QuantType) -> Self {
        match qt {
            QuantType::Q4_0 => GgmlType::Q4_0,
            QuantType::Q4_1 => GgmlType::Q4_1,
            QuantType::Q5_0 => GgmlType::Q5_0,
            QuantType::Q5_1 => GgmlType::Q5_1,
            QuantType::Q8_0 => GgmlType::Q8_0,
            QuantType::Q8_1 => GgmlType::Q8_1,
            QuantType::Q2K => GgmlType::Q2K,
            QuantType::Q3K => GgmlType::Q3K,
            QuantType::Q4K => GgmlType::Q4K,
            QuantType::Q5K => GgmlType::Q5K,
            QuantType::Q6K => GgmlType::Q6K,
            QuantType::IQ4NL => GgmlType::Iq4Nl,
            QuantType::FP8 => GgmlType::F16, // FP8 stored as F16 in GGUF (closest supported)
            QuantType::NF4 => GgmlType::Q4_0, // NF4 approximated as Q4_0 in GGUF
            QuantType::FP4 => GgmlType::Q4_0, // FP4 approximated as Q4_0 in GGUF
            QuantType::GPTQ => GgmlType::Q4_0, // GPTQ INT4 → Q4_0
            QuantType::AWQ => GgmlType::Q4_0, // AWQ INT4 → Q4_0
        }
    }
}

// ============================================================================
// GGUF Metadata Value
// ============================================================================

/// A GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// A GGUF array value
#[derive(Debug, Clone)]
pub enum GgufArray {
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Uint16(Vec<u16>),
    Int16(Vec<i16>),
    Uint32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Uint64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

impl GgufValue {
    fn value_type(&self) -> GgufValueType {
        match self {
            GgufValue::Uint8(_) => GgufValueType::Uint8,
            GgufValue::Int8(_) => GgufValueType::Int8,
            GgufValue::Uint16(_) => GgufValueType::Uint16,
            GgufValue::Int16(_) => GgufValueType::Int16,
            GgufValue::Uint32(_) => GgufValueType::Uint32,
            GgufValue::Int32(_) => GgufValueType::Int32,
            GgufValue::Float32(_) => GgufValueType::Float32,
            GgufValue::Bool(_) => GgufValueType::Bool,
            GgufValue::String(_) => GgufValueType::String,
            GgufValue::Array(_) => GgufValueType::Array,
            GgufValue::Uint64(_) => GgufValueType::Uint64,
            GgufValue::Int64(_) => GgufValueType::Int64,
            GgufValue::Float64(_) => GgufValueType::Float64,
        }
    }
}

impl GgufArray {
    fn element_type(&self) -> GgufValueType {
        match self {
            GgufArray::Uint8(_) => GgufValueType::Uint8,
            GgufArray::Int8(_) => GgufValueType::Int8,
            GgufArray::Uint16(_) => GgufValueType::Uint16,
            GgufArray::Int16(_) => GgufValueType::Int16,
            GgufArray::Uint32(_) => GgufValueType::Uint32,
            GgufArray::Int32(_) => GgufValueType::Int32,
            GgufArray::Float32(_) => GgufValueType::Float32,
            GgufArray::Bool(_) => GgufValueType::Bool,
            GgufArray::String(_) => GgufValueType::String,
            GgufArray::Uint64(_) => GgufValueType::Uint64,
            GgufArray::Int64(_) => GgufValueType::Int64,
            GgufArray::Float64(_) => GgufValueType::Float64,
        }
    }

    fn len(&self) -> usize {
        match self {
            GgufArray::Uint8(v) => v.len(),
            GgufArray::Int8(v) => v.len(),
            GgufArray::Uint16(v) => v.len(),
            GgufArray::Int16(v) => v.len(),
            GgufArray::Uint32(v) => v.len(),
            GgufArray::Int32(v) => v.len(),
            GgufArray::Float32(v) => v.len(),
            GgufArray::Bool(v) => v.len(),
            GgufArray::String(v) => v.len(),
            GgufArray::Uint64(v) => v.len(),
            GgufArray::Int64(v) => v.len(),
            GgufArray::Float64(v) => v.len(),
        }
    }
}

// ============================================================================
// Tensor Info
// ============================================================================

/// Information about a tensor to be written
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Tensor shape (in standard order, will be reversed when written)
    pub shape: Vec<usize>,
    /// Tensor data type
    pub dtype: GgmlType,
    /// Tensor data (packed/quantized bytes)
    pub data: Vec<u8>,
}

impl GgufTensorInfo {
    /// Create new tensor info
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: GgmlType, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            data,
        }
    }

    /// Number of elements in this tensor
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Expected byte size for this tensor
    pub fn expected_bytes(&self) -> usize {
        self.dtype.tensor_bytes(self.n_elements())
    }

    /// Validate tensor data
    pub fn validate(&self) -> Result<()> {
        let expected = self.expected_bytes();
        if self.data.len() != expected {
            bail!(
                "Tensor '{}' has {} bytes but expected {} for dtype {:?} with {} elements",
                self.name,
                self.data.len(),
                expected,
                self.dtype,
                self.n_elements()
            );
        }

        // A block-quantized row is packed as whole blocks, so the row length
        // must be a multiple of the block size. Dimensions are written in
        // reverse, so GGUF's row length `ne[0]` is the LAST dimension of the
        // standard-order shape held here.
        if let Some(&row_len) = self.shape.last() {
            let block_size = self.dtype.block_size();
            if !row_len.is_multiple_of(block_size) {
                bail!(
                    "Tensor '{}' row length {} is not divisible by block size {} for dtype {:?}",
                    self.name,
                    row_len,
                    block_size,
                    self.dtype
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// GGUF Writer
// ============================================================================

/// GGUF file writer with streaming support
pub struct GgufWriter {
    /// Output file handle
    file: Option<BufWriter<File>>,
    /// Data alignment
    alignment: usize,
    /// Metadata key-value pairs
    metadata: Vec<(String, GgufValue)>,
    /// Tensor information (collected before writing)
    tensors: Vec<GgufTensorInfo>,
    /// Whether header has been written
    header_written: bool,
    /// Current offset in data section
    data_offset: u64,
}

impl GgufWriter {
    /// Create a new GGUF writer
    pub fn new() -> Self {
        Self {
            file: None,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            metadata: Vec::new(),
            tensors: Vec::new(),
            header_written: false,
            data_offset: 0,
        }
    }

    /// Set data alignment (must be power of 2)
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        self.alignment = alignment;
        self
    }

    /// Add a metadata key-value pair
    pub fn add_metadata(&mut self, key: impl Into<String>, value: GgufValue) {
        self.metadata.push((key.into(), value));
    }

    /// Add string metadata
    pub fn add_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.add_metadata(key, GgufValue::String(value.into()));
    }

    /// Add uint32 metadata
    pub fn add_u32(&mut self, key: impl Into<String>, value: u32) {
        self.add_metadata(key, GgufValue::Uint32(value));
    }

    /// Add int32 metadata
    pub fn add_i32(&mut self, key: impl Into<String>, value: i32) {
        self.add_metadata(key, GgufValue::Int32(value));
    }

    /// Add float32 metadata
    pub fn add_f32(&mut self, key: impl Into<String>, value: f32) {
        self.add_metadata(key, GgufValue::Float32(value));
    }

    /// Add uint64 metadata
    pub fn add_u64(&mut self, key: impl Into<String>, value: u64) {
        self.add_metadata(key, GgufValue::Uint64(value));
    }

    /// Add a tensor to be written
    pub fn add_tensor(&mut self, tensor: GgufTensorInfo) -> Result<()> {
        tensor.validate()?;

        // Check for duplicate names
        if self.tensors.iter().any(|t| t.name == tensor.name) {
            bail!("Duplicate tensor name: {}", tensor.name);
        }

        self.tensors.push(tensor);
        Ok(())
    }

    /// Set standard model metadata
    pub fn set_model_metadata(&mut self, architecture: &str, name: &str, file_type: u32) {
        self.add_string("general.architecture", architecture);
        self.add_string("general.name", name);
        self.add_u32("general.file_type", file_type);
        self.add_u32("general.quantization_version", 2);

        // Add alignment if non-default
        if self.alignment != GGUF_DEFAULT_ALIGNMENT {
            self.add_u32("general.alignment", self.alignment as u32);
        }
    }

    /// Set LLaMA-specific metadata
    ///
    /// The GGUF spec names these nine fields individually; bundling them into a
    /// struct that exists only to satisfy a lint would obscure that mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn set_llama_metadata(
        &mut self,
        vocab_size: u32,
        context_length: u32,
        embedding_length: u32,
        block_count: u32,
        head_count: u32,
        head_count_kv: u32,
        rope_freq_base: f32,
        rms_norm_eps: f32,
    ) {
        self.add_u32("llama.vocab_size", vocab_size);
        self.add_u32("llama.context_length", context_length);
        self.add_u32("llama.embedding_length", embedding_length);
        self.add_u32("llama.block_count", block_count);
        self.add_u32("llama.attention.head_count", head_count);
        self.add_u32("llama.attention.head_count_kv", head_count_kv);
        self.add_f32("llama.rope.freq_base", rope_freq_base);
        self.add_f32("llama.attention.layer_norm_rms_epsilon", rms_norm_eps);
    }

    /// Write the complete GGUF file
    pub fn write<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create GGUF file: {:?}", path.as_ref()))?;
        self.file = Some(BufWriter::new(file));

        // Write header
        self.write_header()?;

        // Write metadata
        self.write_metadata()?;

        // Write tensor info
        self.write_tensor_info()?;

        // Write padding to alignment boundary
        self.write_alignment_padding()?;

        // Write tensor data
        self.write_tensor_data()?;

        // Flush and close
        if let Some(ref mut file) = self.file {
            file.flush()?;
        }
        self.file = None;

        Ok(())
    }

    /// Write GGUF header
    fn write_header(&mut self) -> Result<()> {
        let file = self.file.as_mut().unwrap();

        // Magic
        file.write_all(&GGUF_MAGIC.to_le_bytes())?;

        // Version
        file.write_all(&GGUF_VERSION.to_le_bytes())?;

        // Tensor count
        file.write_all(&(self.tensors.len() as u64).to_le_bytes())?;

        // KV count
        file.write_all(&(self.metadata.len() as u64).to_le_bytes())?;

        self.header_written = true;
        Ok(())
    }

    /// Write all metadata key-value pairs
    fn write_metadata(&mut self) -> Result<()> {
        for (key, value) in self.metadata.clone() {
            self.write_kv_pair(&key, &value)?;
        }
        Ok(())
    }

    /// Write a single key-value pair
    fn write_kv_pair(&mut self, key: &str, value: &GgufValue) -> Result<()> {
        // Write key (length-prefixed string)
        self.write_string_raw(key)?;

        // Write value type
        let file = self.file.as_mut().unwrap();
        file.write_all(&(value.value_type() as u32).to_le_bytes())?;

        // Write value
        self.write_value(value)?;

        Ok(())
    }

    /// Write a GGUF value
    fn write_value(&mut self, value: &GgufValue) -> Result<()> {
        let file = self.file.as_mut().unwrap();

        match value {
            GgufValue::Uint8(v) => file.write_all(&[*v])?,
            GgufValue::Int8(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Uint16(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Int16(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Uint32(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Int32(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Float32(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Bool(v) => file.write_all(&[if *v { 1u8 } else { 0u8 }])?,
            GgufValue::String(s) => {
                let s = s.clone();
                self.write_string_raw(&s)?;
            }
            GgufValue::Uint64(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Int64(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Float64(v) => file.write_all(&v.to_le_bytes())?,
            GgufValue::Array(arr) => {
                let arr = arr.clone();
                self.write_array(&arr)?;
            }
        }

        Ok(())
    }

    /// Write a GGUF array
    fn write_array(&mut self, array: &GgufArray) -> Result<()> {
        let file = self.file.as_mut().unwrap();

        // Write element type
        file.write_all(&(array.element_type() as u32).to_le_bytes())?;

        // Write length
        file.write_all(&(array.len() as u64).to_le_bytes())?;

        // Write elements
        match array {
            GgufArray::Uint8(v) => {
                for x in v {
                    file.write_all(&[*x])?;
                }
            }
            GgufArray::Int8(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Uint16(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Int16(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Uint32(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Int32(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Float32(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Bool(v) => {
                for x in v {
                    file.write_all(&[if *x { 1u8 } else { 0u8 }])?;
                }
            }
            GgufArray::String(v) => {
                for s in v {
                    self.write_string_raw(s)?;
                }
            }
            GgufArray::Uint64(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Int64(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Float64(v) => {
                for x in v {
                    file.write_all(&x.to_le_bytes())?;
                }
            }
        }

        Ok(())
    }

    /// Write a length-prefixed string (no null terminator)
    fn write_string_raw(&mut self, s: &str) -> Result<()> {
        let file = self.file.as_mut().unwrap();
        let bytes = s.as_bytes();
        file.write_all(&(bytes.len() as u64).to_le_bytes())?;
        file.write_all(bytes)?;
        Ok(())
    }

    /// Write tensor info section
    fn write_tensor_info(&mut self) -> Result<()> {
        // Calculate offsets for each tensor
        let mut offset: u64 = 0;
        let mut tensor_data: Vec<(String, Vec<usize>, GgmlType, u64)> =
            Vec::with_capacity(self.tensors.len());

        for tensor in &self.tensors {
            tensor_data.push((
                tensor.name.clone(),
                tensor.shape.clone(),
                tensor.dtype,
                offset,
            ));
            let size = tensor.data.len() as u64;
            let padded_size = Self::align_offset(size, self.alignment as u64);
            offset = offset.checked_add(padded_size).unwrap();
        }

        // Write tensor info entries
        for (name, shape, dtype, tensor_offset) in tensor_data {
            self.write_tensor_info_entry(&name, &shape, dtype, tensor_offset)?;
        }

        Ok(())
    }

    /// Write a single tensor info entry
    fn write_tensor_info_entry(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: GgmlType,
        offset: u64,
    ) -> Result<()> {
        // Write name (length-prefixed string)
        self.write_string_raw(name)?;

        let file = self.file.as_mut().unwrap();

        // Write n_dims
        file.write_all(&(shape.len() as u32).to_le_bytes())?;

        // Write dimensions in REVERSE order
        for dim in shape.iter().rev() {
            file.write_all(&(*dim as i64).to_le_bytes())?;
        }

        // Write tensor type
        file.write_all(&(dtype as u32).to_le_bytes())?;

        // Write offset (relative to start of data section)
        file.write_all(&offset.to_le_bytes())?;

        Ok(())
    }

    /// Write alignment padding before data section
    fn write_alignment_padding(&mut self) -> Result<()> {
        let file = self.file.as_mut().unwrap();
        let current_pos = file.stream_position()?;
        let aligned_pos = Self::align_offset(current_pos, self.alignment as u64);
        let padding = aligned_pos - current_pos;

        if padding > 0 {
            file.write_all(&vec![0u8; padding as usize])?;
        }

        Ok(())
    }

    /// Write all tensor data
    fn write_tensor_data(&mut self) -> Result<()> {
        for tensor in std::mem::take(&mut self.tensors) {
            self.write_single_tensor_data(&tensor)?;
        }
        Ok(())
    }

    /// Write a single tensor's data with padding
    fn write_single_tensor_data(&mut self, tensor: &GgufTensorInfo) -> Result<()> {
        let file = self.file.as_mut().unwrap();

        // Write tensor data
        file.write_all(&tensor.data)?;

        // Calculate padding needed
        let size = tensor.data.len() as u64;
        let padded_size = Self::align_offset(size, self.alignment as u64);
        let padding = padded_size - size;

        if padding > 0 {
            file.write_all(&vec![0u8; padding as usize])?;
        }

        self.data_offset += padded_size;

        Ok(())
    }

    /// Align offset to boundary
    fn align_offset(offset: u64, alignment: u64) -> u64 {
        offset.div_ceil(alignment) * alignment
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Streaming GGUF Writer (for large models)
// ============================================================================

/// Streaming GGUF writer for O(1) memory usage
///
/// This writer uses a two-pass approach:
/// 1. First pass: collect tensor metadata and write header + info
/// 2. Second pass: stream tensor data directly to disk
pub struct StreamingGgufWriter {
    /// Output path
    path: std::path::PathBuf,
    /// Data alignment
    alignment: usize,
    /// Metadata key-value pairs
    metadata: Vec<(String, GgufValue)>,
    /// Tensor metadata (without data)
    tensor_meta: Vec<TensorMeta>,
    /// File handle (only during data writing phase)
    file: Option<BufWriter<File>>,
    /// Current phase
    phase: WriterPhase,
    /// Data section start offset
    data_section_start: u64,
    /// Current tensor index being written
    current_tensor: usize,
}

/// Tensor metadata (without data)
#[derive(Debug, Clone)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    dtype: GgmlType,
    data_size: usize,
    offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterPhase {
    /// Collecting metadata and tensor info
    Collecting,
    /// Writing header and tensor info
    WritingHeader,
    /// Writing tensor data
    WritingData,
    /// Finalized
    Finalized,
}

impl StreamingGgufWriter {
    /// Create a new streaming GGUF writer
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            metadata: Vec::new(),
            tensor_meta: Vec::new(),
            file: None,
            phase: WriterPhase::Collecting,
            data_section_start: 0,
            current_tensor: 0,
        }
    }

    /// Set data alignment
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two());
        self.alignment = alignment;
        self
    }

    /// Add metadata (same API as GgufWriter)
    pub fn add_metadata(&mut self, key: impl Into<String>, value: GgufValue) -> Result<()> {
        if self.phase != WriterPhase::Collecting {
            bail!(
                "add_metadata called in phase {:?}; metadata is fixed once the header is \
                 written, so metadata added after that point would never reach the file",
                self.phase
            );
        }
        self.metadata.push((key.into(), value));
        Ok(())
    }

    /// Add string metadata
    pub fn add_string(&mut self, key: impl Into<String>, value: impl Into<String>) -> Result<()> {
        self.add_metadata(key, GgufValue::String(value.into()))
    }

    /// Add uint32 metadata
    pub fn add_u32(&mut self, key: impl Into<String>, value: u32) -> Result<()> {
        self.add_metadata(key, GgufValue::Uint32(value))
    }

    /// Add float32 metadata
    pub fn add_f32(&mut self, key: impl Into<String>, value: f32) -> Result<()> {
        self.add_metadata(key, GgufValue::Float32(value))
    }

    /// Register a tensor (metadata only, data will be written later)
    pub fn register_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: GgmlType,
        data_size: usize,
    ) -> Result<()> {
        if self.phase != WriterPhase::Collecting {
            bail!(
                "register_tensor called in phase {:?}: every tensor must be registered before \
                 the first write, because tensor offsets are computed from the registered byte \
                 counts when the header is emitted",
                self.phase
            );
        }

        let name = name.into();

        // Validate
        let n_elements: usize = shape.iter().product();
        let expected_size = dtype.tensor_bytes(n_elements);
        if data_size != expected_size {
            bail!(
                "Tensor '{}' data_size {} != expected {} for shape {:?} dtype {:?}",
                name,
                data_size,
                expected_size,
                shape,
                dtype
            );
        }

        // Check row length divisibility. Dimensions are written in reverse, so
        // GGUF's row length `ne[0]` is the LAST dimension of `shape`.
        if let Some(&row_len) = shape.last() {
            let block_size = dtype.block_size();
            if !row_len.is_multiple_of(block_size) {
                bail!(
                    "Tensor '{}' row length {} not divisible by block size {}",
                    name,
                    row_len,
                    block_size
                );
            }
        }

        // Check for duplicates
        if self.tensor_meta.iter().any(|t| t.name == name) {
            bail!("Duplicate tensor name: {}", name);
        }

        self.tensor_meta.push(TensorMeta {
            name,
            shape,
            dtype,
            data_size,
            offset: 0, // Will be computed in begin_data_write
        });

        Ok(())
    }

    /// Begin writing (writes header and tensor info, prepares for data streaming)
    pub fn begin_data_write(&mut self) -> Result<()> {
        if self.phase != WriterPhase::Collecting {
            bail!(
                "begin_data_write called in phase {:?}; the header is written exactly once, \
                 after every tensor is registered and before the first byte of data",
                self.phase
            );
        }
        self.phase = WriterPhase::WritingHeader;

        // Calculate tensor offsets
        let mut offset: u64 = 0;
        for tensor in &mut self.tensor_meta {
            tensor.offset = offset;
            let padded_size = Self::align(tensor.data_size as u64, self.alignment as u64);
            offset += padded_size;
        }

        // Open file
        let file = File::create(&self.path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(&GGUF_MAGIC.to_le_bytes())?;
        writer.write_all(&GGUF_VERSION.to_le_bytes())?;
        writer.write_all(&(self.tensor_meta.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.metadata.len() as u64).to_le_bytes())?;

        // Write metadata
        for (key, value) in &self.metadata {
            Self::write_kv_to(&mut writer, key, value)?;
        }

        // Write tensor info
        for tensor in &self.tensor_meta {
            Self::write_tensor_info_to(&mut writer, tensor)?;
        }

        // Write alignment padding
        let current_pos = writer.stream_position()?;
        let aligned_pos = Self::align(current_pos, self.alignment as u64);
        let padding = aligned_pos - current_pos;
        if padding > 0 {
            writer.write_all(&vec![0u8; padding as usize])?;
        }

        self.data_section_start = writer.stream_position()?;
        self.file = Some(writer);
        self.phase = WriterPhase::WritingData;
        self.current_tensor = 0;

        Ok(())
    }

    /// Write tensor data (must be called in order of registration)
    ///
    /// The first call closes registration and writes the header, the same way
    /// [`Self::finalize`] does for a file with no tensors. Tensor offsets are
    /// computed from the registered byte counts at that moment, so registering
    /// a tensor after the first write would corrupt every offset already
    /// written — hence the header is emitted here rather than left to the
    /// caller to remember.
    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        if self.phase == WriterPhase::Collecting {
            self.begin_data_write()?;
        }
        if self.phase != WriterPhase::WritingData {
            bail!(
                "write_tensor_data called in phase {:?}, expected WritingData",
                self.phase
            );
        }

        if self.current_tensor >= self.tensor_meta.len() {
            bail!("All tensors already written");
        }

        let tensor = &self.tensor_meta[self.current_tensor];
        if data.len() != tensor.data_size {
            bail!(
                "Tensor '{}' data size {} != expected {}",
                tensor.name,
                data.len(),
                tensor.data_size
            );
        }

        // `begin_data_write` is the only path into `WritingData` and it always
        // installs the file, so this is unreachable rather than a real fallback.
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("GGUF writer is in WritingData with no open file"))?;

        // Write data
        file.write_all(data)?;

        // Write padding
        let padded_size = Self::align(tensor.data_size as u64, self.alignment as u64);
        let padding = padded_size - tensor.data_size as u64;
        if padding > 0 {
            file.write_all(&vec![0u8; padding as usize])?;
        }

        self.current_tensor += 1;
        Ok(())
    }

    /// Finalize the file
    pub fn finalize(&mut self) -> Result<()> {
        if self.phase == WriterPhase::Collecting {
            // No tensors were written, just write empty file
            self.begin_data_write()?;
        }

        if self.phase != WriterPhase::WritingData {
            bail!(
                "finalize called in phase {:?}, expected WritingData",
                self.phase
            );
        }

        if self.current_tensor != self.tensor_meta.len() {
            bail!(
                "Not all tensors written: {} of {}",
                self.current_tensor,
                self.tensor_meta.len()
            );
        }

        if let Some(ref mut file) = self.file {
            file.flush()?;
        }
        self.file = None;
        self.phase = WriterPhase::Finalized;

        Ok(())
    }

    // Helper functions

    fn align(offset: u64, alignment: u64) -> u64 {
        offset.div_ceil(alignment) * alignment
    }

    fn write_string_to(writer: &mut BufWriter<File>, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
        writer.write_all(bytes)?;
        Ok(())
    }

    fn write_kv_to(writer: &mut BufWriter<File>, key: &str, value: &GgufValue) -> Result<()> {
        Self::write_string_to(writer, key)?;
        writer.write_all(&(value.value_type() as u32).to_le_bytes())?;
        Self::write_value_to(writer, value)?;
        Ok(())
    }

    fn write_value_to(writer: &mut BufWriter<File>, value: &GgufValue) -> Result<()> {
        match value {
            GgufValue::Uint8(v) => writer.write_all(&[*v])?,
            GgufValue::Int8(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Uint16(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int16(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Uint32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float32(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Bool(v) => writer.write_all(&[if *v { 1u8 } else { 0u8 }])?,
            GgufValue::String(s) => Self::write_string_to(writer, s)?,
            GgufValue::Uint64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Int64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Float64(v) => writer.write_all(&v.to_le_bytes())?,
            GgufValue::Array(arr) => Self::write_array_to(writer, arr)?,
        }
        Ok(())
    }

    fn write_array_to(writer: &mut BufWriter<File>, array: &GgufArray) -> Result<()> {
        writer.write_all(&(array.element_type() as u32).to_le_bytes())?;
        writer.write_all(&(array.len() as u64).to_le_bytes())?;

        match array {
            GgufArray::Uint8(v) => {
                for x in v {
                    writer.write_all(&[*x])?;
                }
            }
            GgufArray::Int8(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Uint16(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Int16(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Uint32(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Int32(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Float32(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Bool(v) => {
                for x in v {
                    writer.write_all(&[if *x { 1u8 } else { 0u8 }])?;
                }
            }
            GgufArray::String(v) => {
                for s in v {
                    Self::write_string_to(writer, s)?;
                }
            }
            GgufArray::Uint64(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Int64(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
            GgufArray::Float64(v) => {
                for x in v {
                    writer.write_all(&x.to_le_bytes())?;
                }
            }
        }

        Ok(())
    }

    fn write_tensor_info_to(writer: &mut BufWriter<File>, tensor: &TensorMeta) -> Result<()> {
        Self::write_string_to(writer, &tensor.name)?;
        writer.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;

        // Dimensions in REVERSE order
        for dim in tensor.shape.iter().rev() {
            writer.write_all(&(*dim as i64).to_le_bytes())?;
        }

        writer.write_all(&(tensor.dtype as u32).to_le_bytes())?;
        writer.write_all(&tensor.offset.to_le_bytes())?;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F32.bytes_per_block(), 4);

        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.bytes_per_block(), 18);

        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q4K.bytes_per_block(), 144);
    }

    #[test]
    fn test_tensor_bytes() {
        // 1024 f32 elements = 4096 bytes
        assert_eq!(GgmlType::F32.tensor_bytes(1024), 4096);

        // 1024 Q4_0 elements = 32 blocks * 18 bytes = 576 bytes
        assert_eq!(GgmlType::Q4_0.tensor_bytes(1024), 576);

        // 256 Q4_K elements = 1 block * 144 bytes = 144 bytes
        assert_eq!(GgmlType::Q4K.tensor_bytes(256), 144);
    }

    #[test]
    fn test_row_length_check_reads_the_last_dimension() {
        // [8, 2048]: dimensions are written reversed, so the row is 2048
        // elements — a whole number of 256-element Q4_K blocks.
        let ok = GgufTensorInfo::new(
            "embed.weight",
            vec![8, 2048],
            GgmlType::Q4K,
            vec![0u8; GgmlType::Q4K.tensor_bytes(8 * 2048)],
        );
        assert!(ok.validate().is_ok());

        // [256, 64]: 256 is divisible by 256, but the ROW is 64 elements and
        // cannot be packed into whole blocks.
        let bad = GgufTensorInfo::new(
            "proj.weight",
            vec![256, 64],
            GgmlType::Q4K,
            vec![0u8; GgmlType::Q4K.tensor_bytes(256 * 64)],
        );
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("row length 64"), "{}", err);
    }

    #[test]
    fn test_register_tensor_rejects_a_non_block_aligned_row() {
        let mut writer = StreamingGgufWriter::new(std::env::temp_dir().join("test_row_len.gguf"));
        let err = writer
            .register_tensor(
                "proj.weight",
                vec![256, 64],
                GgmlType::Q4K,
                GgmlType::Q4K.tensor_bytes(256 * 64),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("row length 64"), "{}", err);

        // The row of a [73448, 2048] embedding is 2048 elements, so it packs.
        writer
            .register_tensor(
                "base_lm.embed_tokens.weight",
                vec![73448, 2048],
                GgmlType::Q4K,
                GgmlType::Q4K.tensor_bytes(73448 * 2048),
            )
            .unwrap();
    }

    #[test]
    fn test_gguf_writer_basic() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_basic.gguf");

        let mut writer = GgufWriter::new();
        writer.set_model_metadata("llama", "Test Model", 2);
        writer.add_u32("test.value", 42);

        // Create a simple F32 tensor
        let data = vec![0u8; 16]; // 4 f32 values
        let tensor = GgufTensorInfo::new("test.weight", vec![4], GgmlType::F32, data);
        writer.add_tensor(tensor).unwrap();

        writer.write(&path).unwrap();

        // Verify file exists and has content
        let mut file = File::open(&path).unwrap();
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(u32::from_le_bytes(magic), GGUF_MAGIC);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_streaming_writer() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_streaming.gguf");

        let mut writer = StreamingGgufWriter::new(&path);
        writer.add_string("general.architecture", "llama").unwrap();
        writer.add_string("general.name", "Streaming Test").unwrap();
        writer.add_u32("general.file_type", 2).unwrap();

        // Register tensors
        writer
            .register_tensor("layer.0.weight", vec![32], GgmlType::Q4_0, 18)
            .unwrap();
        writer
            .register_tensor("layer.1.weight", vec![32], GgmlType::Q4_0, 18)
            .unwrap();

        // Begin writing
        writer.begin_data_write().unwrap();

        // Write tensor data
        writer.write_tensor_data(&[0u8; 18]).unwrap();
        writer.write_tensor_data(&[0u8; 18]).unwrap();

        // Finalize
        writer.finalize().unwrap();

        // Verify
        let mut file = File::open(&path).unwrap();
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(u32::from_le_bytes(magic), GGUF_MAGIC);

        std::fs::remove_file(&path).ok();
    }
}
