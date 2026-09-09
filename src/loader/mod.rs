//! Model loading, one tensor at a time
//!
//! This module provides a unified interface for loading models from various formats.
//! It uses boostr's SafeTensors, GGUF, TCF and PyTorch readers under the hood.
//!
//! # Memory
//!
//! Opening a SafeTensors, GGUF or TCF model reads its directory only, so a
//! checkpoint of any size opens cheaply. A PyTorch state dict is the
//! exception: boostr's handle answers keys but no shape or element type, so
//! opening one materializes each tensor once to record its shape and drops it
//! again — one tensor at a time, and the file is read twice over a
//! conversion. [`LoadedModel::read_tensor_bytes`] then seeks and reads ONE tensor
//! into an owned buffer per call — it is not a memory map and the bytes are not
//! shared with the file. Peak resident memory over a conversion is therefore
//! roughly the largest tensor plus its converted copy, not the model size.

pub mod tcf;
pub mod torch;

use crate::formats::shard::read_index_shard_files;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub use tcf::{TcfFileFacts, TcfModel, TcfModuleFacts, TcfTensorFacts};
pub use torch::TorchModel;

/// Name of the index that maps each tensor onto the shard holding it.
pub const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

// Re-export boostr's model types
pub use boostr::format::{GgmlType, Gguf, GgufTensorInfo, SafeTensors, TensorInfo};
pub use boostr::DType;

/// Universal model loader supporting multiple formats
pub struct ModelLoader {
    /// Path to the model (file or directory)
    path: String,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
        }
    }

    /// Get the path being loaded
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Auto-detect the model format
    ///
    /// A single file is identified by its CONTENT where the format has a
    /// magic: an extension is a convention a download or a rename can break,
    /// and reading eight bytes settles it. TCF opens with `TCF\0\0\0\0\0`
    /// (Section 4), and a PyTorch archive is a ZIP container. SafeTensors
    /// opens with a header length, which is any eight bytes, so it keeps its
    /// extension, and GGUF keeps the extension it always used here.
    pub fn detect_format(&self) -> Result<crate::InputFormat> {
        let path = Path::new(&self.path);

        if path.is_file() {
            let magic = read_magic(path)?;
            if magic == Some(tcf_core::MAGIC) {
                return Ok(crate::InputFormat::Tcf);
            }
            // A SafeTensors file opens with a header length, so any eight
            // bytes are legal there and the ZIP magic is only trusted for a
            // file that does not already name itself SafeTensors or GGUF.
            let names_another_format = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    ext.eq_ignore_ascii_case("safetensors") || ext.eq_ignore_ascii_case("gguf")
                })
                .unwrap_or(false);
            if !names_another_format && magic.map(|m| m.starts_with(&ZIP_MAGIC)) == Some(true) {
                return Ok(crate::InputFormat::PyTorch);
            }
        }

        // Check for GGUF files first (specific extension)
        if let Some(ext) = path.extension() {
            if ext == "gguf" {
                return Ok(crate::InputFormat::Gguf);
            }
        }

        if path.is_dir() {
            // Check for safetensors files
            if path.join("model.safetensors").exists() || path.join(SAFETENSORS_INDEX_FILE).exists()
            {
                return Ok(crate::InputFormat::Safetensors);
            }

            // Check for sharded safetensors
            let has_shards = std::fs::read_dir(path)
                .ok()
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "safetensors")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

            if has_shards {
                return Ok(crate::InputFormat::Safetensors);
            }

            // Check for PyTorch files
            if path.join("pytorch_model.bin").exists()
                || path.join("pytorch_model.bin.index.json").exists()
            {
                return Ok(crate::InputFormat::PyTorch);
            }
        } else if let Some(ext) = path.extension() {
            match ext.to_str() {
                Some("safetensors") => return Ok(crate::InputFormat::Safetensors),
                Some("bin") | Some("pt") | Some("pth") => return Ok(crate::InputFormat::PyTorch),
                Some("onnx") => return Ok(crate::InputFormat::Onnx),
                // The magic check above already refused this file's content,
                // so the extension is the only thing claiming it is TCF.
                Some("tcf") => anyhow::bail!(
                    "{} is named .tcf but does not start with the TCF magic \
                     54 43 46 00 00 00 00 00 (Section 4). It starts with {}.",
                    self.path,
                    describe_magic(read_magic(path)?)
                ),
                _ => {}
            }
        }

        anyhow::bail!("Could not detect model format for: {}", self.path)
    }

    /// Open the model, reading its headers only
    pub fn load(&self) -> Result<LoadedModel> {
        let format = self.detect_format()?;
        log::info!("Detected format: {:?}", format);

        match format {
            crate::InputFormat::Safetensors => self.load_safetensors(),
            crate::InputFormat::Gguf => self.load_gguf(),
            crate::InputFormat::Tcf => self.load_tcf(),
            crate::InputFormat::PyTorch => self.load_torch(),
            crate::InputFormat::Onnx => {
                anyhow::bail!("ONNX format not yet supported")
            }
        }
    }

    /// Open a TCF file: its directory only, no payload page.
    fn load_tcf(&self) -> Result<LoadedModel> {
        let model = TcfModel::open(Path::new(&self.path))?;
        Ok(LoadedModel {
            inner: LoadedModelInner::Tcf(Box::new(model)),
            format: crate::InputFormat::Tcf,
        })
    }

    /// Open a PyTorch state dict.
    ///
    /// A directory holds its state dict in `pytorch_model.bin`. A sharded
    /// `pytorch_model.bin.index.json` names several, and reading only the
    /// first would silently drop most of the model, so it is refused with the
    /// shard count it found.
    fn load_torch(&self) -> Result<LoadedModel> {
        let path = Path::new(&self.path);
        let file = if path.is_dir() {
            let index = path.join("pytorch_model.bin.index.json");
            let single = path.join("pytorch_model.bin");
            if index.is_file() && !single.is_file() {
                anyhow::bail!(
                    "{} is a sharded PyTorch checkpoint ({}), and compressr reads one .bin \
                     per model. Convert the shards to a single file, or use the \
                     SafeTensors copy of the checkpoint.",
                    self.path,
                    index.display()
                );
            }
            if !single.is_file() {
                anyhow::bail!("No pytorch_model.bin in directory: {}", self.path);
            }
            single
        } else {
            path.to_path_buf()
        };

        let model = TorchModel::open(&file)?;
        Ok(LoadedModel {
            inner: LoadedModelInner::Torch(Box::new(model)),
            format: crate::InputFormat::PyTorch,
        })
    }

    /// Load a SafeTensors model
    fn load_safetensors(&self) -> Result<LoadedModel> {
        let path = Path::new(&self.path);

        if path.is_dir() {
            // Load sharded model
            self.load_sharded_safetensors(path)
        } else {
            // Load single file
            let st = SafeTensors::open(path)
                .with_context(|| format!("Failed to open SafeTensors file: {}", self.path))?;
            Ok(LoadedModel {
                inner: LoadedModelInner::SafeTensors(vec![st]),
                format: crate::InputFormat::Safetensors,
            })
        }
    }

    /// Load a sharded SafeTensors model from a directory
    ///
    /// The `model.safetensors.index.json` is authoritative when it is present:
    /// only the shards its `weight_map` names are loaded. Every other
    /// `.safetensors` file in the directory — a LoRA adapter, a leftover
    /// `consolidated.safetensors`, an earlier conversion's output — is ignored,
    /// because absorbing it would silently add tensors the model does not have.
    fn load_sharded_safetensors(&self, dir: &Path) -> Result<LoadedModel> {
        let mut files = Vec::new();

        let index_path = dir.join(SAFETENSORS_INDEX_FILE);
        if index_path.is_file() {
            let shard_files = read_index_shard_files(&index_path)?;
            log::info!(
                "Loading {} SafeTensors shard(s) named by {}",
                shard_files.len(),
                index_path.display()
            );

            for file_name in shard_files {
                let shard_path = dir.join(&file_name);
                if !shard_path.is_file() {
                    anyhow::bail!(
                        "Shard '{}' named by {} is missing from {}",
                        file_name,
                        index_path.display(),
                        dir.display()
                    );
                }
                let st = SafeTensors::open(&shard_path)
                    .with_context(|| format!("Failed to open shard: {:?}", shard_path))?;
                files.push(st);
            }

            return Ok(LoadedModel {
                inner: LoadedModelInner::SafeTensors(files),
                format: crate::InputFormat::Safetensors,
            });
        }

        // First try single model.safetensors
        let single_file = dir.join("model.safetensors");
        if single_file.exists() {
            let st = SafeTensors::open(&single_file)
                .with_context(|| format!("Failed to open: {:?}", single_file))?;
            return Ok(LoadedModel {
                inner: LoadedModelInner::SafeTensors(vec![st]),
                format: crate::InputFormat::Safetensors,
            });
        }

        // Fallback for a directory that has shards but no index: take every
        // .safetensors file, ordered by name. Without an index there is nothing
        // that says which files belong to the model, so this is a guess and the
        // index path above is always preferred.
        let mut shard_paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        // Sort by name for consistent ordering
        shard_paths.sort();

        if shard_paths.is_empty() {
            anyhow::bail!("No .safetensors files found in directory: {:?}", dir);
        }

        log::info!(
            "No shard index in {}: falling back to {} .safetensors file(s) found by name",
            dir.display(),
            shard_paths.len()
        );

        for shard_path in shard_paths {
            let st = SafeTensors::open(&shard_path)
                .with_context(|| format!("Failed to open shard: {:?}", shard_path))?;
            files.push(st);
        }

        Ok(LoadedModel {
            inner: LoadedModelInner::SafeTensors(files),
            format: crate::InputFormat::Safetensors,
        })
    }

    /// Load a GGUF model
    fn load_gguf(&self) -> Result<LoadedModel> {
        let gguf = Gguf::open(&self.path)
            .with_context(|| format!("Failed to open GGUF file: {}", self.path))?;
        Ok(LoadedModel {
            inner: LoadedModelInner::Gguf(gguf),
            format: crate::InputFormat::Gguf,
        })
    }
}

/// Internal enum for the loaded model type
enum LoadedModelInner {
    /// SafeTensors files (potentially sharded)
    SafeTensors(Vec<SafeTensors>),
    /// GGUF file
    Gguf(Gguf),
    /// TCF file, directory decoded
    Tcf(Box<TcfModel>),
    /// PyTorch state dict
    Torch(Box<TorchModel>),
}

/// An open model: headers in memory, tensor data still on disk
pub struct LoadedModel {
    inner: LoadedModelInner,
    format: crate::InputFormat,
}

impl LoadedModel {
    /// Get the input format
    pub fn format(&self) -> crate::InputFormat {
        self.format
    }

    /// Get the number of tensors in the model
    pub fn tensor_count(&self) -> usize {
        match &self.inner {
            LoadedModelInner::SafeTensors(files) => files.iter().map(|f| f.len()).sum(),
            LoadedModelInner::Gguf(gguf) => gguf.len(),
            LoadedModelInner::Tcf(tcf) => tcf.len(),
            LoadedModelInner::Torch(torch) => torch.len(),
        }
    }

    /// The file-level structure of a TCF input, or `None` for every other
    /// format.
    pub fn tcf_facts(&self) -> Option<&TcfFileFacts> {
        match &self.inner {
            LoadedModelInner::Tcf(tcf) => Some(tcf.facts()),
            _ => None,
        }
    }

    /// Get all tensor names, in a deterministic order.
    ///
    /// Names are sorted within each shard, and shards keep their own order.
    /// The safetensors reader hands back `HashMap` keys, and Rust seeds its
    /// hasher per process, so an unsorted walk gives a different tensor order
    /// on every run. Every downstream artifact inherits that order, which made
    /// two conversions of one checkpoint differ in 99.5% of their bytes while
    /// decoding identically. Determinism is a TCF format guarantee, so the
    /// order is fixed here, at the one place that enumerates tensors.
    pub fn tensor_names(&self) -> Vec<String> {
        match &self.inner {
            LoadedModelInner::SafeTensors(files) => files
                .iter()
                .flat_map(|f| {
                    let mut names: Vec<String> = f.tensor_names().map(|s| s.to_string()).collect();
                    names.sort_unstable();
                    names
                })
                .collect(),
            LoadedModelInner::Gguf(gguf) => gguf.tensor_names().map(|s| s.to_string()).collect(),
            // Both readers fix their own order: TCF keeps file order, and the
            // PyTorch state dict is sorted, because a pickle hands back a
            // `HashMap` exactly as the safetensors reader does.
            LoadedModelInner::Tcf(tcf) => tcf.tensor_names(),
            LoadedModelInner::Torch(torch) => torch.tensor_names(),
        }
    }

    /// Get information about a specific tensor
    pub fn tensor_info(&self, name: &str) -> Option<UnifiedTensorInfo> {
        match &self.inner {
            LoadedModelInner::SafeTensors(files) => {
                for file in files {
                    if let Ok(info) = file.tensor_info(name) {
                        return Some(UnifiedTensorInfo::from_safetensors(info));
                    }
                }
                None
            }
            LoadedModelInner::Gguf(gguf) => gguf
                .tensor_info(name)
                .ok()
                .map(UnifiedTensorInfo::from_gguf),
            LoadedModelInner::Tcf(tcf) => tcf.tensor_info(name),
            LoadedModelInner::Torch(torch) => torch.tensor_info(name),
        }
    }

    /// Iterate over all tensors with their info
    pub fn tensors(&self) -> impl Iterator<Item = (String, UnifiedTensorInfo)> + '_ {
        self.tensor_names()
            .into_iter()
            .filter_map(|name| self.tensor_info(&name).map(|info| (name, info)))
    }

    /// Read one tensor's raw bytes into an owned buffer
    ///
    /// Each call seeks to the tensor and reads it. The buffer is a copy, not a
    /// view into the file, so the caller owns it and it costs one tensor's
    /// worth of memory for as long as it is held.
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>> {
        match &mut self.inner {
            LoadedModelInner::SafeTensors(files) => {
                for file in files.iter_mut() {
                    if file.tensor_info(name).is_ok() {
                        return file.read_tensor_bytes(name).map_err(|e| {
                            anyhow::anyhow!("Failed to read tensor '{}': {}", name, e)
                        });
                    }
                }
                anyhow::bail!("Tensor not found: {}", name)
            }
            LoadedModelInner::Gguf(gguf) => {
                let bytes = gguf
                    .read_tensor_bytes(name)
                    .map_err(|e| anyhow::anyhow!("Failed to read tensor '{}': {}", name, e))?;
                Ok(bytes)
            }
            // A TCF payload is verified and decoded to f32, which is the
            // element type `tensor_info` reports for it.
            LoadedModelInner::Tcf(tcf) => tcf.read_tensor_bytes(name),
            LoadedModelInner::Torch(torch) => torch.read_tensor_bytes(name),
        }
    }

    /// Get model metadata (tries to extract from GGUF or infer from tensor names)
    pub fn metadata(&self) -> ModelMetadata {
        match &self.inner {
            LoadedModelInner::Gguf(gguf) => {
                let meta = gguf.metadata();
                ModelMetadata {
                    architecture: meta
                        .get_string("general.architecture")
                        .map(|s| s.to_string()),
                    num_layers: meta.get_u32("llama.block_count").map(|v| v as usize),
                    hidden_size: meta.get_u32("llama.embedding_length").map(|v| v as usize),
                    vocab_size: meta.get_u32("llama.vocab_size").map(|v| v as usize),
                    num_heads: meta
                        .get_u32("llama.attention.head_count")
                        .map(|v| v as usize),
                    num_kv_heads: meta
                        .get_u32("llama.attention.head_count_kv")
                        .map(|v| v as usize),
                    intermediate_size: meta
                        .get_u32("llama.feed_forward_length")
                        .map(|v| v as usize),
                    rope_theta: meta.get_f32("llama.rope.freq_base").map(|v| v as f64),
                }
            }
            // Neither a TCF directory nor a PyTorch pickle carries an
            // architecture, a head count or a rope base, so the shapes answer
            // what they can, exactly as they do for SafeTensors.
            LoadedModelInner::SafeTensors(_)
            | LoadedModelInner::Tcf(_)
            | LoadedModelInner::Torch(_) => self.infer_metadata_from_tensors(),
        }
    }

    /// Infer model metadata from tensor shapes
    fn infer_metadata_from_tensors(&self) -> ModelMetadata {
        let mut metadata = ModelMetadata::default();

        // Find embedding tensor to get hidden size and vocab size
        for name in self.tensor_names() {
            if name.contains("embed_tokens")
                || name.contains("wte")
                || name.contains("word_embeddings")
            {
                if let Some(info) = self.tensor_info(&name) {
                    if info.shape.len() == 2 {
                        metadata.vocab_size = Some(info.shape[0]);
                        metadata.hidden_size = Some(info.shape[1]);
                    }
                }
                break;
            }
        }

        // Count layers by looking for numbered layer patterns
        let mut max_layer = 0usize;
        for name in self.tensor_names() {
            // Match patterns like "layers.0.", "h.0.", "transformer.h.0.", "model.layers.0."
            for part in name.split('.') {
                if let Ok(layer_num) = part.parse::<usize>() {
                    max_layer = max_layer.max(layer_num + 1);
                    break;
                }
            }
        }
        if max_layer > 0 {
            metadata.num_layers = Some(max_layer);
        }

        metadata
    }

    /// Group tensors by layer for streaming processing
    pub fn layers(&self) -> Vec<LayerGroup> {
        let names = self.tensor_names();
        let mut layers: HashMap<Option<usize>, Vec<String>> = HashMap::new();

        for name in names {
            let layer_idx = extract_layer_index(&name);
            layers.entry(layer_idx).or_default().push(name);
        }

        // Sort and convert to LayerGroup
        let mut result: Vec<_> = layers
            .into_iter()
            .map(|(idx, tensors)| LayerGroup {
                layer_index: idx,
                tensor_names: tensors,
            })
            .collect();

        // Sort by layer index (None = embeddings/non-layer tensors first)
        result.sort_by_key(|g| g.layer_index);
        result
    }

    /// Calculate total model size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.tensors().map(|(_, info)| info.size_bytes).sum()
    }
}

/// First bytes of every ZIP container, and so of every `torch.save` archive.
const ZIP_MAGIC: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];

/// Read the first eight bytes of `path`, or `None` when it is shorter.
///
/// A short read is not an error here: a file too small to hold a magic is
/// simply not one of the formats identified by content, and the extension
/// rules answer for it.
fn read_magic(path: &Path) -> Result<Option<[u8; 8]>> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open model file: {}", path.display()))?;
    let mut magic = [0u8; 8];
    let mut filled = 0usize;
    while filled < magic.len() {
        let read = file
            .read(&mut magic[filled..])
            .with_context(|| format!("Failed to read the first bytes of: {}", path.display()))?;
        if read == 0 {
            return Ok(None);
        }
        filled += read;
    }
    Ok(Some(magic))
}

/// Render a file's first bytes for an error message.
fn describe_magic(magic: Option<[u8; 8]>) -> String {
    match magic {
        Some(magic) => magic
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" "),
        None => "fewer than 8 bytes".to_string(),
    }
}

/// Extract layer index from tensor name
pub fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns: "layers.N.", "h.N.", "transformer.h.N.", "model.layers.N."
    let parts: Vec<&str> = name.split('.').collect();

    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            // Check if previous part suggests this is a layer index
            let prev = parts[i - 1];
            if prev == "layers" || prev == "h" || prev == "block" || prev == "blk" {
                if let Ok(idx) = part.parse::<usize>() {
                    return Some(idx);
                }
            }
        }
    }
    None
}

/// Unified tensor info across formats
#[derive(Debug, Clone)]
pub struct UnifiedTensorInfo {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: UnifiedDType,
    /// Size in bytes, as the tensor is STORED in its input file
    pub size_bytes: usize,
    /// Number of elements
    pub numel: usize,
    /// What a TCF file states beyond shape and element type, or `None` for
    /// every other format.
    ///
    /// A TCF record carries a per-tensor encoding, a fallback reason, a
    /// residency class and the contract, calibration and workload records it
    /// is read against. Flattening that into `dtype` would drop it.
    pub tcf: Option<TcfTensorFacts>,
}

impl UnifiedTensorInfo {
    fn from_safetensors(info: &TensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            shape: info.shape.clone(),
            dtype: UnifiedDType::from_dtype(info.dtype),
            size_bytes: info.size_bytes(),
            numel: info.numel(),
            tcf: None,
        }
    }

    fn from_gguf(info: &GgufTensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            shape: info.shape.clone(),
            dtype: UnifiedDType::from_ggml(info.ggml_type),
            size_bytes: info.size_bytes(),
            numel: info.numel(),
            tcf: None,
        }
    }
}

/// Unified data type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedDType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U32,
    Bool,
    // Quantized types
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // IQ types
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    /// Unknown type
    Unknown,
}

impl UnifiedDType {
    fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F64 => Self::F64,
            DType::I32 => Self::I32,
            DType::I64 => Self::I64,
            DType::U8 => Self::U8,
            DType::U32 => Self::U32,
            DType::Bool => Self::Bool,
            _ => Self::Unknown,
        }
    }

    fn from_ggml(ggml: GgmlType) -> Self {
        match ggml {
            GgmlType::F32 => Self::F32,
            GgmlType::F16 => Self::F16,
            GgmlType::BF16 => Self::BF16,
            GgmlType::F64 => Self::F64,
            GgmlType::I8 => Self::I8,
            GgmlType::I16 => Self::I16,
            GgmlType::I32 => Self::I32,
            GgmlType::I64 => Self::I64,
            GgmlType::Q4_0 => Self::Q4_0,
            GgmlType::Q4_1 => Self::Q4_1,
            GgmlType::Q5_0 => Self::Q5_0,
            GgmlType::Q5_1 => Self::Q5_1,
            GgmlType::Q8_0 => Self::Q8_0,
            GgmlType::Q8_1 => Self::Q8_1,
            GgmlType::Q2K => Self::Q2K,
            GgmlType::Q3K => Self::Q3K,
            GgmlType::Q4K => Self::Q4K,
            GgmlType::Q5K => Self::Q5K,
            GgmlType::Q6K => Self::Q6K,
            GgmlType::Q8K => Self::Q8K,
        }
    }

    /// Check if this is a quantized type
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q8K
                | Self::IQ2XXS
                | Self::IQ2XS
                | Self::IQ3XXS
                | Self::IQ1S
                | Self::IQ4NL
                | Self::IQ3S
                | Self::IQ2S
                | Self::IQ4XS
        )
    }

    /// Check if this is a float type
    ///
    /// This is the discriminator used to decide whether a tensor may be cast to
    /// another element type. It reads the dtype recorded in the file header, not
    /// the tensor name, so integer buffers (token ids, offsets, masks) and
    /// already-quantized blocks pass through untouched.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64)
    }

    /// Get bytes per element, or `None` for quantized and unknown types
    ///
    /// Quantized types have no per-element size: they store whole blocks.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::F64 | Self::I64 => Some(8),
            Self::F32 | Self::I32 | Self::U32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::I8 | Self::U8 | Self::Bool => Some(1),
            _ => None,
        }
    }

    /// Get bits per weight (approximate for quantized)
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F64 => 64.0,
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::I32 | Self::U32 => 32.0,
            Self::I16 => 16.0,
            Self::I8 | Self::U8 => 8.0,
            Self::Bool => 1.0,
            Self::Q8_0 | Self::Q8_1 | Self::Q8K => 8.5, // 8 bits + scale overhead
            Self::Q6K => 6.5,
            Self::Q5_0 | Self::Q5_1 | Self::Q5K => 5.5,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K => 4.5,
            Self::Q3K | Self::IQ3XXS | Self::IQ3S => 3.5,
            Self::Q2K | Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 2.5,
            Self::IQ1S => 1.5,
            Self::IQ4NL | Self::IQ4XS => 4.5,
            Self::I64 => 64.0,
            Self::Unknown => 32.0, // Assume F32
        }
    }
}

impl std::fmt::Display for UnifiedDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::F64 => write!(f, "F64"),
            Self::I8 => write!(f, "I8"),
            Self::I16 => write!(f, "I16"),
            Self::I32 => write!(f, "I32"),
            Self::I64 => write!(f, "I64"),
            Self::U8 => write!(f, "U8"),
            Self::U32 => write!(f, "U32"),
            Self::Bool => write!(f, "Bool"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2K => write!(f, "Q2_K"),
            Self::Q3K => write!(f, "Q3_K"),
            Self::Q4K => write!(f, "Q4_K"),
            Self::Q5K => write!(f, "Q5_K"),
            Self::Q6K => write!(f, "Q6_K"),
            Self::Q8K => write!(f, "Q8_K"),
            Self::IQ2XXS => write!(f, "IQ2_XXS"),
            Self::IQ2XS => write!(f, "IQ2_XS"),
            Self::IQ3XXS => write!(f, "IQ3_XXS"),
            Self::IQ1S => write!(f, "IQ1_S"),
            Self::IQ4NL => write!(f, "IQ4_NL"),
            Self::IQ3S => write!(f, "IQ3_S"),
            Self::IQ2S => write!(f, "IQ2_S"),
            Self::IQ4XS => write!(f, "IQ4_XS"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    pub architecture: Option<String>,
    pub num_layers: Option<usize>,
    pub hidden_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub rope_theta: Option<f64>,
}

/// Group of tensors belonging to a single layer
#[derive(Debug, Clone)]
pub struct LayerGroup {
    /// Layer index (None for non-layer tensors like embeddings)
    pub layer_index: Option<usize>,
    /// Names of tensors in this group
    pub tensor_names: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.0.self_attn.q_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.layers.15.mlp.gate_proj.weight"),
            Some(15)
        );
        assert_eq!(
            extract_layer_index("transformer.h.0.attn.c_attn.weight"),
            Some(0)
        );
        assert_eq!(extract_layer_index("blk.5.attn_q.weight"), Some(5));
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    #[test]
    fn test_unified_dtype_bits() {
        assert_eq!(UnifiedDType::F32.bits_per_weight(), 32.0);
        assert_eq!(UnifiedDType::F16.bits_per_weight(), 16.0);
        assert_eq!(UnifiedDType::Q4_0.bits_per_weight(), 4.5);
    }

    #[test]
    fn test_unified_dtype_is_quantized() {
        assert!(!UnifiedDType::F32.is_quantized());
        assert!(!UnifiedDType::BF16.is_quantized());
        assert!(UnifiedDType::Q4_0.is_quantized());
        assert!(UnifiedDType::Q4K.is_quantized());
    }

    #[test]
    fn test_unified_dtype_is_float() {
        assert!(UnifiedDType::F32.is_float());
        assert!(UnifiedDType::F16.is_float());
        assert!(UnifiedDType::BF16.is_float());
        assert!(UnifiedDType::F64.is_float());

        assert!(!UnifiedDType::I8.is_float());
        assert!(!UnifiedDType::I32.is_float());
        assert!(!UnifiedDType::I64.is_float());
        assert!(!UnifiedDType::U8.is_float());
        assert!(!UnifiedDType::U32.is_float());
        assert!(!UnifiedDType::Bool.is_float());
        assert!(!UnifiedDType::Q4K.is_float());
        assert!(!UnifiedDType::Unknown.is_float());
    }

    #[test]
    fn test_unified_dtype_element_size() {
        assert_eq!(UnifiedDType::F32.element_size(), Some(4));
        assert_eq!(UnifiedDType::BF16.element_size(), Some(2));
        assert_eq!(UnifiedDType::F16.element_size(), Some(2));
        assert_eq!(UnifiedDType::F64.element_size(), Some(8));
        assert_eq!(UnifiedDType::I64.element_size(), Some(8));
        assert_eq!(UnifiedDType::Bool.element_size(), Some(1));
        assert_eq!(UnifiedDType::Q4K.element_size(), None);
        assert_eq!(UnifiedDType::Unknown.element_size(), None);
    }
}
