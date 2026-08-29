//! compressr: model conversion and compression
//!
//! Converts large language model checkpoints between SafeTensors and GGUF,
//! casts their element type, and quantizes them into GGUF block formats.
//!
//! # What is implemented
//!
//! - SafeTensors in, single file or sharded by `model.safetensors.index.json`
//! - GGUF in, for re-conversion
//! - SafeTensors out, sharded at a byte limit with a matching index
//! - GGUF v3 out, one self-contained file
//! - Element-type casts between F32, F16 and BF16, round-to-nearest-even
//! - GGUF block quantization: Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K
//! - TCF v1 out, quantized once from the full-precision source
//! - TCF profile allocation: a base encoding plus a bump encoding on the
//!   worst-reconstructed share of the parameters
//!
//! A TCF profile is allocated from a WEIGHT-ERROR screen. `PROFILES.md`
//! Section 4 step 2 measures a task-metric delta on a calibration set, which
//! compressr cannot run, so `sensitivity_delta`, `sensitivity_ci95`,
//! `sensitivity_samples` and `calibration_id` stay unset and no file compressr
//! writes is profile-solved in that sense.
//!
//! TCF is produced from a full-precision checkpoint only. Converting a GGUF
//! into one would quantize already-quantized weights a second time, so that
//! path does not exist.
//!
//! GPTQ, AWQ, NF4, FP8 and EXL2 are NOT implemented. Each has a
//! [`QuantizationMethod`] or [`OutputFormat`] variant, and each is refused
//! before any output file is created — see [`ConversionConfig::validate`] and
//! [`formats::create_writer`]. PyTorch and ONNX inputs are refused the same way.
//!
//! # Memory
//!
//! Opening a model reads its headers only. Each tensor is then read into an
//! owned buffer, converted, written, and dropped before the next one starts, so
//! peak resident memory is roughly one tensor plus its converted copy. This is
//! bounded by the largest tensor rather than constant, and the tensor bytes are
//! copied out of the file rather than mapped.
//!
//! # Architecture
//!
//! Compressr is a thin orchestration layer - all heavy computation happens in boostr:
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────┐    ┌───────────────┐
//! │   ModelLoader   │───▶│   Quantizer  │───▶│ FormatWriter  │
//! │  (SafeTensors)  │    │   (boostr)   │    │ (GGUF/ST)     │
//! └─────────────────┘    └──────────────┘    └───────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use compressr::{ConversionConfig, OutputFormat, QuantizationConfig, QuantizationMethod};
//!
//! let config = ConversionConfig::new(
//!     "model/".into(),
//!     "model.gguf".into(),
//!     OutputFormat::Gguf,
//! ).with_quantization(QuantizationConfig {
//!     method: QuantizationMethod::Q4K,
//!     bits: 4,
//!     group_size: 128,
//!     calibration_dataset: None,
//!     calibration_samples: 0,
//! });
//!
//! compressr::convert(config)?;
//! ```

pub mod cast;
pub mod formats;
pub mod loader;
pub mod policy;
pub mod quantization;

#[cfg(feature = "python")]
pub mod python;

pub use cast::{cast_bytes, parse_target_dtype, torch_dtype_name, ACCEPTED_DTYPE_NAMES};
pub use formats::{
    parse_shard_size, parse_tcf_encoding, parse_tcf_profile, FormatWriter, GgufBatchWriter,
    GgufFormatWriter, SafeTensorsFormatWriter, StreamingGgufWriter, TcfFormatWriter, TcfOptions,
    TcfProfile, DEFAULT_MAX_SHARD_SIZE, TCF_ENCODING_NAMES, TCF_PROFILE_NAMES,
};
pub use loader::{
    LayerGroup, LoadedModel, ModelLoader, ModelMetadata, UnifiedDType, UnifiedTensorInfo,
};
pub use policy::{
    plan_tcf_tensor, plan_tensor, relative_rms_error, select_bumped, FallbackReason, PlanCounts,
    ScreenedTensor, TcfPlan, TensorPlan,
};
pub use quantization::{
    QuantType, QuantizationConfig, QuantizationMethod, QuantizedTensor, Quantizer,
};

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Version of the compressr library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported input formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// Hugging Face Safetensors (native, single file or sharded with an index)
    Safetensors,
    /// PyTorch state dict (.bin, .pt)
    PyTorch,
    /// ONNX protobuf
    Onnx,
    /// GGUF format (for re-quantization)
    Gguf,
}

/// Supported output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// llama.cpp GGUF format
    Gguf,
    /// Activation-Aware Weight Quantization
    Awq,
    /// GPTQ weight-only quantization
    Gptq,
    /// EXL2 mixed-precision format
    Exl2,
    /// FP8 format (E4M3, E5M2)
    Fp8,
    /// Hugging Face Safetensors
    Safetensors,
    /// Tensor Contract Format v1
    Tcf,
}

impl OutputFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Gguf => "gguf",
            OutputFormat::Awq => "safetensors",
            OutputFormat::Gptq => "safetensors",
            OutputFormat::Exl2 => "safetensors",
            OutputFormat::Fp8 => "safetensors",
            OutputFormat::Safetensors => "safetensors",
            OutputFormat::Tcf => "tcf",
        }
    }
}

/// Main conversion configuration
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Input model path
    pub input_path: String,
    /// Output path
    pub output_path: String,
    /// Input format (auto-detected if None)
    pub input_format: Option<InputFormat>,
    /// Target output format
    pub output_format: OutputFormat,
    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,
    /// Element type to cast every float tensor to on write
    ///
    /// A dtype cast changes the element type only. It is not a quantization
    /// method: there are no scales and no packed blocks. Non-float tensors are
    /// never cast. Mutually exclusive with [`ConversionConfig::quantization`].
    pub target_dtype: Option<UnifiedDType>,
    /// Largest number of bytes in one output shard
    ///
    /// Output that exceeds this is written as numbered shards plus a
    /// `*.safetensors.index.json`; output that fits stays a single file with no
    /// index. `None` writes one file however large. Ignored for GGUF, which is
    /// one self-contained file. Defaults to [`DEFAULT_MAX_SHARD_SIZE`].
    pub max_shard_size: Option<u64>,
    /// Architecture written to GGUF's `general.architecture`
    ///
    /// Overrides the `model_type` of the checkpoint's `config.json` and any
    /// architecture the input itself carried. GGUF only.
    pub arch: Option<String>,
    /// TCF encoding, and the encoding its module declares as preferred
    ///
    /// One of this and [`ConversionConfig::tcf_profile`] is required by
    /// [`OutputFormat::Tcf`], and both are rejected by every other format.
    /// Neither value has a source of truth in a checkpoint, so compressr asks
    /// for both rather than defaulting them.
    pub tcf: Option<TcfOptions>,
    /// TCF allocation profile, the alternative to [`ConversionConfig::tcf`]
    ///
    /// A profile names a base encoding, a bump encoding and the share of
    /// quantizable parameters the bump covers, so it sets one encoding per
    /// tensor rather than one for the whole model.
    pub tcf_profile: Option<TcfProfile>,
    /// Verbose logging
    pub verbose: bool,
}

impl ConversionConfig {
    /// Create a new conversion configuration
    pub fn new(input_path: String, output_path: String, output_format: OutputFormat) -> Self {
        Self {
            input_path,
            output_path,
            input_format: None,
            output_format,
            quantization: None,
            target_dtype: None,
            max_shard_size: Some(DEFAULT_MAX_SHARD_SIZE),
            arch: None,
            tcf: None,
            tcf_profile: None,
            verbose: false,
        }
    }

    /// Set the TCF encoding and the encoding its module prefers
    pub fn with_tcf(mut self, options: TcfOptions) -> Self {
        self.tcf = Some(options);
        self
    }

    /// Set the TCF allocation profile
    pub fn with_tcf_profile(mut self, profile: TcfProfile) -> Self {
        self.tcf_profile = Some(profile);
        self
    }

    /// The TCF options a conversion runs with, from either flag.
    ///
    /// # Errors
    /// Neither was given. [`ConversionConfig::validate`] refuses that first,
    /// so reaching this is an internal error.
    pub fn tcf_options(&self) -> Result<TcfOptions> {
        match (self.tcf, self.tcf_profile) {
            (Some(options), _) => Ok(options),
            (None, Some(profile)) => Ok(TcfOptions::from_profile(profile)),
            (None, None) => bail!("Internal error: --format tcf reached convert with no encoding"),
        }
    }

    /// Set the architecture written to GGUF's `general.architecture`
    pub fn with_arch(mut self, arch: impl Into<String>) -> Self {
        self.arch = Some(arch.into());
        self
    }

    /// Set quantization configuration
    pub fn with_quantization(mut self, config: QuantizationConfig) -> Self {
        self.quantization = Some(config);
        self
    }

    /// Set the largest number of bytes in one output shard
    ///
    /// `None` writes a single weights file however large, and no index.
    pub fn with_max_shard_size(mut self, max_shard_size: Option<u64>) -> Self {
        self.max_shard_size = max_shard_size;
        self
    }

    /// Set the element type every float tensor is cast to
    pub fn with_target_dtype(mut self, dtype: UnifiedDType) -> Self {
        self.target_dtype = Some(dtype);
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Check that the configuration is internally consistent.
    ///
    /// # Errors
    ///
    /// - Both a target dtype and a quantization method are set. Quantizing
    ///   already fixes the stored representation, so a cast target alongside it
    ///   has no meaning; there is no precedence rule.
    /// - The quantization method has no on-disk layout in compressr. See
    ///   [`QuantizationMethod::is_writable`].
    /// - Quantization was requested for a non-GGUF output format.
    /// - The target dtype is not a supported float type.
    /// - `--format tcf` was given with both `--tcf-profile` and
    ///   `--tcf-encoding`, or with neither. A profile already fixes the base
    ///   encoding, so the two flags cannot both describe the same file.
    ///
    /// Every check runs BEFORE any output file is created, because a mismatch
    /// caught by the writer instead surfaces halfway through the tensor loop
    /// and leaves a truncated file behind.
    pub fn validate(&self) -> Result<()> {
        if self.target_dtype.is_some() && self.quantization.is_some() {
            bail!(
                "--dtype and --quantization are mutually exclusive: quantization already \
                 determines the stored representation. Pick one."
            );
        }
        if let Some(ref quant) = self.quantization {
            // The packed payload of an unwritable method is only half of the
            // tensor: NF4 and FP8 leave their scales outside it, GPTQ and AWQ
            // split a layer into several tensors. Writing the payload alone
            // either aborts mid-file on a byte-count mismatch or, for FP8,
            // succeeds and ships a checkpoint with no dequant scale in it.
            if !quant.method.is_writable() {
                bail!(
                    "Quantization method {:?} is not implemented: compressr has no on-disk \
                     layout for its scales, so the output would carry packed weights that \
                     nothing can dequantize. Implemented: q4_0, q4_1, q8_0, q4_k, q5_k, q6_k.",
                    quant.method
                );
            }
            // A block-quantized tensor is a run of whole blocks, not a run of
            // elements of one type. SafeTensors describes a tensor only by
            // element type and shape, so it has nowhere to record the block
            // layout, and a reader would see U8 noise.
            if self.output_format != OutputFormat::Gguf {
                bail!(
                    "Quantization writes GGUF only, but --format is {:?}. A SafeTensors \
                     header records an element type and a shape, which cannot describe a \
                     block-quantized tensor. Use --format gguf, or --dtype for a cast.",
                    self.output_format
                );
            }
        }
        if self.output_format == OutputFormat::Tcf {
            // A TCF module record ranks the encodings its producer prefers,
            // and TCF requires a `fallback_reason` on every tensor whose
            // encoding differs from the top-ranked one. Nothing in a
            // checkpoint names that preference, and compressr does I/O and
            // CLI rather than algorithm policy, so it is asked for.
            if self.tcf.is_some() && self.tcf_profile.is_some() {
                bail!(
                    "--tcf-profile and --tcf-encoding are mutually exclusive: a profile \
                     already names the base encoding, the bump encoding and the share of \
                     parameters the bump covers. Pick one."
                );
            }
            if self.tcf.is_none() && self.tcf_profile.is_none() {
                bail!(
                    "--format tcf needs --tcf-profile, or --tcf-encoding with \
                     --tcf-preferred-encoding. Neither has a default: the encoding fixes the \
                     stored weights, and the preferred encoding decides which tensors record \
                     a fallback reason. Accepted profiles: {}. Accepted encodings: {}",
                    TCF_PROFILE_NAMES.join(", "),
                    TCF_ENCODING_NAMES.join(", ")
                );
            }
            if self.target_dtype.is_some() {
                bail!(
                    "--dtype and --format tcf are mutually exclusive: TCF records an \
                     encoding per tensor, set by --tcf-encoding, so a cast target has no \
                     meaning beside it."
                );
            }
        } else if self.tcf.is_some() || self.tcf_profile.is_some() {
            bail!(
                "--tcf-encoding and --tcf-profile apply to --format tcf only, but --format \
                 is {:?}.",
                self.output_format
            );
        }
        if let Some(dtype) = self.target_dtype {
            if !matches!(
                dtype,
                UnifiedDType::F32 | UnifiedDType::F16 | UnifiedDType::BF16
            ) {
                bail!(
                    "Unsupported dtype: {}. Accepted: {}",
                    dtype,
                    ACCEPTED_DTYPE_NAMES.join(", ")
                );
            }
        }
        Ok(())
    }

    /// The element type a tensor of `source` dtype is stored as.
    ///
    /// Only float tensors follow the cast target. Integer buffers, boolean
    /// masks and already-quantized blocks keep their own type.
    pub fn stored_dtype(&self, source: UnifiedDType) -> UnifiedDType {
        match self.target_dtype {
            Some(target) if source.is_float() => target,
            _ => source,
        }
    }
}

/// Where a conversion writes its weights and its checkpoint sidecars
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLayout {
    /// Path of the weights file itself
    pub weights_path: PathBuf,
    /// Directory the sidecar files are copied into, if the format has any
    pub sidecar_dir: Option<PathBuf>,
}

/// Resolve `output_path` into a weights file plus a sidecar directory.
///
/// GGUF and TCF are each a single self-describing file and carry no sidecars. Every
/// SafeTensors-based format is a *checkpoint directory*: a bare weights file
/// with no tokenizer beside it is not loadable. So an output path that does not
/// already name a `.safetensors` file is treated as the checkpoint directory,
/// and the weights land inside it as `model.safetensors`.
pub fn resolve_output_layout(output_path: &str, output_format: OutputFormat) -> OutputLayout {
    let path = PathBuf::from(output_path);

    // GGUF and TCF are each one self-contained file and carry no sidecars.
    if matches!(output_format, OutputFormat::Gguf | OutputFormat::Tcf) {
        return OutputLayout {
            weights_path: path,
            sidecar_dir: None,
        };
    }

    let names_a_file = path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false);

    if names_a_file {
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        OutputLayout {
            weights_path: path,
            sidecar_dir: Some(parent),
        }
    } else {
        OutputLayout {
            weights_path: path.join("model.safetensors"),
            sidecar_dir: Some(path),
        }
    }
}

/// Statistics from a conversion run
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Number of tensors processed
    pub tensor_count: usize,
    /// Total input size in bytes
    pub input_size: u64,
    /// Total output size in bytes
    pub output_size: u64,
    /// Compression ratio (input / output)
    pub compression_ratio: f64,
    /// Elapsed time in seconds
    pub elapsed_secs: f64,
}

impl std::fmt::Display for ConversionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Processed {} tensors in {:.1}s | {:.2} GB → {:.2} GB | {:.2}x compression",
            self.tensor_count,
            self.elapsed_secs,
            self.input_size as f64 / (1024.0 * 1024.0 * 1024.0),
            self.output_size as f64 / (1024.0 * 1024.0 * 1024.0),
            self.compression_ratio
        )
    }
}

/// Convert a model from one format to another
///
/// This is the main entry point for the conversion pipeline. It:
/// 1. Loads the model using boostr's SafeTensors/GGUF reader
/// 2. Creates a format writer for the target format
/// 3. Iterates through tensors, quantizing and writing each one
/// 4. Returns statistics about the conversion
///
/// # Arguments
///
/// * `config` - Conversion configuration including input/output paths and quantization settings
///
/// # Returns
///
/// Statistics about the conversion, or an error if conversion failed
///
/// # Example
///
/// ```ignore
/// use compressr::{ConversionConfig, OutputFormat};
///
/// let config = ConversionConfig::new(
///     "model/".into(),
///     "model.gguf".into(),
///     OutputFormat::Gguf,
/// );
/// let stats = compressr::convert(config)?;
/// println!("{}", stats);
/// ```
pub fn convert(config: ConversionConfig) -> Result<ConversionStats> {
    let start_time = std::time::Instant::now();

    config.validate()?;

    let layout = resolve_output_layout(&config.output_path, config.output_format);

    log::info!("Starting model conversion");
    log::info!("Input: {}", config.input_path);
    log::info!(
        "Output: {} ({:?})",
        layout.weights_path.display(),
        config.output_format
    );

    if let Some(ref quant) = config.quantization {
        log::info!("Quantization: {:?} ({} bits)", quant.method, quant.bits);
    }
    if let Some(dtype) = config.target_dtype {
        log::info!("Target dtype: {} (float tensors only)", dtype);
    }

    // Step 1: Load the model
    log::info!("Loading model...");
    let mut model = ModelLoader::new(&config.input_path)
        .load()
        .with_context(|| format!("Failed to load model from: {}", config.input_path))?;

    // TCF is quantized once, from full-precision weights. A GGUF input is
    // already quantized, so writing TCF from one would quantize twice and
    // would need a GGML dequantizer per block format. Refused here, before
    // any output file exists.
    if config.output_format == OutputFormat::Tcf && model.format() == InputFormat::Gguf {
        bail!(
            "TCF is produced from a full-precision checkpoint, and '{}' is GGUF. Convert \
             from the SafeTensors source instead: quantizing an already-quantized tensor \
             loses accuracy twice.",
            config.input_path
        );
    }

    let tensor_count = model.tensor_count();
    let input_size = model.total_size_bytes();

    log::info!(
        "Loaded {} tensors ({:.2} GB)",
        tensor_count,
        input_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Step 2: Create the quantizer (if quantization is requested)
    let quant_type = config.quantization.as_ref().map(|q| match q.method {
        QuantizationMethod::Q4_0 => QuantType::Q4_0,
        QuantizationMethod::Q4_1 => QuantType::Q4_1,
        QuantizationMethod::Q8_0 => QuantType::Q8_0,
        QuantizationMethod::Q4K => QuantType::Q4K,
        QuantizationMethod::Q5K => QuantType::Q5K,
        QuantizationMethod::Q6K => QuantType::Q6K,
        QuantizationMethod::Nf4 => QuantType::NF4,
        QuantizationMethod::Fp8 => QuantType::FP8,
        QuantizationMethod::Gptq => QuantType::GPTQ,
        QuantizationMethod::Awq => QuantType::AWQ,
    });

    let quantizer = config
        .quantization
        .as_ref()
        .map(|qc| Quantizer::new(qc.clone()));

    // Step 3: Create the format writer
    log::info!("Creating output writer...");
    let mut writer = formats::create_writer(config.output_format, config.max_shard_size)?;

    // Get model metadata
    let metadata = model.metadata();

    // Configure writer based on format
    if config.output_format == OutputFormat::Gguf {
        // For GGUF, we need to set up metadata. Every consumer keys its
        // tensor-name mapping off `general.architecture`, so it is resolved
        // from the checkpoint (or --arch) rather than guessed.
        let input_path = Path::new(&config.input_path);
        let arch = policy::resolve_architecture(
            config.arch.as_deref(),
            metadata.architecture.as_deref(),
            input_path,
        )?;
        let arch = arch.as_str();

        // We need to configure the writer before initialization
        // Since we can't easily downcast, we'll create a new GGUF-specific writer
        let mut gguf_writer = GgufFormatWriter::new();
        gguf_writer.set_architecture(arch);

        // The name is the checkpoint's own name, not its architecture. With no
        // checkpoint directory to take it from, no name is written.
        if let Some(name) = policy::model_name(input_path) {
            gguf_writer.set_name(&name);
        }

        // Add architecture-specific metadata
        if let Some(vocab_size) = metadata.vocab_size {
            gguf_writer.add_metadata(
                &format!("{}.vocab_size", arch),
                formats::GgufValue::Uint32(vocab_size as u32),
            );
        }
        if let Some(hidden_size) = metadata.hidden_size {
            gguf_writer.add_metadata(
                &format!("{}.embedding_length", arch),
                formats::GgufValue::Uint32(hidden_size as u32),
            );
        }
        if let Some(num_layers) = metadata.num_layers {
            gguf_writer.add_metadata(
                &format!("{}.block_count", arch),
                formats::GgufValue::Uint32(num_layers as u32),
            );
        }

        // Set file type based on quantization
        if let Some(qt) = quant_type {
            let file_type = match qt {
                QuantType::Q4_0 => 2,
                QuantType::Q4_1 => 3,
                QuantType::Q5_0 => 6,
                QuantType::Q5_1 => 7,
                QuantType::Q8_0 => 8,
                QuantType::Q2K => 10,
                QuantType::Q3K => 11,
                QuantType::Q4K => 12,
                QuantType::Q5K => 13,
                QuantType::Q6K => 14,
                _ => 1, // F16 default
            };
            gguf_writer.set_file_type(file_type);
        } else if let Some(dtype) = config.target_dtype {
            // GGUF file_type values for unquantized casts.
            let file_type = match dtype {
                UnifiedDType::F32 => 0,
                UnifiedDType::BF16 => 32,
                _ => 1, // F16
            };
            gguf_writer.set_file_type(file_type);
        }

        writer = Box::new(gguf_writer);
    }

    if config.output_format == OutputFormat::Tcf {
        // TCF quantizes from the full-precision source itself, so the writer
        // needs the input path as well as the encoding: its payload pass
        // re-reads one tensor at a time instead of holding every payload.
        let options = config.tcf_options()?;
        let mut tcf_writer = TcfFormatWriter::new()
            .with_options(options)
            .with_source(&config.input_path);
        if let Some(name) = policy::model_name(Path::new(&config.input_path)) {
            tcf_writer = tcf_writer.with_module_name(&name);
        }
        match options.profile {
            Some(profile) => log::info!(
                "TCF profile {}: base {:?}, bump {:?} on the top {:.0}% of quantizable \
                 parameters",
                profile.name,
                profile.base,
                profile.bump,
                profile.bump_fraction * 100.0
            ),
            None => log::info!(
                "TCF encoding: {:?}, module preferred encoding: {:?}",
                options.encoding,
                options.preferred_encoding
            ),
        }
        log::info!(
            "TCF calibration_id, activation_contract_id and workload_profile_id are left \
             unset: compressr runs no calibration and measures no workload."
        );
        writer = Box::new(tcf_writer);
    }

    // Collect tensor info for registration (need to collect because we iterate twice)
    let tensor_info: Vec<(String, UnifiedTensorInfo)> = model.tensors().collect();

    // Decide once per tensor what happens to it. The registration loop reserves
    // the byte count and the write loop produces the bytes; a tensor the two
    // loops disagreed about would shift every later offset in the output file
    // without raising an error, so both read this one list.
    let plans: Vec<TensorPlan> = tensor_info
        .iter()
        .map(|(_, info)| policy::plan_tensor(&info.shape, info.dtype, quant_type))
        .collect();

    // Step 4: Initialize writer and register all tensors
    if let Some(parent) = layout.weights_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory: {}", parent.display())
            })?;
        }
    }

    writer.initialize(&layout.weights_path).with_context(|| {
        format!(
            "Failed to initialize writer: {}",
            layout.weights_path.display()
        )
    })?;

    // Register all tensors with the writer, using the dtype they will be
    // *stored* as so the reserved byte count matches the data written below.
    log::info!("Registering tensors...");
    for ((name, info), plan) in tensor_info.iter().zip(&plans) {
        match *plan {
            TensorPlan::Quantize(qt) => {
                writer.register_tensor(name, &info.shape, info.dtype, Some(qt))?
            }
            // A tensor that cannot be quantized is stored as F32, so that is
            // the dtype its reservation must use.
            TensorPlan::StoreF32(_) => {
                writer.register_tensor(name, &info.shape, UnifiedDType::F32, None)?
            }
            TensorPlan::StoreAsIs => {
                writer.register_tensor(name, &info.shape, config.stored_dtype(info.dtype), None)?
            }
        }
    }

    // Step 5: Process tensors - quantize and write
    log::info!("Processing tensors...");
    let mut processed = 0;
    let mut output_size = 0u64;
    let mut counts = PlanCounts::default();

    for ((name, info), plan) in tensor_info.iter().zip(&plans) {
        if config.verbose {
            log::debug!("Processing tensor: {} {:?} {:?}", name, info.shape, plan);
        }

        // Read tensor data
        let tensor_data = model.read_tensor_bytes(name)?;

        // Quantize, cast, or pass through untouched, following the plan the
        // registration loop above already reserved space for.
        let output_data = match *plan {
            TensorPlan::Quantize(_) => {
                let quantizer = quantizer
                    .as_ref()
                    .context("Internal error: a tensor was planned for quantization but no quantizer is configured")?;
                let quantized = quantizer.quantize_tensor(&tensor_data, &info.shape, info.dtype)?;
                quantized.packed_data
            }
            TensorPlan::StoreF32(_) => {
                cast::cast_bytes(&tensor_data, info.dtype, UnifiedDType::F32)
                    .with_context(|| format!("Failed to cast tensor '{}' to f32", name))?
            }
            TensorPlan::StoreAsIs => {
                let stored = config.stored_dtype(info.dtype);
                if stored == info.dtype {
                    // Non-float tensors land here and pass through byte-for-byte.
                    tensor_data
                } else {
                    cast::cast_bytes(&tensor_data, info.dtype, stored)
                        .with_context(|| format!("Failed to cast tensor '{}'", name))?
                }
            }
        };
        counts.record(*plan);

        output_size += output_data.len() as u64;

        // Write to output
        writer.write_tensor_data(&output_data)?;

        processed += 1;
        if processed % 50 == 0 {
            log::info!("Processed {}/{} tensors", processed, tensor_count);
        }
    }

    if quantizer.is_some() {
        log::info!("Quantization: {}", counts);
    }

    // Step 6: Finalize
    log::info!("Finalizing output...");
    writer
        .finalize()
        .with_context(|| "Failed to finalize output file")?;

    // The loop above counted the SOURCE bytes it planned each TCF tensor
    // from; the payload the writer produced from them is a different size, so
    // the finished file is the only honest answer.
    if config.output_format == OutputFormat::Tcf {
        output_size = std::fs::metadata(&layout.weights_path)
            .with_context(|| {
                format!(
                    "Failed to measure the output file: {}",
                    layout.weights_path.display()
                )
            })?
            .len();
    }

    // Step 7: Copy the sidecars that make the output a loadable checkpoint
    if let Some(ref sidecar_dir) = layout.sidecar_dir {
        formats::copy_sidecars(
            Path::new(&config.input_path),
            sidecar_dir,
            config.target_dtype,
        )
        .with_context(|| "Failed to copy checkpoint sidecar files")?;
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let compression_ratio = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        1.0
    };

    let stats = ConversionStats {
        tensor_count,
        input_size: input_size as u64,
        output_size,
        compression_ratio,
        elapsed_secs: elapsed,
    };

    log::info!("Conversion complete: {}", stats);

    Ok(stats)
}

/// Get information about a model without converting it
pub fn get_model_info(path: &str) -> Result<ModelInfo> {
    let model = ModelLoader::new(path)
        .load()
        .with_context(|| format!("Failed to load model from: {}", path))?;

    let tensors: Vec<TensorSummary> = model
        .tensors()
        .map(|(name, info)| TensorSummary {
            name,
            shape: info.shape.iter().map(|&x| x as i64).collect(),
            dtype: format!("{:?}", info.dtype),
            size_bytes: info.size_bytes as u64,
        })
        .collect();

    let total_size = model.total_size_bytes();
    let metadata = model.metadata();

    Ok(ModelInfo {
        path: path.to_string(),
        format: format!("{:?}", model.format()),
        tensor_count: tensors.len(),
        total_size_bytes: total_size as u64,
        metadata: Some(metadata),
        tensors,
    })
}

/// Summary information about a model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Path to the model
    pub path: String,
    /// Format name (safetensors, gguf, etc.)
    pub format: String,
    /// Number of tensors
    pub tensor_count: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Model metadata (if available)
    pub metadata: Option<ModelMetadata>,
    /// Per-tensor summaries
    pub tensors: Vec<TensorSummary>,
}

/// Summary of a single tensor
#[derive(Debug, Clone)]
pub struct TensorSummary {
    /// Tensor name
    pub name: String,
    /// Shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: String,
    /// Size in bytes
    pub size_bytes: u64,
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model: {}", self.path)?;
        writeln!(f, "Format: {}", self.format)?;
        writeln!(f, "Tensors: {}", self.tensor_count)?;
        writeln!(
            f,
            "Size: {:.2} GB",
            self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;

        if let Some(ref meta) = self.metadata {
            writeln!(f, "\nMetadata:")?;
            if let Some(ref arch) = meta.architecture {
                writeln!(f, "  Architecture: {}", arch)?;
            }
            if let Some(vocab) = meta.vocab_size {
                writeln!(f, "  Vocab size: {}", vocab)?;
            }
            if let Some(hidden) = meta.hidden_size {
                writeln!(f, "  Hidden size: {}", hidden)?;
            }
            if let Some(layers) = meta.num_layers {
                writeln!(f, "  Layers: {}", layers)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Gguf.extension(), "gguf");
        assert_eq!(OutputFormat::Safetensors.extension(), "safetensors");
        assert_eq!(OutputFormat::Gptq.extension(), "safetensors");
        assert_eq!(OutputFormat::Tcf.extension(), "tcf");
    }

    fn tcf_options() -> TcfOptions {
        TcfOptions::uniform(
            tcf_core::NativeEncoding::Q4S32T64,
            tcf_core::NativeEncoding::Q4S32T64,
        )
    }

    /// The module's preferred encoding decides which tensors record a
    /// fallback reason, and no checkpoint carries it. Guessing one would put
    /// a policy compressr never made into the file, so the configuration is
    /// refused BEFORE any file is opened.
    #[test]
    fn test_tcf_without_an_encoding_is_refused_before_any_file_is_opened() {
        let config = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("--tcf-preferred-encoding"), "{err}");
        assert!(err.contains("--tcf-profile"), "{err}");
        assert!(err.contains("q4s32t64"), "{err}");
        assert!(err.contains("balanced"), "{err}");
        assert!(!Path::new("out.tcf").exists());
    }

    /// A profile and a flat encoding describe the same file two ways. There
    /// is no precedence rule, so both together is refused before any file is
    /// opened.
    #[test]
    fn test_tcf_profile_and_tcf_encoding_are_mutually_exclusive() {
        let config = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf(tcf_options())
            .with_tcf_profile(parse_tcf_profile("balanced").unwrap());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("mutually exclusive"), "{err}");
        assert!(!Path::new("out.tcf").exists());
    }

    #[test]
    fn test_tcf_profile_alone_validates() {
        let config = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf_profile(parse_tcf_profile("compact").unwrap());
        assert!(config.validate().is_ok());
        // A profile is its own preferred encoding, so a bumped tensor differs
        // from the module preference and records a reason.
        let options = config.tcf_options().unwrap();
        assert_eq!(options.encoding, options.preferred_encoding);
        assert_eq!(options.encoding, tcf_core::NativeEncoding::Q4AS32DT64);
    }

    /// A profile applies to TCF alone, exactly as the encoding flags do.
    #[test]
    fn test_tcf_profile_is_refused_for_another_format() {
        let config = ConversionConfig::new("input/".into(), "out.gguf".into(), OutputFormat::Gguf)
            .with_tcf_profile(parse_tcf_profile("high").unwrap());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("--format tcf only"), "{err}");
    }

    #[test]
    fn test_tcf_with_both_encodings_validates() {
        let config = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf(tcf_options());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tcf_encoding_is_refused_for_another_format() {
        let config = ConversionConfig::new("input/".into(), "out.gguf".into(), OutputFormat::Gguf)
            .with_tcf(tcf_options());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("--format tcf only"), "{err}");
    }

    /// A cast target names one element type for every float tensor, which is
    /// what --tcf-encoding already does for a TCF file.
    #[test]
    fn test_tcf_and_dtype_are_mutually_exclusive() {
        let config = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf(tcf_options())
            .with_target_dtype(UnifiedDType::BF16);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("mutually exclusive"), "{err}");
    }

    /// TCF is one self-contained file, as GGUF is: an output path names the
    /// file, not a checkpoint directory.
    #[test]
    fn test_tcf_output_layout_is_a_single_file() {
        let layout = resolve_output_layout("model.tcf", OutputFormat::Tcf);
        assert_eq!(layout.weights_path, PathBuf::from("model.tcf"));
        assert_eq!(layout.sidecar_dir, None);
    }

    #[test]
    fn test_conversion_config() {
        let config =
            ConversionConfig::new("input/".into(), "output.gguf".into(), OutputFormat::Gguf)
                .with_quantization(QuantizationConfig {
                    method: QuantizationMethod::Q4_0,
                    bits: 4,
                    group_size: 32,
                    calibration_dataset: None,
                    calibration_samples: 0,
                })
                .with_verbose(true);

        assert_eq!(config.input_path, "input/");
        assert_eq!(config.output_path, "output.gguf");
        assert_eq!(config.output_format, OutputFormat::Gguf);
        assert!(config.quantization.is_some());
        assert!(config.verbose);
        assert_eq!(config.target_dtype, None);
        assert_eq!(config.max_shard_size, Some(DEFAULT_MAX_SHARD_SIZE));
    }

    #[test]
    fn test_with_max_shard_size_overrides_the_default() {
        let config =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
                .with_max_shard_size(Some(2_000_000_000));
        assert_eq!(config.max_shard_size, Some(2_000_000_000));

        let unlimited =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
                .with_max_shard_size(None);
        assert_eq!(unlimited.max_shard_size, None);
    }

    #[test]
    fn test_target_dtype_and_quantization_are_mutually_exclusive() {
        let config =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
                .with_target_dtype(UnifiedDType::BF16)
                .with_quantization(QuantizationConfig::q4_0());

        let err = config.validate().unwrap_err().to_string();
        assert_eq!(
            err,
            "--dtype and --quantization are mutually exclusive: quantization already \
             determines the stored representation. Pick one."
        );
    }

    #[test]
    fn test_validate_accepts_either_alone() {
        let cast_only =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
                .with_target_dtype(UnifiedDType::BF16);
        assert!(cast_only.validate().is_ok());

        let quant_only =
            ConversionConfig::new("input/".into(), "out.gguf".into(), OutputFormat::Gguf)
                .with_quantization(QuantizationConfig::q4_k());
        assert!(quant_only.validate().is_ok());
    }

    /// NF4 and FP8 pack their scales OUTSIDE `packed_data`, and nothing writes
    /// those scales. FP8 to SafeTensors used to succeed: it produced a U8
    /// tensor of the right byte count whose dequant scale existed nowhere in
    /// the checkpoint, so the weights read back scaled by 448/amax.
    #[test]
    fn test_validate_rejects_methods_with_no_on_disk_layout() {
        for config in [
            QuantizationConfig::nf4(),
            QuantizationConfig::fp8(),
            QuantizationConfig::gptq(4, 128),
            QuantizationConfig::awq(4, 128),
        ] {
            let method = config.method;
            let err = ConversionConfig::new("input/".into(), "out.gguf".into(), OutputFormat::Gguf)
                .with_quantization(config)
                .validate()
                .unwrap_err()
                .to_string();

            assert!(err.contains("not implemented"), "{method:?}: {err}");
            assert!(err.contains(&format!("{method:?}")), "{method:?}: {err}");
        }
    }

    /// A GGML block is not an element type, so a SafeTensors header cannot
    /// describe one. Registering the tensor as U8 reserved one byte per
    /// element and the packed blocks never matched that count.
    #[test]
    fn test_validate_rejects_quantization_into_safetensors() {
        let err = ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
            .with_quantization(QuantizationConfig::q4_k())
            .validate()
            .unwrap_err()
            .to_string();

        assert!(err.contains("Quantization writes GGUF only"), "{err}");
        assert!(err.contains("Safetensors"), "{err}");
    }

    #[test]
    fn test_validate_accepts_every_gguf_block_format() {
        for config in [
            QuantizationConfig::q4_0(),
            QuantizationConfig::q8_0(),
            QuantizationConfig::q4_k(),
            QuantizationConfig::q5_k(),
            QuantizationConfig::q6_k(),
        ] {
            let method = config.method;
            ConversionConfig::new("input/".into(), "out.gguf".into(), OutputFormat::Gguf)
                .with_quantization(config)
                .validate()
                .unwrap_or_else(|e| panic!("{method:?} must validate: {e}"));
        }
    }

    #[test]
    fn test_validate_rejects_non_float_target_dtype() {
        let config =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
                .with_target_dtype(UnifiedDType::I32);

        let err = config.validate().unwrap_err().to_string();
        assert_eq!(
            err,
            "Unsupported dtype: I32. Accepted: f32, fp32, float32, float, f16, fp16, float16, half, bf16, bfloat16, brain16"
        );
    }

    #[test]
    fn test_stored_dtype_casts_floats_and_passes_others_through() {
        let config =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors)
                .with_target_dtype(UnifiedDType::BF16);

        assert_eq!(config.stored_dtype(UnifiedDType::F32), UnifiedDType::BF16);
        assert_eq!(config.stored_dtype(UnifiedDType::F16), UnifiedDType::BF16);
        assert_eq!(config.stored_dtype(UnifiedDType::F64), UnifiedDType::BF16);

        assert_eq!(config.stored_dtype(UnifiedDType::I64), UnifiedDType::I64);
        assert_eq!(config.stored_dtype(UnifiedDType::U8), UnifiedDType::U8);
        assert_eq!(config.stored_dtype(UnifiedDType::Bool), UnifiedDType::Bool);
        assert_eq!(config.stored_dtype(UnifiedDType::Q4K), UnifiedDType::Q4K);
    }

    #[test]
    fn test_stored_dtype_without_target_is_identity() {
        let config =
            ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors);
        assert_eq!(config.stored_dtype(UnifiedDType::F32), UnifiedDType::F32);
        assert_eq!(config.stored_dtype(UnifiedDType::I64), UnifiedDType::I64);
    }

    #[test]
    fn test_resolve_output_layout_gguf_is_a_single_file() {
        let layout = resolve_output_layout("out/model.gguf", OutputFormat::Gguf);
        assert_eq!(
            layout,
            OutputLayout {
                weights_path: PathBuf::from("out/model.gguf"),
                sidecar_dir: None,
            }
        );
    }

    #[test]
    fn test_resolve_output_layout_directory_becomes_a_checkpoint() {
        let layout = resolve_output_layout("llama-3-8b-bf16", OutputFormat::Safetensors);
        assert_eq!(
            layout,
            OutputLayout {
                weights_path: PathBuf::from("llama-3-8b-bf16/model.safetensors"),
                sidecar_dir: Some(PathBuf::from("llama-3-8b-bf16")),
            }
        );
    }

    #[test]
    fn test_resolve_output_layout_explicit_file_keeps_its_name() {
        let layout = resolve_output_layout("out/weights.safetensors", OutputFormat::Safetensors);
        assert_eq!(
            layout,
            OutputLayout {
                weights_path: PathBuf::from("out/weights.safetensors"),
                sidecar_dir: Some(PathBuf::from("out")),
            }
        );
    }

    #[test]
    fn test_resolve_output_layout_bare_file_uses_current_directory() {
        let layout = resolve_output_layout("weights.safetensors", OutputFormat::Safetensors);
        assert_eq!(
            layout,
            OutputLayout {
                weights_path: PathBuf::from("weights.safetensors"),
                sidecar_dir: Some(PathBuf::from(".")),
            }
        );
    }

    #[test]
    fn test_conversion_stats_display() {
        let stats = ConversionStats {
            tensor_count: 100,
            input_size: 10 * 1024 * 1024 * 1024, // 10 GB
            output_size: 5 * 1024 * 1024 * 1024, // 5 GB
            compression_ratio: 2.0,
            elapsed_secs: 30.5,
        };

        let display = format!("{}", stats);
        assert!(display.contains("100 tensors"));
        assert!(display.contains("2.00x compression"));
    }
}
