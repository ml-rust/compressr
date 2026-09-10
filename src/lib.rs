//! compressr: model conversion and compression
//!
//! Converts large language model checkpoints between SafeTensors and GGUF,
//! casts their element type, and quantizes them into GGUF block formats.
//!
//! # What is implemented
//!
//! - SafeTensors in, single file or sharded by `model.safetensors.index.json`
//! - GGUF in, for re-conversion
//! - TCF in, read back through boostr's reader: every payload is verified
//!   against its digests and proof vector, then decoded to f32
//! - PyTorch in, one `.pt` / `.pth` / `.bin` state dict, F32 and F64 tensors
//! - SafeTensors out, sharded at a byte limit with a matching index
//! - GGUF v3 out, one self-contained file
//! - Element-type casts between F32, F16 and BF16, round-to-nearest-even
//! - GGUF block quantization: Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
//! - GGUF mixed precision: Q4_K_S, Q4_K_M, Q5_K_S and Q5_K_M, which assign a
//!   different block format per tensor by role and by layer, reproducing
//!   llama.cpp's `llama_tensor_get_type`. The Q2_K/Q3_K mixes and the I-quant
//!   mixes are refused by name: each needs mix rules that do not exist here,
//!   even though Q2_K and Q3_K are writable as flat formats.
//! - TCF v1 out, quantized once from the full-precision source
//! - TCF profile allocation: a base encoding plus a bump encoding on the
//!   worst-reconstructed share of the parameters
//! - TCF task-metric allocation: the same profile, with the bump set ranked by
//!   a measured sensitivity file instead
//! - TCF error objective: each encoding carries its own default, and
//!   `--tcf-objective` overrides that default on every tensor — `uniform`
//!   (`w = 1`), `squared` (`w = x^2`) or `rms-plus-abs`
//!   (`w = sqrt(mean(x^2)) + |x|` over each group). It is resolved per tensor
//!   from that tensor's encoding, so a mixed-precision profile minimizes each
//!   half of the file under the objective measured for it. It changes which
//!   values the encoder chooses and no bit layout, so the file stays a
//!   conforming TCF v1 file
//! - GGUF task-metric allocation: the same four mixes, with the promotions
//!   ranked by that measurement instead of by llama.cpp's positional rules,
//!   spending the byte budget those rules would have spent. See
//!   [`policy::mix_measured`].
//!
//! A TCF profile is allocated from a WEIGHT-ERROR screen by default.
//! `PROFILES.md` Section 4 step 2 measures a task-metric delta on a
//! calibration set, which compressr cannot run, so on such a run
//! `sensitivity_delta`, `sensitivity_ci95`, `sensitivity_samples` and
//! `calibration_id` stay unset and the file is not profile-solved in that
//! sense.
//!
//! `--sensitivity` supplies that measurement from a harness that did run the
//! model. The bump set is then ranked by measured task damage per element, and
//! `sensitivity_delta`, `sensitivity_samples` and `calibration_id` are written
//! from it alongside a `CalibrationRecord`. `sensitivity_ci95` stays unset:
//! the measurement carries no confidence interval. See
//! [`policy::sensitivity`].
//!
//! TCF is produced from a full-precision checkpoint only. Converting a GGUF
//! or a TCF into one would quantize already-quantized weights a second time,
//! so that path does not exist.
//!
//! GPTQ, AWQ, NF4, FP8 and EXL2 are NOT implemented. Each has a
//! [`QuantizationMethod`] or [`OutputFormat`] variant, and each is refused
//! before any output file is created — see [`ConversionConfig::validate`] and
//! [`formats::create_writer`]. ONNX input is refused the same way.
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
    default_objective_for, objective_policy, parse_shard_size, parse_tcf_encoding,
    parse_tcf_objective, parse_tcf_profile, FormatWriter, GgufBatchWriter, GgufFormatWriter,
    ObjectiveSource, PlannedTensor, SafeTensorsFormatWriter, StreamingGgufWriter, TcfFormatWriter,
    TcfObjective, TcfOptions, TcfProfile, DEFAULT_MAX_SHARD_SIZE, TCF_ENCODING_NAMES,
    TCF_OBJECTIVE_NAMES, TCF_PROFILE_NAMES,
};
pub use loader::{
    extract_layer_index, LayerGroup, LoadedModel, ModelLoader, ModelMetadata, TcfFileFacts,
    TcfModuleFacts, TcfTensorFacts, UnifiedDType, UnifiedTensorInfo,
};
pub use policy::{
    assign_mix, assign_mix_measured, assign_mix_reported, check_measurement_encoding,
    mix_model_facts, parse_gguf_mix, parse_sensitivity, partition_components, plan_format_label,
    plan_mixed_tensor, plan_tcf_tensor, plan_tensor, principal_component, quant_type_label,
    relative_rms_error, select_bumped, select_bumped_measured, split_component, use_more_bits,
    BumpCandidate, ComponentAllocation, FallbackReason, FormatUsage, GgufMix, LadderChoice,
    LadderMenu, LadderOption, LadderOutcome, LadderReport, LayerCountSource, MeasuredBump,
    MeasuredMixAllocation, MeasuredOption, MeasuredTensor, MixAllocation, MixEntry, MixModel,
    MixReport, PlanCounts, ScreenedTensor, SensitivityBaseline, SensitivityMeasurement, TcfPlan,
    TensorPlan, TensorRole, GGUF_MIX_NAMES, UNLAYERED_COMPONENT,
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
    /// PyTorch state dict (.bin, .pt, .pth)
    PyTorch,
    /// ONNX protobuf
    Onnx,
    /// GGUF format (for re-quantization)
    Gguf,
    /// Tensor Contract Format v1, read back through boostr's reader
    Tcf,
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
    /// Error objective the TCF quantizer minimizes, for either TCF flag
    ///
    /// `None` leaves every tensor on the default its own ENCODING carries, so
    /// a mixed-precision profile minimizes each half of the file under the
    /// objective measured for it. `Some` overrides that default on every
    /// tensor, whatever encoding it is written in. Either way the objective
    /// changes which values the encoder chooses and no bit layout, so the file
    /// stays a conforming TCF v1 file. Rejected by every format but
    /// [`OutputFormat::Tcf`].
    pub tcf_objective: Option<TcfObjective>,
    /// Measured task-metric sensitivity for the TCF profile's bump set
    ///
    /// Valid only alongside [`ConversionConfig::tcf_profile`], and only for a
    /// profile that names a bump encoding. Without it the bump set is ranked
    /// by the weight-error screen and no sensitivity field is written.
    pub tcf_sensitivity: Option<SensitivityMeasurement>,
    /// Measured task-metric sensitivity for a GGUF mix's promotion set
    ///
    /// Valid only alongside [`ConversionConfig::gguf_mix`]. Without it the
    /// promotions come from llama.cpp's positional rules, which measured
    /// nothing. With it the same byte budget is spent by measured task damage
    /// instead. See [`policy::mix_measured`].
    pub gguf_sensitivity: Option<SensitivityMeasurement>,
    /// llama.cpp mixed-precision allocation for a GGUF output
    ///
    /// Set alongside [`ConversionConfig::quantization`], whose method is the
    /// mix's base format. When this is set the per-tensor format comes from
    /// [`policy::assign_mix`] and the base method is used only for the tensors
    /// no rule moves.
    pub gguf_mix: Option<GgufMix>,
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
            tcf_objective: None,
            tcf_sensitivity: None,
            gguf_sensitivity: None,
            gguf_mix: None,
            verbose: false,
        }
    }

    /// Set the llama.cpp mixed-precision allocation for a GGUF output
    pub fn with_gguf_mix(mut self, mix: GgufMix) -> Self {
        self.gguf_mix = Some(mix);
        self
    }

    /// Set the measured task-metric sensitivity a GGUF mix's promotions follow
    pub fn with_gguf_sensitivity(mut self, measurement: SensitivityMeasurement) -> Self {
        self.gguf_sensitivity = Some(measurement);
        self
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

    /// Set the error objective the TCF quantizer minimizes
    ///
    /// Unset leaves every tensor on its own encoding's default. Set overrides
    /// that default on every tensor.
    pub fn with_tcf_objective(mut self, objective: TcfObjective) -> Self {
        self.tcf_objective = Some(objective);
        self
    }

    /// Set the measured task-metric sensitivity the bump set is ranked by
    pub fn with_tcf_sensitivity(mut self, measurement: SensitivityMeasurement) -> Self {
        self.tcf_sensitivity = Some(measurement);
        self
    }

    /// The TCF options a conversion runs with, from either flag.
    ///
    /// # Errors
    /// Neither was given. [`ConversionConfig::validate`] refuses that first,
    /// so reaching this is an internal error.
    pub fn tcf_options(&self) -> Result<TcfOptions> {
        // The objective rides on whichever flag named the encodings: it is
        // orthogonal to both, so applying it in one place keeps a flat run
        // and a profile run from diverging. `None` is carried as `None`, not
        // defaulted here: the default belongs to each TENSOR's encoding and
        // is resolved when that encoding is known.
        let objective = self.tcf_objective;
        match (self.tcf, self.tcf_profile) {
            (Some(options), _) => Ok(options.with_objective_override(objective)),
            (None, Some(profile)) => {
                Ok(TcfOptions::from_profile(profile).with_objective_override(objective))
            }
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
    /// - `--sensitivity` was given without a TCF profile that names a bump
    ///   encoding, or with a measurement taken at another encoding. A ranking
    ///   measured at one encoding does not transfer to another.
    /// - `--sensitivity` was given for a GGUF output with no mixed-precision
    ///   quantization, or with a measurement whose bit width does not suit the
    ///   mix's base format. See [`policy::check_measurement_encoding`].
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
                     nothing can dequantize. Implemented: q4_0, q4_1, q8_0, q2_k, q3_k, \
                     q4_k, q5_k, q6_k.",
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
        if let Some(mix) = self.gguf_mix {
            // A mix is a table of per-tensor GGML block types, and GGUF's
            // tensor directory is the only container here that records a type
            // per tensor.
            if self.output_format != OutputFormat::Gguf {
                bail!(
                    "Mixed precision {} writes GGUF only, but --format is {:?}. A mix \
                     assigns a different block type to different tensors, which only \
                     GGUF's tensor directory can record.",
                    mix.name(),
                    self.output_format
                );
            }
            if self.quantization.is_none() {
                bail!(
                    "Internal error: mixed precision {} reached convert with no base \
                     quantization method.",
                    mix.name()
                );
            }
        }
        // A measurement ranks a promotion set against a budget. A flat
        // quantization writes one format for every tensor, so it has no
        // promotion set and nothing to rank.
        if let Some(ref measurement) = self.gguf_sensitivity {
            if self.output_format != OutputFormat::Gguf {
                bail!(
                    "--sensitivity was given for a GGUF mix, but --format is {:?}.",
                    self.output_format
                );
            }
            let Some(mix) = self.gguf_mix else {
                bail!(
                    "--sensitivity needs a mixed-precision quantization for a GGUF output. A \
                     flat method writes one format for every tensor, so there is no promotion \
                     set for a measurement to rank. Accepted mixes: {}.",
                    GGUF_MIX_NAMES.join(", ")
                );
            };
            policy::check_measurement_encoding(mix, measurement)?;
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
            // A measurement ranks one set against one budget, or offers a
            // menu per tensor. Without a profile there is neither: nothing
            // names a base encoding to allocate against.
            if let Some(ref measurement) = self.tcf_sensitivity {
                let Some(profile) = self.tcf_profile else {
                    bail!(
                        "--sensitivity applies to --tcf-profile only. A measurement ranks the \
                         tensors a profile raises to its bump encoding, and --tcf-encoding \
                         writes one encoding for every tensor, so there is no set to rank. \
                         Accepted profiles: {}",
                        TCF_PROFILE_NAMES.join(", ")
                    );
                };
                // A flat profile has nothing to raise a tensor TO, so a file
                // measured at one encoding can do nothing with it. A file
                // measuring several is a menu, and a flat profile allocated
                // from one can still put a tensor on a cheaper rung.
                if profile.bump.is_none() && !measurement.is_ladder() {
                    bail!(
                        "--sensitivity was given with profile {}, which names no bump \
                         encoding, and the file measures one encoding ({}). Every tensor is \
                         written at {:?}, so there is no bump set for a measurement to rank. \
                         A file measuring several encodings per tensor is a menu, and a flat \
                         profile can be allocated from one.",
                        profile.name,
                        measurement.baseline().encoding,
                        profile.base
                    );
                }
                measurement.check_encoding(profile.base)?;
            }
        } else if self.tcf.is_some() || self.tcf_profile.is_some() {
            bail!(
                "--tcf-encoding and --tcf-profile apply to --format tcf only, but --format \
                 is {:?}.",
                self.output_format
            );
        } else if self.tcf_sensitivity.is_some() {
            bail!(
                "--sensitivity applies to --format tcf only, but --format is {:?}. It ranks a \
                 TCF profile's bump set, and no other format has one.",
                self.output_format
            );
        } else if self.tcf_objective.is_some() {
            bail!(
                "--tcf-objective applies to --format tcf only, but --format is {:?}. It names \
                 the error the TCF quantizer minimizes, and every other format's quantizer is \
                 not TCF's. Accepted objectives: {}",
                self.output_format,
                TCF_OBJECTIVE_NAMES.join(", ")
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
    /// The format every tensor was written in, and the bpw that achieved
    ///
    /// Populated for a GGUF or SafeTensors output, not only for a
    /// mixed-precision one: a comparison has to state what it compared, and a
    /// flat run still stores some tensors unquantized. EMPTY for a TCF output,
    /// whose writer produces its payload in a pass this loop does not see.
    pub mix: MixReport,
    /// Which tensors a mixed-precision rule promoted, and every number the
    /// rules were given
    ///
    /// `None` for a run with no mix. A mixed run's allocation is a function of
    /// a layer count and a walk order, so a report that states the formats
    /// without stating those cannot be checked against llama.cpp.
    pub allocation: Option<MixAllocation>,
    /// What a measured GGUF allocation did, against the positional one
    ///
    /// `Some` only for a `--sensitivity` run of a GGUF mix. It carries the
    /// promotion counts of both policies, their overlap, the parameters the
    /// measurement wanted to promote and GGUF's super-block could not hold,
    /// and the bpw each achieved.
    pub measured_allocation: Option<MeasuredMixAllocation>,
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

    // TCF is quantized once, from full-precision weights. A GGUF or TCF input
    // is already quantized, so writing TCF from one would quantize twice: the
    // second pass would work from values the first pass already rounded.
    // Refused here, before any output file exists.
    if config.output_format == OutputFormat::Tcf
        && matches!(model.format(), InputFormat::Gguf | InputFormat::Tcf)
    {
        bail!(
            "TCF is produced from a full-precision checkpoint, and '{}' is {:?}. Convert \
             from the SafeTensors source instead: quantizing an already-quantized tensor \
             loses accuracy twice.",
            config.input_path,
            model.format()
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
    let quant_type = config.quantization.as_ref().map(|q| q.method.quant_type());

    let quantizer = config
        .quantization
        .as_ref()
        .map(|qc| Quantizer::new(qc.clone()));

    // Step 3: Create the format writer
    log::info!("Creating output writer...");
    let mut writer = formats::create_writer(config.output_format, config.max_shard_size)?;

    // Get model metadata
    let metadata = model.metadata();

    // The mix rules read the architecture, so the value written to
    // `general.architecture` is kept rather than resolved a second time.
    let mut gguf_arch: Option<String> = None;

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
        gguf_arch = Some(arch.clone());
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

        // `general.file_type` names the file, and only that: a reader takes
        // each tensor's format from the tensor directory. The values are
        // llama.cpp's `llama_ftype` enum, which is NOT the GGML type id — a
        // file written with GGML ids reports Q3_K_M for a Q4_K model.
        if let Some(mix) = config.gguf_mix {
            gguf_writer.set_file_type(mix.llama_ftype());
        } else if let Some(qt) = quant_type {
            let file_type = match qt {
                QuantType::Q4_0 => 2,
                QuantType::Q4_1 => 3,
                QuantType::Q8_0 => 7,
                QuantType::Q5_0 => 8,
                QuantType::Q5_1 => 9,
                QuantType::Q2K => 10,
                // llama.cpp names no ftype for a flat K-quant, because its own
                // K-quant paths are all mixes. The nearest published value is
                // the `_S` variant of the same base format.
                QuantType::Q3K => 11,
                QuantType::Q4K => 14,
                QuantType::Q5K => 16,
                QuantType::Q6K => 18,
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
        if let Some(measurement) = config.tcf_sensitivity.clone() {
            log::info!(
                "TCF sensitivity: {} measured tensors over {} evaluation rows at {}, \
                 calibration {:#010x}",
                measurement.measured(),
                measurement.baseline().eval_rows,
                measurement.baseline().encoding,
                measurement.calibration_id()
            );
            tcf_writer = tcf_writer.with_sensitivity(measurement);
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
        // Stated on every run, uniform included: a reader comparing two files
        // has to be able to see which error each one minimized, and a line
        // that appears only sometimes cannot answer that. The writer states
        // the RESULT per encoding once the allocation has fixed them; this
        // line states the policy those results come from.
        log::info!("{}", formats::objective_policy(options));
        log::info!(
            "TCF activation contract: exact F32 activations, F32 accumulate, F32 out; a \
             conversion that calibrated no activations tunes nothing for a \
             quantized-activation kernel. workload_profile_id is left unset: compressr \
             measures no workload. calibration_id is set only on a --sensitivity run, from \
             the measurement that run read."
        );
        writer = Box::new(tcf_writer);
    }

    // Collect tensor info for registration (need to collect because we iterate twice)
    let tensor_info: Vec<(String, UnifiedTensorInfo)> = model.tensors().collect();

    // Decide once per tensor what happens to it. The registration loop reserves
    // the byte count and the write loop produces the bytes; a tensor the two
    // loops disagreed about would shift every later offset in the output file
    // without raising an error, so both read this one list.
    let mut allocation: Option<policy::MixAllocation> = None;
    let mut measured_allocation: Option<policy::MeasuredMixAllocation> = None;
    let plans: Vec<TensorPlan> = match config.gguf_mix {
        // A mix assigns a format per tensor from its role and its layer, so
        // the plan list is built from that assignment rather than from one
        // method applied to everything.
        Some(mix) => {
            let arch = gguf_arch.as_deref().unwrap_or_default();
            let facts = policy::mix_model_facts(arch, &metadata, Path::new(&config.input_path))?;
            log::info!(
                "Mixed precision {}: base {:?}, {} layers from {}, {} expert(s), 70B \
                 attention-V rule {}",
                mix.name(),
                mix.base(),
                facts.n_layer,
                facts.layer_count_source,
                facts.n_expert,
                if facts.is_70b { "on" } else { "off" }
            );
            // Without --sensitivity this is llama.cpp's positional
            // allocation and nothing else, byte for byte.
            let (assigned, computed) = match config.gguf_sensitivity.as_ref() {
                Some(measurement) => {
                    let measured_tensors: Vec<policy::MeasuredTensor<'_>> = tensor_info
                        .iter()
                        .map(|(name, info)| (name.as_str(), info.shape.as_slice(), info.dtype))
                        .collect();
                    let (assigned, positional, measured) =
                        policy::assign_mix_measured(mix, &measured_tensors, &facts, measurement)?;
                    log::info!("{measured}");
                    measured_allocation = Some(measured);
                    (assigned, positional)
                }
                None => policy::assign_mix_reported(
                    mix,
                    tensor_info
                        .iter()
                        .map(|(name, info)| (name.as_str(), info.shape.as_slice())),
                    &facts,
                ),
            };
            // A model built from several transformer stacks is not the shape
            // upstream's rules were written for. The conversion still runs —
            // the per-role assignments are unchanged — but the log says what
            // the result is, so nobody reads the file name as a claim.
            if computed.is_multi_component() {
                log::warn!("{}", computed.fidelity());
            }
            log::info!("{computed}");
            allocation = Some(computed);
            tensor_info
                .iter()
                .zip(&assigned)
                .map(|((_, info), assigned)| {
                    policy::plan_mixed_tensor(&info.shape, info.dtype, *assigned)
                })
                .collect()
        }
        None => tensor_info
            .iter()
            .map(|(_, info)| policy::plan_tensor(&info.shape, info.dtype, quant_type))
            .collect(),
    };

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
    let mut mix = MixReport::default();

    for ((name, info), plan) in tensor_info.iter().zip(&plans) {
        if config.verbose {
            log::debug!("Processing tensor: {} {:?} {:?}", name, info.shape, plan);
        }

        // Read tensor data
        let tensor_data = model.read_tensor_bytes(name)?;

        // Quantize, cast, or pass through untouched, following the plan the
        // registration loop above already reserved space for.
        let output_data = match *plan {
            // The plan names the type, not the configuration: a mixed-precision
            // run assigns a different one per tensor, and the registration loop
            // above reserved bytes for exactly this type.
            TensorPlan::Quantize(quant_type) => {
                let quantizer = quantizer
                    .as_ref()
                    .context("Internal error: a tensor was planned for quantization but no quantizer is configured")?;
                let quantized = quantizer.quantize_tensor_as(
                    &tensor_data,
                    &info.shape,
                    info.dtype,
                    quant_type,
                )?;
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
        // The TCF writer produces its payload in a later pass of its own, so
        // `output_data` there is the SOURCE the tensor was planned from and not
        // the bytes the file holds. Recording it would report a bpw the file
        // does not have, so the TCF path contributes no rows.
        if config.output_format != OutputFormat::Tcf {
            mix.record(MixEntry {
                name: name.clone(),
                format: policy::plan_format_label(*plan, config.stored_dtype(info.dtype)),
                elements: info.numel as u64,
                bytes: output_data.len() as u64,
                fallback: match *plan {
                    TensorPlan::StoreF32(reason) => Some(reason),
                    _ => None,
                },
            });
        }

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
        mix,
        allocation,
        measured_allocation,
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
            tcf: info.tcf,
        })
        .collect();

    let total_size = model.total_size_bytes();
    let metadata = model.metadata();
    let tcf = model.tcf_facts().cloned();

    Ok(ModelInfo {
        path: path.to_string(),
        format: format!("{:?}", model.format()),
        tensor_count: tensors.len(),
        total_size_bytes: total_size as u64,
        metadata: Some(metadata),
        tensors,
        tcf,
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
    /// TCF's file-level structure, or `None` for every other format
    pub tcf: Option<TcfFileFacts>,
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
    /// Size in bytes, as the tensor is stored in its input file
    pub size_bytes: u64,
    /// What a TCF record states beyond shape and element type
    pub tcf: Option<TcfTensorFacts>,
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

        // A TCF file states more than a tensor list: modules with their own
        // encoding preference and residency, an encoding per tensor, and the
        // contract, calibration and workload records the tensors are read
        // against. Flattening that into the GGUF shape would hide the part of
        // the file a placement decision is made from.
        if let Some(ref tcf) = self.tcf {
            writeln!(f, "\nTCF v{}.{}:", tcf.major, tcf.minor)?;
            writeln!(f, "  Schema id: {}", tcf.schema_id)?;
            writeln!(f, "  File length: {} bytes", tcf.file_len)?;
            writeln!(f, "  Data section starts at: {}", tcf.data_off)?;
            writeln!(
                f,
                "  Contracts: {} | Calibrations: {} | Relations: {} | Workload profiles: {}",
                tcf.contract_count, tcf.calibration_count, tcf.relation_count, tcf.workload_count
            )?;
            writeln!(
                f,
                "  Tensors at a fallback encoding: {}",
                tcf.fallback_tensors
            )?;

            writeln!(f, "\n  Encodings:")?;
            for (encoding, count) in &tcf.encoding_histogram {
                writeln!(f, "    {:<14} {} tensor(s)", encoding, count)?;
            }

            writeln!(f, "\n  Modules ({}):", tcf.modules.len())?;
            for module in &tcf.modules {
                let name = if module.name.is_empty() {
                    "(unnamed)"
                } else {
                    &module.name
                };
                writeln!(
                    f,
                    "    #{} {} — {}, {} tensor(s), residency {}",
                    module.module_id,
                    name,
                    module.module_role,
                    module.tensor_count,
                    module.default_residency
                )?;
                if !module.preferred_encodings.is_empty() {
                    writeln!(
                        f,
                        "      preferred: {}",
                        module.preferred_encodings.join(" > ")
                    )?;
                }
                if let Some(ref fallback) = module.fallback_encoding {
                    writeln!(f, "      fallback: {}", fallback)?;
                }
                writeln!(f, "      policy: {}", module.policy_flags)?;
            }
        }

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

    /// The objective names the error TCF's own quantizer minimizes, so it is
    /// refused for every other format rather than silently ignored.
    #[test]
    fn test_tcf_objective_is_refused_for_another_format() {
        let config = ConversionConfig::new("input/".into(), "out.gguf".into(), OutputFormat::Gguf)
            .with_tcf_objective(TcfObjective::Squared);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("--tcf-objective"), "{err}");
        assert!(err.contains("--format tcf only"), "{err}");
        assert!(err.contains("rms-plus-abs"), "{err}");
    }

    /// A run that names no objective leaves every tensor on its own
    /// encoding's default, and an explicit one reaches a profile run and a
    /// flat run alike.
    #[test]
    fn test_the_objective_defaults_per_encoding_and_reaches_both_tcf_flags() {
        let flagless = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf(tcf_options());
        assert!(flagless.validate().is_ok());
        let flagless = flagless.tcf_options().unwrap();
        assert_eq!(flagless.objective_override, None);
        // Q4S32_T64 is a 4-bit rung, whose default is the unweighted
        // objective every earlier run wrote.
        assert_eq!(
            flagless.objective_for(tcf_core::NativeEncoding::Q4S32T64),
            TcfObjective::Uniform
        );
        // The same flagless run resolves a 6-bit rung to the weighted
        // objective a measurement chose for it.
        assert_eq!(
            flagless.objective_for(tcf_core::NativeEncoding::Q6S16DT64),
            TcfObjective::RmsPlusAbs
        );

        let flat = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf(tcf_options())
            .with_tcf_objective(TcfObjective::Squared);
        assert!(flat.validate().is_ok());
        assert_eq!(
            flat.tcf_options().unwrap().objective_override,
            Some(TcfObjective::Squared)
        );

        let profiled = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf_profile(parse_tcf_profile("compact").unwrap())
            .with_tcf_objective(TcfObjective::RmsPlusAbs);
        assert!(profiled.validate().is_ok());
        let profiled = profiled.tcf_options().unwrap();
        // An explicit flag wins on BOTH halves of a mixed profile.
        for encoding in [
            tcf_core::NativeEncoding::Q4AS32DT64,
            tcf_core::NativeEncoding::Q6S16DT64,
        ] {
            assert_eq!(
                profiled.objective_for(encoding),
                TcfObjective::RmsPlusAbs,
                "{encoding:?}"
            );
            assert!(profiled.objective_choice(encoding).1.is_explicit());
        }
        // The objective moves no encoding: the profile still names its base.
        assert_eq!(profiled.encoding, tcf_core::NativeEncoding::Q4AS32DT64);
    }

    /// A mixed profile with no flag resolves per tensor, never once per file:
    /// its 4-bit base and its 6-bit bump take different objectives.
    #[test]
    fn test_a_mixed_profile_resolves_the_objective_per_tensor() {
        let config = ConversionConfig::new("input/".into(), "out.tcf".into(), OutputFormat::Tcf)
            .with_tcf_profile(parse_tcf_profile("compact").unwrap());
        assert!(config.validate().is_ok());
        let options = config.tcf_options().unwrap();
        assert_eq!(
            options.objective_for(tcf_core::NativeEncoding::Q4AS32DT64),
            TcfObjective::Uniform
        );
        assert_eq!(
            options.objective_for(tcf_core::NativeEncoding::Q6S16DT64),
            TcfObjective::RmsPlusAbs
        );
        assert!(!options
            .objective_choice(tcf_core::NativeEncoding::Q6S16DT64)
            .1
            .is_explicit());
        // The policy line has to say the file is mixed rather than name one
        // objective for all of it.
        let line = objective_policy(options);
        assert!(line.contains("MIXED precision"), "{line}");
        assert!(line.contains("does NOT reproduce byte for byte"), "{line}");
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
            mix: MixReport::default(),
            allocation: None,
            measured_allocation: None,
        };

        let display = format!("{}", stats);
        assert!(display.contains("100 tensors"));
        assert!(display.contains("2.00x compression"));
    }
}
