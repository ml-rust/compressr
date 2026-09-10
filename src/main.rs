//! compressr CLI - Model conversion and compression tool
//!
//! High-performance model conversion and compression utility.
//!
//! # Usage
//!
//! ```bash
//! # Convert to GGUF with Q4_K quantization
//! compressr convert model/ --format gguf --quantization q4_k
//!
//! # Or with llama.cpp's Q4_K_M mix: Q6_K on the output weight and on
//! # selected attention-V and FFN-down tensors, Q4_K on the rest
//! compressr convert model/ --format gguf --quantization q4_k_m
//!
//! # Name the architecture when the checkpoint's config.json does not
//! compressr convert model/ --format gguf --quantization q4_k --arch voxcpm2
//!
//! # Cast an fp32 checkpoint to bf16, keeping it loadable
//! compressr convert model/ --format safetensors --dtype bf16
//!
//! # Quantize a full-precision checkpoint into TCF v1
//! compressr convert model/ --format tcf --tcf-encoding q4s32t64 \
//!     --tcf-preferred-encoding q4s32t64
//!
//! # Or let a profile allocate the encodings, mixed precision
//! compressr convert model/ --format tcf --tcf-profile balanced
//!
//! # Override every encoding's default objective with one of your own
//! compressr convert model/ --format tcf --tcf-profile balanced \
//!     --tcf-objective rms-plus-abs
//!
//! # Rank that profile's bump set by a measured task-metric delta instead of
//! # by the weight-error screen
//! compressr convert model/ --format tcf --tcf-profile balanced \
//!     --sensitivity sensitivity.jsonl
//!
//! # Shard the output at 2 GB instead of the 5 GB default
//! compressr convert model/ --format safetensors --dtype bf16 --max-shard-size 2GB
//!
//! # Convert a PyTorch state dict, weight-norm pairs folded on read
//! compressr convert checkpoint.pth --format safetensors
//!
//! # Get model info. A TCF file reports its modules, its per-tensor
//! # encodings, and the records a placement decision reads.
//! compressr info model/
//! compressr info model.tcf --verbose
//! ```
//!
//! The `optimize` subcommand parses its flags but every method it accepts —
//! GPTQ, AWQ, NF4, FP8 — is unimplemented, so it always errors. It exists for
//! the CLI surface, not for use.

mod quant_arg;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use compressr::{
    parse_sensitivity, parse_tcf_encoding, parse_tcf_objective, parse_tcf_profile,
    ConversionConfig, OutputFormat, QuantizationConfig, QuantizationMethod, SensitivityMeasurement,
    TcfObjective, TcfOptions, TcfProfile,
};
use quant_arg::parse_quantization;

#[derive(Parser)]
#[command(name = "compressr")]
#[command(version = compressr::VERSION)]
#[command(about = "High-performance model conversion and compression", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Format conversion and dtype casting (f32/f16/bf16)
    Convert {
        /// Input model path
        #[arg(value_name = "INPUT")]
        input: String,

        /// Output format (gguf, safetensors, tcf, awq, gptq, exl2, fp8)
        #[arg(long, value_name = "FORMAT")]
        format: String,

        /// Output path (default: "<INPUT>.gguf", or a "<INPUT>-<dtype>" checkpoint directory)
        #[arg(short, long, value_name = "PATH")]
        output: Option<String>,

        /// Quantization level (q4_0, q4_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k, nf4, fp8)
        ///
        /// Also accepts llama.cpp's mixed-precision names — q4_k_s, q4_k_m,
        /// q5_k_s, q5_k_m — which assign a different GGUF block format to
        /// different tensors by role and by layer. GGUF output only.
        ///
        /// Mutually exclusive with --dtype.
        #[arg(long, value_name = "LEVEL", conflicts_with = "dtype")]
        quantization: Option<String>,

        /// Cast every float tensor to this element type (f32, f16, bf16)
        ///
        /// Non-float tensors pass through untouched. This is a change of
        /// element type, not quantization, so it cannot be combined with
        /// --quantization.
        #[arg(long, value_name = "DTYPE", conflicts_with = "quantization")]
        dtype: Option<String>,

        /// Largest size of one output shard (e.g. 5GB, 500MB, 1GiB)
        ///
        /// Output over this size is written as numbered shards plus a
        /// model.safetensors.index.json. Output under it stays one file with no
        /// index. Pass "none" to write one file however large. GGUF ignores it.
        #[arg(long, value_name = "SIZE", default_value = "5GB")]
        max_shard_size: String,

        /// TCF encoding (q4s32t64, q4as32t64, q4as32dt64, q4as64t64, q6s32t64, q6s16dt64, q8s32t64)
        ///
        /// Required by --format tcf. Bit width alone does not name an
        /// encoding: symmetry and group size are part of it, so this is a
        /// separate flag from --quantization, which names GGUF block formats.
        #[arg(long, value_name = "ENCODING")]
        tcf_encoding: Option<String>,

        /// Encoding the TCF module records as preferred
        ///
        /// Required by --format tcf. TCF records a fallback reason on every
        /// tensor whose encoding differs from this one, and no checkpoint
        /// carries the preference, so compressr never defaults it.
        #[arg(long, value_name = "ENCODING")]
        tcf_preferred_encoding: Option<String>,

        /// TCF allocation profile (ref, high, balanced, compact)
        ///
        /// The alternative to --tcf-encoding, and mutually exclusive with it.
        /// A profile is mixed precision: a base encoding for every quantizable
        /// tensor, and a bump encoding on the worst-reconstructed share of the
        /// parameters. That share is ranked by a weight-error screen unless
        /// --sensitivity supplies a measured task-metric delta.
        #[arg(long, value_name = "PROFILE")]
        tcf_profile: Option<String>,

        /// TCF error objective (uniform, squared, rms-plus-abs)
        ///
        /// --format tcf only. Names WHICH error the quantizer minimizes:
        /// uniform is w = 1, the plain squared reconstruction error; squared
        /// is w = x^2; rms-plus-abs is w = sqrt(mean(x^2)) + |x| over each
        /// group. Unset, each tensor takes the default its own ENCODING
        /// carries, so a mixed-precision profile minimizes each half of the
        /// file under the objective measured for it. Set, this objective
        /// OVERRIDES that default on every tensor. Either way it changes
        /// which values the encoder chooses and no bit layout, so the file it
        /// writes is a normal conforming TCF file.
        #[arg(long, value_name = "OBJECTIVE")]
        tcf_objective: Option<String>,

        /// Measured task-metric sensitivity, as JSON lines
        ///
        /// One "baseline" record naming the model, the evaluation set and the
        /// encoding measured, then one "sensitivity" record per tensor
        /// carrying its task delta. Whatever the promotion set is, it is
        /// ranked by measured damage per element instead of by a rule that
        /// measured nothing.
        ///
        /// With --format tcf: requires --tcf-profile with a bump encoding.
        /// The profile's bump set is ranked by the measurement, and
        /// sensitivity_delta, sensitivity_samples and calibration_id are
        /// written from it. Without the flag the ranking is a weight-error
        /// screen and those fields stay unset.
        ///
        /// With --format gguf: requires a mixed-precision --quantization
        /// (q4_k_s, q4_k_m, q5_k_s, q5_k_m). The mix's promotions are then
        /// ranked by the measurement, spending exactly the bytes llama.cpp's
        /// positional rules would have spent on the same checkpoint, so the
        /// two artifacts are size matched. Tensors whose row is not a whole
        /// number of 256-element super-blocks cannot be promoted at all, and
        /// are counted and reported rather than skipped.
        #[arg(long, value_name = "FILE")]
        sensitivity: Option<String>,

        /// Architecture written to GGUF's general.architecture (e.g. llama, qwen2)
        ///
        /// Defaults to the "model_type" field of the checkpoint's config.json.
        /// Required when the checkpoint carries no architecture of its own.
        #[arg(long, value_name = "NAME")]
        arch: Option<String>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Advanced calibration-based compression (GPTQ, AWQ)
    Optimize {
        /// Input model path
        #[arg(value_name = "INPUT")]
        input: String,

        /// Quantization method (gptq, awq, nf4, fp8)
        #[arg(long, value_name = "METHOD")]
        method: String,

        /// Number of bits (2, 3, 4, 8)
        #[arg(long, value_name = "BITS", default_value = "4")]
        bits: u8,

        /// Group size for quantization
        #[arg(long, value_name = "SIZE", default_value = "128")]
        group_size: usize,

        /// Calibration dataset (wikitext2, c4, ptb)
        #[arg(long, value_name = "DATASET")]
        calib_dataset: Option<String>,

        /// Number of calibration samples
        #[arg(long, value_name = "N", default_value = "128")]
        calib_samples: usize,

        /// Output path
        #[arg(long, value_name = "PATH")]
        out: String,

        /// Device for calibration (cpu, cuda:0, etc.)
        #[arg(long, value_name = "DEVICE", default_value = "cpu")]
        device: String,

        /// Largest size of one output shard (e.g. 5GB, 500MB, 1GiB)
        ///
        /// Pass "none" to write one file however large.
        #[arg(long, value_name = "SIZE", default_value = "5GB")]
        max_shard_size: String,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Display model information
    Info {
        /// Model path
        #[arg(value_name = "MODEL")]
        model: String,

        /// Show all tensors (verbose)
        #[arg(short, long)]
        verbose: bool,

        /// Show only tensor count and total size
        #[arg(short, long)]
        summary: bool,
    },
}

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Convert {
            input,
            format,
            output,
            quantization,
            dtype,
            max_shard_size,
            tcf_encoding,
            tcf_preferred_encoding,
            tcf_profile,
            tcf_objective,
            sensitivity,
            arch,
            verbose,
        } => {
            let output_format = parse_format(&format)?;
            let max_shard_size = compressr::parse_shard_size(&max_shard_size)?;

            // Parse both before defaulting the output path, so a bad value
            // fails before any directory is created.
            let target_dtype = dtype
                .as_deref()
                .map(compressr::parse_target_dtype)
                .transpose()?;
            let quant_choice = quantization
                .as_deref()
                .map(parse_quantization)
                .transpose()?;
            let tcf_options =
                parse_tcf_options(tcf_encoding.as_deref(), tcf_preferred_encoding.as_deref())?;
            let tcf_profile = tcf_profile.as_deref().map(parse_tcf_profile).transpose()?;
            let tcf_objective = tcf_objective
                .as_deref()
                .map(parse_tcf_objective)
                .transpose()?;
            let sensitivity = sensitivity.as_deref().map(read_sensitivity).transpose()?;

            let output_path = output.unwrap_or_else(|| {
                default_output_path(
                    &input,
                    output_format,
                    dtype.as_deref(),
                    quantization.as_deref(),
                )
            });

            let mut config = ConversionConfig::new(input, output_path, output_format)
                .with_max_shard_size(max_shard_size)
                .with_verbose(verbose);

            if let Some(quant_choice) = quant_choice {
                config = config.with_quantization(quant_choice.config);
                if let Some(mix) = quant_choice.mix {
                    config = config.with_gguf_mix(mix);
                }
            }
            if let Some(target_dtype) = target_dtype {
                config = config.with_target_dtype(target_dtype);
            }
            if let Some(arch) = arch {
                config = config.with_arch(arch);
            }
            if let Some(tcf_options) = tcf_options {
                config = config.with_tcf(tcf_options);
            }
            if let Some(tcf_profile) = tcf_profile {
                config = config.with_tcf_profile(tcf_profile);
            }
            if let Some(tcf_objective) = tcf_objective {
                config = config.with_tcf_objective(tcf_objective);
            }
            let measured = sensitivity.clone();
            // A measurement ranks whichever promotion set the output format
            // has: a TCF profile's bump set, or a GGUF mix's promotions.
            if let Some(measurement) = sensitivity {
                config = match output_format {
                    OutputFormat::Gguf => config.with_gguf_sensitivity(measurement),
                    _ => config.with_tcf_sensitivity(measurement),
                };
            }

            let stats = compressr::convert(config)?;
            println!("\n{}", stats);
            // A comparison has to state what it compared, so the formats the
            // conversion actually wrote are printed, not the ones the name
            // implies. `--verbose` adds the per-tensor listing behind them.
            // A TCF run records no rows: its payload is written in a later
            // pass, so its byte counts are not the file's.
            if !stats.mix.entries().is_empty() {
                println!("{}", stats.mix);
            }
            // A mix names an upstream allocation, so the file name carries a
            // claim about which tensors were promoted. Printing the inputs the
            // rules were given — the layer count per stack and where each was
            // read — is what makes that claim checkable, and the fidelity line
            // says outright when the model is not the shape upstream's rules
            // assume.
            if let Some(allocation) = &stats.allocation {
                println!("{allocation}");
            }
            // With --sensitivity the line above is the allocation llama.cpp's
            // rules WOULD have made, which is the budget this run matched.
            // What was written is this.
            if let Some(measured) = &stats.measured_allocation {
                println!("{measured}");
            }
            if verbose && !stats.mix.entries().is_empty() {
                // A mix moves a tensor OFF its base format, so marking those
                // lines is what makes the listing an audit: the marked tensors
                // ARE the allocation. The promoted formats come from the
                // allocation itself rather than from "anything not the base",
                // because a passed-through integer buffer is not the base
                // format either and was promoted by nothing.
                let promoted: std::collections::BTreeSet<&str> = stats
                    .allocation
                    .iter()
                    .flat_map(|allocation| allocation.components.iter())
                    .flat_map(|component| component.promoted_formats.keys())
                    .map(String::as_str)
                    .collect();
                let base = stats
                    .allocation
                    .as_ref()
                    .map(|allocation| allocation.base_format.as_str());
                for entry in stats.mix.entries() {
                    let note = match (entry.fallback, base) {
                        (Some(reason), _) => format!("  (unquantized: {reason:?})"),
                        (None, Some(base)) if promoted.contains(entry.format.as_str()) => {
                            format!("  (promoted from {base})")
                        }
                        _ => String::new(),
                    };
                    println!(
                        "  {:<60} {:<6} {:>12} bytes{}",
                        entry.name, entry.format, entry.bytes, note
                    );
                }
            }
            if output_format == OutputFormat::Tcf {
                // Printed on a flagless run too: the summary already names the
                // encoding, and the objective is the other half of what the
                // encoder was told to do. The line states the POLICY and every
                // encoding it covers, never one objective for a mixed file;
                // the conversion log states what each encoding resolved to.
                match tcf_objective_policy(tcf_options, tcf_profile, tcf_objective) {
                    Ok(policy) => println!("{policy}"),
                    Err(error) => println!("TCF objective: {error}"),
                }
                println!(
                    "TCF activation contract: exact F32 activations, F32 accumulate, F32 out. \
                     workload_profile_id is unset: compressr measures no workload. No \
                     activation calibration was run either, which is what that contract says."
                );
            }
            if let Some(profile) = tcf_profile {
                println!(
                    "TCF profile {}: base {:?}, bump {:?} over the top {:.0}% of quantizable \
                     parameters, counted in elements.",
                    profile.name,
                    profile.base,
                    profile.bump,
                    profile.bump_fraction * 100.0
                );
                match measured {
                    Some(ref measurement) => println!(
                        "That share was ranked by a MEASURED task-metric delta per element: {} \
                         tensors measured over {} evaluation rows at {}. sensitivity_delta, \
                         sensitivity_samples and calibration_id are written from that \
                         measurement, which this file declares as calibration {:#010x}. \
                         sensitivity_ci95 is unset: the measurement carries no confidence \
                         interval, and the calibration's dataset digest and acceptance margin \
                         are unset because the measurement carries neither.",
                        measurement.measured(),
                        measurement.baseline().eval_rows,
                        measurement.baseline().encoding,
                        measurement.calibration_id()
                    ),
                    None => println!(
                        "That share was ranked by a WEIGHT-ERROR screen: each tensor was \
                         quantized at the base encoding, dequantized again, and scored by the \
                         relative RMS of the error. No task-metric calibration was run, so \
                         sensitivity_delta, sensitivity_ci95, sensitivity_samples and \
                         calibration_id are unset and this file is NOT profile-solved per \
                         PROFILES.md Section 4."
                    ),
                }
                println!(
                    "The ladder's {:.2} bpw for {} assumes that allocation. The bpw this file \
                     achieved is measured from the bytes written and is logged as \
                     \"bpw achieved\".",
                    profile.ladder_bpw, profile.name
                );
            }
        }

        Commands::Optimize {
            input,
            method,
            bits,
            group_size,
            calib_dataset,
            calib_samples,
            out,
            device,
            max_shard_size,
            verbose,
        } => {
            let method = parse_method(&method)?;
            let max_shard_size = compressr::parse_shard_size(&max_shard_size)?;
            let output_format = match method {
                QuantizationMethod::Gptq => OutputFormat::Gptq,
                QuantizationMethod::Awq => OutputFormat::Awq,
                QuantizationMethod::Nf4 => OutputFormat::Safetensors,
                QuantizationMethod::Fp8 => OutputFormat::Fp8,
                QuantizationMethod::Q4_0
                | QuantizationMethod::Q4_1
                | QuantizationMethod::Q8_0
                | QuantizationMethod::Q2K
                | QuantizationMethod::Q3K
                | QuantizationMethod::Q4K
                | QuantizationMethod::Q5K
                | QuantizationMethod::Q6K => OutputFormat::Gguf,
            };

            let quant_config = QuantizationConfig {
                method,
                bits,
                group_size,
                calibration_dataset: calib_dataset,
                calibration_samples: calib_samples,
            };

            let config = ConversionConfig::new(input, out, output_format)
                .with_quantization(quant_config)
                .with_max_shard_size(max_shard_size)
                .with_verbose(verbose);

            log::info!("Optimization method: {:?}", method);
            log::info!("Target device: {}", device);

            let stats = compressr::convert(config)?;
            println!("\n{}", stats);
        }

        Commands::Info {
            model,
            verbose,
            summary,
        } => {
            let info = compressr::get_model_info(&model)?;

            if summary {
                // Just show summary
                println!("Tensors: {}", info.tensor_count);
                println!(
                    "Size: {:.2} GB",
                    info.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            } else {
                // Show full info
                println!("{}", info);

                if verbose {
                    println!("\nTensors:");
                    println!("{:-<80}", "");
                    for tensor in &info.tensors {
                        let shape_str: String = tensor
                            .shape
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join("x");
                        // A TCF tensor is described by its encoding, not by an
                        // element type: `dtype` is the type it DECODES to, so
                        // printing that instead would say F32 for every
                        // tensor in a 4-bit file.
                        match tensor.tcf {
                            Some(ref tcf) => {
                                let bpw = tcf
                                    .bits_per_weight
                                    .map(|b| format!("{b:.2} bpw"))
                                    .unwrap_or_else(|| "raw".to_string());
                                println!(
                                    "  {} [{:>12}] {:>12} {:>10} {:>12}",
                                    tensor.name,
                                    shape_str,
                                    tcf.encoding,
                                    bpw,
                                    format_bytes(tensor.size_bytes),
                                );
                                println!(
                                    "      module {} | role {} | exec {} | residency {}",
                                    tcf.module, tcf.role, tcf.execution_role, tcf.residency_class
                                );
                                if let Some(ref reason) = tcf.fallback_reason {
                                    println!("      fallback: {}", reason);
                                }
                                if let Some((delta, ci95)) = tcf.sensitivity {
                                    println!("      sensitivity: {delta} +/- {ci95}");
                                }
                                if tcf.activation_contract_id != 0
                                    || tcf.calibration_id != 0
                                    || tcf.workload_profile_id != 0
                                {
                                    println!(
                                        "      contract {} | calibration {} | workload {}",
                                        tcf.activation_contract_id,
                                        tcf.calibration_id,
                                        tcf.workload_profile_id
                                    );
                                }
                            }
                            None => println!(
                                "  {} [{:>12}] {:>10} {:>12}",
                                tensor.name,
                                shape_str,
                                tensor.dtype,
                                format_bytes(tensor.size_bytes),
                            ),
                        }
                    }
                    println!("{:-<80}", "");
                }
            }
        }
    }

    Ok(())
}

/// Format bytes in human-readable form
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Build the default output path for `convert`.
///
/// GGUF is one self-contained file. Every other format is a checkpoint
/// directory, because the weights need their tokenizer and config beside them.
/// The directory is tagged with the dtype or quantization level so repeated
/// conversions of one model do not collide.
fn default_output_path(
    input: &str,
    output_format: OutputFormat,
    dtype: Option<&str>,
    quantization: Option<&str>,
) -> String {
    let base = input.trim_end_matches('/');

    // GGUF and TCF are each one self-contained file, named by its extension.
    if matches!(output_format, OutputFormat::Gguf | OutputFormat::Tcf) {
        return format!("{}.{}", base, output_format.extension());
    }

    let tag = dtype
        .or(quantization)
        .map(|t| t.to_lowercase())
        .unwrap_or_else(|| format!("{:?}", output_format).to_lowercase());

    format!("{}-{}", base, tag)
}

fn parse_format(s: &str) -> Result<OutputFormat> {
    match s.to_lowercase().as_str() {
        "gguf" => Ok(OutputFormat::Gguf),
        "awq" => Ok(OutputFormat::Awq),
        "gptq" => Ok(OutputFormat::Gptq),
        "exl2" => Ok(OutputFormat::Exl2),
        "fp8" => Ok(OutputFormat::Fp8),
        "safetensors" | "st" => Ok(OutputFormat::Safetensors),
        "tcf" => Ok(OutputFormat::Tcf),
        _ => anyhow::bail!(
            "Unsupported format: {}. Supported: gguf, safetensors, tcf, awq, gptq, exl2, fp8",
            s
        ),
    }
}

/// Parse the two TCF encoding flags, which are set together or not at all.
///
/// Returning `None` for neither leaves `ConversionConfig::validate` to refuse
/// `--format tcf` by name, so one message covers the CLI and the library API.
fn parse_tcf_options(
    encoding: Option<&str>,
    preferred: Option<&str>,
) -> Result<Option<TcfOptions>> {
    match (encoding, preferred) {
        (None, None) => Ok(None),
        (Some(encoding), Some(preferred)) => Ok(Some(TcfOptions::uniform(
            parse_tcf_encoding(encoding)?,
            parse_tcf_encoding(preferred)?,
        ))),
        (Some(_), None) => anyhow::bail!(
            "--tcf-encoding was given without --tcf-preferred-encoding. TCF records a \
             fallback reason on every tensor whose encoding differs from its module's \
             preferred one, so both are required."
        ),
        (None, Some(_)) => anyhow::bail!(
            "--tcf-preferred-encoding was given without --tcf-encoding, so nothing names \
             the encoding the weights are written in."
        ),
    }
}

/// What the summary says about the objective this TCF run quantized under.
///
/// Built from the same two flags the conversion is, so the summary cannot
/// name a policy the conversion did not run. It states whether the objective
/// came from the per-encoding defaults or from an explicit
/// `--tcf-objective`, and for a mixed-precision profile it names every
/// encoding rather than implying one objective covered the file.
///
/// # Errors
/// Neither TCF flag was given, which the conversion refuses first.
fn tcf_objective_policy(
    options: Option<TcfOptions>,
    profile: Option<TcfProfile>,
    objective: Option<TcfObjective>,
) -> Result<String> {
    let options = match (options, profile) {
        (Some(options), _) => options,
        (None, Some(profile)) => TcfOptions::from_profile(profile),
        (None, None) => anyhow::bail!(
            "--format tcf needs --tcf-encoding or --tcf-profile, and neither was given"
        ),
    };
    Ok(compressr::objective_policy(
        options.with_objective_override(objective),
    ))
}

/// Read and parse a `--sensitivity` file.
///
/// Parsed here rather than in the writer so a malformed or empty file fails
/// before any output is created, as every other conversion check does.
fn read_sensitivity(path: &str) -> Result<SensitivityMeasurement> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read the sensitivity file '{path}'"))?;
    parse_sensitivity(&text).with_context(|| format!("Failed to parse '{path}'"))
}

fn parse_method(s: &str) -> Result<QuantizationMethod> {
    match s.to_lowercase().as_str() {
        "gptq" => Ok(QuantizationMethod::Gptq),
        "awq" => Ok(QuantizationMethod::Awq),
        "nf4" => Ok(QuantizationMethod::Nf4),
        "fp8" => Ok(QuantizationMethod::Fp8),
        "q4_0" => Ok(QuantizationMethod::Q4_0),
        "q4_1" => Ok(QuantizationMethod::Q4_1),
        "q8_0" => Ok(QuantizationMethod::Q8_0),
        "q2_k" | "q2k" => Ok(QuantizationMethod::Q2K),
        "q3_k" | "q3k" => Ok(QuantizationMethod::Q3K),
        "q4_k" | "q4k" => Ok(QuantizationMethod::Q4K),
        "q5_k" | "q5k" => Ok(QuantizationMethod::Q5K),
        "q6_k" | "q6k" => Ok(QuantizationMethod::Q6K),
        _ => anyhow::bail!("Unsupported method: {}. Supported: gptq, awq, nf4, fp8, q4_0, q4_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k", s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_dtype_and_quantization_conflict_at_the_cli() {
        let err = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "safetensors",
            "--dtype",
            "bf16",
            "--quantization",
            "q4_k",
        ]);
        // `Cli` is not `Debug`, so `unwrap_err` is unusable here.
        let Err(err) = err else {
            panic!("--dtype and --quantization must conflict");
        };

        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
        assert!(err.to_string().contains("--dtype"), "error was: {}", err);
    }

    #[test]
    fn test_dtype_alone_parses() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "safetensors",
            "--dtype",
            "bf16",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert {
                dtype,
                quantization,
                ..
            } => {
                assert_eq!(dtype.as_deref(), Some("bf16"));
                assert_eq!(quantization, None);
            }
            _ => panic!("expected the convert subcommand"),
        }
    }

    #[test]
    fn test_convert_parses_arch() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "gguf",
            "--arch",
            "voxcpm2",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert { arch, .. } => assert_eq!(arch.as_deref(), Some("voxcpm2")),
            _ => panic!("expected the convert subcommand"),
        }
    }

    #[test]
    fn test_convert_parses_the_tcf_encoding_flags() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "tcf",
            "--tcf-encoding",
            "q4as64t64",
            "--tcf-preferred-encoding",
            "q6s32t64",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert {
                tcf_encoding,
                tcf_preferred_encoding,
                ..
            } => {
                let options =
                    parse_tcf_options(tcf_encoding.as_deref(), tcf_preferred_encoding.as_deref())
                        .unwrap()
                        .expect("both flags given");
                assert_eq!(options.encoding, parse_tcf_encoding("q4as64t64").unwrap());
                assert_eq!(
                    options.preferred_encoding,
                    parse_tcf_encoding("q6s32t64").unwrap()
                );
            }
            _ => panic!("expected the convert subcommand"),
        }
    }

    #[test]
    fn test_convert_parses_the_tcf_profile_flag() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "tcf",
            "--tcf-profile",
            "balanced",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert { tcf_profile, .. } => {
                let profile = parse_tcf_profile(tcf_profile.as_deref().expect("the flag")).unwrap();
                assert_eq!(profile.name, "TCF-BALANCED");
                assert_eq!(profile.bump_fraction, 0.15);
            }
            _ => panic!("expected the convert subcommand"),
        }
    }

    /// The flag parses, and a flagless run names no objective at all — every
    /// tensor then takes the default its own encoding carries.
    #[test]
    fn test_convert_parses_the_tcf_objective_flag() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "tcf",
            "--tcf-profile",
            "balanced",
            "--tcf-objective",
            "rms-plus-abs",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert { tcf_objective, .. } => {
                let objective =
                    parse_tcf_objective(tcf_objective.as_deref().expect("the flag")).unwrap();
                assert_eq!(objective, compressr::TcfObjective::RmsPlusAbs);
            }
            _ => panic!("expected the convert subcommand"),
        }

        let flagless = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "tcf",
            "--tcf-profile",
            "balanced",
        ])
        .unwrap();
        match flagless.command {
            Commands::Convert { tcf_objective, .. } => {
                assert!(tcf_objective.is_none());
                assert_eq!(
                    tcf_objective
                        .as_deref()
                        .map(parse_tcf_objective)
                        .transpose()
                        .unwrap(),
                    None
                );
            }
            _ => panic!("expected the convert subcommand"),
        }
    }

    /// The summary states where the objective came from, and a mixed profile
    /// is never reported as though one objective covered the whole file.
    #[test]
    fn test_the_summary_states_the_objective_policy() {
        let compact = parse_tcf_profile("compact").unwrap();
        let defaults = tcf_objective_policy(None, Some(compact), None).unwrap();
        assert!(defaults.contains("none named"), "{defaults}");
        assert!(defaults.contains("MIXED precision"), "{defaults}");
        assert!(
            defaults.contains("Q6S16D_T64 quantizes under rms-plus-abs"),
            "{defaults}"
        );
        assert!(
            defaults.contains("does NOT reproduce byte for byte"),
            "{defaults}"
        );

        let explicit =
            tcf_objective_policy(None, Some(compact), Some(compressr::TcfObjective::Squared))
                .unwrap();
        assert!(explicit.contains("explicit --tcf-objective"), "{explicit}");
        assert!(
            explicit.contains("overrides the per-encoding default"),
            "{explicit}"
        );

        assert!(tcf_objective_policy(None, None, None).is_err());
    }

    #[test]
    fn test_unknown_tcf_objective_is_refused() {
        let err = parse_tcf_objective("imatrix").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF objective: imatrix"), "{err}");
        assert!(err.contains("rms-plus-abs"), "{err}");
    }

    /// A profile name is a rung of the ladder, so an unknown one is refused
    /// with the accepted names.
    #[test]
    fn test_unknown_tcf_profile_is_refused() {
        let err = parse_tcf_profile("tiny").unwrap_err().to_string();
        assert!(err.contains("Unsupported TCF profile: tiny"), "{err}");
        assert!(err.contains("balanced"), "{err}");
    }

    /// Bit width alone names no TCF encoding, so an unknown value is refused
    /// with the accepted names rather than resolved to a near match.
    #[test]
    fn test_unknown_tcf_encoding_is_refused() {
        let err = parse_tcf_options(Some("q4"), Some("q4s32t64"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Unsupported TCF encoding: q4"), "{err}");
    }

    #[test]
    fn test_the_two_tcf_flags_are_set_together() {
        assert!(parse_tcf_options(None, None).unwrap().is_none());
        let err = parse_tcf_options(Some("q4s32t64"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--tcf-preferred-encoding"), "{err}");
        let err = parse_tcf_options(None, Some("q4s32t64"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--tcf-encoding"), "{err}");
    }

    #[test]
    fn test_default_output_path_tcf_is_a_file() {
        assert_eq!(
            default_output_path("llama-3-8b/", OutputFormat::Tcf, None, None),
            "llama-3-8b.tcf"
        );
    }

    #[test]
    fn test_parse_format_accepts_tcf() {
        assert_eq!(parse_format("tcf").unwrap(), OutputFormat::Tcf);
        let err = parse_format("tcf2").unwrap_err().to_string();
        assert!(err.contains("Supported: gguf, safetensors, tcf"), "{err}");
    }

    #[test]
    fn test_default_output_path_gguf_is_a_file() {
        assert_eq!(
            default_output_path("llama-3-8b/", OutputFormat::Gguf, None, Some("q4_k")),
            "llama-3-8b.gguf"
        );
    }

    #[test]
    fn test_default_output_path_safetensors_is_a_tagged_directory() {
        assert_eq!(
            default_output_path("llama-3-8b/", OutputFormat::Safetensors, Some("BF16"), None),
            "llama-3-8b-bf16"
        );
        assert_eq!(
            default_output_path("llama-3-8b", OutputFormat::Safetensors, None, Some("q4_k")),
            "llama-3-8b-q4_k"
        );
        assert_eq!(
            default_output_path("llama-3-8b", OutputFormat::Safetensors, None, None),
            "llama-3-8b-safetensors"
        );
    }

    #[test]
    fn test_max_shard_size_defaults_to_five_gigabytes() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "safetensors",
            "--dtype",
            "bf16",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert { max_shard_size, .. } => {
                assert_eq!(max_shard_size, "5GB");
                assert_eq!(
                    compressr::parse_shard_size(&max_shard_size).unwrap(),
                    Some(5_000_000_000)
                );
            }
            _ => panic!("expected the convert subcommand"),
        }
    }

    #[test]
    fn test_max_shard_size_none_disables_sharding() {
        let cli = Cli::try_parse_from([
            "compressr",
            "convert",
            "model/",
            "--format",
            "safetensors",
            "--max-shard-size",
            "none",
        ])
        .unwrap();

        match cli.command {
            Commands::Convert { max_shard_size, .. } => {
                assert_eq!(compressr::parse_shard_size(&max_shard_size).unwrap(), None);
            }
            _ => panic!("expected the convert subcommand"),
        }
    }

    #[test]
    fn test_parse_format_rejects_unknown() {
        assert!(parse_format("onnx").is_err());
        assert_eq!(
            parse_format("SAFETENSORS").unwrap(),
            OutputFormat::Safetensors
        );
    }
}
