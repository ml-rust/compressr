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
//! # Name the architecture when the checkpoint's config.json does not
//! compressr convert model/ --format gguf --quantization q4_k --arch voxcpm2
//!
//! # Cast an fp32 checkpoint to bf16, keeping it loadable
//! compressr convert model/ --format safetensors --dtype bf16
//!
//! # Shard the output at 2 GB instead of the 5 GB default
//! compressr convert model/ --format safetensors --dtype bf16 --max-shard-size 2GB
//!
//! # Get model info
//! compressr info model/
//! ```
//!
//! The `optimize` subcommand parses its flags but every method it accepts —
//! GPTQ, AWQ, NF4, FP8 — is unimplemented, so it always errors. It exists for
//! the CLI surface, not for use.

use anyhow::Result;
use clap::{Parser, Subcommand};
use compressr::{ConversionConfig, OutputFormat, QuantizationConfig, QuantizationMethod};

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

        /// Output format (gguf, safetensors, awq, gptq, exl2, fp8)
        #[arg(long, value_name = "FORMAT")]
        format: String,

        /// Output path (default: "<INPUT>.gguf", or a "<INPUT>-<dtype>" checkpoint directory)
        #[arg(short, long, value_name = "PATH")]
        output: Option<String>,

        /// Quantization level (q4_0, q4_1, q8_0, q4_k, q5_k, q6_k, nf4, fp8)
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
            let quant_config = quantization
                .as_deref()
                .map(parse_quantization)
                .transpose()?;

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

            if let Some(quant_config) = quant_config {
                config = config.with_quantization(quant_config);
            }
            if let Some(target_dtype) = target_dtype {
                config = config.with_target_dtype(target_dtype);
            }
            if let Some(arch) = arch {
                config = config.with_arch(arch);
            }

            let stats = compressr::convert(config)?;
            println!("\n{}", stats);
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
                        println!(
                            "  {} [{:>12}] {:>10} {:>12}",
                            tensor.name,
                            shape_str,
                            tensor.dtype,
                            format_bytes(tensor.size_bytes),
                        );
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

    if output_format == OutputFormat::Gguf {
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
        _ => anyhow::bail!(
            "Unsupported format: {}. Supported: gguf, safetensors, awq, gptq, exl2, fp8",
            s
        ),
    }
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
        "q4_k" | "q4k" => Ok(QuantizationMethod::Q4K),
        "q5_k" | "q5k" => Ok(QuantizationMethod::Q5K),
        "q6_k" | "q6k" => Ok(QuantizationMethod::Q6K),
        _ => anyhow::bail!("Unsupported method: {}. Supported: gptq, awq, nf4, fp8, q4_0, q4_1, q8_0, q4_k, q5_k, q6_k", s),
    }
}

fn parse_quantization(s: &str) -> Result<QuantizationConfig> {
    // Parse quantization levels like "q4_0", "q4_k", "q4_k_m", etc.
    let s_lower = s.to_lowercase();

    // Extract base quantization type (ignoring _s, _m, _l suffixes for now)
    let base = s_lower.trim_end_matches(['s', 'm', 'l', '_']);

    let method = match base {
        "q4_0" | "q4.0" => QuantizationMethod::Q4_0,
        "q4_1" | "q4.1" => QuantizationMethod::Q4_1,
        "q8_0" | "q8.0" => QuantizationMethod::Q8_0,
        "q4_k" | "q4k" => QuantizationMethod::Q4K,
        "q5_k" | "q5k" => QuantizationMethod::Q5K,
        "q6_k" | "q6k" => QuantizationMethod::Q6K,
        "nf4" => QuantizationMethod::Nf4,
        "fp8" => QuantizationMethod::Fp8,
        _ => anyhow::bail!(
            "Unsupported quantization: {}. Supported: q4_0, q4_1, q8_0, q4_k, q5_k, q6_k, nf4, fp8",
            s
        ),
    };

    // Determine bits and block size from method
    let (bits, group_size) = match method {
        QuantizationMethod::Q4_0 | QuantizationMethod::Q4_1 | QuantizationMethod::Nf4 => (4, 32),
        QuantizationMethod::Q8_0 => (8, 32),
        QuantizationMethod::Q4K => (4, 256),
        QuantizationMethod::Q5K => (5, 256),
        QuantizationMethod::Q6K => (6, 256),
        QuantizationMethod::Fp8 => (8, 1),
        _ => (4, 128),
    };

    Ok(QuantizationConfig {
        method,
        bits,
        group_size,
        calibration_dataset: None,
        calibration_samples: 0,
    })
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
