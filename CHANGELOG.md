# Changelog

All notable changes to compressr are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-11-06

### Changed

- `README`, crate description: state what the crate does.

## [0.2.0] - 2026-11-01

compressr writes TCF, reads TCF and PyTorch input, and adds GGUF formats, mixes and allocation controls.

### Breaking

- **`cuda` feature removed** — it enabled `boostr/cuda` and changed nothing. Every boostr call runs on `CpuRuntime`.
- **`optimize` takes `--method` only** — it always errors without writing a file. Before, it mapped a flat GGUF method onto `convert` and ignored `--bits`, `--group-size`, `--calib-dataset` and `--device`. The flags `--bits`, `--group-size`, `--calib-dataset`, `--calib-samples`, `--out`, `--device`, `--max-shard-size` and `--verbose` are removed.
- **`formats::create_writer` errors for `OutputFormat::Awq`, `Gptq`, `Exl2` and `Fp8`** — before, it wrote plain SafeTensors under those names.
- **`_s`, `_m` and `_l` suffixes on `--quantization` name a mix** — before, the suffix was ignored and one format covered the whole model. A repeated suffix (`q4_k_mmm`) and the mixes `q2_k_s`, `q3_k_s`, `q3_k_m` and `q3_k_l` are refused.
- **Alignment setters return `Result`** — `GgufBatchWriter::with_alignment`, `StreamingGgufWriter::with_alignment` and `GgufFormatWriter::set_alignment` error on an alignment that is not a power of two or does not fit a `u32`. Before, they panicked.
- **Dependencies** — `boostr` 0.10 and `splintr` 0.22 are required.

### Added

- **TCF v1 output** — `--format tcf` takes every `--quantization` method and mix and stores the GGML block streams a GGUF run writes, byte for byte. Each tensor carries a proof vector and a `fallback_reason`. TCF is written from a full-precision input only.
- **TCF input** — read through boostr's reader, which checks every payload against its digests and proof vector. `info` reports a TCF file's modules, encodings and records.
- **PyTorch input** — one `.pt`, `.pth` or `.bin` state dict, as a file or as `pytorch_model.bin` in a directory. F32, F64, I32 and I64 tensors load.
- **GGUF formats and mixes** — Q2_K, Q3_K, IQ4_NL and IQ4_XS, plus the mixes `q4_k_s`, `q4_k_m`, `q5_k_s` and `q5_k_m` with llama.cpp's per-tensor allocation.
- **`--sensitivity FILE`** — ranks a mix's promotions by measured task damage per element. A file with one record per (tensor, format) pair can also demote a tensor.
- **`--imatrix FILE`** — weights K-quant, IQ4_NL and IQ4_XS choices by an importance matrix. A run without it writes the same bytes as before.
- **Lowbit repack** — `--quantization pq2_0` on a PTQ1_0 GGUF and `--quantization ptq1_0` on a PQ2_0 GGUF. No value rounds.
- **GGUF header for HuggingFace checkpoints** — `convert_hf_to_gguf.py`'s keys, tensor names and tokenizer metadata for `llama`, `mistral`, `mixtral`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `gemma`, `gemma2`, `gemma3` and `phi3`.
- **GGUF header carry** — a GGUF input's key-value pairs are written whole into a GGUF output.
- **VoxCPM2 companion** — a `voxcpm2` checkpoint directory folds its AudioVAE into the same GGUF or TCF file. A GGUF output also carries `config.json` and `tokenizer.json` as string keys.
- **Conversion report** — lists each tensor's output format, the measured bits per weight, and the allocation a mix made.
- **API** — `ConversionConfig::with_input_format` and `ModelLoader::with_format` skip auto-detection, and `ModelLoader::check_format` errors on a content mismatch. `QuantizationMethod` gains `ALL`, `name`, `from_name`, `quantize_names`, `repack_names` and `writable_names`. New: `formats::check_alignment` and the `importance` module.

### Fixed

- **`ConversionConfig::input_format`** — the field is honored. Before, it was ignored.
- **`--dtype` on a block-quantized input** — each block decodes through boostr's reader, then casts. Before, quantized blocks passed through unchanged.
- **`--quantization` error** — it lists exactly the names that convert. Before, it listed `nf4` and `fp8` as flat formats.
- **`ConversionConfig::validate` error** — it names `pq2_0` and `ptq1_0` among the writable methods. Before, it omitted them.
