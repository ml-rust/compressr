# Changelog

All notable changes to compressr are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Changes since 0.1.0.

### Added

- TCF v1 output (`--format tcf`), written through `boostr::tcf`. It takes every `--quantization` method and mix and stores the GGML block streams a GGUF run writes, byte for byte. Each tensor carries a proof vector and a `fallback_reason`. A `--sensitivity` run also records the calibration record and per-tensor task delta. TCF is written from a full-precision input only.
- TCF input, read back through boostr's reader. Every payload is checked against its digests and proof vector before it is decoded. `info` reports a TCF file's modules, encodings and records.
- PyTorch input: one `.pt`, `.pth` or `.bin` state dict, as a file or as `pytorch_model.bin` in a directory. F32, F64, I32 and I64 tensors. A ZIP container is detected by content under any extension. A `state_dict.` root is stripped, and `weight_norm` pairs are folded on read.
- GGUF quantization formats Q2_K, Q3_K, IQ4_NL and IQ4_XS, beside Q4_0, Q4_1, Q8_0, Q4_K, Q5_K and Q6_K.
- GGUF mixed precision: `q4_k_s`, `q4_k_m`, `q5_k_s` and `q5_k_m`, reproducing llama.cpp's `llama_tensor_get_type` per-tensor allocation.
- `--sensitivity FILE`: ranks a mix's promotions by measured task damage per element, at the byte budget llama.cpp's rules spend. A file with one record per (tensor, format) pair is a menu that can also demote a tensor.
- `--imatrix FILE`: weights K-quant, IQ4_NL and IQ4_XS choices by an importance matrix, through boostr's reader. Off by default. A run without it writes the same bytes as before.
- Lowbit repack: `--quantization pq2_0` on a PTQ1_0 GGUF and `--quantization ptq1_0` on a PQ2_0 GGUF. Codes and scales move between layouts and no value rounds.
- GGUF header for HuggingFace checkpoints: `convert_hf_to_gguf.py`'s keys, tensor names and tokenizer metadata for `llama`, `mistral`, `mixtral`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `gemma`, `gemma2`, `gemma3` and `phi3`.
- GGUF header carry: a GGUF input's key-value pairs are written whole into a GGUF output.
- VoxCPM2 companion: a `voxcpm2` checkpoint directory folds its AudioVAE into the same GGUF or TCF file. A GGUF output also carries `config.json` and `tokenizer.json` as string keys.
- Conversion report: the format every tensor was written in, the measured bits per weight, and the allocation a mix made.
- Public modules: `importance` (new). `policy` grows the mix, ladder and sensitivity modules behind the new flags.
- `ConversionConfig::with_input_format` and `ModelLoader::with_format`. An explicit input format skips auto-detection, and `ModelLoader::check_format` errors when the input's content does not match.
- `QuantizationMethod::ALL`, `name`, `from_name`, `quantize_names`, `repack_names` and `writable_names`. Error messages list accepted names from these.
- `formats::check_alignment`.

### Changed

- `formats::create_writer` errors for `OutputFormat::Awq`, `Gptq`, `Exl2` and `Fp8`. It used to write plain SafeTensors under those names.
- `optimize` takes `--method` only and always errors without writing a file. It used to map a flat GGUF method onto `convert` and ignore `--bits`, `--group-size`, `--calib-dataset` and `--device`. The flat-method error now names the `convert` command that writes it. `--bits`, `--group-size`, `--calib-dataset`, `--calib-samples`, `--out`, `--device`, `--max-shard-size` and `--verbose` are removed from `optimize`.
- A `_s`, `_m` or `_l` suffix on `--quantization` names a mix. It used to be ignored, and one format was written for the whole model. A repeated suffix (`q4_k_mmm`) and a mix with no rule table (`q2_k_s`, `q3_k_s`, `q3_k_m`, `q3_k_l`) are refused.
- `--dtype` on a block-quantized input decodes each block through boostr's reader, then casts it. It used to pass quantized blocks through unchanged.
- `ConversionConfig::input_format` is honored. It used to be ignored.
- `GgufBatchWriter::with_alignment` and `StreamingGgufWriter::with_alignment` return `Result` and error on an alignment that is not a power of two or does not fit a `u32`. They used to panic. `GgufFormatWriter::set_alignment` returns `Result` and checks the same.
- `boostr` 0.9 and `splintr` 0.22 are required.

### Removed

- The `cuda` feature. It enabled `boostr/cuda` and changed nothing: every boostr call runs on `CpuRuntime`.
- The `gptq` keyword from `Cargo.toml` and `pyproject.toml`. GPTQ is a roadmap item that compressr refuses.
- `CLAUDE.md` and `AGENTS.md` from the published crate.

### Fixed

- The `--quantization` error listed `nf4` and `fp8` as flat formats. It lists exactly the names that convert.
- The `ConversionConfig::validate` error for an unwritable method omitted `pq2_0` and `ptq1_0`.
