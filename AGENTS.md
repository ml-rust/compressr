# compressr - Model Conversion & Compression

## Project Overview

compressr converts LLM checkpoints between SafeTensors and GGUF, casts their element type, and quantizes them into GGUF block formats. It is a thin CLI/API layer: all quantization kernels live in boostr.

**Core Philosophy:**

- **Unified Pipeline**: Single entry point for all conversion needs (no switching between llama.cpp scripts, AutoGPTQ Python modules, exllamav2 converters)
- **Performance First**: Pure Rust for bare-metal performance, no Python runtime in the conversion path
- **Minimal Footprint**: One tensor is read, converted and written at a time, so peak memory is bounded by the largest tensor plus its converted copy, not by the model size
- **Hardware Acceleration**: Direct boostr integration for GPU-accelerated quantization

## Implemented vs Roadmap

This section is the SINGLE source of truth on what works. Everything below assumes it. NEVER describe a roadmap item as working, in code comments, docs, or replies.

**Implemented and tested:**

| Capability         | Detail                                                     |
| :----------------- | :--------------------------------------------------------- |
| SafeTensors input  | Single file, or sharded via `model.safetensors.index.json` |
| GGUF input         | Read back for re-conversion                                |
| TCF input          | Read back through boostr's reader: directory, and payloads verified and decoded |
| PyTorch input      | One `.pt` / `.pth` / `.bin` state dict, F32, F64, I32 and I64 tensors |
| VoxCPM2 companion  | A safetensors directory whose architecture is `voxcpm2` (by `config.json` or `--arch`) folds `audiovae.safetensors`, else `audiovae.pth`, into the same GGUF or TCF: every tensor under `vae.` (`vae.decoder.*`, `vae.encoder.*`), weight-norm pairs folded, float tensors stored F32 whatever the quantization or dtype cast, the I32 boundaries vector copied through. GGUF also carries `config.json` verbatim as the `voxcpm2.config_json` string key and `tokenizer.json` verbatim as the `tokenizer.huggingface.json` string key, each only when the file exists. boostr loads the VAE, config and tokenizer from that one file and needs nothing beside it. No other architecture gains a companion |
| SafeTensors output | Sharded at `--max-shard-size`, with a matching index       |
| GGUF output        | GGUF v3, one self-contained file                           |
| GGUF header from HF | `src/formats/gguf_hf/` writes `convert_hf_to_gguf.py`'s keys, names and types for `llama` (incl. `mistral`, `mixtral`), `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `gemma`, `gemma2`, `gemma3`, `phi3`: hyperparameters, RoPE keys and tensors (`rope_freqs.weight`), and `tokenizer.*` from `tokenizer.model` or a byte-level BPE `tokenizer.json`. A key whose source field is absent is OMITTED, never defaulted. `llama`, `qwen2`, `qwen3` tensors take llama.cpp names (all or none), and `llama` Q/K rows take llama.cpp's pair order via `boostr::format::gguf::rope_rows`. Other architectures keep HF names: llama.cpp's converter also changes their values |
| TCF output         | TCF v1, one self-contained file, written through `boostr::tcf`. `--format tcf` takes every `--quantization` method and mix and stores the GGML block streams a GGUF run writes, byte for byte, plus the container's records: the `GGML_REFERENCE` activation contract, a proof vector per tensor, a `fallback_reason` per tensor, and on a `--sensitivity` run the calibration record and per-tensor task delta. NEVER add a TCF-only quantizer or encoding: TCF's payload IS the GGUF payload |
| dtype cast         | F32, F16, BF16, round-to-nearest-even. A block-quantized input tensor is decoded through boostr's reader first, then cast |
| GGUF quantization  | Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS |
| Lowbit repack      | `--quantization pq2_0` on a PTQ1_0 GGUF, `--quantization ptq1_0` on a PQ2_0 GGUF: each source tensor's codes and scales move between the two layouts through `boostr::quant::repack`, no value rounded. Tensors already in the target format, float norms and every other type pass through unchanged. PQ2_0 code 3 has no trit, so that direction refuses the block by tensor and index. Neither format is quantized from float weights: a non-GGUF input, or a GGUF holding no source tensor, is refused before any output file is created |
| GGUF header carry  | A GGUF input's key-value pairs are written whole into a GGUF output (tokenizer, chat template, hyperparameters, `prism.hadamard.*`); only `general.architecture`, `general.file_type` and `general.quantization_version` are the writer's |
| Sidecar copying    | Tokenizer and config files follow the weights              |
| GGUF mixed precision | Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M — llama.cpp's per-tensor allocation, reproduced from `llama_tensor_get_type` |
| GGUF measured allocation | `--sensitivity` ranks a mix's promotions by measured task damage per element, spending the byte budget llama.cpp's positional rules would have spent on the same checkpoint |
| Measured ladder allocation | A `--sensitivity` file carrying one record per (tensor, format) pair is a MENU: each tensor's format is chosen from its measured options under the same byte budget, so a tensor measured as undamaged at a cheaper format is demoted instead of paying the base price. A file names the GGML formats boostr's sweep measured at (`Q4_K`, `Q6_K`, ...); a file measuring one format still allocates two ways, byte for byte. GGUF and TCF outputs take the same path; only TCF records the measurement |
| Importance matrix  | `--imatrix FILE` weights the quantizer's choices by the per-column mean square activation a calibration run accumulated, through boostr's `ImportanceMatrix` reader. OPTIONAL and OFF by default: a run that names no file writes exactly the bytes it always wrote, for every format, and a file written without one is a normal conforming file. Accepted with `--format gguf` or `--format tcf` when the quantization is a K-quant, IQ4_NL or IQ4_XS (`q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `iq4_nl`, `iq4_xs`) or a K-quant mix; refused for anything else. Both formats take boostr's `quantize_with_importance`. A tensor the run never exercised is quantized UNWEIGHTED, never against a synthesized zero vector, and a matrix whose column counts disagree with the checkpoint is refused before any output file is created. Changes which values the encoder chooses, never a bit layout |
| Conversion report  | The format every tensor was written in, and the measured bits per weight |

**Roadmap. Each has an enum variant and each is REFUSED before any output file is created:**

| Capability          | What is missing                                                         |
| :------------------ | :---------------------------------------------------------------------- |
| GPTQ, AWQ           | Calibration, plus the `qweight`/`scales`/`qzeros`/`g_idx` tensor layout |
| NF4, FP8            | An on-disk home for the side-band scales the packed payload omits       |
| EXL2                | The whole format driver                                                 |
| ONNX input          | The `.onnx` reader                                                      |
| Sharded PyTorch     | A `pytorch_model.bin.index.json` walk; one `.bin` per model today       |
| PyTorch F16, BF16   | boostr's `.pt` reader decodes F32, F64, I32 and I64 only                |
| Python API          | The PyO3 module exports `__version__` only                              |
| Triple-buffered I/O | A prefetch/compute/write pipeline; reads and writes are serial today    |
| Parallel shard I/O  | Reads and writes are sequential                                         |
| Q2_K/Q3_K mixes, IQ mixes, other I-quants | Mix rules for Q2_K/Q3_K and for IQ mixes, and boostr WRITERS for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S and IQ1_M. IQ4_NL and IQ4_XS are written flat; `q2_k`, `q2_k_s`, `q3_k_s`, `q3_k_m` and `q3_k_l` are refused by name |

The `optimize` subcommand parses its flags but every method it accepts is a roadmap item, so it always errors.

SafeTensors output is NEVER quantized: its header records an element type and a shape, which cannot describe a block-quantized tensor. Block quantization writes GGUF, or TCF with `--format tcf`.

TCF is produced from a full-precision source ONLY. NEVER add a GGUF-to-TCF or TCF-to-TCF path: either would quantize already-quantized weights a second time, from values the first pass had already rounded. A TCF input is read for `info` and for conversion to another format; it is never a TCF source.

## Build Commands

```bash
cd compressr
cargo build --release                            # CPU-only build
cargo build --release --features cuda            # With CUDA support
cargo test                                       # Run tests
cargo test --features cuda                       # CUDA tests

# CLI usage
compressr convert ./llama-3-8b --format gguf --quantization q4_k_m
compressr convert ./llama-3-8b --format safetensors --dtype bf16
compressr convert ./llama-3-8b --format tcf --quantization q4_k_m --sensitivity llama.sens.jsonl
compressr info ./llama-3-8b --verbose
```

`.cargo/config.toml` is gitignored but REQUIRED to build: it patches numr, splintr, regexr and nexar to their monorepo paths. Copy `.cargo/config.toml.example` to `.cargo/config.toml` on a fresh checkout. Without it cargo resolves numr from crates.io and boostr fails to compile.

## Architecture

### Core Components

- `src/loader/` - Format detection and reading; headers up front, then one tensor per call (NOT memory-mapped)
- `src/policy/` - Per-tensor plan: quantize, fall back to F32, or pass through
- `src/quantization/` - Config types and dispatch into boostr's kernels
- `src/cast/` - Element-type conversion, round-to-nearest-even
- `src/formats/` - GGUF, SafeTensors and TCF writers, sharding, sidecar copying
- `src/python/` - PyO3 bindings

### Invariants

- The per-tensor plan is computed ONCE and read by both the registration loop and the write loop. If they disagreed by one tensor, every later offset in the file would be wrong with no error raised. A mixed-precision run puts a DIFFERENT block type in that plan per tensor, so the write loop takes its format from the plan and never from the configuration.
- `general.file_type` is llama.cpp's `llama_ftype` enum, NOT a GGML type id. A reader takes each tensor's format from the tensor directory and reads this key only to name the file, so a wrong value mislabels the model everywhere it is displayed.
- GGUF stores dimensions REVERSED, so the row length `ne[0]` is the LAST dimension of a standard-order shape. The loader reverses a GGUF input's dimensions into standard order and the writer reverses them back. A rank-1 tensor or a row that is not a whole number of blocks is stored as F32, as llama.cpp does.
- `ConversionConfig::validate` rejects every unwritable configuration BEFORE a file is opened. A mismatch caught by the writer instead aborts mid-loop and leaves a truncated file.
- SafeTensors headers are padded with spaces to a multiple of 8 bytes, so the data block starts 8-byte aligned and a reader can map an F32 tensor in place.
- The loader COPIES tensor bytes out of the file. NEVER describe it as zero-copy or memory-mapped.
- A TCF read VERIFIES before it decodes: `payload_digest`, `semantic_digest`, then the proof vector with boostr's own block decoder (Section 15). boostr's reader does this and compressr never bypasses it.
- A `torch.nn.utils.weight_norm` pair is folded on read, through `boostr::nn::fuse_weight_norm` and nothing else. compressr NEVER holds a second copy of that arithmetic, exactly as it holds no block layout.
- GGUF output names and row order come from `GgufTensorLayout`, applied ONLY inside `GgufFormatWriter`. Plans, mix rules, the importance matrix and sensitivity files key on the CHECKPOINT's names.
- A `.pth` whose keys ALL start with `state_dict.` has that root stripped. The pickle flattener carries the top-level key into every name, and no consumer of the output expects it.

### Streaming

Tensors are processed one at a time: read, quantize or cast, write, drop both buffers. Peak resident memory is roughly the largest tensor plus its converted copy — bounded by the widest layer, not constant, and not proportional to the model size.

## Feature Flags

- `cuda` - Enable CUDA GPU acceleration
- `python` - Build Python bindings via PyO3

## Architecture Philosophy: Thin Layer on boostr

**CRITICAL**: compressr is a thin, user-friendly CLI/API layer. **ALL quantization kernels and algorithms live in boostr**.

### Development Rule

If you're implementing a quantization feature, ask:

- "Is this a kernel/algorithm?" → **boostr**
- "Is this file I/O or CLI?" → **compressr**

NEVER add a second copy of a block layout to compressr — GGUF blocks live in boostr, and the TCF container's offsets, digests and proofs live in `boostr::tcf`. A writer checked against a reader that sits beside it proves nothing: Q6_K once shipped with the wrong field order and compressr's own reader agreed with it while every real reader produced NaN. `tests/gguf_roundtrip.rs` decodes with boostr's reader kernels for exactly this reason, and the TCF tests read back through boostr's reader.

### Why This Matters

1. **Reusability**: oxidizr and blazr can use boostr quantization directly
2. **No vendor lock-in**: boostr's `Runtime` trait works for CPU, CUDA, ROCm, etc.
3. **Single source of truth**: One optimized implementation for all tools
4. **Focused optimization**: All kernel work happens in one place

## Performance Targets

Targets, not measurements. NEVER quote one as an achieved result.

1. **Throughput**: 5x faster than AutoGPTQ
2. **Peak RAM**: bounded by the largest tensor plus its converted copy, not by the model size
3. **Perplexity**: Zero degradation vs reference implementations

## Testing

```bash
cargo test                    # Rust unit and integration tests
cargo test --features cuda    # CUDA-accelerated tests
```

## Integration with Ecosystem

- **Input**: Models trained with oxidizr or HuggingFace
- **Output**: GGUF for blazr or llama.cpp; SafeTensors checkpoints for HuggingFace loaders; TCF for boostr, which reads TCF and never writes it
- **Compute**: boostr's multi-backend GPU acceleration
- **Tokenization**: Compatible with splintr vocabularies
