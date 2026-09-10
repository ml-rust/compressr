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
| PyTorch input      | One `.pt` / `.pth` / `.bin` state dict, F32 and F64 tensors |
| SafeTensors output | Sharded at `--max-shard-size`, with a matching index       |
| GGUF output        | GGUF v3, one self-contained file                           |
| TCF output         | TCF v1, one self-contained file, written through tcf-core  |
| TCF quantization   | Q4S32_T64, Q4AS32_T64, Q4AS64_T64, Q6S32_T64, Q6S16D_T64, Q4AS32D_T64, Q8S32_T64 |
| TCF threading      | One tensor's tile chunks quantize across rayon's pool, bytes unchanged |
| dtype cast         | F32, F16, BF16, round-to-nearest-even                      |
| GGUF quantization  | Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K              |
| Sidecar copying    | Tokenizer and config files follow the weights              |
| GGUF mixed precision | Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M — llama.cpp's per-tensor allocation, reproduced from `llama_tensor_get_type` |
| GGUF measured allocation | `--sensitivity` ranks a mix's promotions by measured task damage per element, spending the byte budget llama.cpp's positional rules would have spent on the same checkpoint |
| Conversion report  | The format every tensor was written in, and the measured bits per weight |

**Roadmap. Each has an enum variant and each is REFUSED before any output file is created:**

| Capability          | What is missing                                                         |
| :------------------ | :---------------------------------------------------------------------- |
| GPTQ, AWQ           | Calibration, plus the `qweight`/`scales`/`qzeros`/`g_idx` tensor layout |
| NF4, FP8            | An on-disk home for the side-band scales the packed payload omits       |
| EXL2                | The whole format driver                                                 |
| ONNX input          | The `.onnx` reader                                                      |
| Sharded PyTorch     | A `pytorch_model.bin.index.json` walk; one `.bin` per model today       |
| PyTorch F16, BF16   | boostr's `.pt` reader decodes F32 and F64 only                          |
| Python API          | The PyO3 module exports `__version__` only                              |
| Triple-buffered I/O | A prefetch/compute/write pipeline; reads and writes are serial today    |
| Parallel shard I/O  | Reads are sequential; rayon parallelizes TCF quantization, never I/O    |
| Q2_K/Q3_K mixes, IQ mixes | Mix rules for Q2_K/Q3_K, and an I-quant WRITER in boostr; `q2_k`, `q2_k_s`, `q3_k_s`, `q3_k_m` and `q3_k_l` are refused by name |

The `optimize` subcommand parses its flags but every method it accepts is a roadmap item, so it always errors.

SafeTensors output is NEVER quantized: its header records an element type and a shape, which cannot describe a block-quantized tensor. GGUF block quantization writes GGUF; `--tcf-encoding` writes TCF.

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
compressr convert ./llama-3-8b --format tcf --tcf-encoding q4s32t64 --tcf-preferred-encoding q4s32t64
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
- GGUF stores dimensions REVERSED, so the row length `ne[0]` is the LAST dimension of a standard-order shape. A rank-1 tensor or a row that is not a whole number of blocks is stored as F32, as llama.cpp does.
- `ConversionConfig::validate` rejects every unwritable configuration BEFORE a file is opened. A mismatch caught by the writer instead aborts mid-loop and leaves a truncated file.
- SafeTensors headers are padded with spaces to a multiple of 8 bytes, so the data block starts 8-byte aligned and a reader can map an F32 tensor in place.
- The loader COPIES tensor bytes out of the file. NEVER describe it as zero-copy or memory-mapped.
- A TCF read VERIFIES before it decodes: `payload_digest`, the recomputed logical stream against `semantic_digest`, then the proof vector (Section 15). boostr's reader does this and compressr never bypasses it.
- A `torch.nn.utils.weight_norm` pair is folded on read, through `boostr::nn::fuse_weight_norm` and nothing else. compressr NEVER holds a second copy of that arithmetic, exactly as it holds no block layout.
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

NEVER add a second copy of a block layout to compressr — GGUF blocks live in boostr, and TCF bit positions, tile order and plane order live in tcf-core. A writer checked against a reader that sits beside it proves nothing: Q6_K once shipped with the wrong field order and compressr's own reader agreed with it while every real reader produced NaN. `tests/gguf_roundtrip.rs` decodes with boostr's reader kernels for exactly this reason, and the TCF tests read back through `tcf-core`'s own reader.

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
