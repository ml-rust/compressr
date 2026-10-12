<div align="center">

# compressr

<h3>Model conversion and quantization for LLM checkpoints.</h3>

<p>
Converts SafeTensors, GGUF, TCF, and PyTorch checkpoints to GGUF, TCF, or SafeTensors, and quantizes to GGUF block formats. It reads one tensor at a time.
</p>

<p>
  <a href="#status"><strong>Status</strong></a>
  ·
  <a href="#options"><strong>Options</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/ml-rust/compressr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/compressr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/compressr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

compressr is a Rust CLI and library. It converts LLM checkpoints between formats, casts their element type, and quantizes them into GGUF block formats. Its kernels and block layouts live in [boostr](https://github.com/ml-rust/boostr). compressr handles file I/O, planning, and the command line.

## Status

Implemented:

| Capability         | Detail                                                     |
| ------------------ | ---------------------------------------------------------- |
| SafeTensors input  | Single file, or sharded via `model.safetensors.index.json` |
| GGUF input         | Read back for re-conversion                                |
| TCF input          | Read back, verified and decoded, through boostr's reader. Used for `info` and for conversion to another format |
| PyTorch input      | One `.pt`, `.pth`, or `.bin` state dict with F32, F64, I32, or I64 tensors |
| VoxCPM2 companion  | A safetensors directory with architecture `voxcpm2` folds its audio VAE into the output under `vae.`. Float tensors are stored as F32 |
| SafeTensors output | Sharded at `--max-shard-size`, with a matching index       |
| GGUF output        | GGUF v3, one self-contained file                           |
| GGUF header from a HuggingFace checkpoint | Hyperparameters, RoPE keys, and tokenizer metadata under llama.cpp's key names. See [GGUF headers](#gguf-headers-from-huggingface-checkpoints) |
| TCF output         | TCF v1, one self-contained file. Takes every `--quantization` method and mix |
| dtype cast         | F32, F16, BF16, round-to-nearest-even. A block-quantized input tensor is decoded first |
| GGUF quantization  | Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS |
| GGUF mixed precision | Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, using llama.cpp's per-tensor allocation |
| Measured allocation | `--sensitivity` ranks mix promotions by measured task damage per element |
| Importance matrix  | `--imatrix` weights quantizer choices by calibration activations. Off by default |
| Lowbit repack      | `pq2_0` from a PTQ1_0 GGUF, `ptq1_0` from a PQ2_0 GGUF. Codes and scales move between layouts. No value rounds |
| GGUF header carry  | A GGUF input's key-value pairs are written whole into a GGUF output |
| Sidecar copying    | Tokenizer and config files follow the weights              |
| Conversion report  | The format of every tensor, and the measured bits per weight |

Roadmap. Each capability below has an enum variant. compressr refuses each one before it creates an output file:

| Capability          | What is missing                                                         |
| ------------------- | ----------------------------------------------------------------------- |
| GPTQ, AWQ           | Calibration, plus the `qweight`/`scales`/`qzeros`/`g_idx` tensor layout |
| NF4, FP8            | An on-disk home for the side-band scales the packed payload omits       |
| EXL2                | The whole format driver                                                 |
| ONNX input          | The `.onnx` reader                                                      |
| Sharded PyTorch     | A `pytorch_model.bin.index.json` walk. One `.bin` per model today       |
| PyTorch F16, BF16   | boostr's `.pt` reader decodes F32, F64, I32, and I64 only               |
| Python API          | The PyO3 module exports `__version__` only                              |
| Triple-buffered I/O | A prefetch, compute, and write pipeline. Reads and writes are serial    |
| Parallel shard I/O  | Reads and writes are sequential                                         |
| Q2_K/Q3_K and IQ mixes, other I-quants | Mix rules for Q2_K and Q3_K, and boostr writers for IQ2, IQ3, IQ1 formats. The mixes `q2_k_s`, `q3_k_s`, `q3_k_m`, and `q3_k_l` are refused by name. Flat `q2_k` and `q3_k` are written |

The `optimize` subcommand parses its flags. Every method it accepts is on the roadmap, so it always errors.

SafeTensors output is never quantized. Block quantization writes GGUF, or TCF with `--format tcf`.

TCF comes from a full-precision checkpoint only. A GGUF or TCF input is refused for TCF output.

## Quick Start

compressr depends on `boostr` by path, so it builds inside the `rust-llm` monorepo checkout, not standalone. Copy the local build overrides first. The file is gitignored.

```bash
cp .cargo/config.toml.example .cargo/config.toml
cargo install --path .
```

Add `--features cuda` to `cargo install` for GPU-accelerated quantization through boostr.

Convert a checkpoint to a quantized GGUF file:

```bash
compressr convert ./llama-3-8b --format gguf --quantization q4_k_m
```

Cast a checkpoint to BF16 SafeTensors:

```bash
compressr convert ./llama-3-8b --format safetensors --dtype bf16
```

Inspect a checkpoint:

```bash
compressr info ./llama-3-8b            # format, tensor count, size, metadata
compressr info ./llama-3-8b --verbose  # one line per tensor
compressr info ./llama-3-8b --summary  # tensor count and size only
```

## Usage

### Quantization

`--quantization` takes these flat formats: `q4_0`, `q4_1`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `iq4_nl`, and `iq4_xs`. It also takes the lowbit repack names `pq2_0` and `ptq1_0`.

A mixed-precision name assigns different GGUF block formats to different tensors. The mix names are `q4_k_s`, `q4_k_m`, `q5_k_s`, and `q5_k_m`. The mix reproduces llama.cpp's per-tensor allocation by role and layer.

`--quantization` and `--dtype` are mutually exclusive. Quantization already fixes the stored representation.

### Measured sensitivity and importance matrices

`--sensitivity FILE` reads measured task deltas as JSON lines. It requires a mixed-precision `--quantization` and `--format gguf` or `--format tcf`. The promotions of the mix are ranked by measured damage per element. They spend the same bytes that llama.cpp's rules spend on the checkpoint. Tensors whose rows are not a whole number of 256-element super-blocks cannot be promoted. compressr counts and reports them.

`--imatrix FILE` reads an importance matrix from boostr's `imatrix` example. It weights each quantizer choice by the mean squared activation of each input column. A conversion without the flag writes the same bytes it always wrote. `--imatrix` works with `--format gguf` or `--format tcf`, and with K-quants, `iq4_nl`, `iq4_xs`, and K-quant mixes. A tensor the matrix does not measure is quantized without weights.

```bash
compressr convert ./llama-3-8b --format tcf --quantization q4_k_m --sensitivity llama.sens.jsonl
compressr convert ./llama-3-8b --format gguf --quantization q4_k_m --imatrix llama.imatrix
```

### Lowbit repack

```bash
compressr convert model-PTQ1_0.gguf --format gguf --quantization pq2_0
compressr convert model-PQ2_0.gguf --format gguf --quantization ptq1_0
```

PTQ1_0 and PQ2_0 store the same ternary codes in two block layouts. A repack moves the codes and scales of each source tensor, and it rounds nothing. Tensors already in the target format, float norms, and other types pass through. A PQ2_0 block that holds code 3 has no trit, so `ptq1_0` refuses it. The error names the tensor and block. Neither format is quantized from float weights. A SafeTensors or F16 input is refused.

### Output shards

```bash
compressr convert ./llama-3-8b --format safetensors --dtype bf16 --max-shard-size 2GB
compressr convert ./llama-3-8b --format safetensors --dtype bf16 --max-shard-size none
```

SafeTensors output over the limit becomes `model-00001-of-000NN.safetensors` shards plus `model.safetensors.index.json`. Output under the limit stays one `model.safetensors` with no index. A tensor is never split, so a tensor larger than the limit gets a shard of its own. The flag takes a byte count, a decimal suffix (`5GB`, `500MB`), or a binary suffix (`1GiB`, `512MiB`). The default `5GB` matches HuggingFace. GGUF ignores the flag.

### Input resolution

A directory with a `model.safetensors.index.json` loads exactly the shards that index names. Other `.safetensors` files beside them, such as a LoRA adapter or an earlier conversion's output, are ignored.

A directory with shards and no index loads every `.safetensors` file in name order.

### GGUF headers from HuggingFace checkpoints

For a HuggingFace checkpoint, compressr writes the GGUF header that `convert_hf_to_gguf.py` writes. The architecture comes from `model_type` in `config.json`, then `architecture`, then the first entry of `architectures`. Pass `--arch` when the checkpoint names none.

| Architecture (`model_type`)                                   | `general.architecture` | Tensor names                     |
| ------------------------------------------------------------- | ---------------------- | -------------------------------- |
| `llama`, `mistral`, `mixtral`                                 | `llama`                | llama.cpp's, Q/K rows reordered  |
| `qwen2`, `qwen3`                                              | as named               | llama.cpp's                      |
| `qwen2_moe`, `qwen3_moe`, `gemma`, `gemma2`, `gemma3`, `phi3` | `qwen2moe`, `qwen3moe`, or as named | HuggingFace's              |

- **Hyperparameters**: context length, embedding length, block count, feed-forward length, head counts, and the norm epsilon. It also writes `.rope.*` keys with scaling, expert counts, and key and value lengths when `head_dim` is stated. A key whose `config.json` field is absent is omitted.
- **RoPE tensors**: llama3 scaling writes `rope_freqs.weight`. LongRoPE writes `rope_factors_long.weight` and `rope_factors_short.weight`.
- **Tokenizer**: `tokenizer.model` gives `tokenizer.ggml.model = "llama"`. A byte-level BPE `tokenizer.json` gives `"gpt2"`, with merges and a `tokenizer.ggml.pre` name from llama.cpp's fingerprint table. An unknown fingerprint omits `pre` and logs a warning. Special ids, add-token flags, and `tokenizer.chat_template` follow llama.cpp.
- **Q/K order**: llama.cpp's `llama` architecture rotates adjacent pairs. compressr reorders `q_proj` and `k_proj` rows to match. boostr's GGUF loader restores them.
- **HuggingFace names**: the architectures in the last row keep HuggingFace tensor names. llama.cpp's converter changes their values: Gemma norms, Phi-3 fused projections, and merged MoE experts. llama.cpp cannot load those files.

## Options

### `compressr convert <INPUT>`

| Flag               | Default            | Effect                                                                 |
| ------------------ | ------------------ | ---------------------------------------------------------------------- |
| `--format`         | required           | `gguf`, `safetensors`, or `tcf`. Other values are roadmap items and fail |
| `-o`, `--output`   | derived from input | Output file, or checkpoint directory. Defaults to `<INPUT>.gguf` or `<INPUT>-<dtype>` |
| `--quantization`   | none               | GGUF block format or mix. Repack name for a lowbit GGUF. Not used with `--dtype` |
| `--dtype`          | none               | Cast target: `f32`, `f16`, or `bf16`                                   |
| `--max-shard-size` | `5GB`              | Shard limit for SafeTensors output, or `none`                          |
| `--sensitivity`    | none               | Measured sensitivity JSON lines. Needs a mixed-precision quantization  |
| `--imatrix`        | none               | Importance matrix from boostr's `imatrix` example                      |
| `--arch`           | from `config.json` | Value written to GGUF `general.architecture`                           |
| `-v`, `--verbose`  | off                | Log every tensor as it is processed                                    |

### `compressr info <MODEL>`

| Flag              | Effect                           |
| ----------------- | -------------------------------- |
| `-v`, `--verbose` | Print one line per tensor        |
| `-s`, `--summary` | Print tensor count and size only |

## Memory

Opening a model reads its headers only. Each tensor is then read into an owned buffer, converted, written, and dropped before the next one starts. Peak resident memory is roughly the largest tensor plus its converted copy. It depends on the widest layer, not on the model size.

The loader copies tensor bytes out of the file. It does not memory-map them.

## Architecture

compressr is a thin layer over boostr. Every quantization kernel and block layout lives in boostr, behind its `Runtime` trait.

| Module              | Responsibility                                               |
| ------------------- | ------------------------------------------------------------ |
| `src/loader/`       | Format detection, header reading, one-tensor-at-a-time reads |
| `src/policy/`       | Per-tensor plan: quantize, fall back to F32, or pass through |
| `src/quantization/` | Dispatch into boostr's kernels                               |
| `src/cast/`         | Element-type conversion with round-to-nearest-even           |
| `src/formats/`      | GGUF, SafeTensors, and TCF writers, sharding, sidecar copying |
| `src/importance/`   | Importance matrix input for the quantizer                    |
| `src/python/`       | PyO3 module (`python` feature)                               |

A tensor that cannot be block-quantized is stored as F32 instead of failing the run. A rank-1 tensor has no row to pack blocks along. A row length that is not a multiple of the block size cannot be packed. llama.cpp makes the same choice. The conversion report counts both cases.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
