# compressr

A Rust CLI and library that converts large language model checkpoints between SafeTensors and GGUF, casts their element type, and quantizes them into GGUF block formats.

## Status

Implemented and tested:

| Capability         | Detail                                                     |
| ------------------ | ---------------------------------------------------------- |
| SafeTensors input  | Single file, or sharded via `model.safetensors.index.json` |
| GGUF input         | Read back for re-conversion                                |
| TCF input          | Read back, verified and decoded, through boostr's reader   |
| PyTorch input      | One `.pt` / `.pth` / `.bin` state dict, F32, F64, I32 and I64 tensors |
| VoxCPM2 companion  | A `voxcpm2` checkpoint directory folds its `audiovae.pth` into the same GGUF/TCF under `vae.`, F32, with `config.json` as the `voxcpm2.config_json` key and `tokenizer.json` as the `tokenizer.huggingface.json` key |
| SafeTensors output | Sharded at `--max-shard-size`, with a matching index       |
| GGUF output        | GGUF v3, one self-contained file                           |
| dtype cast         | `f32`, `f16`, `bf16`, round-to-nearest-even; block-quantized input tensors are decoded through boostr first |
| GGUF quantization  | `q4_0`, `q4_1`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `iq4_nl`, `iq4_xs` |
| Lowbit repack      | `pq2_0` from a PTQ1_0 GGUF, `ptq1_0` from a PQ2_0 GGUF: codes and scales move between the two layouts, nothing is rounded, the header is carried whole |
| Sidecar copying    | Tokenizer and config files follow the weights              |

Not implemented. Each errors out rather than writing a file:

| Capability          | What is missing                                                         |
| ------------------- | ----------------------------------------------------------------------- |
| GPTQ, AWQ           | Calibration, plus the `qweight`/`scales`/`qzeros`/`g_idx` tensor layout |
| NF4, FP8            | An on-disk home for the side-band scales the packed payload omits       |
| EXL2                | The whole format driver                                                 |
| ONNX input          | The `.onnx` reader                                                      |
| Sharded PyTorch     | A `pytorch_model.bin.index.json` walk; one `.bin` per model today       |
| PyTorch F16, BF16   | boostr's `.pt` reader decodes F32, F64, I32 and I64 only                |
| Python API          | The PyO3 module exports `__version__` only                              |

The `optimize` subcommand parses its flags but every method it accepts is in the second table, so it always errors. It exists for the CLI surface, not for use.

## Install

```bash
cargo install --path .
```

Add `--features cuda` for GPU-accelerated quantization through boostr.

compressr depends on `boostr` by path, so it builds inside the `rust-llm` monorepo checkout, not standalone.

## Use

### Convert to GGUF

```bash
compressr convert ./llama-3-8b --format gguf --quantization q4_k_m
```

`--quantization` takes `q4_0`, `q4_1`, `q8_0`, `q4_k`, `q5_k` or `q6_k`. A trailing `_s`, `_m` or `_l` is accepted and ignored: compressr picks one block format for the whole model.

GGUF records the architecture in `general.architecture`, and every consumer keys its tensor-name mapping off it. compressr reads it from the input checkpoint's `config.json`, trying `model_type`, then `architecture`, then the first entry of `architectures`. Pass `--arch` when the checkpoint names none:

```bash
compressr convert ./voxcpm2 --format gguf --quantization q4_k --arch voxcpm2
```

### Repack a lowbit GGUF

```bash
compressr convert model-PTQ1_0.gguf --format gguf --quantization pq2_0
compressr convert model-PQ2_0.gguf --format gguf --quantization ptq1_0
```

PTQ1_0 and PQ2_0 store the same ternary codes in two block layouts. A repack moves each source tensor's codes and scales through boostr and rounds nothing; tensors already in the target format, float norms and every other type pass through. A PQ2_0 block holding code 3 (`2 * d`) has no trit, so `ptq1_0` refuses it by tensor and block. Neither format is quantized from float weights: a SafeTensors or F16 input is refused.

### Cast the element type

```bash
compressr convert ./llama-3-8b --format safetensors --dtype bf16
```

Float tensors are cast. Block-quantized tensors of a GGUF input are decoded through boostr's reader, then cast, so a lowbit GGUF becomes an F16 GGUF or SafeTensors checkpoint. Integer buffers and boolean masks pass through byte for byte. The output `config.json` has its `torch_dtype` rewritten to match the weights beside it.

`--dtype` and `--quantization` are mutually exclusive. Quantization already fixes the stored representation.

### Inspect a model

```bash
compressr info ./llama-3-8b            # format, tensor count, size, metadata
compressr info ./llama-3-8b --verbose  # one line per tensor
compressr info ./llama-3-8b --summary  # tensor count and size only
```

### Shard the output

```bash
compressr convert ./llama-3-8b --format safetensors --dtype bf16 --max-shard-size 2GB
compressr convert ./llama-3-8b --format safetensors --dtype bf16 --max-shard-size none
```

SafeTensors output over the limit becomes `model-00001-of-000NN.safetensors` shards plus `model.safetensors.index.json`. Output under it stays one `model.safetensors` with no index. A tensor is never split, so one larger than the limit gets a shard to itself.

The flag takes a byte count, a decimal suffix (`5GB`, `500MB`) or a binary one (`1GiB`, `512MiB`). The default `5GB` is HuggingFace's own. GGUF ignores the flag: GGUF is one self-contained file.

## Options

### `compressr convert <INPUT>`

| Flag               | Default            | Effect                                       |
| ------------------ | ------------------ | -------------------------------------------- |
| `--format`         | required           | `gguf` or `safetensors`                      |
| `-o`, `--output`   | derived from input | Output file or checkpoint directory          |
| `--quantization`   | none               | GGUF block format; requires `--format gguf`  |
| `--dtype`          | none               | Cast target: `f32`, `f16` or `bf16`          |
| `--max-shard-size` | `5GB`              | Shard limit, or `none`                       |
| `--arch`           | from `config.json` | Value written to GGUF `general.architecture` |
| `-v`, `--verbose`  | off                | Log every tensor as it is processed          |

### `compressr info <MODEL>`

| Flag              | Effect                           |
| ----------------- | -------------------------------- |
| `-v`, `--verbose` | Print one line per tensor        |
| `-s`, `--summary` | Print tensor count and size only |

## Input resolution

A directory holding a `model.safetensors.index.json` loads exactly the shards that index names. Every other `.safetensors` file beside them — a LoRA adapter, an earlier conversion's output — is ignored, because absorbing it would add tensors the model does not have.

A directory with shards but no index falls back to loading every `.safetensors` file in name order.

## Memory

Opening a model reads its headers only. Each tensor is then read into an owned buffer, converted, written, and dropped before the next one starts. Peak resident memory is roughly the largest tensor plus its converted copy — bounded by the widest layer, not by the model size, and not constant.

The loader copies tensor bytes out of the file. It does not memory-map them.

## Architecture

compressr is a thin layer. Every quantization kernel lives in boostr, which owns the CPU and GPU implementations behind its `Runtime` trait.

| Module              | Responsibility                                               |
| ------------------- | ------------------------------------------------------------ |
| `src/loader/`       | Format detection, header reading, one-tensor-at-a-time reads |
| `src/policy/`       | Per-tensor plan: quantize, fall back to F32, or pass through |
| `src/quantization/` | Dispatch into boostr's kernels                               |
| `src/cast/`         | Element-type conversion with round-to-nearest-even           |
| `src/formats/`      | GGUF and SafeTensors writers, sharding, sidecar copying      |
| `src/python/`       | PyO3 module                                                  |

A tensor that cannot be block-quantized is stored as F32 rather than failing the run: a rank-1 buffer has no row to pack blocks along, and a row length that is not a multiple of the block size cannot be packed at all. llama.cpp makes the same choice. The conversion summary counts both cases.

## Test

```bash
cargo test
cargo test --features cuda
```

`tests/gguf_roundtrip.rs` quantizes with compressr and dequantizes with boostr's reader kernels — the same ones llama.cpp-compatible loaders use. A writer checked only against a reader that sits beside it proves nothing: compressr once shipped a Q6_K writer with the wrong field order whose own reader agreed with it.

## License

Apache-2.0

## Acknowledgments

- llama.cpp for the GGUF format specification
- boostr for the quantization kernels
