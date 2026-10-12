# Contributing to compressr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and test

compressr builds inside the `rust-llm` monorepo, because it depends on `boostr` by path. On a fresh checkout, copy the local build overrides first:

```bash
cp .cargo/config.toml.example .cargo/config.toml
```

Then run these commands from the `compressr` directory:

```bash
cargo build --release
cargo test
cargo test --features cuda
```

## Conventions

- Put quantization kernels and block layouts in boostr. compressr holds file I/O, planning, and the command line.
- Never add a second copy of a GGML block layout to compressr. Its GGUF blocks live in boostr, and so do TCF offsets, digests, and proofs.
- Check a writer with boostr's reader kernels, not with a reader that sits beside the writer. `tests/gguf_roundtrip.rs` follows this rule.
- Keep one tensor in memory at a time in the conversion loop.
- Refuse an unsupported configuration before the first output file opens.

## Pull requests

- Add tests for every behavior change.
- Update the Status tables in `README.md` when a capability changes state.
