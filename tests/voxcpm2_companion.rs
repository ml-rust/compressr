//! The VoxCPM2 companion: a conversion of a checkpoint directory that holds
//! `audiovae.pth` beside `model.safetensors` writes ONE file carrying the
//! transformer stack, the AudioVAE under `vae.`, and (for GGUF) the
//! `config.json` and `tokenizer.json` as metadata.
//!
//! Read back through boostr's own GGUF and TCF readers, never through
//! compressr's: a writer checked against a reader beside it proves nothing.
//! The folded weight-norm pair is compared against
//! `boostr::nn::fuse_weight_norm` evaluated on the raw pair, the one copy of
//! that arithmetic.

use boostr::format::{GgmlType, Gguf, TcfLoader, TorchStateDict};
use boostr::nn::fuse_weight_norm;
use boostr::tcf::Encoding;
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};
use compressr::{
    ConversionConfig, FormatWriter, GgufMix, OutputFormat, QuantizationConfig,
    SafeTensorsFormatWriter, UnifiedDType, AUDIO_VAE_PREFIX, HF_TOKENIZER_JSON_KEY,
    VOXCPM2_CONFIG_JSON_KEY,
};
use std::path::{Path, PathBuf};

/// The committed AudioVAE fixture, produced by PyTorch: one plain conv
/// weight and bias, one `weight_g`/`weight_v` pair, one `[4, 64]` embed
/// (the shape a flat plan would quantize), the I32 boundaries vector, and
/// one encoder tensor, all under a `state_dict` root.
fn audiovae_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audiovae_state_dict.pth")
}

/// The `config.json` a VoxCPM2 checkpoint ships, trimmed to what the test
/// needs. The bytes are what must round-trip, so they are kept verbatim.
const CONFIG_JSON: &str = "{\n  \"architecture\": \"voxcpm2\",\n  \"patch_size\": 4\n}\n";

/// A `tokenizer.json` trimmed the same way. Nothing parses it here: the
/// bytes round-trip, and boostr's tokenizer loader owns their meaning.
const TOKENIZER_JSON: &str =
    "{\n  \"version\": \"1.0\",\n  \"model\": {\"type\": \"BPE\", \"vocab\": {\"a\": 0}}\n}\n";

/// Reproducible values with enough spread for a quantizer to work on.
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / 16_777_216.0 - 0.5
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The transformer-stack tensors of the fixture checkpoint.
fn stack_tensors() -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    vec![
        (
            "base_lm.layers.0.self_attn.q_proj.weight",
            vec![8, 256],
            values(8 * 256, 1),
        ),
        ("base_lm.norm.weight", vec![8], values(8, 2)),
        ("enc_to_lm_proj.weight", vec![4, 256], values(4 * 256, 3)),
    ]
}

/// Build a VoxCPM2-shaped checkpoint directory: `config.json`,
/// `model.safetensors`, the AudioVAE fixture as `audiovae.pth`, and
/// `tokenizer.json` when `tokenizer_json` is given.
fn write_checkpoint(dir: &Path, config_json: &str, tokenizer_json: Option<&str>) {
    std::fs::write(dir.join("config.json"), config_json).expect("writes config.json");
    if let Some(text) = tokenizer_json {
        std::fs::write(dir.join("tokenizer.json"), text).expect("writes tokenizer.json");
    }
    std::fs::copy(audiovae_fixture(), dir.join("audiovae.pth")).expect("copies the AudioVAE");
    let mut writer = SafeTensorsFormatWriter::new();
    let tensors = stack_tensors();
    for (name, shape, _) in &tensors {
        writer
            .register_tensor(name, shape, UnifiedDType::F32, None)
            .expect("registers a stack tensor");
    }
    writer
        .initialize(&dir.join("model.safetensors"))
        .expect("initializes the checkpoint");
    for (_, _, data) in &tensors {
        writer
            .write_tensor_data(&f32_bytes(data))
            .expect("writes a stack tensor");
    }
    writer.finalize().expect("finalizes the checkpoint");
}

fn convert(dir: &Path, out: &Path, format: OutputFormat) {
    let config = ConversionConfig::new(
        dir.to_string_lossy().to_string(),
        out.to_string_lossy().to_string(),
        format,
    )
    .with_quantization(QuantizationConfig::q4_k());
    compressr::convert(config).expect("converts the checkpoint");
}

/// `fuse_weight_norm` over the fixture's raw pair, on the CPU.
fn expected_folded_pair() -> Vec<f32> {
    let dict = TorchStateDict::open(audiovae_fixture()).expect("opens the fixture");
    let device = CpuDevice::new();
    let v: Tensor<CpuRuntime> = dict
        .load_tensor("state_dict.decoder.model.1.weight_v", &device)
        .expect("weight_v");
    let g: Tensor<CpuRuntime> = dict
        .load_tensor("state_dict.decoder.model.1.weight_g", &device)
        .expect("weight_g");
    let client = CpuClient::new(device);
    fuse_weight_norm(&client, &v, &g, 0)
        .expect("folds")
        .to_vec::<f32>()
}

/// Every name the companion contributes, in its output spelling.
const VAE_NAMES: [&str; 6] = [
    "vae.decoder.model.0.weight",
    "vae.decoder.model.0.bias",
    "vae.decoder.model.1.weight",
    "vae.decoder.sr_cond_model.2.scale_embed.weight",
    "vae.decoder.sr_bin_boundaries",
    "vae.encoder.block.0.weight",
];

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: element count");
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
            "{name}[{index}]: {g} vs {w}"
        );
    }
}

#[test]
fn a_voxcpm2_gguf_carries_the_folded_vae_dense_its_config_and_its_tokenizer() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_checkpoint(dir.path(), CONFIG_JSON, Some(TOKENIZER_JSON));
    let out = dir.path().join("model.gguf");
    convert(dir.path(), &out, OutputFormat::Gguf);

    let mut gguf = Gguf::open(&out).expect("boostr opens the GGUF");
    let device = CpuDevice::new();

    // Every VAE tensor is present, under `vae.`, and dense.
    for name in VAE_NAMES {
        let info = gguf
            .tensor_info(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = if name == "vae.decoder.sr_bin_boundaries" {
            GgmlType::I32
        } else {
            GgmlType::F32
        };
        assert_eq!(info.ggml_type, want, "{name}");
        assert!(name.starts_with(AUDIO_VAE_PREFIX));
    }
    // The `[4, 64]` embed is the shape a flat plan would quantize; it stays
    // F32 by the AudioVAE rule, not by luck of its shape.
    // `GgufTensorInfo::shape` is the file's reversed order; the loaded
    // tensor is in standard order.
    let embed = gguf
        .tensor_info("vae.decoder.sr_cond_model.2.scale_embed.weight")
        .expect("embed");
    assert_eq!(embed.ggml_type, GgmlType::F32);
    let embed: Tensor<CpuRuntime> = gguf
        .load_tensor_f32("vae.decoder.sr_cond_model.2.scale_embed.weight", &device)
        .expect("embed values");
    assert_eq!(embed.shape(), &[4, 64]);

    // The weight-norm pair arrives folded, equal to `fuse_weight_norm`.
    let folded: Tensor<CpuRuntime> = gguf
        .load_tensor_f32("vae.decoder.model.1.weight", &device)
        .expect("folded weight");
    assert_eq!(folded.shape(), &[8, 4, 1]);
    assert_close(
        "vae.decoder.model.1.weight",
        &folded.to_vec::<f32>(),
        &expected_folded_pair(),
    );
    // The raw pair itself is NOT written.
    assert!(gguf.tensor_info("vae.decoder.model.1.weight_v").is_err());
    assert!(gguf.tensor_info("vae.decoder.model.1.weight_g").is_err());

    // The I32 vector is copied through, values intact.
    let bytes = gguf
        .read_tensor_bytes("vae.decoder.sr_bin_boundaries")
        .expect("boundaries");
    let boundaries: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    assert_eq!(boundaries, vec![16000, 24000, 44100]);

    // `config.json` and `tokenizer.json` round-trip byte for byte.
    assert_eq!(
        gguf.metadata().get_string(VOXCPM2_CONFIG_JSON_KEY),
        Some(CONFIG_JSON)
    );
    assert_eq!(
        gguf.metadata().get_string(HF_TOKENIZER_JSON_KEY),
        Some(TOKENIZER_JSON)
    );
    assert_eq!(gguf.metadata().architecture(), Some("voxcpm2"));

    // The transformer stack is unaffected: the block-aligned projection is
    // quantized, the norm falls back to F32, and the values decode to what
    // was written.
    assert_eq!(
        gguf.tensor_info("base_lm.layers.0.self_attn.q_proj.weight")
            .expect("q_proj")
            .ggml_type,
        GgmlType::Q4K
    );
    assert_eq!(
        gguf.tensor_info("base_lm.norm.weight")
            .expect("norm")
            .ggml_type,
        GgmlType::F32
    );
    let norm: Tensor<CpuRuntime> = gguf
        .load_tensor_f32("base_lm.norm.weight", &device)
        .expect("norm values");
    assert_close("base_lm.norm.weight", &norm.to_vec::<f32>(), &values(8, 2));
    assert_eq!(gguf.len(), stack_tensors().len() + VAE_NAMES.len());
}

/// No `tokenizer.json` in the directory: the key is not written, rather
/// than written empty.
#[test]
fn a_voxcpm2_gguf_without_a_tokenizer_file_carries_no_tokenizer_key() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_checkpoint(dir.path(), CONFIG_JSON, None);
    let out = dir.path().join("model.gguf");
    convert(dir.path(), &out, OutputFormat::Gguf);

    let gguf = Gguf::open(&out).expect("boostr opens the GGUF");
    assert_eq!(gguf.metadata().get_string(HF_TOKENIZER_JSON_KEY), None);
    assert_eq!(
        gguf.metadata().get_string(VOXCPM2_CONFIG_JSON_KEY),
        Some(CONFIG_JSON)
    );
}

#[test]
fn a_voxcpm2_tcf_carries_the_folded_vae_dense() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_checkpoint(dir.path(), CONFIG_JSON, Some(TOKENIZER_JSON));
    let out = dir.path().join("model.tcf");
    convert(dir.path(), &out, OutputFormat::Tcf);

    let loader = TcfLoader::open(&out).expect("boostr opens the TCF");
    loader.verify_all().expect("every payload verifies");

    for name in VAE_NAMES {
        let info = loader
            .tensor_info(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            matches!(info.encoding(), Encoding::Raw(_)),
            "{name}: stored {:?}, must be raw",
            info.encoding()
        );
    }
    let embed = loader
        .tensor_info("vae.decoder.sr_cond_model.2.scale_embed.weight")
        .expect("embed");
    assert_eq!(embed.shape(), vec![4, 64]);
    assert!(matches!(embed.encoding(), Encoding::Raw(_)));

    let folded = loader
        .load_tensor_f32("vae.decoder.model.1.weight")
        .expect("folded weight");
    assert_close(
        "vae.decoder.model.1.weight",
        &folded,
        &expected_folded_pair(),
    );
    assert!(loader.tensor_info("vae.decoder.model.1.weight_v").is_err());

    // Rank 3 F32 is accepted as written.
    let front = loader
        .tensor_info("vae.decoder.model.0.weight")
        .expect("front conv");
    assert_eq!(front.shape(), vec![8, 1, 7]);

    // The stack is still block-encoded where it can be.
    assert!(matches!(
        loader
            .tensor_info("base_lm.layers.0.self_attn.q_proj.weight")
            .expect("q_proj")
            .encoding(),
        Encoding::Block(_)
    ));
    assert_eq!(loader.len(), stack_tensors().len() + VAE_NAMES.len());
}

/// A mix allocates the transformer stack only: the VAE stays F32 whatever
/// the mix would assign, and its conv stacks do not appear in the
/// allocation report as layers.
#[test]
fn a_mix_keeps_the_vae_f32_and_out_of_its_report() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_checkpoint(dir.path(), CONFIG_JSON, None);
    let out = dir.path().join("model.gguf");
    let config = ConversionConfig::new(
        dir.path().to_string_lossy().to_string(),
        out.to_string_lossy().to_string(),
        OutputFormat::Gguf,
    )
    .with_quantization(QuantizationConfig::q4_k())
    .with_gguf_mix(GgufMix::Q4KM);
    let stats = compressr::convert(config).expect("converts the checkpoint");

    let gguf = Gguf::open(&out).expect("boostr opens the GGUF");
    assert_eq!(gguf.len(), stack_tensors().len() + VAE_NAMES.len());
    assert_eq!(
        gguf.tensor_info("vae.decoder.sr_cond_model.2.scale_embed.weight")
            .expect("embed")
            .ggml_type,
        GgmlType::F32
    );
    assert_ne!(
        gguf.tensor_info("base_lm.layers.0.self_attn.q_proj.weight")
            .expect("q_proj")
            .ggml_type,
        GgmlType::F32
    );
    // The per-tensor report still covers every tensor written, while the
    // allocation describes the transformer stack alone.
    assert_eq!(
        stats.mix.entries().len(),
        stack_tensors().len() + VAE_NAMES.len()
    );
    let allocation = stats.allocation.expect("a mix records its allocation");
    assert!(
        allocation
            .components
            .iter()
            .all(|c| !c.name.starts_with(AUDIO_VAE_PREFIX)),
        "the VAE is not a stack the mix allocates:\n{allocation}"
    );
}

/// The companion is opened for VoxCPM2 only: the same directory under
/// another architecture converts exactly as before, `audiovae.pth` ignored.
#[test]
fn a_non_voxcpm2_directory_gets_no_companion() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_checkpoint(
        dir.path(),
        "{\"model_type\": \"llama\"}",
        Some(TOKENIZER_JSON),
    );
    let out = dir.path().join("model.gguf");
    convert(dir.path(), &out, OutputFormat::Gguf);

    let gguf = Gguf::open(&out).expect("boostr opens the GGUF");
    assert_eq!(gguf.len(), stack_tensors().len());
    assert!(gguf
        .tensor_names()
        .all(|n| !n.starts_with(AUDIO_VAE_PREFIX)));
    assert_eq!(gguf.metadata().get_string(VOXCPM2_CONFIG_JSON_KEY), None);
    assert_eq!(gguf.metadata().get_string(HF_TOKENIZER_JSON_KEY), None);
}

/// `--arch` outranks `config.json` for the companion rule, exactly as it
/// does for `general.architecture`.
#[test]
fn the_arch_override_decides_the_companion() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_checkpoint(dir.path(), "{\"model_type\": \"llama\"}", None);
    let out = dir.path().join("model.gguf");
    let config = ConversionConfig::new(
        dir.path().to_string_lossy().to_string(),
        out.to_string_lossy().to_string(),
        OutputFormat::Gguf,
    )
    .with_quantization(QuantizationConfig::q4_k())
    .with_arch("voxcpm2");
    compressr::convert(config).expect("converts the checkpoint");

    let gguf = Gguf::open(&out).expect("boostr opens the GGUF");
    assert_eq!(gguf.len(), stack_tensors().len() + VAE_NAMES.len());
}
