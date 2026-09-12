//! A `--format tcf --quantization` run stores the bytes a `--format gguf`
//! run of the same checkpoint stores, tensor for tensor.
//!
//! That byte identity is the whole claim of the block encodings: TCF is never
//! behind the GGUF of the same model, because it IS the GGUF's block streams
//! in TCF's container. Both files are read back by the crates that own their
//! readers — boostr for both — never by anything compressr owns.

use boostr::format::tcf::TcfLoader;
use boostr::format::Gguf;
use boostr::quant::{DequantOps, QuantFormat};
use boostr::tcf::{
    BlockEncoding, DotAccumulator, Encoding, FallbackReason, InputRepresentation, TcfFile,
};
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType};
use compressr::formats::safetensors::{SafeTensorsDType, SafeTensorsWriter};
use compressr::{ConversionConfig, OutputFormat, QuantizationConfig};
use std::path::{Path, PathBuf};

fn values(count: usize, spread: f32) -> Vec<f32> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..count)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as f32) / 16_777_216.0 - 0.5;
            unit * spread * (1.0 + (i % 11) as f32)
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A checkpoint with a quantizable weight, a rank-1 norm the plan stores as
/// F32, a weight whose row of 96 is whole 32-element blocks but no whole
/// 256-element one, and an integer buffer that passes through.
fn write_source(dir: &Path) -> PathBuf {
    let mut writer = SafeTensorsWriter::new();
    writer
        .add_tensor(
            "block.0.weight",
            vec![8, 512],
            SafeTensorsDType::F32,
            f32_bytes(&values(8 * 512, 1.0)),
        )
        .expect("adds");
    writer
        .add_tensor(
            "block.0.norm",
            vec![512],
            SafeTensorsDType::F32,
            f32_bytes(&values(512, 1.0)),
        )
        .expect("adds");
    writer
        .add_tensor(
            "block.0.proj",
            vec![4, 96],
            SafeTensorsDType::F32,
            f32_bytes(&values(4 * 96, 1.0)),
        )
        .expect("adds");
    writer
        .add_tensor(
            "block.0.ids",
            vec![3],
            SafeTensorsDType::I32,
            [7i32, 8, 9].iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .expect("adds");
    let path = dir.join("model.safetensors");
    writer.write(&path).expect("writes the fixture");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"llama","torch_dtype":"float32","hidden_size":512}"#,
    )
    .expect("writes config");
    path
}

fn convert(source: &Path, out: &Path, format: OutputFormat, quant: QuantizationConfig) {
    let config = ConversionConfig::new(
        source.to_string_lossy().to_string(),
        out.to_string_lossy().to_string(),
        format,
    )
    .with_quantization(quant);
    compressr::convert(config).expect("conversion succeeds");
}

/// Both outputs of one checkpoint under one quantization.
fn convert_both(quant: QuantizationConfig) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let source = write_source(dir.path());
    let gguf = dir.path().join("model.gguf");
    let tcf = dir.path().join("model.tcf");
    convert(&source, &gguf, OutputFormat::Gguf, quant.clone());
    convert(&source, &tcf, OutputFormat::Tcf, quant);
    (dir, gguf, tcf)
}

fn tcf_payloads_match_gguf(quant: QuantizationConfig, block: BlockEncoding, format: QuantFormat) {
    let (_dir, gguf_path, tcf_path) = convert_both(quant);

    let mut gguf = Gguf::open(&gguf_path).expect("boostr opens the gguf");
    let bytes = std::fs::read(&tcf_path).expect("reads the tcf");
    let file = TcfFile::open(&bytes).expect("tcf-core accepts the file");
    assert_eq!(file.tensors().len(), 4);

    // Every tensor's payload is the GGUF's bytes for that tensor, and the
    // quantized one is the block encoding of the GGUF's ggml_type. The norm
    // has rank 1, the projection's row is 96, the ids are integers.
    let mut seen = 0;
    for tensor in file.tensors() {
        let name = file.string(tensor.name).expect("name resolves");
        let payload = file.payload(tensor).expect("payload");
        let from_gguf = gguf.read_tensor_bytes(name).expect("gguf tensor bytes");
        assert_eq!(
            payload,
            from_gguf.as_slice(),
            "{name}: payload differs from the GGUF"
        );
        let contract = file.contract(tensor).expect("contract resolves");
        assert_eq!(
            contract.input_representation,
            InputRepresentation::GgmlReference
        );
        assert_eq!(contract.dot_accumulator, DotAccumulator::GgmlReference);
        assert_eq!(contract.quant_group, 0);
        let expected_reason = match name {
            "block.0.weight" => {
                assert_eq!(tensor.encoding, Encoding::Block(block));
                FallbackReason::None
            }
            "block.0.norm" => FallbackReason::RankLt2,
            // A row of 96 is three 32-element blocks and no whole 256 one.
            "block.0.proj" if 96 % block.block_elems() == 0 => {
                assert_eq!(tensor.encoding, Encoding::Block(block));
                FallbackReason::None
            }
            "block.0.proj" => FallbackReason::ShapeIncompatibleWithEncoding,
            "block.0.ids" => FallbackReason::UnsupportedEncoding,
            other => panic!("unexpected tensor {other}"),
        };
        assert_eq!(tensor.fallback_reason, expected_reason, "{name}");
        seen += 1;
    }
    assert_eq!(seen, 4);

    // boostr reads it back as the GGUF-format tensor, proofs included, and
    // dequantizes to what the GGUF dequantizes to.
    let loader = TcfLoader::open(&tcf_path).expect("boostr opens the tcf");
    loader.verify_all().expect("digests and proofs hold");
    assert_eq!(
        loader.quant_format("block.0.weight").expect("format"),
        format
    );
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let qt = loader
        .load_quant_tensor::<CpuRuntime>("block.0.weight", &device)
        .expect("loads packed");
    let from_tcf = client.dequantize(&qt, DType::F32).expect("dequantizes");
    let from_gguf = gguf
        .load_tensor_f32::<CpuRuntime>("block.0.weight", &device)
        .expect("gguf dequantizes");
    assert_eq!(from_tcf.to_vec::<f32>(), from_gguf.to_vec::<f32>());
}

#[test]
fn q8_0_in_tcf_is_the_gguf_byte_for_byte() {
    tcf_payloads_match_gguf(
        QuantizationConfig::q8_0(),
        BlockEncoding::Q8_0,
        QuantFormat::Q8_0,
    );
}

#[test]
fn q4_k_in_tcf_is_the_gguf_byte_for_byte() {
    tcf_payloads_match_gguf(
        QuantizationConfig::q4_k(),
        BlockEncoding::Q4K,
        QuantFormat::Q4K,
    );
}

/// Two block conversions of one checkpoint produce the same bytes.
#[test]
fn two_block_conversions_produce_the_same_bytes() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let source = write_source(dir.path());
    let first = dir.path().join("first.tcf");
    let second = dir.path().join("second.tcf");
    convert(
        &source,
        &first,
        OutputFormat::Tcf,
        QuantizationConfig::q4_k(),
    );
    convert(
        &source,
        &second,
        OutputFormat::Tcf,
        QuantizationConfig::q4_k(),
    );
    assert_eq!(
        std::fs::read(&first).expect("reads"),
        std::fs::read(&second).expect("reads")
    );
}
