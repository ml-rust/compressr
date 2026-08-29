//! Basic integration tests for compressr

use compressr::{OutputFormat, QuantizationConfig, QuantizationMethod};

#[test]
fn test_quantization_config_creation() {
    let gptq = QuantizationConfig::gptq(4, 128);
    assert_eq!(gptq.method, QuantizationMethod::Gptq);
    assert_eq!(gptq.bits, 4);
    assert_eq!(gptq.group_size, 128);

    let awq = QuantizationConfig::awq(4, 128);
    assert_eq!(awq.method, QuantizationMethod::Awq);
    assert!(awq.calibration_dataset.is_some());

    let nf4 = QuantizationConfig::nf4();
    assert_eq!(nf4.method, QuantizationMethod::Nf4);
    assert_eq!(nf4.bits, 4);

    let fp8 = QuantizationConfig::fp8();
    assert_eq!(fp8.method, QuantizationMethod::Fp8);
    assert_eq!(fp8.bits, 8);
}

#[test]
fn test_output_format_variants() {
    let formats = [
        OutputFormat::Gguf,
        OutputFormat::Awq,
        OutputFormat::Gptq,
        OutputFormat::Exl2,
        OutputFormat::Fp8,
        OutputFormat::Safetensors,
        OutputFormat::Tcf,
    ];

    // Just verify they can be created
    assert_eq!(formats.len(), 7);
}

// ============================================================================
// dtype cast conversion
// ============================================================================

use compressr::formats::safetensors::{SafeTensorsDType, SafeTensorsWriter};
use compressr::{ConversionConfig, UnifiedDType};
use std::path::Path;

/// Split a SafeTensors file into its parsed JSON header and its data block.
fn read_safetensors(path: &Path) -> (serde_json::Value, Vec<u8>) {
    let bytes = std::fs::read(path).expect("output weights file must exist");
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("header must be valid JSON");
    (header, bytes[8 + header_len..].to_vec())
}

/// Read the bytes of one tensor out of a SafeTensors data block.
fn tensor_bytes<'a>(header: &serde_json::Value, data: &'a [u8], name: &str) -> &'a [u8] {
    let offsets = &header[name]["data_offsets"];
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;
    &data[start..end]
}

fn u16le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn test_convert_casts_floats_to_bf16_and_keeps_the_checkpoint_loadable() {
    let input = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();

    // 1.005859375 (0x3F80_C000) is the value that separates the two rounding
    // rules: round-to-nearest-even gives bf16 0x3F81, truncation gives 0x3F80.
    let weights: [f32; 4] = [1.0, f32::from_bits(0x3F80_C000), -2.0, 0.5];
    let weight_bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

    // A non-float tensor that must survive byte-for-byte.
    let ids: [i64; 3] = [7, -1, 4096];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut writer = SafeTensorsWriter::new();
    writer
        .add_tensor(
            "model.embed_tokens.weight",
            vec![2, 2],
            SafeTensorsDType::F32,
            weight_bytes,
        )
        .unwrap();
    writer
        .add_tensor(
            "model.position_ids",
            vec![3],
            SafeTensorsDType::I64,
            id_bytes.clone(),
        )
        .unwrap();
    writer
        .write(input.path().join("model.safetensors"))
        .unwrap();

    std::fs::write(
        input.path().join("config.json"),
        r#"{"model_type":"llama","torch_dtype":"float32","hidden_size":2}"#,
    )
    .unwrap();
    std::fs::write(input.path().join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    std::fs::write(
        input.path().join("tokenizer_config.json"),
        r#"{"bos_token":"<s>"}"#,
    )
    .unwrap();

    let out_dir = output.path().join("model-bf16");
    let config = ConversionConfig::new(
        input.path().to_string_lossy().to_string(),
        out_dir.to_string_lossy().to_string(),
        OutputFormat::Safetensors,
    )
    .with_target_dtype(UnifiedDType::BF16);

    let stats = compressr::convert(config).expect("conversion must succeed");
    assert_eq!(stats.tensor_count, 2);

    // Float tensor: cast to bf16 with round-to-nearest-even.
    let (header, data) = read_safetensors(&out_dir.join("model.safetensors"));
    assert_eq!(header["model.embed_tokens.weight"]["dtype"], "BF16");
    assert_eq!(header["model.embed_tokens.weight"]["shape"][0], 2);
    assert_eq!(header["model.embed_tokens.weight"]["shape"][1], 2);
    assert_eq!(
        u16le(tensor_bytes(&header, &data, "model.embed_tokens.weight")),
        vec![0x3F80, 0x3F81, 0xC000, 0x3F00]
    );

    // Non-float tensor: dtype and bytes untouched.
    assert_eq!(header["model.position_ids"]["dtype"], "I64");
    assert_eq!(
        tensor_bytes(&header, &data, "model.position_ids"),
        id_bytes.as_slice()
    );

    // Sidecars: present, and config.json now advertises the new dtype.
    let out_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir.join("config.json")).unwrap())
            .unwrap();
    assert_eq!(out_config["torch_dtype"], "bfloat16");
    assert_eq!(out_config["model_type"], "llama");
    assert_eq!(out_config["hidden_size"], 2);

    assert_eq!(
        std::fs::read_to_string(out_dir.join("tokenizer.json")).unwrap(),
        r#"{"version":"1.0"}"#
    );
    assert_eq!(
        std::fs::read_to_string(out_dir.join("tokenizer_config.json")).unwrap(),
        r#"{"bos_token":"<s>"}"#
    );
}

#[test]
fn test_convert_to_bf16_halves_the_weight_bytes() {
    let input = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();

    let numel = 64 * 64;
    let weight_bytes: Vec<u8> = (0..numel)
        .flat_map(|i| (i as f32 * 0.125).to_le_bytes())
        .collect();
    assert_eq!(weight_bytes.len(), numel * 4);

    let mut writer = SafeTensorsWriter::new();
    writer
        .add_tensor("w", vec![64, 64], SafeTensorsDType::F32, weight_bytes)
        .unwrap();
    let in_file = input.path().join("model.safetensors");
    writer.write(&in_file).unwrap();

    let out_dir = output.path().join("out");
    let config = ConversionConfig::new(
        input.path().to_string_lossy().to_string(),
        out_dir.to_string_lossy().to_string(),
        OutputFormat::Safetensors,
    )
    .with_target_dtype(UnifiedDType::BF16);

    let stats = compressr::convert(config).unwrap();

    assert_eq!(stats.input_size, (numel * 4) as u64);
    assert_eq!(stats.output_size, (numel * 2) as u64);
    assert_eq!(stats.compression_ratio, 2.0);

    // The file itself is half the input plus the small JSON header.
    let in_len = std::fs::metadata(&in_file).unwrap().len();
    let out_len = std::fs::metadata(out_dir.join("model.safetensors"))
        .unwrap()
        .len();
    assert!(
        out_len < in_len * 3 / 5,
        "output {} bytes is not roughly half of input {} bytes",
        out_len,
        in_len
    );
}

#[test]
fn test_convert_rejects_dtype_with_quantization() {
    let config =
        ConversionConfig::new("input/".into(), "output/".into(), OutputFormat::Safetensors)
            .with_target_dtype(UnifiedDType::BF16)
            .with_quantization(QuantizationConfig::q4_k());

    let err = compressr::convert(config).unwrap_err().to_string();
    assert_eq!(
        err,
        "--dtype and --quantization are mutually exclusive: quantization already \
         determines the stored representation. Pick one."
    );
}

#[test]
fn test_parse_target_dtype_rejects_unknown_value() {
    let err = compressr::parse_target_dtype("bfloat")
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "Unsupported dtype: bfloat. Accepted: f32, fp32, float32, float, f16, fp16, float16, half, bf16, bfloat16, brain16"
    );
}

#[test]
fn test_parse_target_dtype_accepts_aliases() {
    assert_eq!(
        compressr::parse_target_dtype("BFloat16").unwrap(),
        UnifiedDType::BF16
    );
    assert_eq!(
        compressr::parse_target_dtype("fp32").unwrap(),
        UnifiedDType::F32
    );
    assert_eq!(
        compressr::parse_target_dtype("half").unwrap(),
        UnifiedDType::F16
    );
}
