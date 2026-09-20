//! The lowbit repack: PTQ1_0 to PQ2_0 and back, and a `--dtype` cast of
//! either.
//!
//! The PTQ1_0 bytes come from boostr's block encoder, the file from
//! compressr's GGUF writer, and every check decodes through boostr's reader
//! kernels. compressr holds no layout of its own to agree with.
//!
//! The real-file case runs only with `BOOSTR_BONSAI2_DIR` set to a directory
//! holding `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and
//! `Ternary-Bonsai-2-27B-PQ2_0.gguf`. Its output goes under the cache
//! directory, never `/tmp`: the files are gigabytes.

use std::path::{Path, PathBuf};

use boostr::format::gguf::{GgmlType as ReaderType, Gguf};
use boostr::quant::cpu::kernels::lowbit_codec::{encode_ptq1_0_block, LOWBIT_BLOCK_SIZE};
use boostr::{CpuDevice, CpuRuntime};
use compressr::formats::{GgmlType, GgufArray, GgufBatchWriter, GgufTensorInfo, GgufValue};
use compressr::{ConversionConfig, OutputFormat, QuantizationConfig, UnifiedDType};
use half::f16;

const WEIGHT: &str = "blk.0.ffn_up.weight";
const NORM: &str = "blk.0.attn_norm.weight";
const ROWS: usize = 4;
const ROW: usize = 2 * LOWBIT_BLOCK_SIZE;

/// PTQ1_0 bytes for `ROWS x ROW` pseudo-random trits, one scale per block.
fn ptq1_0_weight() -> Vec<u8> {
    let mut state = 0x2468_ACE0u32;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state >> 24
    };
    let mut bytes = Vec::new();
    for block in 0..ROWS * ROW / LOWBIT_BLOCK_SIZE {
        let codes: [u8; LOWBIT_BLOCK_SIZE] = std::array::from_fn(|_| (next() % 3) as u8);
        let d = f16::from_f32(0.25 + block as f32 * 0.0625).to_le_bytes();
        bytes.extend_from_slice(&encode_ptq1_0_block(&codes, d).expect("ternary codes pack"));
    }
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A GGUF holding one lowbit weight, one F32 norm and a producer key.
fn write_lowbit_gguf(path: &Path, dtype: GgmlType, weight: Vec<u8>) {
    let mut writer = GgufBatchWriter::new();
    writer.set_model_metadata("qwen35", "synthetic", 143);
    writer.add_u32("prism.hadamard.version", 1);
    writer.add_metadata(
        "prism.hadamard.sign_values",
        GgufValue::Array(GgufArray::Int32(vec![-1, 1, 1, -1])),
    );
    writer.add_metadata(
        "tokenizer.ggml.tokens",
        GgufValue::Array(GgufArray::String(vec!["<s>".into(), "a".into()])),
    );
    writer
        .add_tensor(GgufTensorInfo::new(WEIGHT, vec![ROWS, ROW], dtype, weight))
        .expect("the weight registers");
    let norm: Vec<f32> = (0..ROW).map(|i| 1.0 + i as f32 * 0.001).collect();
    writer
        .add_tensor(GgufTensorInfo::new(
            NORM,
            vec![ROW],
            GgmlType::F32,
            f32_bytes(&norm),
        ))
        .expect("the norm registers");
    writer.write(path).expect("the fixture writes");
}

fn convert(input: &Path, output: &Path, config: impl FnOnce(ConversionConfig) -> ConversionConfig) {
    let base = ConversionConfig::new(
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        OutputFormat::Gguf,
    );
    compressr::convert(config(base)).expect("the conversion succeeds");
}

fn load_f32(gguf: &mut Gguf, name: &str) -> Vec<f32> {
    gguf.load_tensor_f32::<CpuRuntime>(name, &CpuDevice::new())
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .to_vec::<f32>()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn ptq1_0_repacks_to_pq2_0_with_the_header_carried() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");
    write_lowbit_gguf(&input, GgmlType::Ptq1_0, ptq1_0_weight());

    convert(&input, &output, |c| {
        c.with_quantization(QuantizationConfig::pq2_0())
    });

    let mut source = Gguf::open(&input).expect("the input opens");
    let mut result = Gguf::open(&output).expect("the output opens");
    assert_eq!(
        result.tensor_info(WEIGHT).unwrap().ggml_type,
        ReaderType::PQ2_0
    );
    assert_eq!(result.tensor_info(NORM).unwrap().ggml_type, ReaderType::F32);
    assert_eq!(
        result.tensor_info(WEIGHT).unwrap().shape,
        source.tensor_info(WEIGHT).unwrap().shape,
        "the dimension order survives a GGUF round trip"
    );

    assert_eq!(
        bits(&load_f32(&mut source, WEIGHT)),
        bits(&load_f32(&mut result, WEIGHT)),
        "the repacked weight decodes to the same values"
    );
    assert_eq!(
        source.read_tensor_bytes(NORM).unwrap(),
        result.read_tensor_bytes(NORM).unwrap(),
        "the norm passes through untouched"
    );

    let header = result.metadata();
    assert_eq!(header.get_u32("general.file_type"), Some(141));
    assert_eq!(header.get_string("general.architecture"), Some("qwen35"));
    assert_eq!(header.get_string("general.name"), Some("synthetic"));
    assert_eq!(header.get_u32("prism.hadamard.version"), Some(1));
    assert_eq!(
        header.get_i64_array("prism.hadamard.sign_values"),
        Some(vec![-1, 1, 1, -1])
    );
    assert_eq!(
        header.get_string_array("tokenizer.ggml.tokens"),
        Some(vec!["<s>".to_string(), "a".to_string()])
    );
}

#[test]
fn pq2_0_repacks_back_to_the_same_ptq1_0_bytes() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let ptq = dir.path().join("ptq.gguf");
    let pq = dir.path().join("pq.gguf");
    let back = dir.path().join("back.gguf");
    let original = ptq1_0_weight();
    write_lowbit_gguf(&ptq, GgmlType::Ptq1_0, original.clone());

    convert(&ptq, &pq, |c| {
        c.with_quantization(QuantizationConfig::pq2_0())
    });
    convert(&pq, &back, |c| {
        c.with_quantization(QuantizationConfig::ptq1_0())
    });

    let mut result = Gguf::open(&back).expect("the output opens");
    assert_eq!(
        result.tensor_info(WEIGHT).unwrap().ggml_type,
        ReaderType::PTQ1_0
    );
    assert_eq!(result.read_tensor_bytes(WEIGHT).unwrap(), original);
    assert_eq!(result.metadata().get_u32("general.file_type"), Some(143));
}

/// A tensor already in the target format passes through byte for byte.
#[test]
fn a_tensor_already_in_the_target_format_passes_through() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let ptq = dir.path().join("ptq.gguf");
    let pq = dir.path().join("pq.gguf");
    let again = dir.path().join("again.gguf");
    write_lowbit_gguf(&ptq, GgmlType::Ptq1_0, ptq1_0_weight());
    convert(&ptq, &pq, |c| {
        c.with_quantization(QuantizationConfig::pq2_0())
    });

    let err = compressr::convert(
        ConversionConfig::new(
            pq.to_string_lossy().into_owned(),
            again.to_string_lossy().into_owned(),
            OutputFormat::Gguf,
        )
        .with_quantization(QuantizationConfig::pq2_0()),
    )
    .expect_err("a file with no PTQ1_0 tensor has nothing to repack")
    .to_string();
    assert!(err.contains("holds none"), "{err}");
    assert!(!again.exists(), "no output file is created");
}

#[test]
fn a_float_input_cannot_be_quantized_to_ternary() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let input = dir.path().join("f32.gguf");
    let output = dir.path().join("out.gguf");
    let weight: Vec<f32> = (0..ROWS * ROW).map(|i| (i as f32).sin()).collect();
    write_lowbit_gguf(&input, GgmlType::F32, f32_bytes(&weight));

    let err = compressr::convert(
        ConversionConfig::new(
            input.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
            OutputFormat::Gguf,
        )
        .with_quantization(QuantizationConfig::ptq1_0()),
    )
    .expect_err("float weights are not quantized to ternary")
    .to_string();
    assert!(err.contains("not a conversion compressr performs"), "{err}");
    assert!(!output.exists(), "no output file is created");
}

/// `--dtype f16` decodes the lowbit blocks through boostr and casts.
#[test]
fn a_dtype_cast_decodes_the_lowbit_blocks() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let input = dir.path().join("in.gguf");
    let gguf_out = dir.path().join("f16.gguf");
    let st_out = dir.path().join("f16-st");
    write_lowbit_gguf(&input, GgmlType::Ptq1_0, ptq1_0_weight());

    convert(&input, &gguf_out, |c| {
        c.with_target_dtype(UnifiedDType::F16)
    });
    compressr::convert(
        ConversionConfig::new(
            input.to_string_lossy().into_owned(),
            st_out.to_string_lossy().into_owned(),
            OutputFormat::Safetensors,
        )
        .with_target_dtype(UnifiedDType::F16),
    )
    .expect("the SafeTensors cast succeeds");

    let mut source = Gguf::open(&input).expect("the input opens");
    let expected: Vec<u8> = load_f32(&mut source, WEIGHT)
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();

    let mut result = Gguf::open(&gguf_out).expect("the F16 GGUF opens");
    assert_eq!(
        result.tensor_info(WEIGHT).unwrap().ggml_type,
        ReaderType::F16
    );
    assert_eq!(result.read_tensor_bytes(WEIGHT).unwrap(), expected);
    assert_eq!(result.metadata().get_u32("general.file_type"), Some(1));

    let mut st = compressr::ModelLoader::new(&st_out)
        .load()
        .expect("the SafeTensors checkpoint loads");
    let info = st.tensor_info(WEIGHT).expect("the weight is listed");
    assert_eq!(info.dtype, UnifiedDType::F16);
    assert_eq!(info.shape, vec![ROWS, ROW]);
    assert_eq!(st.read_tensor_bytes(WEIGHT).unwrap(), expected);
}

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const REAL_PTQ1_0: &str = "Ternary-Bonsai-2-27B-PTQ1_0.gguf";
const REAL_PQ2_0: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const REAL_TENSORS: &[&str] = &[
    "blk.0.attn_gate.weight",
    "blk.0.attn_qkv.weight",
    "output_norm.weight",
    "blk.0.ssm_alpha.weight",
];

/// The cache directory the gigabyte-scale output goes under.
fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .expect("HOME or XDG_CACHE_HOME is set");
    base.join("compressr-tests")
}

#[test]
fn real_ptq1_0_file_repacks_to_the_shipped_pq2_0_values() {
    let Some(dir) = std::env::var_os(ENV_DIR).map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return;
    };
    let ptq = dir.join(REAL_PTQ1_0);
    let shipped = dir.join(REAL_PQ2_0);
    for path in [&ptq, &shipped] {
        if !path.is_file() {
            println!("skip: {} not found", path.display());
            return;
        }
    }
    let out_dir = cache_dir();
    std::fs::create_dir_all(&out_dir).expect("the cache directory exists");
    let output = out_dir.join("Ternary-Bonsai-2-27B-repacked-PQ2_0.gguf");

    convert(&ptq, &output, |c| {
        c.with_quantization(QuantizationConfig::pq2_0())
    });

    let mut reference = Gguf::open(&shipped).expect("the shipped PQ2_0 opens");
    let mut result = Gguf::open(&output).expect("the repacked file opens");
    for key in ["prism.hadamard.version", "prism.hadamard.block_size"] {
        assert_eq!(
            result.metadata().get_u32(key),
            reference.metadata().get_u32(key),
            "{key}"
        );
    }
    assert_eq!(
        result.metadata().get_string_array("tokenizer.ggml.tokens"),
        reference
            .metadata()
            .get_string_array("tokenizer.ggml.tokens")
    );
    for name in REAL_TENSORS {
        assert_eq!(
            result.tensor_info(name).unwrap().ggml_type,
            reference.tensor_info(name).unwrap().ggml_type,
            "{name}"
        );
        assert_eq!(
            bits(&load_f32(&mut result, name)),
            bits(&load_f32(&mut reference, name)),
            "{name}: repacked values differ from the shipped PQ2_0"
        );
        let same_bytes =
            result.read_tensor_bytes(name).unwrap() == reference.read_tensor_bytes(name).unwrap();
        println!("{name}: bytes identical to the shipped file: {same_bytes}");
    }
    std::fs::remove_file(&output).ok();
}
