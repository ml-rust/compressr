//! The input formats compressr reads, exercised end to end.
//!
//! Two gaps these tests hold shut:
//!
//! 1. compressr could not read the format it writes. A TCF file is converted
//!    here and read back through the input path, so the writer and the reader
//!    are checked against each other over a whole file rather than over one
//!    encoder's opinion of itself.
//! 2. PyTorch input was refused. The fixtures are produced by PyTorch 2.8
//!    itself and committed, so the suite needs no Python: `.pth` files carry
//!    conventions (a `state_dict` root, a weight-norm pair) that only a real
//!    `torch.save` produces.

use compressr::{
    ConversionConfig, FormatWriter, InputFormat, ModelLoader, OutputFormat, QuantizationConfig,
    SafeTensorsFormatWriter, UnifiedDType,
};
use std::path::{Path, PathBuf};

/// Directory holding the committed PyTorch fixtures.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Reproducible values with enough spread for a quantizer to work on.
fn values(count: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
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

/// One tensor of the SafeTensors fixture a TCF conversion starts from.
struct Source {
    name: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// A rank-2 tensor the encoding accepts, and a rank-1 tensor it refuses.
fn sources() -> Vec<Source> {
    vec![
        Source {
            name: "block.0.weight",
            shape: vec![8, 128],
            values: values(8 * 128),
        },
        Source {
            name: "block.1.weight",
            shape: vec![4, 64],
            values: values(4 * 64),
        },
        // Rank 1 has no row to tile, so TCF stores it raw and records why.
        Source {
            name: "block.0.norm",
            shape: vec![256],
            values: values(256),
        },
    ]
}

/// Write the SafeTensors checkpoint a conversion reads.
fn write_source(path: &Path, tensors: &[Source]) {
    let mut writer = SafeTensorsFormatWriter::new();
    for tensor in tensors {
        writer
            .register_tensor(tensor.name, &tensor.shape, UnifiedDType::F32, None)
            .expect("registers a fixture tensor");
    }
    writer.initialize(path).expect("initializes the fixture");
    for tensor in tensors {
        writer
            .write_tensor_data(&f32_bytes(&tensor.values))
            .expect("writes a fixture tensor");
    }
    writer.finalize().expect("finalizes the fixture");
}

/// Convert the fixture to a TCF file at `out`, every quantizable tensor at
/// Q8_0.
fn convert_to_tcf(source: &Path, out: &Path) {
    let config = ConversionConfig::new(
        source.to_string_lossy().to_string(),
        out.to_string_lossy().to_string(),
        OutputFormat::Tcf,
    )
    .with_quantization(QuantizationConfig::q8_0());
    compressr::convert(config).expect("converts the fixture to TCF");
}

/// A TCF file compressr wrote must come back through compressr's own input
/// path with the inventory it was written from.
#[test]
fn a_tcf_file_reads_back_with_the_inventory_it_was_written_from() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = sources();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let out = dir.path().join("model.tcf");
    convert_to_tcf(&source, &out);

    let info = compressr::get_model_info(out.to_str().expect("a utf8 path"))
        .expect("reads the TCF file back");

    assert_eq!(info.format, "Tcf");
    assert_eq!(info.tensor_count, tensors.len());

    // The conversion visits a SafeTensors checkpoint in sorted name order, so
    // the TCF file's own order is that one, not the fixture's.
    for expected in &tensors {
        let summary = info
            .tensors
            .iter()
            .find(|t| t.name == expected.name)
            .unwrap_or_else(|| panic!("{} is missing from the TCF file", expected.name));
        let shape: Vec<i64> = expected.shape.iter().map(|d| *d as i64).collect();
        assert_eq!(summary.shape, shape, "{}", expected.name);

        let facts = summary
            .tcf
            .as_ref()
            .unwrap_or_else(|| panic!("{} carries no TCF facts", expected.name));
        if expected.shape.len() == 1 {
            // Rank 1 holds no block row, so it is stored raw and the record
            // says so. A reader that lost the reason could not tell this
            // tensor from one the producer chose to keep at full precision.
            assert!(
                facts.fallback_reason.is_some(),
                "{} is raw and must record a reason",
                expected.name
            );
        } else {
            assert_eq!(facts.encoding, "Q8_0", "{}", expected.name);
            assert_eq!(facts.fallback_reason, None, "{}", expected.name);
            assert!(facts.bits_per_weight.is_some(), "{}", expected.name);
        }
    }

    // The structure TCF has and GGUF does not: a module with its own encoding
    // preference, and the record counts a placement decision reads.
    let facts = info.tcf.expect("a TCF file states its own structure");
    assert_eq!(facts.major, 1);
    assert_eq!(facts.modules.len(), 1);
    let module = &facts.modules[0];
    assert_eq!(module.tensor_count, tensors.len());
    assert_eq!(
        module.preferred_encodings.first().map(String::as_str),
        Some("Q8_0")
    );
    assert_eq!(facts.fallback_tensors, 1);
    assert_eq!(
        facts.file_len,
        std::fs::metadata(&out).expect("the output exists").len()
    );
}

/// Reading a TCF file must return the weights it stores, not just its
/// directory. The tolerance is the encoding's, and a reader that decoded the
/// wrong plane or tile would miss it by far more.
#[test]
fn a_tcf_file_decodes_to_the_weights_it_was_quantized_from() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = sources();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let tcf = dir.path().join("model.tcf");
    convert_to_tcf(&source, &tcf);

    let mut model = ModelLoader::new(&tcf).load().expect("opens the TCF file");
    assert_eq!(model.format(), InputFormat::Tcf);

    for expected in &tensors {
        let bytes = model
            .read_tensor_bytes(expected.name)
            .expect("decodes a tensor");
        let decoded: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(decoded.len(), expected.values.len(), "{}", expected.name);

        let worst = decoded
            .iter()
            .zip(expected.values.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        // Q8_0 carries 8 bits over a block of 32; the fixture's values span
        // roughly [-0.5, 0.5], so a correct decode lands well inside a
        // hundredth. A raw-stored tensor is exact.
        assert!(worst < 0.01, "{} decoded {worst} away", expected.name);
    }
}

/// TCF is quantized once, from full precision. A TCF input would be quantized
/// a second time, from values the first pass already rounded.
#[test]
fn tcf_output_from_a_tcf_input_is_refused() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = sources();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let tcf = dir.path().join("model.tcf");
    convert_to_tcf(&source, &tcf);

    let out = dir.path().join("again.tcf");
    let err = compressr::convert(
        ConversionConfig::new(
            tcf.to_string_lossy().to_string(),
            out.to_string_lossy().to_string(),
            OutputFormat::Tcf,
        )
        .with_quantization(QuantizationConfig::q8_0()),
    )
    .expect_err("a TCF input cannot produce a TCF file")
    .to_string();

    assert!(err.contains("full-precision checkpoint"), "{err}");
    assert!(!out.exists(), "the refusal must precede any output file");
}

/// The magic decides, not the extension: a TCF file renamed by a download or
/// a benchmark harness is still a TCF file.
#[test]
fn tcf_is_detected_by_its_magic_under_any_extension() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = sources();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let tcf = dir.path().join("model.tcf");
    convert_to_tcf(&source, &tcf);

    let renamed = dir.path().join("model.weights");
    std::fs::copy(&tcf, &renamed).expect("copies the output");
    assert_eq!(
        ModelLoader::new(&renamed)
            .detect_format()
            .expect("detects by content"),
        InputFormat::Tcf
    );
}

/// The converse: a `.tcf` extension on a file that is not TCF must name what
/// was actually found, so the mistake is diagnosable from the message alone.
#[test]
fn a_file_named_tcf_that_is_not_tcf_names_its_own_first_bytes() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let impostor = dir.path().join("model.tcf");
    std::fs::write(&impostor, b"not a tcf file at all").expect("writes the impostor");

    let err = ModelLoader::new(&impostor)
        .detect_format()
        .expect_err("the magic does not hold")
        .to_string();

    assert!(err.contains("TCF magic"), "{err}");
    assert!(err.contains("6e 6f 74 20"), "{err}");
    assert!(!err.contains("python"), "{err}");
}

/// A `.pth` saved as `{"metadata": ..., "state_dict": {...}}` arrives with
/// that root carried into every key, and a `weight_norm` module stores a
/// `(weight_g, weight_v)` pair instead of a weight. Both are conventions of
/// the file, not of the tensor data, so both are resolved on read.
#[test]
fn a_pytorch_state_dict_strips_its_root_and_folds_weight_norm() {
    let path = fixtures().join("weight_norm_state_dict.pth");
    let info = compressr::get_model_info(path.to_str().expect("a utf8 path"))
        .expect("reads the PyTorch fixture");

    assert_eq!(info.format, "PyTorch");
    let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["norm.bias", "norm.weight", "proj.weight"]);

    let proj = info
        .tensors
        .iter()
        .find(|t| t.name == "proj.weight")
        .expect("the folded weight");
    assert_eq!(proj.shape, vec![4, 3, 2]);
    assert_eq!(proj.dtype, "F32");
}

/// The fold must reproduce what `torch.nn.utils.weight_norm` reconstructs.
///
/// The expected values come from PyTorch 2.8 evaluating
/// `g * v / ||v||` on the fixture, not from a second implementation of the
/// same arithmetic here — compressr folds through `boostr::nn::fuse_weight_norm`
/// and nothing else may hold a copy of it.
#[test]
fn the_weight_norm_fold_matches_pytorch() {
    const EXPECTED: [f32; 24] = [
        -0.975_184_4,
        0.470_426_9,
        1.068_850_9,
        -1.650_887_5,
        -0.198_567_32,
        0.339_058_76,
        -0.395_924_7,
        -0.551_902_83,
        -0.524_268_2,
        -0.751_550_14,
        -0.050_037_23,
        0.840_824_66,
        -0.561_215,
        0.037_671_536,
        -0.596_417_37,
        0.216_853_89,
        -0.373_234_87,
        -0.242_882_88,
        -0.807_565_7,
        -0.140_083_7,
        0.430_230_65,
        0.189_418_8,
        0.549_350_4,
        0.552_073_54,
    ];

    let path = fixtures().join("weight_norm_state_dict.pth");
    let mut model = ModelLoader::new(&path).load().expect("opens the fixture");
    let bytes = model
        .read_tensor_bytes("proj.weight")
        .expect("folds the pair");
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    assert_eq!(got.len(), EXPECTED.len());
    for (index, (got, want)) in got.iter().zip(EXPECTED.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "element {index}: folded {got}, PyTorch reconstructs {want}"
        );
    }
}

/// A plain state dict has no root to strip and no pair to fold, so its keys
/// arrive untouched.
#[test]
fn a_plain_pytorch_state_dict_keeps_its_keys() {
    let path = fixtures().join("plain_state_dict.pt");
    let info = compressr::get_model_info(path.to_str().expect("a utf8 path"))
        .expect("reads the PyTorch fixture");

    let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["a.bias", "a.weight"]);
    assert_eq!(info.tensor_count, 2);
}

/// A `torch.save` archive is a ZIP container, so it is recognized whatever it
/// is called. `.bin`, `.pt` and `.pth` are conventions, and a checkpoint
/// downloaded under another name still has to load.
#[test]
fn pytorch_is_detected_by_its_container_under_any_extension() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let renamed = dir.path().join("checkpoint.weights");
    std::fs::copy(fixtures().join("plain_state_dict.pt"), &renamed).expect("copies the fixture");

    assert_eq!(
        ModelLoader::new(&renamed)
            .detect_format()
            .expect("detects by content"),
        InputFormat::PyTorch
    );
}

/// A PyTorch input must convert, not merely open.
#[test]
fn a_pytorch_state_dict_converts_to_safetensors() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let out = dir.path().join("converted");
    let stats = compressr::convert(ConversionConfig::new(
        fixtures()
            .join("weight_norm_state_dict.pth")
            .to_string_lossy()
            .to_string(),
        out.to_string_lossy().to_string(),
        OutputFormat::Safetensors,
    ))
    .expect("converts the PyTorch fixture");

    assert_eq!(stats.tensor_count, 3);

    let info =
        compressr::get_model_info(out.join("model.safetensors").to_str().expect("a utf8 path"))
            .expect("reads the conversion back");
    let names: Vec<&str> = info.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["norm.bias", "norm.weight", "proj.weight"]);
}
