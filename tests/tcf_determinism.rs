//! Two TCF conversions of one checkpoint, under one set of flags, must
//! produce the same bytes.
//!
//! Determinism is a TCF format guarantee, and it is the property every
//! work-elimination change in the writer risks: a payload read back from pass
//! 1's spill, a payload quantized again in pass 2, and a payload from a run
//! that never spilled at all are all required to be the same bytes. A defect
//! in exactly this area — a nondeterministic tensor order — shipped because
//! nothing compared two runs.
//!
//! The fixture is written by compressr's own SafeTensors writer and read back
//! by `tcf-core`'s reader, never by anything compressr owns.

use compressr::{
    parse_tcf_encoding, parse_tcf_profile, FormatWriter, PlannedTensor, SafeTensorsFormatWriter,
    TcfFormatWriter, TcfOptions, UnifiedDType,
};
use std::path::Path;
use tcf_core::TcfFile;

/// A tensor the fixture holds, and the plan it is meant to exercise.
struct Fixture {
    name: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// Reproducible values with a wide dynamic range, so the screen ranks
/// tensors apart and a profile has something to bump.
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

/// Every plan the TCF writer can settle, in one checkpoint.
fn fixtures() -> Vec<Fixture> {
    let mut nonfinite = values(4 * 640, 0.5);
    nonfinite[513] = f32::NAN;
    // Group extremes at +-4e-7.
    let tiny: Vec<f32> = (0..4 * 640)
        .map(|i| 1e-7 * ((i % 8) as f32 - 4.0))
        .collect();
    vec![
        // Above the parallel path's threshold, and a tile count that is not a
        // multiple of the super-block.
        Fixture {
            name: "block.0.weight",
            shape: vec![65, 640],
            values: values(65 * 640, 1.0),
        },
        // Quantizable, serial path, and a different error from the screen's
        // point of view.
        Fixture {
            name: "block.1.weight",
            shape: vec![4, 640],
            values: values(4 * 640, 0.02),
        },
        // Rank 1: stored raw, no screen, no bump candidacy.
        Fixture {
            name: "block.0.norm",
            shape: vec![4096],
            values: values(4096, 1.0),
        },
        // A row of 96 is not a whole number of 64-element tiles: stored raw.
        Fixture {
            name: "block.1.proj",
            shape: vec![2, 96],
            values: values(192, 1.0),
        },
        // A NaN is invisible to every shape check: the quantizer re-plans it.
        Fixture {
            name: "block.2.weight",
            shape: vec![4, 640],
            values: nonfinite,
        },
        // `(hi - lo) / 15` is a binary16 subnormal, so `Q4AS32D_T64` carries
        // these values, while `max_abs / 30.1` is under 2^-25, so
        // `Q6S16D_T64` cannot. Quantizable at the base and no bump candidate
        // — the case the bump check exists to find.
        Fixture {
            name: "block.3.weight",
            shape: vec![4, 640],
            values: tiny,
        },
    ]
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Write the fixture checkpoint that both conversions read.
fn write_source(path: &Path, tensors: &[Fixture]) {
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

/// Convert the fixture to TCF at `out`, and answer how many tensors a profile
/// bumped.
fn convert(source: &Path, out: &Path, options: TcfOptions, tensors: &[Fixture]) -> usize {
    let mut writer = TcfFormatWriter::new()
        .with_options(options)
        .with_source(source.to_str().expect("a utf8 fixture path"));
    writer.initialize(out).expect("initializes the tcf writer");
    for tensor in tensors {
        writer
            .register_tensor(tensor.name, &tensor.shape, UnifiedDType::F32, None)
            .expect("registers");
    }
    for tensor in tensors {
        writer
            .write_tensor_data(&f32_bytes(&tensor.values))
            .expect("settles a plan");
    }
    writer.finalize().expect("writes the tcf file");
    writer.planned().iter().filter(|plan| plan.bumped).count()
}

/// Convert the fixture at `out` and answer the plan the writer settled for
/// every tensor, in registration order.
fn plans(
    source: &Path,
    out: &Path,
    options: TcfOptions,
    tensors: &[Fixture],
) -> Vec<PlannedTensor> {
    let mut writer = TcfFormatWriter::new()
        .with_options(options)
        .with_source(source.to_str().expect("a utf8 fixture path"));
    writer.initialize(out).expect("initializes the tcf writer");
    for tensor in tensors {
        writer
            .register_tensor(tensor.name, &tensor.shape, UnifiedDType::F32, None)
            .expect("registers");
    }
    for tensor in tensors {
        writer
            .write_tensor_data(&f32_bytes(&tensor.values))
            .expect("settles a plan");
    }
    writer.finalize().expect("writes the tcf file");
    writer.planned().to_vec()
}

/// Read `path` back through `tcf-core` and check every digest and proof.
fn verify(path: &Path, expected_tensors: usize) {
    let bytes = std::fs::read(path).expect("reads the output back");
    let file = TcfFile::open(&bytes).expect("tcf-core accepts the file");
    assert_eq!(file.tensors().len(), expected_tensors);
    for tensor in file.tensors() {
        file.verify_tensor(tensor).expect("digests and proofs hold");
    }
}

/// A `--tcf-encoding` run: one encoding for every tensor, no ranking, no
/// bump.
#[test]
fn two_uniform_conversions_produce_the_same_bytes() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = fixtures();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let encoding = parse_tcf_encoding("q4s32t64").expect("a known encoding");
    let options = TcfOptions::uniform(encoding, encoding);

    let first = dir.path().join("first.tcf");
    let second = dir.path().join("second.tcf");
    assert_eq!(convert(&source, &first, options, &tensors), 0);
    assert_eq!(convert(&source, &second, options, &tensors), 0);

    verify(&first, tensors.len());
    verify(&second, tensors.len());
    let left = std::fs::read(&first).expect("reads the first output");
    let right = std::fs::read(&second).expect("reads the second output");
    assert_eq!(left, right, "two identical runs wrote different bytes");
}

/// A `--tcf-profile` run: the bump set is ranked across tensors, so the
/// encoding a payload is written at is decided after every tensor is screened.
#[test]
fn two_profile_conversions_produce_the_same_bytes() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = fixtures();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let profile = parse_tcf_profile("compact").expect("a ladder rung");
    let options = TcfOptions::from_profile(profile);

    let first = dir.path().join("first.tcf");
    let second = dir.path().join("second.tcf");
    let bumped = convert(&source, &first, options, &tensors);
    assert_eq!(
        convert(&source, &second, options, &tensors),
        bumped,
        "the same allocation must fall out of the same input"
    );
    assert!(
        bumped > 0,
        "the fixture must exercise the bumped payload path"
    );

    verify(&first, tensors.len());
    verify(&second, tensors.len());
    let left = std::fs::read(&first).expect("reads the first output");
    let right = std::fs::read(&second).expect("reads the second output");
    assert_eq!(left, right, "two identical runs wrote different bytes");
}

/// The bump set must be the set a full quantize at the bump encoding produces.
///
/// The screen answers representability from each group's extremes rather than
/// from a quantize it throws away. That is a work-elimination change, so the
/// verdict it reaches for every tensor is checked here against the quantize it
/// replaced — the oracle, run directly against `tcf-core`.
#[test]
fn the_bump_set_matches_a_full_quantize() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let tensors = fixtures();
    let source = dir.path().join("model.safetensors");
    write_source(&source, &tensors);

    let profile = parse_tcf_profile("compact").expect("a ladder rung");
    let bump = profile.bump.expect("the compact rung bumps");
    let out = dir.path().join("plans.tcf");
    let settled = plans(&source, &out, TcfOptions::from_profile(profile), &tensors);
    assert_eq!(settled.len(), tensors.len());

    let mut candidates = 0usize;
    let mut refused = 0usize;
    for (plan, fixture) in settled.iter().zip(tensors.iter()) {
        assert_eq!(plan.name, fixture.name);
        if plan.fallback.is_some() {
            // A tensor stored raw has no base encoding to be bumped from.
            assert!(!plan.bumpable, "{} is raw and no candidate", plan.name);
            continue;
        }
        let dims: Vec<u64> = fixture.shape.iter().map(|d| *d as u64).collect();
        let rank = u32::try_from(dims.len()).expect("a small rank");
        let quantized = tcf_core::encoding::quantize(&fixture.values, &dims, rank, bump.layout());
        let oracle = quantized.is_ok();
        assert_eq!(
            plan.bumpable, oracle,
            "{} disagreed with a full quantize at {bump:?}",
            plan.name
        );
        if oracle {
            candidates += 1;
        } else {
            refused += 1;
        }
    }
    assert!(candidates > 0, "the fixture must hold a bump candidate");
    assert!(
        refused > 0,
        "the fixture must hold a tensor the bump encoding refuses"
    );
}
