//! Round-trip checks for every GGUF block format compressr writes.
//!
//! The bytes come from `boostr::QuantizeOps`; compressr only chooses the format
//! and hands the blocks to a file writer. These tests cover the whole path —
//! compressr's dispatch, boostr's writer, boostr's reader — with the same
//! readers blazr and llama.cpp-compatible loaders use. Checking a writer
//! against a reader that lives next to it proves nothing: compressr once
//! shipped a Q6_K writer whose field order and nibble interleave were both
//! wrong, and its own in-crate reader agreed with it, so every self-consistency
//! check passed while real readers produced NaN. That writer is gone; this
//! suite is what keeps a replacement from growing back.
//!
//! Each test asserts two things:
//! 1. every decoded value is FINITE — a misplaced `d` field alone breaks this,
//! 2. the relative RMS error stays inside the band below. A wrong interleave
//!    permutes weights within a block and lands near 140%.

use boostr::quant::cpu::kernels::dequant;
use boostr::quant::{QuantFormat, QuantizeOps};
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};
use compressr::loader::UnifiedDType;
use compressr::quantization::{QuantizationConfig, QuantizationMethod, Quantizer};
use half::f16;

/// 1024 elements: 4 K-quant super-blocks, or 32 simple 32-element blocks.
/// More than one block, so a wrong block stride shows up.
const N: usize = 1024;

/// Elements in one K-quant super-block.
const SUPER_BLOCK: usize = 256;

/// Deterministic synthetic weights.
///
/// An inline LCG feeds an Irwin-Hall sum (12 uniforms minus 6) for an
/// approximately standard-normal distribution, then each 16-element group is
/// scaled by a cycling amplitude so per-sub-block scales genuinely differ.
/// A constant or single-scale input can pass with a wrong nibble interleave.
fn synthetic_weights() -> Vec<f32> {
    const AMPLITUDES: [f32; 5] = [1.0, 0.75, 1.5, 0.9, 1.25];

    let mut state: u32 = 0x1234_5678;
    let mut next_unit = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };

    (0..N)
        .map(|i| {
            let mut acc = -6.0f32;
            for _ in 0..12 {
                acc += next_unit();
            }
            acc * AMPLITUDES[(i / 16) % AMPLITUDES.len()]
        })
        .collect()
}

fn to_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// RMS of the error divided by RMS of the signal, after asserting finiteness.
fn relative_rms(original: &[f32], decoded: &[f32]) -> f32 {
    assert_eq!(original.len(), decoded.len(), "length mismatch");

    let mut err = 0.0f64;
    let mut signal = 0.0f64;
    for (i, (&a, &b)) in original.iter().zip(decoded.iter()).enumerate() {
        assert!(b.is_finite(), "decoded element {i} is not finite: {b}");
        err += ((a - b) as f64).powi(2);
        signal += (a as f64).powi(2);
    }

    (err / signal).sqrt() as f32
}

/// Quantize the synthetic tensor, decode it with `kernel`, return relative RMS.
fn round_trip(
    config: QuantizationConfig,
    block_elems: usize,
    block_bytes: usize,
    kernel: fn(&[u8], &mut [f32]),
) -> f32 {
    let weights = synthetic_weights();
    let quantized = Quantizer::new(config)
        .quantize_tensor(&to_le_bytes(&weights), &[N], UnifiedDType::F32)
        .expect("quantization must succeed");

    assert_eq!(
        quantized.packed_data.len(),
        (N / block_elems) * block_bytes,
        "packed size must match the GGML block stride"
    );

    let mut decoded = vec![0.0f32; N];
    kernel(&quantized.packed_data, &mut decoded);

    relative_rms(&weights, &decoded)
}

fn assert_within(name: &str, error: f32, bound: f32) {
    println!("{name}: relative RMS = {:.3}%", error * 100.0);
    assert!(
        error < bound,
        "{name} relative RMS {:.3}% exceeds the {:.1}% bound for this bit width \
         — the packed layout does not match what boostr reads",
        error * 100.0,
        bound * 100.0
    );
}

fn q4_1_config() -> QuantizationConfig {
    QuantizationConfig {
        method: QuantizationMethod::Q4_1,
        bits: 4,
        group_size: 32,
        calibration_dataset: None,
        calibration_samples: 0,
    }
}

// Bounds for the simple formats (Q4_0, Q4_1, Q8_0) come from the format: the
// expected error is step/sqrt(12) relative to sigma, where step is the group
// range divided by the number of levels.
//
// Bounds for the K-quants (Q4_K, Q5_K, Q6_K) do NOT. Those writers run
// llama.cpp's iterative per-sub-block scale search, so their error sits well
// below the plain absmax fit the bit width alone would predict — the bound
// encodes what the SEARCH achieves, not the format's floor. A bound loose
// enough to pass with the search disabled would pass a silent regression back
// to absmax, which is the failure worth catching. `boostr`'s own
// `q4k_search_beats_absmax` test pins the same claim from the other side.

#[test]
fn q4_0_round_trips_through_boostr() {
    // 32-element group, 16 levels, symmetric: expected ~8%.
    let error = round_trip(QuantizationConfig::q4_0(), 32, 18, dequant::dequant_q4_0);
    assert_within("Q4_0", error, 0.12);
}

#[test]
fn q4_1_round_trips_through_boostr() {
    // 32-element group, 16 levels, asymmetric: expected ~8%.
    let error = round_trip(q4_1_config(), 32, 20, dequant::dequant_q4_1);
    assert_within("Q4_1", error, 0.12);
}

#[test]
fn q8_0_round_trips_through_boostr() {
    // 32-element group, 256 levels: expected ~0.5%, against a ~0.57% floor.
    let error = round_trip(QuantizationConfig::q8_0(), 32, 34, dequant::dequant_q8_0);
    assert_within("Q8_0", error, 0.008);
}

#[test]
fn q4_k_round_trips_through_boostr() {
    // Search result, not the bit width: absmax scores ~8% here, the search ~7%.
    let error = round_trip(QuantizationConfig::q4_k(), 256, 144, dequant::dequant_q4k);
    assert_within("Q4_K", error, 0.09);
}

#[test]
fn q5_k_round_trips_through_boostr() {
    // Search result, not the bit width: 32 levels plus the sweep land near 4%.
    let error = round_trip(QuantizationConfig::q5_k(), 256, 176, dequant::dequant_q5k);
    assert_within("Q5_K", error, 0.05);
}

#[test]
fn q6_k_round_trips_through_boostr() {
    // Search result, not the bit width: the symmetric fit lands near 2%.
    let error = round_trip(QuantizationConfig::q6_k(), 256, 210, dequant::dequant_q6k);
    assert_within("Q6_K", error, 0.025);
}

/// compressr's bytes must BE boostr's bytes, not merely close to them.
///
/// The round-trip bounds above would still pass if compressr grew a second
/// K-quant writer that was nearly as good — which is how the duplication got
/// here the first time. Byte equality leaves no room for a near miss.
#[test]
fn q4_k_output_is_byte_identical_to_boostr() {
    let weights = synthetic_weights();

    let quantized = Quantizer::new(QuantizationConfig::q4_k())
        .quantize_tensor(&to_le_bytes(&weights), &[N], UnifiedDType::F32)
        .expect("quantization must succeed");

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(&weights, &[N], &device)
        .expect("staging the reference tensor must succeed");
    let reference = client
        .quantize(&input, QuantFormat::Q4K)
        .expect("boostr quantization must succeed")
        .to_bytes()
        .expect("reading back the reference blocks must succeed");

    assert_eq!(
        quantized.packed_data, reference,
        "compressr's Q4_K bytes diverged from boostr's — a second writer is back"
    );
}

/// Pin the Q6_K field order without running a decoder.
///
/// GGML's `block_q6_K` is `ql[128] || qh[64] || scales[16] || d`, so the f16
/// scale occupies the LAST two bytes of each 210-byte block. compressr used to
/// write `d` first, which made every reader take two quant bytes as the scale —
/// arbitrary bit patterns, many of them f16 NaN.
///
/// `d` is checked by MAGNITUDE, not sign. Q6_K scales are signed: llama.cpp
/// takes `iscale = -128 / max_scale`, so `d = 1/iscale` is negative whenever
/// the extreme sub-block scale is positive, and the sign cancels against the
/// matching negative entry in `scales`. An earlier version of this test
/// asserted `d > 0`, which only held for compressr's own non-conformant
/// writer and failed the moment the writer became llama.cpp-equivalent.
///
/// The magnitude bound is what makes this a layout check: a dequantized value
/// is `d * scales[j] * q` with `|scales[j]| <= 128` and `|q| <= 32`, so
/// `|d| * 4096` must land near the block's own peak. Two quant bytes read as
/// an f16 miss that by orders of magnitude.
#[test]
fn q6_k_writes_the_scale_in_the_last_two_bytes() {
    const BLOCK_BYTES: usize = 210;
    const D_OFFSET: usize = 208;

    let weights = synthetic_weights();
    let quantized = Quantizer::new(QuantizationConfig::q6_k())
        .quantize_tensor(&to_le_bytes(&weights), &[N], UnifiedDType::F32)
        .expect("quantization must succeed");

    for (b, block) in quantized
        .packed_data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = f16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]).to_f32();
        assert!(
            d.is_finite() && d != 0.0,
            "block {b}: bytes 208..210 must be a finite non-zero f16 scale, got {d}"
        );

        let peak = weights[b * SUPER_BLOCK..(b + 1) * SUPER_BLOCK]
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let reach = d.abs() * 4096.0;
        assert!(
            reach > peak / 10.0 && reach < peak * 10.0,
            "block {b}: scale {d} reaches {reach}, nowhere near the block peak {peak} \
             — bytes 208..210 are probably not the scale"
        );
        // The scales array sits just before it and must not be all zero for
        // non-zero input — that would mean the sub-block scales were misplaced.
        assert!(
            block[192..208].iter().any(|&s| s != 0),
            "block {b}: sub-block scales at 192..208 are all zero"
        );
    }
}
