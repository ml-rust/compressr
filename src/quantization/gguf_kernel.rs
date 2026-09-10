//! The one call into boostr's GGUF block writers.
//!
//! Every GGUF block format compressr writes is packed here and nowhere else,
//! by `boostr::QuantizeOps`. A second writer in compressr can agree with
//! compressr's own reader and still disagree with every real reader — that is
//! exactly how Q6_K once shipped with a wrong field order.
//!
//! # The two entry points
//!
//! boostr carries two writers per K-quant, as `ggml-quants.c` does: the plain
//! one, and the importance-weighted one taken the moment a matrix is
//! supplied. [`quantize_gguf`] calls the plain one when `importance` is
//! `None` rather than calling the weighted one with no vector, so a run that
//! named no matrix cannot diverge from every earlier run even in principle.

use anyhow::{anyhow, bail, Result};

use super::tensor::QuantizedTensor;
use super::types::QuantType;

use boostr::quant::{QuantFormat, QuantizeOps};
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};

/// Quantize into a GGUF block format with boostr's writer kernels.
///
/// boostr packs blocks along the LAST axis of `shape`, so a
/// `[out_features, in_features]` weight blocks along `in_features` — the same
/// axis GGUF calls `ne[0]`. It requires that axis to be a whole number of
/// blocks, which [`crate::policy::plan_tensor`] already checks before a tensor
/// gets here; its error is surfaced rather than swallowed, because the two
/// rules drifting apart is precisely the failure worth seeing.
///
/// The returned blocks carry their own scales inside them, so
/// [`QuantizedTensor::scales`] stays empty for every format written this way.
///
/// `importance` is the OPTIONAL per-column importance, one entry per element
/// of the last axis. `None` takes the unweighted writer.
///
/// # Errors
/// The shape does not describe `data`, boostr cannot stage the tensor, or
/// boostr's writer refuses the format, the block alignment or the importance
/// vector's length.
pub(super) fn quantize_gguf(
    data: &[f32],
    shape: &[usize],
    format: QuantFormat,
    quant_type: QuantType,
    importance: Option<&[f32]>,
) -> Result<QuantizedTensor> {
    let numel: usize = shape.iter().product();
    if shape.is_empty() || numel != data.len() {
        bail!(
            "shape {:?} describes {} elements but the tensor has {}",
            shape,
            numel,
            data.len()
        );
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(data, shape, &device).map_err(|e| {
        anyhow!(
            "failed to stage tensor for {} quantization: {e}",
            format.name()
        )
    })?;

    // The unweighted path calls `quantize` ITSELF, never
    // `quantize_with_importance(.., None)`: a flagless conversion has to take
    // the entry point it has always taken.
    let quantized = match importance {
        None => client.quantize(&input, format),
        Some(importance) => client.quantize_with_importance(&input, format, Some(importance)),
    }
    .map_err(|e| anyhow!("boostr {} quantization failed: {e}", format.name()))?;

    let packed_data = quantized
        .to_bytes()
        .map_err(|e| anyhow!("failed to read back {} blocks: {e}", format.name()))?;

    Ok(QuantizedTensor {
        packed_data,
        scales: Vec::new(),
        original_shape: shape.to_vec(),
        original_numel: numel,
        quant_type,
        block_size: format.block_size(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row of 256 values, which is exactly one K-quant super-block.
    fn row() -> Vec<f32> {
        (0..256)
            .map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32))
            .collect()
    }

    /// The no-importance path is boostr's plain `quantize`, byte for byte.
    ///
    /// This is the flagless conversion every earlier run performed, so its
    /// bytes are compared against the unweighted entry point directly rather
    /// than against a recorded constant.
    #[test]
    fn the_unweighted_path_calls_boostrs_plain_quantize() {
        let data = row();
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let input =
            Tensor::<CpuRuntime>::from_slice(&data, &[1, 256], &device).expect("stages a tensor");
        let expected = client
            .quantize(&input, QuantFormat::Q4K)
            .expect("boostr quantizes")
            .to_bytes()
            .expect("reads back");

        let produced = quantize_gguf(&data, &[1, 256], QuantFormat::Q4K, QuantType::Q4K, None)
            .expect("compressr quantizes");
        assert_eq!(produced.packed_data, expected);
        assert!(produced.scales.is_empty(), "K-quant blocks carry their own");
    }

    /// A supplied importance reaches the weighted writer, which is a
    /// different writer: it picks different values for the same input.
    #[test]
    fn an_importance_vector_reaches_the_weighted_writer() {
        let data = row();
        let plain = quantize_gguf(&data, &[1, 256], QuantFormat::Q4K, QuantType::Q4K, None)
            .expect("unweighted quantizes");
        let importance: Vec<f32> = (0..256).map(|i| 1.0 + (i % 11) as f32).collect();
        let weighted = quantize_gguf(
            &data,
            &[1, 256],
            QuantFormat::Q4K,
            QuantType::Q4K,
            Some(&importance),
        )
        .expect("weighted quantizes");
        assert_eq!(
            plain.packed_data.len(),
            weighted.packed_data.len(),
            "an importance changes no bit layout, so the payload is the same size"
        );
        assert_ne!(
            plain.packed_data, weighted.packed_data,
            "a non-uniform importance selects different values"
        );
    }

    /// A vector that is not exactly as long as the last axis is refused by
    /// boostr rather than quietly ignored.
    #[test]
    fn a_wrong_length_importance_vector_is_refused() {
        let data = row();
        let importance = vec![1.0f32; 128];
        let err = quantize_gguf(
            &data,
            &[1, 256],
            QuantFormat::Q4K,
            QuantType::Q4K,
            Some(&importance),
        )
        .expect_err("a mis-sized vector is an error");
        assert!(err.to_string().contains("Q4_K"), "{err}");
    }
}
