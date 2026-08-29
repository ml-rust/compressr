//! What one TCF tensor is planned as, and which tensors a profile bumps.
//!
//! This module decides encodings. It reads no file and writes none: the writer
//! hands it a tensor's registration and its bytes, and it answers with a
//! [`PlannedTensor`]. Packing, tiling and dequantization all come from
//! `tcf-core`.
//!
//! # The screen, and what it is not
//!
//! [`settle`] quantizes a tensor once to see whether the values themselves
//! refuse the encoding, and the same call answers how well the encoding
//! reconstructs them. That relative RMS error ranks bump candidates in
//! [`allocate`]. It is a weight error. `PROFILES.md` Section 4 step 2 measures
//! a task-metric delta on a calibration set, which compressr cannot run, so
//! `sensitivity_delta`, `sensitivity_ci95`, `sensitivity_samples` and
//! `calibration_id` are never written from anything here.
//!
//! # The screen is hot
//!
//! [`settle`] quantizes the whole tensor, and under a profile
//! [`bump_accepts`] quantizes it a second time at the bump encoding. With the
//! writer's payload pass that is two or three FULL quantizes per tensor, so
//! the screen is most of a conversion's arithmetic, not a side cost. Every one
//! of them goes through [`quantize_tensor`], which splits the tensor's tiles
//! over rayon's pool. Parallelizing the payload pass alone would have left two
//! thirds of the work single-threaded.

use anyhow::{anyhow, Context, Result};

use tcf_core::{encoding::dequantize, Encoding, TcfError};

use super::encoding::{dims_of, rank_of, raw_encoding_for, values_f32, TcfOptions};
use super::parallel::quantize_tensor;
use crate::loader::UnifiedDType;
use crate::policy::{
    plan_tcf_tensor, relative_rms_error, select_bumped, tcf_fallback_reason, FallbackReason,
    ScreenedTensor, TcfPlan,
};

/// One tensor as registered, before its values decide anything.
#[derive(Debug, Clone)]
pub(super) struct Registration {
    /// Tensor name, as the source checkpoint spells it
    pub(super) name: String,
    /// Standard-order shape, whose LAST dimension is the row length
    pub(super) shape: Vec<usize>,
    /// Source element type
    pub(super) dtype: UnifiedDType,
}

/// One tensor's settled plan: what its record says, and how pass 2 produces
/// its payload.
#[derive(Debug, Clone)]
pub struct PlannedTensor {
    /// Tensor name, as the source checkpoint spells it
    pub name: String,
    /// Source element type, which the raw path stores verbatim
    pub dtype: UnifiedDType,
    /// `None` for a quantized tensor, otherwise why it is stored raw
    pub fallback: Option<FallbackReason>,
    /// The encoding written to the record
    pub encoding: Encoding,
    /// Relative RMS reconstruction error at the base encoding
    ///
    /// `None` for a tensor stored raw, which has no base encoding to be
    /// measured against and is therefore no bump candidate. This is a
    /// weight-error screen, never a task-metric delta: no TCF sensitivity
    /// field is written from it.
    pub screen_error: Option<f64>,
    /// Elements in the tensor, which is what a profile's bump budget spends
    pub elements: u64,
    /// True when the profile's bump encoding also accepts these values
    ///
    /// A wider encoding divides a group by a larger code range, so its scale
    /// is smaller and a tensor of very small magnitudes can pass at the base
    /// and underflow binary16 at the bump. Such a tensor is no candidate: the
    /// bump set is fixed before any payload is written, and a quantizer error
    /// during the payload pass would abort the run over a truncated file.
    pub bumpable: bool,
    /// True when a profile allocation raised this tensor to its bump encoding
    pub bumped: bool,
}

/// What the screen learned about one quantizable tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Screen {
    /// Relative RMS reconstruction error at the base encoding
    pub(super) relative_rms: f64,
    /// True when the profile's bump encoding also accepts these values
    pub(super) bumpable: bool,
}

/// Settle one tensor's plan from its shape and, when the shape allows
/// quantization, from its values. A quantized tensor is screened at the
/// same time.
///
/// The quantizer is the only thing that can see a NaN or a scale that
/// underflows binary16, so those two fallbacks are decided here, in the
/// conversion loop, and never abort the run.
///
/// The screen rides along on that same quantize: the tiles it produced are
/// dequantized back to values and compared against the source, which
/// answers how badly the base encoding reconstructs this tensor. The
/// returned [`Screen`] holds the relative RMS of that error, and is `None`
/// for a tensor stored raw. It ranks bump candidates and nothing else — it is
/// a weight error, not a task-metric delta, so no sensitivity field is ever
/// written from it.
pub(super) fn settle(
    options: TcfOptions,
    registration: &Registration,
    data: &[u8],
) -> Result<(TcfPlan, Option<Screen>)> {
    let layout = options.encoding.layout();
    let plan = plan_tcf_tensor(
        &registration.shape,
        registration.dtype,
        layout.geometry.tile,
    );
    if plan != TcfPlan::Quantize {
        return Ok((plan, None));
    }

    let name = registration.name.as_str();
    let values = values_f32(data, registration.dtype, name)?;
    let dims = dims_of(&registration.shape, name)?;
    let rank = rank_of(&registration.shape, name)?;
    // The tiles are dropped at the end of this match: pass 2 recomputes
    // them one tensor at a time, so keeping them would break the
    // one-tensor-in-flight bound. Only the screen scalar survives.
    // `quantize` wants EXACTLY `rank` dims; the padded `[u64; 8]` is the
    // record's wire shape, not the codec's. Passing all eight is
    // `E_INVALID_RANK`.
    let extent = usize::try_from(rank).unwrap_or(dims.len());
    match quantize_tensor(&values, &dims[..extent], rank, layout) {
        Ok(tiles) => {
            // A row that holds a whole number of tiles covers the element
            // space in order, so the reconstruction lines up with `values`
            // one for one. `plan_tcf_tensor` already rejected every other
            // row length.
            let reconstructed = dequantize(&tiles, layout)
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Failed to screen tensor '{name}'"))?;
            let relative_rms = relative_rms_error(&values, &reconstructed)
                .with_context(|| format!("Failed to screen tensor '{name}'"))?;
            let bumpable = bump_accepts(options, &values, &dims[..extent], rank, name)?;
            Ok((
                TcfPlan::Quantize,
                Some(Screen {
                    relative_rms,
                    bumpable,
                }),
            ))
        }
        Err(TcfError::SourceNonfinite { .. }) => {
            log::warn!("Tensor '{name}' holds a NaN or an infinity: storing it unquantized");
            Ok((TcfPlan::Raw(FallbackReason::SourceNonfinite), None))
        }
        Err(TcfError::NumericRange { .. }) => {
            log::warn!(
                "Tensor '{name}' has a group whose scale is not representable at {:?}: \
                 storing it unquantized",
                options.encoding
            );
            Ok((TcfPlan::Raw(FallbackReason::NumericRange), None))
        }
        Err(other) => {
            Err(anyhow!(other)).with_context(|| format!("Failed to quantize tensor '{name}'"))
        }
    }
}

/// Whether the profile's bump encoding can represent these values.
///
/// Returns `false` when no profile names a bump encoding, and when the bump
/// encoding refuses the values a base-encoded quantize accepted. The check
/// costs one extra quantize per candidate and buys the guarantee that the
/// payload pass never meets an encoding its record already fixed.
///
/// # Errors
/// The quantizer failed for a reason that is not value-dependent.
fn bump_accepts(
    options: TcfOptions,
    values: &[f32],
    dims: &[u64],
    rank: u32,
    name: &str,
) -> Result<bool> {
    let Some(bump) = options.profile.and_then(|profile| profile.bump) else {
        return Ok(false);
    };
    match quantize_tensor(values, dims, rank, bump.layout()) {
        Ok(_) => Ok(true),
        Err(TcfError::SourceNonfinite { .. }) | Err(TcfError::NumericRange { .. }) => {
            log::warn!(
                "Tensor '{name}' cannot be represented at {bump:?}: it stays at the base \
                 encoding and is no bump candidate"
            );
            Ok(false)
        }
        Err(other) => Err(anyhow!(other))
            .with_context(|| format!("Failed to screen tensor '{name}' at {bump:?}")),
    }
}

/// Keep one settled tensor's plan, at the base encoding.
///
/// Nothing is registered here. A profile's bump set ranks tensors against
/// each other, so the encoding a record carries is only final once every
/// tensor has been screened. [`allocate`] settles that, and the writer
/// registers the records afterwards.
pub(super) fn plan_one(
    options: TcfOptions,
    registration: &Registration,
    plan: TcfPlan,
    screen: Option<Screen>,
) -> Result<PlannedTensor> {
    let name = registration.name.as_str();

    let mut elements: u64 = 1;
    for dim in &registration.shape {
        let dim = u64::try_from(*dim)
            .with_context(|| format!("Tensor '{name}' has a dimension larger than u64"))?;
        elements = elements
            .checked_mul(dim)
            .with_context(|| format!("Tensor '{name}' holds more elements than u64 counts"))?;
    }

    let (encoding, fallback) = match plan {
        TcfPlan::Quantize => (Encoding::Native(options.encoding), None),
        TcfPlan::Raw(reason) => (
            Encoding::Raw(raw_encoding_for(registration.dtype, name)?),
            Some(reason),
        ),
    };

    Ok(PlannedTensor {
        name: registration.name.clone(),
        dtype: registration.dtype,
        fallback,
        encoding,
        screen_error: screen.map(|screen| screen.relative_rms),
        elements,
        bumpable: screen.is_some_and(|screen| screen.bumpable),
        bumped: false,
    })
}

/// Raise the profile's bump set from the base encoding.
///
/// Candidates are the quantized tensors the bump encoding also accepts. A
/// tensor stored raw has no base encoding to be bumped from, and one the bump
/// encoding refuses cannot carry it, so neither is ranked and neither counts
/// toward the element total the profile's fraction is taken of. The budget
/// is spent in elements, not in tensors: the ladder's rungs are bpw
/// numbers, and only elements track bytes.
///
/// The ranking is the weight-error screen. `PROFILES.md` Section 4 step 2
/// — the task-metric measurement that separates a solved profile from a
/// heuristic — needs a model and a calibration set, which compressr does
/// not have, so it never runs and every sensitivity field stays unset.
pub(super) fn allocate(planned: &mut [PlannedTensor], options: TcfOptions) -> Result<()> {
    let Some(profile) = options.profile else {
        return Ok(());
    };
    let Some(bump) = profile.bump else {
        log::info!(
            "TCF profile {}: {:?} for every quantizable tensor, no bump set",
            profile.name,
            profile.base
        );
        return Ok(());
    };

    let candidates: Vec<ScreenedTensor> = planned
        .iter()
        .enumerate()
        .filter_map(|(index, plan)| {
            plan.screen_error
                .filter(|_| plan.bumpable)
                .map(|relative_rms| ScreenedTensor {
                    index,
                    relative_rms,
                    elements: plan.elements,
                })
        })
        .collect();
    let candidate_count = candidates.len();
    let candidate_elements: u64 = candidates.iter().map(|c| c.elements).sum();
    let bumped = select_bumped(&candidates, profile.bump_fraction);

    let mut bumped_elements: u64 = 0;
    for index in &bumped {
        let plan = planned
            .get_mut(*index)
            .ok_or_else(|| anyhow!("Allocation named tensor {index}, which was never planned"))?;
        plan.encoding = Encoding::Native(bump);
        plan.bumped = true;
        bumped_elements = bumped_elements.saturating_add(plan.elements);
    }

    let share = if candidate_elements == 0 {
        0.0
    } else {
        bumped_elements as f64 * 100.0 / candidate_elements as f64
    };
    log::info!(
        "TCF profile {}: base {:?}, {} of {candidate_count} quantizable tensors bumped to \
         {bump:?}, {share:.1}% of quantizable parameters against a {:.0}% budget",
        profile.name,
        profile.base,
        bumped.len(),
        profile.bump_fraction * 100.0
    );
    log::info!(
        "TCF profile {}: the bump set was ranked by a weight-error screen — relative RMS of \
         the dequantized weights against the source. No task-metric calibration ran, so \
         sensitivity_delta, sensitivity_ci95, sensitivity_samples and calibration_id stay \
         unset and this file is NOT profile-solved per PROFILES.md Section 4.",
        profile.name
    );
    Ok(())
}

/// The `fallback_reason` one settled tensor carries.
///
/// SPECIFICATION.md Section 8.6 makes the field mandatory whenever the
/// encoding differs from the module's highest-ranked preference. Three
/// cases reach that:
///
/// - A raw tensor carries the reason its plan gave.
/// - A tensor a profile bumped carries `TaskSensitivity`: the allocation
///   raised it above the module preference on purpose.
/// - A `--tcf-encoding` run whose two flags disagree carries
///   `ProducerPolicy`, which is what the flags say and nothing more.
pub(super) fn reason_for(plan: &PlannedTensor, options: TcfOptions) -> tcf_core::FallbackReason {
    if let Some(reason) = plan.fallback {
        return tcf_fallback_reason(reason);
    }
    if plan.bumped {
        return tcf_core::FallbackReason::TaskSensitivity;
    }
    if plan.encoding == Encoding::Native(options.preferred_encoding) {
        tcf_core::FallbackReason::None
    } else {
        tcf_core::FallbackReason::ProducerPolicy
    }
}

#[cfg(test)]
mod tests {
    use super::super::profile::{TCF_COMPACT, TCF_REF};
    use super::*;
    use tcf_core::{NativeEncoding, RawEncoding};

    fn planned(
        encoding: Encoding,
        bumped: bool,
        fallback: Option<FallbackReason>,
    ) -> PlannedTensor {
        PlannedTensor {
            name: "w".to_string(),
            dtype: UnifiedDType::F32,
            fallback,
            encoding,
            screen_error: fallback.is_none().then_some(0.1),
            elements: 64,
            bumpable: fallback.is_none(),
            bumped,
        }
    }

    /// A bumped tensor differs from the module preference because the
    /// allocation raised it, which Section 8.6 wants a reason for.
    #[test]
    fn a_bumped_tensor_records_task_sensitivity() {
        let options = TcfOptions::from_profile(TCF_COMPACT);
        let bumped = planned(Encoding::Native(NativeEncoding::Q6S32T64), true, None);
        assert_eq!(
            reason_for(&bumped, options),
            tcf_core::FallbackReason::TaskSensitivity
        );
    }

    /// A tensor left at the base encoding IS the module preference, so it
    /// records no reason.
    #[test]
    fn a_base_encoded_tensor_records_no_reason() {
        let options = TcfOptions::from_profile(TCF_COMPACT);
        let base = planned(Encoding::Native(NativeEncoding::Q4AS64T64), false, None);
        assert_eq!(reason_for(&base, options), tcf_core::FallbackReason::None);
    }

    /// Two `--tcf-encoding` flags that disagree are a producer's choice, and
    /// nothing measured stands behind them.
    #[test]
    fn a_disagreeing_encoding_pair_records_producer_policy() {
        let options = TcfOptions::uniform(NativeEncoding::Q4S32T64, NativeEncoding::Q6S32T64);
        let tensor = planned(Encoding::Native(NativeEncoding::Q4S32T64), false, None);
        assert_eq!(
            reason_for(&tensor, options),
            tcf_core::FallbackReason::ProducerPolicy
        );
    }

    #[test]
    fn a_raw_tensor_records_its_own_reason() {
        let options = TcfOptions::from_profile(TCF_COMPACT);
        let raw = planned(
            Encoding::Raw(RawEncoding::F32),
            false,
            Some(FallbackReason::Rank),
        );
        assert_eq!(reason_for(&raw, options), tcf_core::FallbackReason::RankLt2);
    }

    /// The budget counts the quantizable elements alone, and the raw tensor
    /// here is larger than both candidates together.
    #[test]
    fn allocation_excludes_a_raw_tensor_from_the_ranking_and_the_budget() {
        let mut tensors = vec![
            planned(Encoding::Native(NativeEncoding::Q4AS64T64), false, None),
            planned(Encoding::Native(NativeEncoding::Q4AS64T64), false, None),
            planned(
                Encoding::Raw(RawEncoding::F32),
                false,
                Some(FallbackReason::Rank),
            ),
        ];
        tensors[0].screen_error = Some(0.4);
        tensors[1].screen_error = Some(0.9);
        tensors[2].elements = 4096;

        allocate(&mut tensors, TcfOptions::from_profile(TCF_COMPACT)).expect("allocates");

        assert!(!tensors[0].bumped);
        assert!(tensors[1].bumped, "the worst-reconstructed candidate");
        assert_eq!(
            tensors[1].encoding,
            Encoding::Native(NativeEncoding::Q6S32T64)
        );
        assert!(!tensors[2].bumped, "a raw tensor is no candidate");
        assert_eq!(tensors[2].encoding, Encoding::Raw(RawEncoding::F32));
    }

    /// A tensor the bump encoding cannot represent stays at the base, and is
    /// no more a candidate than a raw one is.
    #[test]
    fn a_tensor_the_bump_encoding_refuses_is_no_candidate() {
        let base = Encoding::Native(NativeEncoding::Q4AS64T64);
        let mut tensors = vec![planned(base, false, None), planned(base, false, None)];
        tensors[0].screen_error = Some(0.9);
        tensors[0].bumpable = false;
        tensors[1].screen_error = Some(0.1);

        allocate(&mut tensors, TcfOptions::from_profile(TCF_COMPACT)).expect("allocates");

        assert!(!tensors[0].bumped, "the bump encoding refuses these values");
        assert!(tensors[1].bumped);
    }

    /// `TCF-REF` names no bump encoding, so every tensor keeps the base.
    #[test]
    fn a_flat_profile_bumps_nothing() {
        let mut tensors = vec![planned(
            Encoding::Native(NativeEncoding::Q8S32T64),
            false,
            None,
        )];
        allocate(&mut tensors, TcfOptions::from_profile(TCF_REF)).expect("allocates");
        assert!(!tensors[0].bumped);
    }

    /// A `--tcf-encoding` run has no profile, so nothing is bumped and the
    /// encoding the flag named stands.
    #[test]
    fn a_run_without_a_profile_bumps_nothing() {
        let options = TcfOptions::uniform(NativeEncoding::Q4S32T64, NativeEncoding::Q4S32T64);
        let mut tensors = vec![planned(
            Encoding::Native(NativeEncoding::Q4S32T64),
            false,
            None,
        )];
        allocate(&mut tensors, options).expect("allocates");
        assert!(!tensors[0].bumped);
    }
}
