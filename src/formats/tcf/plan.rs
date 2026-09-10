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
//! [`super::allocate`] by default. It is a weight error, never a task delta,
//! and no sensitivity field is ever written from it: a run without
//! `--sensitivity` leaves `sensitivity_delta`, `sensitivity_ci95`,
//! `sensitivity_samples` and `calibration_id` unset, and its file is NOT
//! profile-solved per `PROFILES.md` Section 4.
//!
//! `--sensitivity` supplies the measurement compressr cannot take itself: a
//! task-metric delta per tensor, measured elsewhere at the profile's base
//! encoding. [`super::allocate`] then ranks by that instead, and the writer
//! records `sensitivity_delta`, `sensitivity_samples` and `calibration_id`
//! from it. `sensitivity_ci95` stays unset either way — the measurement
//! carries no confidence interval. See [`crate::policy::sensitivity`].
//!
//! # The screen is hot
//!
//! [`settle`] quantizes the whole tensor, so the screen is most of a
//! conversion's arithmetic, not a side cost. It goes through
//! [`quantize_tensor`], which splits the tensor's tiles over rayon's pool.
//! Parallelizing the payload pass alone would have left the rest of the work
//! single-threaded.
//!
//! Under a profile [`bump_accepts`] asks a second question of the same
//! values — whether the bump encoding can represent them — and that question
//! is answered by [`accepts_tensor`], not by a second quantize. A quantize
//! would have run a 19- or 22-candidate scale search plus a sub-scale
//! refinement over every group and kept one bool.
//!
//! The screen's own tiles are no longer thrown away: [`settle`] hands them
//! back, the writer spills them, and pass 2 reads them instead of quantizing
//! the same values again. See [`super::spill`] for why that keeps one tensor
//! in flight.

use anyhow::{anyhow, Context, Result};

use tcf_core::{encoding::dequantize, Encoding, LogicalTile, TcfError};

use tcf_core::NativeEncoding;

use super::encoding::{raw_encoding_for, values_f32, TcfOptions, TensorShape};
use super::parallel::quantize_tensor;
use super::screen::{bump_accepts, ladder_accepts};
use crate::loader::UnifiedDType;
use crate::policy::{
    plan_tcf_tensor, relative_rms_error, tcf_fallback_reason, FallbackReason, TcfPlan,
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
    /// Encodings this tensor's VALUES can be written at
    ///
    /// The base encoding is not in it: a quantized tensor took the base by
    /// definition. Every other entry was asked of the same values by
    /// [`accepts_tensor`], because a ladder allocation may put this tensor at
    /// any of them and the payload pass must never meet an encoding its record
    /// already fixed.
    pub accepts: Vec<NativeEncoding>,
    /// True when an allocation raised this tensor above the profile's base
    pub bumped: bool,
    /// True when a ladder allocation put this tensor BELOW the profile's base
    ///
    /// Only a measurement that scored the cheaper encoding on this tensor can
    /// set it. A two-way allocation never demotes: its base is a floor.
    pub demoted: bool,
    /// The tensor's measured task-metric delta, when `--sensitivity` gave one
    ///
    /// `None` for every run without the flag, and for a tensor the measurement
    /// does not name. This is `PROFILES.md` Section 4 step 2's number and the
    /// only thing `sensitivity_delta` is ever written from; the weight-error
    /// screen never reaches that field.
    pub sensitivity_delta: Option<f64>,
}

/// What the screen learned about one quantizable tensor.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Screen {
    /// Relative RMS reconstruction error at the base encoding
    pub(super) relative_rms: f64,
    /// True when the profile's bump encoding also accepts these values
    pub(super) bumpable: bool,
    /// Every other encoding the ladder may reach that accepts these values
    pub(super) accepts: Vec<NativeEncoding>,
}

/// One tensor's screening verdict, and the payload that verdict came from.
#[derive(Debug)]
pub(super) struct Settled {
    /// What the tensor is stored as
    pub(super) plan: TcfPlan,
    /// How the base encoding reconstructs it, `None` for a raw tensor
    pub(super) screen: Option<Screen>,
    /// The tiles the screen quantized, at the BASE encoding
    ///
    /// `None` for a raw tensor. The writer spills these and pass 2 reads
    /// them back rather than quantizing the same values again; a tensor a
    /// profile later bumps needs a different encoding, so its spilled payload
    /// is refused and requantized.
    pub(super) tiles: Option<Vec<LogicalTile>>,
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
///
/// Those same tiles come back in [`Settled::tiles`]. They are the payload of
/// every tensor the allocation leaves at the base encoding, which is most of
/// them, so recomputing them in pass 2 was the single largest avoidable cost
/// in a conversion.
pub(super) fn settle(
    options: TcfOptions,
    registration: &Registration,
    data: &[u8],
    ladder: &[NativeEncoding],
) -> Result<Settled> {
    let layout = options.encoding.layout();
    let plan = plan_tcf_tensor(
        &registration.shape,
        registration.dtype,
        layout.geometry.tile,
    );
    if plan != TcfPlan::Quantize {
        return Ok(Settled {
            plan,
            screen: None,
            tiles: None,
        });
    }

    let name = registration.name.as_str();
    let values = values_f32(data, registration.dtype, name)?;
    let shape = TensorShape::of(&registration.shape, name)?;
    let raw = |reason| Settled {
        plan: TcfPlan::Raw(reason),
        screen: None,
        tiles: None,
    };
    match quantize_tensor(&values, shape.extent(), shape.rank(), layout) {
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
            let bumpable = bump_accepts(options, &values, &shape, name)?;
            let accepts = ladder_accepts(options, ladder, &values, &shape, registration)?;
            Ok(Settled {
                plan: TcfPlan::Quantize,
                screen: Some(Screen {
                    relative_rms,
                    bumpable,
                    accepts,
                }),
                tiles: Some(tiles),
            })
        }
        Err(TcfError::SourceNonfinite { .. }) => {
            log::warn!("Tensor '{name}' holds a NaN or an infinity: storing it unquantized");
            Ok(raw(FallbackReason::SourceNonfinite))
        }
        Err(TcfError::NumericRange { .. }) => {
            log::warn!(
                "Tensor '{name}' has a group whose scale is not representable at {:?}: \
                 storing it unquantized",
                options.encoding
            );
            Ok(raw(FallbackReason::NumericRange))
        }
        Err(other) => {
            Err(anyhow!(other)).with_context(|| format!("Failed to quantize tensor '{name}'"))
        }
    }
}

/// Keep one settled tensor's plan, at the base encoding.
///
/// Nothing is registered here. A profile's bump set ranks tensors against
/// each other, so the encoding a record carries is only final once every
/// tensor has been screened. [`super::allocate`] settles that, and the writer
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

    // A raw tensor was never screened, so it has no error, no bump verdict
    // and no ladder: it is a candidate for nothing.
    let (screen_error, bumpable, accepts) = match screen {
        Some(screen) => (Some(screen.relative_rms), screen.bumpable, screen.accepts),
        None => (None, false, Vec::new()),
    };

    Ok(PlannedTensor {
        name: registration.name.clone(),
        dtype: registration.dtype,
        fallback,
        encoding,
        screen_error,
        elements,
        bumpable,
        accepts,
        bumped: false,
        demoted: false,
        sensitivity_delta: None,
    })
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
    if plan.bumped || plan.demoted {
        // A demotion is a task-sensitivity decision exactly as a promotion is:
        // the measurement said this tensor takes no damage at the cheaper
        // encoding, and the allocation spent the difference elsewhere.
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
    use super::super::profile::TCF_COMPACT;
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
            accepts: Vec::new(),
            bumped,
            demoted: false,
            sensitivity_delta: None,
        }
    }

    /// A bumped tensor differs from the module preference because the
    /// allocation raised it, which Section 8.6 wants a reason for.
    #[test]
    fn a_bumped_tensor_records_task_sensitivity() {
        let options = TcfOptions::from_profile(TCF_COMPACT);
        let bumped = planned(Encoding::Native(NativeEncoding::Q6S16DT64), true, None);
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
        let base = planned(Encoding::Native(NativeEncoding::Q4AS32DT64), false, None);
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
}
