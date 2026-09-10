//! Whether one tensor's VALUES can be written at an encoding, asked of every
//! encoding an allocation might reach.
//!
//! [`super::plan::settle`] quantizes a tensor once at the base encoding, and
//! that answers the base. Every other rung has to be asked separately, and
//! before the allocation is settled: the bump set and the ladder are both
//! fixed before the first payload is written, so a quantizer error during the
//! payload pass would abort the run over a truncated file.
//!
//! The question is representability, so it is answered by
//! [`super::parallel::accepts_tensor`] and never by a second quantize.

use anyhow::{anyhow, Context, Result};

use tcf_core::NativeEncoding;

use super::encoding::{TcfOptions, TensorShape};
use super::parallel::accepts_tensor;
use super::plan::Registration;
use crate::policy::{plan_tcf_tensor, TcfPlan};

/// Whether the profile's bump encoding can represent these values.
///
/// Returns `false` when no profile names a bump encoding, and when the bump
/// encoding refuses the values a base-encoded quantize accepted. That
/// guarantees the payload pass never meets an encoding its record already
/// fixed.
///
/// This is a representability question, so it is answered by
/// [`accepts_tensor`] and never by a quantize. `accepts_tensor` reads each
/// group once for its extremes and decides Section 13.1's two value-dependent
/// rejections from those; a quantize would have searched 19 or 22 scale
/// candidates per group, refined a sub-scale, built every tile, and then
/// dropped all of it to keep one bool. The verdict is the same one, never an
/// approximation of it: a group the extremes cannot settle runs the full fit.
///
/// The bump payload is not produced here, and it was not kept when it was. A
/// profile bumps a small share of its candidates, so spilling every
/// candidate's bump payload would write a second model's worth of bytes to
/// save requantizing that share. The base payload is the opposite trade:
/// every tensor the allocation leaves alone reads its own back.
///
/// # Errors
/// The bump layout was refused for a reason that is not value-dependent.
pub(super) fn bump_accepts(
    options: TcfOptions,
    values: &[f32],
    shape: &TensorShape,
    name: &str,
) -> Result<bool> {
    let Some(bump) = options.profile.and_then(|profile| profile.bump) else {
        return Ok(false);
    };
    let accepted = accepts_tensor(values, shape.extent(), shape.rank(), bump.layout())
        .map_err(|e| anyhow!(e))
        .with_context(|| format!("Failed to screen tensor '{name}' at {bump:?}"))?;
    if !accepted {
        log::warn!(
            "Tensor '{name}' cannot be represented at {bump:?}: it stays at the base \
             encoding and is no bump candidate"
        );
    }
    Ok(accepted)
}

/// Which of a ladder's encodings this tensor can be written at.
///
/// A ladder allocation may put a tensor at any encoding the measurement scored
/// it at, and two questions decide whether it can go there. The SHAPE question
/// is [`plan_tcf_tensor`]: a row that is not a whole number of that encoding's
/// tiles holds none of them. The VALUE question is [`accepts_tensor`]: a wider
/// encoding divides a group by a larger code range, so a tensor of very small
/// magnitudes can pass at one encoding and underflow binary16 at another.
///
/// Both are asked here, before any record is fixed, for the reason
/// [`super::plan::PlannedTensor::bumpable`] gives: the allocation is settled before the
/// first payload is written, and a quantizer error during the payload pass
/// would abort the run over a truncated file.
///
/// The base encoding is left out — the tensor was just quantized at it — and
/// so is any encoding whose layout is refused for a reason that is not
/// value-dependent, which is an error rather than a verdict.
///
/// # Errors
/// A ladder layout was refused for a reason that is not value-dependent.
pub(super) fn ladder_accepts(
    options: TcfOptions,
    ladder: &[NativeEncoding],
    values: &[f32],
    shape: &TensorShape,
    registration: &Registration,
) -> Result<Vec<NativeEncoding>> {
    let name = registration.name.as_str();
    let mut accepted = Vec::new();
    for encoding in ladder {
        if *encoding == options.encoding {
            continue;
        }
        let layout = encoding.layout();
        if plan_tcf_tensor(
            &registration.shape,
            registration.dtype,
            layout.geometry.tile,
        ) != TcfPlan::Quantize
        {
            continue;
        }
        let takes = accepts_tensor(values, shape.extent(), shape.rank(), layout)
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to screen tensor '{name}' at {encoding:?}"))?;
        if takes {
            accepted.push(*encoding);
        }
    }
    Ok(accepted)
}
