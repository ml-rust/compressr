//! The VoxCPM2 companion: which output tensors the embedded AudioVAE owns,
//! why they stay dense, and the `config.json` metadata written beside them.
//!
//! A VoxCPM2 checkpoint directory holds its `AudioVAE` codec beside the
//! transformer stack as `audiovae.pth`. A conversion folds that companion
//! into the same GGUF/TCF under [`AUDIO_VAE_PREFIX`], so one file loads the
//! whole model. The codec is never quantized: boostr's decoder is verified
//! against F32 fixtures and its loader refuses a block-quantized `vae.*`
//! tensor outright, so a converter that quantized one would write a file
//! nothing loads, and boostr's encoder always runs at F32. Every float
//! tensor under the prefix is therefore planned `StoreF32(AudioVae)` — in a
//! flat run, a mix, and a plain dtype cast alike, so a `--dtype bf16`
//! conversion never rounds the codec either; a non-float one (the
//! `sr_bin_boundaries` I32 vector) passes through as it does anywhere.
//!
//! The rule is a name test layered over [`plan_tensor`] and
//! [`plan_mixed_tensor`] rather than a change to them: those two are shape
//! and dtype rules with many callers that carry no name.

use super::{checkpoint_dir, plan_mixed_tensor, plan_tensor, FallbackReason, TensorPlan};
use crate::loader::UnifiedDType;
use crate::quantization::QuantType;
use anyhow::{Context, Result};
use std::path::Path;

/// Prefix every tensor of an embedded AudioVAE carries in the output.
///
/// The companion's own keys (`decoder.*`, `encoder.*`) sit under it
/// verbatim, so `decoder.model.0.weight` is written as
/// `vae.decoder.model.0.weight` — the spelling boostr's VoxCPM2 loader
/// probes for.
pub const AUDIO_VAE_PREFIX: &str = "vae.";

/// The `general.architecture` value that names a VoxCPM2 checkpoint, in
/// the spelling its own `config.json` uses.
const VOXCPM2_ARCHITECTURE: &str = "voxcpm2";

/// GGUF metadata key holding a VoxCPM2 checkpoint's `config.json`, byte for
/// byte. The same string boostr's `gguf_loader` reads, so a GGUF carrying
/// it loads with no `config.json` path beside it.
pub const VOXCPM2_CONFIG_JSON_KEY: &str = "voxcpm2.config_json";

/// Is `arch` VoxCPM2, whatever its case?
pub fn is_voxcpm2(arch: &str) -> bool {
    arch.trim().eq_ignore_ascii_case(VOXCPM2_ARCHITECTURE)
}

/// The `config.json` beside a VoxCPM2 input, verbatim, or `None` when
/// there is none.
///
/// # Errors
/// The file exists and cannot be read.
pub fn voxcpm2_config_json(input_path: &Path) -> Result<Option<String>> {
    let Some(dir) = checkpoint_dir(input_path) else {
        return Ok(None);
    };
    let path = dir.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    std::fs::read_to_string(&path)
        .map(Some)
        .with_context(|| format!("Failed to read {}", path.display()))
}

/// Is `name` an embedded AudioVAE tensor?
pub fn is_audio_vae_tensor(name: &str) -> bool {
    name.starts_with(AUDIO_VAE_PREFIX)
}

/// Plan a float AudioVAE tensor, or `None` when the name is not one.
fn audio_vae_plan(name: &str, dtype: UnifiedDType) -> Option<TensorPlan> {
    if !is_audio_vae_tensor(name) {
        return None;
    }
    Some(if dtype.is_float() {
        TensorPlan::StoreF32(FallbackReason::AudioVae)
    } else {
        TensorPlan::StoreAsIs
    })
}

/// [`plan_tensor`], with the AudioVAE rule in front of it.
pub fn plan_named_tensor(
    name: &str,
    shape: &[usize],
    dtype: UnifiedDType,
    quant_type: Option<QuantType>,
) -> TensorPlan {
    if let Some(plan) = audio_vae_plan(name, dtype) {
        return plan;
    }
    plan_tensor(shape, dtype, quant_type)
}

/// [`plan_mixed_tensor`], with the AudioVAE rule in front of it.
///
/// A mix assigns a format per role, and the VAE's names match no role the
/// rules know, so without this a `[4, N]` sr-cond embed would take the base
/// format. The rule applies whatever the mix assigned.
pub fn plan_named_mixed_tensor(
    name: &str,
    shape: &[usize],
    dtype: UnifiedDType,
    assigned: Option<QuantType>,
) -> TensorPlan {
    if let Some(plan) = audio_vae_plan(name, dtype) {
        return plan;
    }
    plan_mixed_tensor(shape, dtype, assigned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape `plan_tensor` would quantize: rank 2, row a whole block.
    const EMBED: &[usize] = &[4, 256];

    #[test]
    fn the_architecture_test_ignores_case() {
        assert!(is_voxcpm2("voxcpm2"));
        assert!(is_voxcpm2("VoxCPM2"));
        assert!(is_voxcpm2(" voxcpm2 "));
        assert!(!is_voxcpm2("voxcpm"));
        assert!(!is_voxcpm2("llama"));
    }

    #[test]
    fn a_vae_embed_stays_f32_in_a_flat_run() {
        assert_eq!(
            plan_tensor(EMBED, UnifiedDType::F32, Some(QuantType::Q4K)),
            TensorPlan::Quantize(QuantType::Q4K)
        );
        assert_eq!(
            plan_named_tensor(
                "vae.decoder.sr_cond_model.2.scale_embed.weight",
                EMBED,
                UnifiedDType::F32,
                Some(QuantType::Q4K)
            ),
            TensorPlan::StoreF32(FallbackReason::AudioVae)
        );
    }

    #[test]
    fn a_vae_embed_stays_f32_in_a_mix_whatever_was_assigned() {
        assert_eq!(
            plan_named_mixed_tensor(
                "vae.decoder.sr_cond_model.2.scale_embed.weight",
                EMBED,
                UnifiedDType::F32,
                Some(QuantType::Q6K)
            ),
            TensorPlan::StoreF32(FallbackReason::AudioVae)
        );
        assert_eq!(
            plan_named_mixed_tensor("vae.encoder.block.0.bias", &[8], UnifiedDType::F32, None),
            TensorPlan::StoreF32(FallbackReason::AudioVae)
        );
    }

    /// The I32 boundaries vector passes through as any integer tensor does.
    #[test]
    fn a_non_float_vae_tensor_passes_through() {
        assert_eq!(
            plan_named_tensor(
                "vae.decoder.sr_bin_boundaries",
                &[3],
                UnifiedDType::I32,
                Some(QuantType::Q4K)
            ),
            TensorPlan::StoreAsIs
        );
        assert_eq!(
            plan_named_mixed_tensor(
                "vae.decoder.sr_bin_boundaries",
                &[3],
                UnifiedDType::I32,
                Some(QuantType::Q4K)
            ),
            TensorPlan::StoreAsIs
        );
    }

    /// No quantizer: the codec is still pinned to F32, so a dtype cast
    /// never reaches it.
    #[test]
    fn a_dense_run_keeps_the_vae_f32() {
        assert_eq!(
            plan_named_tensor(
                "vae.decoder.model.0.weight",
                &[8, 1, 7],
                UnifiedDType::F32,
                None
            ),
            TensorPlan::StoreF32(FallbackReason::AudioVae)
        );
        assert_eq!(
            plan_named_tensor("model.norm.weight", &[8], UnifiedDType::F32, None),
            TensorPlan::StoreAsIs
        );
    }

    #[test]
    fn the_config_json_is_read_verbatim_or_not_at_all() {
        let dir = tempfile::TempDir::new().unwrap();
        assert_eq!(voxcpm2_config_json(dir.path()).unwrap(), None);
        let text = "{\n  \"architecture\": \"voxcpm2\",\n  \"patch_size\": 4\n}\n";
        std::fs::write(dir.path().join("config.json"), text).unwrap();
        assert_eq!(
            voxcpm2_config_json(dir.path()).unwrap().as_deref(),
            Some(text)
        );
    }

    #[test]
    fn a_stack_tensor_is_planned_as_before() {
        assert_eq!(
            plan_named_tensor(
                "base_lm.layers.0.self_attn.q_proj.weight",
                EMBED,
                UnifiedDType::BF16,
                Some(QuantType::Q4K)
            ),
            TensorPlan::Quantize(QuantType::Q4K)
        );
        assert_eq!(
            plan_named_mixed_tensor("model.norm.weight", &[8], UnifiedDType::F32, None),
            TensorPlan::StoreF32(FallbackReason::Rank)
        );
    }
}
