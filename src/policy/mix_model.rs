//! The model facts a mixed-precision GGUF rule reads.
//!
//! llama.cpp's `llama_tensor_get_type` branches on more than a tensor's name:
//! the layer count divides every FFN-down rule, the expert count switches the
//! attention tensors to Q8_0, the architecture selects the Falcon branches,
//! and one rule fires only on the 70B shape. compressr has to gather those
//! four from a checkpoint that was never asked for them.

use crate::loader::ModelMetadata;
use anyhow::{Context, Result};
use std::path::Path;

/// The model facts a mix rule reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixModel {
    /// `general.architecture`, lowercased
    pub arch: String,
    /// Transformer blocks in the model
    ///
    /// Every FFN-down rule divides by this, so an allocation computed from a
    /// wrong layer count is a different allocation.
    pub n_layer: usize,
    /// Routed experts, or 0 for a dense model
    pub n_expert: usize,
    /// Whether this is the 70B shape upstream bumps attention-V for
    pub is_70b: bool,
}

impl MixModel {
    /// Whether the Falcon-specific branches apply.
    pub fn is_falcon(&self) -> bool {
        self.arch.contains("falcon")
    }
}

/// Gather the facts a mix needs, from the model metadata and `config.json`.
///
/// The checkpoint's `config.json` is read when there is one: a SafeTensors
/// header carries no head count and no expert count, and both change the
/// allocation. A GGUF input supplies them through `metadata` instead.
///
/// # Errors
/// The layer count is unavailable from either source. Every FFN-down rule is
/// a fraction of it, so guessing one would silently produce an allocation that
/// is not the mix that was asked for.
pub fn mix_model_facts(
    arch: &str,
    metadata: &ModelMetadata,
    input_path: &Path,
) -> Result<MixModel> {
    let config = read_model_config(input_path)?;
    let from_config = |key: &str| -> Option<usize> {
        config
            .as_ref()?
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
    };

    let n_layer = metadata
        .num_layers
        .or_else(|| from_config("num_hidden_layers"))
        .or_else(|| from_config("n_layer"))
        .or_else(|| from_config("num_layers"))
        .context(
            "A mixed-precision allocation needs the model's layer count, and neither the \
             checkpoint metadata nor a config.json beside it names one. Every FFN-down rule \
             is a fraction of that count, so there is no safe default. Use a flat \
             quantization such as q4_k instead.",
        )?;

    let n_expert = from_config("num_local_experts")
        .or_else(|| from_config("num_experts"))
        .or_else(|| from_config("n_routed_experts"))
        .unwrap_or(0);

    let n_head = metadata
        .num_heads
        .or_else(|| from_config("num_attention_heads"));
    let n_head_kv = metadata
        .num_kv_heads
        .or_else(|| from_config("num_key_value_heads"));

    Ok(MixModel {
        arch: arch.to_ascii_lowercase(),
        n_layer,
        n_expert,
        is_70b: is_70b_shape(n_layer, n_head, n_head_kv),
    })
}

/// Whether the model is the shape upstream labels 70B.
///
/// llama.cpp reads that label out of a per-architecture hparams table. Its
/// llama row is 80 layers with grouped-query attention, which is the row
/// reproduced here; a model of another architecture that upstream also labels
/// 70B is NOT recognised, and takes the base allocation on attention-V.
fn is_70b_shape(n_layer: usize, n_head: Option<usize>, n_head_kv: Option<usize>) -> bool {
    n_layer == 80 && matches!((n_head, n_head_kv), (Some(q), Some(kv)) if q != kv)
}

/// Read the checkpoint's `config.json`, if it has one.
///
/// # Errors
/// The file exists but is unreadable or is not JSON. Ignoring that would
/// silently drop the expert count and change the allocation.
fn read_model_config(input_path: &Path) -> Result<Option<serde_json::Value>> {
    let Some(dir) = super::checkpoint_dir(input_path) else {
        return Ok(None);
    };
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", config_path.display()))?;
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn the_expert_count_comes_from_config_json() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"num_hidden_layers": 32, "num_local_experts": 8}"#,
        )
        .unwrap();

        let facts = mix_model_facts("llama", &ModelMetadata::default(), dir.path()).unwrap();
        assert_eq!(facts.n_layer, 32);
        assert_eq!(facts.n_expert, 8);
        assert!(!facts.is_70b);
    }

    /// A dense checkpoint names no experts, and 0 is the answer that makes
    /// every mixture-of-experts rule inapplicable.
    #[test]
    fn a_dense_model_reports_no_experts() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"num_hidden_layers": 32}"#,
        )
        .unwrap();

        assert_eq!(
            mix_model_facts("llama", &ModelMetadata::default(), dir.path())
                .unwrap()
                .n_expert,
            0
        );
    }

    /// The layer count divides every FFN-down rule, so its absence is an
    /// error rather than a default.
    #[test]
    fn a_missing_layer_count_is_an_error() {
        let dir = TempDir::new().unwrap();
        let err = mix_model_facts("llama", &ModelMetadata::default(), dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("layer count"), "{err}");
    }

    /// The metadata wins over config.json: a GGUF input carries the count the
    /// file was actually built with.
    #[test]
    fn the_metadata_layer_count_wins() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"num_hidden_layers": 32}"#,
        )
        .unwrap();
        let metadata = ModelMetadata {
            num_layers: Some(48),
            ..ModelMetadata::default()
        };

        assert_eq!(
            mix_model_facts("llama", &metadata, dir.path())
                .unwrap()
                .n_layer,
            48
        );
    }

    /// 80 layers alone is not the 70B shape: the 65B model has 80 layers too
    /// and does not use grouped-query attention.
    #[test]
    fn the_70b_shape_needs_grouped_query_attention() {
        assert!(is_70b_shape(80, Some(64), Some(8)));
        assert!(!is_70b_shape(80, Some(64), Some(64)));
        assert!(!is_70b_shape(80, None, None));
        assert!(!is_70b_shape(32, Some(32), Some(8)));
    }

    #[test]
    fn the_falcon_branches_key_off_the_architecture() {
        let model = MixModel {
            arch: "falcon".into(),
            n_layer: 32,
            n_expert: 0,
            is_70b: false,
        };
        assert!(model.is_falcon());
        assert!(!MixModel {
            arch: "llama".into(),
            ..model
        }
        .is_falcon());
    }
}
