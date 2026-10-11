//! A HuggingFace `config.json`, read the way `convert_hf_to_gguf.py` reads it.
//!
//! llama.cpp's converter moves a multimodal checkpoint's `text_config` to the
//! top level, then takes the first spelling present from a list of synonyms.
//! This module does the same, so each GGUF key comes from the field llama.cpp
//! reads for it.

use anyhow::{Context, Result};
use serde_json::{Map, Value};
use std::path::Path;

/// The fields of one `config.json`, with `text_config` merged over the root.
#[derive(Debug, Clone, Default)]
pub struct HfConfig {
    fields: Map<String, Value>,
}

impl HfConfig {
    /// A config from parsed JSON. A non-object value yields an empty config.
    pub fn from_value(value: Value) -> Self {
        let mut fields = match value {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        if let Some(Value::Object(text)) = fields.get("text_config").cloned() {
            for (key, value) in text {
                fields.insert(key, value);
            }
        }
        Self { fields }
    }

    /// The `config.json` in `dir`, or `None` when the directory has none.
    ///
    /// # Errors
    /// The file exists but cannot be read or is not valid JSON.
    pub fn read(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("config.json");
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("{} is not valid JSON", path.display()))?;
        Ok(Some(Self::from_value(value)))
    }

    /// The value of `key`, when present and not null.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.get(key).filter(|v| !v.is_null())
    }

    /// The first of `keys` present and not null, in list order.
    pub fn find(&self, keys: &[&str]) -> Option<&Value> {
        keys.iter().find_map(|key| self.get(key))
    }

    /// The first of `keys` present, as an unsigned 32-bit integer.
    ///
    /// A value that is not a whole number in range is logged and skipped, so
    /// the key it feeds is omitted rather than written wrong.
    pub fn find_u32(&self, keys: &[&str]) -> Option<u32> {
        let value = self.find(keys)?;
        let found = value.as_u64().and_then(|v| u32::try_from(v).ok());
        if found.is_none() {
            log::warn!(
                "config.json {keys:?} is {value}, not a 32-bit count; the GGUF key is omitted"
            );
        }
        found
    }

    /// The first of `keys` present, as a 32-bit float.
    pub fn find_f32(&self, keys: &[&str]) -> Option<f32> {
        let value = self.find(keys)?;
        let found = value.as_f64().map(|v| v as f32);
        if found.is_none() {
            log::warn!("config.json {keys:?} is {value}, not a number; the GGUF key is omitted");
        }
        found
    }

    /// `num_attention_heads`, under llama.cpp's synonyms.
    pub fn head_count(&self) -> Option<u32> {
        self.find_u32(&["num_attention_heads", "n_head", "n_heads"])
    }

    /// `num_key_value_heads`, under llama.cpp's synonyms.
    pub fn head_count_kv(&self) -> Option<u32> {
        self.find_u32(&["num_key_value_heads", "n_kv_heads"])
    }

    /// `hidden_size`, under llama.cpp's synonyms.
    pub fn embedding_length(&self) -> Option<u32> {
        self.find_u32(&["hidden_size", "n_embd", "dim"])
    }

    /// The per-head size: `head_dim` when stated, else `hidden_size / heads`.
    pub fn head_dim(&self) -> Option<u32> {
        if let Some(dim) = self.get("head_dim").and_then(Value::as_u64) {
            return u32::try_from(dim).ok();
        }
        self.embedding_length()?.checked_div(self.head_count()?)
    }

    /// The RoPE parameters, with the top-level fields llama.cpp mirrors in.
    ///
    /// The source is `rope_parameters`, else `rope_scaling`. A model without
    /// per-attention-type blocks also takes `rope_theta`, the `type` spelling
    /// of `rope_type`, `partial_rotary_factor` and
    /// `original_max_position_embeddings` from the top level.
    pub fn rope_parameters(&self) -> Map<String, Value> {
        let mut params = match self
            .get("rope_parameters")
            .or_else(|| self.get("rope_scaling"))
        {
            Some(Value::Object(map)) => map.clone(),
            _ => Map::new(),
        };
        if params.contains_key("full_attention") || params.contains_key("sliding_attention") {
            return params;
        }
        let local_theta = self.find(&[
            "local_rope_theta",
            "rope_local_theta",
            "rope_theta_local",
            "swa_rope_theta",
            "rope_local_base_freq",
        ]);
        if let Some(theta) = local_theta {
            let mut sliding = Map::new();
            sliding.insert("rope_theta".into(), theta.clone());
            params.insert("sliding_attention".into(), Value::Object(sliding));
        }
        let mirrored = [
            (
                "rope_theta",
                self.find(&[
                    "global_rope_theta",
                    "rope_global_theta",
                    "rope_theta_global",
                    "rope_theta",
                    "rotary_emb_base",
                ]),
            ),
            ("rope_type", params.get("type").filter(|v| !v.is_null())),
            (
                "partial_rotary_factor",
                self.find(&["partial_rotary_factor", "rope_pct", "rope_percent"]),
            ),
            (
                "original_max_position_embeddings",
                self.get("original_max_position_embeddings"),
            ),
        ];
        let mirrored: Vec<(&str, Value)> = mirrored
            .into_iter()
            .filter_map(|(key, value)| value.map(|v| (key, v.clone())))
            .collect();
        for (key, value) in mirrored {
            params.entry(key).or_insert(value);
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_config_fields_override_the_root() {
        let config = HfConfig::from_value(json!({
            "hidden_size": 1, "text_config": {"hidden_size": 64, "num_attention_heads": 4}
        }));
        assert_eq!(config.embedding_length(), Some(64));
        assert_eq!(config.head_dim(), Some(16));
    }

    #[test]
    fn the_first_present_synonym_wins_and_null_is_absent() {
        let config = HfConfig::from_value(json!({"n_head": 8, "num_attention_heads": null}));
        assert_eq!(config.head_count(), Some(8));
        assert_eq!(config.head_count_kv(), None);
    }

    #[test]
    fn rope_parameters_mirror_the_top_level_fields() {
        let config = HfConfig::from_value(json!({
            "rope_theta": 500000.0,
            "rope_scaling": {"type": "linear", "factor": 2.0}
        }));
        let rope = config.rope_parameters();
        assert_eq!(rope["rope_theta"], json!(500000.0));
        assert_eq!(rope["rope_type"], json!("linear"));
        assert_eq!(rope["factor"], json!(2.0));
    }
}
