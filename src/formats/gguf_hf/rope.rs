//! RoPE keys and RoPE tensors for a GGUF header, from `config.json`.
//!
//! llama.cpp writes linear and YaRN scaling as `{arch}.rope.scaling.*` keys,
//! LongRoPE as a key plus two divisor tensors, and llama3 scaling as the
//! `rope_freqs.weight` divisor tensor alone.

use super::config::HfConfig;
use super::hparams::{i32_array, ArchKeys};
use crate::formats::gguf::GgufValue;
use serde_json::{Map, Value};

/// A 1-D F32 tensor the converter adds beside the checkpoint's own tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtraTensor {
    /// The GGUF tensor name.
    pub name: String,
    /// The values, stored as F32.
    pub values: Vec<f32>,
}

/// The rope block of `TextModel.set_gguf_parameters`, plus the rotary width.
pub(crate) fn rope_metadata(arch: &str, config: &HfConfig, out: &mut ArchKeys) {
    let params = config.rope_parameters();
    let full = full_attention(&params);
    let num = |key: &str| full.get(key).and_then(Value::as_f64);

    out.u32(
        "rope.dimension_count",
        rope_dimension_count(config, &params),
    );

    let rope_type = full
        .get("rope_type")
        .or_else(|| full.get("type"))
        .and_then(Value::as_str);
    let factor = num("factor").map(|f| f as f32);
    let original = full
        .get("original_max_position_embeddings")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok());
    match (rope_type, factor) {
        (Some("linear"), Some(factor)) => {
            out.put("rope.scaling.type", GgufValue::String("linear".into()));
            out.f32("rope.scaling.factor", Some(factor));
        }
        (Some("yarn"), Some(factor)) => {
            out.put("rope.scaling.type", GgufValue::String("yarn".into()));
            out.f32("rope.scaling.factor", Some(factor));
            out.u32("rope.scaling.original_context_length", original);
            out.f32(
                "rope.scaling.yarn_ext_factor",
                num("extrapolation_factor").map(|v| v as f32),
            );
            let attn = num("attention_factor").or_else(|| num("attn_factor"));
            out.f32("rope.scaling.yarn_attn_factor", attn.map(|v| v as f32));
            out.f32(
                "rope.scaling.yarn_beta_fast",
                num("beta_fast").map(|v| v as f32),
            );
            out.f32(
                "rope.scaling.yarn_beta_slow",
                num("beta_slow").map(|v| v as f32),
            );
        }
        (Some("su" | "longrope"), _) => {
            out.put("rope.scaling.type", GgufValue::String("longrope".into()));
        }
        // llama3 scaling is the `rope_freqs.weight` tensor, and `default`
        // and `dynamic` write nothing, as in llama.cpp.
        (Some(_), _) | (None, _) => {}
    }

    if let Some(Value::Array(sections)) = params.get("mrope_section") {
        let mut values: Vec<i32> = sections
            .iter()
            .filter_map(Value::as_i64)
            .filter_map(|v| i32::try_from(v).ok())
            .collect();
        values.resize(values.len().max(4), 0);
        values.truncate(4);
        out.put("rope.dimension_sections", i32_array(values));
    }

    out.f32("rope.freq_base", num("rope_theta").map(|v| v as f32));
    let swa_theta = params
        .get("sliding_attention")
        .and_then(|s| s.get("rope_theta"))
        .and_then(Value::as_f64);
    out.f32("rope.freq_base_swa", swa_theta.map(|v| v as f32));

    if arch == "phi3" {
        let original = params
            .get("original_max_position_embeddings")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());
        out.u32("rope.scaling.original_context_length", original);
    }
}

/// The rotated width of one head: `head_dim`, else `hidden_size / heads`,
/// scaled by `partial_rotary_factor` when the config states one.
fn rope_dimension_count(config: &HfConfig, params: &Map<String, Value>) -> Option<u32> {
    let head_dim = config.head_dim()?;
    match params.get("partial_rotary_factor").and_then(Value::as_f64) {
        Some(factor) => Some((factor * f64::from(head_dim)) as u32),
        None => Some(head_dim),
    }
}

/// The `full_attention` block when present, else the whole parameter map.
fn full_attention(params: &Map<String, Value>) -> &Map<String, Value> {
    match params.get("full_attention") {
        Some(Value::Object(full)) => full,
        _ => params,
    }
}

/// The RoPE tensors llama.cpp adds for this config.
///
/// llama3 scaling gives `rope_freqs.weight`, computed exactly as
/// `LlamaModel.generate_extra_tensors` computes it. LongRoPE gives
/// `rope_factors_long.weight` and `rope_factors_short.weight`, copied from
/// the config's two lists.
pub fn rope_tensors(config: &HfConfig) -> Vec<ExtraTensor> {
    let params = config.rope_parameters();
    let full = full_attention(&params);
    let rope_type = full
        .get("rope_type")
        .or_else(|| full.get("type"))
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase);
    match rope_type.as_deref() {
        Some("llama3") => llama3_factors(config, full)
            .map(|values| {
                vec![ExtraTensor {
                    name: "rope_freqs.weight".into(),
                    values,
                }]
            })
            .unwrap_or_default(),
        Some("su" | "longrope") => {
            let list = |key: &str| -> Option<Vec<f32>> {
                let items = full.get(key)?.as_array()?;
                items.iter().map(|v| v.as_f64().map(|f| f as f32)).collect()
            };
            let mut out = Vec::new();
            for (key, name) in [
                ("long_factor", "rope_factors_long.weight"),
                ("short_factor", "rope_factors_short.weight"),
            ] {
                match list(key) {
                    Some(values) => out.push(ExtraTensor {
                        name: name.into(),
                        values,
                    }),
                    None => log::warn!(
                        "config.json declares LongRoPE with no numeric '{key}' list; \
                         {name} is omitted"
                    ),
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// llama3's per-dimension divisors, or `None` when the base or width is absent.
///
/// The defaults for `factor`, `low_freq_factor`, `high_freq_factor` and
/// `original_max_position_embeddings` are llama.cpp's and Meta's.
fn llama3_factors(config: &HfConfig, params: &Map<String, Value>) -> Option<Vec<f32>> {
    let num = |key: &str, default: f64| params.get(key).and_then(Value::as_f64).unwrap_or(default);
    let Some(base) = params.get("rope_theta").and_then(Value::as_f64) else {
        log::warn!(
            "config.json declares llama3 RoPE with no rope_theta; rope_freqs.weight is omitted"
        );
        return None;
    };
    let dim = config.head_dim()?;
    let factor = num("factor", 8.0) as f32;
    let low = num("low_freq_factor", 1.0) as f32;
    let high = num("high_freq_factor", 4.0) as f32;
    let old_context = num("original_max_position_embeddings", 8192.0) as f32;
    let low_wavelen = old_context / low;
    let high_wavelen = old_context / high;
    // llama.cpp runs this loop over float32 torch tensors, so every step is
    // float32 arithmetic. The same steps in f32 give the same bits.
    let two_pi = (2.0 * std::f64::consts::PI) as f32;
    let factors = (0..dim / 2)
        .map(|i| {
            let exponent = (2 * i) as f32 / dim as f32;
            let freq = 1.0f32 / (base as f32).powf(exponent);
            let wavelen = two_pi / freq;
            if wavelen < high_wavelen {
                1.0
            } else if wavelen > low_wavelen {
                factor
            } else {
                let smooth = (old_context / wavelen - low) / (high - low);
                1.0 / ((1.0 - smooth) / factor + smooth)
            }
        })
        .collect();
    Some(factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn llama32() -> HfConfig {
        HfConfig::from_value(json!({
            "hidden_size": 2048, "num_attention_heads": 32, "head_dim": 64,
            "rope_theta": 500000.0,
            "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                             "original_max_position_embeddings": 8192, "rope_type": "llama3"}
        }))
    }

    /// The divisors `convert_hf_to_gguf.py` writes for Llama 3.2 1B: 1 for
    /// the fast dimensions, 32 for the slow ones, a smooth ramp between.
    #[test]
    fn llama3_scaling_becomes_rope_freqs() {
        let tensors = rope_tensors(&llama32());
        assert_eq!(tensors.len(), 1);
        let freqs = &tensors[0];
        assert_eq!(freqs.name, "rope_freqs.weight");
        assert_eq!(freqs.values.len(), 32);
        // Bit patterns of llama.cpp's torch float32 result for this config.
        let ramp: Vec<u32> = freqs.values[14..19].iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            ramp,
            [1065353216, 1070816962, 1079161967, 1092266734, 1107296256]
        );
        assert!(freqs.values[..15].iter().all(|v| *v == 1.0));
        assert!(freqs.values[18..].iter().all(|v| *v == 32.0));
        let mut out = ArchKeys::new("llama");
        rope_metadata("llama", &llama32(), &mut out);
        assert!(out.keys.iter().all(|(k, _)| !k.contains("scaling")));
    }

    #[test]
    fn linear_and_yarn_scaling_become_keys() {
        let config = HfConfig::from_value(json!({
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                             "original_max_position_embeddings": 32768, "beta_fast": 32.0}
        }));
        let mut out = ArchKeys::new("qwen2");
        rope_metadata("qwen2", &config, &mut out);
        let names: Vec<&str> = out.keys.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            names,
            [
                "qwen2.rope.scaling.type",
                "qwen2.rope.scaling.factor",
                "qwen2.rope.scaling.original_context_length",
                "qwen2.rope.scaling.yarn_beta_fast",
            ]
        );
        assert!(rope_tensors(&config).is_empty());
    }
}
