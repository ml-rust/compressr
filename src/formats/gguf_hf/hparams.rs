//! Architecture hyperparameters for a GGUF header, from `config.json`.
//!
//! Each key carries llama.cpp's name and value type, as `convert_hf_to_gguf.py`
//! writes it: the generic `TextModel.set_gguf_parameters` set, plus the extras
//! of the architecture's own class. A key whose source field is absent is
//! omitted. llama.cpp's converter fills a few Gemma 3 fields with defaults
//! when absent. compressr writes no default.

use super::config::HfConfig;
use super::rope::rope_metadata;
use crate::formats::gguf::{GgufArray, GgufValue};
use serde_json::Value;

/// Collects `{arch}.<name>` keys in the order llama.cpp writes them.
pub(crate) struct ArchKeys {
    arch: String,
    pub(crate) keys: Vec<(String, GgufValue)>,
}

impl ArchKeys {
    pub(crate) fn new(arch: &str) -> Self {
        Self {
            arch: arch.to_string(),
            keys: Vec::new(),
        }
    }

    pub(crate) fn put(&mut self, name: &str, value: GgufValue) {
        let key = format!("{}.{name}", self.arch);
        // A later write of the same key replaces the earlier one, as
        // llama.cpp's writer does when a class overrides a base key.
        self.keys.retain(|(k, _)| *k != key);
        self.keys.push((key, value));
    }

    pub(crate) fn u32(&mut self, name: &str, value: Option<u32>) {
        if let Some(v) = value {
            self.put(name, GgufValue::Uint32(v));
        }
    }

    pub(crate) fn f32(&mut self, name: &str, value: Option<f32>) {
        if let Some(v) = value {
            self.put(name, GgufValue::Float32(v));
        }
    }
}

/// Every `{arch}.*` key `config` supplies, for llama.cpp architecture `arch`.
pub fn arch_metadata(arch: &str, config: &HfConfig) -> Vec<(String, GgufValue)> {
    let mut out = ArchKeys::new(arch);
    base_keys(arch, config, &mut out);
    arch_keys(arch, config, &mut out);
    out.keys
}

/// `TextModel.set_gguf_parameters`, plus the vocabulary size and rotary width
/// llama.cpp's `llama` class adds.
fn base_keys(arch: &str, config: &HfConfig, out: &mut ArchKeys) {
    out.u32("vocab_size", config.find_u32(&["vocab_size"]));
    out.u32(
        "block_count",
        config.find_u32(&["n_layers", "num_hidden_layers", "n_layer", "num_layers"]),
    );
    out.u32(
        "context_length",
        config.find_u32(&[
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
            "max_sequence_length",
            "model_max_length",
        ]),
    );
    out.u32("embedding_length", config.embedding_length());
    out.u32(
        "feed_forward_length",
        config.find_u32(&[
            "prefix_dense_intermediate_size",
            "dense_intermediate_size",
            "intermediate_size",
            "n_inner",
            "hidden_dim",
        ]),
    );
    out.u32("attention.head_count", config.head_count());
    out.u32("attention.head_count_kv", config.head_count_kv());
    if config.get("is_causal").and_then(Value::as_bool) == Some(false) {
        out.put("attention.causal", GgufValue::Bool(false));
    }

    rope_metadata(arch, config, out);

    out.f32(
        "attention.layer_norm_rms_epsilon",
        config.find_f32(&["rms_norm_eps", "norm_eps"]),
    );
    out.f32(
        "attention.layer_norm_epsilon",
        config.find_f32(&["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"]),
    );
    out.u32(
        "expert_count",
        config.find_u32(&["num_local_experts", "num_experts", "n_routed_experts"]),
    );
    out.u32(
        "expert_used_count",
        config.find_u32(&[
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
        ]),
    );
    out.u32("expert_group_count", config.find_u32(&["n_group"]));
    out.u32("expert_group_used_count", config.find_u32(&["topk_group"]));
    let gating = config
        .find(&[
            "score_function",
            "scoring_func",
            "score_func",
            "moe_router_activation",
            "moe_router_activation_func",
            "expert_selection_fn",
        ])
        .and_then(Value::as_str);
    // llama.cpp's `ExpertGatingFuncType`.
    match gating {
        Some("softmax") => out.u32("expert_gating_func", Some(1)),
        Some("sigmoid") => out.u32("expert_gating_func", Some(2)),
        Some("sqrtsoftplus") => out.u32("expert_gating_func", Some(4)),
        Some(other) => log::warn!(
            "config.json names the expert gating function '{other}', which llama.cpp has no \
             value for; expert_gating_func is omitted"
        ),
        None => {}
    }
    let head_dim = config.get("head_dim").and_then(Value::as_u64);
    if let Some(dim) = head_dim.and_then(|d| u32::try_from(d).ok()) {
        out.u32("attention.key_length", Some(dim));
        out.u32("attention.value_length", Some(dim));
    }
}

/// The keys an architecture's own converter class adds.
fn arch_keys(arch: &str, config: &HfConfig, out: &mut ArchKeys) {
    match arch {
        "qwen2moe" | "qwen3moe" => {
            out.u32(
                "expert_feed_forward_length",
                config.find_u32(&["moe_intermediate_size"]),
            );
            out.u32(
                "expert_shared_feed_forward_length",
                config.find_u32(&["shared_expert_intermediate_size"]),
            );
        }
        "gemma2" => {
            out.f32(
                "attn_logit_softcapping",
                config.find_f32(&["attn_logit_softcapping"]),
            );
            out.f32(
                "final_logit_softcapping",
                config.find_f32(&["final_logit_softcapping"]),
            );
            out.u32(
                "attention.sliding_window",
                config.find_u32(&["sliding_window"]),
            );
        }
        "gemma3" => {
            let softcap = config.find_f32(&["final_logit_softcapping"]);
            out.f32("final_logit_softcapping", softcap.filter(|v| *v != 0.0));
            if config.find_u32(&["sliding_window_pattern"]) != Some(1) {
                out.u32(
                    "attention.sliding_window",
                    config.find_u32(&["sliding_window"]),
                );
            }
        }
        "phi3" => {
            out.u32(
                "attention.sliding_window",
                config.find_u32(&["sliding_window"]),
            );
        }
        _ => {}
    }
}

/// A `GgufValue` array of 32-bit integers.
pub(crate) fn i32_array(values: Vec<i32>) -> GgufValue {
    GgufValue::Array(GgufArray::Int32(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn keys(arch: &str, config: Value) -> Vec<(String, GgufValue)> {
        arch_metadata(arch, &HfConfig::from_value(config))
    }

    fn find<'a>(keys: &'a [(String, GgufValue)], key: &str) -> Option<&'a GgufValue> {
        keys.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    #[test]
    fn llama_gqa_config_writes_both_head_counts() {
        let out = keys(
            "llama",
            json!({
                "vocab_size": 128256, "num_hidden_layers": 16, "max_position_embeddings": 131072,
                "hidden_size": 2048, "intermediate_size": 8192, "num_attention_heads": 32,
                "num_key_value_heads": 8, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
                "head_dim": 64
            }),
        );
        assert!(matches!(
            find(&out, "llama.attention.head_count"),
            Some(GgufValue::Uint32(32))
        ));
        assert!(matches!(
            find(&out, "llama.attention.head_count_kv"),
            Some(GgufValue::Uint32(8))
        ));
        assert!(matches!(
            find(&out, "llama.feed_forward_length"),
            Some(GgufValue::Uint32(8192))
        ));
        assert!(matches!(
            find(&out, "llama.context_length"),
            Some(GgufValue::Uint32(131072))
        ));
        assert!(matches!(
            find(&out, "llama.rope.dimension_count"),
            Some(GgufValue::Uint32(64))
        ));
        assert!(matches!(
            find(&out, "llama.attention.key_length"),
            Some(GgufValue::Uint32(64))
        ));
        assert!(matches!(
            find(&out, "llama.attention.layer_norm_rms_epsilon"),
            Some(GgufValue::Float32(v)) if (*v - 1e-5).abs() < 1e-12
        ));
        assert!(matches!(
            find(&out, "llama.rope.freq_base"),
            Some(GgufValue::Float32(v)) if *v == 500000.0
        ));
    }

    /// A key whose source is absent is omitted, never filled in.
    #[test]
    fn absent_sources_write_no_key() {
        let out = keys(
            "llama",
            json!({"hidden_size": 64, "num_attention_heads": 4}),
        );
        assert!(find(&out, "llama.attention.head_count_kv").is_none());
        assert!(find(&out, "llama.context_length").is_none());
        assert!(find(&out, "llama.attention.key_length").is_none());
        assert!(find(&out, "llama.rope.freq_base").is_none());
        // The rotary width follows from hidden_size / heads, as llama.cpp derives it.
        assert!(matches!(
            find(&out, "llama.rope.dimension_count"),
            Some(GgufValue::Uint32(16))
        ));
    }

    #[test]
    fn moe_and_gemma_extras_use_llama_cpp_names() {
        let moe = keys(
            "qwen3moe",
            json!({"num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768}),
        );
        assert!(matches!(
            find(&moe, "qwen3moe.expert_count"),
            Some(GgufValue::Uint32(128))
        ));
        assert!(matches!(
            find(&moe, "qwen3moe.expert_used_count"),
            Some(GgufValue::Uint32(8))
        ));
        assert!(matches!(
            find(&moe, "qwen3moe.expert_feed_forward_length"),
            Some(GgufValue::Uint32(768))
        ));
        let gemma2 = keys(
            "gemma2",
            json!({"attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
                   "sliding_window": 4096, "head_dim": 256}),
        );
        assert!(matches!(
            find(&gemma2, "gemma2.attn_logit_softcapping"),
            Some(GgufValue::Float32(v)) if *v == 50.0
        ));
        assert!(matches!(
            find(&gemma2, "gemma2.attention.sliding_window"),
            Some(GgufValue::Uint32(4096))
        ));
        assert!(matches!(
            find(&gemma2, "gemma2.attention.value_length"),
            Some(GgufValue::Uint32(256))
        ));
    }
}
