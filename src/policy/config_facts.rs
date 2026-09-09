//! Reading the numbers a mixed-precision rule needs out of a `config.json`.
//!
//! # Why a top-level lookup is not enough
//!
//! A single-stack checkpoint puts `num_hidden_layers` at the top level of its
//! `config.json`, and a top-level lookup finds it. A checkpoint that carries
//! several components does not: it nests the language model's own block under
//! a key of its own (`lm_config`, `text_config`, `llm_config`) and keeps the
//! other components beside it. A top-level lookup finds nothing there and the
//! caller falls back to a count inferred from tensor names, which is the count
//! of ONE component rather than the number the config states.
//!
//! So the search here descends. The order is load-bearing and is documented on
//! [`find_config_usize`]: shallowest first, and within one depth the key
//! spellings in priority order. A model whose nested blocks disagree — a
//! decoder stack of 12 beside a language stack of 28 — must not have the
//! search's traversal order decide which one answers.

use serde_json::Value;

/// How deep the search descends before it stops.
///
/// Every nesting convention observed in a checkpoint config puts the component
/// block one level down (`lm_config.num_hidden_layers`). Two spare levels cover
/// a config that wraps its components in a container key; beyond that a hit is
/// far more likely to be an unrelated number than the model's layer count.
const MAX_CONFIG_DEPTH: usize = 3;

/// Layer-count spellings, in the order a config is searched for them.
///
/// `num_hidden_layers` is the HuggingFace convention and is checked first
/// everywhere before `n_layer` is checked anywhere.
pub const LAYER_COUNT_KEYS: &[&str] = &["num_hidden_layers", "n_layer", "num_layers"];

/// Routed-expert-count spellings, in search order.
pub const EXPERT_COUNT_KEYS: &[&str] = &["num_local_experts", "num_experts", "n_routed_experts"];

/// Attention head-count spellings, in search order.
pub const HEAD_COUNT_KEYS: &[&str] = &["num_attention_heads"];

/// Key-value head-count spellings, in search order.
pub const KV_HEAD_COUNT_KEYS: &[&str] = &["num_key_value_heads"];

/// Where a layer count came from.
///
/// A mixed-precision allocation is a function of this number, so a report that
/// states the number without stating where it was read is not auditable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerCountSource {
    /// The input file's own metadata: a GGUF `block_count` key
    Metadata,
    /// A `config.json` key, named by its full path
    Config(String),
    /// The highest layer index in the tensor names, plus one
    ///
    /// A multi-component checkpoint has several layer ranges, so this is the
    /// highest index of ANY of them and not the size of a single stack.
    InferredFromTensorNames,
}

impl std::fmt::Display for LayerCountSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerCountSource::Metadata => write!(f, "input metadata (block_count)"),
            LayerCountSource::Config(path) => write!(f, "config.json:{path}"),
            LayerCountSource::InferredFromTensorNames => {
                write!(f, "inferred from tensor names (no config key)")
            }
        }
    }
}

/// One non-negative integer read out of a config, with the path it was read from.
///
/// The traversal is breadth-first, so a key at the top level always wins over
/// the same key nested inside a component block. Within one depth the `keys`
/// list decides, and objects are visited in the order `serde_json` holds them,
/// which is sorted by key. Both orders are fixed rather than incidental: the
/// same config must always yield the same allocation.
///
/// Arrays are not descended: a list of per-layer settings holds numbers that
/// are not counts of anything.
pub fn find_config_usize(config: &Value, keys: &[&str]) -> Option<(String, usize)> {
    let mut frontier: Vec<(String, &Value)> = vec![(String::new(), config)];

    for _ in 0..=MAX_CONFIG_DEPTH {
        // Every key spelling is tried across the whole of this depth before
        // the search moves down, so a primary spelling one level down beats a
        // secondary spelling one level down.
        for key in keys {
            for (path, value) in &frontier {
                let Some(found) = value.get(key).and_then(Value::as_u64) else {
                    continue;
                };
                let full = if path.is_empty() {
                    (*key).to_string()
                } else {
                    format!("{path}.{key}")
                };
                return Some((full, found as usize));
            }
        }

        let mut next: Vec<(String, &Value)> = Vec::new();
        for (path, value) in &frontier {
            let Some(object) = value.as_object() else {
                continue;
            };
            for (name, child) in object {
                if !child.is_object() {
                    continue;
                }
                let child_path = if path.is_empty() {
                    name.clone()
                } else {
                    format!("{path}.{name}")
                };
                next.push((child_path, child));
            }
        }
        if next.is_empty() {
            return None;
        }
        frontier = next;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> Value {
        serde_json::from_str(text).expect("test fixture is valid JSON")
    }

    #[test]
    fn a_top_level_key_is_found() {
        let config = json(r#"{"num_hidden_layers": 32}"#);
        assert_eq!(
            find_config_usize(&config, LAYER_COUNT_KEYS),
            Some(("num_hidden_layers".to_string(), 32))
        );
    }

    /// A multi-component checkpoint nests its language model's own block, and
    /// the count is only reachable by descending into it.
    #[test]
    fn a_nested_key_is_found_with_its_path() {
        let config = json(r#"{"architecture": "voxcpm2", "lm_config": {"num_hidden_layers": 28}}"#);
        assert_eq!(
            find_config_usize(&config, LAYER_COUNT_KEYS),
            Some(("lm_config.num_hidden_layers".to_string(), 28))
        );
    }

    /// The primary spelling is checked across every block of a depth before
    /// any secondary spelling is checked in any of them. `dit_config` sorts
    /// before `lm_config`, so a traversal that took the first hit per object
    /// would answer 12 — the diffusion stack — for the language model.
    #[test]
    fn the_primary_spelling_wins_across_a_whole_depth() {
        let config = json(
            r#"{
                "dit_config": {"num_layers": 12},
                "encoder_config": {"num_layers": 12},
                "lm_config": {"num_hidden_layers": 28}
            }"#,
        );
        assert_eq!(
            find_config_usize(&config, LAYER_COUNT_KEYS),
            Some(("lm_config.num_hidden_layers".to_string(), 28))
        );
    }

    /// Depth beats spelling: a top-level `num_layers` is the model's own, and
    /// a `num_hidden_layers` inside a component block is that component's.
    #[test]
    fn a_shallower_key_wins_over_a_deeper_one() {
        let config = json(r#"{"num_layers": 32, "vision_config": {"num_hidden_layers": 24}}"#);
        assert_eq!(
            find_config_usize(&config, LAYER_COUNT_KEYS),
            Some(("num_layers".to_string(), 32))
        );
    }

    #[test]
    fn a_config_naming_no_count_answers_nothing() {
        let config = json(r#"{"patch_size": 4, "audio": {"sample_rate": 16000}}"#);
        assert_eq!(find_config_usize(&config, LAYER_COUNT_KEYS), None);
    }

    /// An array holds per-layer settings, not counts, so it is not descended.
    #[test]
    fn arrays_are_not_searched() {
        let config = json(r#"{"stages": [{"num_hidden_layers": 4}]}"#);
        assert_eq!(find_config_usize(&config, LAYER_COUNT_KEYS), None);
    }

    #[test]
    fn the_source_names_itself_in_the_report() {
        assert_eq!(
            LayerCountSource::Config("lm_config.num_hidden_layers".into()).to_string(),
            "config.json:lm_config.num_hidden_layers"
        );
        assert_eq!(
            LayerCountSource::InferredFromTensorNames.to_string(),
            "inferred from tensor names (no config key)"
        );
    }
}
