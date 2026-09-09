//! Splitting a checkpoint into the transformer stacks it is built from.
//!
//! # Why a mix needs this
//!
//! llama.cpp's `llama_tensor_get_type` reads two running counters and one
//! denominator, and all three describe ONE stack of `n_layer` blocks indexed
//! `0..n_layer`. `use_more_bits(i, n)` then means "the first eighth of the
//! stack, the last eighth, and every third block of the middle".
//!
//! A checkpoint that holds several stacks — a language model, a residual
//! language model, an audio encoder and a decoder, each with its own layer
//! range starting at 0 — has no single `n`. Walking all of them as one run
//! gives the rule a numerator that counts past the denominator: the layers of
//! every stack after the first land in the "last eighth" branch and are all
//! promoted, and the layers of the first stack are compared against a
//! denominator that is not its own size.
//!
//! So compressr partitions the tensors by the stack they belong to and applies
//! the rule once per stack. That is compressr's own choice — llama.cpp has no
//! allocation for a multi-stack model — and [`super::mix_allocation`] states it
//! in the conversion report rather than letting it pass as an upstream rule.
//!
//! The alternative, one combined ordering over every stack, is rejected for the
//! reason above: it makes "the last eighth of the model" select whole small
//! components, which is not what the rule means.

/// The name the report gives tensors that belong to no stack.
///
/// Embeddings, the output head and the cross-component projections carry no
/// layer index. Their rules read no layer number, so they need no stack.
pub const UNLAYERED_COMPONENT: &str = "(unlayered)";

/// Path segments that mark the segment after them as a layer index.
///
/// The same set [`crate::loader::extract_layer_index`] matches, so a tensor
/// this module assigns to a stack is one that module also reads an index from.
const LAYER_MARKERS: &[&str] = &["layers", "h", "block", "blk"];

/// One transformer stack of a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    /// The tensor-name prefix up to and including the layer marker
    ///
    /// `base_lm.layers`, `model.layers`, `blk`, or [`UNLAYERED_COMPONENT`].
    pub name: String,
    /// Blocks in this stack: the highest index seen, plus one
    ///
    /// Zero for the unlayered component, which is not a stack.
    pub layer_count: usize,
    /// Positions in the caller's tensor list that belong to this stack
    pub tensor_indices: Vec<usize>,
}

/// Split one tensor name into its stack prefix and its layer index.
///
/// Returns `None` for a tensor that carries no layer index at all.
pub fn split_component(name: &str) -> Option<(&str, usize)> {
    let mut previous: Option<&str> = None;
    let mut previous_end = 0usize;
    let mut offset = 0usize;

    for segment in name.split('.') {
        if let Some(marker) = previous {
            if LAYER_MARKERS.contains(&marker) {
                if let Ok(index) = segment.parse::<usize>() {
                    return Some((&name[..previous_end], index));
                }
            }
        }
        previous = Some(segment);
        previous_end = offset + segment.len();
        offset = previous_end + 1;
    }

    None
}

/// Group a tensor list into the stacks it is built from.
///
/// Components appear in the order their first tensor appears, so the result is
/// a function of the checkpoint alone and not of any map iteration order.
pub fn partition_components(names: &[&str]) -> Vec<Component> {
    let mut components: Vec<Component> = Vec::new();
    let mut position_of: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

    for (index, name) in names.iter().copied().enumerate() {
        let (key, layer) = match split_component(name) {
            Some((prefix, layer)) => (prefix, Some(layer)),
            None => (UNLAYERED_COMPONENT, None),
        };
        let position = match position_of.get(key).copied() {
            Some(position) => position,
            None => {
                components.push(Component {
                    name: key.to_string(),
                    layer_count: 0,
                    tensor_indices: Vec::new(),
                });
                position_of.insert(key, components.len() - 1);
                components.len() - 1
            }
        };
        let component = &mut components[position];
        component.tensor_indices.push(index);
        if let Some(layer) = layer {
            component.layer_count = component.layer_count.max(layer + 1);
        }
    }

    components
}

/// The stack a stated layer count is taken to describe: the largest one.
///
/// A `config.json` states one layer count for the model even when the model
/// holds several stacks, and no convention says which stack it belongs to.
/// The largest is the only defensible reading — it is the model's language
/// backbone in every multi-component checkpoint seen — and it is compressr's
/// own choice, reported as such. Ties go to the stack that appears first.
///
/// Returns `None` when no component carries a layer index.
pub fn principal_component(components: &[Component]) -> Option<usize> {
    components
        .iter()
        .enumerate()
        .filter(|(_, component)| component.layer_count > 0)
        .max_by_key(|(position, component)| (component.layer_count, std::cmp::Reverse(*position)))
        .map(|(position, _)| position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::extract_layer_index;

    /// The prefix this module reads and the index the loader reads must come
    /// from the same segment, or a tensor would be counted into one stack and
    /// indexed as if it were in another.
    #[test]
    fn the_split_agrees_with_the_loader_index() {
        for name in [
            "model.layers.7.mlp.down_proj.weight",
            "base_lm.layers.27.self_attn.v_proj.weight",
            "residual_lm.layers.0.mlp.down_proj.weight",
            "blk.5.attn_q.weight",
            "transformer.h.11.attn.c_attn.weight",
            "lm_head.weight",
        ] {
            assert_eq!(
                split_component(name).map(|(_, index)| index),
                extract_layer_index(name),
                "{name}"
            );
        }
    }

    #[test]
    fn the_prefix_is_the_path_up_to_the_layer_marker() {
        assert_eq!(
            split_component("base_lm.layers.27.mlp.down_proj.weight"),
            Some(("base_lm.layers", 27))
        );
        assert_eq!(
            split_component("feat_encoder.encoder.layers.3.mlp.down_proj.weight"),
            Some(("feat_encoder.encoder.layers", 3))
        );
        assert_eq!(split_component("blk.5.attn_q.weight"), Some(("blk", 5)));
        assert_eq!(split_component("lm_head.weight"), None);
        // A number that follows no layer marker is not a layer index.
        assert_eq!(split_component("experts.3.w1.weight"), None);
    }

    /// The stacks of a multi-component checkpoint each carry their own layer
    /// range starting at 0, and each must be counted on its own.
    #[test]
    fn a_multi_component_checkpoint_splits_by_stack() {
        let names = vec![
            "base_lm.embed_tokens.weight",
            "base_lm.layers.0.mlp.down_proj.weight",
            "base_lm.layers.27.mlp.down_proj.weight",
            "residual_lm.layers.0.mlp.down_proj.weight",
            "residual_lm.layers.7.mlp.down_proj.weight",
            "feat_encoder.encoder.layers.11.mlp.down_proj.weight",
        ];
        let components = partition_components(&names);

        assert_eq!(components.len(), 4);
        assert_eq!(components[0].name, UNLAYERED_COMPONENT);
        assert_eq!(components[0].layer_count, 0);
        assert_eq!(components[1].name, "base_lm.layers");
        assert_eq!(components[1].layer_count, 28);
        assert_eq!(components[2].name, "residual_lm.layers");
        assert_eq!(components[2].layer_count, 8);
        assert_eq!(components[3].layer_count, 12);

        // The largest stack is the one a stated layer count is read against.
        assert_eq!(principal_component(&components), Some(1));
    }

    /// A single-stack checkpoint is one component, so the per-component walk
    /// reduces to exactly the upstream walk.
    #[test]
    fn a_single_stack_checkpoint_is_one_component() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.31.mlp.down_proj.weight",
            "lm_head.weight",
        ];
        let components = partition_components(&names);

        assert_eq!(components.len(), 2);
        assert_eq!(components[0].name, UNLAYERED_COMPONENT);
        assert_eq!(components[0].tensor_indices, vec![0, 3]);
        assert_eq!(components[1].name, "model.layers");
        assert_eq!(components[1].layer_count, 32);
        assert_eq!(principal_component(&components), Some(1));
    }

    #[test]
    fn a_checkpoint_with_no_layer_index_has_no_principal_stack() {
        let names = vec!["lm_head.weight", "model.embed_tokens.weight"];
        let components = partition_components(&names);
        assert_eq!(components.len(), 1);
        assert_eq!(principal_component(&components), None);
    }
}
