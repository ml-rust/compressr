//! Walking a checkpoint's tensors and assigning each one its mixed-precision format.
//!
//! Two of llama.cpp's rules read a RUNNING COUNTER rather than a layer number,
//! so the order the tensors are visited in decides which of them are promoted.
//! Upstream walks one map ordered by transformer block and then by name; this
//! module reproduces that walk, once per transformer stack, and records every
//! number the rules were given so the result can be audited by hand.
//!
//! [`super::mix_components`] explains why the walk is per stack and what that
//! costs in fidelity to upstream.

use crate::quantization::QuantType;
use std::collections::BTreeMap;

use super::config_facts::LayerCountSource;
use super::gguf_mix::GgufMix;
use super::mix_allocation::{ComponentAllocation, MixAllocation};
use super::mix_components::{
    partition_components, principal_component, split_component, Component, UNLAYERED_COMPONENT,
};
use super::mix_model::MixModel;
use super::mix_report::quant_type_label;
use super::mix_rules::{step_down_unquantizable, tensor_type, MixCounters};
use super::tensor_role::{classify, is_quantizable_role, TensorRole};

/// Assign one GGUF block format per tensor, or `None` to leave it unquantized.
///
/// The returned vector is in the caller's order. See [`assign_mix_reported`]
/// for the walk and for the allocation record this discards.
pub fn assign_mix<'a>(
    mix: GgufMix,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    model: &MixModel,
) -> Vec<Option<QuantType>> {
    assign_mix_reported(mix, tensors, model).0
}

/// Assign one GGUF block format per tensor, and record how it was decided.
///
/// The returned vector is in the caller's order. Internally the tensors are
/// walked PER TRANSFORMER STACK — see [`super::mix_components`] — because two
/// rules read a running counter rather than a layer number, and a walk that ran
/// those counters across several stacks would compare each stack's layer index
/// against a denominator that is not its own size.
///
/// Within one stack the walk is llama.cpp's own order, by block and then by
/// name, so a single-stack checkpoint takes exactly upstream's allocation.
///
/// The [`MixAllocation`] carries every number the rules were given and where it
/// came from, and states plainly whether the result is upstream's allocation or
/// an approximation of it.
pub fn assign_mix_reported<'a>(
    mix: GgufMix,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    model: &MixModel,
) -> (Vec<Option<QuantType>>, MixAllocation) {
    let tensors: Vec<(&str, &[usize])> = tensors.into_iter().collect();
    let roles: Vec<TensorRole> = tensors.iter().map(|(name, _)| classify(name)).collect();
    let names: Vec<&str> = tensors.iter().map(|(name, _)| *name).collect();

    let components = partition_components(&names);
    let principal = principal_component(&components);
    // A `config.json` states one layer count, and [`principal_component`]
    // documents which stack it is taken to describe. A count that was itself
    // inferred from the tensor names carries no more authority than the
    // per-stack counts, so it never overrides one.
    let stated_count = matches!(
        model.layer_count_source,
        LayerCountSource::Metadata | LayerCountSource::Config(_)
    );

    // Whether the model has an output weight distinct from its embedding is a
    // property of the whole model, not of one stack.
    let has_output = roles.contains(&TensorRole::Output);

    let mut assigned = vec![None; tensors.len()];
    let mut rows = Vec::with_capacity(components.len());

    for (position, component) in components.iter().enumerate() {
        let is_principal = principal == Some(position) && stated_count;
        let (n_ffn_down, layer_count_source) =
            component_layer_count(component, &roles, model, is_principal);

        // Upstream counts a fused QKV and a latent key-value projection towards
        // the attention-V denominator while only attention-V itself advances
        // the numerator; that is reproduced rather than corrected.
        let n_attention_wv = component
            .tensor_indices
            .iter()
            .filter(|&&index| {
                matches!(
                    roles[index],
                    TensorRole::AttnV | TensorRole::AttnQkv | TensorRole::AttnKvB
                )
            })
            .count();

        let mut counters = MixCounters {
            n_attention_wv: n_attention_wv as i64,
            n_ffn_down: n_ffn_down as i64,
            has_output,
            ..MixCounters::default()
        };

        let mut order = component.tensor_indices.clone();
        order.sort_by_key(|&index| {
            (
                split_component(names[index]).map_or(-1i64, |(_, layer)| layer as i64),
                names[index],
            )
        });

        let mut row = ComponentAllocation {
            name: component.name.clone(),
            layer_count: n_ffn_down,
            layer_count_source,
            attn_v_denominator: n_attention_wv,
            tensors: component.tensor_indices.len(),
            quantized: 0,
            promoted: 0,
            unquantized: 0,
            promoted_formats: BTreeMap::new(),
        };

        for index in order {
            let (name, shape) = tensors[index];
            let role = roles[index];
            if !is_quantizable_role(name, role, shape) {
                row.unquantized += 1;
                continue;
            }
            let new_type = tensor_type(mix, model, name, role, shape, &mut counters);
            let final_type = step_down_unquantizable(new_type, shape);
            assigned[index] = Some(final_type);
            row.quantized += 1;
            if final_type != mix.base() {
                row.promoted += 1;
                *row.promoted_formats
                    .entry(quant_type_label(final_type).to_string())
                    .or_insert(0) += 1;
            }
        }

        rows.push(row);
    }

    let allocation = MixAllocation {
        mix_name: mix.name().to_string(),
        base_format: quant_type_label(mix.base()).to_string(),
        model_layer_count: model.n_layer,
        layer_count_source: model.layer_count_source.clone(),
        n_expert: model.n_expert,
        components: rows,
    };

    (assigned, allocation)
}

/// The `n` one component's rules divide by, and where it came from.
///
/// - The principal stack takes the STATED count when there is one, because a
///   config that names a layer count names it for the model's own backbone.
/// - Every other stack takes its own highest layer index plus one. A config
///   carries no per-component layer counts under any convention compressr can
///   rely on, so this is compressr's own choice, not an upstream rule.
/// - The unlayered group is not a stack. It divides by the number of FFN-down
///   tensors it holds, which is zero in every checkpoint seen, so the rule
///   never runs there. That too is compressr's choice: upstream has no
///   unlayered FFN-down tensor.
fn component_layer_count(
    component: &Component,
    roles: &[TensorRole],
    model: &MixModel,
    is_principal: bool,
) -> (usize, String) {
    if component.name == UNLAYERED_COMPONENT {
        let ffn_down = component
            .tensor_indices
            .iter()
            .filter(|&&index| roles[index] == TensorRole::FfnDown)
            .count();
        return (
            ffn_down,
            "FFN-down tensors in this group (compressr's choice: not a stack)".to_string(),
        );
    }
    if is_principal {
        return (model.n_layer, model.layer_count_source.to_string());
    }
    (
        component.layer_count,
        "highest layer index in this stack, plus one (compressr's choice)".to_string(),
    )
}
