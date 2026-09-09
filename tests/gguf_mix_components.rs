//! The allocation a multi-component checkpoint gets, and how it is reported.
//!
//! llama.cpp's `llama_tensor_get_type` assumes ONE transformer stack of
//! `n_layer` blocks indexed `0..n_layer`. A checkpoint built from several
//! stacks — a language model, a residual language model, an audio encoder and
//! a decoder, each numbering its own layers from 0 — breaks that assumption in
//! a way no error reports: the running counters run past the denominator and
//! the promotion pattern lands on tensors upstream's rule does not select.
//!
//! These tests pin the walk compressr does instead, and pin the report that
//! states it. The report is the point: a baseline whose allocation cannot be
//! audited by hand is not a baseline.

use compressr::{
    assign_mix_reported, extract_layer_index, use_more_bits, GgufMix, LayerCountSource, MixModel,
    QuantType,
};

fn dense(n_layer: usize) -> MixModel {
    MixModel {
        arch: "llama".into(),
        n_layer,
        layer_count_source: LayerCountSource::Config("num_hidden_layers".into()),
        n_expert: 0,
        is_70b: false,
    }
}

fn voxcpm_like() -> MixModel {
    MixModel {
        arch: "voxcpm2".into(),
        n_layer: 28,
        layer_count_source: LayerCountSource::Config("lm_config.num_hidden_layers".into()),
        n_expert: 0,
        is_70b: false,
    }
}

fn per_layer(pattern: &str, n_layer: usize) -> Vec<String> {
    (0..n_layer)
        .map(|i| pattern.replace("{}", &i.to_string()))
        .collect()
}

fn as_tensors<'a>(names: &'a [String], shape: &'a [usize]) -> Vec<(&'a str, &'a [usize])> {
    names.iter().map(|n| (n.as_str(), shape)).collect()
}

/// A single-stack checkpoint's allocation IS upstream's, and the report
/// says so rather than hedging.
#[test]
fn a_single_stack_reports_the_upstream_allocation() {
    let names = per_layer("model.layers.{}.mlp.down_proj.weight", 32);
    let shape = [11008usize, 4096];
    let (assigned, allocation) =
        assign_mix_reported(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(32));

    assert_eq!(assigned[0], Some(QuantType::Q6K));
    assert!(!allocation.is_multi_component());
    assert!(allocation.fidelity().contains("this is llama.cpp's Q4_K_M"));
    assert_eq!(allocation.components.len(), 1);
    assert_eq!(allocation.components[0].name, "model.layers");
    assert_eq!(allocation.components[0].layer_count, 32);
    assert_eq!(
        allocation.components[0].layer_count_source,
        "config.json:num_hidden_layers"
    );
}

/// The defect this walk exists for: several stacks share one counter, so
/// the FFN-down numerator runs past the denominator and everything after
/// the seven-eighths point of the FIRST stack is promoted.
///
/// Per stack, each stack's own layer count is the denominator, so the
/// smaller stack keeps the same first-eighth / last-eighth shape the rule
/// means instead of being promoted whole.
#[test]
fn each_stack_divides_by_its_own_layer_count() {
    let mut names = per_layer("base_lm.layers.{}.mlp.down_proj.weight", 28);
    names.extend(per_layer("residual_lm.layers.{}.mlp.down_proj.weight", 8));
    let shape = [6144usize, 2048];

    let model = voxcpm_like();
    let (assigned, allocation) =
        assign_mix_reported(GgufMix::Q4KM, as_tensors(&names, &shape), &model);

    // The base stack takes exactly the allocation it would take alone.
    for (position, name) in names.iter().enumerate().take(28) {
        let layer = extract_layer_index(name).expect("a layer index");
        let expected = if use_more_bits(layer as i64, 28) {
            QuantType::Q6K
        } else {
            QuantType::Q4K
        };
        assert_eq!(assigned[position], Some(expected), "{name}");
    }
    // The residual stack divides by 8, not by 28 and not by 36.
    for (offset, name) in names.iter().enumerate().skip(28) {
        let layer = extract_layer_index(name).expect("a layer index");
        let expected = if use_more_bits(layer as i64, 8) {
            QuantType::Q6K
        } else {
            QuantType::Q4K
        };
        assert_eq!(assigned[offset], Some(expected), "{name}");
    }
    // Layer 4 of an 8-layer stack is in neither eighth and is not the
    // third of the middle run, so a shared counter promoting it would be
    // the defect showing.
    assert!(!use_more_bits(4, 8));
    assert_eq!(assigned[32], Some(QuantType::Q4K));

    assert!(allocation.is_multi_component());
    assert!(allocation.fidelity().contains("APPROXIMATION"));
}

/// Every stack's denominator and its source appear in the report, and the
/// stated count is applied to the principal stack alone.
#[test]
fn the_allocation_report_names_each_stack_denominator() {
    let mut names = per_layer("base_lm.layers.{}.mlp.down_proj.weight", 28);
    names.extend(per_layer("residual_lm.layers.{}.mlp.down_proj.weight", 8));
    names.push("base_lm.embed_tokens.weight".to_string());
    let shape = [6144usize, 2048];

    let model = voxcpm_like();
    let (_, allocation) = assign_mix_reported(GgufMix::Q4KM, as_tensors(&names, &shape), &model);

    let base = allocation
        .components
        .iter()
        .find(|c| c.name == "base_lm.layers")
        .expect("the base stack");
    assert_eq!(base.layer_count, 28);
    assert_eq!(
        base.layer_count_source,
        "config.json:lm_config.num_hidden_layers"
    );

    let residual = allocation
        .components
        .iter()
        .find(|c| c.name == "residual_lm.layers")
        .expect("the residual stack");
    assert_eq!(residual.layer_count, 8);
    assert!(
        residual.layer_count_source.contains("compressr's choice"),
        "{}",
        residual.layer_count_source
    );

    // The unlayered group is a row, but it is not a stack and divides by
    // no layer count.
    let unlayered = allocation
        .components
        .iter()
        .find(|c| c.name == "(unlayered)")
        .expect("the unlayered group");
    assert_eq!(unlayered.layer_count, 0);
    assert_eq!(allocation.layered_components().count(), 2);

    let text = allocation.to_string();
    assert!(text.contains("n for base_lm.layers: 28"), "{text}");
    assert!(text.contains("n for residual_lm.layers: 8"), "{text}");
}

/// A count that was itself inferred from the tensor names carries no more
/// authority than the per-stack counts, so it never overrides one.
#[test]
fn an_inferred_count_never_overrides_a_stack_count() {
    let mut names = per_layer("base_lm.layers.{}.mlp.down_proj.weight", 28);
    names.extend(per_layer("residual_lm.layers.{}.mlp.down_proj.weight", 8));
    let shape = [6144usize, 2048];

    let model = MixModel {
        layer_count_source: LayerCountSource::InferredFromTensorNames,
        ..voxcpm_like()
    };
    let (_, allocation) = assign_mix_reported(GgufMix::Q4KM, as_tensors(&names, &shape), &model);

    for component in allocation.layered_components() {
        assert!(
            component.layer_count_source.contains("compressr's choice"),
            "{}",
            component.layer_count_source
        );
    }
}

/// The promoted count is per component, which is what a human audits.
#[test]
fn the_promoted_count_is_recorded_per_component() {
    let names = per_layer("model.layers.{}.mlp.down_proj.weight", 32);
    let shape = [11008usize, 4096];
    let (assigned, allocation) =
        assign_mix_reported(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(32));

    let promoted = assigned
        .iter()
        .filter(|format| **format == Some(QuantType::Q6K))
        .count();
    assert_eq!(allocation.components[0].promoted, promoted);
    assert_eq!(allocation.total_promoted(), promoted);
    assert_eq!(
        allocation.components[0].promoted_formats.get("Q6_K"),
        Some(&promoted)
    );
}
