//! llama.cpp's per-tensor mix table, reproduced tensor by tensor.
//!
//! Every case here is a branch of `llama_tensor_get_type` in llama.cpp's
//! `src/llama-quant.cpp`. They pin the allocation a SINGLE-STACK checkpoint
//! gets, which is the shape upstream wrote the rules for and the only shape
//! compressr claims upstream's allocation for. The multi-stack approximation
//! is pinned separately, in `gguf_mix_components.rs`.

use compressr::{
    assign_mix, extract_layer_index, use_more_bits, GgufMix, LayerCountSource, MixModel, QuantType,
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

fn per_layer(pattern: &str, n_layer: usize) -> Vec<String> {
    (0..n_layer)
        .map(|i| pattern.replace("{}", &i.to_string()))
        .collect()
}

fn as_tensors<'a>(names: &'a [String], shape: &'a [usize]) -> Vec<(&'a str, &'a [usize])> {
    names.iter().map(|n| (n.as_str(), shape)).collect()
}

/// The output weight takes Q6_K, which is the headline of every `_M` mix.
#[test]
fn the_output_weight_takes_q6k() {
    let shape = [32000usize, 4096];
    let tensors: Vec<(&str, &[usize])> = vec![("output.weight", &shape)];
    assert_eq!(
        assign_mix(GgufMix::Q4KM, tensors, &dense(32))[0],
        Some(QuantType::Q6K)
    );
}

/// With no separate output weight, the embedding table IS the un-embedding.
/// An unquantizable output row takes Q8_0 before any step-down runs.
#[test]
fn a_tied_embedding_takes_the_output_rule() {
    let shape = [32000usize, 4096];
    let tied: Vec<(&str, &[usize])> = vec![("token_embd.weight", &shape)];
    assert_eq!(
        assign_mix(GgufMix::Q4KM, tied, &dense(32))[0],
        Some(QuantType::Q6K)
    );

    let untied: Vec<(&str, &[usize])> =
        vec![("token_embd.weight", &shape), ("output.weight", &shape)];
    let assigned = assign_mix(GgufMix::Q4KM, untied, &dense(32));
    assert_eq!(assigned[0], Some(QuantType::Q4K));
    assert_eq!(assigned[1], Some(QuantType::Q6K));

    let unaligned = [32000usize, 96];
    let short: Vec<(&str, &[usize])> = vec![("output.weight", &unaligned)];
    assert_eq!(
        assign_mix(GgufMix::Q4KM, short, &dense(32))[0],
        Some(QuantType::Q8_0)
    );
}

/// Q4_K_M bumps FFN-down on the layers `use_more_bits` selects.
#[test]
fn q4_k_m_bumps_ffn_down_by_layer() {
    let names = per_layer("blk.{}.ffn_down.weight", 32);
    let shape = [4096usize, 11008];
    let assigned = assign_mix(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(32));

    assert_eq!(assigned[0], Some(QuantType::Q6K));
    assert_eq!(assigned[31], Some(QuantType::Q6K));
    assert_eq!(assigned[5], Some(QuantType::Q4K));
    assert_eq!(assigned[6], Some(QuantType::Q6K));
}

/// Q4_K_S raises the first eighth of FFN-down to Q5_K, not Q6_K.
#[test]
fn q4_k_s_raises_only_the_first_eighth_of_ffn_down() {
    let names = per_layer("blk.{}.ffn_down.weight", 32);
    let shape = [4096usize, 11008];
    let assigned = assign_mix(GgufMix::Q4KS, as_tensors(&names, &shape), &dense(32));

    assert_eq!(assigned[3], Some(QuantType::Q5K));
    assert_eq!(assigned[4], Some(QuantType::Q4K));
    assert_eq!(assigned[31], Some(QuantType::Q4K));
}

/// The counter that decides an FFN-down layer must advance in layer order,
/// not in the order the caller happens to list the tensors. Names sorted
/// as plain text put layer 10 before layer 2.
#[test]
fn the_layer_counter_follows_block_order_not_string_order() {
    let mut names = per_layer("blk.{}.ffn_down.weight", 32);
    names.sort();
    let shape = [4096usize, 11008];
    let assigned = assign_mix(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(32));

    for (name, format) in names.iter().zip(&assigned) {
        let layer = extract_layer_index(name).expect("a block index");
        let expected = if use_more_bits(layer as i64, 32) {
            QuantType::Q6K
        } else {
            QuantType::Q4K
        };
        assert_eq!(*format, Some(expected), "{name}");
    }
}

/// Q4_K_S raises the first four attention-V tensors; `_M` uses the layer
/// formula and reaches Q6_K.
#[test]
fn attention_v_follows_the_mix() {
    let names = per_layer("blk.{}.attn_v.weight", 32);
    let shape = [4096usize, 4096];

    let small = assign_mix(GgufMix::Q4KS, as_tensors(&names, &shape), &dense(32));
    assert_eq!(small[3], Some(QuantType::Q5K));
    assert_eq!(small[4], Some(QuantType::Q4K));

    let medium = assign_mix(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(32));
    assert_eq!(medium[0], Some(QuantType::Q6K));
    assert_eq!(medium[5], Some(QuantType::Q4K));
}

/// The 70B shape raises attention-V on layers no other rule raised.
#[test]
fn the_70b_shape_raises_attention_v() {
    let names = per_layer("blk.{}.attn_v.weight", 80);
    let shape = [8192usize, 1024];
    let model = MixModel {
        is_70b: true,
        ..dense(80)
    };
    let assigned = assign_mix(GgufMix::Q4KM, as_tensors(&names, &shape), &model);

    // Layer 10 is not in the `use_more_bits` set for 80 layers, so only
    // the 70B rule can raise it.
    assert!(!use_more_bits(10, 80));
    assert_eq!(assigned[10], Some(QuantType::Q5K));
    assert_eq!(
        assign_mix(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(80))[10],
        Some(QuantType::Q4K)
    );
}

/// An 8-expert model puts Q8_0 on attention-K and -V, Q5_K on the output.
#[test]
fn an_eight_expert_model_takes_the_moe_rules() {
    let model = MixModel {
        n_expert: 8,
        ..dense(32)
    };
    let shape = [4096usize, 4096];
    let tensors: Vec<(&str, &[usize])> = vec![
        ("blk.0.attn_v.weight", &shape),
        ("blk.0.attn_k.weight", &shape),
        ("blk.0.attn_output.weight", &shape),
    ];
    let assigned = assign_mix(GgufMix::Q4KM, tensors, &model);

    assert_eq!(assigned[0], Some(QuantType::Q8_0));
    assert_eq!(assigned[1], Some(QuantType::Q8_0));
    assert_eq!(assigned[2], Some(QuantType::Q5K));
}

/// The router decides which experts a token reaches, so no mix rounds it.
#[test]
fn the_router_and_the_norms_stay_unquantized() {
    let router = [8usize, 4096];
    let norm = [4096usize];
    let tensors: Vec<(&str, &[usize])> = vec![
        ("blk.0.ffn_gate_inp.weight", &router),
        ("blk.0.attn_norm.weight", &norm),
    ];
    assert_eq!(
        assign_mix(GgufMix::Q4KM, tensors, &dense(32)),
        vec![None, None]
    );
}

/// A fused QKV projection gets a step more than the base in both `_M` mixes.
#[test]
fn a_fused_qkv_projection_gets_a_step_more() {
    let shape = [12288usize, 4096];
    let tensors: Vec<(&str, &[usize])> = vec![("blk.0.attn_qkv.weight", &shape)];
    assert_eq!(
        assign_mix(GgufMix::Q4KM, tensors.clone(), &dense(32))[0],
        Some(QuantType::Q5K)
    );
    assert_eq!(
        assign_mix(GgufMix::Q5KM, tensors, &dense(32))[0],
        Some(QuantType::Q6K)
    );
}

/// HuggingFace names take the same rules as their GGUF spellings.
#[test]
fn huggingface_names_get_the_same_allocation() {
    let names = per_layer("model.layers.{}.mlp.down_proj.weight", 32);
    let shape = [11008usize, 4096];
    let assigned = assign_mix(GgufMix::Q4KM, as_tensors(&names, &shape), &dense(32));

    assert_eq!(assigned[0], Some(QuantType::Q6K));
    assert_eq!(assigned[5], Some(QuantType::Q4K));
}

/// Q5_K_S moves nothing but the output weight.
#[test]
fn q5_k_s_moves_only_the_output_weight() {
    let shape = [4096usize, 4096];
    let embd = [32000usize, 4096];
    let tensors: Vec<(&str, &[usize])> = vec![
        ("output.weight", &embd),
        ("blk.0.attn_v.weight", &shape),
        ("blk.0.ffn_down.weight", &shape),
        ("blk.0.attn_output.weight", &shape),
    ];
    let assigned = assign_mix(GgufMix::Q5KS, tensors, &dense(32));

    assert_eq!(assigned[0], Some(QuantType::Q6K));
    assert_eq!(assigned[1], Some(QuantType::Q5K));
    assert_eq!(assigned[2], Some(QuantType::Q5K));
    assert_eq!(assigned[3], Some(QuantType::Q5K));
}
