//! What part of a transformer one tensor is, by name.
//!
//! # Two naming conventions, one set of roles
//!
//! llama.cpp's mixed-precision rules are written against GGUF tensor names:
//! `output.weight`, `blk.7.attn_v.weight`, `blk.7.ffn_down.weight`. compressr
//! writes the names its INPUT carried, so a SafeTensors checkpoint keeps
//! HuggingFace names: `lm_head.weight`, `model.layers.7.self_attn.v_proj.weight`,
//! `model.layers.7.mlp.down_proj.weight`.
//!
//! Both spellings map onto the same role here, so one rule table drives both
//! inputs. The GGUF spellings are matched exactly as llama.cpp matches them.
//! The HuggingFace spellings are compressr's own mapping onto those roles, not
//! an upstream rule: a checkpoint that names its projections something else
//! lands in [`TensorRole::Other`] and takes the mix's base format.

/// The role a mixed-precision rule keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// The un-embedding / LM head
    Output,
    /// The token embedding table
    TokenEmbd,
    /// Attention value projection
    AttnV,
    /// Attention key projection
    AttnK,
    /// Attention query projection
    AttnQ,
    /// Fused query-key-value projection
    AttnQkv,
    /// Compressed key-value projection of multi-head latent attention
    AttnKvB,
    /// Attention output projection
    AttnOutput,
    /// FFN down projection (the one that reads the wide hidden state)
    FfnDown,
    /// FFN gate projection
    FfnGate,
    /// FFN up projection
    FfnUp,
    /// Mixture-of-experts router
    ///
    /// llama.cpp never quantizes this: a rounded router changes which experts
    /// a token is sent to, which no dequantization recovers.
    FfnGateInp,
    /// A normalization weight
    Norm,
    /// Anything else
    Other,
}

/// Substrings llama.cpp excludes from quantization outright, whatever their
/// shape. Reproduced from `llama_model_quantize_impl`'s `quantize &= ...`
/// chain: small 2D weights of Mamba, RWKV, T5 and the multimodal projectors,
/// which are cheap to keep exact and expensive to round.
///
/// The RWKV entries are the exact names upstream lists. `time_mix_key.weight`
/// is absent from that list and IS quantized, so this must stay a list of
/// whole names rather than a `time_mix_` prefix test.
pub const NEVER_QUANTIZED_SUBSTRINGS: &[&str] = &[
    "altup",
    "laurel",
    "per_layer_model_proj",
    "position_embd",
    "token_types.weight",
    "ssm_conv1d.weight",
    "shortconv.conv.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
];

/// Classify one tensor name into the role a mix rule reads.
///
/// The order of the tests is load-bearing in two places: `ffn_gate_inp` is
/// tested before `ffn_gate`, because the router's name contains the gate's;
/// and `attn_kv_b` before `attn_k`, for the same reason.
pub fn classify(name: &str) -> TensorRole {
    let lower = name.to_ascii_lowercase();

    // GGUF spellings, matched the way llama.cpp matches them.
    if lower == "output.weight" {
        return TensorRole::Output;
    }
    if lower == "token_embd.weight" || lower == "per_layer_token_embd.weight" {
        return TensorRole::TokenEmbd;
    }
    if lower.contains("ffn_gate_inp") {
        return TensorRole::FfnGateInp;
    }
    if lower.contains("attn_kv_b.weight") {
        return TensorRole::AttnKvB;
    }
    if lower.contains("attn_qkv.weight") {
        return TensorRole::AttnQkv;
    }
    if lower.contains("attn_v.weight") {
        return TensorRole::AttnV;
    }
    if lower.contains("attn_k.weight") {
        return TensorRole::AttnK;
    }
    if lower.contains("attn_q.weight") {
        return TensorRole::AttnQ;
    }
    if lower.contains("attn_output.weight") {
        return TensorRole::AttnOutput;
    }
    if lower.contains("ffn_down") {
        return TensorRole::FfnDown;
    }
    if lower.contains("ffn_gate") {
        return TensorRole::FfnGate;
    }
    if lower.contains("ffn_up") {
        return TensorRole::FfnUp;
    }
    if lower.contains("_norm.weight") {
        return TensorRole::Norm;
    }

    classify_huggingface(&lower)
}

/// compressr's mapping from HuggingFace projection names onto the same roles.
fn classify_huggingface(lower: &str) -> TensorRole {
    if lower == "lm_head.weight" || lower.ends_with(".lm_head.weight") {
        return TensorRole::Output;
    }
    if lower.contains("embed_tokens.weight") || lower.contains("wte.weight") {
        return TensorRole::TokenEmbd;
    }
    // The MoE router: `mlp.gate.weight` (Qwen, DeepSeek) and
    // `block_sparse_moe.gate.weight` (Mixtral). Tested before the gate
    // projection, whose name is `gate_proj`.
    if lower.ends_with("mlp.gate.weight") || lower.ends_with("block_sparse_moe.gate.weight") {
        return TensorRole::FfnGateInp;
    }
    if lower.contains("qkv_proj.weight") || lower.contains("wqkv.weight") {
        return TensorRole::AttnQkv;
    }
    if lower.contains("kv_b_proj.weight") {
        return TensorRole::AttnKvB;
    }
    if lower.contains("v_proj.weight") {
        return TensorRole::AttnV;
    }
    if lower.contains("k_proj.weight") {
        return TensorRole::AttnK;
    }
    if lower.contains("q_proj.weight") {
        return TensorRole::AttnQ;
    }
    if lower.contains("o_proj.weight") || lower.contains("out_proj.weight") {
        return TensorRole::AttnOutput;
    }
    if lower.contains("down_proj.weight") || lower.contains("w2.weight") {
        return TensorRole::FfnDown;
    }
    if lower.contains("gate_proj.weight") || lower.contains("w1.weight") {
        return TensorRole::FfnGate;
    }
    if lower.contains("up_proj.weight") || lower.contains("w3.weight") {
        return TensorRole::FfnUp;
    }
    if lower.contains("norm.weight") {
        return TensorRole::Norm;
    }
    TensorRole::Other
}

/// Whether a mix quantizes this tensor at all.
///
/// Reproduces llama.cpp's `quantize &= ...` chain: the name ends with
/// `weight`, the tensor has at least two dimensions, and it is neither a norm,
/// a router, nor one of [`NEVER_QUANTIZED_SUBSTRINGS`].
///
/// Block alignment is NOT checked here. That is a property of the assigned
/// format rather than of the role, so the caller applies it after the rules
/// have picked a format.
pub fn is_quantizable_role(name: &str, role: TensorRole, shape: &[usize]) -> bool {
    if !name.ends_with("weight") {
        return false;
    }
    if shape.len() < 2 {
        return false;
    }
    if matches!(role, TensorRole::Norm | TensorRole::FfnGateInp) {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    !NEVER_QUANTIZED_SUBSTRINGS
        .iter()
        .any(|excluded| lower.contains(excluded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_names_classify() {
        assert_eq!(classify("output.weight"), TensorRole::Output);
        assert_eq!(classify("token_embd.weight"), TensorRole::TokenEmbd);
        assert_eq!(classify("blk.7.attn_v.weight"), TensorRole::AttnV);
        assert_eq!(classify("blk.7.attn_k.weight"), TensorRole::AttnK);
        assert_eq!(classify("blk.7.attn_q.weight"), TensorRole::AttnQ);
        assert_eq!(classify("blk.7.attn_qkv.weight"), TensorRole::AttnQkv);
        assert_eq!(classify("blk.7.attn_output.weight"), TensorRole::AttnOutput);
        assert_eq!(classify("blk.7.ffn_down.weight"), TensorRole::FfnDown);
        assert_eq!(classify("blk.7.ffn_down_exps.weight"), TensorRole::FfnDown);
        assert_eq!(classify("blk.7.ffn_gate.weight"), TensorRole::FfnGate);
        assert_eq!(classify("blk.7.ffn_up.weight"), TensorRole::FfnUp);
        assert_eq!(classify("blk.7.attn_norm.weight"), TensorRole::Norm);
    }

    /// The router's name contains the gate projection's, so the order of the
    /// two tests decides whether a MoE model's routing weights get rounded.
    #[test]
    fn the_router_is_not_the_gate_projection() {
        assert_eq!(
            classify("blk.3.ffn_gate_inp.weight"),
            TensorRole::FfnGateInp
        );
        assert_eq!(
            classify("model.layers.3.mlp.gate.weight"),
            TensorRole::FfnGateInp
        );
        assert_eq!(
            classify("model.layers.3.mlp.gate_proj.weight"),
            TensorRole::FfnGate
        );
    }

    /// `attn_kv_b` contains `attn_k`, and the two take different rules.
    #[test]
    fn latent_attention_is_not_the_key_projection() {
        assert_eq!(classify("blk.1.attn_kv_b.weight"), TensorRole::AttnKvB);
        assert_eq!(
            classify("model.layers.1.self_attn.kv_b_proj.weight"),
            TensorRole::AttnKvB
        );
    }

    #[test]
    fn huggingface_names_classify() {
        assert_eq!(classify("lm_head.weight"), TensorRole::Output);
        assert_eq!(classify("model.embed_tokens.weight"), TensorRole::TokenEmbd);
        assert_eq!(
            classify("model.layers.7.self_attn.v_proj.weight"),
            TensorRole::AttnV
        );
        assert_eq!(
            classify("model.layers.7.self_attn.o_proj.weight"),
            TensorRole::AttnOutput
        );
        assert_eq!(
            classify("model.layers.7.mlp.down_proj.weight"),
            TensorRole::FfnDown
        );
        assert_eq!(
            classify("model.layers.7.input_layernorm.weight"),
            TensorRole::Norm
        );
    }

    /// A norm is rank 1 in most checkpoints, but the role test must refuse it
    /// even where it is not: llama.cpp excludes it by name, not by shape.
    #[test]
    fn norms_and_routers_are_never_quantized() {
        assert!(!is_quantizable_role(
            "blk.0.attn_norm.weight",
            TensorRole::Norm,
            &[4096, 4096]
        ));
        assert!(!is_quantizable_role(
            "blk.0.ffn_gate_inp.weight",
            TensorRole::FfnGateInp,
            &[8, 4096]
        ));
    }

    #[test]
    fn biases_and_rank_one_buffers_are_never_quantized() {
        assert!(!is_quantizable_role(
            "blk.0.attn_q.bias",
            TensorRole::AttnQ,
            &[4096]
        ));
        assert!(!is_quantizable_role(
            "blk.0.attn_q.weight",
            TensorRole::AttnQ,
            &[4096]
        ));
        assert!(is_quantizable_role(
            "blk.0.attn_q.weight",
            TensorRole::AttnQ,
            &[4096, 4096]
        ));
    }

    /// `time_mix_key.weight` is absent from upstream's exclusion list, so a
    /// prefix test on `time_mix_` would wrongly keep it unquantized.
    #[test]
    fn the_rwkv_exclusions_are_whole_names() {
        assert!(!is_quantizable_role(
            "blk.0.time_mix_w1.weight",
            TensorRole::Other,
            &[64, 4096]
        ));
        assert!(is_quantizable_role(
            "blk.0.time_mix_key.weight",
            TensorRole::Other,
            &[4096, 4096]
        ));
    }
}
