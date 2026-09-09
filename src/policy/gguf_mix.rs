//! llama.cpp's mixed-precision GGUF allocations, reproduced.
//!
//! `Q4_K_M` is not "Q4_K everywhere". It is a base format plus a table of
//! per-role exceptions: `Q6_K` on the output weight, `Q6_K` on the attention-V
//! and FFN-down tensors of selected layers, `Q5_K` on a fused QKV projection,
//! `Q8_0` on the attention tensors of an 8-expert model. The rules live in
//! `llama_tensor_get_type` and `llama_model_quantize_impl` in llama.cpp's
//! `src/llama-quant.cpp`; [`super::mix_rules`] holds the table and names every
//! place compressr's version approximates it.
//!
//! Q4_K_S, Q4_K_M, Q5_K_S and Q5_K_M draw only on Q4_K, Q5_K, Q6_K and Q8_0,
//! which boostr writes. Q2_K, Q2_K_S, Q3_K_S, Q3_K_M and Q3_K_L need a Q2_K or
//! Q3_K writer, and the IQ mixes need the I-quant grids; boostr reads those and
//! writes none of them, so [`parse_gguf_mix`] refuses them by name rather than
//! substituting a nearer format.
//!
//! Upstream matches GGUF tensor names; compressr writes the names its input
//! carried, so [`super::tensor_role`] maps both onto one role set.

use crate::loader::extract_layer_index;
use crate::quantization::QuantType;
use anyhow::{bail, Result};

use super::mix_model::MixModel;
use super::mix_rules::{step_down_unquantizable, tensor_type, MixCounters};
use super::tensor_role::{classify, is_quantizable_role, TensorRole};

/// A mixed-precision GGUF allocation, named as llama.cpp names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMix {
    /// Q4_K base, small variant
    Q4KS,
    /// Q4_K base, medium variant
    Q4KM,
    /// Q5_K base, small variant
    Q5KS,
    /// Q5_K base, medium variant
    Q5KM,
}

/// The mix names [`parse_gguf_mix`] accepts.
pub const GGUF_MIX_NAMES: &[&str] = &["q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m"];

impl GgufMix {
    /// The format every quantizable tensor takes unless a rule moves it.
    pub fn base(self) -> QuantType {
        match self {
            GgufMix::Q4KS | GgufMix::Q4KM => QuantType::Q4K,
            GgufMix::Q5KS | GgufMix::Q5KM => QuantType::Q5K,
        }
    }

    /// The name llama.cpp's `llama-quantize` accepts for this mix.
    pub fn name(self) -> &'static str {
        match self {
            GgufMix::Q4KS => "Q4_K_S",
            GgufMix::Q4KM => "Q4_K_M",
            GgufMix::Q5KS => "Q5_K_S",
            GgufMix::Q5KM => "Q5_K_M",
        }
    }

    /// The `llama_ftype` this mix writes to `general.file_type`.
    ///
    /// The values are llama.cpp's own enum, from `include/llama.h`. They are
    /// NOT GGML type ids: a reader takes the type of each tensor from the
    /// tensor directory, and reads this key only to name the file.
    pub fn llama_ftype(self) -> u32 {
        match self {
            GgufMix::Q4KS => 14,
            GgufMix::Q4KM => 15,
            GgufMix::Q5KS => 16,
            GgufMix::Q5KM => 17,
        }
    }
}

/// Resolve a `--quantization` value that names a mix.
///
/// One spelling per mix, the one `llama-quantize` accepts.
///
/// # Errors
/// The name is a real llama.cpp mix whose base format has no writer in
/// boostr, or it is not a mix name at all.
pub fn parse_gguf_mix(name: &str) -> Result<GgufMix> {
    match name.to_ascii_lowercase().as_str() {
        "q4_k_s" => Ok(GgufMix::Q4KS),
        "q4_k_m" => Ok(GgufMix::Q4KM),
        "q5_k_s" => Ok(GgufMix::Q5KS),
        "q5_k_m" => Ok(GgufMix::Q5KM),
        "q2_k" | "q2_k_s" | "q3_k_s" | "q3_k_m" | "q3_k_l" => bail!(
            "Mix `{name}` is not implemented: its base format is Q2_K or Q3_K, and boostr \
             has a reader for those but no writer, so compressr cannot pack them. \
             Implemented mixes: {}.",
            GGUF_MIX_NAMES.join(", ")
        ),
        _ => bail!(
            "Unsupported mixed-precision name: {name}. Implemented mixes: {}.",
            GGUF_MIX_NAMES.join(", ")
        ),
    }
}

/// Assign one GGUF block format per tensor, or `None` to leave it unquantized.
///
/// The returned vector is in the caller's order. Internally the tensors are
/// walked in llama.cpp's own order — by transformer block, then by name —
/// because two rules read a running counter rather than a layer number, and a
/// different walk order would move which layers get the richer format.
pub fn assign_mix<'a>(
    mix: GgufMix,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    model: &MixModel,
) -> Vec<Option<QuantType>> {
    let tensors: Vec<(&str, &[usize])> = tensors.into_iter().collect();
    let roles: Vec<TensorRole> = tensors.iter().map(|(name, _)| classify(name)).collect();

    // The counting pre-pass runs over every tensor, before the quantize gate.
    // Upstream counts a fused QKV and a latent key-value projection towards the
    // attention-V denominator while only attention-V itself advances the
    // numerator; that is reproduced rather than corrected.
    let mut counters = MixCounters {
        n_attention_wv: roles
            .iter()
            .filter(|role| {
                matches!(
                    role,
                    TensorRole::AttnV | TensorRole::AttnQkv | TensorRole::AttnKvB
                )
            })
            .count() as i64,
        n_ffn_down: model.n_layer as i64,
        has_output: roles.contains(&TensorRole::Output),
        ..MixCounters::default()
    };

    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&a, &b| {
        let key = |i: usize| {
            (
                extract_layer_index(tensors[i].0).map_or(-1i64, |v| v as i64),
                tensors[i].0,
            )
        };
        key(a).cmp(&key(b))
    });

    let mut assigned = vec![None; tensors.len()];
    for index in order {
        let (name, shape) = tensors[index];
        let role = roles[index];
        if is_quantizable_role(name, role, shape) {
            let new_type = tensor_type(mix, model, name, role, shape, &mut counters);
            assigned[index] = Some(step_down_unquantizable(new_type, shape));
        }
    }
    assigned
}

#[cfg(test)]
mod tests {
    use super::super::mix_rules::use_more_bits;
    use super::*;

    fn dense(n_layer: usize) -> MixModel {
        MixModel {
            arch: "llama".into(),
            n_layer,
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

    /// A real llama.cpp mix with no writer here is refused by name, and the
    /// ftype values are llama.cpp's own enum rather than GGML type ids.
    #[test]
    fn names_and_ftypes_match_llama_cpp() {
        let err = parse_gguf_mix("q3_k_m").unwrap_err().to_string();
        assert!(err.contains("Q2_K or Q3_K"), "{err}");
        assert!(err.contains("q4_k_m"), "{err}");
        assert!(parse_gguf_mix("q4_k_mmm").is_err());
        assert_eq!(parse_gguf_mix("Q4_K_M").unwrap(), GgufMix::Q4KM);

        assert_eq!(GgufMix::Q4KS.llama_ftype(), 14);
        assert_eq!(GgufMix::Q4KM.llama_ftype(), 15);
        assert_eq!(GgufMix::Q5KS.llama_ftype(), 16);
        assert_eq!(GgufMix::Q5KM.llama_ftype(), 17);
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
}
