//! Tensor names and row orders for a GGUF written from a HuggingFace checkpoint.
//!
//! llama.cpp finds tensors by its own names (`blk.0.attn_q.weight`), so a
//! file that keeps HuggingFace names loads in boostr and nowhere else. For an
//! architecture in [`RENAMED_ARCHS`], every tensor takes llama.cpp's name
//! from boostr's shared name table. The `llama` architecture also stores
//! query and key rows in llama.cpp's pair order (see
//! `boostr::format::gguf::rope_rows`). A whole row moves at a time, so the
//! reorder runs on the written bytes of any block format and changes no value.
//!
//! The rename is all or nothing. One tensor outside the table leaves every
//! name as the checkpoint has it, because a file with half its names
//! translated loads nowhere.

use super::arch::RENAMED_ARCHS;
use super::rope::ExtraTensor;
use anyhow::{bail, Result};
use boostr::format::gguf::{hf_to_gguf_name, rope_row_heads, rope_row_sources, RopeRowOrder};
use std::collections::HashMap;

/// How each checkpoint tensor is named and ordered in the GGUF output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufTensorLayout {
    renames: HashMap<String, String>,
    row_heads: HashMap<String, usize>,
    /// Tensors the converter adds, written before the checkpoint's own.
    pub extra: Vec<ExtraTensor>,
}

impl GgufTensorLayout {
    /// The layout for llama.cpp architecture `arch`.
    ///
    /// `n_head` and `n_head_kv` group the query and key rows. They are needed
    /// only for an architecture that reorders rows.
    ///
    /// # Errors
    /// The architecture reorders query/key rows and the head count is absent.
    pub fn build(
        arch: &str,
        names: &[String],
        n_head: Option<u32>,
        n_head_kv: Option<u32>,
        extra: Vec<ExtraTensor>,
    ) -> Result<Self> {
        let mut layout = Self {
            extra,
            ..Self::default()
        };
        if !RENAMED_ARCHS.contains(&arch) {
            return Ok(layout);
        }
        let mut renames = HashMap::with_capacity(names.len());
        for name in names {
            match hf_to_gguf_name(name) {
                Some(gguf) => {
                    renames.insert(name.clone(), gguf);
                }
                None => {
                    log::warn!(
                        "Tensor '{name}' has no llama.cpp name, so every tensor keeps its \
                         HuggingFace name. llama.cpp cannot load the output."
                    );
                    return Ok(layout);
                }
            }
        }
        for (name, gguf) in &renames {
            // HuggingFace's own default: no `num_key_value_heads` means one
            // key head per query head.
            let probe = rope_row_heads(arch, gguf, 1, 1);
            if probe.is_none() {
                continue;
            }
            let Some(n_head) = n_head else {
                bail!(
                    "config.json states no num_attention_heads, and '{name}' needs it: the \
                     {arch} architecture stores query and key rows grouped by head."
                );
            };
            let n_head_kv = n_head_kv.unwrap_or(n_head);
            if let Some(heads) = rope_row_heads(arch, gguf, n_head as usize, n_head_kv as usize) {
                layout.row_heads.insert(name.clone(), heads);
            }
        }
        layout.renames = renames;
        Ok(layout)
    }

    /// The GGUF name of checkpoint tensor `name`.
    pub fn output_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.renames.get(name).map_or(name, String::as_str)
    }

    /// The head count `name`'s rows are reordered by, when they are.
    pub fn row_heads(&self, name: &str) -> Option<usize> {
        self.row_heads.get(name).copied()
    }

    /// Whether the layout renames any tensor.
    pub fn renames_tensors(&self) -> bool {
        !self.renames.is_empty()
    }
}

/// `data`, the written bytes of a tensor with `rows` rows, with its rows in
/// llama.cpp's order for `heads` heads.
///
/// # Errors
/// The byte count is not a whole number of rows, or the rows do not split
/// into `heads` heads of an even size.
pub fn reorder_rows(name: &str, data: &[u8], rows: usize, heads: usize) -> Result<Vec<u8>> {
    if rows == 0 || !data.len().is_multiple_of(rows) {
        bail!(
            "Tensor '{name}' is {} bytes, which do not split into {rows} rows",
            data.len()
        );
    }
    let row_bytes = data.len() / rows;
    let sources = rope_row_sources(rows, heads, RopeRowOrder::ToGguf)
        .map_err(|e| anyhow::anyhow!("Tensor '{name}': {e}"))?;
    let mut out = Vec::with_capacity(data.len());
    for source in sources {
        out.extend_from_slice(&data[source * row_bytes..(source + 1) * row_bytes]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn llama_renames_every_tensor_and_reorders_query_and_key() {
        let layout = GgufTensorLayout::build(
            "llama",
            &names(&[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ]),
            Some(4),
            Some(2),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            layout.output_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
        assert_eq!(
            layout.output_name("model.layers.0.self_attn.k_proj.weight"),
            "blk.0.attn_k.weight"
        );
        assert_eq!(
            layout.row_heads("model.layers.0.self_attn.q_proj.weight"),
            Some(4)
        );
        assert_eq!(
            layout.row_heads("model.layers.0.self_attn.k_proj.weight"),
            Some(2)
        );
        assert_eq!(
            layout.row_heads("model.layers.0.self_attn.v_proj.weight"),
            None
        );
    }

    #[test]
    fn qwen2_renames_without_reordering() {
        let layout = GgufTensorLayout::build(
            "qwen2",
            &names(&["model.layers.1.self_attn.q_proj.bias"]),
            None,
            None,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            layout.output_name("model.layers.1.self_attn.q_proj.bias"),
            "blk.1.attn_q.bias"
        );
        assert_eq!(
            layout.row_heads("model.layers.1.self_attn.q_proj.bias"),
            None
        );
    }

    #[test]
    fn one_unknown_tensor_keeps_every_name() {
        let list = names(&["model.embed_tokens.weight", "model.rotary_emb.inv_freq"]);
        let layout = GgufTensorLayout::build("llama", &list, Some(4), None, Vec::new()).unwrap();
        assert!(!layout.renames_tensors());
        assert_eq!(
            layout.output_name("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        let gemma = GgufTensorLayout::build("gemma", &list[..1], None, None, Vec::new()).unwrap();
        assert!(!gemma.renames_tensors());
    }

    /// Two-byte rows tagged with their HF index come out in converter order.
    #[test]
    fn rows_move_whole() {
        let data: Vec<u8> = (0..8u8).flat_map(|r| [r, r]).collect();
        let out = reorder_rows("t", &data, 8, 2).unwrap();
        let order: Vec<u8> = out.chunks(2).map(|row| row[0]).collect();
        assert_eq!(order, [0, 2, 1, 3, 4, 6, 5, 7]);
        assert!(reorder_rows("t", &data[..15], 8, 2).is_err());
    }
}
