//! A byte-level BPE vocabulary (`tokenizer.ggml.model = "gpt2"`) from
//! `tokenizer.json`.
//!
//! A port of `get_vocab_base` and of `SpecialVocab`'s merge reading in
//! llama.cpp's converter. The token list runs over every id below the
//! vocabulary size. An id with no token is `[PAD<id>]` of type UNUSED, so
//! the list length matches the embedding rows.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;

/// llama.cpp's `TokenType`.
pub mod token_type {
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

/// The `gpt2` token list, token types and merges.
#[derive(Debug, Clone, PartialEq)]
pub struct BpeVocab {
    /// Token text by id.
    pub tokens: Vec<String>,
    /// [`token_type`] by id.
    pub token_types: Vec<i32>,
    /// `"left right"` merges in priority order.
    pub merges: Vec<String>,
}

/// One entry of `tokenizer.json`'s `added_tokens`.
struct AddedToken {
    content: String,
    special: bool,
}

/// `ModelBase.does_token_look_special`: added tokens some models leave
/// unmarked although they are control tokens.
pub fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<｜") && token.ends_with("｜>"))
        || (token.starts_with("<unused") && token.ends_with('>'))
}

/// Read the `gpt2` vocabulary of a parsed `tokenizer.json`.
///
/// `vocab_size` is `config.json`'s, when stated. Without it the size is the
/// number of distinct ids.
///
/// # Errors
/// The file has no BPE `model.vocab`, a token id is not a whole number, an id
/// reaches `vocab_size`, or the merges are in neither known layout.
pub fn read_bpe_vocab(tokenizer: &Value, vocab_size: Option<u32>) -> Result<BpeVocab> {
    let model = tokenizer
        .get("model")
        .context("tokenizer.json has no 'model' block")?;
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .context("tokenizer.json has no 'model.vocab' map")?;
    let mut by_id: HashMap<u64, String> = HashMap::with_capacity(vocab.len());
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .with_context(|| format!("tokenizer.json gives '{token}' the id {id}"))?;
        by_id.insert(id, token.clone());
    }
    let mut added: HashMap<u64, AddedToken> = HashMap::new();
    for entry in tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let (Some(id), Some(content)) = (
            entry.get("id").and_then(Value::as_u64),
            entry.get("content").and_then(Value::as_str),
        ) else {
            continue;
        };
        let special = entry
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        by_id.insert(id, content.to_string());
        added.insert(
            id,
            AddedToken {
                content: content.to_string(),
                special,
            },
        );
    }
    let size = match vocab_size {
        Some(size) => u64::from(size),
        None => by_id.len() as u64,
    };
    if let Some(max) = by_id.keys().max().filter(|max| **max >= size) {
        bail!(
            "tokenizer.json holds token id {max}, and config.json gives vocab_size {size}. \
             Every id is below vocab_size."
        );
    }
    let added_contents: std::collections::HashSet<&str> =
        added.values().map(|t| t.content.as_str()).collect();

    let mut tokens = Vec::with_capacity(size as usize);
    let mut token_types = Vec::with_capacity(size as usize);
    for id in 0..size {
        let Some(token) = by_id.get(&id) else {
            tokens.push(format!("[PAD{id}]"));
            token_types.push(token_type::UNUSED);
            continue;
        };
        if added_contents.contains(token.as_str()) {
            let special = added.get(&id).is_some_and(|t| t.special);
            if special || looks_special(token) {
                tokens.push(token.clone());
                token_types.push(token_type::CONTROL);
            } else {
                // llama.cpp stores a user-defined token's spaces as spaces.
                tokens.push(token.replace('\u{2581}', " "));
                token_types.push(token_type::USER_DEFINED);
            }
        } else {
            tokens.push(token.clone());
            token_types.push(token_type::NORMAL);
        }
    }
    let merges = read_merges(model)?;
    Ok(BpeVocab {
        tokens,
        token_types,
        merges,
    })
}

/// `model.merges` as `"left right"` strings.
///
/// The pair layout (transformers 4.45 and later) stores a space inside a
/// merge part as U+0120, as llama.cpp does.
fn read_merges(model: &Value) -> Result<Vec<String>> {
    let Some(merges) = model.get("merges").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    match merges.first() {
        None => Ok(Vec::new()),
        Some(Value::String(_)) => Ok(merges
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()),
        Some(Value::Array(_)) => merges
            .iter()
            .map(|pair| {
                let parts = pair.as_array().filter(|p| p.len() == 2);
                let parts: Option<Vec<String>> = parts.map(|p| {
                    p.iter()
                        .filter_map(Value::as_str)
                        .map(|s| s.replace(' ', "\u{120}"))
                        .collect()
                });
                match parts {
                    Some(p) if p.len() == 2 => Ok(p.join(" ")),
                    _ => bail!("tokenizer.json merge {pair} is not a pair of strings"),
                }
            })
            .collect(),
        Some(other) => bail!("tokenizer.json merges start with {other}, in no known layout"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tokenizer() -> Value {
        json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2},
                      "merges": [["a", "b"], ["a b", "c"]]},
            "added_tokens": [
                {"id": 4, "content": "<|begin_of_text|>", "special": true},
                {"id": 5, "content": "<tool>", "special": false},
                {"id": 6, "content": "<|look|>", "special": false},
                {"id": 7, "content": "x\u{2581}y", "special": false}
            ]
        })
    }

    #[test]
    fn ids_without_tokens_become_pad_and_added_tokens_get_their_type() {
        let vocab = read_bpe_vocab(&tokenizer(), Some(9)).unwrap();
        assert_eq!(
            vocab.tokens,
            [
                "a",
                "b",
                "ab",
                "[PAD3]",
                "<|begin_of_text|>",
                "<tool>",
                "<|look|>",
                "x y",
                "[PAD8]"
            ]
        );
        use token_type::*;
        assert_eq!(
            vocab.token_types,
            [
                NORMAL,
                NORMAL,
                NORMAL,
                UNUSED,
                CONTROL,
                USER_DEFINED,
                CONTROL,
                USER_DEFINED,
                UNUSED
            ]
        );
        assert_eq!(vocab.merges, ["a b", "a\u{120}b c"]);
    }

    #[test]
    fn an_id_at_or_past_vocab_size_is_an_error() {
        let err = read_bpe_vocab(&tokenizer(), Some(7))
            .unwrap_err()
            .to_string();
        assert!(err.contains("vocab_size 7"), "{err}");
    }

    #[test]
    fn string_merges_pass_through() {
        let t = json!({"model": {"vocab": {"a": 0}, "merges": ["a b", "c d"]}});
        assert_eq!(read_bpe_vocab(&t, None).unwrap().merges, ["a b", "c d"]);
    }
}
