//! The `tokenizer.*` block of a GGUF header, from a checkpoint's tokenizer files.
//!
//! The vocabulary source follows llama.cpp's converter: `tokenizer.model`
//! when present (`tokenizer.ggml.model = "llama"`), else a byte-level BPE
//! `tokenizer.json` (`"gpt2"`). A SentencePiece-style BPE `tokenizer.json`
//! with no `tokenizer.model` beside it, and a Unigram or WordPiece one, are
//! not written: the block is omitted with a warning.

use super::bpe_vocab::read_bpe_vocab;
use super::config::HfConfig;
use super::pre_tokenizer::{pre_tokenizer_name, Boundary};
use super::special_vocab::SpecialVocab;
use super::spm_model::read_spm_vocab;
use crate::formats::gguf::{GgufArray, GgufValue};
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

/// Every `tokenizer.*` key the checkpoint in `dir` supplies.
///
/// `arch` is llama.cpp's architecture, which decides `add_space_prefix`.
///
/// # Errors
/// A tokenizer file is present but malformed.
pub fn tokenizer_metadata(
    arch: &str,
    dir: &Path,
    config: &HfConfig,
) -> Result<Vec<(String, GgufValue)>> {
    let json_path = dir.join("tokenizer.json");
    let json_bytes = if json_path.is_file() {
        Some(
            std::fs::read(&json_path)
                .with_context(|| format!("Failed to read {}", json_path.display()))?,
        )
    } else {
        None
    };
    let json: Option<Value> = match &json_bytes {
        Some(bytes) => Some(
            serde_json::from_slice(bytes)
                .with_context(|| format!("{} is not valid JSON", json_path.display()))?,
        ),
        None => None,
    };
    let vocab_size = config.find_u32(&["vocab_size"]);
    let mut out: Vec<(String, GgufValue)> = Vec::new();
    let string = |s: &str| GgufValue::String(s.to_string());

    if arch == "llama" {
        let tokenizer_config =
            super::special_vocab::read_json_object(&dir.join("tokenizer_config.json"))?;
        if let Some(prefix) = tokenizer_config
            .as_ref()
            .and_then(|c| c.get("add_prefix_space"))
            .and_then(Value::as_bool)
        {
            out.push((
                "tokenizer.ggml.add_space_prefix".into(),
                GgufValue::Bool(prefix),
            ));
        }
    }

    let special = if dir.join("tokenizer.model").is_file() {
        let vocab = read_spm_vocab(dir, vocab_size)?;
        let n_vocab = vocab.tokens.len();
        out.push(("tokenizer.ggml.model".into(), string("llama")));
        out.push(("tokenizer.ggml.pre".into(), string("default")));
        out.push(("tokenizer.ggml.tokens".into(), strings(vocab.tokens)));
        out.push((
            "tokenizer.ggml.scores".into(),
            GgufValue::Array(GgufArray::Float32(vocab.scores)),
        ));
        out.push((
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(GgufArray::Int32(vocab.token_types)),
        ));
        SpecialVocab::load(dir, json.as_ref(), Some(n_vocab))?
    } else if let (Some(json), Some(bytes)) = (&json, &json_bytes) {
        let model = json.get("model").unwrap_or(&Value::Null);
        if model.get("type").and_then(Value::as_str) != Some("BPE") {
            log::warn!(
                "tokenizer.json is a {} model with no tokenizer.model beside it; the GGUF \
                 carries no tokenizer",
                model.get("type").unwrap_or(&Value::Null)
            );
            return Ok(out);
        }
        if arch == "llama" && is_sentencepiece_bpe(json) {
            log::warn!(
                "tokenizer.json is a SentencePiece-style BPE with no tokenizer.model beside it; \
                 the GGUF carries no tokenizer. Add tokenizer.model to the checkpoint."
            );
            return Ok(out);
        }
        let vocab = read_bpe_vocab(json, vocab_size)?;
        let special = SpecialVocab::load(dir, Some(json), None)?;
        let flag = |typ: &str| special.add.iter().any(|(t, add)| t == typ && *add);
        let id = |typ: &str| {
            special
                .ids
                .iter()
                .find(|(t, _)| t == typ)
                .map(|(_, id)| *id)
        };
        let boundary = Boundary {
            bos: id("bos").filter(|_| flag("bos")),
            eos: id("eos").filter(|_| flag("eos")),
        };
        out.push(("tokenizer.ggml.model".into(), string("gpt2")));
        if let Some(pre) = pre_tokenizer_name(bytes, boundary) {
            out.push(("tokenizer.ggml.pre".into(), string(pre)));
        }
        out.push(("tokenizer.ggml.tokens".into(), strings(vocab.tokens)));
        out.push((
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(GgufArray::Int32(vocab.token_types)),
        ));
        if vocab.merges.is_empty() {
            log::warn!("tokenizer.json holds no merges; a gpt2 vocabulary needs them");
        } else {
            out.push(("tokenizer.ggml.merges".into(), strings(vocab.merges)));
        }
        special
    } else {
        log::warn!(
            "{} has neither tokenizer.model nor tokenizer.json; the GGUF carries no tokenizer",
            dir.display()
        );
        return Ok(out);
    };
    out.extend(special.metadata());
    if super::arch::NO_SPACE_PREFIX_ARCHS.contains(&arch) {
        out.retain(|(k, _)| k != "tokenizer.ggml.add_space_prefix");
        out.push((
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ));
    }
    Ok(out)
}

/// llama.cpp's `LlamaHfVocab` test: a BPE model with byte fallback and a
/// `Sequence` decoder, which is not the Llama 3 layout.
fn is_sentencepiece_bpe(json: &Value) -> bool {
    let model = json.get("model").unwrap_or(&Value::Null);
    let flag = |key: &str| model.get(key).and_then(Value::as_bool);
    let is_llama3 = flag("ignore_merges") == Some(true) && !flag("byte_fallback").unwrap_or(true);
    !is_llama3
        && flag("byte_fallback") == Some(true)
        && json
            .get("decoder")
            .and_then(|d| d.get("type"))
            .and_then(Value::as_str)
            == Some("Sequence")
}

fn strings(values: Vec<String>) -> GgufValue {
    GgufValue::Array(GgufArray::String(values))
}
