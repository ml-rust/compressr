//! Special token ids, add-token flags and the chat template for a GGUF header.
//!
//! A port of `gguf.SpecialVocab` from llama.cpp's `gguf-py`, minus merges,
//! which [`super::bpe_vocab`] reads. The sources and their precedence are
//! llama.cpp's: `tokenizer.json` (post-processor and normalizer), then
//! `tokenizer_config.json`, then `config.json`. The first id found for a
//! token type wins.

use super::chat_template::{named_templates, read_chat_template, ChatTemplate};
use crate::formats::gguf::GgufValue;
use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use std::path::Path;

/// The token types `SpecialVocab` reads by default.
const DEFAULT_TYPES: &[&str] = &["bos", "eos", "unk", "sep", "pad", "cls", "mask"];

/// The GGUF key of a special token id, or `None` when llama.cpp has no key.
fn id_key(typ: &str) -> Option<&'static str> {
    Some(match typ {
        "bos" => "tokenizer.ggml.bos_token_id",
        "eos" => "tokenizer.ggml.eos_token_id",
        "unk" => "tokenizer.ggml.unknown_token_id",
        // llama.cpp's own spelling.
        "sep" => "tokenizer.ggml.seperator_token_id",
        "pad" => "tokenizer.ggml.padding_token_id",
        "mask" => "tokenizer.ggml.mask_token_id",
        "eot" => "tokenizer.ggml.eot_token_id",
        "eom" => "tokenizer.ggml.eom_token_id",
        _ => return None,
    })
}

/// The GGUF key of an add-token flag, or `None` when llama.cpp has no key.
fn add_key(typ: &str) -> Option<&'static str> {
    Some(match typ {
        "bos" => "tokenizer.ggml.add_bos_token",
        "eos" => "tokenizer.ggml.add_eos_token",
        "sep" => "tokenizer.ggml.add_sep_token",
        _ => return None,
    })
}

/// The special-token facts of one checkpoint directory.
#[derive(Debug, Clone, Default)]
pub struct SpecialVocab {
    /// `(type, id)` in the order found.
    pub ids: Vec<(String, u32)>,
    /// `(type, add)` in the order found.
    pub add: Vec<(String, bool)>,
    /// The chat template, when the checkpoint ships one.
    pub chat_template: Option<ChatTemplate>,
    /// `tokenizer.json`'s normalizer lowercases.
    pub normalizer_lowercase: Option<bool>,
    /// `tokenizer.json`'s normalizer strips accents.
    pub normalizer_strip_accents: Option<bool>,
    types: Vec<String>,
    n_vocab: Option<usize>,
}

impl SpecialVocab {
    /// Read the special-token facts of `dir`.
    ///
    /// `n_vocab` bounds the ids, as llama.cpp's SentencePiece path passes it.
    /// `tokenizer` is the parsed `tokenizer.json`, when the directory has one.
    ///
    /// # Errors
    /// A JSON file is present but malformed, or a stated token id is negative.
    pub fn load(dir: &Path, tokenizer: Option<&Value>, n_vocab: Option<usize>) -> Result<Self> {
        let mut vocab = Self {
            types: DEFAULT_TYPES.iter().map(|t| t.to_string()).collect(),
            n_vocab,
            ..Self::default()
        };
        vocab.load_from_tokenizer_json(dir, tokenizer)?;
        vocab.load_from_config_json(dir)?;
        Ok(vocab)
    }

    fn set_id(&mut self, typ: &str, id: Option<&Value>) -> Result<()> {
        let Some(id) = id.and_then(Value::as_i64) else {
            return Ok(());
        };
        if id < 0 {
            bail!("The {typ} token id is {id}. A token id is zero or more.");
        }
        let id = id as u64;
        if self.n_vocab.is_some_and(|n| id >= n as u64) {
            log::warn!("The {typ} token id {id} is outside the vocabulary; it is skipped");
            return Ok(());
        }
        if self.ids.iter().all(|(t, _)| t != typ) {
            self.ids.push((typ.to_string(), id as u32));
        }
        Ok(())
    }

    /// Set a flag. A flag set twice keeps its first position, as a Python
    /// dict does.
    fn set_add(&mut self, typ: &str, add: bool) {
        match self.add.iter_mut().find(|(t, _)| t == typ) {
            Some(entry) => entry.1 = add,
            None => self.add.push((typ.to_string(), add)),
        }
    }

    fn parse_normalizer(&mut self, normalizer: &Value) {
        match normalizer.get("type").and_then(Value::as_str) {
            Some("Lowercase") => self.normalizer_lowercase = Some(true),
            Some("StripAccents") => self.normalizer_strip_accents = Some(true),
            Some("BertNormalizer") => {
                if let Some(v) = normalizer.get("lowercase").and_then(Value::as_bool) {
                    self.normalizer_lowercase = Some(v);
                }
                if let Some(v) = normalizer.get("strip_accents").and_then(Value::as_bool) {
                    self.normalizer_strip_accents = Some(v);
                }
            }
            Some("Sequence") => {
                if let Some(Value::Array(items)) = normalizer.get("normalizers") {
                    for item in items {
                        self.parse_normalizer(item);
                    }
                }
            }
            _ => {}
        }
    }

    fn load_from_tokenizer_json(&mut self, dir: &Path, tokenizer: Option<&Value>) -> Result<()> {
        if let Some(normalizer) = tokenizer.and_then(|t| t.get("normalizer")) {
            if !normalizer.is_null() {
                self.parse_normalizer(normalizer);
            }
        }
        let added_tokens: Vec<Value> = tokenizer
            .and_then(|t| t.get("added_tokens"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut config = read_json_object(&dir.join("tokenizer_config.json"))?;
        if let Some(tokenizer) = tokenizer {
            self.read_post_processor(tokenizer, &mut config);
        }
        let Some(config) = config else {
            return Ok(());
        };
        self.chat_template = read_chat_template(dir, &config)?;
        for typ in self.types.clone() {
            if let Some(add) = config
                .get(&format!("add_{typ}_token"))
                .and_then(Value::as_bool)
            {
                self.set_add(&typ, add);
            }
            let content = match config.get(&format!("{typ}_token")) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Object(entry)) => match entry.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    _ => continue,
                },
                _ => continue,
            };
            let id = added_tokens
                .iter()
                .find(|t| t.get("content").and_then(Value::as_str) == Some(content.as_str()))
                .and_then(|t| t.get("id"))
                .cloned();
            self.set_id(&typ, id.as_ref())?;
        }
        Ok(())
    }

    fn load_from_config_json(&mut self, dir: &Path) -> Result<()> {
        let Some(config) = read_json_object(&dir.join("config.json"))? else {
            return Ok(());
        };
        for typ in self.types.clone() {
            let key = format!("{typ}_token_id");
            let id = config.get(&key).filter(|v| !v.is_null()).or_else(|| {
                config
                    .get("text_config")
                    .and_then(|t| t.get(&key))
                    .filter(|v| !v.is_null())
            });
            self.set_id(&typ, id)?;
        }
        Ok(())
    }

    /// The `RobertaProcessing` and `TemplateProcessing` reading of
    /// `_try_load_from_tokenizer_json`, which sets the add-token flags and can
    /// name BOS, EOS, CLS and SEP in the tokenizer config.
    fn read_post_processor(&mut self, tokenizer: &Value, config: &mut Option<Map<String, Value>>) {
        let get = |config: &Option<Map<String, Value>>, key: &str| -> Value {
            config
                .as_ref()
                .and_then(|c| c.get(key))
                .cloned()
                .unwrap_or(Value::Null)
        };
        let mut bos = get(config, "bos_token");
        let mut cls = get(config, "cls_token");
        let mut eos = get(config, "eos_token");
        let mut sep = get(config, "sep_token");
        if !truthy(&bos) && truthy(&cls) {
            if let Some(c) = config.as_mut() {
                c.insert("bos_token".into(), cls.clone());
                bos = cls.clone();
            }
        }
        if !truthy(&eos) && truthy(&sep) {
            if let Some(c) = config.as_mut() {
                c.insert("eos_token".into(), sep.clone());
                eos = sep.clone();
            }
        }
        let Some(post) = tokenizer.get("post_processor").filter(|p| truthy(p)) else {
            return;
        };
        let processors: Vec<Value> = match post.get("processors").and_then(Value::as_array) {
            Some(list) => list.clone(),
            None => vec![post.clone()],
        };
        for processor in &processors {
            match processor.get("type").and_then(Value::as_str) {
                Some("RobertaProcessing") => {
                    for typ in ["bos", "eos", "sep"] {
                        self.set_add(typ, true);
                    }
                    if !truthy(&cls) {
                        if let Some(c) = config.as_mut() {
                            cls = first_of(processor.get("cls")).unwrap_or(bos.clone());
                            c.insert("cls_token".into(), cls.clone());
                        }
                    }
                    if !truthy(&sep) {
                        if let Some(c) = config.as_mut() {
                            sep = first_of(processor.get("sep")).unwrap_or(eos.clone());
                            c.insert("sep_token".into(), sep.clone());
                        }
                    }
                }
                Some("TemplateProcessing") => {
                    let mut ctx = TemplateCtx {
                        bos: &mut bos,
                        cls: &mut cls,
                        eos: &mut eos,
                        sep: &mut sep,
                    };
                    self.read_template(processor, config, &mut ctx);
                }
                _ => {}
            }
        }
    }

    fn read_template(
        &mut self,
        processor: &Value,
        config: &mut Option<Map<String, Value>>,
        ctx: &mut TemplateCtx<'_>,
    ) {
        let list = |key: &str| -> Vec<Value> {
            processor
                .get(key)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let single = list("single");
        let mut pair = list("pair");
        let special = |item: Option<&Value>| -> Option<Value> {
            item.and_then(|i| i.get("SpecialToken"))
                .and_then(|s| s.get("id"))
                .filter(|v| truthy(v))
                .cloned()
        };
        let mut first = None;
        let mut last = None;
        if single.len() > 1 {
            first = special(single.first());
            if let Some(f) = &first {
                match config.as_mut() {
                    None => *ctx.bos = f.clone(),
                    Some(c) => {
                        if f != ctx.bos && f != ctx.cls {
                            if !truthy(ctx.bos) {
                                c.insert("bos_token".into(), f.clone());
                                *ctx.bos = f.clone();
                            }
                            if !truthy(ctx.cls) {
                                c.insert("cls_token".into(), f.clone());
                                *ctx.cls = f.clone();
                            }
                        }
                    }
                }
                let known = f == ctx.bos || f == ctx.cls;
                self.set_add("bos", known);
                if !known {
                    log::warn!("Unknown leading special token {f} in TemplateProcessing<single>");
                }
            }
            last = special(single.last());
            if let Some(l) = &last {
                match config.as_mut() {
                    None => *ctx.eos = l.clone(),
                    Some(c) => {
                        if l != ctx.eos {
                            if !self.types.iter().any(|t| t == "eot") {
                                self.types.push("eot".into());
                                c.insert("eot_token".into(), ctx.eos.clone());
                            } else if !self.types.iter().any(|t| t == "eom") {
                                self.types.push("eom".into());
                                c.insert("eom_token".into(), ctx.eos.clone());
                            } else {
                                log::warn!("Overriding EOS token {} with {l}", ctx.eos);
                            }
                            c.insert("eos_token".into(), l.clone());
                            *ctx.eos = l.clone();
                        }
                    }
                }
                self.set_add("eos", l == ctx.eos);
            }
        }
        if pair.is_empty() {
            return;
        }
        let start = usize::from(first.is_some() && special(pair.first()) == first);
        let stop_trim = last.is_some() && special(pair.last()) == last;
        let end = pair.len() - usize::from(stop_trim);
        pair = pair
            .get(start..end.max(start))
            .map(<[Value]>::to_vec)
            .unwrap_or_default();
        if pair.is_empty() {
            return;
        }
        let seq = |item: Option<&Value>| {
            item.and_then(|i| i.get("Sequence"))
                .and_then(|s| s.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let (a, b) = (seq(pair.first()), seq(pair.last()));
        if a.as_deref() != Some("A") || b.as_deref() != Some("B") || pair.len() < 3 {
            return;
        }
        let middle = &pair[1..pair.len() - 1];
        let mut add_sep = false;
        let is_sep = |v: &Value, ctx: &TemplateCtx<'_>| v == ctx.sep || v == ctx.eos;
        if let Some(entry) = special(middle.first()) {
            if is_sep(&entry, ctx) && last.is_none() {
                add_sep = true;
            }
            if !is_sep(&entry, ctx) {
                log::warn!("Unknown separator token {entry} in TemplateProcessing<pair>");
            }
        }
        if middle.len() == 2 {
            if let Some(entry) = special(middle.get(1)) {
                if is_sep(&entry, ctx) {
                    add_sep = true;
                }
            }
        }
        self.set_add("sep", add_sep);
        if add_sep && !truthy(ctx.sep) {
            if let Some(c) = config.as_mut() {
                c.insert("sep_token".into(), ctx.eos.clone());
            }
        }
    }

    /// The `tokenizer.*` keys, in the order `SpecialVocab.add_to_gguf` writes them.
    pub fn metadata(&self) -> Vec<(String, GgufValue)> {
        let mut out = Vec::new();
        for (typ, id) in &self.ids {
            if let Some(key) = id_key(typ) {
                out.push((key.to_string(), GgufValue::Uint32(*id)));
            }
        }
        for (typ, add) in &self.add {
            if let Some(key) = add_key(typ) {
                out.push((key.to_string(), GgufValue::Bool(*add)));
            }
        }
        match &self.chat_template {
            Some(ChatTemplate::One(text)) => {
                out.push((
                    "tokenizer.chat_template".into(),
                    GgufValue::String(text.clone()),
                ));
            }
            Some(ChatTemplate::Named(list)) => out.extend(named_templates(list)),
            None => {}
        }
        if let Some(v) = self.normalizer_lowercase {
            out.push((
                "tokenizer.ggml.normalizer.lowercase".into(),
                GgufValue::Bool(v),
            ));
        }
        if let Some(v) = self.normalizer_strip_accents {
            out.push((
                "tokenizer.ggml.normalizer.strip_accents".into(),
                GgufValue::Bool(v),
            ));
        }
        out
    }
}

/// The token references a `TemplateProcessing` reading updates.
struct TemplateCtx<'a> {
    bos: &'a mut Value,
    cls: &'a mut Value,
    eos: &'a mut Value,
    sep: &'a mut Value,
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
    }
}

/// The first element of a `[token, id]` pair.
fn first_of(value: Option<&Value>) -> Option<Value> {
    value
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .cloned()
}

/// A JSON object file, or `None` when the file is absent.
///
/// # Errors
/// The file exists but cannot be read or is not a JSON object.
pub(crate) fn read_json_object(path: &Path) -> Result<Option<Map<String, Value>>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    match serde_json::from_str(&text) {
        Ok(Value::Object(map)) => Ok(Some(map)),
        Ok(_) => bail!("{} is not a JSON object", path.display()),
        Err(e) => Err(e).with_context(|| format!("{} is not valid JSON", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A BOS-prepending template with EOS as the pair separator, names in the
    /// tokenizer config (one as a `content` object), ids in `added_tokens`,
    /// and a list-valued EOS in `config.json` that llama.cpp skips.
    #[test]
    fn template_and_configs_give_ids_and_add_flags() {
        let dir = tempfile::tempdir().unwrap();
        let config =
            r#"{"bos_token": "<s>", "eos_token": "</s>", "pad_token": {"content": "</s>"}}"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), config).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": [2, 9], "unk_token_id": 0}"#,
        )
        .unwrap();
        let tokenizer = json!({
            "added_tokens": [{"id": 1, "content": "<s>"}, {"id": 2, "content": "</s>"}],
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "Lowercase"}]},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}}],
                "pair": [{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}},
                         {"SpecialToken": {"id": "</s>"}}, {"Sequence": {"id": "B"}}]}
        });
        let vocab = SpecialVocab::load(dir.path(), Some(&tokenizer), Some(10)).unwrap();
        let keys: Vec<(String, String)> = vocab
            .metadata()
            .into_iter()
            .map(|(k, v)| (k, format!("{v:?}")))
            .collect();
        let expected = [
            ("tokenizer.ggml.bos_token_id", "Uint32(1)"),
            ("tokenizer.ggml.eos_token_id", "Uint32(2)"),
            ("tokenizer.ggml.seperator_token_id", "Uint32(2)"),
            ("tokenizer.ggml.padding_token_id", "Uint32(2)"),
            ("tokenizer.ggml.unknown_token_id", "Uint32(0)"),
            ("tokenizer.ggml.add_bos_token", "Bool(true)"),
            ("tokenizer.ggml.add_sep_token", "Bool(true)"),
            ("tokenizer.ggml.normalizer.lowercase", "Bool(true)"),
        ];
        let expected: Vec<(String, String)> = expected
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn a_negative_id_is_an_error_and_an_out_of_range_id_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"bos_token_id": 50}"#).unwrap();
        let vocab = SpecialVocab::load(dir.path(), None, Some(10)).unwrap();
        assert!(vocab.ids.is_empty());
        std::fs::write(dir.path().join("config.json"), r#"{"bos_token_id": -1}"#).unwrap();
        assert!(SpecialVocab::load(dir.path(), None, None).is_err());
    }
}
