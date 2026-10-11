//! The chat template of a checkpoint, and its GGUF keys.
//!
//! The sources are llama.cpp's, in its order: `tokenizer_config.json`'s
//! `chat_template`, then `chat_template.jinja` with any
//! `additional_chat_templates/*.jinja`, then `chat_template.json`.

use super::special_vocab::read_json_object;
use crate::formats::gguf::{GgufArray, GgufValue};
use anyhow::{Context, Result};
use serde_json::{Map, Value};
use std::path::Path;

/// A chat template: one string, or named templates with an optional default.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    /// `tokenizer.chat_template`.
    One(String),
    /// `(name, template)` pairs. `default` is `tokenizer.chat_template`.
    Named(Vec<(String, String)>),
}

/// `GGUFWriter.add_chat_template` for a list of named templates.
pub(crate) fn named_templates(list: &[(String, String)]) -> Vec<(String, GgufValue)> {
    let mut out = Vec::new();
    let mut names = Vec::new();
    let mut default = None;
    for (name, template) in list {
        let name: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        if name.is_empty() {
            continue;
        }
        if name == "default" {
            default = Some(template.clone());
        } else {
            out.push((
                format!("tokenizer.chat_template.{name}"),
                GgufValue::String(template.clone()),
            ));
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    if !names.is_empty() {
        out.push((
            "tokenizer.chat_templates".into(),
            GgufValue::Array(GgufArray::String(names)),
        ));
    }
    if let Some(template) = default {
        out.push((
            "tokenizer.chat_template".into(),
            GgufValue::String(template),
        ));
    }
    out
}

/// `tokenizer_config.json`'s `chat_template`, else `chat_template.jinja` (with
/// any `additional_chat_templates/*.jinja`), else `chat_template.json`.
pub(crate) fn read_chat_template(
    dir: &Path,
    config: &Map<String, Value>,
) -> Result<Option<ChatTemplate>> {
    match config.get("chat_template") {
        Some(Value::String(text)) => return Ok(Some(ChatTemplate::One(text.clone()))),
        Some(Value::Array(items)) => {
            let list = items
                .iter()
                .filter_map(|item| {
                    let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                    let text = item.get("template").and_then(Value::as_str)?;
                    Some((name.to_string(), text.to_string()))
                })
                .collect();
            return Ok(Some(ChatTemplate::Named(list)));
        }
        Some(Value::Null) | None => {}
        Some(_) => {
            log::warn!(
                "tokenizer_config.json chat_template is neither a string nor a list; ignored"
            );
            return Ok(None);
        }
    }
    let jinja = dir.join("chat_template.jinja");
    if jinja.is_file() {
        let text = std::fs::read_to_string(&jinja)
            .with_context(|| format!("Failed to read {}", jinja.display()))?;
        let extra_dir = dir.join("additional_chat_templates");
        let mut extra: Vec<_> = match std::fs::read_dir(&extra_dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "jinja"))
                .collect(),
            Err(_) => Vec::new(),
        };
        if extra.is_empty() {
            return Ok(Some(ChatTemplate::One(text)));
        }
        extra.sort();
        let mut list = vec![("default".to_string(), text)];
        for path in extra {
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let body = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            list.push((name, body));
        }
        return Ok(Some(ChatTemplate::Named(list)));
    }
    let json = dir.join("chat_template.json");
    if let Some(map) = read_json_object(&json)? {
        if let Some(Value::String(text)) = map.get("chat_template") {
            return Ok(Some(ChatTemplate::One(text.clone())));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn named_templates_write_llama_cpp_keys() {
        let list = vec![
            ("default".to_string(), "D".to_string()),
            ("tool use".to_string(), "T".to_string()),
        ];
        let keys: Vec<String> = named_templates(&list).into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            [
                "tokenizer.chat_template.tool_use",
                "tokenizer.chat_templates",
                "tokenizer.chat_template"
            ]
        );
    }

    #[test]
    fn the_tokenizer_config_wins_then_jinja_then_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("chat_template.jinja"), "J").unwrap();
        let config = json!({"chat_template": "C"});
        let from_config = read_chat_template(dir.path(), config.as_object().unwrap()).unwrap();
        assert_eq!(from_config, Some(ChatTemplate::One("C".into())));
        let empty = Map::new();
        let from_jinja = read_chat_template(dir.path(), &empty).unwrap();
        assert_eq!(from_jinja, Some(ChatTemplate::One("J".into())));
        std::fs::remove_file(dir.path().join("chat_template.jinja")).unwrap();
        std::fs::write(
            dir.path().join("chat_template.json"),
            r#"{"chat_template": "S"}"#,
        )
        .unwrap();
        let from_json = read_chat_template(dir.path(), &empty).unwrap();
        assert_eq!(from_json, Some(ChatTemplate::One("S".into())));
    }
}
