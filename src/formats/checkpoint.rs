//! Checkpoint sidecar handling
//!
//! A weights file on its own is not a loadable checkpoint. Every downstream
//! loader also needs the tokenizer and the model config that sat beside the
//! input weights. This module copies those sidecars into the output directory
//! and rewrites `torch_dtype` when the weights were cast to a new element type.

use crate::cast::torch_dtype_name;
use crate::loader::UnifiedDType;
use anyhow::{Context, Result};
use std::path::Path;

/// Sidecar files that make a weights directory a loadable checkpoint
///
/// A file that is absent from the input is skipped, not an error: no
/// architecture carries all of them.
pub const SIDECAR_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
];

/// Copy the checkpoint sidecars from `input_dir` to `output_dir`.
///
/// When `target_dtype` is set and `config.json` carries a `torch_dtype` field,
/// that field is rewritten to the new dtype rather than copied verbatim: a
/// config claiming float32 next to bf16 weights misleads every loader.
///
/// Returns the names of the files that were written.
///
/// Does nothing and returns an empty list when `input_dir` is not a directory
/// (a single-file input carries no sidecars) or when the two directories are
/// the same path (the sidecars are already in place).
pub fn copy_sidecars(
    input_dir: &Path,
    output_dir: &Path,
    target_dtype: Option<UnifiedDType>,
) -> Result<Vec<String>> {
    if !input_dir.is_dir() {
        return Ok(Vec::new());
    }

    if same_directory(input_dir, output_dir) {
        log::debug!("Sidecars already in place: input and output directory are the same");
        return Ok(Vec::new());
    }

    std::fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    let mut copied = Vec::new();

    for name in SIDECAR_FILES {
        let src = input_dir.join(name);
        if !src.is_file() {
            continue;
        }
        let dst = output_dir.join(name);

        if *name == "config.json" {
            if let Some(dtype) = target_dtype {
                let json = std::fs::read_to_string(&src)
                    .with_context(|| format!("Failed to read: {}", src.display()))?;
                let rewritten = rewrite_torch_dtype(&json, dtype).with_context(|| {
                    format!("Failed to rewrite torch_dtype in: {}", src.display())
                })?;
                std::fs::write(&dst, rewritten)
                    .with_context(|| format!("Failed to write: {}", dst.display()))?;
                copied.push((*name).to_string());
                continue;
            }
        }

        std::fs::copy(&src, &dst)
            .with_context(|| format!("Failed to copy {} to {}", src.display(), dst.display()))?;
        copied.push((*name).to_string());
    }

    log::info!(
        "Copied {} sidecar file(s) to {}",
        copied.len(),
        output_dir.display()
    );
    Ok(copied)
}

/// Rewrite the `torch_dtype` field of a HuggingFace `config.json`.
///
/// Leaves the document untouched when it carries no `torch_dtype` field, when
/// it is not a JSON object, or when `dtype` has no `torch_dtype` spelling.
/// Nested `text_config` / `vision_config` blocks are updated too, because
/// multimodal configs record the dtype per sub-config.
pub fn rewrite_torch_dtype(json: &str, dtype: UnifiedDType) -> Result<String> {
    let Some(name) = torch_dtype_name(dtype) else {
        return Ok(json.to_string());
    };

    let mut value: serde_json::Value =
        serde_json::from_str(json).context("config.json is not valid JSON")?;

    set_torch_dtype(&mut value, name);

    serde_json::to_string_pretty(&value).context("Failed to serialize rewritten config.json")
}

/// Set `torch_dtype` on an object and on any nested `*_config` object that has it.
fn set_torch_dtype(value: &mut serde_json::Value, name: &str) {
    let Some(map) = value.as_object_mut() else {
        return;
    };

    // Set it on the TOP-LEVEL config even when the source omitted the field.
    // This config describes the checkpoint we just produced, and its weights
    // really are `name` — a loader would otherwise have to open the safetensors
    // header to find out. `Scicom-intl/Multilingual-Expressive-TTS-1.7B` ships
    // no `torch_dtype`, so a "never add" rule loses the fact entirely.
    map.insert(
        "torch_dtype".to_string(),
        serde_json::Value::String(name.to_string()),
    );

    // Nested sub-configs are only UPDATED, never given the field: a multimodal
    // config's sub-blocks may legitimately differ, and inventing a value for a
    // block we did not inspect would be a claim we cannot support.
    for (key, child) in map.iter_mut() {
        if key.ends_with("_config") {
            if let Some(sub) = child.as_object_mut() {
                if sub.contains_key("torch_dtype") {
                    sub.insert(
                        "torch_dtype".to_string(),
                        serde_json::Value::String(name.to_string()),
                    );
                }
            }
        }
    }
}

/// Compare two directories by canonical path, falling back to a literal compare.
fn same_directory(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_torch_dtype_replaces_value() {
        let json =
            r#"{"architectures":["LlamaForCausalLM"],"torch_dtype":"float32","hidden_size":4096}"#;
        let out = rewrite_torch_dtype(json, UnifiedDType::BF16).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["torch_dtype"], "bfloat16");
        assert_eq!(parsed["hidden_size"], 4096);
        assert_eq!(parsed["architectures"][0], "LlamaForCausalLM");
    }

    #[test]
    /// The output config describes the checkpoint we just wrote, so the dtype
    /// is recorded even when the SOURCE omitted it. The real
    /// `Multilingual-Expressive-TTS-1.7B` config ships no `torch_dtype`, and a
    /// "never add" rule silently lost the fact that the output is bf16.
    fn test_rewrite_torch_dtype_absent_field_is_added_at_top_level() {
        let json = r#"{"hidden_size":128}"#;
        let out = rewrite_torch_dtype(json, UnifiedDType::F16).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["hidden_size"], 128);
        assert_eq!(parsed["torch_dtype"], "float16");
    }

    #[test]
    /// A nested sub-config is only UPDATED, never given the field: a multimodal
    /// config's blocks may legitimately hold different dtypes, and inventing a
    /// value for a block we never inspected would be an unsupported claim.
    fn test_rewrite_torch_dtype_absent_nested_field_is_not_added() {
        let json = r#"{"torch_dtype":"float32","vision_config":{"hidden_size":8}}"#;
        let out = rewrite_torch_dtype(json, UnifiedDType::BF16).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["torch_dtype"], "bfloat16");
        assert_eq!(parsed["vision_config"].as_object().unwrap().len(), 1);
        assert_eq!(parsed["vision_config"]["hidden_size"], 8);
    }

    #[test]
    fn test_rewrite_torch_dtype_nested_configs() {
        let json = r#"{"torch_dtype":"float32","text_config":{"torch_dtype":"float32"},"vision_config":{"torch_dtype":"float16"}}"#;
        let out = rewrite_torch_dtype(json, UnifiedDType::BF16).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["torch_dtype"], "bfloat16");
        assert_eq!(parsed["text_config"]["torch_dtype"], "bfloat16");
        assert_eq!(parsed["vision_config"]["torch_dtype"], "bfloat16");
    }

    #[test]
    fn test_rewrite_torch_dtype_invalid_json_errors() {
        assert!(rewrite_torch_dtype("not json", UnifiedDType::BF16).is_err());
    }

    #[test]
    fn test_copy_sidecars_copies_and_rewrites() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        std::fs::write(
            src.path().join("config.json"),
            r#"{"torch_dtype":"float32","hidden_size":8}"#,
        )
        .unwrap();
        std::fs::write(src.path().join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
        std::fs::write(src.path().join("merges.txt"), "a b\n").unwrap();
        // Not a sidecar: must not be copied.
        std::fs::write(src.path().join("model.safetensors"), b"weights").unwrap();

        let copied = copy_sidecars(src.path(), dst.path(), Some(UnifiedDType::BF16)).unwrap();

        assert_eq!(
            copied,
            vec![
                "config.json".to_string(),
                "tokenizer.json".to_string(),
                "merges.txt".to_string()
            ]
        );
        assert!(!dst.path().join("model.safetensors").exists());
        assert_eq!(
            std::fs::read_to_string(dst.path().join("merges.txt")).unwrap(),
            "a b\n"
        );
        assert_eq!(
            std::fs::read_to_string(dst.path().join("tokenizer.json")).unwrap(),
            r#"{"version":"1.0"}"#
        );

        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dst.path().join("config.json")).unwrap())
                .unwrap();
        assert_eq!(config["torch_dtype"], "bfloat16");
        assert_eq!(config["hidden_size"], 8);
    }

    #[test]
    fn test_copy_sidecars_without_dtype_copies_config_verbatim() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        let original = r#"{"torch_dtype":"float32"}"#;
        std::fs::write(src.path().join("config.json"), original).unwrap();

        let copied = copy_sidecars(src.path(), dst.path(), None).unwrap();

        assert_eq!(copied, vec!["config.json".to_string()]);
        assert_eq!(
            std::fs::read_to_string(dst.path().join("config.json")).unwrap(),
            original
        );
    }

    #[test]
    fn test_copy_sidecars_same_directory_is_a_no_op() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(
            src.path().join("config.json"),
            r#"{"torch_dtype":"float32"}"#,
        )
        .unwrap();

        let copied = copy_sidecars(src.path(), src.path(), Some(UnifiedDType::BF16)).unwrap();

        assert_eq!(copied, Vec::<String>::new());
        assert_eq!(
            std::fs::read_to_string(src.path().join("config.json")).unwrap(),
            r#"{"torch_dtype":"float32"}"#
        );
    }

    #[test]
    fn test_copy_sidecars_file_input_is_a_no_op() {
        let src = tempfile::tempdir().unwrap();
        let file = src.path().join("model.safetensors");
        std::fs::write(&file, b"weights").unwrap();
        let dst = tempfile::tempdir().unwrap();

        let copied = copy_sidecars(&file, dst.path(), Some(UnifiedDType::BF16)).unwrap();
        assert_eq!(copied, Vec::<String>::new());
    }
}
