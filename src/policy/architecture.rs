//! Which architecture a conversion is for, and what the output model is
//! called: `general.architecture` and `general.name` for GGUF, and the
//! VoxCPM2 test the companion-AudioVAE rule keys off.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Resolve the value written to `general.architecture`.
///
/// GGUF is self-describing: every consumer keys its tensor-name mapping off
/// this string, so a guess here silently mislabels the model. Precedence:
///
/// 1. `arch_override` — what the user passed as `--arch`.
/// 2. The architecture field of the checkpoint's `config.json`. Three spellings
///    are read, in order: `model_type` (the HuggingFace convention), then
///    `architecture` (a bare string, as VoxCPM2 writes it), then the first entry
///    of `architectures` (HuggingFace's class-name list). They are alternatives,
///    not a merge — the first one present wins.
/// 3. `metadata_arch` — set only when the input already carried it (GGUF in).
///
/// A safetensors input has no architecture in its header, so when none of the
/// three is available this errors rather than inventing one.
pub fn resolve_architecture(
    arch_override: Option<&str>,
    metadata_arch: Option<&str>,
    input_path: &Path,
) -> Result<String> {
    if let Some(arch) = arch_override.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(arch.to_string());
    }
    if let Some(arch) = read_config_architecture(input_path)? {
        return Ok(arch);
    }
    if let Some(arch) = metadata_arch.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(arch.to_string());
    }

    bail!(
        "Cannot determine the model architecture for '{}': the input carries none and no \
         config.json naming it (\"model_type\", \"architecture\" or \"architectures\") was found \
         beside it. Pass --arch <NAME> to set \
         GGUF's general.architecture explicitly.",
        input_path.display()
    )
}

/// Read the architecture from the `config.json` of the input checkpoint.
///
/// Returns `Ok(None)` when there is no `config.json` or it names no
/// architecture under any of the three spellings; a `config.json` that is
/// present but unreadable or malformed is an error, because silently ignoring
/// it would fabricate an architecture again.
fn read_config_architecture(input_path: &Path) -> Result<Option<String>> {
    let Some(dir) = checkpoint_dir(input_path) else {
        return Ok(None);
    };
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", config_path.display()))?;

    let from_key = |key: &str| value.get(key).and_then(serde_json::Value::as_str);
    let from_list = || {
        value
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .and_then(|list| list.first())
            .and_then(serde_json::Value::as_str)
    };

    Ok(from_key("model_type")
        .or_else(|| from_key("architecture"))
        .or_else(from_list)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string))
}

/// The checkpoint directory an input path belongs to.
///
/// An input names either the directory itself or one weights file inside it.
pub(crate) fn checkpoint_dir(input_path: &Path) -> Option<PathBuf> {
    if input_path.is_dir() {
        return Some(input_path.to_path_buf());
    }
    match input_path.parent() {
        // A bare file name sits in the current directory.
        Some(parent) if parent.as_os_str().is_empty() => Some(PathBuf::from(".")),
        Some(parent) => Some(parent.to_path_buf()),
        None => None,
    }
}

/// The model name for `general.name`: the checkpoint directory's own name.
///
/// The architecture is not a name — `llama` names a family, not this model — so
/// when the directory name is unavailable no name is written at all.
pub fn model_name(input_path: &Path) -> Option<String> {
    let dir = checkpoint_dir(input_path)?;
    // A relative path such as "." or "model/." carries no usable file name of
    // its own, so resolve it first; an unresolvable path keeps its literal form.
    let dir = match dir.canonicalize() {
        Ok(resolved) => resolved,
        Err(_) => dir,
    };
    dir.file_name().and_then(|n| n.to_str()).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_architecture_comes_from_config_model_type() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "voxcpm2", "hidden_size": 2048}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "voxcpm2");
    }

    /// VoxCPM2's own `config.json` spells it `architecture`, a bare string,
    /// with no `model_type` anywhere. Reading only the HuggingFace spelling
    /// makes that checkpoint unconvertible without `--arch`.
    #[test]
    fn test_architecture_comes_from_config_architecture_key() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"architecture": "voxcpm2", "patch_size": 4}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "voxcpm2");
    }

    /// HuggingFace's `architectures` list holds class names; its first entry is
    /// the last resort before erroring.
    #[test]
    fn test_architecture_falls_back_to_the_architectures_list() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["LlamaForCausalLM"]}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "LlamaForCausalLM");
    }

    /// The three spellings are alternatives in a fixed order, not a merge.
    #[test]
    fn test_model_type_wins_over_the_other_two_spellings() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "first", "architecture": "second", "architectures": ["third"]}"#,
        )
        .unwrap();

        let arch = resolve_architecture(None, None, dir.path()).unwrap();
        assert_eq!(arch, "first");
    }

    #[test]
    fn test_arch_override_wins_over_config() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "voxcpm2"}"#,
        )
        .unwrap();

        let arch = resolve_architecture(Some("qwen2"), None, dir.path()).unwrap();
        assert_eq!(arch, "qwen2");
    }

    #[test]
    fn test_missing_architecture_is_an_error() {
        let dir = TempDir::new().unwrap();
        let err = resolve_architecture(None, None, dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("--arch"), "{}", err);
    }

    #[test]
    fn test_input_metadata_architecture_is_used_when_present() {
        let dir = TempDir::new().unwrap();
        let arch = resolve_architecture(None, Some("llama"), dir.path()).unwrap();
        assert_eq!(arch, "llama");
    }

    #[test]
    fn test_config_json_beside_a_weights_file_is_found() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "mamba2"}"#,
        )
        .unwrap();
        let weights = dir.path().join("model.safetensors");
        fs::write(&weights, b"not a real checkpoint").unwrap();

        assert_eq!(
            resolve_architecture(None, None, &weights).unwrap(),
            "mamba2"
        );
    }

    #[test]
    fn test_model_name_is_the_checkpoint_directory_name() {
        let dir = TempDir::new().unwrap();
        let checkpoint = dir.path().join("voxcpm2-0.5b");
        fs::create_dir(&checkpoint).unwrap();

        assert_eq!(model_name(&checkpoint).as_deref(), Some("voxcpm2-0.5b"));
    }
}
