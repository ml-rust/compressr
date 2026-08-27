//! Output sharding for SafeTensors checkpoints
//!
//! A checkpoint that exceeds the shard-size limit is written as
//! `model-00001-of-000NN.safetensors` … plus a `model.safetensors.index.json`
//! naming which shard holds each tensor. A checkpoint that fits stays a single
//! `model.safetensors` with no index, which is what every small model wants.
//!
//! Tensors are never split across shards: a shard boundary always falls between
//! two tensors, so a single tensor larger than the limit gets a shard of its own.

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// Default shard-size limit, in bytes.
///
/// 5 GB is HuggingFace's `max_shard_size` default, so a compressr shard set
/// splits at the same points `save_pretrained` would. HuggingFace spells the
/// unit decimally — `5GB` is 5,000,000,000 bytes, not 5 GiB — and this constant
/// follows that spelling.
pub const DEFAULT_MAX_SHARD_SIZE: u64 = 5_000_000_000;

/// Suffix of the index file that sits beside a sharded weights file.
pub const INDEX_SUFFIX: &str = ".safetensors.index.json";

/// The index file name for a weights stem (`model` -> `model.safetensors.index.json`).
pub fn index_file_name(stem: &str) -> String {
    format!("{}{}", stem, INDEX_SUFFIX)
}

/// Parse a `--max-shard-size` value into a byte count.
///
/// Accepts a bare byte count (`5000000000`), a decimal suffix (`5GB`, `500MB`,
/// `2TB`, `800KB`) or a binary one (`1GiB`, `512MiB`). `none`, `off` and any
/// value that evaluates to zero disable sharding, and return `None`.
///
/// # Errors
///
/// Returns an error naming the rejected value when it is neither a byte count
/// nor a recognised suffixed size, or when the value overflows a `u64`.
pub fn parse_shard_size(s: &str) -> Result<Option<u64>> {
    // Longest suffixes first, so `gb` is matched before the bare `b`.
    const UNITS: &[(&str, u64)] = &[
        ("kib", 1 << 10),
        ("mib", 1 << 20),
        ("gib", 1 << 30),
        ("tib", 1 << 40),
        ("kb", 1_000),
        ("mb", 1_000_000),
        ("gb", 1_000_000_000),
        ("tb", 1_000_000_000_000),
        ("b", 1),
    ];

    let lower = s.trim().to_lowercase();

    if matches!(lower.as_str(), "none" | "off") {
        return Ok(None);
    }

    let (digits, unit) = UNITS
        .iter()
        .find_map(|(suffix, unit)| lower.strip_suffix(suffix).map(|d| (d.trim(), *unit)))
        .unwrap_or((lower.as_str(), 1));

    let Ok(value) = digits.parse::<u64>() else {
        bail!(
            "Invalid shard size: {}. Use a byte count or a suffixed size such as 5GB, 500MB or 1GiB, or \"none\" to write one file",
            s
        );
    };

    let Some(bytes) = value.checked_mul(unit) else {
        bail!("Shard size overflows a 64-bit byte count: {}", s);
    };

    // A zero limit cannot be honoured — every tensor would need its own shard
    // and an empty one after it. Read it as "do not shard".
    if bytes == 0 {
        return Ok(None);
    }

    Ok(Some(bytes))
}

/// Which shard each tensor is written to, and what the shards are called
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardPlan {
    /// Shard file names, in write order
    pub shard_files: Vec<String>,
    /// Shard index for each tensor, in registration order
    pub shard_of: Vec<usize>,
}

impl ShardPlan {
    /// Whether this plan writes more than one file, and so needs an index.
    pub fn is_sharded(&self) -> bool {
        self.shard_files.len() > 1
    }

    /// Map every tensor name onto the shard file that holds it.
    ///
    /// `names` must be in the same order as [`ShardPlan::shard_of`].
    pub fn weight_map(&self, names: &[String]) -> BTreeMap<String, String> {
        names
            .iter()
            .zip(&self.shard_of)
            .map(|(name, &shard)| (name.clone(), self.shard_files[shard].clone()))
            .collect()
    }
}

/// Assign tensors to shards, filling each shard up to `max_shard_size`.
///
/// `sizes` is the stored byte size of each tensor, in write order. A tensor is
/// never split: it opens a new shard instead, even when it alone exceeds the
/// limit. `max_shard_size` of `None` puts everything in one file.
pub fn plan_shards(stem: &str, sizes: &[u64], max_shard_size: Option<u64>) -> ShardPlan {
    let mut shard_of = Vec::with_capacity(sizes.len());
    let mut shard_count = 1usize;
    let mut current_bytes = 0u64;

    for &size in sizes {
        let overflows = match max_shard_size {
            Some(max) => current_bytes > 0 && current_bytes.saturating_add(size) > max,
            None => false,
        };
        if overflows {
            shard_count += 1;
            current_bytes = 0;
        }
        shard_of.push(shard_count - 1);
        current_bytes = current_bytes.saturating_add(size);
    }

    let shard_files = if shard_count == 1 {
        vec![format!("{}.safetensors", stem)]
    } else {
        (1..=shard_count)
            .map(|i| format!("{}-{:05}-of-{:05}.safetensors", stem, i, shard_count))
            .collect()
    };

    ShardPlan {
        shard_files,
        shard_of,
    }
}

/// Write a `*.safetensors.index.json` for a sharded checkpoint.
///
/// `total_size` is the sum of the tensor byte sizes, excluding shard headers —
/// the same quantity HuggingFace records.
pub fn write_index(
    path: &Path,
    weight_map: &BTreeMap<String, String>,
    total_size: u64,
) -> Result<()> {
    let index = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });

    let json = serde_json::to_string_pretty(&index).context("Failed to serialize shard index")?;

    std::fs::write(path, json)
        .with_context(|| format!("Failed to write shard index: {}", path.display()))
}

/// Read the shard file names a `*.index.json` names, sorted and de-duplicated.
///
/// The `weight_map` is authoritative: a file in the same directory that no
/// entry names is not part of the model and is never loaded.
///
/// # Errors
///
/// Returns an error when the index is not valid JSON, when it carries no
/// `weight_map` object, or when that object is empty.
pub fn read_index_shard_files(index_path: &Path) -> Result<Vec<String>> {
    let json = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read shard index: {}", index_path.display()))?;

    let index: serde_json::Value = serde_json::from_str(&json)
        .with_context(|| format!("Shard index is not valid JSON: {}", index_path.display()))?;

    let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) else {
        bail!(
            "Shard index has no \"weight_map\" object: {}",
            index_path.display()
        );
    };

    let mut files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    files.sort();
    files.dedup();

    if files.is_empty() {
        bail!("Shard index names no shard files: {}", index_path.display());
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shard_size_decimal_suffixes() {
        assert_eq!(parse_shard_size("5GB").unwrap(), Some(5_000_000_000));
        assert_eq!(parse_shard_size("500MB").unwrap(), Some(500_000_000));
        assert_eq!(parse_shard_size("800kb").unwrap(), Some(800_000));
        assert_eq!(parse_shard_size(" 2 TB ").unwrap(), Some(2_000_000_000_000));
    }

    #[test]
    fn test_parse_shard_size_binary_suffixes() {
        assert_eq!(parse_shard_size("1GiB").unwrap(), Some(1_073_741_824));
        assert_eq!(parse_shard_size("512MiB").unwrap(), Some(536_870_912));
    }

    #[test]
    fn test_parse_shard_size_bare_bytes() {
        assert_eq!(parse_shard_size("4096").unwrap(), Some(4096));
        assert_eq!(parse_shard_size("4096B").unwrap(), Some(4096));
    }

    #[test]
    fn test_parse_shard_size_disabling_values() {
        assert_eq!(parse_shard_size("none").unwrap(), None);
        assert_eq!(parse_shard_size("OFF").unwrap(), None);
        assert_eq!(parse_shard_size("0").unwrap(), None);
        assert_eq!(parse_shard_size("0MB").unwrap(), None);
    }

    #[test]
    fn test_parse_shard_size_rejects_garbage() {
        let err = parse_shard_size("huge").unwrap_err().to_string();
        assert_eq!(
            err,
            "Invalid shard size: huge. Use a byte count or a suffixed size such as 5GB, 500MB or 1GiB, or \"none\" to write one file"
        );

        let err = parse_shard_size("1.5GB").unwrap_err().to_string();
        assert_eq!(
            err,
            "Invalid shard size: 1.5GB. Use a byte count or a suffixed size such as 5GB, 500MB or 1GiB, or \"none\" to write one file"
        );
    }

    #[test]
    fn test_parse_shard_size_rejects_overflow() {
        let err = parse_shard_size("99999999999999999999TB")
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "Invalid shard size: 99999999999999999999TB. Use a byte count or a suffixed size such as 5GB, 500MB or 1GiB, or \"none\" to write one file"
        );

        let err = parse_shard_size("18446744073709551TB")
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "Shard size overflows a 64-bit byte count: 18446744073709551TB"
        );
    }

    #[test]
    fn test_plan_shards_single_file_when_it_fits() {
        let plan = plan_shards("model", &[100, 200, 300], Some(1000));
        assert_eq!(plan.shard_files, vec!["model.safetensors".to_string()]);
        assert_eq!(plan.shard_of, vec![0, 0, 0]);
        assert!(!plan.is_sharded());
    }

    #[test]
    fn test_plan_shards_no_limit_is_one_file() {
        let plan = plan_shards("model", &[u64::MAX, u64::MAX], None);
        assert_eq!(plan.shard_files, vec!["model.safetensors".to_string()]);
        assert_eq!(plan.shard_of, vec![0, 0]);
    }

    #[test]
    fn test_plan_shards_splits_on_the_limit() {
        let plan = plan_shards("model", &[400, 400, 400, 400], Some(1000));
        assert_eq!(
            plan.shard_files,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
        assert_eq!(plan.shard_of, vec![0, 0, 1, 1]);
        assert!(plan.is_sharded());
    }

    #[test]
    fn test_plan_shards_oversized_tensor_gets_its_own_shard() {
        let plan = plan_shards("model", &[10, 5000, 10], Some(100));
        assert_eq!(plan.shard_of, vec![0, 1, 2]);
        assert_eq!(plan.shard_files.len(), 3);
        assert_eq!(plan.shard_files[1], "model-00002-of-00003.safetensors");
    }

    #[test]
    fn test_plan_shards_uses_the_weights_stem() {
        let plan = plan_shards("weights", &[600, 600], Some(1000));
        assert_eq!(
            plan.shard_files,
            vec![
                "weights-00001-of-00002.safetensors".to_string(),
                "weights-00002-of-00002.safetensors".to_string(),
            ]
        );
    }

    #[test]
    fn test_weight_map_points_each_name_at_its_shard() {
        let plan = plan_shards("model", &[600, 600], Some(1000));
        let names = vec!["a".to_string(), "b".to_string()];
        let map = plan.weight_map(&names);

        assert_eq!(map["a"], "model-00001-of-00002.safetensors");
        assert_eq!(map["b"], "model-00002-of-00002.safetensors");
    }

    #[test]
    fn test_index_file_name() {
        assert_eq!(index_file_name("model"), "model.safetensors.index.json");
        assert_eq!(index_file_name("weights"), "weights.safetensors.index.json");
    }

    #[test]
    fn test_write_then_read_index_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(index_file_name("model"));

        let mut map = BTreeMap::new();
        map.insert(
            "a".to_string(),
            "model-00001-of-00002.safetensors".to_string(),
        );
        map.insert(
            "b".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
        );
        map.insert(
            "c".to_string(),
            "model-00001-of-00002.safetensors".to_string(),
        );

        write_index(&path, &map, 1234).unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(parsed["metadata"]["total_size"], 1234);
        assert_eq!(
            parsed["weight_map"]["b"],
            "model-00002-of-00002.safetensors"
        );

        // Three entries, two distinct shards, sorted and de-duplicated.
        assert_eq!(
            read_index_shard_files(&path).unwrap(),
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );
    }

    #[test]
    fn test_read_index_rejects_a_missing_weight_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&path, r#"{"metadata":{"total_size":0}}"#).unwrap();

        let err = read_index_shard_files(&path).unwrap_err().to_string();
        assert_eq!(
            err,
            format!(
                "Shard index has no \"weight_map\" object: {}",
                path.display()
            )
        );
    }

    #[test]
    fn test_read_index_rejects_an_empty_weight_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&path, r#"{"weight_map":{}}"#).unwrap();

        let err = read_index_shard_files(&path).unwrap_err().to_string();
        assert_eq!(
            err,
            format!("Shard index names no shard files: {}", path.display())
        );
    }

    #[test]
    fn test_read_index_rejects_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&path, "not json").unwrap();

        let err = read_index_shard_files(&path).unwrap_err().to_string();
        assert_eq!(
            err,
            format!("Shard index is not valid JSON: {}", path.display())
        );
    }
}
