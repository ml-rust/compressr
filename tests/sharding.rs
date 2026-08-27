//! Sharded SafeTensors input and output
//!
//! Covers the shard index on the way in (which files are loaded, and which are
//! deliberately not) and on the way out (when an index is written at all, and
//! whether it points at the shard that really holds each tensor).

use compressr::formats::safetensors::{SafeTensorsDType, SafeTensorsWriter};
use compressr::{ConversionConfig, ModelLoader, OutputFormat};
use std::collections::BTreeMap;
use std::path::Path;

/// Write one `.safetensors` file holding the given f32 tensors.
fn write_shard(path: &Path, tensors: &[(&str, Vec<f32>)]) {
    let mut writer = SafeTensorsWriter::new();
    for (name, values) in tensors {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer
            .add_tensor(*name, vec![values.len()], SafeTensorsDType::F32, bytes)
            .unwrap();
    }
    writer.write(path).unwrap();
}

/// Write a `model.safetensors.index.json` mapping each tensor to its shard.
fn write_index(dir: &Path, entries: &[(&str, &str)]) {
    let weight_map: BTreeMap<&str, &str> = entries.iter().copied().collect();
    let index = serde_json::json!({
        "metadata": { "total_size": 0 },
        "weight_map": weight_map,
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .unwrap();
}

/// Decode a tensor read back out of a loaded model into f32 values.
fn read_f32(model: &mut compressr::LoadedModel, name: &str) -> Vec<f32> {
    model
        .read_tensor_bytes(name)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ============================================================================
// Input: the index decides which files are the model
// ============================================================================

#[test]
fn test_index_loads_only_the_shards_it_names() {
    let dir = tempfile::tempdir().unwrap();

    write_shard(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("a", vec![1.0, 2.0])],
    );
    write_shard(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("b", vec![3.0])],
    );
    // A decoy the index does not name. Globbing the directory would absorb it.
    write_shard(
        &dir.path().join("adapter.safetensors"),
        &[
            ("lora.down.weight", vec![9.0]),
            ("lora.up.weight", vec![8.0]),
        ],
    );

    write_index(
        dir.path(),
        &[
            ("a", "model-00001-of-00002.safetensors"),
            ("b", "model-00002-of-00002.safetensors"),
        ],
    );

    let mut model = ModelLoader::new(dir.path()).load().unwrap();

    let mut names = model.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(model.tensor_count(), 2);

    assert_eq!(read_f32(&mut model, "a"), vec![1.0, 2.0]);
    assert_eq!(read_f32(&mut model, "b"), vec![3.0]);

    // The decoy's tensors are absent, by name and by lookup.
    assert!(model.tensor_info("lora.down.weight").is_none());
    assert!(model.tensor_info("lora.up.weight").is_none());
    assert_eq!(
        model
            .read_tensor_bytes("lora.up.weight")
            .unwrap_err()
            .to_string(),
        "Tensor not found: lora.up.weight"
    );

    // 3 f32 elements across the two real shards, 4 bytes each.
    assert_eq!(model.total_size_bytes(), 12);
}

#[test]
fn test_missing_shard_errors_naming_the_shard_and_the_index() {
    let dir = tempfile::tempdir().unwrap();

    write_shard(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("a", vec![1.0])],
    );
    // model-00002-of-00002.safetensors is deliberately never written.
    write_index(
        dir.path(),
        &[
            ("a", "model-00001-of-00002.safetensors"),
            ("b", "model-00002-of-00002.safetensors"),
        ],
    );

    // `LoadedModel` is not `Debug`, so `unwrap_err` is unusable here.
    let Err(err) = ModelLoader::new(dir.path()).load() else {
        panic!("a shard named by the index but missing must be an error");
    };
    let err = err.to_string();
    assert_eq!(
        err,
        format!(
            "Shard 'model-00002-of-00002.safetensors' named by {} is missing from {}",
            dir.path().join("model.safetensors.index.json").display(),
            dir.path().display()
        )
    );
}

#[test]
fn test_no_index_falls_back_to_the_glob() {
    let dir = tempfile::tempdir().unwrap();

    write_shard(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("a", vec![1.0, 2.0])],
    );
    write_shard(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("b", vec![3.0])],
    );
    // No index at all.

    let mut model = ModelLoader::new(dir.path()).load().unwrap();

    let mut names = model.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(read_f32(&mut model, "a"), vec![1.0, 2.0]);
    assert_eq!(read_f32(&mut model, "b"), vec![3.0]);
}

// ============================================================================
// Output: sharding, the index, and the round trip
// ============================================================================

#[test]
fn test_convert_shards_the_output_and_the_index_round_trips() {
    let input = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();

    // Three 16-element f32 tensors: 64 bytes each. At a 130-byte limit the
    // first two share a shard and the third opens the next one.
    let expected: Vec<(&str, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight",
            (0..16).map(|i| i as f32).collect(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            (0..16).map(|i| -(i as f32) * 0.5).collect(),
        ),
        (
            "lm_head.weight",
            (0..16).map(|i| i as f32 + 100.0).collect(),
        ),
    ];
    write_shard(&input.path().join("model.safetensors"), &expected);

    let out_dir = output.path().join("sharded");
    let stats = compressr::convert(
        ConversionConfig::new(
            input.path().to_string_lossy().to_string(),
            out_dir.to_string_lossy().to_string(),
            OutputFormat::Safetensors,
        )
        .with_max_shard_size(Some(130)),
    )
    .unwrap();

    assert_eq!(stats.tensor_count, 3);
    assert_eq!(stats.output_size, 3 * 64);

    // Two shards, no single-file weights, and an index beside them.
    assert!(!out_dir.join("model.safetensors").exists());
    assert!(out_dir.join("model-00001-of-00002.safetensors").is_file());
    assert!(out_dir.join("model-00002-of-00002.safetensors").is_file());

    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(index["metadata"]["total_size"], 192);

    // Every name maps to the shard that really carries it: check the mapped
    // file's own header, not just the index.
    for (name, _) in &expected {
        let shard = index["weight_map"][*name].as_str().unwrap_or_else(|| {
            panic!("index has no weight_map entry for {}", name);
        });
        let bytes = std::fs::read(out_dir.join(shard)).unwrap();
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        assert!(
            header.get(*name).is_some(),
            "index sends {} to {}, which does not contain it",
            name,
            shard
        );
    }
    // Deliberately NOT asserting which shard a given tensor lands in. That
    // depends on registration order and the size limit, so pinning it makes the
    // test brittle without making it stronger — the loop above already proves
    // the invariant that matters: every `weight_map` entry names the shard whose
    // header actually holds that tensor. Both shards must be used, though.
    let used: std::collections::BTreeSet<&str> = index["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        used.len(),
        2,
        "the limit must have forced two shards: {used:?}"
    );

    // Load the output back through the index: every name and value survives.
    let mut reloaded = ModelLoader::new(&out_dir).load().unwrap();
    assert_eq!(reloaded.tensor_count(), 3);

    let mut names = reloaded.tensor_names();
    names.sort();
    assert_eq!(
        names,
        vec![
            "lm_head.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.mlp.up_proj.weight".to_string(),
        ]
    );

    for (name, values) in &expected {
        assert_eq!(read_f32(&mut reloaded, name), *values, "tensor {}", name);
    }
}

#[test]
fn test_convert_that_fits_writes_one_file_and_no_index() {
    let input = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();

    write_shard(
        &input.path().join("model.safetensors"),
        &[("w", (0..16).map(|i| i as f32).collect())],
    );

    let out_dir = output.path().join("single");
    let stats = compressr::convert(
        ConversionConfig::new(
            input.path().to_string_lossy().to_string(),
            out_dir.to_string_lossy().to_string(),
            OutputFormat::Safetensors,
        )
        .with_max_shard_size(Some(5_000_000_000)),
    )
    .unwrap();

    assert_eq!(stats.tensor_count, 1);
    assert!(out_dir.join("model.safetensors").is_file());
    assert!(!out_dir.join("model.safetensors.index.json").exists());
    assert!(!out_dir.join("model-00001-of-00001.safetensors").exists());

    let mut reloaded = ModelLoader::new(&out_dir).load().unwrap();
    assert_eq!(reloaded.tensor_names(), vec!["w".to_string()]);
    assert_eq!(
        read_f32(&mut reloaded, "w"),
        (0..16).map(|i| i as f32).collect::<Vec<f32>>()
    );
}

#[test]
fn test_default_max_shard_size_is_five_decimal_gigabytes() {
    let config = ConversionConfig::new("input/".into(), "out/".into(), OutputFormat::Safetensors);
    assert_eq!(config.max_shard_size, Some(5_000_000_000));
    assert_eq!(compressr::DEFAULT_MAX_SHARD_SIZE, 5_000_000_000);
}
