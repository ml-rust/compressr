//! Carrying a GGUF input's header into a GGUF output.
//!
//! A GGUF re-conversion keeps every key the input carried: the tokenizer,
//! the chat template, the architecture's hyperparameters and any producer
//! keys such as `prism.hadamard.*`. The keys the writer sets itself are
//! the only ones left out.

use super::gguf::{GgufArray, GgufValue};
use anyhow::{bail, Result};
use boostr::format::gguf::{GgufMetadata, GgufValue as ReaderValue};

/// Keys the writer sets from the conversion itself, never copied.
///
/// `general.architecture` is the resolved architecture, which `--arch`
/// overrides; `general.file_type` names the output's format; the
/// quantization version is the writer's own.
pub const WRITER_OWNED_KEYS: &[&str] = &[
    "general.architecture",
    "general.file_type",
    "general.quantization_version",
];

/// Every key-value pair of `metadata` the writer does not own, sorted by key.
///
/// # Errors
/// An array mixes element types or nests another array. GGUF gives an array
/// one element type, so neither can be written back.
pub fn carry_gguf_metadata(metadata: &GgufMetadata) -> Result<Vec<(String, GgufValue)>> {
    let mut carried: Vec<(String, GgufValue)> = Vec::with_capacity(metadata.len());
    for (key, value) in metadata.iter() {
        if WRITER_OWNED_KEYS.contains(&key) {
            continue;
        }
        carried.push((key.to_string(), writer_value(key, value)?));
    }
    carried.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(carried)
}

/// The writer's value for one reader value.
fn writer_value(key: &str, value: &ReaderValue) -> Result<GgufValue> {
    Ok(match value {
        ReaderValue::Uint8(v) => GgufValue::Uint8(*v),
        ReaderValue::Int8(v) => GgufValue::Int8(*v),
        ReaderValue::Uint16(v) => GgufValue::Uint16(*v),
        ReaderValue::Int16(v) => GgufValue::Int16(*v),
        ReaderValue::Uint32(v) => GgufValue::Uint32(*v),
        ReaderValue::Int32(v) => GgufValue::Int32(*v),
        ReaderValue::Float32(v) => GgufValue::Float32(*v),
        ReaderValue::Bool(v) => GgufValue::Bool(*v),
        ReaderValue::String(v) => GgufValue::String(v.clone()),
        ReaderValue::Uint64(v) => GgufValue::Uint64(*v),
        ReaderValue::Int64(v) => GgufValue::Int64(*v),
        ReaderValue::Float64(v) => GgufValue::Float64(*v),
        ReaderValue::Array(items) => GgufValue::Array(writer_array(key, items)?),
    })
}

/// A typed writer array from reader elements that all share one type.
fn writer_array(key: &str, items: &[ReaderValue]) -> Result<GgufArray> {
    macro_rules! collect {
        ($variant:ident, $ty:ty, $get:expr) => {{
            let values: Vec<$ty> = items
                .iter()
                .map(|item| match item {
                    ReaderValue::$variant(v) => Ok($get(v)),
                    other => Err(mixed_array(key, other)),
                })
                .collect::<Result<_>>()?;
            GgufArray::$variant(values)
        }};
    }
    let Some(first) = items.first() else {
        // The reader keeps no element type for an empty array. Int32 is what
        // the common writers emit for an empty integer list.
        log::warn!("GGUF key '{key}' is an empty array; written back as an empty Int32 array");
        return Ok(GgufArray::Int32(Vec::new()));
    };
    Ok(match first {
        ReaderValue::Uint8(_) => collect!(Uint8, u8, |v: &u8| *v),
        ReaderValue::Int8(_) => collect!(Int8, i8, |v: &i8| *v),
        ReaderValue::Uint16(_) => collect!(Uint16, u16, |v: &u16| *v),
        ReaderValue::Int16(_) => collect!(Int16, i16, |v: &i16| *v),
        ReaderValue::Uint32(_) => collect!(Uint32, u32, |v: &u32| *v),
        ReaderValue::Int32(_) => collect!(Int32, i32, |v: &i32| *v),
        ReaderValue::Float32(_) => collect!(Float32, f32, |v: &f32| *v),
        ReaderValue::Bool(_) => collect!(Bool, bool, |v: &bool| *v),
        ReaderValue::String(_) => collect!(String, String, |v: &String| v.clone()),
        ReaderValue::Uint64(_) => collect!(Uint64, u64, |v: &u64| *v),
        ReaderValue::Int64(_) => collect!(Int64, i64, |v: &i64| *v),
        ReaderValue::Float64(_) => collect!(Float64, f64, |v: &f64| *v),
        ReaderValue::Array(_) => {
            bail!("GGUF key '{key}' holds an array of arrays, which the writer cannot store")
        }
    })
}

fn mixed_array(key: &str, item: &ReaderValue) -> anyhow::Error {
    anyhow::anyhow!("GGUF key '{key}' mixes element types: found {item:?} after another type")
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::format::gguf::Gguf;

    /// A GGUF header with no tensors and the given key-value pairs, written
    /// with the crate's own batch writer and read back through boostr.
    fn header_from(pairs: Vec<(&str, GgufValue)>) -> GgufMetadata {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("h.gguf");
        let mut writer = super::super::gguf::GgufWriter::new();
        for (key, value) in pairs {
            writer.add_metadata(key, value);
        }
        writer.write(&path).unwrap();
        Gguf::open(&path).unwrap().metadata().clone()
    }

    #[test]
    fn carries_every_key_the_writer_does_not_own() {
        let metadata = header_from(vec![
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("general.file_type", GgufValue::Uint32(143)),
            ("general.name", GgufValue::String("Hf".into())),
            ("prism.hadamard.version", GgufValue::Uint32(1)),
            (
                "prism.hadamard.sign_values",
                GgufValue::Array(GgufArray::Int32(vec![-1, 1, -1])),
            ),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(vec!["a".into(), "b".into()])),
            ),
            ("qwen35.rope.freq_base", GgufValue::Float32(1e7)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
        ]);
        let carried = carry_gguf_metadata(&metadata).unwrap();
        let keys: Vec<&str> = carried.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "general.name",
                "prism.hadamard.sign_values",
                "prism.hadamard.version",
                "qwen35.rope.freq_base",
                "tokenizer.ggml.add_bos_token",
                "tokenizer.ggml.tokens",
            ]
        );
        let signs = carried
            .iter()
            .find(|(k, _)| k == "prism.hadamard.sign_values")
            .map(|(_, v)| v)
            .unwrap();
        assert!(matches!(
            signs,
            GgufValue::Array(GgufArray::Int32(v)) if v == &[-1, 1, -1]
        ));
        let tokens = carried
            .iter()
            .find(|(k, _)| k == "tokenizer.ggml.tokens")
            .map(|(_, v)| v)
            .unwrap();
        assert!(matches!(
            tokens,
            GgufValue::Array(GgufArray::String(v)) if v == &["a", "b"]
        ));
    }
}
