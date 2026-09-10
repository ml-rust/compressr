//! One line of a sensitivity file, read into a typed record.
//!
//! The file is JSON lines, and every line names its own kind in a `record`
//! field. Reading one line is mechanical, and it is here rather than in
//! [`super::sensitivity`] so that module holds the measurement and its
//! identity alone.
//!
//! # Why an encoding may be absent
//!
//! A file written before the ladder measured one encoding, named it once in
//! the baseline record, and left it off every tensor record. Such a record is
//! read with no encoding and resolved against the baseline afterwards, so the
//! two-way file and the per-encoding file take the same path.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use super::sensitivity::SensitivityBaseline;
use super::sensitivity_options::MeasuredOption;

/// One line of the file.
#[derive(Debug)]
pub enum SensitivityRecord {
    /// The one `baseline` record, naming the calibration
    Baseline(SensitivityBaseline),
    /// A `sensitivity` or `skipped` record, for one (tensor, encoding) pair
    Measured {
        /// The tensor, as the measuring harness spells it
        name: String,
        /// The pair's measurement, before its encoding is resolved
        option: PendingOption,
    },
    /// A record kind this version does not know, which is ignored
    Unknown,
}

/// One measured pair whose encoding may still be the baseline's.
#[derive(Debug)]
pub struct PendingOption {
    /// The encoding the record names, `None` when it names none
    pub encoding: Option<String>,
    /// The measured delta, `0.0` for a skipped pair
    pub delta: f64,
    /// Payload bytes at that encoding, when the record carries them
    pub bytes: Option<u64>,
    /// True for a `skipped` record
    pub skipped: bool,
}

impl PendingOption {
    /// Fill an absent encoding from the file's baseline record.
    pub fn resolve(self, baseline_encoding: &str) -> MeasuredOption {
        MeasuredOption {
            encoding: self
                .encoding
                .unwrap_or_else(|| baseline_encoding.to_string()),
            delta: self.delta,
            bytes: self.bytes,
            skipped: self.skipped,
        }
    }
}

/// Read one line's record.
///
/// # Errors
/// The line names no kind, or a kind this reads is missing a field it needs.
pub fn read_record(value: &Value, number: usize) -> Result<SensitivityRecord> {
    let kind = value.get("record").and_then(Value::as_str).ok_or_else(|| {
        anyhow!("Sensitivity file line {number} has no \"record\" field naming its kind")
    })?;
    match kind {
        "baseline" => Ok(SensitivityRecord::Baseline(read_baseline(value, number)?)),
        "sensitivity" => Ok(SensitivityRecord::Measured {
            name: string_field(value, "name", number)?,
            option: PendingOption {
                encoding: optional_string(value, "encoding"),
                delta: f64_field(value, "delta", number)?,
                bytes: optional_u64(value, "bytes")
                    .or_else(|| optional_u64(value, "payload_bytes")),
                skipped: false,
            },
        }),
        // A skipped pair carries no delta and no byte count: the harness is
        // saying this tensor cannot take this encoding at all.
        "skipped" => Ok(SensitivityRecord::Measured {
            name: string_field(value, "name", number)?,
            option: PendingOption {
                encoding: optional_string(value, "encoding"),
                delta: 0.0,
                bytes: None,
                skipped: true,
            },
        }),
        _ => Ok(SensitivityRecord::Unknown),
    }
}

/// Read the one `baseline` record.
///
/// # Errors
/// A field the baseline is read for is missing or is the wrong kind.
pub fn read_baseline(value: &Value, number: usize) -> Result<SensitivityBaseline> {
    let eval_rows = u64_field(value, "eval_rows", number)?;
    Ok(SensitivityBaseline {
        checkpoint: string_field(value, "ckpt", number)?,
        dataset: string_field(value, "manifest", number)?,
        encoding: string_field(value, "encoding", number)?,
        eval_rows: u32::try_from(eval_rows).with_context(|| {
            format!("Sensitivity file line {number} reports more eval rows than TCF records")
        })?,
        eval_seed: u64_field(value, "eval_seed", number)?,
        baseline_total: f64_field(value, "baseline_total", number)?,
    })
}

/// One string field, or an error naming it and its line.
fn string_field(value: &Value, field: &str, number: usize) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Sensitivity file line {number} has no string field \"{field}\""))
}

/// One string field when the record carries it.
fn optional_string(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_string)
}

/// One unsigned integer field when the record carries it.
fn optional_u64(value: &Value, field: &str) -> Option<u64> {
    value.get(field).and_then(Value::as_u64)
}

/// One finite float field, or an error naming it and its line.
fn f64_field(value: &Value, field: &str, number: usize) -> Result<f64> {
    let read = value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("Sensitivity file line {number} has no number field \"{field}\""))?;
    if !read.is_finite() {
        bail!("Sensitivity file line {number} has a non-finite \"{field}\"");
    }
    Ok(read)
}

/// One unsigned integer field, or an error naming it and its line.
fn u64_field(value: &Value, field: &str, number: usize) -> Result<u64> {
    value.get(field).and_then(Value::as_u64).ok_or_else(|| {
        anyhow!("Sensitivity file line {number} has no unsigned integer field \"{field}\"")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> Value {
        serde_json::from_str(text).expect("a JSON object")
    }

    /// A record naming its own encoding keeps it; one that names none takes
    /// the baseline's.
    #[test]
    fn an_absent_encoding_resolves_to_the_baseline() {
        let bare = json(r#"{"record":"sensitivity","name":"a.weight","delta":0.5}"#);
        let SensitivityRecord::Measured { name, option } = read_record(&bare, 2).expect("reads")
        else {
            panic!("a sensitivity record");
        };
        assert_eq!(name, "a.weight");
        let resolved = option.resolve("Q4AS32D_T64");
        assert_eq!(resolved.encoding, "Q4AS32D_T64");
        assert_eq!(resolved.delta, 0.5);
        assert_eq!(resolved.bytes, None);
        assert!(resolved.is_measured());

        let named = json(
            r#"{"record":"sensitivity","name":"a.weight","encoding":"Q6S16D_T64","delta":0.1,"bytes":210}"#,
        );
        let SensitivityRecord::Measured { option, .. } = read_record(&named, 3).expect("reads")
        else {
            panic!("a sensitivity record");
        };
        let resolved = option.resolve("Q4AS32D_T64");
        assert_eq!(resolved.encoding, "Q6S16D_T64");
        assert_eq!(resolved.bytes, Some(210));
    }

    /// A skipped pair is a refusal, not a measurement of zero damage.
    #[test]
    fn a_skipped_record_carries_no_delta() {
        let value = json(
            r#"{"record":"skipped","name":"c.weight","encoding":"Q6S16D_T64","reason":"row not a whole number of tiles"}"#,
        );
        let SensitivityRecord::Measured { option, .. } = read_record(&value, 4).expect("reads")
        else {
            panic!("a skipped record");
        };
        let resolved = option.resolve("Q4AS32D_T64");
        assert!(resolved.skipped);
        assert!(!resolved.is_measured());
        assert_eq!(resolved.encoding, "Q6S16D_T64");
    }

    #[test]
    fn a_record_with_no_kind_is_refused() {
        let error = read_record(&json(r#"{"name":"a.weight"}"#), 7)
            .expect_err("no kind")
            .to_string();
        assert!(error.contains("line 7"), "{error}");
        assert!(error.contains("\"record\""), "{error}");
    }

    #[test]
    fn an_unknown_kind_is_ignored() {
        let value = json(r#"{"record":"note","text":"anything"}"#);
        assert!(matches!(
            read_record(&value, 1).expect("reads"),
            SensitivityRecord::Unknown
        ));
    }

    #[test]
    fn a_record_with_no_delta_is_refused() {
        let value = json(r#"{"record":"sensitivity","name":"a.weight"}"#);
        let error = read_record(&value, 5).expect_err("no delta").to_string();
        assert!(error.contains("line 5"), "{error}");
        assert!(error.contains("\"delta\""), "{error}");
    }
}
