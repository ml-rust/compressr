//! A measured task-metric sensitivity file: what it holds, and what it names.
//!
//! `PROFILES.md` Section 4 step 2 measures a task-metric delta per bump
//! candidate on a calibration set. compressr does not run models, so it never
//! produces that measurement itself. It reads one: a JSON-lines file holding a
//! single `baseline` record and one `sensitivity` record per tensor, written by
//! whatever harness ran the model. `--sensitivity` supplies it.
//!
//! This module parses that file, checks it was measured at the encoding the
//! profile bumps from, and derives the calibration identity the TCF records
//! point at. [`super::sensitivity_rank`] turns it into a bump set. Without the
//! flag neither runs, and the weight-error screen in [`super::allocation`] is
//! unchanged.
//!
//! # Why the encoding must match
//!
//! A delta says how much task damage one encoding did to one tensor. At
//! another encoding the damage is a different number and, across tensors, a
//! different order. A ranking therefore does not carry over, and a file
//! measured elsewhere is refused rather than reused.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use tcf_core::{hash_128, NativeEncoding};

use crate::formats::parse_tcf_encoding;

/// The measurement's identity, from the file's one `baseline` record.
///
/// Every field here is part of what the numbers mean. Two files differing in
/// any of them measured different experiments, so the calibration identity is
/// derived from all of them together.
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityBaseline {
    /// The checkpoint the measurement ran against, as the file spells it
    pub checkpoint: String,
    /// The evaluation manifest the metric was computed over
    pub dataset: String,
    /// The encoding each candidate was perturbed to, as the file spells it
    pub encoding: String,
    /// Rows of the manifest each measurement was taken over
    pub eval_rows: u32,
    /// The seed that fixed the row sample
    pub eval_seed: u64,
    /// The unperturbed task metric every delta is a difference from
    pub baseline_total: f64,
}

/// A parsed sensitivity file: its identity, and one delta per tensor.
#[derive(Debug, Clone)]
pub struct SensitivityMeasurement {
    baseline: SensitivityBaseline,
    deltas: HashMap<String, f64>,
}

/// Read a JSON-lines sensitivity file.
///
/// A `baseline` record and at least one `sensitivity` record are required.
/// `skipped` records, and any record kind this version does not know, are
/// ignored: the harness reports what it could not measure, and a tensor it
/// could not measure is simply absent from the ranking.
///
/// # Errors
/// A line is not JSON, a record is missing a field this reads, the file
/// carries no baseline or no sensitivity record, or one tensor is measured
/// twice.
pub fn parse_sensitivity(text: &str) -> Result<SensitivityMeasurement> {
    let mut baseline: Option<SensitivityBaseline> = None;
    let mut deltas: HashMap<String, f64> = HashMap::new();

    for (offset, line) in text.lines().enumerate() {
        let number = offset + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("Sensitivity file line {number} is not one JSON object"))?;
        let kind = value.get("record").and_then(Value::as_str).ok_or_else(|| {
            anyhow!("Sensitivity file line {number} has no \"record\" field naming its kind")
        })?;
        match kind {
            "baseline" => {
                if baseline.is_some() {
                    bail!(
                        "Sensitivity file line {number} is a second \"baseline\" record. One \
                         file measures one calibration, so two baselines name two experiments \
                         whose deltas do not compare."
                    );
                }
                baseline = Some(read_baseline(&value, number)?);
            }
            "sensitivity" => {
                let name = string_field(&value, "name", number)?;
                let delta = f64_field(&value, "delta", number)?;
                if deltas.insert(name.clone(), delta).is_some() {
                    bail!(
                        "Sensitivity file measures tensor '{name}' twice, so nothing names \
                         which delta is this tensor's."
                    );
                }
            }
            _ => {}
        }
    }

    let baseline = baseline.ok_or_else(|| {
        anyhow!(
            "Sensitivity file holds no \"baseline\" record, so nothing names the encoding, the \
             model or the row count the deltas were measured at."
        )
    })?;
    if deltas.is_empty() {
        bail!(
            "Sensitivity file holds no \"sensitivity\" record, so there is no measured tensor \
             to rank."
        );
    }
    Ok(SensitivityMeasurement { baseline, deltas })
}

impl SensitivityMeasurement {
    /// The measurement's identity.
    pub fn baseline(&self) -> &SensitivityBaseline {
        &self.baseline
    }

    /// Tensors the file carries a delta for.
    pub fn measured(&self) -> usize {
        self.deltas.len()
    }

    /// Every measured tensor name, ascending.
    pub fn measured_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.deltas.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// One tensor's measured task delta, `None` when the file has no record
    /// for that name.
    pub fn delta_for(&self, name: &str) -> Option<f64> {
        self.deltas.get(name).copied()
    }

    /// Check the measurement was taken at the encoding the profile bumps from.
    ///
    /// # Errors
    /// The baseline names an encoding compressr does not recognize, or one
    /// that is not `expected`.
    pub fn check_encoding(&self, expected: NativeEncoding) -> Result<()> {
        let measured = parse_tcf_encoding(&self.baseline.encoding).with_context(|| {
            format!(
                "Sensitivity file's baseline names encoding '{}', which compressr does not \
                 recognize",
                self.baseline.encoding
            )
        })?;
        if measured != expected {
            bail!(
                "Sensitivity file was measured at {measured:?} but the profile's base encoding \
                 is {expected:?}. A ranking measured at one encoding does not transfer to \
                 another: the delta is that encoding's damage on that tensor. Re-run the \
                 measurement at {expected:?}, or pick the profile whose base is {measured:?}."
            );
        }
        Ok(())
    }

    /// The string two files built from different calibrations differ in.
    ///
    /// Everything that changes what a delta means is in it: the model, the
    /// evaluation manifest, the encoding measured, the row count, the seed
    /// that fixed the rows, and the unperturbed metric they produced.
    pub fn calibration_identity(&self) -> String {
        format!(
            "tcf-sensitivity/1\nmodel={}\ndataset={}\nencoding={}\nrows={}\nseed={}\nbaseline={:e}",
            self.baseline.checkpoint,
            self.baseline.dataset,
            self.baseline.encoding,
            self.baseline.eval_rows,
            self.baseline.eval_seed,
            self.baseline.baseline_total
        )
    }

    /// BLAKE3-128 of [`SensitivityMeasurement::calibration_identity`], which
    /// is what `CalibrationRecord.evaluator_config_digest` carries.
    pub fn calibration_digest(&self) -> [u8; 16] {
        *hash_128(self.calibration_identity().as_bytes()).as_bytes()
    }

    /// The `calibration_id` every measured `TensorRecord` points at.
    ///
    /// The leading four digest bytes, so two files built from different
    /// calibrations carry different ids and the same calibration reproduces
    /// its own. `0` is the format's "no calibration" value, so a digest that
    /// lands there is moved off it.
    pub fn calibration_id(&self) -> u32 {
        let digest = self.calibration_digest();
        let id = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
        if id == 0 {
            1
        } else {
            id
        }
    }

    /// The evaluator name the calibration record interns.
    ///
    /// The last path component of the checkpoint the measurement ran against,
    /// which is the part naming the model revision. The directories above it
    /// describe one machine's filesystem and identify nothing.
    pub fn evaluator_name(&self) -> &str {
        self.baseline
            .checkpoint
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(self.baseline.checkpoint.as_str())
    }
}

/// Read the one `baseline` record.
fn read_baseline(value: &Value, number: usize) -> Result<SensitivityBaseline> {
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

    /// A two-tensor file at the encoding `TCF-BALANCED` and `TCF-COMPACT`
    /// share as their base.
    fn file() -> String {
        [
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4AS32D_T64","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#,
            r#"{"record":"sensitivity","name":"a.weight","elements":1024,"delta":0.5}"#,
            r#"{"record":"sensitivity","name":"b.weight","elements":1024,"delta":-0.25}"#,
            r#"{"record":"skipped","name":"c.weight","reason":"rank below 2"}"#,
            "",
        ]
        .join("\n")
    }

    #[test]
    fn a_file_parses_into_a_baseline_and_one_delta_per_tensor() {
        let measurement = parse_sensitivity(&file()).expect("parses");
        assert_eq!(measurement.measured(), 2);
        assert_eq!(measurement.delta_for("a.weight"), Some(0.5));
        assert_eq!(measurement.delta_for("b.weight"), Some(-0.25));
        // A `skipped` record is not a measurement.
        assert_eq!(measurement.delta_for("c.weight"), None);
        assert_eq!(measurement.baseline().eval_rows, 12);
        assert_eq!(measurement.evaluator_name(), "abcdef");
    }

    #[test]
    fn an_empty_file_is_refused() {
        let error = parse_sensitivity("   \n\n")
            .expect_err("no records")
            .to_string();
        assert!(error.contains("no \"baseline\" record"), "{error}");
    }

    #[test]
    fn a_baseline_with_no_tensor_is_refused() {
        let baseline = file().lines().take(1).collect::<Vec<&str>>().join("\n");
        let error = parse_sensitivity(&baseline)
            .expect_err("no tensors")
            .to_string();
        assert!(error.contains("no \"sensitivity\" record"), "{error}");
    }

    #[test]
    fn a_malformed_line_is_refused() {
        let error = parse_sensitivity("{not json}")
            .expect_err("not json")
            .to_string();
        assert!(error.contains("line 1 is not one JSON object"), "{error}");
    }

    #[test]
    fn a_tensor_measured_twice_is_refused() {
        let doubled = format!(
            "{}\n{}",
            file(),
            r#"{"record":"sensitivity","name":"a.weight","delta":0.9}"#
        );
        let error = parse_sensitivity(&doubled)
            .expect_err("two deltas")
            .to_string();
        assert!(
            error.contains("measures tensor 'a.weight' twice"),
            "{error}"
        );
    }

    /// The measurement is only a ranking of the encoding it was taken at.
    #[test]
    fn a_different_base_encoding_is_refused() {
        let measurement = parse_sensitivity(&file()).expect("parses");
        assert!(measurement
            .check_encoding(NativeEncoding::Q4AS32DT64)
            .is_ok());
        let error = measurement
            .check_encoding(NativeEncoding::Q8S32T64)
            .expect_err("a different encoding")
            .to_string();
        assert!(error.contains("Q4AS32DT64"), "{error}");
        assert!(error.contains("Q8S32T64"), "{error}");
        assert!(error.contains("does not transfer"), "{error}");
    }

    /// Two calibrations must be distinguishable, and one must reproduce.
    #[test]
    fn the_calibration_id_follows_the_measurement_identity() {
        let measurement = parse_sensitivity(&file()).expect("parses");
        let other = parse_sensitivity(&file().replace("\"eval_seed\":7", "\"eval_seed\":8"))
            .expect("parses");
        assert_ne!(measurement.calibration_id(), other.calibration_id());
        assert_ne!(measurement.calibration_id(), 0);
        assert_eq!(
            measurement.calibration_id(),
            parse_sensitivity(&file()).expect("parses").calibration_id()
        );
    }
}
