//! [`TcfFormatWriter`]: the [`FormatWriter`] TCF implementation.
//!
//! # Two passes over the source, one tensor in flight
//!
//! TCF fixes a tensor's encoding in its directory entry, and the directory is
//! written before any payload. A value-dependent fallback — a NaN, or a scale
//! that underflows binary16 — is only knowable once the quantizer has seen the
//! values, which is after `ConversionConfig::validate` and after registration
//! would otherwise be over. So this writer settles each plan in the conversion
//! loop, from the bytes that loop already read, and produces payloads
//! afterwards from a reader of its own:
//!
//! 1. [`FormatWriter::write_tensor_data`] quantizes the tensor, keeps the
//!    verdict and the screen error, and appends the tiles to a temporary
//!    payload spill. A tensor the quantizer rejects is re-planned as raw with
//!    the reason it gave, and the run continues.
//! 2. [`FormatWriter::finalize`] runs the profile allocation, registers every
//!    record, and then lets `TcfWriter::finish_streaming` pull one payload at
//!    a time, write it at its precomputed offset, and drop it before asking
//!    for the next.
//!
//! Registration waits for `finalize` because a profile's bump set is a
//! comparison across tensors: which tensors carry the bump encoding is unknown
//! until the last one has been screened.
//!
//! # The payload each tensor is written from
//!
//! Pass 1 already computed the winning tiles for every tensor it settles, so
//! pass 2 asks [`super::spill`] for them rather than quantizing the same
//! values a second time. Three tensors miss that and are read from the source
//! again: one stored raw, whose payload is its own source bytes; one a profile
//! bumped, whose spilled payload is at the base encoding its record no longer
//! carries; and one whose spill could not be written or read, which costs time
//! and never correctness. The bytes are identical either way — `quantize` is
//! deterministic, and `unpack` is `tcf-core`'s own inverse of the `pack` that
//! spilled them.
//!
//! At most one tensor's values are resident at any point, in both passes, and
//! the spill holds the rest on disk. That is the one-tensor-in-flight bound
//! the alternative — holding every payload until the directory is complete,
//! which is what the older `add_*_tensor` path does — gives up.
//!
//! Nothing is created at the output path before pass 2, so a tensor compressr
//! cannot represent fails with no output file at all rather than truncating
//! one.

use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use tcf_core::{Encoding, LogicalTile, NativeEncoding, Payload, TcfError, TcfWriter};

use super::allocate::allocate;
use super::encoding::{values_f32, TcfOptions, TensorShape, TCF_ENCODING_NAMES};
use super::ladder::ladder_encodings;
use super::parallel::quantize_named;
use super::plan::{plan_one, reason_for, settle, PlannedTensor, Registration};
use super::record::{
    calibration_record, contract_record, module_record, tensor_record, TensorSensitivity,
    SENSITIVITY_SEED_COUNT,
};
use super::spill::PayloadSpill;
use crate::formats::FormatWriter;
use crate::loader::{LoadedModel, ModelLoader, UnifiedDType};
use crate::policy::{SensitivityMeasurement, TcfPlan};
use crate::quantization::QuantType;

/// What every measured `TensorRecord` copies from the declared calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CalibrationFacts {
    /// The `CalibrationRecord` this file declares
    id: u32,
    /// Evaluation rows every delta was measured over
    samples: u32,
}

/// TCF format writer.
///
/// The trait's call order is GGUF's — `initialize`, every `register_tensor`,
/// then one `write_tensor_data` per tensor, then `finalize`. This writer holds
/// the records that order produces and writes the file in `finalize`; see the
/// module documentation for why the payloads come from a second read.
pub struct TcfFormatWriter {
    options: Option<TcfOptions>,
    source_path: Option<String>,
    module_name: String,
    path: Option<PathBuf>,
    registrations: Vec<Registration>,
    planned: Vec<PlannedTensor>,
    /// Each tensor's shape in both the forms TCF asks for, filled by
    /// [`TcfFormatWriter::register_all`] and read again by the payload pass
    shapes: Vec<TensorShape>,
    /// Pass 1's payloads, `None` once a spill error has sent every remaining
    /// tensor back to the quantizer
    spill: Option<PayloadSpill>,
    /// The measured task-metric sensitivity `--sensitivity` supplied
    ///
    /// `None` runs the weight-error screen and writes no sensitivity field.
    sensitivity: Option<SensitivityMeasurement>,
    /// Every encoding the measurement scores, which a ladder allocation may
    /// reach and which each tensor is therefore screened against
    ///
    /// Empty without `--sensitivity`, and empty for a file measured at one
    /// encoding: that file allocates two ways and reaches nothing but the
    /// profile's base and bump.
    ladder: Vec<NativeEncoding>,
    /// The calibration this file declares, filled by
    /// [`TcfFormatWriter::declare_calibration`] for a measured run only
    calibration: Option<CalibrationFacts>,
    writer: Option<TcfWriter>,
}

impl TcfFormatWriter {
    /// A writer with no encoding chosen yet.
    ///
    /// [`FormatWriter::initialize`] errors until both
    /// [`TcfFormatWriter::with_options`] and [`TcfFormatWriter::with_source`]
    /// have been called.
    pub fn new() -> Self {
        Self {
            options: None,
            source_path: None,
            module_name: "model".to_string(),
            path: None,
            registrations: Vec::new(),
            planned: Vec::new(),
            shapes: Vec::new(),
            spill: None,
            sensitivity: None,
            ladder: Vec::new(),
            calibration: None,
            writer: None,
        }
    }

    /// Set the encoding to quantize into and the module's preferred encoding.
    pub fn with_options(mut self, options: TcfOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Set the measured task-metric sensitivity the bump set is ranked by.
    ///
    /// Without this the ranking is the weight-error screen and every
    /// sensitivity field stays unset. See [`crate::policy::sensitivity`].
    pub fn with_sensitivity(mut self, measurement: SensitivityMeasurement) -> Self {
        self.ladder = ladder_encodings(&measurement);
        self.sensitivity = Some(measurement);
        self
    }

    /// Set the input checkpoint the payload pass reads from.
    pub fn with_source(mut self, input_path: &str) -> Self {
        self.source_path = Some(input_path.to_string());
        self
    }

    /// Set the name of the single root module every tensor belongs to.
    pub fn with_module_name(mut self, name: &str) -> Self {
        self.module_name = name.to_string();
        self
    }

    /// Every tensor's settled plan, in directory order.
    pub fn planned(&self) -> &[PlannedTensor] {
        &self.planned
    }

    /// The chosen options, or an error naming the flags that set them.
    fn options(&self) -> Result<TcfOptions> {
        self.options.ok_or_else(|| {
            anyhow!(
                "TCF output needs an encoding: pass --tcf-profile, or --tcf-encoding with \
                 --tcf-preferred-encoding. Accepted encodings: {}",
                TCF_ENCODING_NAMES.join(", ")
            )
        })
    }

    /// Declare the calibration a measured run read, and remember what every
    /// measured tensor copies from it.
    ///
    /// Does nothing without `--sensitivity`: a run that measured nothing
    /// declares no calibration, and its tensors keep `calibration_id` zero.
    ///
    /// # Errors
    /// The writer was not initialized, or the string table refused a name.
    fn declare_calibration(&mut self) -> Result<()> {
        let Some(measurement) = self.sensitivity.clone() else {
            return Ok(());
        };
        let dataset = measurement.baseline().dataset.clone();
        let evaluator = measurement.evaluator_name().to_string();
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;
        let dataset_name = writer
            .intern(&dataset)
            .map_err(|e| anyhow!(e))
            .context("Failed to intern the sensitivity file's dataset name")?;
        let evaluator_name = writer
            .intern(&evaluator)
            .map_err(|e| anyhow!(e))
            .context("Failed to intern the sensitivity file's evaluator name")?;
        let record = calibration_record(&measurement, dataset_name, evaluator_name);
        let facts = CalibrationFacts {
            id: record.calibration_id,
            samples: record.sample_count,
        };
        writer
            .add_calibration(record)
            .map_err(|e| anyhow!(e))
            .context("Failed to declare the sensitivity file's calibration")?;
        self.calibration = Some(facts);
        Ok(())
    }

    /// Register every settled record with `tcf-core`, in directory order.
    ///
    /// Each tensor's [`TensorShape`] is derived here and kept, so the payload
    /// pass reads the same value rather than deriving it again.
    fn register_all(&mut self) -> Result<()> {
        let options = self.options()?;
        self.shapes.clear();
        for index in 0..self.planned.len() {
            let plan = self
                .planned
                .get(index)
                .cloned()
                .ok_or_else(|| anyhow!("TCF plan {index} vanished before registration"))?;
            let shape = self
                .registrations
                .get(index)
                .map(|registration| registration.shape.clone())
                .ok_or_else(|| anyhow!("TCF planned tensor {index} was never registered"))?;

            let name = plan.name.as_str();
            let shape = TensorShape::of(&shape, name)?;
            self.shapes.push(shape);
            let (dims, rank) = (shape.dims(), shape.rank());
            let tensor_id =
                u32::try_from(index).context("More tensors than a TCF file can hold")?;
            let reason = reason_for(&plan, options);
            // A delta with no calibration behind it is a number with no
            // authority (Section 10.5.1), so the two are written together or
            // not at all.
            let measured =
                plan.sensitivity_delta
                    .zip(self.calibration)
                    .map(|(delta, calibration)| TensorSensitivity {
                        calibration_id: calibration.id,
                        delta,
                        samples: calibration.samples,
                        seed_count: SENSITIVITY_SEED_COUNT,
                    });

            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;
            let interned = writer
                .intern(name)
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Failed to intern tensor name '{name}'"))?;
            let record = tensor_record(
                tensor_id,
                interned,
                plan.encoding,
                reason,
                dims,
                rank,
                measured,
            );
            let registered = match plan.encoding {
                Encoding::Native(_) => writer.register_quantized_tensor(record),
                _ => writer.register_raw_tensor(record),
            };
            registered
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Failed to register tensor '{name}'"))?;
        }
        Ok(())
    }

    /// Append pass 1's payload for the tensor just settled, keeping one spill
    /// slot per tensor whether or not there was anything to append.
    ///
    /// A write that fails drops the spill and is reported, never raised: pass
    /// 2 then quantizes again and produces the same bytes.
    fn spill_payload(
        &mut self,
        tiles: Option<&[LogicalTile]>,
        encoding: NativeEncoding,
        name: &str,
    ) {
        let Some(spill) = self.spill.as_mut() else {
            return;
        };
        let appended = match tiles {
            Some(tiles) => spill.push(tiles, encoding),
            None => {
                spill.push_absent();
                Ok(())
            }
        };
        if let Err(error) = appended {
            log::warn!(
                "TCF: the payload spill stopped at tensor '{name}' ({error}); the payload pass \
                 quantizes again instead"
            );
            self.spill = None;
        }
    }

    /// Produce one tensor's payload, from the spill when pass 1 left one
    /// there and from the source checkpoint otherwise.
    ///
    /// The payload is returned by value and dropped by `tcf-core` as soon as
    /// it is written, so one tensor is resident at a time.
    fn payload(
        source: &str,
        model: &mut Option<LoadedModel>,
        spill: &mut Option<PayloadSpill>,
        plan: &PlannedTensor,
        shape: &TensorShape,
        index: usize,
    ) -> Result<Payload> {
        let name = plan.name.as_str();
        // The plan's own encoding drives this, not the run's base encoding: a
        // profile bumps some tensors, and quantizing one at the base here
        // would write a payload its record does not describe.
        let encoding = match plan.encoding {
            Encoding::Native(encoding) => encoding,
            // A raw tensor keeps its source element type, so its bytes are
            // the payload.
            _ => return Ok(Payload::Raw(Self::reread(source, model, name)?)),
        };
        if let Some(tiles) = Self::spilled(spill, plan, encoding, index) {
            return Ok(Payload::Quantized(tiles));
        }
        let bytes = Self::reread(source, model, name)?;
        let values = values_f32(&bytes, plan.dtype, name)?;
        let tiles = quantize_named(
            &values,
            shape.extent(),
            shape.rank(),
            encoding.layout(),
            name,
        )?;
        Ok(Payload::Quantized(tiles))
    }

    /// The payload pass 1 spilled for this tensor, if it is the one the
    /// record now needs.
    ///
    /// A spill that cannot be read is dropped and reported, never raised: the
    /// quantizer is the source of truth and produces the same bytes, so a
    /// broken temporary file costs a conversion time and nothing else.
    fn spilled(
        spill: &mut Option<PayloadSpill>,
        plan: &PlannedTensor,
        encoding: NativeEncoding,
        index: usize,
    ) -> Option<Vec<LogicalTile>> {
        // Bound before the match so the `&mut` ends here: an arm clears
        // `spill`.
        let read = spill.as_mut().map(|store| store.take(index, encoding));
        match read {
            None => None,
            Some(Ok(tiles)) => tiles,
            Some(Err(error)) => {
                log::warn!(
                    "TCF: the payload spill is unusable ({error}); tensor '{}' and every \
                     tensor after it is quantized again",
                    plan.name
                );
                *spill = None;
                None
            }
        }
    }

    /// Whether any tensor's payload still has to come from the source
    /// checkpoint.
    ///
    /// True for a raw tensor, whose payload is its own source bytes, and for
    /// one a profile bumped, whose spilled payload is at the base encoding.
    fn needs_source(&self, spill: Option<&PayloadSpill>) -> bool {
        self.planned
            .iter()
            .enumerate()
            .any(|(index, plan)| match plan.encoding {
                Encoding::Native(encoding) => {
                    !spill.is_some_and(|spill| spill.holds(index, encoding))
                }
                _ => true,
            })
    }

    /// Re-open the input checkpoint the payload pass reads from.
    fn open_source(source: &str) -> Result<LoadedModel> {
        ModelLoader::new(source)
            .load()
            .with_context(|| format!("Failed to re-open the input checkpoint: {source}"))
    }

    /// Read one tensor's source bytes, opening the checkpoint on first need.
    fn reread(source: &str, model: &mut Option<LoadedModel>, name: &str) -> Result<Vec<u8>> {
        if model.is_none() {
            *model = Some(Self::open_source(source)?);
        }
        let model = model
            .as_mut()
            .ok_or_else(|| anyhow!("The TCF source checkpoint vanished after it was opened"))?;
        model
            .read_tensor_bytes(name)
            .with_context(|| format!("Failed to re-read tensor '{name}'"))
    }

    /// Log the bits per weight the finished file achieved.
    ///
    /// Measured from the bytes on disk over every tensor's element count, raw
    /// tensors and container overhead included. A profile's ladder bpw is a
    /// target computed under `PROFILES.md` Section 4's own allocation, so the
    /// two numbers are never claimed to be equal.
    fn report_bpw(&self, path: &Path) -> Result<()> {
        let bytes = std::fs::metadata(path)
            .with_context(|| format!("Failed to measure {}", path.display()))?
            .len();
        let elements: u64 = self
            .planned
            .iter()
            .fold(0u64, |acc, plan| acc.saturating_add(plan.elements));
        if elements == 0 {
            return Ok(());
        }
        let bpw = bytes as f64 * 8.0 / elements as f64;
        let target = match self.options.and_then(|options| options.profile) {
            Some(profile) => format!(
                ", against the {} ladder target of {:.2} bpw, which assumes its own allocation",
                profile.name, profile.ladder_bpw
            ),
            None => String::new(),
        };
        log::info!("TCF: {bytes} bytes over {elements} weights, {bpw:.2} bpw achieved{target}");
        Ok(())
    }
}

impl Default for TcfFormatWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatWriter for TcfFormatWriter {
    /// Record the destination and open a `tcf-core` writer holding one root
    /// module and the one activation contract every record names.
    ///
    /// Nothing is created on disk here: the file is written once in
    /// [`FormatWriter::finalize`].
    fn initialize(&mut self, path: &Path) -> Result<()> {
        let options = self.options()?;
        if self.source_path.is_none() {
            bail!(
                "TCF output needs the input checkpoint path: the payload pass re-reads each \
                 tensor rather than holding every payload in memory."
            );
        }

        let mut writer = TcfWriter::new();
        let name = writer.intern(&self.module_name).map_err(|e| anyhow!(e))?;
        // Section 3: every `activation_contract_id` must resolve to a
        // `ContractRecord` in this file, so the contract is added before any
        // record that names it.
        writer
            .add_contract(contract_record())
            .map_err(|e| anyhow!(e))?;
        writer
            .add_module(module_record(name, options))
            .map_err(|e| anyhow!(e))?;

        // Beside the output, so the spill lands on the filesystem already
        // sized for the file being written. A directory that refuses it is no
        // reason to refuse the conversion: every tensor is then quantized a
        // second time, as it was before the spill existed.
        self.spill = match PayloadSpill::create(path) {
            Ok(spill) => Some(spill),
            Err(error) => {
                log::warn!(
                    "TCF: no payload spill ({error}); every tensor is quantized twice instead"
                );
                None
            }
        };

        self.path = Some(path.to_path_buf());
        self.writer = Some(writer);
        Ok(())
    }

    /// Remember one tensor. Its encoding is settled in
    /// [`FormatWriter::write_tensor_data`], once its values are available.
    fn register_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: UnifiedDType,
        quant_type: Option<QuantType>,
    ) -> Result<()> {
        if let Some(quant_type) = quant_type {
            bail!(
                "Tensor '{name}' was registered as {quant_type:?}, a GGUF block format. TCF \
                 quantizes from full-precision values, selected with --tcf-encoding; a GGML \
                 block would have to be dequantized and quantized a second time."
            );
        }
        self.registrations.push(Registration {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype,
        });
        Ok(())
    }

    /// Settle and screen the next tensor, spill the payload that screen
    /// produced, then drop its values.
    fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        let index = self.planned.len();
        let registration = self
            .registrations
            .get(index)
            .ok_or_else(|| {
                anyhow!(
                    "TCF writer received data for tensor {index} but only {} were registered",
                    self.registrations.len()
                )
            })?
            .clone();
        let options = self.options()?;
        let settled = settle(options, &registration, data, &self.ladder)?;
        self.spill_payload(
            settled.tiles.as_deref(),
            options.encoding,
            &registration.name,
        );
        self.planned.push(plan_one(
            options,
            &registration,
            settled.plan,
            settled.screen,
        )?);
        Ok(())
    }

    /// Allocate, register every record, then write the file one payload at a
    /// time.
    fn finalize(&mut self) -> Result<()> {
        if self.writer.is_none() {
            return Ok(());
        }
        if self.planned.len() != self.registrations.len() {
            bail!(
                "TCF writer has {} registered tensors but {} payloads",
                self.registrations.len(),
                self.planned.len()
            );
        }
        let mut counts = crate::policy::PlanCounts::default();
        for plan in &self.planned {
            counts.record_tcf(match plan.fallback {
                None => TcfPlan::Quantize,
                Some(reason) => TcfPlan::Raw(reason),
            });
        }
        log::info!("TCF: {counts}");

        let options = self.options()?;
        allocate(&mut self.planned, options, self.sensitivity.as_ref())?;
        self.declare_calibration()?;
        self.register_all()?;
        let writer = self
            .writer
            .take()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;

        let path = self
            .path
            .clone()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;
        let source = self
            .source_path
            .clone()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;

        let mut spill = self.spill.take();
        // Bound before the match so the `&mut` ends here: the arms reassign
        // `spill`.
        let sealed = spill.as_mut().map(PayloadSpill::seal);
        match sealed {
            Some(Ok(())) => {
                log::info!("TCF: the payload pass reads pass 1's payloads back from a spill")
            }
            Some(Err(error)) => {
                log::warn!(
                    "TCF: the payload spill could not be sealed ({error}); every tensor is \
                     quantized again instead"
                );
                spill = None;
            }
            None => {}
        }

        // Opened up front when any tensor needs it, so a source that has gone
        // away fails before the output file exists rather than truncating it.
        // A run whose every payload is spilled never touches the checkpoint
        // again, and never reparses its header.
        let mut model = if self.needs_source(spill.as_ref()) {
            Some(Self::open_source(&source)?)
        } else {
            None
        };
        let file = std::fs::File::create(&path)
            .with_context(|| format!("Failed to create {}", path.display()))?;
        let mut sink = BufWriter::new(file);

        let planned = &self.planned;
        let shapes = &self.shapes;
        // A `produce` error must reach the caller as itself: TcfError has no
        // room for one, so the real error is kept here and raised below.
        let mut failure: Option<anyhow::Error> = None;
        let result = writer.finish_streaming(&mut sink, |index| {
            let produced = match (planned.get(index), shapes.get(index)) {
                (Some(plan), Some(shape)) => {
                    Self::payload(&source, &mut model, &mut spill, plan, shape, index)
                }
                _ => Err(anyhow!(
                    "TCF asked for tensor {index}, which was never planned"
                )),
            };
            match produced {
                Ok(payload) => Ok(payload),
                Err(error) => {
                    let message = error.to_string();
                    failure = Some(error);
                    Err(TcfError::Io {
                        kind: std::io::ErrorKind::Other,
                        message,
                    })
                }
            }
        });

        if let Some(error) = failure {
            return Err(error);
        }
        result
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to write {}", path.display()))?;
        // `finish_streaming` never flushes: the buffer is this writer's.
        sink.flush()
            .with_context(|| format!("Failed to flush {}", path.display()))?;
        self.report_bpw(&path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{parse_tcf_profile, SafeTensorsFormatWriter};
    use crate::policy::FallbackReason;
    use tcf_core::{NativeEncoding, RawEncoding, TcfFile};

    fn options() -> TcfOptions {
        TcfOptions::uniform(NativeEncoding::Q4S32T64, NativeEncoding::Q4S32T64)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A writer initialized against a path it never writes to: every test but
    /// the round-trip stops before `finalize`, and `initialize` creates
    /// nothing.
    fn planned_writer() -> TcfFormatWriter {
        let mut writer = TcfFormatWriter::new()
            .with_options(options())
            .with_source("checkpoint");
        writer
            .initialize(Path::new("unused.tcf"))
            .expect("initializes");
        writer
    }

    fn plan_one(writer: &mut TcfFormatWriter, shape: &[usize], values: &[f32]) -> PlannedTensor {
        writer
            .register_tensor("w", shape, UnifiedDType::F32, None)
            .expect("registers");
        writer
            .write_tensor_data(&f32_bytes(values))
            .expect("settles a plan instead of failing");
        writer.planned().first().expect("one plan").clone()
    }

    #[test]
    fn initialize_without_an_encoding_names_the_flags() {
        let mut writer = TcfFormatWriter::new().with_source("checkpoint");
        let err = writer
            .initialize(Path::new("out.tcf"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--tcf-encoding"), "{err}");
    }

    #[test]
    fn initialize_without_a_source_says_why_it_needs_one() {
        let mut writer = TcfFormatWriter::new().with_options(options());
        let err = writer
            .initialize(Path::new("out.tcf"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("input checkpoint path"), "{err}");
    }

    /// A NaN is invisible to every shape check, so the tensor is re-planned
    /// when the quantizer rejects it and the run continues.
    #[test]
    fn a_nonfinite_tensor_is_replanned_as_raw() {
        let mut writer = planned_writer();
        let mut values = vec![1.0f32; 64];
        values[7] = f32::NAN;
        let plan = plan_one(&mut writer, &[1, 64], &values);

        assert_eq!(plan.fallback, Some(FallbackReason::SourceNonfinite));
        assert_eq!(plan.encoding, Encoding::Raw(RawEncoding::F32));
    }

    #[test]
    fn a_finite_tensor_is_quantized_with_no_reason() {
        let mut writer = planned_writer();
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let plan = plan_one(&mut writer, &[1, 64], &values);

        assert_eq!(plan.fallback, None);
        assert_eq!(plan.encoding, Encoding::Native(NativeEncoding::Q4S32T64));
    }

    /// A row of 96 is not a whole number of 64-element tiles.
    #[test]
    fn a_misaligned_row_is_replanned_as_raw() {
        let mut writer = planned_writer();
        let plan = plan_one(&mut writer, &[2, 96], &vec![0.5f32; 192]);
        assert_eq!(plan.fallback, Some(FallbackReason::RowLength));
    }

    #[test]
    fn a_rank_1_tensor_is_replanned_as_raw() {
        let mut writer = planned_writer();
        let plan = plan_one(&mut writer, &[64], &vec![0.5f32; 64]);
        assert_eq!(plan.fallback, Some(FallbackReason::Rank));
    }

    /// A GGUF block type cannot reach this writer: TCF quantizes from
    /// full-precision values, and re-quantizing a GGML block quantizes twice.
    #[test]
    fn refuses_a_gguf_quantization_type() {
        let mut writer = planned_writer();
        let err = writer
            .register_tensor("w", &[1, 64], UnifiedDType::F32, Some(QuantType::Q4K))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("quantizes from full-precision values"),
            "{err}"
        );
    }

    /// A ramp spreads evenly over the reconstruction levels. Every value of
    /// [`off_grid`] but the two endpoints sits halfway between two of them,
    /// which is the worst place a value can sit.
    fn ramp() -> Vec<f32> {
        (0..64).map(|i| i as f32).collect()
    }

    /// 62 values half a step away from any reconstruction level, between two
    /// endpoints that fix the range.
    fn off_grid() -> Vec<f32> {
        let mut values = vec![0.5f32; 64];
        values[0] = 0.0;
        values[1] = 15.0;
        values
    }

    /// The screen ranks by how badly the base encoding reconstructs a tensor,
    /// so the evenly spread ramp scores below the off-grid tensor.
    #[test]
    fn the_screen_scores_an_off_grid_tensor_above_a_ramp() {
        let mut writer = planned_writer();
        writer
            .register_tensor("ramp", &[1, 64], UnifiedDType::F32, None)
            .expect("registers");
        writer
            .register_tensor("off_grid", &[1, 64], UnifiedDType::F32, None)
            .expect("registers");
        writer
            .write_tensor_data(&f32_bytes(&ramp()))
            .expect("settles");
        writer
            .write_tensor_data(&f32_bytes(&off_grid()))
            .expect("settles");

        let planned = writer.planned();
        let ramp_error = planned[0]
            .screen_error
            .expect("a quantized tensor is screened");
        let off_grid_error = planned[1]
            .screen_error
            .expect("a quantized tensor is screened");
        assert!(
            off_grid_error > ramp_error,
            "{off_grid_error} vs {ramp_error}"
        );
        assert_eq!(planned[0].elements, 64);
    }

    /// A raw tensor has no base encoding to be measured against, so it is
    /// screened not at all and is no bump candidate.
    #[test]
    fn a_raw_tensor_carries_no_screen_error() {
        let mut writer = planned_writer();
        let plan = plan_one(&mut writer, &[64], &vec![0.5f32; 64]);
        assert_eq!(plan.fallback, Some(FallbackReason::Rank));
        assert_eq!(plan.screen_error, None);
        assert_eq!(plan.elements, 64);
    }

    /// A profile is mixed precision. The worst-reconstructed share of the
    /// QUANTIZABLE parameters carries the bump encoding and records
    /// `TaskSensitivity` for it; a tensor left at the base records no reason
    /// at all.
    ///
    /// The rank-1 tensor here is 4096 elements against the two candidates'
    /// 64 each. Counted in the budget it would raise `TCF-COMPACT`'s 10% from
    /// 12.8 elements to 422.4 and bump both candidates, so the ramp staying
    /// at the base is what proves the exclusion.
    #[test]
    fn a_profile_bumps_the_worst_reconstructed_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model.safetensors");
        let ramp = ramp();
        let off_grid = off_grid();
        let bias = vec![0.5f32; 4096];
        let tensors: [(&str, &[usize], &[f32]); 3] = [
            ("ramp", &[1, 64], &ramp),
            ("off_grid", &[1, 64], &off_grid),
            ("bias", &[4096], &bias),
        ];

        let mut fixture = SafeTensorsFormatWriter::new();
        for (name, shape, _) in tensors {
            fixture
                .register_tensor(name, shape, UnifiedDType::F32, None)
                .expect("registers");
        }
        fixture.initialize(&source).expect("initializes");
        for (_, _, values) in tensors {
            fixture
                .write_tensor_data(&f32_bytes(values))
                .expect("writes");
        }
        fixture.finalize().expect("finalizes");

        let profile = parse_tcf_profile("compact").expect("a ladder rung");
        let out = dir.path().join("model.tcf");
        let mut writer = TcfFormatWriter::new()
            .with_options(TcfOptions::from_profile(profile))
            .with_source(source.to_str().expect("utf8 path"));
        writer.initialize(&out).expect("initializes");
        for (name, shape, _) in tensors {
            writer
                .register_tensor(name, shape, UnifiedDType::F32, None)
                .expect("registers");
        }
        for (_, _, values) in tensors {
            writer
                .write_tensor_data(&f32_bytes(values))
                .expect("settles a plan");
        }
        writer.finalize().expect("writes");

        assert_eq!(
            writer.planned().iter().filter(|plan| plan.bumped).count(),
            1,
            "one candidate fills TCF-COMPACT's element budget"
        );

        let bytes = std::fs::read(&out).expect("reads the output back");
        let file = TcfFile::open(&bytes).expect("tcf-core accepts the file");
        let records = file.tensors();
        let at = |wanted: &str| -> usize {
            records
                .iter()
                .position(|tensor| file.string(tensor.name).ok() == Some(wanted))
                .expect("a tensor of that name")
        };

        let bumped = &records[at("off_grid")];
        assert_eq!(bumped.encoding, Encoding::Native(NativeEncoding::Q6S16DT64));
        assert_eq!(
            bumped.fallback_reason,
            tcf_core::FallbackReason::TaskSensitivity
        );

        let base = &records[at("ramp")];
        assert_eq!(base.encoding, Encoding::Native(NativeEncoding::Q4AS32DT64));
        assert_eq!(base.fallback_reason, tcf_core::FallbackReason::None);

        let raw = &records[at("bias")];
        assert_eq!(raw.encoding, Encoding::Raw(RawEncoding::F32));
        assert_eq!(raw.fallback_reason, tcf_core::FallbackReason::RankLt2);

        for tensor in records {
            file.verify_tensor(tensor).expect("digests and proofs hold");
        }
    }

    /// The file is read back by `tcf-core`'s own reader, never by anything
    /// compressr owns: a writer checked against a reader beside it proves
    /// nothing.
    #[test]
    fn writes_a_file_tcf_core_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model.safetensors");
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.03).collect();

        let mut fixture = SafeTensorsFormatWriter::new();
        fixture
            .register_tensor("w", &[1, 64], UnifiedDType::F32, None)
            .expect("registers");
        fixture.initialize(&source).expect("initializes");
        fixture
            .write_tensor_data(&f32_bytes(&values))
            .expect("writes");
        fixture.finalize().expect("finalizes");

        let out = dir.path().join("model.tcf");
        let mut writer = TcfFormatWriter::new()
            .with_options(options())
            .with_source(source.to_str().expect("utf8 path"));
        writer.initialize(&out).expect("initializes");
        writer
            .register_tensor("w", &[1, 64], UnifiedDType::F32, None)
            .expect("registers");
        writer
            .write_tensor_data(&f32_bytes(&values))
            .expect("settles a plan");
        writer.finalize().expect("writes");

        let bytes = std::fs::read(&out).expect("reads the output back");
        let file = TcfFile::open(&bytes).expect("tcf-core accepts the file");
        let tensor = file.tensors().first().expect("one tensor");
        assert_eq!(tensor.encoding, Encoding::Native(NativeEncoding::Q4S32T64));
        assert_eq!(tensor.shape(), &[1, 64]);
        assert_eq!(tensor.fallback_reason, tcf_core::FallbackReason::None);
        assert_eq!(file.string(tensor.name).expect("utf8 name"), "w");
        file.verify_tensor(tensor).expect("digests and proofs hold");
    }
}
