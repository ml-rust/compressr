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
//!    verdict and the screen error, and drops the tiles. A tensor the
//!    quantizer rejects is re-planned as raw with the reason it gave, and the
//!    run continues.
//! 2. [`FormatWriter::finalize`] runs the profile allocation, registers every
//!    record, and then lets `TcfWriter::finish_streaming` pull one payload at
//!    a time, write it at its precomputed offset, and drop it before asking
//!    for the next.
//!
//! Registration waits for `finalize` because a profile's bump set is a
//! comparison across tensors: which tensors carry the bump encoding is unknown
//! until the last one has been screened.
//!
//! Each tensor is therefore read and quantized twice, and at most one tensor's
//! values are resident at any point. That second read is the price of a
//! value-dependent `fallback_reason`: the alternative holds every payload
//! until the directory is complete, which is what the older `add_*_tensor`
//! path does.
//!
//! Nothing is created on disk before pass 2, so a tensor compressr cannot
//! represent fails with no output file at all rather than truncating one.

use anyhow::{anyhow, bail, Context, Result};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use tcf_core::{Encoding, Payload, TcfError, TcfWriter};

use super::encoding::{dims_of, rank_of, values_f32, TcfOptions, TCF_ENCODING_NAMES};
use super::parallel::quantize_named;
use super::plan::{allocate, plan_one, reason_for, settle, PlannedTensor, Registration};
use super::record::{module_record, tensor_record};
use crate::formats::FormatWriter;
use crate::loader::{LoadedModel, ModelLoader, UnifiedDType};
use crate::policy::TcfPlan;
use crate::quantization::QuantType;

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
            writer: None,
        }
    }

    /// Set the encoding to quantize into and the module's preferred encoding.
    pub fn with_options(mut self, options: TcfOptions) -> Self {
        self.options = Some(options);
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

    /// Register every settled record with `tcf-core`, in directory order.
    fn register_all(&mut self) -> Result<()> {
        let options = self.options()?;
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
            let dims = dims_of(&shape, name)?;
            let rank = rank_of(&shape, name)?;
            let tensor_id =
                u32::try_from(index).context("More tensors than a TCF file can hold")?;
            let reason = reason_for(&plan, options);

            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;
            let interned = writer
                .intern(name)
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Failed to intern tensor name '{name}'"))?;
            let record = tensor_record(tensor_id, interned, plan.encoding, reason, dims, rank);
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

    /// Produce one tensor's payload, reading it from the source checkpoint.
    ///
    /// The payload is returned by value and dropped by `tcf-core` as soon as
    /// it is written, so one tensor is resident at a time.
    fn payload(model: &mut LoadedModel, plan: &PlannedTensor, shape: &[usize]) -> Result<Payload> {
        let name = plan.name.as_str();
        let bytes = model
            .read_tensor_bytes(name)
            .with_context(|| format!("Failed to re-read tensor '{name}'"))?;
        // The plan's own encoding drives this, not the run's base encoding: a
        // profile bumps some tensors, and quantizing one at the base here
        // would write a payload its record does not describe.
        let encoding = match plan.encoding {
            Encoding::Native(encoding) => encoding,
            // A raw tensor keeps its source element type, so its bytes are
            // the payload.
            _ => return Ok(Payload::Raw(bytes)),
        };
        let values = values_f32(&bytes, plan.dtype, name)?;
        let dims = dims_of(shape, name)?;
        let rank = rank_of(shape, name)?;
        // Exactly `rank` dims, as in `settle` above.
        let extent = usize::try_from(rank).unwrap_or(dims.len());
        let tiles = quantize_named(&values, &dims[..extent], rank, encoding.layout(), name)?;
        Ok(Payload::Quantized(tiles))
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
    /// module.
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
        writer
            .add_module(module_record(name, options))
            .map_err(|e| anyhow!(e))?;

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

    /// Settle and screen the next tensor, then drop its values.
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
        let (plan, screen) = settle(options, &registration, data)?;
        self.planned
            .push(plan_one(options, &registration, plan, screen)?);
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
        allocate(&mut self.planned, options)?;
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

        let mut model = ModelLoader::new(&source)
            .load()
            .with_context(|| format!("Failed to re-open the input checkpoint: {source}"))?;
        let file = std::fs::File::create(&path)
            .with_context(|| format!("Failed to create {}", path.display()))?;
        let mut sink = BufWriter::new(file);

        let planned = &self.planned;
        let registrations = &self.registrations;
        // A `produce` error must reach the caller as itself: TcfError has no
        // room for one, so the real error is kept here and raised below.
        let mut failure: Option<anyhow::Error> = None;
        let result = writer.finish_streaming(&mut sink, |index| {
            let produced = match (planned.get(index), registrations.get(index)) {
                (Some(plan), Some(registration)) => {
                    Self::payload(&mut model, plan, &registration.shape)
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
        assert_eq!(bumped.encoding, Encoding::Native(NativeEncoding::Q6S32T64));
        assert_eq!(
            bumped.fallback_reason,
            tcf_core::FallbackReason::TaskSensitivity
        );

        let base = &records[at("ramp")];
        assert_eq!(base.encoding, Encoding::Native(NativeEncoding::Q4AS64T64));
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
