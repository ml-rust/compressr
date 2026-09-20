//! [`TcfBlockFormatWriter`]: TCF carrying GGML block encodings.
//!
//! A `--format tcf` run with `--quantization` (flat or a mix) plans and
//! quantizes exactly as a GGUF run does: the same per-tensor plan, the same
//! `ggml-quants.c` writers in boostr, the same bytes. This writer stores
//! those bytes in TCF's container instead of GGUF's — a `TensorRecord` per
//! tensor whose encoding is `Encoding::Block(ggml_type)`, the ggml-family
//! activation contract every block kernel satisfies, a proof vector checked
//! with boostr's own decoder, and header-only placement. A reader loads each
//! tensor as the `QuantTensor` it would have loaded from the GGUF.
//!
//! # Measured allocation
//!
//! On a `--sensitivity` run the file also carries the measurement the mix was
//! ranked by: one `CalibrationRecord` for the measurement, and on every
//! measured tensor the task-metric delta at the format it was written in
//! (or, when the file did not measure that format, the delta the allocation
//! ranked on), with `SENSITIVITY_VALID` set. That is what makes a measured
//! TCF identifiable where a measured GGUF is indistinguishable from a
//! positional one.
//!
//! # One tensor in flight
//!
//! The trait hands over one finished payload at a time, before the directory
//! can be written. Each is appended to a [`BlockSpill`] and read back once in
//! `finalize`, so peak memory stays at the largest tensor.
//!
//! # Fallback reasons
//!
//! A tensor not at the module's preferred block format carries the reason
//! TCF requires (Section 8.6). The conversion loop's own [`TensorPlan`] list
//! is the source: a `StoreF32` reason maps through
//! [`crate::policy::tcf_fallback_reason`], a mix's other block format is
//! `ProducerPolicy`, and a pass-through integer tensor is
//! `UnsupportedEncoding`.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use boostr::format::tcf::BoostrBlockDecoder;
use boostr::tcf::{block_proof_values, BlockEncoding, Encoding, Payload, TcfError, TcfWriter};

use super::block_spill::BlockSpill;
use super::encoding::{raw_encoding_for, TensorShape};
use super::record::{
    block_module_record, calibration_record, ggml_reference_contract_record, tensor_record,
    TensorSensitivity, SENSITIVITY_SEED_COUNT,
};
use crate::formats::gguf::GgmlType;
use crate::formats::FormatWriter;
use crate::loader::UnifiedDType;
use crate::policy::{quant_type_label, tcf_fallback_reason, SensitivityMeasurement, TensorPlan};
use crate::quantization::QuantType;

/// The TCF block encoding a GGUF quantization type is stored as.
///
/// The identifier is `ggml_type` itself, so the mapping is the one the GGUF
/// writer uses, looked up in the container's table.
///
/// # Errors
/// The type is not a GGML block format (NF4, FP8, GPTQ, AWQ have no block
/// stream to carry).
pub fn block_encoding_for(quant_type: QuantType) -> Result<BlockEncoding> {
    if !matches!(
        quant_type,
        QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q8_0
            | QuantType::Q8_1
            | QuantType::Q2K
            | QuantType::Q3K
            | QuantType::Q4K
            | QuantType::Q5K
            | QuantType::Q6K
            | QuantType::IQ4NL
            | QuantType::IQ4XS
    ) {
        bail!("{quant_type:?} is not a GGML block format, so TCF has no block encoding for it");
    }
    let ggml = GgmlType::from_quant_type(quant_type) as u32;
    let raw = u16::try_from(ggml)
        .ok()
        .and_then(|id| id.checked_add(0x0200))
        .ok_or_else(|| anyhow!("ggml_type {ggml} is outside TCF's block encoding range"))?;
    BlockEncoding::try_from(raw)
        .map_err(|e| anyhow!("{quant_type:?}: TCF has no block encoding for it: {e}"))
}

/// One tensor as the trait registered it.
#[derive(Debug, Clone)]
struct Registration {
    name: String,
    shape: Vec<usize>,
    dtype: UnifiedDType,
    quant_type: Option<QuantType>,
}

/// One record's payload description, kept for the write pass.
enum Pending {
    Block { proof: Vec<u16> },
    Raw,
}

/// What every measured `TensorRecord` copies from the declared calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CalibrationFacts {
    /// The `CalibrationRecord` this file declares
    id: u32,
    /// Evaluation rows every delta was measured over
    samples: u32,
}

/// TCF writer for GGML block encodings.
pub struct TcfBlockFormatWriter {
    /// The run's base quantization type: the module's preferred encoding
    base: Option<QuantType>,
    module_name: String,
    /// The conversion loop's plan per tensor, in registration order
    plans: Vec<TensorPlan>,
    /// The measurement a `--sensitivity` run ranked its mix by
    sensitivity: Option<SensitivityMeasurement>,
    /// The calibration this file declares, on a measured run only
    calibration: Option<CalibrationFacts>,
    path: Option<PathBuf>,
    writer: Option<TcfWriter>,
    registrations: Vec<Registration>,
    pending: Vec<Pending>,
    spill: Option<BlockSpill>,
}

impl TcfBlockFormatWriter {
    /// A writer with no base format and no plans; set both before
    /// `initialize`.
    pub fn new() -> Self {
        Self {
            base: None,
            module_name: "model".to_string(),
            plans: Vec::new(),
            sensitivity: None,
            calibration: None,
            path: None,
            writer: None,
            registrations: Vec::new(),
            pending: Vec::new(),
            spill: None,
        }
    }

    /// The base quantization type, written as the module's preferred
    /// encoding.
    pub fn with_base(mut self, base: QuantType) -> Self {
        self.base = Some(base);
        self
    }

    /// The root module's name.
    pub fn with_module_name(mut self, name: &str) -> Self {
        self.module_name = name.to_string();
        self
    }

    /// The conversion loop's plan per tensor, in the order tensors are
    /// registered. Supplies each fallback reason.
    pub fn with_plans(mut self, plans: Vec<TensorPlan>) -> Self {
        self.plans = plans;
        self
    }

    /// The measurement the mix was ranked by. Declares a calibration record
    /// and writes each measured tensor's delta; see the module docs.
    pub fn with_sensitivity(mut self, measurement: SensitivityMeasurement) -> Self {
        self.sensitivity = Some(measurement);
        self
    }

    /// Declare the calibration a measured run read, and remember what every
    /// measured tensor copies from it. Does nothing without `--sensitivity`.
    fn declare_calibration(&mut self) -> Result<()> {
        let Some(measurement) = self.sensitivity.clone() else {
            return Ok(());
        };
        let dataset = measurement.baseline().dataset.clone();
        let evaluator = measurement.evaluator_name().to_string();
        let writer = self.writer_mut()?;
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

    /// The sensitivity one tensor's record carries: the measured delta at
    /// the format it was written in, else the delta the allocation ranked
    /// on, else nothing. A delta with no calibration behind it is a number
    /// with no authority (Section 10.5.1), so the two come together or not
    /// at all.
    fn sensitivity_for(&self, registration: &Registration) -> Option<TensorSensitivity> {
        let measurement = self.sensitivity.as_ref()?;
        let calibration = self.calibration?;
        let name = registration.name.as_str();
        let delta = registration
            .quant_type
            .and_then(|written| measurement.delta_at(name, quant_type_label(written)))
            .or_else(|| measurement.delta_for(name))?;
        Some(TensorSensitivity {
            calibration_id: calibration.id,
            delta,
            samples: calibration.samples,
            seed_count: SENSITIVITY_SEED_COUNT,
        })
    }

    fn writer_mut(&mut self) -> Result<&mut TcfWriter> {
        self.writer
            .as_mut()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))
    }

    /// The reason tensor `index` is not at the preferred encoding, or `None`
    /// when it is.
    fn reason_for(&self, index: usize, registration: &Registration) -> boostr::tcf::FallbackReason {
        if registration.quant_type == self.base {
            return boostr::tcf::FallbackReason::None;
        }
        match self.plans.get(index) {
            Some(TensorPlan::StoreF32(reason)) => tcf_fallback_reason(*reason),
            // A mix put this tensor at another format. When a measurement
            // drove that, the record says so and carries the delta.
            Some(TensorPlan::Quantize(_)) if self.sensitivity.is_some() => {
                boostr::tcf::FallbackReason::TaskSensitivity
            }
            Some(TensorPlan::Quantize(_)) => boostr::tcf::FallbackReason::ProducerPolicy,
            Some(TensorPlan::StoreAsIs) | Some(TensorPlan::Repack(_)) | None => {
                boostr::tcf::FallbackReason::UnsupportedEncoding
            }
        }
    }

    /// Register tensor `index`'s record from its finished bytes.
    fn register(&mut self, index: usize, registration: &Registration, data: &[u8]) -> Result<()> {
        let name = registration.name.as_str();
        let shape = TensorShape::of(&registration.shape, name)?;
        let tensor_id = u32::try_from(index).context("TCF stores at most u32::MAX tensors")?;
        let reason = self.reason_for(index, registration);
        let measured = self.sensitivity_for(registration);
        let writer = self.writer_mut()?;
        let interned = writer
            .intern(name)
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to intern tensor name '{name}'"))?;

        let (encoding, pending) = match registration.quant_type {
            Some(quant_type) => {
                let block = block_encoding_for(quant_type)?;
                let proof = block_proof_values(
                    &BoostrBlockDecoder,
                    block,
                    data,
                    shape.extent(),
                    shape.rank(),
                    tensor_id,
                )
                .map_err(|e| anyhow!(e))
                .with_context(|| format!("Failed to compute the proof vector of '{name}'"))?;
                (Encoding::Block(block), Pending::Block { proof })
            }
            None => (
                Encoding::Raw(raw_encoding_for(registration.dtype, name)?),
                Pending::Raw,
            ),
        };

        let record = tensor_record(
            tensor_id,
            interned,
            encoding,
            reason,
            shape.dims(),
            shape.rank(),
            measured,
        );
        let registered = match encoding {
            Encoding::Block(_) => writer.register_block_tensor(record),
            _ => writer.register_raw_tensor(record),
        };
        registered
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to register tensor '{name}'"))?;
        self.pending.push(pending);
        Ok(())
    }
}

impl Default for TcfBlockFormatWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// An `anyhow` error from the spill, carried through `finish_streaming`.
fn spill_error(error: anyhow::Error) -> TcfError {
    TcfError::Io {
        kind: std::io::ErrorKind::Other,
        message: format!("{error:#}"),
    }
}

impl FormatWriter for TcfBlockFormatWriter {
    /// Open the container writer with the block contract, the root module
    /// and, on a measured run, the calibration record, and the spill beside
    /// the output. Nothing is created at the output path until `finalize`.
    fn initialize(&mut self, path: &Path) -> Result<()> {
        let base = self.base.ok_or_else(|| {
            anyhow!("TCF block output needs a base quantization type before initialize()")
        })?;
        let base = block_encoding_for(base)?;
        let mut writer = TcfWriter::new();
        let name = writer.intern(&self.module_name).map_err(|e| anyhow!(e))?;
        writer
            .add_contract(ggml_reference_contract_record())
            .map_err(|e| anyhow!(e))?;
        writer
            .add_module(block_module_record(name, base))
            .map_err(|e| anyhow!(e))?;
        self.spill = Some(BlockSpill::create(path)?);
        self.path = Some(path.to_path_buf());
        self.writer = Some(writer);
        self.declare_calibration()
    }

    /// Remember one tensor. Its record is registered when its bytes arrive,
    /// because the proof vector is computed from them.
    fn register_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: UnifiedDType,
        quant_type: Option<QuantType>,
    ) -> Result<()> {
        if let Some(quant_type) = quant_type {
            block_encoding_for(quant_type)
                .with_context(|| format!("Tensor '{name}' cannot be stored in TCF"))?;
        }
        self.registrations.push(Registration {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype,
            quant_type,
        });
        Ok(())
    }

    /// Register the next tensor's record and spill its bytes.
    fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        let index = self.pending.len();
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
        self.register(index, &registration, data)?;
        self.spill
            .as_mut()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?
            .push(data)
    }

    /// Write the file, one spilled payload at a time.
    fn finalize(&mut self) -> Result<()> {
        let Some(writer) = self.writer.take() else {
            return Ok(());
        };
        if self.pending.len() != self.registrations.len() {
            bail!(
                "TCF writer has {} registered tensors but {} payloads",
                self.registrations.len(),
                self.pending.len()
            );
        }
        let path = self
            .path
            .clone()
            .ok_or_else(|| anyhow!("TCF writer finalized before initialize()"))?;
        let mut spill = self
            .spill
            .take()
            .ok_or_else(|| anyhow!("TCF writer finalized before initialize()"))?;
        spill.seal()?;
        if spill.len() != self.pending.len() {
            bail!(
                "TCF writer spilled {} payloads for {} records",
                spill.len(),
                self.pending.len()
            );
        }
        let pending = std::mem::take(&mut self.pending);

        let file =
            File::create(&path).with_context(|| format!("Failed to create {}", path.display()))?;
        let mut sink = BufWriter::new(file);
        writer
            .finish_streaming(&mut sink, |index| {
                let bytes = spill.take(index).map_err(spill_error)?;
                Ok(match pending.get(index) {
                    Some(Pending::Block { proof }) => Payload::Block {
                        bytes,
                        proof: proof.clone(),
                    },
                    _ => Payload::Raw(bytes),
                })
            })
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to write {}", path.display()))?;
        sink.into_inner()
            .map_err(|e| anyhow!("Failed to flush {}: {e}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::format::tcf::TcfLoader;
    use boostr::quant::QuantFormat;
    use boostr::tcf::TensorFlags;
    use boostr::CpuRuntime;

    /// Two rows of 64 as Q8_0: `f16 d` then 32 `i8` codes per block.
    fn q8_0_stream() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 * 34);
        for block in 0..4u8 {
            let d = half::f16::from_f32(0.5 + f32::from(block)).to_bits();
            bytes.extend_from_slice(&d.to_le_bytes());
            for code in 0..32u8 {
                bytes.push((code as i8 - 16 + block as i8) as u8);
            }
        }
        bytes
    }

    #[test]
    fn every_writable_block_type_has_an_encoding() {
        for (quant_type, expected) in [
            (QuantType::Q4_0, BlockEncoding::Q4_0),
            (QuantType::Q8_0, BlockEncoding::Q8_0),
            (QuantType::Q4K, BlockEncoding::Q4K),
            (QuantType::Q6K, BlockEncoding::Q6K),
            (QuantType::IQ4NL, BlockEncoding::Iq4Nl),
            (QuantType::IQ4XS, BlockEncoding::Iq4Xs),
        ] {
            assert_eq!(block_encoding_for(quant_type).expect("maps"), expected);
        }
        assert!(block_encoding_for(QuantType::NF4).is_err());
    }

    /// The file a GGUF-style run produces: a block weight, a raw bias, and a
    /// pass-through integer tensor. boostr reads it back as the `QuantTensor`
    /// the GGUF would have given, under the ggml-family contract.
    #[test]
    fn a_block_file_round_trips_through_boostr() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("model.tcf");
        let mut writer = TcfBlockFormatWriter::new()
            .with_base(QuantType::Q8_0)
            .with_module_name("m")
            .with_plans(vec![
                TensorPlan::Quantize(QuantType::Q8_0),
                TensorPlan::StoreF32(crate::policy::FallbackReason::Rank),
                TensorPlan::StoreAsIs,
            ]);
        writer.initialize(&path).expect("initializes");
        writer
            .register_tensor("w", &[2, 64], UnifiedDType::F32, Some(QuantType::Q8_0))
            .expect("registers");
        writer
            .register_tensor("b", &[4], UnifiedDType::F32, None)
            .expect("registers");
        writer
            .register_tensor("ids", &[3], UnifiedDType::I32, None)
            .expect("registers");
        writer.write_tensor_data(&q8_0_stream()).expect("writes");
        let bias: Vec<u8> = [1.0f32, -2.5, 0.0, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_tensor_data(&bias).expect("writes");
        let ids: Vec<u8> = [7i32, 8, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_tensor_data(&ids).expect("writes");
        writer.finalize().expect("finalizes");

        let loader = TcfLoader::open(&path).expect("boostr opens");
        loader.verify_all().expect("digests and proofs check");
        assert_eq!(loader.quant_format("w").expect("format"), QuantFormat::Q8_0);
        let device = boostr::CpuDevice::new();
        let qt = loader
            .load_quant_tensor::<CpuRuntime>("w", &device)
            .expect("loads packed");
        assert_eq!(qt.storage_bytes(), 4 * 34);
        let contract = qt.activation_contract().expect("carries the contract");
        assert_eq!(
            contract.input_representation,
            boostr::tcf::InputRepresentation::GgmlReference
        );
        // The family contract admits the block kernel: the weight runs.
        let client = boostr::CpuClient::new(device.clone());
        let act = boostr::Tensor::<CpuRuntime>::from_slice(&[1.0f32; 64], &[1, 64], &device)
            .expect("activation");
        boostr::QuantMatmulOps::quant_matmul(&client, &act, &qt).expect("runs");
        let b = loader.load_tensor_f32("b").expect("loads dense");
        assert_eq!(b, vec![1.0, -2.5, 0.0, 3.25]);
        let ids = loader.load_tensor_f32("ids").expect("loads dense");
        assert_eq!(ids, vec![7.0, 8.0, 9.0]);

        let infos: Vec<_> = ["w", "b", "ids"]
            .iter()
            .map(|n| loader.tensor_info(n).expect("info").record.fallback_reason)
            .collect();
        assert_eq!(
            infos,
            vec![
                boostr::tcf::FallbackReason::None,
                boostr::tcf::FallbackReason::RankLt2,
                boostr::tcf::FallbackReason::UnsupportedEncoding,
            ]
        );
        let w = loader.tensor_info("w").expect("info").record;
        assert!(!w.flags.contains(TensorFlags::SENSITIVITY_VALID));
        assert_eq!(w.calibration_id, 0);
    }

    /// A measured run declares its calibration and writes each measured
    /// tensor's delta against it; an unmeasured tensor stays unmarked.
    #[test]
    fn a_measured_run_records_the_calibration_and_each_delta() {
        let measurement = crate::policy::parse_sensitivity(
            &[
                r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q8_0","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#,
                r#"{"record":"sensitivity","name":"w","delta":0.25}"#,
            ]
            .join("\n"),
        )
        .expect("parses");
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("model.tcf");
        let mut writer = TcfBlockFormatWriter::new()
            .with_base(QuantType::Q8_0)
            .with_plans(vec![
                TensorPlan::Quantize(QuantType::Q8_0),
                TensorPlan::StoreF32(crate::policy::FallbackReason::Rank),
            ])
            .with_sensitivity(measurement.clone());
        writer.initialize(&path).expect("initializes");
        writer
            .register_tensor("w", &[2, 64], UnifiedDType::F32, Some(QuantType::Q8_0))
            .expect("registers");
        writer
            .register_tensor("b", &[4], UnifiedDType::F32, None)
            .expect("registers");
        writer.write_tensor_data(&q8_0_stream()).expect("writes");
        let bias: Vec<u8> = [1.0f32, -2.5, 0.0, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_tensor_data(&bias).expect("writes");
        writer.finalize().expect("finalizes");

        let loader = TcfLoader::open(&path).expect("boostr opens");
        loader.verify_all().expect("digests and proofs check");
        let w = loader.tensor_info("w").expect("info").record;
        assert!(w.flags.contains(TensorFlags::SENSITIVITY_VALID));
        assert_eq!(w.sensitivity_delta, 0.25);
        assert_eq!(w.sensitivity_samples, 12);
        assert_eq!(w.calibration_id, measurement.calibration_id());
        let b = loader.tensor_info("b").expect("info").record;
        assert!(!b.flags.contains(TensorFlags::SENSITIVITY_VALID));
        assert_eq!(b.calibration_id, 0);
        let calibrations = loader.calibrations();
        assert_eq!(calibrations.len(), 1);
        assert_eq!(calibrations[0].calibration_id, measurement.calibration_id());
        assert_eq!(calibrations[0].sample_count, 12);
    }

    /// A payload whose length disagrees with its registration is refused
    /// before any output file exists.
    #[test]
    fn a_short_payload_is_refused_with_no_output() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("model.tcf");
        let mut writer = TcfBlockFormatWriter::new()
            .with_base(QuantType::Q8_0)
            .with_plans(vec![TensorPlan::Quantize(QuantType::Q8_0)]);
        writer.initialize(&path).expect("initializes");
        writer
            .register_tensor("w", &[2, 64], UnifiedDType::F32, Some(QuantType::Q8_0))
            .expect("registers");
        assert!(writer.write_tensor_data(&q8_0_stream()[..100]).is_err());
        assert!(!path.exists());
    }
}
