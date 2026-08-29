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
//!    verdict, drops the tiles, and registers the record. A tensor the
//!    quantizer rejects is re-planned as raw with the reason it gave, and the
//!    run continues.
//! 2. [`FormatWriter::finalize`] adds no records: every one is already
//!    registered, so `TcfWriter::finish_streaming` pulls one payload at a
//!    time, writes it at its precomputed offset, and drops it before asking
//!    for the next.
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

use tcf_core::{encoding::quantize, Encoding, NativeEncoding, Payload, TcfError, TcfWriter};

use super::encoding::{
    dims_of, rank_of, raw_encoding_for, values_f32, TcfOptions, TCF_ENCODING_NAMES,
};
use super::record::{module_record, tensor_record};
use crate::formats::FormatWriter;
use crate::loader::{LoadedModel, ModelLoader, UnifiedDType};
use crate::policy::{plan_tcf_tensor, tcf_fallback_reason, FallbackReason, TcfPlan};
use crate::quantization::QuantType;

/// One tensor as registered, before its values decide anything.
#[derive(Debug, Clone)]
struct Registration {
    name: String,
    shape: Vec<usize>,
    dtype: UnifiedDType,
}

/// One tensor's settled plan: what its record says, and how pass 2 produces
/// its payload.
#[derive(Debug, Clone)]
pub struct PlannedTensor {
    /// Tensor name, as the source checkpoint spells it
    pub name: String,
    /// Source element type, which the raw path stores verbatim
    pub dtype: UnifiedDType,
    /// `None` for a quantized tensor, otherwise why it is stored raw
    pub fallback: Option<FallbackReason>,
    /// The encoding written to the record
    pub encoding: Encoding,
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
                "TCF output needs an encoding: pass --tcf-encoding and \
                 --tcf-preferred-encoding. Accepted: {}",
                TCF_ENCODING_NAMES.join(", ")
            )
        })
    }

    /// The `fallback_reason` a quantized tensor carries.
    ///
    /// SPECIFICATION.md Section 8.6 makes the field mandatory whenever the
    /// encoding differs from the module's highest-ranked preference, which a
    /// quantized tensor does whenever the two flags disagree.
    fn quantized_reason(options: TcfOptions) -> tcf_core::FallbackReason {
        if options.encoding == options.preferred_encoding {
            tcf_core::FallbackReason::None
        } else {
            tcf_core::FallbackReason::ProducerPolicy
        }
    }

    /// Settle one tensor's plan from its shape and, when the shape allows
    /// quantization, from its values.
    ///
    /// The quantizer is the only thing that can see a NaN or a scale that
    /// underflows binary16, so those two fallbacks are decided here, in the
    /// conversion loop, and never abort the run.
    fn settle(&self, registration: &Registration, data: &[u8]) -> Result<TcfPlan> {
        let options = self.options()?;
        let geometry = options.encoding.geometry();
        let plan = plan_tcf_tensor(&registration.shape, registration.dtype, geometry.tile);
        if plan != TcfPlan::Quantize {
            return Ok(plan);
        }

        let name = registration.name.as_str();
        let values = values_f32(data, registration.dtype, name)?;
        let dims = dims_of(&registration.shape, name)?;
        let rank = rank_of(&registration.shape, name)?;
        // The tiles are dropped here: pass 2 recomputes them one tensor at a
        // time, so keeping them would break the one-tensor-in-flight bound.
        // `quantize` wants EXACTLY `rank` dims; the padded `[u64; 8]` is the
        // record's wire shape, not the codec's. Passing all eight is
        // `E_INVALID_RANK`.
        let extent = usize::try_from(rank).unwrap_or(dims.len());
        match quantize(&values, &dims[..extent], rank, geometry) {
            Ok(_) => Ok(TcfPlan::Quantize),
            Err(TcfError::SourceNonfinite { .. }) => {
                log::warn!("Tensor '{name}' holds a NaN or an infinity: storing it unquantized");
                Ok(TcfPlan::Raw(FallbackReason::SourceNonfinite))
            }
            Err(TcfError::NumericRange { .. }) => {
                log::warn!(
                    "Tensor '{name}' has a group whose scale is not representable at {:?}: \
                     storing it unquantized",
                    options.encoding
                );
                Ok(TcfPlan::Raw(FallbackReason::NumericRange))
            }
            Err(other) => {
                Err(anyhow!(other)).with_context(|| format!("Failed to quantize tensor '{name}'"))
            }
        }
    }

    /// Register one settled tensor's record with `tcf-core`.
    fn register(&mut self, registration: &Registration, plan: TcfPlan) -> Result<()> {
        let options = self.options()?;
        let index = self.planned.len();
        let name = registration.name.as_str();
        let dims = dims_of(&registration.shape, name)?;
        let rank = rank_of(&registration.shape, name)?;

        let (encoding, reason) = match plan {
            TcfPlan::Quantize => (
                Encoding::Native(options.encoding),
                Self::quantized_reason(options),
            ),
            TcfPlan::Raw(reason) => (
                Encoding::Raw(raw_encoding_for(registration.dtype, name)?),
                tcf_fallback_reason(reason),
            ),
        };

        let tensor_id = u32::try_from(index).context("More tensors than a TCF file can hold")?;
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))?;
        let interned = writer
            .intern(name)
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to intern tensor name '{name}'"))?;
        let record = tensor_record(tensor_id, interned, encoding, reason, dims, rank);
        let registered = match plan {
            TcfPlan::Quantize => writer.register_quantized_tensor(record),
            TcfPlan::Raw(_) => writer.register_raw_tensor(record),
        };
        registered
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to register tensor '{name}'"))?;

        self.planned.push(PlannedTensor {
            name: registration.name.clone(),
            dtype: registration.dtype,
            fallback: match plan {
                TcfPlan::Quantize => None,
                TcfPlan::Raw(reason) => Some(reason),
            },
            encoding,
        });
        Ok(())
    }

    /// Produce one tensor's payload, reading it from the source checkpoint.
    ///
    /// The payload is returned by value and dropped by `tcf-core` as soon as
    /// it is written, so one tensor is resident at a time.
    fn payload(
        model: &mut LoadedModel,
        plan: &PlannedTensor,
        encoding: NativeEncoding,
        shape: &[usize],
    ) -> Result<Payload> {
        let name = plan.name.as_str();
        let bytes = model
            .read_tensor_bytes(name)
            .with_context(|| format!("Failed to re-read tensor '{name}'"))?;
        if plan.fallback.is_some() {
            // A raw tensor keeps its source element type, so its bytes are
            // the payload.
            return Ok(Payload::Raw(bytes));
        }
        let values = values_f32(&bytes, plan.dtype, name)?;
        let dims = dims_of(shape, name)?;
        let rank = rank_of(shape, name)?;
        // Exactly `rank` dims, as in `settle` above.
        let extent = usize::try_from(rank).unwrap_or(dims.len());
        let tiles = quantize(&values, &dims[..extent], rank, encoding.geometry())
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("Failed to quantize tensor '{name}'"))?;
        Ok(Payload::Quantized(tiles))
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

    /// Settle and register the next tensor's record, then drop its values.
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
        let plan = self.settle(&registration, data)?;
        self.register(&registration, plan)
    }

    /// Write the file, pulling one payload at a time.
    fn finalize(&mut self) -> Result<()> {
        let Some(writer) = self.writer.take() else {
            return Ok(());
        };
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
                    Self::payload(&mut model, plan, options.encoding, &registration.shape)
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::SafeTensorsFormatWriter;
    use tcf_core::{RawEncoding, TcfFile};

    fn options() -> TcfOptions {
        TcfOptions {
            encoding: NativeEncoding::Q4S32T64,
            preferred_encoding: NativeEncoding::Q4S32T64,
        }
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
