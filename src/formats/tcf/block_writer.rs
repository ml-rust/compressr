//! [`TcfBlockFormatWriter`]: TCF carrying GGML block encodings.
//!
//! A `--format tcf` run with `--quantization` (flat or a mix) plans and
//! quantizes exactly as a GGUF run does: the same per-tensor plan, the same
//! `ggml-quants.c` writers in boostr, the same bytes. This writer stores
//! those bytes in TCF's container instead of GGUF's — a `TensorRecord` per
//! tensor whose encoding is `Encoding::Block(ggml_type)`, the 8-bit dynamic
//! activation contract every block kernel implements, a proof vector checked
//! with boostr's own decoder, and header-only placement. A reader loads each
//! tensor as the `QuantTensor` it would have loaded from the GGUF.
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
use tcf_core::{block_proof_values, BlockEncoding, Encoding, Payload, TcfError, TcfWriter};

use super::block_spill::BlockSpill;
use super::encoding::{raw_encoding_for, TensorShape};
use super::record::{
    a8_dynamic_contract_record, block_module_record, tensor_record, A8_DYNAMIC_CONTRACT_ID,
};
use crate::formats::gguf::GgmlType;
use crate::formats::FormatWriter;
use crate::loader::UnifiedDType;
use crate::policy::{tcf_fallback_reason, TensorPlan};
use crate::quantization::QuantType;

/// The TCF block encoding a GGUF quantization type is stored as.
///
/// The identifier is `ggml_type` itself, so the mapping is the one the GGUF
/// writer uses, looked up in tcf-core's table.
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
        .map_err(|e| anyhow!("{quant_type:?}: tcf-core has no block encoding for it: {e}"))
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

/// TCF writer for GGML block encodings.
pub struct TcfBlockFormatWriter {
    /// The run's base quantization type: the module's preferred encoding
    base: Option<QuantType>,
    module_name: String,
    /// The conversion loop's plan per tensor, in registration order
    plans: Vec<TensorPlan>,
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

    fn writer_mut(&mut self) -> Result<&mut TcfWriter> {
        self.writer
            .as_mut()
            .ok_or_else(|| anyhow!("TCF writer used before initialize()"))
    }

    /// The reason tensor `index` is not at the preferred encoding, or `None`
    /// when it is.
    fn reason_for(&self, index: usize, registration: &Registration) -> tcf_core::FallbackReason {
        if registration.quant_type == self.base {
            return tcf_core::FallbackReason::None;
        }
        match self.plans.get(index) {
            Some(TensorPlan::StoreF32(reason)) => tcf_fallback_reason(*reason),
            Some(TensorPlan::Quantize(_)) => tcf_core::FallbackReason::ProducerPolicy,
            Some(TensorPlan::StoreAsIs) | None => tcf_core::FallbackReason::UnsupportedEncoding,
        }
    }

    /// Register tensor `index`'s record from its finished bytes.
    fn register(&mut self, index: usize, registration: &Registration, data: &[u8]) -> Result<()> {
        let name = registration.name.as_str();
        let shape = TensorShape::of(&registration.shape, name)?;
        let tensor_id = u32::try_from(index).context("TCF stores at most u32::MAX tensors")?;
        let reason = self.reason_for(index, registration);
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

        let mut record = tensor_record(
            tensor_id,
            interned,
            encoding,
            reason,
            shape.dims(),
            shape.rank(),
            None,
        );
        record.activation_contract_id = A8_DYNAMIC_CONTRACT_ID;
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
    /// Open the `tcf-core` writer with the block contract and the root
    /// module, and the spill beside the output. Nothing is created at the
    /// output path until `finalize`.
    fn initialize(&mut self, path: &Path) -> Result<()> {
        let base = self.base.ok_or_else(|| {
            anyhow!("TCF block output needs a base quantization type before initialize()")
        })?;
        let base = block_encoding_for(base)?;
        let mut writer = TcfWriter::new();
        let name = writer.intern(&self.module_name).map_err(|e| anyhow!(e))?;
        writer
            .add_contract(a8_dynamic_contract_record())
            .map_err(|e| anyhow!(e))?;
        writer
            .add_module(block_module_record(name, base))
            .map_err(|e| anyhow!(e))?;
        self.spill = Some(BlockSpill::create(path)?);
        self.path = Some(path.to_path_buf());
        self.writer = Some(writer);
        Ok(())
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
    use boostr::quant::{QuantFormat, QuantScheme};
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
    /// the GGUF would have given, under the 8-bit activation contract.
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
        assert_eq!(
            loader.quant_scheme("w").expect("scheme"),
            QuantScheme::Gguf(QuantFormat::Q8_0)
        );
        let device = boostr::CpuDevice::new();
        let qt = loader
            .load_quant_tensor::<CpuRuntime>("w", &device)
            .expect("loads packed");
        assert_eq!(qt.storage_bytes(), 4 * 34);
        let contract = qt.activation_contract().expect("carries the contract");
        assert_eq!(
            contract.input_representation,
            tcf_core::InputRepresentation::A8S32Dynamic
        );
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
                tcf_core::FallbackReason::None,
                tcf_core::FallbackReason::RankLt2,
                tcf_core::FallbackReason::UnsupportedEncoding,
            ]
        );
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
