//! TCF input: the directory up front, one decoded tensor per read.
//!
//! Every byte of format knowledge here comes from `boostr::format::tcf`, which
//! decodes through `tcf-core`'s reference codec. compressr holds no TCF bit
//! position, tile order or plane order — the same rule that keeps the GGUF
//! block layouts in boostr (AGENTS.md, MIGRATION.md Section 4.5.3).
//!
//! # Memory
//!
//! `TcfLoader::open` maps the file and decodes the directory only, so opening
//! a checkpoint of any size costs a directory pass. A payload page is touched
//! only by [`TcfModel::read_tensor_bytes`], which verifies the tensor's
//! digests and proof vector (Section 15) and then decodes it to f32.
//!
//! # Names
//!
//! TCF names are provenance, never identity (Section 6), so a file MAY repeat
//! one. Everything downstream of the loader keys a tensor by its name, and
//! `TcfLoader` resolves a repeated name to its first occurrence, so a repeat
//! would silently convert one tensor twice and drop the other. A file that
//! repeats a name is refused at open instead.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

use boostr::format::tcf::{encoding_name, TcfLoader};

use super::{UnifiedDType, UnifiedTensorInfo};

/// What a TCF file states about one tensor, beyond shape and element type.
///
/// TCF records a per-tensor encoding, why that encoding was chosen, where the
/// producer intends the tensor to live, and which contract, calibration and
/// workload records describe it. None of that survives a flattening into the
/// GGUF shape, so it is carried whole.
#[derive(Debug, Clone)]
pub struct TcfTensorFacts {
    /// Spec name of the on-disk encoding, e.g. `Q8S32_T64`. Section 12.
    pub encoding: String,
    /// Honest bits per weight, scales and minima included. `None` for a raw
    /// encoding, whose width is fixed by its element size. Section 12.2.
    pub bits_per_weight: Option<f64>,
    /// Why this tensor departs from its module's preferred encoding, or
    /// `None` when it is at the preference. Section 8.6.
    pub fallback_reason: Option<String>,
    /// Semantic role. Never a dispatch key. Section 8.7.
    pub role: String,
    /// The dispatch key. Section 8.6.1.
    pub execution_role: String,
    /// Placement intent. Section 8.5.
    pub residency_class: String,
    /// Name of the owning module, or its numeric id when the file names none.
    pub module: String,
    /// `ContractRecord` this tensor is read against, 0 when unset.
    pub activation_contract_id: u32,
    /// `CalibrationRecord` behind its sensitivity figures, 0 when unset.
    pub calibration_id: u32,
    /// `WorkloadProfileRecord` its access figures are measured under, 0 when
    /// unset.
    pub workload_profile_id: u32,
    /// Stored payload bytes, alignment padding excluded. Section 8.0.1.
    pub payload_bytes: u64,
    /// Task sensitivity delta and its 95% interval, present only when the
    /// producer measured one. Section 8.2.
    pub sensitivity: Option<(f32, f32)>,
}

/// One module's policy, as the file states it. Section 7.
#[derive(Debug, Clone)]
pub struct TcfModuleFacts {
    /// Module identifier, unique within the file.
    pub module_id: u32,
    /// Parent module identifier.
    pub parent_id: u32,
    /// Resolved name, empty when the record carries none.
    pub name: String,
    /// What the module is.
    pub module_role: String,
    /// Producer preference, highest first.
    pub preferred_encodings: Vec<String>,
    /// Declared fallback encoding, if any.
    pub fallback_encoding: Option<String>,
    /// Producer policy bits, e.g. `FORBID_REQUANT`.
    pub policy_flags: String,
    /// Default residency for the module's tensors.
    pub default_residency: String,
    /// Number of tensors this module owns.
    pub tensor_count: usize,
}

/// The file-level structure TCF has and GGUF does not.
#[derive(Debug, Clone)]
pub struct TcfFileFacts {
    /// Format major version.
    pub major: u16,
    /// Format minor version.
    pub minor: u16,
    /// Layout schema identifier.
    pub schema_id: u32,
    /// Total file length in bytes, directory and padding included.
    pub file_len: u64,
    /// First byte of the tensor data section. Section 4.1.
    pub data_off: u64,
    /// Number of `ContractRecord`s.
    pub contract_count: u32,
    /// Number of `CalibrationRecord`s.
    pub calibration_count: u32,
    /// Number of `RelationRecord`s.
    pub relation_count: u32,
    /// Number of `WorkloadProfileRecord`s.
    pub workload_count: u32,
    /// Every module, in file order.
    pub modules: Vec<TcfModuleFacts>,
    /// How many tensors carry each encoding, ordered by descending count.
    pub encoding_histogram: Vec<(String, usize)>,
    /// How many tensors record a departure from their module's preference.
    pub fallback_tensors: usize,
}

/// An open TCF file: directory in memory, payloads still on disk.
pub struct TcfModel {
    loader: TcfLoader,
    /// Tensor names in file order, so a conversion reads them in the order the
    /// producer wrote them.
    names: Vec<String>,
    infos: HashMap<String, UnifiedTensorInfo>,
    facts: TcfFileFacts,
}

impl TcfModel {
    /// Open `path` and decode its directory. No payload byte is read.
    ///
    /// # Errors
    ///
    /// - The file cannot be opened or mapped.
    /// - Any header, section, record or digest check fails. The error carries
    ///   the spec's `E_*` code.
    /// - The file repeats a tensor name, which no name-keyed caller can
    ///   resolve.
    pub fn open(path: &Path) -> Result<Self> {
        let loader = TcfLoader::open(path)
            .map_err(|e| anyhow!("Failed to open TCF file {}: {}", path.display(), e))?;

        let mut names = Vec::with_capacity(loader.len());
        let mut infos = HashMap::with_capacity(loader.len());
        for tensor in loader.tensors() {
            let record = &tensor.record;
            let module = loader
                .module(tensor.module_id())
                .map(|m| {
                    if m.name.is_empty() {
                        format!("#{}", m.module_id)
                    } else {
                        m.name.clone()
                    }
                })
                .unwrap_or_else(|| format!("#{}", tensor.module_id()));

            let fallback = if tensor.is_fallback() {
                Some(format!("{:?}", tensor.fallback_reason()))
            } else {
                None
            };

            let facts = TcfTensorFacts {
                encoding: encoding_name(tensor.encoding()),
                bits_per_weight: tensor.bits_per_weight(),
                fallback_reason: fallback,
                role: format!("{:?}", tensor.role()),
                execution_role: format!("{:?}", record.execution_role),
                residency_class: format!("{:?}", tensor.residency_class()),
                module,
                activation_contract_id: record.activation_contract_id,
                calibration_id: record.calibration_id,
                workload_profile_id: record.workload_profile_id,
                payload_bytes: tensor.payload_bytes(),
                sensitivity: tensor.sensitivity(),
            };

            let shape = tensor.shape();
            let numel: usize = shape.iter().product();
            // A payload decodes to f32, so that is the element type every
            // later stage sees. `size_bytes` stays the STORED byte count: the
            // honest answer to "how big is this file's tensor".
            let info = UnifiedTensorInfo {
                name: tensor.name.clone(),
                shape,
                dtype: UnifiedDType::F32,
                size_bytes: tensor.payload_bytes() as usize,
                numel,
                tcf: Some(facts),
            };

            if infos.insert(tensor.name.clone(), info).is_some() {
                return Err(anyhow!(
                    "TCF file {} declares the tensor name '{}' twice. Names are provenance, \
                     not identity (Section 6), so a repeated name cannot be converted: one \
                     tensor would be read twice and the other dropped.",
                    path.display(),
                    tensor.name
                ));
            }
            names.push(tensor.name.clone());
        }

        let facts = file_facts(&loader);
        Ok(Self {
            loader,
            names,
            infos,
            facts,
        })
    }

    /// Tensor names, in file order.
    pub fn tensor_names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// True when the file declares no tensor.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The directory entry for `name`, or `None` when the file has no such
    /// tensor.
    pub fn tensor_info(&self, name: &str) -> Option<UnifiedTensorInfo> {
        self.infos.get(name).cloned()
    }

    /// The file-level structure, for `compressr info`.
    pub fn facts(&self) -> &TcfFileFacts {
        &self.facts
    }

    /// Verify `name` and decode it into little-endian f32 bytes.
    ///
    /// The verification is not optional and not skippable: a reader that
    /// decodes without checking `payload_digest`, `semantic_digest` and the
    /// proof vector cannot tell a correct file from a corrupted one, which is
    /// the failure TCF exists to prevent (Section 15).
    ///
    /// # Errors
    /// The file has no tensor of that name, a digest or proof check fails, or
    /// the encoding has no decode path in boostr's reader.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let values = self
            .loader
            .load_tensor_f32(name)
            .map_err(|e| anyhow!("Failed to read TCF tensor '{}': {}", name, e))?;
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
}

/// Summarize a file's directory into the facts `compressr info` prints.
fn file_facts(loader: &TcfLoader) -> TcfFileFacts {
    let header = loader.header();

    let mut owned: HashMap<u32, usize> = HashMap::new();
    let mut encodings: HashMap<String, usize> = HashMap::new();
    let mut fallback_tensors = 0usize;
    for tensor in loader.tensors() {
        *owned.entry(tensor.module_id()).or_default() += 1;
        *encodings
            .entry(encoding_name(tensor.encoding()))
            .or_default() += 1;
        if tensor.is_fallback() {
            fallback_tensors += 1;
        }
    }

    let modules = loader
        .modules()
        .iter()
        .map(|m| TcfModuleFacts {
            module_id: m.module_id,
            parent_id: m.parent_id,
            name: m.name.clone(),
            module_role: format!("{:?}", m.module_role),
            preferred_encodings: m
                .preferred_encoding
                .iter()
                .flatten()
                .map(|e| encoding_name(*e))
                .collect(),
            fallback_encoding: m.fallback_encoding.map(encoding_name),
            policy_flags: format!("{:?}", m.policy_flags),
            default_residency: format!("{:?}", m.default_residency),
            tensor_count: owned.get(&m.module_id).copied().unwrap_or(0),
        })
        .collect();

    // Ties break on the encoding name, so two runs over one file print the
    // same histogram.
    let mut encoding_histogram: Vec<(String, usize)> = encodings.into_iter().collect();
    encoding_histogram.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    TcfFileFacts {
        major: header.major,
        minor: header.minor,
        schema_id: header.schema_id,
        file_len: header.file_len,
        data_off: header.data_off,
        contract_count: header.contract_count,
        calibration_count: header.calibration_count,
        relation_count: header.relation_count,
        workload_count: header.workload_count,
        modules,
        encoding_histogram,
        fallback_tensors,
    }
}
