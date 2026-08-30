//! PyTorch `.pt` / `.pth` / `.bin` input.
//!
//! The reader is `boostr::format::torch_pt`: a ZIP container walk plus a
//! pickle VM that never executes Python. compressr adds the two conventions a
//! checkpoint carries that a raw state dict does not describe:
//!
//! 1. A `.pth` top level is often `{"metadata": ..., "state_dict": {...}}`,
//!    and the pickle flattener carries that root into the keys, so a tensor
//!    arrives as `state_dict.encoder.weight`. The root is stripped, because
//!    every consumer of the output expects `encoder.weight`.
//! 2. A checkpoint saved through `torch.nn.utils.weight_norm` stores
//!    `X.weight_g` / `X.weight_v` instead of `X.weight`. The pair is folded
//!    back into `X.weight` by `boostr::nn::fuse_weight_norm`, the one copy of
//!    that arithmetic.
//!
//! Folding is a conversion, not a re-serialization: the output of a `convert`
//! is a GGUF, TCF or SafeTensors artifact meant to be RUN, and no consumer of
//! those understands a weight-norm reparameterization. A faithful copy of the
//! pair would produce a checkpoint that loads nowhere.
//!
//! # Memory
//!
//! `boostr`'s state-dict handle exposes tensor keys but no shape or element
//! type without materializing the tensor, so the open pass loads each tensor
//! once, records its shape and dtype, and drops it. Peak resident memory is
//! still one tensor, but the file is read twice over a conversion.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use boostr::format::TorchStateDict;
use boostr::nn::fuse_weight_norm;
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor};

use super::{UnifiedDType, UnifiedTensorInfo};

/// The root a `torch.save({"state_dict": ...})` checkpoint puts in front of
/// every key.
const STATE_DICT_ROOT: &str = "state_dict.";

/// Suffix of the weight-norm direction tensor.
const WEIGHT_NORM_V: &str = ".weight_v";

/// Suffix of the weight-norm magnitude tensor.
const WEIGHT_NORM_G: &str = ".weight_g";

/// The output-channel axis `torch.nn.utils.weight_norm` defaults to.
const WEIGHT_NORM_DIM: usize = 0;

/// Which state-dict entries produce one exposed tensor.
#[derive(Debug, Clone)]
enum TorchSource {
    /// One state-dict key, copied through.
    Direct(String),
    /// A `(weight_v, weight_g)` pair, folded into one weight.
    WeightNorm { v: String, g: String },
}

/// An open PyTorch state dict: metadata in memory, storages still in the ZIP.
pub struct TorchModel {
    dict: TorchStateDict,
    path: PathBuf,
    device: CpuDevice,
    /// Exposed names, sorted, so a conversion of one file always visits its
    /// tensors in the same order.
    names: Vec<String>,
    sources: HashMap<String, TorchSource>,
    infos: HashMap<String, UnifiedTensorInfo>,
}

impl TorchModel {
    /// Open `path` and record every tensor's shape and element type.
    ///
    /// # Errors
    ///
    /// - The file is not a PyTorch ZIP container, or its pickle names a
    ///   construct the reader does not implement.
    /// - A tensor's element type has no reader in boostr. The message names
    ///   the tensor and the type.
    pub fn open(path: &Path) -> Result<Self> {
        let dict = TorchStateDict::open(path)
            .map_err(|e| anyhow!("Failed to open PyTorch file {}: {}", path.display(), e))?;

        let mut keys: Vec<String> = dict.keys().map(str::to_string).collect();
        keys.sort_unstable();
        if keys.is_empty() {
            return Err(anyhow!(
                "PyTorch file {} holds no tensor: its pickle declared none.",
                path.display()
            ));
        }
        let strip = keys.iter().all(|k| k.starts_with(STATE_DICT_ROOT));

        // Exposed name -> state-dict key, before the weight-norm fold.
        let exposed: Vec<(String, String)> = keys
            .iter()
            .map(|key| (exposed_name(key, strip, path), key.clone()))
            .collect();
        let by_name: HashMap<&str, &str> = exposed
            .iter()
            .map(|(name, key)| (name.as_str(), key.as_str()))
            .collect();

        let mut sources: HashMap<String, TorchSource> = HashMap::with_capacity(exposed.len());
        let mut names: Vec<String> = Vec::with_capacity(exposed.len());
        for (name, key) in &exposed {
            // A magnitude tensor is consumed by its direction tensor below.
            if let Some(base) = name.strip_suffix(WEIGHT_NORM_G) {
                if by_name.contains_key(format!("{base}{WEIGHT_NORM_V}").as_str()) {
                    continue;
                }
            }
            if let Some(base) = name.strip_suffix(WEIGHT_NORM_V) {
                if let Some(g_key) = by_name.get(format!("{base}{WEIGHT_NORM_G}").as_str()) {
                    let fused = format!("{base}.weight");
                    if sources.contains_key(&fused) {
                        return Err(anyhow!(
                            "PyTorch file {} holds both '{}' and the weight-norm pair that \
                             folds into it. compressr cannot tell which one the model uses.",
                            path.display(),
                            fused
                        ));
                    }
                    log::info!("Folding weight-norm pair into '{}'", fused);
                    sources.insert(
                        fused.clone(),
                        TorchSource::WeightNorm {
                            v: key.clone(),
                            g: (*g_key).to_string(),
                        },
                    );
                    names.push(fused);
                    continue;
                }
            }
            if sources
                .insert(name.clone(), TorchSource::Direct(key.clone()))
                .is_some()
            {
                return Err(anyhow!(
                    "PyTorch file {} holds the tensor name '{}' twice.",
                    path.display(),
                    name
                ));
            }
            names.push(name.clone());
        }
        names.sort_unstable();

        let device = CpuDevice::new();
        let mut infos = HashMap::with_capacity(names.len());
        for name in &names {
            let source = sources.get(name).ok_or_else(|| {
                anyhow!("Internal error: PyTorch tensor '{name}' has no source entry")
            })?;
            // The state-dict handle answers keys only, so the shape and the
            // element type come from the tensor itself. It is dropped again
            // before the next one is read.
            let key = match source {
                TorchSource::Direct(key) => key,
                TorchSource::WeightNorm { v, .. } => v,
            };
            let tensor = load_key(&dict, key, path, &device)?;
            let shape = tensor.shape().to_vec();
            let dtype = tensor.dtype();
            let numel: usize = shape.iter().product();
            let element_size = dtype.size_in_bytes();
            infos.insert(
                name.clone(),
                UnifiedTensorInfo {
                    name: name.clone(),
                    shape,
                    dtype: UnifiedDType::from_dtype(dtype),
                    size_bytes: numel * element_size,
                    numel,
                    tcf: None,
                },
            );
        }

        Ok(Self {
            dict,
            path: path.to_path_buf(),
            device,
            names,
            sources,
            infos,
        })
    }

    /// Exposed tensor names, sorted.
    pub fn tensor_names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// Number of exposed tensors. A folded weight-norm pair counts once.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// True when the file exposes no tensor.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Shape and element type of `name`, or `None` when there is no such
    /// tensor.
    pub fn tensor_info(&self, name: &str) -> Option<UnifiedTensorInfo> {
        self.infos.get(name).cloned()
    }

    /// Read `name` into little-endian bytes of its own element type.
    ///
    /// # Errors
    /// There is no such tensor, a storage is short or misaligned, or the
    /// weight-norm fold rejects the pair.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let source = self
            .sources
            .get(name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?;
        let tensor = match source {
            TorchSource::Direct(key) => load_key(&self.dict, key, &self.path, &self.device)?,
            TorchSource::WeightNorm { v, g } => {
                let v_tensor = load_key(&self.dict, v, &self.path, &self.device)?;
                let g_tensor = load_key(&self.dict, g, &self.path, &self.device)?;
                let client = CpuClient::new(self.device.clone());
                fuse_weight_norm(&client, &v_tensor, &g_tensor, WEIGHT_NORM_DIM).map_err(|e| {
                    anyhow!(
                        "Failed to fold the weight-norm pair ('{}', '{}') of {} into '{}': {}",
                        v,
                        g,
                        self.path.display(),
                        name,
                        e
                    )
                })?
            }
        };
        tensor_bytes(&tensor, name)
    }
}

/// The name a state-dict key is exposed under.
///
/// `torch.save(tensor)` produces a bare tensor whose flattened key is empty.
/// An empty name is not writable into any output format, so the file's own
/// stem names it.
fn exposed_name(key: &str, strip_root: bool, path: &Path) -> String {
    let key = if strip_root {
        key.strip_prefix(STATE_DICT_ROOT).unwrap_or(key)
    } else {
        key
    };
    if !key.is_empty() {
        return key.to_string();
    }
    path.file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "tensor".to_string())
}

/// Materialize one state-dict key on the host.
fn load_key(
    dict: &TorchStateDict,
    key: &str,
    path: &Path,
    device: &CpuDevice,
) -> Result<Tensor<CpuRuntime>> {
    dict.load_tensor::<CpuRuntime>(key, device)
        .map_err(|e| anyhow!("{}: {}", path.display(), e))
        .with_context(|| format!("Failed to read PyTorch tensor '{key}'"))
}

/// Copy a host tensor out as little-endian bytes of its own element type.
fn tensor_bytes(tensor: &Tensor<CpuRuntime>, name: &str) -> Result<Vec<u8>> {
    match tensor.dtype() {
        DType::F32 => Ok(tensor
            .to_vec::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()),
        DType::F64 => Ok(tensor
            .to_vec::<f64>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()),
        other => Err(anyhow!(
            "PyTorch tensor '{}' is {:?}, which boostr's .pt reader does not decode. Its \
             reader covers F32 and F64.",
            name,
            other
        )),
    }
}
