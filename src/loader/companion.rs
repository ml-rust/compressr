//! A companion source: a second file whose tensors a [`LoadedModel`] exposes
//! under a prefix, beside the primary checkpoint's own.
//!
//! VoxCPM2 ships its `AudioVAE` as `audiovae.pth` next to
//! `model.safetensors`. A conversion of that directory folds the codec into
//! the same output, so the loader opens the companion when the input is a
//! safetensors directory whose architecture is VoxCPM2 and chains it into
//! enumeration and lookup with every name under
//! [`AUDIO_VAE_PREFIX`](crate::policy::AUDIO_VAE_PREFIX). The
//! output writer's duplicate-name check stays the guard against a
//! collision; the prefix keeps one from happening.
//!
//! Format detection is untouched: the companion is opened AFTER the primary
//! is classified, and only for this one architecture. Every other input
//! loads exactly as before.
//!
//! [`LoadedModel`]: super::LoadedModel

use super::torch::TorchModel;
use super::{SafeTensors, UnifiedTensorInfo};
use crate::policy::{is_voxcpm2, resolve_architecture, AUDIO_VAE_PREFIX};
use anyhow::{Context, Result};
use std::path::Path;

/// The AudioVAE files a VoxCPM2 directory can hold, in preference order.
///
/// A converted `audiovae.safetensors` is already weight-norm-folded and is
/// preferred, as boostr's own `VaeCheckpoint` prefers it; the published
/// `.pth` is folded by [`TorchModel`] on read.
const AUDIO_VAE_FILES: [&str; 2] = ["audiovae.safetensors", "audiovae.pth"];

/// The reader behind a companion.
enum CompanionSource {
    /// An already folded `audiovae.safetensors`.
    SafeTensors(SafeTensors),
    /// The published `audiovae.pth`, folded on read.
    Torch(Box<TorchModel>),
}

/// A second file exposed under a name prefix.
pub struct Companion {
    prefix: &'static str,
    source: CompanionSource,
}

impl Companion {
    /// Open the AudioVAE beside a VoxCPM2 checkpoint directory, or `None`.
    ///
    /// `None` when the directory is not VoxCPM2 (by `--arch` or by its
    /// `config.json`, exactly as [`resolve_architecture`] decides for
    /// `general.architecture`), when no architecture can be resolved at all,
    /// or when neither AudioVAE file is present.
    ///
    /// # Errors
    /// An AudioVAE file is present and fails to open.
    pub fn for_voxcpm2(dir: &Path, arch_override: Option<&str>) -> Result<Option<Self>> {
        let Ok(arch) = resolve_architecture(arch_override, None, dir) else {
            return Ok(None);
        };
        if !is_voxcpm2(&arch) {
            return Ok(None);
        }
        let Some(file) = AUDIO_VAE_FILES
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file())
        else {
            return Ok(None);
        };
        log::info!(
            "VoxCPM2 checkpoint: folding {} into the output under `{}`",
            file.display(),
            AUDIO_VAE_PREFIX
        );
        let source = if file.extension().is_some_and(|ext| ext == "safetensors") {
            CompanionSource::SafeTensors(
                SafeTensors::open(&file)
                    .with_context(|| format!("Failed to open AudioVAE {}", file.display()))?,
            )
        } else {
            CompanionSource::Torch(Box::new(TorchModel::open(&file)?))
        };
        Ok(Some(Self {
            prefix: AUDIO_VAE_PREFIX,
            source,
        }))
    }

    /// The companion's own name for an exposed `name`, or `None` when the
    /// name is not under the prefix.
    fn inner_name<'n>(&self, name: &'n str) -> Option<&'n str> {
        name.strip_prefix(self.prefix)
    }

    /// Number of tensors the companion exposes.
    pub fn tensor_count(&self) -> usize {
        match &self.source {
            CompanionSource::SafeTensors(st) => st.len(),
            CompanionSource::Torch(torch) => torch.len(),
        }
    }

    /// Exposed names, prefixed and sorted.
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = match &self.source {
            CompanionSource::SafeTensors(st) => st
                .tensor_names()
                .map(|name| format!("{}{name}", self.prefix))
                .collect(),
            // Already sorted, but the prefix is what callers see, so the
            // order is fixed here too.
            CompanionSource::Torch(torch) => torch
                .tensor_names()
                .into_iter()
                .map(|name| format!("{}{name}", self.prefix))
                .collect(),
        };
        names.sort_unstable();
        names
    }

    /// Shape and element type of an exposed `name`, reported under that name.
    pub fn tensor_info(&self, name: &str) -> Option<UnifiedTensorInfo> {
        let inner = self.inner_name(name)?;
        let mut info = match &self.source {
            CompanionSource::SafeTensors(st) => st
                .tensor_info(inner)
                .ok()
                .map(UnifiedTensorInfo::from_safetensors)?,
            CompanionSource::Torch(torch) => torch.tensor_info(inner)?,
        };
        info.name = name.to_string();
        Some(info)
    }

    /// Read an exposed `name`'s bytes, or `None` when the name is not the
    /// companion's.
    pub fn read_tensor_bytes(&mut self, name: &str) -> Option<Result<Vec<u8>>> {
        let inner = self.inner_name(name)?;
        match &mut self.source {
            CompanionSource::SafeTensors(st) => {
                st.tensor_info(inner).ok()?;
                Some(
                    st.read_tensor_bytes(inner)
                        .map_err(|e| anyhow::anyhow!("Failed to read tensor '{name}': {e}")),
                )
            }
            CompanionSource::Torch(torch) => {
                torch.tensor_info(inner)?;
                Some(torch.read_tensor_bytes(inner))
            }
        }
    }
}
