//! The GGUF header and tensor layout for one HuggingFace checkpoint directory.

use super::arch::llama_cpp_arch;
use super::config::HfConfig;
use super::hparams::arch_metadata;
use super::rope::rope_tensors;
use super::tensor_layout::GgufTensorLayout;
use super::tokenizer::tokenizer_metadata;
use crate::formats::gguf::GgufValue;
use anyhow::Result;
use std::path::Path;

/// What a GGUF output takes from a HuggingFace checkpoint beside its weights.
#[derive(Debug, Clone)]
pub struct HfGgufHeader {
    /// Architecture and tokenizer keys, in llama.cpp's write order.
    pub metadata: Vec<(String, GgufValue)>,
    /// Tensor names, row orders and added RoPE tensors.
    pub layout: GgufTensorLayout,
}

/// The header for the checkpoint in `dir`, written as llama.cpp architecture
/// `arch`. `tensor_names` are the checkpoint's own names.
///
/// `None` when `arch` is outside the architectures llama.cpp's converter
/// handles here: such a file keeps the keys compressr derives from the
/// weights alone.
///
/// # Errors
/// A config or tokenizer file is present but malformed, or the tensor layout
/// needs a head count `config.json` does not state.
pub fn hf_gguf_header(
    arch: &str,
    dir: &Path,
    tensor_names: &[String],
) -> Result<Option<HfGgufHeader>> {
    let Some(arch) = llama_cpp_arch(arch) else {
        return Ok(None);
    };
    let config = match HfConfig::read(dir)? {
        Some(config) => config,
        None => {
            log::warn!(
                "{} has no config.json; the GGUF carries no architecture hyperparameters",
                dir.display()
            );
            HfConfig::default()
        }
    };
    let mut metadata = arch_metadata(arch, &config);
    metadata.extend(tokenizer_metadata(arch, dir, &config)?);
    let layout = GgufTensorLayout::build(
        arch,
        tensor_names,
        config.head_count(),
        config.head_count_kv(),
        rope_tensors(&config),
    )?;
    Ok(Some(HfGgufHeader { metadata, layout }))
}
