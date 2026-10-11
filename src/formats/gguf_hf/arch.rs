//! Which llama.cpp architecture a HuggingFace checkpoint is written as.
//!
//! `general.architecture` names llama.cpp's architecture, which is not always
//! the checkpoint's `model_type`: `convert_hf_to_gguf.py` writes Mistral and
//! Mixtral as `llama`, and `qwen2_moe` as `qwen2moe`.

/// llama.cpp's architecture for a `model_type` or `architectures` class name.
///
/// `None` for a name outside the architectures this module writes metadata for.
pub fn llama_cpp_arch(name: &str) -> Option<&'static str> {
    Some(match name {
        "llama" | "mistral" | "mixtral" | "LlamaForCausalLM" | "MistralForCausalLM"
        | "MixtralForCausalLM" => "llama",
        "qwen2" | "Qwen2ForCausalLM" => "qwen2",
        "qwen2_moe" | "qwen2moe" | "Qwen2MoeForCausalLM" => "qwen2moe",
        "qwen3" | "Qwen3ForCausalLM" => "qwen3",
        "qwen3_moe" | "qwen3moe" | "Qwen3MoeForCausalLM" => "qwen3moe",
        "gemma" | "GemmaForCausalLM" => "gemma",
        "gemma2" | "Gemma2ForCausalLM" => "gemma2",
        "gemma3" | "gemma3_text" | "Gemma3ForCausalLM" => "gemma3",
        "phi3" | "Phi3ForCausalLM" => "phi3",
        _ => return None,
    })
}

/// The architectures whose HuggingFace tensors map onto llama.cpp's names
/// one to one, with no change to any value but a row reorder.
///
/// The others need a value change llama.cpp's converter makes: Gemma adds 1
/// to every norm weight, Phi-3 fuses projections, and the MoE models merge
/// their experts into one 3-D tensor per projection.
pub const RENAMED_ARCHS: &[&str] = &["llama", "qwen2", "qwen3"];

/// The architectures whose tokenizer splits no space prefix off the first word.
pub const NO_SPACE_PREFIX_ARCHS: &[&str] = &["gemma", "gemma2", "gemma3"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_names_map_to_llama_cpp_architectures() {
        assert_eq!(llama_cpp_arch("llama"), Some("llama"));
        assert_eq!(llama_cpp_arch("mistral"), Some("llama"));
        assert_eq!(llama_cpp_arch("MixtralForCausalLM"), Some("llama"));
        assert_eq!(llama_cpp_arch("qwen2_moe"), Some("qwen2moe"));
        assert_eq!(llama_cpp_arch("gemma3_text"), Some("gemma3"));
        assert_eq!(llama_cpp_arch("voxcpm2"), None);
    }
}
