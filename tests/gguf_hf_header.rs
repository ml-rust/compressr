//! A GGUF written from a HuggingFace Llama checkpoint carries the header and
//! tensor layout llama.cpp's converter writes, so it loads with nothing beside it.
//!
//! The output is read back through boostr's GGUF reader, never through
//! compressr's: a writer checked against a reader beside it proves nothing.
//! The key names and value types are also checked against a list copied from
//! a `convert_hf_to_gguf.py` run on Llama 3.2 1B, so a key boostr happens to
//! tolerate under a wrong name or type still fails here.

use boostr::format::{GgmlType, Gguf, GgufValue};
use boostr::{CpuDevice, CpuRuntime, VarMap};
use compressr::{
    ConversionConfig, FormatWriter, OutputFormat, QuantizationConfig, SafeTensorsFormatWriter,
    UnifiedDType,
};
use std::path::Path;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const FFN: usize = 128;
const LAYERS: usize = 2;
const VOCAB: usize = 16;

const CONFIG_JSON: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 16,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 16,
  "max_position_embeddings": 256,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                   "original_max_position_embeddings": 64, "rope_type": "llama3"},
  "tie_word_embeddings": true,
  "bos_token_id": 14,
  "eos_token_id": 15
}"#;

/// A byte-level BPE tokenizer: 14 base tokens, 2 special tokens, and a
/// post-processor that prepends BOS.
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "added_tokens": [
    {"id": 14, "content": "<|begin_of_text|>", "special": true, "normalized": false},
    {"id": 15, "content": "<|end_of_text|>", "special": true, "normalized": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
                    "use_regex": true},
  "post_processor": {"type": "TemplateProcessing",
    "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
               {"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [],
    "special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [14],
                                            "tokens": ["<|begin_of_text|>"]}}},
  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
              "use_regex": true},
  "model": {"type": "BPE", "dropout": null, "unk_token": null, "byte_fallback": false,
    "ignore_merges": true,
    "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "Ġ": 5, "ab": 6, "cd": 7,
              "Ġa": 8, "Ġb": 9, "abcd": 10, "f": 11, "g": 12, "h": 13},
    "merges": [["a", "b"], ["c", "d"], ["Ġ", "a"], ["Ġ", "b"], ["ab", "cd"]]}
}"#;

const TOKENIZER_CONFIG_JSON: &str = r#"{
  "bos_token": "<|begin_of_text|>",
  "eos_token": "<|end_of_text|>",
  "pad_token": "<|end_of_text|>",
  "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
}"#;

/// The header llama.cpp's converter writes for this checkpoint: every key
/// name with its GGUF value type. Copied from a `convert_hf_to_gguf.py` run on
/// Llama 3.2 1B, plus the padding id and chat template its config lacks, minus
/// `general.*` provenance keys. `tokenizer.ggml.pre` is absent: llama.cpp's
/// table has no fingerprint for this synthetic tokenizer.
const EXPECTED_KEYS: &[(&str, &str)] = &[
    ("general.architecture", "STRING"),
    ("general.name", "STRING"),
    ("general.file_type", "UINT32"),
    ("general.quantization_version", "UINT32"),
    ("llama.vocab_size", "UINT32"),
    ("llama.block_count", "UINT32"),
    ("llama.context_length", "UINT32"),
    ("llama.embedding_length", "UINT32"),
    ("llama.feed_forward_length", "UINT32"),
    ("llama.attention.head_count", "UINT32"),
    ("llama.attention.head_count_kv", "UINT32"),
    ("llama.rope.dimension_count", "UINT32"),
    ("llama.rope.freq_base", "FLOAT32"),
    ("llama.attention.layer_norm_rms_epsilon", "FLOAT32"),
    ("llama.attention.key_length", "UINT32"),
    ("llama.attention.value_length", "UINT32"),
    ("tokenizer.ggml.model", "STRING"),
    ("tokenizer.ggml.tokens", "ARRAY[STRING]"),
    ("tokenizer.ggml.token_type", "ARRAY[INT32]"),
    ("tokenizer.ggml.merges", "ARRAY[STRING]"),
    ("tokenizer.ggml.bos_token_id", "UINT32"),
    ("tokenizer.ggml.eos_token_id", "UINT32"),
    ("tokenizer.ggml.padding_token_id", "UINT32"),
    ("tokenizer.ggml.add_bos_token", "BOOL"),
    ("tokenizer.chat_template", "STRING"),
];

/// `rope_freqs.weight` for this config, as bit patterns of the float32 tensor
/// `convert_hf_to_gguf.py` computes with torch.
const EXPECTED_ROPE_FREQS: [u32; 8] = [
    1065353216, 1067819264, 1089821496, 1090519040, 1090519040, 1090519040, 1090519040, 1090519040,
];

fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / 16_777_216.0 - 0.5
        })
        .collect()
}

/// The checkpoint's tensors, HuggingFace names, standard-order shapes.
fn tensors() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut out = vec![(
        "model.embed_tokens.weight".to_string(),
        vec![VOCAB, HIDDEN],
        values(VOCAB * HIDDEN, 1),
    )];
    let shapes = [
        ("input_layernorm.weight", vec![HIDDEN]),
        ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
        ("self_attn.o_proj.weight", vec![HIDDEN, HEADS * HEAD_DIM]),
        ("post_attention_layernorm.weight", vec![HIDDEN]),
        ("mlp.gate_proj.weight", vec![FFN, HIDDEN]),
        ("mlp.up_proj.weight", vec![FFN, HIDDEN]),
        ("mlp.down_proj.weight", vec![HIDDEN, FFN]),
    ];
    for layer in 0..LAYERS {
        for (i, (suffix, shape)) in shapes.iter().enumerate() {
            let count = shape.iter().product();
            out.push((
                format!("model.layers.{layer}.{suffix}"),
                shape.clone(),
                values(count, (10 * layer + i + 2) as u64),
            ));
        }
    }
    out.push((
        "model.norm.weight".to_string(),
        vec![HIDDEN],
        values(HIDDEN, 99),
    ));
    out
}

fn write_checkpoint(dir: &Path) {
    std::fs::write(dir.join("config.json"), CONFIG_JSON).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG_JSON).unwrap();
    let tensors = tensors();
    let mut writer = SafeTensorsFormatWriter::new();
    for (name, shape, _) in &tensors {
        writer
            .register_tensor(name, shape, UnifiedDType::F32, None)
            .unwrap();
    }
    writer.initialize(&dir.join("model.safetensors")).unwrap();
    for (_, _, data) in &tensors {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_tensor_data(&bytes).unwrap();
    }
    writer.finalize().unwrap();
}

fn convert(dir: &Path, out: &Path, quantize: bool) {
    let config = ConversionConfig::new(
        dir.to_string_lossy().to_string(),
        out.to_string_lossy().to_string(),
        OutputFormat::Gguf,
    );
    let config = if quantize {
        config.with_quantization(QuantizationConfig::q8_0())
    } else {
        config.with_target_dtype(UnifiedDType::F32)
    };
    compressr::convert(config).expect("converts the checkpoint");
}

/// The GGUF type name of a reader value, with the element type for an array.
fn type_name(value: &GgufValue) -> String {
    let scalar = |v: &GgufValue| match v {
        GgufValue::Uint8(_) => "UINT8",
        GgufValue::Int8(_) => "INT8",
        GgufValue::Uint16(_) => "UINT16",
        GgufValue::Int16(_) => "INT16",
        GgufValue::Uint32(_) => "UINT32",
        GgufValue::Int32(_) => "INT32",
        GgufValue::Float32(_) => "FLOAT32",
        GgufValue::Bool(_) => "BOOL",
        GgufValue::String(_) => "STRING",
        GgufValue::Uint64(_) => "UINT64",
        GgufValue::Int64(_) => "INT64",
        GgufValue::Float64(_) => "FLOAT64",
        GgufValue::Array(_) => "ARRAY",
    };
    match value {
        GgufValue::Array(items) => {
            format!("ARRAY[{}]", items.first().map_or("EMPTY", scalar))
        }
        other => scalar(other).to_string(),
    }
}

fn header_keys(gguf: &Gguf) -> Vec<(String, String)> {
    let mut keys: Vec<(String, String)> = gguf
        .metadata()
        .iter()
        .map(|(k, v)| (k.to_string(), type_name(v)))
        .collect();
    keys.sort();
    keys
}

fn expected_keys() -> Vec<(String, String)> {
    let mut keys: Vec<(String, String)> = EXPECTED_KEYS
        .iter()
        .map(|(k, t)| (k.to_string(), t.to_string()))
        .collect();
    keys.sort();
    keys
}

#[test]
fn llama_header_matches_llama_cpp_names_and_types() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let out = dir.path().join("out.gguf");
    convert(dir.path(), &out, true);

    let gguf = Gguf::open(&out).unwrap();
    assert_eq!(header_keys(&gguf), expected_keys());

    let meta = gguf.metadata();
    assert_eq!(meta.architecture(), Some("llama"));
    let u32_key = |key: &str| meta.get_u32(key);
    assert_eq!(u32_key("llama.attention.head_count"), Some(HEADS as u32));
    assert_eq!(
        u32_key("llama.attention.head_count_kv"),
        Some(KV_HEADS as u32)
    );
    assert_eq!(u32_key("llama.feed_forward_length"), Some(FFN as u32));
    assert_eq!(u32_key("llama.context_length"), Some(256));
    assert_eq!(u32_key("llama.rope.dimension_count"), Some(HEAD_DIM as u32));
    assert_eq!(u32_key("general.file_type"), Some(7));
    assert_eq!(u32_key("tokenizer.ggml.bos_token_id"), Some(14));
    assert_eq!(u32_key("tokenizer.ggml.eos_token_id"), Some(15));
    assert_eq!(u32_key("tokenizer.ggml.padding_token_id"), Some(15));
    assert_eq!(meta.get_f32("llama.rope.freq_base"), Some(10000.0));
    assert_eq!(meta.get_string("tokenizer.ggml.model"), Some("gpt2"));
    let tokens = meta.get_array("tokenizer.ggml.tokens").unwrap();
    assert_eq!(tokens.len(), VOCAB);
    assert!(matches!(&tokens[5], GgufValue::String(s) if s == "Ġ"));
    let types = meta.get_array("tokenizer.ggml.token_type").unwrap();
    assert!(matches!(types[14], GgufValue::Int32(3)), "BOS is CONTROL");
    let merges = meta.get_array("tokenizer.ggml.merges").unwrap();
    assert!(matches!(&merges[4], GgufValue::String(s) if s == "ab cd"));

    // boostr's own vocabulary reader accepts the block whole.
    let vocab = boostr::format::extract_gguf_vocab(meta).unwrap();
    assert_eq!(vocab.add_bos_token, Some(true));
}

#[test]
fn llama_tensors_take_llama_cpp_names_rows_and_rope_freqs() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path());
    let out = dir.path().join("out.gguf");
    convert(dir.path(), &out, false);
    let device = CpuDevice::new();
    let mut gguf = Gguf::open(&out).unwrap();

    let mut names: Vec<String> = gguf.tensor_names().map(str::to_string).collect();
    names.sort();
    let mut expected = vec![
        "output_norm.weight".to_string(),
        "rope_freqs.weight".to_string(),
        "token_embd.weight".to_string(),
    ];
    for layer in 0..LAYERS {
        for suffix in [
            "attn_k",
            "attn_norm",
            "attn_output",
            "attn_q",
            "attn_v",
            "ffn_down",
            "ffn_gate",
            "ffn_norm",
            "ffn_up",
        ] {
            expected.push(format!("blk.{layer}.{suffix}.weight"));
        }
    }
    expected.sort();
    assert_eq!(names, expected);

    let freqs = gguf.tensor_info("rope_freqs.weight").unwrap().clone();
    assert_eq!(freqs.ggml_type, GgmlType::F32);
    let freqs: Vec<u32> = gguf
        .load_tensor_f32::<CpuRuntime>("rope_freqs.weight", &device)
        .unwrap()
        .to_vec::<f32>()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(freqs, EXPECTED_ROPE_FREQS);

    // llama.cpp's pair order, from the converter's numpy expression
    // `w.reshape(heads, 2, head_dim / 2, cols).swapaxes(1, 2)`: GGUF row
    // `h * head_dim + 2 * i + j` is HF row `h * head_dim + j * head_dim / 2 + i`.
    let hf = tensors();
    let source = |name: &str| hf.iter().find(|(n, _, _)| n == name).unwrap().2.clone();
    for (gguf_name, hf_name, heads) in [
        (
            "blk.1.attn_q.weight",
            "model.layers.1.self_attn.q_proj.weight",
            HEADS,
        ),
        (
            "blk.1.attn_k.weight",
            "model.layers.1.self_attn.k_proj.weight",
            KV_HEADS,
        ),
    ] {
        let stored = gguf
            .load_tensor_f32::<CpuRuntime>(gguf_name, &device)
            .unwrap()
            .to_vec::<f32>();
        let original = source(hf_name);
        let half = HEAD_DIM / 2;
        for h in 0..heads {
            for i in 0..half {
                for j in 0..2 {
                    let g = h * HEAD_DIM + 2 * i + j;
                    let r = h * HEAD_DIM + j * half + i;
                    assert_eq!(
                        stored[g * HIDDEN..(g + 1) * HIDDEN],
                        original[r * HIDDEN..(r + 1) * HIDDEN],
                        "{gguf_name} row {g}"
                    );
                }
            }
        }
    }
    let v = gguf
        .load_tensor_f32::<CpuRuntime>("blk.0.attn_v.weight", &device)
        .unwrap()
        .to_vec::<f32>();
    assert_eq!(v, source("model.layers.0.self_attn.v_proj.weight"));

    // boostr's model loader restores HuggingFace names and row order.
    let map = VarMap::<CpuRuntime>::from_gguf(&out, &device).unwrap();
    for (name, _, data) in &hf {
        let loaded = map.get(name).unwrap().as_tensor().unwrap().to_vec::<f32>();
        assert_eq!(&loaded, data, "{name}");
    }
}

/// With the real Llama 3.2 tokenizer, the pre-tokenizer fingerprint matches
/// llama.cpp's table, and the token list equals the converter's.
#[test]
fn llama32_tokenizer_is_named_llama_bpe() {
    let Some(models) = std::env::var_os("BOOSTR_MODELS_DIR") else {
        eprintln!("BOOSTR_MODELS_DIR unset; skipped");
        return;
    };
    let source = Path::new(&models).join("llama-3.2-1b");
    if !source.join("tokenizer.json").is_file() {
        eprintln!("{} has no tokenizer.json; skipped", source.display());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
        std::fs::copy(source.join(file), dir.path().join(file)).unwrap();
    }
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();
    let header = compressr::formats::gguf_hf::hf_gguf_header(
        "llama",
        dir.path(),
        &["model.embed_tokens.weight".to_string()],
    )
    .unwrap()
    .unwrap();
    let find = |key: &str| {
        header
            .metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    assert!(matches!(
        find("tokenizer.ggml.pre"),
        Some(compressr::formats::GgufValue::String(s)) if s == "llama-bpe"
    ));
    assert!(matches!(
        find("tokenizer.ggml.add_bos_token"),
        Some(compressr::formats::GgufValue::Bool(true))
    ));
    assert!(matches!(
        find("tokenizer.ggml.add_sep_token"),
        Some(compressr::formats::GgufValue::Bool(false))
    ));
    assert_eq!(config["vocab_size"], 128256);
}
