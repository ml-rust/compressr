//! A mixed-precision GGUF conversion writes a DIFFERENT block type per tensor.
//!
//! The allocation is unit-tested in `policy::gguf_mix` against the rules from
//! llama.cpp's `llama_tensor_get_type`. What those tests cannot show is that
//! the type survives the writer: GGUF records a type per tensor in its tensor
//! directory, and a writer that reserved bytes for one type and wrote another
//! shifts every later offset in the file with no error raised. So this test
//! converts a checkpoint and reads the finished file back through boostr's
//! GGUF reader — the same reader a loader uses — and checks the type of each
//! tensor against the mix.

use compressr::formats::safetensors::{SafeTensorsDType, SafeTensorsWriter};
use compressr::{ConversionConfig, GgufMix, OutputFormat, QuantizationConfig};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Transformer blocks in the synthetic checkpoint.
///
/// `use_more_bits` divides the layer count by 8 and by 8/7, so four layers put
/// layers 2 and 3 in the richer set and layers 0 and 1 outside it. That is the
/// smallest count that exercises both sides of the rule.
const N_LAYER: usize = 4;

/// Row length of every 2D tensor: one Q4_K / Q6_K super-block.
const ROW: usize = 256;

/// Rows of a layer weight. Small, because the count changes no rule.
const ROWS: usize = 32;

/// Deterministic weights with a varying per-group amplitude, so no sub-block
/// is constant and every format's scale search has something to find.
fn weights(count: usize) -> Vec<u8> {
    let mut state: u32 = 0x9E37_79B9;
    (0..count)
        .flat_map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let amplitude = 1.0 + (i % 7) as f32 * 0.25;
            (unit * amplitude).to_le_bytes()
        })
        .collect()
}

/// Write a llama-shaped SafeTensors checkpoint with a matching `config.json`.
fn write_checkpoint(dir: &Path) {
    let mut writer = SafeTensorsWriter::new();
    let mut add = |name: String, shape: Vec<usize>| {
        let count: usize = shape.iter().product();
        writer
            .add_tensor(name, shape, SafeTensorsDType::F32, weights(count))
            .expect("the checkpoint must accept every tensor");
    };

    add("model.embed_tokens.weight".into(), vec![64, ROW]);
    add("lm_head.weight".into(), vec![64, ROW]);
    add("model.norm.weight".into(), vec![ROW]);

    for layer in 0..N_LAYER {
        for leaf in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            add(
                format!("model.layers.{layer}.{leaf}.weight"),
                vec![ROWS, ROW],
            );
        }
        add(
            format!("model.layers.{layer}.input_layernorm.weight"),
            vec![ROW],
        );
    }

    writer
        .write(dir.join("model.safetensors"))
        .expect("the checkpoint must be writable");
    std::fs::write(
        dir.join("config.json"),
        format!(
            r#"{{"model_type": "llama", "num_hidden_layers": {N_LAYER}, "num_attention_heads": 4, "num_key_value_heads": 4}}"#
        ),
    )
    .expect("config.json must be writable");
}

/// Convert the checkpoint with `mix` and report the GGML type name of every
/// tensor in the finished GGUF, read back through boostr's reader.
fn convert_and_read(mix: GgufMix) -> HashMap<String, String> {
    let input = tempfile::tempdir().expect("a temporary input directory");
    let output = tempfile::tempdir().expect("a temporary output directory");
    write_checkpoint(input.path());

    let out_path = output.path().join("model.gguf");
    let config = ConversionConfig::new(
        input.path().to_string_lossy().into_owned(),
        out_path.to_string_lossy().into_owned(),
        OutputFormat::Gguf,
    )
    .with_quantization(QuantizationConfig::q4_k())
    .with_gguf_mix(mix);

    compressr::convert(config).expect("the conversion must succeed");

    let info = compressr::get_model_info(&out_path.to_string_lossy())
        .expect("the finished file must read back");
    // A `llama` GGUF stores llama.cpp's names. boostr's map gives back the
    // checkpoint names the mix assigned by.
    info.tensors
        .into_iter()
        .map(|tensor| {
            let name = boostr::format::gguf::gguf_to_hf_name(&tensor.name);
            (name, tensor.dtype)
        })
        .collect()
}

/// Q4_K_M is not Q4_K everywhere. The un-embedding takes Q6_K, the
/// attention-V and FFN-down tensors of layers 2 and 3 take Q6_K, and
/// everything else takes the base format.
#[test]
fn q4_k_m_writes_a_different_type_per_tensor() {
    let types = convert_and_read(GgufMix::Q4KM);

    assert_eq!(types["lm_head.weight"], "Q6K");
    assert_eq!(types["model.embed_tokens.weight"], "Q4K");

    for layer in 0..N_LAYER {
        let expected = if layer >= 2 { "Q6K" } else { "Q4K" };
        assert_eq!(
            types[&format!("model.layers.{layer}.self_attn.v_proj.weight")],
            expected,
            "attention-V of layer {layer}"
        );
        assert_eq!(
            types[&format!("model.layers.{layer}.mlp.down_proj.weight")],
            expected,
            "FFN-down of layer {layer}"
        );
        assert_eq!(
            types[&format!("model.layers.{layer}.self_attn.q_proj.weight")],
            "Q4K",
            "attention-Q of layer {layer}"
        );
        // A norm has no row to pack blocks along, so it stays F32.
        assert_eq!(
            types[&format!("model.layers.{layer}.input_layernorm.weight")],
            "F32"
        );
    }

    // More than one block type in one file is the whole point of a mix.
    let distinct: BTreeSet<&String> = types.values().collect();
    assert!(
        distinct.len() >= 3,
        "a mix must write several types, got {distinct:?}"
    );
}

/// Q4_K_S moves the un-embedding and the leading tensors only, and never
/// reaches Q6_K on a layer weight.
#[test]
fn q4_k_s_is_a_smaller_mix_than_q4_k_m() {
    let types = convert_and_read(GgufMix::Q4KS);

    assert_eq!(types["lm_head.weight"], "Q6K");
    // With four attention-V tensors, all four are inside the `i < 4` rule.
    for layer in 0..N_LAYER {
        assert_eq!(
            types[&format!("model.layers.{layer}.self_attn.v_proj.weight")],
            "Q5K",
            "attention-V of layer {layer}"
        );
        assert_eq!(
            types[&format!("model.layers.{layer}.mlp.up_proj.weight")],
            "Q4K"
        );
    }
}

/// The report is measured from the bytes the writer produced, so a mix reads
/// back with several formats and a bits-per-weight between them.
#[test]
fn the_report_states_what_was_written() {
    let input = tempfile::tempdir().expect("a temporary input directory");
    let output = tempfile::tempdir().expect("a temporary output directory");
    write_checkpoint(input.path());

    let stats = compressr::convert(
        ConversionConfig::new(
            input.path().to_string_lossy().into_owned(),
            output
                .path()
                .join("model.gguf")
                .to_string_lossy()
                .into_owned(),
            OutputFormat::Gguf,
        )
        .with_quantization(QuantizationConfig::q4_k())
        .with_gguf_mix(GgufMix::Q4KM),
    )
    .expect("the conversion must succeed");

    let formats: Vec<String> = stats
        .mix
        .per_format()
        .into_iter()
        .map(|usage| usage.format)
        .collect();
    assert!(formats.contains(&"Q4_K".to_string()), "{formats:?}");
    assert!(formats.contains(&"Q6_K".to_string()), "{formats:?}");
    assert!(formats.contains(&"F32".to_string()), "{formats:?}");

    // Q4_K is 4.5 bpw and Q6_K is 6.5, and a handful of F32 norms pull the
    // measured figure up. It cannot sit at or below the base format's rate.
    let bpw = stats.mix.bits_per_weight();
    assert!(bpw > 4.5, "bpw {bpw} must exceed the base format's 4.5");
    assert!(bpw < 8.0, "bpw {bpw} must stay under a byte per weight");
}
