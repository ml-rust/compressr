//! What a conversion actually wrote, per format and per tensor.
//!
//! A mixed-precision comparison is only meaningful if it states what it
//! compared. `Q4_K_M` names an allocation, not a size: the achieved bits per
//! weight moves with the model's shape, with how many tensors were too small
//! or too unaligned to quantize, and with the embedding table's share of the
//! parameters. So the numbers here are MEASURED from the bytes the writer
//! produced, never taken from a format's nominal bit width.

use super::{FallbackReason, TensorPlan};
use crate::loader::UnifiedDType;
use crate::quantization::QuantType;
use std::collections::BTreeMap;

/// One tensor as it was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixEntry {
    /// Tensor name, as written to the output
    pub name: String,
    /// The format its bytes are in
    pub format: String,
    /// Elements in the tensor
    pub elements: u64,
    /// Bytes written for it
    pub bytes: u64,
    /// Why it is not in the mix's base format, when it is not
    pub fallback: Option<FallbackReason>,
}

/// How much of the output one format accounts for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatUsage {
    /// The format
    pub format: String,
    /// Tensors written in it
    pub tensors: usize,
    /// Elements those tensors hold
    pub elements: u64,
    /// Bytes they occupy
    pub bytes: u64,
}

impl FormatUsage {
    /// Bits per weight this format achieved, over its own tensors.
    pub fn bits_per_weight(&self) -> f64 {
        if self.elements == 0 {
            return 0.0;
        }
        self.bytes as f64 * 8.0 / self.elements as f64
    }
}

/// Every tensor a conversion wrote, in write order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MixReport {
    entries: Vec<MixEntry>,
}

impl MixReport {
    /// Record one tensor.
    pub fn record(&mut self, entry: MixEntry) {
        self.entries.push(entry);
    }

    /// The tensors, in write order.
    pub fn entries(&self) -> &[MixEntry] {
        &self.entries
    }

    /// Elements over every tensor written.
    pub fn total_elements(&self) -> u64 {
        self.entries.iter().map(|e| e.elements).sum()
    }

    /// Bytes over every tensor written.
    ///
    /// This is the weight payload only. A GGUF file also carries its header
    /// and its metadata, so the file on disk is larger.
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// Bits per weight over every tensor written, including the ones no mix
    /// quantizes.
    ///
    /// This is the number a size comparison quotes, because the norms, the
    /// router and the unquantizable rows are in the file too.
    pub fn bits_per_weight(&self) -> f64 {
        let elements = self.total_elements();
        if elements == 0 {
            return 0.0;
        }
        self.total_bytes() as f64 * 8.0 / elements as f64
    }

    /// One row per format, ordered by format name.
    pub fn per_format(&self) -> Vec<FormatUsage> {
        let mut totals: BTreeMap<&str, FormatUsage> = BTreeMap::new();
        for entry in &self.entries {
            let usage = totals
                .entry(entry.format.as_str())
                .or_insert_with(|| FormatUsage {
                    format: entry.format.clone(),
                    tensors: 0,
                    elements: 0,
                    bytes: 0,
                });
            usage.tensors += 1;
            usage.elements += entry.elements;
            usage.bytes += entry.bytes;
        }
        totals.into_values().collect()
    }
}

impl std::fmt::Display for MixReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Per-tensor formats written:")?;
        for usage in self.per_format() {
            writeln!(
                f,
                "  {:<6} {:>5} tensor(s)  {:>14} bytes  {:>5.2} bpw",
                usage.format,
                usage.tensors,
                usage.bytes,
                usage.bits_per_weight()
            )?;
        }
        write!(
            f,
            "  bpw achieved over {} weights: {:.3} (weight payload only, {} bytes)",
            self.total_elements(),
            self.bits_per_weight(),
            self.total_bytes()
        )
    }
}

/// The format label one plan writes its bytes in.
///
/// `stored` is the element type an unquantized tensor keeps, which the caller
/// already resolved from the conversion's cast target.
pub fn plan_format_label(plan: TensorPlan, stored: UnifiedDType) -> String {
    match plan {
        TensorPlan::Quantize(quant_type) => quant_type_label(quant_type).to_string(),
        TensorPlan::StoreF32(_) => "F32".to_string(),
        TensorPlan::StoreAsIs => stored.to_string(),
    }
}

/// The GGML name of a quantization type.
///
/// These are the names a GGUF reader prints, so a report that uses them can
/// be compared against `llama-quantize` output directly.
pub fn quant_type_label(quant_type: QuantType) -> &'static str {
    match quant_type {
        QuantType::Q4_0 => "Q4_0",
        QuantType::Q4_1 => "Q4_1",
        QuantType::Q5_0 => "Q5_0",
        QuantType::Q5_1 => "Q5_1",
        QuantType::Q8_0 => "Q8_0",
        QuantType::Q8_1 => "Q8_1",
        QuantType::Q2K => "Q2_K",
        QuantType::Q3K => "Q3_K",
        QuantType::Q4K => "Q4_K",
        QuantType::Q5K => "Q5_K",
        QuantType::Q6K => "Q6_K",
        QuantType::NF4 => "NF4",
        QuantType::FP4 => "FP4",
        QuantType::FP8 => "FP8",
        QuantType::GPTQ => "GPTQ",
        QuantType::AWQ => "AWQ",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, format: &str, elements: u64, bytes: u64) -> MixEntry {
        MixEntry {
            name: name.into(),
            format: format.into(),
            elements,
            bytes,
            fallback: None,
        }
    }

    /// Q4_K packs 256 weights into 144 bytes, which is 4.5 bits each.
    #[test]
    fn the_bpw_is_measured_from_the_bytes() {
        let mut report = MixReport::default();
        report.record(entry("blk.0.ffn_up.weight", "Q4_K", 256, 144));
        assert!((report.bits_per_weight() - 4.5).abs() < 1e-9);
    }

    /// A tensor no mix quantizes is in the file, so it is in the denominator
    /// and in the numerator of the achieved bpw.
    #[test]
    fn unquantized_tensors_count_towards_the_achieved_bpw() {
        let mut report = MixReport::default();
        report.record(entry("blk.0.ffn_up.weight", "Q4_K", 256, 144));
        report.record(entry("blk.0.attn_norm.weight", "F32", 256, 1024));
        // 1168 bytes over 512 weights.
        assert!((report.bits_per_weight() - 18.25).abs() < 1e-9);
    }

    #[test]
    fn the_per_format_rows_aggregate_by_format() {
        let mut report = MixReport::default();
        report.record(entry("a", "Q4_K", 256, 144));
        report.record(entry("b", "Q4_K", 512, 288));
        report.record(entry("c", "Q6_K", 256, 210));

        let rows = report.per_format();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].format, "Q4_K");
        assert_eq!(rows[0].tensors, 2);
        assert_eq!(rows[0].elements, 768);
        assert_eq!(rows[0].bytes, 432);
        assert!((rows[0].bits_per_weight() - 4.5).abs() < 1e-9);
        assert_eq!(rows[1].format, "Q6_K");
    }

    #[test]
    fn an_empty_report_has_no_bpw_rather_than_a_nan() {
        assert_eq!(MixReport::default().bits_per_weight(), 0.0);
    }

    #[test]
    fn a_plan_labels_itself_with_the_ggml_name() {
        assert_eq!(
            plan_format_label(TensorPlan::Quantize(QuantType::Q6K), UnifiedDType::F32),
            "Q6_K"
        );
        assert_eq!(
            plan_format_label(
                TensorPlan::StoreF32(FallbackReason::Rank),
                UnifiedDType::F16
            ),
            "F32"
        );
    }
}
