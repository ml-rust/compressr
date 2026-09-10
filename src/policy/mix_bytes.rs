//! What one tensor costs in a GGUF mix, in the bytes the writer will write.
//!
//! Every allocation that spends a byte budget — the two-way promotion in
//! [`super::mix_measured`] and the ladder in [`super::mix_ladder`] — has to
//! agree with the registration loop about how many bytes a tensor takes, or
//! the budget it thinks it spent is not the budget the file spends. So the
//! byte count is read from the GGUF writer's own block table, never from a
//! nominal bit width, and both allocations read it here.

use crate::formats::GgmlType;
use crate::loader::UnifiedDType;
use crate::quantization::QuantType;

use super::mix_rules::plan_mixed_tensor;
use super::TensorPlan;

/// Elements in a tensor of this shape.
pub fn element_count(shape: &[usize]) -> u64 {
    shape.iter().product::<usize>() as u64
}

/// Whether a row of this shape is a whole number of the format's blocks.
///
/// GGUF packs blocks along a ROW, which is the LAST dimension of the standard
/// order shape. For `Q4_K` and `Q6_K` that block is 256 elements, so a row
/// that is not a multiple of 256 can hold neither.
pub fn holds_format(shape: &[usize], format: QuantType) -> bool {
    shape.len() >= 2 && shape[shape.len() - 1].is_multiple_of(format.block_size())
}

/// Extra bytes moving one tensor from `base` to `format` costs.
///
/// Zero when `format` is the cheaper of the two: the caller spends a budget,
/// and a saving is accounted where the allocation starts rather than as a
/// negative cost here.
pub fn promotion_cost(
    shape: &[usize],
    dtype: UnifiedDType,
    elements: u64,
    base: QuantType,
    format: QuantType,
) -> u64 {
    format_bytes(shape, dtype, elements, format)
        .saturating_sub(format_bytes(shape, dtype, elements, base))
}

/// Bytes one tensor takes when the mix assigns it this format.
pub fn format_bytes(shape: &[usize], dtype: UnifiedDType, elements: u64, format: QuantType) -> u64 {
    plan_bytes(
        plan_mixed_tensor(shape, dtype, Some(format)),
        elements,
        dtype,
    )
}

/// Bytes one plan writes for a tensor of this many elements, read from the
/// GGUF writer's own block table rather than from a nominal bit width.
pub fn plan_bytes(plan: TensorPlan, elements: u64, dtype: UnifiedDType) -> u64 {
    match plan {
        TensorPlan::Quantize(format) => {
            GgmlType::from_quant_type(format).tensor_bytes(elements as usize) as u64
        }
        // An unquantizable tensor is stored as F32, which is what the
        // registration loop reserves for it.
        TensorPlan::StoreF32(_) => elements * 4,
        TensorPlan::StoreAsIs => elements * dtype.element_size().unwrap_or(4) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 256-element row holds a K-quant super-block; a 96-element row holds
    /// none, and a rank-1 buffer has no row at all.
    #[test]
    fn a_row_holds_a_format_or_it_does_not() {
        assert!(holds_format(&[512, 512], QuantType::Q4K));
        assert!(!holds_format(&[512, 96], QuantType::Q4K));
        assert!(!holds_format(&[512], QuantType::Q4K));
        assert!(holds_format(&[512, 96], QuantType::Q8_0));
    }

    /// A wider format costs more, and the difference is what a promotion
    /// spends.
    #[test]
    fn a_promotion_costs_the_difference() {
        let shape = [512usize, 512];
        let elements = element_count(&shape);
        assert_eq!(elements, 512 * 512);
        let at_q4 = format_bytes(&shape, UnifiedDType::F32, elements, QuantType::Q4K);
        let at_q6 = format_bytes(&shape, UnifiedDType::F32, elements, QuantType::Q6K);
        assert!(at_q6 > at_q4);
        assert_eq!(
            promotion_cost(
                &shape,
                UnifiedDType::F32,
                elements,
                QuantType::Q4K,
                QuantType::Q6K
            ),
            at_q6 - at_q4
        );
        // A demotion is no cost, which is how a budget reads it.
        assert_eq!(
            promotion_cost(
                &shape,
                UnifiedDType::F32,
                elements,
                QuantType::Q6K,
                QuantType::Q4K
            ),
            0
        );
    }
}
