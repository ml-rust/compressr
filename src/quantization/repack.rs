//! The one call into boostr's lowbit repack.
//!
//! PTQ1_0 and PQ2_0 hold the same codes under the same block size, so a
//! move between them copies codes and scales and rounds nothing. boostr
//! owns both layouts; compressr names the direction and hands over the
//! bytes.

use anyhow::{anyhow, bail, Result};

use super::types::QuantType;
use crate::loader::UnifiedDType;
use boostr::quant::repack::{repack_pq2_0_to_ptq1_0, repack_ptq1_0_to_pq2_0};

/// Repack one tensor's blocks from `source` into `target`.
///
/// `numel` is the tensor's element count. The output is the byte stream a
/// GGUF tensor of `target` stores, in the same block order as `data`.
///
/// # Errors
/// - `(source, target)` is not `(PTQ1_0, PQ2_0)` or `(PQ2_0, PTQ1_0)`.
/// - `data` is not the byte count `source` stores for `numel` elements.
/// - A PQ2_0 block holds code 3, whose value `2 * d` has no trit. The
///   error names the tensor and the block.
pub fn repack_lowbit(
    name: &str,
    data: &[u8],
    source: UnifiedDType,
    target: QuantType,
    numel: usize,
) -> Result<Vec<u8>> {
    let repacked = match (source, target) {
        (UnifiedDType::PTQ1_0, QuantType::PQ2_0) => repack_ptq1_0_to_pq2_0(data, numel),
        (UnifiedDType::PQ2_0, QuantType::PTQ1_0) => repack_pq2_0_to_ptq1_0(data, numel),
        (source, target) => bail!(
            "tensor '{name}': no repack from {source} to {target:?}; the only moves are \
             PTQ1_0 to PQ2_0 and PQ2_0 to PTQ1_0"
        ),
    };
    repacked.map_err(|e| anyhow!("tensor '{name}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::quant::cpu::kernels::lowbit_codec::{encode_ptq1_0_block, LOWBIT_BLOCK_SIZE};

    fn one_block() -> Vec<u8> {
        let codes: [u8; LOWBIT_BLOCK_SIZE] = std::array::from_fn(|i| (i % 3) as u8);
        encode_ptq1_0_block(&codes, [0x00, 0x3C]).unwrap().to_vec()
    }

    #[test]
    fn both_directions_round_trip() {
        let ptq = one_block();
        let pq = repack_lowbit("w", &ptq, UnifiedDType::PTQ1_0, QuantType::PQ2_0, 128).unwrap();
        assert_eq!(pq.len(), 34);
        let back = repack_lowbit("w", &pq, UnifiedDType::PQ2_0, QuantType::PTQ1_0, 128).unwrap();
        assert_eq!(back, ptq);
    }

    #[test]
    fn an_unrelated_pair_is_refused_by_name() {
        let err = repack_lowbit("w", &[0; 18], UnifiedDType::Q4_0, QuantType::PQ2_0, 32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tensor 'w'"), "{err}");
        assert!(err.contains("Q4_0"), "{err}");
    }

    #[test]
    fn code_3_names_the_tensor_and_the_block() {
        let mut pq = vec![0u8; 34 * 2];
        pq[34 + 2] = 0x03;
        let err = repack_lowbit("blk.0.w", &pq, UnifiedDType::PQ2_0, QuantType::PTQ1_0, 256)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tensor 'blk.0.w'"), "{err}");
        assert!(err.contains("block 1"), "{err}");
    }
}
