//! The TCF records compressr fills in, and the fields it deliberately leaves
//! unset.
//!
//! Every writer-owned field stays zero: `tcf-core` computes `data_offset`,
//! `logical_payload_bytes`, `physical_span_bytes`, `resident_bytes`,
//! `transfer_bytes`, both digests and the proof fields, and rejects a
//! caller-supplied value by name rather than overwriting it.
//!
//! `calibration_id`, `activation_contract_id` and `workload_profile_id` stay
//! zero as well. compressr runs no calibration and measures no workload, and a
//! fabricated id names a record no consumer can resolve.

use tcf_core::{
    consts::ROOT_PARENT_ID, Encoding, ExecutionRole, LayoutId, ModuleRecord, ModuleRole,
    PolicyFlags, ProofFormat, RawEncoding, ResidencyClass, Role, StateDtype, StateFlags, StringRef,
    TensorFlags, TensorRecord,
};

use super::encoding::TcfOptions;

/// The single root module every tensor belongs to.
///
/// compressr reads a checkpoint's bytes, not its architecture, so the module
/// role stays `Other` rather than guessing at attention or FFN. The declared
/// fallback is F32, which is what a tensor the quantizer rejects lands in.
pub fn module_record(name: StringRef, options: TcfOptions) -> ModuleRecord {
    ModuleRecord {
        module_id: 0,
        parent_id: ROOT_PARENT_ID,
        name,
        module_role: ModuleRole::Other,
        fallback_encoding: Some(Encoding::Raw(RawEncoding::F32)),
        preferred_encoding: [
            Some(Encoding::Native(options.preferred_encoding)),
            None,
            None,
            None,
        ],
        activation_contract_id: 0,
        policy_flags: PolicyFlags::NONE,
        min_quant_k: u32::from(options.encoding.geometry().tile),
        default_residency: ResidencyClass::Warm,
        state_dtype: StateDtype::None,
        state_flags: StateFlags::NONE,
        policy_digest: [0u8; 16],
    }
}

/// One tensor's record.
///
/// `role` stays `Other` and `execution_role` follows the rank alone: a rank-2
/// tensor is a matmul operand and a rank-1 one is elementwise. Naming a tensor
/// an embedding or a router from its name would be a guess, and
/// `execution_role` is the dispatch key.
pub fn tensor_record(
    tensor_id: u32,
    name: StringRef,
    encoding: Encoding,
    fallback_reason: tcf_core::FallbackReason,
    dims: [u64; 8],
    rank: u32,
) -> TensorRecord {
    TensorRecord {
        tensor_id,
        module_id: 0,
        name,
        role: Role::Other,
        encoding,
        fallback_reason,
        residency_class: ResidencyClass::Warm,
        flags: TensorFlags::NONE,
        rank,
        calibration_id: 0,
        dims,
        activation_contract_id: 0,
        layout_id: LayoutId::RowMajorDense,
        data_offset: 0,
        logical_payload_bytes: 0,
        physical_span_bytes: 0,
        resident_bytes: 0,
        transfer_bytes: 0,
        sensitivity_delta: 0.0,
        sensitivity_ci95: 0.0,
        accesses_per_generation: 0.0,
        bytes_read_per_generation: 0.0,
        sensitivity_samples: 0,
        sensitivity_seed_count: 0,
        access_profile_samples: 0,
        execution_role: if rank >= 2 {
            ExecutionRole::Matmul
        } else {
            ExecutionRole::Elementwise
        },
        workload_profile_id: 0,
        semantic_digest: [0u8; 16],
        payload_digest: [0u8; 16],
        proof_rel_off: 0,
        proof_count: 0,
        proof_format: ProofFormat::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::NativeEncoding;

    fn options() -> TcfOptions {
        TcfOptions {
            encoding: NativeEncoding::Q4S32T64,
            preferred_encoding: NativeEncoding::Q6S32T64,
        }
    }

    #[test]
    fn the_module_ranks_the_preferred_encoding_first() {
        let module = module_record(StringRef::new(0, 5), options());
        assert_eq!(
            module.top_preferred_encoding(),
            Some(Encoding::Native(NativeEncoding::Q6S32T64))
        );
        assert!(module.is_root());
        assert_eq!(module.min_quant_k, 64);
    }

    /// Every id compressr cannot honestly fill stays zero, and every field
    /// tcf-core owns stays zero for it to compute.
    #[test]
    fn ids_and_writer_owned_fields_stay_unset() {
        let record = tensor_record(
            3,
            StringRef::new(0, 1),
            Encoding::Native(NativeEncoding::Q4S32T64),
            tcf_core::FallbackReason::None,
            [1, 64, 0, 0, 0, 0, 0, 0],
            2,
        );
        assert_eq!(record.calibration_id, 0);
        assert_eq!(record.activation_contract_id, 0);
        assert_eq!(record.workload_profile_id, 0);
        assert_eq!(record.data_offset, 0);
        assert_eq!(record.logical_payload_bytes, 0);
        assert_eq!(record.physical_span_bytes, 0);
        assert_eq!(record.semantic_digest, [0u8; 16]);
        assert_eq!(record.payload_digest, [0u8; 16]);
        assert_eq!(record.execution_role, ExecutionRole::Matmul);
    }

    #[test]
    fn a_rank_1_tensor_dispatches_elementwise() {
        let record = tensor_record(
            0,
            StringRef::new(0, 1),
            Encoding::Raw(RawEncoding::F32),
            tcf_core::FallbackReason::RankLt2,
            [64, 0, 0, 0, 0, 0, 0, 0],
            1,
        );
        assert_eq!(record.execution_role, ExecutionRole::Elementwise);
    }
}
