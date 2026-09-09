//! The TCF records compressr fills in, and the fields it deliberately leaves
//! unset.
//!
//! Every writer-owned field stays zero: `tcf-core` computes `data_offset`,
//! `logical_payload_bytes`, `physical_span_bytes`, `resident_bytes`,
//! `transfer_bytes`, both digests and the proof fields, and rejects a
//! caller-supplied value by name rather than overwriting it.
//!
//! `calibration_id` and `workload_profile_id` stay zero. compressr runs no
//! calibration and measures no workload, and a fabricated id names a record no
//! consumer can resolve.
//!
//! `activation_contract_id` is different: Section 3 requires every tensor to
//! name a `ContractRecord` in the same file, and Section 9 defines no "no
//! contract" value. So this module declares the contract an uncalibrated
//! conversion actually produces — exact F32 activations, F32 accumulation,
//! F32 output — and points every record at it. That is a truthful statement,
//! not a placeholder: these weights were prepared for an f32-activation
//! kernel, and a consumer that dispatches them through an 8-bit-activation
//! kernel is the mismatch the contract rule exists to refuse, because
//! compressr never calibrated them for it.

use tcf_core::{
    consts::ROOT_PARENT_ID, ContractFlags, ContractRecord, DotAccumulator, Encoding, ExecutionRole,
    InputRepresentation, LayoutId, MathMode, ModuleRecord, ModuleRole, OutputDtype, PolicyFlags,
    ProofFormat, QuantAxis, RawEncoding, ResidencyClass, Role, RoundingMode, ScaleComputeDtype,
    StateDtype, StateFlags, StringRef, TensorFlags, TensorRecord,
};

use super::encoding::TcfOptions;

/// The id every `ModuleRecord` and `TensorRecord` compressr writes points at.
///
/// Section 9 gives `contract_id` no reserved "unset" value, so `0` would be an
/// ordinary id that happens to read like an omission. `1` says the reference
/// is deliberate.
pub const EXACT_F32_CONTRACT_ID: u32 = 1;

/// The single activation contract every compressr-written TCF file declares.
///
/// compressr quantizes weights with no calibration and no activation-side
/// measurement, and tunes nothing for a quantized-activation kernel. The
/// contract that describes that output exactly is the full-precision one:
/// F32 activations in, F32 dot accumulation, F32 out. `quant_group`, `qmin`
/// and `qmax` describe an activation quantization grid that `F32` does not
/// have, so they stay zero; Section 9.2 defines them for `A8S32_DYNAMIC`
/// alone. `quant_axis`, `rounding_mode` and `scale_compute_dtype` carry the
/// one value v1 defines for each. `kernel_semantics_id` and `calibration_id`
/// stay zero: compressr names no kernel and runs no calibration.
///
/// `math_mode` is `REASSOCIATION_ALLOWED`. compressr places no constraint on
/// the order a consumer's kernel accumulates in — it never measured an error
/// budget that a fixed reduction order would protect. `REASSOCIATION_FORBIDDEN`
/// would read as safer while asserting a requirement compressr never
/// established, and would refuse backends for a reason this producer cannot
/// justify.
///
/// `contract_digest` stays zero for `tcf-core` to compute, as every other
/// writer-owned field does. Every value here is a constant, so two runs
/// produce byte-identical contract records.
pub fn contract_record() -> ContractRecord {
    ContractRecord {
        contract_id: EXACT_F32_CONTRACT_ID,
        input_representation: InputRepresentation::F32,
        quant_group: 0,
        quant_axis: QuantAxis::Last,
        rounding_mode: RoundingMode::RnEven,
        qmin: 0,
        qmax: 0,
        scale_compute_dtype: ScaleComputeDtype::F32,
        dot_accumulator: DotAccumulator::F32,
        output_dtype: OutputDtype::F32,
        math_mode: MathMode::ReassociationAllowed,
        kernel_semantics_id: 0,
        calibration_id: 0,
        flags: ContractFlags::NONE,
        contract_digest: [0u8; 16],
    }
}

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
        activation_contract_id: EXACT_F32_CONTRACT_ID,
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
        activation_contract_id: EXACT_F32_CONTRACT_ID,
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
        TcfOptions::uniform(NativeEncoding::Q4S32T64, NativeEncoding::Q6S32T64)
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

    /// The declared contract is the exact f32-activation one, with no
    /// activation quantization grid and no kernel or calibration named.
    #[test]
    fn the_declared_contract_is_exact_f32() {
        let contract = contract_record();
        assert_eq!(contract.contract_id, EXACT_F32_CONTRACT_ID);
        assert_eq!(contract.input_representation, InputRepresentation::F32);
        assert_eq!(contract.dot_accumulator, DotAccumulator::F32);
        assert_eq!(contract.output_dtype, OutputDtype::F32);
        assert_eq!(contract.math_mode, MathMode::ReassociationAllowed);
        assert_eq!(contract.quant_group, 0);
        assert_eq!(contract.qmin, 0);
        assert_eq!(contract.qmax, 0);
        assert_eq!(contract.kernel_semantics_id, 0);
        assert_eq!(contract.calibration_id, 0);
        assert_eq!(contract.contract_digest, [0u8; 16]);
    }

    /// Two calls produce the same bytes: nothing in the record is read from
    /// the clock, a hash map, or the source checkpoint.
    #[test]
    fn the_contract_is_the_same_on_every_call() {
        assert_eq!(contract_record(), contract_record());
    }

    /// The module and every tensor name the contract this file carries.
    #[test]
    fn every_record_points_at_the_declared_contract() {
        let module = module_record(StringRef::new(0, 5), options());
        assert_eq!(module.activation_contract_id, contract_record().contract_id);
    }

    /// Every id compressr cannot honestly fill stays zero, every field
    /// tcf-core owns stays zero for it to compute, and the contract id
    /// resolves to the record this module writes.
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
        assert_eq!(record.activation_contract_id, EXACT_F32_CONTRACT_ID);
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
