//! The TCF records compressr fills in, and the fields it deliberately leaves
//! unset.
//!
//! Every writer-owned field stays zero: `boostr::tcf` computes `data_offset`,
//! `logical_payload_bytes`, `physical_span_bytes`, `resident_bytes`,
//! `transfer_bytes`, both digests and the proof fields, and rejects a
//! caller-supplied value by name rather than overwriting it.
//!
//! `workload_profile_id` stays zero. compressr measures no workload, and a
//! fabricated id names a record no consumer can resolve.
//!
//! `calibration_id` stays zero on every run that measures nothing, which is
//! every run without `--sensitivity`. With that flag a measurement exists, so
//! [`calibration_record`] declares it and every measured tensor points at it.
//! Section 10.5.1 is why that pairing is not optional: `sensitivity_delta`
//! without `calibration_id` is a number with no authority, and a scheduler
//! MUST NOT rank on it.
//!
//! `activation_contract_id` names the ggml-family contract (Section 9.3): a
//! GGML block type fixes its weight bytes and not its activation path, and
//! every block kernel in boostr satisfies that family.

use boostr::tcf::{
    consts::ROOT_PARENT_ID, BlockEncoding, CalibrationFlags, CalibrationRecord, ContractFlags,
    ContractRecord, DotAccumulator, Encoding, ExecutionRole, InputRepresentation, LayoutId,
    MathMode, ModuleRecord, ModuleRole, OutputDtype, PolicyFlags, PrimaryMetric, ProofFormat,
    QuantAxis, RawEncoding, ResidencyClass, Role, RoundingMode, ScaleComputeDtype, StateDtype,
    StateFlags, StringRef, TensorFlags, TensorRecord,
};

use crate::policy::SensitivityMeasurement;

/// The id every `ModuleRecord` and `TensorRecord` compressr writes points at.
///
/// Section 9 gives `contract_id` no reserved "unset" value, so `0` would be an
/// ordinary id that happens to read like an omission. `1` says the reference
/// is deliberate.
pub const GGML_REFERENCE_CONTRACT_ID: u32 = 1;

/// The activation contract of every block tensor compressr writes.
///
/// `GGML_REFERENCE` names the kernel family `ggml-quants.c` defines for the
/// block type on the executing backend — Q8_K activations for K-quants on the
/// CPU, Q8_1 on CUDA, f32 where a backend dequantizes first. compressr
/// quantizes weights with no activation-side measurement, so it cannot pin one
/// of those without refusing the others; the family is the truthful claim.
/// `quant_group`, `qmin` and `qmax` are nominal because the family shares no
/// single grid. `quant_axis`, `rounding_mode` and `scale_compute_dtype` carry
/// the one value v1 defines for each.
///
/// `math_mode` is `REASSOCIATION_ALLOWED`. compressr never measured an error
/// budget that a fixed reduction order would protect, and every block kernel
/// reassociates.
///
/// `contract_digest` stays zero for `boostr::tcf` to compute, as every other
/// writer-owned field does. Every value here is a constant, so two runs
/// produce byte-identical contract records.
pub fn ggml_reference_contract_record() -> ContractRecord {
    ContractRecord {
        contract_id: GGML_REFERENCE_CONTRACT_ID,
        input_representation: InputRepresentation::GgmlReference,
        quant_group: 0,
        quant_axis: QuantAxis::Last,
        rounding_mode: RoundingMode::RnEven,
        qmin: 0,
        qmax: 0,
        scale_compute_dtype: ScaleComputeDtype::F32,
        dot_accumulator: DotAccumulator::GgmlReference,
        output_dtype: OutputDtype::F32,
        math_mode: MathMode::ReassociationAllowed,
        calibration_id: 0,
        flags: ContractFlags::NONE,
        contract_digest: [0u8; 16],
    }
}

/// The single root module every tensor belongs to.
///
/// compressr reads a checkpoint's bytes, not its architecture, so the module
/// role stays `Other` rather than guessing at attention or FFN. The preferred
/// encoding is the run's base block format, the declared fallback is F32,
/// which is what a tensor the plan rejects lands in, and `min_quant_k` is the
/// base format's block length.
pub fn block_module_record(name: StringRef, base: BlockEncoding) -> ModuleRecord {
    ModuleRecord {
        module_id: 0,
        parent_id: ROOT_PARENT_ID,
        name,
        module_role: ModuleRole::Other,
        fallback_encoding: Some(Encoding::Raw(RawEncoding::F32)),
        preferred_encoding: [Some(Encoding::Block(base)), None, None, None],
        activation_contract_id: GGML_REFERENCE_CONTRACT_ID,
        policy_flags: PolicyFlags::NONE,
        min_quant_k: u32::try_from(base.block_elems()).unwrap_or(u32::MAX),
        default_residency: ResidencyClass::Warm,
        state_dtype: StateDtype::None,
        state_flags: StateFlags::NONE,
        policy_digest: [0u8; 16],
    }
}

/// One tensor's measured sensitivity, and the calibration that measured it.
///
/// Every field is read from a `--sensitivity` file. A run without that flag
/// passes `None` and the record's sensitivity fields stay zero with
/// `SENSITIVITY_VALID` clear, which Section 8.2 requires: zero MUST NOT be
/// overloaded to mean "not measured".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorSensitivity {
    /// The `CalibrationRecord` this delta was measured under
    pub calibration_id: u32,
    /// The task-metric delta quantizing this tensor cost
    pub delta: f64,
    /// Evaluation rows the delta was measured over
    pub samples: u32,
    /// Seeds those rows were drawn under
    pub seed_count: u32,
}

/// One tensor's record.
///
/// `role` stays `Other` and `execution_role` follows the rank alone: a rank-2
/// tensor is a matmul operand and a rank-1 one is elementwise. Naming a tensor
/// an embedding or a router from its name would be a guess, and
/// `execution_role` is the dispatch key.
///
/// `sensitivity_ci95` stays zero with or without a measurement. The
/// sensitivity file carries one delta per tensor and no confidence interval,
/// and Section 8.2 reads the field only against `SENSITIVITY_VALID`, so a
/// fabricated interval would be read as measured.
///
/// `TASK_CRITICAL` is never set. Section 8.2 names the bit and defines no
/// threshold for it, so a producer setting it from its own promotion decision
/// would be asserting a claim the format does not define.
pub fn tensor_record(
    tensor_id: u32,
    name: StringRef,
    encoding: Encoding,
    fallback_reason: boostr::tcf::FallbackReason,
    dims: [u64; 8],
    rank: u32,
    sensitivity: Option<TensorSensitivity>,
) -> TensorRecord {
    TensorRecord {
        tensor_id,
        module_id: 0,
        name,
        role: Role::Other,
        encoding,
        fallback_reason,
        residency_class: ResidencyClass::Warm,
        flags: match sensitivity {
            Some(_) => TensorFlags::SENSITIVITY_VALID,
            None => TensorFlags::NONE,
        },
        rank,
        calibration_id: sensitivity.map_or(0, |measured| measured.calibration_id),
        dims,
        activation_contract_id: GGML_REFERENCE_CONTRACT_ID,
        layout_id: LayoutId::RowMajorDense,
        data_offset: 0,
        logical_payload_bytes: 0,
        physical_span_bytes: 0,
        resident_bytes: 0,
        transfer_bytes: 0,
        sensitivity_delta: sensitivity.map_or(0.0, |measured| measured.delta as f32),
        sensitivity_ci95: 0.0,
        accesses_per_generation: 0.0,
        bytes_read_per_generation: 0.0,
        sensitivity_samples: sensitivity.map_or(0, |measured| measured.samples),
        sensitivity_seed_count: sensitivity.map_or(0, |measured| measured.seed_count),
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

/// Seeds a sensitivity file's row sample was drawn under.
///
/// The measurement names one `eval_seed`, so one seed. A larger count would
/// claim a seed-to-seed variance estimate the file does not hold.
pub const SENSITIVITY_SEED_COUNT: u32 = 1;

/// The calibration a `--sensitivity` run's tensors point at. Section 10.
///
/// Section 10.5.1 makes the pairing load-bearing: a `sensitivity_delta` with
/// no calibration behind it is a number with no authority. So a measured run
/// declares the calibration it actually read.
///
/// `primary_metric` is `TASK_SPECIFIC`. The measurement is a task loss, not a
/// word or character error rate and not a perplexity, and Section 10 forbids
/// naming a tensor RMS here.
///
/// Three fields stay zero because compressr cannot fill them honestly, and
/// each is a real gap rather than a formality:
///
/// - `dataset_digest`, a BLAKE3-256 over the evaluation set's content.
///   compressr never sees that data; it reads a file of numbers computed from
///   it. The dataset is named, not fingerprinted.
/// - `acceptance_margin`. Nothing in the measurement declares a margin a delta
///   must stay inside, and inventing one would fix an acceptance criterion the
///   producer never set.
/// - `producer_timestamp`. The measurement carries no time, and the conversion's
///   own clock would date the file rather than the calibration. Leaving it zero
///   also keeps two identical runs byte-identical.
pub fn calibration_record(
    measurement: &SensitivityMeasurement,
    dataset_name: StringRef,
    evaluator_name: StringRef,
) -> CalibrationRecord {
    let baseline = measurement.baseline();
    CalibrationRecord {
        calibration_id: measurement.calibration_id(),
        primary_metric: PrimaryMetric::TaskSpecific,
        flags: CalibrationFlags::NONE,
        sample_count: baseline.eval_rows,
        seed_count: SENSITIVITY_SEED_COUNT,
        baseline_metric: baseline.baseline_total as f32,
        acceptance_margin: 0.0,
        dataset_name,
        evaluator_name,
        dataset_digest: [0u8; 32],
        evaluator_config_digest: measurement.calibration_digest(),
        producer_timestamp: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement() -> SensitivityMeasurement {
        crate::policy::parse_sensitivity(&[
            r#"{"record":"baseline","ckpt":"cache/models/example/abcdef","manifest":"manifests/eval.tsv","encoding":"Q4_K","eval_rows":12,"eval_seed":7,"baseline_total":1.17}"#,
            r#"{"record":"sensitivity","name":"a.weight","delta":0.5}"#,
        ]
        .join("\n"))
        .expect("parses")
    }

    #[test]
    fn the_module_ranks_the_base_block_format_first() {
        let module = block_module_record(StringRef::new(0, 5), BlockEncoding::Q4K);
        assert_eq!(
            module.top_preferred_encoding(),
            Some(Encoding::Block(BlockEncoding::Q4K))
        );
        assert!(module.is_root());
        assert_eq!(module.min_quant_k, 256);
        assert_eq!(module.activation_contract_id, GGML_REFERENCE_CONTRACT_ID);
    }

    /// The declared contract names the ggml family, with no activation grid
    /// and no calibration named.
    #[test]
    fn the_declared_contract_is_the_ggml_family() {
        let contract = ggml_reference_contract_record();
        assert_eq!(contract.contract_id, GGML_REFERENCE_CONTRACT_ID);
        assert_eq!(
            contract.input_representation,
            InputRepresentation::GgmlReference
        );
        assert_eq!(contract.dot_accumulator, DotAccumulator::GgmlReference);
        assert_eq!(contract.output_dtype, OutputDtype::F32);
        assert_eq!(contract.math_mode, MathMode::ReassociationAllowed);
        assert_eq!(contract.quant_group, 0);
        assert_eq!(contract.qmin, 0);
        assert_eq!(contract.qmax, 0);
        assert_eq!(contract.calibration_id, 0);
        assert_eq!(contract.contract_digest, [0u8; 16]);
    }

    /// Two calls produce the same bytes: nothing in the record is read from
    /// the clock, a hash map, or the source checkpoint.
    #[test]
    fn the_contract_is_the_same_on_every_call() {
        assert_eq!(
            ggml_reference_contract_record(),
            ggml_reference_contract_record()
        );
    }

    /// Every id compressr cannot honestly fill stays zero, every field the
    /// container owns stays zero for it to compute, and the contract id
    /// resolves to the record this module writes.
    #[test]
    fn ids_and_writer_owned_fields_stay_unset() {
        let record = tensor_record(
            3,
            StringRef::new(0, 1),
            Encoding::Block(BlockEncoding::Q4K),
            boostr::tcf::FallbackReason::None,
            [1, 256, 0, 0, 0, 0, 0, 0],
            2,
            None,
        );
        assert_eq!(record.calibration_id, 0);
        assert_eq!(record.activation_contract_id, GGML_REFERENCE_CONTRACT_ID);
        assert_eq!(record.workload_profile_id, 0);
        assert_eq!(record.data_offset, 0);
        assert_eq!(record.logical_payload_bytes, 0);
        assert_eq!(record.physical_span_bytes, 0);
        assert_eq!(record.semantic_digest, [0u8; 16]);
        assert_eq!(record.payload_digest, [0u8; 16]);
        assert_eq!(record.execution_role, ExecutionRole::Matmul);
        assert!(!record.flags.contains(TensorFlags::SENSITIVITY_VALID));
    }

    /// A measured tensor carries the delta, the sample count and the
    /// calibration behind them, and says so with `SENSITIVITY_VALID`. The
    /// confidence interval stays zero: the measurement has none.
    #[test]
    fn a_measured_tensor_records_its_delta_and_its_calibration() {
        let measurement = measurement();
        let record = tensor_record(
            0,
            StringRef::new(0, 1),
            Encoding::Block(BlockEncoding::Q6K),
            boostr::tcf::FallbackReason::TaskSensitivity,
            [1, 256, 0, 0, 0, 0, 0, 0],
            2,
            Some(TensorSensitivity {
                calibration_id: measurement.calibration_id(),
                delta: 0.5,
                samples: 12,
                seed_count: SENSITIVITY_SEED_COUNT,
            }),
        );
        assert!(record.flags.contains(TensorFlags::SENSITIVITY_VALID));
        assert_eq!(record.sensitivity_delta, 0.5);
        assert_eq!(record.sensitivity_samples, 12);
        assert_eq!(record.sensitivity_seed_count, 1);
        assert_eq!(record.sensitivity_ci95, 0.0);
        assert_eq!(record.calibration_id, measurement.calibration_id());
        assert_ne!(record.calibration_id, 0);
    }

    /// The calibration names a task metric, never an RMS, and leaves the
    /// three fields compressr cannot fill honestly at zero.
    #[test]
    fn the_calibration_declares_the_measurement_it_read() {
        let measurement = measurement();
        let record = calibration_record(&measurement, StringRef::new(0, 4), StringRef::new(4, 6));
        assert_eq!(record.primary_metric, PrimaryMetric::TaskSpecific);
        assert_eq!(record.calibration_id, measurement.calibration_id());
        assert_eq!(record.sample_count, 12);
        assert_eq!(record.seed_count, SENSITIVITY_SEED_COUNT);
        assert_eq!(
            record.evaluator_config_digest,
            measurement.calibration_digest()
        );
        assert_eq!(record.dataset_digest, [0u8; 32]);
        assert_eq!(record.acceptance_margin, 0.0);
        assert_eq!(record.producer_timestamp, 0);
    }

    #[test]
    fn a_rank_1_tensor_dispatches_elementwise() {
        let record = tensor_record(
            0,
            StringRef::new(0, 1),
            Encoding::Raw(RawEncoding::F32),
            boostr::tcf::FallbackReason::RankLt2,
            [64, 0, 0, 0, 0, 0, 0, 0],
            1,
            None,
        );
        assert_eq!(record.execution_role, ExecutionRole::Elementwise);
    }
}
