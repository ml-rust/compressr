//! The one table every TCF encoding fact is read out of.
//!
//! compressr fixes three things per encoding: the name the CLI accepts, the
//! name a report prints, and the error objective the encoder minimizes when a
//! run names none. They are one ROW here rather than three lists, because
//! three lists drift. A rung added to the name list and forgotten in the
//! objective list would not fail to compile; it would quietly quantize under
//! an objective nobody chose for it, which is the whole defect the per-encoding
//! default exists to fix.
//!
//! [`super::encoding`] derives its name constant, its parser and its display
//! name from this table, so those three cannot disagree either.

use tcf_core::NativeEncoding;

use super::objective::TcfObjective;

/// How many encodings compressr can write.
///
/// The table below must hold exactly this many rows, so a rung added without
/// a default objective is a compile error rather than a silent uniform.
pub const TCF_ENCODING_COUNT: usize = 7;

/// Everything compressr fixes about one encoding.
#[derive(Debug, Clone, Copy)]
pub(super) struct TcfEncodingDef {
    /// The `tcf-core` value this row describes
    pub(super) encoding: NativeEncoding,
    /// The `--tcf-encoding` spelling, already normalized
    pub(super) cli_name: &'static str,
    /// The spelling `PROFILES.md` and a sensitivity file use
    pub(super) display_name: &'static str,
    /// The objective this encoding quantizes under when a run names none
    pub(super) default_objective: TcfObjective,
}

/// Every encoding compressr can write, in the order the CLI help lists them.
///
/// # Why the 4-bit and 8-bit rows are uniform
///
/// `TcfObjective::Uniform` is `w = 1`, the plain squared error TCF has always
/// minimized. A measurement over two disjoint evaluation splits of one text
/// causal LM found the weighted objectives no better than uniform at 8 bits,
/// where the tier is already lossless, and WORSE than uniform at 4 bits on
/// both splits. So these rows keep the unweighted objective, and their bytes
/// are the bytes every earlier run wrote.
///
/// # Why the 6-bit rows are `RmsPlusAbs`
///
/// The same measurement separated weighted from unweighted at 6 bits: both
/// weighted objectives beat uniform on both splits. It did NOT separate the
/// two weighted forms from each other — one won each split — and it is one
/// model, so this row is the weaker of the two claims and moves if a broader
/// measurement separates them.
///
/// `RmsPlusAbs` is the one taken, for the mechanism it applies: `w = sigma +
/// |x|` weights a group's large elements more than its small ones while
/// keeping the group's own root mean square as a FLOOR under every weight, so
/// no element's error is discounted to nothing. `w = x^2` has no such floor,
/// and a 6-bit code range is wide enough that the elements it discounts still
/// carry reconstruction error worth spending scale on.
pub(super) const TCF_ENCODINGS: [TcfEncodingDef; TCF_ENCODING_COUNT] = [
    TcfEncodingDef {
        encoding: NativeEncoding::Q4S32T64,
        cli_name: "q4s32t64",
        display_name: "Q4S32_T64",
        default_objective: TcfObjective::Uniform,
    },
    TcfEncodingDef {
        encoding: NativeEncoding::Q4AS32T64,
        cli_name: "q4as32t64",
        display_name: "Q4AS32_T64",
        default_objective: TcfObjective::Uniform,
    },
    TcfEncodingDef {
        encoding: NativeEncoding::Q4AS64T64,
        cli_name: "q4as64t64",
        display_name: "Q4AS64_T64",
        default_objective: TcfObjective::Uniform,
    },
    TcfEncodingDef {
        encoding: NativeEncoding::Q6S32T64,
        cli_name: "q6s32t64",
        display_name: "Q6S32_T64",
        default_objective: TcfObjective::RmsPlusAbs,
    },
    TcfEncodingDef {
        encoding: NativeEncoding::Q6S16DT64,
        cli_name: "q6s16dt64",
        display_name: "Q6S16D_T64",
        default_objective: TcfObjective::RmsPlusAbs,
    },
    TcfEncodingDef {
        encoding: NativeEncoding::Q4AS32DT64,
        cli_name: "q4as32dt64",
        display_name: "Q4AS32D_T64",
        default_objective: TcfObjective::Uniform,
    },
    TcfEncodingDef {
        encoding: NativeEncoding::Q8S32T64,
        cli_name: "q8s32t64",
        display_name: "Q8S32_T64",
        default_objective: TcfObjective::Uniform,
    },
];

/// The CLI names, in table order.
///
/// Built from the table rather than written beside it, so a rung cannot be
/// listed in the help and missing from the defaults.
const fn cli_names() -> [&'static str; TCF_ENCODING_COUNT] {
    let mut names = [""; TCF_ENCODING_COUNT];
    let mut index = 0;
    while index < TCF_ENCODING_COUNT {
        names[index] = TCF_ENCODINGS[index].cli_name;
        index += 1;
    }
    names
}

/// The `--tcf-encoding` values, in the order the CLI help lists them.
pub const TCF_ENCODING_NAMES: [&str; TCF_ENCODING_COUNT] = cli_names();

/// The row for one encoding, or `None` for a rung compressr cannot write.
///
/// Returned by value: the table is a `const`, so a reference into it would
/// borrow the temporary each use materializes.
pub(super) fn definition(encoding: NativeEncoding) -> Option<TcfEncodingDef> {
    TCF_ENCODINGS
        .into_iter()
        .find(|def| def.encoding == encoding)
}

/// The row whose CLI name matches an already-normalized spelling.
pub(super) fn definition_by_name(normalized: &str) -> Option<TcfEncodingDef> {
    TCF_ENCODINGS
        .into_iter()
        .find(|def| def.cli_name == normalized)
}

/// The objective this encoding quantizes under when a run names none.
///
/// `NativeEncoding` is `#[non_exhaustive]`: a rung `tcf-core` adds and this
/// table has no row for defaults to [`TcfObjective::Uniform`], the unweighted
/// objective, rather than to a weighting nothing measured for it.
pub fn default_objective_for(encoding: NativeEncoding) -> TcfObjective {
    match definition(encoding) {
        Some(def) => def.default_objective,
        None => TcfObjective::Uniform,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The documented default of every encoding, row for row. A row that
    /// moves without its measurement moving fails here.
    #[test]
    fn every_encoding_maps_to_its_documented_default() {
        let expected = [
            (NativeEncoding::Q4S32T64, TcfObjective::Uniform),
            (NativeEncoding::Q4AS32T64, TcfObjective::Uniform),
            (NativeEncoding::Q4AS64T64, TcfObjective::Uniform),
            (NativeEncoding::Q4AS32DT64, TcfObjective::Uniform),
            (NativeEncoding::Q6S32T64, TcfObjective::RmsPlusAbs),
            (NativeEncoding::Q6S16DT64, TcfObjective::RmsPlusAbs),
            (NativeEncoding::Q8S32T64, TcfObjective::Uniform),
        ];
        for (encoding, objective) in expected {
            assert_eq!(default_objective_for(encoding), objective, "{:?}", encoding);
        }
    }

    /// 4-bit and 8-bit output must stay byte-identical to the runs written
    /// before a per-encoding default existed, which starts with their
    /// objective staying the unweighted one.
    #[test]
    fn the_four_bit_and_eight_bit_defaults_are_still_uniform() {
        for encoding in [
            NativeEncoding::Q4S32T64,
            NativeEncoding::Q4AS32T64,
            NativeEncoding::Q4AS64T64,
            NativeEncoding::Q4AS32DT64,
            NativeEncoding::Q8S32T64,
        ] {
            assert!(
                default_objective_for(encoding).is_uniform(),
                "{:?}",
                encoding
            );
        }
    }

    /// The 6-bit rungs are the ones the measurement moved, and both moved.
    #[test]
    fn both_six_bit_encodings_take_the_weighted_default() {
        for encoding in [NativeEncoding::Q6S32T64, NativeEncoding::Q6S16DT64] {
            assert_eq!(
                default_objective_for(encoding),
                TcfObjective::RmsPlusAbs,
                "{:?}",
                encoding
            );
        }
    }

    /// One row per encoding, and the name constant is that table's own
    /// column: a rung cannot be in the help and absent from the defaults.
    #[test]
    fn the_table_is_the_only_list_of_encodings() {
        assert_eq!(TCF_ENCODING_NAMES.len(), TCF_ENCODINGS.len());
        for (index, def) in TCF_ENCODINGS.iter().enumerate() {
            assert_eq!(TCF_ENCODING_NAMES[index], def.cli_name);
            assert!(definition(def.encoding).is_some(), "{}", def.cli_name);
            assert!(
                definition_by_name(def.cli_name).is_some(),
                "{}",
                def.cli_name
            );
        }
        for def in &TCF_ENCODINGS {
            let same_encoding = TCF_ENCODINGS
                .iter()
                .filter(|other| other.encoding == def.encoding)
                .count();
            assert_eq!(same_encoding, 1, "{}", def.cli_name);
        }
    }

    /// A CLI name and a display name are two spellings of one row, so the
    /// normalized forms agree.
    #[test]
    fn a_display_name_normalizes_to_its_cli_name() {
        for def in &TCF_ENCODINGS {
            assert_eq!(
                super::super::encoding::normalized_encoding_name(def.display_name),
                def.cli_name
            );
        }
    }
}
