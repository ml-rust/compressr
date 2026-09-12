//! The menu of encodings a sensitivity file measures for one tensor.
//!
//! A two-way allocation needed one number per tensor: the damage the base
//! encoding did. A ladder allocation needs a menu — one measured record per
//! (tensor, encoding) pair, each carrying that pair's delta and the payload
//! bytes it costs. This module is where those records are held, keyed by name.
//!
//! # Why a pair, and not a name, is the unit
//!
//! A delta says what ONE encoding did to ONE tensor. Two records for the same
//! tensor at two encodings are two measurements, not a contradiction, so the
//! duplicate check is on the pair. Two records for the same pair are a
//! contradiction, and nothing names which of them is the tensor's delta at
//! that encoding, so that is still refused.
//!
//! # Why a skipped pair is kept
//!
//! A `skipped` record says the harness could not measure that pair, and the
//! usual reason is that the tensor cannot physically take the encoding: a row
//! that is not a whole number of blocks holds no block at all. An allocator
//! that treated the pair as merely absent could still reach it from its own
//! shape arithmetic, so the refusal is stored rather than dropped.

use std::collections::BTreeMap;

use anyhow::{bail, Result};

/// One encoding name with its case, underscores and hyphens removed.
///
/// Two spellings of one format are one format: a file spelling `Q4_K` and
/// one spelling `q4k` name the same rung rather than two.
pub fn normalized_encoding_name(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// One measured (tensor, encoding) pair.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredOption {
    /// The encoding, as the file spells it
    pub encoding: String,
    /// The measured task-metric delta at that encoding
    ///
    /// `0.0` for a skipped pair, which carries no measurement at all.
    pub delta: f64,
    /// Payload bytes the tensor takes at that encoding, when the file says
    ///
    /// A harness that quantized the tensor knows this exactly. A file that
    /// omits it leaves the byte count to the format's own arithmetic.
    pub bytes: Option<u64>,
    /// True when the harness reported the pair as unmeasurable
    pub skipped: bool,
}

impl MeasuredOption {
    /// True when this option carries a measurement an allocator can rank.
    pub fn is_measured(&self) -> bool {
        !self.skipped
    }
}

/// Every measured option, keyed by tensor name.
///
/// The map is ordered so that two runs over the same file walk the names in
/// the same order, and an allocation ranked from it is reproducible.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeasuredOptions {
    by_name: BTreeMap<String, Vec<MeasuredOption>>,
}

impl MeasuredOptions {
    /// Add one record.
    ///
    /// # Errors
    /// The file already carries a record for this (name, encoding) pair.
    pub fn insert(&mut self, name: &str, option: MeasuredOption) -> Result<()> {
        let key = normalized_encoding_name(&option.encoding);
        let options = self.by_name.entry(name.to_string()).or_default();
        if options
            .iter()
            .any(|held| normalized_encoding_name(&held.encoding) == key)
        {
            bail!(
                "Sensitivity file measures tensor '{name}' at encoding '{}' twice, so nothing \
                 names which delta is this tensor's at that encoding.",
                option.encoding
            );
        }
        options.push(option);
        Ok(())
    }

    /// Tensors the file carries at least one measurement for.
    pub fn measured_tensors(&self) -> usize {
        self.by_name
            .values()
            .filter(|options| options.iter().any(MeasuredOption::is_measured))
            .count()
    }

    /// Every tensor name the file measures, ascending.
    ///
    /// A name carrying nothing but skipped pairs is not measured and is left
    /// out, exactly as a tensor the file never names is.
    pub fn measured_names(&self) -> Vec<&str> {
        self.by_name
            .iter()
            .filter(|(_, options)| options.iter().any(MeasuredOption::is_measured))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Every option the file carries for one tensor, skipped pairs included.
    pub fn options_for(&self, name: &str) -> &[MeasuredOption] {
        self.by_name
            .get(name)
            .map_or(&[][..], |options| options.as_slice())
    }

    /// Distinct encodings the file measures, ascending by the spelling it
    /// uses.
    ///
    /// One encoding is a two-way file: every tensor was measured at one
    /// encoding and at no other, which is what a file written before the
    /// ladder carries. More than one is a menu.
    pub fn encodings(&self) -> Vec<&str> {
        let mut seen: BTreeMap<String, &str> = BTreeMap::new();
        for option in self.by_name.values().flatten() {
            if option.is_measured() {
                seen.entry(normalized_encoding_name(&option.encoding))
                    .or_insert(option.encoding.as_str());
            }
        }
        seen.into_values().collect()
    }

    /// One tensor's measured delta at one encoding, `None` when the file has
    /// no measured record for that pair.
    pub fn delta_at(&self, name: &str, encoding: &str) -> Option<f64> {
        let key = normalized_encoding_name(encoding);
        self.options_for(name)
            .iter()
            .find(|option| {
                option.is_measured() && normalized_encoding_name(&option.encoding) == key
            })
            .map(|option| option.delta)
    }

    /// The delta a two-way allocation reads for one tensor.
    ///
    /// A tensor measured at exactly one encoding has exactly one delta, and
    /// that is the number every file written before the ladder carries. A
    /// tensor with a menu is read at `baseline`, which is the encoding the
    /// file says the two-way allocation ranks from.
    pub fn baseline_delta(&self, name: &str, baseline: &str) -> Option<f64> {
        let measured: Vec<&MeasuredOption> = self
            .options_for(name)
            .iter()
            .filter(|option| option.is_measured())
            .collect();
        match measured.as_slice() {
            [only] => Some(only.delta),
            _ => self.delta_at(name, baseline),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option(encoding: &str, delta: f64) -> MeasuredOption {
        MeasuredOption {
            encoding: encoding.to_string(),
            delta,
            bytes: None,
            skipped: false,
        }
    }

    fn skipped(encoding: &str) -> MeasuredOption {
        MeasuredOption {
            encoding: encoding.to_string(),
            delta: 0.0,
            bytes: None,
            skipped: true,
        }
    }

    /// Two encodings for one tensor are two measurements, not a duplicate.
    #[test]
    fn one_tensor_holds_a_menu() {
        let mut options = MeasuredOptions::default();
        options
            .insert("a.weight", option("Q4_K", 0.5))
            .expect("first");
        options
            .insert("a.weight", option("Q6_K", 0.1))
            .expect("second");
        assert_eq!(options.options_for("a.weight").len(), 2);
        assert_eq!(options.measured_tensors(), 1);
        assert_eq!(options.encodings(), vec!["Q4_K", "Q6_K"]);
        assert_eq!(options.delta_at("a.weight", "q4k"), Some(0.5));
    }

    /// The same pair twice is what nothing can name a delta for.
    #[test]
    fn one_pair_twice_is_refused() {
        let mut options = MeasuredOptions::default();
        options
            .insert("a.weight", option("Q4_K", 0.5))
            .expect("first");
        let error = options
            .insert("a.weight", option("q4_k", 0.7))
            .expect_err("the same pair")
            .to_string();
        assert!(error.contains("'a.weight'"), "{error}");
        assert!(error.contains("twice"), "{error}");
    }

    /// A tensor the harness could not measure is no measured tensor, however
    /// many skipped pairs it carries.
    #[test]
    fn a_skipped_pair_is_no_measurement() {
        let mut options = MeasuredOptions::default();
        options
            .insert("c.weight", skipped("Q6_K"))
            .expect("skipped");
        assert_eq!(options.measured_tensors(), 0);
        assert!(options.measured_names().is_empty());
        assert_eq!(options.delta_at("c.weight", "Q6_K"), None);
        assert_eq!(options.options_for("c.weight").len(), 1);
    }

    /// A file measured at one encoding answers the two-way read whatever the
    /// baseline is spelled as.
    #[test]
    fn a_single_option_answers_the_two_way_read() {
        let mut options = MeasuredOptions::default();
        options
            .insert("a.weight", option("Q4_K", 0.5))
            .expect("only");
        assert_eq!(options.baseline_delta("a.weight", "Q4_K"), Some(0.5));
        assert_eq!(options.baseline_delta("a.weight", "Q8_0"), Some(0.5));
        assert_eq!(options.baseline_delta("b.weight", "Q4_K"), None);
    }

    /// With a menu the two-way read is the baseline encoding's delta.
    #[test]
    fn a_menu_answers_the_two_way_read_at_the_baseline() {
        let mut options = MeasuredOptions::default();
        options
            .insert("a.weight", option("Q4_K", 0.5))
            .expect("first");
        options
            .insert("a.weight", option("Q6_K", 0.1))
            .expect("second");
        assert_eq!(options.baseline_delta("a.weight", "Q4_K"), Some(0.5));
        assert_eq!(options.baseline_delta("a.weight", "Q8_0"), None);
    }
}
