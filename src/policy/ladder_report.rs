//! What a ladder allocation did, in the numbers that decide whether it was
//! worth doing.
//!
//! The demotion count leads, because it is the whole point of the change. A
//! two-way allocation cannot demote at all: its base encoding is a floor, and
//! a tensor measured as undamaged at that floor still pays for it. A ladder
//! measured at several encodings can move that tensor down and spend what it
//! frees somewhere the measurement says it matters. If the demotion count is
//! zero, the ladder bought nothing over the allocation it replaced, and this
//! report is where that shows.

use std::collections::BTreeMap;

use super::ladder::LadderOutcome;

/// One ladder allocation, reported.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderReport {
    /// The allocation this describes, e.g. a profile name or a mix name
    pub scope: String,
    /// The encoding a two-way allocation would have put every tensor at
    pub base_encoding: String,
    /// Tensors the ladder allocated
    pub tensors: usize,
    /// Tensors that landed on each encoding, by encoding name
    pub per_encoding: Vec<(String, usize)>,
    /// Tensors that landed BELOW the base encoding
    pub demoted: usize,
    /// Tensors that landed above it
    pub promoted: usize,
    /// Tensors that stayed on it
    pub held: usize,
    /// Measured damage of the chosen allocation
    pub chosen_damage: f64,
    /// Measured damage an all-base allocation would have done
    pub base_damage: f64,
    /// Payload bytes the chosen allocation writes
    pub chosen_bytes: u64,
    /// Payload bytes an all-base allocation would write
    pub base_bytes: u64,
    /// Payload bytes the all-cheapest allocation the greedy starts from writes
    pub start_bytes: u64,
    /// Bytes the format allows spending above an all-base allocation
    pub budget_bytes: u64,
    /// Bytes the greedy could spend: the budget, plus what demotion freed
    pub available_bytes: u64,
    /// Bytes it spent
    pub spent_bytes: u64,
    /// Upgrade steps taken
    pub upgrades: usize,
    /// Steps refused for a damage reduction that is not positive
    pub nonpositive_steps: usize,
    /// Steps the remaining budget could not pay for
    pub unaffordable_steps: usize,
    /// (tensor, encoding) pairs no allocation may use
    ///
    /// The measurement marked the pair skipped, or the tensor's row is not a
    /// whole number of that encoding's blocks. Either way the tensor cannot
    /// physically take it, so it was never an option.
    pub unusable_pairs: usize,
    /// Tensors left with nothing but their base option
    pub single_option_tensors: usize,
}

impl LadderReport {
    /// Read a report off one allocation.
    pub fn of(
        scope: String,
        base_encoding: String,
        budget_bytes: u64,
        unusable_pairs: usize,
        outcome: &LadderOutcome,
    ) -> Self {
        let mut per_encoding: BTreeMap<&str, usize> = BTreeMap::new();
        let mut demoted = 0usize;
        let mut promoted = 0usize;
        for choice in &outcome.chosen {
            *per_encoding.entry(choice.encoding.as_str()).or_insert(0) += 1;
            if choice.is_demoted() {
                demoted += 1;
            } else if choice.is_promoted() {
                promoted += 1;
            }
        }
        let tensors = outcome.chosen.len();
        Self {
            scope,
            base_encoding,
            tensors,
            per_encoding: per_encoding
                .into_iter()
                .map(|(encoding, count)| (encoding.to_string(), count))
                .collect(),
            demoted,
            promoted,
            held: tensors.saturating_sub(demoted + promoted),
            chosen_damage: outcome.chosen_damage,
            base_damage: outcome.base_damage,
            chosen_bytes: outcome.chosen_bytes,
            base_bytes: outcome.base_bytes,
            start_bytes: outcome.start_bytes,
            budget_bytes,
            available_bytes: outcome.available_bytes,
            spent_bytes: outcome.spent_bytes,
            upgrades: outcome.upgrades,
            nonpositive_steps: outcome.nonpositive_steps,
            unaffordable_steps: outcome.unaffordable_steps,
            unusable_pairs,
            single_option_tensors: 0,
        }
    }

    /// Tensors left with nothing but their base option, set by the caller.
    pub fn with_single_option_tensors(mut self, count: usize) -> Self {
        self.single_option_tensors = count;
        self
    }

    /// Budget bytes the allocation could not place.
    pub fn unspent_bytes(&self) -> u64 {
        self.available_bytes.saturating_sub(self.spent_bytes)
    }

    /// Measured damage the ladder removed against an all-base allocation.
    ///
    /// Positive means less damage than the allocation it replaced, which is
    /// the direction the whole comparison is about.
    pub fn damage_removed(&self) -> f64 {
        self.base_damage - self.chosen_damage
    }
}

impl std::fmt::Display for LadderReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Measured ladder {}: {} tensor(s) allocated from measured options, against a base \
             encoding of {}",
            self.scope, self.tensors, self.base_encoding
        )?;
        writeln!(
            f,
            "  DEMOTED below the base: {} tensor(s). Promoted above it: {}. Left on it: {}. A \
             two-way allocation could demote none of them.",
            self.demoted, self.promoted, self.held
        )?;
        let chosen: Vec<String> = self
            .per_encoding
            .iter()
            .map(|(encoding, count)| format!("{encoding} {count}"))
            .collect();
        writeln!(f, "  encodings chosen: {}", chosen.join(", "))?;
        writeln!(
            f,
            "  measured damage: {:e} chosen against {:e} for an all-base allocation, {:e} removed",
            self.chosen_damage,
            self.base_damage,
            self.damage_removed()
        )?;
        writeln!(
            f,
            "  byte budget {} above an all-base allocation of {} byte(s); starting every tensor \
             at its cheapest measured option costs {}, so {} byte(s) were available and {} were \
             spent over {} upgrade step(s), leaving {} unspent",
            self.budget_bytes,
            self.base_bytes,
            self.start_bytes,
            self.available_bytes,
            self.spent_bytes,
            self.upgrades,
            self.unspent_bytes()
        )?;
        writeln!(
            f,
            "  {} step(s) refused for a damage reduction that is not positive; {} step(s) the \
             remaining budget could not pay for",
            self.nonpositive_steps, self.unaffordable_steps
        )?;
        write!(
            f,
            "  {} (tensor, encoding) pair(s) no allocation may use, and {} tensor(s) had nothing \
             but their base option; payload bytes {} chosen against {} for the all-base \
             allocation",
            self.unusable_pairs, self.single_option_tensors, self.chosen_bytes, self.base_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::ladder::{allocate_ladder, LadderMenu, LadderOption};

    fn option(encoding: &str, bytes: u64, delta: f64) -> LadderOption {
        LadderOption {
            encoding: encoding.to_string(),
            bytes,
            delta,
        }
    }

    /// The report states the demotion count first, and the damage it removed
    /// against the allocation it replaced.
    #[test]
    fn the_report_leads_with_the_demotions() {
        let menus = vec![
            LadderMenu {
                index: 0,
                name: "free.weight".into(),
                options: vec![option("cheap", 100, 0.0), option("base", 200, 0.0)],
                base: 1,
            },
            LadderMenu {
                index: 1,
                name: "damaged.weight".into(),
                options: vec![option("base", 200, 0.9), option("dear", 260, 0.1)],
                base: 0,
            },
        ];
        let outcome = allocate_ladder(&menus, 100).expect("allocates");
        let report = LadderReport::of("profile TEST".into(), "base".into(), 0, 3, &outcome)
            .with_single_option_tensors(1);

        assert_eq!(report.demoted, 1);
        assert_eq!(report.promoted, 1);
        assert_eq!(report.held, 0);
        assert!(report.damage_removed() > 0.0);
        assert_eq!(report.per_encoding.len(), 2);

        let text = report.to_string();
        assert!(text.contains("DEMOTED below the base: 1"), "{text}");
        assert!(text.contains("encodings chosen: cheap 1, dear 1"), "{text}");
        assert!(text.contains("3 (tensor, encoding) pair(s)"), "{text}");
    }
}
