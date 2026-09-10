//! A few names out of a list, for an error that has to be actionable.
//!
//! Every strict-match refusal in this crate names what did not match, and a
//! refusal that printed thousands of tensor names would be unreadable while
//! one that printed none would be unusable. The rule is the same everywhere,
//! so it is written once.

/// How many names one list shows before it is summarized.
const EXAMPLE_NAMES: usize = 5;

/// Up to [`EXAMPLE_NAMES`] names, quoted, with a count of the rest.
pub fn examples(names: &[&str]) -> String {
    let shown: Vec<String> = names
        .iter()
        .take(EXAMPLE_NAMES)
        .map(|name| format!("'{name}'"))
        .collect();
    let rest = names.len().saturating_sub(shown.len());
    if rest == 0 {
        shown.join(", ")
    } else {
        format!("{} and {rest} more", shown.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_list_is_shown_whole() {
        assert_eq!(examples(&["a", "b"]), "'a', 'b'");
        assert_eq!(examples(&[]), "");
    }

    #[test]
    fn a_long_list_is_summarized() {
        let names = ["a", "b", "c", "d", "e", "f", "g"];
        assert_eq!(
            examples(&names),
            "'a', 'b', 'c', 'd', 'e' and 2 more".to_string()
        );
    }
}
