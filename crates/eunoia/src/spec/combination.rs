//! Set combination representation.

use std::convert::Infallible;
use std::fmt::{self, Display};
use std::str::FromStr;

/// Represents a combination of sets (e.g., "A", "A&B", "A&B&C").
///
/// Combinations are stored as sorted vectors of set names to ensure
/// consistent representation (e.g., "A&B" and "B&A" are equivalent).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Combination {
    sets: Vec<String>,
}

impl Combination {
    /// Creates a new combination from a slice of set names.
    ///
    /// The sets are sorted to ensure canonical representation.
    pub fn new(sets: &[&str]) -> Self {
        let mut set_vec: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        set_vec.sort();
        Combination { sets: set_vec }
    }

    /// Returns the set names in this combination.
    pub fn sets(&self) -> &[String] {
        &self.sets
    }

    /// Returns the number of sets in this combination.
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Returns true if this is an empty combination.
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    pub fn contains_all(&self, other: &Combination) -> bool {
        other.sets.iter().all(|s| self.sets.contains(s))
    }
}

impl Display for Combination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sets.join("&"))
    }
}

impl FromStr for Combination {
    type Err = Infallible;

    /// Parse a combination from its [`Display`] form (`"A&B&C"`). Splits on
    /// `&`, trims whitespace on each part, and drops empty parts so that
    /// `"".parse()` and `"A&"` round-trip cleanly.
    ///
    /// Always succeeds — the error type is [`Infallible`] — so callers can
    /// use [`str::parse`] without unwrap noise.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::spec::Combination;
    ///
    /// let combo: Combination = "A&B&C".parse().unwrap();
    /// assert_eq!(combo.to_string(), "A&B&C");
    ///
    /// // Whitespace around parts is trimmed.
    /// let combo: Combination = "  A &  B ".parse().unwrap();
    /// assert_eq!(combo.to_string(), "A&B");
    ///
    /// // Empty input parses to the empty combination.
    /// let combo: Combination = "".parse().unwrap();
    /// assert!(combo.is_empty());
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s
            .split('&')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        Ok(Combination::new(&parts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_new() {
        let combo = Combination::new(&["A", "B"]);
        assert_eq!(combo.sets(), &["A", "B"]);
        assert_eq!(combo.len(), 2);
    }

    #[test]
    fn test_combination_sorted() {
        let combo1 = Combination::new(&["B", "A"]);
        let combo2 = Combination::new(&["A", "B"]);
        assert_eq!(combo1, combo2);
    }

    #[test]
    fn test_combination_to_string() {
        let combo = Combination::new(&["A", "B", "C"]);
        assert_eq!(combo.to_string(), "A&B&C");
    }

    #[test]
    fn test_combination_from_str_round_trips_with_display() {
        // `Combination::new` canonicalises by sorting, and `Display` emits
        // that canonical form, so round-trip equality holds for inputs that
        // are already sorted.
        for s in ["A", "A&B", "A&B&C", "W&X&Y&Z"] {
            let combo: Combination = s.parse().unwrap();
            assert_eq!(combo.to_string(), s);
        }
    }

    #[test]
    fn test_combination_from_str_trims_whitespace_and_empty_parts() {
        let combo: Combination = "  A & B & ".parse().unwrap();
        assert_eq!(combo.to_string(), "A&B");

        let combo: Combination = "".parse().unwrap();
        assert!(combo.is_empty());

        let combo: Combination = "&&&".parse().unwrap();
        assert!(combo.is_empty());
    }

    #[test]
    fn test_combination_from_str_canonicalises() {
        let a: Combination = "B&A".parse().unwrap();
        let b: Combination = "A&B".parse().unwrap();
        assert_eq!(a, b);
    }
}
