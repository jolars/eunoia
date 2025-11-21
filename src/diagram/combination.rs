//! Set combination representation.

use std::fmt::{self, Display};

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
}

impl Display for Combination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sets.join("&"))
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
}
