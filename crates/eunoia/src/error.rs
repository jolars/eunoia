//! Error types for diagram operations.

use std::fmt::{self, Display};

/// Errors that can occur when building or working with diagrams.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagramError {
    /// A combination references a set that was never defined.
    UndefinedSet(String),

    /// A value provided is negative or invalid.
    InvalidValue { combination: String, value: f64 },

    /// No sets were defined.
    EmptySets,

    /// Duplicate combination definition.
    DuplicateCombination(String),

    /// Invalid combination format.
    InvalidCombination(String),
}

impl Display for DiagramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagramError::UndefinedSet(set) => {
                write!(f, "Set '{}' is referenced but never defined", set)
            }
            DiagramError::InvalidValue { combination, value } => {
                write!(
                    f,
                    "Invalid value {} for combination '{}'",
                    value, combination
                )
            }
            DiagramError::EmptySets => {
                write!(f, "No sets defined in diagram")
            }
            DiagramError::DuplicateCombination(combo) => {
                write!(f, "Combination '{}' defined multiple times", combo)
            }
            DiagramError::InvalidCombination(combo) => {
                write!(f, "Invalid combination format: '{}'", combo)
            }
        }
    }
}

impl std::error::Error for DiagramError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undefined_set_error() {
        let error = DiagramError::UndefinedSet("X".to_string());
        assert_eq!(error, DiagramError::UndefinedSet("X".to_string()));
        assert_eq!(
            format!("{}", error),
            "Set 'X' is referenced but never defined"
        );
    }

    #[test]
    fn test_invalid_value_error() {
        let error = DiagramError::InvalidValue {
            combination: "A&B".to_string(),
            value: -5.0,
        };
        assert_eq!(
            error,
            DiagramError::InvalidValue {
                combination: "A&B".to_string(),
                value: -5.0,
            }
        );
        assert_eq!(
            format!("{}", error),
            "Invalid value -5 for combination 'A&B'"
        );
    }

    #[test]
    fn test_empty_sets_error() {
        let error = DiagramError::EmptySets;
        assert_eq!(error, DiagramError::EmptySets);
        assert_eq!(format!("{}", error), "No sets defined in diagram");
    }

    #[test]
    fn test_duplicate_combination_error() {
        let error = DiagramError::DuplicateCombination("A&B".to_string());
        assert_eq!(error, DiagramError::DuplicateCombination("A&B".to_string()));
        assert_eq!(
            format!("{}", error),
            "Combination 'A&B' defined multiple times"
        );
    }

    #[test]
    fn test_invalid_combination_error() {
        let error = DiagramError::InvalidCombination("A&".to_string());
        assert_eq!(error, DiagramError::InvalidCombination("A&".to_string()));
        assert_eq!(format!("{}", error), "Invalid combination format: 'A&'");
    }

    #[test]
    fn test_error_trait_implementation() {
        let error: Box<dyn std::error::Error> = Box::new(DiagramError::EmptySets);
        assert_eq!(format!("{}", error), "No sets defined in diagram");
    }

    #[test]
    fn test_debug_implementation() {
        let error = DiagramError::UndefinedSet("A".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("UndefinedSet"));
        assert!(debug_str.contains("A"));
    }

    #[test]
    fn test_clone_implementation() {
        let error = DiagramError::InvalidValue {
            combination: "test".to_string(),
            value: 1.5,
        };
        let cloned = error.clone();
        assert_eq!(error, cloned);
    }
}
