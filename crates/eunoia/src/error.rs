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
