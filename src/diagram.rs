//! Diagram specification and construction.
//!
//! This module provides types for defining Euler and Venn diagram specifications through
//! set combinations and their values.

mod builder;
mod combination;
mod input;

pub use builder::DiagramSpecBuilder;
pub use combination::Combination;
pub use input::InputType;

use std::collections::{HashMap, HashSet};

/// Represents a complete Euler or Venn diagram specification.
///
/// This contains the input data (set sizes and intersections) that describes
/// what the diagram should represent. The actual geometric shapes will be
/// computed during the fitting process.
#[derive(Debug, Clone)]
pub struct DiagramSpec {
    /// Map from combinations to their values.
    pub(crate) combinations: HashMap<Combination, f64>,

    /// How the input values should be interpreted.
    pub(crate) input_type: InputType,

    /// Set of all unique set names in the diagram.
    pub(crate) set_names: HashSet<String>,
}

impl DiagramSpec {
    /// Returns the input type for this diagram specification.
    pub fn input_type(&self) -> InputType {
        self.input_type
    }

    /// Returns the set names in this diagram specification.
    pub fn set_names(&self) -> &HashSet<String> {
        &self.set_names
    }

    /// Returns the combinations and their values.
    pub fn combinations(&self) -> &HashMap<Combination, f64> {
        &self.combinations
    }

    /// Gets the value for a specific combination.
    pub fn get_combination(&self, combination: &Combination) -> Option<f64> {
        self.combinations.get(combination).copied()
    }
}
