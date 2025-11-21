//! # Eunoia
//!
//! A Rust library for creating area-proportional Euler and Venn diagrams.
//!
//! Eunoia generates optimal layouts for set visualizations using various geometric shapes
//! (circles, ellipses, rectangles, triangles). The library uses a two-phase optimization
//! approach:
//!
//! 1. **Initial layout**: Multi-dimensional scaling (MDS) to place fixed-size shapes
//! 2. **Refinement**: Comprehensive optimization to minimize loss functions (RegionError or stress)
//!
//! ## Example
//!
//! ```rust
//! use eunoia::{DiagramBuilder, InputType};
//!
//! let diagram = DiagramBuilder::new()
//!     .set("A", 5.0)
//!     .set("B", 2.0)
//!     .intersection(&["A", "B"], 1.0)
//!     .input_type(InputType::Disjoint)
//!     .build()
//!     .expect("Failed to build diagram");
//! ```

pub mod geometry;

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};

/// Trait for shapes that can be parameterized for optimization.
///
/// This trait enables shapes to expose their degrees of freedom (position, size, rotation)
/// as a parameter vector for use with optimization algorithms.
pub trait Parameters {
    /// Returns the number of parameters needed to fully describe this shape.
    ///
    /// For example, a circle has 3 parameters (x, y, radius), while an ellipse
    /// has 5 (x, y, semi-major axis, semi-minor axis, rotation angle).
    fn n_params(&self) -> usize;

    /// Updates the shape's parameters from a parameter vector.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice containing the new parameter values
    ///
    /// # Panics
    ///
    /// May panic if the length of `params` doesn't match `n_params()`.
    fn update(&mut self, params: &[f64]);
}

/// Specifies how input values should be interpreted.
///
/// This determines whether the provided values represent disjoint regions
/// or complete set sizes including overlaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputType {
    /// Values represent disjoint regions that sum to the total set size.
    ///
    /// For example: `A=5, B=2, A&B=1` means set A has total size 6 (5+1)
    /// and set B has total size 3 (2+1).
    Disjoint,

    /// Values represent complete set sizes including overlaps.
    ///
    /// For example: `A=6, B=3, A&B=1` means set A has total size 6
    /// and set B has total size 3.
    #[default]
    Union,
}

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

/// Represents a complete Euler or Venn diagram specification.
///
/// This contains the input data (set sizes and intersections) along with
/// metadata needed for optimization. The actual geometric shapes will be
/// computed during the layout process.
#[derive(Debug, Clone)]
pub struct Diagram {
    /// Map from combinations to their values.
    combinations: HashMap<Combination, f64>,

    /// How the input values should be interpreted.
    input_type: InputType,

    /// Set of all unique set names in the diagram.
    set_names: HashSet<String>,
}

impl Diagram {
    /// Returns the input type for this diagram.
    pub fn input_type(&self) -> InputType {
        self.input_type
    }

    /// Returns the set names in this diagram.
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

/// Builder for creating diagrams with a fluent API.
///
/// # Examples
///
/// ```
/// use eunoia::{DiagramBuilder, InputType};
///
/// let diagram = DiagramBuilder::new()
///     .set("A", 5.0)
///     .set("B", 2.0)
///     .intersection(&["A", "B"], 1.0)
///     .input_type(InputType::Disjoint)
///     .build()
///     .expect("Failed to build diagram");
/// ```
#[derive(Debug, Default)]
pub struct DiagramBuilder {
    combinations: HashMap<Combination, f64>,
    input_type: Option<InputType>,
}

impl DiagramBuilder {
    /// Creates a new diagram builder.
    pub fn new() -> Self {
        DiagramBuilder {
            combinations: HashMap::new(),
            input_type: None,
        }
    }

    /// Adds a single set with the given value.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the set
    /// * `value` - The size/value for this set
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::DiagramBuilder;
    ///
    /// let builder = DiagramBuilder::new()
    ///     .set("A", 10.0);
    /// ```
    pub fn set(mut self, name: impl Into<String>, value: f64) -> Self {
        let combination = Combination::new(&[&name.into()]);
        self.combinations.insert(combination, value);
        self
    }

    /// Adds an intersection of multiple sets with the given value.
    ///
    /// # Arguments
    ///
    /// * `sets` - A slice of set names to intersect
    /// * `value` - The size/value for this intersection
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::DiagramBuilder;
    ///
    /// let builder = DiagramBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .intersection(&["A", "B"], 2.0);
    /// ```
    pub fn intersection(mut self, sets: &[&str], value: f64) -> Self {
        let combination = Combination::new(sets);
        self.combinations.insert(combination, value);
        self
    }

    /// Sets how the input values should be interpreted.
    ///
    /// # Arguments
    ///
    /// * `input_type` - Whether values are disjoint or union-based
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramBuilder, InputType};
    ///
    /// let builder = DiagramBuilder::new()
    ///     .input_type(InputType::Disjoint);
    /// ```
    pub fn input_type(mut self, input_type: InputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Builds the diagram, validating all inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No sets are defined
    /// - Any value is negative
    /// - An intersection references undefined sets
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramBuilder, InputType};
    ///
    /// let diagram = DiagramBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 2.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .build()
    ///     .expect("Failed to build diagram");
    /// ```
    pub fn build(self) -> Result<Diagram, DiagramError> {
        // Check that we have at least one set
        if self.combinations.is_empty() {
            return Err(DiagramError::EmptySets);
        }

        // Collect all unique set names and find single-set combinations
        let mut set_names = HashSet::new();
        let mut single_sets = HashSet::new();

        for combination in self.combinations.keys() {
            for set_name in combination.sets() {
                set_names.insert(set_name.clone());
            }
            if combination.len() == 1 {
                single_sets.insert(combination.sets()[0].clone());
            }
        }

        // Validate that all values are non-negative
        for (combination, &value) in &self.combinations {
            if value < 0.0 {
                return Err(DiagramError::InvalidValue {
                    combination: combination.to_string(),
                    value,
                });
            }
        }

        // Validate that all sets in intersections are defined as single sets
        for combination in self.combinations.keys() {
            if combination.len() > 1 {
                for set_name in combination.sets() {
                    if !single_sets.contains(set_name) {
                        return Err(DiagramError::UndefinedSet(set_name.clone()));
                    }
                }
            }
        }

        Ok(Diagram {
            combinations: self.combinations,
            input_type: self.input_type.unwrap_or_default(),
            set_names,
        })
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
    fn test_builder_simple() {
        let diagram = DiagramBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .build()
            .unwrap();

        assert_eq!(diagram.set_names().len(), 2);
        assert!(diagram.set_names().contains("A"));
        assert!(diagram.set_names().contains("B"));
    }

    #[test]
    fn test_builder_with_intersection() {
        let diagram = DiagramBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Disjoint)
            .build()
            .unwrap();

        assert_eq!(diagram.input_type(), InputType::Disjoint);
        assert_eq!(diagram.combinations().len(), 3);
    }

    #[test]
    fn test_builder_undefined_set_error() {
        let result = DiagramBuilder::new()
            .set("A", 5.0)
            .intersection(&["A", "B"], 1.0)
            .build();

        assert!(matches!(result, Err(DiagramError::UndefinedSet(_))));
    }

    #[test]
    fn test_builder_negative_value_error() {
        let result = DiagramBuilder::new().set("A", -5.0).build();

        assert!(matches!(result, Err(DiagramError::InvalidValue { .. })));
    }

    #[test]
    fn test_builder_empty_error() {
        let result = DiagramBuilder::new().build();
        assert!(matches!(result, Err(DiagramError::EmptySets)));
    }

    #[test]
    fn test_input_type_default() {
        let diagram = DiagramBuilder::new().set("A", 5.0).build().unwrap();

        assert_eq!(diagram.input_type(), InputType::Union);
    }

    #[test]
    fn test_three_way_intersection() {
        let diagram = DiagramBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .set("C", 12.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "C"], 3.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.5)
            .build()
            .unwrap();

        assert_eq!(diagram.set_names().len(), 3);
        assert_eq!(diagram.combinations().len(), 7);
    }

    #[test]
    fn test_get_combination() {
        let diagram = DiagramBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .build()
            .unwrap();

        let combo_ab = Combination::new(&["A", "B"]);
        assert_eq!(diagram.get_combination(&combo_ab), Some(1.0));

        let combo_ac = Combination::new(&["A", "C"]);
        assert_eq!(diagram.get_combination(&combo_ac), None);
    }
}
