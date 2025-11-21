//! Builder for constructing diagrams.

use super::{Combination, Diagram, InputType};
use crate::error::DiagramError;
use std::collections::{HashMap, HashSet};

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
