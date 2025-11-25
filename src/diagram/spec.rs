//! Builder for constructing diagram specifications.

use super::{Combination, DiagramSpec, InputType};
use crate::error::DiagramError;
use std::collections::{HashMap, HashSet};

/// Builder for creating diagram specifications with a fluent API.
///
/// # Examples
///
/// ```
/// use eunoia::{DiagramSpecBuilder, InputType};
///
/// let spec = DiagramSpecBuilder::new()
///     .set("A", 5.0)
///     .set("B", 2.0)
///     .intersection(&["A", "B"], 1.0)
///     .input_type(InputType::Disjoint)
///     .build()
///     .expect("Failed to build diagram specification");
/// ```
#[derive(Debug, Default)]
pub struct DiagramSpecBuilder {
    combinations: HashMap<Combination, f64>,
    input_type: Option<InputType>,
    set_order: Vec<String>,
}

impl DiagramSpecBuilder {
    /// Creates a new diagram builder.
    pub fn new() -> Self {
        DiagramSpecBuilder {
            combinations: HashMap::new(),
            input_type: None,
            set_order: Vec::new(),
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
    /// use eunoia::DiagramSpecBuilder;
    ///
    /// let builder = DiagramSpecBuilder::new()
    ///     .set("A", 10.0);
    /// ```
    pub fn set(mut self, name: impl Into<String>, value: f64) -> Self {
        let name_string = name.into();
        let combination = Combination::new(&[&name_string]);
        // Track order of first occurrence
        if !self.set_order.contains(&name_string) {
            self.set_order.push(name_string.clone());
        }
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
    /// use eunoia::DiagramSpecBuilder;
    ///
    /// let builder = DiagramSpecBuilder::new()
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
    /// use eunoia::{DiagramSpecBuilder, InputType};
    ///
    /// let builder = DiagramSpecBuilder::new()
    ///     .input_type(InputType::Disjoint);
    /// ```
    pub fn input_type(mut self, input_type: InputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Builds the diagram specification, validating all inputs.
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
    /// use eunoia::{DiagramSpecBuilder, InputType};
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 2.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .build()
    ///     .expect("Failed to build diagram specification");
    /// ```
    pub fn build(self) -> Result<DiagramSpec, DiagramError> {
        // Check that we have at least one set
        if self.combinations.is_empty() {
            return Err(DiagramError::EmptySets);
        }

        // Collect all unique set names from all combinations
        let mut all_set_names = HashSet::new();
        let mut single_sets = HashSet::new();

        for combination in self.combinations.keys() {
            for set_name in combination.sets() {
                all_set_names.insert(set_name.clone());
            }
            if combination.len() == 1 {
                single_sets.insert(combination.sets()[0].clone());
            }
        }

        // For sets that appear in intersections but not as single sets,
        // implicitly add them with value 0.0 (they exist entirely within intersections)
        let mut combinations = self.combinations;
        for set_name in &all_set_names {
            if !single_sets.contains(set_name) {
                let combination = Combination::new(&[set_name.as_str()]);
                combinations.insert(combination, 0.0);
                single_sets.insert(set_name.clone());
            }
        }

        // Use set_order to create an ordered vector of set names
        let ordered_set_names: Vec<String> = self
            .set_order
            .iter()
            .filter(|name| all_set_names.contains(*name))
            .cloned()
            .collect();

        // Validate that all values are non-negative
        for (combination, &value) in &combinations {
            if value < 0.0 {
                return Err(DiagramError::InvalidValue {
                    combination: combination.to_string(),
                    value,
                });
            }
        }

        // Convert to both disjoint and union representations
        let input_type = self.input_type.unwrap_or_default();
        let (disjoint_areas, union_areas) = match input_type {
            InputType::Disjoint => {
                let disjoint = combinations;
                let union = DiagramSpec::disjoint_to_union_static(&disjoint)?;
                (disjoint, union)
            }
            InputType::Union => {
                let union = combinations;
                let disjoint = DiagramSpec::union_to_disjoint_static(&union)?;
                (disjoint, union)
            }
        };

        Ok(DiagramSpec {
            disjoint_areas,
            union_areas,
            input_type,
            set_names: ordered_set_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_simple() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .build()
            .unwrap();

        assert_eq!(spec.set_names().len(), 2);
        assert!(spec.set_names().contains(&"A".to_string()));
        assert!(spec.set_names().contains(&"B".to_string()));

        // Both representations should be available
        assert!(spec.get_union(&Combination::new(&["A"])).is_some());
        assert!(spec.get_disjoint(&Combination::new(&["A"])).is_some());
    }

    #[test]
    fn test_builder_with_intersection() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Disjoint)
            .build()
            .unwrap();

        assert_eq!(spec.input_type(), InputType::Disjoint);
        assert_eq!(spec.disjoint_areas().len(), 3);
        assert_eq!(spec.union_areas().len(), 3);
    }

    #[test]
    fn test_builder_implicit_zero_set() {
        // When a set is referenced in an intersection but not defined as a single set,
        // it should be implicitly added with value 0.0
        let spec = DiagramSpecBuilder::new()
            .set("B", 5.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Disjoint)
            .build()
            .unwrap();

        // Set A should be implicitly added with disjoint value 0.0
        let combo_a = Combination::new(&["A"]);
        assert_eq!(spec.get_disjoint(&combo_a), Some(0.0));

        // In union representation, A should have total size = 1.0 (just the intersection)
        assert_eq!(spec.get_union(&combo_a), Some(1.0));
    }

    #[test]
    fn test_contained_set_disjoint() {
        // Test case: A=0, B=5, A&B=1 (disjoint)
        // This means A is entirely contained within B
        let spec = DiagramSpecBuilder::new()
            .set("A", 0.0)
            .set("B", 5.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Disjoint)
            .build()
            .unwrap();

        // Disjoint areas
        assert_eq!(spec.get_disjoint(&Combination::new(&["A"])), Some(0.0));
        assert_eq!(spec.get_disjoint(&Combination::new(&["B"])), Some(5.0));
        assert_eq!(spec.get_disjoint(&Combination::new(&["A", "B"])), Some(1.0));

        // Union areas (total sizes)
        assert_eq!(spec.get_union(&Combination::new(&["A"])), Some(1.0)); // A total = 0 + 1
        assert_eq!(spec.get_union(&Combination::new(&["B"])), Some(6.0)); // B total = 5 + 1
        assert_eq!(spec.get_union(&Combination::new(&["A", "B"])), Some(1.0));
    }

    #[test]
    fn test_builder_negative_value_error() {
        let result = DiagramSpecBuilder::new().set("A", -5.0).build();

        assert!(matches!(result, Err(DiagramError::InvalidValue { .. })));
    }

    #[test]
    fn test_builder_empty_error() {
        let result = DiagramSpecBuilder::new().build();
        assert!(matches!(result, Err(DiagramError::EmptySets)));
    }

    #[test]
    fn test_input_type_default() {
        let spec = DiagramSpecBuilder::new().set("A", 5.0).build().unwrap();

        assert_eq!(spec.input_type(), InputType::Union);
    }

    #[test]
    fn test_three_way_intersection() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .set("C", 12.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "C"], 3.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.5)
            .build()
            .unwrap();

        assert_eq!(spec.set_names().len(), 3);
        assert_eq!(spec.disjoint_areas().len(), 7);
        assert_eq!(spec.union_areas().len(), 7);
    }

    #[test]
    fn test_get_combination() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .build()
            .unwrap();

        let combo_ab = Combination::new(&["A", "B"]);
        assert_eq!(spec.get_union(&combo_ab), Some(1.0));

        let combo_ac = Combination::new(&["A", "C"]);
        assert_eq!(spec.get_union(&combo_ac), None);
    }
}
