//! Builder for constructing diagram specifications.

use super::{Combination, DiagramSpec, InputType};
use crate::constants::MAX_SETS;
use crate::error::DiagramError;
use std::collections::{HashMap, HashSet};

/// Builder for creating diagram specifications with a fluent API.
///
/// The specification is shape-agnostic - the shape type is determined when
/// fitting the diagram using a `Fitter`, not when building the spec.
///
/// # Examples
///
/// ```
/// use eunoia::{DiagramSpecBuilder, InputType, Fitter};
/// use eunoia::geometry::shapes::Circle;
///
/// let spec = DiagramSpecBuilder::new()
///     .set("A", 5.0)
///     .set("B", 2.0)
///     .intersection(&["A", "B"], 1.0)
///     .input_type(InputType::Exclusive)
///     .build()
///     .expect("Failed to build diagram specification");
///
/// // Shape type is chosen when fitting
/// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
/// ```
#[derive(Debug)]
pub struct DiagramSpecBuilder {
    combinations: HashMap<Combination, f64>,
    input_type: Option<InputType>,
    set_order: Vec<String>,
}

impl Default for DiagramSpecBuilder {
    fn default() -> Self {
        Self::new()
    }
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
    /// * `input_type` - Whether values are exclusive or inclusive
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, InputType};
    ///
    ///
    /// let builder = DiagramSpecBuilder::new()
    ///     .input_type(InputType::Exclusive);
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

        // Use set_order to create an ordered vector of set names, then add any
        // implicitly defined sets that weren't in the original order
        let mut ordered_set_names: Vec<String> = self
            .set_order
            .iter()
            .filter(|name| all_set_names.contains(*name))
            .cloned()
            .collect();

        // Add any sets that were implicitly discovered but not in set_order
        for set_name in &all_set_names {
            if !ordered_set_names.contains(set_name) {
                ordered_set_names.push(set_name.clone());
            }
        }

        // Enforce the hard cap on set count. The internal RegionMask is a
        // usize bitset, but we cap well below the bit limit because a fully
        // overlapping diagram has 2^n - 1 regions — sparse inputs scale fine
        // up to MAX_SETS, but the worst case is bounded.
        if ordered_set_names.len() > MAX_SETS {
            return Err(DiagramError::TooManySets {
                requested: ordered_set_names.len(),
                max: MAX_SETS,
            });
        }

        // Validate that all values are non-negative
        for (combination, &value) in &combinations {
            if value < 0.0 {
                return Err(DiagramError::InvalidValue {
                    combination: combination.to_string(),
                    value,
                });
            }
        }

        // Reduce input to the canonical exclusive representation. The
        // inclusive view is computed on demand by `DiagramSpec` and is
        // never stored in the spec; this keeps build cost proportional to
        // input size, with no `2^|c|` subset walk per input combination.
        let input_type = self.input_type.unwrap_or_default();
        let exclusive_areas = match input_type {
            InputType::Exclusive => combinations,
            InputType::Inclusive => DiagramSpec::inclusive_to_exclusive_static(&combinations)?,
        };

        Ok(DiagramSpec {
            exclusive_areas,
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
        assert!(spec.get_inclusive(&Combination::new(&["A"])).is_some());
        assert!(spec.get_exclusive(&Combination::new(&["A"])).is_some());
    }

    #[test]
    fn test_builder_with_intersection() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 2.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        assert_eq!(spec.input_type(), InputType::Exclusive);
        assert_eq!(spec.exclusive_areas().len(), 3);
        assert_eq!(spec.inclusive_areas().len(), 3);
    }

    #[test]
    fn test_builder_implicit_zero_set() {
        // When a set is referenced in an intersection but not defined as a single set,
        // it should be implicitly added with value 0.0
        let spec = DiagramSpecBuilder::new()
            .set("B", 5.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Set A should be implicitly added with exclusive value 0.0
        let combo_a = Combination::new(&["A"]);
        assert_eq!(spec.get_exclusive(&combo_a), Some(0.0));

        // In inclusive representation, A should have total size = 1.0 (just the intersection)
        assert_eq!(spec.get_inclusive(&combo_a), Some(1.0));
    }

    #[test]
    fn test_contained_set_exclusive() {
        // Test case: A=0, B=5, A&B=1 (exclusive)
        // This means A is entirely contained within B
        let spec = DiagramSpecBuilder::new()
            .set("A", 0.0)
            .set("B", 5.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Exclusive areas
        assert_eq!(spec.get_exclusive(&Combination::new(&["A"])), Some(0.0));
        assert_eq!(spec.get_exclusive(&Combination::new(&["B"])), Some(5.0));
        assert_eq!(
            spec.get_exclusive(&Combination::new(&["A", "B"])),
            Some(1.0)
        );

        // Inclusive areas (total sizes)
        assert_eq!(spec.get_inclusive(&Combination::new(&["A"])), Some(1.0)); // A total = 0 + 1
        assert_eq!(spec.get_inclusive(&Combination::new(&["B"])), Some(6.0)); // B total = 5 + 1
        assert_eq!(
            spec.get_inclusive(&Combination::new(&["A", "B"])),
            Some(1.0)
        );
    }

    #[test]
    fn test_implicit_set_from_three_way() {
        // Test case: A=0, B=5, A&B=1, A&B&C=0.1 (disjoint)
        // Set C is only referenced in the 3-way intersection
        let spec = DiagramSpecBuilder::new()
            .set("A", 0.0)
            .set("B", 5.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["A", "B", "C"], 0.1)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // All three sets should exist
        assert_eq!(spec.set_names().len(), 3);
        assert!(spec.set_names().contains(&"A".to_string()));
        assert!(spec.set_names().contains(&"B".to_string()));
        assert!(spec.set_names().contains(&"C".to_string()));

        // Check that C was implicitly added with value 0.0
        assert_eq!(spec.get_exclusive(&Combination::new(&["C"])), Some(0.0));

        // C's inclusive area should be 0.1 (just the 3-way intersection)
        assert_eq!(spec.get_inclusive(&Combination::new(&["C"])), Some(0.1));
    }

    #[test]
    fn test_nested_containment() {
        // Test case: B=5, A&B=2, A&B&C=1 (disjoint)
        // A is contained in B, and C is contained in A&B
        let spec = DiagramSpecBuilder::new()
            .set("B", 5.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // All three sets should exist
        assert_eq!(spec.set_names().len(), 3);

        // Expected exclusive areas
        assert_eq!(spec.get_exclusive(&Combination::new(&["A"])), Some(0.0));
        assert_eq!(spec.get_exclusive(&Combination::new(&["B"])), Some(5.0));
        assert_eq!(spec.get_exclusive(&Combination::new(&["C"])), Some(0.0));
        assert_eq!(
            spec.get_exclusive(&Combination::new(&["A", "B"])),
            Some(2.0)
        );
        assert_eq!(
            spec.get_exclusive(&Combination::new(&["A", "B", "C"])),
            Some(1.0)
        );

        // Expected inclusive areas (total sizes)
        assert_eq!(spec.get_inclusive(&Combination::new(&["A"])), Some(3.0)); // 0 + 2 + 1
        assert_eq!(spec.get_inclusive(&Combination::new(&["B"])), Some(8.0)); // 5 + 2 + 1
        assert_eq!(spec.get_inclusive(&Combination::new(&["C"])), Some(1.0)); // 0 + 1
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
        assert_eq!(spec.exclusive_areas().len(), 7);
        assert_eq!(spec.inclusive_areas().len(), 7);
    }

    #[test]
    fn test_too_many_sets_rejected() {
        // MAX_SETS + 1 distinct singletons should be rejected with TooManySets,
        // before any 2^n preprocessing step has a chance to allocate.
        let mut builder = DiagramSpecBuilder::new();
        for i in 0..(MAX_SETS + 1) {
            builder = builder.set(format!("S{i}"), 1.0);
        }
        let result = builder.build();
        assert!(
            matches!(
                result,
                Err(DiagramError::TooManySets { requested, max })
                if requested == MAX_SETS + 1 && max == MAX_SETS
            ),
            "expected TooManySets for n = MAX_SETS + 1, got {:?}",
            result
        );
    }

    #[test]
    fn test_max_sets_accepted() {
        // Exactly MAX_SETS singletons should build cleanly. The sparse
        // exclusive→inclusive path must not blow up at this size.
        let mut builder = DiagramSpecBuilder::new();
        for i in 0..MAX_SETS {
            builder = builder.set(format!("S{i}"), 1.0);
        }
        let spec = builder.build().expect("MAX_SETS singletons should build");
        assert_eq!(spec.set_names().len(), MAX_SETS);
        // Sparse: exactly one inclusive entry per singleton, no power-set blowup.
        assert_eq!(spec.inclusive_areas().len(), MAX_SETS);
    }

    #[test]
    fn test_sparse_exclusive_to_inclusive_no_power_set_blowup() {
        // 25 disjoint singletons + one pair. With the old dense
        // exclusive→inclusive, this would walk 2^25 ≈ 33M masks. The sparse
        // version touches only the input combinations + their subsets.
        let mut builder = DiagramSpecBuilder::new();
        for i in 0..25 {
            builder = builder.set(format!("S{i}"), 1.0);
        }
        let spec = builder
            .intersection(&["S0", "S1"], 0.5)
            .input_type(InputType::Exclusive)
            .build()
            .expect("sparse 25-set spec should build");

        // S0 inclusive = own exclusive (1.0) + pair (0.5) = 1.5
        assert!(
            (spec
                .get_inclusive(&Combination::new(&["S0"]))
                .expect("S0 inclusive")
                - 1.5)
                .abs()
                < 1e-10
        );
        // Untouched singletons keep their input area as inclusive area.
        assert!(
            (spec
                .get_inclusive(&Combination::new(&["S5"]))
                .expect("S5 inclusive")
                - 1.0)
                .abs()
                < 1e-10
        );
        // The pair appears in the inclusive map exactly once.
        assert!(
            (spec
                .get_inclusive(&Combination::new(&["S0", "S1"]))
                .expect("S0&S1 inclusive")
                - 0.5)
                .abs()
                < 1e-10
        );
        // No spurious 3+ way regions get materialized.
        assert!(spec
            .get_inclusive(&Combination::new(&["S0", "S1", "S2"]))
            .is_none());
        // Total inclusive entries = 25 singletons + 1 pair = 26 (sparse).
        assert_eq!(spec.inclusive_areas().len(), 26);
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
        assert_eq!(spec.get_inclusive(&combo_ab), Some(1.0));

        let combo_ac = Combination::new(&["A", "C"]);
        assert_eq!(spec.get_inclusive(&combo_ac), None);
    }

    #[test]
    fn test_build_scales_with_large_kway_intersection() {
        // A single 30-way intersection used to walk 2^30 ≈ 1B subsets at
        // build time via the eager exclusive→inclusive expansion. With the
        // lazy inclusive view, build cost is bounded by the input size.
        let n = 30;
        let names: Vec<String> = (0..n).map(|i| format!("S{i}")).collect();
        let mut builder = DiagramSpecBuilder::new();
        for name in &names {
            builder = builder.set(name.as_str(), 1.0);
        }
        let intersection_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let start = std::time::Instant::now();
        let spec = builder
            .intersection(&intersection_refs, 0.5)
            .input_type(InputType::Exclusive)
            .build()
            .expect("30-way spec should build");
        let elapsed = start.elapsed();

        // 30 singletons + 1 thirty-way intersection.
        assert_eq!(spec.exclusive_areas().len(), n + 1);

        // A loose timing bound — anything under a second proves we're not
        // walking 2^30 subsets. On a typical dev machine this finishes in
        // microseconds.
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "30-way build took {:?}; expected sub-second",
            elapsed
        );
    }
}
