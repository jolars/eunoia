//! Diagram specification and construction.
//!
//! This module provides types for defining Euler and Venn diagram specifications through
//! set combinations and their values.

mod combination;
mod input;
mod spec_builder;

pub use crate::error::DiagramError;
pub use combination::Combination;
pub use input::InputType;
pub use spec_builder::DiagramSpecBuilder;

use std::collections::HashMap;

/// Represents a complete Euler or Venn diagram specification.
///
/// This contains the input data (set sizes and intersections) that describes
/// what the diagram should represent. The actual geometric shapes will be
/// computed during the fitting process.
///
/// Both exclusive and inclusive representations are stored for efficient access.
#[derive(Debug, Clone)]
pub struct DiagramSpec {
    /// Exclusive areas (unique parts of each combination)
    pub(crate) exclusive_areas: HashMap<Combination, f64>,

    /// Inclusive areas (inclusive of all subsets)
    pub(crate) inclusive_areas: HashMap<Combination, f64>,

    /// How the input values were originally specified.
    pub(crate) input_type: InputType,

    /// Set of all unique set names in the diagram (ordered).
    pub(crate) set_names: Vec<String>,
}

impl DiagramSpec {
    /// Returns the input type for this diagram specification.
    pub fn input_type(&self) -> InputType {
        self.input_type
    }

    /// Returns the set names in this diagram specification (in order).
    pub fn set_names(&self) -> &[String] {
        &self.set_names
    }

    /// Returns the exclusive areas.
    pub fn exclusive_areas(&self) -> &HashMap<Combination, f64> {
        &self.exclusive_areas
    }

    /// Returns the inclusive areas.
    pub fn inclusive_areas(&self) -> &HashMap<Combination, f64> {
        &self.inclusive_areas
    }

    /// Gets the exclusive area for a specific combination.
    pub fn get_exclusive(&self, combination: &Combination) -> Option<f64> {
        self.exclusive_areas.get(combination).copied()
    }

    /// Gets the inclusive area for a specific combination.
    pub fn get_inclusive(&self, combination: &Combination) -> Option<f64> {
        self.inclusive_areas.get(combination).copied()
    }

    /// Preprocess the specification for fitting (internal use).
    ///
    /// This:
    /// 1. Removes empty sets (area < ε)
    /// 2. Removes combinations containing empty sets
    /// 3. Computes pairwise relationships (subset, disjoint)
    #[allow(dead_code)]
    pub(crate) fn preprocess(&self) -> Result<PreprocessedSpec, DiagramError> {
        const EPSILON: f64 = 1e-10; // sqrt of machine epsilon

        // 1. Find empty sets (use inclusive areas to determine empty sets)
        let mut non_empty_sets = Vec::new();
        let mut set_to_idx = HashMap::new();

        for set_name in self.set_names.iter() {
            let combo = Combination::new(&[set_name]);
            if let Some(&area) = self.inclusive_areas.get(&combo) {
                if area >= EPSILON {
                    let idx = non_empty_sets.len();
                    non_empty_sets.push(set_name.clone());
                    set_to_idx.insert(set_name.clone(), idx);
                }
            }
        }

        let n_sets = non_empty_sets.len();

        if n_sets <= 1 {
            return Err(DiagramError::InvalidCombination(
                "Need at least 2 non-empty sets".to_string(),
            ));
        }

        // 2. Filter combinations to only include non-empty sets
        let mut filtered_exclusive = HashMap::new();
        let mut filtered_inclusive = HashMap::new();

        // First, add all combinations from exclusive
        for (combo, &area) in self.exclusive_areas.iter() {
            // Check if all sets in this combination are non-empty
            let all_non_empty = combo.sets().iter().all(|s| set_to_idx.contains_key(s));

            if all_non_empty {
                filtered_exclusive.insert(combo.clone(), area);
                if let Some(&inclusive_area) = self.inclusive_areas.get(combo) {
                    filtered_inclusive.insert(combo.clone(), inclusive_area);
                }
            }
        }

        // Also add combinations from inclusive that might have zero exclusive area
        for (combo, &inclusive_area) in self.inclusive_areas.iter() {
            let all_non_empty = combo.sets().iter().all(|s| set_to_idx.contains_key(s));

            if all_non_empty && inclusive_area > 1e-10 && !filtered_inclusive.contains_key(combo) {
                filtered_inclusive.insert(combo.clone(), inclusive_area);
                // Add to exclusive with 0 if not already there
                if !filtered_exclusive.contains_key(combo) {
                    filtered_exclusive.insert(combo.clone(), 0.0);
                }
            }
        }

        // 3. Compute set areas
        let mut set_areas = vec![0.0; n_sets];
        for (i, set_name) in non_empty_sets.iter().enumerate() {
            let combo = Combination::new(&[set_name]);
            if let Some(&area) = filtered_inclusive.get(&combo) {
                set_areas[i] = area;
            }
        }

        // 4. Compute pairwise relationships
        let relationships = Self::compute_pairwise_relations(&non_empty_sets, &filtered_inclusive)?;

        Ok(PreprocessedSpec {
            set_names: non_empty_sets,
            set_to_idx,
            exclusive_areas: filtered_exclusive,
            inclusive_areas: filtered_inclusive,
            n_sets,
            set_areas,
            relationships,
        })
    }

    #[allow(dead_code)]
    fn compute_pairwise_relations(
        set_names: &[String],
        inclusive_areas: &HashMap<Combination, f64>,
    ) -> Result<PairwiseRelations, DiagramError> {
        let n = set_names.len();

        // Initialize relationship matrices
        let mut subset = vec![vec![false; n]; n];
        let mut disjoint = vec![vec![false; n]; n];
        let mut overlap_areas = vec![vec![0.0; n]; n];

        // Check all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let set_i = &set_names[i];
                let set_j = &set_names[j];

                let combo_i = Combination::new(&[set_i]);
                let combo_j = Combination::new(&[set_j]);
                let combo_ij = Combination::new(&[set_i, set_j]);

                let area_i = inclusive_areas.get(&combo_i).copied().unwrap_or(0.0);
                let area_j = inclusive_areas.get(&combo_j).copied().unwrap_or(0.0);
                let area_ij_inclusive = inclusive_areas.get(&combo_ij).copied().unwrap_or(0.0);

                // Store overlap area - use inclusive intersection
                // This represents the total geometric intersection including higher-order overlaps
                overlap_areas[i][j] = area_ij_inclusive;
                overlap_areas[j][i] = area_ij_inclusive;

                // Check if disjoint (intersection is zero)
                if area_ij_inclusive < 1e-10 {
                    disjoint[i][j] = true;
                    disjoint[j][i] = true;
                }

                // Check if one is subset of another
                // j ⊆ i if inclusive area(i ∩ j) == area(j)
                if (area_ij_inclusive - area_j).abs() < 1e-10 {
                    subset[i][j] = true; // j is subset of i
                }
                if (area_ij_inclusive - area_i).abs() < 1e-10 {
                    subset[j][i] = true; // i is subset of j
                }
            }
        }

        Ok(PairwiseRelations {
            n_sets: n,
            subset,
            disjoint,
            overlap_areas,
        })
    }

    /// Convert exclusive areas to inclusive areas (static version for builder).
    fn exclusive_to_inclusive_static(
        exclusive: &HashMap<Combination, f64>,
    ) -> Result<HashMap<Combination, f64>, DiagramError> {
        let mut inclusive: HashMap<Combination, f64> = HashMap::new();

        // First, collect all unique set names
        let mut all_sets = std::collections::HashSet::new();
        for combo in exclusive.keys() {
            for set_name in combo.sets() {
                all_sets.insert(set_name.clone());
            }
        }
        let all_sets: Vec<String> = all_sets.into_iter().collect();
        let n_sets = all_sets.len();

        // Generate all possible combinations (power set excluding empty)
        for mask in 1..(1 << n_sets) {
            let mut combo_sets = Vec::new();
            for (i, set_name) in all_sets.iter().enumerate() {
                if (mask & (1 << i)) != 0 {
                    combo_sets.push(set_name.as_str());
                }
            }
            let combo = Combination::new(&combo_sets);

            // Compute inclusive area = sum of exclusive areas of this combo and all its supersets
            let mut inclusive_area = 0.0;
            for (other_combo, &other_excl) in exclusive.iter() {
                // Include if other_combo contains all sets in combo
                if other_combo.contains_all(&combo) {
                    inclusive_area += other_excl;
                }
            }

            // Only include if non-zero
            if inclusive_area > 1e-10 {
                inclusive.insert(combo, inclusive_area);
            }
        }

        Ok(inclusive)
    }

    /// Convert inclusive areas to exclusive areas (static version for builder).
    fn inclusive_to_exclusive_static(
        inclusive: &HashMap<Combination, f64>,
    ) -> Result<HashMap<Combination, f64>, DiagramError> {
        let mut exclusive: HashMap<Combination, f64> = HashMap::new();

        // Sort combinations by size (process from largest to smallest)
        let mut sorted_combos: Vec<_> = inclusive.keys().collect();
        sorted_combos.sort_by_key(|c| std::cmp::Reverse(c.len()));

        for combo in sorted_combos {
            let inclusive_area = inclusive[combo];
            let mut exclusive_area = inclusive_area;

            // Subtract exclusive areas of all proper supersets (combinations that contain this one)
            for (other_combo, &other_excl) in exclusive.iter() {
                if other_combo != combo && other_combo.contains_all(combo) {
                    exclusive_area -= other_excl;
                }
            }

            if exclusive_area < -1e-10 {
                return Err(DiagramError::InvalidValue {
                    combination: combo.to_string(),
                    value: exclusive_area,
                });
            }

            exclusive.insert(combo.clone(), exclusive_area.max(0.0));
        }

        Ok(exclusive)
    }
}

/// Preprocessed specification ready for fitting (internal).
///
/// This is created by filtering out empty sets from a DiagramSpec and
/// computing additional metadata needed for optimization.
#[allow(dead_code)]
pub(crate) struct PreprocessedSpec {
    /// Non-empty set names in canonical order
    pub(crate) set_names: Vec<String>,

    /// Mapping from set name to index in set_names
    pub(crate) set_to_idx: HashMap<String, usize>,

    /// All non-empty combinations with their exclusive areas
    pub(crate) exclusive_areas: HashMap<Combination, f64>,

    /// All non-empty combinations with their inclusive areas  
    pub(crate) inclusive_areas: HashMap<Combination, f64>,

    /// Number of non-empty sets
    pub(crate) n_sets: usize,

    /// Areas for each set (for shape sizing)
    pub(crate) set_areas: Vec<f64>,

    /// Pairwise relationships
    pub(crate) relationships: PairwiseRelations,
}

/// Pairwise relationships between sets (internal).
pub(crate) struct PairwiseRelations {
    /// Number of sets
    #[allow(dead_code)]
    pub(crate) n_sets: usize,

    /// subset[i][j] = true if set j ⊆ set i
    pub(crate) subset: Vec<Vec<bool>>,

    /// disjoint[i][j] = true if sets i and j are disjoint
    pub(crate) disjoint: Vec<Vec<bool>>,

    /// Desired overlap areas between pairs [i][j]
    pub(crate) overlap_areas: Vec<Vec<f64>>,
}

impl PairwiseRelations {
    /// Check if set j is a subset of set i.
    #[allow(dead_code)]
    pub(crate) fn is_subset(&self, i: usize, j: usize) -> bool {
        self.subset[i][j]
    }

    /// Check if sets i and j are disjoint.
    #[allow(dead_code)]
    pub(crate) fn is_disjoint(&self, i: usize, j: usize) -> bool {
        self.disjoint[i][j]
    }

    /// Get the desired overlap area between sets i and j.
    #[allow(dead_code)]
    pub(crate) fn overlap_area(&self, i: usize, j: usize) -> f64 {
        self.overlap_areas[i][j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_both_representations_available() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .input_type(InputType::Inclusive)
            .build()
            .unwrap();

        // Check inclusive areas (what we input)
        assert_eq!(spec.get_inclusive(&Combination::new(&["A"])), Some(10.0));
        assert_eq!(spec.get_inclusive(&Combination::new(&["B"])), Some(8.0));
        assert_eq!(
            spec.get_inclusive(&Combination::new(&["A", "B"])),
            Some(2.0)
        );

        // Check exclusive areas (computed)
        assert_eq!(spec.get_exclusive(&Combination::new(&["A"])), Some(8.0)); // 10 - 2
        assert_eq!(spec.get_exclusive(&Combination::new(&["B"])), Some(6.0)); // 8 - 2
        assert_eq!(
            spec.get_exclusive(&Combination::new(&["A", "B"])),
            Some(2.0)
        );
    }

    #[test]
    fn test_exclusive_input() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0) // exclusive A-only
            .set("B", 2.0) // exclusive B-only
            .intersection(&["A", "B"], 1.0) // exclusive overlap
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Check exclusive areas (what we input)
        assert_eq!(spec.get_exclusive(&Combination::new(&["A"])), Some(5.0));
        assert_eq!(spec.get_exclusive(&Combination::new(&["B"])), Some(2.0));
        assert_eq!(
            spec.get_exclusive(&Combination::new(&["A", "B"])),
            Some(1.0)
        );

        // Check inclusive areas (computed)
        assert_eq!(spec.get_inclusive(&Combination::new(&["A"])), Some(6.0)); // 5 + 1
        assert_eq!(spec.get_inclusive(&Combination::new(&["B"])), Some(3.0)); // 2 + 1
        assert_eq!(
            spec.get_inclusive(&Combination::new(&["A", "B"])),
            Some(1.0)
        );
    }
}
