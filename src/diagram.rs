//! Diagram specification and construction.
//!
//! This module provides types for defining Euler and Venn diagram specifications through
//! set combinations and their values.

mod combination;
mod input;
mod spec;

pub use crate::error::DiagramError;
pub use combination::Combination;
pub use input::InputType;
pub use spec::DiagramSpecBuilder;

use std::collections::{HashMap, HashSet};

/// Represents a complete Euler or Venn diagram specification.
///
/// This contains the input data (set sizes and intersections) that describes
/// what the diagram should represent. The actual geometric shapes will be
/// computed during the fitting process.
///
/// Both disjoint and union representations are stored for efficient access.
#[derive(Debug, Clone)]
pub struct DiagramSpec {
    /// Disjoint areas (unique parts of each combination)
    pub(crate) disjoint_areas: HashMap<Combination, f64>,

    /// Union areas (inclusive of all subsets)
    pub(crate) union_areas: HashMap<Combination, f64>,

    /// How the input values were originally specified.
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

    /// Returns the disjoint areas.
    pub fn disjoint_areas(&self) -> &HashMap<Combination, f64> {
        &self.disjoint_areas
    }

    /// Returns the union areas.
    pub fn union_areas(&self) -> &HashMap<Combination, f64> {
        &self.union_areas
    }

    /// Gets the disjoint area for a specific combination.
    pub fn get_disjoint(&self, combination: &Combination) -> Option<f64> {
        self.disjoint_areas.get(combination).copied()
    }

    /// Gets the union area for a specific combination.
    pub fn get_union(&self, combination: &Combination) -> Option<f64> {
        self.union_areas.get(combination).copied()
    }

    /// Preprocess the specification for fitting.
    ///
    /// This:
    /// 1. Removes empty sets (area < ε)
    /// 2. Removes combinations containing empty sets
    /// 3. Computes pairwise relationships (subset, disjoint)
    pub fn preprocess(&self) -> Result<PreprocessedSpec, DiagramError> {
        const EPSILON: f64 = 1e-10; // sqrt of machine epsilon

        // 1. Find empty sets (use union areas to determine empty sets)
        let mut non_empty_sets = Vec::new();
        let mut set_to_idx = HashMap::new();

        for set_name in self.set_names.iter() {
            let combo = Combination::new(&[set_name]);
            if let Some(&area) = self.union_areas.get(&combo) {
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
        let mut filtered_disjoint = HashMap::new();
        let mut filtered_union = HashMap::new();

        for (combo, &area) in self.disjoint_areas.iter() {
            // Check if all sets in this combination are non-empty
            let all_non_empty = combo.sets().iter().all(|s| set_to_idx.contains_key(s));

            if all_non_empty {
                filtered_disjoint.insert(combo.clone(), area);
                if let Some(&union_area) = self.union_areas.get(combo) {
                    filtered_union.insert(combo.clone(), union_area);
                }
            }
        }

        // 3. Compute pairwise relationships
        let relationships = Self::compute_pairwise_relations(&non_empty_sets, &filtered_union)?;

        Ok(PreprocessedSpec {
            set_names: non_empty_sets,
            set_to_idx,
            disjoint_areas: filtered_disjoint,
            union_areas: filtered_union,
            n_sets,
            relationships,
        })
    }

    fn compute_pairwise_relations(
        set_names: &[String],
        union_areas: &HashMap<Combination, f64>,
    ) -> Result<PairwiseRelations, DiagramError> {
        let n = set_names.len();

        // Compute radii for single sets
        let mut radii = vec![0.0; n];
        for (i, set_name) in set_names.iter().enumerate() {
            let combo = Combination::new(&[set_name]);
            if let Some(&area) = union_areas.get(&combo) {
                radii[i] = (area / std::f64::consts::PI).sqrt();
            }
        }

        // Initialize relationship matrices
        let mut subset = vec![vec![false; n]; n];
        let mut disjoint = vec![vec![false; n]; n];

        // Check all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let set_i = &set_names[i];
                let set_j = &set_names[j];

                let combo_i = Combination::new(&[set_i]);
                let combo_j = Combination::new(&[set_j]);
                let combo_ij = Combination::new(&[set_i, set_j]);

                let area_i = union_areas.get(&combo_i).copied().unwrap_or(0.0);
                let area_j = union_areas.get(&combo_j).copied().unwrap_or(0.0);
                let area_ij = union_areas.get(&combo_ij).copied().unwrap_or(0.0);

                // Check if disjoint (intersection is zero)
                if area_ij < 1e-10 {
                    disjoint[i][j] = true;
                    disjoint[j][i] = true;
                }

                // Check if one is subset of another
                // j ⊆ i if area(i ∩ j) == area(j)
                if (area_ij - area_j).abs() < 1e-10 {
                    subset[i][j] = true; // j is subset of i
                }
                if (area_ij - area_i).abs() < 1e-10 {
                    subset[j][i] = true; // i is subset of j
                }
            }
        }

        Ok(PairwiseRelations {
            radii,
            subset,
            disjoint,
        })
    }

    /// Convert disjoint areas to union areas (static version for builder).
    fn disjoint_to_union_static(
        disjoint: &HashMap<Combination, f64>,
    ) -> Result<HashMap<Combination, f64>, DiagramError> {
        let mut union: HashMap<Combination, f64> = HashMap::new();

        // For each combination, its union area = its disj area + disj areas of all supersets
        for (combo, &disj_area) in disjoint.iter() {
            let mut union_area = disj_area;

            // Add disjoint areas of all proper supersets (combinations that contain this one)
            for (other_combo, &other_disj) in disjoint.iter() {
                // other_combo is a proper superset of combo (contains all sets in combo, plus more)
                if other_combo != combo && other_combo.contains_all(combo) {
                    union_area += other_disj;
                }
            }

            union.insert(combo.clone(), union_area);
        }

        Ok(union)
    }

    /// Convert union areas to disjoint areas (static version for builder).
    fn union_to_disjoint_static(
        union: &HashMap<Combination, f64>,
    ) -> Result<HashMap<Combination, f64>, DiagramError> {
        let mut disjoint: HashMap<Combination, f64> = HashMap::new();

        // Sort combinations by size (process from largest to smallest)
        let mut sorted_combos: Vec<_> = union.keys().collect();
        sorted_combos.sort_by_key(|c| std::cmp::Reverse(c.len()));

        for combo in sorted_combos {
            let union_area = union[combo];
            let mut disjoint_area = union_area;

            // Subtract disjoint areas of all proper supersets (combinations that contain this one)
            for (other_combo, &other_disj) in disjoint.iter() {
                if other_combo != combo && other_combo.contains_all(combo) {
                    disjoint_area -= other_disj;
                }
            }

            if disjoint_area < -1e-10 {
                return Err(DiagramError::InvalidValue {
                    combination: combo.to_string(),
                    value: disjoint_area,
                });
            }

            disjoint.insert(combo.clone(), disjoint_area.max(0.0));
        }

        Ok(disjoint)
    }
}

pub struct PreprocessedSpec {
    /// Non-empty set names in canonical order
    pub set_names: Vec<String>,

    /// Mapping from set name to index in set_names
    pub set_to_idx: HashMap<String, usize>,

    /// All non-empty combinations with their disjoint areas
    pub disjoint_areas: HashMap<Combination, f64>,

    /// All non-empty combinations with their union areas  
    pub union_areas: HashMap<Combination, f64>,

    /// Number of non-empty sets
    pub n_sets: usize,

    /// Pairwise relationships
    pub relationships: PairwiseRelations,
}

pub struct PairwiseRelations {
    /// radii[i] = sqrt(area[i] / π) for set i
    pub radii: Vec<f64>,

    /// subset[i][j] = true if set j ⊆ set i
    pub subset: Vec<Vec<bool>>,

    /// disjoint[i][j] = true if sets i and j are disjoint
    pub disjoint: Vec<Vec<bool>>,
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
            .input_type(InputType::Union)
            .build()
            .unwrap();

        // Check union areas (what we input)
        assert_eq!(spec.get_union(&Combination::new(&["A"])), Some(10.0));
        assert_eq!(spec.get_union(&Combination::new(&["B"])), Some(8.0));
        assert_eq!(spec.get_union(&Combination::new(&["A", "B"])), Some(2.0));

        // Check disjoint areas (computed)
        assert_eq!(spec.get_disjoint(&Combination::new(&["A"])), Some(8.0)); // 10 - 2
        assert_eq!(spec.get_disjoint(&Combination::new(&["B"])), Some(6.0)); // 8 - 2
        assert_eq!(spec.get_disjoint(&Combination::new(&["A", "B"])), Some(2.0));
    }

    #[test]
    fn test_disjoint_input() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0) // disjoint A-only
            .set("B", 2.0) // disjoint B-only
            .intersection(&["A", "B"], 1.0) // disjoint overlap
            .input_type(InputType::Disjoint)
            .build()
            .unwrap();

        // Check disjoint areas (what we input)
        assert_eq!(spec.get_disjoint(&Combination::new(&["A"])), Some(5.0));
        assert_eq!(spec.get_disjoint(&Combination::new(&["B"])), Some(2.0));
        assert_eq!(spec.get_disjoint(&Combination::new(&["A", "B"])), Some(1.0));

        // Check union areas (computed)
        assert_eq!(spec.get_union(&Combination::new(&["A"])), Some(6.0)); // 5 + 1
        assert_eq!(spec.get_union(&Combination::new(&["B"])), Some(3.0)); // 2 + 1
        assert_eq!(spec.get_union(&Combination::new(&["A", "B"])), Some(1.0));
    }
}
