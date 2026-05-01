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
/// Only the exclusive view is stored; the inclusive view is computed on
/// demand by [`DiagramSpec::inclusive_areas`] and
/// [`DiagramSpec::get_inclusive`]. This keeps build-time cost proportional
/// to the input size — a single large k-way intersection no longer forces a
/// `2^k` walk over its subsets at build time.
///
/// The diagram specification is shape-agnostic - the shape type is determined
/// when fitting the diagram, not when building the specification.
#[derive(Debug, Clone)]
pub struct DiagramSpec {
    /// Exclusive areas (unique parts of each combination)
    pub(crate) exclusive_areas: HashMap<Combination, f64>,

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

    /// Returns the inclusive areas, computed on demand from `exclusive_areas`.
    ///
    /// Cost is `O(Σ 2^|c|)` over input combinations — every non-empty subset
    /// of every input combination is visited. The build path no longer pays
    /// this; only callers of this method do. For a single-entry lookup, prefer
    /// [`DiagramSpec::get_inclusive`].
    pub fn inclusive_areas(&self) -> HashMap<Combination, f64> {
        let mut inclusive: HashMap<Combination, f64> = HashMap::new();

        for (combo_super, &area) in self.exclusive_areas.iter() {
            if area.abs() < crate::constants::EPSILON {
                continue;
            }

            let sets = combo_super.sets();
            let k = sets.len();
            for mask in 1u64..(1u64 << k) {
                let mut subset_sets: Vec<&str> = Vec::with_capacity(k);
                for (i, name) in sets.iter().enumerate() {
                    if (mask >> i) & 1 == 1 {
                        subset_sets.push(name.as_str());
                    }
                }
                let subset_combo = Combination::new(&subset_sets);
                *inclusive.entry(subset_combo).or_insert(0.0) += area;
            }
        }

        inclusive.retain(|_, area| *area > crate::constants::EPSILON);
        inclusive
    }

    /// Gets the exclusive area for a specific combination.
    pub fn get_exclusive(&self, combination: &Combination) -> Option<f64> {
        self.exclusive_areas.get(combination).copied()
    }

    /// Computes the inclusive area for a specific combination on demand.
    ///
    /// Returns `Some(area)` if any superset of `combination` in
    /// `exclusive_areas` contributes a positive area, otherwise `None` —
    /// matching the documented "missing = zero" convention. Cost is
    /// `O(|exclusive_areas|)` per call; no `2^k` subset walk.
    pub fn get_inclusive(&self, combination: &Combination) -> Option<f64> {
        let mut sum = 0.0;
        for (combo, &area) in self.exclusive_areas.iter() {
            if combo.contains_all(combination) {
                sum += area;
            }
        }
        if sum > crate::constants::EPSILON {
            Some(sum)
        } else {
            None
        }
    }

    /// Preprocess the specification for fitting (internal use).
    ///
    /// This:
    /// 1. Removes empty sets (area < ε) — a set is empty iff no input
    ///    combination containing it has positive exclusive area.
    /// 2. Removes combinations touching empty sets.
    /// 3. Computes singleton + pair inclusive areas needed for set sizes
    ///    and pairwise relationships, directly from `exclusive_areas`. No
    ///    full `2^|c|` subset walk over input combinations.
    pub(crate) fn preprocess(&self) -> Result<PreprocessedSpec, DiagramError> {
        const EPSILON: f64 = 1e-10; // sqrt of machine epsilon

        // 1. Compute singleton inclusive areas in a single pass over the
        //    exclusive map: for each (c, a), every set s ∈ c gets +a.
        let mut singleton_inclusive: HashMap<&str, f64> = HashMap::new();
        for (combo, &area) in self.exclusive_areas.iter() {
            if area.abs() < EPSILON {
                continue;
            }
            for set_name in combo.sets() {
                *singleton_inclusive.entry(set_name.as_str()).or_insert(0.0) += area;
            }
        }

        // 2. Determine non-empty sets, preserving canonical order from
        //    `set_names`.
        let mut non_empty_sets: Vec<String> = Vec::new();
        let mut set_to_idx: HashMap<String, usize> = HashMap::new();
        for set_name in self.set_names.iter() {
            let inclusive = singleton_inclusive
                .get(set_name.as_str())
                .copied()
                .unwrap_or(0.0);
            if inclusive >= EPSILON {
                let idx = non_empty_sets.len();
                non_empty_sets.push(set_name.clone());
                set_to_idx.insert(set_name.clone(), idx);
            }
        }

        let n_sets = non_empty_sets.len();
        if n_sets <= 1 {
            return Err(DiagramError::InvalidCombination(
                "Need at least 2 non-empty sets".to_string(),
            ));
        }

        // 3. Filter exclusive areas to combinations whose sets are all
        //    non-empty, and convert to RegionMask format for the fitter.
        use crate::geometry::diagram;
        let mut exclusive_areas_mask = HashMap::new();
        for (combo, &area) in self.exclusive_areas.iter() {
            if combo.sets().iter().all(|s| set_to_idx.contains_key(s)) {
                let mask = diagram::combination_to_mask(combo, &non_empty_sets);
                exclusive_areas_mask.insert(mask, area);
            }
        }

        // 4. Set areas in canonical order.
        let set_areas: Vec<f64> = non_empty_sets
            .iter()
            .map(|s| singleton_inclusive.get(s.as_str()).copied().unwrap_or(0.0))
            .collect();

        // 5. Pairwise relationships, computed sparsely from `exclusive_areas`.
        let relationships = Self::compute_pairwise_relations(
            &non_empty_sets,
            &set_to_idx,
            &set_areas,
            &self.exclusive_areas,
        );

        Ok(PreprocessedSpec {
            set_names: non_empty_sets,
            set_to_idx,
            exclusive_areas: exclusive_areas_mask,
            n_sets,
            set_areas,
            relationships,
        })
    }

    /// Compute pairwise relationships sparsely from exclusive areas.
    ///
    /// Pair-inclusive overlap of (i, j) is the sum of exclusive areas across
    /// every input combination that contains both i and j. One pass over the
    /// exclusive map populates the full `overlap_areas` matrix in
    /// `O(Σ |c|²)` — no `2^|c|` subset walk.
    fn compute_pairwise_relations(
        set_names: &[String],
        set_to_idx: &HashMap<String, usize>,
        set_areas: &[f64],
        exclusive_areas: &HashMap<Combination, f64>,
    ) -> PairwiseRelations {
        const EPSILON: f64 = 1e-10;
        let n = set_names.len();

        let mut subset = vec![vec![false; n]; n];
        let mut disjoint = vec![vec![false; n]; n];
        let mut overlap_areas = vec![vec![0.0; n]; n];

        for (combo, &area) in exclusive_areas.iter() {
            if area.abs() < EPSILON {
                continue;
            }
            // Translate combo's sets into canonical indices; skip if any set
            // is empty (and therefore filtered out).
            let mut indices: Vec<usize> = Vec::with_capacity(combo.sets().len());
            let mut all_non_empty = true;
            for s in combo.sets() {
                match set_to_idx.get(s) {
                    Some(&idx) => indices.push(idx),
                    None => {
                        all_non_empty = false;
                        break;
                    }
                }
            }
            if !all_non_empty || indices.len() < 2 {
                continue;
            }
            for a in 0..indices.len() {
                for b in (a + 1)..indices.len() {
                    let i = indices[a];
                    let j = indices[b];
                    overlap_areas[i][j] += area;
                    overlap_areas[j][i] += area;
                }
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let overlap = overlap_areas[i][j];
                if overlap < EPSILON {
                    disjoint[i][j] = true;
                    disjoint[j][i] = true;
                }
                // j ⊆ i iff overlap(i, j) == area(j).
                if (overlap - set_areas[j]).abs() < EPSILON {
                    subset[i][j] = true;
                }
                if (overlap - set_areas[i]).abs() < EPSILON {
                    subset[j][i] = true;
                }
            }
        }

        PairwiseRelations {
            n_sets: n,
            subset,
            disjoint,
            overlap_areas,
        }
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
#[derive(Clone)]
pub(crate) struct PreprocessedSpec {
    /// Non-empty set names in canonical order
    #[allow(dead_code)] // Canonical order is also encoded in `set_to_idx`.
    pub(crate) set_names: Vec<String>,

    /// Mapping from set name to index in set_names
    pub(crate) set_to_idx: HashMap<String, usize>,

    /// All non-empty combinations with their exclusive areas (internal RegionMask format)
    pub(crate) exclusive_areas: HashMap<crate::geometry::diagram::RegionMask, f64>,

    /// Number of non-empty sets
    pub(crate) n_sets: usize,

    /// Areas for each set (for shape sizing)
    pub(crate) set_areas: Vec<f64>,

    /// Pairwise relationships
    pub(crate) relationships: PairwiseRelations,
}

/// Pairwise relationships between sets (internal).
#[derive(Clone)]
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
    fn test_inclusive_three_set_decomposition() {
        // Known 3-set case: A=10, B=8, C=6; A∩B=3, A∩C=2, B∩C=1, A∩B∩C=0
        // Expected exclusive decomposition via inclusion-exclusion:
        //   A∩B∩C (exclusive)           = 0
        //   A∩B only = 3 - 0            = 3
        //   A∩C only = 2 - 0            = 2
        //   B∩C only = 1 - 0            = 1
        //   A only   = 10 - 3 - 2 + 0   = 5
        //   B only   = 8  - 3 - 1 + 0   = 4
        //   C only   = 6  - 2 - 1 + 0   = 3
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .set("C", 6.0)
            .intersection(&["A", "B"], 3.0)
            .intersection(&["A", "C"], 2.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.0)
            .input_type(InputType::Inclusive)
            .build()
            .unwrap();

        let g = |names: &[&str]| {
            spec.get_exclusive(&Combination::new(names))
                .expect("exclusive area should be defined")
        };
        assert!((g(&["A"]) - 5.0).abs() < 1e-10);
        assert!((g(&["B"]) - 4.0).abs() < 1e-10);
        assert!((g(&["C"]) - 3.0).abs() < 1e-10);
        assert!((g(&["A", "B"]) - 3.0).abs() < 1e-10);
        assert!((g(&["A", "C"]) - 2.0).abs() < 1e-10);
        assert!((g(&["B", "C"]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inclusive_rejects_negative_disjoint_area() {
        // A∩B is larger than either A or B — impossible set relationships.
        // The exclusive decomposition would yield a negative A-only area.
        let result = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 5.0)
            .intersection(&["A", "B"], 10.0)
            .input_type(InputType::Inclusive)
            .build();

        assert!(
            matches!(result, Err(DiagramError::InvalidValue { .. })),
            "expected InvalidValue for negative-disjoint inclusive input, got {:?}",
            result
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

    #[test]
    fn test_get_inclusive_recovers_implicit_subsets() {
        // 3-way intersection with no explicit pair entries: the lazy
        // get_inclusive must still report the pair-level inclusive areas
        // contributed by the higher-order term (matches what the eager
        // exclusive→inclusive map produced before).
        let spec = DiagramSpecBuilder::new()
            .set("A", 0.0)
            .set("B", 5.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["A", "B", "C"], 0.1)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // A∩C inclusive comes only from the 3-way term.
        assert_eq!(
            spec.get_inclusive(&Combination::new(&["A", "C"])),
            Some(0.1)
        );
        assert_eq!(
            spec.get_inclusive(&Combination::new(&["B", "C"])),
            Some(0.1)
        );
        // Singletons sum every superset that touches them.
        assert!((spec.get_inclusive(&Combination::new(&["A"])).unwrap() - 1.1).abs() < 1e-10);
        assert!((spec.get_inclusive(&Combination::new(&["B"])).unwrap() - 6.1).abs() < 1e-10);
        assert!((spec.get_inclusive(&Combination::new(&["C"])).unwrap() - 0.1).abs() < 1e-10);

        // Combinations with no contributing supersets stay None
        // (matching the "missing = zero" convention).
        assert_eq!(
            spec.get_inclusive(&Combination::new(&["A", "B", "C", "D"])),
            None
        );
    }

    #[test]
    fn test_inclusive_areas_matches_get_inclusive() {
        // The bulk `inclusive_areas()` accessor must agree with
        // pointwise `get_inclusive` on every key it produces.
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .set("C", 6.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "C"], 3.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.5)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let bulk = spec.inclusive_areas();
        for (combo, &area) in bulk.iter() {
            let got = spec
                .get_inclusive(combo)
                .expect("get_inclusive should agree with inclusive_areas keys");
            assert!(
                (got - area).abs() < 1e-10,
                "mismatch for {combo}: bulk={area}, get_inclusive={got}"
            );
        }
    }
}
