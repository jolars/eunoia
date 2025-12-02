//! Region decomposition for diagram visualization.
//!
//! This module provides utilities for decomposing fitted shapes into
//! exclusive regions (one per set combination) for plotting.

use crate::geometry::shapes::Polygon;
use crate::geometry::traits::{DiagramShape, Polygonize};
use crate::plotting::clip::{polygon_clip_many, ClipOperation};
use crate::spec::{Combination, DiagramSpec};
use std::collections::HashMap;

/// Collection of polygons for each exclusive region in a diagram.
///
/// Each key is a combination of set names, and the value is a list of
/// polygons that together represent that exclusive region. Multiple polygons
/// can occur when a region is disconnected.
#[derive(Debug, Clone)]
pub struct RegionPolygons {
    regions: HashMap<Combination, Vec<Polygon>>,
}

impl RegionPolygons {
    /// Creates a new empty RegionPolygons collection.
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
        }
    }

    /// Adds polygons for a given region.
    pub fn insert(&mut self, combination: Combination, polygons: Vec<Polygon>) {
        self.regions.insert(combination, polygons);
    }

    /// Gets polygons for a given region.
    pub fn get(&self, combination: &Combination) -> Option<&Vec<Polygon>> {
        self.regions.get(combination)
    }

    /// Returns an iterator over all regions and their polygons.
    pub fn iter(&self) -> impl Iterator<Item = (&Combination, &Vec<Polygon>)> {
        self.regions.iter()
    }

    /// Returns the number of regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Returns true if there are no regions.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Computes the total area for each region.
    pub fn areas(&self) -> HashMap<Combination, f64> {
        self.regions
            .iter()
            .map(|(combo, polys)| {
                let area = polys.iter().map(|p| p.area()).sum();
                (combo.clone(), area)
            })
            .collect()
    }
}

impl Default for RegionPolygons {
    fn default() -> Self {
        Self::new()
    }
}

/// Decomposes fitted shapes into exclusive region polygons.
///
/// This function takes a set of fitted shapes and the diagram specification,
/// and produces a collection of polygons for each exclusive region (set combination).
///
/// The algorithm:
/// 1. Convert each shape to a polygon
/// 2. For each exclusive region in the spec:
///    a. Start with polygons of sets that should be present
///    b. Intersect them together
///    c. Subtract polygons of sets that should NOT be present
///
/// # Arguments
///
/// * `shapes` - The fitted diagram shapes (one per set)
/// * `set_names` - Names of the sets (in same order as shapes)
/// * `spec` - The diagram specification
/// * `n_vertices` - Number of vertices to use when converting shapes to polygons
///
/// # Examples
///
/// ```
/// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
/// use eunoia::geometry::shapes::Circle;
/// use eunoia::plotting::decompose_regions;
///
/// let spec = DiagramSpecBuilder::new()
///     .set("A", 5.0)
///     .set("B", 3.0)
///     .intersection(&["A", "B"], 1.0)
///     .input_type(InputType::Exclusive)
///     .build()
///     .unwrap();
///
/// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
/// let shapes: Vec<_> = spec.set_names().iter()
///     .map(|name| *layout.shape_for_set(name).unwrap())
///     .collect();
///
/// let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);
/// ```
pub fn decompose_regions<S: DiagramShape + Polygonize>(
    shapes: &[S],
    set_names: &[String],
    spec: &DiagramSpec,
    n_vertices: usize,
) -> RegionPolygons {
    if shapes.is_empty() {
        return RegionPolygons::new();
    }

    // Convert all shapes to polygons
    let shape_polygons: Vec<Polygon> = shapes.iter().map(|s| s.polygonize(n_vertices)).collect();

    // Create a map from set name to polygon index
    let set_to_idx: HashMap<&str, usize> = set_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let mut result = RegionPolygons::new();

    // Iterate through exclusive areas (combinations)
    for (combination, &area) in spec.exclusive_areas() {
        // Skip regions with effectively zero area
        if area < 1e-10 {
            continue;
        }

        let set_indices_in_combo: Vec<usize> = combination
            .sets()
            .iter()
            .filter_map(|set_name| set_to_idx.get(set_name.as_str()).copied())
            .collect();

        if set_indices_in_combo.is_empty() {
            continue;
        }

        // Start with the first shape in the combination
        let mut current_polygons = vec![shape_polygons[set_indices_in_combo[0]].clone()];

        // Intersect with remaining shapes that should be present
        for &idx in &set_indices_in_combo[1..] {
            current_polygons = polygon_clip_many(
                &current_polygons,
                &shape_polygons[idx],
                ClipOperation::Intersection,
            );

            if current_polygons.is_empty() {
                break;
            }
        }

        // Skip if intersection is empty
        if current_polygons.is_empty() {
            continue;
        }

        // Subtract all shapes that should NOT be present
        for (idx, _) in shapes.iter().enumerate() {
            if !set_indices_in_combo.contains(&idx) {
                current_polygons = polygon_clip_many(
                    &current_polygons,
                    &shape_polygons[idx],
                    ClipOperation::Difference,
                );

                if current_polygons.is_empty() {
                    break;
                }
            }
        }

        // Add to result if non-empty
        if !current_polygons.is_empty() {
            result.insert(combination.clone(), current_polygons);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::spec::{DiagramSpecBuilder, InputType};

    #[test]
    fn test_decompose_two_circles() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        // Should have regions for A-only, B-only, and A&B
        assert!(regions.len() >= 2); // At least A-only and B-only

        // Check that we have some polygons
        for (combo, polys) in regions.iter() {
            assert!(!polys.is_empty(), "Region {:?} should have polygons", combo);
        }
    }

    #[test]
    fn test_decompose_three_circles() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .set("C", 4.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["A", "C"], 1.0)
            .intersection(&["A", "B", "C"], 0.5)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(123).fit().unwrap();

        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        // Should have multiple regions
        assert!(regions.len() >= 3);
    }

    #[test]
    fn test_region_areas() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 3.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        let shapes: Vec<Circle> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 128);
        let areas = regions.areas();

        // Total area should be close to sum of fitted values
        let total_area: f64 = areas.values().sum();
        let expected_total: f64 = spec.exclusive_areas().values().sum();

        // Allow some tolerance due to polygonization
        assert!(
            (total_area - expected_total).abs() < 0.5,
            "Total area {:.3} should be close to expected {:.3}",
            total_area,
            expected_total
        );
    }
}
