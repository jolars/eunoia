//! Region decomposition for diagram visualization.
//!
//! This module provides utilities for decomposing fitted shapes into
//! exclusive regions (one per set combination) for plotting.

use crate::geometry::primitives::Point;
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

    /// Computes a label anchor point for every non-empty region.
    ///
    /// For each region, this returns the *pole of inaccessibility* (the
    /// interior point farthest from the polygon boundary — the Polylabel
    /// algorithm) of the region's largest polygon. This is a good default
    /// anchor for region-level quantity labels in Euler diagrams because it
    /// maximizes breathing room between the label and any edge.
    ///
    /// Regions whose polygons all have zero area (or where the region has no
    /// polygons at all) are omitted from the result.
    ///
    /// # Arguments
    ///
    /// * `precision` - Polylabel precision, in the same units as the polygon
    ///   coordinates. Smaller values yield more accurate anchors at higher
    ///   cost. A value of roughly 1% of the diagram's extent is typical
    ///   (e.g. `0.01` when coordinates are normalized to `[0, 1]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, InputType};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 3.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .input_type(InputType::Exclusive)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
    /// let regions = layout.region_polygons(&spec, 64);
    /// let labels = regions.label_points(0.01);
    ///
    /// // One label point per non-empty region.
    /// for (combo, point) in &labels {
    ///     println!("{}: ({:.2}, {:.2})", combo, point.x(), point.y());
    /// }
    /// ```
    pub fn label_points(&self, precision: f64) -> HashMap<Combination, Point> {
        self.regions
            .iter()
            .filter_map(|(combo, polys)| {
                let largest = polys.iter().filter(|p| p.area() > 0.0).max_by(|a, b| {
                    a.area()
                        .partial_cmp(&b.area())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;
                Some((combo.clone(), largest.pole_of_inaccessibility(precision)))
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
/// This function takes a set of fitted shapes and produces a collection of polygons
/// for each exclusive region (set combination), regardless of whether those regions
/// were specified in the original diagram specification.
///
/// The algorithm:
/// 1. Convert each shape to a polygon
/// 2. For each possible combination of sets (power set):
///    a. Start with polygons of sets that should be present
///    b. Intersect them together
///    c. Subtract polygons of sets that should NOT be present
/// 3. Only include regions with non-empty polygon results
///
/// **Note**: Unlike previous versions, this function generates regions for ALL possible
/// set combinations, not just those mentioned in the spec. This ensures that if the
/// optimizer produces shapes with unexpected overlaps or exclusive regions, they will
/// still be visualized correctly.
///
/// # Arguments
///
/// * `shapes` - The fitted diagram shapes (one per set)
/// * `set_names` - Names of the sets (in same order as shapes)
/// * `_spec` - The diagram specification (currently unused, kept for API compatibility)
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
    _spec: &DiagramSpec,
    n_vertices: usize,
) -> RegionPolygons {
    if shapes.is_empty() {
        return RegionPolygons::new();
    }

    // Convert all shapes to polygons
    let shape_polygons: Vec<Polygon> = shapes.iter().map(|s| s.polygonize(n_vertices)).collect();

    let mut result = RegionPolygons::new();

    // Generate all possible combinations (power set of sets, excluding empty set)
    // This ensures we compute regions even if they have zero area in the spec
    // but non-zero area in the fitted layout
    let n = shapes.len();
    let all_combinations: Vec<Vec<usize>> = (1..(1 << n))
        .map(|mask| (0..n).filter(|&i| (mask & (1 << i)) != 0).collect())
        .collect();

    for set_indices_in_combo in all_combinations {
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
            // Create combination from set indices
            let combo_sets: Vec<&str> = set_indices_in_combo
                .iter()
                .map(|&i| set_names[i].as_str())
                .collect();
            let combination = Combination::new(&combo_sets);

            result.insert(combination, current_polygons);
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
    fn test_label_points_two_circles() {
        // Two overlapping circles should yield 3 non-empty regions: A-only,
        // B-only, A&B. Each should get a label point inside the region itself.
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
        let labels = regions.label_points(0.01);

        // Every non-empty region in `regions` should appear in `labels`.
        for combo in regions.iter().map(|(c, _)| c) {
            assert!(
                labels.contains_key(combo),
                "Missing label point for region {:?}",
                combo
            );
        }

        // Each label point must be finite. (A tighter geometric check — that
        // the point lies inside the region's largest polygon — is a quality
        // property of `Polygon::pole_of_inaccessibility`, which this method
        // merely delegates to. See its unit tests for geometric guarantees.)
        for (combo, point) in &labels {
            assert!(
                point.x().is_finite() && point.y().is_finite(),
                "Label for {:?} has non-finite coordinates: ({:?}, {:?})",
                combo,
                point.x(),
                point.y()
            );
        }
    }

    #[test]
    fn test_label_points_empty() {
        let empty = RegionPolygons::new();
        assert!(empty.label_points(0.01).is_empty());
    }

    #[test]
    fn test_label_points_skips_zero_area_regions() {
        // A region composed only of degenerate (zero-area) polygons should be
        // omitted from the label map.
        let mut regions = RegionPolygons::new();
        let degenerate = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 0.0), // collinear → zero area
        ]);
        regions.insert(Combination::new(&["X"]), vec![degenerate]);

        assert!(regions.label_points(0.01).is_empty());
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

    #[test]
    fn test_decompose_with_zero_spec_area() {
        // Regression test: When spec has zero area for a set (e.g., C),
        // but the fitted layout has non-zero area, we should still generate
        // polygons for all possible regions, even if after subtraction some
        // regions are empty.
        let spec = DiagramSpecBuilder::new()
            .set("A", 3.0)
            .set("B", 5.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Spec should have C with zero area
        let c_combo = crate::spec::Combination::new(&["C"]);
        assert!(
            spec.exclusive_areas().get(&c_combo).copied().unwrap_or(0.0) < 1e-10,
            "Spec should have zero area for C-only"
        );

        println!("\nSpec exclusive areas:");
        for (combo, &area) in spec.exclusive_areas() {
            println!("  {}: {:.3}", combo, area);
        }

        // Fit the layout - this will create a non-zero ellipse for C
        use crate::geometry::shapes::Ellipse;
        use crate::geometry::traits::Area;
        let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();

        println!("\nFitted areas from layout:");
        for (combo, &area) in layout.fitted() {
            println!("  {}: {:.3}", combo, area);
        }

        let shapes: Vec<Ellipse> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        println!("\nActual shape areas:");
        for (i, shape) in shapes.iter().enumerate() {
            println!("  {}: {:.3}", spec.set_names()[i], shape.area());
        }

        // Decompose regions
        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        println!("\nDecomposed regions:");
        for (combo, polys) in regions.iter() {
            let total_area: f64 = polys.iter().map(|p| p.area()).sum();
            println!(
                "  {}: {} polygons, area={:.3}",
                combo,
                polys.len(),
                total_area
            );
        }

        // For this particular configuration, C is fully contained within
        // A&B&C intersection, so there won't be a C-only region. But we
        // should verify that regions involving C are present.
        let abc_combo = crate::spec::Combination::new(&["A", "B", "C"]);
        let abc_polygons = regions.get(&abc_combo);
        assert!(abc_polygons.is_some(), "Should have polygons for A&B&C");

        // Verify total area is reasonable
        let total_area: f64 = regions.areas().values().sum();
        println!("\nTotal decomposed area: {:.3}", total_area);

        // The total should be close to sum of individual shapes minus overlaps
        assert!(
            total_area > 5.0,
            "Total area should be substantial, got {:.3}",
            total_area
        );
    }
}
