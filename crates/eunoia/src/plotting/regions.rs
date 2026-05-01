//! Region decomposition for diagram visualization.
//!
//! This module provides utilities for decomposing fitted shapes into
//! exclusive regions (one per set combination) for plotting.

use crate::geometry::diagram::{discover_regions, IntersectionPoint};
use crate::geometry::primitives::Point;
use crate::geometry::shapes::Polygon;
use crate::geometry::traits::{Closed, DiagramShape, Polygonize};
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
    /// algorithm). When a region is disconnected into several polygons, the
    /// anchor with the largest pole-to-boundary distance is chosen — i.e. the
    /// piece with the most breathing room for a label, which is not always
    /// the piece with the largest area (a thin crescent can have large area
    /// but no spot wide enough to fit text).
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
                let best = polys
                    .iter()
                    .filter(|p| p.area() > 0.0)
                    .map(|p| p.pole_of_inaccessibility_with_distance(precision))
                    .max_by(|(_, da), (_, db)| {
                        da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal)
                    })?;
                Some((combo.clone(), best.0))
            })
            .collect()
    }

    /// Computes a label anchor point for every set in `set_names`.
    ///
    /// For each set, this unions every region polygon that contains the set
    /// (i.e. all of the set's exclusive regions) into a single shape, splits
    /// it into connected components, and returns the pole of inaccessibility
    /// of the component with the largest pole-to-boundary distance. This
    /// matches eulerr's `locate_centers` strategy for placing per-set labels:
    /// pick the cluster of regions where the label has the most breathing
    /// room.
    ///
    /// Sets that are absent from every region are omitted from the result.
    ///
    /// # Arguments
    ///
    /// * `set_names` - The full list of set names in the diagram (typically
    ///   `spec.set_names()`).
    /// * `precision` - Polylabel precision, in the same units as the polygon
    ///   coordinates.
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
    /// let set_labels = regions.set_label_points(spec.set_names(), 0.01);
    ///
    /// assert!(set_labels.contains_key("A"));
    /// assert!(set_labels.contains_key("B"));
    /// ```
    pub fn set_label_points(&self, set_names: &[String], precision: f64) -> HashMap<String, Point> {
        let mut result = HashMap::new();

        for name in set_names {
            let mut owned: Vec<Polygon> = self
                .regions
                .iter()
                .filter(|(combo, _)| combo.sets().iter().any(|s| s == name))
                .flat_map(|(_, polys)| polys.iter().cloned())
                .collect();

            if owned.is_empty() {
                continue;
            }

            let mut merged = vec![owned.remove(0)];
            for p in owned {
                merged = polygon_clip_many(&merged, &p, ClipOperation::Union);
                if merged.is_empty() {
                    break;
                }
            }

            let best = merged
                .iter()
                .filter(|p| p.area() > 0.0)
                .map(|p| p.pole_of_inaccessibility_with_distance(precision))
                .max_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((point, _)) = best {
                result.insert(name.clone(), point);
            }
        }

        result
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
/// 2. Discover candidate regions sparsely from the actual fitted geometry (via
///    the shape boundary intersection points), instead of enumerating the full
///    `2^n - 1` power set. Only regions that can geometrically be non-empty
///    are considered.
/// 3. For each candidate region:
///    a. Start with polygons of sets that should be present
///    b. Intersect them together
///    c. Subtract polygons of sets that should NOT be present
/// 4. Only include regions with non-empty polygon results
///
/// **Note**: This function generates regions based on the actual fitted shapes,
/// not the spec — so if the optimizer produces shapes with unexpected overlaps
/// or exclusive regions, they will still be visualized. Unlike a power-set
/// scan, the cost scales with the number of geometrically real regions, which
/// keeps large `n` tractable on sparse layouts.
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

    let n = shapes.len();

    // Sparse region discovery: walk only candidate masks the actual fitted
    // geometry can populate. Avoids the 2^n power-set scan that the previous
    // implementation required and that put a hard practical ceiling on `n`.
    let intersections = collect_intersections_generic(shapes, n);
    let mut region_masks = discover_regions(shapes, &intersections, n);
    region_masks.sort_unstable();

    for mask in region_masks {
        let set_indices_in_combo: Vec<usize> = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();

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

/// Collect pairwise boundary intersection points across arbitrary closed
/// shapes. Uses only the `Closed` trait, so it works for any `DiagramShape`
/// implementor — including future shapes that don't have a hand-rolled
/// `collect_intersections_*` helper.
///
/// The resulting `IntersectionPoint` list is what `discover_regions` consumes
/// to figure out which region masks the geometry can actually populate.
fn collect_intersections_generic<S: Closed>(shapes: &[S], n_sets: usize) -> Vec<IntersectionPoint> {
    let mut intersections = Vec::new();

    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = shapes[i].intersection_points(&shapes[j]);
            for point in pts {
                let mut adopters = vec![i, j];
                for (k, shape) in shapes.iter().enumerate() {
                    if k != i && k != j && shape.contains_point(&point) {
                        adopters.push(k);
                    }
                }
                adopters.sort_unstable();

                intersections.push(IntersectionPoint::new(point, (i, j), adopters));
            }
        }
    }

    intersections
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::spec::{DiagramSpecBuilder, InputType};

    #[test]
    fn test_decompose_disjoint_circles_skips_pairwise() {
        // Two disjoint circles. The previous power-set scanner would still
        // walk the {A,B} mask and only discover emptiness via polygon
        // clipping. The sparse path should skip {A,B} entirely (no boundary
        // intersection points, no containment), keeping cost proportional to
        // the number of geometrically real regions.
        let circles = [
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(10.0, 0.0), 1.0),
        ];
        let set_names = vec!["A".to_string(), "B".to_string()];
        let spec = DiagramSpecBuilder::new()
            .set("A", std::f64::consts::PI)
            .set("B", std::f64::consts::PI)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let regions = decompose_regions(&circles, &set_names, &spec, 64);

        // Exactly the two singleton regions, never the pair.
        assert_eq!(regions.len(), 2, "expected only A-only and B-only regions");
        assert!(regions.get(&Combination::new(&["A"])).is_some());
        assert!(regions.get(&Combination::new(&["B"])).is_some());
        assert!(
            regions.get(&Combination::new(&["A", "B"])).is_none(),
            "disjoint pair should not appear in region polygons"
        );
    }

    #[test]
    fn test_decompose_many_disjoint_circles_scales_sparsely() {
        // 20 disjoint circles in a row. Power-set decomposition would walk
        // 2^20 - 1 ≈ 1M masks; the sparse path walks only the 20 singletons.
        // This test asserts both correctness (only singletons appear) and
        // implicitly that the call returns in well under a second.
        let n = 20;
        let circles: Vec<Circle> = (0..n)
            .map(|i| Circle::new(Point::new(10.0 * i as f64, 0.0), 1.0))
            .collect();
        let set_names: Vec<String> = (0..n).map(|i| format!("S{i}")).collect();

        let mut builder = DiagramSpecBuilder::new();
        for name in &set_names {
            builder = builder.set(name.clone(), std::f64::consts::PI);
        }
        let spec = builder.input_type(InputType::Exclusive).build().unwrap();

        let regions = decompose_regions(&circles, &set_names, &spec, 32);
        assert_eq!(regions.len(), n, "expected one region per disjoint circle");
    }

    #[test]
    fn test_decompose_two_squares() {
        // The plotting path is generic over any `DiagramShape + Polygonize`.
        // Square implements both, so polygon decomposition into per-region
        // polygons should work without any plotting-side changes.
        use crate::geometry::shapes::Square;

        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Square>::new(&spec).seed(42).fit().unwrap();
        let shapes: Vec<Square> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        // n_vertices is ignored by Square::polygonize (always 4 corners).
        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);
        assert!(!regions.is_empty(), "no regions decomposed");
        for (combo, polys) in regions.iter() {
            assert!(!polys.is_empty(), "Region {:?} should have polygons", combo);
        }
    }

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

        // Each label point must sit inside the axis-aligned bounding box of
        // its region's largest polygon.
        for (combo, polys) in regions.iter() {
            let label = labels.get(combo).unwrap();
            let largest = polys
                .iter()
                .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap())
                .unwrap();
            let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
            let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for v in largest.vertices() {
                min_x = min_x.min(v.x());
                min_y = min_y.min(v.y());
                max_x = max_x.max(v.x());
                max_y = max_y.max(v.y());
            }
            assert!(
                label.x() >= min_x - 1e-9
                    && label.x() <= max_x + 1e-9
                    && label.y() >= min_y - 1e-9
                    && label.y() <= max_y + 1e-9,
                "Label for {:?} at ({:.3}, {:.3}) is outside its region's bounding box [{:.3}, {:.3}] x [{:.3}, {:.3}]",
                combo,
                label.x(),
                label.y(),
                min_x,
                max_x,
                min_y,
                max_y
            );
        }
    }

    #[test]
    fn test_set_label_points_two_circles() {
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
        let set_labels = regions.set_label_points(spec.set_names(), 0.01);

        assert_eq!(set_labels.len(), 2);
        for name in ["A", "B"] {
            let label = set_labels.get(name).expect("missing set label");
            // The set label must lie inside the corresponding circle.
            let circle = layout.shape_for_set(name).unwrap();
            let dx = label.x() - circle.center().x();
            let dy = label.y() - circle.center().y();
            let r = circle.radius();
            assert!(
                dx * dx + dy * dy <= r * r + 1e-6,
                "set label for {} at ({:.3}, {:.3}) is outside circle (center=({:.3}, {:.3}), r={:.3})",
                name,
                label.x(),
                label.y(),
                circle.center().x(),
                circle.center().y(),
                r,
            );
        }
    }

    #[test]
    fn test_set_label_points_skips_absent_sets() {
        let regions = RegionPolygons::new();
        let names = vec!["A".to_string()];
        assert!(regions.set_label_points(&names, 0.01).is_empty());
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

        // Fit the layout - this will create a non-zero ellipse for C
        use crate::geometry::shapes::Ellipse;
        let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();

        let shapes: Vec<Ellipse> = spec
            .set_names()
            .iter()
            .map(|name| *layout.shape_for_set(name).unwrap())
            .collect();

        // Decompose regions
        let regions = decompose_regions(&shapes, spec.set_names(), &spec, 64);

        // For this particular configuration, C is fully contained within
        // A&B&C intersection, so there won't be a C-only region. But we
        // should verify that regions involving C are present.
        let abc_combo = crate::spec::Combination::new(&["A", "B", "C"]);
        let abc_polygons = regions.get(&abc_combo);
        assert!(abc_polygons.is_some(), "Should have polygons for A&B&C");

        // Verify total area is reasonable
        let total_area: f64 = regions.areas().values().sum();

        // The total should be close to sum of individual shapes minus overlaps
        assert!(
            total_area > 5.0,
            "Total area should be substantial, got {:.3}",
            total_area
        );
    }
}
