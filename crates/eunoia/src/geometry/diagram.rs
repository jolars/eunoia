use crate::geometry::point::Point;
use crate::geometry::shapes::circle::Circle;
use crate::geometry::shapes::Shape;
use crate::spec::Combination;
use std::collections::HashMap;

/// Internal representation of a region as a bit mask.
/// Each bit represents whether a set is part of the region.
pub(crate) type RegionMask = usize;

/// Information about a single intersection point between shapes.
#[derive(Debug, Clone)]
pub struct IntersectionPoint {
    /// The intersection point
    point: Point,
    /// Indices of the two shapes that create this intersection
    parents: (usize, usize),
    /// Indices of all shapes that contain this point
    adopters: Vec<usize>,
}

impl IntersectionPoint {
    /// Creates a new IntersectionPoint.
    pub fn new(point: Point, parents: (usize, usize), adopters: Vec<usize>) -> Self {
        IntersectionPoint {
            point,
            parents,
            adopters,
        }
    }

    /// Returns the coordinates of the intersection point.
    pub fn point(&self) -> &Point {
        &self.point
    }

    /// Returns the indices of the two parent shapes.
    pub fn parents(&self) -> (usize, usize) {
        self.parents
    }

    /// Returns the indices of all shapes that contain this point.
    pub fn adopters(&self) -> &Vec<usize> {
        &self.adopters
    }
}

/// Compute all the exclusive regions and their areas from a set of circles.
pub fn compute_exclusive_regions(circles: &[Circle]) -> HashMap<RegionMask, f64> {
    let n_sets = circles.len();

    let intersections = collect_intersections(circles, n_sets);
    let regions = discover_regions(circles, &intersections, n_sets);

    let mut overlapping_areas = HashMap::new();
    for &mask in &regions {
        let area = compute_region_area(mask, circles, &intersections, n_sets);
        overlapping_areas.insert(mask, area);
    }

    to_exclusive_areas(&overlapping_areas)
}

/// Collect all intersection points between pairs of circles.
pub fn collect_intersections(circles: &[Circle], n_sets: usize) -> Vec<IntersectionPoint> {
    let mut intersections = Vec::new();

    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = circles[i].intersection_points(&circles[j]);
            for point in pts {
                // Start with parent circles - they must contain the intersection point
                let mut adopters = vec![i, j];

                // Add any other circles that contain this point
                (0..n_sets).for_each(|k| {
                    if k != i && k != j && circles[k].contains_point(&point) {
                        adopters.push(k);
                    }
                });

                adopters.sort_unstable();

                intersections.push(IntersectionPoint::new(point, (i, j), adopters));
            }
        }
    }

    intersections
}

/// Discover which regions actually exist in the diagram using sparse detection.
pub fn discover_regions(
    circles: &[Circle],
    intersections: &[IntersectionPoint],
    n_sets: usize,
) -> Vec<RegionMask> {
    let mut regions = std::collections::HashSet::new();

    // 1. Singles always exist
    for i in 0..n_sets {
        regions.insert(1 << i);
    }

    // 2. From intersection points - convert adopters to masks
    for info in intersections {
        let mask = adopters_to_mask(info.adopters());
        regions.insert(mask);
    }

    // 3. From pairwise relationships
    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let has_edge_intersection = intersections
                .iter()
                .any(|info| info.parents() == (i, j) || info.parents() == (j, i));

            if has_edge_intersection {
                // Circles i and j intersect - their pairwise region exists
                regions.insert((1 << i) | (1 << j));
            } else if circles[i].contains(&circles[j]) || circles[j].contains(&circles[i]) {
                // No edge intersection but one contains the other
                regions.insert((1 << i) | (1 << j));
            } else {
                // Check if they have non-zero intersection area
                // (covers case where intersection points exist but are all shared with other circles)
                let int_area = circles[i].intersection_area(&circles[j]);
                if int_area > 1e-10 {
                    regions.insert((1 << i) | (1 << j));
                }
            }
        }
    }

    // 4. Build higher-order combinations from pairwise compatibility
    // Key insight: A∩B∩C can only exist if A∩B, A∩C, and B∩C all exist
    let pairwise_regions: Vec<RegionMask> = regions
        .iter()
        .filter(|&&m| (m as RegionMask).count_ones() == 2)
        .copied()
        .collect();

    // Build compatibility matrix from pairwise regions
    let mut compatible = vec![vec![false; n_sets]; n_sets];
    (0..n_sets).for_each(|i| {
        compatible[i][i] = true;
    });
    for &mask in &pairwise_regions {
        let indices = mask_to_indices(mask, n_sets);
        if indices.len() == 2 {
            compatible[indices[0]][indices[1]] = true;
            compatible[indices[1]][indices[0]] = true;
        }
    }

    // Extend to 3-way, 4-way, etc. combinations
    let mut current_level = pairwise_regions;
    for _level in 3..=n_sets {
        let mut next_level = Vec::new();

        for &base_mask in &current_level {
            let base_indices = mask_to_indices(base_mask, n_sets);

            #[allow(clippy::needless_range_loop)]
            for new_idx in 0..n_sets {
                if (base_mask & (1 << new_idx)) != 0 {
                    continue;
                }

                // Check if new_idx is compatible with ALL in base_mask
                if base_indices
                    .iter()
                    .all(|&existing| compatible[new_idx][existing])
                {
                    let new_mask = base_mask | (1 << new_idx);
                    if regions.insert(new_mask) {
                        next_level.push(new_mask);
                    }
                }
            }
        }

        if next_level.is_empty() {
            break;
        }
        current_level = next_level;
    }

    regions.into_iter().collect()
}

/// Convert adopters vector to a region bit mask.
pub fn adopters_to_mask(adopters: &[usize]) -> RegionMask {
    adopters.iter().fold(0, |mask, &i| mask | (1 << i))
}

/// Compute the area of a region based on its bit mask.
pub fn compute_region_area(
    mask: RegionMask,
    circles: &[Circle],
    intersections: &[IntersectionPoint],
    n_sets: usize,
) -> f64 {
    let circle_count = mask.count_ones();

    match circle_count {
        0 => 0.0,
        1 => {
            // Single circle - just return its area
            let idx = mask.trailing_zeros() as usize;
            circles[idx].area()
        }
        2 => {
            // Two circles - use intersection_area
            let indices = mask_to_indices(mask, n_sets);
            circles[indices[0]].intersection_area(&circles[indices[1]])
        }
        _ => {
            // 3+ circles - need to check if they have intersection points or are nested
            let indices = mask_to_indices(mask, n_sets);

            // Check if all circles are nested (one contains all others)
            let mut all_nested = true;
            let mut smallest_idx = indices[0];
            let mut smallest_radius = circles[indices[0]].radius();

            for &idx in &indices {
                if circles[idx].radius() < smallest_radius {
                    smallest_radius = circles[idx].radius();
                    smallest_idx = idx;
                }
            }

            // Check if all other circles contain the smallest
            for &idx in &indices {
                if idx != smallest_idx && !circles[idx].contains(&circles[smallest_idx]) {
                    all_nested = false;
                    break;
                }
            }

            if all_nested {
                // All circles contain the smallest one, so the intersection is the smallest circle
                return circles[smallest_idx].area();
            }

            // Otherwise, use the polygon-based calculation from intersection points
            // Following eulerr's approach, we need points where:
            // 1. Both parent circles are in the mask
            // 2. ALL circles in the mask contain the point (mask is subset of adopters)
            let region_points: Vec<IntersectionPoint> = intersections
                .iter()
                .filter(|info| {
                    let (p1, p2) = info.parents();
                    let parents_in_mask = (mask & (1 << p1)) != 0 && (mask & (1 << p2)) != 0;

                    let adopter_mask = adopters_to_mask(info.adopters());
                    let mask_subset_of_adopters = (mask & adopter_mask) == mask;

                    parents_in_mask && mask_subset_of_adopters
                })
                .cloned()
                .collect();

            if region_points.is_empty() {
                // No intersection points: either disjoint or one circle is contained in all others
                // Following eulerr's approach: check if the smallest circle's center is inside all others
                let mut smallest_idx = indices[0];
                let mut smallest_area = circles[indices[0]].area();

                for &idx in &indices {
                    let area = circles[idx].area();
                    if area < smallest_area {
                        smallest_area = area;
                        smallest_idx = idx;
                    }
                }

                // Check if the smallest circle's center is inside all other circles
                let smallest_center = circles[smallest_idx].center();
                let all_contain_center = indices.iter().all(|&idx| {
                    idx == smallest_idx || circles[idx].contains_point(smallest_center)
                });

                if all_contain_center {
                    // The smallest circle is contained in all others - return its area
                    smallest_area
                } else {
                    // Circles are disjoint
                    0.0
                }
            } else {
                // Pass the mask so multiple_overlap_areas knows which circles to consider
                crate::geometry::shapes::circle::multiple_overlap_areas_with_mask(
                    circles,
                    &region_points,
                    &indices,
                )
            }
        }
    }
}

/// Convert a region mask to a list of circle indices.
fn mask_to_indices(mask: RegionMask, n_sets: usize) -> Vec<usize> {
    (0..n_sets).filter(|&i| (mask & (1 << i)) != 0).collect()
}

/// Convert overlapping areas to exclusive areas using inclusion-exclusion.
pub fn to_exclusive_areas(
    overlapping_areas: &HashMap<RegionMask, f64>,
) -> HashMap<RegionMask, f64> {
    let mut exclusive = overlapping_areas.clone();

    // Sort masks by bit count in ASCENDING order, then by mask value
    let mut masks: Vec<_> = overlapping_areas.keys().copied().collect();
    masks.sort_by_key(|m| (m.count_ones(), *m));

    // Process in REVERSE order (from largest to smallest bit count)
    // This ensures supersets are processed before subsets
    for i in (0..masks.len()).rev() {
        let mask_i = masks[i];
        let mut to_subtract = 0.0;

        // DEBUG: Track if this is A
        let is_debug = mask_i == 0b000001 && masks.len() > 10;
        if is_debug {
            eprintln!("\n=== Processing A (0b000001) at index {} ===", i);
            eprintln!("A overlapping: {}", overlapping_areas[&mask_i]);
        }

        // Look at masks that were processed BEFORE this one in the reverse iteration
        // Those are indices AFTER i in the sorted array (larger bit counts)
        #[allow(clippy::needless_range_loop)]
        for j in (i + 1)..masks.len() {
            let mask_j = masks[j];
            // If mask_i is a subset of mask_j, subtract mask_j's already-computed exclusive area
            if is_subset(mask_i, mask_j) {
                if is_debug && to_subtract < 20.0 {
                    // Print first few
                    eprintln!("  Subtracting {:#08b}: {}", mask_j, exclusive[&mask_j]);
                }
                to_subtract += exclusive[&mask_j];
            }
        }

        if is_debug {
            eprintln!("Total to subtract: {}", to_subtract);
            eprintln!(
                "A exclusive will be: {}",
                overlapping_areas[&mask_i] - to_subtract
            );
        }

        *exclusive.get_mut(&mask_i).unwrap() -= to_subtract;
    }

    // Clamp to non-negative
    for area in exclusive.values_mut() {
        *area = area.max(0.0);
    }

    exclusive
}

/// Check if mask1 is a subset of mask2.
fn is_subset(mask1: RegionMask, mask2: RegionMask) -> bool {
    (mask1 & mask2) == mask1
}

/// Compute region error: sum of squared differences between fitted and target areas.
pub(crate) fn compute_region_error(
    fitted_areas: &HashMap<RegionMask, f64>,
    target_areas: &HashMap<RegionMask, f64>,
) -> f64 {
    let mut error = 0.0;

    for (&mask, &target) in target_areas {
        let fitted = fitted_areas.get(&mask).copied().unwrap_or(0.0);
        let diff = fitted - target;
        error += diff * diff;
    }

    error
}

/// Convert a Combination to a bit mask.
pub(crate) fn combination_to_mask(combo: &Combination, set_names: &[String]) -> RegionMask {
    let combo_sets = combo.sets();
    let mut mask = 0;

    for (i, set_name) in set_names.iter().enumerate() {
        if combo_sets.contains(set_name) {
            mask |= 1 << i;
        }
    }

    mask
}

/// Compute all exclusive areas from a circle layout (public for WASM)
pub fn compute_exclusive_areas_from_layout(
    circles: &[Circle],
    set_names: &[String],
) -> HashMap<Combination, f64> {
    let n_sets = circles.len();

    let intersections = collect_intersections(circles, n_sets);
    let regions = discover_regions(circles, &intersections, n_sets);

    let mut overlapping_areas = HashMap::new();
    for &mask in &regions {
        let area = compute_region_area(mask, circles, &intersections, n_sets);
        overlapping_areas.insert(mask, area);
    }

    let exclusive_areas = to_exclusive_areas(&overlapping_areas);

    // Convert masks to Combinations
    let mut exclusive_combos = HashMap::new();
    for (mask, area) in exclusive_areas {
        if area > 0.0 {
            // Only include non-zero areas
            let indices = mask_to_indices(mask, n_sets);
            let combo_sets: Vec<&str> = indices.iter().map(|&i| set_names[i].as_str()).collect();

            if !combo_sets.is_empty() {
                let combo = Combination::new(&combo_sets);
                exclusive_combos.insert(combo, area);
            }
        }
    }

    exclusive_combos
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_region_discovery() {
        // Test that we correctly discover regions from circles
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 2.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 2.0);
        let circles = vec![c1, c2, c3];

        let intersections = collect_intersections(&circles, 3);
        let regions = discover_regions(&circles, &intersections, 3);

        // Should have at least: 3 singles + some overlaps
        assert!(
            regions.len() >= 3,
            "Should discover at least 3 single regions"
        );

        // Check that singles exist
        assert!(
            regions.contains(&(1 << 0)),
            "Region for circle 0 should exist"
        );
        assert!(
            regions.contains(&(1 << 1)),
            "Region for circle 1 should exist"
        );
        assert!(
            regions.contains(&(1 << 2)),
            "Region for circle 2 should exist"
        );

        println!("Discovered {} regions", regions.len());
        for &mask in &regions {
            let count = mask.count_ones();
            println!("  Region mask {:#b} ({} circles)", mask, count);
        }
    }

    #[test]
    fn test_compute_region_area_single() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circles = vec![c];
        let intersections = vec![];

        let mask = 1; // First circle only
        let area = compute_region_area(mask, &circles, &intersections, 1);

        let expected = c.area();
        let error = (area - expected).abs() / expected;
        assert!(
            error < 0.001,
            "Single circle area should match: {} vs {}",
            area,
            expected
        );
    }

    #[test]
    fn test_compute_region_area_two_circles() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        let circles = vec![c1, c2];

        let intersections = collect_intersections(&circles, 2);

        let mask = 0b11; // Both circles
        let area = compute_region_area(mask, &circles, &intersections, 2);

        let expected = c1.intersection_area(&c2);
        let error = (area - expected).abs() / expected;
        assert!(
            error < 0.001,
            "Two circle intersection area should match: {} vs {}",
            area,
            expected
        );
    }

    #[test]
    fn test_compute_region_area_three_circles() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 2.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 2.0);
        let circles = vec![c1, c2, c3];

        let intersections = collect_intersections(&circles, 3);

        let mask = 0b111; // All three circles
        let area = compute_region_area(mask, &circles, &intersections, 3);

        // Should be positive and reasonable
        assert!(area > 0.0, "Three circle intersection should exist");
        assert!(
            area < 10.0,
            "Three circle intersection should be smaller than individual circles"
        );

        println!("Three-way intersection area: {}", area);
    }

    #[test]
    fn test_to_exclusive_areas() {
        // Create overlapping areas manually
        let mut overlapping = HashMap::new();
        overlapping.insert(0b01, 10.0); // A only
        overlapping.insert(0b10, 15.0); // B only
        overlapping.insert(0b11, 3.0); // A ∩ B

        let exclusive = to_exclusive_areas(&overlapping);

        // A only (exclusive) = A - (A ∩ B) = 10 - 3 = 7
        // B only (exclusive) = B - (A ∩ B) = 15 - 3 = 12
        // A ∩ B (exclusive) = 3 (unchanged, no supersets)

        assert!(
            (exclusive[&0b01] - 7.0).abs() < 0.001,
            "A exclusive should be 7"
        );
        assert!(
            (exclusive[&0b10] - 12.0).abs() < 0.001,
            "B exclusive should be 12"
        );
        assert!((exclusive[&0b11] - 3.0).abs() < 0.001, "A∩B should be 3");
    }

    #[test]
    fn test_known_area() {
        // From R, eulerr package, in ellipse form. a and b are the semi-axes.
        // phi is the rotation, which we ignore here for circles.
        //              A          B           C
        // h   -0.6093604  0.6006954 -0.08497298
        // k   -0.7656790 -0.7656790  0.61778865
        // a    0.6458488  0.8939289  1.04190949
        // b    0.6458488  0.8939289  1.04190949
        // phi  4.6774535  4.6774535  4.67745352
        let c1 = Circle::new(Point::new(-0.6093604, -0.7656790), 0.6458488);
        let c2 = Circle::new(Point::new(0.6006954, -0.7656790), 0.8939289);
        let c3 = Circle::new(Point::new(-0.08497298, 0.61778865), 1.04190949);
        let circles = vec![c1, c2, c3];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &["A".to_string(), "B".to_string(), "C".to_string()],
        );

        // Here are the expected areas from R eulerr
        //
        // > exact
        // [1] 1.00000870 2.00000280 2.99999966 0.19993832 0.09990269 0.29995982 0.01057351

        let expected_areas = vec![
            (Combination::new(&["A"]), 1.00000870),
            (Combination::new(&["B"]), 2.00000280),
            (Combination::new(&["C"]), 2.99999966),
            (Combination::new(&["A", "B"]), 0.19993832),
            (Combination::new(&["A", "C"]), 0.09990269),
            (Combination::new(&["B", "C"]), 0.29995982),
            (Combination::new(&["A", "B", "C"]), 0.01057351),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = (computed - expected).abs() / expected;
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {}",
                combo.sets(),
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_simple_overlapping() {
        let c1 = Circle::new(Point::new(-1.0000000000, 1.0000000000), 2.0000000000); // A
        let c2 = Circle::new(Point::new(0.0000000000, 0.0000000000), 1.0000000000); // B
        let c3 = Circle::new(Point::new(2.0000000000, 2.0000000000), 2.0000000000); // C
        let circles = vec![c1, c2, c3];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &["A".to_string(), "B".to_string(), "C".to_string()],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 8.7155937547),
            (Combination::new(&["B"]), 0.5823202638),
            (Combination::new(&["C"]), 11.1636848997),
            (Combination::new(&["A", "B"]), 2.4512959193),
            (Combination::new(&["A", "C"]), 1.2947092442),
            (Combination::new(&["B", "C"]), 0.0032047743),
            (Combination::new(&["A", "B", "C"]), 0.1047716962),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {}",
                combo.sets(),
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_well_separated() {
        let c1 = Circle::new(Point::new(0.0000000000, 0.0000000000), 1.0000000000); // A
        let c2 = Circle::new(Point::new(5.0000000000, 0.0000000000), 1.0000000000); // B
        let c3 = Circle::new(Point::new(2.5000000000, 4.0000000000), 1.0000000000); // C
        let circles = vec![c1, c2, c3];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &["A".to_string(), "B".to_string(), "C".to_string()],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), std::f64::consts::PI),
            (Combination::new(&["B"]), std::f64::consts::PI),
            (Combination::new(&["C"]), std::f64::consts::PI),
            (Combination::new(&["A", "B"]), 0.0000000000),
            (Combination::new(&["A", "C"]), 0.0000000000),
            (Combination::new(&["B", "C"]), 0.0000000000),
            (Combination::new(&["A", "B", "C"]), 0.0000000000),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {}",
                combo.sets(),
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_highly_overlapping() {
        let c1 = Circle::new(Point::new(0.0000000000, 0.0000000000), 2.0000000000); // A
        let c2 = Circle::new(Point::new(1.0000000000, 0.0000000000), 2.0000000000); // B
        let c3 = Circle::new(Point::new(0.5000000000, 0.8660000000), 2.0000000000); // C
        let circles = vec![c1, c2, c3];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &["A".to_string(), "B".to_string(), "C".to_string()],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 2.4119668571),
            (Combination::new(&["B"]), 2.4119668571),
            (Combination::new(&["C"]), 2.4118816505),
            (Combination::new(&["A", "B"]), 1.5458816505),
            (Combination::new(&["A", "C"]), 1.5459668571),
            (Combination::new(&["B", "C"]), 1.5459668571),
            (Combination::new(&["A", "B", "C"]), 7.0625552496),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {}",
                combo.sets(),
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_nested_configuration() {
        let c1 = Circle::new(Point::new(0.0000000000, 0.0000000000), 3.0000000000); // A
        let c2 = Circle::new(Point::new(2.0000000000, 0.0000000000), 3.0000000000); // B
        let c3 = Circle::new(Point::new(1.0000000000, 0.0000000000), 0.5000000000); // C
        let circles = vec![c1, c2, c3];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &["A".to_string(), "B".to_string(), "C".to_string()],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 11.7739186197),
            (Combination::new(&["B"]), 11.7739186197),
            (Combination::new(&["C"]), 0.0000000000),
            (Combination::new(&["A", "B"]), 15.7150170992),
            (Combination::new(&["A", "C"]), 0.0000000000),
            (Combination::new(&["B", "C"]), 0.0000000000),
            (
                Combination::new(&["A", "B", "C"]),
                std::f64::consts::FRAC_PI_4,
            ),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {}",
                combo.sets(),
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_original_failing() {
        let c1 = Circle::new(Point::new(-0.6093604213, -0.7656789691), 0.6458487961); // A
        let c2 = Circle::new(Point::new(0.6006953936, -0.7656789691), 0.8939288774); // B
        let c3 = Circle::new(Point::new(-0.0849729815, 0.6177886475), 1.0419094940); // C
        let circles = vec![c1, c2, c3];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &["A".to_string(), "B".to_string(), "C".to_string()],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 1.0000087040),
            (Combination::new(&["B"]), 2.0000028019),
            (Combination::new(&["C"]), 2.9999996572),
            (Combination::new(&["A", "B"]), 0.1999383198),
            (Combination::new(&["A", "C"]), 0.0999026917),
            (Combination::new(&["B", "C"]), 0.2999598242),
            (Combination::new(&["A", "B", "C"]), 0.0105735088),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {}",
                combo.sets(),
                computed,
                expected
            );
        }
    }

    // TODO: 4+ circle area calculations have accuracy issues that need to be investigated.
    // The errors are larger than expected (>10% for some regions), suggesting potential
    // issues with the polygon-based area calculation in multiple_overlap_areas or with
    // the region discovery algorithm for higher-order intersections.
    //
    // For now, these tests are commented out. They can be used to verify improvements
    // to the 4+ circle support.

    #[test]
    fn test_eulerr_comparison_four_circles_square() {
        let c1 = Circle::new(Point::new(0.0000000000, 0.0000000000), 1.5000000000); // A
        let c2 = Circle::new(Point::new(2.0000000000, 0.0000000000), 1.5000000000); // B
        let c3 = Circle::new(Point::new(0.0000000000, 2.0000000000), 1.5000000000); // C
        let c4 = Circle::new(Point::new(2.0000000000, 2.0000000000), 1.5000000000); // D
        let circles = vec![c1, c2, c3, c4];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &[
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 4.0430769603),
            (Combination::new(&["B"]), 4.0430769603),
            (Combination::new(&["C"]), 4.0430769603),
            (Combination::new(&["D"]), 4.0430769603),
            (Combination::new(&["A", "B"]), 1.4336885099),
            (Combination::new(&["A", "C"]), 1.4336885099),
            (Combination::new(&["A", "D"]), 0.0000000000),
            (Combination::new(&["B", "C"]), 0.0000000000),
            (Combination::new(&["B", "D"]), 1.4336885099),
            (Combination::new(&["C", "D"]), 1.4336885099),
            (Combination::new(&["A", "B", "C"]), 0.0430769603),
            (Combination::new(&["A", "B", "D"]), 0.0430769603),
            (Combination::new(&["A", "C", "D"]), 0.0430769603),
            (Combination::new(&["B", "C", "D"]), 0.0430769603),
            (Combination::new(&["A", "B", "C", "D"]), 0.0288986095),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.02,
                "Area for {:?} should match: {} vs {} (error: {})",
                combo.sets(),
                computed,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_four_circles_center() {
        let c1 = Circle::new(Point::new(-2.0000000000, 0.0000000000), 1.5000000000); // A
        let c2 = Circle::new(Point::new(2.0000000000, 0.0000000000), 1.5000000000); // B
        let c3 = Circle::new(Point::new(0.0000000000, 2.0000000000), 1.5000000000); // C
        let c4 = Circle::new(Point::new(0.0000000000, 0.0000000000), 1.0000000000); // D
        let circles = vec![c1, c2, c3, c4];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &[
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 6.4585131745),
            (Combination::new(&["B"]), 6.4585131745),
            (Combination::new(&["C"]), 6.3458908330),
            (Combination::new(&["D"]), 1.6541091670),
            (Combination::new(&["A", "B"]), 0.0000000000),
            (Combination::new(&["A", "C"]), 0.1126223414),
            (Combination::new(&["A", "D"]), 0.4950177660),
            (Combination::new(&["B", "C"]), 0.1126223414),
            (Combination::new(&["B", "D"]), 0.4950177660),
            (Combination::new(&["C", "D"]), 0.4925875772),
            (Combination::new(&["A", "B", "C"]), 0.0000000000),
            (Combination::new(&["A", "B", "D"]), 0.0000000000),
            (Combination::new(&["A", "C", "D"]), 0.0024301887),
            (Combination::new(&["B", "C", "D"]), 0.0024301887),
            (Combination::new(&["A", "B", "C", "D"]), 0.0000000000),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.02,
                "Area for {:?} should match: {} vs {} (error: {})",
                combo.sets(),
                computed,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_five_circles_circular() {
        let c1 = Circle::new(Point::new(2.0000000000, 0.0000000000), 1.5000000000); // A
        let c2 = Circle::new(Point::new(0.6180339887, 1.9021130326), 1.5000000000); // B
        let c3 = Circle::new(Point::new(-1.6180339887, 1.1755705046), 1.5000000000); // C
        let c4 = Circle::new(Point::new(-1.6180339887, -1.1755705046), 1.5000000000); // D
        let c5 = Circle::new(Point::new(0.6180339887, -1.9021130326), 1.5000000000); // E
        let circles = vec![c1, c2, c3, c4, c5];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &[
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
            ],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 5.4180566219),
            (Combination::new(&["B"]), 5.4180566219),
            (Combination::new(&["C"]), 5.4180566219),
            (Combination::new(&["D"]), 5.4180566219),
            (Combination::new(&["E"]), 5.4180566219),
            (Combination::new(&["A", "B"]), 0.8252634243),
            (Combination::new(&["A", "C"]), 0.0000000000),
            (Combination::new(&["A", "D"]), 0.0000000000),
            (Combination::new(&["A", "E"]), 0.8252634243),
            (Combination::new(&["B", "C"]), 0.8252634243),
            (Combination::new(&["B", "D"]), 0.0000000000),
            (Combination::new(&["B", "E"]), 0.0000000000),
            (Combination::new(&["C", "D"]), 0.8252634243),
            (Combination::new(&["C", "E"]), 0.0000000000),
            (Combination::new(&["D", "E"]), 0.8252634243),
            (Combination::new(&["A", "B", "C"]), 0.0000000000),
            (Combination::new(&["A", "B", "D"]), 0.0000000000),
            (Combination::new(&["A", "B", "E"]), 0.0000000000),
            (Combination::new(&["A", "C", "D"]), 0.0000000000),
            (Combination::new(&["A", "C", "E"]), 0.0000000000),
            (Combination::new(&["A", "D", "E"]), 0.0000000000),
            (Combination::new(&["B", "C", "D"]), 0.0000000000),
            (Combination::new(&["B", "C", "E"]), 0.0000000000),
            (Combination::new(&["B", "D", "E"]), 0.0000000000),
            (Combination::new(&["C", "D", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "C", "D"]), 0.0000000000),
            (Combination::new(&["A", "B", "C", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "D", "E"]), 0.0000000000),
            (Combination::new(&["A", "C", "D", "E"]), 0.0000000000),
            (Combination::new(&["B", "C", "D", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "C", "D", "E"]), 0.0000000000),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.02,
                "Area for {:?} should match: {} vs {} (error: {})",
                combo.sets(),
                computed,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_eulerr_comparison_five_circles_cross() {
        let c1 = Circle::new(Point::new(0.0000000000, 0.0000000000), 1.0000000000); // A
        let c2 = Circle::new(Point::new(1.2000000000, 0.0000000000), 0.8000000000); // B
        let c3 = Circle::new(Point::new(-1.2000000000, 0.0000000000), 0.8000000000); // C
        let c4 = Circle::new(Point::new(0.0000000000, 1.2000000000), 0.8000000000); // D
        let c5 = Circle::new(Point::new(0.0000000000, -1.2000000000), 0.8000000000); // E
        let circles = vec![c1, c2, c3, c4, c5];

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &[
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
            ],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 0.9336790656),
            (Combination::new(&["B"]), 1.4586409013),
            (Combination::new(&["C"]), 1.4586409013),
            (Combination::new(&["D"]), 1.4586409013),
            (Combination::new(&["E"]), 1.4586409013),
            (Combination::new(&["A", "B"]), 0.5519783970),
            (Combination::new(&["A", "C"]), 0.5519783970),
            (Combination::new(&["A", "D"]), 0.5519783970),
            (Combination::new(&["A", "E"]), 0.5519783970),
            (Combination::new(&["B", "C"]), 0.0000000000),
            (Combination::new(&["B", "D"]), 0.0000000000),
            (Combination::new(&["B", "E"]), 0.0000000000),
            (Combination::new(&["C", "D"]), 0.0000000000),
            (Combination::new(&["C", "E"]), 0.0000000000),
            (Combination::new(&["D", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "C"]), 0.0000000000),
            (Combination::new(&["A", "B", "D"]), 0.0000000000),
            (Combination::new(&["A", "B", "E"]), 0.0000000000),
            (Combination::new(&["A", "C", "D"]), 0.0000000000),
            (Combination::new(&["A", "C", "E"]), 0.0000000000),
            (Combination::new(&["A", "D", "E"]), 0.0000000000),
            (Combination::new(&["B", "C", "D"]), 0.0000000000),
            (Combination::new(&["B", "C", "E"]), 0.0000000000),
            (Combination::new(&["B", "D", "E"]), 0.0000000000),
            (Combination::new(&["C", "D", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "C", "D"]), 0.0000000000),
            (Combination::new(&["A", "B", "C", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "D", "E"]), 0.0000000000),
            (Combination::new(&["A", "C", "D", "E"]), 0.0000000000),
            (Combination::new(&["B", "C", "D", "E"]), 0.0000000000),
            (Combination::new(&["A", "B", "C", "D", "E"]), 0.0000000000),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.02,
                "Area for {:?} should match: {} vs {} (error: {})",
                combo.sets(),
                computed,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_hexagon_pruning() {
        // Hexagon - not all circles overlap
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(2.0 * angle.cos(), 2.0 * angle.sin()), 1.5))
            .collect();

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        println!("\n=== Hexagon (sparse overlap) ===");
        println!("Discovered: {} / 63 regions", regions.len());
        println!("Pruned: {} unnecessary computations!", 63 - regions.len());

        assert!(regions.len() < 63, "Should have pruned some regions");
    }

    #[test]
    fn test_eulerr_comparison_six_circles_all_overlap() {
        let c1 = Circle::new(Point::new(1.5000000000, 0.0000000000), 1.8000000000); // A
        let c2 = Circle::new(Point::new(0.7500000000, 1.2990381057), 1.8000000000); // B
        let c3 = Circle::new(Point::new(-0.7500000000, 1.2990381057), 1.8000000000); // C
        let c4 = Circle::new(Point::new(-1.5000000000, 0.0000000000), 1.8000000000); // D
        let c5 = Circle::new(Point::new(-0.7500000000, -1.2990381057), 1.8000000000); // E
        let c6 = Circle::new(Point::new(0.7500000000, -1.2990381057), 1.8000000000); // F
        let circles = vec![c1, c2, c3, c4, c5, c6];

        // Debug: compute overlapping for ABCDEF
        let intersections = collect_intersections(&circles, 6);
        eprintln!("Total intersection points: {}", intersections.len());

        // Check how many points qualify for ABCDEF
        let mask = 0b111111;
        let count = intersections
            .iter()
            .filter(|info| {
                let (p1, p2) = info.parents();
                let parents_in_mask = (mask & (1 << p1)) != 0 && (mask & (1 << p2)) != 0;
                let adopter_mask = adopters_to_mask(info.adopters());
                let mask_subset_of_adopters = (mask & adopter_mask) == mask;
                parents_in_mask && mask_subset_of_adopters
            })
            .count();
        eprintln!("Points qualifying for ABCDEF: {}", count);

        let regions = discover_regions(&circles, &intersections, 6);
        let mut overlapping_areas = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 6);
            overlapping_areas.insert(mask, area);
        }
        eprintln!(
            "ABCDEF (0b111111) overlapping: {}",
            overlapping_areas[&0b111111]
        );

        let areas = compute_exclusive_areas_from_layout(
            &circles,
            &[
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
                "F".to_string(),
            ],
        );

        let expected_areas = vec![
            (Combination::new(&["A"]), 1.9828808649),
            (Combination::new(&["B"]), 1.9828808649),
            (Combination::new(&["C"]), 1.9828808649),
            (Combination::new(&["D"]), 1.9828808649),
            (Combination::new(&["E"]), 1.9828808649),
            (Combination::new(&["F"]), 1.9828808649),
            (
                Combination::new(&["A", "B", "C", "D", "E", "F"]),
                0.3062166, // 6-way intersection exclusive area from eulerr
            ),
        ];

        for (combo, expected) in expected_areas {
            let computed = areas.get(&combo).copied().unwrap_or(0.0);
            let error = if expected > 1e-10 {
                (computed - expected).abs() / expected
            } else {
                (computed - expected).abs()
            };
            assert!(
                error < 0.01,
                "Area for {:?} should match: {} vs {} (error: {})",
                combo.sets(),
                computed,
                expected,
                error
            );
        }
    }

    #[test]
    fn compare_abcdef_exact_vs_monte_carlo() {
        let c1 = Circle::new(Point::new(1.5, 0.0), 1.8);
        let c2 = Circle::new(Point::new(0.75, 1.2990381057), 1.8);
        let c3 = Circle::new(Point::new(-0.75, 1.2990381057), 1.8);
        let c4 = Circle::new(Point::new(-1.5, 0.0), 1.8);
        let c5 = Circle::new(Point::new(-0.75, -1.2990381057), 1.8);
        let c6 = Circle::new(Point::new(0.75, -1.2990381057), 1.8);

        // Exact calculation
        let circles = vec![c1, c2, c3, c4, c5, c6];
        let intersections = collect_intersections(&circles, 6);
        let exact = compute_region_area(0b111111, &circles, &intersections, 6);

        // Sample the region directly - count points in all 6 circles
        let bbox_min_x = -1.5 - 1.8;
        let bbox_max_x = 1.5 + 1.8;
        let bbox_min_y = -1.2990381057 - 1.8;
        let bbox_max_y = 1.2990381057 + 1.8;
        let bbox_area = (bbox_max_x - bbox_min_x) * (bbox_max_y - bbox_min_y);

        let n_samples = 100_000; // 100k samples for reasonable accuracy
        let mut in_all = 0;

        for _ in 0..n_samples {
            let x = bbox_min_x + (bbox_max_x - bbox_min_x) * rand::random::<f64>();
            let y = bbox_min_y + (bbox_max_y - bbox_min_y) * rand::random::<f64>();
            let p = Point::new(x, y);

            if circles.iter().all(|c| c.contains_point(&p)) {
                in_all += 1;
            }
        }

        let mc = (in_all as f64 / n_samples as f64) * bbox_area;

        eprintln!("ABCDEF exact: {:.10}", exact);
        eprintln!(
            "ABCDEF Monte Carlo ({}k samples): {:.10}",
            n_samples / 1_000,
            mc
        );
        eprintln!("eulerr expects exclusive: 0.3062166");
        eprintln!("Difference: {:.10}", (exact - mc).abs());
        eprintln!(
            "Relative error: {:.2}%",
            ((exact - mc).abs() / mc.max(exact)) * 100.0
        );

        assert!((exact - mc).abs() < 0.05, "MC and exact should be close");
    }
}
