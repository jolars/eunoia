use crate::geometry::point::Point;
use crate::geometry::shapes::circle::Circle;
use crate::geometry::shapes::Shape;
use crate::spec::Combination;
use std::collections::HashMap;

type RegionMask = usize;

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
                for k in 0..n_sets {
                    if k != i && k != j && circles[k].contains_point(&point) {
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
    for i in 0..n_sets {
        compatible[i][i] = true;
    }
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
            // We need points where ALL circles in the mask contain the point
            // (i.e., mask is a subset of adopters), not just exact matches
            let region_points: Vec<IntersectionPoint> = intersections
                .iter()
                .filter(|info| {
                    let adopter_mask = adopters_to_mask(info.adopters());
                    // Check if mask is a subset of adopter_mask
                    (mask & adopter_mask) == mask
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
pub fn compute_region_error(
    fitted_areas: &HashMap<RegionMask, f64>,
    target_areas: &HashMap<Combination, f64>,
    set_names: &[String],
) -> f64 {
    let mut error = 0.0;

    for (combo, &target) in target_areas {
        // Convert combination to mask
        let mask = combination_to_mask(combo, set_names);
        let fitted = fitted_areas.get(&mask).copied().unwrap_or(0.0);
        let diff = fitted - target;
        error += diff * diff;
    }

    error
}

/// Convert a Combination to a bit mask.
fn combination_to_mask(combo: &Combination, set_names: &[String]) -> RegionMask {
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
            (Combination::new(&["A"]), 3.1415926536),
            (Combination::new(&["B"]), 3.1415926536),
            (Combination::new(&["C"]), 3.1415926536),
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
            (Combination::new(&["A", "B", "C"]), 0.7853981634),
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
}
#[cfg(test)]
mod analyze_sparse {
    use super::*;

    #[test]
    fn analyze_why_sparse_fails() {
        // Six circles all overlapping at center
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        let intersections = collect_intersections(&circles, 6);

        println!("\n=== Intersection Analysis ===");
        println!("Total intersection points: {}", intersections.len());

        // Check what masks we get from intersection points
        let mut masks_from_points = std::collections::HashSet::new();
        for info in &intersections {
            let mask = adopters_to_mask(info.adopters());
            masks_from_points.insert(mask);
        }

        println!(
            "Unique masks from intersection points: {}",
            masks_from_points.len()
        );

        // Check some specific combinations
        println!("\n=== Checking specific combinations ===");

        // Check if A-B (0b000011) has intersection points
        let ab_mask = 0b000011;
        let ab_points: Vec<_> = intersections
            .iter()
            .filter(|info| adopters_to_mask(info.adopters()) == ab_mask)
            .collect();
        println!("A-B only points: {}", ab_points.len());

        // Check if A-B have ANY shared points
        let ab_shared: Vec<_> = intersections
            .iter()
            .filter(|info| {
                let adopters = info.adopters();
                adopters.contains(&0) && adopters.contains(&1)
            })
            .collect();
        println!("A-B shared points (any mask): {}", ab_shared.len());

        // Key question: do A and B actually intersect?
        let ab_int_area = circles[0].intersection_area(&circles[1]);
        println!("A-B intersection area: {:.6}", ab_int_area);

        // The problem: when ALL six circles meet at a central region,
        // the A-B intersection points are ALSO in C, D, E, F
        // So there are NO points with mask exactly 0b000011
        // But A-B DO have a pairwise intersection area!

        println!("\n=== The Problem ===");
        println!(
            "A-B have intersection area {:.3}, but no points with mask exactly A-B",
            ab_int_area
        );
        println!("All A-B boundary points are also in other circles (higher-order intersections)");
        println!("Sparse discovery looks for unique points per region, but in dense overlap,");
        println!("many regions don't have unique points - their boundaries are all shared!");
    }
}

#[cfg(test)]
mod test_improved_discovery {
    use super::*;

    #[test]
    fn test_discover_six_overlap() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        println!("\n=== Improved Discovery ===");
        println!("Discovered: {} / 63 regions", regions.len());
        println!(
            "Saved: {} region computations vs full enumeration",
            63 - regions.len()
        );
    }
}

#[cfg(test)]
mod test_hexagon_pruning {
    use super::*;

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
    }
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
            0.0343237063,
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

#[cfg(test)]
mod debug_ie_six {
    use super::*;

    #[test]
    fn debug_ie_six_overlap() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        println!("\n=== Circle A area: {:.6}", circles[0].area());

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        println!("Regions discovered: {}", regions.len());

        let mut overlapping = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 6);
            overlapping.insert(mask, area);
        }

        println!("\n=== Before IE ===");
        println!(
            "A (0b000001): {:.6}",
            overlapping.get(&0b000001).copied().unwrap_or(0.0)
        );
        println!(
            "ABCDEF (0b111111): {:.6}",
            overlapping.get(&0b111111).copied().unwrap_or(0.0)
        );

        let exclusive = to_exclusive_areas(&overlapping);

        println!("\n=== After IE ===");
        println!(
            "A (0b000001): {:.6}",
            exclusive.get(&0b000001).copied().unwrap_or(0.0)
        );
        println!(
            "ABCDEF (0b111111): {:.6}",
            exclusive.get(&0b111111).copied().unwrap_or(0.0)
        );

        println!("\n=== Expected ===");
        println!("A: 1.9828808649");
        println!("ABCDEF: 0.0343237063");
    }
}

#[cfg(test)]
mod check_zero_regions {
    use super::*;

    #[test]
    fn check_zero_area_regions() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        let mut zero_count = 0;
        let mut nonzero_count = 0;

        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 6);
            if area < 1e-10 {
                zero_count += 1;
            } else {
                nonzero_count += 1;
            }
        }

        println!("\n=== Region areas ===");
        println!("Zero area: {}", zero_count);
        println!("Non-zero area: {}", nonzero_count);
        println!("Total: {}", regions.len());
        println!(
            "\nWe're discovering {} regions that don't actually exist!",
            zero_count
        );
    }
}

#[cfg(test)]
mod trace_ie {
    use super::*;

    #[test]
    fn trace_inclusion_exclusion() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        let mut overlapping = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 6);
            overlapping.insert(mask, area);
        }

        // Manually trace IE for mask 0b000001 (A)
        let a_mask = 0b000001;
        let a_overlapping = overlapping[&a_mask];

        println!("\n=== Tracing IE for A (0b000001) ===");
        println!("Initial overlapping area: {:.6}", a_overlapping);

        let mut a_exclusive = a_overlapping;
        let mut subtracted = Vec::new();

        for (&mask, &area) in &overlapping {
            if mask != a_mask && is_subset(a_mask, mask) && mask.count_ones() > 1 {
                println!("Subtracting {:#08b} (area {:.6})", mask, area);
                a_exclusive -= area;
                subtracted.push((mask, area));
            }
        }

        println!(
            "\nTotal subtracted: {:.6}",
            subtracted.iter().map(|(_, a)| a).sum::<f64>()
        );
        println!("Final exclusive area: {:.6}", a_exclusive);
        println!("Expected: 1.9828808649");
    }
}

#[cfg(test)]
mod monte_carlo_verification {
    use super::*;

    #[test]
    fn verify_six_overlap_monte_carlo() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        // Use Monte Carlo to estimate exclusive areas
        let n_samples = 1_000_000;

        println!(
            "\n=== Monte Carlo Verification ({}M samples) ===",
            n_samples / 1_000_000
        );

        // Find bounding box
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for circle in &circles {
            let c = circle.center();
            let r = circle.radius();
            min_x = min_x.min(c.x() - r);
            max_x = max_x.max(c.x() + r);
            min_y = min_y.min(c.y() - r);
            max_y = max_y.max(c.y() + r);
        }

        let bbox_area = (max_x - min_x) * (max_y - min_y);

        // Count points in each exclusive region
        let mut region_counts = std::collections::HashMap::new();

        for _ in 0..n_samples {
            let x = min_x + rand::random::<f64>() * (max_x - min_x);
            let y = min_y + rand::random::<f64>() * (max_y - min_y);
            let pt = Point::new(x, y);

            // Check which circles contain this point
            let mut mask = 0;
            for (i, circle) in circles.iter().enumerate() {
                if circle.contains_point(&pt) {
                    mask |= 1 << i;
                }
            }

            if mask != 0 {
                *region_counts.entry(mask).or_insert(0) += 1;
            }
        }

        println!("\n=== Monte Carlo Results ===");
        println!(
            "A (0b000001): {:.6}",
            region_counts.get(&0b000001).copied().unwrap_or(0) as f64 / n_samples as f64
                * bbox_area
        );
        println!(
            "ABCDEF (0b111111): {:.6}",
            region_counts.get(&0b111111).copied().unwrap_or(0) as f64 / n_samples as f64
                * bbox_area
        );

        println!("\n=== Our Implementation ===");
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
        println!(
            "A: {:.6}",
            areas.get(&Combination::new(&["A"])).copied().unwrap_or(0.0)
        );
        println!(
            "ABCDEF: {:.6}",
            areas
                .get(&Combination::new(&["A", "B", "C", "D", "E", "F"]))
                .copied()
                .unwrap_or(0.0)
        );

        println!("\n=== Expected from eulerr ===");
        println!("A: 1.9828808649");
        println!("ABCDEF: 0.0343237063");
    }
}

#[cfg(test)]
mod trace_ie_order {
    use super::*;

    #[test]
    fn trace_ie_correct_order() {
        // Simple 3-circle case to understand the algorithm
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 1.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 1.0);
        let circles = vec![c1, c2, c3];

        let intersections = collect_intersections(&circles, 3);
        let regions = discover_regions(&circles, &intersections, 3);

        let mut overlapping = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 3);
            overlapping.insert(mask, area);
        }

        println!("\n=== Overlapping areas ===");
        println!(
            "A (0b001): {:.6}",
            overlapping.get(&0b001).copied().unwrap_or(0.0)
        );
        println!(
            "B (0b010): {:.6}",
            overlapping.get(&0b010).copied().unwrap_or(0.0)
        );
        println!(
            "C (0b100): {:.6}",
            overlapping.get(&0b100).copied().unwrap_or(0.0)
        );
        println!(
            "AB (0b011): {:.6}",
            overlapping.get(&0b011).copied().unwrap_or(0.0)
        );
        println!(
            "AC (0b101): {:.6}",
            overlapping.get(&0b101).copied().unwrap_or(0.0)
        );
        println!(
            "BC (0b110): {:.6}",
            overlapping.get(&0b110).copied().unwrap_or(0.0)
        );
        println!(
            "ABC (0b111): {:.6}",
            overlapping.get(&0b111).copied().unwrap_or(0.0)
        );

        println!("\n=== IE should process in this order ===");
        println!("1. ABC (0b111) - largest, no supersets, stays as is");
        println!("2. AB (0b011) - subtract ABC");
        println!("3. AC (0b101) - subtract ABC");
        println!("4. BC (0b110) - subtract ABC");
        println!("5. A (0b001) - subtract AB, AC, ABC");
        println!("6. B (0b010) - subtract AB, BC, ABC");
        println!("7. C (0b100) - subtract AC, BC, ABC");

        println!("\n=== Manual correct IE for A ===");
        let a_over = overlapping[&0b001];
        let ab_over = overlapping[&0b011];
        let ac_over = overlapping[&0b101];
        let abc_over = overlapping[&0b111];

        // ABC exclusive = ABC overlapping (no supersets)
        let abc_excl = abc_over;

        // AB exclusive = AB overlapping - ABC exclusive
        let ab_excl = ab_over - abc_excl;

        // AC exclusive = AC overlapping - ABC exclusive
        let ac_excl = ac_over - abc_excl;

        // A exclusive = A overlapping - AB exclusive - AC exclusive - ABC exclusive
        let a_excl = a_over - ab_excl - ac_excl - abc_excl;

        println!("A exclusive should be: {:.6}", a_excl);
    }
}

#[cfg(test)]
mod monte_carlo_check {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn compare_abcdef_with_monte_carlo() {
        let c1 = Circle::new(Point::new(1.5, 0.0), 1.8);
        let c2 = Circle::new(Point::new(0.75, 1.2990381057), 1.8);
        let c3 = Circle::new(Point::new(-0.75, 1.2990381057), 1.8);
        let c4 = Circle::new(Point::new(-1.5, 0.0), 1.8);
        let c5 = Circle::new(Point::new(-0.75, -1.2990381057), 1.8);
        let c6 = Circle::new(Point::new(0.75, -1.2990381057), 1.8);

        // Monte Carlo estimate
        let all_circles = vec![
            c1.clone(),
            c2.clone(),
            c3.clone(),
            c4.clone(),
            c5.clone(),
            c6.clone(),
        ];
        let mut rng = StdRng::seed_from_u64(42);

        // Use the public function from overlaps module
        let mc_area = crate::geometry::operations::overlaps::compute_overlaps_circles(&all_circles);

        // Wait, that's exact too. Let me manually do MC...
        let circles = vec![c1, c2, c3, c4, c5, c6];
        let intersections = collect_intersections(&circles, 6);
        let exact_area = compute_region_area(0b111111, &circles, &intersections, 6);

        eprintln!("Alternative exact ABCDEF overlapping: {:.10}", mc_area);
        eprintln!("Our exact ABCDEF overlapping: {:.10}", exact_area);
        eprintln!("eulerr expects ABCDEF exclusive: 0.0343237063");
    }
}

#[cfg(test)]
mod check_our_ie {
    use super::*;

    #[test]
    fn check_our_ie_algorithm() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 1.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 1.0);
        let circles = vec![c1, c2, c3];

        let intersections = collect_intersections(&circles, 3);
        let regions = discover_regions(&circles, &intersections, 3);

        let mut overlapping = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 3);
            overlapping.insert(mask, area);
        }

        let exclusive = to_exclusive_areas(&overlapping);

        println!("\n=== Our implementation ===");
        println!(
            "A exclusive: {:.6}",
            exclusive.get(&0b001).copied().unwrap_or(0.0)
        );
        println!("Expected: 2.319140");
    }
}

#[cfg(test)]
mod trace_six_ie_detail {
    use super::*;

    #[test]
    fn trace_six_ie_in_detail() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        let mut overlapping = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 6);
            overlapping.insert(mask, area);
        }

        // Sort like our IE does
        let mut masks: Vec<_> = overlapping.keys().copied().collect();
        masks.sort_by_key(|m| (m.count_ones(), *m));

        println!("\n=== Sorted masks (first 10) ===");
        for i in 0..10.min(masks.len()) {
            println!("{}: {:#08b} ({} bits)", i, masks[i], masks[i].count_ones());
        }

        println!("\n=== Sorted masks (last 10) ===");
        for i in (masks.len() - 10).max(0)..masks.len() {
            println!("{}: {:#08b} ({} bits)", i, masks[i], masks[i].count_ones());
        }

        // Find A and ABCDEF
        let a_idx = masks.iter().position(|&m| m == 0b000001).unwrap();
        let abcdef_idx = masks.iter().position(|&m| m == 0b111111).unwrap();

        println!("\n=== Key positions ===");
        println!("A (0b000001) at index: {}", a_idx);
        println!("ABCDEF (0b111111) at index: {}", abcdef_idx);
        println!("\nProcessing order (reverse): {} -> 0", masks.len() - 1);
        println!(
            "So ABCDEF processed at position: {}",
            masks.len() - 1 - abcdef_idx
        );
        println!("And A processed at position: {}", masks.len() - 1 - a_idx);
    }
}

#[cfg(test)]
mod trace_a_subtractions {
    use super::*;

    #[test]
    fn trace_what_a_subtracts() {
        let angles: Vec<f64> = (0..6)
            .map(|i| i as f64 * std::f64::consts::PI / 3.0)
            .collect();
        let circles: Vec<Circle> = angles
            .iter()
            .map(|&angle| Circle::new(Point::new(1.5 * angle.cos(), 1.5 * angle.sin()), 1.8))
            .collect();

        let intersections = collect_intersections(&circles, 6);
        let regions = discover_regions(&circles, &intersections, 6);

        let mut overlapping = std::collections::HashMap::new();
        for &mask in &regions {
            let area = compute_region_area(mask, &circles, &intersections, 6);
            overlapping.insert(mask, area);
        }

        // Manually do IE for A
        let mut exclusive = overlapping.clone();
        let mut masks: Vec<_> = overlapping.keys().copied().collect();
        masks.sort_by_key(|m| (m.count_ones(), *m));

        let a_mask = 0b000001;
        let a_idx = masks.iter().position(|&m| m == a_mask).unwrap();

        println!("\n=== When processing A (index {}) ===", a_idx);
        println!("A overlapping: {:.6}", overlapping[&a_mask]);

        let mut to_subtract = 0.0;
        let mut count = 0;

        for j in (a_idx + 1)..masks.len() {
            let mask_j = masks[j];
            if is_subset(a_mask, mask_j) {
                println!("  Subtracting {:#08b}: {:.6}", mask_j, exclusive[&mask_j]);
                to_subtract += exclusive[&mask_j];
                count += 1;
                if count >= 10 {
                    println!("  ... and {} more", masks.len() - j - 1);
                    break;
                }
            }
        }

        println!("\nTotal to subtract: {:.6}", to_subtract);
        println!(
            "A exclusive would be: {:.6}",
            overlapping[&a_mask] - to_subtract
        );
        println!("Expected: ~1.983");
    }
}
