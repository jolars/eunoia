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

    // 3. From containment (pairs with no edge intersection)
    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let has_edge_intersection = intersections
                .iter()
                .any(|info| info.parents() == (i, j) || info.parents() == (j, i));

            if !has_edge_intersection
                && (circles[i].contains(&circles[j]) || circles[j].contains(&circles[i]))
            {
                regions.insert((1 << i) | (1 << j));
            }
        }
    }

    // 4. Higher-order intersections from containment
    // For any circle, check which other circles contain it (or it contains)
    // If circle i is contained in multiple circles, their intersection exists
    for i in 0..n_sets {
        let mut containing_circles = vec![i]; // Start with itself

        for j in 0..n_sets {
            if i != j && circles[j].contains(&circles[i]) {
                containing_circles.push(j);
            }
        }

        // If this circle is contained in others, their intersection exists
        if containing_circles.len() > 1 {
            let mask = containing_circles
                .iter()
                .fold(0, |acc, &idx| acc | (1 << idx));
            if mask != 0 {
                // Don't add empty set
                regions.insert(mask);
            }
        }
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
            let region_points: Vec<IntersectionPoint> = intersections
                .iter()
                .filter(|info| adopters_to_mask(info.adopters()) == mask)
                .cloned()
                .collect();

            if region_points.is_empty() {
                0.0 // No geometry
            } else {
                crate::geometry::shapes::circle::multiple_overlap_areas(circles, &region_points)
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

    // Sort masks by bit count (process larger sets first)
    let mut masks: Vec<_> = overlapping_areas.keys().copied().collect();
    masks.sort_by_key(|m| std::cmp::Reverse(m.count_ones()));

    // For each region, subtract all its proper supersets
    for &mask_i in &masks {
        for &mask_j in &masks {
            // mask_j is a proper superset of mask_i if:
            // 1. mask_i's bits are all in mask_j (mask_i is subset of mask_j)
            // 2. mask_j has more bits than mask_i
            if mask_i != mask_j
                && is_subset(mask_i, mask_j)
                && mask_j.count_ones() > mask_i.count_ones()
            {
                *exclusive.get_mut(&mask_i).unwrap() -= exclusive[&mask_j];
            }
        }
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
}
