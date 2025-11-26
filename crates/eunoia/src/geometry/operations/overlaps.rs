use crate::geometry::diagram::IntersectionPoint;
use crate::geometry::point::Point;
use crate::geometry::shapes::circle::Circle;
use crate::geometry::shapes::Shape;
use rand::Rng;
use std::collections::HashMap;

pub enum OverlapMethod {
    MonteCarlo,
    Exact,
}

pub fn compute_overlaps<S: Shape>(
    shapes: &[S],
    method: OverlapMethod,
    rng: &mut dyn rand::RngCore,
) -> f64 {
    match method {
        OverlapMethod::MonteCarlo => monte_carlo_overlap(shapes, 100_000, rng),
        OverlapMethod::Exact => {
            // Exact computation requires shape-specific implementations
            // For generic shapes, we fall back to Monte Carlo
            // Use compute_overlaps_circles() for exact circle computation
            monte_carlo_overlap(shapes, 100_000, rng)
        }
    }
}

/// Compute exact overlap areas for circles using geometric calculation.
///
/// This uses the `multiple_overlap_areas` function from the circle module
/// which computes exact areas using polygon + circular segment calculations.
/// For 2 circles, it falls back to the analytical intersection_area method.
///
/// # Arguments
///
/// * `circles` - The circles to compute overlaps for
///
/// # Returns
///
/// Exact intersection area where ALL circles overlap
pub fn compute_overlaps_circles(circles: &[Circle]) -> f64 {
    let n_sets = circles.len();

    if n_sets == 1 {
        return circles[0].area();
    }

    if n_sets == 2 {
        return circles[0].intersection_area(&circles[1]);
    }

    // For 3+ circles: collect intersection points and use polygon method
    let mut intersections: Vec<IntersectionPoint> = Vec::new();

    for i in 0..n_sets {
        for j in (i + 1)..n_sets {
            let pts = circles[i].intersection_points(&circles[j]);
            for point in pts {
                let adopters = (0..n_sets)
                    .filter(|&k| circles[k].contains_point(&point))
                    .collect();

                intersections.push(IntersectionPoint::new(point, (i, j), adopters));
            }
        }
    }

    // Use the exact circle overlap calculation for 3+ circles
    crate::geometry::shapes::circle::multiple_overlap_areas(circles, &intersections)
}

/// Compute overlap areas using Monte Carlo integration.
///
/// # Algorithm
///
/// 1. Filter intersection points to only those in ALL shapes
/// 2. Compute tight bounding box around the intersection region
/// 3. Generate N random points uniformly in the tight bounding box
/// 4. Count points that are in ALL shapes (intersection)
/// 5. Estimate area as: (intersection_count / total_points) × bounding_box_area
///
/// This is much more efficient than sampling the entire shape region because
/// the intersection is typically much smaller than the inclusive area.
///
/// # Arguments
///
/// * `shapes` - The shapes to compute overlaps for
/// * `n_samples` - Number of Monte Carlo samples (more = better accuracy)
///
/// # Returns
///
/// Intersection area estimate (area where ALL shapes overlap)
fn monte_carlo_overlap<S: Shape>(
    shapes: &[S],
    n_samples: usize,
    rng: &mut dyn rand::RngCore,
) -> f64 {
    let n_sets = shapes.len();
    let all_shapes_mask = (1 << n_sets) - 1; // All bits set

    // Compute bounding box for sampling
    // We need a region that's guaranteed to contain the intersection.
    // The intersection of all shapes is contained within the intersection
    // of their bounding boxes.
    let (x_min, x_max, y_min, y_max) = if shapes.len() == 1 {
        // Single shape - use its bounding box
        let bbox = shapes[0].bounding_box();
        let (min_pt, max_pt) = bbox.to_points();
        (min_pt.x(), max_pt.x(), min_pt.y(), max_pt.y())
    } else {
        // Multiple shapes - compute intersection of all bounding boxes
        let mut x_min = f64::NEG_INFINITY;
        let mut x_max = f64::INFINITY;
        let mut y_min = f64::NEG_INFINITY;
        let mut y_max = f64::INFINITY;

        for shape in shapes {
            let bbox = shape.bounding_box();
            let (min_pt, max_pt) = bbox.to_points();

            // Intersection of bounding boxes
            x_min = x_min.max(min_pt.x());
            x_max = x_max.min(max_pt.x());
            y_min = y_min.max(min_pt.y());
            y_max = y_max.min(max_pt.y());
        }

        // If bounding boxes don't overlap, there's no intersection
        if x_min >= x_max || y_min >= y_max {
            return 0.0;
        }

        (x_min, x_max, y_min, y_max)
    };

    let bbox_area = (x_max - x_min) * (y_max - y_min);

    // Track region counts: region_mask -> count
    let mut region_counts: HashMap<usize, usize> = HashMap::new();

    // Generate random points and test containment
    for _ in 0..n_samples {
        let x = rng.random_range(x_min..x_max);
        let y = rng.random_range(y_min..y_max);
        let point = Point::new(x, y);

        // Determine which shapes contain this point
        let mut region_mask = 0_usize;
        for (i, shape) in shapes.iter().enumerate() {
            if shape.contains_point(&point) {
                region_mask |= 1 << i;
            }
        }

        // Increment count for this region
        if region_mask > 0 {
            *region_counts.entry(region_mask).or_insert(0) += 1;
        }
    }

    // Compute intersection area (only points in ALL shapes)
    let intersection_count = region_counts.get(&all_shapes_mask).copied().unwrap_or(0);
    (intersection_count as f64 / n_samples as f64) * bbox_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::shapes::circle::Circle;
    use crate::geometry::shapes::rectangle::Rectangle;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_monte_carlo_single_circle() {
        let mut rng = StdRng::seed_from_u64(42);
        let circle = Circle::new(Point::new(0.0, 0.0), 1.0);
        let shapes = vec![circle];

        // Expected area: π ≈ 3.14159 (single shape = full area)
        let estimated = monte_carlo_overlap(&shapes, 100_000, &mut rng);
        let expected = std::f64::consts::PI;

        // Monte Carlo should be within ~1% for 100k samples
        let error = (estimated - expected).abs() / expected;
        assert!(
            error < 0.01,
            "Error too large: {} (estimated: {}, expected: {})",
            error,
            estimated,
            expected
        );
    }

    #[test]
    fn test_monte_carlo_two_separate_circles() {
        let mut rng = StdRng::seed_from_u64(42);
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(10.0, 0.0), 1.0);
        let shapes = vec![c1, c2];

        // Expected area: 0 (no overlap/intersection)
        let estimated = monte_carlo_overlap(&shapes, 100_000, &mut rng);

        assert!(
            estimated.abs() < 0.001,
            "Should be near zero but got: {}",
            estimated
        );
    }

    #[test]
    fn test_monte_carlo_two_overlapping_circles() {
        let mut rng = StdRng::seed_from_u64(42);
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        let shapes = vec![c1, c2];

        // Compute intersection points
        let pts = c1.intersection_points(&c2);
        let mut intersections = Vec::new();
        for point in pts {
            let adopters = vec![0, 1]; // Both circles contain these points
            intersections.push(IntersectionPoint::new(point, (0, 1), adopters));
        }

        // Expected area: intersection area only
        let expected = c1.intersection_area(&c2);

        let estimated = monte_carlo_overlap(&shapes, 100_000, &mut rng);

        let error = (estimated - expected).abs() / expected;
        assert!(
            error < 0.02,
            "Error too large: {} (estimated: {}, expected: {})",
            error,
            estimated,
            expected
        );
    }

    #[test]
    fn test_monte_carlo_rectangle() {
        let mut rng = StdRng::seed_from_u64(42);
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);
        let shapes = vec![rect];

        // Expected area: 8.0 (single shape = full area)
        let estimated = monte_carlo_overlap(&shapes, 100_000, &mut rng);
        let expected = 8.0;

        let error = (estimated - expected).abs() / expected;
        assert!(
            error < 0.01,
            "Error too large: {} (estimated: {}, expected: {})",
            error,
            estimated,
            expected
        );
    }

    #[test]
    fn test_monte_carlo_two_overlapping_rectangles() {
        let mut rng = StdRng::seed_from_u64(42);
        let r1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let r2 = Rectangle::new(Point::new(2.0, 0.0), 4.0, 4.0);
        let shapes = vec![r1, r2];

        // Expected area: intersection only (2x4 = 8)
        let expected = r1.intersection_area(&r2);

        let estimated = monte_carlo_overlap(&shapes, 100_000, &mut rng);

        let error = (estimated - expected).abs() / expected;
        assert!(
            error < 0.02,
            "Error too large: {} (estimated: {}, expected: {})",
            error,
            estimated,
            expected
        );
    }

    #[test]
    fn test_monte_carlo_three_overlapping_circles() {
        let mut rng = StdRng::seed_from_u64(42);
        // Three circles with a common intersection region
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 2.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 2.0);
        let shapes = vec![c1, c2, c3];

        // Should have some 3-way intersection
        let estimated = monte_carlo_overlap(&shapes, 100_000, &mut rng);

        // Just verify it's positive and reasonable
        assert!(
            estimated > 0.5 && estimated < 5.0,
            "Expected reasonable 3-way intersection area, got: {}",
            estimated
        );
    }

    #[test]
    fn test_exact_circles_two_overlapping() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        let circles = vec![c1, c2];

        // Exact computation for circles
        let exact = compute_overlaps_circles(&circles);

        // Should match the analytical intersection area
        let expected = c1.intersection_area(&c2);

        let error = (exact - expected).abs() / expected;
        assert!(
            error < 0.001,
            "Exact should be very close: {} vs {}",
            exact,
            expected
        );
    }

    #[test]
    fn test_exact_circles_three_overlapping() {
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 2.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 2.0);
        let circles = vec![c1, c2, c3];

        // Exact computation for 3-way intersection
        let exact = compute_overlaps_circles(&circles);

        // Should be positive (there is a 3-way intersection)
        assert!(
            exact > 0.0,
            "Expected positive 3-way intersection area, got: {}",
            exact
        );
    }

    #[test]
    fn test_exact_vs_monte_carlo_two_circles() {
        let mut rng = StdRng::seed_from_u64(42);
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        let circles = vec![c1, c2];

        // Exact computation
        let exact = compute_overlaps_circles(&circles);

        // Monte Carlo computation
        let monte_carlo = monte_carlo_overlap(&circles, 100_000, &mut rng);

        // They should be within ~2% of each other
        let error = (exact - monte_carlo).abs() / exact;
        assert!(
            error < 0.02,
            "Exact and Monte Carlo should agree: exact={}, monte_carlo={}, error={}",
            exact,
            monte_carlo,
            error
        );
    }

    #[test]
    fn test_exact_vs_monte_carlo_three_circles() {
        let mut rng = StdRng::seed_from_u64(42);
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 2.0);
        let c3 = Circle::new(Point::new(0.75, 1.3), 2.0);
        let circles = vec![c1, c2, c3];

        // Exact computation
        let exact = compute_overlaps_circles(&circles);

        // Monte Carlo computation
        let monte_carlo = monte_carlo_overlap(&circles, 100_000, &mut rng);

        // They should be within ~3% of each other (3-way is harder)
        let error = (exact - monte_carlo).abs() / exact;
        assert!(
            error < 0.03,
            "Exact and Monte Carlo should agree: exact={}, monte_carlo={}, error={}",
            exact,
            monte_carlo,
            error
        );
    }

    #[test]
    fn test_exact_vs_monte_carlo_no_overlap() {
        let mut rng = StdRng::seed_from_u64(42);
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(10.0, 0.0), 1.0);
        let circles = vec![c1, c2];

        // Exact computation
        let exact = compute_overlaps_circles(&circles);

        // Monte Carlo computation
        let monte_carlo = monte_carlo_overlap(&circles, 100_000, &mut rng);

        // Both should be essentially zero
        assert!(
            exact.abs() < 0.001 && monte_carlo.abs() < 0.001,
            "Both should be near zero: exact={}, monte_carlo={}",
            exact,
            monte_carlo
        );
    }

    #[test]
    fn test_exact_vs_monte_carlo_four_circles() {
        let mut rng = StdRng::seed_from_u64(42);
        // Four circles arranged in a square pattern with central overlap
        let c1 = Circle::new(Point::new(-0.5, -0.5), 1.5);
        let c2 = Circle::new(Point::new(0.5, -0.5), 1.5);
        let c3 = Circle::new(Point::new(-0.5, 0.5), 1.5);
        let c4 = Circle::new(Point::new(0.5, 0.5), 1.5);
        let circles = vec![c1, c2, c3, c4];

        // Exact computation
        let exact = compute_overlaps_circles(&circles);

        // Monte Carlo computation
        let monte_carlo = monte_carlo_overlap(&circles, 100_000, &mut rng);

        // For 4-way intersection, if there is one
        if exact > 0.1 && monte_carlo > 0.1 {
            // They should be within ~5% of each other (4-way is harder)
            let error = (exact - monte_carlo).abs() / exact.max(monte_carlo);
            assert!(
                error < 0.05,
                "Exact and Monte Carlo should agree: exact={}, monte_carlo={}, error={}",
                exact,
                monte_carlo,
                error
            );
        } else if exact < 0.1 && monte_carlo < 0.1 {
            // Both agree there's minimal/no intersection
            // This is expected behavior
        } else {
            // One says yes, one says no - that's a problem
            panic!(
                "Methods disagree on existence of intersection: exact={}, monte_carlo={}",
                exact, monte_carlo
            );
        }
    }

    #[test]
    fn test_exact_vs_monte_carlo_highly_overlapping() {
        let mut rng = StdRng::seed_from_u64(42);
        // Three circles that overlap significantly
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.5);
        let c2 = Circle::new(Point::new(1.0, 0.0), 2.5);
        let c3 = Circle::new(Point::new(0.5, 0.8), 2.5);
        let circles = vec![c1, c2, c3];

        // Exact computation
        let exact = compute_overlaps_circles(&circles);

        // Monte Carlo computation
        let monte_carlo = monte_carlo_overlap(&circles, 100_000, &mut rng);

        // They should be within ~3% of each other
        let error = (exact - monte_carlo).abs() / exact;
        assert!(
            error < 0.03,
            "Exact and Monte Carlo should agree: exact={}, monte_carlo={}, error={}",
            exact,
            monte_carlo,
            error
        );

        // Should have substantial area (large overlapping circles)
        assert!(
            exact > 5.0,
            "Expected large 3-way intersection for highly overlapping circles, got: {}",
            exact
        );
    }

    #[test]
    fn test_exact_vs_monte_carlo_barely_overlapping() {
        let mut rng = StdRng::seed_from_u64(42);
        // Three circles that barely overlap
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.5);
        let c2 = Circle::new(Point::new(2.5, 0.0), 1.5);
        let c3 = Circle::new(Point::new(1.25, 2.0), 1.5);
        let circles = vec![c1, c2, c3];

        // Exact computation
        let exact = compute_overlaps_circles(&circles);

        // Monte Carlo computation
        let monte_carlo = monte_carlo_overlap(&circles, 100_000, &mut rng);

        // For small intersections, allow slightly larger error
        if exact > 0.1 && monte_carlo > 0.1 {
            let error = (exact - monte_carlo).abs() / exact;
            assert!(
                error < 0.05,
                "Exact and Monte Carlo should agree: exact={}, monte_carlo={}, error={}",
                exact,
                monte_carlo,
                error
            );
        } else {
            // Both should be very small
            assert!(
                exact < 0.5 && monte_carlo < 0.5,
                "Both should be small for barely overlapping circles: exact={}, monte_carlo={}",
                exact,
                monte_carlo
            );
        }
    }

    #[test]
    fn test_exact_single_circle() {
        let c = Circle::new(Point::new(2.0, 3.0), 1.5);
        let circles = vec![c];

        let exact = compute_overlaps_circles(&circles);
        let expected = c.area();

        let error = (exact - expected).abs() / expected;
        assert!(
            error < 0.001,
            "Single circle should return its area: exact={}, expected={}",
            exact,
            expected
        );
    }

    #[test]
    fn test_exact_contained_circles() {
        // One circle completely inside another
        let c1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let c2 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circles = vec![c1, c2];

        let exact = compute_overlaps_circles(&circles);
        let expected = c2.area(); // Intersection is the smaller circle

        let error = (exact - expected).abs() / expected;
        assert!(
            error < 0.001,
            "Contained circle intersection should be smaller circle area: exact={}, expected={}",
            exact,
            expected
        );
    }
}
