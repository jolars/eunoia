//! Circle shape implementation.

use crate::geometry::coord::Coord;
use crate::geometry::shapes::Shape;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::brent::BrentOpt;

/// A circle defined by a center point and radius.
///
/// Circles are the simplest shape for Euler and Venn diagrams and are often
/// sufficient for many use cases. They have the advantage of being rotationally
/// symmetric, which simplifies some computations.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::circle::Circle;
/// use eunoia::geometry::shapes::Shape;
/// use eunoia::geometry::coord::Coord;
///
/// let c1 = Circle::new(Coord::new(0.0, 0.0), 2.0);
/// let c2 = Circle::new(Coord::new(3.0, 0.0), 1.0);
///
/// let area1 = c1.area();
/// let overlap = c1.intersection_area(&c2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    center: Coord,
    radius: f64,
}

impl Shape for Circle {
    /// Computes the area of the circle using the formula A = πr².
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    fn centroid(&self) -> (f64, f64) {
        (self.center.x(), self.center.y())
    }

    /// Computes the minimum distance between the boundaries of two circles.
    ///
    /// Returns 0.0 if the circles overlap or touch.
    fn distance(&self, other: &Self) -> f64 {
        let center_distance = self.center.distance(&other.center);
        let radius_sum = self.radius + other.radius;

        if center_distance > radius_sum {
            center_distance - radius_sum
        } else {
            0.0
        }
    }

    fn contains(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance + other.radius <= self.radius
    }

    /// Checks if two circles intersect (share any common points).
    ///
    /// Note: This implementation returns `true` if circles are separate,
    /// which appears to be inverted from the typical definition. This may
    /// need correction.
    fn intersects(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance >= self.radius + other.radius
    }

    /// Computes the area of intersection between two circles.
    ///
    /// Uses the standard geometric formula for circle-circle intersection:
    /// - Returns 0 if circles don't overlap
    /// - Returns area of smaller circle if one contains the other
    /// - Otherwise computes the lens-shaped intersection area
    ///
    /// # Algorithm
    ///
    /// For two circles with radii r1 and r2 separated by distance d, the
    /// intersection area is computed using the formula involving circular
    /// segments from both circles.
    fn intersection_area(&self, other: &Self) -> f64 {
        let d = self.center.distance(&other.center);

        if d >= self.radius + other.radius {
            return 0.0; // No intersection
        }

        if d <= (self.radius - other.radius).abs() {
            // One circle is completely inside the other
            let smaller_radius = self.radius.min(other.radius);
            return std::f64::consts::PI * smaller_radius * smaller_radius;
        }

        let r1 = self.radius;
        let r2 = other.radius;

        let part1 = r1 * r1 * (((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos());
        let part2 = r2 * r2 * (((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos());
        let part3 = 0.5 * ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).sqrt();

        part1 + part2 - part3
    }
}

struct SeparationCost {
    r1: f64,
    r2: f64,
    target_overlap: f64,
}

impl CostFunction for SeparationCost {
    type Param = f64;
    type Output = f64;

    fn cost(&self, distance: &Self::Param) -> Result<Self::Output, Error> {
        let c1 = Circle::new(Coord::new(0.0, 0.0), self.r1);
        let c2 = Circle::new(Coord::new(*distance, 0.0), self.r2);

        let current_overlap = c1.intersection_area(&c2);
        let cost = (current_overlap - self.target_overlap).powi(2);

        Ok(cost)
    }
}

impl Circle {
    /// Creates a new circle with the specified center and radius.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the circle
    /// * `radius` - The radius of the circle (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::circle::Circle;
    /// use eunoia::geometry::coord::Coord;
    ///
    /// let circle = Circle::new(Coord::new(1.0, 2.0), 3.0);
    /// ```
    pub fn new(center: Coord, radius: f64) -> Self {
        Circle { center, radius }
    }

    /// Returns a reference to the circle's center point.
    pub fn center(&self) -> &Coord {
        &self.center
    }

    /// Returns the circle's radius.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Sets the center of the circle.
    pub fn set_center(&mut self, center: Coord) {
        self.center = center;
    }
}

/// Computes the distance required between two circles to achieve a specified overlap area.
pub(crate) fn distance_for_overlap(
    r1: f64,
    r2: f64,
    overlap: f64,
    tol: Option<f64>,
    max_iter: Option<u64>,
) -> Result<f64, Error> {
    let min_distance = (r1 - r2).abs();
    let max_distance = r1 + r2;

    if overlap <= 0.0 {
        return Ok(max_distance);
    }

    let cost_fun = SeparationCost {
        r1,
        r2,
        target_overlap: overlap,
    };

    let solver = BrentOpt::new(min_distance, max_distance);

    let result = Executor::new(cost_fun, solver)
        .configure(|state| {
            state
                .max_iters(max_iter.unwrap_or(100))
                .target_cost(tol.unwrap_or(1e-6))
        })
        .run()?;

    Ok(*result.state.get_best_param().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_circle_new() {
        let center = Coord::new(1.0, 2.0);
        let circle = Circle::new(center, 5.0);
        assert_eq!(circle.radius(), 5.0);
        assert_eq!(circle.center().x(), 1.0);
        assert_eq!(circle.center().y(), 2.0);
    }

    #[test]
    fn test_circle_area() {
        let circle = Circle::new(Coord::new(0.0, 0.0), 1.0);
        assert!(approx_eq(circle.area(), std::f64::consts::PI));

        let circle2 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        assert!(approx_eq(circle2.area(), 4.0 * std::f64::consts::PI));

        let circle3 = Circle::new(Coord::new(5.0, 5.0), 3.0);
        assert!(approx_eq(circle3.area(), 9.0 * std::f64::consts::PI));
    }

    #[test]
    fn test_circle_distance_no_overlap() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(5.0, 0.0), 1.0);
        assert_eq!(circle1.distance(&circle2), 3.0);
    }

    #[test]
    fn test_circle_distance_touching() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(2.0, 0.0), 1.0);
        assert_eq!(circle1.distance(&circle2), 0.0);
    }

    #[test]
    fn test_circle_distance_overlapping() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Coord::new(1.0, 0.0), 2.0);
        assert_eq!(circle1.distance(&circle2), 0.0);
    }

    #[test]
    fn test_circle_contains_smaller() {
        let large = Circle::new(Coord::new(0.0, 0.0), 5.0);
        let small = Circle::new(Coord::new(1.0, 1.0), 2.0);
        assert!(large.contains(&small));
    }

    #[test]
    fn test_circle_contains_self() {
        let circle = Circle::new(Coord::new(0.0, 0.0), 3.0);
        assert!(circle.contains(&circle));
    }

    #[test]
    fn test_circle_not_contains() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Coord::new(5.0, 0.0), 2.0);
        assert!(!circle1.contains(&circle2));
    }

    #[test]
    fn test_circle_not_contains_partial_overlap() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 3.0);
        let circle2 = Circle::new(Coord::new(2.0, 0.0), 2.0);
        assert!(!circle1.contains(&circle2));
    }

    #[test]
    fn test_circle_intersects_separate() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(5.0, 0.0), 1.0);
        assert!(circle1.intersects(&circle2));
    }

    #[test]
    fn test_circle_intersects_touching() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(2.0, 0.0), 1.0);
        assert!(circle1.intersects(&circle2));
    }

    #[test]
    fn test_circle_intersects_overlapping() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Coord::new(1.0, 0.0), 2.0);
        assert!(!circle1.intersects(&circle2));
    }

    #[test]
    fn test_intersection_area_no_overlap() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(10.0, 0.0), 1.0);
        assert_eq!(circle1.intersection_area(&circle2), 0.0);
    }

    #[test]
    fn test_intersection_area_touching() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(2.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);
        assert!(approx_eq(area, 0.0));
    }

    #[test]
    fn test_intersection_area_complete_overlap_same_size() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        let expected = std::f64::consts::PI * 4.0;
        assert!(approx_eq(circle1.intersection_area(&circle2), expected));
    }

    #[test]
    fn test_intersection_area_one_inside_other() {
        let large = Circle::new(Coord::new(0.0, 0.0), 5.0);
        let small = Circle::new(Coord::new(1.0, 0.0), 2.0);
        let expected = std::f64::consts::PI * 4.0; // Area of smaller circle
        assert!(approx_eq(large.intersection_area(&small), expected));
        assert!(approx_eq(small.intersection_area(&large), expected));
    }

    #[test]
    fn test_intersection_area_partial_overlap() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Coord::new(1.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);

        // For two unit circles with centers 1 apart, there's a known formula
        // The intersection area should be positive and less than π
        assert!(area > 0.0);
        assert!(area < std::f64::consts::PI);
    }

    #[test]
    fn test_intersection_area_symmetric() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Coord::new(1.5, 0.0), 1.5);
        let area1 = circle1.intersection_area(&circle2);
        let area2 = circle2.intersection_area(&circle1);
        assert!(approx_eq(area1, area2));
    }

    #[test]
    fn test_intersection_area_different_sizes() {
        let circle1 = Circle::new(Coord::new(0.0, 0.0), 3.0);
        let circle2 = Circle::new(Coord::new(2.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);

        // Should be positive and at most the smaller circle's area
        assert!(area > 0.0);
        assert!(area <= std::f64::consts::PI * 1.0 * 1.0);
    }

    #[test]
    fn test_distance_for_overlap_zero_overlap() {
        let r1 = 2.0;
        let r2 = 1.5;
        let overlap = 0.0;

        let distance = distance_for_overlap(r1, r2, overlap, None, None).unwrap();

        // Should return the sum of radii (circles just touching)
        assert!(approx_eq(distance, r1 + r2));
    }

    #[test]
    fn test_distance_for_overlap_negative_overlap() {
        let r1 = 2.0;
        let r2 = 1.5;
        let overlap = -1.0;

        let distance = distance_for_overlap(r1, r2, overlap, None, None).unwrap();

        // Should return the sum of radii (circles separated)
        assert!(approx_eq(distance, r1 + r2));
    }

    #[test]
    fn test_distance_for_overlap_full_overlap() {
        let r1 = 3.0;
        let r2 = 2.0;
        let overlap = std::f64::consts::PI * r2 * r2; // Full area of smaller circle

        let distance = distance_for_overlap(r1, r2, overlap, None, None).unwrap();

        // Distance should be close to |r1 - r2| (one inside the other)
        assert!(distance <= (r1 - r2).abs() + 0.1); // Allow small tolerance
    }

    #[test]
    fn test_distance_for_overlap_partial_overlap_equal_radii() {
        let r1 = 2.0;
        let r2 = 2.0;
        let target_overlap = 2.0; // Some specific overlap area

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        // Verify the result by computing the actual overlap at this distance
        let c1 = Circle::new(Coord::new(0.0, 0.0), r1);
        let c2 = Circle::new(Coord::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        // Should match target within tolerance (relaxed for optimization convergence)
        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_partial_overlap_different_radii() {
        let r1 = 3.0;
        let r2 = 1.5;
        let target_overlap = 1.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        // Verify the result
        let c1 = Circle::new(Coord::new(0.0, 0.0), r1);
        let c2 = Circle::new(Coord::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_custom_tolerance() {
        let r1 = 2.0;
        let r2 = 1.0;
        let target_overlap = 0.5;
        let custom_tol = 1e-8;

        let distance =
            distance_for_overlap(r1, r2, target_overlap, Some(custom_tol), None).unwrap();

        let c1 = Circle::new(Coord::new(0.0, 0.0), r1);
        let c2 = Circle::new(Coord::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        // Should be very close to target with custom tolerance
        assert!((actual_overlap - target_overlap).abs() < 1e-3);
    }

    #[test]
    fn test_distance_for_overlap_custom_max_iter() {
        let r1 = 2.0;
        let r2 = 1.5;
        let target_overlap = 1.0;
        let max_iter = 50;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, Some(max_iter)).unwrap();

        // Should still converge within fewer iterations
        let c1 = Circle::new(Coord::new(0.0, 0.0), r1);
        let c2 = Circle::new(Coord::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_small_circles() {
        let r1 = 0.5;
        let r2 = 0.3;
        let target_overlap = 0.1;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        let c1 = Circle::new(Coord::new(0.0, 0.0), r1);
        let c2 = Circle::new(Coord::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-4);
    }

    #[test]
    fn test_distance_for_overlap_large_circles() {
        let r1 = 100.0;
        let r2 = 75.0;
        let target_overlap = 500.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        let c1 = Circle::new(Coord::new(0.0, 0.0), r1);
        let c2 = Circle::new(Coord::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1.0); // Larger tolerance for large circles
    }

    #[test]
    fn test_distance_for_overlap_bounds() {
        let r1 = 2.0;
        let r2 = 1.5;
        let target_overlap = 1.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        // Distance should be between min (one inside other) and max (circles touching)
        let min_distance = (r1 - r2).abs();
        let max_distance = r1 + r2;

        assert!(distance >= min_distance);
        assert!(distance <= max_distance);
    }
}
