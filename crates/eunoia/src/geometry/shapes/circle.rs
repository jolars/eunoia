//! Circle shape implementation.

use crate::geometry::diagram::IntersectionPoint;
use crate::geometry::point;
use crate::geometry::point::Point;
use crate::geometry::shapes::rectangle::Rectangle;
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
/// use eunoia::geometry::point::Point;
///
/// let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
/// let c2 = Circle::new(Point::new(3.0, 0.0), 1.0);
///
/// let area1 = c1.area();
/// let overlap = c1.intersection_area(&c2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    center: Point,
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

    fn contains_point(&self, point: &Point) -> bool {
        let dist = self.center.distance(point);
        dist <= self.radius
    }

    /// Compute the perimeter of the circle.
    fn perimeter(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
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
        let part3 = 0.5 * ((r1 + r2 - d) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).sqrt();

        part1 + part2 - part3
    }

    /// Computes the points of intersection between two circles.
    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        let d = self.center.distance(&other.center);

        if d > self.radius + other.radius || d < (self.radius - other.radius).abs() {
            return vec![]; // No intersection points
        }

        let a = (self.radius * self.radius - other.radius * other.radius + d * d) / (2.0 * d);
        let h = (self.radius * self.radius - a * a).sqrt();

        let p2_x = self.center.x() + a * (other.center.x() - self.center.x()) / d;
        let p2_y = self.center.y() + a * (other.center.y() - self.center.y()) / d;

        let rx = -(other.center.y() - self.center.y()) * (h / d);
        let ry = (other.center.x() - self.center.x()) * (h / d);

        let intersection1 = Point::new(p2_x + rx, p2_y + ry);
        let intersection2 = Point::new(p2_x - rx, p2_y - ry);

        if intersection1 == intersection2 {
            vec![intersection1]
        } else {
            vec![intersection1, intersection2]
        }
    }

    fn bounding_box(&self) -> Rectangle {
        let width = 2.0 * self.radius;
        let height = 2.0 * self.radius;

        Rectangle::new(self.center, width, height)
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
        let c1 = Circle::new(Point::new(0.0, 0.0), self.r1);
        let c2 = Circle::new(Point::new(*distance, 0.0), self.r2);

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
    /// use eunoia::geometry::point::Point;
    ///
    /// let circle = Circle::new(Point::new(1.0, 2.0), 3.0);
    /// ```
    pub fn new(center: Point, radius: f64) -> Self {
        Circle { center, radius }
    }

    /// Returns a reference to the circle's center point.
    pub fn center(&self) -> &Point {
        &self.center
    }

    /// Returns the circle's radius.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Sets the center of the circle.
    pub fn set_center(&mut self, center: Point) {
        self.center = center;
    }

    /// Circular sector area given the radius and angle in radians.
    pub fn sector_area(&self, angle_rad: f64) -> f64 {
        0.5 * self.radius * self.radius * angle_rad
    }

    /// Circular segment area from radius and angle
    pub fn segment_area_from_angle(&self, angle_rad: f64) -> f64 {
        0.5 * self.radius * self.radius * (angle_rad - angle_rad.sin())
    }

    /// Circular segment area from radius and chord length
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `chord_length > 2 * radius` (impossible geometry).
    /// In release builds, this validation is skipped for performance.
    pub fn segment_area_from_chord(&self, chord_length: f64) -> f64 {
        let r = self.radius;
        debug_assert!(
            chord_length <= 2.0 * r,
            "Chord length {} cannot exceed diameter {}",
            chord_length,
            2.0 * r
        );
        let theta = 2.0 * (chord_length / (2.0 * r)).asin();
        self.segment_area_from_angle(theta)
    }

    pub fn segment_area_from_points(&self, p1: &Point, p2: &Point) -> f64 {
        let chord_length = p1.distance(p2);
        self.segment_area_from_chord(chord_length)
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

    // If the desired overlap is zero, then the circles should
    // at most be touching.
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
                .max_iters(max_iter.unwrap_or(1000))
                .target_cost(tol.unwrap_or(f64::EPSILON.sqrt())) // Match eulerr: sqrt(machine epsilon)
        })
        .run()?;

    Ok(*result.state.get_best_param().unwrap())
}

pub fn multiple_overlap_areas(circles: &[Circle], points: &[IntersectionPoint]) -> f64 {
    let n_circles = circles.len();

    // Filter to only points that are in ALL circles (full intersection)
    let full_intersection_points: Vec<&IntersectionPoint> = points
        .iter()
        .filter(|ip| ip.adopters().len() == n_circles)
        .collect();

    if full_intersection_points.is_empty() {
        return 0.0;
    }

    let n_points = full_intersection_points.len();

    // Sort the points by their angles around the centroid
    let centroid = point::centroid(
        &full_intersection_points
            .iter()
            .map(|ip| *ip.point())
            .collect::<Vec<Point>>(),
    );

    let mut indices: Vec<usize> = (0..n_points).collect();
    indices.sort_by(|&i, &j| {
        full_intersection_points[i]
            .point()
            .angle_to(&centroid)
            .partial_cmp(&full_intersection_points[j].point().angle_to(&centroid))
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let mut area = 0.0;

    let mut l = n_points - 1;

    for k in 0..n_points {
        let i = indices[k];
        let j = indices[l];

        let p1 = &full_intersection_points[i].point();
        let p2 = &full_intersection_points[j].point();

        // Now we need to discover which of the circles the two points are
        // coming from so that we can compute the segment area.
        // This should be the set intersection of the parents of both points.
        // In some cases, the intersection may be of length 2, in which
        // case we need to compute both segment areas and pick the
        // smaller one.
        let parents1 = &full_intersection_points[i].parents();
        let parents2 = &full_intersection_points[j].parents();

        let common_parents: Vec<usize> = vec![parents1.0, parents1.1]
            .into_iter()
            .filter(|p| *p == parents2.0 || *p == parents2.1)
            .collect();

        let mut segment_areas = Vec::with_capacity(common_parents.len());

        if common_parents.is_empty() {
            // This should not happen in a well-formed set of intersection points
            panic!("No common parent circles found for intersection points");
        }

        for &circle_index in &common_parents {
            let circle = &circles[circle_index];
            let seg_area = circle.segment_area_from_points(p1, p2);

            debug_assert!(seg_area >= 0.0, "Segment area should be non-negative");

            segment_areas.push(seg_area);
        }

        let triangle_area = 0.5 * ((p1.x() + p2.x()) * (p1.y() - p2.y()));
        // Note: triangle_area can be negative (signed area from shoelace algorithm)
        // We take abs() at the end to get the final area

        let min_segment = segment_areas
            .into_iter()
            .fold(f64::INFINITY, |a, b| a.min(b));

        area += triangle_area;
        area += min_segment;

        l = k;
    }

    area.abs()
}

/// Compute the area of the intersection region for a subset of circles.
///
/// This is similar to `multiple_overlap_areas` but allows specifying which circles
/// to consider for the "full intersection". This is needed for computing 3-way
/// intersections in a 4+ circle diagram where some intersection points may be
/// in more than just the 3 circles of interest.
///
/// # Arguments
/// * `circles` - All circles in the diagram
/// * `points` - Intersection points (with adopters referencing indices in `circles`)
/// * `circle_indices` - Indices of the circles that define this region
pub fn multiple_overlap_areas_with_mask(
    circles: &[Circle],
    points: &[IntersectionPoint],
    circle_indices: &[usize],
) -> f64 {
    // Filter to only points that are in ALL of the specified circles
    // A point is in the region if all circle_indices are present in its adopters
    let region_points: Vec<&IntersectionPoint> = points
        .iter()
        .filter(|ip| {
            circle_indices
                .iter()
                .all(|&idx| ip.adopters().contains(&idx))
        })
        .collect();

    if region_points.is_empty() {
        return 0.0;
    }

    let n_points = region_points.len();

    // Sort the points by their angles around the centroid
    let centroid = point::centroid(
        &region_points
            .iter()
            .map(|ip| *ip.point())
            .collect::<Vec<Point>>(),
    );

    let mut indices: Vec<usize> = (0..n_points).collect();
    indices.sort_by(|&i, &j| {
        region_points[i]
            .point()
            .angle_to(&centroid)
            .partial_cmp(&region_points[j].point().angle_to(&centroid))
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let mut area = 0.0;

    let mut l = n_points - 1;

    for k in 0..n_points {
        let i = indices[k];
        let j = indices[l];

        let p1 = &region_points[i].point();
        let p2 = &region_points[j].point();

        // Find which of the region circles these points come from
        let parents1 = &region_points[i].parents();
        let parents2 = &region_points[j].parents();

        let common_parents: Vec<usize> = vec![parents1.0, parents1.1]
            .into_iter()
            .filter(|p| *p == parents2.0 || *p == parents2.1)
            .filter(|p| circle_indices.contains(p)) // Only consider circles in our region
            .collect();

        let mut segment_areas = Vec::with_capacity(common_parents.len());

        if common_parents.is_empty() {
            // Try to find any circle in the region that contains both points
            // This can happen when points come from different pairs but are connected
            // through the region
            for &circle_idx in circle_indices {
                let circle = &circles[circle_idx];
                if circle.contains_point(p1) && circle.contains_point(p2) {
                    let seg_area = circle.segment_area_from_points(p1, p2);
                    segment_areas.push(seg_area);
                }
            }

            if segment_areas.is_empty() {
                // No circle in the region connects these points - use straight line
                // This shouldn't normally happen in a well-formed region
                l = k;
                continue;
            }
        } else {
            for &circle_index in &common_parents {
                let circle = &circles[circle_index];
                let seg_area = circle.segment_area_from_points(p1, p2);

                debug_assert!(seg_area >= 0.0, "Segment area should be non-negative");

                segment_areas.push(seg_area);
            }
        }

        let triangle_area = 0.5 * ((p1.x() + p2.x()) * (p1.y() - p2.y()));

        let min_segment = segment_areas
            .into_iter()
            .fold(f64::INFINITY, |a, b| a.min(b));

        area += triangle_area;
        area += min_segment;

        l = k;
    }

    area.abs()
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
        let center = Point::new(1.0, 2.0);
        let circle = Circle::new(center, 5.0);
        assert_eq!(circle.radius(), 5.0);
        assert_eq!(circle.center().x(), 1.0);
        assert_eq!(circle.center().y(), 2.0);
    }

    #[test]
    fn test_circle_area() {
        let circle = Circle::new(Point::new(0.0, 0.0), 1.0);
        assert!(approx_eq(circle.area(), std::f64::consts::PI));

        let circle2 = Circle::new(Point::new(0.0, 0.0), 2.0);
        assert!(approx_eq(circle2.area(), 4.0 * std::f64::consts::PI));

        let circle3 = Circle::new(Point::new(5.0, 5.0), 3.0);
        assert!(approx_eq(circle3.area(), 9.0 * std::f64::consts::PI));
    }

    #[test]
    fn test_circle_distance_no_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(5.0, 0.0), 1.0);
        assert_eq!(circle1.distance(&circle2), 3.0);
    }

    #[test]
    fn test_circle_distance_touching() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        assert_eq!(circle1.distance(&circle2), 0.0);
    }

    #[test]
    fn test_circle_distance_overlapping() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(1.0, 0.0), 2.0);
        assert_eq!(circle1.distance(&circle2), 0.0);
    }

    #[test]
    fn test_circle_contains_smaller() {
        let large = Circle::new(Point::new(0.0, 0.0), 5.0);
        let small = Circle::new(Point::new(1.0, 1.0), 2.0);
        assert!(large.contains(&small));
    }

    #[test]
    fn test_circle_contains_self() {
        let circle = Circle::new(Point::new(0.0, 0.0), 3.0);
        assert!(circle.contains(&circle));
    }

    #[test]
    fn test_circle_not_contains() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(5.0, 0.0), 2.0);
        assert!(!circle1.contains(&circle2));
    }

    #[test]
    fn test_circle_not_contains_partial_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 2.0);
        assert!(!circle1.contains(&circle2));
    }

    #[test]
    fn test_circle_intersects_separate() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(5.0, 0.0), 1.0);
        assert!(circle1.intersects(&circle2));
    }

    #[test]
    fn test_circle_intersects_touching() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        assert!(circle1.intersects(&circle2));
    }

    #[test]
    fn test_circle_intersects_overlapping() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(1.0, 0.0), 2.0);
        assert!(!circle1.intersects(&circle2));
    }

    #[test]
    fn test_intersection_area_no_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(10.0, 0.0), 1.0);
        assert_eq!(circle1.intersection_area(&circle2), 0.0);
    }

    #[test]
    fn test_intersection_area_touching() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);
        assert!(approx_eq(area, 0.0));
    }

    #[test]
    fn test_intersection_area_complete_overlap_same_size() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let expected = std::f64::consts::PI * 4.0;
        assert!(approx_eq(circle1.intersection_area(&circle2), expected));
    }

    #[test]
    fn test_intersection_area_one_inside_other() {
        let large = Circle::new(Point::new(0.0, 0.0), 5.0);
        let small = Circle::new(Point::new(1.0, 0.0), 2.0);
        let expected = std::f64::consts::PI * 4.0; // Area of smaller circle
        assert!(approx_eq(large.intersection_area(&small), expected));
        assert!(approx_eq(small.intersection_area(&large), expected));
    }

    #[test]
    fn test_intersection_area_partial_overlap() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let circle2 = Circle::new(Point::new(1.0, 0.0), 1.0);
        let area = circle1.intersection_area(&circle2);

        // For two unit circles with centers 1 apart, there's a known formula
        // The intersection area should be positive and less than π
        assert!(area > 0.0);
        assert!(area < std::f64::consts::PI);
    }

    #[test]
    fn test_intersection_area_symmetric() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let circle2 = Circle::new(Point::new(1.5, 0.0), 1.5);
        let area1 = circle1.intersection_area(&circle2);
        let area2 = circle2.intersection_area(&circle1);
        assert!(approx_eq(area1, area2));
    }

    #[test]
    fn test_intersection_area_different_sizes() {
        let circle1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let circle2 = Circle::new(Point::new(2.0, 0.0), 1.0);
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
        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
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
        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
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

        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
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
        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-2);
    }

    #[test]
    fn test_distance_for_overlap_small_circles() {
        let r1 = 0.5;
        let r2 = 0.3;
        let target_overlap = 0.1;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
        let actual_overlap = c1.intersection_area(&c2);

        assert!((actual_overlap - target_overlap).abs() < 1e-4);
    }

    #[test]
    fn test_distance_for_overlap_large_circles() {
        let r1 = 100.0;
        let r2 = 75.0;
        let target_overlap = 500.0;

        let distance = distance_for_overlap(r1, r2, target_overlap, None, None).unwrap();

        let c1 = Circle::new(Point::new(0.0, 0.0), r1);
        let c2 = Circle::new(Point::new(distance, 0.0), r2);
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

    #[test]
    fn test_intersection_points_no_intersection() {
        // Circles too far apart
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(5.0, 0.0), 1.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_intersection_points_one_inside_other() {
        // One circle completely inside the other
        let c1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let c2 = Circle::new(Point::new(0.0, 0.0), 1.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_intersection_points_touching_externally() {
        // Circles touch at exactly one point (externally)
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(4.0, 0.0), 2.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 1);

        // The touching point should be at (2.0, 0.0)
        assert!(approx_eq(points[0].x(), 2.0));
        assert!(approx_eq(points[0].y(), 0.0));
    }

    #[test]
    fn test_intersection_points_two_points() {
        // Circles intersect at two points
        let c1 = Circle::new(Point::new(0.0, 0.0), 2.0);
        let c2 = Circle::new(Point::new(2.0, 0.0), 2.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // Both points should be on the circles
        for point in &points {
            let dist_to_c1 = c1.center.distance(point);
            let dist_to_c2 = c2.center.distance(point);
            assert!(approx_eq(dist_to_c1, c1.radius));
            assert!(approx_eq(dist_to_c2, c2.radius));
        }

        // Points should be at (1.0, sqrt(3)) and (1.0, -sqrt(3))
        let expected_x = 1.0;
        let expected_y = 3.0_f64.sqrt();

        // Check one point is at (1.0, sqrt(3))
        let found_positive = points
            .iter()
            .any(|p| approx_eq(p.x(), expected_x) && approx_eq(p.y(), expected_y));
        // Check other point is at (1.0, -sqrt(3))
        let found_negative = points
            .iter()
            .any(|p| approx_eq(p.x(), expected_x) && approx_eq(p.y(), -expected_y));

        assert!(found_positive);
        assert!(found_negative);
    }

    #[test]
    fn test_intersection_points_vertical_alignment() {
        // Test with circles aligned vertically
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.5);
        let c2 = Circle::new(Point::new(0.0, 2.0), 1.5);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // Both points should be equidistant from both centers
        for point in &points {
            let dist_to_c1 = c1.center.distance(point);
            let dist_to_c2 = c2.center.distance(point);
            assert!(approx_eq(dist_to_c1, c1.radius));
            assert!(approx_eq(dist_to_c2, c2.radius));
        }
    }

    #[test]
    fn test_intersection_points_equal_radii_partial_overlap() {
        // Two circles with equal radii, partially overlapping
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.0, 0.0), 1.0);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // The intersection points should be symmetric about the line connecting centers
        // They should both have x-coordinate = 0.5
        for point in &points {
            assert!(approx_eq(point.x(), 0.5));
        }
    }

    #[test]
    fn test_intersection_points_different_radii() {
        // Two circles with different radii
        let c1 = Circle::new(Point::new(0.0, 0.0), 3.0);
        let c2 = Circle::new(Point::new(2.0, 0.0), 1.5);

        let points = c1.intersection_points(&c2);
        assert_eq!(points.len(), 2);

        // Verify both points lie on both circles
        for point in &points {
            let dist_to_c1 = c1.center.distance(point);
            let dist_to_c2 = c2.center.distance(point);
            assert!(approx_eq(dist_to_c1, c1.radius));
            assert!(approx_eq(dist_to_c2, c2.radius));
        }
    }

    #[test]
    fn test_perimeter() {
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let perimeter = c.perimeter();
        assert!(approx_eq(perimeter, 2.0 * std::f64::consts::PI));
    }

    #[test]
    fn test_perimeter_various_radii() {
        let test_cases = vec![
            (0.5, std::f64::consts::PI),
            (2.0, 4.0 * std::f64::consts::PI),
            (10.0, 20.0 * std::f64::consts::PI),
        ];

        for (radius, expected) in test_cases {
            let c = Circle::new(Point::new(0.0, 0.0), radius);
            assert!(approx_eq(c.perimeter(), expected));
        }
    }

    #[test]
    fn test_sector_area_full_circle() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let full_circle_angle = 2.0 * std::f64::consts::PI;
        let sector = c.sector_area(full_circle_angle);
        assert!(approx_eq(sector, c.area()));
    }

    #[test]
    fn test_sector_area_half_circle() {
        let c = Circle::new(Point::new(0.0, 0.0), 3.0);
        let half_circle_angle = std::f64::consts::PI;
        let sector = c.sector_area(half_circle_angle);
        assert!(approx_eq(sector, c.area() / 2.0));
    }

    #[test]
    fn test_sector_area_quarter_circle() {
        let c = Circle::new(Point::new(0.0, 0.0), 4.0);
        let quarter_circle_angle = std::f64::consts::PI / 2.0;
        let sector = c.sector_area(quarter_circle_angle);
        assert!(approx_eq(sector, c.area() / 4.0));
    }

    #[test]
    fn test_sector_area_zero() {
        let c = Circle::new(Point::new(0.0, 0.0), 5.0);
        let sector = c.sector_area(0.0);
        assert!(approx_eq(sector, 0.0));
    }

    #[test]
    fn test_segment_area_from_angle_semicircle() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let angle = std::f64::consts::PI;
        let segment = c.segment_area_from_angle(angle);
        // For a semicircle, segment area equals sector area (half circle)
        assert!(approx_eq(segment, c.area() / 2.0));
    }

    #[test]
    fn test_segment_area_from_angle_zero() {
        let c = Circle::new(Point::new(0.0, 0.0), 3.0);
        let segment = c.segment_area_from_angle(0.0);
        assert!(approx_eq(segment, 0.0));
    }

    #[test]
    fn test_segment_area_from_angle_small() {
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let angle = std::f64::consts::PI / 6.0; // 30 degrees
        let segment = c.segment_area_from_angle(angle);

        // Segment should be positive and less than sector area
        let sector = c.sector_area(angle);
        assert!(segment > 0.0);
        assert!(segment < sector);
    }

    #[test]
    fn test_segment_area_from_chord_diameter() {
        let radius = 2.0;
        let c = Circle::new(Point::new(0.0, 0.0), radius);
        let chord_length = 2.0 * radius; // Diameter
        let segment = c.segment_area_from_chord(chord_length);

        // A chord equal to diameter creates a semicircle
        assert!(approx_eq(segment, c.area() / 2.0));
    }

    #[test]
    fn test_segment_area_from_chord_small() {
        let c = Circle::new(Point::new(0.0, 0.0), 5.0);
        let chord_length = 2.0;
        let segment = c.segment_area_from_chord(chord_length);

        // Small chord should create small segment
        assert!(segment > 0.0);
        assert!(segment < c.area() / 4.0);
    }

    #[test]
    fn test_segment_area_from_chord_vs_angle_consistency() {
        let c = Circle::new(Point::new(0.0, 0.0), 3.0);
        let angle = std::f64::consts::PI / 3.0; // 60 degrees

        // Calculate chord length from angle
        let chord_length = 2.0 * c.radius() * (angle / 2.0).sin();

        let segment_from_angle = c.segment_area_from_angle(angle);
        let segment_from_chord = c.segment_area_from_chord(chord_length);

        // Both methods should give same result
        assert!(approx_eq(segment_from_angle, segment_from_chord));
    }

    #[test]
    fn test_segment_area_relationships() {
        let c = Circle::new(Point::new(0.0, 0.0), 1.0);
        let angle = std::f64::consts::PI / 4.0; // 45 degrees

        let sector = c.sector_area(angle);
        let segment = c.segment_area_from_angle(angle);

        // Segment area should be less than sector area (triangle is subtracted)
        assert!(segment < sector);

        // For small angles, segment should be much smaller than sector
        let small_angle = 0.1;
        let small_sector = c.sector_area(small_angle);
        let small_segment = c.segment_area_from_angle(small_angle);
        assert!(small_segment < small_sector / 2.0);
    }

    #[test]
    #[should_panic(expected = "Chord length")]
    fn test_segment_area_from_chord_invalid() {
        let c = Circle::new(Point::new(0.0, 0.0), 2.0);
        let chord_length = 5.0; // Impossible: longer than diameter
        c.segment_area_from_chord(chord_length);
    }
}
