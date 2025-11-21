use crate::geometry::coord::Coord;
use crate::geometry::operations::Area;
use crate::geometry::operations::Contains;
use crate::geometry::operations::Distance;
use crate::geometry::operations::IntersectionArea;
use crate::geometry::operations::Intersects;

pub struct Circle {
    center: Coord,
    radius: f64,
}

impl Area for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

impl Distance for Circle {
    fn distance(&self, other: &Self) -> f64 {
        let center_distance = self.center.distance(&other.center);
        let radius_sum = self.radius + other.radius;

        if center_distance > radius_sum {
            center_distance - radius_sum
        } else {
            0.0
        }
    }
}

impl Contains for Circle {
    fn contains(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance + other.radius <= self.radius
    }
}

impl Intersects for Circle {
    fn intersects(&self, other: &Self) -> bool {
        let center_distance = self.center.distance(&other.center);
        center_distance >= self.radius + other.radius
    }
}

impl IntersectionArea for Circle {
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

impl Circle {
    pub fn new(center: Coord, radius: f64) -> Self {
        Circle { center, radius }
    }

    pub fn center(&self) -> &Coord {
        &self.center
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
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
}
