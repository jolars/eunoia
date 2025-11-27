//! Axis-aligned rectangle shape implementation.

use crate::geometry::primitives::Point;
use crate::geometry::traits::{Area, BoundingBox, Centroid, Closed, Distance, Perimeter};

/// An axis-aligned rectangle defined by a center point, width, and height.
///
/// The rectangle's edges are parallel to the x and y axes. This simplifies
/// many geometric computations compared to rotated rectangles.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::shapes::Rectangle;
/// use eunoia::geometry::traits::Area;
/// use eunoia::geometry::traits::Closed;
/// use eunoia::geometry::primitives::Point;
///
/// let r1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);
/// let r2 = Rectangle::new(Point::new(3.0, 0.0), 2.0, 3.0);
///
/// let area1 = r1.area();
/// let overlap = r1.intersection_area(&r2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    center: Point,
    width: f64,
    height: f64,
}

#[allow(dead_code)]
impl Rectangle {
    /// Creates a new axis-aligned rectangle with the specified center, width, and height.
    ///
    /// # Arguments
    ///
    /// * `center` - The center point of the rectangle
    /// * `width` - The width of the rectangle (must be positive)
    /// * `height` - The height of the rectangle (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Rectangle;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let rect = Rectangle::new(Point::new(1.0, 2.0), 4.0, 3.0);
    /// ```
    pub fn new(center: Point, width: f64, height: f64) -> Self {
        Rectangle {
            center,
            width,
            height,
        }
    }

    /// Create a rectangle from two corner points (bottom-left and top-right).
    pub fn from_corners(bottom_left: Point, top_right: Point) -> Self {
        let center_x = (bottom_left.x() + top_right.x()) / 2.0;
        let center_y = (bottom_left.y() + top_right.y()) / 2.0;

        let width = top_right.x() - bottom_left.x();
        let height = top_right.y() - bottom_left.y();

        Rectangle::new(Point::new(center_x, center_y), width, height)
    }

    /// Returns a reference to the rectangle's center point.
    pub fn center(&self) -> &Point {
        &self.center
    }

    /// Returns the rectangle's width.
    pub fn width(&self) -> f64 {
        self.width
    }

    /// Returns the rectangle's height.
    pub fn height(&self) -> f64 {
        self.height
    }

    /// Sets the center of the rectangle.
    pub fn set_center(&mut self, center: Point) {
        self.center = center;
    }

    /// Returns the bounds of the rectangle as (x_min, x_max, y_min, y_max).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;
        (
            self.center.x() - half_width,
            self.center.x() + half_width,
            self.center.y() - half_height,
            self.center.y() + half_height,
        )
    }

    /// Returns the bottom-left and top-right corner points of the rectangle.
    pub fn to_points(self) -> (Point, Point) {
        let (x_min, x_max, y_min, y_max) = self.bounds();
        (Point::new(x_min, y_min), Point::new(x_max, y_max))
    }

    /// Returns the four corner points of the rectangle.
    pub fn corners(&self) -> [Point; 4] {
        let (x_min, x_max, y_min, y_max) = self.bounds();
        [
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ]
    }
}

impl Area for Rectangle {
    /// Computes the area of the rectangle using the formula A = width × height.
    fn area(&self) -> f64 {
        self.width * self.height
    }
}

impl Perimeter for Rectangle {
    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
}

impl BoundingBox for Rectangle {
    fn bounding_box(&self) -> Rectangle {
        *self
    }
}

impl Centroid for Rectangle {
    /// Returns the centroid (center point) of the rectangle.
    fn centroid(&self) -> (f64, f64) {
        (self.center.x(), self.center.y())
    }
}

impl Distance for Rectangle {
    /// Computes the minimum distance between the boundaries of two rectangles.
    ///
    /// Returns 0.0 if the rectangles overlap or touch.
    fn distance(&self, other: &Self) -> f64 {
        let (x1_min, x1_max, y1_min, y1_max) = self.bounds();
        let (x2_min, x2_max, y2_min, y2_max) = other.bounds();

        let dx = if x1_max < x2_min {
            x2_min - x1_max
        } else if x2_max < x1_min {
            x1_min - x2_max
        } else {
            0.0
        };

        let dy = if y1_max < y2_min {
            y2_min - y1_max
        } else if y2_max < y1_min {
            y1_min - y2_max
        } else {
            0.0
        };

        (dx * dx + dy * dy).sqrt()
    }
}

#[allow(dead_code)]
impl Closed for Rectangle {
    fn contains(&self, other: &Self) -> bool {
        let (x1_min, x1_max, y1_min, y1_max) = self.bounds();
        let (x2_min, x2_max, y2_min, y2_max) = other.bounds();

        x2_min >= x1_min && x2_max <= x1_max && y2_min >= y1_min && y2_max <= y1_max
    }

    /// Checks if a point is inside the rectangle (including the boundary).
    fn contains_point(&self, point: &Point) -> bool {
        let (x_min, x_max, y_min, y_max) = self.bounds();
        point.x() >= x_min && point.x() <= x_max && point.y() >= y_min && point.y() <= y_max
    }

    fn intersects(&self, other: &Self) -> bool {
        let (x1_min, x1_max, y1_min, y1_max) = self.bounds();
        let (x2_min, x2_max, y2_min, y2_max) = other.bounds();

        !(x1_max < x2_min || x2_max < x1_min || y1_max < y2_min || y2_max < y1_min)
    }

    /// Computes the area of intersection between two axis-aligned rectangles.
    ///
    /// Returns 0 if rectangles don't overlap.
    fn intersection_area(&self, other: &Self) -> f64 {
        let (x1_min, x1_max, y1_min, y1_max) = self.bounds();
        let (x2_min, x2_max, y2_min, y2_max) = other.bounds();

        let x_overlap = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0);
        let y_overlap = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0);

        x_overlap * y_overlap
    }

    /// Computes the intersection points between two rectangles.
    ///
    /// For axis-aligned rectangles, intersection points are at the corners
    /// of the overlapping region.
    fn intersection_points(&self, other: &Self) -> Vec<Point> {
        if !self.intersects(other) {
            return vec![];
        }

        let (x1_min, x1_max, y1_min, y1_max) = self.bounds();
        let (x2_min, x2_max, y2_min, y2_max) = other.bounds();

        let x_min = x1_min.max(x2_min);
        let x_max = x1_max.min(x2_max);
        let y_min = y1_min.max(y2_min);
        let y_max = y1_max.min(y2_max);

        if x_min >= x_max || y_min >= y_max {
            return vec![];
        }

        vec![
            Point::new(x_min, y_min),
            Point::new(x_max, y_min),
            Point::new(x_max, y_max),
            Point::new(x_min, y_max),
        ]
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
    fn test_rectangle_new() {
        let center = Point::new(1.0, 2.0);
        let rect = Rectangle::new(center, 4.0, 3.0);
        assert_eq!(rect.width(), 4.0);
        assert_eq!(rect.height(), 3.0);
        assert_eq!(rect.center().x(), 1.0);
        assert_eq!(rect.center().y(), 2.0);
    }

    #[test]
    fn test_rectangle_area() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        assert!(approx_eq(rect.area(), 12.0));

        let rect2 = Rectangle::new(Point::new(5.0, 5.0), 2.0, 5.0);
        assert!(approx_eq(rect2.area(), 10.0));
    }

    #[test]
    fn test_rectangle_perimeter() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        assert!(approx_eq(rect.perimeter(), 14.0));

        let rect2 = Rectangle::new(Point::new(1.0, 1.0), 2.0, 2.0);
        assert!(approx_eq(rect2.perimeter(), 8.0));
    }

    #[test]
    fn test_rectangle_bounds() {
        let rect = Rectangle::new(Point::new(2.0, 3.0), 4.0, 6.0);
        let (x_min, x_max, y_min, y_max) = rect.bounds();
        assert!(approx_eq(x_min, 0.0));
        assert!(approx_eq(x_max, 4.0));
        assert!(approx_eq(y_min, 0.0));
        assert!(approx_eq(y_max, 6.0));
    }

    #[test]
    fn test_rectangle_corners() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let corners = rect.corners();
        assert_eq!(corners.len(), 4);

        let expected = [
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ];

        for (corner, &expected_corner) in corners.iter().zip(expected.iter()) {
            assert!(approx_eq(corner.x(), expected_corner.x()));
            assert!(approx_eq(corner.y(), expected_corner.y()));
        }
    }

    #[test]
    fn test_rectangle_centroid() {
        let rect = Rectangle::new(Point::new(3.0, 4.0), 2.0, 2.0);
        let (cx, cy) = rect.centroid();
        assert!(approx_eq(cx, 3.0));
        assert!(approx_eq(cy, 4.0));
    }

    #[test]
    fn test_rectangle_contains_point() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 4.0, 2.0);

        assert!(rect.contains_point(&Point::new(0.0, 0.0)));
        assert!(rect.contains_point(&Point::new(1.0, 0.5)));
        assert!(rect.contains_point(&Point::new(-1.0, -0.5)));
        assert!(rect.contains_point(&Point::new(2.0, 1.0)));

        assert!(!rect.contains_point(&Point::new(3.0, 0.0)));
        assert!(!rect.contains_point(&Point::new(0.0, 2.0)));
    }

    #[test]
    fn test_rectangle_distance_no_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (4, 6, -1, 1)
        // Distance between edges: 4 - 1 = 3
        assert!(approx_eq(rect1.distance(&rect2), 3.0));
    }

    #[test]
    fn test_rectangle_distance_touching() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(3.0, 0.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (2, 4, -1, 1)
        // Distance between edges: 2 - 1 = 1
        assert!(approx_eq(rect1.distance(&rect2), 1.0));
    }

    #[test]
    fn test_rectangle_distance_overlapping() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        assert!(approx_eq(rect1.distance(&rect2), 0.0));
    }

    #[test]
    fn test_rectangle_distance_diagonal() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 5.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (4, 6, 4, 6)
        // dx = 4 - 1 = 3, dy = 4 - 1 = 3
        let expected = ((3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
        assert!(approx_eq(rect1.distance(&rect2), expected));
    }

    #[test]
    fn test_rectangle_contains_smaller() {
        let large = Rectangle::new(Point::new(0.0, 0.0), 10.0, 10.0);
        let small = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        assert!(large.contains(&small));
    }

    #[test]
    fn test_rectangle_contains_self() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), 3.0, 3.0);
        assert!(rect.contains(&rect));
    }

    #[test]
    fn test_rectangle_not_contains() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);
        assert!(!rect1.contains(&rect2));
    }

    #[test]
    fn test_rectangle_not_contains_partial_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 3.0, 3.0);
        assert!(!rect1.contains(&rect2));
    }

    #[test]
    fn test_rectangle_intersects_separate() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);
        assert!(!rect1.intersects(&rect2));
    }

    #[test]
    fn test_rectangle_intersects_touching() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 2.0, 2.0);
        // rect1 bounds: (-1, 1, -1, 1), rect2 bounds: (1, 3, -1, 1)
        // They touch at x=1, so they intersect
        assert!(rect1.intersects(&rect2));
    }

    #[test]
    fn test_rectangle_intersects_overlapping() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(1.0, 0.0), 2.0, 2.0);
        assert!(rect1.intersects(&rect2));
    }

    #[test]
    fn test_intersection_area_no_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(10.0, 0.0), 2.0, 2.0);
        assert!(approx_eq(rect1.intersection_area(&rect2), 0.0));
    }

    #[test]
    fn test_intersection_area_touching() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(3.0, 0.0), 2.0, 2.0);
        assert!(approx_eq(rect1.intersection_area(&rect2), 0.0));
    }

    #[test]
    fn test_intersection_area_complete_overlap_same_size() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        let rect2 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        assert!(approx_eq(rect1.intersection_area(&rect2), 12.0));
    }

    #[test]
    fn test_intersection_area_one_inside_other() {
        let large = Rectangle::new(Point::new(0.0, 0.0), 10.0, 10.0);
        let small = Rectangle::new(Point::new(1.0, 0.0), 4.0, 4.0);
        let expected = 16.0;
        assert!(approx_eq(large.intersection_area(&small), expected));
        assert!(approx_eq(small.intersection_area(&large), expected));
    }

    #[test]
    fn test_intersection_area_partial_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 4.0, 4.0);

        let x_overlap = 2.0;
        let y_overlap = 4.0;
        let expected = x_overlap * y_overlap;

        assert!(approx_eq(rect1.intersection_area(&rect2), expected));
    }

    #[test]
    fn test_intersection_area_symmetric() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 3.0);
        let rect2 = Rectangle::new(Point::new(1.5, 0.0), 3.0, 2.0);
        let area1 = rect1.intersection_area(&rect2);
        let area2 = rect2.intersection_area(&rect1);
        assert!(approx_eq(area1, area2));
    }

    #[test]
    fn test_intersection_area_different_sizes() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 6.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 2.0, 2.0);
        let area = rect1.intersection_area(&rect2);

        assert!(area > 0.0);
        assert!(area <= 4.0);
    }

    #[test]
    fn test_intersection_points_no_intersection() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        let rect2 = Rectangle::new(Point::new(5.0, 0.0), 2.0, 2.0);

        let points = rect1.intersection_points(&rect2);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_intersection_points_overlapping() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(1.0, 1.0), 2.0, 2.0);

        let points = rect1.intersection_points(&rect2);
        assert_eq!(points.len(), 4);

        for point in &points {
            assert!(rect1.contains_point(point));
            assert!(rect2.contains_point(point));
        }
    }

    #[test]
    fn test_intersection_points_partial_overlap() {
        let rect1 = Rectangle::new(Point::new(0.0, 0.0), 4.0, 4.0);
        let rect2 = Rectangle::new(Point::new(2.0, 0.0), 4.0, 4.0);

        let points = rect1.intersection_points(&rect2);
        assert_eq!(points.len(), 4);

        for point in &points {
            assert!(rect1.contains_point(point));
            assert!(rect2.contains_point(point));
        }
    }

    #[test]
    fn test_set_center() {
        let mut rect = Rectangle::new(Point::new(0.0, 0.0), 2.0, 2.0);
        rect.set_center(Point::new(5.0, 3.0));
        assert_eq!(rect.center().x(), 5.0);
        assert_eq!(rect.center().y(), 3.0);
    }
}
