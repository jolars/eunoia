//! 2D point representation.

use crate::geometry::shapes::Ellipse;
use crate::geometry::traits::Distance;

/// A point in 2D Cartesian space.
///
/// `Point` represents a location using x and y coordinates. It is used as the
/// foundational type for positioning shapes in the diagram plane.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::primitives::Point;
///
/// let origin = Point::new(0.0, 0.0);
/// let point = Point::new(3.0, 4.0);
/// let dist = origin.distance(&point);
/// # assert_eq!(dist, 5.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    /// Creates a new coordinate at the specified position.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate
    /// * `y` - The y-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let point = Point::new(1.5, 2.5);
    /// ```
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn at_origin() -> Self {
        Point { x: 0.0, y: 0.0 }
    }

    /// Returns the x-coordinate.
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Returns the y-coordinate.
    pub fn y(&self) -> f64 {
        self.y
    }

    pub fn distance(&self, other: &Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    pub fn angle_to(&self, other: &Self) -> f64 {
        (other.y - self.y).atan2(other.x - self.x)
    }

    pub fn angle_from_origin(&self) -> f64 {
        self.y.atan2(self.x)
    }

    pub fn rotate_around(&self, other: &Self, angle: f64) -> Self {
        let (sin_theta, cos_theta) = angle.sin_cos();
        let dx = self.x - other.x;
        let dy = self.y - other.y;

        Self {
            x: other.x + (dx * cos_theta - dy * sin_theta),
            y: other.y + (dx * sin_theta + dy * cos_theta),
        }
    }

    // Convenience method for rotating around origin
    pub fn rotate(&self, angle: f64) -> Self {
        self.rotate_around(&Self::ORIGIN, angle)
    }

    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }

    pub fn reflect_across_x(&self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    pub fn reflect_across_y(&self) -> Self {
        Self {
            x: -self.x,
            y: self.y,
        }
    }

    pub fn reflect_across_origin(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }

    pub fn to_ellipse_frame(self, e: &Ellipse) -> Self {
        self.translate(-e.center().x(), -e.center().y())
            .rotate(-e.rotation())
    }

    pub const ORIGIN: Point = Point { x: 0.0, y: 0.0 };
}

impl Distance for Point {
    fn distance(&self, other: &Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

/// Centroid of a set of points.
pub fn centroid(points: &[Point]) -> Point {
    let (sum_x, sum_y) = points.iter().fold((0.0, 0.0), |(acc_x, acc_y), p| {
        (acc_x + p.x(), acc_y + p.y())
    });
    let n = points.len() as f64;
    Point::new(sum_x / n, sum_y / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_new() {
        let point = Point::new(3.0, 4.0);
        assert_eq!(point.x(), 3.0);
        assert_eq!(point.y(), 4.0);
    }

    #[test]
    fn test_point_getters() {
        let point = Point::new(-2.5, 7.8);
        assert_eq!(point.x(), -2.5);
        assert_eq!(point.y(), 7.8);
    }

    #[test]
    fn test_distance_same_point() {
        let point1 = Point::new(1.0, 1.0);
        let point2 = Point::new(1.0, 1.0);
        assert_eq!(point1.distance(&point2), 0.0);
    }

    #[test]
    fn test_distance_horizontal() {
        let point1 = Point::new(0.0, 0.0);
        let point2 = Point::new(3.0, 0.0);
        assert_eq!(point1.distance(&point2), 3.0);
    }

    #[test]
    fn test_distance_vertical() {
        let point1 = Point::new(0.0, 0.0);
        let point2 = Point::new(0.0, 4.0);
        assert_eq!(point1.distance(&point2), 4.0);
    }

    #[test]
    fn test_distance_pythagorean() {
        let point1 = Point::new(0.0, 0.0);
        let point2 = Point::new(3.0, 4.0);
        assert_eq!(point1.distance(&point2), 5.0);
    }

    #[test]
    fn test_distance_negative_points() {
        let point1 = Point::new(-1.0, -1.0);
        let point2 = Point::new(2.0, 3.0);
        assert_eq!(point1.distance(&point2), 5.0);
    }

    #[test]
    fn test_distance_symmetric() {
        let point1 = Point::new(1.5, 2.5);
        let point2 = Point::new(4.5, 6.5);
        assert_eq!(point1.distance(&point2), point2.distance(&point1));
    }

    #[test]
    fn test_at_origin() {
        let origin = Point::at_origin();
        assert_eq!(origin.x(), 0.0);
        assert_eq!(origin.y(), 0.0);
    }

    #[test]
    fn test_rotate_around_origin_90_degrees() {
        let point = Point::new(1.0, 0.0);
        let origin = Point::new(0.0, 0.0);
        let rotated = point.rotate_around(&origin, std::f64::consts::FRAC_PI_2);
        assert!((rotated.x() - 0.0).abs() < 1e-10);
        assert!((rotated.y() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_around_center() {
        let point = Point::new(2.0, 1.0);
        let center = Point::new(1.0, 1.0);
        let rotated = point.rotate_around(&center, std::f64::consts::PI);
        assert!((rotated.x() - 0.0).abs() < 1e-10);
        assert!((rotated.y() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_360_degrees() {
        let point = Point::new(3.5, 2.5);
        let center = Point::new(1.0, 1.0);
        let rotated = point.rotate_around(&center, 2.0 * std::f64::consts::PI);
        assert!((rotated.x() - point.x()).abs() < 1e-10);
        assert!((rotated.y() - point.y()).abs() < 1e-10);
    }

    #[test]
    fn test_translate() {
        let point = Point::new(1.0, 2.0);
        let translated = point.translate(3.0, 4.0);
        assert_eq!(translated.x(), 4.0);
        assert_eq!(translated.y(), 6.0);
    }

    #[test]
    fn test_translate_negative() {
        let point = Point::new(5.0, 5.0);
        let translated = point.translate(-2.0, -3.0);
        assert_eq!(translated.x(), 3.0);
        assert_eq!(translated.y(), 2.0);
    }

    #[test]
    fn test_scale() {
        let point = Point::new(2.0, 3.0);
        let scaled = point.scale(2.0);
        assert_eq!(scaled.x(), 4.0);
        assert_eq!(scaled.y(), 6.0);
    }

    #[test]
    fn test_scale_fractional() {
        let point = Point::new(10.0, 20.0);
        let scaled = point.scale(0.5);
        assert_eq!(scaled.x(), 5.0);
        assert_eq!(scaled.y(), 10.0);
    }

    #[test]
    fn test_scale_negative() {
        let point = Point::new(3.0, 4.0);
        let scaled = point.scale(-1.0);
        assert_eq!(scaled.x(), -3.0);
        assert_eq!(scaled.y(), -4.0);
    }
}
