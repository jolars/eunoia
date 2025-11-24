//! 2D point representation.

/// A point in 2D Cartesian space.
///
/// `Point` represents a location using x and y coordinates. It is used as the
/// foundational type for positioning shapes in the diagram plane.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::point::Point;
///
/// let origin = Point::new(0.0, 0.0);
/// let point = Point::new(3.0, 4.0);
/// let dist = origin.distance(&point);
/// assert_eq!(dist, 5.0);
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
    /// use eunoia::geometry::point::Point;
    ///
    /// let point = Point::new(1.5, 2.5);
    /// ```
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
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
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_new() {
        let coord = Point::new(3.0, 4.0);
        assert_eq!(coord.x(), 3.0);
        assert_eq!(coord.y(), 4.0);
    }

    #[test]
    fn test_coord_getters() {
        let coord = Point::new(-2.5, 7.8);
        assert_eq!(coord.x(), -2.5);
        assert_eq!(coord.y(), 7.8);
    }

    #[test]
    fn test_distance_same_point() {
        let coord1 = Point::new(1.0, 1.0);
        let coord2 = Point::new(1.0, 1.0);
        assert_eq!(coord1.distance(&coord2), 0.0);
    }

    #[test]
    fn test_distance_horizontal() {
        let coord1 = Point::new(0.0, 0.0);
        let coord2 = Point::new(3.0, 0.0);
        assert_eq!(coord1.distance(&coord2), 3.0);
    }

    #[test]
    fn test_distance_vertical() {
        let coord1 = Point::new(0.0, 0.0);
        let coord2 = Point::new(0.0, 4.0);
        assert_eq!(coord1.distance(&coord2), 4.0);
    }

    #[test]
    fn test_distance_pythagorean() {
        let coord1 = Point::new(0.0, 0.0);
        let coord2 = Point::new(3.0, 4.0);
        assert_eq!(coord1.distance(&coord2), 5.0);
    }

    #[test]
    fn test_distance_negative_coords() {
        let coord1 = Point::new(-1.0, -1.0);
        let coord2 = Point::new(2.0, 3.0);
        assert_eq!(coord1.distance(&coord2), 5.0);
    }

    #[test]
    fn test_distance_symmetric() {
        let coord1 = Point::new(1.5, 2.5);
        let coord2 = Point::new(4.5, 6.5);
        assert_eq!(coord1.distance(&coord2), coord2.distance(&coord1));
    }
}
