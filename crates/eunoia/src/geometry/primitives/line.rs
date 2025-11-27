//! Line representation.

use crate::geometry::primitives::Point;

/// An infinite line in 2D space.
///
/// `Line` represents an infinite line defined by a point and a direction (slope and intercept,
/// or vertical line). It provides operations for calculating distance from points, intersections,
/// and other geometric properties.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::primitives::Line;
/// use eunoia::geometry::primitives::Point;
///
/// let line = Line::from_points(Point::new(0.0, 0.0), Point::new(1.0, 2.0));
/// # assert_eq!(line.slope(), Some(2.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Line {
    /// A non-vertical line: y = mx + b
    SlopeIntercept { slope: f64, intercept: f64 },
    /// A vertical line: x = c
    Vertical { x: f64 },
}

impl Line {
    /// Creates a line from two points.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::primitives::Line;
    /// use eunoia::geometry::primitives::Point;
    ///
    /// let line = Line::from_points(Point::new(0.0, 0.0), Point::new(2.0, 4.0));
    /// # assert_eq!(line.slope(), Some(2.0));
    /// ```
    pub fn from_points(p1: Point, p2: Point) -> Self {
        let dx = p2.x() - p1.x();
        if dx.abs() < f64::EPSILON {
            Line::Vertical { x: p1.x() }
        } else {
            let slope = (p2.y() - p1.y()) / dx;
            let intercept = p1.y() - slope * p1.x();
            Line::SlopeIntercept { slope, intercept }
        }
    }

    /// Creates a line from slope and y-intercept.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::primitives::Line;
    ///
    /// let line = Line::from_slope_intercept(2.0, 3.0);
    /// # assert_eq!(line.slope(), Some(2.0));
    /// ```
    pub fn from_slope_intercept(slope: f64, intercept: f64) -> Self {
        Line::SlopeIntercept { slope, intercept }
    }

    /// Creates a vertical line at the given x-coordinate.
    pub fn vertical(x: f64) -> Self {
        Line::Vertical { x }
    }

    /// Returns the slope of the line, or `None` if vertical.
    pub fn slope(&self) -> Option<f64> {
        match self {
            Line::SlopeIntercept { slope, .. } => Some(*slope),
            Line::Vertical { .. } => None,
        }
    }

    /// Returns the y-intercept of the line, or `None` if vertical.
    pub fn y_intercept(&self) -> Option<f64> {
        match self {
            Line::SlopeIntercept { intercept, .. } => Some(*intercept),
            Line::Vertical { .. } => None,
        }
    }

    /// Checks if the line contains a given point.
    pub fn contains(&self, point: &Point) -> bool {
        match self {
            Line::SlopeIntercept { slope, intercept } => {
                let expected_y = slope * point.x() + intercept;
                (point.y() - expected_y).abs() < f64::EPSILON
            }
            Line::Vertical { x } => (point.x() - x).abs() < f64::EPSILON,
        }
    }

    /// Checks if this line is parallel to another line.
    pub fn is_parallel(&self, other: &Line) -> bool {
        match (self, other) {
            (Line::Vertical { .. }, Line::Vertical { .. }) => true,
            (Line::SlopeIntercept { slope: s1, .. }, Line::SlopeIntercept { slope: s2, .. }) => {
                (s1 - s2).abs() < f64::EPSILON
            }
            _ => false,
        }
    }

    /// Checks if this line is perpendicular to another line.
    pub fn is_perpendicular(&self, other: &Line) -> bool {
        match (self, other) {
            (Line::Vertical { .. }, Line::SlopeIntercept { slope, .. }) => {
                slope.abs() < f64::EPSILON
            }
            (Line::SlopeIntercept { slope, .. }, Line::Vertical { .. }) => {
                slope.abs() < f64::EPSILON
            }
            (Line::SlopeIntercept { slope: s1, .. }, Line::SlopeIntercept { slope: s2, .. }) => {
                (s1 * s2 + 1.0).abs() < f64::EPSILON
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_from_points() {
        let line = Line::from_points(Point::new(0.0, 0.0), Point::new(2.0, 4.0));
        assert_eq!(line.slope(), Some(2.0));
        assert_eq!(line.y_intercept(), Some(0.0));
    }

    #[test]
    fn test_line_from_points_with_intercept() {
        let line = Line::from_points(Point::new(0.0, 3.0), Point::new(2.0, 7.0));
        assert_eq!(line.slope(), Some(2.0));
        assert_eq!(line.y_intercept(), Some(3.0));
    }

    #[test]
    fn test_line_vertical() {
        let line = Line::from_points(Point::new(5.0, 0.0), Point::new(5.0, 10.0));
        assert_eq!(line.slope(), None);
        assert_eq!(line.y_intercept(), None);
    }

    #[test]
    fn test_line_from_slope_intercept() {
        let line = Line::from_slope_intercept(2.0, 3.0);
        assert_eq!(line.slope(), Some(2.0));
        assert_eq!(line.y_intercept(), Some(3.0));
    }

    #[test]
    fn test_line_contains() {
        let line = Line::from_slope_intercept(2.0, 1.0);
        assert!(line.contains(&Point::new(0.0, 1.0)));
        assert!(line.contains(&Point::new(1.0, 3.0)));
        assert!(line.contains(&Point::new(2.0, 5.0)));
        assert!(!line.contains(&Point::new(1.0, 1.0)));
    }

    #[test]
    fn test_line_vertical_contains() {
        let line = Line::vertical(5.0);
        assert!(line.contains(&Point::new(5.0, 0.0)));
        assert!(line.contains(&Point::new(5.0, 100.0)));
        assert!(!line.contains(&Point::new(4.0, 0.0)));
    }

    #[test]
    fn test_is_parallel() {
        let line1 = Line::from_slope_intercept(2.0, 1.0);
        let line2 = Line::from_slope_intercept(2.0, 5.0);
        let line3 = Line::from_slope_intercept(3.0, 1.0);
        assert!(line1.is_parallel(&line2));
        assert!(!line1.is_parallel(&line3));
    }

    #[test]
    fn test_is_parallel_vertical() {
        let line1 = Line::vertical(1.0);
        let line2 = Line::vertical(5.0);
        let line3 = Line::from_slope_intercept(2.0, 1.0);
        assert!(line1.is_parallel(&line2));
        assert!(!line1.is_parallel(&line3));
    }

    #[test]
    fn test_is_perpendicular() {
        let line1 = Line::from_slope_intercept(2.0, 1.0);
        let line2 = Line::from_slope_intercept(-0.5, 3.0);
        let line3 = Line::from_slope_intercept(2.0, 5.0);
        assert!(line1.is_perpendicular(&line2));
        assert!(!line1.is_perpendicular(&line3));
    }

    #[test]
    fn test_is_perpendicular_vertical() {
        let vertical = Line::vertical(1.0);
        let horizontal = Line::from_slope_intercept(0.0, 5.0);
        let diagonal = Line::from_slope_intercept(2.0, 1.0);
        assert!(vertical.is_perpendicular(&horizontal));
        assert!(!vertical.is_perpendicular(&diagonal));
    }
}
