//! Line and line segment representation.

use super::point::Point;

/// An infinite line in 2D space.
///
/// `Line` represents an infinite line defined by a point and a direction (slope and intercept,
/// or vertical line). It provides operations for calculating distance from points, intersections,
/// and other geometric properties.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::line::Line;
/// use eunoia::geometry::point::Point;
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
    /// use eunoia::geometry::line::Line;
    /// use eunoia::geometry::point::Point;
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
    /// use eunoia::geometry::line::Line;
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

/// A line segment in 2D space defined by two endpoints.
///
/// `LineSegment` represents a finite line segment between two points. It provides
/// operations for calculating length, midpoint, and other geometric properties.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::line::LineSegment;
/// use eunoia::geometry::point::Point;
///
/// let start = Point::new(0.0, 0.0);
/// let end = Point::new(3.0, 4.0);
/// let segment = LineSegment::new(start, end);
/// # assert_eq!(segment.length(), 5.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSegment {
    start: Point,
    end: Point,
}

impl LineSegment {
    /// Creates a new line segment from two points.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting point of the line segment
    /// * `end` - The ending point of the line segment
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::line::LineSegment;
    /// use eunoia::geometry::point::Point;
    ///
    /// let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
    /// ```
    pub fn new(start: Point, end: Point) -> Self {
        LineSegment { start, end }
    }

    /// Returns the starting point of the line segment.
    pub fn start(&self) -> Point {
        self.start
    }

    /// Returns the ending point of the line segment.
    pub fn end(&self) -> Point {
        self.end
    }

    /// Calculates the length of the line segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::line::LineSegment;
    /// use eunoia::geometry::point::Point;
    ///
    /// let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
    /// # assert_eq!(segment.length(), 5.0);
    /// ```
    pub fn length(&self) -> f64 {
        self.start.distance(&self.end)
    }

    /// Calculates the midpoint of the line segment.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::line::LineSegment;
    /// use eunoia::geometry::point::Point;
    ///
    /// let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(4.0, 6.0));
    /// let mid = segment.midpoint();
    /// # assert_eq!(mid, Point::new(2.0, 3.0));
    /// ```
    pub fn midpoint(&self) -> Point {
        Point::new(
            (self.start.x() + self.end.x()) / 2.0,
            (self.start.y() + self.end.y()) / 2.0,
        )
    }

    /// Returns the squared length of the line segment.
    ///
    /// This is more efficient than `length()` when you only need to compare
    /// lengths, as it avoids the square root calculation.
    pub fn length_squared(&self) -> f64 {
        let dx = self.end.x() - self.start.x();
        let dy = self.end.y() - self.start.y();
        dx * dx + dy * dy
    }

    /// Calculates the slope of the line segment.
    ///
    /// Returns `None` if the line is vertical (infinite slope).
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::line::LineSegment;
    /// use eunoia::geometry::point::Point;
    ///
    /// let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(2.0, 4.0));
    /// # assert_eq!(segment.slope(), Some(2.0));
    /// ```
    pub fn slope(&self) -> Option<f64> {
        let dx = self.end.x() - self.start.x();
        if dx.abs() < f64::EPSILON {
            None // Vertical line
        } else {
            let dy = self.end.y() - self.start.y();
            Some(dy / dx)
        }
    }

    /// Reverses the direction of the line segment.
    ///
    /// Returns a new line with start and end points swapped.
    pub fn reverse(&self) -> Self {
        LineSegment::new(self.end, self.start)
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

    #[test]
    fn test_segment_new() {
        let start = Point::new(1.0, 2.0);
        let end = Point::new(3.0, 4.0);
        let segment = LineSegment::new(start, end);
        assert_eq!(segment.start(), start);
        assert_eq!(segment.end(), end);
    }

    #[test]
    fn test_length() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert_eq!(segment.length(), 5.0);
    }

    #[test]
    fn test_length_horizontal() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(5.0, 0.0));
        assert_eq!(segment.length(), 5.0);
    }

    #[test]
    fn test_length_vertical() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(0.0, 7.0));
        assert_eq!(segment.length(), 7.0);
    }

    #[test]
    fn test_midpoint() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(4.0, 6.0));
        let mid = segment.midpoint();
        assert_eq!(mid.x(), 2.0);
        assert_eq!(mid.y(), 3.0);
    }

    #[test]
    fn test_midpoint_negative() {
        let segment = LineSegment::new(Point::new(-2.0, -4.0), Point::new(2.0, 4.0));
        let mid = segment.midpoint();
        assert_eq!(mid.x(), 0.0);
        assert_eq!(mid.y(), 0.0);
    }

    #[test]
    fn test_length_squared() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert_eq!(segment.length_squared(), 25.0);
    }

    #[test]
    fn test_slope_positive() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(2.0, 4.0));
        assert_eq!(segment.slope(), Some(2.0));
    }

    #[test]
    fn test_slope_negative() {
        let segment = LineSegment::new(Point::new(0.0, 4.0), Point::new(2.0, 0.0));
        assert_eq!(segment.slope(), Some(-2.0));
    }

    #[test]
    fn test_slope_horizontal() {
        let segment = LineSegment::new(Point::new(0.0, 5.0), Point::new(10.0, 5.0));
        assert_eq!(segment.slope(), Some(0.0));
    }

    #[test]
    fn test_slope_vertical() {
        let segment = LineSegment::new(Point::new(5.0, 0.0), Point::new(5.0, 10.0));
        assert_eq!(segment.slope(), None);
    }

    #[test]
    fn test_reverse() {
        let start = Point::new(1.0, 2.0);
        let end = Point::new(3.0, 4.0);
        let segment = LineSegment::new(start, end);
        let reversed = segment.reverse();
        assert_eq!(reversed.start(), end);
        assert_eq!(reversed.end(), start);
    }

    #[test]
    fn test_reverse_length_unchanged() {
        let segment = LineSegment::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        let reversed = segment.reverse();
        assert_eq!(segment.length(), reversed.length());
    }
}
