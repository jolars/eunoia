//! Line segment representation.

use crate::geometry::point::Point;

/// A line segment in 2D space defined by two endpoints.
///
/// `LineSegment` represents a finite line segment between two points. It provides
/// operations for calculating length, midpoint, and other geometric properties.
///
/// # Examples
///
/// ```
/// use eunoia::geometry::line_segment::LineSegment;
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
    /// use eunoia::geometry::line_segment::LineSegment;
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
    /// use eunoia::geometry::line_segment::LineSegment;
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
    /// use eunoia::geometry::line_segment::LineSegment;
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
    /// use eunoia::geometry::line_segment::LineSegment;
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
