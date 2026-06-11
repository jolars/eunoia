//! Axis-aligned bounding box primitive.

use crate::geometry::primitives::Point;

/// An axis-aligned bounding box, described by its extents along each axis.
///
/// This is the canonical bounding-box type: it is what
/// [`BoundingBox::bounds`](crate::geometry::traits::BoundingBox::bounds)
/// returns for every shape. The fields are public and named so callers bind
/// them by name rather than by position — destructuring
/// `let Bounds { x_min, x_max, y_min, y_max } = shape.bounds();` is as terse as
/// a tuple but immune to the silent axis/min-max mix-ups a bare
/// `(f64, f64, f64, f64)` invites.
///
/// A degenerate box where `x_min == x_max` or `y_min == y_max` (zero width or
/// height) is permitted by design, mirroring the zero-sized bounding boxes that
/// [`Rectangle`](crate::geometry::shapes::Rectangle) accepts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Smallest x coordinate (left edge).
    pub x_min: f64,
    /// Largest x coordinate (right edge).
    pub x_max: f64,
    /// Smallest y coordinate (bottom edge).
    pub y_min: f64,
    /// Largest y coordinate (top edge).
    pub y_max: f64,
}

impl Bounds {
    /// Creates a bounding box from its four extents.
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Bounds {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Width of the box (`x_max - x_min`).
    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    /// Height of the box (`y_max - y_min`).
    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    /// Area of the box (`width · height`).
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Returns the bottom-left and top-right corner points.
    pub fn to_points(&self) -> (Point, Point) {
        (
            Point::new(self.x_min, self.y_min),
            Point::new(self.x_max, self.y_max),
        )
    }

    /// Smallest box containing both `self` and `other`.
    pub fn union(&self, other: &Bounds) -> Bounds {
        Bounds {
            x_min: self.x_min.min(other.x_min),
            x_max: self.x_max.max(other.x_max),
            y_min: self.y_min.min(other.y_min),
            y_max: self.y_max.max(other.y_max),
        }
    }

    /// Overlap of `self` and `other`, or `None` if they are disjoint.
    pub fn intersection(&self, other: &Bounds) -> Option<Bounds> {
        let x_min = self.x_min.max(other.x_min);
        let x_max = self.x_max.min(other.x_max);
        let y_min = self.y_min.max(other.y_min);
        let y_max = self.y_max.min(other.y_max);
        (x_min < x_max && y_min < y_max).then_some(Bounds {
            x_min,
            x_max,
            y_min,
            y_max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_and_dimensions() {
        let bb = Bounds::new(-1.0, 3.0, 0.0, 5.0);
        assert_eq!(bb.x_min, -1.0);
        assert_eq!(bb.x_max, 3.0);
        assert_eq!(bb.y_min, 0.0);
        assert_eq!(bb.y_max, 5.0);
        assert_eq!(bb.width(), 4.0);
        assert_eq!(bb.height(), 5.0);
        assert_eq!(bb.area(), 20.0);
    }

    #[test]
    fn to_points_returns_corners() {
        let bb = Bounds::new(-1.0, 3.0, 0.0, 5.0);
        let (bl, tr) = bb.to_points();
        assert_eq!(bl, Point::new(-1.0, 0.0));
        assert_eq!(tr, Point::new(3.0, 5.0));
    }

    #[test]
    fn union_covers_both() {
        let a = Bounds::new(0.0, 2.0, 0.0, 2.0);
        let b = Bounds::new(1.0, 4.0, -1.0, 1.0);
        assert_eq!(a.union(&b), Bounds::new(0.0, 4.0, -1.0, 2.0));
    }

    #[test]
    fn intersection_overlap_and_disjoint() {
        let a = Bounds::new(0.0, 2.0, 0.0, 2.0);
        let b = Bounds::new(1.0, 4.0, 1.0, 3.0);
        assert_eq!(a.intersection(&b), Some(Bounds::new(1.0, 2.0, 1.0, 2.0)));

        let c = Bounds::new(5.0, 6.0, 5.0, 6.0);
        assert_eq!(a.intersection(&c), None);
    }
}
