//! Geometric operation traits.
//!
//! This module defines traits for common geometric operations that can be
//! implemented by various shape types.

use crate::geometry::coord::Coord;

/// Trait for computing the area of a shape.
pub trait Area {
    /// Returns the area of the shape in square units.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::circle::Circle;
    /// use eunoia::geometry::coord::Coord;
    /// use eunoia::geometry::operations::Area;
    ///
    /// let circle = Circle::new(Coord::new(0.0, 0.0), 1.0);
    /// let area = circle.area();
    /// assert!((area - std::f64::consts::PI).abs() < 1e-10);
    /// ```
    fn area(&self) -> f64;
}

/// Trait for computing the centroid (geometric center) of a shape.
pub trait Centroid {
    /// Returns the centroid of the shape.
    ///
    /// For simple shapes, this is often the geometric center. For complex
    /// polygons, this is the center of mass assuming uniform density.
    fn centroid(&self) -> Coord;
}

/// Trait for checking if two shapes intersect.
pub trait Intersects {
    /// Returns `true` if this shape intersects with another shape.
    ///
    /// Two shapes intersect if they share any common points, including
    /// touching at a single point or edge.
    ///
    /// # Arguments
    ///
    /// * `other` - The other shape to check intersection with
    fn intersects(&self, other: &Self) -> bool;
}

/// Trait for checking if one shape contains another.
pub trait Contains {
    /// Returns `true` if this shape completely contains the other shape.
    ///
    /// A shape contains another if all points of the other shape lie within
    /// or on the boundary of this shape.
    ///
    /// # Arguments
    ///
    /// * `other` - The shape to check for containment
    fn contains(&self, other: &Self) -> bool;
}

/// Trait for computing the distance between two geometric objects.
pub trait Distance {
    /// Computes the Euclidean distance between two objects.
    ///
    /// For points, this is the straight-line distance. For shapes, this is
    /// typically the minimum distance between their boundaries (0 if overlapping).
    ///
    /// # Arguments
    ///
    /// * `other` - The other object to measure distance to
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::coord::Coord;
    /// use eunoia::geometry::operations::Distance;
    ///
    /// let p1 = Coord::new(0.0, 0.0);
    /// let p2 = Coord::new(3.0, 4.0);
    /// assert_eq!(p1.distance(&p2), 5.0);
    /// ```
    fn distance(&self, other: &Self) -> f64;
}

/// Trait for computing the perimeter of a shape.
pub trait Perimeter {
    /// Returns the perimeter (total boundary length) of the shape.
    fn perimeter(&self) -> f64;
}

/// Trait for computing intersection area between two shapes.
pub trait IntersectionArea {
    /// Computes the area of overlap between this shape and another.
    ///
    /// Returns 0.0 if the shapes don't intersect. If one shape completely
    /// contains the other, returns the area of the smaller shape.
    ///
    /// # Arguments
    ///
    /// * `other` - The other shape to compute intersection with
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::circle::Circle;
    /// use eunoia::geometry::coord::Coord;
    /// use eunoia::geometry::operations::IntersectionArea;
    ///
    /// let c1 = Circle::new(Coord::new(0.0, 0.0), 1.0);
    /// let c2 = Circle::new(Coord::new(10.0, 0.0), 1.0);
    /// assert_eq!(c1.intersection_area(&c2), 0.0);
    /// ```
    fn intersection_area(&self, other: &Self) -> f64;
}
