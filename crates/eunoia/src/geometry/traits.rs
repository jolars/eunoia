//! Shape implementations for Euler and Venn diagrams.
//!
//! This module defines composable traits for geometric operations and diagram shapes.
//! The trait system is decomposed to allow code reuse across different geometric types:
//!
//! - Basic geometric properties: `Area`, `Centroid`, `Perimeter`, `BoundingBox`
//! - Spatial relations between same-type objects: `Distance`, `Closed`
//! - Diagram-specific operations: `DiagramShape` (composes all of the above)

use crate::geometry::primitives::Point;
use crate::geometry::shapes::Rectangle;

/// Trait for objects that have a computable distance to another object of the same type.
///
/// This can be implemented by both shapes (Circle, Point, LineSegment, etc.) to enable
/// generic distance-based algorithms.
pub trait Distance<Rhs = Self> {
    /// Computes the minimum distance between this object and another.
    /// For shapes, this is the distance between boundaries (0.0 if overlapping).
    /// For points, this is the Euclidean distance.
    fn distance(&self, other: &Rhs) -> f64;
}

/// Trait for objects that have a measurable area.
pub trait Area {
    /// Returns the area of the object.
    fn area(&self) -> f64;
}

/// Trait for objects that have a definable centroid (center of mass).
pub trait Centroid {
    /// Returns the centroid (center point) as a Point.
    fn centroid(&self) -> Point;
}

/// Trait for objects that have a measurable perimeter.
pub trait Perimeter {
    /// Computes the perimeter (boundary length) of the object.
    fn perimeter(&self) -> f64;
}

/// Trait for objects that can be bounded by a rectangle.
pub trait BoundingBox {
    /// Computes the axis-aligned bounding box as a Rectangle.
    fn bounding_box(&self) -> Rectangle;
}

/// Trait for spatial relationships between objects of the same type.
///
/// This covers containment, intersection testing, and computing intersection areas/points.
pub trait Closed: Sized + Area + BoundingBox + Perimeter + Centroid {
    /// Checks if this object contains another object entirely within its boundaries.
    fn contains(&self, other: &Self) -> bool;

    /// Checks if a point is inside the object (on the boundary or interior).
    fn contains_point(&self, point: &Point) -> bool;

    /// Checks if this object intersects with another object.
    fn intersects(&self, other: &Self) -> bool;

    /// Computes the area of intersection between this object and another.
    fn intersection_area(&self, other: &Self) -> f64;

    /// Computes the points where this object's boundary intersects another's.
    fn intersection_points(&self, other: &Self) -> Vec<Point>;
}

/// Trait for shapes that can be used in Euler and Venn diagrams.
///
/// This is a supertrait that combines all the geometric capabilities needed for
/// diagram construction and optimization. Types implementing this trait can be used
/// with the diagram fitter.
///
/// # Type Parameters
///
/// The associated methods for parameter conversion enable the optimization process:
/// 1. Initial layout uses circles (MDS algorithm)
/// 2. Circle parameters are converted to shape-specific parameters via `params_from_circle`
/// 3. Final optimization operates on shape-specific parameters
/// 4. Shapes are constructed from optimized parameters via `from_params`
pub trait DiagramShape: Closed {
    /// Compute all exclusive regions and their areas from a collection of shapes.
    ///
    /// This method should use exact geometric computation for the shape type.
    /// Returns a map from RegionMask (bit representation) to exclusive area.
    ///
    /// This is used during optimization to compute loss functions.
    fn compute_exclusive_regions(
        shapes: &[Self],
    ) -> std::collections::HashMap<crate::geometry::diagram::RegionMask, f64>
    where
        Self: Sized;

    /// Convert initial circle parameters to shape-specific parameters.
    ///
    /// Takes circle parameters (x, y, radius) and converts them to whatever
    /// parameters this shape type needs for optimization.
    ///
    /// For Circle: returns [x, y, r]
    /// For Ellipse: might return [x, y, a, b, angle] where a=b=r initially
    fn params_from_circle(x: f64, y: f64, radius: f64) -> Vec<f64>
    where
        Self: Sized;

    /// Get the number of parameters needed for this shape type.
    ///
    /// For Circle: 3 (x, y, r)
    /// For Ellipse: 5 (x, y, a, b, angle)
    fn n_params() -> usize
    where
        Self: Sized;

    /// Construct a shape from optimized parameters.
    ///
    /// Takes a slice of parameters specific to this shape and constructs the shape.
    /// The parameters should match what params_from_circle produces.
    fn from_params(params: &[f64]) -> Self
    where
        Self: Sized;
}

/// Trait for converting shapes to polygons for visualization.
///
/// This trait allows analytical shapes (circles, ellipses) to be converted
/// to discrete polygon representations for plotting and export.
pub trait Polygonize {
    /// Convert the shape to a polygon with the specified number of vertices.
    ///
    /// # Arguments
    ///
    /// * `n_vertices` - Number of vertices in the resulting polygon (minimum 3)
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::geometry::shapes::Circle;
    /// use eunoia::geometry::primitives::Point;
    /// use eunoia::geometry::traits::Polygonize;
    ///
    /// let circle = Circle::new(Point::new(0.0, 0.0), 5.0);
    /// let polygon = circle.polygonize(64);
    /// assert_eq!(polygon.vertices().len(), 64);
    /// ```
    fn polygonize(&self, n_vertices: usize) -> crate::geometry::shapes::Polygon;
}

/// Compute the bounding box for a collection of shapes.
pub fn bounding_box<S: BoundingBox>(shapes: &[S]) -> Rectangle {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;

    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for shape in shapes {
        let points = shape.bounding_box().to_points();
        min_x = min_x.min(points.0.x());
        min_y = min_y.min(points.0.y());
        max_x = max_x.max(points.1.x());
        max_y = max_y.max(points.1.y());
    }

    Rectangle::from_corners(Point::new(min_x, min_y), Point::new(max_x, max_y))
}
