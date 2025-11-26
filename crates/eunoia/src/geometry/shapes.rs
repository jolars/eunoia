//! Shape implementations for Euler and Venn diagrams.
//!
//! This module contains various geometric shape types that can be used
//! to represent sets in diagrams.

pub mod circle;

use crate::geometry::point::Point;
use crate::geometry::rectangle::Rectangle;

pub trait Shape {
    /// Returns the area of the shape.
    fn area(&self) -> f64;

    /// Computes the minimum distance between this shape and another shape.
    /// Returns 0.0 if the shapes overlap.
    fn distance(&self, other: &Self) -> f64;

    /// Checks if a shape contains another shape entirely within its boundaries
    fn contains(&self, other: &Self) -> bool;

    /// Checks if this shape intersects with another shape.
    fn intersects(&self, other: &Self) -> bool;

    /// Computes the area of intersection between this shape and another shape.
    fn intersection_area(&self, other: &Self) -> f64;

    /// Computes the intersection points between this shape and another shape.
    fn intersection_points(&self, other: &Self) -> Vec<Point>;

    /// Returns the centroid (center point) of the shape as (x, y) coordinates.
    fn centroid(&self) -> (f64, f64);

    /// Compute the perimiter of the shape.
    fn perimeter(&self) -> f64;

    /// Check if a point is inside the shape.
    fn contains_point(&self, point: &Point) -> bool;

    /// Compute the bounding box of the shape as a Rectangle.
    fn bounding_box(&self) -> Rectangle;
}

/// Compute the bounding box for a collection of shapes.
pub fn bounding_box<S: Shape>(shapes: &[S]) -> Rectangle {
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
