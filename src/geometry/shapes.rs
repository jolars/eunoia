//! Shape implementations for Euler and Venn diagrams.
//!
//! This module contains various geometric shape types that can be used
//! to represent sets in diagrams.

pub mod circle;

use crate::geometry::point::Point;

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
}
