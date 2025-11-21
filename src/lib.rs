//! # Eunoia
//!
//! A Rust library for creating area-proportional Euler and Venn diagrams.
//!
//! Eunoia generates optimal layouts for set visualizations using various geometric shapes
//! (circles, ellipses, rectangles, triangles). The library uses a two-phase optimization
//! approach:
//!
//! 1. **Initial layout**: Multi-dimensional scaling (MDS) to place fixed-size shapes
//! 2. **Refinement**: Comprehensive optimization to minimize loss functions (RegionError or stress)
//!
//! ## Example
//!
//! ```rust
//! use eunoia::geometry::shapes::circle::Circle;
//! use eunoia::geometry::coord::Coord;
//! use eunoia::geometry::operations::Area;
//!
//! let circle = Circle::new(Coord::new(0.0, 0.0), 1.0);
//! let area = circle.area();
//! ```

pub mod geometry;

/// Trait for computing intersection areas between shapes.
///
/// This trait allows different shape types to compute their area of overlap,
/// which is essential for evaluating diagram quality and computing loss functions.
pub trait Intersects<S> {
    /// Computes the area of intersection between this shape and another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other shape to intersect with
    ///
    /// # Returns
    ///
    /// The area of intersection in square units. Returns 0.0 if shapes don't intersect.
    fn intersection_area(&self, other: &S) -> f64;
}

/// Trait for shapes that can be parameterized for optimization.
///
/// This trait enables shapes to expose their degrees of freedom (position, size, rotation)
/// as a parameter vector for use with optimization algorithms.
pub trait Parameters {
    /// Returns the number of parameters needed to fully describe this shape.
    ///
    /// For example, a circle has 3 parameters (x, y, radius), while an ellipse
    /// has 5 (x, y, semi-major axis, semi-minor axis, rotation angle).
    fn n_params(&self) -> usize;
    
    /// Updates the shape's parameters from a parameter vector.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice containing the new parameter values
    ///
    /// # Panics
    ///
    /// May panic if the length of `params` doesn't match `n_params()`.
    fn update(&mut self, params: &[f64]);
}

/// Represents a complete Euler or Venn diagram with multiple shapes.
///
/// A diagram contains a collection of shapes that represent sets, along with
/// metadata about their relationships and optimization state.
///
/// # Type Parameters
///
/// * `S` - The shape type used in the diagram (e.g., `Circle`, `Ellipse`)
pub struct Diagram<S> {
    shapes: Vec<S>,
}

#[cfg(test)]
mod tests {}
