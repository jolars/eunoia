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
//! use eunoia::{DiagramBuilder, InputType};
//!
//! let diagram = DiagramBuilder::new()
//!     .set("A", 5.0)
//!     .set("B", 2.0)
//!     .intersection(&["A", "B"], 1.0)
//!     .input_type(InputType::Disjoint)
//!     .build()
//!     .expect("Failed to build diagram");
//! ```

pub mod diagram;
pub mod error;
pub mod geometry;

// Re-export main API types
pub use diagram::{Combination, Diagram, DiagramBuilder, InputType};
pub use error::DiagramError;

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
