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
//! use eunoia::{DiagramSpecBuilder, Fitter, InputType};
//! use eunoia::geometry::shapes::Circle;
//!
//! // Build specification (shape-agnostic)
//! let spec = DiagramSpecBuilder::new()
//!     .set("A", 5.0)
//!     .set("B", 2.0)
//!     .intersection(&["A", "B"], 1.0)
//!     .input_type(InputType::Exclusive)
//!     .build()
//!     .expect("Failed to build diagram specification");
//!
//! // Choose shape type when fitting
//! let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
//! ```

pub mod error;
pub mod fitter;
pub mod geometry;
pub mod loss;
pub mod math;
pub mod spec;

pub mod constants;

// Internal test utilities
#[cfg(test)]
pub mod test_utils;

// Re-export main API types
pub use error::DiagramError;
pub use fitter::{Fitter, Layout, Optimizer};
pub use spec::{Combination, DiagramSpecBuilder, InputType};
