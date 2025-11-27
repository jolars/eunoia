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
//! use eunoia::{DiagramSpecBuilder, InputType};
//! use eunoia::geometry::shapes::Circle;
//!
//! let spec = DiagramSpecBuilder::<Circle>::new()
//!     .set("A", 5.0)
//!     .set("B", 2.0)
//!     .intersection(&["A", "B"], 1.0)
//!     .input_type(InputType::Exclusive)
//!     .build()
//!     .expect("Failed to build diagram specification");
//! ```

pub mod error;
pub mod fitter;
pub mod geometry;
pub mod loss;
pub mod spec;

// Re-export main API types
pub use error::DiagramError;
pub use fitter::{Fitter, Layout};
pub use spec::{Combination, DiagramSpecBuilder, InputType};
