//! Geometric primitives and operations.
//!
//! This module provides the foundational geometric types and traits used throughout
//! the library, including points, shapes, and geometric operations.

pub mod diagram;
// The overlap helpers in `operations` exist only to cross-check the circle/area
// math against a Monte-Carlo oracle in tests; nothing in the library or its
// public API uses them, so the module is compiled for tests only.
#[cfg(test)]
mod operations;
pub mod primitives;
pub(crate) mod projective;
pub mod shapes;
pub mod traits;
