//! Projective geometry primitives and operations.
//!
//! This module provides types and operations for projective geometry, which extends
//! Euclidean geometry by adding points at infinity and using homogeneous coordinates.
//!
//! # Homogeneous Coordinates
//!
//! In projective geometry, 2D points are represented using three coordinates [x, y, w]
//! where scalar multiples represent the same point. A point [x, y, w] with w ≠ 0
//! corresponds to the Euclidean point (x/w, y/w). Points with w = 0 are "ideal points"
//! or points at infinity, representing directions.
//!
//! Similarly, lines are represented as [a, b, c] where points [x, y, w] on the line
//! satisfy ax + by + cw = 0.
//!
//! # Use Cases
//!
//! - **Ellipse intersection**: Computing intersection points of two ellipses using
//!   conic sections in projective coordinates
//! - **Degeneracies**: Handling tangent cases and points at infinity gracefully
//! - **Transformations**: Projective transformations unify affine and perspective operations

pub mod line;
pub mod point;

pub use line::HomogeneousLine;
pub use point::HomogeneousPoint;
