//! Shape implementations for Euler and Venn diagrams.
//!
//! This module defines composable traits for geometric operations and diagram shapes.
//! The trait system is decomposed to allow code reuse across different geometric types:
//!
//! - Basic geometric properties: `Area`, `Centroid`, `Perimeter`, `BoundingBox`
//! - Spatial relations between same-type objects: `Distance`, `Closed`
//! - Diagram-specific operations: `DiagramShape` (composes all of the above)

pub mod circle;
pub mod ellipse;
pub mod rectangle;

pub use circle::Circle;
pub use ellipse::Ellipse;
pub use rectangle::Rectangle;
