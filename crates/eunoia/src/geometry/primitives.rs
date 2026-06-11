//! Geometric primitives (lines, points, etc).
pub mod bounds;
pub mod line;
pub mod point;

// Re-export common types for convenience
pub use bounds::Bounds;
pub use line::Line;
pub use point::Point;
