//! Mathematical and numerical constants used throughout the library.

/// Tolerance for numerical comparisons and zero-thresholding.
///
/// This value represents approximately the square root of machine epsilon for f64,
/// which is appropriate for geometric calculations where small errors can accumulate.
///
/// Used for:
/// - Determining if floating-point values are effectively zero
/// - Comparing floating-point values for approximate equality
/// - Thresholding matrix elements to zero
/// - Checking if points lie on geometric objects
pub const EPSILON: f64 = 1e-10;
