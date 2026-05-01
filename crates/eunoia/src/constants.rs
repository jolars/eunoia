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

/// Maximum number of sets supported in a single diagram specification.
///
/// Region masks are encoded in a `usize`, so the absolute representational
/// ceiling is 63 on 64-bit platforms. The practical ceiling is much lower:
/// a fully-overlapping diagram has `2^n - 1` non-empty regions, so memory
/// and runtime explode well before the bit limit. We cap at 32 to keep the
/// worst case bounded (2³² ≈ 4.3 billion regions is already untenable, but
/// well-formed sparse inputs at this size are still tractable).
///
/// Sparse inputs (where most subsets are empty) will only allocate memory
/// proportional to the number of non-empty regions, not `2^n`.
pub const MAX_SETS: usize = 32;
