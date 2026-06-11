//! Mathematical and numerical constants used throughout the library.
//!
//! These constants are part of the public API: downstream code can name them,
//! and—because Rust does not treat a `const`'s *value* as part of its
//! type—a value change compiles cleanly at the call site while silently
//! altering behavior. Each constant below therefore documents an explicit
//! **stability** stance so callers know what they may rely on:
//!
//! - [`MAX_SETS_HARD_CAP`] is a structural invariant of the region-mask
//!   representation and will not change on a given target.
//! - [`MAX_SETS`] is a *default* and may be raised in a future minor release;
//!   read it at runtime rather than hard-coding `32`.
//! - [`EPSILON`] is a numerical-tuning value and may be adjusted; do not pin
//!   cross-version behavior to its exact magnitude.

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
///
/// # Stability
///
/// The exact magnitude is a numerical-tuning choice, not a stability
/// guarantee: it may be adjusted in a future release if the geometric
/// algorithms are retuned. Use it as "the library's notion of negligible"
/// (e.g. to stay consistent with eunoia's own comparisons), but do not pin
/// cross-version behavior to it being exactly `1e-10`.
pub const EPSILON: f64 = 1e-10;

/// Default maximum number of sets supported in a single diagram specification.
///
/// Region masks are encoded in a `usize`, so the absolute representational
/// ceiling is [`MAX_SETS_HARD_CAP`]. The practical ceiling is much lower:
/// a fully-overlapping diagram has `2^n - 1` non-empty regions, so memory
/// and runtime explode well before the bit limit. We default to 32 to keep
/// the worst case bounded (2³² ≈ 4.3 billion regions is already untenable,
/// but well-formed sparse inputs at this size are still tractable).
///
/// Sparse inputs (where most subsets are empty) will only allocate memory
/// proportional to the number of non-empty regions, not `2^n`.
///
/// Callers that knowingly need to exceed this default can override it on
/// a per-spec basis with [`crate::DiagramSpecBuilder::max_sets`], up to
/// [`MAX_SETS_HARD_CAP`].
///
/// # Stability
///
/// This is a *default*, not a fixed limit, and may be raised in a future
/// minor release as the fitter handles larger sparse inputs. Read it at
/// runtime (or override it explicitly) rather than hard-coding `32`; the
/// only guaranteed upper bound is [`MAX_SETS_HARD_CAP`].
pub const MAX_SETS: usize = 32;

/// Absolute upper bound on the number of sets in a single diagram.
///
/// Region masks are encoded in a `usize` bitset, so we reserve one bit for
/// the empty mask and cap the supported count at 63 (the largest count that
/// fits on 32-bit platforms is lower, but the API still accepts 63 — a
/// caller running on a 32-bit target will hit the platform limit naturally).
/// [`crate::DiagramSpecBuilder::max_sets`] silently clamps any override to
/// this value, so values above 63 cannot be requested.
///
/// # Stability
///
/// This is the one value here that is a genuine API contract: it is fixed by
/// the region-mask representation (a `usize` bitset with one bit reserved for
/// the empty mask), not a tunable policy. It will not change on a 64-bit
/// target. Callers may rely on `63` as the absolute ceiling.
pub const MAX_SETS_HARD_CAP: usize = 63;
