pub mod linear_algebra;
pub mod polynomial;

use crate::constants::EPSILON;

/// Zeros out small values below a threshold.
///
/// # Arguments
///
/// * `val` - The value to check
/// * `epsilon` - The threshold below which values are zeroed (default: [`EPSILON`])
///
/// # Examples
///
/// ```
/// use eunoia::math::{zap_small, zap_small_with};
/// use nalgebra::Matrix3;
///
/// // Using default epsilon
/// let matrix = Matrix3::new(1.0, 1e-12, 0.0, 1e-12, 2.0, 0.0, 0.0, 0.0, 3.0);
/// let cleaned = matrix.map(zap_small);
///
/// // Using custom epsilon
/// let cleaned = matrix.map(|x| zap_small_with(x, 1e-8));
/// ```
pub fn zap_small(val: f64) -> f64 {
    zap_small_with(val, EPSILON)
}

/// Zeros out small values below a custom threshold.
///
/// # Arguments
///
/// * `val` - The value to check
/// * `epsilon` - The threshold below which values are zeroed
pub fn zap_small_with(val: f64, epsilon: f64) -> f64 {
    if val.abs() < epsilon {
        0.0
    } else {
        val
    }
}
