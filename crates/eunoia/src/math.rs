pub(crate) mod linear_algebra;
pub(crate) mod polynomial;

use crate::constants::EPSILON;

/// Zeros out small values below a threshold.
///
/// # Arguments
///
/// * `val` - The value to check
/// * `epsilon` - The threshold below which values are zeroed (default: [`EPSILON`])
pub(crate) fn zap_small(val: f64) -> f64 {
    zap_small_with(val, EPSILON)
}

/// Zeros out small values below a custom threshold.
///
/// # Arguments
///
/// * `val` - The value to check
/// * `epsilon` - The threshold below which values are zeroed
pub(crate) fn zap_small_with(val: f64, epsilon: f64) -> f64 {
    if val.abs() < epsilon { 0.0 } else { val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zap_small_zeros_below_default_epsilon() {
        assert_eq!(zap_small(1e-12), 0.0);
        assert_eq!(zap_small(2.0), 2.0);
    }

    #[test]
    fn zap_small_with_respects_custom_epsilon() {
        assert_eq!(zap_small_with(1e-9, 1e-8), 0.0);
        assert_eq!(zap_small_with(1e-7, 1e-8), 1e-7);
    }
}
