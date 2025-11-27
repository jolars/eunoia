//! Polynomial equation solvers.
//!
//! This module provides solvers for polynomial equations, primarily for use in
//! geometric computations involving conic sections.

use num_complex::Complex64;

/// Solves a cubic equation of the form αx³ + βx² + γx + δ = 0.
///
/// Returns up to three complex roots. Real roots have imaginary parts close to zero.
///
/// # Algorithm
///
/// Uses the cubic formula (Cardano's method) with careful handling of the discriminant:
/// - If R² < Q³: Three real roots (trigonometric solution)
/// - If R² ≥ Q³: One real root and two complex conjugate roots
///
/// # Examples
///
/// ```
/// use eunoia::math::polynomial::solve_cubic;
///
/// // Solve x³ - 6x² + 11x - 6 = 0
/// // Roots are x = 1, 2, 3
/// let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
///
/// // All three roots should be real
/// for root in &roots {
///     assert!(root.im.abs() < 1e-10);
/// }
/// ```
///
/// # References
///
/// This implementation is based on RConics (GPL-3), adapted from the C++ version
/// by Emanuel Huber.
pub fn solve_cubic(alpha: f64, beta: f64, gamma: f64, delta: f64) -> [Complex64; 3] {
    use std::f64::consts::PI;

    // Normalize to x³ + ax² + bx + c = 0
    let a = beta / alpha;
    let b = gamma / alpha;
    let c = delta / alpha;

    // Compute intermediate values Q and R
    let q = (a * a - 3.0 * b) / 9.0;
    let r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;

    let discriminant = r * r - q * q * q;

    if discriminant < 0.0 {
        // Three distinct real roots (trigonometric solution)
        let theta = (r / (q * q * q).sqrt()).acos();
        let sqrt_q = q.sqrt();

        [
            Complex64::new(-2.0 * sqrt_q * (theta / 3.0).cos() - a / 3.0, 0.0),
            Complex64::new(
                -2.0 * sqrt_q * ((theta + 2.0 * PI) / 3.0).cos() - a / 3.0,
                0.0,
            ),
            Complex64::new(
                -2.0 * sqrt_q * ((theta - 2.0 * PI) / 3.0).cos() - a / 3.0,
                0.0,
            ),
        ]
    } else {
        // One real root and two complex conjugate roots
        let sqrt_discriminant = discriminant.sqrt();
        let a_val = -r.signum() * (r.abs() + sqrt_discriminant).cbrt();
        let b_val = if a_val.abs() < 1e-10 { 0.0 } else { q / a_val };

        let real_part = -0.5 * (a_val + b_val) - a / 3.0;
        let imag_part = 3.0_f64.sqrt() * (a_val - b_val) / 2.0;

        [
            Complex64::new(a_val + b_val - a / 3.0, 0.0),
            Complex64::new(real_part, imag_part),
            Complex64::new(real_part, -imag_part),
        ]
    }
}

/// Extracts real roots from complex roots.
///
/// Returns only roots with imaginary parts smaller than the given tolerance.
///
/// # Examples
///
/// ```
/// use eunoia::math::polynomial::{solve_cubic, extract_real_roots};
///
/// let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
/// let real_roots = extract_real_roots(&roots, 1e-10);
///
/// assert_eq!(real_roots.len(), 3);
/// ```
pub fn extract_real_roots(roots: &[Complex64], tolerance: f64) -> Vec<f64> {
    roots
        .iter()
        .filter(|r| r.im.abs() < tolerance)
        .map(|r| r.re)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_cubic_three_real_roots() {
        // (x - 1)(x - 2)(x - 3) = x³ - 6x² + 11x - 6
        let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);

        // Extract real parts
        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // All roots should be real
        for root in &roots {
            assert!(root.im.abs() < 1e-10, "Root should be real: {:?}", root);
        }

        // Check root values
        assert!(approx_eq(real_roots[0], 1.0, 1e-10));
        assert!(approx_eq(real_roots[1], 2.0, 1e-10));
        assert!(approx_eq(real_roots[2], 3.0, 1e-10));
    }

    #[test]
    fn test_cubic_one_real_root() {
        // x³ - 1 = 0 has roots: 1, -1/2 ± i√3/2
        let roots = solve_cubic(1.0, 0.0, 0.0, -1.0);

        let real_roots = extract_real_roots(&roots, 1e-10);
        assert_eq!(real_roots.len(), 1);
        assert!(approx_eq(real_roots[0], 1.0, 1e-10));

        // Count complex roots
        let complex_roots: Vec<_> = roots.iter().filter(|r| r.im.abs() > 1e-10).collect();
        assert_eq!(complex_roots.len(), 2);
    }

    #[test]
    fn test_cubic_repeated_root() {
        // (x - 2)³ = x³ - 6x² + 12x - 8
        let roots = solve_cubic(1.0, -6.0, 12.0, -8.0);

        // All roots should be approximately 2
        for root in &roots {
            assert!(
                approx_eq(root.re, 2.0, 1e-9),
                "Root should be 2.0, got {}",
                root.re
            );
            assert!(root.im.abs() < 1e-9, "Root should be real");
        }
    }

    #[test]
    fn test_extract_real_roots() {
        let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
        let real_roots = extract_real_roots(&roots, 1e-10);

        assert_eq!(real_roots.len(), 3);

        let mut sorted = real_roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(approx_eq(sorted[0], 1.0, 1e-10));
        assert!(approx_eq(sorted[1], 2.0, 1e-10));
        assert!(approx_eq(sorted[2], 3.0, 1e-10));
    }

    #[test]
    fn test_cubic_negative_leading_coefficient() {
        // -x³ + 6x² - 11x + 6 = 0 (same roots as before, but negated equation)
        let roots = solve_cubic(-1.0, 6.0, -11.0, 6.0);

        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(approx_eq(real_roots[0], 1.0, 1e-10));
        assert!(approx_eq(real_roots[1], 2.0, 1e-10));
        assert!(approx_eq(real_roots[2], 3.0, 1e-10));
    }

    #[test]
    fn test_cubic_with_zero_root() {
        // x³ - x = x(x² - 1) = x(x-1)(x+1)
        // Roots: -1, 0, 1
        let roots = solve_cubic(1.0, 0.0, -1.0, 0.0);

        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!(approx_eq(real_roots[0], -1.0, 1e-10));
        assert!(approx_eq(real_roots[1], 0.0, 1e-10));
        assert!(approx_eq(real_roots[2], 1.0, 1e-10));
    }
}
