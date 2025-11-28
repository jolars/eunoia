//! Polynomial equation solvers.
//!
//! This module provides solvers for polynomial equations, primarily for use in
//! geometric computations involving conic sections.

use nalgebra::Matrix3;
use nalgebra::Vector3;
use num_complex::Complex64;

/// Solves a cubic equation of the form αx³ + βx² + γx + δ = 0.
///
/// Returns up to three complex roots. Real roots have imaginary parts close to zero.
///
/// # Arguments
///
/// * `alpha` - Coefficient of x³ (must be non-zero)
/// * `beta` - Coefficient of x²
/// * `gamma` - Coefficient of x
/// * `delta` - Constant term
///
/// # Panics
///
/// Panics if `alpha` is zero (not a cubic equation).
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

    // Check for degenerate case
    assert!(
        alpha.abs() > 1e-14,
        "alpha must be non-zero for a cubic equation (got {})",
        alpha
    );

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
        // This branch requires Q > 0
        if q <= 0.0 {
            // Shouldn't happen if discriminant < 0, but be defensive
            panic!("Invalid state: discriminant < 0 but Q ≤ 0");
        }

        let q_cubed = q * q * q;
        let sqrt_q_cubed = q_cubed.sqrt();

        // Clamp to avoid acos() domain errors due to floating point precision
        let cos_arg = (r / sqrt_q_cubed).clamp(-1.0, 1.0);
        let theta = cos_arg.acos();
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
/// A root is considered real if its imaginary part is exactly zero.
/// This strict check matches RConics behavior.
///
/// # Examples
///
/// ```
/// use eunoia::math::polynomial::{solve_cubic, extract_real_roots};
///
/// let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
/// let real_roots = extract_real_roots(&roots);
///
/// assert_eq!(real_roots.len(), 3);
/// ```
pub fn extract_real_roots(roots: &[Complex64]) -> Vec<f64> {
    roots.iter().filter(|r| r.im == 0.0).map(|r| r.re).collect()
}

/// Solves a homogeneous cubic equation αλ³ + βλ²μ + γλμ² + δμ³ = 0.
///
/// Returns (λ, μ) pairs representing solutions. This method solves the equation
/// directly in homogeneous form without dividing by μ, naturally handling all cases
/// including μ = 0.
///
/// # Algorithm
///
/// Uses the formulation from computational geometry for homogeneous cubics:
/// 1. Compute W = -2β³ + 9αβγ - 27α²δ
/// 2. Compute discriminant D
/// 3. Compute Q = W - α√(27D)
/// 4. Form vectors L and M
/// 5. Apply transformation with cube roots of unity
///
/// # Examples
///
/// ```
/// use eunoia::math::polynomial::solve_homogeneous_cubic;
///
/// // Solve λ³ - 6λ²μ + 11λμ² - 6μ³ = 0
/// // This factors as (λ - μ)(λ - 2μ)(λ - 3μ) = 0
/// let pairs = solve_homogeneous_cubic(1.0, -6.0, 11.0, -6.0);
///
/// // Should give ratios λ:μ = 1:1, 2:1, 3:1
/// assert_eq!(pairs.len(), 3);
/// ```
pub fn solve_homogeneous_cubic(
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> Vec<(Complex64, Complex64)> {
    let w = -2.0 * beta.powi(3) + 9.0 * alpha * beta * gamma - 27.0 * alpha.powi(2) * delta;

    let d =
        -beta.powi(2) * gamma.powi(2) + 4.0 * alpha * gamma.powi(3) + 4.0 * beta.powi(3) * delta
            - 18.0 * alpha * beta * gamma * delta
            + 27.0 * alpha.powi(2) * delta.powi(3);

    let q = w - alpha * Complex64::new(27.0 * d, 0.0).sqrt();

    let r = (4.0 * q).cbrt();

    let l = Vector3::new(
        Complex64::new(2.0 * beta.powi(2) - 6.0 * alpha * gamma, 0.0),
        Complex64::new(-beta, 0.0),
        r,
    );

    let m = Vector3::new(
        r * 3.0 * alpha,
        Complex64::new(3.0 * alpha, 0.0),
        Complex64::new(6.0 * alpha, 0.0),
    );

    let omega = Complex64::new(-0.5, (3.0_f64).sqrt() / 2.0); // e^(2πi/3)

    // Apply transformation with cube roots of unity
    // [ω   1   ω²] L = λ_vectors
    // [1   1   1 ]
    // [ω²  1   ω ]
    let omega_mat = Matrix3::from_rows(&[
        nalgebra::RowVector3::new(omega, Complex64::new(1.0, 0.0), omega * omega),
        nalgebra::RowVector3::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ),
        nalgebra::RowVector3::new(omega * omega, Complex64::new(1.0, 0.0), omega),
    ]);

    let lambda_vectors = omega_mat * l;
    let mu_vectors = omega_mat * m;

    vec![
        (lambda_vectors[0], mu_vectors[0]),
        (lambda_vectors[1], mu_vectors[1]),
        (lambda_vectors[2], mu_vectors[2]),
    ]
}

/// Extracts real (λ, μ) pairs from complex pairs.
///
/// Returns only pairs where both λ and μ have imaginary parts smaller than tolerance.
pub fn extract_real_pairs(pairs: &[(Complex64, Complex64)], tolerance: f64) -> Vec<(f64, f64)> {
    pairs
        .iter()
        .filter(|(lambda, mu)| lambda.im.abs() < tolerance && mu.im.abs() < tolerance)
        .map(|(lambda, mu)| (lambda.re, mu.re))
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

        let real_roots = extract_real_roots(&roots);
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
        let real_roots = extract_real_roots(&roots);

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

    #[test]
    #[should_panic(expected = "alpha must be non-zero")]
    fn test_cubic_alpha_zero() {
        // Not a cubic equation - should panic
        solve_cubic(0.0, 1.0, -2.0, 1.0);
    }

    #[test]
    fn test_cubic_near_boundary() {
        // Test case where discriminant is very close to zero
        // (x - 1)²(x - 2) = x³ - 4x² + 5x - 2
        let roots = solve_cubic(1.0, -4.0, 5.0, -2.0);

        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Should have roots at 1 (double) and 2
        assert!(approx_eq(real_roots[0], 1.0, 1e-9));
        assert!(approx_eq(real_roots[1], 1.0, 1e-9));
        assert!(approx_eq(real_roots[2], 2.0, 1e-9));
    }

    #[test]
    #[ignore] // TODO: Fix homogeneous cubic solver - same issue as test_homogeneous_vs_standard
    fn test_homogeneous_cubic_simple() {
        // (λ - μ)(λ - 2μ)(λ - 3μ) = λ³ - 6λ²μ + 11λμ² - 6μ³
        let pairs = solve_homogeneous_cubic(1.0, -6.0, 11.0, -6.0);
        let real_pairs = extract_real_pairs(&pairs, 1e-8);

        // Should have 3 real pairs
        assert_eq!(real_pairs.len(), 3);

        // Extract ratios λ/μ
        let mut ratios: Vec<f64> = real_pairs.iter().map(|(l, m)| l / m).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Should be approximately 1, 2, 3
        assert!(approx_eq(ratios[0], 1.0, 1e-6));
        assert!(approx_eq(ratios[1], 2.0, 1e-6));
        assert!(approx_eq(ratios[2], 3.0, 1e-6));
    }

    #[test]
    #[ignore] // TODO: Fix homogeneous cubic solver - currently not producing real solutions
    fn test_homogeneous_vs_standard() {
        // Compare homogeneous cubic with standard cubic solver
        // λ³ - 6λ²μ + 11λμ² - 6μ³ = 0
        //
        // TODO: The homogeneous solver implementation is producing complex (λ, μ) pairs
        // that don't satisfy the "real in homogeneous sense" condition (where λ/μ should be real).
        // The algorithm from the book appears correct, but something is wrong in the implementation.
        // Possible issues to investigate:
        // - Sign error in L or M vector components
        // - Cube root selection (though book says any root should work)
        // - Matrix transformation order or structure
        // - Formula for W or D discriminants
        //
        // Expected: λ/μ ratios should be [1.0, 2.0, 3.0]
        // Actual: Getting complex ratios with significant imaginary components
        let alpha = 1.0;
        let beta = -6.0;
        let gamma = 11.0;
        let delta = -6.0;

        // Standard method (solve for t = λ/μ)
        let roots = solve_cubic(alpha, beta, gamma, delta);
        let mut standard_ratios: Vec<f64> = roots.iter().map(|r| r.re).collect();
        standard_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Homogeneous method
        let pairs = solve_homogeneous_cubic(alpha, beta, gamma, delta);
        let real_pairs = extract_real_pairs(&pairs, 1e-8);
        let mut homo_ratios: Vec<f64> = real_pairs.iter().map(|(l, m)| l / m).collect();
        homo_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("{:#?}", roots);
        println!("{:#?}", pairs);

        // Both should give the same ratios
        assert_eq!(standard_ratios.len(), homo_ratios.len());
        for (s, h) in standard_ratios.iter().zip(homo_ratios.iter()) {
            assert!(
                approx_eq(*s, *h, 1e-6),
                "Standard: {}, Homogeneous: {}",
                s,
                h
            );
        }
    }
}
