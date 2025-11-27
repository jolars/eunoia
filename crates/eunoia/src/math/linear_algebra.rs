//! Linear algebra utilities and extensions.
//!
//! This module provides extensions to `nalgebra` types for domain-specific
//! operations used in Euler diagram computations.
//!
//! # Mathematical Background
//!
//! ## Cofactor Matrix
//!
//! For a 3×3 matrix A, the cofactor matrix C is defined as:
//! ```text
//! C[i,j] = (-1)^(i+j) × minor(i,j)
//! ```
//! where `minor(i,j)` is the determinant of the 2×2 submatrix obtained by
//! removing row i and column j from A.
//!
//! ## Adjugate Matrix
//!
//! The adjugate is the transpose of the cofactor matrix:
//! ```text
//! adj(A) = C^T
//! ```
//!
//! Key property: `A × adj(A) = det(A) × I`, therefore `A^(-1) = adj(A) / det(A)`

use nalgebra::Matrix3;

/// Extension trait for `nalgebra::Matrix3`.
///
/// Provides cofactor and adjugate matrix operations.
pub trait Matrix3Ext {
    /// Computes the cofactor matrix.
    ///
    /// Each element C[i,j] = (-1)^(i+j) × minor(i,j), where the minor is the
    /// determinant of the 2×2 submatrix formed by removing row i and column j.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::math::linear_algebra::Matrix3Ext;
    /// use nalgebra::Matrix3;
    ///
    /// let m: Matrix3<f64> = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 4.0, 5.0,
    ///     1.0, 0.0, 6.0,
    /// );
    /// let cof = m.cofactor_matrix();
    /// ```
    fn cofactor_matrix(&self) -> Matrix3<f64>;

    /// Computes the adjugate (classical adjoint) matrix.
    ///
    /// The adjugate is the transpose of the cofactor matrix: adj(A) = cofactor(A)^T.
    ///
    /// # Properties
    ///
    /// - A × adj(A) = det(A) × I
    /// - A^(-1) = adj(A) / det(A) when det(A) ≠ 0
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::math::linear_algebra::Matrix3Ext;
    /// use nalgebra::Matrix3;
    ///
    /// let m: Matrix3<f64> = Matrix3::new(
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0,
    /// );
    /// let adj = m.adjugate();
    ///
    /// // Verify: A × adj(A) = det(A) × I
    /// let product = m * adj;
    /// let det = m.determinant();
    /// ```
    fn adjugate(&self) -> Matrix3<f64>;

    // Add more methods here as needed
}

impl Matrix3Ext for Matrix3<f64> {
    fn cofactor_matrix(&self) -> Matrix3<f64> {
        let m = self;

        // Each element is (-1)^(i+j) * minor(i,j)
        // Minor(i,j) is the determinant of the 2×2 submatrix obtained by
        // removing row i and column j from the original matrix.
        //
        // The sign pattern follows a checkerboard: + - + / - + - / + - +
        Matrix3::new(
            // Row 0
            m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)], // C00: +minor(0,0)
            -(m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]), // C01: -minor(0,1)
            m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)], // C02: +minor(0,2)
            // Row 1
            -(m[(0, 1)] * m[(2, 2)] - m[(0, 2)] * m[(2, 1)]), // C10: -minor(1,0)
            m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)],    // C11: +minor(1,1)
            -(m[(0, 0)] * m[(2, 1)] - m[(0, 1)] * m[(2, 0)]), // C12: -minor(1,2)
            // Row 2
            m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)], // C20: +minor(2,0)
            -(m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)]), // C21: -minor(2,1)
            m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)], // C22: +minor(2,2)
        )
    }

    fn adjugate(&self) -> Matrix3<f64> {
        self.cofactor_matrix().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_cofactor_identity() {
        let identity: Matrix3<f64> = Matrix3::identity();
        let cof = identity.cofactor_matrix();

        // Cofactor of identity is identity
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(approx_eq(cof[(i, j)], 1.0));
                } else {
                    assert!(approx_eq(cof[(i, j)], 0.0));
                }
            }
        }
    }

    #[test]
    fn test_adjugate_is_cofactor_transpose() {
        let m: Matrix3<f64> = Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);

        let adj = m.adjugate();
        let cof_transpose = m.cofactor_matrix().transpose();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(adj[(i, j)], cof_transpose[(i, j)]),
                    "At ({}, {}): adj={}, cof^T={}",
                    i,
                    j,
                    adj[(i, j)],
                    cof_transpose[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_adjugate_identity() {
        let identity: Matrix3<f64> = Matrix3::identity();
        let adj = identity.adjugate();

        // Adjugate of identity is identity
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(approx_eq(adj[(i, j)], 1.0));
                } else {
                    assert!(approx_eq(adj[(i, j)], 0.0));
                }
            }
        }
    }

    #[test]
    fn test_adjugate_property() {
        let m: Matrix3<f64> = Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);

        let adj = m.adjugate();
        let product = m * adj;
        let det = m.determinant();

        // m * adj(m) = det(m) * I
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { det } else { 0.0 };
                assert!(
                    approx_eq(product[(i, j)], expected),
                    "At ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    product[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_adjugate_inverse_relationship() {
        let m: Matrix3<f64> = Matrix3::new(2.0, 1.0, 3.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0);

        let adj = m.adjugate();
        let det = m.determinant();

        // adj(m) / det(m) should equal inverse
        if det.abs() > 1e-10 {
            let inv = m.try_inverse().unwrap();
            let adj_over_det = adj / det;

            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        approx_eq(adj_over_det[(i, j)], inv[(i, j)]),
                        "At ({}, {}): adjugate/det={}, inverse={}",
                        i,
                        j,
                        adj_over_det[(i, j)],
                        inv[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_adjugate_known() {
        let m: Matrix3<f64> = Matrix3::new(-3.0, 2.0, -5.0, -1.0, 0.0, -2.0, 3.0, -4.0, 1.0);
        let adj = m.adjugate();
        let expected: Matrix3<f64> =
            Matrix3::new(-8.0, 18.0, -4.0, -5.0, 12.0, -1.0, 4.0, -6.0, 2.0);
        assert_eq!(adj, expected);
    }
}
