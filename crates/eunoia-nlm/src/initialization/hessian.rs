//! Hessian initialization for secant updates
//!
//! Port of hsnint from nlm.c:1531-1556

use crate::types::Method;
use nalgebra::{DMatrix, DVector};

/// Provide initial Hessian when using secant updates
///
/// Port of `hsnint` from nlm.c:1531-1556
///
/// Initializes the Hessian approximation for secant update methods.
/// The initialization depends on the method being used:
/// - Methods 1 & 2 (factored secant): Initialize with diagonal = sx
/// - Method 3 (unfactored secant): Initialize with diagonal = sx²
///
/// # Arguments
/// * `n` - Dimension of problem
/// * `sx` - Diagonal scaling matrix for x
/// * `method` - Algorithm method (affects initialization)
///
/// # Returns
/// Initial Hessian as lower triangular matrix with appropriate diagonal
///
/// # Note
/// Returns a matrix with:
/// - Diagonal elements set based on method
/// - Off-diagonal elements (lower triangle) set to zero
pub fn hsnint(n: usize, sx: &DVector<f64>, method: Method) -> DMatrix<f64> {
    let mut a = DMatrix::zeros(n, n);

    for i in 0..n {
        // Set diagonal based on method
        a[(i, i)] = match method {
            Method::MoreHebdon => sx[i] * sx[i], // Method 3: unfactored (H diag = sx^2)
            _ => sx[i],                          // Methods 1,2: factored (L diag = sx)
        };

        // Off-diagonal elements are zero (lower triangle)
        for j in 0..i {
            a[(i, j)] = 0.0;
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_hsnint_line_search() {
        let sx = dvector![2.0, 3.0, 4.0];
        let a = hsnint(3, &sx, Method::LineSearch);

        // For methods 1 and 2, diagonal should be sx (factored L diag)
        assert_eq!(a[(0, 0)], 2.0);
        assert_eq!(a[(1, 1)], 3.0);
        assert_eq!(a[(2, 2)], 4.0);

        // Off-diagonal should be zero
        assert_eq!(a[(1, 0)], 0.0);
        assert_eq!(a[(2, 0)], 0.0);
        assert_eq!(a[(2, 1)], 0.0);
    }

    #[test]
    fn test_hsnint_dogleg() {
        let sx = dvector![2.0, 3.0];
        let a = hsnint(2, &sx, Method::DoubleDogleg);

        // For method 2, diagonal should be sx (factored L diag)
        assert_eq!(a[(0, 0)], 2.0);
        assert_eq!(a[(1, 1)], 3.0);

        // Off-diagonal should be zero
        assert_eq!(a[(1, 0)], 0.0);
    }

    #[test]
    fn test_hsnint_more_hebdon() {
        let sx = dvector![2.0, 3.0, 4.0];
        let a = hsnint(3, &sx, Method::MoreHebdon);

        // For method 3, diagonal should be sx²
        assert_eq!(a[(0, 0)], 4.0); // 2²
        assert_eq!(a[(1, 1)], 9.0); // 3²
        assert_eq!(a[(2, 2)], 16.0); // 4²

        // Off-diagonal should be zero
        assert_eq!(a[(1, 0)], 0.0);
        assert_eq!(a[(2, 0)], 0.0);
        assert_eq!(a[(2, 1)], 0.0);
    }

    #[test]
    fn test_hsnint_single_dimension() {
        let sx = dvector![5.0];

        let a1 = hsnint(1, &sx, Method::LineSearch);
        assert_eq!(a1[(0, 0)], 5.0);

        let a3 = hsnint(1, &sx, Method::MoreHebdon);
        assert_eq!(a3[(0, 0)], 25.0);
    }

    #[test]
    fn test_hsnint_identity_scale() {
        let sx = dvector![1.0, 1.0, 1.0];

        let a = hsnint(3, &sx, Method::LineSearch);

        // With unit scaling, L diag should be 1.0
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 1)], 1.0);
        assert_eq!(a[(2, 2)], 1.0);
    }
}
