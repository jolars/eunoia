//! Matrix-vector multiplication operations
//!
//! Port of mvmltl, mvmltu, and mvmlts from nlm.c:130-210

use nalgebra::{DMatrix, DVector};

/// Compute y = L*x where L is a lower triangular matrix
///
/// Port of `mvmltl` from nlm.c:130-156
///
/// # Arguments
/// * `a` - Lower triangular (n×n) matrix stored in column-major order
/// * `x` - Operand vector of length n
///
/// # Returns
/// Result vector y = L*x
///
/// # Note
/// Only the lower triangular part of `a` is used
pub fn mvmltl(a: &DMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    let n = x.len();
    let mut y = DVector::zeros(n);

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..=i {
            // C uses column-major: a[i + j * nr]
            // nalgebra uses (row, col): a[(i, j)]
            sum += a[(i, j)] * x[j];
        }
        y[i] = sum;
    }

    y
}

/// Compute y = L^T * x where L is a lower triangular matrix
///
/// Port of `mvmltu` from nlm.c:158-179
///
/// # Arguments
/// * `a` - Lower triangular (n×n) matrix
/// * `x` - Operand vector of length n
///
/// # Returns
/// Result vector y = L^T * x
///
/// # Note
/// L-transpose is taken implicitly. Only lower triangular part is used.
pub fn mvmltu(a: &DMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    let n = x.len();
    let mut y = DVector::zeros(n);

    // For each row i of the result:
    // y[i] = dot(column i of L^T, x[i:])
    //      = dot(row i of L, x[i:])
    for i in 0..n {
        let mut sum = 0.0;
        // C code: ddot(&length, &a[i + i * nr], &one, &x[i], &one)
        // This computes dot product of a[i:n, i] with x[i:n]
        for j in i..n {
            sum += a[(j, i)] * x[j];
        }
        y[i] = sum;
    }

    y
}

/// Compute y = A*x where A is a symmetric matrix stored in lower triangular form
///
/// Port of `mvmlts` from nlm.c:181-210
///
/// # Arguments
/// * `a` - Symmetric (n×n) matrix with only lower triangle and diagonal stored
/// * `x` - Operand vector of length n
///
/// # Returns
/// Result vector y = A*x
///
/// # Note
/// Only the lower triangular part and diagonal of `a` are accessed.
/// The upper triangular part is assumed symmetric.
pub fn mvmlts(a: &DMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    let n = x.len();
    let mut y = DVector::zeros(n);

    for i in 0..n {
        let mut sum = 0.0;

        // Sum over lower triangular part (j <= i)
        for j in 0..=i {
            sum += a[(i, j)] * x[j];
        }

        // Sum over upper triangular part using symmetry (j > i)
        // a[i,j] = a[j,i] for symmetric matrix
        for j in (i + 1)..n {
            sum += a[(j, i)] * x[j];
        }

        y[i] = sum;
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_mvmltl() {
        // Lower triangular matrix:
        // [2  0  0]
        // [3  4  0]
        // [5  6  7]
        let a = dmatrix![
            2.0, 0.0, 0.0;
            3.0, 4.0, 0.0;
            5.0, 6.0, 7.0
        ];
        let x = dvector![1.0, 2.0, 3.0];

        // Expected: [2*1, 3*1+4*2, 5*1+6*2+7*3]
        //         = [2, 11, 38]
        let y = mvmltl(&a, &x);

        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
        assert!((y[2] - 38.0).abs() < 1e-10);
    }

    #[test]
    fn test_mvmltu() {
        // Lower triangular matrix L:
        // [2  0  0]
        // [3  4  0]
        // [5  6  7]
        // L^T:
        // [2  3  5]
        // [0  4  6]
        // [0  0  7]
        let a = dmatrix![
            2.0, 0.0, 0.0;
            3.0, 4.0, 0.0;
            5.0, 6.0, 7.0
        ];
        let x = dvector![1.0, 2.0, 3.0];

        // Expected: L^T * x
        // [2*1+3*2+5*3, 4*2+6*3, 7*3]
        // [23, 26, 21]
        let y = mvmltu(&a, &x);

        assert!((y[0] - 23.0).abs() < 1e-10);
        assert!((y[1] - 26.0).abs() < 1e-10);
        assert!((y[2] - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_mvmlts() {
        // Symmetric matrix stored in lower triangle:
        // [2  3  5]
        // [3  4  6]
        // [5  6  7]
        let a = dmatrix![
            2.0, 0.0, 0.0;
            3.0, 4.0, 0.0;
            5.0, 6.0, 7.0
        ];
        let x = dvector![1.0, 2.0, 3.0];

        // Expected: A * x
        // [2*1+3*2+5*3, 3*1+4*2+6*3, 5*1+6*2+7*3]
        // [23, 29, 38]
        let y = mvmlts(&a, &x);

        assert!((y[0] - 23.0).abs() < 1e-10);
        assert!((y[1] - 29.0).abs() < 1e-10);
        assert!((y[2] - 38.0).abs() < 1e-10);
    }

    #[test]
    fn test_mvmltl_identity() {
        let a = DMatrix::identity(3, 3);
        let x = dvector![1.0, 2.0, 3.0];
        let y = mvmltl(&a, &x);

        assert_eq!(y, x);
    }

    #[test]
    fn test_mvmltu_identity() {
        let a = DMatrix::identity(3, 3);
        let x = dvector![1.0, 2.0, 3.0];
        let y = mvmltu(&a, &x);

        assert_eq!(y, x);
    }

    #[test]
    fn test_mvmlts_identity() {
        let a = DMatrix::identity(3, 3);
        let x = dvector![1.0, 2.0, 3.0];
        let y = mvmlts(&a, &x);

        assert_eq!(y, x);
    }
}
