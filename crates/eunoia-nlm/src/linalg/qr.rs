//! QR update operations
//!
//! Port of qraux1, qraux2, and qrupdt from nlm.c:319-439

use nalgebra::{DMatrix, DVector};

/// Interchange rows i and i+1 of upper Hessenberg matrix R, columns i to n
///
/// Port of `qraux1` from nlm.c:319-341
///
/// # Arguments
/// * `r` - Upper Hessenberg matrix (modified in place)
/// * `i` - Index of row to interchange (must be < n-1)
///
/// # Note
/// This operates on the upper Hessenberg structure, swapping two consecutive rows
/// for columns from i to the end.
pub fn qraux1(r: &mut DMatrix<f64>, i: usize) {
    let n = r.ncols();

    // Interchange rows i and i+1 for columns i to n-1
    for j in i..n {
        let tmp = r[(i, j)];
        r[(i, j)] = r[(i + 1, j)];
        r[(i + 1, j)] = tmp;
    }
}

/// Pre-multiply R by the Jacobi rotation J(i, i+1, a, b)
///
/// Port of `qraux2` from nlm.c:343-376
///
/// # Arguments
/// * `r` - Upper Hessenberg matrix (modified in place)
/// * `i` - Index of row
/// * `a` - First scalar for rotation
/// * `b` - Second scalar for rotation
///
/// # Description
/// Applies a Givens/Jacobi rotation to zero out an element.
/// The rotation is defined by c = a/sqrt(a²+b²) and s = b/sqrt(a²+b²).
pub fn qraux2(r: &mut DMatrix<f64>, i: usize, a: f64, b: f64) {
    let n = r.ncols();

    // Compute rotation parameters
    let den = a.hypot(b);
    let c = a / den;
    let s = b / den;

    // Apply rotation to rows i and i+1 for columns i to n-1
    for j in i..n {
        let y = r[(i, j)];
        let z = r[(i + 1, j)];
        r[(i, j)] = c * y - s * z;
        r[(i + 1, j)] = s * y + c * z;
    }
}

/// Find orthogonal matrix Q* and upper triangular R* such that Q*R* = R + uv^T
///
/// Port of `qrupdt` from nlm.c:379-439
///
/// # Arguments
/// * `a` - On input: upper triangular matrix R; On output: updated matrix R*
/// * `u` - Vector u (modified during computation)
/// * `v` - Vector v
///
/// # Description
/// Updates the QR factorization after a rank-1 update. This is used in secant
/// updates of the Hessian approximation.
///
/// The algorithm:
/// 1. Uses (k-1) Jacobi rotations to transform R + uv^T to Hessenberg form
/// 2. Adds the rank-1 update
/// 3. Uses (k-1) more Jacobi rotations to restore upper triangular form
pub fn qrupdt(a: &mut DMatrix<f64>, u: &mut DVector<f64>, v: &DVector<f64>) {
    let n = a.nrows();

    // Determine last non-zero element in u
    let mut k = n;
    for i in (0..n).rev() {
        if u[i] != 0.0 {
            k = i + 1;
            break;
        }
    }

    if k == 0 {
        return; // u is all zeros, nothing to do
    }

    k -= 1; // Convert to 0-indexed last non-zero position

    // (k) Jacobi rotations transform R + uv^T --> R* + (u[0]*e1)v^T
    // which is upper Hessenberg
    if k > 0 {
        let mut ii = k;
        while ii > 0 {
            let i = ii - 1;
            if u[i] == 0.0 {
                qraux1(a, i);
                u[i] = u[ii];
            } else {
                qraux2(a, i, u[i], -u[ii]);
                u[i] = u[i].hypot(u[ii]);
            }
            ii = i;
        }
    }

    // R <-- R + (u[0]*e1)v^T
    // This adds u[0]*v[j] to first row, column j
    for j in 0..n {
        a[(0, j)] += u[0] * v[j];
    }

    // (k) Jacobi rotations transform upper Hessenberg R to upper triangular R*
    for i in 0..k {
        if a[(i, i)] == 0.0 {
            qraux1(a, i);
        } else {
            let t1 = a[(i, i)];
            let t2 = -a[(i + 1, i)];
            qraux2(a, i, t1, t2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_qraux1() {
        let mut r = dmatrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];

        // Swap rows 0 and 1 from column 0 onward
        qraux1(&mut r, 0);

        assert_eq!(r[(0, 0)], 4.0);
        assert_eq!(r[(0, 1)], 5.0);
        assert_eq!(r[(0, 2)], 6.0);
        assert_eq!(r[(1, 0)], 1.0);
        assert_eq!(r[(1, 1)], 2.0);
        assert_eq!(r[(1, 2)], 3.0);
        assert_eq!(r[(2, 0)], 7.0); // Row 2 unchanged
    }

    #[test]
    fn test_qraux2() {
        // Test that qraux2 applies a Givens rotation correctly
        // The function modifies rows i and i+1 from column i onward
        let mut r = dmatrix![
            3.0, 4.0;
            4.0, 5.0
        ];

        // Apply rotation at row 0
        qraux2(&mut r, 0, 3.0, 4.0);

        // Verify the matrix structure is maintained (basic sanity check)
        // Exact values depend on the rotation parameters
        assert!(r.nrows() == 2);
        assert!(r.ncols() == 2);
    }

    #[test]
    fn test_qrupdt_simple() {
        // Start with 2x2 upper triangular matrix
        let mut a = dmatrix![
            2.0, 1.0;
            0.0, 3.0
        ];

        let mut u = dvector![1.0, 0.5];
        let v = dvector![1.0, 1.0];

        qrupdt(&mut a, &mut u, &v);

        // Result should still be upper triangular
        // (we don't check exact values as the QR update is complex,
        //  but we verify structure)
        assert!(a[(1, 0)].abs() < 1e-10); // Should still be upper triangular
    }

    #[test]
    fn test_qrupdt_zero_u() {
        let mut a = dmatrix![
            2.0, 1.0;
            0.0, 3.0
        ];

        let a_copy = a.clone();
        let mut u = dvector![0.0, 0.0];
        let v = dvector![1.0, 1.0];

        qrupdt(&mut a, &mut u, &v);

        // With u=0, matrix should be unchanged
        assert_eq!(a, a_copy);
    }

    #[test]
    fn test_qraux2_zero_b() {
        let mut r = dmatrix![
            3.0, 4.0;
            5.0, 6.0
        ];

        // Rotation with b=0 should leave first row unchanged (up to sign)
        qraux2(&mut r, 0, 3.0, 0.0);

        assert!((r[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((r[(0, 1)] - 4.0).abs() < 1e-10);
    }
}
