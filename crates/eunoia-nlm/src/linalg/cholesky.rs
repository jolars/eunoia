//! Cholesky decomposition operations
//!
//! Port of choldc, chlhsn, and lltslv from nlm.c:212-238, 240-317, 1355-1528

use nalgebra::{DMatrix, DVector};

/// Solve Ax=b where A has the form L*L^T (Cholesky factorization)
///
/// Port of `lltslv` from nlm.c:212-238
///
/// # Arguments
/// * `a` - Matrix in L*L^T form (only lower triangular part is used)
/// * `b` - Right-hand side vector
///
/// # Returns
/// Solution vector x such that A*x = b
///
/// # Note
/// This function uses forward and back substitution to solve the system.
/// The C version uses LINPACK's dtrsl routine.
pub fn lltslv(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    let n = b.len();
    let mut x = b.clone();

    // Forward substitution: L*y = b
    // Solve for y (stored in x)
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..i {
            sum -= a[(i, j)] * x[j];
        }
        x[i] = sum / a[(i, i)];
    }

    // Back substitution: L^T*x = y
    // Solve for x (overwriting x)
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= a[(j, i)] * x[j];
        }
        x[i] = sum / a[(i, i)];
    }

    x
}

/// Find perturbed L*L^T decomposition of A+D
///
/// Port of `choldc` from nlm.c:240-317
///
/// Finds the Cholesky decomposition of a+d, where d is a non-negative diagonal
/// matrix added if necessary to make the decomposition possible.
///
/// # Arguments
/// * `a` - Matrix to decompose (modified in place)
/// * `diagmx` - Maximum diagonal element of original A
/// * `tol` - Tolerance for perturbation
///
/// # Returns
/// Maximum amount added to diagonal (addmax)
///
/// # Description
/// Normal Cholesky decomposition is performed. However, if at any point
/// L(i,i) would be set to sqrt(temp) with temp < tol*diagmx, then L(i,i)
/// is set to sqrt(tol*diagmx) instead. This is equivalent to adding
/// tol*diagmx-temp to A(i,i).
///
/// On return, the lower triangular part and diagonal of `a` contain L.
pub fn choldc(a: &mut DMatrix<f64>, diagmx: f64, tol: f64) -> f64 {
    let n = a.nrows();
    let mut addmax = 0.0;
    let aminl = (diagmx * tol).sqrt();
    let amnlsq = aminl * aminl;

    // Form row i of L
    for i in 0..n {
        // Find i,j element of lower triangular matrix L
        for j in 0..i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += a[(i, k)] * a[(j, k)];
            }
            a[(i, j)] = (a[(i, j)] - sum) / a[(j, j)];
        }

        // Find diagonal element of L
        let mut sum = 0.0;
        for k in 0..i {
            sum += a[(i, k)] * a[(i, k)];
        }

        let tmp1 = a[(i, i)] - sum;

        if tmp1 >= amnlsq {
            // Normal Cholesky
            a[(i, i)] = tmp1.sqrt();
        } else {
            // Augment diagonal of L
            // Find maximum off-diagonal element in row
            let mut offmax = 0.0;
            for j in 0..i {
                let tmp2 = a[(i, j)].abs();
                if offmax < tmp2 {
                    offmax = tmp2;
                }
            }
            if offmax <= amnlsq {
                offmax = amnlsq;
            }

            // Add to diagonal element to allow Cholesky decomposition to continue
            a[(i, i)] = offmax.sqrt();
            let tmp2 = offmax - tmp1;
            if addmax < tmp2 {
                addmax = tmp2;
            }
        }
    }

    addmax
}

/// Find L*L^T decomposition of perturbed model Hessian matrix A+μ*I
///
/// Port of `chlhsn` from nlm.c:1355-1528
///
/// Finds the L*L^T decomposition of A+μ*I where μ≥0 and I is the identity
/// matrix, ensuring the result is safely positive definite. If A is safely
/// positive definite upon entry, then μ=0.
///
/// # Arguments
/// * `a` - Model Hessian matrix (modified in place)
/// * `epsm` - Machine epsilon
/// * `sx` - Diagonal scaling matrix for x
/// * `udiag` - Storage for diagonal of Hessian (output)
///
/// # Description
/// Three-step process:
///
/// 1. If A has negative diagonal elements, choose μ>0 such that diagonal
///    of A+μ*I is all positive with ratio of smallest to largest on order
///    of sqrt(epsm).
///
/// 2. A undergoes perturbed Cholesky decomposition resulting in L*L^T
///    decomposition of A+D, where D is a non-negative diagonal matrix
///    implicitly added during decomposition if A is not positive definite.
///
/// 3. If addmax=0, A was positive definite. Otherwise, calculate minimum
///    sdd to make A safely strictly diagonally dominant, choose
///    μ=min(addmax, sdd) and decompose A+μ*I.
///
/// On entry: A contains model Hessian (lower triangle and diagonal)
/// On exit: A contains L in lower triangle, original Hessian in upper triangle
///          and udiag contains diagonal of Hessian
pub fn chlhsn(a: &mut DMatrix<f64>, epsm: f64, sx: &DVector<f64>, udiag: &mut DVector<f64>) {
    let n = a.nrows();

    // Scale Hessian: pre- and post-multiply A by inv(sx)
    for j in 0..n {
        for i in j..n {
            a[(i, j)] /= sx[i] * sx[j];
        }
    }

    // Step 1: Ensure positive diagonal
    let tol = epsm.sqrt();

    let mut diagmx = a[(0, 0)];
    let mut diagmn = a[(0, 0)];

    if n > 1 {
        for i in 1..n {
            let tmp = a[(i, i)];
            if diagmn > tmp {
                diagmn = tmp;
            }
            if diagmx < tmp {
                diagmx = tmp;
            }
        }
    }

    let posmax = diagmx.max(0.0);

    if diagmn <= posmax * tol {
        let mut amu = tol * (posmax - diagmn) - diagmn;
        if amu == 0.0 {
            // Find largest off-diagonal element of A
            let mut offmax = 0.0;
            for i in 1..n {
                for j in 0..i {
                    let tmp = a[(i, j)].abs();
                    if offmax < tmp {
                        offmax = tmp;
                    }
                }
            }

            if offmax == 0.0 {
                amu = 1.0;
            } else {
                amu = offmax * (tol + 1.0);
            }
        }

        // A = A + μ*I
        for i in 0..n {
            a[(i, i)] += amu;
        }
        diagmx += amu;
    }

    // Step 2: Copy lower triangular part to upper and diagonal to udiag
    for i in 0..n {
        udiag[i] = a[(i, i)];
        for j in 0..i {
            a[(j, i)] = a[(i, j)];
        }
    }

    let addmax = choldc(a, diagmx, tol);

    // Step 3: If addmax > 0, perturb A to be safely diagonally dominant
    if addmax > 0.0 {
        // Restore original A (lower triangular part and diagonal)
        for i in 0..n {
            a[(i, i)] = udiag[i];
            for j in 0..i {
                a[(i, j)] = a[(j, i)];
            }
        }

        // Find sdd such that A+sdd*I is safely positive definite
        let mut evmin = 0.0;
        let mut evmax = a[(0, 0)];

        for i in 0..n {
            let mut offrow = 0.0;
            for j in 0..i {
                offrow += a[(i, j)].abs();
            }
            for j in (i + 1)..n {
                offrow += a[(j, i)].abs();
            }

            let tmp = a[(i, i)] - offrow;
            if evmin > tmp {
                evmin = tmp;
            }

            let tmp = a[(i, i)] + offrow;
            if evmax < tmp {
                evmax = tmp;
            }
        }

        let sdd = tol * (evmax - evmin) - evmin;

        // Perturb A and decompose again
        let amu = sdd.min(addmax);
        for i in 0..n {
            a[(i, i)] += amu;
            udiag[i] = a[(i, i)];
        }

        // A now guaranteed safely positive definite
        choldc(a, 0.0, tol);
    }

    // Unscale Hessian and Cholesky decomposition matrix
    for j in 0..n {
        for i in j..n {
            a[(i, j)] *= sx[i];
        }
        for i in 0..j {
            a[(i, j)] *= sx[i] * sx[j];
        }
        udiag[j] *= sx[j] * sx[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_lltslv_simple() {
        // A = L*L^T where L = [[2, 0], [1, 3]]
        // A = [[4, 2], [2, 10]]
        let l = dmatrix![
            2.0, 0.0;
            1.0, 3.0
        ];
        // For x = [1, 5]: A*x = [[4, 2], [2, 10]] * [1, 5] = [14, 52]
        let b = dvector![14.0, 52.0];
        let x = lltslv(&l, &b);

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_choldc_positive_definite() {
        // Positive definite matrix: [[4, 2], [2, 3]]
        let mut a = dmatrix![
            4.0, 0.0;
            2.0, 3.0
        ];

        let diagmx = 4.0;
        let tol = 1e-8;
        let addmax = choldc(&mut a, diagmx, tol);

        // Should be zero perturbation for positive definite matrix
        assert!(addmax < 1e-10);

        // Verify L*L^T = original matrix
        // L = [[2, 0], [1, sqrt(2)]]
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_choldc_needs_perturbation() {
        // Nearly singular matrix
        let mut a = dmatrix![
            1.0, 0.0;
            0.9, 1e-10
        ];

        let diagmx = 1.0;
        let tol = 1e-6;
        let addmax = choldc(&mut a, diagmx, tol);

        // Should need perturbation
        assert!(addmax > 0.0);
    }

    #[test]
    fn test_chlhsn_positive_definite() {
        // Positive definite Hessian
        let mut a = dmatrix![
            4.0, 0.0;
            2.0, 3.0
        ];

        let epsm = f64::EPSILON;
        let sx = dvector![1.0, 1.0];
        let mut udiag = dvector![0.0, 0.0];

        chlhsn(&mut a, epsm, &sx, &mut udiag);

        // Check that udiag contains original diagonal scaled back
        assert!((udiag[0] - 4.0).abs() < 1e-6);
        assert!((udiag[1] - 3.0).abs() < 1e-6);
    }
}
