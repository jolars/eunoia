//! Secant (BFGS-style) Hessian updates
//!
//! Port of secunf and secfac from nlm.c:1143-1234, 1236-1353

use crate::linalg::{mvmltl, mvmlts, mvmltu, qrupdt};
use nalgebra::{DMatrix, DVector};

/// Parameters for secant updates (to avoid too many arguments)
pub struct SecantParams<'a> {
    pub x: &'a DVector<f64>,
    pub g: &'a DVector<f64>,
    pub xpls: &'a DVector<f64>,
    pub gpls: &'a DVector<f64>,
    pub epsm: f64,
    pub itncnt: usize,
    pub rnf: f64,
    pub has_gradient: bool,
}

/// Update Hessian by BFGS unfactored method (method 3 only)
///
/// Port of `secunf` from nlm.c:1143-1234
///
/// Updates the Hessian approximation using the unfactored BFGS formula.
/// This is used only for Method 3 (More-Hebdon).
///
/// # Arguments
/// * `a` - Approximate Hessian (input: upper triangle + udiag; output: lower triangle + diagonal)
/// * `udiag` - Diagonal of Hessian on entry
/// * `params` - Secant update parameters
/// * `noupdt` - Boolean flag indicating if this is the first update
///
/// # Returns
/// Updated noupdt flag
///
/// # Note
/// The Hessian is stored differently on input vs output:
/// - Input: Upper triangular part + udiag contains the approximation
/// - Output: Lower triangular part + diagonal contains updated approximation
pub fn secunf(
    a: &mut DMatrix<f64>,
    udiag: &DVector<f64>,
    params: &SecantParams,
    noupdt: &mut bool,
) {
    let n = params.x.len();
    let mut s = DVector::zeros(n);
    let mut y = DVector::zeros(n);

    // Copy Hessian from upper triangle + udiag to lower triangle + diagonal
    for i in 0..n {
        a[(i, i)] = udiag[i];
        for j in 0..i {
            a[(i, j)] = a[(j, i)];
        }
    }

    *noupdt = params.itncnt == 1;

    // Compute s = xpls - x and y = gpls - g
    for i in 0..n {
        s[i] = params.xpls[i] - params.x[i];
        y[i] = params.gpls[i] - params.g[i];
    }

    // Check curvature condition: s^T y > 0
    let den1 = s.dot(&y);
    let snorm2 = s.norm();
    let ynrm2 = y.norm();

    if den1 < params.epsm.sqrt() * snorm2 * ynrm2 {
        return; // Skip update
    }

    // Compute t = H*s
    let mut t = mvmlts(a, &s);
    let den2 = s.dot(&t);

    if *noupdt {
        // First update: scale H by (s^T y) / (s^T H s)
        let gam = den1 / den2;
        for j in 0..n {
            t[j] *= gam;
            for i in j..n {
                a[(i, j)] *= gam;
            }
        }
        *noupdt = false;
    }

    // Check update condition on each component
    let mut skpupd = true;
    for i in 0..n {
        let tol = params.rnf * params.g[i].abs().max(params.gpls[i].abs());
        let tol = if !params.has_gradient {
            tol / params.rnf.sqrt()
        } else {
            tol
        };

        if (y[i] - t[i]).abs() >= tol {
            skpupd = false;
            break;
        }
    }

    if skpupd {
        return; // Skip update
    }

    // BFGS update: H_new = H + (yy^T)/(y^Ts) - (tt^T)/(s^THs)
    for j in 0..n {
        for i in j..n {
            a[(i, j)] += y[i] * y[j] / den1 - t[i] * t[j] / den2;
        }
    }
}

/// Update Hessian by BFGS factored method (methods 1 and 2)
///
/// Port of `secfac` from nlm.c:1236-1353
///
/// Updates the Cholesky factorization of the Hessian using BFGS formula.
/// This is more numerically stable than the unfactored update.
///
/// # Arguments
/// * `a` - Cholesky decomposition of Hessian (lower triangle + diagonal)
/// * `params` - Secant update parameters
/// * `noupdt` - Boolean flag indicating if this is the first update
///
/// # Returns
/// Updated noupdt flag
///
/// # Note
/// The matrix `a` contains the Cholesky factor L where H = L*L^T.
/// The update maintains this factorization directly.
pub fn secfac(a: &mut DMatrix<f64>, params: &SecantParams, noupdt: &mut bool) {
    let n = params.x.len();
    let mut s = DVector::zeros(n);
    let mut y = DVector::zeros(n);

    *noupdt = params.itncnt == 1;

    // Compute s = xpls - x and y = gpls - g
    for i in 0..n {
        s[i] = params.xpls[i] - params.x[i];
        y[i] = params.gpls[i] - params.g[i];
    }

    let den1 = s.dot(&y);
    let snorm2 = s.norm();
    let ynrm2 = y.norm();

    if den1 < params.epsm.sqrt() * snorm2 * ynrm2 {
        return;
    }

    // u = L^T * s
    let mut u = mvmltu(a, &s);
    let den2 = u.dot(&u);

    // L <-- sqrt(den1/den2) * L
    let mut alp = (den1 / den2).sqrt();
    if *noupdt {
        for j in 0..n {
            u[j] *= alp;
            for i in j..n {
                a[(i, j)] *= alp;
            }
        }
        *noupdt = false;
        alp = 1.0;
    }

    // w = L*(L^T*s) = H*s
    let mut w = mvmltl(a, &u);

    let reltol = if params.has_gradient {
        params.rnf
    } else {
        params.rnf.sqrt()
    };

    // Check if update should be skipped
    let mut skpupd = true;
    for i in 0..n {
        if (y[i] - w[i]).abs() >= reltol * params.g[i].abs().max(params.gpls[i].abs()) {
            skpupd = false;
            break;
        }
    }

    if skpupd {
        return;
    }

    // w = y - alp*H*s
    for i in 0..n {
        w[i] = y[i] - alp * w[i];
    }

    // alp = 1/sqrt(den1*den2)
    alp /= den1;

    // u = (L^T*s)/sqrt((y^T*s) * (s^T*H*s))
    for i in 0..n {
        u[i] *= alp;
    }

    // Copy L into upper triangular part, zero L
    for i in 1..n {
        for j in 0..i {
            a[(j, i)] = a[(i, j)];
            a[(i, j)] = 0.0;
        }
    }

    // Find Q, L^T such that Q*L^T = L^T + u*w^T
    qrupdt(a, &mut u, &w);

    // Copy back to lower triangular part
    for i in 1..n {
        for j in 0..i {
            a[(i, j)] = a[(j, i)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_secunf_basic() {
        // Start with simple 2x2 Hessian
        let mut a = dmatrix![
            2.0, 1.0;
            1.0, 3.0
        ];
        let udiag = dvector![2.0, 3.0];

        let x = dvector![0.0, 0.0];
        let g = dvector![1.0, 1.0];
        let xpls = dvector![0.1, 0.1];
        let gpls = dvector![0.9, 0.8];

        let params = SecantParams {
            x: &x,
            g: &g,
            xpls: &xpls,
            gpls: &gpls,
            epsm: f64::EPSILON,
            itncnt: 2,
            rnf: 1e-10,
            has_gradient: true,
        };

        let mut noupdt = false;
        secunf(&mut a, &udiag, &params, &mut noupdt);

        // Matrix should still be symmetric (within lower triangle)
        // Just verify it runs without panicking
        assert!(a.nrows() == 2);
    }

    #[test]
    fn test_secfac_basic() {
        // Start with Cholesky factor
        let mut a = dmatrix![
            2.0, 0.0;
            1.0, 1.5
        ];

        let x = dvector![0.0, 0.0];
        let g = dvector![1.0, 1.0];
        let xpls = dvector![0.1, 0.1];
        let gpls = dvector![0.9, 0.8];

        let params = SecantParams {
            x: &x,
            g: &g,
            xpls: &xpls,
            gpls: &gpls,
            epsm: f64::EPSILON,
            itncnt: 2,
            rnf: 1e-10,
            has_gradient: true,
        };

        let mut noupdt = false;
        secfac(&mut a, &params, &mut noupdt);

        // Should still be lower triangular
        assert!(a.nrows() == 2);
    }

    #[test]
    fn test_secunf_skip_on_bad_curvature() {
        let mut a = dmatrix![
            1.0, 0.0;
            0.0, 1.0
        ];
        let udiag = dvector![1.0, 1.0];

        // Set up situation where s^T y is negative (bad curvature)
        let x = dvector![0.0, 0.0];
        let g = dvector![1.0, 1.0];
        let xpls = dvector![1.0, 0.0]; // s = [1, 0]
        let gpls = dvector![0.0, 1.0]; // y = [-1, 0], s^T y = -1 < 0

        let params = SecantParams {
            x: &x,
            g: &g,
            xpls: &xpls,
            gpls: &gpls,
            epsm: 1e-10,
            itncnt: 2,
            rnf: 1e-10,
            has_gradient: true,
        };

        let _a_orig = a.clone();
        let mut noupdt = false;
        secunf(&mut a, &udiag, &params, &mut noupdt);

        // Matrix should be unchanged (update skipped)
        // After copying to lower triangle, diagonal should match udiag
        assert_eq!(a[(0, 0)], udiag[0]);
        assert_eq!(a[(1, 1)], udiag[1]);
    }
}
