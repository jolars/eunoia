//! Hessian computation using finite differences
//!
//! Port of fdhess and sndofd from nlm.c:48-114, 1676-1748

use crate::types::ObjectiveFn;
use nalgebra::{DMatrix, DVector};

/// Finite difference Hessian approximation (Algorithm A5.6.2)
///
/// Port of `fdhess` from nlm.c:48-114
///
/// Calculates numerical approximation to upper triangular portion of
/// the second derivative matrix (Hessian).
///
/// # Algorithm
///
/// Algorithm A5.6.2 from Dennis and Schnabel (1983), pp. 321-322
///
/// # Arguments
/// * `x` - Vector of parameter values
/// * `f_val` - Function value at x
/// * `func` - Objective function
/// * `ndigit` - Number of good digits in function evaluations
/// * `typx` - Typical size for each component of x
///
/// # Returns
/// Hessian matrix (symmetric, but only lower triangle is computed)
pub fn fdhess(
    x: &DVector<f64>,
    f_val: f64,
    func: &ObjectiveFn,
    ndigit: i32,
    typx: &DVector<f64>,
) -> DMatrix<f64> {
    let n = x.len();
    let eta = 10.0_f64.powf(-ndigit as f64 / 3.0);

    let mut h = DMatrix::zeros(n, n);
    let mut step = DVector::zeros(n);
    let mut f = DVector::zeros(n);

    // Compute step sizes and function values at x + step[i]
    for i in 0..n {
        step[i] = eta * x[i].abs().max(typx[i].abs());
        if typx[i] < 0.0 {
            step[i] = -step[i];
        }

        let mut x_temp = x.clone();
        x_temp[i] += step[i];
        step[i] = x_temp[i] - x[i]; // Actual step taken
        f[i] = func(&x_temp);
    }

    // Compute Hessian elements
    for i in 0..n {
        // Diagonal element
        let mut x_temp = x.clone();
        x_temp[i] += 2.0 * step[i];
        let fii = func(&x_temp);
        h[(i, i)] = (f_val - f[i] + (fii - f[i])) / (step[i] * step[i]);

        // Off-diagonal elements (lower triangle)
        x_temp[i] = x[i] + step[i];
        for j in (i + 1)..n {
            x_temp[j] = x[j] + step[j];
            let fij = func(&x_temp);
            h[(i, j)] = (f_val - f[i] + (fij - f[j])) / (step[i] * step[j]);
            h[(j, i)] = h[(i, j)]; // Symmetric
            x_temp[j] = x[j];
        }
    }

    h
}

/// Second-order forward finite difference Hessian approximation
///
/// Port of `sndofd` from nlm.c:1676-1748
///
/// Finds second order forward finite difference approximation to the
/// Hessian of the function. Used when no analytical gradient or Hessian
/// is available and the optimization function is inexpensive to evaluate.
///
/// # Arguments
/// * `xpls` - Current iterate
/// * `fpls` - Function value at xpls
/// * `func` - Objective function
/// * `sx` - Diagonal scaling matrix for x
/// * `rnoise` - Relative noise in function evaluations
///
/// # Returns
/// Finite difference approximation to Hessian (lower triangle + diagonal)
///
/// # Note
/// Only the lower triangular part and diagonal are computed and returned.
pub fn sndofd(
    xpls: &DVector<f64>,
    fpls: f64,
    func: &ObjectiveFn,
    sx: &DVector<f64>,
    rnoise: f64,
) -> DMatrix<f64> {
    let n = xpls.len();
    let mut a = DMatrix::zeros(n, n);
    let mut stepsz = DVector::zeros(n);
    let mut anbr = DVector::zeros(n);

    // Find i-th stepsize and evaluate neighbor in direction of i-th unit vector
    for i in 0..n {
        let xtmpi = xpls[i];
        stepsz[i] = rnoise.powf(1.0 / 3.0) * (xtmpi.abs()).max(1.0 / sx[i]);

        let mut x_temp = xpls.clone();
        x_temp[i] = xtmpi + stepsz[i];
        anbr[i] = func(&x_temp);
    }

    // Calculate row i of a
    for i in 0..n {
        let xtmpi = xpls[i];

        // Diagonal element
        let mut x_temp = xpls.clone();
        x_temp[i] = xtmpi + stepsz[i] * 2.0;
        let fhat = func(&x_temp);
        a[(i, i)] = ((fpls - anbr[i]) + (fhat - anbr[i])) / (stepsz[i] * stepsz[i]);

        // Calculate sub-diagonal elements of column
        if i == 0 {
            continue;
        }

        x_temp[i] = xtmpi + stepsz[i];
        for j in 0..i {
            let xtmpj = x_temp[j];
            x_temp[j] = xtmpj + stepsz[j];
            let fhat = func(&x_temp);
            a[(i, j)] = ((fpls - anbr[i]) + (fhat - anbr[j])) / (stepsz[i] * stepsz[j]);
            x_temp[j] = xtmpj;
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_fdhess_quadratic() {
        // Test with quadratic function f(x,y) = x^2 + 2xy + 3y^2
        // Hessian should be [[2, 2], [2, 6]]
        let func = |x: &DVector<f64>| x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1];

        let x = dvector![1.0, 1.0];
        let f_val = func(&x);
        let typx = dvector![1.0, 1.0];
        let ndigit = 15;

        let h = fdhess(&x, f_val, &func, ndigit, &typx);

        // Check Hessian values (allowing for numerical error)
        assert!((h[(0, 0)] - 2.0).abs() < 0.1, "H[0,0] = {}", h[(0, 0)]);
        assert!((h[(0, 1)] - 2.0).abs() < 0.1, "H[0,1] = {}", h[(0, 1)]);
        assert!((h[(1, 0)] - 2.0).abs() < 0.1, "H[1,0] = {}", h[(1, 0)]);
        assert!((h[(1, 1)] - 6.0).abs() < 0.1, "H[1,1] = {}", h[(1, 1)]);
    }

    #[test]
    fn test_fdhess_simple() {
        // Simple function: f(x) = x^2
        // Hessian should be [[2]]
        let func = |x: &DVector<f64>| x[0] * x[0];

        let x = dvector![2.0];
        let f_val = func(&x);
        let typx = dvector![1.0];
        let ndigit = 15;

        let h = fdhess(&x, f_val, &func, ndigit, &typx);

        assert!((h[(0, 0)] - 2.0).abs() < 0.01, "H[0,0] = {}", h[(0, 0)]);
    }

    #[test]
    fn test_sndofd_quadratic() {
        // Test with f(x,y) = x^2 + y^2
        // Hessian should be [[2, 0], [0, 2]]
        let func = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];

        let xpls = dvector![1.0, 1.0];
        let fpls = func(&xpls);
        let sx = dvector![1.0, 1.0];
        let rnoise = 1e-10;

        let a = sndofd(&xpls, fpls, &func, &sx, rnoise);

        // Diagonal should be close to 2
        assert!((a[(0, 0)] - 2.0).abs() < 0.1, "a[0,0] = {}", a[(0, 0)]);
        assert!((a[(1, 1)] - 2.0).abs() < 0.1, "a[1,1] = {}", a[(1, 1)]);

        // Off-diagonal should be close to 0
        assert!(a[(1, 0)].abs() < 0.1, "a[1,0] = {}", a[(1, 0)]);
    }

    #[test]
    fn test_sndofd_simple() {
        // f(x) = x^2, Hessian = [[2]]
        let func = |x: &DVector<f64>| x[0] * x[0];

        let xpls = dvector![1.0];
        let fpls = func(&xpls);
        let sx = dvector![1.0];
        let rnoise = 1e-10;

        let a = sndofd(&xpls, fpls, &func, &sx, rnoise);

        assert!((a[(0, 0)] - 2.0).abs() < 0.01, "a[0,0] = {}", a[(0, 0)]);
    }
}
