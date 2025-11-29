//! Gradient computation using finite differences
//!
//! Port of fstofd and fstocd from nlm.c:1558-1636, 1638-1674

use crate::types::{GradientFn, ObjectiveFn};
use nalgebra::{DMatrix, DVector};

/// Forward finite difference approximation to derivatives
///
/// Port of `fstofd` from nlm.c:1558-1636
///
/// Finds first order forward finite difference approximation to the first
/// derivative of the function. Can be used to estimate:
/// 1. Gradient of optimization function (m=1, icase=1)
/// 2. Hessian via gradient differences (m=n, icase=3)
///
/// # Arguments
/// * `m` - Number of rows in output (1 for gradient, n for Hessian)
/// * `n` - Dimension of problem
/// * `xpls` - Current point
/// * `func_or_grad` - Either objective function (m=1) or gradient function (m=n)
/// * `fpls` - Function/gradient value at xpls
/// * `sx` - Diagonal scaling matrix
/// * `rnoise` - Relative noise in function evaluations
/// * `icase` - 1: optimization (gradient), 3: optimization (hessian)
///
/// # Returns
/// Matrix a where:
/// - For gradient (m=1): a is 1×n gradient vector
/// - For Hessian (m=n): a is n×n Hessian matrix (lower triangle + diagonal)
pub fn fstofd_generic(
    m: usize,
    n: usize,
    xpls: &DVector<f64>,
    func_or_grad: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    fpls: &DVector<f64>,
    sx: &DVector<f64>,
    rnoise: f64,
) -> DMatrix<f64> {
    let mut a = DMatrix::zeros(m, n);

    // Find j-th column of a
    // Each column is derivative with respect to xpls[j]
    for j in 0..n {
        let temp1 = xpls[j].abs();
        let temp2 = 1.0 / sx[j];
        let stepsz = rnoise.sqrt() * temp1.max(temp2);

        let mut x_temp = xpls.clone();
        x_temp[j] += stepsz;
        let fhat = func_or_grad(&x_temp);

        for i in 0..m {
            a[(i, j)] = (fhat[i] - fpls[i]) / stepsz;
        }
    }

    a
}

/// Forward difference gradient approximation (convenience wrapper)
///
/// Estimates gradient of objective function using forward differences.
///
/// # Arguments
/// * `xpls` - Current point
/// * `fpls` - Function value at xpls
/// * `func` - Objective function
/// * `sx` - Diagonal scaling matrix
/// * `rnoise` - Relative noise in function evaluations
///
/// # Returns
/// Gradient vector
pub fn fstofd_gradient(
    xpls: &DVector<f64>,
    fpls: f64,
    func: &ObjectiveFn,
    sx: &DVector<f64>,
    rnoise: f64,
) -> DVector<f64> {
    let n = xpls.len();
    let mut g = DVector::zeros(n);

    // Wrapper to make objective function return a vector
    let func_vec = |x: &DVector<f64>| {
        let val = func(x);
        DVector::from_element(1, val)
    };

    let fpls_vec = DVector::from_element(1, fpls);
    let a = fstofd_generic(1, n, xpls, &func_vec, &fpls_vec, sx, rnoise);

    // Extract gradient from first row
    for j in 0..n {
        g[j] = a[(0, j)];
    }

    g
}

/// Forward difference Hessian from gradient
///
/// Estimates Hessian by taking finite differences of the gradient.
///
/// # Arguments
/// * `xpls` - Current point
/// * `gpls` - Gradient at xpls
/// * `grad_func` - Gradient function
/// * `sx` - Diagonal scaling matrix
/// * `rnoise` - Relative noise in function evaluations
///
/// # Returns
/// Hessian matrix (symmetrized)
pub fn fstofd_hessian(
    xpls: &DVector<f64>,
    gpls: &DVector<f64>,
    grad_func: &GradientFn,
    sx: &DVector<f64>,
    rnoise: f64,
) -> DMatrix<f64> {
    let n = xpls.len();
    let mut a = fstofd_generic(n, n, xpls, grad_func, gpls, sx, rnoise);

    // If computing Hessian, must be symmetric (symmetrize lower triangle)
    if n > 1 {
        for i in 1..n {
            for j in 0..i {
                a[(i, j)] = (a[(i, j)] + a[(j, i)]) / 2.0;
                a[(j, i)] = a[(i, j)];
            }
        }
    }

    a
}

/// Central difference gradient approximation
///
/// Port of `fstocd` from nlm.c:1638-1674
///
/// Finds central difference approximation to the gradient. More accurate
/// than forward differences but requires twice as many function evaluations.
/// Used as a fallback when forward differences fail to find a satisfactory step.
///
/// # Arguments
/// * `x` - Point at which to approximate gradient
/// * `func` - Objective function
/// * `sx` - Diagonal scaling matrix
/// * `rnoise` - Relative noise in function evaluations
///
/// # Returns
/// Central difference approximation to gradient
pub fn fstocd(
    x: &DVector<f64>,
    func: &ObjectiveFn,
    sx: &DVector<f64>,
    rnoise: f64,
) -> DVector<f64> {
    let n = x.len();
    let mut g = DVector::zeros(n);

    // Find i-th stepsize, evaluate two neighbors in direction of i-th unit vector,
    // and evaluate i-th component of gradient
    for i in 0..n {
        let xtempi = x[i];
        let temp1 = xtempi.abs();
        let temp2 = 1.0 / sx[i];
        let stepi = rnoise.powf(1.0 / 3.0) * temp1.max(temp2);

        // Function value at x + stepi*e_i
        let mut x_plus = x.clone();
        x_plus[i] = xtempi + stepi;
        let fplus = func(&x_plus);

        // Function value at x - stepi*e_i
        let mut x_minus = x.clone();
        x_minus[i] = xtempi - stepi;
        let fminus = func(&x_minus);

        // Central difference
        g[i] = (fplus - fminus) / (stepi * 2.0);
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_fstofd_gradient_simple() {
        // f(x) = x^2, gradient = 2x
        let func = |x: &DVector<f64>| x[0] * x[0];

        let x = dvector![2.0];
        let f = func(&x);
        let sx = dvector![1.0];
        let rnoise = 1e-10;

        let g = fstofd_gradient(&x, f, &func, &sx, rnoise);

        // Gradient at x=2 should be 4
        assert!((g[0] - 4.0).abs() < 0.01, "g[0] = {}", g[0]);
    }

    #[test]
    fn test_fstofd_gradient_multivariate() {
        // f(x,y) = x^2 + 2y^2
        // gradient = [2x, 4y]
        let func = |x: &DVector<f64>| x[0] * x[0] + 2.0 * x[1] * x[1];

        let x = dvector![1.0, 2.0];
        let f = func(&x);
        let sx = dvector![1.0, 1.0];
        let rnoise = 1e-10;

        let g = fstofd_gradient(&x, f, &func, &sx, rnoise);

        // Gradient at (1,2) should be [2, 8]
        assert!((g[0] - 2.0).abs() < 0.01, "g[0] = {}", g[0]);
        assert!((g[1] - 8.0).abs() < 0.01, "g[1] = {}", g[1]);
    }

    #[test]
    fn test_fstocd_simple() {
        // f(x) = x^2, gradient = 2x
        let func = |x: &DVector<f64>| x[0] * x[0];

        let x = dvector![3.0];
        let sx = dvector![1.0];
        let rnoise = 1e-10;

        let g = fstocd(&x, &func, &sx, rnoise);

        // Gradient at x=3 should be 6
        assert!((g[0] - 6.0).abs() < 0.001, "g[0] = {}", g[0]);
    }

    #[test]
    fn test_fstocd_multivariate() {
        // f(x,y) = x^2 + 3y^2
        // gradient = [2x, 6y]
        let func = |x: &DVector<f64>| x[0] * x[0] + 3.0 * x[1] * x[1];

        let x = dvector![2.0, 1.0];
        let sx = dvector![1.0, 1.0];
        let rnoise = 1e-10;

        let g = fstocd(&x, &func, &sx, rnoise);

        // Gradient at (2,1) should be [4, 6]
        assert!((g[0] - 4.0).abs() < 0.001, "g[0] = {}", g[0]);
        assert!((g[1] - 6.0).abs() < 0.001, "g[1] = {}", g[1]);
    }

    #[test]
    fn test_fstofd_hessian() {
        // f(x,y) = x^2 + 2xy + 3y^2
        // gradient = [2x + 2y, 2x + 6y]
        // Hessian = [[2, 2], [2, 6]]
        let grad_func =
            |x: &DVector<f64>| dvector![2.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 6.0 * x[1]];

        let x = dvector![1.0, 1.0];
        let g = grad_func(&x);
        let sx = dvector![1.0, 1.0];
        let rnoise = 1e-10;

        let h = fstofd_hessian(&x, &g, &grad_func, &sx, rnoise);

        // Check Hessian (with loose tolerance for numerical differentiation)
        assert!((h[(0, 0)] - 2.0).abs() < 0.5, "H[0,0] = {}", h[(0, 0)]);
        assert!((h[(0, 1)] - 2.0).abs() < 0.5, "H[0,1] = {}", h[(0, 1)]);
        assert!((h[(1, 0)] - 2.0).abs() < 0.5, "H[1,0] = {}", h[(1, 0)]);
        assert!((h[(1, 1)] - 6.0).abs() < 0.5, "H[1,1] = {}", h[(1, 1)]);
    }

    #[test]
    fn test_central_more_accurate_than_forward() {
        // For smooth functions, central differences should be more accurate
        let func = |x: &DVector<f64>| x[0].powi(3);

        let x = dvector![2.0];
        let f = func(&x);
        let sx = dvector![1.0];
        let rnoise = 1e-6;

        // True gradient at x=2 is 3*4 = 12
        let g_forward = fstofd_gradient(&x, f, &func, &sx, rnoise);
        let g_central = fstocd(&x, &func, &sx, rnoise);

        let error_forward = (g_forward[0] - 12.0).abs();
        let error_central = (g_central[0] - 12.0).abs();

        // Central differences should be more accurate
        assert!(
            error_central < error_forward,
            "Central: {}, Forward: {}",
            error_central,
            error_forward
        );
    }
}
