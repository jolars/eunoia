//! Gradient and Hessian validation functions
//!
//! Port of grdchk and heschk from nlm.c:1751-1793, 1795-1871

use super::gradient::{fstofd_gradient, fstofd_hessian};
use super::hessian::sndofd;
use crate::types::{GradientFn, HessianFn, NlmError, ObjectiveFn, Result};
use nalgebra::{DMatrix, DVector};

/// Parameters for gradient checking
pub struct GrdchkParams<'a> {
    pub x: &'a DVector<f64>,
    pub func: &'a ObjectiveFn,
    pub f: f64,
    pub g: &'a DVector<f64>,
    pub typsiz: &'a DVector<f64>,
    pub sx: &'a DVector<f64>,
    pub fscale: f64,
    pub rnf: f64,
    pub analtl: f64,
}

/// Check analytic gradient against estimated gradient
///
/// Port of `grdchk` from nlm.c:1751-1793
///
/// Compares user-supplied analytic gradient against finite difference estimate.
/// Returns error if they disagree beyond tolerance.
///
/// # Returns
/// Ok(()) if gradient is correct, Err with code -21 if probable coding error
pub fn grdchk(params: GrdchkParams) -> Result<()> {
    let n = params.x.len();

    // Compute first order finite difference gradient
    let wrk1 = fstofd_gradient(params.x, params.f, params.func, params.sx, params.rnf);

    // Compare to analytic gradient
    for i in 0..n {
        let gs = params.f.abs().max(params.fscale) / params.x[i].abs().max(params.typsiz[i]);
        if (params.g[i] - wrk1[i]).abs() > params.g[i].abs().max(gs) * params.analtl {
            return Err(NlmError::OptimizationFailure(
                "Probable coding error in gradient (code -21)".to_string(),
            ));
        }
    }

    Ok(())
}

/// Parameters for Hessian checking
pub struct HeschkParams<'a> {
    pub x: &'a DVector<f64>,
    pub func: &'a ObjectiveFn,
    pub grad_func: Option<&'a GradientFn>,
    pub hess_func: &'a HessianFn,
    pub f: f64,
    pub g: &'a DVector<f64>,
    pub typsiz: &'a DVector<f64>,
    pub sx: &'a DVector<f64>,
    pub rnf: f64,
    pub analtl: f64,
    pub has_grad: bool,
}

/// Check analytic Hessian against estimated Hessian
///
/// Port of `heschk` from nlm.c:1795-1871
///
/// Compares user-supplied analytic Hessian against finite difference estimate.
/// Can use either gradient-based or function-based finite differences.
///
/// # Returns
/// Ok with Hessian if correct, Err with code -22 if probable coding error
pub fn heschk(params: HeschkParams) -> Result<DMatrix<f64>> {
    let n = params.x.len();

    // Compute finite difference approximation to Hessian
    let mut a_fd = if params.has_grad {
        if let Some(grad_fn) = params.grad_func {
            // Use gradient-based finite differences
            fstofd_hessian(params.x, params.g, grad_fn, params.sx, params.rnf)
        } else {
            // Use function-based finite differences
            sndofd(params.x, params.f, params.func, params.sx, params.rnf)
        }
    } else {
        // Use function-based finite differences
        sndofd(params.x, params.f, params.func, params.sx, params.rnf)
    };

    // Copy lower triangular part to upper and save diagonal
    let mut udiag = DVector::zeros(n);
    for j in 0..n {
        udiag[j] = a_fd[(j, j)];
        for i in (j + 1)..n {
            a_fd[(j, i)] = a_fd[(i, j)];
        }
    }

    // Compute analytic Hessian
    let a_analytic = (params.hess_func)(params.x);

    // Compare analytic to finite difference approximation
    for j in 0..n {
        let hs = params.g[j].abs().max(1.0) / params.x[j].abs().max(params.typsiz[j]);

        // Check diagonal
        if (a_analytic[(j, j)] - udiag[j]).abs() > udiag[j].abs().max(hs) * params.analtl {
            return Err(NlmError::OptimizationFailure(format!(
                "Probable coding error in Hessian diagonal at index {} (code -22)",
                j
            )));
        }

        // Check off-diagonal elements
        for i in (j + 1)..n {
            let temp1 = a_analytic[(i, j)];
            let temp2 = (temp1 - a_fd[(j, i)]).abs();
            let temp1_abs = temp1.abs();

            if temp2 > temp1_abs.max(hs) * params.analtl {
                return Err(NlmError::OptimizationFailure(format!(
                    "Probable coding error in Hessian at ({}, {}) (code -22)",
                    i, j
                )));
            }
        }
    }

    Ok(a_analytic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_grdchk_correct_gradient() {
        // f(x,y) = x^2 + 2y^2
        // gradient = [2x, 4y]
        let func = |x: &DVector<f64>| x[0] * x[0] + 2.0 * x[1] * x[1];

        let x = dvector![1.0, 2.0];
        let f = func(&x);
        let g = dvector![2.0, 8.0]; // Correct gradient
        let typsiz = dvector![1.0, 1.0];
        let sx = dvector![1.0, 1.0];
        let fscale = 1.0;
        let rnf = 1e-10;
        let analtl = 1e-3;

        // Should pass with correct gradient
        let params = GrdchkParams {
            x: &x,
            func: &func,
            f,
            g: &g,
            typsiz: &typsiz,
            sx: &sx,
            fscale,
            rnf,
            analtl,
        };
        assert!(grdchk(params).is_ok());
    }

    #[test]
    fn test_grdchk_incorrect_gradient() {
        // f(x,y) = x^2 + 2y^2
        // gradient = [2x, 4y]
        let func = |x: &DVector<f64>| x[0] * x[0] + 2.0 * x[1] * x[1];

        let x = dvector![1.0, 2.0];
        let f = func(&x);
        let g = dvector![2.0, 10.0]; // Incorrect gradient (should be 8, not 10)
        let typsiz = dvector![1.0, 1.0];
        let sx = dvector![1.0, 1.0];
        let fscale = 1.0;
        let rnf = 1e-10;
        let analtl = 1e-3;

        // Should fail with incorrect gradient
        let params = GrdchkParams {
            x: &x,
            func: &func,
            f,
            g: &g,
            typsiz: &typsiz,
            sx: &sx,
            fscale,
            rnf,
            analtl,
        };
        assert!(grdchk(params).is_err());
    }

    #[test]
    fn test_heschk_correct_hessian() {
        // f(x,y) = x^2 + 3y^2
        // gradient = [2x, 6y]
        // Hessian = [[2, 0], [0, 6]]
        let func = |x: &DVector<f64>| x[0] * x[0] + 3.0 * x[1] * x[1];

        let grad_func = |x: &DVector<f64>| dvector![2.0 * x[0], 6.0 * x[1]];

        let hess_func = |_x: &DVector<f64>| {
            dmatrix![
                2.0, 0.0;
                0.0, 6.0
            ]
        };

        let x = dvector![1.0, 1.0];
        let f = func(&x);
        let g = grad_func(&x);
        let typsiz = dvector![1.0, 1.0];
        let sx = dvector![1.0, 1.0];
        let rnf = 1e-8;
        let analtl = 0.5; // Loose tolerance for numerical errors

        // Should pass with correct Hessian
        let params = HeschkParams {
            x: &x,
            func: &func,
            grad_func: Some(&grad_func),
            hess_func: &hess_func,
            f,
            g: &g,
            typsiz: &typsiz,
            sx: &sx,
            rnf,
            analtl,
            has_grad: true,
        };
        let result = heschk(params);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_heschk_incorrect_hessian() {
        // f(x) = x^2, Hessian = [[2]]
        let func = |x: &DVector<f64>| x[0] * x[0];
        let grad_func = |x: &DVector<f64>| dvector![2.0 * x[0]];

        // Incorrect Hessian (should be 2, not 3)
        let hess_func = |_x: &DVector<f64>| dmatrix![3.0];

        let x = dvector![1.0];
        let f = func(&x);
        let g = grad_func(&x);
        let typsiz = dvector![1.0];
        let sx = dvector![1.0];
        let rnf = 1e-10;
        let analtl = 0.1;

        // Should fail with incorrect Hessian
        let params = HeschkParams {
            x: &x,
            func: &func,
            grad_func: Some(&grad_func),
            hess_func: &hess_func,
            f,
            g: &g,
            typsiz: &typsiz,
            sx: &sx,
            rnf,
            analtl,
            has_grad: true,
        };
        let result = heschk(params);
        assert!(result.is_err());
    }

    #[test]
    fn test_heschk_without_gradient() {
        // Test using function-based finite differences (no analytic gradient)
        let func = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];

        let hess_func = |_x: &DVector<f64>| {
            dmatrix![
                2.0, 0.0;
                0.0, 2.0
            ]
        };

        let x = dvector![1.0, 1.0];
        let f = func(&x);
        let g = dvector![2.0, 2.0]; // Just for the test
        let typsiz = dvector![1.0, 1.0];
        let sx = dvector![1.0, 1.0];
        let rnf = 1e-8;
        let analtl = 0.5;

        // Should work without gradient function
        let params = HeschkParams {
            x: &x,
            func: &func,
            grad_func: None,
            hess_func: &hess_func,
            f,
            g: &g,
            typsiz: &typsiz,
            sx: &sx,
            rnf,
            analtl,
            has_grad: false,
        };
        let result = heschk(params);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }
}
