//! Dennis-Schnabel Nonlinear Minimization Algorithm
//!
//! Complete Rust port of R's nlm() optimizer, originally from uncmin.f (FORTRAN)
//! and translated to C. This is a faithful implementation with no simplifications.
//!
//! # Reference
//!
//! Dennis, J.E. and Schnabel, R.B. (1983) "Numerical Methods for Unconstrained
//! Optimization and Nonlinear Equations", Prentice-Hall.
//!
//! # Example
//!
//! ```ignore
//! use eunoia_nlm::{optimize, NlmConfig, Method};
//! use nalgebra::DVector;
//!
//! // Define objective function
//! let rosenbrock = |x: &DVector<f64>| {
//!     (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
//! };
//!
//! // Initial point
//! let x0 = DVector::from_vec(vec![-1.2, 1.0]);
//!
//! // Optimize
//! let config = NlmConfig::default();
//! let result = optimize(&x0, &rosenbrock, None, None, &config)?;
//! ```

pub mod differentiation;
pub mod driver;
pub mod initialization;
pub mod linalg;
pub mod methods;
pub mod stopping;
pub mod types;
pub mod updates;

use nalgebra::DVector;

pub use driver::{optdrv, OptimizationConfig, OptimizationResult};
pub use types::{
    GradientFn, HessianFn, Method, NlmConfig, NlmError, NlmResult, ObjectiveFn, Result,
    TerminationCode,
};

/// Convenience function for optimization with gradient
///
/// This is a simplified wrapper around the full `optdrv` function that assumes
/// you have an analytic gradient.
///
/// # Arguments
/// * `x0` - Initial parameter values
/// * `func` - Objective function to minimize
/// * `grad` - Gradient function
/// * `config` - Optimization configuration (use NlmConfig, which will be converted)
///
/// # Returns
/// Optimization result with final values and convergence status
pub fn optimize(
    x0: DVector<f64>,
    func: impl Fn(&DVector<f64>) -> f64 + 'static,
    grad: impl Fn(&DVector<f64>) -> DVector<f64> + 'static,
    config: NlmConfig,
) -> OptimizationResult {
    let n = x0.len();

    // Convert NlmConfig to OptimizationConfig
    let opt_config = OptimizationConfig {
        typsiz: config
            .typsize
            .unwrap_or_else(|| DVector::from_element(n, 1.0)),
        fscale: config.fscale,
        method: config.method,
        expensive: config.expensive,
        ndigit: config.ndigit.unwrap_or(-1),
        itnlim: config.max_iter,
        has_gradient: true,
        has_hessian: false,
        dlt: config.delta.unwrap_or(-1.0),
        gradtl: config.grad_tol,
        stepmx: config.max_step,
        steptl: config.step_tol,
    };

    let func_boxed: Box<ObjectiveFn> = Box::new(func);
    let grad_boxed: Box<GradientFn> = Box::new(grad);

    optdrv(&x0, &*func_boxed, Some(&*grad_boxed), None, &opt_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify module is properly structured
        let config = NlmConfig::default();
        assert_eq!(config.max_iter, 150);
    }
}
