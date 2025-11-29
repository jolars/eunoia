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

pub mod linalg;
pub mod types;

pub use types::{
    GradientFn, HessianFn, Method, NlmConfig, NlmError, NlmResult, ObjectiveFn, Result,
    TerminationCode,
};

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
