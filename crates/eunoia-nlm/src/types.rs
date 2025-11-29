//! Type definitions for the NLM optimization algorithm
//!
//! Port of types and structures from nlm.c

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Result type for NLM operations
pub type Result<T> = std::result::Result<T, NlmError>;

/// Error types for NLM algorithm
#[derive(Error, Debug)]
pub enum NlmError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Function evaluation failed")]
    FunctionEvaluation,

    #[error("Gradient evaluation failed")]
    GradientEvaluation,

    #[error("Hessian evaluation failed")]
    HessianEvaluation,

    #[error("Matrix operation failed: {0}")]
    MatrixOperation(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailure(String),
}

/// Objective function type: R^n -> R
///
/// Takes a vector of parameters and returns the function value
pub type ObjectiveFn = dyn Fn(&DVector<f64>) -> f64;

/// Gradient function type: R^n -> R^n
///
/// Takes a vector of parameters and returns the gradient vector
pub type GradientFn = dyn Fn(&DVector<f64>) -> DVector<f64>;

/// Hessian function type: R^n -> R^(n×n)
///
/// Takes a vector of parameters and returns the Hessian matrix
pub type HessianFn = dyn Fn(&DVector<f64>) -> DMatrix<f64>;

/// Optimization method
///
/// Corresponds to `method` parameter in optif9
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Line search method (method = 1)
    LineSearch = 1,

    /// Double dogleg method (method = 2)
    DoubleDogleg = 2,

    /// More-Hebdon method (method = 3)
    MoreHebdon = 3,
}

/// Termination code returned by opt_stop
///
/// See opt_stop in nlm.c:1874-1962
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationCode {
    /// Relative gradient close to zero (code 1)
    GradientConvergence = 1,

    /// Successive iterates within tolerance (code 2)
    XConvergence = 2,

    /// Both gradient and X-convergence (code 3)
    BothConvergence = 3,

    /// Iteration limit reached (code 4)
    IterationLimit = 4,

    /// Maximum step size exceeded 5 consecutive times (code 5)
    MaxStepExceeded = 5,

    /// Could not find satisfactory step (code 6)
    NoSatisfactoryStep = 6,
}

impl TerminationCode {
    /// Check if the optimization converged successfully
    ///
    /// Returns true for gradient convergence, X convergence, or both.
    /// Returns false for iteration limit, max step exceeded, or no satisfactory step.
    pub fn is_converged(&self) -> bool {
        matches!(
            self,
            TerminationCode::GradientConvergence
                | TerminationCode::XConvergence
                | TerminationCode::BothConvergence
        )
    }
}

/// Configuration for NLM optimization
///
/// Corresponds to parameters in optif9 (nlm.c:2550-2614)
#[derive(Debug, Clone)]
pub struct NlmConfig {
    /// Typical size for each component of x
    /// Default: vector of ones
    pub typsize: Option<DVector<f64>>,

    /// Estimate of scale of objective function
    /// Default: 1.0
    pub fscale: f64,

    /// Algorithm method to use
    /// Default: LineSearch
    pub method: Method,

    /// Whether function is expensive to evaluate
    /// If true, use secant updates instead of recomputing Hessian
    /// Default: true (iexp = 1)
    pub expensive: bool,

    /// Number of good digits in optimization function
    /// If None, computed as floor(-log10(DBL_EPSILON))
    /// Default: None (computed automatically)
    pub ndigit: Option<i32>,

    /// Maximum number of iterations
    /// Default: 150
    pub max_iter: usize,

    /// Whether analytic gradient is provided
    /// Default: false (will use numerical gradient)
    pub has_gradient: bool,

    /// Whether analytic Hessian is provided
    /// Default: false (will use numerical Hessian or secant updates)
    pub has_hessian: bool,

    /// Trust region radius (for methods 2 and 3)
    /// If None, computed in optchk
    /// Default: None (computed automatically)
    pub delta: Option<f64>,

    /// Gradient tolerance
    /// Default: DBL_EPSILON^(1/3)
    pub grad_tol: f64,

    /// Maximum allowable step size
    /// If 0.0, computed in optchk
    /// Default: 0.0 (computed automatically)
    pub max_step: f64,

    /// Step tolerance (relative)
    /// Default: sqrt(DBL_EPSILON)
    pub step_tol: f64,

    /// Message level (controls warnings and checks)
    /// See optchk and optdrv for details
    /// Default: 0 (all checks enabled)
    pub msg: i32,
}

impl Default for NlmConfig {
    fn default() -> Self {
        let epsm = f64::EPSILON;
        Self {
            typsize: None,
            fscale: 1.0,
            method: Method::LineSearch,
            expensive: true,
            ndigit: None,
            max_iter: 150,
            has_gradient: false,
            has_hessian: false,
            delta: None,
            grad_tol: epsm.powf(1.0 / 3.0),
            max_step: 0.0,
            step_tol: epsm.sqrt(),
            msg: 0,
        }
    }
}

/// Result of optimization
///
/// Returned by optimize() function
#[derive(Debug, Clone)]
pub struct NlmResult {
    /// Solution point
    pub x: DVector<f64>,

    /// Function value at solution
    pub f: f64,

    /// Gradient at solution
    pub gradient: DVector<f64>,

    /// Hessian (or approximation) at solution
    pub hessian: DMatrix<f64>,

    /// Termination code
    pub code: TerminationCode,

    /// Number of iterations performed
    pub iterations: usize,

    /// Number of function evaluations
    pub function_evals: usize,

    /// Number of gradient evaluations
    pub gradient_evals: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NlmConfig::default();
        assert_eq!(config.method, Method::LineSearch);
        assert_eq!(config.max_iter, 150);
        assert!(config.expensive);
        assert!(!config.has_gradient);
        assert!(!config.has_hessian);
    }

    #[test]
    fn test_config_with_overrides() {
        let config = NlmConfig {
            method: Method::DoubleDogleg,
            max_iter: 500,
            has_gradient: true,
            ..Default::default()
        };
        assert_eq!(config.method, Method::DoubleDogleg);
        assert_eq!(config.max_iter, 500);
        assert!(config.has_gradient);
    }

    #[test]
    fn test_termination_codes() {
        assert_eq!(TerminationCode::GradientConvergence as i32, 1);
        assert_eq!(TerminationCode::XConvergence as i32, 2);
        assert_eq!(TerminationCode::IterationLimit as i32, 4);
    }
}
