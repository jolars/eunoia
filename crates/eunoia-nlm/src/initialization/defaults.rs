//! Default parameter initialization
//!
//! Port of dfault from nlm.c:2447-2504

use crate::types::{Method, NlmConfig};
use nalgebra::DVector;

/// Set default values for optimization parameters
///
/// Port of `dfault` from nlm.c:2447-2504
///
/// Sets default values for minimization algorithm. This is used when
/// the user doesn't provide explicit values for optional parameters.
///
/// # Arguments
/// * `n` - Dimension of problem
/// * `x` - Initial guess (used to compute max step size)
///
/// # Returns
/// Default configuration with sensible values for all parameters
pub fn dfault(n: usize, _x: &DVector<f64>) -> NlmConfig {
    let epsm = f64::EPSILON; // ~2.22e-16 for IEEE double precision

    NlmConfig {
        // Typical size: all ones
        typsize: Some(DVector::from_element(n, 1.0)),

        // Function scale
        fscale: 1.0,

        // Use line search method by default
        method: Method::LineSearch,

        // Assume function is expensive to evaluate
        expensive: true,

        // Number of digits: computed as floor(-log10(epsilon)) in validation
        ndigit: None,

        // Maximum iterations
        max_iter: 150,

        // No analytic gradient/Hessian by default
        has_gradient: false,
        has_hessian: false,

        // Trust region radius (computed in validation if needed)
        delta: None,

        // Gradient tolerance: epsilon^(1/3) ~6.055e-6
        grad_tol: epsm.powf(1.0 / 3.0),

        // Maximum step: 0 means compute default in validation
        max_step: 0.0,

        // Step tolerance: sqrt(epsilon) ~1.490e-8
        step_tol: epsm.sqrt(),

        // Message level
        msg: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_dfault_basic() {
        let x = dvector![1.0, 2.0, 3.0];
        let config = dfault(3, &x);

        assert_eq!(config.method, Method::LineSearch);
        assert_eq!(config.max_iter, 150);
        assert!(config.expensive);
        assert!(!config.has_gradient);
        assert!(!config.has_hessian);
        assert_eq!(config.fscale, 1.0);
        assert_eq!(config.max_step, 0.0); // Will be computed later
    }

    #[test]
    fn test_dfault_typsize() {
        let x = dvector![1.0, 2.0];
        let config = dfault(2, &x);

        let typsize = config.typsize.unwrap();
        assert_eq!(typsize.len(), 2);
        assert_eq!(typsize[0], 1.0);
        assert_eq!(typsize[1], 1.0);
    }

    #[test]
    fn test_dfault_tolerances() {
        let x = dvector![0.0];
        let config = dfault(1, &x);

        let epsm = f64::EPSILON;

        // grad_tol should be epsilon^(1/3)
        assert!((config.grad_tol - epsm.powf(1.0 / 3.0)).abs() < 1e-15);

        // step_tol should be sqrt(epsilon)
        assert!((config.step_tol - epsm.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn test_dfault_ndigit_none() {
        let x = dvector![1.0];
        let config = dfault(1, &x);

        // ndigit should be None initially (computed in validation)
        assert!(config.ndigit.is_none());
    }
}
