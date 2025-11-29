//! Stopping criteria for optimization
//!
//! Port of opt_stop from nlm.c:1874-1961

use nalgebra::DVector;

/// Termination codes for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationCode {
    /// Continue optimization (not terminated)
    Continue = 0,
    /// Converged: relative gradient small enough
    GradientConverged = 1,
    /// Converged: relative step size small enough
    StepConverged = 2,
    /// Failed: last global step could not find lower point
    GlobalStepFailed = 3,
    /// Iteration limit reached
    IterationLimitReached = 4,
    /// Too many consecutive maximum steps (divergence suspected)
    TooManyMaxSteps = 5,
}

impl TerminationCode {
    /// Check if the optimization converged successfully
    ///
    /// Returns true for gradient convergence or step convergence.
    /// Returns false for iteration limit, global step failure, or too many max steps.
    pub fn is_converged(&self) -> bool {
        matches!(
            self,
            TerminationCode::GradientConverged | TerminationCode::StepConverged
        )
    }
}

/// Parameters for stopping criteria check
pub struct StopParams<'a> {
    pub xpls: &'a DVector<f64>,
    pub fpls: f64,
    pub gpls: &'a DVector<f64>,
    pub x: &'a DVector<f64>,
    pub itncnt: usize,
    pub icscmx: usize,
    pub gradtl: f64,
    pub steptl: f64,
    pub sx: &'a DVector<f64>,
    pub fscale: f64,
    pub itnlim: usize,
    pub iretcd: i32,
    pub mxtake: bool,
}

/// Check stopping criteria for optimization
///
/// Port of `opt_stop` from nlm.c:1874-1961
///
/// Determines whether the algorithm should terminate due to:
/// 1. Problem solved within user tolerance (gradient small)
/// 2. Convergence (step size small)
/// 3. Iteration limit reached
/// 4. Divergence or too restrictive maximum step suspected
///
/// # Arguments
/// * `params` - Stopping criteria parameters
/// * `icscmx` - Number of consecutive steps >= stepmx (updated)
///
/// # Returns
/// Termination code indicating why (or if) to stop
pub fn opt_stop(params: &StopParams, icscmx: &mut usize) -> TerminationCode {
    let n = params.xpls.len();

    // Last global step failed to locate a point lower than x
    if params.iretcd == 1 {
        return TerminationCode::GlobalStepFailed;
    }

    // Find direction in which relative gradient is maximum
    let d = params.fpls.abs().max(params.fscale);
    let mut rgx = 0.0;

    for i in 0..n {
        let relgrd = params.gpls[i].abs() * params.xpls[i].abs().max(1.0 / params.sx[i]) / d;
        if rgx < relgrd {
            rgx = relgrd;
        }
    }

    // Check if gradient is small enough (converged)
    if rgx <= params.gradtl {
        return TerminationCode::GradientConverged;
    }

    // Gradient not small enough
    if params.itncnt == 0 {
        // First iteration - continue
        return TerminationCode::Continue;
    }

    // Find direction in which relative step size is maximum
    let mut rsx = 0.0;
    for i in 0..n {
        let relstp =
            (params.xpls[i] - params.x[i]).abs() / params.xpls[i].abs().max(1.0 / params.sx[i]);
        if rsx < relstp {
            rsx = relstp;
        }
    }

    // Check if step is small enough (converged)
    if rsx <= params.steptl {
        return TerminationCode::StepConverged;
    }

    // Step not small enough - check iteration limit
    if params.itncnt >= params.itnlim {
        return TerminationCode::IterationLimitReached;
    }

    // Check number of consecutive steps >= stepmx
    if !params.mxtake {
        *icscmx = 0;
        return TerminationCode::Continue;
    }

    // Maximum step taken - increment counter
    *icscmx += 1;
    if *icscmx < 5 {
        return TerminationCode::Continue;
    }

    // Too many consecutive max steps - divergence suspected
    TerminationCode::TooManyMaxSteps
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_gradient_converged() {
        let xpls = dvector![1.0, 1.0];
        let x = dvector![0.9, 0.9];
        let gpls = dvector![1e-7, 1e-7]; // Very small gradient
        let sx = dvector![1.0, 1.0];

        let params = StopParams {
            xpls: &xpls,
            fpls: 1.0,
            gpls: &gpls,
            x: &x,
            itncnt: 1,
            icscmx: 0,
            gradtl: 1e-6,
            steptl: 1e-8,
            sx: &sx,
            fscale: 1.0,
            itnlim: 100,
            iretcd: 0,
            mxtake: false,
        };

        let mut icscmx = 0;
        let code = opt_stop(&params, &mut icscmx);
        assert_eq!(code, TerminationCode::GradientConverged);
    }

    #[test]
    fn test_step_converged() {
        let xpls = dvector![1.0, 1.0];
        let x = dvector![0.999999, 0.999999]; // Very small step
        let gpls = dvector![0.1, 0.1]; // Large gradient
        let sx = dvector![1.0, 1.0];

        let params = StopParams {
            xpls: &xpls,
            fpls: 1.0,
            gpls: &gpls,
            x: &x,
            itncnt: 1,
            icscmx: 0,
            gradtl: 1e-6,
            steptl: 1e-5,
            sx: &sx,
            fscale: 1.0,
            itnlim: 100,
            iretcd: 0,
            mxtake: false,
        };

        let mut icscmx = 0;
        let code = opt_stop(&params, &mut icscmx);
        assert_eq!(code, TerminationCode::StepConverged);
    }

    #[test]
    fn test_global_step_failed() {
        let xpls = dvector![1.0, 1.0];
        let x = dvector![0.9, 0.9];
        let gpls = dvector![0.1, 0.1];
        let sx = dvector![1.0, 1.0];

        let params = StopParams {
            xpls: &xpls,
            fpls: 1.0,
            gpls: &gpls,
            x: &x,
            itncnt: 1,
            icscmx: 0,
            gradtl: 1e-6,
            steptl: 1e-8,
            sx: &sx,
            fscale: 1.0,
            itnlim: 100,
            iretcd: 1, // Failure code
            mxtake: false,
        };

        let mut icscmx = 0;
        let code = opt_stop(&params, &mut icscmx);
        assert_eq!(code, TerminationCode::GlobalStepFailed);
    }

    #[test]
    fn test_iteration_limit() {
        let xpls = dvector![1.0, 1.0];
        let x = dvector![0.5, 0.5]; // Large step
        let gpls = dvector![0.1, 0.1]; // Large gradient
        let sx = dvector![1.0, 1.0];

        let params = StopParams {
            xpls: &xpls,
            fpls: 1.0,
            gpls: &gpls,
            x: &x,
            itncnt: 100, // At limit
            icscmx: 0,
            gradtl: 1e-6,
            steptl: 1e-8,
            sx: &sx,
            fscale: 1.0,
            itnlim: 100, // Limit reached
            iretcd: 0,
            mxtake: false,
        };

        let mut icscmx = 0;
        let code = opt_stop(&params, &mut icscmx);
        assert_eq!(code, TerminationCode::IterationLimitReached);
    }

    #[test]
    fn test_too_many_max_steps() {
        let xpls = dvector![1.0, 1.0];
        let x = dvector![0.5, 0.5];
        let gpls = dvector![0.1, 0.1];
        let sx = dvector![1.0, 1.0];

        let params = StopParams {
            xpls: &xpls,
            fpls: 1.0,
            gpls: &gpls,
            x: &x,
            itncnt: 5,
            icscmx: 4, // Already 4 consecutive max steps
            gradtl: 1e-6,
            steptl: 1e-8,
            sx: &sx,
            fscale: 1.0,
            itnlim: 100,
            iretcd: 0,
            mxtake: true, // Max step taken
        };

        let mut icscmx = 4;
        let code = opt_stop(&params, &mut icscmx);
        assert_eq!(code, TerminationCode::TooManyMaxSteps);
        assert_eq!(icscmx, 5);
    }

    #[test]
    fn test_continue_first_iteration() {
        let xpls = dvector![1.0, 1.0];
        let x = dvector![0.5, 0.5];
        let gpls = dvector![0.5, 0.5]; // Large gradient
        let sx = dvector![1.0, 1.0];

        let params = StopParams {
            xpls: &xpls,
            fpls: 1.0,
            gpls: &gpls,
            x: &x,
            itncnt: 0, // First iteration
            icscmx: 0,
            gradtl: 1e-6,
            steptl: 1e-8,
            sx: &sx,
            fscale: 1.0,
            itnlim: 100,
            iretcd: 0,
            mxtake: false,
        };

        let mut icscmx = 0;
        let code = opt_stop(&params, &mut icscmx);
        assert_eq!(code, TerminationCode::Continue);
    }
}
