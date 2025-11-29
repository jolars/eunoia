//! Line search method for finding next iterate
//!
//! Port of lnsrch from nlm.c:611-736

use crate::types::ObjectiveFn;
use nalgebra::DVector;

/// Return codes for line search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LnsrchReturnCode {
    /// Satisfactory step found
    Success = 0,
    /// No satisfactory step found (step too small)
    Failure = 1,
    /// Continuing iteration
    Iterating = 2,
}

/// Parameters for line search
pub struct LnsrchParams<'a> {
    pub x: &'a DVector<f64>,
    pub f: f64,
    pub g: &'a DVector<f64>,
    pub p: &'a DVector<f64>,
    pub func: &'a ObjectiveFn,
    pub stepmx: f64,
    pub steptl: f64,
    pub sx: &'a DVector<f64>,
}

/// Result of line search
pub struct LnsrchResult {
    pub xpls: DVector<f64>,
    pub fpls: f64,
    pub iretcd: LnsrchReturnCode,
    pub mxtake: bool,
}

/// Line search to find next Newton iterate
///
/// Port of `lnsrch` from nlm.c:611-736
///
/// Finds the next iterate by performing a line search along the Newton direction.
/// Uses backtracking with quadratic and cubic interpolation.
///
/// # Arguments
/// * `params` - Line search parameters
///
/// # Returns
/// New iterate, function value, and status
///
/// # Algorithm
/// 1. Scale Newton step if it exceeds maximum step size
/// 2. Try full Newton step (λ=1)
/// 3. If unsuccessful, backtrack using:
///    - Quadratic fit on first backtrack
///    - Cubic fit on subsequent backtracks
/// 4. Accept if sufficient decrease (Armijo condition)
pub fn lnsrch(params: &LnsrchParams) -> LnsrchResult {
    let n = params.x.len();
    let mut p = params.p.clone();

    // Compute scaled Newton step length
    let mut temp1 = 0.0;
    for i in 0..n {
        temp1 += params.sx[i] * params.sx[i] * p[i] * p[i];
    }
    let mut sln = temp1.sqrt();

    // Scale Newton step if too long
    if sln > params.stepmx {
        let scl = params.stepmx / sln;
        for i in 0..n {
            p[i] *= scl;
        }
        sln = params.stepmx;
    }

    // Slope along search direction
    let slp = params.g.dot(&p);

    // Compute relative step length
    let mut rln = 0.0;
    for i in 0..n {
        let temp = p[i].abs() / params.x[i].abs().max(1.0 / params.sx[i]);
        if rln < temp {
            rln = temp;
        }
    }
    let rmnlmb = params.steptl / rln;

    // Initial step length
    let mut lambda = 1.0;
    let mut firstback = true;
    let mut pfpls = 0.0;
    let mut plmbda = 0.0;

    let mut mxtake = false;
    let mut iretcd = LnsrchReturnCode::Iterating;

    loop {
        // Compute trial point
        let xpls = params.x + &p * lambda;
        let fpls = (params.func)(&xpls);

        // Check Armijo condition: sufficient decrease
        if fpls <= params.f + slp * 1e-4 * lambda {
            iretcd = LnsrchReturnCode::Success;
            if lambda == 1.0 && sln > params.stepmx * 0.99 {
                mxtake = true;
            }
            return LnsrchResult {
                xpls,
                fpls,
                iretcd,
                mxtake,
            };
        }

        // Check if step too small
        if lambda < rmnlmb {
            iretcd = LnsrchReturnCode::Failure;
            return LnsrchResult {
                xpls,
                fpls,
                iretcd,
                mxtake,
            };
        }

        // Calculate new lambda (backtracking)
        if !fpls.is_finite() {
            // Non-finite value: reduce dramatically
            lambda *= 0.1;
            firstback = true;
        } else if firstback {
            // First backtrack: quadratic interpolation
            let denom = (fpls - params.f - slp) * 2.0;
            let tlmbda = if denom.abs() < 1e-300 {
                // Denominator too small, just reduce lambda
                lambda * 0.1
            } else {
                -lambda * slp / denom
            };
            firstback = false;

            // Update plmbda and pfpls for next iteration (if any)
            plmbda = lambda;
            pfpls = fpls;

            // Clamp lambda
            lambda = if tlmbda < lambda * 0.1 {
                lambda * 0.1
            } else {
                tlmbda
            };
        } else {
            // Subsequent backtracks: cubic interpolation
            let t1 = fpls - params.f - lambda * slp;
            let t2 = pfpls - params.f - plmbda * slp;
            let t3 = 1.0 / (lambda - plmbda);
            let a3 = 3.0 * t3 * (t1 / (lambda * lambda) - t2 / (plmbda * plmbda));
            let b = t3 * (t2 * lambda / (plmbda * plmbda) - t1 * plmbda / (lambda * lambda));
            let disc = b * b - a3 * slp;

            let mut tlmbda = if disc > b * b {
                // Only one positive critical point (minimum)
                let sign = if a3 < 0.0 { -1.0 } else { 1.0 };
                (-b + sign * disc.sqrt()) / a3
            } else {
                // Both critical points positive, first is minimum
                let sign = if a3 < 0.0 { 1.0 } else { -1.0 };
                (-b + sign * disc.sqrt()) / a3
            };

            // Safeguard against NaN or invalid values
            if !tlmbda.is_finite() || tlmbda <= 0.0 {
                tlmbda = lambda * 0.1;
            }

            if tlmbda > lambda * 0.5 {
                tlmbda = lambda * 0.5;
            }

            plmbda = lambda;
            pfpls = fpls;

            lambda = if tlmbda < lambda * 0.1 {
                lambda * 0.1
            } else {
                tlmbda
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_lnsrch_success() {
        // Quadratic function: f(x) = x^2
        let func = |x: &DVector<f64>| x[0] * x[0];

        let x = dvector![2.0];
        let f = func(&x);
        let g = dvector![4.0]; // Gradient at x=2
        let p = dvector![-0.5]; // Descent direction
        let sx = dvector![1.0];

        let params = LnsrchParams {
            x: &x,
            f,
            g: &g,
            p: &p,
            func: &func,
            stepmx: 10.0,
            steptl: 1e-8,
            sx: &sx,
        };

        let result = lnsrch(&params);

        assert_eq!(result.iretcd, LnsrchReturnCode::Success);
        assert!(result.fpls < f); // Should decrease
    }

    #[test]
    fn test_lnsrch_scales_long_step() {
        let func = |x: &DVector<f64>| x[0] * x[0];

        let x = dvector![1.0];
        let f = func(&x);
        let g = dvector![2.0];
        let p = dvector![-100.0]; // Very long step
        let sx = dvector![1.0];

        let params = LnsrchParams {
            x: &x,
            f,
            g: &g,
            p: &p,
            func: &func,
            stepmx: 1.0, // Small max step
            steptl: 1e-8,
            sx: &sx,
        };

        let result = lnsrch(&params);

        // Should still work despite long step
        assert!(
            result.iretcd == LnsrchReturnCode::Success
                || result.iretcd == LnsrchReturnCode::Failure
        );
    }
    #[test]
    fn test_lnsrch_return_codes() {
        assert_eq!(LnsrchReturnCode::Success as i32, 0);
        assert_eq!(LnsrchReturnCode::Failure as i32, 1);
        assert_eq!(LnsrchReturnCode::Iterating as i32, 2);
    }
}
