//! More-Hebdon trust region method (hookstep algorithm)
//!
//! Port of hook_1step and hookdrv from nlm.c:903-1140

use crate::linalg::{choldc, lltslv};
use crate::types::ObjectiveFn;
use crate::updates::trust_region::{tregup, TregupParams, TregupReturnCode};
use nalgebra::{DMatrix, DVector};

/// State for More-Hebdon algorithm (retained between calls)
pub struct HookState {
    pub amu: f64,
    pub dltp: f64,
    pub phi: f64,
    pub phip0: f64,
    pub fstime: bool,
}

impl HookState {
    pub fn new() -> Self {
        Self {
            amu: 0.0,
            dltp: 0.0,
            phi: 0.0,
            phip0: 0.0,
            fstime: true,
        }
    }
}

impl Default for HookState {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute one step of More-Hebdon algorithm
///
/// Port of `hook_1step` from nlm.c:903-1040
///
/// Finds new step by More-Hebdon (hookstep) algorithm. This method solves
/// the trust region subproblem by finding λ such that ||step|| = δ using
/// an iterative procedure on the augmented Hessian (H + λI).
///
/// # Arguments
/// * `g` - Gradient at current iterate
/// * `a` - Hessian (upper triangle + udiag) and workspace (lower triangle for Cholesky)
/// * `udiag` - Diagonal of Hessian
/// * `p` - Newton step
/// * `sx` - Diagonal scaling matrix
/// * `rnwtln` - Newton step length
/// * `dlt` - Trust region radius (may be updated)
/// * `state` - Hook state (retained between calls)
/// * `epsm` - Machine epsilon
///
/// # Returns
/// * `sc` - Current step
/// * `nwtake` - True if Newton step taken
#[allow(clippy::too_many_arguments)]
pub fn hook_1step(
    g: &DVector<f64>,
    a: &mut DMatrix<f64>,
    udiag: &DVector<f64>,
    p: &DVector<f64>,
    sx: &DVector<f64>,
    rnwtln: f64,
    dlt: &mut f64,
    state: &mut HookState,
    epsm: f64,
) -> (DVector<f64>, bool) {
    let n = g.len();
    const HI: f64 = 1.5;
    const ALO: f64 = 0.75;

    // Shall we take Newton step?
    let nwtake = rnwtln <= HI * *dlt;

    if nwtake {
        *dlt = dlt.min(rnwtln);
        state.amu = 0.0;
        return (p.clone(), true);
    }

    // Newton step not taken
    if state.amu > 0.0 {
        state.amu -=
            (state.phi + state.dltp) * (state.dltp - *dlt + state.phi) / (*dlt * state.phip0);
    }

    state.phi = rnwtln - *dlt;

    if state.fstime {
        // First time: compute phip0
        let mut wrk0 = DVector::zeros(n);
        for i in 0..n {
            wrk0[i] = sx[i] * sx[i] * p[i];
        }

        // Solve L*y = (sx**2)*p using lower triangular solver
        // Note: dtrsl with job=0 solves L*x = b
        wrk0 = solve_lower_triangular(a, &wrk0);

        let temp1 = wrk0.norm();
        state.phip0 = -(temp1 * temp1) / rnwtln;
        state.fstime = false;
    }

    let phip = state.phip0;
    let mut amulo = -state.phi / phip;
    let mut amuup = 0.0;
    for i in 0..n {
        amuup += g[i] * g[i] / (sx[i] * sx[i]);
    }
    amuup = amuup.sqrt() / *dlt;

    // Iterate to find acceptable step (with safety limit)
    let mut iter_count = 0;
    const MAX_ITERS: usize = 100;

    let sc = loop {
        iter_count += 1;
        if iter_count > MAX_ITERS {
            // Safety: return current best step if we've iterated too long
            let mut sc = DVector::zeros(n);
            for i in 0..n {
                sc[i] = p[i] * (*dlt / rnwtln);
            }
            break sc;
        }

        // Test value of amu; generate next if necessary
        if state.amu < amulo || state.amu > amuup {
            state.amu = (amulo * amuup).sqrt().max(amuup * 0.001);
        }

        // Copy (H, udiag) to L where H <-- H + amu*(sx**2)
        for i in 0..n {
            a[(i, i)] = udiag[i] + state.amu * sx[i] * sx[i];
            for j in 0..i {
                a[(i, j)] = a[(j, i)];
            }
        }

        // Factor H = L(L^T) using Cholesky
        let temp1 = epsm.sqrt();
        let _addmax = choldc(a, 0.0, temp1);

        // Solve H*p = L(L^T)*sc = -g
        let mut wrk0 = -g;
        let sc = lltslv(a, &wrk0);

        // Compute step length
        let mut stepln = 0.0;
        for i in 0..n {
            stepln += sx[i] * sx[i] * sc[i] * sc[i];
        }
        stepln = stepln.sqrt();
        state.phi = stepln - *dlt;

        // Compute derivative phip
        for i in 0..n {
            wrk0[i] = sx[i] * sx[i] * sc[i];
        }
        wrk0 = solve_lower_triangular(a, &wrk0);
        let temp1 = wrk0.norm();
        let phip = -(temp1 * temp1) / stepln;

        // Check if sc is acceptable hookstep
        if (ALO * *dlt <= stepln && stepln <= HI * *dlt) || (amuup - amulo <= 0.0) {
            break sc;
        }

        // sc not acceptable: select new amu
        let temp1 = (state.amu - state.phi) / phip;
        amulo = amulo.max(temp1);
        if state.phi < 0.0 {
            amuup = amuup.min(state.amu);
        }
        state.amu -= stepln * state.phi / (*dlt * phip);
    };

    (sc, false)
}

/// Solve lower triangular system L*x = b
fn solve_lower_triangular(l: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    let n = b.len();
    let mut x = DVector::zeros(n);

    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[(i, j)] * x[j];
        }
        x[i] = sum / l[(i, i)];
    }

    x
}

/// More-Hebdon trust region method
///
/// Port of `hookdrv` from nlm.c:1042-1140
///
/// Finds next Newton iterate using the More-Hebdon method. This is the most
/// sophisticated of the three methods, solving the trust region subproblem
/// more accurately than dogleg.
///
/// # Arguments
/// * `x` - Current iterate
/// * `f` - Function value at x
/// * `g` - Gradient at x
/// * `a` - Hessian (upper triangle + udiag) and Cholesky workspace (lower triangle)
/// * `udiag` - Diagonal of Hessian
/// * `p` - Newton step
/// * `func` - Objective function
/// * `sx` - Diagonal scaling matrix
/// * `stepmx` - Maximum step size
/// * `steptl` - Step tolerance
/// * `dlt` - Trust region radius (updated)
/// * `state` - Hook state (retained between calls)
/// * `epsm` - Machine epsilon
/// * `itncnt` - Iteration count
///
/// # Returns
/// * `xpls` - New iterate
/// * `fpls` - Function value at xpls
/// * `iretcd` - Return code (0=success, 1=failure)
/// * `mxtake` - True if maximum step taken
#[allow(clippy::too_many_arguments)]
pub fn hookdrv(
    x: &DVector<f64>,
    f: f64,
    g: &DVector<f64>,
    a: &mut DMatrix<f64>,
    udiag: &DVector<f64>,
    p: &DVector<f64>,
    func: &ObjectiveFn,
    sx: &DVector<f64>,
    stepmx: f64,
    steptl: f64,
    dlt: &mut f64,
    state: &mut HookState,
    epsm: f64,
    itncnt: usize,
) -> (DVector<f64>, f64, i32, bool) {
    let n = x.len();

    // Compute Newton step length
    let mut tmp = 0.0;
    for i in 0..n {
        tmp += sx[i] * sx[i] * p[i] * p[i];
    }
    let rnwtln = tmp.sqrt();

    // Initialize on first iteration
    if itncnt == 1 {
        state.amu = 0.0;

        // Compute initial trust region if not provided
        if *dlt == -1.0 {
            let mut alpha = 0.0;
            for i in 0..n {
                alpha += g[i] * g[i] / (sx[i] * sx[i]);
            }

            let mut bet = 0.0;
            for i in 0..n {
                let mut tmp = 0.0;
                for j in i..n {
                    tmp += a[(j, i)] * g[j] / (sx[j] * sx[j]);
                }
                bet += tmp * tmp;
            }

            *dlt = alpha * alpha.sqrt() / bet;
            if *dlt > stepmx {
                *dlt = stepmx;
            }
        }
    }

    let mut iretcd = TregupReturnCode::IncreaseRadius;
    state.fstime = true;
    let mut xplsp = DVector::zeros(n);
    let mut fplsp = 0.0;
    let mut xpls = DVector::zeros(n);
    let mut fpls = 0.0;
    let mut mxtake = false;

    loop {
        // Find new step by More-Hebdon algorithm
        let (sc, nwtake) = hook_1step(g, a, udiag, p, sx, rnwtln, dlt, state, epsm);
        state.dltp = *dlt;

        // Check new point and update trust region
        let tregup_params = TregupParams {
            x,
            f,
            g,
            a,
            func,
            sc: &sc,
            sx,
            nwtake,
            stepmx,
            steptl,
            method: crate::types::Method::MoreHebdon,
            udiag,
        };

        let result = tregup(&tregup_params, *dlt, iretcd, xplsp, fplsp);

        iretcd = result.iretcd;
        xpls = result.xpls;
        fpls = result.fpls;
        mxtake = result.mxtake;
        *dlt = result.dlt;
        xplsp = result.xplsp;
        fplsp = result.fplsp;

        if iretcd == TregupReturnCode::Accepted || iretcd == TregupReturnCode::AcceptedSmallStep {
            break;
        }
    }

    let ret_code = match iretcd {
        TregupReturnCode::Accepted => 0,
        TregupReturnCode::AcceptedSmallStep => 1,
        _ => 2,
    };

    (xpls, fpls, ret_code, mxtake)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_solve_lower_triangular() {
        // L = [[2, 0], [3, 4]], b = [6, 17]
        // Solution: x = [3, 2]
        let l = dmatrix![
            2.0, 0.0;
            3.0, 4.0
        ];
        let b = dvector![6.0, 17.0];

        let x = solve_lower_triangular(&l, &b);

        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hook_1step_takes_newton() {
        // If Newton step is within 1.5*trust region, take it
        let g = dvector![1.0, 1.0];
        let mut a = dmatrix![
            2.0, 0.5;
            0.5, 2.0
        ];
        let udiag = dvector![2.0, 2.0];
        let p = dvector![-0.5, -0.5];
        let sx = dvector![1.0, 1.0];
        let rnwtln = 0.707; // sqrt(0.5)
        let mut dlt = 1.0; // Large enough: 1.5*1.0 > 0.707
        let mut state = HookState::new();
        let epsm = f64::EPSILON;

        let (sc, nwtake) = hook_1step(
            &g, &mut a, &udiag, &p, &sx, rnwtln, &mut dlt, &mut state, epsm,
        );

        assert!(nwtake);
        assert!((sc[0] - p[0]).abs() < 1e-10);
        assert!((sc[1] - p[1]).abs() < 1e-10);
    }

    #[test]
    fn test_hookdrv_basic() {
        // Simple quadratic: f(x) = x^2 + y^2
        let func = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];

        let x = dvector![2.0, 2.0];
        let f = func(&x);
        let g = dvector![4.0, 4.0];
        let mut a = dmatrix![
            2.0, 0.0;
            0.0, 2.0
        ];
        let udiag = dvector![2.0, 2.0];
        let p = dvector![-2.0, -2.0]; // Newton direction
        let sx = dvector![1.0, 1.0];
        let stepmx = 10.0;
        let steptl = 1e-8;
        let mut dlt = 1.0;
        let mut state = HookState::new();
        let epsm = f64::EPSILON;

        let (xpls, fpls, iretcd, _mxtake) = hookdrv(
            &x, f, &g, &mut a, &udiag, &p, &func, &sx, stepmx, steptl, &mut dlt, &mut state, epsm,
            1,
        );

        // Should find a better point
        assert!(fpls < f);
        assert!(iretcd == 0 || iretcd == 1);
        assert_eq!(xpls.len(), 2);
    }

    #[test]
    fn test_hook_state_default() {
        let state = HookState::default();
        assert_eq!(state.amu, 0.0);
        assert_eq!(state.dltp, 0.0);
        assert!(state.fstime);
    }
}
