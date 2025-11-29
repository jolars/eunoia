//! Double dogleg trust region method
//!
//! Port of dog_1step and dogdrv from nlm.c:738-901

use crate::types::ObjectiveFn;
use crate::updates::trust_region::{tregup, TregupParams, TregupReturnCode};
use nalgebra::{DMatrix, DVector};

/// State for dogleg algorithm (retained between calls)
pub struct DoglegState {
    pub ssd: DVector<f64>,
    pub v: DVector<f64>,
    pub cln: f64,
    pub eta: f64,
    pub fstdog: bool,
}

impl DoglegState {
    pub fn new(n: usize) -> Self {
        Self {
            ssd: DVector::zeros(n),
            v: DVector::zeros(n),
            cln: 0.0,
            eta: 0.0,
            fstdog: true,
        }
    }
}

/// Compute one step of double dogleg algorithm
///
/// Port of `dog_1step` from nlm.c:738-834
///
/// Finds new step by double dogleg algorithm. The dogleg path consists of:
/// 1. Steepest descent direction (Cauchy point)
/// 2. Convex combination toward Newton direction
/// 3. Full Newton step if within trust region
///
/// # Arguments
/// * `g` - Gradient at current iterate
/// * `a` - Cholesky decomposition of Hessian (lower triangle)
/// * `p` - Newton step
/// * `sx` - Diagonal scaling matrix
/// * `rnwtln` - Newton step length
/// * `dlt` - Trust region radius (may be updated)
/// * `state` - Dogleg state (retained between calls)
/// * `stepmx` - Maximum allowable step size
///
/// # Returns
/// * `sc` - Current step
/// * `nwtake` - True if Newton step taken
#[allow(clippy::too_many_arguments)]
pub fn dog_1step(
    g: &DVector<f64>,
    a: &DMatrix<f64>,
    p: &DVector<f64>,
    sx: &DVector<f64>,
    rnwtln: f64,
    dlt: &mut f64,
    state: &mut DoglegState,
    stepmx: f64,
) -> (DVector<f64>, bool) {
    let n = g.len();

    // Can we take Newton step?
    let nwtake = rnwtln <= *dlt;

    if nwtake {
        *dlt = rnwtln;
        return (p.clone(), true);
    }

    // Newton step too long - use dogleg curve
    if state.fstdog {
        // Calculate double dogleg curve (Cauchy step)
        state.fstdog = false;

        // alpha = ||g||^2 / ||sx||^2
        let mut alpha = 0.0;
        for i in 0..n {
            alpha += g[i] * g[i] / (sx[i] * sx[i]);
        }

        // bet = g^T * A * g (quadratic form)
        let mut bet = 0.0;
        for i in 0..n {
            let mut tmp = 0.0;
            for j in i..n {
                tmp += a[(j, i)] * g[j] / (sx[j] * sx[j]);
            }
            bet += tmp * tmp;
        }

        // Cauchy step: steepest descent scaled by alpha/beta
        for i in 0..n {
            state.ssd[i] = -(alpha / bet) * g[i] / sx[i];
        }

        // Cauchy length
        state.cln = alpha * alpha.sqrt() / bet;

        // eta: interpolation parameter between Cauchy and Newton
        state.eta = (0.8 * alpha * alpha / (-bet * g.dot(p))) + 0.2;

        // v = eta*sx*p - ssd (direction from Cauchy to scaled Newton)
        for i in 0..n {
            state.v[i] = state.eta * sx[i] * p[i] - state.ssd[i];
        }

        // Initialize trust region if needed
        if *dlt == -1.0 {
            *dlt = state.cln.min(stepmx);
        }
    }

    let sc = if state.eta * rnwtln <= *dlt {
        // Take partial step in Newton direction
        p * (*dlt / rnwtln)
    } else if state.cln >= *dlt {
        // Take step in steepest descent direction
        state.ssd.component_div(sx) * (*dlt / state.cln)
    } else {
        // Take convex combination of ssd and eta*p with scaled length dlt
        let dot1 = state.v.dot(&state.ssd);
        let dot2 = state.v.dot(&state.v);
        let alam =
            (-dot1 + (dot1 * dot1 - dot2 * (state.cln * state.cln - *dlt * *dlt)).sqrt()) / dot2;

        (&state.ssd + &state.v * alam).component_div(sx)
    };

    (sc, false)
}

/// Double dogleg trust region method
///
/// Port of `dogdrv` from nlm.c:837-901
///
/// Finds next Newton iterate using the double dogleg method. Repeatedly
/// calls dog_1step and tregup until convergence.
///
/// # Arguments
/// * `x` - Current iterate
/// * `f` - Function value at x
/// * `g` - Gradient at x
/// * `a` - Cholesky decomposition of Hessian
/// * `p` - Newton step
/// * `func` - Objective function
/// * `sx` - Diagonal scaling matrix
/// * `stepmx` - Maximum step size
/// * `steptl` - Step tolerance
/// * `dlt` - Trust region radius (updated)
/// * `state` - Dogleg state (retained between calls)
/// * `itncnt` - Iteration count
///
/// # Returns
/// * `xpls` - New iterate
/// * `fpls` - Function value at xpls
/// * `iretcd` - Return code (0=success, 1=failure)
/// * `mxtake` - True if maximum step taken
#[allow(clippy::too_many_arguments)]
pub fn dogdrv(
    x: &DVector<f64>,
    f: f64,
    g: &DVector<f64>,
    a: &DMatrix<f64>,
    p: &DVector<f64>,
    func: &ObjectiveFn,
    sx: &DVector<f64>,
    stepmx: f64,
    steptl: f64,
    dlt: &mut f64,
    state: &mut DoglegState,
    _itncnt: usize,
) -> (DVector<f64>, f64, i32, bool) {
    let n = x.len();

    // Compute Newton step length
    let mut tmp = 0.0;
    for i in 0..n {
        tmp += sx[i] * sx[i] * p[i] * p[i];
    }
    let rnwtln = tmp.sqrt();

    let mut iretcd = TregupReturnCode::IncreaseRadius;
    let mut xplsp = DVector::zeros(n);
    let mut fplsp = 0.0;
    let mut xpls = DVector::zeros(n);
    let mut fpls = 0.0;
    let mut mxtake = false;

    loop {
        // Find new step by double dogleg algorithm
        let (sc, nwtake) = dog_1step(g, a, p, sx, rnwtln, dlt, state, stepmx);

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
            method: crate::types::Method::DoubleDogleg,
            udiag: &DVector::zeros(n), // Not used for method 2
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
    fn test_dog_1step_takes_newton() {
        // If Newton step is within trust region, take it
        let g = dvector![1.0, 1.0];
        let a = dmatrix![
            2.0, 0.0;
            0.0, 2.0
        ];
        let p = dvector![-0.5, -0.5];
        let sx = dvector![1.0, 1.0];
        let rnwtln = 0.707; // sqrt(0.5)
        let mut dlt = 1.0; // Large enough
        let mut state = DoglegState::new(2);
        let stepmx = 10.0;

        let (sc, nwtake) = dog_1step(&g, &a, &p, &sx, rnwtln, &mut dlt, &mut state, stepmx);

        assert!(nwtake);
        assert!((sc[0] - p[0]).abs() < 1e-10);
        assert!((sc[1] - p[1]).abs() < 1e-10);
    }

    #[test]
    fn test_dog_1step_dogleg_path() {
        // Newton step too long - should use dogleg
        let g = dvector![2.0, 2.0];
        let a = dmatrix![
            1.0, 0.0;
            0.0, 1.0
        ];
        let p = dvector![-2.0, -2.0];
        let sx = dvector![1.0, 1.0];
        let rnwtln = 2.828; // sqrt(8)
        let mut dlt = 0.5; // Small trust region
        let mut state = DoglegState::new(2);
        let stepmx = 10.0;

        let (sc, nwtake) = dog_1step(&g, &a, &p, &sx, rnwtln, &mut dlt, &mut state, stepmx);

        assert!(!nwtake);
        // Step should have length approximately dlt
        let step_len = (sc[0] * sc[0] + sc[1] * sc[1]).sqrt();
        assert!((step_len - dlt).abs() < 0.1);
    }

    #[test]
    fn test_dogdrv_basic() {
        // Simple quadratic: f(x) = x^2 + y^2
        let func = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];

        let x = dvector![2.0, 2.0];
        let f = func(&x);
        let g = dvector![4.0, 4.0];
        let a = dmatrix![
            2.0, 0.0;
            0.0, 2.0
        ];
        let p = dvector![-2.0, -2.0]; // Newton direction
        let sx = dvector![1.0, 1.0];
        let stepmx = 10.0;
        let steptl = 1e-8;
        let mut dlt = 1.0;
        let mut state = DoglegState::new(2);

        let (xpls, fpls, iretcd, _mxtake) = dogdrv(
            &x, f, &g, &a, &p, &func, &sx, stepmx, steptl, &mut dlt, &mut state, 1,
        );

        // Should find a better point
        assert!(fpls < f);
        assert!(iretcd == 0 || iretcd == 1);
        assert_eq!(xpls.len(), 2);
    }
}
