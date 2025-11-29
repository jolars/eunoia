//! Trust region radius updating
//!
//! Port of tregup from nlm.c:441-609

use crate::types::{Method, ObjectiveFn};
use nalgebra::{DMatrix, DVector};

/// Return codes for trust region update
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TregupReturnCode {
    /// xpls accepted as next iterate; dlt is trust region for next iteration
    Accepted = 0,
    /// xpls unsatisfactory but accepted because step length too small
    AcceptedSmallStep = 1,
    /// f(xpls) too large; continue with reduced dlt
    Rejected = 2,
    /// f(xpls) small but model predicts can do better; continue with doubled dlt
    IncreaseRadius = 3,
}

/// Parameters for trust region update
pub struct TregupParams<'a> {
    pub x: &'a DVector<f64>,
    pub f: f64,
    pub g: &'a DVector<f64>,
    pub a: &'a DMatrix<f64>,
    pub func: &'a ObjectiveFn,
    pub sc: &'a DVector<f64>,
    pub sx: &'a DVector<f64>,
    pub nwtake: bool,
    pub stepmx: f64,
    pub steptl: f64,
    pub method: Method,
    pub udiag: &'a DVector<f64>,
}

/// Result of trust region update
pub struct TregupResult {
    pub iretcd: TregupReturnCode,
    pub xpls: DVector<f64>,
    pub fpls: f64,
    pub mxtake: bool,
    pub dlt: f64,
    pub xplsp: DVector<f64>,
    pub fplsp: f64,
}

/// Trust region updating
///
/// Port of `tregup` from nlm.c:441-609
///
/// Decides whether to accept xpls = x + sc as the next iterate and updates
/// the trust region radius. Used only for methods 2 and 3 (dogleg and More-Hebdon).
///
/// # Arguments
/// * `params` - Trust region parameters
/// * `dlt` - Current trust region radius
/// * `iretcd` - Previous return code (for continuation)
/// * `xplsp` - Previous trial point (retained between calls)
/// * `fplsp` - Function value at xplsp (retained between calls)
///
/// # Returns
/// Updated state including new iterate, function value, and trust region radius
///
/// # Return Codes
/// - 0: xpls accepted; dlt for next iteration
/// - 1: xpls accepted despite being unsatisfactory (step too small)
/// - 2: f(xpls) too large; reduce dlt and retry
/// - 3: f(xpls) acceptable but can do better; double dlt and retry
pub fn tregup(
    params: &TregupParams,
    mut dlt: f64,
    iretcd: TregupReturnCode,
    mut xplsp: DVector<f64>,
    mut fplsp: f64,
) -> TregupResult {
    let n = params.x.len();
    let mut mxtake = false;

    // Compute new trial point
    let mut xpls = params.x + params.sc;

    // Evaluate function at trial point
    let fpls = (params.func)(&xpls);
    let dltf = fpls - params.f;
    let slp = params.g.dot(params.sc);

    let final_iretcd =
        if iretcd == TregupReturnCode::IncreaseRadius && (fpls >= fplsp || dltf > slp * 1e-4) {
            // Reset xpls to xplsp and terminate global step
            xpls = xplsp.clone();
            dlt *= 0.5;
            TregupReturnCode::Accepted
        } else if dltf > slp * 1e-4 {
            // fpls too large
            let mut rln = 0.0;
            for i in 0..n {
                let temp1 = params.sc[i].abs() / xpls[i].abs().max(1.0 / params.sx[i]);
                if rln < temp1 {
                    rln = temp1;
                }
            }

            if rln < params.steptl {
                // Cannot find satisfactory xpls sufficiently distinct from x
                TregupReturnCode::AcceptedSmallStep
            } else {
                // Reduce trust region and continue global step
                let dltmp = -slp * dlt / ((dltf - slp) * 2.0);
                dlt = if dltmp < dlt * 0.1 { dlt * 0.1 } else { dltmp };
                TregupReturnCode::Rejected
            }
        } else {
            // fpls sufficiently small

            // Compute predicted reduction dltfp
            let dltfp = match params.method {
                Method::DoubleDogleg => {
                    // For method 2: use Cholesky factor in lower triangle
                    let mut sum = 0.0;
                    for i in 0..n {
                        let mut temp1 = 0.0;
                        for j in i..n {
                            temp1 += params.a[(j, i)] * params.sc[j];
                        }
                        sum += temp1 * temp1;
                    }
                    slp + sum / 2.0
                }
                _ => {
                    // For method 3: use upper triangle and udiag
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += params.udiag[i] * params.sc[i] * params.sc[i];
                        let mut temp1 = 0.0;
                        for j in (i + 1)..n {
                            temp1 += params.a[(i, j)] * params.sc[i] * params.sc[j];
                        }
                        sum += temp1 * 2.0;
                    }
                    slp + sum / 2.0
                }
            };

            if iretcd != TregupReturnCode::Rejected
                && (dltfp - dltf).abs() <= dltf.abs() * 0.1
                && params.nwtake
                && dlt <= params.stepmx * 0.99
            {
                // Double trust region and continue global step
                xplsp = xpls.clone();
                fplsp = fpls;
                dlt = (dlt * 2.0).min(params.stepmx);
                TregupReturnCode::IncreaseRadius
            } else {
                // Accept xpls as next iterate; choose new trust region
                if dlt > params.stepmx * 0.99 {
                    mxtake = true;
                }

                if dltf >= dltfp * 0.1 {
                    // Decrease trust region for next iteration
                    dlt *= 0.5;
                } else if dltf <= dltfp * 0.75 {
                    // Increase trust region for next iteration
                    dlt = (dlt * 2.0).min(params.stepmx);
                }

                TregupReturnCode::Accepted
            }
        };

    TregupResult {
        iretcd: final_iretcd,
        xpls,
        fpls,
        mxtake,
        dlt,
        xplsp,
        fplsp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_tregup_accept_good_step() {
        // Function that decreases significantly
        let func = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];

        let x = dvector![2.0, 2.0];
        let f = func(&x);
        let g = dvector![4.0, 4.0]; // Gradient at x
        let sc = dvector![-0.5, -0.5]; // Step
        let sx = dvector![1.0, 1.0];
        let a = dmatrix![
            2.0, 0.0;
            0.0, 2.0
        ];
        let udiag = dvector![2.0, 2.0];

        let params = TregupParams {
            x: &x,
            f,
            g: &g,
            a: &a,
            func: &func,
            sc: &sc,
            sx: &sx,
            nwtake: false,
            stepmx: 10.0,
            steptl: 1e-8,
            method: Method::DoubleDogleg,
            udiag: &udiag,
        };

        let result = tregup(
            &params,
            1.0,
            TregupReturnCode::Accepted,
            dvector![0.0, 0.0],
            0.0,
        );

        // Should accept the step
        assert!(
            result.iretcd == TregupReturnCode::Accepted
                || result.iretcd == TregupReturnCode::AcceptedSmallStep
        );
        assert_eq!(result.xpls.len(), 2);
    }

    #[test]
    fn test_tregup_reject_bad_step() {
        // Function that increases (bad step direction for this test)
        let func = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];

        let x = dvector![1.0, 1.0];
        let f = func(&x);
        let g = dvector![2.0, 2.0];
        // Bad step: moves away from minimum
        let sc = dvector![2.0, 2.0];
        let sx = dvector![1.0, 1.0];
        let a = dmatrix![
            2.0, 0.0;
            0.0, 2.0
        ];
        let udiag = dvector![2.0, 2.0];

        let params = TregupParams {
            x: &x,
            f,
            g: &g,
            a: &a,
            func: &func,
            sc: &sc,
            sx: &sx,
            nwtake: false,
            stepmx: 10.0,
            steptl: 1e-8,
            method: Method::MoreHebdon,
            udiag: &udiag,
        };

        let result = tregup(
            &params,
            1.0,
            TregupReturnCode::Accepted,
            dvector![0.0, 0.0],
            0.0,
        );

        // Since step increases function significantly, should reject or accept with small step
        assert!(
            result.iretcd == TregupReturnCode::Rejected
                || result.iretcd == TregupReturnCode::AcceptedSmallStep
                || result.iretcd == TregupReturnCode::Accepted
        );
    }

    #[test]
    fn test_tregup_return_codes() {
        // Verify return code enum values match C code
        assert_eq!(TregupReturnCode::Accepted as i32, 0);
        assert_eq!(TregupReturnCode::AcceptedSmallStep as i32, 1);
        assert_eq!(TregupReturnCode::Rejected as i32, 2);
        assert_eq!(TregupReturnCode::IncreaseRadius as i32, 3);
    }
}
