//! Option validation (optchk) port
//!
//! Port of optchk from nlm.c:1964-2067 (simplified structure retaining logic)
//! Ensures configuration parameters are within acceptable ranges and sets
//! derived defaults matching R's nlm behavior.

use crate::driver::OptimizationConfig;
use crate::types::Method;
use nalgebra::DVector;

/// Result of option checking (normalized parameters)
pub struct OptchkResult {
    pub method: Method,
    pub expensive: bool,
    pub ndigit: i32,
    pub stepmx: f64,
    pub gradtl: f64,
    pub steptl: f64,
    pub dlt: f64,
    pub fscale: f64,
    pub sx: DVector<f64>,
}

/// Validate and normalize optimization options.
///
/// This function emulates the behavior of `optchk` in the original C implementation:
/// - Enforces method in {1,2,3}
/// - Computes typical scaling (sx) from typsiz
/// - Sets defaults for stepmx, ndigit, gradtl, steptl, dlt, fscale
/// - Adjusts `expensive` flag based on presence of analytic Hessian
pub fn optchk(x0: &DVector<f64>, config: &OptimizationConfig) -> OptchkResult {
    let n = x0.len();
    let epsm = f64::EPSILON;

    // Method validation
    let mut method = config.method;
    if (method as i32) < 1 || (method as i32) > 3 {
        method = Method::LineSearch;
    }

    // Typical sizes (typsiz) -> scaling sx
    let mut sx = DVector::zeros(n);
    for i in 0..n {
        let typsize = if i < config.typsiz.len() {
            config.typsiz[i]
        } else {
            1.0
        };
        // Match R's nlm: sx[i] = max(|typsiz[i]|, 1.0) if typsiz provided; else sx[i] = max(|x0[i]|, 1.0)
        sx[i] = if i < config.typsiz.len() {
            typsize.abs().max(1.0)
        } else {
            x0[i].abs().max(1.0)
        };
    }

    // fscale default (must be positive)
    let fscale = if config.fscale == 0.0 {
        1.0
    } else {
        config.fscale.abs()
    };

    // ndigit default (floor(-log10(eps))) if -1
    let ndigit = if config.ndigit == -1 {
        (-epsm.log10()).floor() as i32
    } else {
        config.ndigit
    };

    // grad tolerance default
    let gradtl = if config.gradtl < 0.0 {
        epsm.powf(1.0 / 3.0)
    } else {
        config.gradtl
    };

    // step tolerance default
    let steptl = if config.steptl <= 0.0 {
        epsm.sqrt()
    } else {
        config.steptl
    };

    // stepmx default (1000 * ||x||_scaled or >=1)
    let mut stpsiz = 0.0;
    for i in 0..n {
        stpsiz += x0[i] * x0[i] * sx[i] * sx[i];
    }
    let stepmx_default = 1000.0 * stpsiz.sqrt().max(1.0);
    let stepmx = if config.stepmx <= 0.0 {
        stepmx_default
    } else {
        config.stepmx
    };

    // trust region radius dlt: default 1.0 (R's nlm optimize.c line 813 sets dlt=1.0)
    let mut dlt = if config.dlt <= 0.0 {
        1.0
    } else {
        config.dlt.min(stepmx)
    };

    // Expensive flag: false if analytic Hessian provided
    let expensive = if config.has_hessian {
        false
    } else {
        config.expensive
    };

    OptchkResult {
        method,
        expensive,
        ndigit,
        stepmx,
        gradtl,
        steptl,
        dlt,
        fscale,
        sx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::OptimizationConfig;
    use crate::types::Method;
    use nalgebra::dvector;

    #[test]
    fn test_optchk_defaults() {
        let x0 = dvector![2.0, -1.0];
        let cfg = OptimizationConfig::default();
        let res = optchk(&x0, &cfg);
        assert_eq!(res.method, Method::LineSearch);
        assert!(res.stepmx > 0.0);
        assert_eq!(res.dlt, 1.0);
        assert!(res.gradtl > 0.0);
        assert_eq!(res.sx[0], 2.0);
        assert_eq!(res.sx[1], 1.0);
    }

    #[test]
    fn test_optchk_custom_method() {
        let x0 = dvector![1.0];
        let mut cfg = OptimizationConfig::default();
        cfg.method = Method::DoubleDogleg;
        let res = optchk(&x0, &cfg);
        assert_eq!(res.method, Method::DoubleDogleg);
    }
}
