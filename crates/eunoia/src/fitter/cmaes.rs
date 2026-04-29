//! Bounded CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
//!
//! Minimal purecma-style implementation used as a global-escape stage in the
//! final-layout pipeline. The caller supplies an initial mean, per-dimension
//! initial standard deviations (used to precondition the search by scaling
//! every coordinate to the unit normal), and a feasible box `[lower, upper]`.
//! Samples that fall outside the box are clipped and a quadratic distance
//! penalty (in the preconditioned space) is added to the objective so the
//! sampling distribution naturally drifts away from the boundary instead of
//! requiring rejection sampling.
//!
//! The implementation follows Hansen's `purecma` reference: rank-µ + rank-1
//! covariance update, cumulative step-size adaptation, lazy eigendecomposition
//! every `1/(c1+cµ)/n/10` generations.
//!
//! This module is internal; its only consumer is
//! `final_layout::Optimizer::CmaEsLm`.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for a single CMA-ES run.
#[derive(Clone, Debug)]
pub(crate) struct CmaEsConfig {
    /// Per-dim lower bound. Use `f64::NEG_INFINITY` for unbounded.
    pub lower: Vec<f64>,
    /// Per-dim upper bound. Use `f64::INFINITY` for unbounded.
    pub upper: Vec<f64>,
    /// Starting mean of the search distribution.
    pub initial_mean: Vec<f64>,
    /// Per-dim initial standard deviation. Used to scale every coordinate so
    /// the internal CMA-ES state runs on a unitless `y = (x - x0) / std`
    /// space — this is the standard preconditioning trick that lets a single
    /// scalar `sigma` work across heterogeneous parameter scales.
    pub initial_std: Vec<f64>,
    /// Hard cap on generations. The actual function-evaluation budget is
    /// `max_iters * lambda` unless [`max_evals`] is set tighter.
    pub max_iters: usize,
    /// Stop when the spread of the best fitness over the most recent
    /// `STAGNATION_GENS` generations falls below this value. Should be set
    /// well above the noise floor of the fitness — for the diagram cost on
    /// `NormalizedSumSquared` losses, ~`1e-10` is a safe choice.
    pub fn_tol: f64,
    /// Seed for sampling.
    pub seed: u64,
    /// Optional override for population size. Default is
    /// `4 + floor(3 ln n)` (Hansen's recommendation). Larger pops give a
    /// stronger global signal at proportional eval cost.
    pub lambda: Option<usize>,
    /// Optional cap on total function evaluations. Default
    /// `max_iters * lambda`.
    pub max_evals: Option<usize>,
}

/// Outcome of a CMA-ES run.
///
/// `best_f`, `iterations`, and `evals` are present for tests and future
/// debug logging — the production caller only reads `best_x`.
#[allow(dead_code)]
pub(crate) struct CmaEsResult {
    /// Best feasible point seen during the search (clipped into `[lower, upper]`).
    pub best_x: Vec<f64>,
    /// Unpenalised objective value at `best_x`.
    pub best_f: f64,
    /// Generations actually executed.
    pub iterations: usize,
    /// Function evaluations actually performed.
    pub evals: usize,
}

/// Number of recent generations the stagnation termination test inspects.
const STAGNATION_GENS: usize = 10;

/// Minimise `f` subject to box bounds via CMA-ES.
///
/// `f` is called on **clipped** parameters (always inside `[lower, upper]`),
/// so callers do not have to defend against out-of-bound inputs. The
/// quadratic boundary penalty is added by this routine to the value `f`
/// returns.
pub(crate) fn minimize<F>(config: &CmaEsConfig, mut f: F) -> CmaEsResult
where
    F: FnMut(&[f64]) -> f64,
{
    let n = config.initial_mean.len();
    assert_eq!(config.lower.len(), n, "lower bound dim mismatch");
    assert_eq!(config.upper.len(), n, "upper bound dim mismatch");
    assert_eq!(config.initial_std.len(), n, "initial_std dim mismatch");
    assert!(n > 0, "CMA-ES requires at least one dimension");

    // Per-dim scale: floor at a tiny positive so `0` initial-std doesn't
    // collapse the search direction. Caller is responsible for choosing
    // sensible scales; this is just a safety net.
    let scale: Vec<f64> = config.initial_std.iter().map(|s| s.max(1e-12)).collect();

    let n_f = n as f64;
    let lambda = config
        .lambda
        .unwrap_or_else(|| 4 + (3.0 * n_f.ln()).floor() as usize)
        .max(4);
    let mu = lambda / 2;
    let weights_raw: Vec<f64> = (1..=mu)
        .map(|i| ((mu as f64 + 1.0) / i as f64).ln())
        .collect();
    let sum_w: f64 = weights_raw.iter().sum();
    let weights: Vec<f64> = weights_raw.iter().map(|w| w / sum_w).collect();
    let mueff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

    // Strategy parameters (Hansen's defaults).
    let c_sigma = (mueff + 2.0) / (n_f + mueff + 5.0);
    let d_sigma = 1.0 + 2.0 * (((mueff - 1.0) / (n_f + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
    let c_c = (4.0 + mueff / n_f) / (n_f + 4.0 + 2.0 * mueff / n_f);
    let c1 = 2.0 / ((n_f + 1.3).powi(2) + mueff);
    let c_mu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_f + 2.0).powi(2) + mueff));
    let c_mu = c_mu.max(0.0);
    let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

    // Mean lives in scaled coordinates: 0 corresponds to `initial_mean`.
    let mut mean = DVector::<f64>::zeros(n);
    let mut sigma: f64 = 1.0;
    let mut p_sigma = DVector::<f64>::zeros(n);
    let mut p_c = DVector::<f64>::zeros(n);
    let mut c_mat = DMatrix::<f64>::identity(n, n);
    let mut b = DMatrix::<f64>::identity(n, n);
    let mut d_diag = DVector::<f64>::from_element(n, 1.0);
    let mut last_eigen_iter: usize = 0;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut best_x = clip_into_bounds(&config.initial_mean, &config.lower, &config.upper);
    let mut best_f: f64 = f(&best_x);
    let mut evals: usize = 1;
    let mut history: Vec<f64> = Vec::new();

    let max_evals = config
        .max_evals
        .unwrap_or(lambda.saturating_mul(config.max_iters.max(1)));

    let mut iter = 0;
    while iter < config.max_iters && evals < max_evals {
        // Sample lambda candidates.
        let mut samples: Vec<Sample> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            if evals >= max_evals {
                break;
            }
            let z = DVector::from_iterator(n, (0..n).map(|_| sample_normal(&mut rng)));
            // BD * z, columnwise scaled by D
            let bd_z = bd_times(&b, &d_diag, &z);
            let y = &mean + sigma * &bd_z;
            // Real-space candidate before clipping.
            let x_unclipped: Vec<f64> = (0..n)
                .map(|i| config.initial_mean[i] + scale[i] * y[i])
                .collect();
            let x_clipped: Vec<f64> = (0..n)
                .map(|i| x_unclipped[i].clamp(config.lower[i], config.upper[i]))
                .collect();
            // Penalty in scaled space so it's commensurate with sigma.
            let mut penalty = 0.0;
            for i in 0..n {
                let dx = (x_unclipped[i] - x_clipped[i]) / scale[i];
                penalty += dx * dx;
            }
            let raw_f = f(&x_clipped);
            evals += 1;
            // Track best on the unpenalised, clipped point — that's what we
            // actually want to return as a feasible solution.
            if raw_f.is_finite() && raw_f < best_f {
                best_f = raw_f;
                best_x = x_clipped.clone();
            }
            // Use the *unclipped* z for state updates so the boundary
            // pressure stays in the gradient signal; the penalty in `fit`
            // does the bound-handling work.
            let fit = if raw_f.is_finite() {
                raw_f + penalty
            } else {
                f64::INFINITY
            };
            samples.push(Sample { y, z, fit });
        }
        if samples.is_empty() {
            break;
        }
        samples.sort_by(|a, b| {
            a.fit
                .partial_cmp(&b.fit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Selection / recombination over top-mu.
        let top = &samples[..mu.min(samples.len())];
        let actual_mu = top.len();
        let w: &[f64] = &weights[..actual_mu];
        let w_norm: f64 = w.iter().sum();
        let w_eff: Vec<f64> = w.iter().map(|wi| wi / w_norm).collect();
        let mut z_bar = DVector::<f64>::zeros(n);
        let mut new_mean = DVector::<f64>::zeros(n);
        for (i, s) in top.iter().enumerate() {
            new_mean += w_eff[i] * &s.y;
            z_bar += w_eff[i] * &s.z;
        }
        mean = new_mean;

        // Conjugate evolution path (uses C^{-1/2} step = B * z_bar).
        let bz = &b * &z_bar;
        p_sigma = (1.0 - c_sigma) * &p_sigma + (c_sigma * (2.0 - c_sigma) * mueff).sqrt() * bz;
        let p_sigma_norm = p_sigma.norm();
        let damp = 1.0 - (1.0 - c_sigma).powi(2 * (iter + 1) as i32);
        let hsig = (p_sigma_norm / damp.max(1e-300).sqrt()) < (1.4 + 2.0 / (n_f + 1.0)) * chi_n;
        let hsig_f: f64 = if hsig { 1.0 } else { 0.0 };

        // Anisotropic evolution path.
        let bd_zbar = bd_times(&b, &d_diag, &z_bar);
        p_c = (1.0 - c_c) * &p_c + hsig_f * (c_c * (2.0 - c_c) * mueff).sqrt() * bd_zbar;

        // Rank-µ update from the selected samples (in BD-space).
        let mut rank_mu_update = DMatrix::<f64>::zeros(n, n);
        for (i, s) in top.iter().enumerate() {
            let bd_zi = bd_times(&b, &d_diag, &s.z);
            rank_mu_update += w_eff[i] * (&bd_zi * bd_zi.transpose());
        }
        let rank_1_update = &p_c * p_c.transpose();
        // hsig deficit term keeps trace(C) invariant when hsig=0.
        let off = c1 * (1.0 - hsig_f) * c_c * (2.0 - c_c);
        c_mat = (1.0 - c1 - c_mu + off) * &c_mat + c1 * rank_1_update + c_mu * rank_mu_update;

        // Step-size adaptation.
        sigma *= ((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1.0)).exp();

        // Lazy eigendecomposition. Hansen's heuristic: refresh once per
        // `1/(c1+cµ)/n/10` generations so the algebraic cost stays O(λn²)
        // amortised.
        let refresh_every = (1.0 / (c1 + c_mu) / n_f / 10.0).ceil() as usize;
        if (iter + 1).saturating_sub(last_eigen_iter) >= refresh_every.max(1) {
            last_eigen_iter = iter + 1;
            // Symmetrise to absorb numerical drift.
            let c_sym = 0.5 * (&c_mat + c_mat.transpose());
            let eig = SymmetricEigen::new(c_sym.clone());
            b = eig.eigenvectors;
            d_diag = eig.eigenvalues.map(|e| e.max(0.0).sqrt());
            c_mat = c_sym;
        }

        // Termination tests.
        history.push(samples[0].fit);
        if history.len() >= STAGNATION_GENS {
            let recent = &history[history.len() - STAGNATION_GENS..];
            let lo = recent.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (hi - lo) < config.fn_tol {
                iter += 1;
                break;
            }
        }
        let max_d = d_diag.iter().cloned().fold(0.0_f64, f64::max);
        if sigma * max_d < 1e-12 {
            iter += 1;
            break;
        }
        if !sigma.is_finite() || !mean.iter().all(|v| v.is_finite()) {
            // Numerical blowup — bail out and return best so far.
            iter += 1;
            break;
        }

        iter += 1;
    }

    CmaEsResult {
        best_x,
        best_f,
        iterations: iter,
        evals,
    }
}

struct Sample {
    /// Sample in scaled (preconditioned) space: y = sigma * BD * z + mean.
    y: DVector<f64>,
    /// Underlying standard-normal draw, kept so the rank-µ update can build
    /// (BD z)(BD z)ᵀ without re-solving for z.
    z: DVector<f64>,
    /// Penalised fitness used for ranking only.
    fit: f64,
}

/// Compute `B * (D ⊙ z)` without materialising the full diagonal matrix.
fn bd_times(b: &DMatrix<f64>, d: &DVector<f64>, z: &DVector<f64>) -> DVector<f64> {
    let n = z.len();
    let dz = DVector::from_iterator(n, (0..n).map(|i| d[i] * z[i]));
    b * dz
}

fn clip_into_bounds(x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lower.iter().zip(upper.iter()))
        .map(|(xi, (lo, hi))| xi.clamp(*lo, *hi))
        .collect()
}

fn sample_normal(rng: &mut StdRng) -> f64 {
    // Box–Muller. We don't bother caching the sin component because the
    // outer loop is dominated by the diagram-cost evaluation, not by the
    // RNG.
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    let mag = (-2.0 * u1.ln()).sqrt();
    mag * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: f(x) = sum x_i^2. Optimum at origin, f=0.
    /// Verifies CMA-ES finds the unconstrained optimum.
    #[test]
    fn sphere_unconstrained() {
        let n = 5;
        let config = CmaEsConfig {
            lower: vec![f64::NEG_INFINITY; n],
            upper: vec![f64::INFINITY; n],
            initial_mean: vec![3.0; n],
            initial_std: vec![1.0; n],
            max_iters: 200,
            fn_tol: 1e-12,
            seed: 42,
            lambda: None,
            max_evals: None,
        };
        let result = minimize(&config, |x| x.iter().map(|v| v * v).sum::<f64>());
        assert!(
            result.best_f < 1e-10,
            "sphere best_f = {} (expected < 1e-10), iters={}",
            result.best_f,
            result.iterations
        );
    }

    /// Rosenbrock with the optimum inside the feasible box. Verifies the
    /// routine handles a non-convex curved valley and respects bounds.
    /// 4D Rosenbrock typically takes O(10^4) evals with default `lambda`,
    /// so the budget is sized accordingly.
    #[test]
    fn rosenbrock_in_bounds() {
        let n = 4;
        let config = CmaEsConfig {
            lower: vec![-2.0; n],
            upper: vec![2.0; n],
            initial_mean: vec![-1.0; n],
            initial_std: vec![0.5; n],
            max_iters: 3000,
            fn_tol: 1e-14,
            seed: 42,
            lambda: Some(20),
            max_evals: None,
        };
        let result = minimize(&config, |x| {
            (0..x.len() - 1)
                .map(|i| {
                    let a = 1.0 - x[i];
                    let b = x[i + 1] - x[i] * x[i];
                    a * a + 100.0 * b * b
                })
                .sum::<f64>()
        });
        assert!(
            result.best_f < 1e-4,
            "rosenbrock best_f = {} (expected < 1e-4), iters={} evals={}",
            result.best_f,
            result.iterations,
            result.evals,
        );
        for v in &result.best_x {
            assert!(*v >= -2.0 && *v <= 2.0, "bound violation: {}", v);
        }
    }

    /// Optimum sits *outside* the feasible box. Verifies the boundary
    /// penalty drives the solution to the closest feasible corner instead
    /// of leaking out.
    #[test]
    fn sphere_with_active_bounds() {
        let n = 3;
        // Optimum of (x-5)^2 + ... is at x=5, but we cap at 2.
        let config = CmaEsConfig {
            lower: vec![-2.0; n],
            upper: vec![2.0; n],
            initial_mean: vec![0.0; n],
            initial_std: vec![1.0; n],
            max_iters: 300,
            fn_tol: 1e-12,
            seed: 7,
            lambda: None,
            max_evals: None,
        };
        let result = minimize(&config, |x| {
            x.iter().map(|v| (v - 5.0).powi(2)).sum::<f64>()
        });
        for v in &result.best_x {
            assert!(*v <= 2.0 + 1e-8 && *v >= -2.0, "bound violation: {}", v);
            // Should land on the upper bound.
            assert!((*v - 2.0).abs() < 1e-2, "expected x≈2.0, got {}", v);
        }
        // Best objective: 3 * (2-5)^2 = 27.
        assert!(
            (result.best_f - 27.0).abs() < 1e-2,
            "best_f = {} (expected ≈ 27)",
            result.best_f
        );
    }

    /// Reproducibility: same seed must yield bit-identical results.
    #[test]
    fn seed_reproducibility() {
        let n = 4;
        let make = || {
            let config = CmaEsConfig {
                lower: vec![-5.0; n],
                upper: vec![5.0; n],
                initial_mean: vec![1.0; n],
                initial_std: vec![1.0; n],
                max_iters: 30,
                fn_tol: 1e-10,
                seed: 12345,
                lambda: None,
                max_evals: None,
            };
            minimize(&config, |x| x.iter().map(|v| v * v).sum::<f64>())
        };
        let a = make();
        let b = make();
        assert_eq!(a.best_f, b.best_f);
        assert_eq!(a.best_x, b.best_x);
    }
}
