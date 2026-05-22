//! Loss function implementations for diagram fitting.
//!
//! This module provides simple loss functions that measure the difference
//! between fitted and target region areas.

use crate::geometry::diagram::RegionMask;
use std::collections::HashMap;

/// Huber-style smooth approximation of `|x|`.
///
/// Returns `√(x² + ε²) − ε`. Equals `|x|` in the limit `ε → 0`,
/// is C¹ everywhere (including at `x = 0`), and matches `|x|` to within
/// `ε` for all `x`. Subtracting `ε` keeps the surrogate exactly zero at
/// the origin so the loss can still hit zero.
#[inline]
fn smooth_abs(x: f64, eps: f64) -> f64 {
    (x * x + eps * eps).sqrt() - eps
}

/// Logsumexp smooth approximation of `max_i x_i`.
///
/// Returns `ε · log Σ exp(x_i/ε)` evaluated in the numerically-stable
/// `m + ε · log Σ exp((x_i − m)/ε)` form (with `m = max_i x_i`). Equals
/// `max_i x_i` in the limit `ε → 0` and is C¹ in every `x_i`. Returns
/// `0.0` for an empty input.
#[inline]
fn smooth_max(values: &[f64], eps: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let inv_eps = 1.0 / eps;
    let sum: f64 = values.iter().map(|&v| ((v - m) * inv_eps).exp()).sum();
    m + eps * sum.ln()
}

/// Softmax weights `p_k = exp((x_k − m)/ε) / Σ exp((x_j − m)/ε)`.
///
/// These are the per-element gradients of [`smooth_max`] with respect to
/// each input: `∂smooth_max/∂x_k = p_k`. Sum to 1.0; uses the same
/// numerically-stable `m`-shifted form as `smooth_max`. Returns an empty
/// vector for empty input.
#[inline]
fn softmax_weights(values: &[f64], eps: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        // All -inf or all +inf — fall back to a uniform distribution so the
        // chain rule doesn't blow up. In practice we never hit this.
        return vec![1.0 / values.len() as f64; values.len()];
    }
    let inv_eps = 1.0 / eps;
    let exps: Vec<f64> = values.iter().map(|&v| ((v - m) * inv_eps).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Loss function type.
///
/// Every variant is **scale-invariant**: the loss magnitude is bounded
/// roughly in `[0, 1]` regardless of input area scale, so the optimizer's
/// tolerance and CMA-ES fallback threshold (`Fitter::tolerance`,
/// `Fitter::cmaes_fallback_threshold`) carry the same meaning across
/// specs from `gene_sets` (areas ~10²) to `issue71_4_set_extreme_scale`
/// (areas up to 38000). Each variant divides by an appropriate
/// target-side norm — `Σtᵢ²`, `Σ|tᵢ|`, `max|tᵢ|`, …
///
/// # Smooth vs non-smooth losses
///
/// Variants built from `|·|` or `max(·)` (`SumAbsoute`,
/// `SumAbsoluteRegionError`, `MaxAbsolute`, `MaxSquared`, `DiagError`)
/// are **non-smooth**: their gradients are zero almost everywhere or
/// discontinuous at every zero crossing, which stalls L-BFGS. The
/// fitter routes them to derivative-free Nelder-Mead — fast but coarse
/// (issue #45).
///
/// For each non-smooth variant there is a `Smooth*` counterpart with an
/// `eps` payload that replaces `|·|` with `√(x² + ε²) − ε` (Huber) and
/// `max(·)` with `ε · log Σ exp(·/ε)` (logsumexp). Those are smooth
/// surrogates: C¹ everywhere, converging to the true loss as `ε → 0`.
/// The fitter dispatches them through the L-BFGS path, which converges
/// to dramatically better minima at higher cost than the NM fallback.
///
/// Pick `eps` ~ 1% of typical residual magnitude. Smaller `eps` is closer
/// to the true loss but inherits more of its gradient pathology; larger
/// `eps` gives crisper gradients but biases the optimum.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LossType {
    /// Normalised sum of squared errors: `Σ(fitted - target)² / Σtarget²`.
    ///
    /// Default. Dividing by `Σt²` is a constant of the spec, so the
    /// descent direction is identical to the un-normalised
    /// `Σ(f - t)²` — only the loss-magnitude scale changes.
    ///
    /// Unlike [`Stress`], there is no β-rescale degree of freedom, so this
    /// loss penalises both shape *and* scale mismatch and won't let small
    /// regions drift the way Stress can on high-arity specs.
    ///
    /// [`Stress`]: Self::Stress
    #[default]
    SumSquared,
    /// Normalised sum of absolute errors: `Σ|fitted - target| / Σ|target|`.
    /// Non-smooth — see [`SmoothSumAbsolute`] for the gradient-friendly
    /// surrogate.
    ///
    /// [`SmoothSumAbsolute`]: Self::SmoothSumAbsolute
    SumAbsoute,
    /// `Σ|fitted/Σfitted - target/Σtarget|`. Non-smooth — see
    /// [`SmoothSumAbsoluteRegionError`].
    ///
    /// [`SmoothSumAbsoluteRegionError`]: Self::SmoothSumAbsoluteRegionError
    SumAbsoluteRegionError,
    /// `Σ(fitted/Σfitted - target/Σtarget)²`. Smooth.
    SumSquaredRegionError,
    /// Normalised maximum absolute error: `max|fitted - target| / max|target|`.
    /// Non-smooth — see [`SmoothMaxAbsolute`].
    ///
    /// [`SmoothMaxAbsolute`]: Self::SmoothMaxAbsolute
    MaxAbsolute,
    /// Normalised maximum squared error: `max(fitted - target)² / max(target²)`.
    /// Non-smooth (the `max` aggregator) — see [`SmoothMaxSquared`].
    ///
    /// [`SmoothMaxSquared`]: Self::SmoothMaxSquared
    MaxSquared,
    /// Normalised root-mean-squared error:
    /// `sqrt(Σ(fitted - target)² / Σtarget²)` (= sqrt of [`SumSquared`]).
    ///
    /// [`SumSquared`]: Self::SumSquared
    RootMeanSquared,
    /// Stress (venneuler-style). Already normalised by `Σf²`.
    Stress,
    /// DiagError `max|fit/Σfit - target/Σtarget|`, EulerAPE style.
    /// Non-smooth — see [`SmoothDiagError`].
    ///
    /// [`SmoothDiagError`]: Self::SmoothDiagError
    DiagError,
    /// Smooth surrogate of [`SumAbsoute`]:
    /// `Σ smooth_abs(f - t, ε) / Σ|target|`.
    ///
    /// [`SumAbsoute`]: Self::SumAbsoute
    SmoothSumAbsolute {
        /// Huber smoothing parameter; converges to true `SumAbsoute` as
        /// `eps → 0`. Pick ~1% of typical residual magnitude.
        eps: f64,
    },
    /// Smooth surrogate of [`SumAbsoluteRegionError`]:
    /// `Σ smooth_abs(f/Σf - t/Σt, ε)`.
    ///
    /// [`SumAbsoluteRegionError`]: Self::SumAbsoluteRegionError
    SmoothSumAbsoluteRegionError {
        /// Huber smoothing parameter; converges to true
        /// `SumAbsoluteRegionError` as `eps → 0`.
        eps: f64,
    },
    /// Smooth surrogate of [`MaxAbsolute`]:
    /// `smooth_max(smooth_abs(f - t, ε)) / max|target|`.
    ///
    /// [`MaxAbsolute`]: Self::MaxAbsolute
    SmoothMaxAbsolute {
        /// Huber/logsumexp smoothing parameter; converges to true
        /// `MaxAbsolute` as `eps → 0`.
        eps: f64,
    },
    /// Smooth surrogate of [`MaxSquared`]:
    /// `smooth_max((f - t)²) / max(target²)`.
    ///
    /// The squared term is already smooth, so only the `max`
    /// aggregator is replaced with logsumexp.
    ///
    /// [`MaxSquared`]: Self::MaxSquared
    SmoothMaxSquared {
        /// Logsumexp smoothing parameter; converges to true
        /// `MaxSquared` as `eps → 0`.
        eps: f64,
    },
    /// Smooth surrogate of [`DiagError`]:
    /// `smooth_max(smooth_abs(f/Σf - t/Σt, ε))`.
    ///
    /// [`DiagError`]: Self::DiagError
    SmoothDiagError {
        /// Huber/logsumexp smoothing parameter; converges to true
        /// `DiagError` as `eps → 0`.
        eps: f64,
    },
}

impl LossType {
    /// Normalised sum of squared errors. Alias for [`LossType::SumSquared`].
    pub fn sse() -> Self {
        Self::SumSquared
    }

    /// Root mean squared error
    pub fn rmse() -> Self {
        Self::RootMeanSquared
    }

    /// Stress loss (venneuler-style)
    pub fn stress() -> Self {
        Self::Stress
    }

    /// Maximum absolute error
    pub fn max_absolute() -> Self {
        Self::MaxAbsolute
    }

    /// Maximum squared error
    pub fn max_squared() -> Self {
        Self::MaxSquared
    }

    /// Sum of absolute errors
    pub fn sum_absolute() -> Self {
        Self::SumAbsoute
    }

    /// Sum of absolute region errors
    pub fn sum_absolute_region_error() -> Self {
        Self::SumAbsoluteRegionError
    }

    /// Sum of squared region errors
    pub fn sum_squared_region_error() -> Self {
        Self::SumSquaredRegionError
    }

    /// Diagonal error (EulerAPE style)
    pub fn diag_error() -> Self {
        Self::DiagError
    }

    /// Smooth surrogate of [`SumAbsoute`]. Converges to it as `eps → 0`.
    ///
    /// [`SumAbsoute`]: Self::SumAbsoute
    pub fn smooth_sum_absolute(eps: f64) -> Self {
        Self::SmoothSumAbsolute { eps }
    }

    /// Smooth surrogate of [`SumAbsoluteRegionError`]. Converges to it as
    /// `eps → 0`.
    ///
    /// [`SumAbsoluteRegionError`]: Self::SumAbsoluteRegionError
    pub fn smooth_sum_absolute_region_error(eps: f64) -> Self {
        Self::SmoothSumAbsoluteRegionError { eps }
    }

    /// Smooth surrogate of [`MaxAbsolute`]. Converges to it as `eps → 0`.
    ///
    /// [`MaxAbsolute`]: Self::MaxAbsolute
    pub fn smooth_max_absolute(eps: f64) -> Self {
        Self::SmoothMaxAbsolute { eps }
    }

    /// Smooth surrogate of [`MaxSquared`]. Converges to it as `eps → 0`.
    ///
    /// [`MaxSquared`]: Self::MaxSquared
    pub fn smooth_max_squared(eps: f64) -> Self {
        Self::SmoothMaxSquared { eps }
    }

    /// Smooth surrogate of [`DiagError`]. Converges to it as `eps → 0`.
    ///
    /// [`DiagError`]: Self::DiagError
    pub fn smooth_diag_error(eps: f64) -> Self {
        Self::SmoothDiagError { eps }
    }

    /// Whether this loss is smooth (continuously differentiable) in the
    /// region areas `f`.
    ///
    /// Returns `false` for losses built from `|·|` or `max(·)`. Their
    /// defect is not a missing gradient — a subgradient exists almost
    /// everywhere — but that the optima lie *on* the non-smooth set:
    /// minimax losses (`MaxAbsolute`, `MaxSquared`, `DiagError`) balance
    /// several equally-active residuals, and L1 losses (`SumAbsolute`,
    /// `SumAbsoluteRegionError`) sit where many regions match their target
    /// exactly. There:
    ///
    /// - the subgradient is discontinuous at every kink (`|·|` flips sign
    ///   across `fᵢ = tᵢ`; `max(·)` jumps as the active region switches),
    ///   so the line search's smoothness assumption breaks and L-BFGS
    ///   thrashes (issue #45);
    /// - its magnitude doesn't vanish approaching the optimum (it stays
    ///   `±const` for `|·|`), so the first-order convergence test never
    ///   trips; and
    /// - a central difference across a kink returns an attenuated,
    ///   step-size-dependent slope — an implicit, uncontrolled smoothing —
    ///   so finite-difference gradients don't rescue the gradient path
    ///   either.
    ///
    /// The fitter therefore routes these to derivative-free Nelder-Mead.
    /// The principled alternative is a `Smooth*` variant: a C¹
    /// Huber/logsumexp surrogate with an analytic gradient and a tunable
    /// smoothing scale — the controlled version of what a finite difference
    /// does by accident. Those report `true`.
    pub fn is_smooth(&self) -> bool {
        match self {
            LossType::SumSquared
            | LossType::RootMeanSquared
            | LossType::Stress
            | LossType::SumSquaredRegionError
            | LossType::SmoothSumAbsolute { .. }
            | LossType::SmoothSumAbsoluteRegionError { .. }
            | LossType::SmoothMaxAbsolute { .. }
            | LossType::SmoothMaxSquared { .. }
            | LossType::SmoothDiagError { .. } => true,
            LossType::SumAbsoute
            | LossType::SumAbsoluteRegionError
            | LossType::MaxAbsolute
            | LossType::MaxSquared
            | LossType::DiagError => false,
        }
    }

    /// Compute loss between fitted and target region areas
    pub fn compute(
        &self,
        fitted: &HashMap<RegionMask, f64>,
        target: &HashMap<RegionMask, f64>,
    ) -> f64 {
        // Collect all unique region masks from both fitted and target, sorted
        // for deterministic iteration order (HashMap/HashSet use RandomState,
        // and ULP-level floating-point differences from different summation
        // orders can flip Nelder-Mead accept/reject decisions).
        let mut all_masks: Vec<RegionMask> = fitted.keys().chain(target.keys()).copied().collect();
        all_masks.sort_unstable();
        all_masks.dedup();

        if all_masks.is_empty() {
            return 0.0;
        }

        match self {
            LossType::SumSquared => {
                let sum_t2: f64 = target.values().map(|&v| v * v).sum();
                if sum_t2 < 1e-20 {
                    return 0.0;
                }
                let sum_sq: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).powi(2)
                    })
                    .sum();
                sum_sq / sum_t2
            }
            LossType::RootMeanSquared => {
                // sqrt(Σ(f-t)² / Σt²) — scale-invariant variant of the
                // classic RMSE. Equals `sqrt(SumSquared)` after this
                // normalisation.
                let sum_t2: f64 = target.values().map(|&v| v * v).sum();
                if sum_t2 < 1e-20 {
                    return 0.0;
                }
                let sum_squared: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).powi(2)
                    })
                    .sum();
                (sum_squared / sum_t2).sqrt()
            }
            LossType::Stress => {
                // venneuler-style stress (matches eulerr):
                // stress = Σ(f - β·t)² / Σf²  where  β = Σ(f·t) / Σt²
                let sum_ft: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        f * t
                    })
                    .sum();
                let sum_t2: f64 = target.values().map(|&v| v * v).sum();
                let sum_f2: f64 = fitted.values().map(|&v| v * v).sum();

                if sum_t2 < 1e-20 || sum_f2 < 1e-20 {
                    return 0.0;
                }

                let beta = sum_ft / sum_t2;
                let numerator: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - beta * t).powi(2)
                    })
                    .sum();
                numerator / sum_f2
            }
            LossType::MaxAbsolute => {
                let max_t: f64 = target.values().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if max_t < 1e-20 {
                    return 0.0;
                }
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).abs()
                    })
                    .fold(0.0, f64::max)
                    / max_t
            }
            LossType::MaxSquared => {
                let max_t2: f64 = target.values().map(|v| v * v).fold(0.0_f64, f64::max);
                if max_t2 < 1e-20 {
                    return 0.0;
                }
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).powi(2)
                    })
                    .fold(0.0, f64::max)
                    / max_t2
            }
            LossType::DiagError => {
                // eulerr's diagError: max|f_i/Σf - t_i/Σt| (linear sum normalization)
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();

                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return f64::MAX;
                }

                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f / ssf - t / sst).abs()
                    })
                    .fold(0.0, f64::max)
            }
            LossType::SumAbsoute => {
                let sum_abs_t: f64 = target.values().map(|v| v.abs()).sum();
                if sum_abs_t < 1e-20 {
                    return 0.0;
                }
                let sum_abs: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).abs()
                    })
                    .sum();
                sum_abs / sum_abs_t
            }
            LossType::SumAbsoluteRegionError => {
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return f64::MAX;
                }
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f / ssf - t / sst).abs()
                    })
                    .sum()
            }
            LossType::SumSquaredRegionError => {
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return f64::MAX;
                }
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f / ssf - t / sst).powi(2)
                    })
                    .sum()
            }
            LossType::SmoothSumAbsolute { eps } => {
                let eps = eps.max(f64::MIN_POSITIVE);
                let sum_abs_t: f64 = target.values().map(|v| v.abs()).sum();
                if sum_abs_t < 1e-20 {
                    return 0.0;
                }
                let sum_abs: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        smooth_abs(f - t, eps)
                    })
                    .sum();
                sum_abs / sum_abs_t
            }
            LossType::SmoothSumAbsoluteRegionError { eps } => {
                let eps = eps.max(f64::MIN_POSITIVE);
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return f64::MAX;
                }
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        smooth_abs(f / ssf - t / sst, eps)
                    })
                    .sum()
            }
            LossType::SmoothMaxAbsolute { eps } => {
                let eps = eps.max(f64::MIN_POSITIVE);
                let max_t: f64 = target.values().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if max_t < 1e-20 {
                    return 0.0;
                }
                let smoothed_abs: Vec<f64> = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        smooth_abs(f - t, eps)
                    })
                    .collect();
                smooth_max(&smoothed_abs, eps) / max_t
            }
            LossType::SmoothMaxSquared { eps } => {
                let eps = eps.max(f64::MIN_POSITIVE);
                let max_t2: f64 = target.values().map(|v| v * v).fold(0.0_f64, f64::max);
                if max_t2 < 1e-20 {
                    return 0.0;
                }
                let squared: Vec<f64> = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).powi(2)
                    })
                    .collect();
                smooth_max(&squared, eps) / max_t2
            }
            LossType::SmoothDiagError { eps } => {
                let eps = eps.max(f64::MIN_POSITIVE);
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return f64::MAX;
                }
                let smoothed_abs: Vec<f64> = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        smooth_abs(f / ssf - t / sst, eps)
                    })
                    .collect();
                smooth_max(&smoothed_abs, eps)
            }
        }
    }

    /// Compute loss and analytical gradient `∂L/∂f_mask`, returning `None` if
    /// no closed-form gradient is implemented for this loss type. Callers
    /// fall back to finite differences when this returns `None`.
    pub fn compute_with_gradient(
        &self,
        fitted: &HashMap<RegionMask, f64>,
        target: &HashMap<RegionMask, f64>,
    ) -> Option<(f64, HashMap<RegionMask, f64>)> {
        let mut all_masks: Vec<RegionMask> = fitted.keys().chain(target.keys()).copied().collect();
        all_masks.sort_unstable();
        all_masks.dedup();

        if all_masks.is_empty() {
            return Some((0.0, HashMap::new()));
        }

        match self {
            LossType::SumSquared => {
                let sum_t2: f64 = target.values().map(|&v| v * v).sum();
                if sum_t2 < 1e-20 {
                    return Some((0.0, HashMap::new()));
                }
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                let mut total = 0.0;
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let diff = f - t;
                    total += diff * diff;
                    grad.insert(mask, 2.0 * diff / sum_t2);
                }
                Some((total / sum_t2, grad))
            }
            LossType::RootMeanSquared => {
                // L = sqrt(SSE / Σt²). ∂L/∂f_m = (f_m − t_m) / (L · Σt²).
                let sum_t2: f64 = target.values().map(|&v| v * v).sum();
                if sum_t2 < 1e-20 {
                    return Some((0.0, HashMap::new()));
                }
                let mut sse = 0.0;
                let diffs: Vec<(RegionMask, f64)> = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        let d = f - t;
                        sse += d * d;
                        (mask, d)
                    })
                    .collect();
                let loss = (sse / sum_t2).sqrt();
                if loss < 1e-20 {
                    // At L=0 the gradient is the subgradient {0}; report it
                    // as zero so the optimiser sees a stationary point.
                    return Some((0.0, HashMap::new()));
                }
                let denom = loss * sum_t2;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(diffs.len());
                for (mask, d) in diffs {
                    grad.insert(mask, d / denom);
                }
                Some((loss, grad))
            }
            LossType::Stress => {
                // L = Σ(f − β·t)² / Σf², β = Σ(f·t)/Σt². β minimises the
                // numerator, so ∂N/∂β = 0 and the envelope theorem gives
                // ∂L/∂f_m = 2[(f_m − β·t_m) − L·f_m] / Σf².
                let mut sum_ft = 0.0;
                let mut sum_t2 = 0.0;
                let mut sum_f2 = 0.0;
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    sum_ft += f * t;
                    sum_t2 += t * t;
                    sum_f2 += f * f;
                }
                if sum_t2 < 1e-20 || sum_f2 < 1e-20 {
                    return Some((0.0, HashMap::new()));
                }
                let beta = sum_ft / sum_t2;
                let mut numerator = 0.0;
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let r = f - beta * t;
                    numerator += r * r;
                }
                let loss = numerator / sum_f2;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let g = 2.0 * ((f - beta * t) - loss * f) / sum_f2;
                    grad.insert(mask, g);
                }
                Some((loss, grad))
            }
            LossType::SumSquaredRegionError => {
                // L = Σ r_k² with r_k = f_k/Σf − t_k/Σt. Then
                // ∂L/∂f_m = (2/Σf)(r_m − c) where c = (1/Σf) Σ_k r_k f_k.
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return Some((0.0, HashMap::new()));
                }
                let mut residuals: HashMap<RegionMask, f64> =
                    HashMap::with_capacity(all_masks.len());
                let mut loss = 0.0;
                let mut c_num = 0.0;
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let r = f / ssf - t / sst;
                    loss += r * r;
                    c_num += r * f;
                    residuals.insert(mask, r);
                }
                let c = c_num / ssf;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let r = residuals[&mask];
                    grad.insert(mask, (2.0 / ssf) * (r - c));
                }
                Some((loss, grad))
            }
            LossType::SmoothSumAbsolute { eps } => {
                // L = Σ smooth_abs(f_m − t_m, ε) / Σ|t|.
                // ∂L/∂f_m = (f_m − t_m) / [√((f_m−t_m)² + ε²) · Σ|t|].
                let eps = eps.max(f64::MIN_POSITIVE);
                let sum_abs_t: f64 = target.values().map(|v| v.abs()).sum();
                if sum_abs_t < 1e-20 {
                    return Some((0.0, HashMap::new()));
                }
                let mut loss = 0.0;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let d = f - t;
                    let denom = (d * d + eps * eps).sqrt();
                    loss += denom - eps;
                    grad.insert(mask, d / (denom * sum_abs_t));
                }
                Some((loss / sum_abs_t, grad))
            }
            LossType::SmoothSumAbsoluteRegionError { eps } => {
                // L = Σ smooth_abs(r_k, ε), r_k = f_k/Σf − t_k/Σt.
                // w_k = r_k / √(r_k² + ε²); ∂L/∂f_m = (1/Σf)[w_m − (1/Σf) Σ_k w_k f_k].
                let eps = eps.max(f64::MIN_POSITIVE);
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return Some((0.0, HashMap::new()));
                }
                let mut loss = 0.0;
                let mut weights: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                let mut wf_sum = 0.0;
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let r = f / ssf - t / sst;
                    let denom = (r * r + eps * eps).sqrt();
                    loss += denom - eps;
                    let w = r / denom;
                    wf_sum += w * f;
                    weights.insert(mask, w);
                }
                let c = wf_sum / ssf;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let w = weights[&mask];
                    grad.insert(mask, (w - c) / ssf);
                }
                Some((loss, grad))
            }
            LossType::SmoothMaxAbsolute { eps } => {
                // L = smooth_max([smooth_abs(f_m − t_m, ε)]) / max|t|.
                // Let p_k be softmax weights; ∂L/∂f_m =
                // p_m · (f_m − t_m) / √((f_m−t_m)² + ε²) / max|t|.
                let eps = eps.max(f64::MIN_POSITIVE);
                let max_t: f64 = target.values().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if max_t < 1e-20 {
                    return Some((0.0, HashMap::new()));
                }
                let mut diffs: Vec<f64> = Vec::with_capacity(all_masks.len());
                let mut smoothed: Vec<f64> = Vec::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let d = f - t;
                    diffs.push(d);
                    smoothed.push(smooth_abs(d, eps));
                }
                let p = softmax_weights(&smoothed, eps);
                let loss = smooth_max(&smoothed, eps) / max_t;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for (i, &mask) in all_masks.iter().enumerate() {
                    let d = diffs[i];
                    let denom = (d * d + eps * eps).sqrt();
                    grad.insert(mask, p[i] * d / denom / max_t);
                }
                Some((loss, grad))
            }
            LossType::SmoothMaxSquared { eps } => {
                // L = smooth_max([(f_m − t_m)²]) / max(t²).
                // ∂L/∂f_m = p_m · 2(f_m − t_m) / max(t²).
                let eps = eps.max(f64::MIN_POSITIVE);
                let max_t2: f64 = target.values().map(|v| v * v).fold(0.0_f64, f64::max);
                if max_t2 < 1e-20 {
                    return Some((0.0, HashMap::new()));
                }
                let mut diffs: Vec<f64> = Vec::with_capacity(all_masks.len());
                let mut squared: Vec<f64> = Vec::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let d = f - t;
                    diffs.push(d);
                    squared.push(d * d);
                }
                let p = softmax_weights(&squared, eps);
                let loss = smooth_max(&squared, eps) / max_t2;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for (i, &mask) in all_masks.iter().enumerate() {
                    grad.insert(mask, p[i] * 2.0 * diffs[i] / max_t2);
                }
                Some((loss, grad))
            }
            LossType::SmoothDiagError { eps } => {
                // L = smooth_max([smooth_abs(r_k, ε)]) where r_k = f_k/Σf − t_k/Σt.
                // ∂L/∂f_m = (1/Σf)[p_m·w_m − (1/Σf) Σ_k p_k·w_k·f_k],
                // with w_k = r_k / √(r_k² + ε²) and p_k softmax over σ_k.
                let eps = eps.max(f64::MIN_POSITIVE);
                let ssf = fitted.values().sum::<f64>();
                let sst = target.values().sum::<f64>();
                if ssf.abs() < 1e-10 || sst.abs() < 1e-10 {
                    return Some((0.0, HashMap::new()));
                }
                let mut weights: Vec<f64> = Vec::with_capacity(all_masks.len());
                let mut smoothed: Vec<f64> = Vec::with_capacity(all_masks.len());
                let mut fs: Vec<f64> = Vec::with_capacity(all_masks.len());
                for &mask in &all_masks {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    let r = f / ssf - t / sst;
                    let denom = (r * r + eps * eps).sqrt();
                    smoothed.push(denom - eps);
                    weights.push(r / denom);
                    fs.push(f);
                }
                let p = softmax_weights(&smoothed, eps);
                let loss = smooth_max(&smoothed, eps);
                let pwf_sum: f64 = (0..all_masks.len())
                    .map(|i| p[i] * weights[i] * fs[i])
                    .sum();
                let c = pwf_sum / ssf;
                let mut grad: HashMap<RegionMask, f64> = HashMap::with_capacity(all_masks.len());
                for (i, &mask) in all_masks.iter().enumerate() {
                    grad.insert(mask, (p[i] * weights[i] - c) / ssf);
                }
                Some((loss, grad))
            }
            // Non-smooth variants (`SumAbsoute`, `SumAbsoluteRegionError`,
            // `MaxAbsolute`, `MaxSquared`, `DiagError`) deliberately fall
            // back to FD here. Their gradients are zero almost everywhere
            // or discontinuous at every zero crossing, so a subgradient
            // would mislead L-BFGS more than help it. Use the matching
            // `Smooth*` variant if you want the analytic path.
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse() {
        let loss = LossType::sse();

        let mut fitted = HashMap::new();
        fitted.insert(0b001, 10.0);
        fitted.insert(0b010, 20.0);
        fitted.insert(0b100, 30.0);

        let mut target = HashMap::new();
        target.insert(0b001, 12.0);
        target.insert(0b010, 18.0);
        target.insert(0b100, 28.0);

        // Σ(f-t)² = 4 + 4 + 4 = 12; Σt² = 144 + 324 + 784 = 1252.
        // SumSquared = 12 / 1252.
        let expected = 12.0 / 1252.0;
        assert!((loss.compute(&fitted, &target) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_rmse() {
        let loss = LossType::rmse();

        let mut fitted = HashMap::new();
        fitted.insert(0b001, 10.0);
        fitted.insert(0b010, 20.0);
        fitted.insert(0b100, 30.0);

        let mut target = HashMap::new();
        target.insert(0b001, 12.0);
        target.insert(0b010, 18.0);
        target.insert(0b100, 28.0);

        // sqrt(Σ(f-t)² / Σt²) = sqrt(12 / 1252).
        let expected = (12.0_f64 / 1252.0).sqrt();
        assert!((loss.compute(&fitted, &target) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_stress() {
        let loss = LossType::stress();

        let mut fitted = HashMap::new();
        fitted.insert(0b001, 10.0);
        fitted.insert(0b010, 20.0);

        let mut target = HashMap::new();
        target.insert(0b001, 12.0);
        target.insert(0b010, 18.0);

        // venneuler/eulerr stress: Σ(f - β·t)² / Σf² where β = Σ(f·t) / Σt²
        // Σft = 10·12 + 20·18 = 480
        // Σt² = 144 + 324 = 468  →  β = 480/468 = 40/39
        // (10 - 40/39·12)² + (20 - 40/39·18)² = (90/39)² + (60/39)² = 11700/1521
        // Σf² = 100 + 400 = 500  →  stress = 11700/1521/500 ≈ 0.015385
        let result = loss.compute(&fitted, &target);
        assert!(
            (result - 0.015385).abs() < 1e-5,
            "expected 0.015385, got {}",
            result
        );
    }

    #[test]
    fn test_max_absolute() {
        let loss = LossType::max_absolute();

        let mut fitted = HashMap::new();
        fitted.insert(0b001, 10.0);
        fitted.insert(0b010, 20.0);
        fitted.insert(0b100, 30.0);

        let mut target = HashMap::new();
        target.insert(0b001, 8.0);
        target.insert(0b010, 25.0);
        target.insert(0b100, 28.0);

        // max(|10-8|, |20-25|, |30-28|) / max|t| = 5 / 28
        assert!((loss.compute(&fitted, &target) - 5.0 / 28.0).abs() < 1e-12);
    }

    #[test]
    fn test_empty_target() {
        let loss = LossType::sse();
        let fitted = HashMap::new();
        let target = HashMap::new();
        assert_eq!(loss.compute(&fitted, &target), 0.0);
    }

    #[test]
    fn test_missing_fitted_area() {
        let loss = LossType::sse();

        let fitted = HashMap::new(); // Empty - no fitted areas

        let mut target = HashMap::new();
        target.insert(0b001, 5.0);
        target.insert(0b010, 3.0);

        // Σ(0 - t)² = 25 + 9 = 34;  Σt² = 25 + 9 = 34;  loss = 1.0
        assert!((loss.compute(&fitted, &target) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_extra_fitted_area() {
        let loss = LossType::sse();

        let mut fitted = HashMap::new();
        fitted.insert(0b001, 5.0);
        fitted.insert(0b010, 3.0);
        fitted.insert(0b100, 7.0); // Extra region not in target

        let mut target = HashMap::new();
        target.insert(0b001, 5.0);
        target.insert(0b010, 3.0);
        // 0b100 missing from target

        // Σ(f-t)² = 0 + 0 + 49 = 49;  Σt² = 25 + 9 = 34;  loss = 49/34.
        let expected = 49.0 / 34.0;
        assert!((loss.compute(&fitted, &target) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_stress_with_zero_target() {
        let loss = LossType::stress();

        let mut fitted = HashMap::new();
        fitted.insert(0b001, 5.0);
        fitted.insert(0b010, 0.0);
        fitted.insert(0b100, 3.0);

        let mut target = HashMap::new();
        target.insert(0b001, 0.0);
        target.insert(0b010, 0.0);
        target.insert(0b100, 3.0);

        // Σft = 0 + 0 + 9 = 9;  Σt² = 9  →  β = 1
        // numerator = (5-0)² + (0-0)² + (3-3)² = 25
        // Σf² = 25 + 0 + 9 = 34  →  stress = 25/34 ≈ 0.735294
        let result = loss.compute(&fitted, &target);
        assert!(
            (result - 25.0 / 34.0).abs() < 1e-10,
            "expected 0.735294, got {}",
            result
        );
    }

    #[test]
    fn test_equality() {
        assert_eq!(LossType::sse(), LossType::SumSquared);
        assert_eq!(LossType::stress(), LossType::Stress);
        assert_ne!(LossType::sse(), LossType::rmse());
    }

    #[test]
    fn test_clone() {
        let loss = LossType::sse();
        let cloned = loss;
        assert_eq!(loss, cloned);
    }

    #[test]
    fn test_is_smooth() {
        assert!(LossType::SumSquared.is_smooth());
        assert!(LossType::RootMeanSquared.is_smooth());
        assert!(LossType::Stress.is_smooth());
        assert!(LossType::SumSquaredRegionError.is_smooth());

        assert!(!LossType::SumAbsoute.is_smooth());
        assert!(!LossType::SumAbsoluteRegionError.is_smooth());
        assert!(!LossType::MaxAbsolute.is_smooth());
        assert!(!LossType::MaxSquared.is_smooth());
        assert!(!LossType::DiagError.is_smooth());
    }

    #[test]
    fn test_smooth_abs_basics() {
        // smooth_abs(x, ε) → |x| as ε → 0; matches |x| within ε.
        assert!((smooth_abs(0.0, 1e-3) - 0.0).abs() < 1e-3);
        assert!((smooth_abs(1.0, 1e-6) - 1.0).abs() < 1e-6);
        assert!((smooth_abs(-2.5, 1e-6) - 2.5).abs() < 1e-6);
        // Always non-negative.
        assert!(smooth_abs(0.0, 0.5) >= 0.0);
        assert!(smooth_abs(-3.0, 0.5) >= 0.0);
    }

    #[test]
    fn test_smooth_max_basics() {
        // smooth_max(xs, ε) → max(xs) as ε → 0.
        let xs = vec![1.0, 2.0, 3.0];
        assert!((smooth_max(&xs, 1e-6) - 3.0).abs() < 1e-3);
        // Empty is 0.
        assert_eq!(smooth_max(&[], 1.0), 0.0);
        // Single value: smoothed max equals that value.
        assert!((smooth_max(&[5.0], 1e-6) - 5.0).abs() < 1e-9);
        // Numerically stable for large values: m subtracted before exp.
        let big = vec![1e6, 1e6 - 1.0];
        let res = smooth_max(&big, 1.0);
        assert!(res.is_finite());
        assert!((res - 1e6).abs() < 5.0); // within a few ε
    }

    #[test]
    fn test_smooth_variants_converge_to_true_loss() {
        // Each Smooth* variant must converge to its non-smooth counterpart
        // as ε → 0.
        let mut fitted = HashMap::new();
        fitted.insert(0b001, 10.0);
        fitted.insert(0b010, 20.0);
        fitted.insert(0b100, 30.0);

        let mut target = HashMap::new();
        target.insert(0b001, 8.0);
        target.insert(0b010, 25.0);
        target.insert(0b100, 28.0);

        let pairs: &[(LossType, LossType)] = &[
            (LossType::SumAbsoute, LossType::smooth_sum_absolute(1e-9)),
            (LossType::MaxAbsolute, LossType::smooth_max_absolute(1e-9)),
            (LossType::MaxSquared, LossType::smooth_max_squared(1e-9)),
            (
                LossType::SumAbsoluteRegionError,
                LossType::smooth_sum_absolute_region_error(1e-9),
            ),
            (LossType::DiagError, LossType::smooth_diag_error(1e-9)),
        ];

        for &(true_loss, smooth_loss) in pairs {
            let exact = true_loss.compute(&fitted, &target);
            let smoothed = smooth_loss.compute(&fitted, &target);
            assert!(
                (smoothed - exact).abs() < 1e-3 * exact.abs().max(1e-3),
                "{:?} vs {:?}: smoothed = {}, exact = {}",
                true_loss,
                smooth_loss,
                smoothed,
                exact
            );
        }
    }

    #[test]
    fn test_smooth_variants_are_smooth() {
        assert!(LossType::smooth_sum_absolute(1e-3).is_smooth());
        assert!(LossType::smooth_sum_absolute_region_error(1e-3).is_smooth());
        assert!(LossType::smooth_max_absolute(1e-3).is_smooth());
        assert!(LossType::smooth_max_squared(1e-3).is_smooth());
        assert!(LossType::smooth_diag_error(1e-3).is_smooth());
    }

    /// Build the synthetic 4-region (f, t) input used by every analytic-vs-FD
    /// loss-gradient test. Asymmetric values so per-mask gradients are all
    /// distinct and degenerate cancellations don't hide bugs.
    fn fixture_for_grad() -> (HashMap<RegionMask, f64>, HashMap<RegionMask, f64>) {
        let mut fitted = HashMap::new();
        fitted.insert(0b0001, 10.0);
        fitted.insert(0b0010, 22.5);
        fitted.insert(0b0100, 31.0);
        fitted.insert(0b1000, 4.0);

        let mut target = HashMap::new();
        target.insert(0b0001, 8.0);
        target.insert(0b0010, 25.0);
        target.insert(0b0100, 28.0);
        target.insert(0b1000, 6.0);
        (fitted, target)
    }

    /// Verify the analytic per-mask gradient returned by `compute_with_gradient`
    /// matches a central-difference estimate of `compute()` on the same
    /// (fitted, target) pair, within `tol` relative error.
    fn assert_loss_grad_matches_fd(loss: LossType, h: f64, tol: f64) {
        let (fitted, target) = fixture_for_grad();
        let (loss_val, analytic) = loss
            .compute_with_gradient(&fitted, &target)
            .expect("analytic gradient");
        // Sanity: the analytic loss agrees with `compute()`.
        let plain = loss.compute(&fitted, &target);
        assert!(
            (loss_val - plain).abs() <= 1e-9 + 1e-9 * plain.abs(),
            "{:?}: analytic loss {} vs compute() {}",
            loss,
            loss_val,
            plain
        );

        let masks: Vec<RegionMask> = {
            let mut m: Vec<_> = fitted.keys().chain(target.keys()).copied().collect();
            m.sort_unstable();
            m.dedup();
            m
        };
        for &mask in &masks {
            let mut plus = fitted.clone();
            let mut minus = fitted.clone();
            *plus.entry(mask).or_insert(0.0) += h;
            *minus.entry(mask).or_insert(0.0) -= h;
            let fd = (loss.compute(&plus, &target) - loss.compute(&minus, &target)) / (2.0 * h);
            let an = analytic.get(&mask).copied().unwrap_or(0.0);
            let scale = fd.abs().max(1e-6);
            let rel = (an - fd).abs() / scale;
            assert!(
                rel < tol,
                "{:?}, mask {:b}: analytic={} fd={} rel={:.3e}",
                loss,
                mask,
                an,
                fd,
                rel
            );
        }
    }

    #[test]
    fn test_grad_sum_squared_matches_fd() {
        assert_loss_grad_matches_fd(LossType::SumSquared, 1e-6, 1e-5);
    }

    #[test]
    fn test_grad_root_mean_squared_matches_fd() {
        assert_loss_grad_matches_fd(LossType::RootMeanSquared, 1e-6, 1e-5);
    }

    #[test]
    fn test_grad_stress_matches_fd() {
        assert_loss_grad_matches_fd(LossType::Stress, 1e-6, 1e-5);
    }

    #[test]
    fn test_grad_sum_squared_region_error_matches_fd() {
        assert_loss_grad_matches_fd(LossType::SumSquaredRegionError, 1e-6, 1e-5);
    }

    #[test]
    fn test_grad_smooth_sum_absolute_matches_fd() {
        // ε ≈ 1% of typical residual (~2.5).
        assert_loss_grad_matches_fd(LossType::smooth_sum_absolute(0.05), 1e-6, 1e-5);
    }

    #[test]
    fn test_grad_smooth_sum_absolute_region_error_matches_fd() {
        // r_k is ~O(1/Σf) ≈ 0.015, so use a smaller ε to stay in the smooth
        // regime without dominating the loss.
        assert_loss_grad_matches_fd(LossType::smooth_sum_absolute_region_error(1e-3), 1e-6, 1e-4);
    }

    #[test]
    fn test_grad_smooth_max_absolute_matches_fd() {
        // Larger ε keeps the softmax weights bounded away from a one-hot
        // vector — which makes FD comparisons stable on this fixture (one
        // residual is ~3× the next-largest, so a tiny ε would put nearly all
        // mass on a single mask and amplify FD noise).
        assert_loss_grad_matches_fd(LossType::smooth_max_absolute(0.5), 1e-6, 1e-4);
    }

    #[test]
    fn test_grad_smooth_max_squared_matches_fd() {
        assert_loss_grad_matches_fd(LossType::smooth_max_squared(0.5), 1e-6, 1e-4);
    }

    #[test]
    fn test_grad_smooth_diag_error_matches_fd() {
        assert_loss_grad_matches_fd(LossType::smooth_diag_error(1e-3), 1e-6, 1e-4);
    }

    #[test]
    fn test_grad_degenerate_inputs() {
        // Empty target: every smooth loss reports (0.0, empty) — gradient is
        // either undefined or a stationary subgradient at 0.
        let empty = HashMap::<RegionMask, f64>::new();
        let mut fitted = HashMap::new();
        fitted.insert(0b001, 10.0);
        fitted.insert(0b010, 5.0);

        let smooth_losses = [
            LossType::SumSquared,
            LossType::RootMeanSquared,
            LossType::Stress,
            LossType::SumSquaredRegionError,
            LossType::smooth_sum_absolute(1e-3),
            LossType::smooth_sum_absolute_region_error(1e-3),
            LossType::smooth_max_absolute(1e-3),
            LossType::smooth_max_squared(1e-3),
            LossType::smooth_diag_error(1e-3),
        ];
        for loss in smooth_losses {
            let (l, g) = loss
                .compute_with_gradient(&fitted, &empty)
                .expect("smooth losses always return Some");
            assert_eq!(l, 0.0, "{:?}: loss should be 0 for empty target", loss);
            assert!(g.is_empty(), "{:?}: grad should be empty", loss);
        }

        // Non-smooth losses still fall back to FD (return None).
        let non_smooth = [
            LossType::SumAbsoute,
            LossType::SumAbsoluteRegionError,
            LossType::MaxAbsolute,
            LossType::MaxSquared,
            LossType::DiagError,
        ];
        for loss in non_smooth {
            assert!(
                loss.compute_with_gradient(&fitted, &empty).is_none(),
                "{:?}: should return None to trigger FD fallback",
                loss
            );
        }
    }
}
