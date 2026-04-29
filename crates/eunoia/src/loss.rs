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
    /// Returns `false` for losses built from `|·|` or `max(·)`: the
    /// gradient is zero almost everywhere except at a single active
    /// region (`Max*`) or has a discontinuity at every zero crossing
    /// (`SumAbsolute`, `DiagError`, `SumAbsoluteRegionError`). On those
    /// losses, central-difference gradients return mostly zeros and
    /// L-BFGS thrashes against the line search; the fitter routes
    /// non-smooth losses to derivative-free Nelder-Mead instead.
    /// `Smooth*` variants have C¹ surrogates and report `true`. See
    /// issue #45.
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
            // Other loss variants don't yet have analytical gradients; the
            // optimiser falls back to central finite differences.
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
}
