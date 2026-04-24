//! Loss function implementations for diagram fitting.
//!
//! This module provides simple loss functions that measure the difference
//! between fitted and target region areas.

use crate::geometry::diagram::RegionMask;
use std::collections::HashMap;

/// Loss function type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LossType {
    /// Sum of squared errors: Σ(fitted - target)²
    #[default]
    SumSquared,
    /// Sums of absolute errors: Σ|fitted - target|
    SumAbsoute,
    /// SumRegionError sum(|fitted / sum(fitted) - target / sum(target)|)
    SumAbsoluteRegionError,
    /// SumSquaredRegionError sum((fitted / sum(fitted) - target / sum(target))²)
    SumSquaredRegionError,
    /// Maximum absolute error: max(|fitted - target|)
    MaxAbsolute,
    /// Maximum squared error: max((fitted - target)²)
    MaxSquared,
    /// Root mean squared error: sqrt(mean((fitted - target)²))
    RootMeanSquared,
    /// Stress (venneuler-style)
    Stress,
    /// DiagError max(|fit / sum(fit) - target / sum(target)|), EulerAPE style
    DiagError,
}

impl LossType {
    /// Sum of squared errors
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
            LossType::SumSquared => all_masks
                .iter()
                .map(|&mask| {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    (f - t).powi(2)
                })
                .sum(),
            LossType::RootMeanSquared => {
                let sum_squared: f64 = all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).powi(2)
                    })
                    .sum();
                (sum_squared / all_masks.len() as f64).sqrt()
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
            LossType::MaxAbsolute => all_masks
                .iter()
                .map(|&mask| {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    (f - t).abs()
                })
                .fold(0.0, f64::max),
            LossType::MaxSquared => all_masks
                .iter()
                .map(|&mask| {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    (f - t).powi(2)
                })
                .fold(0.0, f64::max),
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
            LossType::SumAbsoute => all_masks
                .iter()
                .map(|&mask| {
                    let f = fitted.get(&mask).copied().unwrap_or(0.0);
                    let t = target.get(&mask).copied().unwrap_or(0.0);
                    (f - t).abs()
                })
                .sum(),
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

        // (10-12)² + (20-18)² + (30-28)² = 4 + 4 + 4 = 12
        assert_eq!(loss.compute(&fitted, &target), 12.0);
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

        // sqrt((4 + 4 + 4) / 3) = sqrt(4) = 2.0
        assert_eq!(loss.compute(&fitted, &target), 2.0);
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

        // max(|10-8|, |20-25|, |30-28|) = max(2, 5, 2) = 5
        assert_eq!(loss.compute(&fitted, &target), 5.0);
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

        // (0-5)² + (0-3)² = 25 + 9 = 34
        assert_eq!(loss.compute(&fitted, &target), 34.0);
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

        // (5-5)² + (3-3)² + (7-0)² = 0 + 0 + 49 = 49
        assert_eq!(loss.compute(&fitted, &target), 49.0);
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
}
