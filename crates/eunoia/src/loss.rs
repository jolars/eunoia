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
    Sse,
    /// Root mean squared error: sqrt(mean((fitted - target)²))
    Rmse,
    /// Stress (venneuler-style): Σ((fitted - target) / target)²
    Stress,
    /// Maximum absolute error: max(|fitted - target|)
    MaxAbsolute,
}

impl LossType {
    /// Sum of squared errors
    pub fn sse() -> Self {
        Self::Sse
    }

    /// Root mean squared error
    pub fn rmse() -> Self {
        Self::Rmse
    }

    /// Stress loss (venneuler-style)
    pub fn stress() -> Self {
        Self::Stress
    }

    /// Maximum absolute error
    pub fn max_absolute() -> Self {
        Self::MaxAbsolute
    }

    /// Compute loss between fitted and target region areas
    pub fn compute(
        &self,
        fitted: &HashMap<RegionMask, f64>,
        target: &HashMap<RegionMask, f64>,
    ) -> f64 {
        // Collect all unique region masks from both fitted and target
        let all_masks: std::collections::HashSet<RegionMask> =
            fitted.keys().chain(target.keys()).copied().collect();

        if all_masks.is_empty() {
            return 0.0;
        }

        match self {
            LossType::Sse => {
                // Sum of squared errors
                all_masks
                    .iter()
                    .map(|&mask| {
                        let fitted_area = fitted.get(&mask).copied().unwrap_or(0.0);
                        let target_area = target.get(&mask).copied().unwrap_or(0.0);
                        (fitted_area - target_area).powi(2)
                    })
                    .sum()
            }
            LossType::Rmse => {
                // Root mean squared error
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
                // Stress: sum of squared relative errors
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        if t.abs() < 1e-10 {
                            if f.abs() < 1e-10 {
                                0.0
                            } else {
                                f.powi(2) // Target is zero, penalize any fitted value
                            }
                        } else {
                            ((f - t) / t).powi(2)
                        }
                    })
                    .sum()
            }
            LossType::MaxAbsolute => {
                // Maximum absolute error
                all_masks
                    .iter()
                    .map(|&mask| {
                        let f = fitted.get(&mask).copied().unwrap_or(0.0);
                        let t = target.get(&mask).copied().unwrap_or(0.0);
                        (f - t).abs()
                    })
                    .fold(0.0, f64::max)
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

        // ((10-12)/12)² + ((20-18)/18)² = (1/6)² + (1/9)² = 0.02778 + 0.01235 ≈ 0.04013
        let result = loss.compute(&fitted, &target);
        assert!((result - 0.04013).abs() < 0.001);
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

        // First: target=0, fitted=5: 5² = 25
        // Second: target=0, fitted=0: 0
        // Third: target=3, fitted=3: 0
        let result = loss.compute(&fitted, &target);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_equality() {
        assert_eq!(LossType::sse(), LossType::Sse);
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
