//! Loss function implementations for diagram fitting.
//!
//! This module provides simple loss functions that measure the difference
//! between fitted and target region areas.

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

    /// Compute loss between fitted and target values
    pub fn compute(&self, fitted: &[f64], target: &[f64]) -> f64 {
        assert_eq!(
            fitted.len(),
            target.len(),
            "Fitted and target vectors must have the same length"
        );

        if fitted.is_empty() {
            return 0.0;
        }

        match self {
            LossType::Sse => {
                // Sum of squared errors
                fitted
                    .iter()
                    .zip(target.iter())
                    .map(|(f, t)| (f - t).powi(2))
                    .sum()
            }
            LossType::Rmse => {
                // Root mean squared error
                let sum_squared: f64 = fitted
                    .iter()
                    .zip(target.iter())
                    .map(|(f, t)| (f - t).powi(2))
                    .sum();
                (sum_squared / fitted.len() as f64).sqrt()
            }
            LossType::Stress => {
                // Stress: sum of squared relative errors
                fitted
                    .iter()
                    .zip(target.iter())
                    .map(|(f, t)| {
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
                fitted
                    .iter()
                    .zip(target.iter())
                    .map(|(f, t)| (f - t).abs())
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
        let fitted = vec![10.0, 20.0, 30.0];
        let target = vec![12.0, 18.0, 28.0];

        // (10-12)² + (20-18)² + (30-28)² = 4 + 4 + 4 = 12
        assert_eq!(loss.compute(&fitted, &target), 12.0);
    }

    #[test]
    fn test_rmse() {
        let loss = LossType::rmse();
        let fitted = vec![10.0, 20.0, 30.0];
        let target = vec![12.0, 18.0, 28.0];

        // sqrt((4 + 4 + 4) / 3) = sqrt(4) = 2.0
        assert_eq!(loss.compute(&fitted, &target), 2.0);
    }

    #[test]
    fn test_stress() {
        let loss = LossType::stress();
        let fitted = vec![10.0, 20.0];
        let target = vec![12.0, 18.0];

        // ((10-12)/12)² + ((20-18)/18)² = (1/6)² + (1/9)² = 0.02778 + 0.01235 ≈ 0.04013
        let result = loss.compute(&fitted, &target);
        assert!((result - 0.04013).abs() < 0.001);
    }

    #[test]
    fn test_max_absolute() {
        let loss = LossType::max_absolute();
        let fitted = vec![10.0, 20.0, 30.0];
        let target = vec![8.0, 25.0, 28.0];

        // max(|10-8|, |20-25|, |30-28|) = max(2, 5, 2) = 5
        assert_eq!(loss.compute(&fitted, &target), 5.0);
    }

    #[test]
    fn test_empty_vectors() {
        let loss = LossType::sse();
        assert_eq!(loss.compute(&[], &[]), 0.0);
    }

    #[test]
    fn test_stress_with_zero_target() {
        let loss = LossType::stress();
        let fitted = vec![5.0, 0.0, 3.0];
        let target = vec![0.0, 0.0, 3.0];

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

    #[test]
    #[should_panic(expected = "Fitted and target vectors must have the same length")]
    fn test_mismatched_lengths() {
        let loss = LossType::sse();
        loss.compute(&[1.0, 2.0], &[1.0]);
    }
}
