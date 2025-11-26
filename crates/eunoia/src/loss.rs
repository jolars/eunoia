use crate::geometry::diagram::RegionMask;
use std::collections::HashMap;

/// How to aggregate errors across regions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aggregation {
    /// Sum of squared errors (default)
    SumSquared,
    /// Sum of absolute errors
    SumAbsolute,
    /// Maximum absolute error
    MaxAbsolute,
    /// Maximum relative error
    MaxRelative,
    /// Root mean squared error
    RootMeanSquared,
}

impl Default for Aggregation {
    fn default() -> Self {
        Self::SumSquared
    }
}

/// What to measure for each region
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorMetric {
    /// Absolute difference: |fitted - target|
    Absolute,
    /// Squared difference: (fitted - target)²
    Squared,
    /// Relative difference: |fitted - target| / target
    Relative,
    /// Relative squared: ((fitted - target) / target)²
    RelativeSquared,
}

impl Default for ErrorMetric {
    fn default() -> Self {
        Self::Squared
    }
}

/// Public API for loss function configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossType {
    pub metric: ErrorMetric,
    pub aggregation: Aggregation,
}

impl Default for LossType {
    fn default() -> Self {
        Self::region_error()
    }
}

impl LossType {
    /// Classic region error: sum of squared errors (eulerr default)
    pub fn region_error() -> Self {
        Self {
            metric: ErrorMetric::Squared,
            aggregation: Aggregation::SumSquared,
        }
    }

    /// Stress loss: sum of squared relative errors (venneuler-style)
    pub fn stress() -> Self {
        Self {
            metric: ErrorMetric::RelativeSquared,
            aggregation: Aggregation::SumSquared,
        }
    }

    /// DiagError: sum of absolute relative errors
    pub fn diag_error() -> Self {
        Self {
            metric: ErrorMetric::Relative,
            aggregation: Aggregation::SumAbsolute,
        }
    }

    /// Minimize maximum absolute error
    pub fn minimax() -> Self {
        Self {
            metric: ErrorMetric::Absolute,
            aggregation: Aggregation::MaxAbsolute,
        }
    }

    /// Minimize maximum relative error
    pub fn minimax_relative() -> Self {
        Self {
            metric: ErrorMetric::Relative,
            aggregation: Aggregation::MaxRelative,
        }
    }

    /// Custom loss function
    pub fn custom(metric: ErrorMetric, aggregation: Aggregation) -> Self {
        Self {
            metric,
            aggregation,
        }
    }

    pub(crate) fn create(&self) -> Box<dyn LossFunction> {
        Box::new(ConfigurableLoss {
            metric: self.metric,
            aggregation: self.aggregation,
        })
    }
}

pub(crate) trait LossFunction {
    fn evaluate(
        &self,
        fitted_areas: &HashMap<RegionMask, f64>,
        target_areas: &HashMap<RegionMask, f64>,
    ) -> f64;

    #[allow(dead_code)]
    fn name(&self) -> &str;
}

struct ConfigurableLoss {
    metric: ErrorMetric,
    aggregation: Aggregation,
}

impl ConfigurableLoss {
    fn compute_error(&self, fitted: f64, target: f64) -> f64 {
        match self.metric {
            ErrorMetric::Absolute => (fitted - target).abs(),
            ErrorMetric::Squared => {
                let diff = fitted - target;
                diff * diff
            }
            ErrorMetric::Relative => {
                if target.abs() < 1e-10 {
                    if fitted.abs() < 1e-10 {
                        0.0 // Both zero, no error
                    } else {
                        fitted.abs() // Target is zero, error is the fitted value
                    }
                } else {
                    ((fitted - target) / target).abs()
                }
            }
            ErrorMetric::RelativeSquared => {
                if target.abs() < 1e-10 {
                    if fitted.abs() < 1e-10 {
                        0.0
                    } else {
                        fitted * fitted
                    }
                } else {
                    let rel = (fitted - target) / target;
                    rel * rel
                }
            }
        }
    }

    fn aggregate(&self, errors: Vec<f64>) -> f64 {
        if errors.is_empty() {
            return 0.0;
        }

        match self.aggregation {
            Aggregation::SumSquared => errors.iter().sum(),
            Aggregation::SumAbsolute => errors.iter().sum(),
            Aggregation::MaxAbsolute => errors.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Aggregation::MaxRelative => errors.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Aggregation::RootMeanSquared => {
                let sum: f64 = errors.iter().sum();
                (sum / errors.len() as f64).sqrt()
            }
        }
    }
}

impl LossFunction for ConfigurableLoss {
    fn evaluate(
        &self,
        fitted_areas: &HashMap<RegionMask, f64>,
        target_areas: &HashMap<RegionMask, f64>,
    ) -> f64 {
        let errors: Vec<f64> = target_areas
            .iter()
            .map(|(&mask, &target_area)| {
                let fitted_area = fitted_areas.get(&mask).copied().unwrap_or(0.0);
                self.compute_error(fitted_area, target_area)
            })
            .collect();

        self.aggregate(errors)
    }

    fn name(&self) -> &str {
        match (self.metric, self.aggregation) {
            (ErrorMetric::Squared, Aggregation::SumSquared) => "region_error",
            (ErrorMetric::RelativeSquared, Aggregation::SumSquared) => "stress",
            (ErrorMetric::Relative, Aggregation::SumAbsolute) => "diag_error",
            (ErrorMetric::Absolute, Aggregation::MaxAbsolute) => "minimax",
            (ErrorMetric::Relative, Aggregation::MaxRelative) => "minimax_relative",
            _ => "custom",
        }
    }
}
