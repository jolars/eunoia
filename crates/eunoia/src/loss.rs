use crate::geometry::diagram::RegionMask;
use std::collections::HashMap;

/// How to aggregate errors across regions
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Aggregation {
    /// Sum of squared errors (default)
    #[default]
    Sum,
    /// Maximum absolute error
    MaxAbsolute,
    /// Maximum relative error
    MaxRelative,
    /// Root mean squared error
    RootMeanSquared,
}

/// What to measure for each region
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ErrorMetric {
    /// Absolute difference: |fitted - target|
    Absolute,
    /// Squared difference: (fitted - target)²
    #[default]
    Squared,
    /// Relative difference: |fitted - target| / target
    Relative,
    /// Relative squared: ((fitted - target) / target)²
    RelativeSquared,
    /// Region error: |(fitted / sum_fitted) - (target / sum_target)|
    RegionError,
}

/// Public API for loss function configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossType {
    pub metric: ErrorMetric,
    pub aggregation: Aggregation,
}

impl Default for LossType {
    fn default() -> Self {
        Self::sse()
    }
}

impl LossType {
    /// Sum of squared errors
    pub fn sse() -> Self {
        Self {
            metric: ErrorMetric::Squared,
            aggregation: Aggregation::Sum,
        }
    }

    /// Stress loss: sum of squared relative errors (venneuler-style)
    pub fn stress() -> Self {
        Self {
            metric: ErrorMetric::RelativeSquared,
            aggregation: Aggregation::Sum,
        }
    }

    /// Sum of absolute relative errors
    pub fn sse_relative() -> Self {
        Self {
            metric: ErrorMetric::Relative,
            aggregation: Aggregation::Sum,
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

    /// DiagError (from EulerAPE): maximize the minimum of fitted/target ratios across regions
    pub fn diag_error() -> Self {
        Self {
            metric: ErrorMetric::RegionError,
            aggregation: Aggregation::MaxAbsolute,
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
    /// Compute per-region error, with access to full maps for complex metrics.
    fn compute_error(
        &self,
        fitted: f64,
        target: f64,
        fitted_all: &HashMap<RegionMask, f64>,
        target_all: &HashMap<RegionMask, f64>,
    ) -> f64 {
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
            ErrorMetric::RegionError => {
                let sum_fitted = fitted_all.values().sum::<f64>();
                let sum_target = target_all.values().sum::<f64>();

                (fitted / sum_fitted - target / sum_target).abs()
            }
        }
    }

    fn aggregate(&self, errors: Vec<f64>) -> f64 {
        if errors.is_empty() {
            return 0.0;
        }

        match self.aggregation {
            Aggregation::Sum => errors.iter().sum(),
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
                self.compute_error(fitted_area, target_area, fitted_areas, target_areas)
            })
            .collect();

        self.aggregate(errors)
    }

    fn name(&self) -> &str {
        match (self.metric, self.aggregation) {
            (ErrorMetric::Squared, Aggregation::Sum) => "sse",
            (ErrorMetric::RelativeSquared, Aggregation::Sum) => "stress",
            (ErrorMetric::Relative, Aggregation::Sum) => "sse_relative",
            (ErrorMetric::Absolute, Aggregation::MaxAbsolute) => "minimax",
            (ErrorMetric::Relative, Aggregation::MaxRelative) => "minimax_relative",
            (ErrorMetric::RegionError, Aggregation::MaxAbsolute) => "diag_error",
            _ => "custom",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregation_default() {
        assert_eq!(Aggregation::default(), Aggregation::Sum);
    }

    #[test]
    fn test_aggregation_clone() {
        let agg = Aggregation::MaxAbsolute;
        let cloned = agg;
        assert_eq!(agg, cloned);
    }

    #[test]
    fn test_error_metric_default() {
        assert_eq!(ErrorMetric::default(), ErrorMetric::Squared);
    }

    #[test]
    fn test_error_metric_equality() {
        assert_eq!(ErrorMetric::Absolute, ErrorMetric::Absolute);
        assert_ne!(ErrorMetric::Absolute, ErrorMetric::Squared);
    }

    #[test]
    fn test_loss_type_default() {
        let loss = LossType::default();
        assert_eq!(loss.metric, ErrorMetric::Squared);
        assert_eq!(loss.aggregation, Aggregation::Sum);
    }

    #[test]
    fn test_loss_type_region_error() {
        let loss = LossType::sse();
        assert_eq!(loss.metric, ErrorMetric::Squared);
        assert_eq!(loss.aggregation, Aggregation::Sum);
    }

    #[test]
    fn test_loss_type_stress() {
        let loss = LossType::stress();
        assert_eq!(loss.metric, ErrorMetric::RelativeSquared);
        assert_eq!(loss.aggregation, Aggregation::Sum);
    }

    #[test]
    fn test_loss_type_diag_error() {
        let loss = LossType::sse_relative();
        assert_eq!(loss.metric, ErrorMetric::Relative);
        assert_eq!(loss.aggregation, Aggregation::Sum);
    }

    #[test]
    fn test_loss_type_minimax() {
        let loss = LossType::minimax();
        assert_eq!(loss.metric, ErrorMetric::Absolute);
        assert_eq!(loss.aggregation, Aggregation::MaxAbsolute);
    }

    #[test]
    fn test_loss_type_minimax_relative() {
        let loss = LossType::minimax_relative();
        assert_eq!(loss.metric, ErrorMetric::Relative);
        assert_eq!(loss.aggregation, Aggregation::MaxRelative);
    }

    #[test]
    fn test_loss_type_custom() {
        let loss = LossType::custom(ErrorMetric::Absolute, Aggregation::RootMeanSquared);
        assert_eq!(loss.metric, ErrorMetric::Absolute);
        assert_eq!(loss.aggregation, Aggregation::RootMeanSquared);
    }

    #[test]
    fn test_loss_type_equality() {
        let loss1 = LossType::sse();
        let loss2 = LossType::sse();
        let loss3 = LossType::stress();
        assert_eq!(loss1, loss2);
        assert_ne!(loss1, loss3);
    }

    #[test]
    fn test_loss_type_clone() {
        let loss = LossType::minimax();
        let cloned = loss;
        assert_eq!(loss, cloned);
    }

    #[test]
    fn test_aggregate_sum_squared() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Absolute,
            aggregation: Aggregation::Sum,
        };
        assert_eq!(loss.aggregate(vec![1.0, 2.0, 3.0]), 6.0);
        assert_eq!(loss.aggregate(vec![]), 0.0);
    }

    #[test]
    fn test_aggregate_sum_absolute() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Absolute,
            aggregation: Aggregation::Sum,
        };
        assert_eq!(loss.aggregate(vec![1.0, 2.0, 3.0]), 6.0);
        assert_eq!(loss.aggregate(vec![]), 0.0);
    }

    #[test]
    fn test_aggregate_max_absolute() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Absolute,
            aggregation: Aggregation::MaxAbsolute,
        };
        assert_eq!(loss.aggregate(vec![1.0, 5.0, 3.0]), 5.0);
        assert_eq!(loss.aggregate(vec![7.0]), 7.0);
    }

    #[test]
    fn test_aggregate_max_relative() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Relative,
            aggregation: Aggregation::MaxRelative,
        };
        assert_eq!(loss.aggregate(vec![0.1, 0.5, 0.3]), 0.5);
    }

    #[test]
    fn test_aggregate_root_mean_squared() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Squared,
            aggregation: Aggregation::RootMeanSquared,
        };
        let result = loss.aggregate(vec![4.0, 9.0, 16.0]);
        assert!((result - 3.1091).abs() < 0.001);
        assert_eq!(loss.aggregate(vec![16.0]), 4.0);
    }

    #[test]
    fn test_loss_function_evaluate() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Squared,
            aggregation: Aggregation::Sum,
        };

        let mut fitted_areas = HashMap::new();
        fitted_areas.insert(0b001, 10.0);
        fitted_areas.insert(0b010, 20.0);

        let mut target_areas = HashMap::new();
        target_areas.insert(0b001, 12.0);
        target_areas.insert(0b010, 18.0);

        // (10-12)^2 + (20-18)^2 = 4 + 4 = 8
        assert_eq!(loss.evaluate(&fitted_areas, &target_areas), 8.0);
    }

    #[test]
    fn test_loss_function_evaluate_missing_fitted() {
        let loss = ConfigurableLoss {
            metric: ErrorMetric::Squared,
            aggregation: Aggregation::Sum,
        };

        let fitted_areas = HashMap::new();
        let mut target_areas = HashMap::new();
        target_areas.insert(0b001, 5.0);

        // fitted=0, target=5: (0-5)^2 = 25
        assert_eq!(loss.evaluate(&fitted_areas, &target_areas), 25.0);
    }

    #[test]
    fn test_loss_function_name() {
        let tests = vec![
            (ErrorMetric::Squared, Aggregation::Sum, "sse"),
            (ErrorMetric::RelativeSquared, Aggregation::Sum, "stress"),
            (ErrorMetric::Relative, Aggregation::Sum, "sse_relative"),
            (ErrorMetric::Absolute, Aggregation::MaxAbsolute, "minimax"),
            (
                ErrorMetric::Relative,
                Aggregation::MaxRelative,
                "minimax_relative",
            ),
            (
                ErrorMetric::Absolute,
                Aggregation::RootMeanSquared,
                "custom",
            ),
        ];

        for (metric, aggregation, expected_name) in tests {
            let loss = ConfigurableLoss {
                metric,
                aggregation,
            };
            assert_eq!(loss.name(), expected_name);
        }
    }

    #[test]
    fn test_loss_type_create() {
        let loss_type = LossType::sse();
        let loss_fn = loss_type.create();

        let mut fitted_areas = HashMap::new();
        fitted_areas.insert(0b001, 10.0);

        let mut target_areas = HashMap::new();
        target_areas.insert(0b001, 8.0);

        // (10-8)^2 = 4
        assert_eq!(loss_fn.evaluate(&fitted_areas, &target_areas), 4.0);
        assert_eq!(loss_fn.name(), "sse");
    }

    #[test]
    fn test_debug_implementations() {
        let agg = Aggregation::Sum;
        let debug = format!("{:?}", agg);
        assert!(debug.contains("Sum"));

        let metric = ErrorMetric::Relative;
        let debug = format!("{:?}", metric);
        assert!(debug.contains("Relative"));

        let loss = LossType::stress();
        let debug = format!("{:?}", loss);
        assert!(debug.contains("RelativeSquared"));
    }
}
