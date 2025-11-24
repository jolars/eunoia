//! Final layout optimization using region error minimization.
//!
//! This module implements the second optimization step that refines the initial
//! layout by minimizing the difference between target disjoint areas and actual
//! fitted areas in the diagram.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use nalgebra::DVector;

use crate::diagram::{Combination, PreprocessedSpec};
use crate::geometry::coord::Coord;
use crate::geometry::shapes::circle::Circle;
use crate::geometry::shapes::Shape;
use std::collections::{HashMap, HashSet};

/// Threshold for considering an intersection area as "viable" (non-zero).
const VIABLE_THRESHOLD: f64 = 1e-8;

/// Configuration for final layout optimization.
#[derive(Debug, Clone)]
pub(crate) struct FinalLayoutConfig {
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
}

impl Default for FinalLayoutConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Optimize the final layout by minimizing region error.
///
/// This takes the initial layout (positions and radii) and refines them to better
/// match the target disjoint areas specified by the user.
pub(crate) fn optimize_layout(
    spec: &PreprocessedSpec,
    initial_positions: &[f64], // [x0, y0, x1, y1, ..., xn, yn]
    initial_radii: &[f64],     // [r0, r1, ..., rn]
    config: FinalLayoutConfig,
) -> Result<(Vec<f64>, Vec<f64>, f64), Error> {
    let n_sets = spec.n_sets;

    // Combine positions and radii into single parameter vector
    // Layout: [x0, y0, ..., xn, yn, r0, r1, ..., rn]
    let mut initial_params = Vec::with_capacity(n_sets * 2 + n_sets);
    initial_params.extend_from_slice(initial_positions);
    initial_params.extend_from_slice(initial_radii);

    let initial_param = DVector::from_vec(initial_params);

    let cost_function = RegionErrorCost { spec };

    // NelderMead Takes a vector of parameter vectors. The number of parameter vectors must be n +
    // 1 where n is the number of optimization parameters.
    // We need to provide n + 1 initial points for the simplex.

    let initial_simplex = {
        let n_params = initial_param.len();
        let mut simplex = Vec::with_capacity(n_params + 1);
        simplex.push(initial_param.clone());

        let step_size = 0.1; // 10% perturbation

        for i in 0..n_params {
            let mut perturbed = initial_param.clone();
            perturbed[i] *= 1.0 + step_size;
            simplex.push(perturbed);
        }

        simplex
    };

    let solver = NelderMead::new(initial_simplex);

    let result = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(config.max_iterations as u64))
        .run()?;

    let final_params = result.state().get_best_param().unwrap();
    let loss = result.state().get_cost();

    // Split back into positions and radii
    let (positions, radii) = final_params.as_slice().split_at(n_sets * 2);

    Ok((positions.to_vec(), radii.to_vec(), loss))
}

/// Cost function for region error optimization.
///
/// Computes the discrepancy between target disjoint areas and actual fitted areas.
struct RegionErrorCost<'a> {
    spec: &'a PreprocessedSpec,
}

impl<'a> RegionErrorCost<'a> {
    /// Extract circles from parameter vector.
    fn params_to_circles(&self, params: &DVector<f64>) -> Vec<Circle> {
        let n_sets = self.spec.n_sets;
        let positions = params.rows(0, n_sets * 2);
        let radii = params.rows(n_sets * 2, n_sets);

        (0..n_sets)
            .map(|i| {
                let x = positions[i * 2];
                let y = positions[i * 2 + 1];
                let r = radii[i].max(0.01); // Ensure positive radius
                Circle::new(Coord::new(x, y), r)
            })
            .collect()
    }

    /// Compute all actual disjoint areas from fitted circles.
    ///
    /// This computes the area of each region in the Venn diagram by inclusion-exclusion.
    fn compute_fitted_disjoint_areas(&self, circles: &[Circle]) -> HashMap<Combination, f64> {
        let mut fitted = HashMap::new();

        // For each combination in the spec, compute its disjoint area
        for combo in self.spec.disjoint_areas.keys() {
            let area = self.compute_disjoint_region_area(circles, combo);
            fitted.insert(combo.clone(), area);
        }

        fitted
    }

    /// Compute the disjoint area for a specific combination.
    ///
    /// This is the area that belongs to ALL sets in the combination but NO other sets.
    fn compute_disjoint_region_area(&self, circles: &[Circle], combo: &Combination) -> f64 {
        // For now, implement for up to 2-way intersections
        // TODO: Implement general n-way intersection computation

        if combo.len() == 1 {
            // Single set: its total area minus all intersections with other sets
            let set_name = &combo.sets()[0];
            let idx = self.spec.set_to_idx[set_name];
            let circle = &circles[idx];

            let mut disjoint_area = std::f64::consts::PI * circle.radius().powi(2);

            // Subtract all 2-way intersections
            for (other_combo, _) in self.spec.disjoint_areas.iter() {
                if other_combo.len() == 2 && other_combo.sets().contains(set_name) {
                    // This is a 2-way intersection involving our set
                    let other_set = other_combo.sets().iter().find(|s| *s != set_name).unwrap();
                    let other_idx = self.spec.set_to_idx[other_set];
                    disjoint_area -= circle.intersection_area(&circles[other_idx]);
                }
            }

            disjoint_area.max(0.0)
        } else if combo.len() == 2 {
            // 2-way intersection: just the intersection area
            let sets = combo.sets();
            let idx1 = self.spec.set_to_idx[&sets[0]];
            let idx2 = self.spec.set_to_idx[&sets[1]];

            circles[idx1].intersection_area(&circles[idx2])
        } else {
            // TODO: Implement 3+ way intersections
            0.0
        }
    }
}

impl<'a> CostFunction for RegionErrorCost<'a> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let circles = self.params_to_circles(param);
        let fitted = self.compute_fitted_disjoint_areas(&circles);

        // Compute region error: sum of squared differences
        let mut error = 0.0;

        for (combo, &target_area) in self.spec.disjoint_areas.iter() {
            let fitted_area = fitted.get(combo).copied().unwrap_or(0.0);
            let diff = target_area - fitted_area;
            error += diff.powi(2);
        }

        Ok(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagram::DiagramSpecBuilder;

    #[test]
    fn test_optimize_layout_simple() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 3.0)
            .set("B", 4.0)
            .intersection(&["A", "B"], 1.0)
            .build()
            .unwrap();

        let preprocessed = spec.preprocess().unwrap();

        // Start with circles that are too far apart
        let positions = vec![0.0, 0.0, 5.0, 0.0];
        let radii = vec![
            (3.0 / std::f64::consts::PI).sqrt(),
            (4.0 / std::f64::consts::PI).sqrt(),
        ];

        let config = FinalLayoutConfig {
            max_iterations: 50,
            tolerance: 1e-4,
        };

        let result = optimize_layout(&preprocessed, &positions, &radii, config);
        assert!(result.is_ok());

        let (final_positions, final_radii, loss) = result.unwrap();
        assert_eq!(final_positions.len(), 4);
        assert_eq!(final_radii.len(), 2);
        assert!(loss >= 0.0);
    }
}
