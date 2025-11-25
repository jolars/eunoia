//! Final layout optimization using region error minimization.
//!
//! This module implements the second optimization step that refines the initial
//! layout by minimizing the difference between target disjoint areas and actual
//! fitted areas in the diagram.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use nalgebra::DVector;

use crate::diagram::PreprocessedSpec;
use crate::geometry::point::Point;
use crate::geometry::shapes::circle::Circle;
use crate::geometry::shapes::Shape;

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
                Circle::new(Point::new(x, y), r)
            })
            .collect()
    }
}

/// Information about a single intersection point between shapes.
#[derive(Debug, Clone)]
pub struct IntersectionPoint {
    /// The intersection point
    point: Point,
    /// Indices of the two shapes that create this intersection
    parents: (usize, usize),
    /// Indices of all shapes that contain this point
    adopters: Vec<usize>,
}

impl IntersectionPoint {
    /// Creates a new IntersectionPoint.
    pub fn new(point: Point, parents: (usize, usize), adopters: Vec<usize>) -> Self {
        IntersectionPoint {
            point,
            parents,
            adopters,
        }
    }

    /// Returns the coordinates of the intersection point.
    pub fn point(&self) -> &Point {
        &self.point
    }

    /// Returns the indices of the two parent shapes.
    pub fn parents(&self) -> (usize, usize) {
        self.parents
    }

    /// Returns the indices of all shapes that contain this point.
    pub fn adopters(&self) -> &Vec<usize> {
        &self.adopters
    }
}

impl<'a> CostFunction for RegionErrorCost<'a> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let shapes = self.params_to_circles(param);

        let n_sets = self.spec.n_sets;

        let mut intersections: Vec<IntersectionPoint> = Vec::new();

        // We collect all intersection points between pairs of shapes.
        // We need this because if we want the area of an intersection region,
        // it is defined by the intersection points that are adopted (contained
        // within the shapes of the intersection).
        for i in 0..n_sets {
            for j in (i + 1)..n_sets {
                let pts = shapes[i].intersection_points(&shapes[j]);
                for point in pts {
                    let adopters = (0..n_sets)
                        .filter(|&k| shapes[k].contains_point(&point))
                        .collect();

                    intersections.push(IntersectionPoint {
                        point,
                        parents: (i, j),
                        adopters,
                    });
                }
            }
        }

        // Next, we need to compute the area of each region defined by the
        // intersections between the shapes (circles). We need to figure
        // out which intersections actually exist in the diagram.
        // In eulerr, this was done naively by checking all combinations,
        // but this is very inefficient when there are many sets.
        // For eunoia, we want to do this more intelligently.
        // How?

        // Compute region error: sum of squared differences
        let error = 0.0;

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
