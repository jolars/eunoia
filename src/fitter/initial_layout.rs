use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::DVector;
use rand::Rng;

use crate::diagram::PairwiseRelations;

/// Configuration for the initial layout optimization.
#[derive(Debug, Clone)]
pub(crate) struct InitialLayoutConfig {
    /// Maximum number of optimization attempts
    pub max_attempts: usize,
    /// Number of attempts without improvement before stopping
    pub patience: usize,
    /// Relative improvement threshold (e.g., 0.01 for 1% improvement)
    pub improvement_threshold: f64,
    /// Absolute loss threshold for perfect fit early stopping
    pub perfect_fit_threshold: f64,
}

impl Default for InitialLayoutConfig {
    fn default() -> Self {
        Self {
            max_attempts: 100,
            patience: 5,
            improvement_threshold: 0.001, // 0.1% improvement
            perfect_fit_threshold: 1e-8,  // Near-zero loss
        }
    }
}

/// Compute initial layout using patience-based optimization with default configuration.
pub(crate) fn compute_initial_layout(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
) -> Result<Vec<f64>, Error> {
    compute_initial_layout_with_config(distances, relationships, InitialLayoutConfig::default())
}

/// Compute initial layout using patience-based optimization.
///
/// This function tries random initializations until no improvement is seen
/// for `patience` consecutive attempts, or until `max_attempts` is reached.
pub(crate) fn compute_initial_layout_with_config(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    config: InitialLayoutConfig,
) -> Result<Vec<f64>, Error> {
    let n_sets = distances.len();

    let mut best_params = Vec::new();
    let mut best_loss = f64::INFINITY;
    let mut attempts_without_improvement = 0;
    let mut rng = rand::rng();

    // Compute scale for random initialization
    let max_distance = distances
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(0.0_f64, f64::max);

    let scale = if max_distance > 0.0 {
        max_distance
    } else {
        10.0
    };

    for _attempt in 0..config.max_attempts {
        // Initialize with random positions in a reasonable range
        let mut initial_values = vec![0.0; n_sets * 2];
        for value in &mut initial_values {
            *value = rng.random_range(-scale..scale);
        }
        let initial_param = DVector::from_vec(initial_values);

        let cost_function = MdsCost {
            distances,
            relationships,
        };

        let line_search = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(line_search, 7);

        let result = Executor::new(cost_function, solver)
            .configure(|state| state.param(initial_param).max_iters(100))
            .run()?;

        let loss = result.state().get_cost();

        // Early stopping if we've achieved perfect fit
        if loss < config.perfect_fit_threshold {
            return Ok(result.state().get_best_param().unwrap().as_slice().to_vec());
        }

        // Check if this is an improvement
        if loss < best_loss {
            let relative_improvement = if best_loss.is_finite() && best_loss > 0.0 {
                (best_loss - loss) / best_loss
            } else {
                f64::INFINITY
            };

            // Accept first valid result or improvements above threshold
            if !best_loss.is_finite() || relative_improvement > config.improvement_threshold {
                best_loss = loss;
                best_params = result.state().get_best_param().unwrap().as_slice().to_vec();
                attempts_without_improvement = 0;
            } else {
                attempts_without_improvement += 1;
            }
        } else {
            attempts_without_improvement += 1;
        }

        // Early stopping if no improvement for patience attempts
        if attempts_without_improvement >= config.patience {
            break;
        }
    }

    Ok(best_params)
}

struct MdsCost<'a> {
    distances: &'a Vec<Vec<f64>>,
    relationships: &'a PairwiseRelations,
}

impl<'a> CostFunction for MdsCost<'a> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let n_sets = param.len() / 2;
        let x = param.rows(0, n_sets);
        let y = param.rows(n_sets, n_sets);

        let mut loss = 0.0;

        for i in 0..n_sets {
            for j in 0..n_sets {
                if i == j {
                    continue;
                }

                let xd = x[i] - x[j];
                let yd = y[i] - y[j];
                let d = xd.powi(2) + yd.powi(2) - self.distances[i][j].powi(2);

                if self.relationships.is_disjoint(i, j) && d >= 0.0 {
                    continue;
                }

                if (self.relationships.is_subset(i, j) || self.relationships.is_subset(j, i))
                    && d <= 0.0
                {
                    continue;
                }

                loss += d.powi(2);
            }
        }

        Ok(loss)
    }
}

impl<'a> Gradient for MdsCost<'a> {
    type Param = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let n_sets = param.len() / 2;
        let x = param.rows(0, n_sets);
        let y = param.rows(n_sets, n_sets);

        let mut grad = DVector::from_element(param.len(), 0.0);

        for i in 0..n_sets {
            for j in 0..n_sets {
                if i == j {
                    continue;
                }

                let xd = x[i] - x[j];
                let yd = y[i] - y[j];
                let d = xd.powi(2) + yd.powi(2) - self.distances[i][j].powi(2);

                if self.relationships.is_disjoint(i, j) && d >= 0.0 {
                    continue;
                }

                if (self.relationships.is_subset(i, j) || self.relationships.is_subset(j, i))
                    && d <= 0.0
                {
                    continue;
                }

                grad[i] += 4.0 * d * xd;
                grad[n_sets + i] += 4.0 * d * yd;
            }
        }

        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn create_test_relationships(n_sets: usize) -> PairwiseRelations {
        PairwiseRelations {
            n_sets,
            subset: vec![vec![false; n_sets]; n_sets],
            disjoint: vec![vec![false; n_sets]; n_sets],
            overlap_areas: vec![vec![0.0; n_sets]; n_sets],
        }
    }

    #[test]
    fn test_compute_initial_layout_two_sets_touching() {
        // Two circles with radii r1=1, r2=1, touching (distance = 2)
        let distances = vec![vec![0.0, 2.0], vec![2.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        // Should have 4 parameters (2 x, 2 y)
        assert_eq!(result.len(), 4);

        let (x, y) = result.split_at(2);

        // Calculate actual distance between the two points
        let dx = x[0] - x[1];
        let dy = y[0] - y[1];
        let actual_distance = (dx * dx + dy * dy).sqrt();

        // Should be close to 2.0
        assert!(
            approx_eq(actual_distance, 2.0, 0.1),
            "Distance {} should be close to 2.0",
            actual_distance
        );
    }

    #[test]
    fn test_compute_initial_layout_two_sets_overlapping() {
        // Two circles with centers 1 unit apart (overlapping)
        let distances = vec![vec![0.0, 1.0], vec![1.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 4);

        let (x, y) = result.split_at(2);

        let dx = x[0] - x[1];
        let dy = y[0] - y[1];
        let actual_distance = (dx * dx + dy * dy).sqrt();

        assert!(
            approx_eq(actual_distance, 1.0, 0.1),
            "Distance {} should be close to 1.0",
            actual_distance
        );
    }

    #[test]
    fn test_compute_initial_layout_two_sets_separated() {
        // Two circles far apart
        let distances = vec![vec![0.0, 5.0], vec![5.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 4);

        let (x, y) = result.split_at(2);

        let dx = x[0] - x[1];
        let dy = y[0] - y[1];
        let actual_distance = (dx * dx + dy * dy).sqrt();

        assert!(
            approx_eq(actual_distance, 5.0, 0.1),
            "Distance {} should be close to 5.0",
            actual_distance
        );
    }

    #[test]
    fn test_compute_initial_layout_two_sets_disjoint() {
        // Two disjoint sets (distance should be at least r1 + r2)
        let distances = vec![vec![0.0, 3.0], vec![3.0, 0.0]];

        let mut relationships = create_test_relationships(2);
        relationships.disjoint[0][1] = true;
        relationships.disjoint[1][0] = true;

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 4);

        let (x, y) = result.split_at(2);

        let dx = x[0] - x[1];
        let dy = y[0] - y[1];
        let actual_distance = (dx * dx + dy * dy).sqrt();

        // For disjoint sets, distance should be at least the specified distance
        assert!(
            actual_distance >= 3.0 - 0.1,
            "Distance {} should be at least 3.0",
            actual_distance
        );
    }

    #[test]
    fn test_compute_initial_layout_three_sets_triangle() {
        // Three sets arranged in a triangle
        let d = 2.0; // Equal distances
        let distances = vec![vec![0.0, d, d], vec![d, 0.0, d], vec![d, d, 0.0]];

        let relationships = create_test_relationships(3);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 6); // 3 x, 3 y

        let (x, y) = result.split_at(3);

        // Check all pairwise distances
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dx = x[i] - x[j];
                let dy = y[i] - y[j];
                let actual_distance = (dx * dx + dy * dy).sqrt();

                assert!(
                    approx_eq(actual_distance, d, 0.2),
                    "Distance between {} and {} is {}, should be close to {}",
                    i,
                    j,
                    actual_distance,
                    d
                );
            }
        }
    }

    #[test]
    fn test_compute_initial_layout_three_sets_collinear() {
        // Three sets in a line: A---B---C
        let distances = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let relationships = create_test_relationships(3);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 6);

        let (x, y) = result.split_at(3);

        // Verify distances
        let d01 = ((x[0] - x[1]).powi(2) + (y[0] - y[1]).powi(2)).sqrt();
        let d12 = ((x[1] - x[2]).powi(2) + (y[1] - y[2]).powi(2)).sqrt();
        let d02 = ((x[0] - x[2]).powi(2) + (y[0] - y[2]).powi(2)).sqrt();

        assert!(approx_eq(d01, 1.0, 0.2), "Distance 0-1: {}", d01);
        assert!(approx_eq(d12, 1.0, 0.2), "Distance 1-2: {}", d12);
        assert!(approx_eq(d02, 2.0, 0.2), "Distance 0-2: {}", d02);
    }

    #[test]
    fn test_compute_initial_layout_four_sets_square() {
        // Four sets in a square pattern
        let side = 1.0;
        let diag = side * 2.0_f64.sqrt();
        let distances = vec![
            vec![0.0, side, diag, side],
            vec![side, 0.0, side, diag],
            vec![diag, side, 0.0, side],
            vec![side, diag, side, 0.0],
        ];

        let relationships = create_test_relationships(4);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 8);

        let (x, y) = result.split_at(4);

        // Check that opposite corners are farther than adjacent sides
        let d01 = ((x[0] - x[1]).powi(2) + (y[0] - y[1]).powi(2)).sqrt();
        let d02 = ((x[0] - x[2]).powi(2) + (y[0] - y[2]).powi(2)).sqrt();

        // Adjacent sides should be closer than diagonal
        assert!(d01 < d02);
    }

    #[test]
    fn test_compute_initial_layout_with_restarts() {
        // Test that multiple restarts work
        let distances = vec![vec![0.0, 1.5], vec![1.5, 0.0]];

        let relationships = create_test_relationships(2);

        let result1 = compute_initial_layout(&distances, &relationships).unwrap();
        let result2 = compute_initial_layout(&distances, &relationships).unwrap();

        // Both should produce valid results
        assert_eq!(result1.len(), 4);
        assert_eq!(result2.len(), 4);

        // Both should satisfy the distance constraint approximately
        let (x1, y1) = result1.split_at(2);
        let d1 = ((x1[0] - x1[1]).powi(2) + (y1[0] - y1[1]).powi(2)).sqrt();

        let (x2, y2) = result2.split_at(2);
        let d2 = ((x2[0] - x2[1]).powi(2) + (y2[0] - y2[1]).powi(2)).sqrt();

        assert!(approx_eq(d1, 1.5, 0.2), "Distance with first run: {}", d1);
        assert!(approx_eq(d2, 1.5, 0.2), "Distance with second run: {}", d2);
    }

    #[test]
    fn test_compute_initial_layout_zero_distance() {
        // Two sets at the same position (fully overlapping)
        let distances = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships).unwrap();

        assert_eq!(result.len(), 4);

        let (x, y) = result.split_at(2);

        let dx = x[0] - x[1];
        let dy = y[0] - y[1];
        let actual_distance = (dx * dx + dy * dy).sqrt();

        // Should be very close to 0
        assert!(
            actual_distance < 0.1,
            "Distance {} should be close to 0.0",
            actual_distance
        );
    }

    #[test]
    fn test_compute_initial_layout_asymmetric_distances() {
        // Test with slightly asymmetric input (should still work)
        let distances = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.5],
            vec![2.0, 1.5, 0.0],
        ];

        let relationships = create_test_relationships(3);

        let result = compute_initial_layout(&distances, &relationships);

        // Should succeed even with asymmetric distances
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 6);
    }

    #[test]
    fn test_patience_based_optimization() {
        // Test that patience-based optimization stops early when no improvement
        let distances = vec![vec![0.0, 1.5], vec![1.5, 0.0]];
        let relationships = create_test_relationships(2);

        let config = InitialLayoutConfig {
            max_attempts: 20,
            patience: 3,
            improvement_threshold: 0.01,
            perfect_fit_threshold: 1e-8,
        };

        let result = compute_initial_layout_with_config(&distances, &relationships, config);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.len(), 4);

        // Verify the result satisfies distance constraints
        let (x, y) = params.split_at(2);
        let d = ((x[0] - x[1]).powi(2) + (y[0] - y[1]).powi(2)).sqrt();
        assert!(approx_eq(d, 1.5, 0.2), "Distance: {}", d);
    }

    #[test]
    fn test_patience_with_zero_threshold() {
        // Test with zero threshold - should only improve on any decrease
        let distances = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let relationships = create_test_relationships(2);

        let config = InitialLayoutConfig {
            max_attempts: 10,
            patience: 2,
            improvement_threshold: 0.0,
            perfect_fit_threshold: 1e-8,
        };

        let result = compute_initial_layout_with_config(&distances, &relationships, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_early_stopping_on_perfect_fit() {
        // Test that optimization stops immediately when loss reaches zero
        // Two circles at exact distance (should achieve near-zero loss quickly)
        let distances = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let relationships = create_test_relationships(2);

        let config = InitialLayoutConfig {
            max_attempts: 100,
            patience: 5,
            improvement_threshold: 0.01,
            perfect_fit_threshold: 1e-8,
        };

        let result = compute_initial_layout_with_config(&distances, &relationships, config);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.len(), 4);

        // Verify the distance is correct
        let (x, y) = params.split_at(2);
        let d = ((x[0] - x[1]).powi(2) + (y[0] - y[1]).powi(2)).sqrt();
        assert!(approx_eq(d, 1.0, 0.1), "Distance: {}", d);
    }
}
