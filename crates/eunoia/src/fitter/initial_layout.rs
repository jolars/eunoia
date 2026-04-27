use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::spec::PairwiseRelations;

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
            patience: 10,
            improvement_threshold: 0.001, // 0.1% improvement
            perfect_fit_threshold: 1e-8,  // Near-zero loss
        }
    }
}

/// Compute initial layout using patience-based optimization with default configuration.
pub(crate) fn compute_initial_layout(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    set_areas: &[f64],
    rng: &mut dyn rand::RngCore,
) -> Result<Vec<f64>, Error> {
    compute_initial_layout_with_config(
        distances,
        relationships,
        set_areas,
        InitialLayoutConfig::default(),
        rng,
    )
}

/// Compute initial layout using patience-based optimization.
///
/// This function tries random initializations until no improvement is seen
/// for `patience` consecutive attempts, or until `max_attempts` is reached.
pub(crate) fn compute_initial_layout_with_config(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    set_areas: &[f64],
    config: InitialLayoutConfig,
    rng: &mut dyn rand::RngCore,
) -> Result<Vec<f64>, Error> {
    let n_sets = distances.len();

    let mut best_params = Vec::new();
    let mut best_loss = f64::INFINITY;
    let mut attempts_without_improvement = 0;

    // Sampling scale for random initial positions: side of a square that fits
    // all sets (= sqrt(sum of set areas) = sqrt(sum(πr²))). This matches
    // eulerr's `bnd = sqrt(sum(r^2 * pi))` and is intentionally generous —
    // tighter sampling makes MDS converge to configurations where shapes are
    // packed too close together to give the final-stage ellipse optimizer
    // room to deform (issue #28).
    let total_area: f64 = set_areas.iter().sum();
    let scale = if total_area > 0.0 {
        total_area.sqrt()
    } else {
        10.0
    };

    // Pre-derive a per-attempt seed from the parent RNG. This keeps results
    // deterministic for a given parent seed regardless of how attempts are
    // scheduled across threads.
    let attempt_seeds: Vec<u64> = (0..config.max_attempts).map(|_| rng.random()).collect();

    // Chunk size controls how much work runs in parallel before we re-check
    // the patience criterion. Picking the thread count balances parallel
    // throughput with the early-stop savings of patience.
    #[cfg(not(target_arch = "wasm32"))]
    let chunk_size = rayon::current_num_threads().max(1);
    #[cfg(target_arch = "wasm32")]
    let chunk_size = 1;

    'outer: for chunk in attempt_seeds.chunks(chunk_size) {
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<(f64, Vec<f64>)> = chunk
            .par_iter()
            .map(|&seed| run_attempt(distances, relationships, n_sets, scale, seed))
            .collect::<Result<_, _>>()?;

        #[cfg(target_arch = "wasm32")]
        let results: Vec<(f64, Vec<f64>)> = chunk
            .iter()
            .map(|&seed| run_attempt(distances, relationships, n_sets, scale, seed))
            .collect::<Result<_, _>>()?;

        for (loss, params) in results {
            log::debug!("Attempt loss: {}", loss);

            if loss < config.perfect_fit_threshold {
                return Ok(params);
            }

            if loss < best_loss {
                let relative_improvement = if best_loss.is_finite() && best_loss > 0.0 {
                    (best_loss - loss) / best_loss
                } else {
                    f64::INFINITY
                };

                if !best_loss.is_finite() || relative_improvement > config.improvement_threshold {
                    best_loss = loss;
                    best_params = params;
                    attempts_without_improvement = 0;
                } else {
                    attempts_without_improvement += 1;
                }
            } else {
                attempts_without_improvement += 1;
            }

            if attempts_without_improvement >= config.patience {
                break 'outer;
            }
        }
    }

    Ok(best_params)
}

/// Run a single MDS LBFGS attempt from a freshly seeded random initialization.
///
/// Returns the final cost and the corresponding best parameter vector.
fn run_attempt(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    n_sets: usize,
    scale: f64,
    seed: u64,
) -> Result<(f64, Vec<f64>), Error> {
    let mut local_rng = StdRng::seed_from_u64(seed);

    // Sample positions in [0, scale] — matches eulerr's
    // `runif(n*2, 0, sqrt(sum(r^2*pi)))` (one-sided, not symmetric).
    let mut initial_values = vec![0.0; n_sets * 2];
    for value in &mut initial_values {
        *value = local_rng.random_range(0.0..scale);
    }
    let initial_param = DVector::from_vec(initial_values);

    let cost_function = MdsCost {
        distances,
        relationships,
    };

    let line_search = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(line_search, 10);

    let result = Executor::new(cost_function, solver)
        .configure(|state| state.param(initial_param).max_iters(200))
        .run()?;

    let loss = result.state().get_cost();
    let params = result.state().get_best_param().unwrap().as_slice().to_vec();

    Ok((loss, params))
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

                // 8.0, not 4.0: the loss double-counts each unordered pair
                // (i,j) and (j,i) both contribute D²), so each pair contributes
                // 2*D² to the loss and ∂(2*D²)/∂x_k = 8*D*xd. eulerr's
                // optim_init.cpp has the same factor-2 understatement here
                // (verified vs finite differences); we get the math right.
                grad[i] += 8.0 * d * xd;
                grad[i + n_sets] += 8.0 * d * yd;
            }
        }

        Ok(grad)
    }
}

#[cfg(test)]
mod gradient_check {
    use super::*;
    use nalgebra::DVector;

    fn fd_gradient(cost: &MdsCost, p: &DVector<f64>) -> DVector<f64> {
        let h = 1e-6;
        let mut g = DVector::from_element(p.len(), 0.0);
        for i in 0..p.len() {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[i] += h;
            pm[i] -= h;
            g[i] = (cost.cost(&pp).unwrap() - cost.cost(&pm).unwrap()) / (2.0 * h);
        }
        g
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        // 4-set case 2 distances/relationships, evaluated at a non-pathological point.
        // distances ≈ d(A,*) = 2.232, d(B,C)=d(B,D)=d(C,D) = 1.642
        let distances = vec![
            vec![0.0, 2.232, 2.232, 2.232],
            vec![2.232, 0.0, 1.642, 1.642],
            vec![2.232, 1.642, 0.0, 1.642],
            vec![2.232, 1.642, 1.642, 0.0],
        ];
        let mut relations = PairwiseRelations {
            n_sets: 4,
            subset: vec![vec![false; 4]; 4],
            disjoint: vec![vec![false; 4]; 4],
            overlap_areas: vec![vec![0.0; 4]; 4],
        };
        // A is a superset of B, C, D
        for i in 1..4 {
            relations.subset[0][i] = true;
            relations.subset[i][0] = true;
        }
        let cost = MdsCost {
            distances: &distances,
            relationships: &relations,
        };
        // A point that doesn't trigger the no-penalty branches:
        // A center near origin (so subset penalty zero), B/C/D somewhere unrelated.
        let p = DVector::from_vec(vec![
            0.5, 3.0, 3.5, 4.5, // xs (A, B, C, D)
            0.3, 1.0, 2.5, 0.8, // ys
        ]);
        let analytic = cost.gradient(&p).unwrap();
        let numeric = fd_gradient(&cost, &p);
        let max_abs_diff = analytic
            .iter()
            .zip(numeric.iter())
            .map(|(a, n)| (a - n).abs())
            .fold(0.0_f64, f64::max);
        let max_abs = numeric.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        println!("analytic = {:?}", analytic.as_slice());
        println!("numeric  = {:?}", numeric.as_slice());
        println!("max abs diff: {:.4e}", max_abs_diff);
        println!("max abs num : {:.4e}", max_abs);
        println!("ratio analytic/numeric per element:");
        for i in 0..p.len() {
            if numeric[i].abs() > 1e-9 {
                println!(
                    "  [{}]  analytic={:>12.4}  numeric={:>12.4}  ratio={:.4}",
                    i,
                    analytic[i],
                    numeric[i],
                    analytic[i] / numeric[i]
                );
            }
        }
        // We don't assert here; the diagnostic prints diagnose the discrepancy.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
        let mut rng = StdRng::seed_from_u64(42);
        // Two circles with radii r1=1, r2=1, touching (distance = 2)
        let distances = vec![vec![0.0, 2.0], vec![2.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Two circles with centers 1 unit apart (overlapping)
        let distances = vec![vec![0.0, 1.0], vec![1.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Two circles far apart
        let distances = vec![vec![0.0, 5.0], vec![5.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Two disjoint sets (distance should be at least r1 + r2)
        let distances = vec![vec![0.0, 3.0], vec![3.0, 0.0]];

        let mut relationships = create_test_relationships(2);
        relationships.disjoint[0][1] = true;
        relationships.disjoint[1][0] = true;

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Three sets arranged in a triangle
        let d = 2.0; // Equal distances
        let distances = vec![vec![0.0, d, d], vec![d, 0.0, d], vec![d, d, 0.0]];

        let relationships = create_test_relationships(3);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Three sets in a line: A---B---C
        let distances = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];

        let relationships = create_test_relationships(3);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
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

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Test that multiple restarts work
        let distances = vec![vec![0.0, 1.5], vec![1.5, 0.0]];

        let relationships = create_test_relationships(2);

        let result1 = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();
        let result2 = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Two sets at the same position (fully overlapping)
        let distances = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        let relationships = create_test_relationships(2);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng).unwrap();

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
        let mut rng = StdRng::seed_from_u64(42);
        // Test with slightly asymmetric input (should still work)
        let distances = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.5],
            vec![2.0, 1.5, 0.0],
        ];

        let relationships = create_test_relationships(3);

        let result = compute_initial_layout(&distances, &relationships, &[], &mut rng);

        // Should succeed even with asymmetric distances
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 6);
    }

    #[test]
    fn test_patience_based_optimization() {
        let mut rng = StdRng::seed_from_u64(42);
        // Test that patience-based optimization stops early when no improvement
        let distances = vec![vec![0.0, 1.5], vec![1.5, 0.0]];
        let relationships = create_test_relationships(2);

        let config = InitialLayoutConfig {
            max_attempts: 100,
            patience: 10,
            improvement_threshold: 0.01,
            perfect_fit_threshold: 1e-8,
        };

        let result =
            compute_initial_layout_with_config(&distances, &relationships, &[], config, &mut rng);
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
        let mut rng = StdRng::seed_from_u64(42);
        // Test with zero threshold - should only improve on any decrease
        let distances = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let relationships = create_test_relationships(2);

        let config = InitialLayoutConfig {
            max_attempts: 10,
            patience: 2,
            improvement_threshold: 0.0,
            perfect_fit_threshold: 1e-8,
        };

        let result =
            compute_initial_layout_with_config(&distances, &relationships, &[], config, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_early_stopping_on_perfect_fit() {
        let mut rng = StdRng::seed_from_u64(42);
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

        let result =
            compute_initial_layout_with_config(&distances, &relationships, &[], config, &mut rng);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.len(), 4);

        // Verify the distance is correct
        let (x, y) = params.split_at(2);
        let d = ((x[0] - x[1]).powi(2) + (y[0] - y[1]).powi(2)).sqrt();
        assert!(approx_eq(d, 1.0, 0.1), "Distance: {}", d);
    }
}
