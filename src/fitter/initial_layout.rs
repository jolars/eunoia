use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::DVector;

use crate::diagram::PairwiseRelations;

pub(crate) fn compute_initial_layout(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    n_restarts: usize,
) -> Result<Vec<f64>, Error> {
    let n_sets = distances.len();

    let mut best_params = Vec::new();
    let mut best_loss = f64::INFINITY;

    for _ in 0..n_restarts {
        let initial_param = DVector::from_element(n_sets * 2, 0.0);

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

        if loss < best_loss {
            best_loss = loss;
            best_params = result.state().get_best_param().unwrap().as_slice().to_vec();
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
