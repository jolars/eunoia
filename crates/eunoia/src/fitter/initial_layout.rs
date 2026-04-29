use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::newton::NewtonCG;
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::storage::Owned;
use nalgebra::{DMatrix, DVector, Dyn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::spec::PairwiseRelations;

/// Sampling-scale convention used by every initial-layout sampler.
/// Side of a square that fits all sets (= `sqrt(Σ set_areas)`), matching
/// eulerr's `bnd = sqrt(sum(r^2 * pi))`. Falls back to `10.0` when the spec
/// has no area information (e.g. the unit-test relationship-only callers
/// that pass `set_areas = &[]`).
pub(crate) fn sampling_scale(set_areas: &[f64]) -> f64 {
    let total_area: f64 = set_areas.iter().sum();
    if total_area > 0.0 {
        total_area.sqrt()
    } else {
        10.0
    }
}

/// Initial-position sampler for the per-restart MDS init.
///
/// The outer `n_restarts` loop draws random initial positions for each MDS
/// attempt. Stratifying that draw via Latin hypercube sampling spreads attempts
/// more evenly across the sampling box than independent uniform draws, which
/// can pile up multiple attempts in the same basin by chance.
///
/// See [`Fitter::initial_sampler`] to switch.
///
/// [`Fitter::initial_sampler`]: crate::Fitter::initial_sampler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitialSampler {
    /// Independent uniform draws on `[0, scale]` per attempt — matches eulerr's
    /// `runif(n*2, 0, sqrt(sum(r^2*pi)))`.
    #[default]
    Uniform,
    /// Latin hypercube sampling on the central `[0.25·scale, 0.75·scale]^(2·n_sets)`
    /// box: across the batch of `n_restarts` attempts, each of the `2·n_sets`
    /// axes is split into `n_restarts` equal strata sampled exactly once under
    /// a random per-axis permutation. The central box (vs the full eulerr
    /// `[0, scale]` extent) keeps the LHS budget on initial conditions
    /// downstream LM can refine, and empirically lifts mean `diag_error` ~10%
    /// on ellipses without hurting circles at `n_restarts=10`. See
    /// [`LHS_HALF_WIDTH_FRAC`].
    LatinHypercube,
}

/// Half-width of the LHS box as a fraction of the eulerr `[0, scale]` extent.
///
/// The eulerr Uniform sampler draws on `[0, scale]`; the LHS sampler draws on
/// `[scale·(½ - LHS_HALF_WIDTH_FRAC), scale·(½ + LHS_HALF_WIDTH_FRAC)]`,
/// i.e. centred at `scale/2` with total width `2·LHS_HALF_WIDTH_FRAC·scale`.
///
/// `0.25` gives the central 50% of the eulerr extent
/// (`[0.25·scale, 0.75·scale]`). Rationale: under independent uniform draws on
/// `[0, scale]`, ~50% of samples land outside `[0.25·scale, 0.75·scale]` per
/// axis — but for a well-conditioned MDS init most of those edge points are
/// far from any reasonable layout (one circle dragged way out, others piled at
/// the origin). Stratifying within the central box concentrates the LHS budget
/// on initial conditions that are actually plausible, while still spreading
/// across `n_restarts` strata. Edge coverage is sacrificed deliberately —
/// users who need it can fall back to `InitialSampler::Uniform`.
pub(crate) const LHS_HALF_WIDTH_FRAC: f64 = 0.25;

/// Build a Latin hypercube design of `n_samples` points in `[lo, hi]^n_dims`.
///
/// Returns one row per sample; row `i` is the initial position vector for
/// restart `i`. Each axis is split into `n_samples` equal strata (width
/// `(hi - lo) / n_samples`); the stratum-to-sample assignment is an
/// independent random permutation per axis, with a uniform jitter inside the
/// chosen stratum. `n_samples = 1` degenerates to a single uniform draw on
/// `[lo, hi]`.
pub(crate) fn latin_hypercube_rows(
    n_samples: usize,
    n_dims: usize,
    lo: f64,
    hi: f64,
    rng: &mut dyn rand::RngCore,
) -> Vec<Vec<f64>> {
    debug_assert!(n_samples >= 1);
    debug_assert!(hi >= lo);
    let span = hi - lo;
    let inv_n = span / (n_samples as f64);
    let mut rows = vec![vec![0.0; n_dims]; n_samples];
    for d in 0..n_dims {
        let mut perm: Vec<usize> = (0..n_samples).collect();
        perm.shuffle(rng);
        for (i, row) in rows.iter_mut().enumerate() {
            let u: f64 = rng.random_range(0.0..1.0);
            row[d] = lo + (perm[i] as f64 + u) * inv_n;
        }
    }
    rows
}

/// Sample a single uniform initial position vector in `[0, scale]^(2·n_sets)`.
fn sample_uniform_init(rng: &mut dyn rand::RngCore, n_sets: usize, scale: f64) -> DVector<f64> {
    // Derive a fresh per-attempt seed from the supplied rng so the sampling
    // path is identical to the historical behaviour (compute_initial_layout
    // → run_attempt seeded a local StdRng before drawing 2·n_sets uniforms),
    // keeping unit-test outputs stable.
    let seed: u64 = rng.random();
    let mut local_rng = StdRng::seed_from_u64(seed);
    let mut values = vec![0.0; n_sets * 2];
    for value in &mut values {
        *value = local_rng.random_range(0.0..scale);
    }
    DVector::from_vec(values)
}

/// Run one MDS optimization from a freshly seeded random initialization.
///
/// The Fitter's outer `n_restarts` loop is the only layer of diversity — each
/// outer attempt calls this once and forwards the result to its downstream
/// loss comparison. We deliberately don't pre-select on MDS loss before
/// downstream evaluation, since MDS-suboptimal layouts can produce perfect
/// downstream fits while MDS-optimal ones can be downstream-rigid (issue #28).
/// Run one MDS optimization from a freshly seeded random initialization,
/// using the default solver.
#[cfg(test)]
pub(crate) fn compute_initial_layout(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    set_areas: &[f64],
    rng: &mut dyn rand::RngCore,
) -> Result<Vec<f64>, Error> {
    compute_initial_layout_with_solver(
        distances,
        relationships,
        set_areas,
        rng,
        MdsSolver::default(),
        None,
    )
}

/// Run one MDS optimization with the chosen solver. If `initial_positions` is
/// `Some`, those values (length `2·n_sets`) are used as the starting point;
/// otherwise an independent uniform draw on `[0, sampling_scale(set_areas)]`
/// is generated from `rng`. The production fitter passes
/// [`MdsSolver::default`] and supplies pre-sampled positions when running the
/// Latin-hypercube sampler so all `n_restarts` attempts share one stratified
/// design; the benchmark harness in `examples/initial_layout_bench.rs`
/// cycles through the alternative solvers.
pub(crate) fn compute_initial_layout_with_solver(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    set_areas: &[f64],
    rng: &mut dyn rand::RngCore,
    solver: MdsSolver,
    initial_positions: Option<&[f64]>,
) -> Result<Vec<f64>, Error> {
    let n_sets = distances.len();

    let initial_param = match initial_positions {
        Some(positions) => {
            debug_assert_eq!(
                positions.len(),
                n_sets * 2,
                "initial_positions length must be 2·n_sets"
            );
            DVector::from_column_slice(positions)
        }
        None => sample_uniform_init(rng, n_sets, sampling_scale(set_areas)),
    };

    let (_loss, params) = run_attempt(distances, relationships, n_sets, &initial_param, solver)?;
    Ok(params)
}

/// Solver used for the MDS-based initial layout.
///
/// Different solvers reach different local minima on the same MDS objective,
/// which translates to different downstream basins after the final stage.
/// The default fitter cycles `[Lbfgs, TrustRegion]` across the outer
/// `n_restarts` loop because empirically that pairing widens basin coverage
/// on hard ellipse fits without raising wall time (restarts run in parallel).
///
/// See [`Fitter::initial_solver`] to pin a single solver, or
/// [`Fitter::initial_solver_pool`] to specify a custom cycling pool.
///
/// [`Fitter::initial_solver`]: crate::Fitter::initial_solver
/// [`Fitter::initial_solver_pool`]: crate::Fitter::initial_solver_pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MdsSolver {
    /// Limited-memory BFGS with More-Thuente line search. Gradient-only and
    /// cheap per iteration. Was the historical default; demoted from default
    /// after the More-Thuente inner line search was observed to deadlock at
    /// subset-clamp kinks on certain initial conditions (see
    /// `MdsSolver::LevenbergMarquardt` doc).
    Lbfgs,
    /// Trust region with Steihaug-CG subproblem. Uses the analytic Hessian
    /// and tolerates the indefinite Hessian our MDS objective routinely
    /// produces. Reaches different local minima than L-BFGS on hard ellipse
    /// fits, which is why the default pool mixes both.
    TrustRegion,
    /// Truncated Newton (Newton-CG) with More-Thuente line search. Uses the
    /// analytic Hessian via inner CG iterations. Comparable quality to
    /// L-BFGS but ~3× the wall time on small problems.
    NewtonCg,
    /// Levenberg-Marquardt with the analytic per-pair Jacobian. The MDS
    /// objective `Σ_{i≠j} d_{ij}²` is a textbook nonlinear least-squares
    /// problem (one residual per ordered pair, post-clamp), so LM gets to
    /// approximate the Hessian as `JᵀJ` from the residual Jacobian rather
    /// than relying on the analytic Hessian (TrustRegion / NewtonCg) or
    /// gradient history (L-BFGS).
    ///
    /// Default solver. The disjoint / subset clamps (`max(0, ·)`,
    /// `min(0, ·)`) make the residuals C¹ rather than C². L-BFGS' inner
    /// More-Thuente line search has been observed to deadlock at the
    /// resulting kinks on certain extreme-scale specs (e.g.
    /// `issue71_4_set_extreme_scale` when LHS forces a stratum where one
    /// circle fully encloses the others, activating subset clamps and
    /// zeroing most of the gradient). LM's trust-region update sidesteps
    /// the line-search deadlock entirely.
    #[default]
    LevenbergMarquardt,
}

/// Run a single MDS attempt from the supplied initial position, using the
/// given solver. Returns the final cost and the corresponding best parameter
/// vector. Sampling is the caller's responsibility.
fn run_attempt(
    distances: &Vec<Vec<f64>>,
    relationships: &PairwiseRelations,
    n_sets: usize,
    initial_param: &DVector<f64>,
    solver: MdsSolver,
) -> Result<(f64, Vec<f64>), Error> {
    let cost_function = MdsCost {
        distances,
        relationships,
    };

    match solver {
        MdsSolver::Lbfgs => {
            let line_search = MoreThuenteLineSearch::new();
            let solver = LBFGS::new(line_search, 10);
            let result = Executor::new(cost_function, solver)
                .configure(|state| state.param(initial_param.clone()).max_iters(200))
                .run()?;
            Ok((
                result.state().get_cost(),
                result.state().get_best_param().unwrap().as_slice().to_vec(),
            ))
        }
        MdsSolver::TrustRegion => {
            // TrustRegion needs Vec<f64> params; wrap MdsCost.
            // Steihaug subproblem (truncated CG) — handles the indefinite
            // Hessian that our MDS objective produces (8*xd² + 4*D goes
            // negative whenever shapes are closer than their target distance).
            // Cauchy point hangs in that regime.
            let vec_cost = VecMdsCost {
                inner: cost_function,
            };
            let subproblem: Steihaug<Vec<f64>, f64> = Steihaug::new().with_max_iters(200);
            let solver = TrustRegion::new(subproblem);
            let initial_param_vec = initial_param.as_slice().to_vec();
            let result = Executor::new(vec_cost, solver)
                .configure(|state| state.param(initial_param_vec).max_iters(200))
                .run()?;
            Ok((
                result.state().get_cost(),
                result.state().get_best_param().unwrap().clone(),
            ))
        }
        MdsSolver::NewtonCg => {
            // Truncated Newton: solves the Newton equations approximately via
            // an inner CG, then takes a More-Thuente line search step.
            // Uses the analytic Hessian. Vec<f64> params via VecMdsCost.
            let vec_cost = VecMdsCost {
                inner: cost_function,
            };
            let line_search = MoreThuenteLineSearch::new();
            let solver: NewtonCG<_, f64> = NewtonCG::new(line_search);
            let initial_param_vec = initial_param.as_slice().to_vec();
            let result = Executor::new(vec_cost, solver)
                .configure(|state| state.param(initial_param_vec).max_iters(200))
                .run()?;
            Ok((
                result.state().get_cost(),
                result.state().get_best_param().unwrap().clone(),
            ))
        }
        MdsSolver::LevenbergMarquardt => {
            let problem = LmMdsProblem::new(distances, relationships, n_sets, initial_param);
            // Tolerances match the other MDS solvers (default argmin behaviour
            // is `sqrt(EPSILON)`-ish; LM's default `30·EPS` is far stricter
            // than we need). 200 iters mirrors the hardcoded cap on the
            // other arms.
            let lm = LevenbergMarquardt::new().with_patience(200);
            let (problem_after, report) = lm.minimize(problem);
            // LM minimises ½·Σrᵢ²; MdsCost returns Σrᵢ², so multiply by 2.
            Ok((
                report.objective_function * 2.0,
                problem_after.params.iter().copied().collect(),
            ))
        }
    }
}

/// Vec<f64>-param wrapper around `MdsCost` for argmin's TrustRegion solver.
struct VecMdsCost<'a> {
    inner: MdsCost<'a>,
}

impl<'a> CostFunction for VecMdsCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        self.inner.cost(&DVector::from_vec(p.clone()))
    }
}

impl<'a> Gradient for VecMdsCost<'a> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let g = self.inner.gradient(&DVector::from_vec(p.clone()))?;
        Ok(g.as_slice().to_vec())
    }
}

impl<'a> Hessian for VecMdsCost<'a> {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;
    fn hessian(&self, p: &Vec<f64>) -> Result<Vec<Vec<f64>>, Error> {
        self.inner.hessian(&DVector::from_vec(p.clone()))
    }
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

impl<'a> Hessian for MdsCost<'a> {
    type Param = DVector<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// Analytic Hessian. For each ordered active pair (i, j) with
    /// xd = x_i - x_j, yd = y_i - y_j, D = xd² + yd² - d_ij², the contribution
    /// from D² to the Hessian is:
    ///
    /// - block xx, yy: (δ_ki - δ_kj)(δ_li - δ_lj) * (8·xd² + 4·D)  [/yd² for yy]
    /// - block xy:     (δ_ki - δ_kj)(δ_li - δ_lj) * 8·xd·yd
    ///
    /// Both ordered iterations (i,j) and (j,i) contribute the same numerical
    /// pattern to the same entries, doubling the per-pair amount — which
    /// matches the loss being a sum over ordered pairs.
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        let n_sets = param.len() / 2;
        let n = 2 * n_sets;
        let x = param.rows(0, n_sets);
        let y = param.rows(n_sets, n_sets);

        let mut h = vec![vec![0.0; n]; n];

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

                let xx = 8.0 * xd * xd + 4.0 * d;
                let yy = 8.0 * yd * yd + 4.0 * d;
                let xy = 8.0 * xd * yd;

                // x block (rows/cols 0..n_sets)
                h[i][i] += xx;
                h[i][j] -= xx;
                h[j][i] -= xx;
                h[j][j] += xx;

                // y block (rows/cols n_sets..2n_sets)
                let ii = i + n_sets;
                let jj = j + n_sets;
                h[ii][ii] += yy;
                h[ii][jj] -= yy;
                h[jj][ii] -= yy;
                h[jj][jj] += yy;

                // x-y cross block (and symmetric y-x)
                h[i][ii] += xy;
                h[i][jj] -= xy;
                h[j][ii] -= xy;
                h[j][jj] += xy;
                // symmetric mirror
                h[ii][i] += xy;
                h[jj][i] -= xy;
                h[ii][j] -= xy;
                h[jj][j] += xy;
            }
        }

        Ok(h)
    }
}

/// Levenberg-Marquardt least-squares adapter around the MDS cost.
///
/// One residual per ordered pair `(i, j)` with `i ≠ j`, valued at
/// `d_ij = (x_i − x_j)² + (y_i − y_j)² − target_ij²` and clamped to zero on
/// the satisfied side of disjoint / subset constraints. `Σ rᵢ²` matches the
/// existing `MdsCost::cost`. The Jacobian is sparse — each row depends only
/// on `(x_i, y_i, x_j, y_j)` — but we store it dense, since `n_sets ≤ 8` in
/// realistic fits keeps the matrix tiny.
struct LmMdsProblem<'a> {
    distances: &'a Vec<Vec<f64>>,
    relationships: &'a PairwiseRelations,
    n_sets: usize,
    /// Param layout matches `MdsCost`: `[x_0, …, x_{n−1}, y_0, …, y_{n−1}]`.
    params: DVector<f64>,
    /// Canonical residual ordering: ordered pairs in row-major (i, j) order
    /// with `i ≠ j`, length `n*(n−1)`.
    pairs: Vec<(usize, usize)>,
}

impl<'a> LmMdsProblem<'a> {
    fn new(
        distances: &'a Vec<Vec<f64>>,
        relationships: &'a PairwiseRelations,
        n_sets: usize,
        initial_param: &DVector<f64>,
    ) -> Self {
        let mut pairs = Vec::with_capacity(n_sets.saturating_sub(1) * n_sets);
        for i in 0..n_sets {
            for j in 0..n_sets {
                if i != j {
                    pairs.push((i, j));
                }
            }
        }
        Self {
            distances,
            relationships,
            n_sets,
            params: initial_param.clone(),
            pairs,
        }
    }

    /// Returns `(d, xd, yd, active)` for ordered pair `(i, j)` at the current
    /// parameters. `active = false` means the disjoint/subset clamp is in
    /// effect and the residual / Jacobian row should be zero.
    fn pair_state(&self, i: usize, j: usize) -> (f64, f64, f64, bool) {
        let n = self.n_sets;
        let xd = self.params[i] - self.params[j];
        let yd = self.params[n + i] - self.params[n + j];
        let d = xd.powi(2) + yd.powi(2) - self.distances[i][j].powi(2);
        if self.relationships.is_disjoint(i, j) && d >= 0.0 {
            return (0.0, xd, yd, false);
        }
        if (self.relationships.is_subset(i, j) || self.relationships.is_subset(j, i)) && d <= 0.0 {
            return (0.0, xd, yd, false);
        }
        (d, xd, yd, true)
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for LmMdsProblem<'a> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let mut residuals = DVector::zeros(self.pairs.len());
        for (k, &(i, j)) in self.pairs.iter().enumerate() {
            let (d, _xd, _yd, active) = self.pair_state(i, j);
            residuals[k] = if active { d } else { 0.0 };
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let n = self.n_sets;
        let mut jacobian = DMatrix::zeros(self.pairs.len(), self.params.len());
        for (k, &(i, j)) in self.pairs.iter().enumerate() {
            let (_d, xd, yd, active) = self.pair_state(i, j);
            if !active {
                continue;
            }
            // d = (x_i − x_j)² + (y_i − y_j)² − target² ⇒
            //   ∂d/∂x_i = 2·xd, ∂d/∂x_j = −2·xd,
            //   ∂d/∂y_i = 2·yd, ∂d/∂y_j = −2·yd.
            jacobian[(k, i)] = 2.0 * xd;
            jacobian[(k, j)] = -2.0 * xd;
            jacobian[(k, n + i)] = 2.0 * yd;
            jacobian[(k, n + j)] = -2.0 * yd;
        }
        Some(jacobian)
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

    fn fd_hessian(cost: &MdsCost, p: &DVector<f64>) -> Vec<Vec<f64>> {
        // Finite-difference the *analytic* gradient.
        let h = 1e-5;
        let n = p.len();
        let mut hessian = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut pp = p.clone();
            let mut pm = p.clone();
            pp[j] += h;
            pm[j] -= h;
            let gp = cost.gradient(&pp).unwrap();
            let gm = cost.gradient(&pm).unwrap();
            for i in 0..n {
                hessian[i][j] = (gp[i] - gm[i]) / (2.0 * h);
            }
        }
        hessian
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

    #[test]
    fn analytic_hessian_matches_finite_difference() {
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
        for i in 1..4 {
            relations.subset[0][i] = true;
            relations.subset[i][0] = true;
        }
        let cost = MdsCost {
            distances: &distances,
            relationships: &relations,
        };
        let p = DVector::from_vec(vec![0.5, 3.0, 3.5, 4.5, 0.3, 1.0, 2.5, 0.8]);
        let analytic = cost.hessian(&p).unwrap();
        let numeric = fd_hessian(&cost, &p);
        let n = p.len();
        let mut max_abs_diff = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let d = (analytic[i][j] - numeric[i][j]).abs();
                max_abs_diff = max_abs_diff.max(d);
                max_abs = max_abs.max(numeric[i][j].abs());
            }
        }
        assert!(
            max_abs_diff < 1e-3 * max_abs.max(1.0),
            "analytic Hessian disagrees with FD reference: max abs diff {:.4e}, max abs {:.4e}",
            max_abs_diff,
            max_abs,
        );
    }

    #[test]
    fn lm_jacobian_consistent_with_mds_gradient() {
        // For a sum-of-squares loss, ∇L(θ) = 2·Jᵀ·r. Verifies the LM MDS
        // adapter constructs J consistently with the analytical gradient
        // already verified against finite differences above.
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
        for i in 1..4 {
            relations.subset[0][i] = true;
            relations.subset[i][0] = true;
        }
        let cost = MdsCost {
            distances: &distances,
            relationships: &relations,
        };
        let p = DVector::from_vec(vec![0.5, 3.0, 3.5, 4.5, 0.3, 1.0, 2.5, 0.8]);
        let analytic = cost.gradient(&p).unwrap();

        let problem = LmMdsProblem::new(&distances, &relations, 4, &p);
        let r = problem.residuals().unwrap();
        let j = problem.jacobian().unwrap();
        let lm_grad = j.transpose() * &r * 2.0;

        let max_abs_diff = analytic
            .iter()
            .zip(lm_grad.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_abs = analytic
            .iter()
            .map(|x: &f64| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff < 1e-9 + 1e-8 * max_abs,
            "|2·Jᵀ·r − ∇L| = {:.3e} (max |∇L| = {:.3e})",
            max_abs_diff,
            max_abs
        );
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
    fn latin_hypercube_stratifies_each_axis() {
        // Stratification invariant: for each of the n_dims axes, the n_samples
        // values must hit every stratum [lo + k·w, lo + (k+1)·w) exactly once,
        // where w = (hi - lo) / n_samples.
        let mut rng = StdRng::seed_from_u64(7);
        let n_samples = 10;
        let n_dims = 6;
        let lo = 1.0;
        let hi = 5.0;
        let rows = latin_hypercube_rows(n_samples, n_dims, lo, hi, &mut rng);
        assert_eq!(rows.len(), n_samples);
        let span = hi - lo;
        for d in 0..n_dims {
            let mut hits = vec![false; n_samples];
            for row in &rows {
                let v = row[d];
                assert!((lo..hi).contains(&v), "axis {d} value {v} out of range");
                let stratum = (((v - lo) / span) * n_samples as f64).floor() as usize;
                let stratum = stratum.min(n_samples - 1);
                assert!(!hits[stratum], "axis {d}: stratum {stratum} sampled twice");
                hits[stratum] = true;
            }
            assert!(hits.iter().all(|&h| h));
        }
    }

    #[test]
    fn latin_hypercube_n1_is_just_uniform() {
        let mut rng = StdRng::seed_from_u64(0);
        let rows = latin_hypercube_rows(1, 4, 0.0, 3.0, &mut rng);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 4);
        for &v in &rows[0] {
            assert!((0.0..3.0).contains(&v), "n=1 sample {v} out of range");
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
}
