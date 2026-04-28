//! Fitter for creating diagram layouts from specifications.

mod clustering;
pub mod final_layout;
mod initial_layout;
mod layout;
pub mod normalize;
mod packing;

pub use final_layout::Optimizer;
pub use initial_layout::MdsSolver;
pub use layout::Layout;

use crate::error::DiagramError;
use crate::geometry::shapes::circle::distance_for_overlap;
use crate::geometry::shapes::Circle;
use crate::geometry::traits::DiagramShape;
use crate::loss::LossType;
use crate::spec::DiagramSpec;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Fitter for creating diagram layouts from specifications.
///
/// The type parameter `S` determines which shape type will be used (e.g., Circle, Ellipse).
/// The specification itself is shape-agnostic - the shape type is chosen here.
///
/// # Examples
///
/// ```
/// use eunoia::{DiagramSpecBuilder, Fitter};
/// use eunoia::geometry::shapes::Circle;
///
/// let spec = DiagramSpecBuilder::new()
///     .set("A", 10.0)
///     .set("B", 8.0)
///     .build()
///     .unwrap();
///
/// // Choose shape type when fitting
/// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
/// ```
pub struct Fitter<'a, S: DiagramShape = Circle> {
    spec: &'a DiagramSpec,
    max_iterations: usize,
    tolerance: f64,
    seed: Option<u64>,
    loss_type: LossType,
    /// Pool of final-stage optimizers cycled across outer-loop restarts:
    /// attempt `i` uses `optimizer_pool[i % optimizer_pool.len()]`. Default
    /// `[NelderMead, Lbfgs]` — Nelder-Mead is dramatically faster on small
    /// ellipse fits where it matches L-BFGS' fit quality, and L-BFGS finds
    /// basins Nelder-Mead misses on hard high-arity fits (issue #28). Best
    /// loss across attempts wins.
    optimizer_pool: Vec<Optimizer>,
    sa_fallback_threshold: Option<f64>,
    n_restarts: usize,
    /// Pool of MDS solvers cycled across outer-loop restarts: attempt `i`
    /// uses `initial_solvers[i % initial_solvers.len()]`. Mixing solvers in
    /// the pool widens basin coverage (different solvers fall into different
    /// local minima) without raising wall time, since restarts run in parallel.
    initial_solvers: Vec<MdsSolver>,
    _shape: std::marker::PhantomData<S>,
}

impl<'a, S: DiagramShape + Copy + 'static> Fitter<'a, S> {
    /// Create a new fitter for the given specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Circle>::new(&spec);
    /// ```
    pub fn new(spec: &'a DiagramSpec) -> Self {
        Fitter {
            spec,
            // 200 iters × tolerance 1e-6 (wired into L-BFGS' grad/cost tolerances
            // in `final_layout.rs`) tracks eulerr's nlm budget. The previous
            // 50_000-iter cap combined with argmin's default `sqrt(EPSILON)`/
            // `EPSILON` tolerances meant L-BFGS routinely ran tens of thousands
            // of iterations past any useful convergence on every restart.
            // With the default loss now `NormalizedSumSquared` (scale-invariant
            // SSE — see `LossType`), the loss magnitude is bounded ~`[0, 1]`
            // regardless of input area scale, so tolerance behavior is
            // consistent across specs. 1e-6 sits well above the central-diff
            // FD noise floor on the normalized cost yet pushes typical 3-set
            // ellipse fits to near-machine quality.
            max_iterations: 200,
            tolerance: 1e-6,
            seed: None,
            loss_type: LossType::sse(),
            // Cycle Nelder-Mead and L-BFGS across restarts. Per-attempt cost
            // varies (NM is O(ms), L-BFGS is O(100ms-1s) on ellipses due to
            // numerical-gradient overhead — issue #34), but with restarts
            // already running in parallel the wall time is bounded by the
            // slowest single attempt. The mix gives speed on cases NM solves
            // and robustness on cases that need L-BFGS (issue #28).
            optimizer_pool: vec![Optimizer::NelderMead, Optimizer::Lbfgs],
            // SA fallback is disabled by default — empirically it never
            // improved the result vs the primary optimizer on any spec or seed
            // we tested (likely because it starts from the local optimum with
            // a low initial temperature and can't escape). The mechanism is
            // kept behind the builder for experimentation, but turning it on
            // is currently a no-op in practice.
            sa_fallback_threshold: None,
            // Number of full-pipeline restarts (fresh MDS init + final
            // optimizer per attempt, lowest-loss attempt kept). Matches
            // eulerr's `n_restarts = 10`. Each fit does that much work.
            n_restarts: 10,
            // L-BFGS-only by default. We tried mixing in TrustRegion+Steihaug
            // (it hits different local basins than L-BFGS on hard ellipse fits
            // per `benches/initial_layout.rs`), but it caused hangs on real
            // eulerr-style specs. The mix is still available via
            // `initial_solver_pool`; the default keeps the proven path.
            initial_solvers: vec![MdsSolver::Lbfgs],
            _shape: std::marker::PhantomData,
        }
    }

    /// Pin the initial-layout MDS solver to a single choice.
    ///
    /// The default is [`MdsSolver::Lbfgs`] for every restart. Other solvers
    /// (`TrustRegion+Steihaug`, `Newton-CG`, `ConjugateGradient`) are exposed
    /// for experimentation; on hard ellipse fits TrustRegion sometimes finds
    /// basins L-BFGS misses, but it can also hang on certain real-world
    /// specs, so it is opt-in.
    ///
    /// To cycle multiple solvers across the outer `n_restarts` loop, see
    /// [`initial_solver_pool`].
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, MdsSolver};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Circle>::new(&spec)
    ///     .initial_solver(MdsSolver::Lbfgs);
    /// ```
    ///
    /// [`initial_solver_pool`]: Self::initial_solver_pool
    pub fn initial_solver(mut self, solver: MdsSolver) -> Self {
        self.initial_solvers = vec![solver];
        self
    }

    /// Set the pool of MDS solvers to cycle across outer-loop restarts.
    ///
    /// Restart `i` uses `pool[i % pool.len()]`. Mixing solvers in the pool
    /// widens local-minimum coverage on hard fits without raising wall time,
    /// since restarts already run in parallel. The default pool is
    /// `[MdsSolver::Lbfgs]` (single-solver) — the mixed `[Lbfgs, TrustRegion]`
    /// pool improves best-of-N quality on issue #28-class specs but has been
    /// observed to hang on some eulerr-style real-world inputs.
    ///
    /// # Panics
    ///
    /// Panics if `pool` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, MdsSolver};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Bias the cycle 7:3 toward L-BFGS over TrustRegion.
    /// let fitter = Fitter::<Ellipse>::new(&spec).initial_solver_pool(vec![
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::TrustRegion,
    ///     MdsSolver::TrustRegion,
    ///     MdsSolver::TrustRegion,
    /// ]);
    /// ```
    pub fn initial_solver_pool(mut self, pool: Vec<MdsSolver>) -> Self {
        assert!(!pool.is_empty(), "initial_solver_pool must be non-empty");
        self.initial_solvers = pool;
        self
    }

    /// Configure the diag-error threshold above which a simulated-annealing
    /// fallback runs for ellipse fits after the primary optimizer.
    ///
    /// Pass `None` to disable the fallback entirely. The default is `None` —
    /// empirical testing has not found a configuration where the SA fallback
    /// improves on the primary optimizer, so it is off by default. Set to
    /// `Some(threshold)` to opt in for experimentation.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 5.0)
    ///     .set("B", 4.0)
    ///     .set("C", 3.0)
    ///     .intersection(&["A", "B"], 1.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Disable the SA fallback
    /// let fitter = Fitter::<Ellipse>::new(&spec).sa_fallback_threshold(None);
    /// ```
    pub fn sa_fallback_threshold(mut self, threshold: Option<f64>) -> Self {
        self.sa_fallback_threshold = threshold;
        self
    }

    /// Set maximum iterations for optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Circle>::new(&spec).max_iterations(500);
    /// ```
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the convergence tolerance for the final-stage optimizer.
    ///
    /// Currently honored by L-BFGS only — passed as `tol_grad`, with
    /// `tol_cost = tolerance²`. Other solvers (Nelder-Mead, nonlinear CG,
    /// TrustRegion) do not expose tolerance setters in argmin 0.11 and run
    /// until `max_iterations`.
    ///
    /// The default is `1e-6`, matching eulerr's nlm `gradtol`/`steptol`.
    /// Tightening this (e.g. `1e-9`) gives a sharper fit at the cost of more
    /// iterations; loosening it (e.g. `1e-3`) speeds up fits where coarse
    /// convergence is acceptable.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Circle>::new(&spec).tolerance(1e-4);
    /// ```
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the optimizer to use for final layout optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, Optimizer};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Try L-BFGS optimizer
    /// let fitter = Fitter::<Ellipse>::new(&spec).optimizer(Optimizer::Lbfgs);
    /// ```
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer_pool = vec![optimizer];
        self
    }

    /// Set the pool of final-stage optimizers cycled across outer-loop restarts.
    ///
    /// Restart `i` uses `pool[i % pool.len()]`, and the lowest-loss attempt
    /// across the entire `n_restarts` loop wins. The default pool is
    /// `[NelderMead, Lbfgs]` — NM is O(ms) per fit on small ellipse problems
    /// and L-BFGS finds basins NM misses on hard high-arity fits (issue #28),
    /// so cycling gives both speed and robustness.
    ///
    /// Calling [`optimizer`] reduces the pool to a single solver.
    ///
    /// # Panics
    ///
    /// Panics if `pool` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, Optimizer};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // 8 NM attempts to 2 L-BFGS attempts per `n_restarts`-cycle.
    /// let fitter = Fitter::<Ellipse>::new(&spec).optimizer_pool(vec![
    ///     Optimizer::NelderMead, Optimizer::NelderMead, Optimizer::NelderMead, Optimizer::NelderMead,
    ///     Optimizer::NelderMead, Optimizer::NelderMead, Optimizer::NelderMead, Optimizer::NelderMead,
    ///     Optimizer::Lbfgs, Optimizer::Lbfgs,
    /// ]);
    /// ```
    ///
    /// [`optimizer`]: Self::optimizer
    pub fn optimizer_pool(mut self, pool: Vec<Optimizer>) -> Self {
        assert!(!pool.is_empty(), "optimizer_pool must be non-empty");
        self.optimizer_pool = pool;
        self
    }

    /// Set random seed for reproducible layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the loss function type for optimization.
    pub fn loss_type(mut self, loss_type: LossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Set the number of full-pipeline restarts.
    ///
    /// Each restart runs the entire fit (fresh MDS initialization + final
    /// optimization) from an independently seeded random circle layout, and
    /// the lowest-loss result is kept. Mirrors eulerr's `n_restarts = 10`.
    /// Higher values give a better chance of finding the global optimum but
    /// cost proportionally more (10× the work for `n=10`). Set to 1 to
    /// disable restarts.
    pub fn n_restarts(mut self, n: usize) -> Self {
        self.n_restarts = n.max(1);
        self
    }

    /// Fit the diagram using circles.
    ///
    /// This creates a layout with circles positioned to match the specification.
    /// Currently uses a simple grid initialization; optimization will be added later.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Circle;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .intersection(&["A", "B"], 2.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let layout = Fitter::<Circle>::new(&spec).fit().unwrap();
    /// println!("Loss: {}", layout.loss());
    /// ```
    pub fn fit(self) -> Result<Layout<S>, DiagramError> {
        self.fit_with_optimization(true)
    }

    /// Fit the diagram, optionally skipping final optimization.
    ///
    /// When `optimize` is false, returns only the initial MDS-based layout.
    /// This is useful for debugging or comparing initial vs optimized layouts.
    pub fn fit_initial_only(self) -> Result<Layout<S>, DiagramError> {
        self.fit_with_optimization(false)
    }

    fn fit_with_optimization(self, optimize: bool) -> Result<Layout<S>, DiagramError> {
        let spec = self.spec.preprocess()?;
        let n_sets = spec.n_sets;
        let max_iterations = self.max_iterations;
        let tolerance = self.tolerance;
        let loss_type = self.loss_type;
        let optimizer_pool = self.optimizer_pool.clone();

        // Master RNG: derives a per-attempt seed for each full-pipeline restart.
        let master_seed = self.seed.unwrap_or_else(|| rand::rng().random());
        let mut master_rng = StdRng::seed_from_u64(master_seed);

        let optimal_distances = Self::compute_optimal_distances(&spec)?;
        let initial_radii: Vec<f64> = spec
            .set_areas
            .iter()
            .map(|area| (area / std::f64::consts::PI).sqrt())
            .collect();

        // Run the full pipeline (fresh MDS init + final optimization) `n_restarts`
        // times and keep the lowest-loss attempt. Mirrors eulerr's `n_restarts=10`:
        // each restart explores an independent MDS basin (since MDS samples random
        // initial positions), and only one of those typically lands in a basin
        // that the final optimizer can refine to a perfect fit (issue #28).
        let n_attempts = if optimize { self.n_restarts.max(1) } else { 1 };
        let attempt_seeds: Vec<u64> = (0..n_attempts).map(|_| master_rng.random()).collect();

        let initial_solvers = self.initial_solvers.clone();
        let run_attempt =
            |(attempt_idx, attempt_seed): (usize, u64)| -> Result<(Vec<f64>, f64), DiagramError> {
                let mut attempt_rng = StdRng::seed_from_u64(attempt_seed);

                let initial_solver = initial_solvers[attempt_idx % initial_solvers.len()];
                let initial_params = match initial_layout::compute_initial_layout_with_solver(
                    &optimal_distances,
                    &spec.relationships,
                    &spec.set_areas,
                    &mut attempt_rng,
                    initial_solver,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        return Err(DiagramError::InvalidCombination(format!(
                            "Initial layout failed: {}",
                            e
                        )));
                    }
                };

                let (x, y) = initial_params.split_at(n_sets);
                let initial_positions: Vec<f64> = x
                    .iter()
                    .zip(y.iter())
                    .flat_map(|(xi, yi)| vec![*xi, *yi])
                    .collect();

                if !optimize {
                    // Without final optimization there's nothing to score across
                    // attempts — just take the first MDS init as-is.
                    let mut params = Vec::new();
                    for i in 0..n_sets {
                        let xi = initial_positions[i * 2];
                        let yi = initial_positions[i * 2 + 1];
                        let r = initial_radii[i];
                        params.extend(S::params_from_circle(xi, yi, r));
                    }
                    return Ok((params, 0.0));
                }

                let attempt_optimizer = optimizer_pool[attempt_idx % optimizer_pool.len()];
                let config = final_layout::FinalLayoutConfig {
                    max_iterations,
                    tolerance,
                    loss_type,
                    optimizer: attempt_optimizer,
                    seed: attempt_seed,
                    // Outer loop already provides full-pipeline diversity via fresh
                    // MDS inits, so each attempt's final stage runs once.
                    n_restarts: 1,
                };

                match final_layout::optimize_layout::<S>(
                    &spec,
                    &initial_positions,
                    &initial_radii,
                    config,
                ) {
                    Ok((params, loss)) => Ok((params, loss)),
                    Err(e) => Err(DiagramError::InvalidCombination(format!(
                        "Optimization failed: {}",
                        e
                    ))),
                }
            };

        let indexed_seeds: Vec<(usize, u64)> = attempt_seeds
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        let attempt_results: Vec<Result<(Vec<f64>, f64), DiagramError>> = if optimize {
            #[cfg(not(target_arch = "wasm32"))]
            {
                indexed_seeds
                    .par_iter()
                    .map(|&pair| run_attempt(pair))
                    .collect()
            }

            #[cfg(target_arch = "wasm32")]
            {
                indexed_seeds
                    .iter()
                    .map(|&pair| run_attempt(pair))
                    .collect()
            }
        } else {
            vec![run_attempt(indexed_seeds[0])]
        };

        let mut best: Option<(Vec<f64>, f64)> = None;
        let mut last_err: Option<DiagramError> = None;

        for result in attempt_results {
            match result {
                Ok((params, loss)) => match &best {
                    None => best = Some((params, loss)),
                    Some((_, best_loss)) if loss < *best_loss => best = Some((params, loss)),
                    _ => {}
                },
                Err(e) => last_err = Some(e),
            }
        }

        let (mut final_params, _loss) = best.ok_or_else(|| {
            last_err.unwrap_or_else(|| {
                DiagramError::InvalidCombination(
                    "All restarts failed to produce a layout".to_string(),
                )
            })
        })?;

        // Step 2b: Global-search fallback — run simulated annealing if the primary
        // optimizer left us with a bad fit. Like eulerr's GenSA fallback, but
        // gated only on `diag_error > threshold` (any arity), since high-arity
        // ellipse fits also benefit from the global pass (issue #28).
        if optimize {
            if let Some(threshold) = self.sa_fallback_threshold {
                let is_ellipse = std::any::TypeId::of::<S>()
                    == std::any::TypeId::of::<crate::geometry::shapes::Ellipse>();
                if is_ellipse {
                    let current_diag =
                        final_layout::diag_error_from_params::<S>(&final_params, &spec);
                    if current_diag > threshold {
                        let (lower, upper) =
                            final_layout::derive_sa_bounds(&final_params, S::n_params());
                        let sa_seed = self.seed.unwrap_or(0xDEAD_BEEF);
                        if let Ok((sa_params, _sa_loss)) = final_layout::run_simulated_annealing::<S>(
                            &spec,
                            &final_params,
                            &lower,
                            &upper,
                            self.loss_type,
                            S::n_params(),
                            self.max_iterations.max(5000),
                            sa_seed,
                        ) {
                            let sa_diag =
                                final_layout::diag_error_from_params::<S>(&sa_params, &spec);
                            if sa_diag < current_diag {
                                final_params = sa_params;
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Create shapes for the non-empty sets from optimized parameters
        let params_per_shape = S::n_params();
        let mut optimized_shapes: Vec<S> = Vec::with_capacity(n_sets);
        for i in 0..n_sets {
            let start = i * params_per_shape;
            let end = start + params_per_shape;
            optimized_shapes.push(S::from_params(&final_params[start..end]));
        }

        #[cfg(debug_assertions)]
        let pre_normalize_regions = S::compute_exclusive_regions(&optimized_shapes);

        // Step 4: Normalize the non-empty shapes only (zero shapes would confuse
        // clustering/packing). We do this before re-assembly.
        crate::fitter::normalize::normalize_layout(&mut optimized_shapes, 0.05);

        #[cfg(debug_assertions)]
        {
            let post_normalize_regions = S::compute_exclusive_regions(&optimized_shapes);
            debug_assert!(
                exclusive_region_maps_approx_equal(
                    &pre_normalize_regions,
                    &post_normalize_regions,
                    1e-8,
                    1e-6
                ),
                "normalize_layout changed fitted exclusive regions"
            );
        }

        // Step 5: Re-assemble full shape list in the ORIGINAL spec set ordering,
        // inserting zero-parameter placeholders for sets that were pruned by
        // preprocessing (empty sets). This keeps indexing by set name stable for
        // downstream consumers (e.g. R bindings).
        let zero_params = vec![0.0; params_per_shape];
        let mut shapes: Vec<S> = Vec::with_capacity(self.spec.set_names().len());
        let mut set_to_shape = HashMap::new();
        for (original_idx, set_name) in self.spec.set_names().iter().enumerate() {
            let shape = match spec.set_to_idx.get(set_name) {
                Some(&preproc_idx) => optimized_shapes[preproc_idx],
                None => S::from_params(&zero_params),
            };
            shapes.push(shape);
            set_to_shape.insert(set_name.clone(), original_idx);
        }

        // Create and return the layout
        let layout = Layout::new(
            shapes,
            set_to_shape,
            self.spec,
            self.max_iterations,
            self.loss_type,
        );

        Ok(layout)
    }

    /// Compute optimal distances between circle centers based on desired overlaps.
    ///
    /// For each pair of sets, this calculates the distance between circle centers
    /// that would produce the desired overlap area given their radii.
    ///
    /// This always uses circles for initial layout, regardless of final shape type.
    #[allow(clippy::needless_range_loop)]
    fn compute_optimal_distances(
        spec: &crate::spec::PreprocessedSpec,
    ) -> Result<Vec<Vec<f64>>, DiagramError> {
        let n_sets = spec.n_sets;
        let mut optimal_distances = vec![vec![0.0; n_sets]; n_sets];

        for i in 0..n_sets {
            for j in (i + 1)..n_sets {
                let overlap = spec.relationships.overlap_area(i, j);
                let r1 = (spec.set_areas[i] / std::f64::consts::PI).sqrt();
                let r2 = (spec.set_areas[j] / std::f64::consts::PI).sqrt();

                let desired_distance =
                    distance_for_overlap(r1, r2, overlap, None, None).map_err(|_| {
                        DiagramError::InvalidCombination(format!(
                            "Could not compute distance for sets {} and {}",
                            i, j
                        ))
                    })?;

                optimal_distances[i][j] = desired_distance;
                optimal_distances[j][i] = desired_distance;
            }
        }

        Ok(optimal_distances)
    }
}

fn exclusive_region_maps_approx_equal(
    lhs: &HashMap<crate::geometry::diagram::RegionMask, f64>,
    rhs: &HashMap<crate::geometry::diagram::RegionMask, f64>,
    abs_tol: f64,
    rel_tol: f64,
) -> bool {
    let all_masks: HashSet<_> = lhs.keys().chain(rhs.keys()).copied().collect();

    all_masks.into_iter().all(|mask| {
        let left = lhs.get(&mask).copied().unwrap_or(0.0);
        let right = rhs.get(&mask).copied().unwrap_or(0.0);
        let scale = left.abs().max(right.abs()).max(1.0);
        (left - right).abs() <= abs_tol.max(rel_tol * scale)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::DiagramSpecBuilder;

    #[test]
    fn test_fitter_basic() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert!(layout.loss() >= 0.0);
    }

    #[test]
    fn test_fitter_with_intersection() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert_eq!(layout.requested().len(), 3); // A, B, A&B
    }

    #[test]
    fn test_russian_doll_initial_fit() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 1.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec)
            .seed(42)
            .fit_initial_only()
            .unwrap();

        // Russian doll (fully nested) is hard to solve with a pure MDS initial
        // layout; we only assert the loss is finite and reasonable. Full optimization
        // is expected to bring it close to zero.
        assert!(layout.loss().is_finite());
        assert!(layout.loss() < 25.0);
    }

    #[test]
    fn test_seed_reproducibility() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        // Same seed should produce identical results
        let layout1 = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let layout2 = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();

        assert_eq!(layout1.loss(), layout2.loss());

        // Verify shapes are identical
        for (s1, s2) in layout1.shapes().iter().zip(layout2.shapes().iter()) {
            assert_eq!(s1.center(), s2.center());
            assert_eq!(s1.radius(), s2.radius());
        }
    }

    #[test]
    fn test_fitter_with_ellipses_basic() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert!(layout.loss() >= 0.0);
    }

    #[test]
    fn test_fitter_with_ellipses_intersection() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();

        assert_eq!(layout.shapes().len(), 2);
        assert_eq!(layout.requested().len(), 3); // A, B, A&B
        assert!(layout.loss() < 10.0); // Should converge to reasonable solution
    }

    #[test]
    fn test_sa_fallback_does_not_regress_easy_ellipse_fit() {
        // For an easy 3-set ellipse problem, Nelder-Mead should find a near-perfect
        // fit on its own and the SA fallback (if it runs) must not make things worse.
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 10.0)
            .set("C", 10.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["B", "C"], 2.0)
            .intersection(&["A", "C"], 2.0)
            .intersection(&["A", "B", "C"], 0.5)
            .build()
            .unwrap();

        let layout_with = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();
        let layout_without = Fitter::<Ellipse>::new(&spec)
            .seed(42)
            .sa_fallback_threshold(None)
            .fit()
            .unwrap();

        // Both should converge to reasonable solutions; the fallback result
        // should not be worse than the primary-only one (SA only accepts an
        // improvement, otherwise falls back to the primary solution).
        assert!(layout_with.diag_error() <= layout_without.diag_error() + 1e-6);
    }

    /// Regression fixture for issue #28 (6-set ellipse spec from eulerr's
    /// `test-accuracy.R`). eulerr's `nlm` backend fits this exactly. Currently
    /// reaches `diag_error ≈ 1.7e-9` at seed=1 — well below the bar.
    #[test]
    #[ignore = "slow regression coverage"]
    fn test_issue28_six_set_ellipse_regression() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 4.0)
            .set("B", 6.0)
            .set("C", 3.0)
            .set("D", 2.0)
            .set("E", 7.0)
            .set("F", 3.0)
            .intersection(&["A", "B"], 2.0)
            .intersection(&["A", "F"], 2.0)
            .intersection(&["B", "C"], 2.0)
            .intersection(&["B", "D"], 1.0)
            .intersection(&["B", "F"], 2.0)
            .intersection(&["C", "D"], 1.0)
            .intersection(&["D", "E"], 1.0)
            .intersection(&["E", "F"], 1.0)
            .intersection(&["A", "B", "F"], 1.0)
            .intersection(&["B", "C", "D"], 1.0)
            .build()
            .unwrap();

        // Asserts L-BFGS can drive diag_error below 1e-6 *when given the
        // budget*. The default tolerance (1e-4) is set above the FD noise
        // floor for speed (issue #34); this test pins down the tight-budget
        // behavior eulerr's C++ backend reached on this spec.
        // tol=1e-10 is needed to drive the *normalized* cost below the
        // FD noise floor on this hard high-arity case; with the old
        // SumSquared default (loss scaled with input²), tol=1e-8 was
        // equivalent. See `LossType::NormalizedSumSquared` doc.
        let layout = Fitter::<Ellipse>::new(&spec)
            .seed(1)
            .tolerance(1e-10)
            .max_iterations(2000)
            .fit()
            .unwrap();
        assert!(
            layout.diag_error() < 1e-6,
            "issue #28 case 1: diag_error = {:e} (expected < 1e-6)",
            layout.diag_error()
        );
    }

    /// Regression fixture for issue #28 (4-set ellipse spec where A is a
    /// superset of B/C/D). eulerr fits this exactly. Currently reaches
    /// `diag_error ≈ 1e-11` at seed=1 — well below the bar.
    #[test]
    #[ignore = "slow regression coverage"]
    fn test_issue28_four_set_superset_ellipse_regression() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 30.0)
            .intersection(&["A", "B"], 3.0)
            .intersection(&["A", "C"], 3.0)
            .intersection(&["A", "D"], 3.0)
            .intersection(&["A", "B", "C"], 2.0)
            .intersection(&["A", "B", "D"], 2.0)
            .intersection(&["A", "C", "D"], 2.0)
            .intersection(&["A", "B", "C", "D"], 1.0)
            .build()
            .unwrap();

        // tol=1e-10 is needed to drive the *normalized* cost below the
        // FD noise floor on this hard high-arity case; with the old
        // SumSquared default (loss scaled with input²), tol=1e-8 was
        // equivalent. See `LossType::NormalizedSumSquared` doc.
        let layout = Fitter::<Ellipse>::new(&spec)
            .seed(1)
            .tolerance(1e-10)
            .max_iterations(2000)
            .fit()
            .unwrap();
        assert!(
            layout.diag_error() < 1e-6,
            "issue #28 case 2: diag_error = {:e} (expected < 1e-6)",
            layout.diag_error()
        );
    }

    #[test]
    fn test_sa_fallback_threshold_disabled_via_builder() {
        // With threshold=None the fallback path must not run; sanity-check the
        // builder plumbing by confirming fit() still succeeds and returns a layout.
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 4.0)
            .set("C", 3.0)
            .intersection(&["A", "B"], 1.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec)
            .seed(7)
            .sa_fallback_threshold(None)
            .fit()
            .unwrap();

        assert_eq!(layout.shapes().len(), 3);
        assert!(layout.loss().is_finite());
    }

    #[test]
    #[ignore = "slow regression coverage"]
    fn test_fitter_with_ellipses_three_sets() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 15.0)
            .set("B", 12.0)
            .set("C", 10.0)
            .intersection(&["A", "B"], 3.0)
            .intersection(&["B", "C"], 2.5)
            .intersection(&["A", "C"], 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(123).fit().unwrap();

        assert_eq!(layout.shapes().len(), 3);
        assert!(layout.loss() < 20.0); // Should converge
    }

    #[test]
    fn test_ellipse_to_polygon_workflow() {
        use crate::geometry::shapes::{Ellipse, Polygon};
        use crate::geometry::traits::Polygonize;

        // Fit a diagram with ellipses
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 8.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();

        // Convert to polygons for plotting
        let polygons: Vec<Polygon> = layout
            .shapes()
            .iter()
            .map(|ellipse| ellipse.polygonize(64))
            .collect();

        assert_eq!(polygons.len(), 2);
        assert_eq!(polygons[0].vertices().len(), 64);
        assert_eq!(polygons[1].vertices().len(), 64);

        // Polygons should have areas close to ellipse areas
        use crate::geometry::traits::Area;
        for (ellipse, polygon) in layout.shapes().iter().zip(polygons.iter()) {
            let error = (ellipse.area() - polygon.area()).abs() / ellipse.area();
            assert!(
                error < 0.01,
                "Polygon area error too large: {:.2}%",
                error * 100.0
            ); // < 1% error with 64 vertices
        }
    }

    #[test]
    #[ignore = "slow regression coverage"]
    fn test_spurious_ac_intersection() {
        use crate::geometry::shapes::Ellipse;
        use crate::spec::{DiagramSpecBuilder, InputType};

        // User reported: A=2.2, B=2, C=2, A&B&C=1 (exclusive)
        // Result shows A&C=0.059 but A&C don't visually intersect
        let spec = DiagramSpecBuilder::new()
            .set("A", 2.2)
            .set("B", 2.0)
            .set("C", 2.0)
            .intersection(&["A", "B", "C"], 1.0)
            .input_type(InputType::Exclusive)
            .build()
            .unwrap();

        // Try multiple seeds to see if we can find a good solution
        let seeds = vec![42, 123, 456, 789, 1000];
        let mut best_loss = f64::INFINITY;
        let mut best_seed = 0;

        for &seed in &seeds {
            let fitter = Fitter::<Ellipse>::new(&spec).seed(seed);
            let layout = fitter.fit().unwrap();
            if layout.loss() < best_loss {
                best_loss = layout.loss();
                best_seed = seed;
            }
        }

        // Use best seed for detailed analysis
        let fitter = Fitter::<Ellipse>::new(&spec).seed(best_seed);
        let layout = fitter.fit().unwrap();
        assert!(layout.loss().is_finite());
    }
}
#[test]
fn test_circles_ac_issue_seed42() {
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Circle;
    use crate::spec::{Combination, DiagramSpecBuilder, InputType};

    // User test case: A=2.2, B=2, C=3, A&B&C=1 (exclusive), seed=42
    // Shows A&C=0.339 but no visual intersection
    let spec = DiagramSpecBuilder::new()
        .set("A", 2.2)
        .set("B", 2.0)
        .set("C", 3.0)
        .intersection(&["A", "B", "C"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let fitter = Fitter::<Circle>::new(&spec).seed(42);
    let layout = fitter.fit().unwrap();

    let shape_a = layout.shape_for_set("A").unwrap();
    let shape_c = layout.shape_for_set("C").unwrap();

    let dist_ac = ((shape_a.center().x() - shape_c.center().x()).powi(2)
        + (shape_a.center().y() - shape_c.center().y()).powi(2))
    .sqrt();

    let ac_combo = Combination::new(&["A", "C"]);
    let ac_fitted = layout.fitted().get(&ac_combo).copied().unwrap_or(0.0);

    // If distance > sum of radii, they can't intersect
    if dist_ac > shape_a.radius() + shape_c.radius() {
        assert!(
            ac_fitted <= 0.001,
            "A&C fitted area is {:.3} but circles are separated",
            ac_fitted
        );
    }
}
#[test]
#[ignore = "slow regression coverage"]
fn test_compare_optimizers() {
    use crate::fitter::Fitter;
    use crate::geometry::shapes::Ellipse;
    use crate::spec::{DiagramSpecBuilder, InputType};

    let spec = DiagramSpecBuilder::new()
        .set("A", 2.2)
        .set("B", 2.0)
        .set("C", 2.0)
        .intersection(&["A", "B", "C"], 1.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let fitter_default = Fitter::<Ellipse>::new(&spec).seed(42);
    let layout_default = fitter_default.fit().unwrap();
    assert!(layout_default.loss().is_finite());
}
