//! Fitter for creating diagram layouts from specifications.

mod clustering;
pub mod final_layout;
mod initial_layout;
mod layout;
pub mod normalize;
mod packing;

#[cfg(test)]
mod corpus_quality;

#[cfg(test)]
mod synthetic_groundtruth;

pub use final_layout::Optimizer;
pub use initial_layout::{InitialSampler, MdsSolver};
pub use layout::Layout;

use crate::error::DiagramError;
use crate::geometry::shapes::Circle;
use crate::geometry::traits::DiagramShape;
use crate::loss::LossType;
use crate::spec::{DiagramSpec, PreprocessedSpec};
use crate::venn::VennDiagram;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use rayon::prelude::*;
use std::collections::HashMap;

/// Maximum `n_sets` for which the Venn warm-start is attempted under
/// [`Fitter::<Circle>`]. The canonical Venn is genuinely circular only for
/// `n ∈ {1, 2, 3}` ([`crate::venn`]); n=4..=5 use Wilkinson/Edwards
/// ellipse arrangements, so we cap circles slightly above the supported
/// range and let [`venn_warm_start_params`] return `None` for the
/// non-circular cases.
const VENN_SEED_MAX_SETS_CIRCLE: usize = 4;
/// Maximum `n_sets` for which the Venn warm-start is attempted under
/// [`Fitter::<Ellipse>`]. Matches the n=1..=5 hardcoded arrangements
/// in [`crate::venn`]; beyond that no clean Venn ellipse exists.
const VENN_SEED_MAX_SETS_ELLIPSE: usize = 5;
/// Maximum `n_sets` for which the Venn warm-start is attempted under
/// [`Fitter::<Square>`]. Matches the n=1..=3 axis-aligned arrangements in
/// [`crate::venn`] — n ≥ 4 has no axis-aligned-square Venn (any four
/// axis-aligned rectangles miss at least one of the `2ⁿ − 1` regions).
const VENN_SEED_MAX_SETS_SQUARE: usize = 3;
/// Maximum `n_sets` for which the Venn warm-start is attempted under
/// [`Fitter::<Rectangle>`]. Same cap as [`VENN_SEED_MAX_SETS_SQUARE`] — the
/// Venn topology obstruction at n ≥ 4 applies to any axis-aligned shape
/// regardless of width/height freedom.
const VENN_SEED_MAX_SETS_RECTANGLE: usize = 3;

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
    /// Per-knob LM stopping overrides. `None` inherits `tolerance`. Only
    /// `Optimizer::LevenbergMarquardt` (and the LM polish step inside
    /// `Optimizer::CmaEsLm`) honours these — other optimizers don't expose
    /// the underlying knobs separately. See `tolerance` for default
    /// rationale; per-knob overrides exist for benchmarking / tuning the
    /// LM stopping behaviour without touching the shared `tolerance`.
    xtol: Option<f64>,
    ftol: Option<f64>,
    gtol: Option<f64>,
    seed: Option<u64>,
    loss_type: LossType,
    /// Pool of final-stage optimizers cycled across outer-loop restarts:
    /// attempt `i` uses `optimizer_pool[i % optimizer_pool.len()]`. Default
    /// `[Lbfgs]`. The previous default mixed Nelder-Mead in (`[NelderMead,
    /// Lbfgs]`) on the theory that NM is faster on small ellipse fits and
    /// L-BFGS handles the hard ones, but the `examples/quality_report` sweep
    /// showed NM-only ellipse fits land ~6 orders of magnitude worse on
    /// median loss than L-BFGS-only with no time advantage at our default
    /// `n_restarts`, so cycling NM in just dilutes the pool. Best loss across
    /// attempts still wins.
    optimizer_pool: Vec<Optimizer>,
    n_restarts: usize,
    /// Pool of MDS solvers cycled across outer-loop restarts: attempt `i`
    /// uses `initial_solvers[i % initial_solvers.len()]`. Mixing solvers in
    /// the pool widens basin coverage (different solvers fall into different
    /// local minima) without raising wall time, since restarts run in parallel.
    initial_solvers: Vec<MdsSolver>,
    /// Loss threshold above which `Optimizer::CmaEsLm` invokes the global
    /// CMA-ES escape stage. When plain LM lands at or below this value the
    /// CMA-ES step is skipped, so easy specs pay no extra wall time. Other
    /// optimizers ignore this field.
    cmaes_fallback_threshold: f64,
    /// Strategy for drawing per-restart MDS initial positions. The default
    /// `Uniform` matches eulerr's `runif`-per-restart behaviour;
    /// `LatinHypercube` stratifies the batch of `n_restarts` draws across
    /// `[0, scale]^(2·n_sets)`.
    initial_sampler: InitialSampler,
    /// Restart-loop thread count, honoured only when the `parallel` feature is
    /// on and the target is not wasm. `None` (the default) uses rayon's
    /// current/global pool — i.e. all logical cores, or whatever
    /// `RAYON_NUM_THREADS` / a caller-installed pool dictates. `Some(n)` with
    /// `n >= 1` runs the restarts in a private, scoped pool of `n` threads,
    /// leaving the caller's global pool untouched. `Some(0)` is treated as
    /// `None`. See [`Fitter::jobs`].
    jobs: Option<usize>,
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
            // 200 iters × tolerance 1e-3 tracks eulerr's nlm budget for the
            // iter cap; the cost tolerance was tuned via the `final_tolerance`
            // bench (see `crates/eunoia/benches/final_tolerance.rs`). With the
            // default loss `SumSquared` (scale-invariant SSE — see `LossType`)
            // the loss magnitude is bounded ~`[0, 1]` regardless of input area
            // scale, so cost-tolerance behaviour is consistent across specs.
            // `tolerance` is wired as the LM `ftol` (cost-change exit) and as
            // L-BFGS' grad+cost tolerance; the LM `xtol`/`gtol` knobs keep
            // their own fixed `1e-6` defaults so loosening `tolerance` only
            // shortens the cost-converged tail without trading off
            // parameter-space precision. The corpus validation pass over all
            // 27 specs × 16 seeds × {Circle, Ellipse} showed zero regressions
            // at `tolerance = 1e-3` versus the previous `1e-6`, with up to
            // ~170× wall-time wins on the slow ellipse specs (`gene_sets`
            // 91.9 ms → 0.54 ms). Tighten via `Fitter::tolerance(_)` if a
            // future spec needs a sharper cost tail.
            max_iterations: 200,
            tolerance: 1e-3,
            xtol: None,
            ftol: None,
            gtol: None,
            seed: None,
            // `SumSquared` is the scale-invariant `Σ(f-t)² / Σt²`. The
            // bounded-`[0, 1]` magnitude keeps `tolerance` and
            // `cmaes_fallback_threshold` portable across specs.
            loss_type: LossType::SumSquared,
            // L-BFGS only. We previously cycled `[NelderMead, Lbfgs]` so each
            // restart attempt traded off NM's per-call speed against L-BFGS'
            // basin coverage, but the `examples/quality_report` sweep showed
            // NM-only ellipse fits land ~6 orders of magnitude worse on median
            // loss than L-BFGS-only with no wall-time advantage at the default
            // `n_restarts=10` (NM's per-call speed didn't beat L-BFGS in the
            // restart-parallelised total). Mixing NM into the pool just
            // diluted the result. Trust-region landed in the same basins as
            // L-BFGS but ~10× slower. Levenberg-Marquardt then crushed
            // L-BFGS in turn (~20 orders of magnitude lower median loss on
            // ellipses, ~3× faster wall time) — see the `lm_final` row in
            // `examples/quality_report`.
            // CMA-ES global escape with LM polish, threshold-fired (see
            // `cmaes_fallback_threshold`) so easy specs pay no extra wall
            // time. On the `examples/quality_report` ellipse sweep this
            // closes specs LM-on-LM gets stuck on (`issue92_3_set_dropped_pair`
            // 1.3e-4 → 1.5e-29, `random_4_set` 8.5e-3 → 4.1e-3) without
            // regressing the easy ones — every spec where `lm_full` lands
            // at machine precision sits below the threshold and skips
            // CMA-ES entirely.
            optimizer_pool: vec![Optimizer::CmaEsLm],
            // Default loss threshold below which the CMA-ES global stage
            // is skipped. Only consulted by `Optimizer::CmaEsLm`. See
            // `FinalLayoutConfig::cmaes_fallback_threshold` for the
            // empirical justification of `1e-3`.
            cmaes_fallback_threshold: 1e-3,
            // Number of full-pipeline restarts (fresh MDS init + final
            // optimizer per attempt, lowest-loss attempt kept). Matches
            // eulerr's `n_restarts = 10`. Each fit does that much work.
            n_restarts: 10,
            // Levenberg-Marquardt for the MDS init. The previous default was
            // L-BFGS with a More-Thuente line search, which under certain
            // starting points (e.g. the LHS-induced
            // `issue71_4_set_extreme_scale` config where one circle fully
            // contains the others) stalled inside the inner line search at a
            // subset-clamp kink. LM's trust-region update sidesteps the
            // line-search stall, and `MdsSolver::LevenbergMarquardt` is already
            // wired with the analytic per-pair Jacobian. The mix
            // `initial_solver_pool([LevenbergMarquardt, Lbfgs])` is still
            // available for experimentation.
            initial_solvers: vec![MdsSolver::LevenbergMarquardt],
            // Independent uniform draws per restart, matching eulerr. See
            // `initial_sampler` to switch to a stratified Latin-hypercube
            // design across the `n_restarts` batch.
            initial_sampler: InitialSampler::default(),
            // Use rayon's current/global pool when parallelism is compiled in;
            // a no-op otherwise. Callers pin a count via `Fitter::jobs`.
            jobs: None,
            _shape: std::marker::PhantomData,
        }
    }

    /// Pin the initial-layout MDS solver to a single choice.
    ///
    /// The default is [`MdsSolver::LevenbergMarquardt`] for every restart.
    /// [`MdsSolver::Lbfgs`] is also available for experimentation; it reaches
    /// different basins on hard ellipse fits but its line search can stall at
    /// the MDS objective's subset-clamp kinks, so LM is the default.
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
    /// `[MdsSolver::LevenbergMarquardt]` (single-solver); mixing in
    /// `MdsSolver::Lbfgs` can widen best-of-N basin coverage on
    /// issue #28-class specs.
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
    /// // Bias the cycle 7:3 toward Levenberg-Marquardt over L-BFGS.
    /// let fitter = Fitter::<Ellipse>::new(&spec).initial_solver_pool(vec![
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::LevenbergMarquardt,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    ///     MdsSolver::Lbfgs,
    /// ]);
    /// ```
    pub fn initial_solver_pool(mut self, pool: Vec<MdsSolver>) -> Self {
        assert!(!pool.is_empty(), "initial_solver_pool must be non-empty");
        self.initial_solvers = pool;
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

    /// Set the cost-change convergence tolerance for the final-stage
    /// optimizer.
    ///
    /// - **Levenberg-Marquardt** (incl. the LM polish step of
    ///   [`Optimizer::CmaEsLm`]): wired as `with_ftol`, the relative-cost
    ///   exit. The LM parameter-change (`xtol`) and gradient (`gtol`) knobs
    ///   keep their own fixed `1e-6` defaults; override them independently
    ///   via [`xtol`] / [`gtol`].
    /// - **L-BFGS**: wired as both `tol_grad` and `tol_cost`.
    /// - **Nelder-Mead**: no tolerance setter is exposed, so it runs until
    ///   `max_iterations`.
    ///
    /// The default is `1e-3`, chosen to maximise the timing win on
    /// cost-converged fits without regressing the corpus (validated across
    /// all 27 specs × 16 seeds × {Circle, Ellipse} via the `final_tolerance`
    /// bench). Tighten this (e.g. `1e-6`) if a spec needs a sharper cost
    /// tail; loosen it (e.g. `1e-2`) for even faster coarse fits.
    ///
    /// [`xtol`]: Self::xtol
    /// [`gtol`]: Self::gtol
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

    /// Override the LM parameter-change tolerance (`with_xtol`).
    ///
    /// Only honoured by [`Optimizer::LevenbergMarquardt`] and the LM polish
    /// step of [`Optimizer::CmaEsLm`]. `None` (the default) uses the fixed
    /// LM `xtol` default of `1e-6` — independent of [`tolerance`], which
    /// targets only the LM `ftol` (cost) knob. LM stops when
    /// `‖Δp‖ ≤ xtol·‖p‖`.
    ///
    /// [`tolerance`]: Self::tolerance
    pub fn xtol(mut self, xtol: f64) -> Self {
        self.xtol = Some(xtol);
        self
    }

    /// Override the LM cost-change tolerance (`with_ftol`).
    ///
    /// Only honoured by [`Optimizer::LevenbergMarquardt`] and the LM polish
    /// step of [`Optimizer::CmaEsLm`]. `None` (the default) inherits
    /// [`tolerance`] (the unified cost-tolerance setter). LM stops when
    /// relative cost change drops below `ftol`.
    ///
    /// [`tolerance`]: Self::tolerance
    pub fn ftol(mut self, ftol: f64) -> Self {
        self.ftol = Some(ftol);
        self
    }

    /// Override the LM gradient tolerance (`with_gtol`).
    ///
    /// Only honoured by [`Optimizer::LevenbergMarquardt`] and the LM polish
    /// step of [`Optimizer::CmaEsLm`]. `None` (the default) uses the fixed
    /// LM `gtol` default of `1e-6` — independent of [`tolerance`], which
    /// targets only the LM `ftol` (cost) knob. LM stops when the ∞-norm of
    /// `Jᵀr` drops below `gtol`.
    ///
    /// [`tolerance`]: Self::tolerance
    pub fn gtol(mut self, gtol: f64) -> Self {
        self.gtol = Some(gtol);
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
    /// `[Lbfgs]` (single-solver) — `examples/quality_report` showed NM-only
    /// ellipse fits land ~6 orders of magnitude worse on median loss with no
    /// wall-time advantage at the default `n_restarts`, so mixing NM in
    /// merely diluted the pool. Use this builder to opt back in to a mixed
    /// pool for experimentation.
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

    /// Set the number of threads used to fan out the `n_restarts` loop.
    ///
    /// This is a pure wall-time knob: the restarts are independently seeded and
    /// reduced by lowest loss, so the chosen layout is byte-for-byte identical
    /// no matter the thread count. It exists so that integrators — language
    /// bindings, async services — can map their own "cores"/"jobs" setting
    /// straight through, and so a single fit can be pinned single-threaded
    /// (`jobs(1)`) without recompiling.
    ///
    /// - `Some(n)` with `n >= 1` runs the restarts in a *private, scoped*
    ///   thread pool of `n` threads. eunoia never calls
    ///   `rayon::ThreadPoolBuilder::build_global`, so the caller's own global
    ///   rayon pool is left untouched.
    /// - `Some(0)` and `None` (the default) defer to rayon's current/global
    ///   pool, which honours `RAYON_NUM_THREADS` and any pool the caller has
    ///   installed — typically all logical cores.
    ///
    /// Has an effect only when the crate is built with the `parallel` feature
    /// and the target is not wasm; otherwise the fit runs single-threaded and
    /// this setting is accepted but inert (so binding FFI surfaces need no
    /// feature gating).
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
    /// // Pin this fit to a single thread (e.g. inside an async task).
    /// let fitter = Fitter::<Circle>::new(&spec).jobs(1);
    /// ```
    pub fn jobs(mut self, n: usize) -> Self {
        self.jobs = Some(n);
        self
    }

    /// Set the loss threshold above which `Optimizer::CmaEsLm` invokes its
    /// CMA-ES global escape stage.
    ///
    /// `Optimizer::CmaEsLm` always runs plain Levenberg-Marquardt first; if
    /// that lands at or below `threshold`, the (expensive) CMA-ES step is
    /// skipped and the LM result is returned. If LM stalls above
    /// `threshold` — e.g. on `issue92_3_set_dropped_pair` (1.3e-4 across
    /// every seed under plain LM) or `eulerape_3_set` (4.4e-4) — CMA-ES
    /// fires and the lower-loss of {LM, CMA-ES → LM polish} is kept.
    ///
    /// Default `1e-3` was picked empirically from `examples/quality_report`:
    /// every spec where `lm_full` lands at machine precision sits well
    /// below 1e-20, every spec where it stalls in a wrong basin sits at or
    /// above 1e-4, so 1e-3 cleanly separates the two populations.
    /// Tightening (e.g. `1e-6`) makes CMA-ES fire on more specs at higher
    /// wall-time cost; loosening (e.g. `1e-1`) gives back ~all of LM's
    /// runtime at the price of leaving the few hard-stuck specs in LM's
    /// suboptimal basin. Other optimizers ignore this setting.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Tighter threshold — fire CMA-ES on more specs.
    /// let fitter = Fitter::<Ellipse>::new(&spec).cmaes_fallback_threshold(1e-6);
    /// ```
    pub fn cmaes_fallback_threshold(mut self, threshold: f64) -> Self {
        self.cmaes_fallback_threshold = threshold;
        self
    }

    /// Set the strategy for drawing initial MDS positions across the
    /// `n_restarts` batch.
    ///
    /// The default [`InitialSampler::Uniform`] preserves eulerr's behaviour
    /// (independent uniform draws per restart on `[0, sqrt(Σ areas)]`).
    /// [`InitialSampler::LatinHypercube`] replaces the batch of `n_restarts`
    /// independent draws with a single stratified design — each of the
    /// `2·n_sets` axes is split into `n_restarts` equal strata sampled exactly
    /// once — so attempts cover the search space more evenly. This costs no
    /// additional MDS work, only a one-time `O(n_restarts · n_sets)` setup,
    /// and is intended to lift best-of-N quality on hard specs where multiple
    /// uniform draws happen to land in the same basin.
    ///
    /// # Examples
    ///
    /// ```
    /// use eunoia::{DiagramSpecBuilder, Fitter, InitialSampler};
    /// use eunoia::geometry::shapes::Ellipse;
    ///
    /// let spec = DiagramSpecBuilder::new()
    ///     .set("A", 10.0)
    ///     .set("B", 8.0)
    ///     .build()
    ///     .unwrap();
    ///
    /// let fitter = Fitter::<Ellipse>::new(&spec)
    ///     .initial_sampler(InitialSampler::LatinHypercube);
    /// ```
    pub fn initial_sampler(mut self, sampler: InitialSampler) -> Self {
        self.initial_sampler = sampler;
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

        // Container/complement fitting is only wired for shapes that
        // implement `compute_exclusive_regions_clipped`. As of S5 every
        // currently-implemented shape (Circle, Ellipse, Square, Rectangle)
        // does; this gate stays for forward-compatibility when new shape
        // types land without a clipping path yet.
        if spec.complement.is_some() && !shape_supports_container_clipping::<S>() {
            return Err(DiagramError::InvalidCombination(
                "complement (container/universe) fitting is not supported for this \
                 shape type; the shape lacks a `compute_exclusive_regions_clipped` \
                 implementation"
                    .to_string(),
            ));
        }

        // Multi-cluster + complement is rejected by design. A single container
        // rectangle is a single universe — putting two disjoint clusters
        // inside it makes the "items outside every set" region span both
        // clusters and the gap between them, which the packer would normally
        // separate. Per-cluster containers were considered (one universe per
        // disjoint sub-diagram) but rejected: the complement count is one
        // number, with no obvious per-cluster split, and the visual framing
        // would imply mutually exclusive universes that don't reflect the
        // user's intent. Users with multiple disjoint clusters should fit
        // them as separate diagrams, each with its own complement.
        if spec.complement.is_some() && spec_is_multi_cluster(&spec) {
            return Err(DiagramError::InvalidCombination(
                "complement (container/universe) fitting requires that every set overlap \
                 (directly or transitively) with every other set; fit each disjoint \
                 cluster as its own diagram with its own complement instead"
                    .to_string(),
            ));
        }
        let max_iterations = self.max_iterations;
        let tolerance = self.tolerance;
        let xtol = self.xtol;
        let ftol = self.ftol;
        let gtol = self.gtol;
        let loss_type = self.loss_type;
        let optimizer_pool = self.optimizer_pool.clone();
        let cmaes_fallback_threshold = self.cmaes_fallback_threshold;
        let jobs = self.jobs;

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

        // Pre-compute the Latin-hypercube design from the master RNG (so each
        // parallel attempt receives a deterministic precomputed initial
        // position) when LHS is selected. Uniform sampling is left to each
        // attempt's local rng for behavioural parity with eulerr's
        // independent-restart `runif`.
        let lhs_rows: Option<Vec<Vec<f64>>> = match self.initial_sampler {
            InitialSampler::Uniform => None,
            InitialSampler::LatinHypercube => {
                // Stratify on the central `2·LHS_HALF_WIDTH_FRAC` fraction of
                // the eulerr `[0, scale]` extent (centred at `scale/2`),
                // not the full extent. The full-extent design over-spreads
                // into edge regions Uniform sampling rarely visits, which
                // both wastes restarts on implausible inits and (empirically)
                // forces some specs into pathological strata that deadlock
                // L-BFGS-MDS at subset clamps. See `LHS_HALF_WIDTH_FRAC`.
                let scale = initial_layout::sampling_scale(&spec.set_areas);
                let half_width = initial_layout::LHS_HALF_WIDTH_FRAC * scale;
                let lo = 0.5 * scale - half_width;
                let hi = 0.5 * scale + half_width;
                Some(initial_layout::latin_hypercube_rows(
                    n_attempts,
                    n_sets * 2,
                    lo,
                    hi,
                    &mut master_rng,
                ))
            }
        };

        // Precompute the Venn warm-start params (full shape parameters,
        // already at the spec scale) for slot 0. `None` means slot 0 falls
        // back to the standard MDS path. Only consulted when `optimize`
        // is true — the no-optimization branch returns the first MDS init
        // as-is for diagnostic purposes.
        //
        // `venn_warm_start_params` returns `None` (so slot 0 stays on the
        // MDS path) whenever the canonical Venn isn't applicable: n_sets
        // outside the per-shape support window (≤ 4 circles, ≤ 5 ellipses),
        // any disjoint pair in the spec (the Venn topology forces every
        // region positive, so the optimizer would have to undo the
        // overlap from a basin with little gradient — see TODO.md), or
        // shapes where `n_params()` is neither 3 nor 5. In every other
        // case the warm-start replaces slot 0's random MDS init.
        //
        // Empirically this closes `issue92_3_set_dropped_pair` from a
        // basin LM-on-LM gets stuck in (1.309e-4 across every seed) to
        // ~1.4e-31 across all seeds, lifts ellipse spec-wins from 18 to
        // 19 in `examples/quality_report`, and is a no-op on circles
        // (most circle specs are out of range or already at the optimum).
        // Cost is ~2% wall time when applicable.
        let venn_initial: Option<Vec<f64>> = if optimize && spec.complement.is_none() {
            venn_warm_start_params::<S>(&spec)
        } else {
            None
        };

        let initial_solvers = self.initial_solvers.clone();
        let run_attempt =
            |(attempt_idx, attempt_seed): (usize, u64)| -> Result<(Vec<f64>, f64), DiagramError> {
                let attempt_optimizer = optimizer_pool[attempt_idx % optimizer_pool.len()];
                let final_config = final_layout::FinalLayoutConfig {
                    max_iterations,
                    tolerance,
                    xtol,
                    ftol,
                    gtol,
                    loss_type,
                    optimizer: attempt_optimizer,
                    seed: attempt_seed,
                    // Outer loop already provides full-pipeline diversity via fresh
                    // MDS inits, so each attempt's final stage runs once.
                    n_restarts: 1,
                    cmaes_fallback_threshold,
                };

                if attempt_idx == 0
                    && let Some(venn_params) = venn_initial.as_ref()
                {
                    // Slot 0 with Venn warm-start: skip MDS entirely and
                    // hand the canonical-Venn shape parameters straight
                    // to the final-stage optimizer (flavour (b) in
                    // TODO.md). MDS adds no information when we already
                    // have a layout where every set overlaps every
                    // other.
                    let initial_param = DVector::from_vec(venn_params.clone());
                    return final_layout::optimize_from_initial::<S>(
                        &spec,
                        &initial_param,
                        &final_config,
                    )
                    .map(|(p, l)| (p.as_slice().to_vec(), l))
                    .map_err(|e| {
                        DiagramError::InvalidCombination(format!(
                            "Venn-seed final optimisation failed: {}",
                            e
                        ))
                    });
                }

                let mut attempt_rng = StdRng::seed_from_u64(attempt_seed);

                let initial_solver = initial_solvers[attempt_idx % initial_solvers.len()];
                let initial_positions: Option<&[f64]> =
                    lhs_rows.as_ref().map(|rows| rows[attempt_idx].as_slice());
                let initial_params = match initial_layout::compute_initial_layout_with_solver(
                    &optimal_distances,
                    &spec.relationships,
                    &spec.set_areas,
                    &mut attempt_rng,
                    initial_solver,
                    initial_positions,
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
                        params.extend(S::optimizer_params_from_circle(xi, yi, r));
                    }
                    return Ok((params, 0.0));
                }

                match final_layout::optimize_layout::<S>(
                    &spec,
                    &initial_positions,
                    &initial_radii,
                    final_config,
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
            #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
            {
                // Fan the independent restarts across threads. Results are
                // reduced by lowest loss below, and each attempt is seeded
                // independently, so the chosen layout is identical regardless
                // of thread count or completion order — `jobs` only moves wall
                // time.
                let run = || {
                    indexed_seeds
                        .par_iter()
                        .map(|&pair| run_attempt(pair))
                        .collect()
                };
                match jobs {
                    // Explicit count: a private, scoped pool. We deliberately
                    // never `build_global` — that pool is process-wide and
                    // one-shot, so a library claiming it would hijack the
                    // caller's own rayon usage. A pool that fails to build is
                    // not worth aborting the fit over, so fall back to the
                    // current pool.
                    Some(n) if n >= 1 => {
                        match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
                            Ok(pool) => pool.install(run),
                            Err(_) => run(),
                        }
                    }
                    // `None`/`Some(0)`: rayon's current/global pool, which
                    // respects `RAYON_NUM_THREADS` and any caller-installed
                    // pool.
                    _ => run(),
                }
            }

            #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
            {
                // Serial path: `parallel` feature off, or wasm (no threads).
                // `jobs` is accepted but inert here.
                let _ = jobs;
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

        let (final_params, _loss) = best.ok_or_else(|| {
            last_err.unwrap_or_else(|| {
                DiagramError::InvalidCombination(
                    "All restarts failed to produce a layout".to_string(),
                )
            })
        })?;

        // Step 3: Create shapes for the non-empty sets from optimized parameters
        let params_per_shape = S::n_params();
        let mut optimized_shapes: Vec<S> = Vec::with_capacity(n_sets);
        for i in 0..n_sets {
            let start = i * params_per_shape;
            let end = start + params_per_shape;
            optimized_shapes.push(S::from_optimizer_params(&final_params[start..end]));
        }

        // When the spec carries a complement, the trailing 4 entries of the
        // optimizer parameter vector encode the jointly-optimised container
        // rectangle. Decode it here so it can ride along with `Layout`.
        let mut optimized_container: Option<crate::geometry::shapes::Rectangle> =
            if spec.complement.is_some() {
                let start = n_sets * params_per_shape;
                let trailing = &final_params[start..start + 4];
                Some(crate::geometry::shapes::Rectangle::from_optimizer_params(
                    trailing,
                ))
            } else {
                None
            };

        // Compute the pre-normalize exclusive-region areas and thread them
        // into `normalize_layout` so cluster detection uses the same exact-
        // conic / inclusion-exclusion math the optimizer minimised against.
        // Otherwise the geometric `find_clusters` (built on
        // `Closed::intersects` quick-rejects + boundary crossings) can
        // disagree with the optimizer at near-coincident ellipse geometry,
        // mis-split a cluster, and let `pack_clusters` translate genuinely
        // overlapping shapes apart.
        //
        // No post-normalize sanity assert lives here: a previous version
        // compared the pre/post exclusive-region map (per-mask) and another
        // compared total visible area, but both proved unreliable in
        // practice. `normalize_layout` is rigid (rotation + mirror +
        // translation), so the geometry is mathematically preserved — but
        // `compute_exclusive_regions` re-runs quartic conic intersection
        // on rotated coordinates, and on near-degenerate ellipse fits ULP
        // drift in the rotated coefficients can shift root classifications
        // enough that the recomputed region map differs by O(1e-2) of the
        // total area on `three_inside_fourth`-style "shapes inside one big
        // shape" geometries. That overlaps the original `intersects`
        // clustering-bug magnitude (~6.5e-2 × scale) too closely to give a
        // useful false-positive margin. Coverage of the original bug now
        // comes structurally from `find_clusters_from_exclusive_regions`
        // (which uses the same area math the optimizer consumed) plus
        // end-to-end `diag_error` ceilings in `corpus_quality` and
        // `synthetic_groundtruth`.
        let pre_normalize_regions = S::compute_exclusive_regions(&optimized_shapes);

        // Step 4: Normalize the non-empty shapes only (zero shapes would confuse
        // clustering/packing). We do this before re-assembly.
        //
        // Complement specs use the container-aware path: only translate, so
        // the (axis-aligned) container/shape relationship survives. The
        // standard rotate/mirror/pack path is reserved for non-complement
        // specs where the cluster orientation can be freely chosen.
        if let Some(container) = optimized_container.as_mut() {
            crate::fitter::normalize::normalize_layout_with_container(
                &mut optimized_shapes,
                container,
            );
        } else {
            crate::fitter::normalize::normalize_layout_with_clusters(
                &mut optimized_shapes,
                0.05,
                Some(&pre_normalize_regions),
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
                None => S::from_optimizer_params(&zero_params),
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
            optimized_container,
        );

        Ok(layout)
    }

    /// Compute target center-to-center distances per pair for the MDS phase.
    ///
    /// For each pair of sets, asks the shape `S` to invert its own pairwise
    /// overlap formula along a canonical direction (`DiagramShape::mds_target_distance`).
    /// Circles and ellipses share the closed-form lens-area inversion (the MDS
    /// warm-start treats every set as a circle of equal area); axis-aligned
    /// squares invert along the diagonal `|dx| = |dy|`.
    #[allow(clippy::needless_range_loop)]
    fn compute_optimal_distances(
        spec: &crate::spec::PreprocessedSpec,
    ) -> Result<Vec<Vec<f64>>, DiagramError> {
        let n_sets = spec.n_sets;
        let mut optimal_distances = vec![vec![0.0; n_sets]; n_sets];

        for i in 0..n_sets {
            for j in (i + 1)..n_sets {
                let overlap = spec.relationships.overlap_area(i, j);
                let desired_distance =
                    S::mds_target_distance(spec.set_areas[i], spec.set_areas[j], overlap)?;

                optimal_distances[i][j] = desired_distance;
                optimal_distances[j][i] = desired_distance;
            }
        }

        Ok(optimal_distances)
    }
}

/// Build full shape-parameter vector for the canonical Venn-diagram layout
/// of the preprocessed spec, scaled to roughly match the spec's set sizes.
///
/// Returns `None` (so the caller falls back to the standard MDS path) when:
/// - `n_sets < 2`.
/// - `n_sets` is outside the per-shape support window
///   ([`VENN_SEED_MAX_SETS_CIRCLE`] / [`VENN_SEED_MAX_SETS_ELLIPSE`] /
///   [`VENN_SEED_MAX_SETS_SQUARE`]).
/// - The spec has any disjoint pair (Venn topology forces every region
///   open; specs with hard-zero overlaps would start in a wrong topology
///   the optimizer can't easily escape — see TODO.md).
/// - `S` is not [`Circle`], [`Ellipse`], or [`Square`] — the per-shape
///   parameter encoding is hard-coded below, and other shapes that happen
///   to share an `n_params()` count would silently mis-decode.
/// - The canonical Venn for `n_sets` is non-circular and the shape is a
///   circle (i.e. n_sets ∈ {4, 5} under [`Circle`]); we have no
///   circular-Venn arrangement for those.
/// - The underlying [`VennDiagram::new`] returns `Err`.
///
/// The canonical Venn arrangements live at unit scale (radii / sides ~1).
/// We scale by a shape-appropriate factor of the spec's mean set area so
/// the warm-start sits at the same magnitude as the spec — the loss
/// landscape's gradient is much healthier when the initial shapes are
/// close to the right size:
/// - Circle / Ellipse: scale by `mean(sqrt(area_i / π))` so radii / semi-axes
///   land at the mean spec radius.
/// - Square: scale by `mean(sqrt(area_i))` so side lengths land at the mean
///   spec side, which puts each square's area at the mean spec area.
///
/// [`Circle`]: crate::geometry::shapes::Circle
/// [`Ellipse`]: crate::geometry::shapes::Ellipse
/// [`Square`]: crate::geometry::shapes::Square
fn venn_warm_start_params<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
) -> Option<Vec<f64>> {
    use crate::geometry::shapes::{Ellipse, Rectangle, Square};
    use std::any::TypeId;

    let n_sets = spec.n_sets;
    if n_sets < 2 {
        return None;
    }
    if spec_has_disjoint_pair(spec) {
        return None;
    }

    let type_id = TypeId::of::<S>();
    let pp = S::n_params();

    if type_id == TypeId::of::<Square>() {
        if n_sets > VENN_SEED_MAX_SETS_SQUARE {
            return None;
        }
        // Match the canonical-Venn square footprint (sides ~1) to the spec's
        // mean side length. `mean(sqrt(area_i))` lands a unit-side canonical
        // square at the right area magnitude after scaling.
        let mean_side: f64 = if !spec.set_areas.is_empty() {
            let total: f64 = spec.set_areas.iter().map(|a| a.sqrt()).sum();
            (total / spec.set_areas.len() as f64).max(1e-6)
        } else {
            1.0
        };
        let venn = VennDiagram::<Square>::new(n_sets).ok()?;
        let mut params = Vec::with_capacity(n_sets * pp);
        for sq in venn.shapes() {
            let c = sq.center();
            // Square params are `[x, y, side]`; bypass `optimizer_params_from_circle`
            // (which would re-encode `side` as `r·√π` and shrink the seed).
            params.extend([c.x() * mean_side, c.y() * mean_side, sq.side() * mean_side]);
        }
        return Some(params);
    }

    if type_id == TypeId::of::<Rectangle>() {
        if n_sets > VENN_SEED_MAX_SETS_RECTANGLE {
            return None;
        }
        // Rectangle's canonical Venn at n ≤ 3 uses square footprints
        // (width = height = side); aspect ratio = 1 → ln(ratio) = 0. Scale
        // sides by `mean(sqrt(area_i))` so the area magnitude matches the
        // spec, then encode as optimizer params `[x, y, ln(area), 0]`.
        let mean_side: f64 = if !spec.set_areas.is_empty() {
            let total: f64 = spec.set_areas.iter().map(|a| a.sqrt()).sum();
            (total / spec.set_areas.len() as f64).max(1e-6)
        } else {
            1.0
        };
        let venn = VennDiagram::<Rectangle>::new(n_sets).ok()?;
        let mut params = Vec::with_capacity(n_sets * pp);
        for r in venn.shapes() {
            let c = r.center();
            let scaled_w = r.width() * mean_side;
            let scaled_h = r.height() * mean_side;
            let u = (scaled_w * scaled_h).ln();
            let v = (scaled_w / scaled_h).ln();
            params.extend([c.x() * mean_side, c.y() * mean_side, u, v]);
        }
        return Some(params);
    }

    // Circle / Ellipse path. Both encode their warm-start positions in
    // ellipse parameters `(h, k, a, b, phi)` (a circle is the special case
    // `a == b`), so reach for the ellipse canonical Venn directly rather
    // than `VennDiagram::<S>::new` — circles have no canonical layout for
    // n ∈ {4, 5} and the circular subset of n ∈ {1, 2, 3} is encoded in
    // the ellipse layout already.
    if type_id != TypeId::of::<Circle>() && type_id != TypeId::of::<Ellipse>() {
        return None;
    }
    let max_n = match pp {
        3 => VENN_SEED_MAX_SETS_CIRCLE,
        5 => VENN_SEED_MAX_SETS_ELLIPSE,
        _ => return None,
    };
    if n_sets > max_n {
        return None;
    }

    let venn = VennDiagram::<Ellipse>::new(n_sets).ok()?;

    // Match the canonical-Venn footprint to the spec's mean circle radius.
    // Without this, very large-area specs (e.g. issue91-class with sums
    // ~thousand) would start at unit scale, ~100× too small for the loss
    // landscape.
    let mean_radius: f64 = if !spec.set_areas.is_empty() {
        let total: f64 = spec
            .set_areas
            .iter()
            .map(|a| (a / std::f64::consts::PI).sqrt())
            .sum();
        (total / spec.set_areas.len() as f64).max(1e-6)
    } else {
        1.0
    };

    let mut params = Vec::with_capacity(n_sets * pp);
    for ell in venn.shapes() {
        let h = ell.center().x() * mean_radius;
        let k = ell.center().y() * mean_radius;
        let a = ell.semi_major() * mean_radius;
        let b = ell.semi_minor() * mean_radius;
        let phi = ell.rotation();
        match pp {
            3 => {
                // Circle. Reject when the canonical Venn for this n_sets is
                // non-circular (n ∈ {4, 5}) — we'd have to drop the
                // semi-minor axis and lose the topology guarantee, which
                // defeats the warm-start's purpose.
                if (a - b).abs() > 1e-9 * mean_radius.max(1.0) {
                    return None;
                }
                params.extend(S::optimizer_params_from_circle(h, k, a));
            }
            5 => {
                // Ellipse params layout matches `Ellipse::optimizer_params_from_circle`:
                // [x, y, ln(a), ln(b), phi]. Semi-axes are stored in log
                // space so the unbounded LM solver stays on the positive-
                // axis manifold.
                params.extend([h, k, a.ln(), b.ln(), phi]);
            }
            _ => return None,
        }
    }
    Some(params)
}

/// Returns true iff the preprocessed spec marks at least one pair of sets
/// as fully disjoint (zero inclusive overlap).
fn spec_has_disjoint_pair(spec: &PreprocessedSpec) -> bool {
    let n = spec.n_sets;
    for i in 0..n {
        for j in (i + 1)..n {
            if spec.relationships.is_disjoint(i, j) {
                return true;
            }
        }
    }
    false
}

/// Whether the shape `S` supports the `compute_exclusive_regions_clipped`
/// path needed for joint container/complement fitting. Probed by calling the
/// trait method on an empty shape slice — a no-op call that returns
/// `Some(empty map)` for impls that override the default and `None` for
/// shapes still on the default trait impl.
///
/// As of S5 this returns true for `Circle`, `Ellipse`, `Square`, and
/// `Rectangle`. Used at fitter construction to fail early on unsupported
/// `complement + shape` combinations.
fn shape_supports_container_clipping<S: DiagramShape>() -> bool {
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::Rectangle;
    let probe_container = Rectangle::new(Point::new(0.0, 0.0), 1.0, 1.0);
    S::compute_exclusive_regions_clipped(&[], &probe_container).is_some()
}

/// Returns true iff the spec's pairwise-overlap graph (edge between i and j
/// when their inclusive overlap is non-zero) splits into more than one
/// connected component. Such specs would normally be packed via
/// `skyline_pack` into multiple clusters; with a single shared container
/// rectangle that has no clean semantics — see the rejection block in
/// [`Fitter::fit_with_optimization`]. Multi-cluster + complement is rejected
/// by design.
fn spec_is_multi_cluster(spec: &PreprocessedSpec) -> bool {
    let n = spec.n_sets;
    if n <= 1 {
        return false;
    }
    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            let root = find(parent, parent[i]);
            parent[i] = root;
        }
        parent[i]
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if !spec.relationships.is_disjoint(i, j) {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        roots.insert(r);
        if roots.len() > 1 {
            return true;
        }
    }
    false
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

    /// Regression fixture for issue #28 (6-set ellipse spec from eulerr's
    /// `test-accuracy.R`). eulerr's `nlm` backend fits this exactly. Currently
    /// reaches `diag_error ≈ 1.7e-9` at seed=1 — well below the bar.
    ///
    /// Spec is shared with the corpus (`wilkinson_6_set`); the corpus test
    /// asserts the *default* user experience, this test asserts what
    /// L-BFGS can reach with a tight budget.
    #[test]
    #[ignore = "slow regression coverage"]
    fn test_issue28_six_set_ellipse_regression() {
        use crate::geometry::shapes::Ellipse;
        use crate::test_utils::corpus;

        let spec = (corpus::get("wilkinson_6_set").expect("corpus entry").build)();

        // Asserts L-BFGS can drive diag_error below 1e-6 *when given the
        // budget*. The default tolerance (1e-4) is set above the FD noise
        // floor for speed (issue #34); this test pins down the tight-budget
        // behavior eulerr's C++ backend reached on this spec.
        // tol=1e-10 is needed to drive the cost below the FD noise floor
        // on this hard high-arity case; the un-normalised SumSquared
        // (scaled with input²) used to reach the same bar at tol=1e-8.
        // See `LossType::SumSquared` doc.
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
    ///
    /// Spec is shared with the corpus (`three_inside_fourth`).
    #[test]
    #[ignore = "slow regression coverage"]
    fn test_issue28_four_set_superset_ellipse_regression() {
        use crate::geometry::shapes::Ellipse;
        use crate::test_utils::corpus;

        let spec = (corpus::get("three_inside_fourth")
            .expect("corpus entry")
            .build)();

        // tol=1e-10 is needed to drive the cost below the FD noise floor
        // on this hard high-arity case; the un-normalised SumSquared
        // (scaled with input²) used to reach the same bar at tol=1e-8.
        // See `LossType::SumSquared` doc.
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

    /// Helper: build a 3-set spec with all pairwise + triple overlaps
    /// non-zero (no disjoint pairs), so the Venn warm-start is applicable.
    fn three_set_overlapping_spec() -> DiagramSpec {
        DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 10.0)
            .set("C", 10.0)
            .intersection(&["A", "B"], 4.0)
            .intersection(&["A", "C"], 4.0)
            .intersection(&["B", "C"], 4.0)
            .intersection(&["A", "B", "C"], 2.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap()
    }

    #[test]
    fn venn_warm_start_returns_some_for_supported_ellipse_n() {
        use crate::geometry::shapes::Ellipse;

        for n in 2..=5usize {
            // Build a fully-overlapping n-set spec via DiagramSpecBuilder
            // with set names "A".."A+n-1".
            let names: Vec<&str> = ["A", "B", "C", "D", "E"][..n].to_vec();
            let mut builder = DiagramSpecBuilder::new();
            for &name in &names {
                builder = builder.set(name, 10.0);
            }
            // A few moderate intersections so all pairs overlap.
            for i in 0..n {
                for j in (i + 1)..n {
                    builder = builder.intersection(&[names[i], names[j]], 1.0);
                }
            }
            let spec = builder
                .input_type(crate::InputType::Inclusive)
                .build()
                .unwrap();
            let preprocessed = spec.preprocess().unwrap();
            let params = venn_warm_start_params::<Ellipse>(&preprocessed);
            assert!(params.is_some(), "ellipse n={} should produce params", n);
            let params = params.unwrap();
            assert_eq!(params.len(), n * 5, "ellipse n={} param length", n);
        }
    }

    #[test]
    fn venn_warm_start_returns_some_for_circle_n_2_and_3() {
        for n in 2..=3usize {
            let names: Vec<&str> = ["A", "B", "C"][..n].to_vec();
            let mut builder = DiagramSpecBuilder::new();
            for &name in &names {
                builder = builder.set(name, 10.0);
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    builder = builder.intersection(&[names[i], names[j]], 1.0);
                }
            }
            let spec = builder
                .input_type(crate::InputType::Inclusive)
                .build()
                .unwrap();
            let preprocessed = spec.preprocess().unwrap();
            let params = venn_warm_start_params::<Circle>(&preprocessed);
            assert!(params.is_some(), "circle n={} should produce params", n);
            assert_eq!(params.unwrap().len(), n * 3);
        }
    }

    #[test]
    fn venn_warm_start_rejects_n4_and_n5_circles() {
        // Canonical Venn for n=4, 5 is non-circular (Wilkinson/Edwards
        // ellipse arrangements), so the Circle warm-start path must bail.
        for n in [4usize, 5] {
            let names: Vec<&str> = ["A", "B", "C", "D", "E"][..n].to_vec();
            let mut builder = DiagramSpecBuilder::new();
            for &name in &names {
                builder = builder.set(name, 10.0);
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    builder = builder.intersection(&[names[i], names[j]], 1.0);
                }
            }
            let spec = builder
                .input_type(crate::InputType::Inclusive)
                .build()
                .unwrap();
            let preprocessed = spec.preprocess().unwrap();
            assert!(
                venn_warm_start_params::<Circle>(&preprocessed).is_none(),
                "circle n={} must reject (Venn is non-circular)",
                n
            );
        }
    }

    #[test]
    fn venn_warm_start_rejects_n_above_5_for_ellipses() {
        use crate::geometry::shapes::Ellipse;

        let names = ["A", "B", "C", "D", "E", "F"];
        let mut builder = DiagramSpecBuilder::new();
        for &name in &names {
            builder = builder.set(name, 10.0);
        }
        for i in 0..6 {
            for j in (i + 1)..6 {
                builder = builder.intersection(&[names[i], names[j]], 1.0);
            }
        }
        let spec = builder
            .input_type(crate::InputType::Inclusive)
            .build()
            .unwrap();
        let preprocessed = spec.preprocess().unwrap();
        assert!(
            venn_warm_start_params::<Ellipse>(&preprocessed).is_none(),
            "ellipse n=6 must reject (no canonical Venn)"
        );
    }

    #[test]
    fn venn_warm_start_skips_specs_with_disjoint_pairs() {
        use crate::geometry::shapes::Ellipse;

        // three_disjoint: all sets fully separate, so the Venn topology
        // (every region positive) is wrong as a starting point.
        let spec = DiagramSpecBuilder::new()
            .set("A", 1.0)
            .set("B", 1.0)
            .set("C", 1.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();
        let preprocessed = spec.preprocess().unwrap();
        assert!(spec_has_disjoint_pair(&preprocessed));
        assert!(
            venn_warm_start_params::<Ellipse>(&preprocessed).is_none(),
            "ellipse + disjoint spec must skip Venn warm-start"
        );
    }

    #[test]
    fn venn_warm_start_returns_some_for_square_n_2_and_3() {
        use crate::geometry::shapes::Square;

        for n in 2..=3usize {
            let names: Vec<&str> = ["A", "B", "C"][..n].to_vec();
            let mut builder = DiagramSpecBuilder::new();
            for &name in &names {
                builder = builder.set(name, 10.0);
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    builder = builder.intersection(&[names[i], names[j]], 1.0);
                }
            }
            let spec = builder
                .input_type(crate::InputType::Inclusive)
                .build()
                .unwrap();
            let preprocessed = spec.preprocess().unwrap();
            let params = venn_warm_start_params::<Square>(&preprocessed);
            assert!(params.is_some(), "square n={} should produce params", n);
            let params = params.unwrap();
            assert_eq!(params.len(), n * 3, "square n={} param length", n);
            // Sides are scaled by mean(sqrt(area_i)) ≈ √10 ≈ 3.162. Canonical
            // sides are 1.0 for n ∈ {2, 3}, so every emitted side should be
            // strictly positive and roughly that magnitude.
            for i in 0..n {
                let side = params[3 * i + 2];
                assert!(side > 0.0, "square n={n} shape {i}: side {side} ≤ 0");
                assert!(
                    (1.0..10.0).contains(&side),
                    "square n={n} shape {i}: side {side} far from expected ≈ √10"
                );
            }
        }
    }

    #[test]
    fn venn_warm_start_rejects_n4_squares() {
        use crate::geometry::shapes::Square;

        // Axis-aligned squares cannot form a true Venn for n ≥ 4 (no
        // canonical layout in `Square::canonical_venn_layout`), so the
        // warm-start path must bail and let slot 0 fall back to MDS.
        let names = ["A", "B", "C", "D"];
        let mut builder = DiagramSpecBuilder::new();
        for &name in &names {
            builder = builder.set(name, 10.0);
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                builder = builder.intersection(&[names[i], names[j]], 1.0);
            }
        }
        let spec = builder
            .input_type(crate::InputType::Inclusive)
            .build()
            .unwrap();
        let preprocessed = spec.preprocess().unwrap();
        assert!(
            venn_warm_start_params::<Square>(&preprocessed).is_none(),
            "square n=4 must reject (no axis-aligned-square Venn)"
        );
    }

    #[test]
    fn venn_warm_start_skips_squares_for_specs_with_disjoint_pairs() {
        use crate::geometry::shapes::Square;

        let spec = DiagramSpecBuilder::new()
            .set("A", 1.0)
            .set("B", 1.0)
            .set("C", 1.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();
        let preprocessed = spec.preprocess().unwrap();
        assert!(spec_has_disjoint_pair(&preprocessed));
        assert!(
            venn_warm_start_params::<Square>(&preprocessed).is_none(),
            "square + disjoint spec must skip Venn warm-start"
        );
    }

    #[test]
    fn venn_seed_default_path_yields_finite_loss_for_square() {
        use crate::geometry::shapes::Square;

        // Confirm the slot-0 Venn warm-start path doesn't break a basic
        // 3-set square fit end-to-end.
        let spec = three_set_overlapping_spec();
        let layout = Fitter::<Square>::new(&spec).seed(42).fit().unwrap();
        assert!(layout.loss().is_finite());
        assert_eq!(layout.shapes().len(), 3);
    }

    #[test]
    fn venn_seed_default_path_yields_finite_loss() {
        // Sanity-check that the now-default Venn warm-start in slot 0
        // doesn't break a basic 3-set circle fit.
        let spec = three_set_overlapping_spec();
        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        assert!(layout.loss().is_finite());
    }

    #[test]
    fn venn_seed_falls_back_when_n_sets_too_large_for_circle() {
        // n=4 circle case: canonical Venn for n=4 is non-circular (ellipses),
        // so `venn_warm_start_params` returns `None` and the fitter must
        // fall back to the standard MDS path without panicking.
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 5.0)
            .set("C", 5.0)
            .set("D", 5.0)
            .intersection(&["A", "B"], 1.0)
            .intersection(&["A", "C"], 1.0)
            .intersection(&["A", "D"], 1.0)
            .intersection(&["B", "C"], 1.0)
            .intersection(&["B", "D"], 1.0)
            .intersection(&["C", "D"], 1.0)
            .input_type(crate::InputType::Inclusive)
            .build()
            .unwrap();
        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        assert!(layout.loss().is_finite());
    }

    // ===== Container / complement (Session 1) =====

    /// Two overlapping circles with a *small* complement: the container
    /// area should match `complement + Σ named` (the universe), shapes match
    /// targets, and the residual loss is small.
    #[test]
    fn fit_two_circles_small_complement_container_matches_universe() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 25.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 5.0)
            .complement(20.0) // universe = 75; some breathing room around the shapes
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");

        let area = container.width() * container.height();
        assert!(
            (area - 75.0).abs() / 75.0 < 0.05,
            "container area {} ≠ 75 (within 5%)",
            area
        );
        // L-BFGS with analytical gradients (S2) on the clipped landscape
        // reaches a low residual when the shapes have room inside the box.
        assert!(
            layout.loss() < 5e-3,
            "loss {} should be small",
            layout.loss()
        );
    }

    /// Same A/B sizes, but a *large* complement should leave the container
    /// much larger than the shapes — the shapes don't change relative to each
    /// other but the universe grows.
    #[test]
    fn fit_two_circles_large_complement_universe_matches() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 25.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 5.0)
            .complement(945.0) // universe = 1000
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");

        let area = container.width() * container.height();
        assert!(
            (area - 1000.0).abs() / 1000.0 < 0.05,
            "container area {} ≠ 1000 (within 5%)",
            area
        );
        // The disks together occupy ~55 area; the rest is complement. Container
        // should be much larger than the union.
        let union_target = 25.0 + 25.0 + 5.0; // 55
        assert!(
            area > 10.0 * union_target,
            "container area {} should be much larger than union {}",
            area,
            union_target
        );
    }

    /// Single set + complement: pins the absolute scale (the disk area must
    /// match the spec value, container area must match universe).
    #[test]
    fn fit_one_circle_with_complement_pins_scale() {
        // Single-set specs are rejected by `preprocess` (n_sets <= 1), so use
        // the smallest case that's accepted: one large set and one tiny set
        // that's effectively a placeholder, plus a complement.
        let spec = DiagramSpecBuilder::new()
            .set("A", 100.0)
            .set("B", 1.0)
            .intersection(&["A", "B"], 0.5)
            .complement(50.0) // universe = 100 + 1 + 0.5 + 50 ≈ 151.5
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");

        let area = container.width() * container.height();
        let universe = 151.5;
        assert!(
            (area - universe).abs() / universe < 0.05,
            "container area {} ≠ {} (within 5%)",
            area,
            universe
        );
    }

    /// Multi-cluster + complement should be rejected at fitter construction
    /// time with a clear error.
    #[test]
    fn multi_cluster_with_complement_errors() {
        // Three sets where {A, B} overlap but C is disjoint from both.
        let spec = DiagramSpecBuilder::new()
            .set("A", 10.0)
            .set("B", 10.0)
            .set("C", 10.0)
            .intersection(&["A", "B"], 2.0)
            .complement(50.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let result = Fitter::<Circle>::new(&spec).seed(42).fit();
        assert!(
            matches!(result, Err(DiagramError::InvalidCombination(_))),
            "expected InvalidCombination for multi-cluster + complement, got {:?}",
            result.map(|_| "Ok(_layout)")
        );
    }

    /// End-to-end ellipse + complement (S3): the container area should
    /// match the universe and shape areas should match targets. Mirrors
    /// `fit_two_circles_small_complement_container_matches_universe` but
    /// with `Fitter::<Ellipse>`. Loss tolerance is looser than the circle
    /// case because the ellipse path uses FD gradients (S4 will add
    /// analytical).
    #[test]
    fn fit_two_ellipses_small_complement_container_matches_universe() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 25.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 5.0)
            .complement(20.0) // universe = 75
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");

        let area = container.width() * container.height();
        assert!(
            (area - 75.0).abs() / 75.0 < 0.05,
            "container area {} ≠ 75 (within 5%)",
            area
        );
        // FD-gradient L-BFGS is less accurate than the circle's analytical
        // path; allow a higher residual.
        assert!(
            layout.loss() < 5e-2,
            "loss {} should be small",
            layout.loss()
        );
    }

    /// Ellipse + large complement: shapes occupy a small fraction of the
    /// container, but container area should still match the universe.
    #[test]
    fn fit_two_ellipses_large_complement_universe_matches() {
        use crate::geometry::shapes::Ellipse;

        let spec = DiagramSpecBuilder::new()
            .set("A", 25.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 5.0)
            .complement(945.0) // universe = 1000
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Ellipse>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");

        let area = container.width() * container.height();
        assert!(
            (area - 1000.0).abs() / 1000.0 < 0.05,
            "container area {} ≠ 1000 (within 5%)",
            area
        );
        let union_target = 55.0;
        assert!(
            area > 10.0 * union_target,
            "container area {} should be much larger than union {}",
            area,
            union_target
        );
    }

    /// Square + complement: end-to-end smoke that the S5 clipping path is
    /// wired all the way through `Fitter::<Square>::fit`. Universe size
    /// (set sizes + intersection + complement) should match the fitted
    /// container area to within a generous tolerance.
    #[test]
    fn fit_squares_with_complement_container_matches_universe() {
        use crate::geometry::shapes::Square;
        use crate::geometry::traits::Area;

        let spec = DiagramSpecBuilder::new()
            .set("A", 25.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 5.0)
            .complement(50.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Square>::new(&spec).seed(42).fit().unwrap();
        let container = layout
            .container()
            .expect("complement spec carries container");
        let universe = 25.0 + 25.0 + 5.0 + 50.0;
        assert!(
            (container.area() - universe).abs() / universe < 0.1,
            "container area {} should be near universe {}",
            container.area(),
            universe,
        );
    }

    /// End-to-end inclusive input × complement: union sizes go through
    /// inclusion-exclusion in the builder; the container's area should
    /// match the universe (sum of exclusive areas + complement).
    #[test]
    fn fit_circles_inclusive_input_with_complement() {
        // Inclusive: |A| = 30, |B| = 25, |A∪B| = 45.
        // → exclusive: A only = 30 - intersection, B only = 25 - intersection,
        //   intersection = 30 + 25 - 45 = 10.
        //   Universe = 30 + 25 - 10 + complement = 45 + 30 = 75.
        let spec = DiagramSpecBuilder::new()
            .set("A", 30.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 10.0)
            .complement(30.0)
            .input_type(crate::InputType::Inclusive)
            .build()
            .unwrap();

        let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");
        let universe = container.width() * container.height();
        let expected = 45.0 + 30.0;
        assert!(
            (universe - expected).abs() / expected < 0.05,
            "container area {} should match universe {} within 5%",
            universe,
            expected,
        );
    }

    /// Container-aware normalize: post-fit the container is centred at the
    /// origin (the fitter applies the container-aware path automatically),
    /// and a follow-up `Layout::normalize` is a stable no-op.
    #[test]
    fn fit_centres_container_at_origin_and_normalize_is_idempotent() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 25.0)
            .set("B", 25.0)
            .intersection(&["A", "B"], 5.0)
            .complement(50.0)
            .input_type(crate::InputType::Exclusive)
            .build()
            .unwrap();

        let mut layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
        let container = layout.container().expect("container present");
        assert!(
            container.center().x().abs() < 1e-9 && container.center().y().abs() < 1e-9,
            "container should be centred at origin post-fit, got ({}, {})",
            container.center().x(),
            container.center().y(),
        );

        let snapshot_shapes = layout.shapes().to_vec();
        let snapshot_container = *layout.container().unwrap();
        layout.normalize(0.05);
        assert_eq!(layout.container().copied(), Some(snapshot_container));
        assert_eq!(layout.shapes(), snapshot_shapes.as_slice());
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

/// Pins the CMA-ES-on-LM regression-escape for `issue92_3_set_dropped_pair`.
///
/// At default `LevenbergMarquardt` settings this spec lands at the same
/// `1.309e-4` loss / `5.388e-3 diag_error` from every seed in `QUALITY_SEEDS`
/// — LM gets stuck in a basin where the small `A&B=12` pair sits between
/// the dominant `A&C=459` / `B&C=703` regions and cannot be moved without
/// breaking the larger overlaps. CMA-ES's bounded global step escapes this
/// trap (median ~1e-29 across seeds in `examples/quality_report`). If this
/// test starts failing the global-escape stage has regressed.
#[test]
#[ignore = "slow regression coverage"]
fn test_cmaes_lm_escapes_issue92_dropped_pair() {
    use crate::fitter::{Fitter, Optimizer};
    use crate::geometry::shapes::Ellipse;
    use crate::spec::{DiagramSpecBuilder, InputType};

    // Same numbers as `corpus::issue92_3_set_dropped_pair`.
    let spec = DiagramSpecBuilder::new()
        .set("A", 164.0)
        .set("B", 561.0)
        .set("C", 166.0)
        .intersection(&["A", "B"], 12.0)
        .intersection(&["A", "C"], 459.0)
        .intersection(&["B", "C"], 703.0)
        .intersection(&["A", "B", "C"], 162.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    let layout = Fitter::<Ellipse>::new(&spec)
        .optimizer(Optimizer::CmaEsLm)
        .seed(1)
        .fit()
        .unwrap();
    // Default LM landed at 1.309e-4 across every seed. CMA-ES + LM polish
    // routinely reaches < 1e-10 on this spec; pin loosely at 1e-6 so
    // small numerical drift doesn't false-positive.
    assert!(
        layout.loss() < 1e-6,
        "CmaEsLm loss = {:e} (expected < 1e-6 — global step regressed)",
        layout.loss()
    );
}
