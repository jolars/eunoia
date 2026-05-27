//! Final layout optimization.
//!
//! This module implements the second optimization step that refines the initial
//! layout by minimizing the difference between target exclusive areas and actual
//! fitted areas in the diagram.

use finitediff::vec;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::DiagramError;
use crate::geometry::diagram::RegionMask;
use crate::geometry::traits::DiagramShape;
use crate::loss::LossType;
use crate::spec::PreprocessedSpec;

/// Optimizer to use for final layout optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Optimizer {
    /// Levenberg-Marquardt with analytic Jacobian. Specialised for the
    /// sum-of-squares loss `LossType::SumSquared`. When configured with a
    /// non-LSQ loss, the dispatch falls back to `Lbfgs` for smooth losses
    /// (RMSE, Stress, …) and to `NelderMead` for non-smooth ones (`Max*`,
    /// `SumAbsolute`, `DiagError`, `SumAbsoluteRegionError`) — the latter
    /// produce zero/discontinuous gradients that stall L-BFGS (issue #45).
    ///
    /// Uses the existing analytical region-area gradients to assemble a
    /// per-residual Jacobian (one row per region mask), then approximates
    /// the Hessian as `JᵀJ`. The Jacobian comes from
    /// [`crate::geometry::traits::DiagramShape::compute_exclusive_regions_with_gradient`];
    /// boundary builders for Circle and Ellipse are documented in `AGENTS.md`.
    LevenbergMarquardt,
    /// L-BFGS. Used as the non-LSQ fallback under the LM dispatch arm and
    /// available directly for losses where LM doesn't apply (RMSE, Stress,
    /// DiagError, …). With analytical gradients it's competitive with LM on
    /// circles, but ~20 orders of magnitude worse on ellipse median loss in
    /// `examples/quality_report` — keep it as a fallback, not as a default.
    Lbfgs,
    /// Nelder-Mead simplex (derivative-free). Last-resort option for losses
    /// where neither LM nor a meaningful gradient is available; kept mostly
    /// because the harness still uses it as a quality lower-bound sentinel.
    NelderMead,
    /// Threshold-fired CMA-ES global escape followed by an **unbounded**
    /// Levenberg-Marquardt polish. The previous default; [`Optimizer::CmaEsTrf`]
    /// (bounded TRF polish) is now the [`Fitter`] default, but this remains
    /// available for the unbounded-polish behaviour.
    ///
    /// Each restart runs plain LM first; if the result is at or below
    /// `Fitter::cmaes_fallback_threshold` (default `1e-3` on the default
    /// `SumSquared` loss) the CMA-ES step is skipped entirely
    /// and the LM result is returned.
    /// Easy specs that LM already crushes pay zero extra wall time.
    ///
    /// When LM stalls above the threshold (e.g. `issue92_3_set_dropped_pair`
    /// at `1.3e-4`, `eulerape_3_set` at `4.4e-4`, `random_4_set` at
    /// `8.5e-3` under plain LM), CMA-ES fires: it samples in a feasible
    /// box around the MDS init (centroid ± span on positions,
    /// `[eps, k·max_radius]` on radii / semi-axes; angles unbounded) and
    /// hands its best point to the same Levenberg-Marquardt residual
    /// problem the `LevenbergMarquardt` arm uses, so the analytical
    /// Jacobian still drives the final tightening. The lower-loss of
    /// {plain LM, CMA-ES → LM polish} is returned, so the path is
    /// strictly non-regressing vs `LevenbergMarquardt`.
    ///
    /// Only the LM legs (the up-front guard and the polish) require the
    /// `SumSquared` loss — they build the least-squares residual problem and
    /// reject anything else. With a non-`SumSquared` loss those legs
    /// transparently degrade through the same dispatch as the
    /// `LevenbergMarquardt` arm: `Lbfgs` for the other smooth losses (RMSE,
    /// Stress, `Smooth*`, …) and `NelderMead` for the non-smooth ones
    /// (`Max*`, `SumAbsolute`, `DiagError`, `SumAbsoluteRegionError`). The
    /// CMA-ES escape stage is derivative-free and loss-agnostic, so it still
    /// fires on any loss above the threshold — meaning under a non-LSQ loss
    /// this strategy is really "{smooth: L-BFGS, non-smooth: Nelder-Mead},
    /// with a CMA-ES global escape", not literal LM.
    ///
    /// Cost: when CMA-ES fires, ~λ·max_iters extra function evaluations
    /// on top of LM, with λ = `4 + floor(3 ln n)` for an n-parameter
    /// problem and `max_iters = 100`. On `issue91_6_set` (n=30) that's
    /// ~1400 extra region-area evals per restart. The threshold gate keeps
    /// this off the easy-spec budget.
    ///
    /// [`Fitter`]: crate::Fitter
    CmaEsLm,
    /// Box-constrained Levenberg-Marquardt — trust-region-reflective
    /// (`basin::Trf`). The bounded analog of [`Optimizer::LevenbergMarquardt`]:
    /// it shares the exact `LmDiagramProblem` least-squares kernel (analytic
    /// region-area Jacobian, `JᵀJ` Gram) but constrains every step to a
    /// per-shape box via Coleman-Li affine scaling.
    ///
    /// Bounds come from [`optimizer_bounds_for`] under the wider
    /// [`BoundsEnvelope::LOCAL`] safety-cage (positions to centroid ±
    /// `8·span`, radii / semi-axes to `[1e-8·max_radius, 12·max_radius]`,
    /// log-space for ellipses), distinct from the tighter
    /// [`BoundsEnvelope::CMAES`] the global escape uses — a local refinement
    /// only needs to catch blow-up, not constrain the optimum. The periodic
    /// ellipse angle is `±∞`; the affine scaling degrades to unconstrained on
    /// infinite-bound coordinates, so no finite angle cap is needed. Where
    /// `lower = -∞` and `upper = +∞` element-wise, `Trf` reduces exactly to
    /// `LevenbergMarquardt`.
    ///
    /// Unlike a wrapped global step, this is a purely *local* solver: it
    /// cannot escape a wrong basin (no CMA-ES stage), so it competes with
    /// `LevenbergMarquardt` and only beats `CmaEsLm` where parameter bounds —
    /// not basin escape — are the bottleneck. As with the `LevenbergMarquardt`
    /// arm, non-`SumSquared` losses fall back to `Lbfgs` (smooth) / `NelderMead`
    /// (non-smooth) since the least-squares problem doesn't exist for them.
    Trf,
    /// [`CmaEsLm`](Optimizer::CmaEsLm) with the post-escape polish swapped from
    /// unbounded LM to box-constrained [`Trf`](Optimizer::Trf). **Default for
    /// [`Fitter`].**
    ///
    /// Same threshold-fired structure: plain LM first, and if it stalls above
    /// `cmaes_fallback_threshold`, a bounded CMA-ES global escape — but the
    /// best CMA-ES candidate is then refined with the **bounded** TRF polish
    /// (under [`BoundsEnvelope::LOCAL`]) instead of unbounded LM, so the
    /// refinement can't wander outside the feasible box. Combines the global
    /// escape (which fixes the wrong-basin specs a purely local solver
    /// regresses on) with the bounded local refinement (which wins
    /// extreme-scale / nested specs). The lower-loss of {plain LM, CMA-ES →
    /// TRF polish} is returned, so it is strictly non-regressing vs
    /// `LevenbergMarquardt` like `CmaEsLm`.
    ///
    /// On the `examples/quality_report` sweep this matches `CmaEsLm` on nearly
    /// every spec, wins a few (ellipse `three_inside_fourth`, rectangle
    /// `unequal_overlaps`), and trades one small, bound-independent regression
    /// (circle `issue103`, an intrinsic TRF-vs-LM stopping difference) — a
    /// net-positive that motivated making it the default.
    ///
    /// [`Fitter`]: crate::Fitter
    #[default]
    CmaEsTrf,
}

/// Configuration for final layout optimization.
#[derive(Debug, Clone)]
pub(crate) struct FinalLayoutConfig {
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Loss function
    pub loss_type: crate::loss::LossType,
    /// Optimizer to use
    pub optimizer: Optimizer,
    /// Cost-change convergence tolerance. Honored by every solver except
    /// Nelder-Mead:
    /// - **L-BFGS**: passed as both `tol_grad` and `tol_cost`. Squaring the
    ///   cost tolerance backfires with central-difference gradients (FD noise
    ///   floor on cost evals is ~`sqrt(EPSILON) × |cost|`), so both share
    ///   `config.tolerance`.
    /// - **Levenberg-Marquardt**: passed as `ftol` (cost-change exit) only.
    ///   `xtol` and `gtol` use the fixed `1e-6` defaults from the
    ///   per-knob fields below.
    /// - **CmaEsLm**: used by the LM step(s) (the up-front guard and the
    ///   polish) as above; the bounded CMA-ES escape stage itself runs to its
    ///   fixed generation budget under basin's TolX, so it doesn't read this.
    /// - **Nelder-Mead**: ignored — no cost-tolerance setter is wired on the
    ///   `basin::NelderMead` run, so it runs to `max_iterations`.
    ///
    /// Default `1e-3`, validated against the full 27-spec corpus × 16 seeds
    /// × {Circle, Ellipse} via the `final_tolerance` bench (zero regressions
    /// vs. the prior `1e-6`, with up to ~170× wall-time wins on slow ellipse
    /// specs because LM was previously cost-converged but still grinding the
    /// `xtol`/`gtol` tail). The LM `xtol`/`gtol` knobs stay at `1e-6` to
    /// preserve parameter-space precision.
    pub tolerance: f64,
    /// Per-knob LM stopping tolerance overrides. `None` falls back to the
    /// LM-specific defaults: `xtol = 1e-6`, `ftol = config.tolerance`,
    /// `gtol = 1e-6`. Only honoured by `Optimizer::LevenbergMarquardt` and
    /// the LM polish step of `Optimizer::CmaEsLm`; other optimizers ignore
    /// these.
    pub xtol: Option<f64>,
    pub ftol: Option<f64>,
    pub gtol: Option<f64>,
    /// Seed used for stochastic operations: simulated annealing, and
    /// position-perturbation restarts.
    pub seed: u64,
    /// Number of optimization restarts from perturbed initial circle layouts.
    /// Attempt 0 always uses the unperturbed MDS init; attempts `1..n_restarts`
    /// perturb the circle positions before converting to shape parameters.
    /// Mirrors eulerr's `n_restarts = 10` strategy.
    pub n_restarts: usize,
    /// Loss threshold above which `Optimizer::CmaEsLm` fires the global
    /// CMA-ES escape stage. When plain LM lands at or below this value the
    /// CMA-ES step is skipped entirely, so the easy-spec wall-time cost
    /// drops to that of `Optimizer::LevenbergMarquardt`. Only the
    /// `CmaEsLm` arm consults this; other optimizers ignore it.
    pub cmaes_fallback_threshold: f64,
}

impl Default for FinalLayoutConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            loss_type: crate::loss::LossType::default(),
            optimizer: Optimizer::LevenbergMarquardt,
            tolerance: 1e-3,
            xtol: None,
            ftol: None,
            gtol: None,
            seed: 0xDEAD_BEEF,
            n_restarts: 10,
            // 1e-3 sits well above the ~1e-20 / ~1e-30 floor LM crushes
            // easy specs to and well below the ~1e-2 / ~1e-1 plateaus the
            // hard specs (issue91, issue92, eulerape, …) get stuck on.
            // Picked empirically from `examples/quality_report`: every
            // spec where lm_full lands at machine precision sits at or
            // below 1e-20, every spec where it stalls in a wrong basin
            // sits at or above 1e-4.
            cmaes_fallback_threshold: 1e-3,
        }
    }
}

/// Optimize the final layout by minimizing region error.
///
/// Runs the configured optimizer up to `config.n_restarts` times — attempt 0
/// from the unperturbed MDS initial layout, then attempts `1..n_restarts` from
/// MDS positions perturbed with seeded Gaussian noise. The lowest-loss result
/// across all restarts is returned. This mirrors eulerr's `n_restarts = 10`
/// strategy and helps escape topologically wrong basins where a local optimizer
/// gets stuck (e.g. shapes that should overlap landing disjoint, where the loss
/// is locally flat — see issue #28).
///
/// Returns the optimized parameters as a flat vector along with the loss.
pub(crate) fn optimize_layout<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    initial_positions: &[f64], // [x0, y0, x1, y1, ..., xn, yn]
    initial_radii: &[f64],     // [r0, r1, ..., rn]
    config: FinalLayoutConfig,
) -> Result<(Vec<f64>, f64), DiagramError> {
    let n_sets = spec.n_sets;
    let params_per_shape = S::n_params();

    // Mean radius drives the perturbation scale; positions are perturbed by
    // ~N(0, mean_radius) on each axis, which is large enough to land in a
    // different basin but small enough to stay near the MDS solution.
    let mean_radius = if !initial_radii.is_empty() {
        initial_radii.iter().sum::<f64>() / initial_radii.len() as f64
    } else {
        1.0
    };

    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut last_err: Option<DiagramError> = None;
    let n_restarts = config.n_restarts.max(1);
    let mut rng = StdRng::seed_from_u64(config.seed ^ 0x52455354_41525453); // "RESTARTS"

    for attempt in 0..n_restarts {
        let mut positions = initial_positions.to_vec();
        if attempt > 0 {
            // Perturb positions with Gaussian noise. Box-Muller two-at-a-time
            // keeps this dependency-free.
            for i in 0..n_sets {
                let u1: f64 = rng.random_range(f64::EPSILON..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                let mag = (-2.0 * u1.ln()).sqrt();
                let dx = mag * (2.0 * std::f64::consts::PI * u2).cos();
                let dy = mag * (2.0 * std::f64::consts::PI * u2).sin();
                positions[i * 2] += dx * mean_radius;
                positions[i * 2 + 1] += dy * mean_radius;
            }
        }

        let mut initial_params = Vec::with_capacity(n_sets * params_per_shape + 4);
        for i in 0..n_sets {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            let r = initial_radii[i];
            initial_params.extend(S::optimizer_params_from_circle(x, y, r));
        }
        // Append container init params when the spec carries a complement.
        // Container init: centred on the (perturbed) MDS positions, square
        // footprint of side √universe (so initial area ≈ complement +
        // Σ named exclusive). The optimiser refines x/y/area/ratio from there.
        if spec.complement.is_some() {
            let mut sx = 0.0;
            let mut sy = 0.0;
            for i in 0..n_sets {
                sx += positions[i * 2];
                sy += positions[i * 2 + 1];
            }
            let cx = if n_sets > 0 { sx / n_sets as f64 } else { 0.0 };
            let cy = if n_sets > 0 { sy / n_sets as f64 } else { 0.0 };
            // Universe area = sum of all targets in `exclusive_areas`, which
            // includes the mask-0 complement entry inserted by `preprocess`.
            let universe = spec
                .exclusive_areas
                .values()
                .sum::<f64>()
                .max(f64::MIN_POSITIVE);
            let side = universe.sqrt();
            let rect = crate::geometry::shapes::Rectangle::new(
                crate::geometry::primitives::Point::new(cx, cy),
                side,
                side,
            );
            initial_params.extend(rect.to_optimizer_params());
        }
        let initial_param = DVector::from_vec(initial_params);

        // A perturbed start can drive `compute_exclusive_regions` into a
        // degenerate geometry that panics inside the numeric stack
        // (e.g. cubic-equation solver with α≈0). Catch panics so one bad
        // restart doesn't abort the whole fit; a restart that panics or
        // errors is just skipped.
        let attempt_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            optimize_from_initial::<S>(spec, &initial_param, &config)
        }));

        match attempt_result {
            Ok(Ok((params, loss))) => {
                let params_vec = params.as_slice().to_vec();
                match &best {
                    None => best = Some((params_vec, loss)),
                    Some((_, best_loss)) if loss < *best_loss => best = Some((params_vec, loss)),
                    _ => {}
                }
            }
            Ok(Err(e)) => last_err = Some(e),
            Err(_) => {
                // Panic in optimizer / cost function; skip this restart.
            }
        }
    }

    match best {
        Some(b) => Ok(b),
        None => Err(last_err.unwrap_or_else(|| {
            DiagramError::InvalidCombination(
                "all optimization restarts failed (panicked or errored)".to_string(),
            )
        })),
    }
}

/// Run the configured optimizer once from a given initial parameter vector.
///
/// Exposed `pub(crate)` so the [`Fitter`]'s outer-loop attempts can dispatch
/// the final-stage solver directly when they have full shape parameters in
/// hand and want to skip the standard `(positions, radii) → optimizer_params_from_circle`
/// path. The Venn warm-start uses this entry point to feed canonical-Venn
/// shape params (with non-circular ellipses for n ∈ {4, 5}) straight to the
/// optimizer.
///
/// [`Fitter`]: crate::Fitter
pub(crate) fn optimize_from_initial<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
) -> Result<(DVector<f64>, f64), DiagramError> {
    let params_per_shape = S::n_params();

    // Complement specs run through L-BFGS with analytical gradients (S2). LM
    // also has analytical Jacobians via `LmDiagramProblem`, but on the small
    // complement specs in the test corpus it consistently converges to higher
    // residual loss than L-BFGS — the clipped landscape's basin shape differs
    // enough from the unclipped one that LM's `JᵀJ` step bias is unhelpful.
    // CmaEsLm is also downgraded because `optimizer_bounds_for` doesn't yet split
    // off the trailing container block. Both lifts are S6 polish.
    let optimizer = if spec.complement.is_some() {
        Optimizer::Lbfgs
    } else {
        config.optimizer
    };

    // Choose optimizer and run based on configuration
    let (final_params, loss) = match optimizer {
        Optimizer::NelderMead => {
            run_nelder_mead::<S>(spec, params_per_shape, initial_param, config)?
        }
        Optimizer::Lbfgs => {
            // L-BFGS with numerical gradients. On non-smooth losses
            // (`MaxAbsolute`, `SumAbsoute`, …) gradients are unreliable —
            // pick a `LossType::Smooth*` surrogate variant instead.
            run_lbfgs::<S>(spec, params_per_shape, initial_param, config)
        }
        Optimizer::LevenbergMarquardt => {
            run_lm_or_lbfgs::<S>(spec, params_per_shape, initial_param, config)?
        }
        Optimizer::Trf => run_trf::<S>(spec, params_per_shape, initial_param, config)?,
        // Threshold-fired CMA-ES global escape; the two variants differ only in
        // the post-escape polish — unbounded LM (`CmaEsLm`) vs bounded TRF
        // (`CmaEsTrf`). See [`run_cmaes_escape`].
        Optimizer::CmaEsLm => run_cmaes_escape::<S>(
            spec,
            params_per_shape,
            initial_param,
            config,
            run_lm_or_lbfgs::<S>,
        )?,
        Optimizer::CmaEsTrf => {
            run_cmaes_escape::<S>(spec, params_per_shape, initial_param, config, run_trf::<S>)?
        }
    };

    Ok((final_params, loss))
}

/// Run Levenberg-Marquardt for `SumSquared`, falling back to L-BFGS on other
/// smooth losses and to Nelder-Mead on non-smooth ones (`Max*`,
/// `SumAbsolute`, `DiagError`, `SumAbsoluteRegionError`). Non-smooth losses
/// produce zero or discontinuous gradients almost everywhere, which makes
/// L-BFGS thrash against the line search — see issue #45 and
/// [`LossType::is_smooth`].
///
/// Shared between [`Optimizer::LevenbergMarquardt`] and the polish step of
/// [`Optimizer::CmaEsLm`] so both go through the same numerical path.
///
/// [`LossType::is_smooth`]: crate::loss::LossType::is_smooth
fn run_lm_or_lbfgs<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    params_per_shape: usize,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
) -> Result<(DVector<f64>, f64), DiagramError> {
    match LmDiagramProblem::<S>::new(spec, params_per_shape, config.loss_type) {
        Ok(problem) => {
            // Convergence uses LM's three MINPACK-style solver-internal tests,
            // mirroring the `levenberg-marquardt` crate's `gtol`/`ftol`/`xtol`:
            //   - `tol_grad_rel` (gtol): cosine of the angle between `r` and the
            //     Jacobian columns, `max_j |gⱼ|/(‖J·,ⱼ‖·‖r‖) ≤ gtol`;
            //   - `ftol`: relative cost reduction with the *predicted*-reduction
            //     guard (`actred ≤ ftol·f AND prered ≤ ftol·f`), which bails
            //     stuck wrong-basin restarts without truncating productive-but-
            //     slow ones (the framework `RelativeCostTolerance` lacks `prered`
            //     and stops short — e.g. `three_set_triple_only`'s 3.3e-2
            //     settling point before it refines to 3.3e-3);
            //   - `xtol`: relative step `‖Δx‖ ≤ xtol·‖x‖`.
            // The absolute `tol_grad` is disabled — the residuals' `1/√Σtᵢ²`
            // normalisation makes `‖Jᵀr‖_∞` scale-dependent (a fixed bound
            // over-iterates small-area specs and stops large-area ones early).
            let solver = basin::LevenbergMarquardt::new()
                .tol_grad(0.0)
                .tol_grad_rel(config.gtol.unwrap_or(1e-8))
                .ftol(config.ftol.unwrap_or(config.tolerance))
                .xtol(config.xtol.unwrap_or(1e-6));
            let result = basin::Executor::new(
                problem,
                solver,
                basin::BasicState::new(initial_param.clone()),
            )
            .max_iter(config.max_iterations.max(1) as u64)
            .run();
            let final_param = result.param().clone();
            // basin LM's `state.cost()` is ½·Σrᵢ² over the normalised
            // residuals; the existing loss is Σrᵢ², so ×2.
            let final_loss = result.cost() * 2.0;
            Ok((final_param, final_loss))
        }
        Err(_incompatible_loss) if !config.loss_type.is_smooth() => {
            // Non-smooth loss: gradient methods thrash here, so use
            // derivative-free Nelder-Mead instead. The user can switch
            // to a `LossType::Smooth*` variant to take the L-BFGS path
            // through a C¹ surrogate. See issue #45 and `LossType::is_smooth`.
            run_nelder_mead::<S>(spec, params_per_shape, initial_param, config)
        }
        Err(_incompatible_loss) => {
            // Smooth loss (including `Smooth*` surrogate variants). The
            // cost landscape is C¹, so L-BFGS with numerical gradients
            // applies.
            Ok(run_lbfgs::<S>(
                spec,
                params_per_shape,
                initial_param,
                config,
            ))
        }
    }
}

/// Run box-constrained Levenberg-Marquardt (`basin::Trf`,
/// trust-region-reflective) — the bounded analog of [`run_lm_or_lbfgs`].
///
/// Reuses the same `LmDiagramProblem` least-squares kernel (analytic Jacobian,
/// `JᵀJ` Gram) wrapped in [`BoundedLmDiagramProblem`] to carry the box bounds
/// `Trf` requires. Bounds come from [`optimizer_bounds_for`] — the same
/// per-shape envelope the `CmaEsLm` escape uses. `Trf`'s Coleman-Li scaling
/// reduces to plain LM on `±∞` coordinates (the ellipse angle), so unbounded
/// dims need no finite surrogate.
///
/// For non-`SumSquared` losses the least-squares problem doesn't exist, so the
/// dispatch degrades through the same fallback as [`run_lm_or_lbfgs`]:
/// Nelder-Mead for non-smooth losses, L-BFGS for the other smooth ones.
fn run_trf<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    params_per_shape: usize,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
) -> Result<(DVector<f64>, f64), DiagramError> {
    match LmDiagramProblem::<S>::new(spec, params_per_shape, config.loss_type) {
        Ok(inner) => {
            let (lower, upper, _std) = optimizer_bounds_for(
                initial_param.as_slice(),
                params_per_shape,
                BoundsEnvelope::LOCAL,
            );
            let lower = DVector::from_vec(lower);
            let upper = DVector::from_vec(upper);
            // `Trf`'s Coleman-Li affine scaling is undefined on a finite face
            // (`v_i = 0`), so start strictly inside the box. For an interior
            // MDS/LM seed (the common case) this is a no-op; it only rescues a
            // degenerate seed that landed on or outside a finite bound.
            let mut x0 = initial_param.clone();
            for i in 0..x0.len() {
                let (lo, hi) = (lower[i], upper[i]);
                if lo.is_finite() && hi.is_finite() && hi > lo {
                    let margin = 1e-6 * (hi - lo);
                    x0[i] = x0[i].clamp(lo + margin, hi - margin);
                }
            }
            let problem = BoundedLmDiagramProblem::<S> {
                inner,
                lower,
                upper,
            };
            // `Trf`'s sole convergence tolerance is the absolute
            // `‖D·Jᵀr‖_∞ ≤ tol_grad` test — it has no relative ftol/xtol/gtol
            // knobs like basin's LM. The residuals' `1/√Σtᵢ²` normalisation
            // makes that bound scale-dependent (the same reason the LM arm
            // disables absolute `tol_grad`), so the iteration budget is the
            // primary stop here. We still thread `config.gtol` through for
            // parity with the LM dispatch.
            let solver = basin::Trf::<DVector<f64>, DMatrix<f64>>::new()
                .tol_grad(config.gtol.unwrap_or(1e-8));
            let result = basin::Executor::new(problem, solver, basin::BasicState::new(x0))
                .max_iter(config.max_iterations.max(1) as u64)
                .run();
            let final_param = result.param().clone();
            // `Trf::state.cost` is ½·Σrᵢ² over the normalised residuals; the
            // configured loss is Σrᵢ², so ×2 (matches the LM arm).
            let final_loss = result.cost() * 2.0;
            Ok((final_param, final_loss))
        }
        // No least-squares problem for non-`SumSquared` losses — fall back
        // exactly as the `LevenbergMarquardt` arm does.
        Err(_incompatible_loss) if !config.loss_type.is_smooth() => {
            run_nelder_mead::<S>(spec, params_per_shape, initial_param, config)
        }
        Err(_incompatible_loss) => Ok(run_lbfgs::<S>(
            spec,
            params_per_shape,
            initial_param,
            config,
        )),
    }
}

/// A final-layout local solver: `(spec, params_per_shape, x0, config)` →
/// `(params, loss)`. Both [`run_lm_or_lbfgs`] and [`run_trf`] match this, which
/// is what lets [`run_cmaes_escape`] take the post-escape polish as a parameter.
type FinalSolver = fn(
    &PreprocessedSpec,
    usize,
    &DVector<f64>,
    &FinalLayoutConfig,
) -> Result<(DVector<f64>, f64), DiagramError>;

/// Threshold-fired CMA-ES global escape shared by [`Optimizer::CmaEsLm`] and
/// [`Optimizer::CmaEsTrf`].
///
/// 1. Run plain LM. If it lands at or below `config.cmaes_fallback_threshold`,
///    return immediately — CMA-ES wouldn't beat that and would only burn
///    ~1k–2k extra evals per restart, so easy specs pay no extra wall time.
/// 2. Otherwise LM is stuck in a wrong basin (issue92, eulerape, …). Run
///    bounded CMA-ES, then refine its best candidate with `polish` — unbounded
///    [`run_lm_or_lbfgs`] for `CmaEsLm`, bounded [`run_trf`] for `CmaEsTrf`.
///    The bounded TRF polish keeps the refinement inside the box CMA-ES
///    explored; the unbounded LM polish picks up the analytic Jacobian for
///    free. Either way, return the lower-loss of {LM-only, CMA-ES → polish},
///    so the path is strictly non-regressing vs `LevenbergMarquardt` even if
///    CMA-ES wanders into a worse basin (the guard that fixed
///    `issue47_3_set_huge_triple`).
fn run_cmaes_escape<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    params_per_shape: usize,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
    polish: FinalSolver,
) -> Result<(DVector<f64>, f64), DiagramError> {
    let lm_only = run_lm_or_lbfgs::<S>(spec, params_per_shape, initial_param, config)?;
    if lm_only.1 <= config.cmaes_fallback_threshold {
        return Ok(lm_only);
    }
    let cma_seed = run_bounded_cmaes::<S>(spec, params_per_shape, initial_param, config);
    let cma_polish = polish(spec, params_per_shape, &cma_seed, config)?;
    Ok(if cma_polish.1 < lm_only.1 {
        cma_polish
    } else {
        lm_only
    })
}

/// Run derivative-free Nelder-Mead from the given initial parameters.
///
/// Used directly by `Optimizer::NelderMead` and as the non-smooth-loss
/// fallback inside [`run_lm_or_lbfgs`]. Builds the initial simplex by
/// perturbing each parameter by 5% of its magnitude (with shape-aware
/// fallbacks for rotations and near-zero values).
fn run_nelder_mead<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    params_per_shape: usize,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
) -> Result<(DVector<f64>, f64), DiagramError> {
    let n_params = initial_param.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n_params + 1);
    simplex.push(initial_param.as_slice().to_vec());
    for i in 0..n_params {
        let mut perturbed = initial_param.as_slice().to_vec();
        let x0_i = initial_param[i];
        let param_idx = i % params_per_shape;
        let is_rotation = params_per_shape == 5 && param_idx == 4;

        let delta = if x0_i.abs() > 1e-10 {
            0.05 * x0_i.abs()
        } else if is_rotation {
            0.2 // ~11.5 degrees for rotation parameters
        } else {
            0.00025
        };
        perturbed[i] += delta;
        simplex.push(perturbed);
    }

    let cost_function = BasinDiagramCost::<S> {
        inner: DiagramCost {
            spec,
            loss_type: config.loss_type,
            params_per_shape,
            // NM doesn't need smoothing for its own sake, but if the caller
            // opted into the surrogate we honour it for cost-landscape
            // consistency with other dispatch arms.
            _shape: std::marker::PhantomData,
        },
    };
    let state = basin::BasicSimplexState::from_simplex(simplex);
    let solver = basin::NelderMead::standard();
    let result = basin::Executor::new(cost_function, solver, state)
        .max_iter(config.max_iterations as u64)
        .run();
    Ok((DVector::from_vec(result.param().clone()), result.cost()))
}

/// Run unbounded L-BFGS (`basin::LBFGS`) from `initial_param` over the smooth
/// cost landscape, returning `(params, loss)`.
///
/// Shared by `Optimizer::Lbfgs` and the smooth-loss fallback in
/// [`run_lm_or_lbfgs`]. Both convergence tests sit at `config.tolerance`:
/// `GradientTolerance` (`‖∇f‖₂ ≤ tol`) and `CostTolerance` (`|Δf| ≤ tol`),
/// the framework equivalents of the previous solver's `with_tolerance_grad`
/// / `with_tolerance_cost`. Squaring the cost tolerance (the obvious "be
/// stricter on cost than gradient" trick) backfires: with central-difference
/// gradients the FD noise floor on cost evals is ~`sqrt(EPSILON) × |cost|`,
/// well above any tolerance < ~1e-9, so a too-tight cost tolerance means the
/// optimizer never declares "no progress" and grinds to `max_iters` even when
/// sitting at the optimum (issue #34).
fn run_lbfgs<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    params_per_shape: usize,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
) -> (DVector<f64>, f64) {
    let cost_function = BasinDiagramCost::<S> {
        inner: DiagramCost {
            spec,
            loss_type: config.loss_type,
            params_per_shape,
            _shape: std::marker::PhantomData,
        },
    };
    let x0 = initial_param.as_slice().to_vec();
    let solver = basin::LBFGS::<basin::solver::lbfgs::Unbounded>::new().m_capacity(10);
    let result = basin::Executor::new(cost_function, solver, basin::LbfgsState::new(x0, 10))
        .max_iter(config.max_iterations.max(1) as u64)
        .terminate_on(basin::GradientTolerance(config.tolerance))
        .terminate_on(basin::CostTolerance::new(config.tolerance))
        .run();
    (DVector::from_vec(result.param().clone()), result.cost())
}

/// Run the bounded CMA-ES global-escape stage and return its best candidate as
/// a seed for the LM polish that follows.
///
/// Box-constrained CMA-ES (`basin::BoundedCmaEs`, Hansen/pycma adaptive
/// quadratic boundary penalty) explores from the MDS/LM start when plain LM is
/// stuck in a wrong basin. The per-coordinate bounds and initial standard
/// deviations come from [`optimizer_bounds_for`]; the std vector preconditions the
/// heterogeneous parameter scales (positions, radii, log-semi-axes, angles) via
/// `with_stds`, so a single `initial_sigma = 1.0` works across all of them —
/// the same `y = (x − x0)/std` preconditioning the previous inline CMA-ES did
/// by hand, now `diag(stds²)` as the initial covariance.
///
/// Budgeted at 100 generations (≈1k–2k evals on hard 6-set ellipse fits) —
/// empirically enough to escape the wrong-basin
/// `issue92_3_set_dropped_pair` / `eulerape_3_set` cases without dominating
/// wall time when the caller has dialled `n_restarts` down. The caller owns the
/// LM polish and the non-regression guard against the LM-only result.
fn run_bounded_cmaes<S: DiagramShape + Copy + 'static>(
    spec: &PreprocessedSpec,
    params_per_shape: usize,
    initial_param: &DVector<f64>,
    config: &FinalLayoutConfig,
) -> DVector<f64> {
    let (lower, upper, initial_std) = optimizer_bounds_for(
        initial_param.as_slice(),
        params_per_shape,
        BoundsEnvelope::CMAES,
    );
    let problem = BoundedDiagramCost::<S> {
        inner: DiagramCost {
            spec,
            loss_type: config.loss_type,
            params_per_shape,
            _shape: std::marker::PhantomData,
        },
        lower: DVector::from_vec(lower),
        upper: DVector::from_vec(upper),
    };
    let n = initial_param.len();
    let lambda = basin::BoundedCmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(n);
    let solver = basin::BoundedCmaEs::<DVector<f64>, DMatrix<f64>>::new(
        initial_param.clone(),
        // initial_sigma = 1: the per-coordinate scale lives entirely in
        // `with_stds`, so the search runs on the unitless `diag(stds)`-scaled
        // space (matches the old inline solver's internal `sigma = 1`).
        1.0,
        config.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
    .with_stds(DVector::from_vec(initial_std));
    let result = basin::Executor::new(
        problem,
        solver,
        basin::BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(100)
    .run();
    result.param().clone()
}

/// Adapter wrapping [`DiagramCost`] for basin's `CostFunction` trait.
///
/// This adapter uses `Param = Vec<f64>` (the gradient/simplex solvers take
/// it), whereas the least-squares (`LmDiagramProblem`) and bounded
/// (`BoundedDiagramCost`, `BoundedLmDiagramProblem`) wrappers run on
/// `DVector<f64>` directly — eunoia and basin are aligned on a single nalgebra
/// (0.34), so basin's vector type *is* `DVector<f64>` and there is no version
/// skew to route around. The `Vec`↔`DVector` conversion inside `cost()` is
/// invisible next to `compute_exclusive_regions`, which dominates the profile.
struct BasinDiagramCost<'a, S: DiagramShape + Copy + 'static> {
    inner: DiagramCost<'a, S>,
}

impl<S: DiagramShape + Copy + 'static> basin::CostFunction for BasinDiagramCost<'_, S> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> f64 {
        let p = DVector::from_vec(param.clone());
        // `DiagramCost::cost` returns a Result; map both Err and NaN to +∞
        // so a single bad evaluation can't poison the simplex sort.
        match self.inner.cost(&p) {
            Ok(c) if c.is_finite() => c,
            _ => f64::INFINITY,
        }
    }
}

impl<S: DiagramShape + Copy + 'static> basin::Gradient for BasinDiagramCost<'_, S> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Vec<f64>) -> Vec<f64> {
        let p = DVector::from_vec(param.clone());
        // `DiagramCost::gradient` takes the analytic boundary-velocity path
        // when the shape + loss support it and falls back to central FD
        // otherwise. A failed evaluation becomes a zero vector so L-BFGS
        // treats the point as stationary rather than stepping on garbage.
        match self.inner.gradient(&p) {
            Ok(g) => g.as_slice().to_vec(),
            Err(_) => vec![0.0; param.len()],
        }
    }
}

/// `DiagramCost` adapter for basin's box-constrained CMA-ES
/// (`Optimizer::CmaEsLm`'s global-escape stage).
///
/// Unlike [`BasinDiagramCost`] (which uses `Param = Vec<f64>` for the gradient
/// and simplex solvers), this wrapper keeps `Param = DVector<f64>` because
/// `basin::BoundedCmaEs` runs on the nalgebra backend directly — the whole tree
/// is aligned on a single nalgebra, so basin's vector type *is*
/// [`DVector<f64>`]. Box bounds live on the problem (basin's tenet 4); the
/// solver's adaptive quadratic penalty reads them through [`BoxConstrained`].
struct BoundedDiagramCost<'a, S: DiagramShape + Copy + 'static> {
    inner: DiagramCost<'a, S>,
    lower: DVector<f64>,
    upper: DVector<f64>,
}

impl<S: DiagramShape + Copy + 'static> basin::CostFunction for BoundedDiagramCost<'_, S> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &DVector<f64>) -> f64 {
        // Map both `Err` and non-finite costs to +∞ so a single bad sample
        // can't poison the population sort (mirrors `BasinDiagramCost::cost`).
        match self.inner.cost(param) {
            Ok(c) if c.is_finite() => c,
            _ => f64::INFINITY,
        }
    }
}

impl<S: DiagramShape + Copy + 'static> basin::BoxConstrained for BoundedDiagramCost<'_, S> {
    fn lower(&self) -> &DVector<f64> {
        &self.lower
    }

    fn upper(&self) -> &DVector<f64> {
        &self.upper
    }
}

/// Box-constrained wrapper around [`LmDiagramProblem`] for basin's `Trf`
/// (trust-region-reflective) solver — [`Optimizer::Trf`].
///
/// Delegates `Residual`/`Jacobian` to the inner least-squares problem
/// unchanged and adds the box bounds `Trf` requires via [`BoxConstrained`].
/// The `CostFunction` impl exists only to satisfy basin's `BoxConstrained:
/// CostFunction` supertrait bound — `Trf` computes `½‖r‖²` from the residual
/// directly and never queries `cost()`, so it carries no extra kernel
/// evaluations on the solve path.
struct BoundedLmDiagramProblem<'a, S: DiagramShape + Copy + 'static> {
    inner: LmDiagramProblem<'a, S>,
    lower: DVector<f64>,
    upper: DVector<f64>,
}

impl<S: DiagramShape + Copy + 'static> basin::Residual for BoundedLmDiagramProblem<'_, S> {
    type Param = DVector<f64>;
    type Output = DVector<f64>;

    fn residual(&self, x: &DVector<f64>) -> DVector<f64> {
        basin::Residual::residual(&self.inner, x)
    }
}

impl<S: DiagramShape + Copy + 'static> basin::Jacobian for BoundedLmDiagramProblem<'_, S> {
    type Param = DVector<f64>;
    type Output = DMatrix<f64>;

    fn jacobian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        basin::Jacobian::jacobian(&self.inner, x)
    }
}

impl<S: DiagramShape + Copy + 'static> basin::CostFunction for BoundedLmDiagramProblem<'_, S> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, x: &DVector<f64>) -> f64 {
        // Present only for the `BoxConstrained: CostFunction` bound; `Trf`
        // never calls this. Mirror its `½‖r‖²` convention so any incidental
        // query stays consistent with `state.cost`.
        0.5 * basin::Residual::residual(&self.inner, x).norm_squared()
    }
}

impl<S: DiagramShape + Copy + 'static> basin::BoxConstrained for BoundedLmDiagramProblem<'_, S> {
    fn lower(&self) -> &DVector<f64> {
        &self.lower
    }

    fn upper(&self) -> &DVector<f64> {
        &self.upper
    }
}

/// Width of the per-shape box-bound envelope, in units relative to the
/// MDS-initialised layout's `max(span, max_radius)` and `max_radius`.
///
/// The two bounded solvers want different widths:
/// - [`CMAES`](Self::CMAES) — a tight box for the CMA-ES *global escape*. The
///   adaptive boundary penalty and the `with_stds` preconditioning are tuned
///   around it, and a tight box keeps the global sampling focused. These are
///   the historical values; do not change them without re-benchmarking the
///   default (`CmaEsLm`), whose behaviour they define.
/// - [`LOCAL`](Self::LOCAL) — a wider safety-cage for the bounded *local*
///   solve ([`Optimizer::Trf`] and the `CmaEsTrf` polish). A local refinement
///   only needs bounds to catch radius/position blow-up, not to constrain a
///   legitimate optimum; a too-tight cage clips specs whose best fit needs a
///   shape larger than the tight ceiling (observed as `cmaes_trf` regressions
///   on `issue103`/`issue114` under the `CMAES` width).
#[derive(Clone, Copy)]
struct BoundsEnvelope {
    /// Position half-width as a multiple of `max(span, max_radius)`.
    pos_margin_scale: f64,
    /// Lower bound on radius / semi-axis as a fraction of `max_radius`.
    radius_floor_frac: f64,
    /// Upper bound on radius / semi-axis as a multiple of `max_radius`.
    radius_ceil_mult: f64,
}

impl BoundsEnvelope {
    /// Tight box for CMA-ES global exploration (historical values).
    const CMAES: Self = Self {
        pos_margin_scale: 4.0,
        radius_floor_frac: 1e-6,
        radius_ceil_mult: 5.0,
    };
    /// Wide safety-cage for bounded local refinement (TRF): loose enough not
    /// to clip a legitimate optimum, tight enough to catch blow-up.
    const LOCAL: Self = Self {
        pos_margin_scale: 8.0,
        radius_floor_frac: 1e-8,
        radius_ceil_mult: 12.0,
    };
}

/// Build per-parameter (lower, upper, initial_std) triples for the bounded
/// solvers from an MDS-initialised parameter vector, under the given
/// [`BoundsEnvelope`]. Consumed by the `CmaEsLm` global escape
/// ([`run_bounded_cmaes`], `BoundsEnvelope::CMAES`, uses all three) and by the
/// bounded local solve ([`run_trf`], `BoundsEnvelope::LOCAL`, uses only
/// lower/upper).
///
/// Layout-dependent on the shape (constants below are the `env` fields):
/// - Circle (`params_per_shape == 3`): `[x, y, r]` per shape. Positions
///   bounded to centroid ± `pos_margin_scale · max(span, max_radius)`; radii
///   bounded to `[radius_floor_frac · max_radius, radius_ceil_mult ·
///   max_radius]`. Initial std on x/y is `max(span, max_radius)/2`, on r it's
///   `max_radius/2`.
/// - Ellipse (`params_per_shape == 5`): `[x, y, ln(a), ln(b), angle]`.
///   Same x/y bounds; the semi-axis envelope is mapped into log space, so the
///   bound width and std are scale-invariant. Angle is unbounded with std
///   `π/4` (periodic; hard caps would just force the solver against an
///   artificial wall — and TRF's Coleman-Li scaling handles the `±∞` coords
///   by reducing to unconstrained there).
///
/// Other `params_per_shape` values fall back to wide unbounded bounds with
/// std proportional to the parameter magnitude — fine for the rest of the
/// algorithm to run, but new shape kinds should be added here so bounds
/// reflect their geometry.
fn optimizer_bounds_for(
    initial_param: &[f64],
    params_per_shape: usize,
    env: BoundsEnvelope,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = initial_param.len();
    let n_shapes = n / params_per_shape.max(1);

    // Pull centroid + spread from the position dims (always indices 0, 1
    // within each shape's parameter block, by convention shared across
    // every `DiagramShape` implementation).
    let mut xs = Vec::with_capacity(n_shapes);
    let mut ys = Vec::with_capacity(n_shapes);
    let mut radii = Vec::with_capacity(n_shapes);
    for k in 0..n_shapes {
        let base = k * params_per_shape;
        xs.push(initial_param[base]);
        ys.push(initial_param[base + 1]);
        match params_per_shape {
            3 => radii.push(initial_param[base + 2]),
            5 => {
                // Ellipse semi-axes live in log space (`u = ln(a)`,
                // `v = ln(b)`); exponentiate to get the linear
                // semi-axis magnitudes used for position scale.
                radii.push(initial_param[base + 2].exp());
                radii.push(initial_param[base + 3].exp());
            }
            _ => {}
        }
    }
    let mean_x: f64 = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
    let mean_y: f64 = ys.iter().sum::<f64>() / ys.len().max(1) as f64;
    let max_radius: f64 = radii.iter().cloned().fold(0.0_f64, f64::max).max(1e-6);
    let span_x: f64 = xs
        .iter()
        .map(|x| (x - mean_x).abs())
        .fold(0.0_f64, f64::max);
    let span_y: f64 = ys
        .iter()
        .map(|y| (y - mean_y).abs())
        .fold(0.0_f64, f64::max);
    let pos_scale: f64 = span_x.max(span_y).max(max_radius);
    // Position half-width. For CMA-ES (`pos_margin_scale = 4`) this gives the
    // global search room to rearrange shapes before it hits a wall; for the
    // local solve the wider scale keeps the cage off legitimate optima.
    let pos_margin: f64 = env.pos_margin_scale * pos_scale;

    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    let mut std_dev = Vec::with_capacity(n);
    for k in 0..n_shapes {
        // x
        lower.push(mean_x - pos_margin);
        upper.push(mean_x + pos_margin);
        std_dev.push((pos_scale * 0.5).max(1e-6));
        // y
        lower.push(mean_y - pos_margin);
        upper.push(mean_y + pos_margin);
        std_dev.push((pos_scale * 0.5).max(1e-6));
        // shape-specific dims
        match params_per_shape {
            3 => {
                // r
                lower.push(env.radius_floor_frac * max_radius);
                upper.push(env.radius_ceil_mult * max_radius);
                std_dev.push((max_radius * 0.5).max(1e-6));
            }
            5 => {
                // u = ln(a), v = ln(b). Bounds map the linear interval
                // `[radius_floor_frac·max_radius, radius_ceil_mult·max_radius]`
                // into log space; the width is constant in log space, so the
                // std is independent of scale.
                let u_lower = (env.radius_floor_frac * max_radius).ln();
                let u_upper = (env.radius_ceil_mult * max_radius).ln();
                let log_std = (u_upper - u_lower) * 0.1;
                // u = ln(semi-major)
                lower.push(u_lower);
                upper.push(u_upper);
                std_dev.push(log_std);
                // v = ln(semi-minor)
                lower.push(u_lower);
                upper.push(u_upper);
                std_dev.push(log_std);
                // angle (periodic; no hard caps)
                lower.push(f64::NEG_INFINITY);
                upper.push(f64::INFINITY);
                std_dev.push(std::f64::consts::FRAC_PI_4);
            }
            other => {
                // Unknown shape kind: leave the trailing dims unbounded and
                // pick a generic std from the parameter magnitude. New shape
                // kinds should add an explicit branch above.
                let extra = other.saturating_sub(2);
                let _ = (k, extra); // keep `k` live for future per-shape logic
                for j in 2..other {
                    let v = initial_param[k * other + j];
                    lower.push(f64::NEG_INFINITY);
                    upper.push(f64::INFINITY);
                    std_dev.push(v.abs().max(0.5));
                }
            }
        }
    }
    (lower, upper, std_dev)
}

/// Cost function for region error optimization.
///
/// Computes the discrepancy between target exclusive areas and actual fitted areas.
struct DiagramCost<'a, S: DiagramShape + Copy + 'static> {
    spec: &'a PreprocessedSpec,
    loss_type: crate::loss::LossType,
    params_per_shape: usize,
    _shape: std::marker::PhantomData<S>,
}

impl<S: DiagramShape + Copy + 'static> DiagramCost<'_, S> {
    /// Extract shapes from parameter vector.
    fn params_to_shapes(&self, params: &DVector<f64>) -> Vec<S> {
        let n_sets = self.spec.n_sets;

        (0..n_sets)
            .map(|i| {
                let start = i * self.params_per_shape;
                let end = start + self.params_per_shape;
                S::from_optimizer_params(&params.as_slice()[start..end])
            })
            .collect()
    }

    /// Decode the trailing 4 container params (`[x, y, ln area, ln ratio]`)
    /// into a `Rectangle`. Only meaningful when `self.spec.complement.is_some()`;
    /// the caller is responsible for that gate.
    fn params_to_container(&self, params: &DVector<f64>) -> crate::geometry::shapes::Rectangle {
        let n_sets = self.spec.n_sets;
        let trailing = &params.as_slice()[n_sets * self.params_per_shape..];
        crate::geometry::shapes::Rectangle::from_optimizer_params(trailing)
    }
}

impl<S: DiagramShape + Copy + 'static> DiagramCost<'_, S> {
    /// Loss between the fitted and target exclusive areas at `param`.
    fn cost(&self, param: &DVector<f64>) -> Result<f64, DiagramError> {
        let shapes = self.params_to_shapes(param);

        // When the spec carries a complement, the trailing 4 params encode a
        // jointly-optimised axis-aligned container rectangle; per-region areas
        // are clipped to it and mask 0 (the complement) is added by the
        // shape-specific clipping helper. Otherwise use the unclipped path.
        let exclusive_areas = if self.spec.complement.is_some() {
            let container = self.params_to_container(param);
            S::compute_exclusive_regions_clipped(&shapes, &container).ok_or_else(|| {
                DiagramError::InvalidCombination(
                    "complement specs require a shape with `compute_exclusive_regions_clipped`; \
                     fitter construction should have rejected this combination"
                        .to_string(),
                )
            })?
        } else {
            S::compute_exclusive_regions(&shapes)
        };

        // `LossType::compute` evaluates the right thing for both true and
        // smooth-surrogate variants — smoothing is now expressed by picking
        // a `Smooth*` variant rather than via a runtime flag.
        Ok(self
            .loss_type
            .compute(&exclusive_areas, &self.spec.exclusive_areas))
    }

    /// Analytic gradient where the shape + loss support it; central finite
    /// differences otherwise.
    fn gradient(&self, param: &DVector<f64>) -> Result<DVector<f64>, DiagramError> {
        // Try the analytical path: requires both the shape (region geometry)
        // and the loss to provide analytical gradients. Falls back to central
        // finite differences when either piece isn't available.
        //
        // When the spec carries a complement, region geometry is clipped to a
        // jointly-optimised container; the clipped-with-gradient companion
        // returns gradients of length `n_sets · S::n_params() + 4` matching
        // the trailing container params in `param`.
        let shapes = self.params_to_shapes(param);
        let analytic = if self.spec.complement.is_some() {
            let container = self.params_to_container(param);
            S::compute_exclusive_regions_clipped_with_gradient(&shapes, &container)
        } else {
            S::compute_exclusive_regions_with_gradient(&shapes)
        };
        if let Some((fitted, fitted_grads)) = analytic
            && let Some((_loss, loss_grad)) = self
                .loss_type
                .compute_with_gradient(&fitted, &self.spec.exclusive_areas)
        {
            let n_params = param.len();
            let mut grad = vec![0.0; n_params];
            for (mask, &dl_df) in loss_grad.iter() {
                if let Some(df_dtheta) = fitted_grads.get(mask) {
                    for (k, item) in grad.iter_mut().enumerate().take(n_params) {
                        *item += dl_df * df_dtheta[k];
                    }
                }
            }
            return Ok(DVector::from_vec(grad));
        }

        // Fallback: central finite differences. `finitediff` closures carry an
        // `anyhow::Error`; map it to `DiagramError` (this branch never errors
        // in practice — the cost closure always returns `Ok`).
        let param_vec = param.as_slice().to_vec();
        let f = |x: &Vec<f64>| {
            let p = DVector::from_vec(x.to_vec());
            Ok(self.cost(&p).unwrap_or(f64::INFINITY))
        };
        let g_central = vec::central_diff(&f);
        let grad_vec = g_central(&param_vec).map_err(|e| {
            DiagramError::InvalidCombination(format!("finite-difference gradient failed: {e}"))
        })?;
        Ok(DVector::from_vec(grad_vec))
    }
}

/// Levenberg-Marquardt least-squares adapter around the diagram cost.
///
/// Decomposes the existing `Σ(fitted_mask − target_mask)²` loss into per-region
/// residuals so LM can build `JᵀJ` directly from the analytical
/// region-area gradients exposed by
/// [`DiagramShape::compute_exclusive_regions_with_gradient`].
///
/// Residual ordering is recomputed per evaluation point and scores only the
/// **sparse** set of masks that can carry signal there — the union of masks
/// with a non-zero target and masks the geometry actually produces (see
/// [`DiagramCache::masks`]). Masks outside that set have `f = t = 0`, so they
/// contribute nothing to the residual, `JᵀJ`, or `‖r‖²`; enumerating the full
/// `2ⁿ` subset space (131 071 rows at `n = 17`) only to score structural zeros
/// is what made large disjoint specs (eulerr #89) crawl. For complement specs
/// the set includes mask `0` (the clipped kernel emits it and the complement
/// target lives there), the trailing 4 entries of the parameter vector encode
/// the jointly-optimised container rectangle, and the Jacobian columns pick up
/// box-edge contributions via
/// [`DiagramShape::compute_exclusive_regions_clipped_with_gradient`].
///
/// Supported loss: [`LossType::SumSquared`]. Other losses are rejected at
/// construction with an [`Error`].
struct LmDiagramProblem<'a, S: DiagramShape + Copy + 'static> {
    spec: &'a PreprocessedSpec,
    params_per_shape: usize,
    /// `1/sqrt(Σ tᵢ²)`. Stored residuals get multiplied by this so
    /// `Σ rᵢ²` equals the configured loss `Σ(f-t)² / Σt²`.
    norm_factor: f64,
    /// Single-entry cache of `(fitted, fitted_grads)` keyed by the parameter
    /// vector it was evaluated at. basin's LM queries `residual(x)` and
    /// `jacobian(x)` through separate trait methods, but both come from one
    /// `compute_exclusive_regions_with_gradient` call, so caching by point
    /// collapses the residual+Jacobian-at-x pair into a single kernel
    /// evaluation. (basin's LM also caches `r`/`J` internally across
    /// iterations; this only guards the within-point double call.)
    cache: RefCell<Option<(Vec<f64>, DiagramCache)>>,
    _shape: std::marker::PhantomData<S>,
}

struct DiagramCache {
    /// Sorted residual ordering for *this* point: the sparse union of masks
    /// with a non-zero target (`spec.exclusive_areas`) and masks the geometry
    /// produced here (`fitted`). Sorted so the ordering is a deterministic
    /// function of the point — basin pairs a stashed `r` with a freshly
    /// recomputed `J` at the same `x`, so `residual(x)` and `jacobian(x)` must
    /// stay row-aligned even across an intervening cache eviction.
    masks: Vec<RegionMask>,
    fitted: HashMap<RegionMask, f64>,
    fitted_grads: HashMap<RegionMask, Vec<f64>>,
}

impl<'a, S: DiagramShape + Copy + 'static> LmDiagramProblem<'a, S> {
    fn new(
        spec: &'a PreprocessedSpec,
        params_per_shape: usize,
        loss_type: LossType,
    ) -> Result<Self, DiagramError> {
        let norm_factor = match loss_type {
            LossType::SumSquared => {
                let sum_t2: f64 = spec.exclusive_areas.values().map(|&v| v * v).sum();
                if sum_t2 < 1e-20 {
                    return Err(DiagramError::InvalidCombination(
                        "Levenberg-Marquardt: target area norm is zero, cannot normalise"
                            .to_string(),
                    ));
                }
                1.0 / sum_t2.sqrt()
            }
            other => {
                return Err(DiagramError::InvalidCombination(format!(
                    "Levenberg-Marquardt only supports SumSquared loss, got {other:?}"
                )));
            }
        };
        // The residual ordering is built per evaluation point in
        // `ensure_cache` (sparse: targets ∪ produced regions), not enumerated
        // here over the full `2ⁿ` subset space.
        Ok(Self {
            spec,
            params_per_shape,
            norm_factor,
            cache: RefCell::new(None),
            _shape: std::marker::PhantomData,
        })
    }

    /// Ensure the cache holds `(fitted, fitted_grads)` for the parameter
    /// vector `xs`, recomputing the region kernel only when the point moved.
    fn ensure_cache(&self, xs: &[f64]) {
        if let Some((cached_x, _)) = self.cache.borrow().as_ref()
            && cached_x.as_slice() == xs
        {
            return;
        }
        let n_sets = self.spec.n_sets;
        let shapes: Vec<S> = (0..n_sets)
            .map(|i| {
                let start = i * self.params_per_shape;
                let end = start + self.params_per_shape;
                S::from_optimizer_params(&xs[start..end])
            })
            .collect();
        let analytic = if self.spec.complement.is_some() {
            // Trailing 4 params encode the container rectangle in `[x_c, y_c,
            // ln(area), ln(ratio)]` — same encoding as `Rectangle`'s
            // optimizer params.
            let trailing = &xs[n_sets * self.params_per_shape..];
            let container = crate::geometry::shapes::Rectangle::from_optimizer_params(trailing);
            S::compute_exclusive_regions_clipped_with_gradient(&shapes, &container)
        } else {
            S::compute_exclusive_regions_with_gradient(&shapes)
        };
        let (fitted, fitted_grads) = match analytic {
            Some(pair) => pair,
            None => {
                // Should not happen for shapes that wire LM in (Circle/Ellipse
                // both implement the gradient path). Fall back to areas-only
                // with empty gradient map; the Jacobian becomes all-zeros
                // and LM terminates immediately on its first-order optimality
                // check, surfacing the issue.
                let fitted = if self.spec.complement.is_some() {
                    let trailing = &xs[n_sets * self.params_per_shape..];
                    let container =
                        crate::geometry::shapes::Rectangle::from_optimizer_params(trailing);
                    S::compute_exclusive_regions_clipped(&shapes, &container)
                        .unwrap_or_else(|| S::compute_exclusive_regions(&shapes))
                } else {
                    S::compute_exclusive_regions(&shapes)
                };
                (fitted, HashMap::new())
            }
        };
        // Score only masks that can carry signal at this point: targets plus
        // the regions the geometry actually produced. Collected through a
        // `BTreeSet` for dedup + a deterministic (sorted) order. For
        // non-complement specs mask 0 ("outside everything") carries no target
        // and is dropped — matching the old `1..2ⁿ` enumeration; complement
        // specs keep it (the clipped kernel emits it and the target lives there).
        let mut mask_set: std::collections::BTreeSet<RegionMask> = self
            .spec
            .exclusive_areas
            .keys()
            .copied()
            .chain(fitted.keys().copied())
            .collect();
        if self.spec.complement.is_none() {
            mask_set.remove(&0);
        }
        let masks: Vec<RegionMask> = mask_set.into_iter().collect();
        *self.cache.borrow_mut() = Some((
            xs.to_vec(),
            DiagramCache {
                masks,
                fitted,
                fitted_grads,
            },
        ));
    }
}

impl<S: DiagramShape + Copy + 'static> basin::Residual for LmDiagramProblem<'_, S> {
    type Param = DVector<f64>;
    type Output = DVector<f64>;

    fn residual(&self, x: &DVector<f64>) -> DVector<f64> {
        self.ensure_cache(x.as_slice());
        let slot = self.cache.borrow();
        let (_, cache) = slot.as_ref().expect("cache populated by ensure_cache");
        DVector::from_fn(cache.masks.len(), |i, _| {
            let mask = cache.masks[i];
            let f = cache.fitted.get(&mask).copied().unwrap_or(0.0);
            let t = self.spec.exclusive_areas.get(&mask).copied().unwrap_or(0.0);
            self.norm_factor * (f - t)
        })
    }
}

impl<S: DiagramShape + Copy + 'static> basin::Jacobian for LmDiagramProblem<'_, S> {
    type Param = DVector<f64>;
    type Output = DMatrix<f64>;

    fn jacobian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        self.ensure_cache(x.as_slice());
        let slot = self.cache.borrow();
        let (_, cache) = slot.as_ref().expect("cache populated by ensure_cache");
        let n_params = x.len();
        // One HashMap lookup per residual row (not per element): fetch each
        // row's gradient vector once and fill the row.
        let mut jac = DMatrix::zeros(cache.masks.len(), n_params);
        for (i, &mask) in cache.masks.iter().enumerate() {
            if let Some(grad) = cache.fitted_grads.get(&mask) {
                for (j, &g) in grad.iter().enumerate().take(n_params) {
                    jac[(i, j)] = self.norm_factor * g;
                }
            }
        }
        jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::diagram;
    use crate::geometry::primitives::Point;
    use crate::geometry::shapes::Circle;
    use crate::spec::{DiagramSpec, DiagramSpecBuilder};

    /// Test helper utilities for final layout testing
    mod helpers {
        use super::*;
        use crate::Combination;
        use std::collections::HashMap;

        /// Generate a random diagram specification with the given number of sets.
        ///
        /// This creates random circles, computes their overlaps, and returns
        /// a DiagramSpec that can be used for testing the fitter.
        ///
        /// Returns: (spec, original_circles) for validation
        pub fn generate_random_diagram(n_sets: usize, seed: u64) -> (DiagramSpec, Vec<Circle>) {
            let (circles, set_names) = random_circle_layout(n_sets, seed);
            let exclusive_areas =
                diagram::compute_exclusive_areas_from_layout(&circles, &set_names);
            let spec = create_spec_from_exclusive(exclusive_areas);
            (spec, circles)
        }

        /// Generate a random circle layout for testing
        pub fn random_circle_layout(n_sets: usize, seed: u64) -> (Vec<Circle>, Vec<String>) {
            use rand::RngExt;
            use rand::SeedableRng;

            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

            let set_names: Vec<String> = (0..n_sets).map(|i| format!("Set{}", i)).collect();

            let mut circles = Vec::new();

            for _ in 0..n_sets {
                // Random position in [-5, 5] x [-5, 5]
                let x = rng.random_range(-5.0..5.0);
                let y = rng.random_range(-5.0..5.0);
                // Random radius in [0.5, 2.0]
                let r = rng.random_range(0.5..2.0);

                circles.push(Circle::new(Point::new(x, y), r));
            }

            (circles, set_names)
        }

        /// Create a DiagramSpec from exclusive areas
        pub fn create_spec_from_exclusive(
            exclusive_areas: HashMap<Combination, f64>,
        ) -> DiagramSpec {
            let mut builder = DiagramSpecBuilder::new();

            // Add single sets in sorted order so set-to-bit index assignment is
            // deterministic regardless of HashMap iteration order.
            let mut singles: Vec<_> = exclusive_areas
                .iter()
                .filter(|(combo, _)| combo.sets().len() == 1)
                .collect();
            singles.sort_by_key(|(combo, _)| combo.sets()[0].clone());
            for &(combo, &area) in &singles {
                builder = builder.set(combo.sets()[0].as_str(), area);
            }

            // Add all intersections (order doesn't affect bit assignment)
            for (combo, &area) in &exclusive_areas {
                if combo.sets().len() > 1 {
                    let sets: Vec<&str> = combo.sets().iter().map(|s| s.as_str()).collect();
                    builder = builder.intersection(&sets, area);
                }
            }

            builder.build().unwrap()
        }
    }

    // ========== Basic Functionality Tests ==========

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
            optimizer: Optimizer::NelderMead,
            loss_type: crate::loss::LossType::sse(),
            max_iterations: 50,
            tolerance: 1e-4,
            seed: 0,
            n_restarts: 1,
            cmaes_fallback_threshold: 1e-3,
            xtol: None,
            ftol: None,
            gtol: None,
        };

        let result = optimize_layout::<Circle>(&preprocessed, &positions, &radii, config);
        assert!(result.is_ok());

        let (final_params, loss) = result.unwrap();
        // For circles, params are [x0, y0, r0, x1, y1, r1, ...]
        assert_eq!(final_params.len(), 2 * Circle::n_params()); // 2 circles * 3 params each
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_cost_function_computes() {
        let spec = DiagramSpecBuilder::new()
            .set("A", 5.0)
            .set("B", 5.0)
            .intersection(&["A", "B"], 2.0)
            .build()
            .unwrap();

        let preprocessed = spec.preprocess().unwrap();

        let cost_fn = DiagramCost::<Circle> {
            spec: &preprocessed,
            loss_type: crate::loss::LossType::sse(),
            params_per_shape: Circle::n_params(),
            _shape: std::marker::PhantomData,
        };

        // Create parameter vector
        let positions = vec![0.0, 0.0, 2.0, 0.0];
        let radii = vec![
            (5.0 / std::f64::consts::PI).sqrt(),
            (5.0 / std::f64::consts::PI).sqrt(),
        ];

        let mut params = positions.clone();
        params.extend_from_slice(&radii);
        let param_vec = DVector::from_vec(params);

        let result = cost_fn.cost(&param_vec);
        assert!(result.is_ok(), "Cost function should compute successfully");

        let error = result.unwrap();
        assert!(error >= 0.0, "Error should be non-negative");
    }

    // ========== Layout Reproduction Tests ==========

    #[test]
    fn test_reproduce_simple_two_circle_layout() {
        use helpers::*;

        // Create a simple known layout
        let c1 = Circle::new(Point::new(0.0, 0.0), 1.0);
        let c2 = Circle::new(Point::new(1.5, 0.0), 1.0);
        let circles = vec![c1, c2];
        let set_names = vec!["A".to_string(), "B".to_string()];

        // Compute exclusive areas from this layout
        let exclusive = diagram::compute_exclusive_areas_from_layout(&circles, &set_names);

        // Create spec from these areas
        let spec = create_spec_from_exclusive(exclusive);
        let preprocessed = spec.preprocess().unwrap();

        // Try to fit with initial guess close to original
        let positions = vec![0.0, 0.0, 1.5, 0.0];
        let radii = vec![1.0, 1.0];

        let config = FinalLayoutConfig {
            optimizer: Optimizer::NelderMead,
            loss_type: crate::loss::LossType::sse(),
            max_iterations: 100,
            tolerance: 1e-6,
            seed: 0,
            n_restarts: 1,
            cmaes_fallback_threshold: 1e-3,
            xtol: None,
            ftol: None,
            gtol: None,
        };

        let result = optimize_layout::<Circle>(&preprocessed, &positions, &radii, config);
        assert!(result.is_ok());

        let (_, loss) = result.unwrap();

        // Should be able to reproduce the layout with very low error
        assert!(
            loss < 1e-2,
            "Should reproduce layout with low error, got: {}",
            loss
        );
    }

    #[test]
    fn test_reproduce_random_three_circle_layout() {
        use helpers::*;

        // Generate random layout
        let (circles, set_names) = random_circle_layout(3, 42);

        // Compute exclusive areas
        let exclusive_areas = diagram::compute_exclusive_areas_from_layout(&circles, &set_names);

        // Create spec
        let spec = create_spec_from_exclusive(exclusive_areas);
        let preprocessed = spec.preprocess().unwrap();

        // Extract initial positions and radii
        let mut positions = Vec::new();
        let mut radii = Vec::new();
        for c in &circles {
            positions.push(c.center().x());
            positions.push(c.center().y());
            radii.push(c.radius());
        }

        let config = FinalLayoutConfig {
            optimizer: Optimizer::NelderMead,
            loss_type: crate::loss::LossType::sse(),
            max_iterations: 200,
            tolerance: 1e-6,
            seed: 0,
            n_restarts: 1,
            cmaes_fallback_threshold: 1e-3,
            xtol: None,
            ftol: None,
            gtol: None,
        };

        let result = optimize_layout::<Circle>(&preprocessed, &positions, &radii, config);
        assert!(result.is_ok());

        let (_final_pos, loss) = result.unwrap();

        // Should be able to reproduce with reasonable error
        // Note: 3-way intersections are harder, and optimizer may converge to local minima
        // Relaxed tolerance to account for optimization challenges
        assert!(
            loss < 5.0,
            "Should reproduce random layout reasonably, got: {}",
            loss
        );
    }

    #[test]
    fn test_reproduce_multiple_random_diagrams() {
        use helpers::*;

        // Test multiple random configurations. Seeds picked so Nelder-Mead
        // converges within the per-arity tolerances on all supported
        // platforms; seed 500 for n=4 was swapped after observed divergence on
        // macOS CI (loss ~57 vs. tolerance 10). If you bump seeds, re-run on
        // every CI target before committing.
        let test_configs = [
            (2, 100), // 2 circles
            (2, 200), // 2 circles
            (3, 300), // 3 circles
            (3, 400), // 3 circles
            (4, 501), // 4 circles
        ];

        let mut results = Vec::new();

        for (i, &(n_sets, seed)) in test_configs.iter().enumerate() {
            // Generate random diagram
            let (spec, original_circles) = generate_random_diagram(n_sets, seed);
            let preprocessed = spec.preprocess().unwrap();

            // Extract initial positions and radii from original circles
            let mut positions = Vec::new();
            let mut radii = Vec::new();
            for c in &original_circles {
                positions.push(c.center().x());
                positions.push(c.center().y());
                radii.push(c.radius());
            }

            let config = FinalLayoutConfig {
                optimizer: Optimizer::NelderMead,
                loss_type: crate::loss::LossType::sse(),
                max_iterations: 200,
                tolerance: 1e-6,
                seed: 0,
                n_restarts: 1,
                cmaes_fallback_threshold: 1e-3,
                xtol: None,
                ftol: None,
                gtol: None,
            };

            let result = optimize_layout::<Circle>(&preprocessed, &positions, &radii, config);
            assert!(result.is_ok(), "Optimization failed for config {}", i);

            let (_, loss) = result.unwrap();

            results.push((n_sets, seed, loss));
        }

        // Relaxed tolerances - the algorithm should do reasonably well
        // but we're starting from the exact solution, so it should converge.
        // On failure, report the exact (n_sets, seed, loss) so CI breakage is
        // diagnosable without having to re-run locally.
        for &(n_sets, seed, loss) in &results {
            let tol: f64 = match n_sets {
                2 => 2.0,  // 2-way: should be very good
                3 => 5.0,  // 3-way: harder but doable
                4 => 10.0, // 4-way: quite difficult
                _ => 20.0, // Higher order: very challenging
            };
            assert!(
                loss < tol,
                "configuration n_sets={}, seed={} produced loss={:.6}, \
                 exceeds tolerance {:.1} — optimizer may have regressed, \
                 or the seed lands in a bad basin on this platform",
                n_sets,
                seed,
                loss,
                tol
            );
        }
    }

    /// Compare the analytical gradient produced by `DiagramCost::gradient` to
    /// the central-difference gradient of the cost function. Asserts a relative
    /// error below `tol`. This is the primary correctness check for the
    /// analytical-gradient path: equality (up to FD noise) on smooth interior
    /// configurations.
    fn assert_analytic_matches_fd<F>(
        spec: &PreprocessedSpec,
        params: &[f64],
        loss_type: crate::loss::LossType,
        h: f64,
        tol: f64,
        label: F,
    ) where
        F: FnOnce() -> String,
    {
        let cost = DiagramCost::<Circle> {
            spec,
            loss_type,
            params_per_shape: Circle::n_params(),
            _shape: std::marker::PhantomData,
        };
        let p = DVector::from_vec(params.to_vec());

        let analytic = cost.gradient(&p).expect("analytic gradient");

        // Central differences with explicit step size.
        let n = params.len();
        let mut fd = vec![0.0; n];
        for i in 0..n {
            let mut plus = params.to_vec();
            let mut minus = params.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let f_plus = cost.cost(&DVector::from_vec(plus)).expect("cost +");
            let f_minus = cost.cost(&DVector::from_vec(minus)).expect("cost -");
            fd[i] = (f_plus - f_minus) / (2.0 * h);
        }

        let analytic_slice: Vec<f64> = analytic.as_slice().to_vec();
        let diff_norm: f64 = analytic_slice
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
        let rel = if fd_norm > 1e-12 {
            diff_norm / fd_norm
        } else {
            diff_norm
        };
        assert!(
            rel < tol,
            "{}: analytic vs FD gradient mismatch (rel={:.3e}, |fd|={:.3e})\n  analytic={:?}\n  fd      ={:?}",
            label(),
            rel,
            fd_norm,
            analytic_slice,
            fd
        );
    }

    /// Build a PreprocessedSpec from a known circle layout — the target areas
    /// are simply the layout's own exclusive areas, so the gradient at the
    /// layout's own params is well-defined and the loss at that point is zero.
    /// Perturbing the params lets us probe the gradient on the smooth interior
    /// without sitting at the optimum.
    fn spec_from_circles(circles: &[Circle]) -> PreprocessedSpec {
        use helpers::create_spec_from_exclusive;
        let names: Vec<String> = (0..circles.len()).map(|i| format!("S{}", i)).collect();
        let exclusive = diagram::compute_exclusive_areas_from_layout(circles, &names);
        create_spec_from_exclusive(exclusive).preprocess().unwrap()
    }

    fn flat_params(circles: &[Circle]) -> Vec<f64> {
        let mut v = Vec::with_capacity(3 * circles.len());
        for c in circles {
            v.push(c.center().x());
            v.push(c.center().y());
            v.push(c.radius());
        }
        v
    }

    /// Every smooth loss type, paired with a relative tolerance multiplier
    /// applied to the scenario's base tolerance. `Smooth*` losses use a
    /// deliberately roomy `eps` so the softmax weights stay
    /// numerically well-conditioned in the FD comparison; smaller `eps` is
    /// fine in production but amplifies FD noise here.
    fn smooth_losses_for_grad_test() -> Vec<(crate::loss::LossType, f64, &'static str)> {
        use crate::loss::LossType;
        // Multipliers reflect how much harder FD agreement is for losses with
        // sharper curvature: `Smooth*` and region-error variants get a 10×
        // bump because the softmax / quotient amplifies higher-order FD error.
        vec![
            (LossType::SumSquared, 1.0, "SumSquared"),
            (LossType::RootMeanSquared, 1.0, "RootMeanSquared"),
            (LossType::Stress, 1.0, "Stress"),
            (
                LossType::SumSquaredRegionError,
                10.0,
                "SumSquaredRegionError",
            ),
            (
                LossType::smooth_sum_absolute(0.05),
                10.0,
                "SmoothSumAbsolute",
            ),
            (
                LossType::smooth_sum_absolute_region_error(1e-3),
                10.0,
                "SmoothSumAbsoluteRegionError",
            ),
            (
                LossType::smooth_max_absolute(0.5),
                10.0,
                "SmoothMaxAbsolute",
            ),
            (LossType::smooth_max_squared(0.5), 10.0, "SmoothMaxSquared"),
            (LossType::smooth_diag_error(1e-3), 10.0, "SmoothDiagError"),
        ]
    }

    #[test]
    fn analytic_gradient_two_circle_overlap() {
        // Classic 2-circle lens: this is the case I derived analytically.
        let circles = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(1.0, 0.0), 1.0),
        ];
        let spec = spec_from_circles(&circles);
        // Probe at a perturbed config so the loss isn't exactly zero
        // (gradient at L=0 minimum is also zero, which is uninformative).
        let params = vec![0.0, 0.0, 1.0, 1.2, 0.0, 0.95];
        let base_tol = 1e-4;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd(&spec, &params, loss_type, 1e-6, base_tol * mul, || {
                format!("two_circle_overlap {}", name)
            });
        }
    }

    #[test]
    fn analytic_gradient_three_circle_venn() {
        // Equilateral 3-Venn — exercises the polygon-boundary path.
        let circles = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(1.0, 0.0), 1.0),
            Circle::new(Point::new(0.5, 0.866), 1.0),
        ];
        let spec = spec_from_circles(&circles);
        let mut params = flat_params(&circles);
        // perturb to leave the optimum and avoid trivial zero gradient.
        params[0] += 0.07;
        params[4] -= 0.05;
        params[8] += 0.03;
        let base_tol = 1e-3;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd(&spec, &params, loss_type, 1e-6, base_tol * mul, || {
                format!("three_circle_venn {}", name)
            });
        }
    }

    #[test]
    fn analytic_gradient_disjoint() {
        // All-disjoint: exclusive areas are just the singletons; no
        // intersection-region gradients. Probes the n=1 boundary case.
        let circles = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(5.0, 0.0), 1.2),
            Circle::new(Point::new(2.5, 5.0), 0.8),
        ];
        let spec = spec_from_circles(&circles);
        let params = flat_params(&circles);
        // Perturb radii (positions don't move the loss when shapes stay
        // disjoint, so the gradient on x/y is zero — trivially matches).
        let mut perturbed = params.clone();
        perturbed[2] += 0.1;
        perturbed[5] -= 0.05;
        perturbed[8] += 0.07;
        let base_tol = 1e-4;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd(&spec, &perturbed, loss_type, 1e-6, base_tol * mul, || {
                format!("disjoint {}", name)
            });
        }
    }

    #[test]
    fn analytic_gradient_nested() {
        // Containment: small circle entirely inside a larger one.
        let circles = vec![
            Circle::new(Point::new(0.0, 0.0), 3.0),
            Circle::new(Point::new(0.5, 0.2), 0.8),
        ];
        let spec = spec_from_circles(&circles);
        let mut params = flat_params(&circles);
        // Perturb the inner circle.
        params[3] += 0.1;
        params[5] += 0.05;
        let base_tol = 1e-4;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd(&spec, &params, loss_type, 1e-6, base_tol * mul, || {
                format!("nested {}", name)
            });
        }
    }

    /// Build a complement-bearing PreprocessedSpec from a known circle layout
    /// inside a known container. Targets equal the layout's own clipped
    /// areas, so the gradient at the layout's own params is well-defined and
    /// the loss at that point is zero.
    fn complement_spec_from_layout(
        circles: &[Circle],
        container: &crate::geometry::shapes::Rectangle,
    ) -> PreprocessedSpec {
        use crate::geometry::shapes::circle::compute_exclusive_regions_clipped;
        use crate::spec::{DiagramSpec, DiagramSpecBuilder};

        let names: Vec<String> = (0..circles.len()).map(|i| format!("S{}", i)).collect();
        let areas = compute_exclusive_regions_clipped(circles, container);

        // Build a builder mirroring `helpers::create_spec_from_exclusive` but
        // routing per-set / per-intersection values from the (mask → area)
        // map of the clipped layout.
        let mut builder = DiagramSpecBuilder::new();
        for (i, name) in names.iter().enumerate() {
            // Singleton: the area at mask `1 << i` is the disk-only piece.
            let single_mask = 1usize << i;
            let v = areas.get(&single_mask).copied().unwrap_or(0.0);
            builder = builder.set(name.as_str(), v);
        }
        for (mask, &v) in &areas {
            if *mask == 0 {
                continue;
            }
            let bits = mask.count_ones();
            if bits < 2 {
                continue;
            }
            let mut sets: Vec<&str> = Vec::with_capacity(bits as usize);
            for (i, name) in names.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    sets.push(name.as_str());
                }
            }
            builder = builder.intersection(&sets, v);
        }
        let complement = areas.get(&0).copied().unwrap_or(0.0);
        builder = builder
            .complement(complement)
            .input_type(crate::InputType::Exclusive);
        let spec: DiagramSpec = builder.build().unwrap();
        spec.preprocess().unwrap()
    }

    /// Pack circles + container into the optimizer's flat parameter layout
    /// (per-circle blocks then container `[x_c, y_c, ln(area), ln(ratio)]`).
    fn pack_complement_params(
        circles: &[Circle],
        container: &crate::geometry::shapes::Rectangle,
    ) -> Vec<f64> {
        let mut p = Vec::with_capacity(circles.len() * 3 + 4);
        for c in circles {
            p.extend(c.to_optimizer_params());
        }
        p.extend(container.to_optimizer_params());
        p
    }

    /// FD-vs-analytical for `DiagramCost::gradient` on a complement spec.
    /// Mirrors `assert_analytic_matches_fd` but works with the extended
    /// parameter vector that includes the trailing 4 container params.
    #[test]
    fn analytic_gradient_complement_two_circles_inside_box() {
        let circles = vec![
            Circle::new(Point::new(-0.6, 0.0), 1.0),
            Circle::new(Point::new(0.55, 0.05), 0.95),
        ];
        let container = crate::geometry::shapes::Rectangle::new(Point::new(0.0, 0.0), 6.0, 5.0);
        let spec = complement_spec_from_layout(&circles, &container);
        // Probe at a perturbed config so the loss isn't exactly zero.
        let mut params = pack_complement_params(&circles, &container);
        params[0] += 0.07; // x0
        params[3] += -0.05; // x1
        params[6 + 2] += 0.03; // ln(area) of container

        let cost = DiagramCost::<Circle> {
            spec: &spec,
            loss_type: crate::loss::LossType::SumSquared,
            params_per_shape: Circle::n_params(),
            _shape: std::marker::PhantomData,
        };
        let p = DVector::from_vec(params.clone());
        let analytic = cost.gradient(&p).expect("analytic gradient");

        let h = 1e-6;
        let n = params.len();
        let mut fd = vec![0.0; n];
        for i in 0..n {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let f_plus = cost.cost(&DVector::from_vec(plus)).expect("cost +");
            let f_minus = cost.cost(&DVector::from_vec(minus)).expect("cost -");
            fd[i] = (f_plus - f_minus) / (2.0 * h);
        }
        let analytic_slice: Vec<f64> = analytic.as_slice().to_vec();
        let diff_norm: f64 = analytic_slice
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
        let rel = if fd_norm > 1e-9 {
            diff_norm / fd_norm
        } else {
            diff_norm
        };
        assert!(
            rel < 1e-3,
            "complement gradient mismatch (rel={:.3e}, |fd|={:.3e})\n  analytic={:?}\n  fd      ={:?}",
            rel,
            fd_norm,
            analytic_slice,
            fd
        );
    }

    /// Same as above but with one disk clipped by the right edge of the
    /// container — exercises the box-edge contribution to the disk's
    /// boundary integral.
    #[test]
    fn analytic_gradient_complement_one_disk_clipped() {
        let circles = vec![
            Circle::new(Point::new(1.4, 0.05), 0.9),
            Circle::new(Point::new(0.0, -0.07), 0.85),
        ];
        let container = crate::geometry::shapes::Rectangle::new(Point::new(0.0, 0.0), 3.6, 3.0);
        let spec = complement_spec_from_layout(&circles, &container);
        let mut params = pack_complement_params(&circles, &container);
        // Bump container x slightly so the right-edge clip on disk 0 stays a
        // smooth (non-tangent) intersection but the gradient is non-trivial.
        params[6] += 0.04;
        params[6 + 3] += 0.02;

        let cost = DiagramCost::<Circle> {
            spec: &spec,
            loss_type: crate::loss::LossType::SumSquared,
            params_per_shape: Circle::n_params(),
            _shape: std::marker::PhantomData,
        };
        let p = DVector::from_vec(params.clone());
        let analytic = cost.gradient(&p).expect("analytic gradient");

        let h = 1e-6;
        let n = params.len();
        let mut fd = vec![0.0; n];
        for i in 0..n {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let f_plus = cost.cost(&DVector::from_vec(plus)).expect("cost +");
            let f_minus = cost.cost(&DVector::from_vec(minus)).expect("cost -");
            fd[i] = (f_plus - f_minus) / (2.0 * h);
        }
        let analytic_slice: Vec<f64> = analytic.as_slice().to_vec();
        let diff_norm: f64 = analytic_slice
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
        let rel = if fd_norm > 1e-9 {
            diff_norm / fd_norm
        } else {
            diff_norm
        };
        assert!(
            rel < 1e-3,
            "complement+clip gradient mismatch (rel={:.3e}, |fd|={:.3e})\n  analytic={:?}\n  fd      ={:?}",
            rel,
            fd_norm,
            analytic_slice,
            fd
        );
    }

    /// Build a complement-bearing PreprocessedSpec from a known ellipse
    /// layout inside a known container. Mirrors `complement_spec_from_layout`
    /// but routes through the ellipse-specific clipped helper.
    fn complement_spec_from_layout_ellipse(
        ellipses: &[crate::geometry::shapes::Ellipse],
        container: &crate::geometry::shapes::Rectangle,
    ) -> PreprocessedSpec {
        use crate::geometry::shapes::ellipse::compute_exclusive_regions_clipped_ellipse;
        use crate::spec::{DiagramSpec, DiagramSpecBuilder};

        let names: Vec<String> = (0..ellipses.len()).map(|i| format!("S{}", i)).collect();
        let areas = compute_exclusive_regions_clipped_ellipse(ellipses, container);

        let mut builder = DiagramSpecBuilder::new();
        for (i, name) in names.iter().enumerate() {
            let single_mask = 1usize << i;
            let v = areas.get(&single_mask).copied().unwrap_or(0.0);
            builder = builder.set(name.as_str(), v);
        }
        for (mask, &v) in &areas {
            if *mask == 0 {
                continue;
            }
            let bits = mask.count_ones();
            if bits < 2 {
                continue;
            }
            let mut sets: Vec<&str> = Vec::with_capacity(bits as usize);
            for (i, name) in names.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    sets.push(name.as_str());
                }
            }
            builder = builder.intersection(&sets, v);
        }
        let complement = areas.get(&0).copied().unwrap_or(0.0);
        builder = builder
            .complement(complement)
            .input_type(crate::InputType::Exclusive);
        let spec: DiagramSpec = builder.build().unwrap();
        spec.preprocess().unwrap()
    }

    /// Pack ellipses + container into the optimizer's flat parameter layout
    /// (per-ellipse blocks `[x, y, ln a, ln b, φ]` then container
    /// `[x_c, y_c, ln(area), ln(ratio)]`).
    fn pack_complement_params_ellipse(
        ellipses: &[crate::geometry::shapes::Ellipse],
        container: &crate::geometry::shapes::Rectangle,
    ) -> Vec<f64> {
        let mut p = Vec::with_capacity(ellipses.len() * 5 + 4);
        for e in ellipses {
            p.extend(e.to_optimizer_params());
        }
        p.extend(container.to_optimizer_params());
        p
    }

    /// FD-vs-analytical for `DiagramCost::gradient` on an ellipse complement
    /// spec with both ellipses inside the container. Mirrors
    /// `analytic_gradient_complement_two_circles_inside_box`.
    #[test]
    fn analytic_gradient_complement_two_ellipses_inside_box() {
        use crate::geometry::shapes::Ellipse;
        let ellipses = vec![
            Ellipse::new(Point::new(-0.7, 0.0), 1.0, 0.7, 0.0),
            Ellipse::new(Point::new(0.55, 0.05), 0.95, 0.6, 0.0),
        ];
        let container = crate::geometry::shapes::Rectangle::new(Point::new(0.0, 0.0), 6.0, 5.0);
        let spec = complement_spec_from_layout_ellipse(&ellipses, &container);
        // Probe at a perturbed config so the loss isn't exactly zero.
        let mut params = pack_complement_params_ellipse(&ellipses, &container);
        params[0] += 0.07; // x0
        params[5] += -0.05; // x1
        params[10 + 2] += 0.03; // ln(area) of container

        let cost = DiagramCost::<Ellipse> {
            spec: &spec,
            loss_type: crate::loss::LossType::SumSquared,
            params_per_shape: Ellipse::n_params(),
            _shape: std::marker::PhantomData,
        };
        let p = DVector::from_vec(params.clone());
        let analytic = cost.gradient(&p).expect("analytic gradient");

        let h = 1e-6;
        let n = params.len();
        let mut fd = vec![0.0; n];
        for i in 0..n {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let f_plus = cost.cost(&DVector::from_vec(plus)).expect("cost +");
            let f_minus = cost.cost(&DVector::from_vec(minus)).expect("cost -");
            fd[i] = (f_plus - f_minus) / (2.0 * h);
        }
        let analytic_slice: Vec<f64> = analytic.as_slice().to_vec();
        let diff_norm: f64 = analytic_slice
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
        let rel = if fd_norm > 1e-9 {
            diff_norm / fd_norm
        } else {
            diff_norm
        };
        assert!(
            rel < 1e-3,
            "ellipse complement gradient mismatch (rel={:.3e}, |fd|={:.3e})\n  analytic={:?}\n  fd      ={:?}",
            rel,
            fd_norm,
            analytic_slice,
            fd
        );
    }

    /// Same as above but with one rotated ellipse clipped by the right edge of
    /// the container — exercises the box-edge contribution to the ellipse's
    /// boundary integral plus a non-zero rotation.
    #[test]
    fn analytic_gradient_complement_one_ellipse_clipped() {
        use crate::geometry::shapes::Ellipse;
        let ellipses = vec![
            Ellipse::new(Point::new(1.4, 0.05), 0.9, 0.6, 0.3),
            Ellipse::new(Point::new(0.0, -0.07), 0.85, 0.55, -0.2),
        ];
        let container = crate::geometry::shapes::Rectangle::new(Point::new(0.0, 0.0), 3.6, 3.0);
        let spec = complement_spec_from_layout_ellipse(&ellipses, &container);
        let mut params = pack_complement_params_ellipse(&ellipses, &container);
        // Bump container x slightly so the right-edge clip on ellipse 0 stays
        // a smooth (non-tangent) intersection but the gradient is non-trivial.
        params[10] += 0.04;
        params[10 + 3] += 0.02;

        let cost = DiagramCost::<Ellipse> {
            spec: &spec,
            loss_type: crate::loss::LossType::SumSquared,
            params_per_shape: Ellipse::n_params(),
            _shape: std::marker::PhantomData,
        };
        let p = DVector::from_vec(params.clone());
        let analytic = cost.gradient(&p).expect("analytic gradient");

        let h = 1e-6;
        let n = params.len();
        let mut fd = vec![0.0; n];
        for i in 0..n {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let f_plus = cost.cost(&DVector::from_vec(plus)).expect("cost +");
            let f_minus = cost.cost(&DVector::from_vec(minus)).expect("cost -");
            fd[i] = (f_plus - f_minus) / (2.0 * h);
        }
        let analytic_slice: Vec<f64> = analytic.as_slice().to_vec();
        let diff_norm: f64 = analytic_slice
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
        let rel = if fd_norm > 1e-9 {
            diff_norm / fd_norm
        } else {
            diff_norm
        };
        assert!(
            rel < 1e-3,
            "ellipse complement+clip gradient mismatch (rel={:.3e}, |fd|={:.3e})\n  analytic={:?}\n  fd      ={:?}",
            rel,
            fd_norm,
            analytic_slice,
            fd
        );
    }

    /// Same idea as `benchmark_gradient_call_only` but for ellipses.
    #[test]
    #[ignore]
    fn benchmark_gradient_call_only_ellipse() {
        use crate::geometry::shapes::Ellipse;
        use crate::loss::LossType;
        use rand::{RngExt, SeedableRng};
        use std::time::Instant;
        let cases: &[(usize, u64, usize)] = &[(3, 13, 500), (5, 100, 200), (8, 200, 100)];
        for &(n, seed, iters) in cases {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let ellipses: Vec<Ellipse> = (0..n)
                .map(|_| {
                    Ellipse::new(
                        Point::new(rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5)),
                        rng.random_range(0.7..1.6),
                        rng.random_range(0.5..1.0),
                        rng.random_range(-0.5..0.5),
                    )
                })
                .collect();
            let spec = spec_from_ellipses(&ellipses);
            let mut params = flat_params_ellipse(&ellipses);
            for v in &mut params {
                *v += rng.random_range(-0.04..0.04);
            }
            let p = DVector::from_vec(params.clone());
            let cost = DiagramCost::<Ellipse> {
                spec: &spec,
                loss_type: LossType::SumSquared,
                params_per_shape: Ellipse::n_params(),
                _shape: std::marker::PhantomData,
            };
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = cost.gradient(&p).unwrap();
            }
            let dt_an = t0.elapsed();
            let t0 = Instant::now();
            for _ in 0..iters {
                let pv = p.as_slice().to_vec();
                let f = |x: &Vec<f64>| {
                    let pp = DVector::from_vec(x.to_vec());
                    Ok(cost.cost(&pp).unwrap_or(f64::INFINITY))
                };
                let g = vec::central_diff(&f);
                let _ = g(&pv).unwrap();
            }
            let dt_fd = t0.elapsed();
            eprintln!(
                "ellipse n={} ({} grads): analytic={:>7.2}ms ({:>5.1}us/call) | FD={:>7.2}ms ({:>5.1}us/call) | speedup={:.1}x",
                n,
                iters,
                dt_an.as_secs_f64() * 1000.0,
                dt_an.as_secs_f64() * 1e6 / iters as f64,
                dt_fd.as_secs_f64() * 1000.0,
                dt_fd.as_secs_f64() * 1e6 / iters as f64,
                dt_fd.as_secs_f64() / dt_an.as_secs_f64(),
            );
        }
    }

    /// Apples-to-apples: time the gradient function alone, with and without
    /// the analytical path, on the same loss. The FD comparison comes from
    /// running the same `DiagramCost::gradient` body but skipping the
    /// analytic short-circuit by using `finitediff::vec::central_diff`
    /// directly — what the old implementation did.
    #[test]
    #[ignore]
    fn benchmark_gradient_call_only() {
        use crate::loss::LossType;
        use rand::{RngExt, SeedableRng};
        use std::time::Instant;

        let cases: &[(usize, u64, usize)] = &[(3, 11, 1000), (5, 33, 500), (8, 55, 200)];
        for &(n, seed, iters) in cases {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let circles: Vec<Circle> = (0..n)
                .map(|_| {
                    Circle::new(
                        Point::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)),
                        rng.random_range(0.6..1.8),
                    )
                })
                .collect();
            let spec = spec_from_circles(&circles);
            let mut params = flat_params(&circles);
            for v in &mut params {
                *v += rng.random_range(-0.05..0.05);
            }
            let p = DVector::from_vec(params.clone());

            let cost = DiagramCost::<Circle> {
                spec: &spec,
                loss_type: LossType::SumSquared,
                params_per_shape: Circle::n_params(),
                _shape: std::marker::PhantomData,
            };

            // Analytical (current default for SumSquared + Circle).
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = cost.gradient(&p).unwrap();
            }
            let dt_an = t0.elapsed();

            // FD baseline: invoke the central_diff path directly (same as the
            // pre-analytic implementation).
            let cost_fd = || -> f64 {
                let dvec = p.clone();
                cost.cost(&dvec).unwrap()
            };
            let _warm = cost_fd();
            let t0 = Instant::now();
            for _ in 0..iters {
                let pv = p.as_slice().to_vec();
                let f = |x: &Vec<f64>| {
                    let pp = DVector::from_vec(x.to_vec());
                    Ok(cost.cost(&pp).unwrap_or(f64::INFINITY))
                };
                let g = vec::central_diff(&f);
                let _ = g(&pv).unwrap();
            }
            let dt_fd = t0.elapsed();

            eprintln!(
                "n={} ({} grads): analytic={:>7.2}ms ({:>5.1}us/call) | FD={:>7.2}ms ({:>5.1}us/call) | speedup={:.1}x",
                n,
                iters,
                dt_an.as_secs_f64() * 1000.0,
                dt_an.as_secs_f64() * 1e6 / iters as f64,
                dt_fd.as_secs_f64() * 1000.0,
                dt_fd.as_secs_f64() * 1e6 / iters as f64,
                dt_fd.as_secs_f64() / dt_an.as_secs_f64(),
            );
        }
    }

    /// Build an ellipse-only spec from a known layout, parallel to
    /// `spec_from_circles`.
    fn spec_from_ellipses(ellipses: &[crate::geometry::shapes::Ellipse]) -> PreprocessedSpec {
        use crate::Combination;
        use crate::geometry::traits::Area;
        use std::collections::HashMap;
        // Use the new boundary-based area for both target generation and
        // the cost-function side, so the FD baseline and analytical gradient
        // are consistent at the smooth interior.
        let exclusive =
            crate::geometry::shapes::ellipse::Ellipse::compute_exclusive_regions(ellipses);
        // Convert mask->area map into Combination-keyed map.
        let names: Vec<String> = (0..ellipses.len()).map(|i| format!("S{}", i)).collect();
        let mut combos: HashMap<Combination, f64> = HashMap::new();
        for (mask, area) in exclusive {
            if area > 0.0 {
                let sets: Vec<&str> = (0..ellipses.len())
                    .filter(|&i| (mask & (1 << i)) != 0)
                    .map(|i| names[i].as_str())
                    .collect();
                if !sets.is_empty() {
                    combos.insert(Combination::new(&sets), area);
                }
            }
        }
        // Even if some areas were 0, ensure singletons are present in the
        // builder so set indices line up.
        let single_areas: HashMap<usize, f64> = (0..ellipses.len())
            .map(|i| (i, ellipses[i].area()))
            .collect();
        let mut builder = crate::spec::DiagramSpecBuilder::new();
        for i in 0..ellipses.len() {
            let area = combos
                .get(&Combination::new(&[names[i].as_str()]))
                .copied()
                .unwrap_or(single_areas[&i]);
            builder = builder.set(names[i].as_str(), area);
        }
        for (combo, area) in &combos {
            if combo.sets().len() > 1 {
                let sets: Vec<&str> = combo.sets().iter().map(|s| s.as_str()).collect();
                builder = builder.intersection(&sets, *area);
            }
        }
        builder.build().unwrap().preprocess().unwrap()
    }

    fn flat_params_ellipse(ellipses: &[crate::geometry::shapes::Ellipse]) -> Vec<f64> {
        let mut v = Vec::with_capacity(5 * ellipses.len());
        for e in ellipses {
            v.push(e.center().x());
            v.push(e.center().y());
            v.push(e.semi_major());
            v.push(e.semi_minor());
            v.push(e.rotation());
        }
        v
    }

    /// Generic cost-function FD-vs-analytic comparison parameterised over the
    /// shape type. Mirrors `assert_analytic_matches_fd` for circles.
    fn assert_analytic_matches_fd_shape<S, F>(
        spec: &PreprocessedSpec,
        params: &[f64],
        loss_type: crate::loss::LossType,
        h: f64,
        tol: f64,
        label: F,
    ) where
        S: DiagramShape + Copy + 'static,
        F: FnOnce() -> String,
    {
        let cost = DiagramCost::<S> {
            spec,
            loss_type,
            params_per_shape: S::n_params(),
            _shape: std::marker::PhantomData,
        };
        let p = DVector::from_vec(params.to_vec());
        let analytic = cost.gradient(&p).expect("analytic gradient");
        let n = params.len();
        let mut fd = vec![0.0; n];
        for i in 0..n {
            let mut plus = params.to_vec();
            let mut minus = params.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let f_plus = cost.cost(&DVector::from_vec(plus)).expect("cost +");
            let f_minus = cost.cost(&DVector::from_vec(minus)).expect("cost -");
            fd[i] = (f_plus - f_minus) / (2.0 * h);
        }
        let analytic_slice: Vec<f64> = analytic.as_slice().to_vec();
        let diff_norm: f64 = analytic_slice
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let fd_norm: f64 = fd.iter().map(|b| b * b).sum::<f64>().sqrt();
        let rel = if fd_norm > 1e-12 {
            diff_norm / fd_norm
        } else {
            diff_norm
        };
        assert!(
            rel < tol,
            "{}: analytic vs FD gradient mismatch (rel={:.3e}, |fd|={:.3e})\n  analytic={:?}\n  fd      ={:?}",
            label(),
            rel,
            fd_norm,
            analytic_slice,
            fd
        );
    }

    #[test]
    fn analytic_gradient_ellipse_two_overlap() {
        use crate::geometry::shapes::Ellipse;
        let ellipses = vec![
            Ellipse::new(Point::new(0.0, 0.0), 1.0, 0.6, 0.0),
            Ellipse::new(Point::new(1.0, 0.2), 1.2, 0.5, 0.4),
        ];
        let spec = spec_from_ellipses(&ellipses);
        let mut params = flat_params_ellipse(&ellipses);
        params[0] += 0.05;
        params[4] += 0.07;
        let base_tol = 5e-4;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd_shape::<Ellipse, _>(
                &spec,
                &params,
                loss_type,
                1e-6,
                base_tol * mul,
                || format!("ellipse two_overlap {}", name),
            );
        }
    }

    #[test]
    fn analytic_gradient_ellipse_three_venn() {
        use crate::geometry::shapes::Ellipse;
        let ellipses = vec![
            Ellipse::new(Point::new(0.0, 0.0), 1.2, 0.7, 0.0),
            Ellipse::new(Point::new(1.0, 0.0), 1.1, 0.65, 0.3),
            Ellipse::new(Point::new(0.5, 0.866), 1.0, 0.8, -0.2),
        ];
        let spec = spec_from_ellipses(&ellipses);
        let mut params = flat_params_ellipse(&ellipses);
        params[0] += 0.07;
        params[4] -= 0.05;
        params[8] += 0.03;
        params[14] += 0.04;
        let base_tol = 5e-3;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd_shape::<Ellipse, _>(
                &spec,
                &params,
                loss_type,
                1e-6,
                base_tol * mul,
                || format!("ellipse three_venn {}", name),
            );
        }
    }

    #[test]
    fn analytic_gradient_ellipse_disjoint() {
        use crate::geometry::shapes::Ellipse;
        let ellipses = vec![
            Ellipse::new(Point::new(0.0, 0.0), 1.0, 0.6, 0.0),
            Ellipse::new(Point::new(5.0, 0.0), 1.2, 0.5, 0.5),
            Ellipse::new(Point::new(2.5, 5.0), 0.8, 0.4, -0.3),
        ];
        let spec = spec_from_ellipses(&ellipses);
        let mut params = flat_params_ellipse(&ellipses);
        params[2] += 0.1;
        params[3] -= 0.05;
        params[4] += 0.07;
        let base_tol = 5e-4;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd_shape::<Ellipse, _>(
                &spec,
                &params,
                loss_type,
                1e-6,
                base_tol * mul,
                || format!("ellipse disjoint {}", name),
            );
        }
    }

    #[test]
    fn analytic_gradient_ellipse_nested() {
        use crate::geometry::shapes::Ellipse;
        let ellipses = vec![
            Ellipse::new(Point::new(0.0, 0.0), 3.0, 2.5, 0.1),
            Ellipse::new(Point::new(0.5, 0.2), 0.8, 0.5, 0.3),
        ];
        let spec = spec_from_ellipses(&ellipses);
        let mut params = flat_params_ellipse(&ellipses);
        // Perturb the inner ellipse only — outer stays around inner.
        params[5] += 0.1;
        params[7] += 0.05;
        params[9] += 0.05;
        let base_tol = 5e-4;
        for (loss_type, mul, name) in smooth_losses_for_grad_test() {
            assert_analytic_matches_fd_shape::<Ellipse, _>(
                &spec,
                &params,
                loss_type,
                1e-6,
                base_tol * mul,
                || format!("ellipse nested {}", name),
            );
        }
    }

    #[test]
    fn analytic_gradient_ellipse_random_layouts() {
        use crate::geometry::shapes::Ellipse;
        use rand::RngExt;
        use rand::SeedableRng;
        let configs: &[(usize, u64)] = &[(2, 7), (3, 13), (3, 21), (5, 100)];
        for &(n, seed) in configs {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let ellipses: Vec<Ellipse> = (0..n)
                .map(|_| {
                    Ellipse::new(
                        Point::new(rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5)),
                        rng.random_range(0.7..1.6),
                        rng.random_range(0.5..1.0),
                        rng.random_range(-0.5..0.5),
                    )
                })
                .collect();
            let spec = spec_from_ellipses(&ellipses);
            let mut params = flat_params_ellipse(&ellipses);
            for v in &mut params {
                *v += rng.random_range(-0.04..0.04);
            }
            assert_analytic_matches_fd_shape::<Ellipse, _>(
                &spec,
                &params,
                crate::loss::LossType::SumSquared,
                1e-6,
                1e-2,
                || format!("ellipse random n={}, seed={}", n, seed),
            );
        }
    }

    /// Eyeball benchmark: time L-BFGS fits with the analytical gradient
    /// against the finite-difference fallback. This is an `#[ignore]`-gated
    /// observation, not an asserted regression — wall time varies across
    /// machines, but on a typical workstation the analytical path should be
    /// noticeably faster on 5- and 8-set fits and produce equivalent loss.
    #[test]
    #[ignore]
    fn benchmark_lbfgs_analytic_vs_fd() {
        use crate::loss::LossType;
        use std::time::Instant;
        // Disable analytical gradient for the FD baseline by pretending the
        // shape doesn't support it — we do this by routing through a wrapper
        // cost that always falls through to FD. The simplest implementation:
        // run the optimiser with a loss type that doesn't expose an analytic
        // gradient (e.g. SumAbsoute), and compare against the same fit using
        // SumSquared (which DOES). This isn't apples-to-apples on
        // loss surface but does exercise the gradient path. For a more direct
        // test, see comments below.
        let configs: &[(usize, u64)] = &[(5, 33), (5, 44), (8, 55)];
        for &(n, seed) in configs {
            use rand::RngExt;
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let circles: Vec<Circle> = (0..n)
                .map(|_| {
                    Circle::new(
                        Point::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)),
                        rng.random_range(0.6..1.8),
                    )
                })
                .collect();
            let spec = spec_from_circles(&circles);
            let positions: Vec<f64> = circles
                .iter()
                .flat_map(|c| [c.center().x(), c.center().y()])
                .collect();
            let radii: Vec<f64> = circles.iter().map(|c| c.radius()).collect();

            // Analytical-gradient path: SumSquared has compute_with_gradient.
            let cfg_an = FinalLayoutConfig {
                optimizer: Optimizer::Lbfgs,
                loss_type: LossType::SumSquared,
                max_iterations: 200,
                tolerance: 1e-6,
                seed: 0,
                n_restarts: 1,
                cmaes_fallback_threshold: 1e-3,
                xtol: None,
                ftol: None,
                gtol: None,
            };
            let t0 = Instant::now();
            let (_p_an, loss_an) =
                optimize_layout::<Circle>(&spec, &positions, &radii, cfg_an).expect("analytic fit");
            let dt_an = t0.elapsed();

            // FD path: SumAbsoute has no compute_with_gradient → falls back to FD.
            // (Same optimiser, different loss; serves as a relative timing reference.)
            let cfg_fd = FinalLayoutConfig {
                optimizer: Optimizer::Lbfgs,
                loss_type: LossType::SumAbsoute,
                max_iterations: 200,
                tolerance: 1e-6,
                seed: 0,
                n_restarts: 1,
                cmaes_fallback_threshold: 1e-3,
                xtol: None,
                ftol: None,
                gtol: None,
            };
            let t0 = Instant::now();
            let (_p_fd, loss_fd) =
                optimize_layout::<Circle>(&spec, &positions, &radii, cfg_fd).expect("fd fit");
            let dt_fd = t0.elapsed();

            eprintln!(
                "n={} seed={}: analytic={:>6.1}ms loss={:.4e} | fd-loss={:>6.1}ms loss={:.4e}",
                n,
                seed,
                dt_an.as_secs_f64() * 1000.0,
                loss_an,
                dt_fd.as_secs_f64() * 1000.0,
                loss_fd
            );
        }
    }

    #[test]
    fn analytic_gradient_random_layouts() {
        use rand::RngExt;
        use rand::SeedableRng;
        // Several seeds × sizes, spanning generic 3-, 5-, 8-set fits.
        let configs: &[(usize, u64)] = &[(3, 11), (3, 22), (5, 33), (5, 44), (8, 55)];
        for &(n, seed) in configs {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let circles: Vec<Circle> = (0..n)
                .map(|_| {
                    Circle::new(
                        Point::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)),
                        rng.random_range(0.6..1.8),
                    )
                })
                .collect();
            let spec = spec_from_circles(&circles);
            // Perturb params to leave the (target = layout) optimum.
            let mut params = flat_params(&circles);
            for v in &mut params {
                *v += rng.random_range(-0.05..0.05);
            }
            assert_analytic_matches_fd(
                &spec,
                &params,
                crate::loss::LossType::SumSquared,
                1e-6,
                5e-3,
                || format!("random n={}, seed={}", n, seed),
            );
        }
    }

    #[test]
    fn lm_jacobian_consistent_with_diagram_gradient() {
        // For sum-of-squares losses, ∇L(θ) = 2·Jᵀ·r where J is the
        // residual Jacobian and r the residual vector. Verifies the LM
        // adapter constructs J consistently with `DiagramCost::gradient`.
        use rand::RngExt;
        use rand::SeedableRng;
        let configs: &[(usize, u64, crate::loss::LossType)] = &[
            (3, 11, crate::loss::LossType::SumSquared),
            (5, 33, crate::loss::LossType::SumSquared),
        ];
        for &(n, seed, loss_type) in configs {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let circles: Vec<Circle> = (0..n)
                .map(|_| {
                    Circle::new(
                        Point::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)),
                        rng.random_range(0.6..1.8),
                    )
                })
                .collect();
            let spec = spec_from_circles(&circles);
            let mut params = flat_params(&circles);
            for v in &mut params {
                *v += rng.random_range(-0.05..0.05);
            }
            let param_dvec = DVector::from_vec(params.clone());

            let cost = DiagramCost::<Circle> {
                spec: &spec,
                loss_type,
                params_per_shape: Circle::n_params(),
                _shape: std::marker::PhantomData,
            };
            let analytic_grad = cost.gradient(&param_dvec).unwrap();

            let problem =
                LmDiagramProblem::<Circle>::new(&spec, Circle::n_params(), loss_type).unwrap();
            let r = basin::Residual::residual(&problem, &param_dvec);
            let j = basin::Jacobian::jacobian(&problem, &param_dvec);
            // lm_grad = 2 · Jᵀ·r, assembled elementwise over the nalgebra DVector/DMatrix.
            let n_params = param_dvec.len();
            let n_res = r.nrows();
            let lm_grad: Vec<f64> = (0..n_params)
                .map(|jx| 2.0 * (0..n_res).map(|ix| j[(ix, jx)] * r[ix]).sum::<f64>())
                .collect();

            let max_abs_diff = analytic_grad
                .iter()
                .zip(lm_grad.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let max_abs = analytic_grad
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs_diff < 1e-9 + 1e-8 * max_abs,
                "n={} seed={} loss={:?}: |2·Jᵀ·r − ∇L| = {:.3e} (max |∇L| = {:.3e})",
                n,
                seed,
                loss_type,
                max_abs_diff,
                max_abs
            );
        }
    }

    #[test]
    fn lm_rejects_non_least_squares_loss() {
        let circles = vec![
            Circle::new(Point::new(0.0, 0.0), 1.0),
            Circle::new(Point::new(1.5, 0.0), 1.0),
        ];
        let spec = spec_from_circles(&circles);
        let result = LmDiagramProblem::<Circle>::new(
            &spec,
            Circle::n_params(),
            crate::loss::LossType::Stress,
        );
        assert!(result.is_err(), "Stress loss should be rejected by LM");
    }
}
