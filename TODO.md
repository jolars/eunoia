# TODO

## Test corpus follow-ups

These are deferred items from the fit-quality harness landed alongside this
file. The corpus lives at `crates/eunoia/src/test_utils/corpus.rs`; the
default-suite tests are in `crates/eunoia/src/fitter/corpus_quality.rs` and
`crates/eunoia/src/fitter/synthetic_groundtruth.rs`.

- [x] **Aggregate quality-report binary**
      (`crates/eunoia/examples/quality_report.rs`). Runs the full corpus ×
      `QUALITY_SEEDS` (16) × {Circle, Ellipse} using the default `Fitter`,
      prints a markdown summary to stdout, and writes a JSON snapshot to
      `<target>/quality_report.json`. Captures both fit quality (`diag_error`,
      final loss, iterations) and runtime (per-cell `elapsed_ms`, per-spec
      aggregates, per-shape totals) so a single report characterises both axes
      of any optimizer change.

      Run with:

      ```
      cargo run --release --example quality_report --features corpus
      ```

      The `corpus` feature exposes `eunoia::test_utils::corpus` outside
      `cfg(test)` so the example can build the same fixture set the
      integration tests use; it is internal, not part of the public API
      contract. Default sweep wall time on a developer machine is ~1 s.

- [ ] **Bench dedupe**: move the 4 helpers in
      `crates/eunoia/benches/initial_layout.rs` (`three_circle_easy`,
      `three_circle_user_case`, `issue28_four_set_superset`, `issue28_six_set`)
      into the corpus and have the bench import from there. They overlap with
      the corpus entries `eulerape_3_set` (-ish), `wilkinson_6_set`, and
      `three_inside_fourth`. PR 1 left the bench alone to keep the diff small.

- [ ] **eulerr cross-comparison**: capture R-side `diagError` per spec (eulerr)
      and compare against the Rust corpus output. Useful baseline ahead of
      optimizer redesigns, but needs an R-output capture script and a stable
      harness in eulerr --- out of scope for the eunoia PR.

## Surfaced fitter issues (regressions to investigate)

The corpus / proptest surfaced these. None were introduced by the harness;
they're pre-existing behaviour the harness now exposes.

- [ ] **`eulerape_3_set`ellipses land at `diag_error ≈ 1.16e-2`** at default
      `Fitter` settings on every `TEST_SEEDS` entry. Ellipses can represent this
      layout exactly in principle --- eulerr's eulerAPE article spec is the
      canonical "ellipses fit it perfectly" example. Looks like a default-budget
      local-minimum trap. Loosened ceiling to 2e-2 in the corpus; tighten back
      to \~5e-4 once fixed.

- [ ] **`normalize_layout`debug_assert trips on coincident ellipses**
      (`fitter.rs:627` "normalize_layout changed fitted exclusive regions").
      Reproducer: corpus spec `two_overlapping_completely`
      (`A=0, B=0,   A&B=10`) with `Ellipse` in any debug build. Cluster
      rotation/normalization perturbs the two coincident ellipses by more than
      the assert's `abs=1e-8, rel=1e-6` tolerances. The corpus marks this as
      `Fittable::Skip` for ellipses; the bug itself still needs fixing (loosen
      tolerances, or fix normalize to be a true rigid transform on coincident
      shapes).

- [ ] **`random_4_set`ellipses land at `diag_error ≈ 2.6e-2`**. Random area
      inputs aren't guaranteed to be representable by ellipses, so this may not
      be improvable without more general shapes --- but it's a data point worth
      re-checking after any optimizer redesign.

- [ ] **Synthetic-groundtruth threshold is loose** (5e-2). The generating
      configuration is exactly representable by construction, so a healthy
      fitter would reach near-zero. The current threshold accommodates
      default-budget local minima on randomly drawn ellipse layouts. Tighten it
      as fitter quality improves; treat any loosening of this number as a
      regression.

- [ ] **`test_issue28_four_set_superset_ellipse_regression`slow-test fails under
      default LM at the test's tightened budget** (`tolerance=1e-10`,
      `max_iterations=2000`). Default-budget LM at seed=1 reaches `diag_error`
      well below the test's 1e-6 bar (the `corpus_ellipses_diag_error` fast test
      passes the same spec at seed=1), but with `patience=2000` LM's
      `max_fev = patience·(n+1) ≈ 42000` on n=20 lets it drift past the good
      basin. Pre-existing --- surfaced in the SA-fallback drop. Either tighten
      LM termination on `with_patience` or relax the tightened budget in the
      test; the spec itself is fittable.

## Optimizer work (gated on the harness)

- [x] **Levenberg-Marquardt final stage** --- landed as
      `Optimizer::LevenbergMarquardt` (final stage) and
      `MdsSolver::LevenbergMarquardt` (MDS stage), via the
      `levenberg-marquardt = "0.13"` crate. On the quality_report ellipse sweep,
      `lm_final` drops median loss from `1.6e-7` (L-BFGS default) to `1.1e-28`,
      mean diag_error from `7.4e-3` to `4.8e-3`, and total wall time from 4.5 s
      to 1.55 s --- wins 20 of 27 specs vs default's 5. On circles the gain is
      marginal (already near-optimum). Restricted to `SumSquared` /
      `NormalizedSumSquared`; rejects other losses at construction.

- [ ] **Gauss-Newton final stage** (low-priority bookkeeping after LM landed).
      argmin ships `GaussNewton` with line search; the LM Jacobian plumbing is
      already in place, so GN is a one-line solver swap. With LM hitting \~1e-28
      median ellipse loss out of the box this is no longer a quality candidate
      --- only worth running to characterise how often the linearisation is good
      enough that GN's undamped step matches LM.

- [x] **CMA-ES global step + LM polish** --- landed as `Optimizer::CmaEsLm`
      (final stage). Inline purecma-style CMA-ES (`fitter/cmaes.rs`, \~350 LOC,
      no new deps): bounded box around the MDS init (centroid ± 4·span on
      positions, `[1e-6·max_r, 5·max_r]` on radii / semi-axes, unbounded on
      angles), per-dim std for preconditioning, quadratic boundary penalty (no
      rejection sampling). Each restart runs both plain LM and CMA-ES → LM
      polish in parallel and keeps the lower-loss result, so the path is
      strictly non-regressing vs `Optimizer::LevenbergMarquardt`. On the
      `examples/quality_report` ellipse sweep median loss falls from `1.3e-4` to
      `~1.5e-29` on `issue92_3_set_dropped_pair` (full escape), from `8.5e-3` to
      `4.1e-3` on `random_4_set`, marginally on `issue44_4_set_inclusive`
      (`4.44e-3` → `4.36e-3`), and is unchanged on `issue91_6_set` (`3.86e-1`).
      Aggregate ellipse mean diag drops from `4.0e-3` to `3.2e-3`; 20 spec wins
      (median loss) vs 18 for `lm_full`. Promoted to the `Fitter` default after
      threshold-firing on `Fitter::cmaes_fallback_threshold` (default `1e-3`):
      plain LM runs first and the global step is skipped when LM lands at or
      below the threshold, so easy specs pay no extra wall time. Wall-time delta
      vs `lm_full` on the full `examples/quality_report` ellipse sweep drops
      from \~7× (always-fire) to \~3.8× (threshold-fired); circle sweep is \~5×.
      Mean diag improvement (4.0e-3 → 3.4e-3) and the
      `issue92_3_set_dropped_pair` escape both preserved. Open: `issue91_6_set`
      is **not closed** --- CMA-ES at default budget can't escape its basin in
      30 dims. Either bigger budget or pivot to memetic DE+LM. Not blocking;
      flag as separate follow-up.

- [ ] **Differential Evolution probe (`Optimizer::DeLm`)**. Cheap experiment to
      decide whether `issue91_6_set` (the one spec CMA-ES can't escape at 30
      dims and default budget) is reachable by a different population dynamic,
      or whether it's genuinely outside reach of any non-Jacobian global stage
      at this budget. Reuse the same threshold-fire scaffolding
      `Optimizer::CmaEsLm` uses; swap CMA-ES for a basic DE/rand/1/bin (mutation
      factor `0.5`, crossover `0.9`, population `~10·n`) inline in
      `fitter/de.rs`, \~150 LOC, no new deps. Benchmark via
      `examples/quality_report` against `default` (CmaEsLm) on the full corpus +
      the synthetic-groundtruth proptest. Decision rule: if the `issue91_6_set`
      median drops below `1e-1`, invest in a memetic DE+LM variant (adaptive
      `F`/`CR`, archive-based mutation, ...); if it doesn't, close out the
      global-stage line of work and move on to Latin hypercube starts and the
      bounded `a/b` reparameterisation below --- both likely worth more
      diag-error per hour spent than further global-stage tuning at this budget.

- [ ] **Latin hypercube initial starts** instead of uniform random perturbations
      across `n_restarts`. Cheap potential 10-20% best-of-N quality bump; no
      algorithmic complexity.

- [ ] **Bounded final stage for pathological `a/b` ratios**. With LM as the
      default and L-BFGS-B not in argmin, the cleanest fix is reparameterising
      semi-axes as `a = exp(u)`, `b = exp(v)` so the unbounded solver stays on a
      valid manifold. Re-test first --- LM's stronger basin coverage may have
      already eliminated the proptest tail failures that motivated this.
