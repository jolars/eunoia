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
      `<target>/quality_report.json`. Captures both fit quality
      (`diag_error`, final loss, iterations) and runtime (per-cell
      `elapsed_ms`, per-spec aggregates, per-shape totals) so a single report
      characterises both axes of any optimizer change.

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

## Optimizer work (gated on the harness)

- [ ] **Levenberg-Marquardt final stage** (try this *before* CMA-ES). Our
      objective is `Σ rᵢ²` where `rᵢ = fitted_i - target_i` — textbook
      nonlinear least-squares — and we already have analytical residual
      gradients (per-region area gradients on `NormalizedSumSquared` /
      `SumSquared`, see AGENTS.md status). LM approximates the Hessian via
      `JᵀJ` from the Jacobian, so the "no analytical Hessian" gap doesn't
      bite. argmin 0.10 doesn't ship LM — use the `levenberg-marquardt`
      crate (nalgebra-integrated) and wire it as a new `Optimizer` variant.
      Add a `lm` config to `examples/quality_report.rs` and compare against
      the new L-BFGS-only default. Baseline numbers for the comparison live
      in `target/quality_report.json` at the time this TODO was written.

- [ ] **Gauss-Newton final stage** (cheap follow-on to LM). argmin already
      ships `GaussNewton` with line search. Once LM's residual Jacobian
      plumbing is in place, GN is essentially a different solver-call away —
      useful as a "what does LM's damping buy us" data point, mostly to
      characterise how often the linearisation is good enough that GN's
      undamped step is fine.

- [ ] **CMA-ES + L-BFGS polish pilot**. Bound-constrained final stage, CMA-ES
      global step, L-BFGS final polish using the existing analytical gradients.
      Benchmark against `corpus_circles_diag_error` /
      `corpus_ellipses_diag_error` and the synthetic ground-truth proptest. If
      CMA-ES alone closes the issue #28 / `eulerape_3_set` gap, ship it; only
      then consider memetic DE+L-BFGS. Run *after* the LM/GN comparison —
      they're cheaper to try and may already close the gap on their own.

- [ ] **Latin hypercube initial starts** instead of uniform random perturbations
      across `n_restarts`. Cheap potential 10-20% best-of-N quality bump; no
      algorithmic complexity.

- [ ] **L-BFGS-B (bounded) for the final stage**. Argmin doesn't ship it; either
      project unbounded L-BFGS results onto valid ranges, or reparameterise
      (`a = exp(u)`, `b = exp(v)`) so the unbounded solver stays on a valid
      manifold. Worth doing regardless of the bigger redesign --- pathological
      `a/b` ratios show up at the proptest generation tail.
