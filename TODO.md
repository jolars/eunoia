# TODO

## Test corpus follow-ups

These are deferred items from the fit-quality harness landed alongside this
file. The corpus lives at `crates/eunoia/src/test_utils/corpus.rs`; the
default-suite tests are in `crates/eunoia/src/fitter/corpus_quality.rs` and
`crates/eunoia/src/fitter/synthetic_groundtruth.rs`.

- [ ] **Aggregate quality-report binary**
      (`crates/eunoia/examples/quality_report.rs`). Run the full corpus ×
      `QUALITY_SEEDS` (16) × shapes, dump a markdown table to stdout and a JSON
      snapshot to `target/quality_report.json` for PR-vs-PR diffing. Block on
      this before serious optimizer-comparison work --- it's how we'll measure
      CMA-ES / DE+L-BFGS / etc against the current default.

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

- [ ] **CMA-ES + L-BFGS polish pilot**. Bound-constrained final stage, CMA-ES
      global step, L-BFGS final polish using the existing analytical gradients.
      Benchmark against `corpus_circles_diag_error` /
      `corpus_ellipses_diag_error` and the synthetic ground-truth proptest. If
      CMA-ES alone closes the issue #28 / `eulerape_3_set` gap, ship it; only
      then consider memetic DE+L-BFGS.

- [ ] **Latin hypercube initial starts** instead of uniform random perturbations
      across `n_restarts`. Cheap potential 10-20% best-of-N quality bump; no
      algorithmic complexity.

- [ ] **L-BFGS-B (bounded) for the final stage**. Argmin doesn't ship it; either
      project unbounded L-BFGS results onto valid ranges, or reparameterise
      (`a = exp(u)`, `b = exp(v)`) so the unbounded solver stays on a valid
      manifold. Worth doing regardless of the bigger redesign --- pathological
      `a/b` ratios show up at the proptest generation tail.
