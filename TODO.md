# TODO

## Test corpus follow-ups

These are deferred items from the fit-quality harness landed alongside this
file. The corpus lives at `crates/eunoia/src/test_utils/corpus.rs`; the
default-suite tests are in `crates/eunoia/src/fitter/corpus_quality.rs` and
`crates/eunoia/src/fitter/synthetic_groundtruth.rs`.

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

- [ ] **`issue71_4_set_extreme_scale`ellipse seed=1 lands in a different
      basin on Windows** (diag `~1.5e-1` vs Linux `~1.9e-4`). The spec's
      4-order-of-magnitude area variation (A=38066 vs D=6) makes the
      final-stage optimisation sensitive to FP rounding in
      `sin`/`cos`/conic-intersection math, and Windows's MSVC math
      runtime returns slightly different ULP values than glibc. Other
      `TEST_SEEDS` entries (42, 7) match Linux on Windows. Worked
      around with a platform-conditional ceiling
      (`ISSUE71_ELLIPSE_CEILING` in
      `crates/eunoia/src/test_utils/corpus.rs`): Linux/macOS at `5e-2`,
      Windows at `2e-1`. Real fix would tighten the optimizer's basin-
      of-attraction on extreme-scale specs (e.g. better
      `NormalizedSumSquared` conditioning, scale-aware initial
      perturbation, or a tighter MDS init). Not blocking; the
      platform split keeps the Linux ceiling strict so future
      regressions on dev machines / Linux CI still trip.

- [ ] **`random_4_set`ellipses land at `diag_error ≈ 2.6e-2`**. The corpus
      ceiling is tightened to `3e-2` (was `5e-2`) since this basin is a
      deterministic floor across most master seeds. There are at least two
      distinct local minima:
      - **basin A** (loss `7.786e-3`, diag `2.606e-2`) — reached by ~13/16
        `QUALITY_SEEDS` master seeds at default `n_restarts=10`.
      - **basin B** (loss `4.335e-3`, diag `1.147e-2`) — reached by the other
        ~3/16. An even slightly better basin (loss `4.086e-3`) shows up at
        `n_restarts ≥ 40` but only as the global-min, never the median.

      The basins differ in which ellipse area maps to which set's target
      (basin A nails `C ≈ 4.24`, `E ≈ 4.56` and undershoots `A`/`B`; basin B
      spreads the error more evenly). Neither raising `n_restarts` to 100 nor
      forcing `Optimizer::CmaEsLm` to fire on every restart
      (`cmaes_fallback_threshold = 1.0`) shifts the median off basin A —
      CMA-ES at default budget / box doesn't span the basin gap from this
      MDS init. Probed via a throwaway example (deleted after measurement).

      Worth re-checking after any optimizer redesign that touches the global
      stage; tighten the ceiling further if a future change closes basin A.

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

## MDS architecture follow-ups

- [ ] **Ellipse MDS still warm-starts as a circle**. `Ellipse::mds_target_distance`
      delegates to `Circle::mds_target_distance` (treating each ellipse as
      a circle of equal area), and the MDS phase optimises only 2D centers
      — orientation `φ` and the `a/b` ratio never enter the MDS loss. This
      is defensible (the final-stage optimizer takes over and reshapes from
      there, and most corpus specs hit machine precision) but it has a real
      blind spot: ellipses with large axis ratio overlapping along their
      major vs minor axis produce the same overlap area at very different
      center distances, and the circle-equivalent inversion picks one
      canonical distance somewhere between. Ellipse fits then have no
      rotational seed information; the optimizer rediscovers `φ` from
      scratch, which is part of why ellipse fits need higher `n_restarts`
      and the CMA-ES escape stage. A correct ellipse MDS would optimise
      over `[x, y, a, b, φ]` per shape against overlap targets directly
      (the larger refactor sketched as option (1) during the Square
      design — kept out of the Square PR because the existing MDS solver
      pool — Lbfgs, TrustRegion, NewtonCg, LevenbergMarquardt — all have
      analytical gradients/Hessians/Jacobians wired to the 2D positional
      cost, and rederiving them for ellipse parameters means inheriting
      ellipse's intricate boundary-integral derivative. Worth revisiting
      when (a) we have concrete specs where ellipse fits miss because of
      a wrong rotational basin out of MDS, or (b) we're doing the
      shape-aware-MDS refactor anyway for triangles or another shape that
      can't reasonably warm-start as a circle.

- [ ] **Square Venn warm-start** — see "Square shape follow-ups" below.

## Square shape follow-ups

Deferred from the axis-aligned `Square` PR (`crates/eunoia/src/geometry/shapes/square.rs`).

- [ ] **Analytical final-stage gradient for `Square`**. Currently
      `compute_exclusive_regions_with_gradient` returns `None`, so the
      optimizer falls back to central finite differences. An analytical
      gradient is closed-form: each n-way intersection rectangle has 4
      active-constraint edges; for each active edge, `∂(width·height)/∂θ` is
      a closed-form sum over which `x_i ± side_i/2` and `y_i ± side_i/2`
      are the binding extrema, then chained through inclusion-exclusion via
      the existing `to_exclusive_areas_and_gradients` helper
      (`geometry/diagram.rs`). Should net the same ~10–30× speedup
      Circle/Ellipse get from their boundary-velocity gradients. Until this
      lands, `Fitter::<Square>` fits are FD-bound and don't drive loss to
      machine precision on smooth specs.

- [ ] **Add `Square` to the corpus and `examples/quality_report`**. Requires
      adding a `fittable_square: Fittable` field to every `CorpusEntry`
      in `crates/eunoia/src/test_utils/corpus.rs` (27 entries) plus a
      per-shape ceiling (`max_diag_error_square`, or fall back to the
      category default), then wiring a third shape pass through
      `quality_report::run`. Held back from the initial Square PR to keep
      the diff focused. Without this, Square has no cross-spec quality
      tracking and regressions in its fits won't surface in the standard
      sweep.

- [ ] **WASM bindings for `Square`**. `crates/eunoia-wasm/src/lib.rs` exposes
      `WasmCircle` / `WasmEllipse` wrappers and per-shape entry points
      (`generate_from_spec_circle`, `generate_from_spec_ellipse`); add the
      analogous `WasmSquare`, `SquareResult`, and `generate_from_spec_square`,
      and surface the new shape in the web app's shape selector
      (`web/src/lib/DiagramViewer.svelte`).

- [ ] **Rotated squares / general axis-aligned rectangles**. Axis-aligned
      `Square` keeps n-way intersections trivially axis-aligned. Rotation
      breaks that (the n-way intersection becomes a convex polygon), so a
      rotated variant either gates on the `plotting` feature for polygon
      clipping or pulls `i_overlay` into the core dependencies. Needs a
      design pass before implementation. Same reasoning applies to general
      axis-aligned `Rectangle` (currently a bounding-box primitive only,
      not a `DiagramShape`); promoting it to `DiagramShape` is a separate
      smaller change.

- [ ] **Venn warm-start for `Square`**. `venn_warm_start_params` in
      `fitter.rs` currently `TypeId`-skips Square because the canonical-Venn
      arrangement is parameterised in circle/ellipse terms. A square Venn
      for n=3 (three squares with centers at the vertices of an equilateral
      triangle, scaled so adjacent pairs overlap and all three meet at the
      centroid) is well-defined and would close the slot-0 quality gap on
      easy 2- and 3-set Square specs. n ≥ 4 has no clean canonical Venn
      under axis-aligned squares; leave those on the random MDS path.

- [ ] **Generate Venn diagrams with squares** (broader than the warm-start
      above). `crate::venn::VennDiagram` currently exposes only a
      circle/ellipse arrangement. Adding a square equivalent — at minimum
      for n ∈ {2, 3} where an axis-aligned arrangement exists — would let
      callers ask for a canonical square Venn directly, not just use it as
      a fitter seed. Useful for Venn-input workflows (no spec, just "give
      me a 3-set Venn in squares") and for parity with the circle/ellipse
      Venn API.
