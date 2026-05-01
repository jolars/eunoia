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

- [x] **Square Venn warm-start** — see "Square shape follow-ups" below.

## Square shape follow-ups

Deferred from the axis-aligned `Square` PR (`crates/eunoia/src/geometry/shapes/square.rs`).

- [x] **Analytical final-stage gradient for `Square`**. Done:
      `compute_exclusive_regions_with_gradient` overridden in
      `crates/eunoia/src/geometry/shapes/square.rs`. For each region the
      n-way intersection rectangle's `dx · dy` decomposes via four binding
      extrema (`x_min, x_max, y_min, y_max`); each side's contribution goes
      to the binding shape, with equal split among ties on coincident edges
      (matches the central-FD subgradient at non-smooth points). Chained
      through `geometry::diagram::to_exclusive_areas_and_gradients` for IE.
      Gradient-vs-FD tests cover 1-, 2-, 3-square overlap, disjoint, nested,
      and a generic no-ties config (tight 1e-7 tolerance).

- [x] **Add `Square` to the corpus and `examples/quality_report`**. Done:
      `CorpusEntry` carries `fittable_square: Fittable` and
      `max_diag_error_square: Option<f64>` with a `ceiling_square()`
      accessor; all 27 entries are populated. Per-spec ceilings were
      tightened/loosened against observed default-fitter quality (see
      inline comments). `corpus_quality.rs` adds
      `corpus_squares_diag_error`. `examples/quality_report` runs the
      same config sweep across `Square` as a third shape pass and emits
      it in both the markdown and JSON outputs.

- [x] **WASM bindings for `Square`**. `crates/eunoia-wasm/src/lib.rs` exposes
      `WasmSquare`, `SquareResult`, `generate_from_spec_square`,
      `generate_squares_as_polygons`, and `generate_region_polygons_squares`
      (parallel to the circle/ellipse paths). `PolygonResult` carries an
      additional `squares` field. The web app surfaces a third "Square"
      option in `SpecEditor.svelte`; `fit.ts` dispatches to the
      square-specific WASM entry points and `DiagramSvg.svelte` renders
      `<rect>` outlines and labels.

- [ ] **Rotated squares / general axis-aligned rectangles**. Axis-aligned
      `Square` keeps n-way intersections trivially axis-aligned. Rotation
      breaks that (the n-way intersection becomes a convex polygon), so a
      rotated variant either gates on the `plotting` feature for polygon
      clipping or pulls `i_overlay` into the core dependencies. Needs a
      design pass before implementation. Same reasoning applies to general
      axis-aligned `Rectangle` (currently a bounding-box primitive only,
      not a `DiagramShape`); promoting it to `DiagramShape` is a separate
      smaller change.

- [x] **Venn warm-start for `Square`**. Done: `venn_warm_start_params` in
      `fitter.rs` now dispatches via `TypeId` to a dedicated Square branch
      that pulls from `VennDiagram::<Square>::new(n)` for n ∈ {2, 3} and
      scales by the spec's mean side length (`mean(sqrt(area_i))`) so the
      seed lands at the right area magnitude. n ≥ 4 returns `None` and
      stays on the random MDS path (no axis-aligned-square Venn exists).
      `VENN_SEED_MAX_SETS_SQUARE = 3` is the new cap. With the warm-start
      slot 0 is now seed-independent for Square fits where it applies.

- [x] **Generate Venn diagrams with squares** (broader than the warm-start
      above). Done: `VennDiagram` is now generic over `S: DiagramShape`
      (`VennDiagram::<Square>::new(n)` for n ∈ {1, 2, 3}, returning
      `UnsupportedSetCount` for n ≥ 4). Canonical layouts moved onto each
      shape via `DiagramShape::canonical_venn_layout`, with the existing
      ellipse N1..N5 constants colocated in `geometry/shapes/ellipse.rs`.
      The accessor on `VennDiagram` was renamed `ellipses() → shapes()`.
