# TODO

## Test corpus follow-ups

These are deferred items from the fit-quality harness landed alongside this
file. The corpus lives at `crates/eunoia/src/test_utils/corpus.rs`; the
default-suite tests are in `crates/eunoia/src/fitter/corpus_quality.rs` and
`crates/eunoia/src/fitter/synthetic_groundtruth.rs`.

- [x] **Bench dedupe**: `benches/initial_layout.rs` now imports specs via
      `corpus::get(...)`. The two issue-#28 helpers reuse the existing
      `wilkinson_6_set` / `three_inside_fourth` entries; the two unique
      probes were promoted to corpus entries `three_set_small_overlaps`
      and `three_set_triple_only`. The bench is now gated on
      `required-features = ["corpus"]`.

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

## Spec representation follow-ups

- [x] **Make `DiagramSpec::inclusive_areas` lazy or drop it**. Done (option 2):
      `DiagramSpec` now stores only `exclusive_areas`. Build no longer walks
      the `2^|combo|` subset tree per input combination — a 30-way
      intersection that previously expanded to 2³⁰ subsets now builds in
      microseconds (regression test in
      `spec/spec_builder.rs::test_build_scales_with_large_kway_intersection`).
      Public API: `inclusive_areas()` returns `HashMap<...>` by value (was
      `&HashMap<...>` — minor breaking change), and `get_inclusive` sums
      contributing exclusive entries on demand. `preprocess()` computes
      singleton inclusive areas and pairwise overlaps directly from
      `exclusive_areas` in `O(Σ |c|²)` total — the eager full-inclusive
      map is gone, and `PreprocessedSpec.inclusive_areas` was dropped
      since nothing outside `spec.rs` consumed it. WASM bindings and the
      web app use `exclusive_areas()` only and were unaffected.

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

## Complement / container follow-ups

Loose ends from S6 of the complement roadmap. The feature itself shipped (see
the complement bullet in `AGENTS.md` Status); these are polish items the
roadmap didn't require but would tighten the surface.

- [ ] **Migrate or delete the six legacy WASM entry points**. The S6 WASM
      pass added a shared `build_diagram_spec(specs, input_type, complement)`
      helper and routed the modern `generate_*_as_polygons` /
      `generate_region_polygons_*` / `generate_venn_*` functions through it.
      The older entry points still inline their `DiagramSpecBuilder` block
      and don't accept a `complement` argument:
      `generate_from_spec`, `generate_from_spec_with_debug`, `get_debug_info`,
      `get_debug_info_simple`, `generate_from_spec_initial`,
      `get_debug_info_initial` (in `crates/eunoia-wasm/src/lib.rs`). Neither
      `ts/index.ts` nor the web app calls any of them — they're legacy /
      debug paths from before the polygon-mode backend. Either migrate them
      to the helper (consistency, complement support) or delete them
      outright (smaller WASM surface). Deletion is probably the right call;
      `generate_circles_as_polygons` already covers everything
      `generate_from_spec` did, with strictly more diagnostics.

- [ ] **Web app: validation feedback for the complement input**. The
      `Complement (universe)` numeric in `web/src/lib/components/SpecEditor.svelte`
      uses `min="0"` but `<input type="number">` doesn't actually block
      negative entry — `runFit` (`web/src/lib/fit.ts`) silently treats
      anything `< 0` or `NaN` as "not set", which is a confusing fail-soft.
      Show an inline error (matching the row-input style) when the value is
      invalid, or clamp on input.

- [ ] **Legend entry for the complement region**. `DiagramSvg.svelte`
      renders the container as a dashed grey frame and (when
      `style.showCounts`) labels the count in the top-right corner. The
      legend (`StyleControls` → `legendPosition`) doesn't include a
      "complement" / "outside" swatch when a container is present. Adding
      one means deciding the right label ("complement", "outside", a
      user-supplied name?) and surfacing it through the existing legend
      build path.

## Label placement follow-ups

- [ ] **Leader lines crossing interior labels**. Exterior label
      leaders run from `LabelPlacement.tether` (the region's POI,
      deep inside the region) to the exterior anchor, which means
      a leader can visually cross other regions' interior labels.
      Most visible in dense n=4+ ellipse diagrams where several
      exterior labels' rays sweep across the central interior
      labels. Three approaches, increasing in effort:

      1. **Move the tether to the polygon boundary** — set the
         tether to the first ray-vs-region-boundary intersection
         (the point where the ray *exits* the region) instead of
         the POI. The leader then lives entirely outside the
         region; eliminates most leader-vs-interior-label
         crossings since interior labels also sit at POIs inside
         their regions. Cheap — one ray-vs-polygon intersection
         per exterior label, reusing the scan in
         `last_vertex_clearance_t`. Already noted under
         `AGENTS.md` "Future Considerations" as "Exterior
         leader-line entry-point refinement".

      2. **Add leader-vs-interior-label repulsion to
         ForceDirected**. Treat each leader as a line segment;
         when an interior label's AABB intersects the segment,
         push the exterior anchor tangentially until the segment
         clears. Moderate effort; only affects ForceDirected.
         Some tension with existing forces — convergence not
         guaranteed but a few extra iterations usually settle it.

      3. **Route leaders as polylines around obstacles**. Most
         general; works for both Raycast and ForceDirected. Highest
         effort and changes the visual idiom from "straight ray"
         to "polyline". Skip unless bent leaders are explicitly
         desired.

      Recommendation: do (1) first — cheap, on the existing TODO,
      removes the common case. Reach for (2) only if real diagrams
      still show crossings after (1). Surfaced 2026-05-11 during
      the union-polygon raycast refinement.

## Documentation

- [x] **Add `/docs/` routes to the existing Svelte site** (alongside
      rustdoc / tsdoc / `AGENTS.md`). Rustdoc + tsdoc are reference
      docs; `AGENTS.md` is contributor-internal. There's no narrative
      that serves end users (R/Python wrappers) or binding authors
      wiring up downstream packages.

      **Why expand the existing site rather than ship a separate
      mdbook**: eunoia.bz already has the brand, Tailwind styling,
      `LandingPage.svelte`, the embedded `/app/`, the `/cite/` page,
      and one deploy pipeline. A separate mdbook subdomain would
      duplicate all of that with a different theme. The real win of
      keeping it in Svelte is **embedded live `<DiagramViewer>`
      examples** — "here's force-directed vs raycast on the same
      n=4 spec, drag the label-size slider yourself" — which a
      static mdbook can't do.

      Stack: render markdown chapters with
      [`mdsvex`](https://mdsvex.pngwn.io/) (Svelte-native, supports
      Svelte components inside markdown so live demos drop straight
      in). Nav generated from a small `web/src/lib/docs/SUMMARY.ts`
      that mirrors mdbook's `SUMMARY.md` convention.

      Suggested initial route structure (~8 starter pages, most
      stubbed):

      ```
      web/src/lib/docs/
        SUMMARY.ts                   # nav definition
        introduction.md              # what is eunoia, who is this for
        quickstart/
          rust.md
          javascript.md
        concepts/
          fitter-pipeline.md         # MDS init → final stage → normalize → pack
          shapes.md                  # circle/ellipse/square/rectangle, generic design
          label-placement.md         # full guide — see below
          complement.md              # universe / container
        bindings/
          wasm-contract.md           # JSON shapes, RegionPolygons::from_map
          resize-loops.md            # the size→place→measure pattern, in R/JS/Py
        reference.md                 # links to rustdoc, npm types, AGENTS.md
      ```

      Routing: a single `/docs/[...slug]/+page.svelte` (or whatever
      the project's Svelte router uses) that resolves the slug
      against the markdown tree. Sidebar component reads `SUMMARY.ts`
      so adding a chapter only touches one nav file. Cross-link from
      `LandingPage.svelte` ("Docs" button next to the existing
      "Try it in the browser" / "Source code"). Doc pages can deep-
      link into `/app/` with URL-param-encoded specs and strategy
      knobs; the app's existing controls become the interactive
      sandbox the docs reference.

      Write the **label placement** chapter end-to-end as the first
      real content — it's the gap downstream consumers (eulerr R
      bindings, future Python/Julia wrappers) will hit immediately
      and the API just landed (`place_labels`, `placements_bbox`,
      `place_labels_to_fixed_point`, polygon-aware `ForceDirected`).
      Other chapters can stay as one-line "TODO: cover X" stubs until
      a downstream consumer hits the gap. Cover in the label-placement
      chapter:

      1. Mental model — predicate (`fit_labels_in_regions`) vs
         strategy-driven (`place_labels`); raycast vs force-directed.
      2. Size measurement contract — caller measures in user coords,
         eunoia has no font/text knowledge; what `(w, h)` means.
      3. The resize loop — sketched in R, JS, and Python (six lines
         each), with `placements_bbox` as the canvas-extent helper.
         Note that native Rust callers can shortcut to
         `place_labels_to_fixed_point`; FFI callers iterate in their
         host language.
      4. Strategy decision tree — when to pick raycast vs
         force-directed; which params matter (`margin`, `iterations`).
         **Embed a live `<DiagramViewer>`** showing the same spec
         under both strategies side-by-side.
      5. Rendering recipe — `PlacementKind` switch, leader-line
         drawing for exterior placements (kind ∈
         {`ExteriorRaycast`, `ExteriorForceDirected`}), complement-
         region keying (`""`).

      Hosting: same deploy pipeline as the rest of the Svelte site
      (no new workflow). README should keep its quickstart and add a
      one-line link to `eunoia.bz/docs/`.
