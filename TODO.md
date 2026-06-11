# TODO

## Surfaced fitter issues (regressions to investigate)

The corpus / proptest surfaced these. None were introduced by the harness;
they're pre-existing behaviour the harness now exposes.

- [ ] **`issue71_4_set_extreme_scale`ellipse seed=1 lands in a different basin
      on Windows** (diag `~1.5e-1` vs Linux `~1.9e-4`). The spec's
      4-order-of-magnitude area variation (A=38066 vs D=6) makes the final-stage
      optimisation sensitive to FP rounding in `sin`/`cos`/conic-intersection
      math, and Windows's MSVC math runtime returns slightly different ULP
      values than glibc. Other `TEST_SEEDS` entries (42, 7) match Linux on
      Windows. Worked around with a platform-conditional ceiling
      (`ISSUE71_ELLIPSE_CEILING` in `crates/eunoia/src/test_utils/corpus.rs`):
      Linux/macOS at `5e-2`, Windows at `2e-1`. Real fix would tighten the
      optimizer's basin- of-attraction on extreme-scale specs (e.g. better
      `NormalizedSumSquared` conditioning, scale-aware initial perturbation, or
      a tighter MDS init). Not blocking; the platform split keeps the Linux
      ceiling strict so future regressions on dev machines / Linux CI still
      trip.

- [ ] **`random_4_set`ellipses land at `diag_error ≈ 2.6e-2`**. The corpus
      ceiling is tightened to `3e-2` (was `5e-2`) since this basin is a
      deterministic floor across most master seeds. There are at least two
      distinct local minima:
      - **basin A** (loss `7.786e-3`, diag `2.606e-2`) --- reached by \~13/16
        `QUALITY_SEEDS` master seeds at default `n_restarts=10`.
      - **basin B** (loss `4.335e-3`, diag `1.147e-2`) --- reached by the other
        \~3/16. An even slightly better basin (loss `4.086e-3`) shows up at
        `n_restarts ≥ 40` but only as the global-min, never the median.

      ```
        The basins differ in which ellipse area maps to which set's target
        (basin A nails `C ≈ 4.24`, `E ≈ 4.56` and undershoots `A`/`B`; basin B
        spreads the error more evenly). Neither raising `n_restarts` to 100 nor
        forcing `Optimizer::CmaEsLm` to fire on every restart
        (`cmaes_fallback_threshold = 1.0`) shifts the median off basin A —
        CMA-ES at default budget / box doesn't span the basin gap from this
        MDS init. Probed via a throwaway example (deleted after measurement).

        Worth re-checking after any optimizer redesign that touches the global
        stage; tighten the ceiling further if a future change closes basin A.
      ```

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

- [ ] **issue89 (17-set) ellipse fits are highly multimodal and the global
      escape doesn't help**. Best-of-`n_restarts=10` ellipse stress swings
      wildly with the master seed: `1.8e-3`/`1.9e-3`/`2.3e-3`/`2.4e-3` (good) on
      seeds 8/5/3/7, but `2.9e-2`, `1.6e-1`, `6.1e-1`, `9.8e-1` (poor → failed)
      on seeds 1/4/6/2. The CMA-ES global escape provides **no** benefit here:
      every restart's plain-LM loss is above `cmaes_fallback_threshold` (1e-3)
      so the escape fires on all of them, yet `Optimizer::CmaEsTrf`,
      `Optimizer::CmaEsLm`, and bare `Optimizer::LevenbergMarquardt` produce
      *bit-identical* per-seed stress --- i.e. the escape + polish never beats
      the plain-LM result, and the lower loss kept is always LM's. So quality is
      determined entirely by the MDS init + local LM convergence, and the escape
      stage is pure wasted compute for this spec. Not caused by the LM→TRF
      default switch in 8eda26d (CmaEsLm ≡ CmaEsTrf here) nor by the sparse-mask
      perf fix (mathematically identical; circle fits are bit-identical
      before/after). Most likely root cause is the circle-equivalent ellipse MDS
      warm-start (see "Ellipse MDS still warm-starts as a circle" below): 17
      ellipses seeded with no rotational information land in bad rotational
      basins the local solver can't leave and CMA-ES can't span. Probed via a
      throwaway example (deleted after measurement). Surfaced 2026-05-27.
      Circles fit fine (stress consistently `~2e-3`). Worth re-checking after
      any MDS or global-stage redesign.

## MDS architecture follow-ups

- [ ] **Ellipse MDS still warm-starts as a circle**.
      `Ellipse::mds_target_distance` delegates to `Circle::mds_target_distance`
      (treating each ellipse as a circle of equal area), and the MDS phase
      optimises only 2D centers --- orientation `φ` and the `a/b` ratio never
      enter the MDS loss. This is defensible (the final-stage optimizer takes
      over and reshapes from there, and most corpus specs hit machine precision)
      but it has a real blind spot: ellipses with large axis ratio overlapping
      along their major vs minor axis produce the same overlap area at very
      different center distances, and the circle-equivalent inversion picks one
      canonical distance somewhere between. Ellipse fits then have no rotational
      seed information; the optimizer rediscovers `φ` from scratch, which is
      part of why ellipse fits need higher `n_restarts` and the CMA-ES escape
      stage. A correct ellipse MDS would optimise over `[x, y, a, b, φ]` per
      shape against overlap targets directly (the larger refactor sketched as
      option (1) during the Square design --- kept out of the Square PR because
      the existing MDS solver pool --- Lbfgs, TrustRegion, NewtonCg,
      LevenbergMarquardt --- all have analytical gradients/Hessians/Jacobians
      wired to the 2D positional cost, and rederiving them for ellipse
      parameters means inheriting ellipse's intricate boundary-integral
      derivative. Worth revisiting when (a) we have concrete specs where ellipse
      fits miss because of a wrong rotational basin out of MDS, or (b) we're
      doing the shape-aware-MDS refactor anyway for triangles or another shape
      that can't reasonably warm-start as a circle.

## Complement / container follow-ups

Loose ends from S6 of the complement roadmap. The feature itself shipped (see
the complement bullet in `AGENTS.md` Status); these are polish items the roadmap
didn't require but would tighten the surface.

- [ ] **Migrate or delete the six legacy WASM entry points**. The S6 WASM pass
      added a shared `build_diagram_spec(specs, input_type, complement)` helper
      and routed the modern `generate_*_as_polygons` /
      `generate_region_polygons_*` / `generate_venn_*` functions through it. The
      older entry points still inline their `DiagramSpecBuilder` block and don't
      accept a `complement` argument: `generate_from_spec`,
      `generate_from_spec_with_debug`, `get_debug_info`,
      `get_debug_info_simple`, `generate_from_spec_initial`,
      `get_debug_info_initial` (in `crates/eunoia-wasm/src/lib.rs`). Neither
      `ts/index.ts` nor the web app calls any of them --- they're legacy / debug
      paths from before the polygon-mode backend. Either migrate them to the
      helper (consistency, complement support) or delete them outright (smaller
      WASM surface). Deletion is probably the right call;
      `generate_circles_as_polygons` already covers everything
      `generate_from_spec` did, with strictly more diagnostics.

- [ ] **Web app: validation feedback for the complement input**. The
      `Complement (universe)` numeric in
      `web/src/lib/components/SpecEditor.svelte` uses `min="0"` but
      `<input type="number">` doesn't actually block negative entry --- `runFit`
      (`web/src/lib/fit.ts`) silently treats anything `< 0` or `NaN` as "not
      set", which is a confusing fail-soft. Show an inline error (matching the
      row-input style) when the value is invalid, or clamp on input.

- [ ] **Legend entry for the complement region**. `DiagramSvg.svelte` renders
      the container as a dashed grey frame and (when `style.showCounts`) labels
      the count in the top-right corner. The legend (`StyleControls` →
      `legendPosition`) doesn't include a "complement" / "outside" swatch when a
      container is present. Adding one means deciding the right label
      ("complement", "outside", a user-supplied name?) and surfacing it through
      the existing legend build path.

## Label placement follow-ups

- [ ] **Leader lines crossing interior labels**. Exterior label leaders run from
      `LabelPlacement.tether` (the region's POI, deep inside the region) to the
      exterior anchor, which means a leader can visually cross other regions'
      interior labels. Most visible in dense n=4+ ellipse diagrams where several
      exterior labels' rays sweep across the central interior labels. Three
      approaches, increasing in effort:

      1. **Move the tether to the polygon boundary** --- set the tether to the
         first ray-vs-region-boundary intersection (the point where the ray
         *exits* the region) instead of the POI. The leader then lives entirely
         outside the region; eliminates most leader-vs-interior-label crossings
         since interior labels also sit at POIs inside their regions. Cheap ---
         one ray-vs-polygon intersection per exterior label, reusing the scan in
         `last_vertex_clearance_t`. Already noted under `AGENTS.md` "Future
         Considerations" as "Exterior leader-line entry-point refinement".

      2. **Add leader-vs-interior-label repulsion to ForceDirected**. Treat each
         leader as a line segment; when an interior label's AABB intersects the
         segment, push the exterior anchor tangentially until the segment
         clears. Moderate effort; only affects ForceDirected. Some tension with
         existing forces --- convergence not guaranteed but a few extra
         iterations usually settle it.

      3. **Route leaders as polylines around obstacles**. Most general; works
         for both Raycast and ForceDirected. Highest effort and changes the
         visual idiom from "straight ray" to "polyline". Skip unless bent
         leaders are explicitly desired.

      ```
        Recommendation: do (1) first — cheap, on the existing TODO,
        removes the common case. Reach for (2) only if real diagrams
        still show crossings after (1). Surfaced 2026-05-11 during
        the union-polygon raycast refinement.
      ```

- [ ] **Leader-line entry-point refinement**. Start the leader at the first
      ray--region-boundary intersection (where the ray exits the region) rather
      than at the POI. This is exactly approach (1) of the "Leader lines
      crossing interior labels" item above --- see there for the detail. Moved
      from `AGENTS.md` "Open work" 2026-05-22.

- [ ] **`InteriorPolicy::Loose` and `ExteriorPolicy::None` for `place_labels`**.
      Only `InteriorPolicy::Strict` and the `Raycast` / `ForceDirected` exterior
      policies are implemented; `Loose` interior placement and the `None`
      exterior policy currently return `PlacementError::Unimplemented` (see
      `plotting/placement.rs`). Moved from `AGENTS.md` "Open work" 2026-05-22.

## Language bindings

- [ ] **Julia bindings**. The core is platform-independent and designed for
      multiple language bindings; Julia is a planned target alongside the
      existing WASM/TS surface. Moved from `AGENTS.md` "Open work" 2026-05-22.
