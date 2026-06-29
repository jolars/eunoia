# TODO

## Surfaced fitter issues (regressions to investigate)

The corpus / proptest surfaced these. None were introduced by the harness;
they're pre-existing behaviour the harness now exposes.

- [ ] **`random_4_set`ellipses land at `diag_error ≈ 2.6e-2`**. The corpus
  ceiling is tightened to `3e-2` (was `5e-2`) since this basin is a
  deterministic floor across most master seeds. There are at least two
  distinct local minima: - **basin A** (loss `7.786e-3`, diag `2.606e-2`)
  --- reached by \~13/16 `QUALITY_SEEDS` master seeds at default
  `n_restarts=10`. - **basin B** (loss `4.335e-3`, diag `1.147e-2`) ---
  reached by the other \~3/16. An even slightly better basin (loss
  `4.086e-3`) shows up at `n_restarts ≥ 40` but only as the global-min,
  never the median.

  The basins differ in which ellipse area maps to which set's target (basin A
  nails `C ≈ 4.24`, `E ≈ 4.56` and undershoots `A`/`B`; basin B spreads the
  error more evenly). Neither raising `n_restarts` to 100 nor forcing
  `Optimizer::CmaEsLm` to fire on every restart
  (`cmaes_fallback_threshold = 1.0`) shifts the median off basin A --- CMA-ES at
  default budget / box doesn't span the basin gap from this MDS init. Probed via
  a throwaway example (deleted after measurement).

  Worth re-checking after any optimizer redesign that touches the global stage;
  tighten the ceiling further if a future change closes basin A.

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

  1. **Move the tether to the polygon boundary** --- set the tether to the first
     ray-vs-region-boundary intersection (the point where the ray *exits* the
     region) instead of the POI. The leader then lives entirely outside the
     region; eliminates most leader-vs-interior-label crossings since interior
     labels also sit at POIs inside their regions. Cheap --- one ray-vs-polygon
     intersection per exterior label, reusing the scan in
     `last_vertex_clearance_t`. Already noted under `AGENTS.md` "Future
     Considerations" as "Exterior leader-line entry-point refinement".

  2. **Add leader-vs-interior-label repulsion to ForceDirected**. Treat each
     leader as a line segment; when an interior label's AABB intersects the
     segment, push the exterior anchor tangentially until the segment clears.
     Moderate effort; only affects ForceDirected. Some tension with existing
     forces --- convergence not guaranteed but a few extra iterations usually
     settle it.

  3. **Route leaders as polylines around obstacles**. Most general; works for
     both Raycast and ForceDirected. Highest effort and changes the visual idiom
     from "straight ray" to "polyline". Skip unless bent leaders are explicitly
     desired.

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

## RotatedRectangle follow-ups

The `RotatedRectangle` shape shipped across core/fitter/capi/wasm/ts (commit
`12b272d`, 2026-06-21): an oriented box fitted derivative-free (exact
Sutherland--Hodgman convex-clip overlap is only piecewise-C¹, so it carries no
analytic gradient and the capability-driven default pool routes it to
`[NelderMead, CmaEs]`). These are the loose ends that PR did not cover.

- [x] **Web app doesn't expose the shape** (done). `ShapeType` in
  `web/src/lib/types/diagram.ts` and the `FitResult.shapeType` field now
  carry the `"rotatedRectangle"` variant, the `scaleLayout` dispatch in
  `web/src/lib/fit.ts` has a `rotatedRectangle` case (rotation passes
  through unscaled, like ellipse), and `SpecEditor.svelte` adds a "Rotated
  rectangle" radio. Rendering rides the existing polygons path through
  `@jolars/eunoia/svg`, so no serializer change was needed. No shape-param
  geometry readout exists in the app, so there was nothing to surface
  rotation in. The landing-page `HeroWidget` keeps its curated
  circle/ellipse/square/ rectangle subset (live slider re-fits stay fast).

- [ ] **No quality-harness coverage** (near-term, actionable). `quality_report`
  (`crates/eunoia/examples/quality_report.rs`), `corpus_quality`, and
  `synthetic_groundtruth` run Circle/Ellipse/Square/Rectangle but not
  `RotatedRectangle` (and `corpus.rs` has no per-shape treatment for it).
  There is therefore no regression guardrail and no benchmark of the
  derivative-free fit quality versus the gradient-based shapes. Add a
  `RotatedRectangle` config to `quality_report` and a corpus ceiling before
  relying on the shape's fit quality; treat the numbers as the baseline.

- [ ] **Fit quality may motivate corner-rounding**. The shape is fitted
  derivative-free by design (no analytic gradient). If the quality harness
  above shows it underperforming the gradient shapes, the principled upgrade
  is a rounded-rectangle / superellipse family: rounding the corners makes
  the overlap area C¹ again, which restores a usable analytic gradient and
  re-enables the LM/TRF path (flip `SUPPORTS_ANALYTIC_GRADIENT` to `true`).
  This is the "smooth the shape" option from the original design discussion;
  revisit only if derivative-free quality proves insufficient.

- [ ] **TS handle-freeing is wasteful** (minor cleanup). The `euler` / `venn`
  dispatch in `ts/index.ts` frees the non-active shape arrays per branch by
  *accessing* each getter (which clones the wasm handles) only to free the
  clones --- `result.free()` already drops the internal vectors, so a branch
  that never reads a getter leaks nothing. Pre-existing pattern, now
  extended to `rotated_rectangles`. If that dispatch is ever revisited, drop
  the create-then-free of unused shape arrays.

- [ ] **Narrative docs**. <https://eunoia.bz/docs/> and the rustdoc/README shape
  lists don't mention `RotatedRectangle`. Add it once the web UI and quality
  baseline land.
