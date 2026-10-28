# TODO

## Surfaced fitter issues (regressions to investigate)

The corpus / proptest surfaced these. None were introduced by the harness;
they're pre-existing behaviour the harness now exposes.

- [ ] **`random_4_set`ellipses land at `diag_error ≈ 2.6e-2`**. The corpus
  ceiling is tightened to `3e-2` (was `5e-2`) since this basin is a
  deterministic floor across most master seeds. There are at least two
  distinct local minima: - **basin A** (loss `7.786e-3`, diag `2.606e-2`) ---
  reached by \~13/16 `QUALITY_SEEDS` master seeds at default
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

## Loss / topology follow-ups

- [ ] **The loss has no topology term, so the optimum can be topologically
  wrong**. Every `LossType` scores only per-region area residuals, so
  nothing distinguishes "region with target 0 drawn at positive area" (a
  *false* intersection the data doesn't contain) from an equal-magnitude
  area error on a region that legitimately exists. Both are just residual.

  Issue #133 is the clean demonstration. The spec has `S&U = 0` while
  `A&S&U = 2`, i.e. the whole `S∩U` overlap must nest inside `A`:

  - **Square / `sum_squared`**: the optimum (`Σ(f-t)² = 0.82` after 9dc14dd)
    draws `S&U = 0.354` against a target of `0` --- a visible false intersection
    that is *genuinely optimal* for the objective.
  - **Rectangle / `sum_squared`** fits it exactly since 9dc14dd (`3.9e-25`,
    `S&U = 0`, `A&S&U = 2`). Before that fix rectangles looked like another
    instance of this item; they were not. Do not assume a topologically wrong
    optimum is inherent without first checking the fitter actually reached the
    optimum.
  - **Circle / `sum_squared`**: the optimum (`Σ(f-t)² = 4.0`) instead drops
    `A&S&U` from `2` to `0`, omitting a region that does exist. Confirmed
    global: an independent pure-Python solver (exact analytic circle areas,
    validated to `1e-7` against numerical integration, 3000 multistarts) reaches
    exactly `4.000000`, as do all six eunoia optimizers. Three circles simply
    cannot represent this spec.
  - **Ellipse / `sum_squared`** fits it exactly (loss `1.3e-23`).
  - Under `max_absolute` the effect is worse across the board, which is inherent
    to minimax: it spreads error evenly rather than concentrating it, so several
    regions land \~1.4 off instead of one landing 2 off, and the false
    intersection becomes visible even for ellipses.

  **Plotting is not implicated** and should not be re-investigated: over 160
  fits, `plot_data` region areas match the fitted areas to `1e-8` for squares
  and rectangles (exact polygonal decomposition), with only \~`1e-3`
  inscribed-polygon discretization on curved shapes, erring *downward*.
  `eunoia-py`'s `_plot.py` renders those rings verbatim with no geometry of its
  own.

  Next step is to measure what topological correctness costs: the best
  achievable loss *subject to* no false and no missing regions. If it's cheap
  (squares going from `0.82` to \~`1.5`), a penalty or lexicographic tie-break
  on topology is worth adding; if it's expensive, document the trade-off and
  steer users to ellipses or rectangles. Surfaced 2026-08-14 from issue #133.

  **Python report item 5:** `python_report_repro` now retains a runnable
  reproduction of the same topology: circles fit the triple target `2` as `0`,
  with raw squared residual sum `4`, and no triple polygon. Even tiny
  `0.01 x 0.01` member boxes cannot restore missing geometry. Ellipses and
  rectangles fit the triple as `2` and place both names. Squares and rotated
  rectangles retain a positive triple but have nonzero residuals in this run.
  `Layout::loss()` is normalized; compare `raw_sse` in the example with the
  unnormalized objectives quoted above. This single seed does not establish
  global optimality. The low-level packer intentionally ignores absent region
  keys, so the Python omission diagnostic is a separate renderer follow-up.

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

## Label placement follow-ups

- [ ] **Nested sets in `place_set_labels`**. The exterior set-label mode
  guarantees label boxes don't overlap each other, but clearance from the
  *shapes* is best-effort: a set nested inside a much larger one has a hug
  ring entirely inside its container, so no candidate angle clears it and
  the sweep returns the least-bad one (the box then straddles the
  container's boundary). Options if this proves annoying: fall back to the
  interior anchor when the best clearance is negative, or relax the hug
  constraint and let the label escape to the diagram exterior with a leader
  — which is really asking for a hybrid of `place_set_labels` and
  `place_labels`, so design the two together rather than bolting a leader
  onto this mode. Surfaced 2026-08-14 when the mode landed.

## Glyph placement follow-ups

`place_glyphs` shipped across core/wasm/capi/ts/web (eulerGlyphs-style unit
marks packed per region, `plotting/glyphs.rs`): uniform (hex lattice, spread by
spacing bisection) and random (seeded dart-throwing) arrangements, a
diagram-wide radius auto-sized by feasibility bisection, and a `gap` knob
padding both glyph-vs-glyph spacing and the boundary inset. These are the loose
ends that work deliberately deferred. Surfaced 2026-08-06.

- [ ] **`max_scale` for glyph boxes** (minor). The auto-scale bracket's upper
  end is hardcoded to `1.0`, so a roomy diagram renders sparse text at the
  caller's reference size rather than growing to fill --- deliberately
  asymmetric with `place_glyphs`, which does grow. `GlyphBoxOptions` is
  `#[non_exhaustive]`, so a `max_scale` field defaulting to `1.0` can land
  later without a break if the asymmetry proves annoying in practice. Python
  report items 2/4 demonstrate this behavior; `python_report_repro` prints
  scale `1.0` for both roomy fixtures. Explicit `.scale(k)` already permits
  `k > 1`; the missing capability is automatic growth. Python owns reference
  font sizes, the label/member size ratio, and readability floors.

- [ ] **Relaxation pass for the box footprint**. The disc packer's `random`
  arrangement now finishes with a force-directed relaxation
  (`plotting/glyphs/relax.rs`): neighbour repulsion plus a
  `signed_clearance` boundary push, over a fixed sweep count, with every
  move accepted only if it preserves the packer's own invariants.
  `pack_random_boxes_piece` gets none of it --- heterogeneous footprints
  have no single spacing to relax toward, so the force model would have to
  be per-pair AABB separation and the acceptance test `rect_fits_in_piece`.
  Worth doing only if the scattered box mode ever looks bad enough in
  practice to earn the complexity. A Bridson Poisson-disk sampler remains
  the alternative blue-noise upgrade for the disc packer; phyllotaxis is a
  further arrangement candidate. The uniform mode already spreads
  deterministically and needs neither.

- [ ] **Random-packer spatial hash** (perf, only if needed). Dart acceptance
  checks are O(placed) per dart, O(n²) per piece overall --- fine at the
  hundreds-of-glyphs scale the web app caps at (2000), noticeable beyond. A
  uniform grid keyed at cell size `2r(1+gap)` makes it O(1) per dart. The
  box packer's `pack_random_boxes_piece` has the same shape (per-pair AABB
  separation instead of one squared distance) and would want the same fix,
  as do the relaxation pass's neighbour-force and acceptance loops --- those
  two are now essentially all of that pass's cost (~20ms of an ~85ms random
  pack at 2200 glyphs; it already caches one `signed_clearance` per point so
  the sweeps don't re-walk the rings).

- [ ] **Counts-from-spec convenience**. The web app rounds `metrics.target`
  (exclusive quantities) into counts client-side (`DiagramSvg.svelte`); the
  same convenience could ship in the TS wrapper (e.g.
  `glyphCountsFromLayout(layout)`) so other consumers don't re-derive it.
  Core stays counts-only --- spec quantities are `f64` areas, not
  cardinalities, so the rounding policy belongs to the caller.

## RotatedRectangle follow-ups

The `RotatedRectangle` shape shipped across core/fitter/capi/wasm/ts (commit
`12b272d`, 2026-06-21): an oriented box fitted derivative-free (exact
Sutherland--Hodgman convex-clip overlap is only piecewise-C¹, so it carries no
analytic gradient and the capability-driven default pool routes it to
`[NelderMead, CmaEs]`). These are the loose ends that PR did not cover.

- [ ] **Fit quality may motivate corner-rounding**. The shape is fitted
  derivative-free by design (no analytic gradient). If the quality harness
  above shows it underperforming the gradient shapes, the principled upgrade
  is a rounded-rectangle / superellipse family: rounding the corners makes
  the overlap area C¹ again, which restores a usable analytic gradient and
  re-enables the LM/TRF path (flip `SUPPORTS_ANALYTIC_GRADIENT` to `true`).
  This is the "smooth the shape" option from the original design discussion;
  revisit only if derivative-free quality proves insufficient.
