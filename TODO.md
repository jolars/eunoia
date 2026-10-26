# TODO

## Python 0.6.0 report: core follow-ups

Triaged from `epost-att-lv9vlQ.html` against core commit `4e9b5fc`. The
standalone [Rust example](crates/eunoia/examples/python_report_repro.rs) uses
synthetic measured boxes and the exclusive counts derived from the original
membership data in [issue #133](https://github.com/jolars/eunoia/issues/133):
`A=35, S=32, U=35, A&S=12, A&U=2, S&U=0, A&S&U=2`, where `A/S/U` denote
Atorvastatin, Simvastatin, and Sunitinib. These are **exclusive region counts**,
not total set sizes. Fitting uses the issue's seed 0 and an explicit budget of
10 restarts.

```sh
cargo run -p eunoia --release --example python_report_repro
```

The example prints measurements; it does not assert that today's shortcomings
must persist. Python rendering follow-ups and a script that generates figures
are in the sibling [eunoia-py TODO](../eunoia-py/TODO.md).

  | Report item                                    | Ownership and finding                                                                                                                                                                                                                   |
  | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | 1: Labels, members, and leaders overlap        | Core obstacle fallback permits label/member overlap. Boundary tethers already exist; choosing them when members are shown belongs in Python.                                                                                            |
  | 2: Relative font sizes change with figure size | Python controls font sizes and their ratio. Core's automatic scale ceiling is the existing `max_scale` follow-up below.                                                                                                                 |
  | 3: Tall lists and exterior overlaps            | Python composes and measures list blocks. In the synthetic long-list reproduction, all four core calls returned non-overlapping supplied boxes, but final rendered text overlapped. Fix the renderer's measurement/viewport loop first. |
  | 4: Long rows and whole groups disappearing     | Core chooses the first row count that fits and returns only a prefix of the input. Both policies are reproduced below. Font defaults and list columns belong in Python.                                                                 |
  | 5: Missing triple intersection                 | Core fitting loses the region in this circle fit. This is the existing topology follow-up below. Python must also report members whose region has no geometry.                                                                          |
  | 6: Font-dependent omissions                    | Expected when measured widths differ. Python should expose omissions clearly and verify that measurement matches rendering; core receives only boxes.                                                                                   |
  | 7: Hidden Plotly title                         | Python's `_plotly._finalize_layout` sets the top margin to zero.                                                                                                                                                                        |

- [x] **Offer strict label obstacles for member and glyph packing (item 1).**
  Both options types now accept `ObstaclePolicy::Strict`, which requires
  obstacle clearance and reports overflow in `unplaced`. `BestEffort`
  remains the default. Size floors, final spreading, and the member-box
  prefix contract are preserved. The policy is exposed through WASM, C, and
  TypeScript, with coverage for both arrangements and both packers,
  including zero-padding boxes. The example compares both policies. Python
  adoption and visible overflow handling remain a separate follow-up.

- [x] **Offer balanced rows for packed member names (item 4).**
  `GlyphBoxOptions::target_aspect_ratio` optionally balances measured widths
  toward a preferred block width-to-height ratio, preserving input order,
  boundary clearance, obstacle policy, and the prefix contract. The default
  still returns the first complete row packing. With twelve `2 x 0.5` boxes,
  a target of `1.0` uses six rows in both `20 x 20` and `40 x 40` regions,
  occupying height `3.625` at the unchanged automatic scale `1.0`. The
  example compares both modes. WASM, C, TypeScript, tests, and docs expose
  the option. Python adoption and multicolumn `mode="list"` remain renderer
  follow-ups.

- [x] **Keep prefix packing and compose whole-region callouts (item 4).** The
  core API still returns a prefix. The web renderer now replaces any
  incomplete roster with one measured block containing its heading, optional
  count, and every name, then repacks the remaining regions. Label placement
  handles the block as a unit, keeping it inside when it fits and placing it
  outside with a leader otherwise. Requested regions without drawable
  geometry retain their names in a separate list marked "no geometry". SVG
  rendering and exports include the blocks and expand their viewport. The
  oversized-first-name fixture, both arrangements, all leader styles, and
  missing/degenerate regions have regression coverage. Python adoption
  remains a renderer follow-up.

Any new public packing option must be mirrored through WASM, C, TypeScript,
tests, and binding docs, then exposed by eunoia-py separately.

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

- [x] **Improve leaders that traverse unrelated regions (Python gallery).** The
  [frozen gallery fixture](crates/eunoia/tests/fixtures/leader_routing_gallery.json)
  captures the rings, measured boxes, and original leaders from "Counts and
  percentages" with seed `0`. Raycast and matched originally traversed about
  `2.08` units of `Treat` for `Anti-CCP&Treat` (count `6`); elbow traversed
  about `11` units of `Anti-CCP`. All three straight strategies now refine
  their exit directions after collision resolution, reducing the reported
  crossing to numerical tolerance for both POI and boundary tethers. The
  refinement includes unlabeled regions, holes, and disconnected pieces,
  preserves source attachment and matched ordering, and accepts only valid
  reductions. Core, C, and TypeScript/WASM regressions cover the fixture.
  Enclosed sources can still require traversal. Elbow routing and Python's
  viewport and font-measurement fixes remain separate.

- [x] **Leader lines crossing interior labels**. Raycast and ForceDirected now
  apply `leader_avoidance_push` against interior label boxes; both have
  regression tests in `plotting/placement.rs`. This does not cover packed
  member boxes, which Python places afterward. Report item 1's default
  leaders still traverse their source region; select a boundary tether
  first. Boundary attachment alone does not guarantee clearance from members
  in other regions. General routing around arbitrary member boxes remains a
  possible extension, pending a reproduction that still crosses members with
  boundary tethers.

- [x] **Leader-line entry-point refinement**. Implemented as
  `TetherSource::Boundary`; `Poi` remains the default. The report example
  shows the tether moving from `(1, 1)` to `(1, 2)` for a `2 x 2` region.
  Python already exposes this as
  `labels={"placement": {"tether": "boundary"}}`.

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

- [ ] **`InteriorPolicy::Loose` and `ExteriorPolicy::None` for `place_labels`**.
  Only `InteriorPolicy::Strict` and the `Raycast` / `ForceDirected` exterior
  policies are implemented; `Loose` interior placement and the `None`
  exterior policy currently return `PlacementError::Unimplemented` (see
  `plotting/placement.rs`). Moved from `AGENTS.md` "Open work" 2026-05-22.

## Glyph placement follow-ups

`place_glyphs` shipped across core/wasm/capi/ts/web (eulerGlyphs-style unit
marks packed per region, `plotting/glyphs.rs`): uniform (hex lattice, spread by
spacing bisection) and random (seeded dart-throwing) arrangements, a
diagram-wide radius auto-sized by feasibility bisection, and a `gap` knob
padding both glyph-vs-glyph spacing and the boundary inset. These are the loose
ends that work deliberately deferred. Surfaced 2026-08-06.

- [x] **Label keep-out**. Done: `GlyphOptions::obstacles` takes a diagram-wide
  list of axis-aligned keep-out boxes (`label_boxes` / `labelObstacles`
  build them from a `place_labels` result), mirrored through wasm/capi/ts
  and wired into the web app. Lattice cells and darts within `r * (1 + gap)`
  of a box are rejected, the lattice anchor escapes a blocked pole of
  inaccessibility, and the auto-radius bisection probes with the boxes
  applied. Two deliberate softenings: the obstacle-aware radius is floored
  at half the obstacle-blind one, and a region that still cannot fit packs
  into its box rather than reporting a shortfall --- otherwise one small
  region whose label nearly fills it would shrink every glyph in the
  diagram.

- [x] **Member text labels as glyphs**. Done: `place_glyph_boxes` is a sibling
  entry point taking per-item measured `(w, h)` boxes instead of counts,
  with a single diagram-wide `scale` bisected in `[min_scale, 1.0]`
  (shrink-only --- the caller owns the reference font size). The uniform
  arrangement packs rows over a new exact scan-line band oracle
  (`plotting/glyphs/scan.rs`); the random one throws rectangular darts. The
  obstacle plumbing, apportionment, and `PROBE`/`PACK` split are shared with
  the disc packer, which moved to `plotting/glyphs/discs.rs`. Mirrored
  through wasm/capi/ts + a `toSvg` `glyphBoxes` mode. `boxes[key]` is a
  prefix of `sizes[key]`, so drop order is the caller's item order.

- [x] **Web app doesn't expose member labels** (done). The UI decision was a
  per-row roster: `Row.members` is an optional comma/newline-separated
  string edited on a second line of each combination row in
  `SpecEditor.svelte`, and the default rows ship with names so the mode
  demos itself. The glyph knobs outgrew `StyleControls.svelte` and moved to
  their own collapsed `Glyphs` sidebar section (`GlyphControls.svelte`): the
  `showGlyphs` checkbox became a `glyphMode` select (`none` / `dots` /
  `members`) sharing the arrangement/spacing/seed knobs, plus a
  `memberLabelSize` slider --- its own knob because the packer is
  shrink-only, so the reference size is a ceiling rather than a hint.
  `DiagramSvg.svelte` measures the names in a second hidden `<text>` pass
  inside the existing `data-fit-measure` group, packs them with
  `placeGlyphBoxesForRegions` under the same `labelObstacles` keep-outs as
  the disc packer, and renders via the `glyphBoxes` serializer option at a
  pinned weight of 400 (the Bold toggle belongs to the set names, and
  measuring at 400 while rendering at 700 would overflow every box).
  `web/src/lib/members.svelte.ts` holds the glue the core does not provide:
  roster parsing, order-insensitive matching of a typed `B&A` row to the
  core's canonical `A&B` region key, and the rune store carrying `unplaced`
  to the inline overflow note. Member names are Euler-only (Venn is driven
  by `vennN`, not the rows) and the complement region has no roster. A
  500-name budget mirrors `MAX_GLYPHS`. Persisted `showGlyphs` migrates to
  `glyphMode` in `loadPersisted`.

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

- [x] **Whole-region callouts for overflowing member rosters (web).**
  `web/src/lib/member-layout.ts` composes the existing box packer and label
  placers. Every region with an incomplete packing switches permanently to a
  measured roster block for that pass. The block shares label collision
  resolution with region headings and counts; set names are placed against
  the resulting boxes, and the surviving rosters use strict obstacles.
  Repacking can evict more regions as label obstacles move, but each
  iteration removes at least one roster, so termination does not depend on
  the scale increasing monotonically. All measurements stay in layout units
  rather than feeding viewport expansion back into text sizes.

  `ToSvgOptions.memberCallouts` renders the measured lines, preserves leaders,
  replaces the ordinary region label, and includes each block in the viewport.
  Missing and degenerate regions get a separate labeled list without an invented
  leader target. The core and its prefix contract are unchanged; Python
  integration remains separate. Leader crossings still depend on the selected
  label-placement strategy, as they do for ordinary exterior labels.

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

- [x] **Quality-harness coverage** (done). `quality_report`, `corpus_quality`,
  and `synthetic_groundtruth` now include `RotatedRectangle`. The report
  benchmarks default, Nelder--Mead, CMA-ES, and MADS configurations for it,
  includes all five shapes in Markdown and JSON, and accepts `--config` and
  `--shape` filters. The initial default sweep (2026-10-14, 30 specs × 16
  seeds × 5 shapes) completed without fit errors. Rotated rectangles had
  median SumSquared loss `4.75e-5`; the worst `diag_error` was `8.90e-2` on
  `issue44_4_set_inclusive`. `corpus.rs` records per-spec ceilings with
  headroom over this baseline. The synthetic test checks recovery of 50
  deterministic, nondegenerate rotated-rectangle configurations, with the
  existing `5e-2` diagnostic-error limit.

- [ ] **Fit quality may motivate corner-rounding**. The shape is fitted
  derivative-free by design (no analytic gradient). If the quality harness
  above shows it underperforming the gradient shapes, the principled upgrade
  is a rounded-rectangle / superellipse family: rounding the corners makes
  the overlap area C¹ again, which restores a usable analytic gradient and
  re-enables the LM/TRF path (flip `SUPPORTS_ANALYTIC_GRADIENT` to `true`).
  This is the "smooth the shape" option from the original design discussion;
  revisit only if derivative-free quality proves insufficient.

- [x] **TS handle-freeing is wasteful** (done). The `euler` / `venn` dispatch in
  `ts/index.ts` now reads and frees only the arrays used in the returned
  layout. Unused shape arrays and polygons in `shapes` mode stay inside the
  result until `result.free()` drops them, avoiding getter clones.

- [x] **Narrative docs** (done). README and core rustdoc list all five fitting
  shapes. The Shapes page covers rotated-rectangle parameters,
  derivative-free fitting, an interactive example, and the four-set
  canonical Venn limit. Language quickstarts, the JavaScript reference, C
  ABI Venn docs, page metadata, and `llms.txt` are updated. The TypeScript
  `venn()` range check now accepts four sets for `"rotatedRectangle"`, with
  regression tests for both output modes and rejection of five sets.
