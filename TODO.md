# TODO

## Eunoia 2.0

- [ ] Remove the deprecated TRF optimizer aliases from Rust and WASM (`Trf`,
  `CmaEsTrf`), TypeScript (`trf`, `cmaEsTrf`), and the C ABI (`trf`,
  `cmaes_trf`). Update the compatibility tests and documentation
  accordingly.

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

## Label placement follow-ups

- [x] **Nested sets in `place_set_labels`**. Fully enclosing sets now score
  clearance on their inside, so a nested set's label favors room between its
  own outline and the enclosing boundary. Polygon containment distinguishes
  nesting from partial overlap; coincident outlines still send labels
  outside. The label keeps hugging its own shape, with no leader. Shape
  clearance is still best-effort when the gap cannot fit the label or the
  collision guard moves it. An exterior escape with a leader would require a
  joint design with `place_labels`. Surfaced 2026-08-14 when the mode
  landed.

## Glyph placement follow-ups

`place_glyphs` shipped across core/wasm/capi/ts/web (eulerGlyphs-style unit
marks packed per region, `plotting/glyphs.rs`): uniform (hex lattice, spread by
spacing bisection) and random (seeded dart-throwing) arrangements, a
diagram-wide radius auto-sized by feasibility bisection, and a `gap` knob
padding both glyph-vs-glyph spacing and the boundary inset. These are the loose
ends that work deliberately deferred. Surfaced 2026-08-06.

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
