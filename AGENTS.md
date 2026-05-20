# Eunoia — Copilot Instructions

**Keep this file current.** When changes affect architecture, public API, or
the patterns described here, update this file in the same PR.

## What is Eunoia

A Rust library for area-proportional Euler and Venn diagrams. Modern rewrite
of the eulerr R package (C++). Designed for multiple language bindings; the
core is platform-independent and the WASM bindings live in this repo.

Pipeline:

1. Initial layout via positional MDS with shape-aware target distances.
2. Final shape-specific optimization minimising a selectable loss.
3. Post-fit normalization (cluster, rotate, centre, skyline pack).
4. Optional polygonisation and label placement.

## Supported shapes

Implemented as `DiagramShape`: `Circle`, `Ellipse`, `Square` (axis-aligned),
`Rectangle` (axis-aligned). Planned: rotated rectangles, triangles.

Geometric vs optimizer encodings (`to_params` / `to_optimizer_params`):

| Shape     | Geometric           | Optimizer encoding                   |
| --------- | ------------------- | ------------------------------------ |
| Circle    | `[x, y, r]`         | identity                             |
| Ellipse   | `[x, y, a, b, φ]`   | `[x, y, ln a, ln b, φ]`              |
| Square    | `[x, y, side]`      | identity                             |
| Rectangle | `[x, y, w, h]`      | `[x, y, ln(w·h), ln(w/h)]`           |

FFI / external callers want the geometric encoding (`to_params`,
`from_params`, and per-shape accessors). The optimizer encoding is internal
to the fitter.

For squares and rectangles, every n-way intersection is itself an
axis-aligned rectangle, so `compute_exclusive_regions` is exact in closed
form. For circles and ellipses the intersections use exact
conic/polysegments, including 3+ way.

## Workspace layout

Cargo workspace, edition 2024, MSRV 1.91.1 (pinned for rextendr / R
compatibility). Workspace version is shared.

- `crates/eunoia/` — core library, no platform deps
- `crates/eunoia-wasm/` — wasm-bindgen wrapper (`cdylib`+`rlib`), depends on
  the core with `features = ["wasm", "plotting"]`
- `ts/` — hand-written TypeScript surface (`euler`, `venn`, `Layout`,
  `placeLabelsForRegions`, …) compiled to `npm/` by `prepare-package.mjs`
- `npm/` — generated, gitignored; published as `@jolars/eunoia`
- `web/` — SvelteKit app (Svelte 5, Vite/rolldown-vite, Tailwind v4),
  consumes `@jolars/eunoia` via `file:../npm`

### Core modules (`crates/eunoia/src/`)

- `spec/` — `Combination`, `InputType` (`Exclusive` / `Inclusive`),
  `DiagramSpecBuilder` (shape-agnostic; complement via `.complement(value)`)
- `geometry/`
  - `primitives/` — `Point`, `Line`, `LineSegment`
  - `projective/` — projective helpers (`point`, `line`, `conic`)
  - `traits.rs` — `Area`, `Centroid`, `Distance`, `Perimeter`,
    `BoundingBox`, `Closed`, `DiagramShape`
  - `shapes/` — `circle`, `ellipse`, `square`, `rectangle`, `polygon`
  - `operations/overlaps.rs`
  - `diagram.rs` — region discovery and inclusion–exclusion combiner
- `fitter/`
  - `initial_layout.rs` — MDS warm-start, `InitialSampler`, `MdsSolver`
  - `final_layout.rs` — `Optimizer`, dispatch
  - `cmaes.rs` — inline purecma-style CMA-ES (no extra dep)
  - `clustering.rs`, `normalize.rs`, `packing.rs`
  - `layout.rs` — `Layout<S>`, `Layout::container()`, quality metrics
  - `corpus_quality.rs`, `synthetic_groundtruth.rs` — corpus-driven evaluation
- `loss.rs` — `LossType` (`SumSquared` default, `Stress`, region-error and
  smoothed variants, etc.)
- `plotting/` (feature `plotting`) — `clip`, `regions`, `placement`,
  `inscribed`, `plot_data`
- `venn.rs` — canonical Venn templates (circles n≤3, Wilkinson/Edwards
  ellipses n∈{4,5}, axis-aligned squares n∈{2,3}, rectangles n∈{1,2,3});
  used as warm-start slot 0 in `Fitter`
- `math/`, `error.rs`, `constants.rs`

## Optimization strategy

1. **MDS initial layout** — sizes derived from set areas; only `(x, y)` is
   optimised. Each shape inverts its own pairwise overlap formula via
   `DiagramShape::mds_target_distance`. Ellipses, squares, and rectangles
   all warm-start through their equal-area circle/square approximation.

2. **Optimizer parameter conversion** — `DiagramShape::optimizer_params_from_circle`
   maps `[x, y, r]` into the per-shape optimizer encoding (see table above).

3. **Final optimization** — selectable via `Optimizer`:
   - `LevenbergMarquardt` (`basin::LevenbergMarquardt`, nalgebra backend;
     MINPACK-style `gtol`/`ftol`/`xtol` termination)
   - `Lbfgs` (argmin)
   - `NelderMead` (`basin::NelderMead`)
   - `CmaEsLm` *(default)* — plain LM first; if it stays above
     `Fitter::cmaes_fallback_threshold` (default `1e-3` on
     `NormalizedSumSquared`), runs a CMA-ES → LM polish and keeps the lower
     loss. Strictly non-regressing vs LM.

   `MdsSolver` for the initial stage: `Lbfgs` (default), `TrustRegion`,
   `NewtonCg`, `LevenbergMarquardt`.

   Cycling pools available via `Fitter::initial_solver_pool` /
   `Fitter::optimizer_pool`.

4. **Analytical gradients** for circles and ellipses with `NormalizedSumSquared`
   / `SumSquared`, derived from boundary-velocity on a CCW arc decomposition
   and chained through inclusion–exclusion. Rectangles have analytical
   edge-velocity gradients with full chain-rule into the log-area / log-ratio
   basis. Falls back to central FD when the loss has no analytical form.

5. **Venn warm-start** in slot 0 of `n_restarts` — replaces (not adds to)
   the first random restart with the canonical Venn template for the shape
   and `n_sets`. Auto-skipped when out of range or when the spec has any
   disjoint pair (Venn topology requires all regions positive).

## Complement / "container" support

Opt-in via `DiagramSpecBuilder::complement(value)`. The fitter jointly
optimises an axis-aligned bounding `Rectangle` (4 trailing optimizer params
`[x, y, ln(area), ln(ratio)]`) so its area minus the (clipped) union of
shapes matches the complement target. All four shapes implement
`compute_exclusive_regions_clipped` and `compute_exclusive_regions_clipped_with_gradient`
with analytical boundary-velocity gradients on the inside-container
sub-arcs / box edges. Multi-cluster + complement is rejected by design (one
universe per diagram). `Layout::container()` exposes the fitted rectangle;
container-aware `Layout::normalize` translates shapes + container together
(rotation/mirror/pack are out of scope — the container must stay
axis-aligned). `VennDiagram::complement(value)` attaches a non-proportional
padded-bounding-box container as a visual frame.

## Label placement

Two-axis strategy (`plotting/placement.rs`):

- `InteriorPolicy::{Strict, Loose}` — only `Strict` is implemented
- `ExteriorPolicy::{None, Raycast { margin }, ForceDirected { margin, iterations }}` —
  `Raycast` and `ForceDirected` are implemented; `None` returns
  `Err(PlacementError::Unimplemented)`

Default is `Strict + Raycast` with a per-label proportional margin
(`0.5 * max(label_w, label_h)`). Both exterior strategies work against the
union polygon of the fitted shapes. Force-directed warm-starts from the
raycast positions and iterates a soft spring plus three repulsive
constraints: label–label AABB, union-polygon containment along the raycast
direction, and label–foreign-region repulsion.

Lower-level primitives: `fit_label_in_region` / `fit_labels_in_regions`
(predicate — interior anchor or omit), `largest_inscribed_rect`,
`PlotData::region_anchors` / `set_anchors` (hole-aware POI).

Resize loops: `placements_bbox` returns the union AABB so callers can extend
the canvas; native callers can use `place_labels_to_fixed_point` for the
measure-then-replace loop.

## API design

Specifications are shape-agnostic; shape is chosen when constructing the
`Fitter`:

```rust
use eunoia::{DiagramSpecBuilder, Fitter, InputType};
use eunoia::geometry::shapes::Circle;

let spec = DiagramSpecBuilder::new()
    .set("A", 5.0)
    .set("B", 2.0)
    .intersection(&["A", "B"], 1.0)
    .input_type(InputType::Exclusive)
    .build()
    .unwrap();

let layout = Fitter::<Circle>::new(&spec).seed(42).fit().unwrap();
```

`InputType::Exclusive` ↔ eulerr `"disjoint"`; `InputType::Inclusive` ↔
eulerr `"union"` (decomposed via inclusion–exclusion, rejects negative
regions with `DiagramError::InvalidValue`).

`Layout::region_error()`, `diag_error()`, `stress()`, `residuals()` expose
named quality metrics.

## Adding a new shape

1. Create `crates/eunoia/src/geometry/shapes/<name>.rs`. Implement the
   component traits (`Area`, `Centroid`, `Distance`, `Perimeter`,
   `BoundingBox`, `Closed`) and `DiagramShape`. Mirror an existing impl
   that's closest in topology — `Square`/`Rectangle` for polygonal,
   `Circle`/`Ellipse` for curved.
2. Implement `compute_exclusive_regions` exactly when possible. Prefer
   exact conic/polysegment or closed-form rectangle intersection over Monte
   Carlo (last-resort fallback only).
3. Override `to_optimizer_params` / `from_optimizer_params` /
   `optimizer_params_from_circle` only if the optimizer needs a different
   encoding from the geometric one. Defaults delegate to `to_params` /
   `from_params`.
4. Export from `geometry/shapes.rs`; add unit tests.

The fitter is shape-generic — no fitter changes needed. To get the canonical
Venn warm-start, add an arm to `crate::venn::VennDiagram` and the dispatch
in `fitter.rs::venn_warm_start_params` (TypeId-based because per-shape
encodings aren't interchangeable).

## Conventions

- Edition 2024, MSRV 1.91.1 (workspace-level, inherited via
  `edition.workspace = true` / `rust-version.workspace = true`).
- Module layout: `module.rs` + `module/`, never `module/mod.rs`.
- All public items must have rustdoc. Include examples where useful.
- All code must pass `cargo fmt` and
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- Use `Result<T, E>`; library code does not panic.
- Tests live in `#[cfg(test)]` modules in the file under test. Property
  tests use `proptest`.
- Conventional Commits (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`,
  …; `!` or `BREAKING CHANGE:` for breaks). Atomic commits.
- SemVer. Pre-1.0.0, so breaking changes are acceptable.

## Dependencies

Core (`crates/eunoia/`): `nalgebra` 0.34, `basin` 0.3 (`nalgebra` backend;
final-layout LM), `argmin` 0.11, `argmin-math` 0.5 (`nalgebra_v0_34`;
L-BFGS + MDS-init solvers), `levenberg-marquardt` 0.15 (MDS-init LM only),
`finitediff`, `polylabel-mini`, `num-complex`, `log`, `rand` 0.9,
`i_overlay` ~2.0 (optional, `plotting`), `rayon` (non-wasm only). The whole
tree is aligned on a single nalgebra 0.34 so basin's `nalgebra`-backend
types unify with eunoia's own (no faer, no second nalgebra major).

Features: `wasm` (forwards to `argmin/wasm-bindgen`), `plotting`
(`i_overlay`), `corpus` (exposes `test_utils::corpus` to example binaries —
internal, not part of the public contract).

WASM (`crates/eunoia-wasm/`): `wasm-bindgen` 0.2, `serde` 1.0,
`serde-wasm-bindgen` 0.6, `serde_json` 1.0, `console_error_panic_hook`,
`web-sys`, `getrandom` 0.3 with `wasm_js`.

Keep the dep footprint minimal — compile time, WASM binary size, license,
and maintenance burden are all considerations.

## Workflows

[Task](https://taskfile.dev) wraps the common commands. See `Taskfile.yml`:

- `task fmt` / `task fmt-check`
- `task lint` — clippy with `-D warnings`
- `task dev` — fmt + check + test + lint
- `task test-debug` / `task test-quiet` / `task test-slow` (ignored tests)
- `task coverage` / `task coverage-open`
- `task build-release`
- `task build-wasm` — wasm-pack only
- `task pack-npm` — `build-wasm` + `prepare-package.mjs` (npm publish input)

Direct cargo:

- `cargo build --workspace`, `cargo test -p eunoia`
- Slow regression / stochastic tests: `cargo test --workspace -- --ignored`
- WASM: `wasm-pack build crates/eunoia-wasm --target bundler --out-dir ../../npm`

Web dev server: `cd web && npm install --include=dev && npm run dev`.
The web app and the `ts/` build pin pnpm via `packageManager`; run
`corepack enable` once locally. Publishing is automated by
`.github/workflows/publish-npm.yml` on `v*` tags.

The `eunoia` test profile is built at `opt-level = 3` (with
`debug_assertions` on) so `cargo test -p eunoia --lib` runs in ~2–3 s
instead of ~30 s. A `profiling` profile (release + full debuginfo, no
strip) is configured for `cargo flamegraph`.

## References

- **eulerr** (R package, C++ backend) — reference for algorithms and edge
  cases
- **venneuler** — stress loss
- Classic MDS for initialisation
- Polylabel for poles of inaccessibility

## Open work

- Rotated squares / rectangles, triangles
- Tighter inscribed-rectangle solver (directional clearance instead of the
  current radial bound)
- `InteriorPolicy::Loose` and `ExteriorPolicy::None` for `place_labels`
- Leader-line entry-point refinement (use first ray–region intersection
  rather than the POI)
- R / Python / Julia bindings (separate repositories, thin wrappers)
- Global-search fallback for hard 3-ellipse cases beyond what CMA-ES
  already catches
