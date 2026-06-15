# Eunoia.jl roadmap

Status: **Phases 1–3 complete; Phase 4 slices (a) (fitting knobs), (b) (plot
tuning), and (c) label placement (capi entry point + Julia binding) done — only
the Makie auto-placement rendering remains in slice (c), then Phase 5 (release
polish & split-out).** This document plans the path
to a registerable, plotting-capable package on par with the
[`eunoia-py`](https://github.com/jolars/eunoia-py) sister binding, and the
eventual split into its own repo (`jolars/Eunoia.jl`).

The capi work (Phase 4) deliberately comes **before** the split: while the capi
and the Julia wrapper live in one repo, changing the C ABI and the binding that
consumes it is a single atomic change. Once Julia is split out, every capi change
becomes a two-repo dance (tag capi → build artifacts → regenerate
`Artifacts.toml` → bump Julia), so it is much cheaper to get the control surface
right first.

Progress:

- **Phase 1 — done.** Typed result model (`EulerFit{S}`/`VennFit{S}` under
  `AbstractEulerFit{S}`, the four shape structs, `Point`, `Container`) in
  `src/types.jl`; `Base.show` residual table; input parity in `src/parse.jl`
  (membership-list input, inclusive→fitted reconstruction, `venn` Int/vector/
  mapping inputs).
- **Phase 2 — done.** `eunoia-capi` now emits per-region `region_error` and the
  full `plot_data` bundle (region pieces with outer+holes, region/set anchors,
  region areas, shape outlines) as `[x, y]` pairs. `EulerFit`/`VennFit` carry
  `region_error` and a `plot_data::JSON3.Object` field; the `show` table has its
  `regionError` column. 45 Julia tests + 5 capi tests green.
- **Phase 3 — done.** Makie rendering ships as a package extension
  (`ext/EunoiaMakieExt.jl`, triggered by `Makie` + `GeometryBasics`). Primary
  entry `eunoiaplot(fit)` returns a publication-ready figure (equal aspect, no
  decorations, optional legend); `eunoiaplot!(ax, fit)` composes, and an
  `EunoiaDiagram` recipe makes bare `plot(fit)`/`plot!(ax, fit)` work. Region
  fills are drawn as polygons-with-holes with perceptual OKLab color blending;
  styling kwargs (`colors`, `fills`, `edges`, `labels`, `quantities`, `legend`,
  `complement`) mirror `eunoia-py`'s `plot()`. The global `options()` system is
  deferred (per-call kwargs only). Plotting tests live in the dedicated
  `test/makie/` env and run only under `EUNOIA_TEST_MAKIE=true` so the default
  `]test` stays light. Compat: `Makie = "0.24"`, `GeometryBasics = "0.5"`,
  `CairoMakie = "0.15"`.
- **Phase 4 — in progress.** Extend the capi control surface (fitting knobs:
  loss + solver; plot tuning: `PlotOptions`; label placement/repulsion +
  leaders) and surface it in the Julia API — all additive JSON fields. Done
  before the split so capi changes stay a one-repo change. **Slice (a) (fitting
  knobs) is done**: `EulerInput` forwards optional `loss`/`loss_eps`/
  `n_restarts`/`optimizer`/`mds_solver`/`initial_sampler`/
  `cmaes_fallback_threshold`/`max_iterations`/`tolerance`/`xtol`/`ftol`/`gtol`/
  `jobs` (snake_case enum tokens validated capi-side) to the `Fitter` setters,
  surfaced as `euler` keyword args. **Slice (b) (plot tuning) is also done**:
  `EulerInput` forwards optional `n_vertices`/`label_precision`/
  `sliver_threshold` into the `PlotOptions` that backs `plot_data` (previously
  hardcoded to `PlotOptions::default()`), surfaced as `euler` keyword args; 12
  capi tests green. **Slice (c) (label placement) capi + Julia binding is done**:
  a separate `eunoia_place_labels` entry point (region polygons + caller-measured
  label sizes + a `PlacementStrategy` JSON → resolved placements with leader
  geometry), surfaced as `Eunoia.place_labels(fit, sizes; …)` returning typed
  `LabelPlacement`s; 16 capi + 76 Julia tests green. Only the Makie ext rendering
  of placed labels + leaders remains. See the Phase 4 section below.
- **Phase 5 — after.** Docs site, CI matrix, versioning decision, registration,
  and the split to `jolars/Eunoia.jl`; see the Phase 5 section below.

The Python package is the reference for "full scope." This roadmap closes the
gap to it, adapted to Julia idioms (typed structs, `Base.show`, a Makie
**extension** rather than matplotlib).

## Where we are (Phase 0 --- done)

- `eunoia-capi` cdylib speaking a JSON-in/JSON-out C ABI (`eunoia_euler`,
  `eunoia_venn`, `eunoia_version`, `eunoia_free`).
- `Eunoia.jl` dlopen's it and exposes `euler`, `venn`, `version`, returning
  **raw `JSON3.Object`s**.
- Artifact-based binary distribution ("roll-your-own JLL") via
  `gen/generate_artifacts.jl` + `.github/workflows/julia-artifacts.yml`.
- `EUNOIA_CAPI_LIB` dev override; tests build the cdylib on the fly.

## Gap analysis vs. eunoia-py

  | Capability                                          | eunoia-py       | Eunoia.jl (now)                           |
  | --------------------------------------------------- | --------------- | ----------------------------------------- |
  | Typed result model                                  | ✅ dataclasses  | ❌ raw JSON                               |
  | `euler` area-dict input                             | ✅              | ✅                                        |
  | `euler` membership-list input                       | ✅              | ❌                                        |
  | `input` inclusive/exclusive + fitted reconstruction | ✅              | ⚠️ passed through, no reconstruction      |
  | `venn` int / name-list / mapping inputs             | ✅              | ⚠️ name-list only                         |
  | Metrics: residuals, region_error                    | ✅              | ✅                                        |
  | Pretty `repr` / `show` (residual table)             | ✅              | ✅                                        |
  | Plotting + styling                                  | ✅ matplotlib   | ✅ Makie extension                        |
  | Region polygons / anchors over the FFI              | ✅ (PyO3)       | ✅ (capi emits `plot_data`)               |
  | Docs + gallery                                      | ✅ Sphinx       | ⚠️ README only                            |
  | Typed/strict checking                               | ✅ mypy/pyright | n/a                                       |

### The one structural blocker

The **C ABI is thinner than the PyO3 surface.** PyO3's `_fit_*`/`_venn` emit a
full `plot_data` bundle --- `region_pieces` (outer + holes polygons),
`region_anchors`, `region_areas`, `set_anchors`, `shape_outlines` --- plus a
per-region `region_error` map. The capi `LayoutOut` emits only fitted shapes,
scalar metrics, a per-shape `label_anchor`, and the container.

Those region polygons come from `i_overlay` clipping **inside the core** and
cannot be recomputed Julia-side. So **plotting parity requires extending
`eunoia-capi`** to emit the same `plot_data` PyO3 already does. This is the
critical-path dependency for Phase 3, addressed in Phase 2.

--------------------------------------------------------------------------------

## Phase 1 --- Typed model + API parity (no plotting)

Make the package feel like a real Julia library: typed results, idiomatic
`show`, and full input parity. No new native code needed (capi already emits
`target_areas` + `fitted_areas`; only `region_error` is missing --- defer that
to Phase 2 with the rest of the capi work, or stub `region_error` from residuals
until then).

- **Result structs** (mirror eunoia-py's `_models.py`):
  - `Point`, `Circle`, `Ellipse`, `Square`, `Rectangle`, `Container`.
  - `EulerFit{S}` carrying `shapes`, `original_values`, `fitted_values`,
    `residuals`, `region_error`, `diag_error`, `stress`, `loss`, `container`,
    and a private `plot_data` field (populated in Phase 2).
  - `VennFit{S} <: EulerFit{S}` --- topological; area metrics zeroed.
  - Parse the JSON3 envelope into these in `euler`/`venn` instead of returning
    raw objects. **This is a breaking change to the current return type** ---
    fine pre-1.0, but note it in the changelog.
- **Input parity** (port the pure logic from eunoia-py `_parse.py`):
  - `canonicalize(combo)` --- trim/drop-empty/sort/rejoin on `&`.
  - Membership-list input for `euler` (`Dict("A" => ["x","y"], ...)` → exclusive
    region counts). Reject mixed area/membership maps.
  - `venn` accepts `Int` (default names `A`, `B`, ...), a name vector, or a
    mapping (extract base set names).
  - `input="inclusive"`: reconstruct fitted values in the user's scale
    (`to_inclusive`) so `fitted_values`/`residuals` read back in inclusive form.
- **`Base.show`** --- pretty `text/plain` for `EulerFit` (the residual table:
  `original | fitted | residual | regionError`) and a one-line `VennFit`
  summary. Mirror eunoia-py's `__repr__`.
- **Tests**: extend `runtests.jl` --- typed-field assertions, membership input,
  inclusive reconstruction, `venn` input forms, `show` output.

**Exit criteria:** `euler`/`venn` return typed structs; membership + inclusive
inputs work; `show` prints a readable table; tests green against a locally built
capi.

## Phase 2 --- Extend the C ABI for plotting data

Bring the capi up to the PyO3 surface so the Julia side has everything Makie
needs. All additions are **additive JSON fields** → backward compatible.

- In `crates/eunoia-capi/src/lib.rs`, extend `LayoutOut` (or add a sibling
  `plot` object) with, mirroring PyO3's `fill_plot_data`:
  - `region_pieces`: `combo -> [{outer: [[x,y]...], holes: [[[x,y]...]]}]`
  - `region_anchors`: `combo -> [x,y]`
  - `region_areas`: `combo -> area`
  - `set_anchors`: `name -> [x,y]`
  - `shape_outlines`: `name -> [[x,y]...]`
  - `Metrics.region_error`: `combo -> f64` (per-region; PyO3 already emits
    this).
- Reuse `layout.plot_data(spec, PlotOptions::default())` exactly as PyO3 does
  --- the serialization is the only new code; the data already exists.
- Add capi `#[test]`s asserting the new fields are present and well-formed (the
  corpus guardrail in the core already covers correctness of the geometry).
- Julia side: extend the structs' `plot_data` parsing; surface `region_areas`
  etc. as needed.

**Exit criteria:** `eunoia_euler`/`eunoia_venn` JSON carries region polygons +
anchors; Julia parses them into the `EulerFit.plot_data`; capi tests green.

> Note: this is the only phase touching shared monorepo code. It benefits Julia
> only (Python reaches plot_data through PyO3 directly), but the additive design
> keeps the capi a clean, language-agnostic contract.

## Phase 3 --- Makie extension (done)

Rendering ships as a **package extension** (`ext/EunoiaMakieExt.jl`), so the core
package stays plot-free and Makie loads only when the user has it.

- `Project.toml`: `Makie` + `GeometryBasics` under `[weakdeps]` with
  `[extensions] EunoiaMakieExt = ["Makie", "GeometryBasics"]`. (GeometryBasics is
  listed explicitly so the ext can reference `Polygon`/`Point2f` directly; it is
  a transitive Makie dep so co-triggering is free. `MakieCore`/`Colors` are not
  separate weakdeps — Makie re-exports what's needed.)
- `ext/EunoiaMakieExt.jl`:
  - An `EunoiaDiagram` recipe (`@recipe` + `Makie.plot!`) plus
    `Makie.plottype(::AbstractEulerFit)` so bare `plot(fit)`/`plot!(ax, fit)`
    dispatch to it. The figure-level concerns a recipe can't own (equal aspect,
    hidden decorations, legend) live in the `eunoiaplot(fit)` wrapper, which
    returns a `Makie.FigureAxisPlot`; `eunoiaplot!(ax, fit)` composes.
  - Filled region pieces (outer + holes) as `GeometryBasics.Polygon`s, set
    outlines, set-name labels at `set_anchors`, optional per-region quantities at
    `region_anchors`, and the container box. Region fill color is a perceptual
    **OKLab** blend of the member sets (ported from `eunoia-py`).
  - Styling kwargs mirroring eunoia-py `plot()`: `colors`, `fills`, `edges`,
    `labels`, `quantities`, `legend`, `complement` — using Makie-native attribute
    names. The global `options()` defaults system is **deferred**.
- Tests: a CairoMakie-headless `@testset "Makie extension"` in `test/runtests.jl`,
  gated on `EUNOIA_TEST_MAKIE=true` and run in the dedicated `test/makie/`
  environment, so the default `]test` stays light. Phase 5 wires this into CI
  (needs the env var + a CairoMakie precompile step).

**Exit criteria (met):** `using CairoMakie; eunoiaplot(euler(...))` (and the bare
`plot(euler(...))`) renders an Euler/Venn diagram with labels, quantities,
legend, and complement support.

## Phase 4 --- Extend the C ABI control surface

The capi is intentionally minimal today: `EulerInput`/`VennInput` expose only
`shape`, `complement`, `input_type`, and `seed`. Everything else the core
supports is locked to defaults, so the Julia (and any future C) binding cannot
reach it. Like Phase 2, these are **additive JSON input fields** → backward
compatible, and they benefit Julia (Python reaches the core directly through
PyO3). Land this before release so the registered package isn't missing knobs.

Three independent slices, smallest first. Slice (a) (fitting knobs) was the
agreed starting point — it established the optional-field + capi-test +
Julia-kwarg pattern that (b) and (c) reuse.

- **(a) Fitting knobs — done.** Optional `EulerInput` fields forward to the
  existing `Fitter` builder methods: `loss` (`LossType`, 16 snake_case tokens
  incl. the six `smooth_*` surrogates) + `loss_eps`, `n_restarts`, `optimizer`
  (6 tokens), `mds_solver`, `initial_sampler`, `cmaes_fallback_threshold`,
  `max_iterations`, `tolerance`/`xtol`/`ftol`/`gtol`, `jobs`. Enum strings are
  validated capi-side by `parse_*` helpers (mirroring the `shape`/`input_type`
  match style) and resolved once into a `FitConfig` before the shape match, so a
  bad token errors regardless of shape. The Julia `euler` surfaces each as a
  keyword arg (no client-side validation — the core is the contract). Deferred:
  the cycling-`optimizer_pool`/`initial_solver_pool` array forms (single values
  only for now).
- **(b) Plot tuning — done.** Optional `EulerInput` fields `n_vertices` (200),
  `label_precision` (0.01), and `sliver_threshold` (1e-3) are resolved into a
  `PlotOptions` by `plot_options_from_input` (mirroring `fit`'s `if let Some(..)`
  shape) and threaded through `extract`'s new `plot_opts` parameter into
  `layout.plot_data(spec, …)` (previously hardcoded `PlotOptions::default()`).
  No `parse_*` validation — the knobs are numeric, forwarded as-is. `venn` keeps
  the defaults. Surfaced as `euler` keyword args; the capi test asserts
  `n_vertices` is honored (the set-outline vertex count tracks it) and the Julia
  test mirrors that across the FFI.
- **(c) Label placement / repulsion** *(larger)* — surface the core's
  `PlacementStrategy`/`ExteriorPolicy` (poles-of-inaccessibility is the default;
  `ForceDirected` adds spring/repulsion, plus leader-line `LeaderStrategy`/elbows
  and tethers).

  **Design correction:** the original "emit resolved positions in `plot_data`"
  wording is infeasible. `place_labels()` needs caller-supplied label **box
  sizes** (font metrics the core can't know at fit time), so resolved positions
  can't be baked into `plot_data`. Instead, a **separate `eunoia_place_labels`
  capi entry point** takes region polygons + measured sizes + a strategy and
  returns the resolved placements (`anchor`/`kind`/`tether`/`leader_end`/
  `leader_waypoints`), mirroring the web app's wasm `place_region_labels` and TS
  `placeLabelsForRegions`. **Done (capi + Julia binding):**
  - capi: `eunoia_place_labels` with `PlaceLabelsInput` (regions/sizes/container/
    strategy), snake_case strategy tokens validated by `parse_tether`/
    `strategy_from_input` (mirroring slice (a)'s `parse_*`/`FitConfig`), region
    pieces rebuilt via `classify_into_pieces` + `RegionPolygons::from_map`, and a
    generic `run`/`OkResponse<T>`/`to_json` so the entry point can return a
    `placements` envelope. 4 new capi `#[test]`s (interior/exterior/
    force-directed+elbow/bad-token).
  - Julia: `place_labels(fit, sizes; placement, leader, margin, iterations,
    precision, tether, leader_gap, min_gap)` forwards only-when-set, reading
    `fit.plot_data.region_pieces` + `fit.container`; returns
    `Dict{String,LabelPlacement}` (new typed struct). Exported, docstringed, and
    covered by a `@testset "label placement"`.

  **Makie wiring — done.** `eunoiaplot`/`eunoiaplot!` take a `placement` keyword
  (`false` default; `true` or a `NamedTuple`/`Dict` of `place_labels` strategy
  knobs) plus `leader_style`. The wrapper owns the axis, so it runs the loop the
  recipe can't: it builds one combined name+count box per region (mirroring the
  web's `nestedSets`/`regionTitleLines`, with set→host-region recovered by
  matching `set_anchors` to `region_anchors` rather than a new capi field),
  measures each box with `Makie.text_bb` (pixels), and iterates measure →
  convert-with-current-`finallimits`/`viewport`-scale → `place_labels` →
  grow-limits until the view extent settles (with a divergence guard for labels
  larger than the viewport). It then renders the stacked lines at the resolved
  anchors and any leader polylines (`tether → leader_waypoints… → leader_end`),
  straight or elbow. The recipe gains a `defer_labels` flag so the bare
  `plot(fit)` path keeps the raw-anchor fallback without double-drawing.
  Collision-aware placement is `eunoiaplot`-only (it needs the axis projection);
  `plot(fit)` ignores `placement`. Covered by the `collision-aware placement`
  Makie testset.

This was wiring + serialization only (Julia extension); the core already does
the work and no shared capi change was required.

**Exit criteria:** a caller can select the loss and key solver knobs, tune
`PlotOptions`, and opt into force-directed label placement (with leaders) entirely
through the capi JSON; Julia surfaces them as keyword args; capi + Julia tests
green.

> Note: like Phase 2, this is the only release-blocking work that touches shared
> monorepo code. Keep every field optional and language-neutral.

## Phase 5 --- Release polish & split-out

- **Docs**: Documenter.jl site (API + a gallery mirroring eunoia-py's). Host on
  GitHub Pages.
- **CI**: a Julia test workflow (matrix over OS + Julia LTS/stable) using
  `EUNOIA_CAPI_LIB` from a built capi; keep the existing `julia-artifacts`
  release workflow for binaries.
- **Versioning decision** (see open questions): `Project.toml` currently says
  `0.18.0`, tracking the Rust crate. Decide whether the Julia package versions
  independently (recommended) before first registration.
- **Registration**: register in the General registry (Registrator/JuliaHub). The
  artifact tarballs are fetched lazily from `julia-v*` GitHub releases, so a
  registered release must be preceded by a capi binary release + regenerated
  `Artifacts.toml`.
- **Split to `jolars/Eunoia.jl`**: move `julia/Eunoia/` out. The artifact URLs
  already point at the `jolars/eunoia` release assets, so the split package
  keeps pulling binaries from this repo's releases --- the capi stays here, the
  Julia wrapper lives in its own repo. Document this two-repo release dance (tag
  capi → build artifacts → bump + register Julia). This dance is exactly why the
  capi control surface (Phase 4) is settled first: after the split, every capi
  change pays this cost.

**Exit criteria:** registered `Eunoia.jl` installable via `add Eunoia`, docs
live, binaries fetched lazily, development continues in its own repo.

--------------------------------------------------------------------------------

## Open questions / decisions

1. **Versioning** --- independent semver for the Julia package vs. mirroring the
   Rust crate (`0.18.0` today). Recommendation: independent, start at `0.1.0`.
2. **Return-type break** --- Phase 1 changes `euler`/`venn` from `JSON3.Object`
   to typed structs. Acceptable pre-1.0; call it out in the changelog.
3. **Recipe ergonomics** --- *resolved (Phase 3):* `eunoiaplot(fit)` is the
   primary figure-builder (equal aspect, decorations off, optional legend) since
   a recipe's `plot!` can't own those; an `EunoiaDiagram` recipe + `plottype`
   keeps bare `plot(fit)`/`plot!(ax, fit)` working, and `eunoiaplot!(ax, fit)`
   composes.
4. **capi as shared contract** --- Phase 2 grows the capi for Julia's benefit
   only. Keep it additive and language-neutral so a future C/C++/other binding
   reuses it.

## Sequencing summary

```
Phase 1  typed model + input parity + show     (Julia only)         ✓ done
Phase 2  capi emits plot_data + region_error   (Rust, additive)     ✓ done
Phase 3  Makie extension (recipe + styling)    (Julia, weakdep)     ✓ done
Phase 4  capi control surface (loss/solver/    (Rust + Julia,       ← (a)(b) done,
         plot/label knobs) + Julia surfacing    additive)             (c) capi+Julia
                                                                       done; Makie
                                                                       render left
Phase 5  docs, CI, register, split repo                             ← after
```

Phases 1 and 2 were independent; Phase 3 needed both. Phase 4 grows the capi a
second time (after Phase 2) and must land **before** Phase 5 — the split makes
every later capi change a two-repo release dance, so the control surface should
be settled while it is still one atomic change.
