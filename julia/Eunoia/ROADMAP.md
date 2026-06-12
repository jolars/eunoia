# Eunoia.jl roadmap

Status: **Phases 1–3 complete; Phase 4 (release polish & split-out) is next.**
This document plans the path to a registerable, plotting-capable package on par
with the [`eunoia-py`](https://github.com/jolars/eunoia-py) sister binding, and
the eventual split into its own repo (`jolars/Eunoia.jl`).

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
- **Phase 4 — next.** Docs site, CI matrix, versioning decision, registration,
  and the split to `jolars/Eunoia.jl`; see the Phase 4 section below.

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
  environment, so the default `]test` stays light. Phase 4 wires this into CI
  (needs the env var + a CairoMakie precompile step).

**Exit criteria (met):** `using CairoMakie; eunoiaplot(euler(...))` (and the bare
`plot(euler(...))`) renders an Euler/Venn diagram with labels, quantities,
legend, and complement support.

## Phase 4 --- Release polish & split-out

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
  capi → build artifacts → bump + register Julia).

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
Phase 4  docs, CI, register, split repo                             ← next
```

Phases 1 and 2 were independent; Phase 3 needed both. Phase 4 is the remaining
release work.
