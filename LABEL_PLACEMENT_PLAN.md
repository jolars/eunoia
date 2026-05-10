# Label-placement strategy API

## Context

Phase 1 (already shipped) added a *predicate* layer for label fit-checks:

- Rust core: `fit_label_in_region(pieces, w, h, precision) -> Option<Point>` and `fit_labels_in_regions(regions, sizes, precision) -> HashMap<String, Point>` in `crates/eunoia/src/plotting/inscribed.rs`. Built on top of the existing `largest_inscribed_rect` (radial-conservative bound).
- WASM: `compute_region_label_placements(...)` (re-fits the diagram) and `fit_labels_for_polygons(polygons_json, sizes_json, precision)` (operates on already-decomposed regions, no re-fit) in `crates/eunoia-wasm/src/lib.rs`.
- TS wrapper: `placeRegionLabels({ sets, sizes, ... })` and `placeRegionLabelsForRegions({ regions, sizes, precision })` in `ts/index.ts`.
- Web demo (`web/src/lib/components/DiagramSvg.svelte`): hidden `<text data-fit-region="...">` measured via `getBBox()`, fit-check called per-region, labels gated on the result.
- pnpm dep: `web/package.json` uses `link:../npm` so `task pack-npm` rebuilds reflect immediately (no copy-cache).

The current `fit_labels_in_regions` is a *predicate* — it tells you whether/where a label fits, but for unfit regions it returns nothing. That's not useful as a default placement API: by default a user would get holes in their result.

Phase 2 is to add a real *placement* API that always returns a position for every region, with strategy as a configurable knob.

## API design (the part we want to lock in now)

Two orthogonal axes:

- **Interior policy** — what to do when the label box would fit at the region's POI:
  - `Strict`: anchor at the POI only if `fit_label_in_region` says yes.
  - `Loose`: always anchor at the POI, even when the box overflows the region polygon.
- **Exterior fallback** — what to do for regions where `Strict` says "doesn't fit":
  - `None`: omit the region from the result. (This is the current predicate behavior.)
  - `Raycast`: deterministic ray from diagram centroid through the region's POI; place anchor outside the diagram bounding box (or container, when complement is set), with a configurable margin.
  - `ForceDirected`: iterative spring/repulsion solve; polygon-aware (labels repel foreign region polygons, not just other labels). Eunoia has more leverage here than ggrepel because we know the region geometry — labels can be constrained to not cross unrelated regions.

`Loose` ignores `exterior` (interior always returns). `Strict + None` is the current predicate. `Strict + Raycast` is the default for the new placement API.

### Return type

```rust
pub struct LabelPlacement {
    /// Centre of the label box, in the same coordinate space as the regions.
    pub anchor: Point,
    /// Where the placement landed (interior / overflow / exterior).
    pub kind: PlacementKind,
    /// Inside-region point to draw a leader line to. `None` for interior
    /// placements; `Some` for exterior. Renderers use this to draw the tether
    /// from `anchor` toward `tether`.
    pub tether: Option<Point>,
}

pub enum PlacementKind {
    /// Box fits inside the region's polygon.
    Interior,
    /// Loose policy: anchor is at the region's POI but the box overflows.
    InteriorOverflow,
    /// Anchor is outside the diagram, ray-cast from centroid through POI.
    ExteriorRaycast,
    /// Anchor is outside the diagram, decided by the force-directed solver.
    ExteriorForceDirected,
}
```

`InteriorOverflow` exists so callers can tell which `Loose` placements *would* have failed `Strict` — useful for styling (e.g. dim the label) without needing a second predicate call.

### Function signature

```rust
pub fn place_labels(
    regions: &RegionPolygons,
    sizes: &HashMap<String, (f64, f64)>,
    strategy: &PlacementStrategy,
) -> HashMap<String, LabelPlacement>;

pub struct PlacementStrategy {
    pub interior: InteriorPolicy,
    pub exterior: ExteriorPolicy,
    pub precision: f64,
}

pub enum InteriorPolicy { Strict, Loose }

pub enum ExteriorPolicy {
    None,
    Raycast { margin: f64 },
    ForceDirected { /* knobs TBD */ },
}

impl Default for PlacementStrategy {
    fn default() -> Self {
        Self {
            interior: InteriorPolicy::Strict,
            exterior: ExteriorPolicy::Raycast { margin: /* some sensible default */ },
            precision: 0.01,
        }
    }
}
```

`fit_labels_in_regions` (the predicate) stays as a separate primitive — it's a useful building block and keeps the predicate vs placement distinction clean.

## v1 scope

Implement only **`Strict + Raycast`** end-to-end. Other strategy variants are present in the enum so the surface is extensible, but selecting them returns `Err(...)` (or `unimplemented!()` panic — pick one in v1; I'd lean toward `Err` so callers can detect and fall back).

Concretely v1 implements:
- `PlacementStrategy::default()` (Strict + Raycast).
- `InteriorPolicy::Strict` evaluation (calls `fit_label_in_region` per region).
- `ExteriorPolicy::Raycast` geometry — see "Raycast algorithm" below.
- The `LabelPlacement` / `PlacementKind` return types.

`InteriorPolicy::Loose`, `ExteriorPolicy::None`, and `ExteriorPolicy::ForceDirected` are deferred — the variants exist but the implementation returns an error. `Loose` is trivial to add later (it's `RegionPolygons::label_points` per region, no fit gate). `ForceDirected` is a real implementation effort (springs, repulsion, iteration loop, convergence).

## Raycast algorithm sketch

Inputs: per-region pieces, label `(w, h)`, the diagram's overall bounding box (or the fitted complement container if present), a margin.

Per-region procedure:

1. POI = `poi_with_holes(pieces, precision)` — already exists.
2. Diagram centroid C = centroid of the union of all region polygons (or just bbox centre — simpler, deterministic, fine for v1).
3. Direction `d` = `(POI - C).normalize()`. If `POI == C` (Venn-style centred regions), fall back to the principal axis from `principal_axis(piece)` — the direction with the most "room to extend outward."
4. Anchor candidate = `POI + d * t`, where `t` is chosen so the label box (of size `w × h`, centred on the candidate) is fully outside the diagram bounding box plus `margin`. Closed-form: project the box's bbox onto `d` and walk `t` until both extents clear the diagram bbox.
5. Tether = the point where the ray from anchor toward POI first enters the region's polygon (closest intersection with `pieces[*].outer`). For v1, just use the POI itself — leader-line entry-point refinement is a follow-up.

No collision avoidance between exterior labels in v1. For most diagrams (3–4 sets) this is fine. For crowded ones (Venn n=5 with all five centre regions failing the fit) labels will pile up at similar exterior angles — that's `ForceDirected`'s job.

Open algorithmic questions worth resolving during implementation:
- "Diagram bounding box" — bbox of the union of all region polygons, the overall containing rectangle from the fit, or the fitted `complement` container (when present)? Probably "container if set, else bbox of regions."
- Margin units — fixed user-coord units, or proportional to label size? `margin = max(label_w, label_h) * 0.5` is a sensible default that scales naturally.
- Direction tiebreak when POI ≈ centroid — principal axis is the obvious answer, but pick a sign convention (e.g. +y for symmetric cases).

## Implementation steps

1. **Rust core** — new module `crates/eunoia/src/plotting/placement.rs`:
   - Define `PlacementStrategy`, `InteriorPolicy`, `ExteriorPolicy`, `LabelPlacement`, `PlacementKind`.
   - Implement `place_labels(regions, sizes, strategy) -> Result<HashMap<String, LabelPlacement>, PlacementError>` for `Strict + Raycast`. Other variants return `PlacementError::Unimplemented`.
   - Re-export from `crates/eunoia/src/plotting.rs`.
   - Tests modeled on the existing `inscribed.rs` test patterns: comfortable interior fit, exterior fallback, complement region, two-circle batch.

2. **WASM** (`crates/eunoia-wasm/src/lib.rs`):
   - Add `place_region_labels(polygons_json, sizes_json, strategy_json) -> String` operating on already-decomposed polygons (mirrors `fit_labels_for_polygons`). The strategy is JSON-encoded for now; we can revisit a typed wasm-bindgen enum later.
   - Output JSON: `{ [combo: string]: { anchor: [x, y], kind: "Interior" | "InteriorOverflow" | "ExteriorRaycast" | "ExteriorForceDirected", tether?: [x, y] } }`.

3. **TS wrapper** (`ts/index.ts`):
   - Public types: `PlacementStrategy`, `InteriorPolicy`, `ExteriorPolicy`, `LabelPlacement`, `PlacementKind`.
   - Function: `placeLabelsForRegions({ regions, sizes, strategy? }) -> Record<string, LabelPlacement>`. Defaults to `Strict + Raycast`.
   - The existing `placeRegionLabelsForRegions` (predicate) stays — different semantics, different name.

4. **Web demo** (`web/src/lib/components/DiagramSvg.svelte`):
   - Switch the existing `regionLabelFits` derivation from `placeRegionLabelsForRegions` to `placeLabelsForRegions` (default strategy).
   - Render every label at its returned `anchor` (no more hide-on-fail). Exterior labels render at their exterior anchor; if `tether` is set, draw a thin polyline from `anchor` toward `tether`.
   - Keep the `[fit-check]` console.debug logging while the demo is the canonical test surface.

5. **Docs** (`AGENTS.md`):
   - Update the "Current Status" entry for label-anchor utilities to mention the new placement API + the `Strict + Raycast` default.
   - Note the deferred strategy variants under "Future Considerations" alongside the existing radial-conservative-bound follow-up.

## Open questions to confirm before implementing

- **Default margin** for `Raycast`: fixed (e.g. 5% of diagram bbox short side) or proportional to label size (`max(w, h) * 0.5`)? Proportional is easier to reason about across scales.
- **Unimplemented variants**: `Err(PlacementError::Unimplemented)` (callers can branch) or `unimplemented!()` panic (loud failure)? I'd lean `Err`.
- **`InteriorOverflow`**: keep the dedicated variant or collapse `Loose` results back into `Interior`? Keeping the variant is more honest but adds enum surface; collapsing is simpler but loses information.
- **Diagram bounding box** for raycast: union-of-regions bbox, container (when set), or both (container if set, else union)?
- **Should `place_labels` accept a single `RegionPolygons` only, or also a free `&[(String, Vec<RegionPiece>)]` slice?** The former matches the existing predicate API; the latter is more flexible for callers who don't carry a full `RegionPolygons` value.

## Out of scope for v1

- `InteriorPolicy::Loose` implementation (just plumbing for now).
- `ExteriorPolicy::None` implementation (caller can use `fit_labels_in_regions` predicate directly for that).
- `ExteriorPolicy::ForceDirected` implementation.
- Inter-label collision avoidance (force-directed handles this when implemented).
- Tighter inscribed-rectangle bound (existing follow-up on `largest_inscribed_rect`).
- Exterior leader-line entry-point refinement (use POI as tether for now).

## Verification

End-to-end:
1. `task dev` — Rust unit tests + clippy. Add tests for `place_labels` covering: interior-fits, exterior-raycast-on-overflow, complement region exterior placement, ray-cast-direction-tiebreak when POI ≈ centroid.
2. `task pack-npm` — confirm the new TS wrapper compiles and is exported.
3. `cd web && npm run dev` — open the diagram viewer:
   - Default 3-set diagram at moderate label size: every label has an interior anchor, none missing.
   - Crank label size: previously-hidden labels now appear *outside* the diagram with a leader line.
   - Switch to Venn n=4 / n=5: most centre labels render exteriorly with leader lines (and may pile up at similar angles — that's the forced-directed case for later).
4. Browser console: `[fit-check]` lines now log `kind` per region (Interior vs ExteriorRaycast) instead of just present/absent.
