# Complement / Container Support — Roadmap

In-flight feature: support for area-proportional Euler diagrams with a
known "universe" size and items outside every set (the **complement**),
visualised as a jointly-optimised **axis-aligned bounding rectangle**
(the **container**).

This file tracks multi-session progress. **Delete this file and update
`AGENTS.md` Status when the feature ships.**

## Why

Users frequently want to label items outside every named set
(e.g. "n = 16 not in any") — see eulerr#13 and the related
variance-decomposition follow-up. The eulerr workaround (declaring a
fake containing set) is awkward and treats the universe as a real set.

The principled approach: store the complement count in the spec,
fit shapes inside an axis-aligned bounding `Rectangle` whose area is
also optimised, and let the existing inclusion-exclusion loss handle
the all-zeros region (mask `0`). Shapes that spill outside the
container have their out-of-box area dropped, which naturally pulls
them back in via the per-region residuals.

## Design invariants

- The container is **always axis-aligned**. Rotated containers are
  out of scope, ever.
- The container is `crates/eunoia/src/geometry/shapes/rectangle.rs`'s
  `Rectangle`, with optimiser encoding `[x, y, ln(area), ln(ratio)]`.
  Reused, not subclassed.
- 4 container params are **appended** to the per-shape parameter
  vector (so layout-internal indexing of shape params stays
  `i * S::n_params() + p`).
- All non-container behaviour is unchanged when `spec.complement` is
  `None`.

## Sessions

### S1 — Spec, scaffolding, Circle clipping (FD gradient) — **DONE**

Goal: end-to-end working complement support for **circles only**, with
finite-difference gradient. Other shapes erroring at fitter
construction; analytical gradient deferred.

- [x] Add `complement: Option<f64>` to `DiagramSpec` and
      `DiagramSpecBuilder` (`spec.rs`, `spec/spec_builder.rs`)
- [x] Validate complement ≥ 0 in builder
- [x] Insert mask `0` into `PreprocessedSpec.exclusive_areas` when
      complement is set
- [x] Append 4 container init params after MDS
      (`fitter/final_layout.rs::optimize_layout`)
- [x] Skip Venn warm-start when complement is set
- [x] Error in `Fitter::fit()` when complement is set and
      `S` lacks `compute_exclusive_regions_clipped` (lifted in S3, S5)
- [x] `DiagramCost`: dispatches to clipped path when
      `spec.complement.is_some()`; gradient forces FD for container
      specs (`fitter/final_layout.rs`)
- [x] `optimize_from_initial`: forces L-BFGS for container specs
      (LM has no analytical Jacobian for clipped regions yet)
- [x] `compute_exclusive_regions_clipped(circles, container)` —
      arc-vs-axis-aligned-line clipping, box-edge segment
      stitching, area via Green's theorem
      (`geometry/shapes/circle.rs`)
- [x] `Layout<S>` exposes `container: Option<Rectangle>`
      (`fitter/layout.rs`)
- [x] Skip post-fit normalisation when container is present
      (`fitter.rs::fit_with_optimization` and `Layout::normalize`)
- [x] Reject multi-cluster + complement upfront
      (`fitter.rs::spec_is_multi_cluster`)
- [x] Tests: spec/builder (4), circle clipping (7),
      end-to-end (6) — 17 total
- [x] `task fmt`, `cargo test --workspace`, `task lint` all green

### S2 — Circle analytical gradient with container — **DONE**

- [x] Box-edge boundary integrals contributing to container params
      (`x`, `y`, `ln(area)`, `ln(ratio)`) via the rigid-edge
      boundary-velocity formulas in
      `area_and_gradient_from_clipped_arcs`
- [x] Clipped-arc gradient contributions to circle shape params
      via `clipped_arc_integral_with_gradient` (boundary-velocity
      identity over inside-container sub-arcs)
- [x] Complement region gradient: mask `0` seeded with
      `container_area` plus `∂(w·h)/∂u = container_area`; shape
      params get zero from the seed, container params get the full
      contribution; inclusion-exclusion handles the rest
- [x] Trait method `compute_exclusive_regions_clipped_with_gradient`
      with default `None`; `Circle` overrides
- [x] `DiagramCost::gradient` uses the analytical clipped path when
      `spec.complement.is_some()`; FD fallback retained
- [x] `LmDiagramProblem` threads the container and includes mask `0`
      in the residual list when complement is set
      (LM still gated to L-BFGS in `optimize_from_initial` because
      LM converges to higher-loss basins on the test corpus —
      revisit in S6)
- [x] FD-vs-analytical verification at the per-region level (4
      tests in `circle.rs`) and end-to-end through
      `DiagramCost::gradient` (2 tests in `final_layout.rs`)
- [x] `task fmt`, `cargo test --workspace`, `task lint` all green

### S3 — Ellipse clipping (FD gradient) — **DONE**

- [x] Lift the `S != Circle` gate to allow Ellipse
      (`fitter.rs::fit_with_optimization`)
- [x] Ellipse-vs-axis-aligned-line intersections via
      `A cos t + B sin t = C` (`A = a cos φ`, `B = −b sin φ` for vertical
      edges; `A = a sin φ`, `B = b cos φ` for horizontal) solved as
      `t = atan2(B, A) ± acos(C/√(A²+B²))`
      (`geometry/shapes/ellipse.rs::push_axis_crossings`)
- [x] `compute_exclusive_regions_clipped_ellipse` — arc clipping via
      the parametric crossings, box-edge segment intervals via the
      rotated-ellipse implicit form, area via Green's theorem
      (`geometry/shapes/ellipse.rs`)
- [x] Reuse box-edge stitching pattern from S1 (per-edge inside-mask
      interval intersected across all ellipses in the mask)
- [x] `Ellipse::compute_exclusive_regions_clipped` overrides the trait
      default; `compute_exclusive_regions_clipped_with_gradient` stays
      on the default `None` so `DiagramCost::gradient` falls back to
      central FD (analytical gradient is S4)
- [x] Tests: ellipse clipping (7 unit tests) and end-to-end fit
      (2 tests) — 9 total
- [x] `task fmt`, `cargo test --workspace`, `task lint` all green

### S4 — Ellipse analytical gradient with container

- [ ] Same as S2 but for `EllipseArc` representation
- [ ] FD verification

### S5 — Square + Rectangle clipping & gradient

- [ ] Lift the gate for Square / Rectangle
- [ ] Closed-form rect-vs-rect clipping (n-way intersection of axis-
      aligned rectangles is itself an axis-aligned rectangle, then
      intersect with container — also a rect)
- [ ] Analytical gradient via the existing rectangle log-area /
      log-ratio chain rule, extended with container-edge
      contributions

### S6 — Polish, exposure, render

- [ ] Container preserved through normalise (centring) without
      breaking existing single-cluster / multi-cluster behaviour
- [ ] Multi-cluster + complement: decide whether v2 lifts the
      rejection (e.g. via a per-cluster container) or keeps it
- [ ] `VennDiagram::complement(value)` + container in
      `into_layout`: non-proportional bounding box (shapes' bbox +
      padding) carrying the complement value as metadata. Venn is
      topological, not area-proportional, so the container is purely
      a visual frame — no optimisation involved.
- [ ] `eunoia-wasm` bindings: expose container in result types
- [ ] npm wrapper (`ts/index.ts`): expose container in `Layout`
- [ ] Web app: optional toggle in the picker
- [ ] Inclusive-input × complement deeper validation
- [ ] Update `AGENTS.md` Status section
- [ ] Delete this file

## Out of scope (not in any session)

- Rotated container.
- Non-rectangular containers (circular universe, polygon universe).
- Container with its own `contains` constraints on each shape (we
  rely entirely on the loss to encourage containment).
- Triangle / rotated-rectangle child shapes (not implemented in
  eunoia anywhere).
