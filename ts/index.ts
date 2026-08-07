// High-level TypeScript wrapper around the wasm-bindgen-generated bindings.
// Compiled into the published npm package next to `eunoia_wasm.js`.
//
// Goals:
// - camelCase, idiomatic TS surface (no `Wasm*` prefixes, no snake_case).
// - String-union types for shape / output / optimizer / loss / inputType
//   instead of numeric enums.
// - No JSON-string payloads — everything is parsed before returning.
// - All wasm-bindgen handles freed inside the wrapper; consumers only see
//   plain JS objects.

import * as wasm from "./eunoia_wasm.js";

// ============================================================================
// Public types
// ============================================================================

export type ShapeType =
  | "circle"
  | "ellipse"
  | "square"
  | "rectangle"
  | "rotatedRectangle";
export type InputType = "exclusive" | "inclusive";
export type OutputMode = "shapes" | "polygons" | "regions";
export type Optimizer =
  | "cmaEsTrf"
  | "cmaEsLm"
  | "cmaEs"
  | "levenbergMarquardt"
  | "trf"
  | "lbfgs"
  | "nelderMead";
export type LossType =
  | "sumSquared"
  | "sumAbsolute"
  | "sumAbsoluteRegionError"
  | "sumSquaredRegionError"
  | "maxAbsolute"
  | "maxSquared"
  | "rootMeanSquared"
  | "stress"
  | "diagError";

export interface Point {
  x: number;
  y: number;
}

export interface Circle {
  label: string;
  x: number;
  y: number;
  radius: number;
  labelAnchor: Point;
}

export interface Ellipse {
  label: string;
  x: number;
  y: number;
  semiMajor: number;
  semiMinor: number;
  rotation: number;
  labelAnchor: Point;
}

export interface Square {
  label: string;
  x: number;
  y: number;
  side: number;
  labelAnchor: Point;
}

export interface Rectangle {
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  labelAnchor: Point;
}

export interface RotatedRectangle {
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Rotation in radians, counterclockwise from the +x axis. */
  rotation: number;
  labelAnchor: Point;
}

/**
 * Jointly-optimised container rectangle drawn around the diagram when the
 * spec carried a complement (universe outside every named set).
 *
 * For a fitted diagram (`euler({ complement: ... })`), the container is
 * area-proportional: its area minus the union of the (clipped) shapes equals
 * the complement target, up to optimiser residual. For a Venn diagram
 * (`venn({ complement: ... })`), the container is purely a visual frame
 * around the canonical layout (Venn is topological, not area-proportional).
 */
export interface Container {
  /** X coordinate of the rectangle's centre (same convention as `Rectangle`). */
  x: number;
  /** Y coordinate of the rectangle's centre. */
  y: number;
  /** Full width along x. */
  width: number;
  /** Full height along y. */
  height: number;
}

/**
 * An axis-aligned rectangle: centre plus full extents, the same convention
 * as {@link Container}. Used for glyph keep-out boxes — see
 * {@link GlyphOptions.obstacles} and {@link labelObstacles}.
 */
export interface Rect {
  /** X coordinate of the rectangle's centre. */
  x: number;
  /** Y coordinate of the rectangle's centre. */
  y: number;
  /** Full width along x. */
  width: number;
  /** Full height along y. */
  height: number;
}

export interface Polygon {
  label: string;
  vertices: Point[];
  area: number;
}

export interface RegionPiece {
  outer: Polygon;
  holes: Polygon[];
  area: number;
}

export interface Region {
  /**
   * The set combination this region belongs to, in canonical form (e.g.
   * `"A"`, `"A&B"`, `"A&B&C"`).
   *
   * **Complement (universe minus all sets)** — when the spec was built
   * with `complement: ...`, an extra region with `combination === ""`
   * (the empty string) is included alongside the named regions. Its
   * `pieces` are the container minus the union of fitted shapes, and
   * its `labelAnchor` is the hole-aware POI inside that complement
   * piece. Filter this entry out if you only want named regions.
   */
  combination: string;
  totalArea: number;
  pieces: RegionPiece[];
  labelAnchor: Point;
}

export interface Metrics {
  loss: number;
  stress: number;
  diagError: number;
  iterations: number;
  targetAreas: Record<string, number>;
  fittedAreas: Record<string, number>;
  regionError: Record<string, number>;
  residuals: Record<string, number>;
}

export type Layout = (
  | { mode: "shapes"; shape: "circle"; circles: Circle[]; metrics: Metrics }
  | { mode: "shapes"; shape: "ellipse"; ellipses: Ellipse[]; metrics: Metrics }
  | { mode: "shapes"; shape: "square"; squares: Square[]; metrics: Metrics }
  | {
      mode: "shapes";
      shape: "rectangle";
      rectangles: Rectangle[];
      metrics: Metrics;
    }
  | {
      mode: "shapes";
      shape: "rotatedRectangle";
      rotatedRectangles: RotatedRectangle[];
      metrics: Metrics;
    }
  | {
      mode: "polygons";
      shape: "circle";
      polygons: Polygon[];
      circles: Circle[];
      metrics: Metrics;
    }
  | {
      mode: "polygons";
      shape: "ellipse";
      polygons: Polygon[];
      ellipses: Ellipse[];
      metrics: Metrics;
    }
  | {
      mode: "polygons";
      shape: "square";
      polygons: Polygon[];
      squares: Square[];
      metrics: Metrics;
    }
  | {
      mode: "polygons";
      shape: "rectangle";
      polygons: Polygon[];
      rectangles: Rectangle[];
      metrics: Metrics;
    }
  | {
      mode: "polygons";
      shape: "rotatedRectangle";
      polygons: Polygon[];
      rotatedRectangles: RotatedRectangle[];
      metrics: Metrics;
    }
  | {
      mode: "regions";
      shape: ShapeType;
      regions: Region[];
      setAnchors: Record<string, Point>;
      /**
       * For each set whose label was anchored to a region, the canonical
       * combination string of that region (a key of a `"regions"` entry):
       * an exclusive-area set maps to its own region (`A` → `"A"`), a
       * fully-nested set to its largest containing region (`B` → `"A&B"`).
       * Sets that fell back to the bare-shape POI are omitted. Mirrors the
       * core `PlotData::set_anchor_regions`; use it to pair a set label with
       * its region without matching `setAnchors` against region anchors by
       * coordinate.
       */
      setAnchorRegions: Record<string, string>;
      metrics: Metrics;
    }
) & {
  /**
   * Container rectangle (universe / "outside every named set" frame), present
   * only when the input spec carried a complement.
   */
  container?: Container;
};

export interface EulerOptions {
  /** Set sizes keyed by combination expression (e.g. `{ A: 5, "A&B": 1 }`). */
  sets: Record<string, number>;
  /** Whether `sets` values are exclusive subset sizes or full set unions. Default `"exclusive"`. */
  inputType?: InputType;
  /** Shape primitive to fit. Default `"circle"`. */
  shape?: ShapeType;
  /** What to return: shape parameters, polygon outlines, or exclusive regions. Default `"shapes"`. */
  output?: OutputMode;
  /** RNG seed for reproducible layouts. Accepts `number` or `bigint`. */
  seed?: number | bigint;
  /** Final-stage optimizer. When omitted, the core default (`"cmaEsTrf"`) is used. */
  optimizer?: Optimizer;
  /** Loss function. Defaults to the optimizer's preferred loss. */
  loss?: LossType;
  /** Optimizer convergence tolerance. */
  tolerance?: number;
  /**
   * Number of random restarts of the two-phase fit; the lowest-loss attempt is
   * kept. When omitted, the core default (10, mirroring eulerr) is used.
   * Lowering it trades fit robustness for speed — useful for interactive
   * previews where a fixed `seed` already pins the layout. Clamped to `>= 1`.
   */
  restarts?: number;
  /** Number of vertices per polygon outline (used when `output` is `"polygons"` or `"regions"`). Default 256. */
  polygonVertices?: number;
  /**
   * Items outside every named set (the universe complement). When set, the
   * fitter jointly optimises a bounding rectangular container; the returned
   * `Layout` exposes it via `layout.container`. Multi-cluster specs are
   * rejected with a complement.
   */
  complement?: number;
}

export interface LabelSize {
  /** Label width in diagram coordinates (same units as fitted shapes). */
  w: number;
  /** Label height in diagram coordinates. */
  h: number;
}

/**
 * Minimal region shape accepted by [`placeLabelsForRegions`]. Both the
 * wrapper's [`Region`] type and the web app's `RegionPolygon` satisfy
 * this — only `combination` and `pieces` (with outer + holes vertex
 * lists) are read.
 */
export interface RegionInput {
  combination: string;
  pieces: ReadonlyArray<{
    outer: { vertices: ReadonlyArray<Point> };
    holes: ReadonlyArray<{ vertices: ReadonlyArray<Point> }>;
  }>;
}

/**
 * Exterior fallback solver used when a label doesn't fit inside its
 * region.
 *
 * - `"raycast"` — deterministic ray from the diagram centroid through the
 *   region's POI; anchor lands outside the diagram bbox (or container, when
 *   complement is set), padded by `margin`.
 * - `"forceDirected"` — iterative spring + repulsion solve. Initial
 *   positions come from the raycast, then each label is pulled toward
 *   that "home" by a soft spring while being repelled from other labels
 *   *and* from foreign region polygons (the eunoia-specific bit: ggrepel
 *   can only see labels). Use this when raycast labels visually overlap
 *   unrelated regions or pile up at similar angles.
 * - `"matched"` — boundary-labeling placement. Each label sits on a ring
 *   that hugs the diagram silhouette, at its own outward angle, with the
 *   angles spread proportionally to each label's width so they don't overlap;
 *   a final 2-opt pass uncrosses any crossing leaders. The leaders are
 *   provably non-crossing and the labels stay close to the diagram. Best for
 *   dense diagrams where raycast leaders cross each other or pile up.
 */
export type ExteriorPolicyName = "raycast" | "forceDirected" | "matched";

/**
 * Where the exterior-leader tether attaches on the source region.
 *
 * * `"poi"` (default) — tether is the region's pole of inaccessibility
 *   (deep inside the region). Safe for any rendering style, including
 *   stroke-less fills, because the tether sits well inside the visible
 *   colored area.
 * * `"boundary"` — tether is the point where the `(poi → anchor)` ray
 *   exits the region's outer polygon ring, so the rendered leader
 *   starts on the polygon edge (standard labeling convention). Opt in
 *   when your renderer draws shape strokes.
 */
export type TetherSource = "poi" | "boundary";

/**
 * Leader strategy: the *edge type* drawn between a region and its exterior
 * label, coupled with the *placement algorithm* that suits that edge type.
 * Picking the edge type picks an appropriate placement algorithm —
 * raycasting, for instance, is a straight-line construction, so it's only
 * offered for straight leaders.
 *
 * Two edge types: `"straight"` (raycast / force-directed placement) and
 * d3-pie style `"elbow"` (orthogonal leaders with a column-based placement
 * algorithm of their own).
 */
export type LeaderStrategy =
  | {
      /** Straight leader lines (a single `tether → leaderEnd` segment). */
      type: "straight";
      /**
       * Placement algorithm for straight leaders. Default `"raycast"`.
       */
      placement?: ExteriorPolicyName;
      /**
       * Margin around the diagram bbox/container, applied to `"raycast"`,
       * `"forceDirected"`, and `"matched"` placement. Omit to use a per-region
       * proportional default of `0.5 * max(label_w, label_h)`.
       */
      margin?: number;
      /**
       * Iteration cap for the `"forceDirected"` placement. Ignored otherwise.
       * Defaults to 200; raise for crowded diagrams that haven't converged.
       */
      iterations?: number;
    }
  | {
      /**
       * d3-pie style orthogonal (elbow) leaders. Exterior labels are sorted
       * into a left/right column, stacked vertically without overlap, and
       * reached by an orthogonal three-segment polyline. The bend joints are
       * returned in `leaderWaypoints`.
       */
      type: "elbow";
      /**
       * Horizontal gap between the diagram edge and the column's vertical
       * bend rail. Omit for a per-column proportional default of
       * `0.5 * max(label_w, label_h)`.
       */
      margin?: number;
      /**
       * Minimum vertical centre-to-centre spacing between stacked labels in a
       * column. Omit for a default of `1.5 *` the taller neighbour's height.
       */
      minGap?: number;
    };

export interface PlacementStrategy {
  /**
   * Leader strategy — the edge type and the placement algorithm for
   * exterior labels. Omit for the default: straight leaders placed by
   * raycasting.
   */
  leader?: LeaderStrategy;
  /** Polylabel-style search precision. Default `0.01`. */
  precision?: number;
  /**
   * Where the leader tether attaches on the source region for exterior
   * placements. Default `"poi"`.
   */
  tether?: TetherSource;
  /**
   * Visible gap (in the same coordinate units as the label sizes) between
   * the leader-line tip (`placement.leaderEnd`) and the label's bounding
   * box. The placer inflates the box used to compute `leaderEnd` by this
   * amount on every side, so the leader stops `leaderGap` units short of
   * the rendered text edge. Negative values are clamped to `0`. Default
   * `0` (leader ends exactly at the box edge).
   *
   * Set this when your renderer hands raw measured text bboxes to the
   * placer and you want breathing room between the leader tip and the
   * glyphs. If you instead pre-pad the sizes you pass in, keep
   * `leaderGap = 0` — the padding shows up as the visible gap.
   */
  leaderGap?: number;
}

/**
 * Discriminator on [`LabelPlacement`] — tells the renderer whether the
 * anchor is inside or outside the region.
 */
export type PlacementKind =
  | "interior"
  | "exteriorRaycast"
  | "exteriorForceDirected"
  | "exteriorElbow"
  | "exteriorMatched";

export interface LabelPlacement {
  /** Centre of the label box, in the same coordinates as the regions. */
  anchor: Point;
  /** Where the placement landed. */
  kind: PlacementKind;
  /**
   * Inside-region point to draw a leader line to. `undefined` for interior
   * placements; set for exterior placements. Renderers draw the tether
   * from `anchor` toward `tether`.
   */
  tether?: Point;
  /**
   * Point on the label's bounding box where the leader line should
   * terminate. `undefined` for interior placements; set for exterior. Use
   * this as the leader's terminus instead of `anchor` so the line stops at
   * the box edge rather than continuing through the rendered text.
   */
  leaderEnd?: Point;
  /**
   * Intermediate vertices of the leader polyline, in draw order, running
   * between `tether` and `leaderEnd`. Empty for interior placements (no
   * leader) and for straight leaders, where the leader is the single segment
   * `tether → leaderEnd`. Future edge types (e.g. elbow/orthogonal leaders)
   * populate this with their bend joints, so a renderer always draws the
   * leader as the polyline `tether → leaderWaypoints… → leaderEnd`.
   */
  leaderWaypoints: Point[];
}

export interface PlaceLabelsForRegionsOptions {
  /**
   * Already-decomposed regions in any coordinate space — the placement is
   * scale-invariant as long as `sizes` are in the same units as the
   * polygon vertices.
   */
  regions: ReadonlyArray<RegionInput>;
  /**
   * The complement container (universe rectangle), when the spec was built
   * with a complement. Pass [`Layout.container`] (from `euler({ complement })`
   * / `venn({ complement })`) here so exterior raycasts land outside the
   * frame rather than just outside the cluster.
   */
  container?: Container;
  /**
   * Label dimensions per region, keyed by canonical combination
   * (e.g. `"A"`, `"A&B"`, `""` for the complement region).
   */
  sizes: Record<string, LabelSize>;
  /** Strategy knobs. Defaults to `Strict + Raycast`. */
  strategy?: PlacementStrategy;
}

/**
 * How glyph centers are arranged within a region.
 *
 * - `"uniform"` (default) — centers sit on a hexagonal lattice anchored at
 *   the region's pole of inaccessibility, with the spacing widened so the
 *   glyphs spread across the region. Fully deterministic.
 * - `"random"` — seeded dart throwing with a minimum center-to-center
 *   spacing, giving the scattered look of the original eulerGlyphs tool.
 *   Deterministic for a fixed `seed`.
 */
export type GlyphArrangement = "uniform" | "random";

/** Tuning knobs for [`placeGlyphsForRegions`]. All optional. */
export interface GlyphOptions {
  /** Arrangement of glyph centers. Default `"uniform"`. */
  arrangement?: GlyphArrangement;
  /**
   * Glyph radius, in the same units as the region polygons. Omit for the
   * auto mode: the largest radius at which every region holds its full
   * count, found by bisection. With an explicit radius, overflowing regions
   * place as many glyphs as fit and report the shortfall in
   * [`GlyphPlacements.unplaced`].
   */
  radius?: number;
  /**
   * Extra breathing room around glyphs, as a fraction of the radius. It
   * applies both between glyphs (the minimum center-to-center distance is
   * `2r * (1 + gap)`) and against the region boundary (centers keep a
   * clearance of `r * (1 + gap)` to every ring, so glyphs never sit tangent
   * to a region edge). Default `0.25`.
   */
  gap?: number;
  /** Seed for the `"random"` arrangement. Default `0`. */
  seed?: number;
  /** Pole-of-inaccessibility search precision. Default `0.01`. */
  precision?: number;
  /**
   * `"random"` only: dart throws attempted per glyph before the region is
   * declared full. Default `300`.
   */
  maxAttempts?: number;
  /**
   * Diagram-wide keep-out boxes that glyph centers clear by
   * `r * (1 + gap)`, in the same coordinates as the region polygons.
   * Labels are painted over glyphs, so these are usually the measured label
   * boxes — see {@link labelObstacles}, which builds them from a
   * [`placeLabelsForRegions`] result.
   *
   * Clearance is a strong preference, not a guarantee: the auto radius
   * shrinks to honour the boxes, but not below half the radius it would
   * have chosen without them, so one cramped region cannot shrink every
   * glyph in the diagram. Degenerate boxes are ignored. Default: none.
   */
  obstacles?: ReadonlyArray<Rect>;
}

/** Result of [`placeGlyphsForRegions`]. */
export interface GlyphPlacements {
  /**
   * The radius actually used — auto-chosen, or the caller's `radius` echoed
   * back. `0` when no region could hold any glyph (degenerate input).
   */
  radius: number;
  /**
   * Glyph center points per region, keyed by canonical combination (e.g.
   * `"A"`, `"A&B"`, `""` for the complement region). Regions absent from
   * `counts`, or with a count of zero, are omitted.
   */
  positions: Record<string, Point[]>;
  /**
   * Per-region count that did **not** fit at the used radius. Only present
   * when a fixed radius overflowed a region (or the input was degenerate).
   */
  unplaced?: Record<string, number>;
}

export interface PlaceGlyphsForRegionsOptions {
  /**
   * Already-decomposed regions, same shape as
   * [`PlaceLabelsForRegionsOptions.regions`].
   */
  regions: ReadonlyArray<RegionInput>;
  /**
   * Glyph count per region, keyed by canonical combination (`""` for the
   * complement region). One glyph is placed per unit.
   */
  counts: Record<string, number>;
  /** Tuning knobs. Defaults to a uniform arrangement with auto radius. */
  options?: GlyphOptions;
}

/** Tuning knobs for {@link placeGlyphBoxesForRegions}. All optional. */
export interface GlyphBoxOptions {
  /**
   * Arrangement of boxes. Default `"uniform"`, which here means row/shelf
   * packing: rows of one shared height, boxes left to right at their own
   * widths, the block centered on the region's pole of inaccessibility.
   */
  arrangement?: GlyphArrangement;
  /**
   * Diagram-wide factor applied to every measured box. Omit for the auto
   * mode: the largest factor in `[minScale, 1]` at which every region holds
   * all of its items. Pass one to take control — values above `1` are then
   * honoured — in which case overflowing regions report the shortfall in
   * [`GlyphBoxPlacements.unplaced`].
   */
  scale?: number;
  /**
   * Lower end of the auto-scale bracket: the readability floor below which
   * shrinking the text is worse than dropping items. Clamped to `(0, 1]`,
   * ignored when `scale` is set. Default `0.35`.
   */
  minScale?: number;
  /**
   * Breathing room around every box, as a fraction of the **row height**
   * (the diagram-wide maximum scaled box height — this packer's analogue of
   * the disc packer's radius). Each box carries a halo of
   * `0.5 * gap * rowHeight`, so adjacent boxes end up `gap * rowHeight`
   * apart and every box keeps half that to region boundaries, holes, and
   * obstacles. The default `0.25` therefore means the same thing as in
   * {@link GlyphOptions.gap}: here, "a quarter of a line-height apart, half
   * that from the edge".
   */
  gap?: number;
  /** Seed for the `"random"` arrangement. Default `0`. */
  seed?: number;
  /** Pole-of-inaccessibility search precision. Default `0.01`. */
  precision?: number;
  /**
   * `"random"` only: dart throws attempted per box before the piece is
   * declared full. Default `300`.
   */
  maxAttempts?: number;
  /**
   * Diagram-wide keep-out boxes, with the same semantics as
   * {@link GlyphOptions.obstacles} — best-effort, with the scale allowed to
   * shrink only to half of what it would have been without them. Usually
   * the measured label boxes, via {@link labelObstacles}. Default: none.
   */
  obstacles?: ReadonlyArray<Rect>;
}

/** Result of {@link placeGlyphBoxesForRegions}. */
export interface GlyphBoxPlacements {
  /**
   * The factor actually used — auto-chosen, or the caller's `scale` echoed
   * back. `0` when nothing could be placed (degenerate input). Multiply
   * your reference font size by this to render the text at the size the
   * boxes were packed at.
   */
  scale: number;
  /**
   * Placed boxes per region, keyed by canonical combination (e.g. `"A"`,
   * `"A&B"`, `""` for the complement region).
   *
   * **Index-aligned with the input**: `boxes[key]` is a *prefix* of
   * `sizes[key]`, so `boxes[key][i]` belongs to `sizes[key][i]` — and hence
   * to your `labels[key][i]`. Each rect is that item's own
   * `scale * (w, h)`, centered; the halo is not included.
   */
  boxes: Record<string, Rect[]>;
  /**
   * Per-region item count that did **not** fit. Only present when a region
   * overflowed. Text boxes are much wider than tall, so this is far more
   * common than the disc packer's equivalent — see
   * {@link placeGlyphBoxesForRegions} for what to do about it.
   */
  unplaced?: Record<string, number>;
}

export interface PlaceGlyphBoxesForRegionsOptions {
  /**
   * Already-decomposed regions, same shape as
   * [`PlaceLabelsForRegionsOptions.regions`].
   */
  regions: ReadonlyArray<RegionInput>;
  /**
   * Measured box per item, keyed by canonical combination (`""` for the
   * complement region), **in the order you want the items placed** — the
   * packer returns a prefix. Measure at a reference font size and scale the
   * result by the returned `scale`.
   */
  sizes: Record<string, ReadonlyArray<LabelSize>>;
  /** Tuning knobs. Defaults to row packing with auto scale. */
  options?: GlyphBoxOptions;
}

/**
 * Options for a canonical Venn diagram. Unlike {@link EulerOptions}, there is no
 * `seed`/`optimizer`/`loss`/`tolerance`: the layout is fixed and topological
 * (not area-proportional), so no fitting runs.
 */
export interface VennOptions {
  /**
   * Number of sets in the Venn diagram. The valid range depends on `shape`:
   * `"ellipse"` supports `1 ≤ n ≤ 5`, while `"circle"`, `"square"`,
   * `"rectangle"`, and `"rotatedRectangle"` support `1 ≤ n ≤ 3` (equal-sized
   * axis-aligned/round shapes cannot open all `2ⁿ − 1` regions beyond three
   * sets).
   */
  n: number;
  /**
   * Shape primitive for the canonical layout. Default `"ellipse"` — the only
   * shape that covers all of `n ∈ 1..=5`. `"circle"` gives the classic one-,
   * two-, and three-circle diagrams; `"square"` / `"rectangle"` /
   * `"rotatedRectangle"` are box-based (the rotated variant uses φ = 0
   * footprints). All non-ellipse shapes cap at `n = 3`.
   */
  shape?: ShapeType;
  /** Output mode: polygon outlines per set, or exclusive regions. Default `"polygons"`. */
  output?: "polygons" | "regions";
  /** Number of vertices per polygon outline. Default 256. */
  polygonVertices?: number;
  /**
   * Items outside every named set (the universe complement). Venn is
   * topological, not area-proportional, so the resulting `Layout.container`
   * is a non-proportional visual frame around the canonical layout.
   */
  complement?: number;
}

// ============================================================================
// Internal mappings
// ============================================================================

const OPTIMIZER_MAP: Record<Optimizer, wasm.WasmOptimizer> = {
  cmaEsTrf: wasm.WasmOptimizer.CmaEsTrf,
  cmaEsLm: wasm.WasmOptimizer.CmaEsLm,
  cmaEs: wasm.WasmOptimizer.CmaEs,
  levenbergMarquardt: wasm.WasmOptimizer.LevenbergMarquardt,
  trf: wasm.WasmOptimizer.Trf,
  lbfgs: wasm.WasmOptimizer.Lbfgs,
  nelderMead: wasm.WasmOptimizer.NelderMead,
};

const LOSS_MAP: Record<LossType, wasm.WasmLossType> = {
  sumSquared: wasm.WasmLossType.SumSquared,
  sumAbsolute: wasm.WasmLossType.SumAbsolute,
  sumAbsoluteRegionError: wasm.WasmLossType.SumAbsoluteRegionError,
  sumSquaredRegionError: wasm.WasmLossType.SumSquaredRegionError,
  maxAbsolute: wasm.WasmLossType.MaxAbsolute,
  maxSquared: wasm.WasmLossType.MaxSquared,
  rootMeanSquared: wasm.WasmLossType.RootMeanSquared,
  stress: wasm.WasmLossType.Stress,
  diagError: wasm.WasmLossType.DiagError,
};

interface Disposable {
  free(): void;
}

function freeAll(items: Disposable[] | undefined): void {
  if (!items) return;
  for (const item of items) {
    try {
      item.free();
    } catch {
      // Already freed or never owned — ignore.
    }
  }
}

function toSeed(seed: number | bigint | undefined): bigint | undefined {
  if (seed === undefined) return undefined;
  return typeof seed === "bigint" ? seed : BigInt(seed);
}

function parseRecord(s: string | undefined | null): Record<string, number> {
  if (!s) return {};
  try {
    return JSON.parse(s) as Record<string, number>;
  } catch {
    return {};
  }
}

function parseAnchors(s: string | undefined | null): Record<string, Point> {
  if (!s) return {};
  try {
    const raw = JSON.parse(s) as Record<string, [number, number]>;
    const out: Record<string, Point> = {};
    for (const [k, v] of Object.entries(raw)) out[k] = { x: v[0], y: v[1] };
    return out;
  } catch {
    return {};
  }
}

function parseStringRecord(
  s: string | undefined | null,
): Record<string, string> {
  if (!s) return {};
  try {
    return JSON.parse(s) as Record<string, string>;
  } catch {
    return {};
  }
}

function buildSpecs(sets: Record<string, number>): wasm.DiagramSpec[] {
  const specs: wasm.DiagramSpec[] = [];
  for (const [input, size] of Object.entries(sets)) {
    if (!Number.isFinite(size) || size <= 0) continue;
    specs.push(new wasm.DiagramSpec(input, size));
  }
  return specs;
}

// ----- wasm → plain JS converters --------------------------------------------

function pointFrom(p: wasm.WasmPoint): Point {
  return { x: p.x, y: p.y };
}

function circleFrom(c: wasm.WasmCircle): Circle {
  return {
    label: c.label,
    x: c.x,
    y: c.y,
    radius: c.radius,
    labelAnchor: { x: c.label_x, y: c.label_y },
  };
}

function ellipseFrom(e: wasm.WasmEllipse): Ellipse {
  return {
    label: e.label,
    x: e.x,
    y: e.y,
    semiMajor: e.semi_major,
    semiMinor: e.semi_minor,
    rotation: e.rotation,
    labelAnchor: { x: e.label_x, y: e.label_y },
  };
}

function squareFrom(s: wasm.WasmSquare): Square {
  return {
    label: s.label,
    x: s.x,
    y: s.y,
    side: s.side,
    labelAnchor: { x: s.label_x, y: s.label_y },
  };
}

function rectangleFrom(r: wasm.WasmRectangle): Rectangle {
  return {
    label: r.label,
    x: r.x,
    y: r.y,
    width: r.width,
    height: r.height,
    labelAnchor: { x: r.label_x, y: r.label_y },
  };
}

function rotatedRectangleFrom(r: wasm.WasmRotatedRectangle): RotatedRectangle {
  return {
    label: r.label,
    x: r.x,
    y: r.y,
    width: r.width,
    height: r.height,
    rotation: r.rotation,
    labelAnchor: { x: r.label_x, y: r.label_y },
  };
}

function containerFrom(
  r: wasm.WasmRectangle | undefined,
): Container | undefined {
  if (!r) return undefined;
  const c: Container = { x: r.x, y: r.y, width: r.width, height: r.height };
  r.free();
  return c;
}

function polygonFrom(p: wasm.WasmPolygon): Polygon {
  const verts = p.vertices;
  const out: Point[] = verts.map(pointFrom);
  const area = p.area;
  freeAll(verts);
  return { label: p.label, vertices: out, area };
}

function regionPieceFrom(piece: wasm.WasmRegionPiece): RegionPiece {
  const outer = polygonFrom(piece.outer);
  const holesArr = piece.holes;
  const holes = holesArr.map(polygonFrom);
  const area = piece.area;
  freeAll(holesArr);
  return { outer, holes, area };
}

function regionFrom(r: wasm.WasmRegion): Region {
  const piecesArr = r.pieces;
  const pieces = piecesArr.map(regionPieceFrom);
  const result = {
    combination: r.combination,
    totalArea: r.total_area,
    pieces,
    labelAnchor: { x: r.label_x, y: r.label_y },
  };
  freeAll(piecesArr);
  return result;
}

function metricsFromPolygonResult(
  r: wasm.PolygonResult | wasm.WasmRegionPolygons,
): Metrics {
  return {
    loss: r.loss,
    stress: r.stress,
    diagError: r.diag_error,
    iterations: r.iterations,
    targetAreas: parseRecord(r.target_areas_json),
    fittedAreas: parseRecord(r.fitted_areas_json),
    regionError: parseRecord(r.region_error_json),
    residuals: parseRecord(r.residuals_json),
  };
}

// ============================================================================
// Public entry points
// ============================================================================

/**
 * Fit an Euler diagram from a set-size specification.
 *
 * The fit always runs the full polygon-mode backend so metrics
 * (`loss`, `stress`, `diagError`, `iterations`, …) are populated regardless
 * of `output`. The returned `Layout` is discriminated on `mode`.
 */
export function euler(options: EulerOptions): Layout {
  const {
    sets,
    inputType = "exclusive",
    shape = "circle",
    output = "shapes",
    seed,
    optimizer,
    loss,
    tolerance,
    restarts,
    polygonVertices = 256,
    complement,
  } = options;

  if (!sets || typeof sets !== "object") {
    throw new TypeError("euler: `sets` must be an object of name → size");
  }

  const specs = buildSpecs(sets);
  if (specs.length === 0) {
    throw new Error(
      "euler: `sets` must contain at least one entry with size > 0",
    );
  }

  const seedArg = toSeed(seed);
  const optimizerArg =
    optimizer !== undefined ? OPTIMIZER_MAP[optimizer] : undefined;
  if (optimizer !== undefined && optimizerArg === undefined) {
    throw new RangeError(`euler: unknown optimizer "${optimizer}"`);
  }
  const lossArg = loss !== undefined ? LOSS_MAP[loss] : undefined;
  if (loss !== undefined && lossArg === undefined) {
    throw new RangeError(`euler: unknown loss "${loss}"`);
  }
  const tolArg = tolerance && tolerance > 0 ? tolerance : undefined;
  const restartsArg =
    restarts && restarts > 0 ? Math.max(1, Math.floor(restarts)) : undefined;
  const nVerts = Math.max(3, Math.floor(polygonVertices));

  try {
    if (output === "regions") {
      const fn =
        shape === "circle"
          ? wasm.generate_region_polygons_circles
          : shape === "square"
            ? wasm.generate_region_polygons_squares
            : shape === "rectangle"
              ? wasm.generate_region_polygons_rectangles
              : shape === "rotatedRectangle"
                ? wasm.generate_region_polygons_rotated_rectangles
                : wasm.generate_region_polygons_ellipses;
      const result = fn(
        specs,
        inputType,
        nVerts,
        seedArg,
        optimizerArg,
        lossArg,
        tolArg,
        restartsArg,
        complement,
      );
      try {
        const regionsArr = result.regions;
        const regions = regionsArr.map(regionFrom);
        freeAll(regionsArr);
        const container = containerFrom(result.container);
        return {
          mode: "regions",
          shape,
          regions,
          setAnchors: parseAnchors(result.set_anchors_json),
          setAnchorRegions: parseStringRecord(result.set_anchor_regions_json),
          metrics: metricsFromPolygonResult(result),
          ...(container ? { container } : {}),
        };
      } finally {
        result.free();
      }
    }

    // "shapes" and "polygons" both use the *_as_polygons backend.
    const fn =
      shape === "circle"
        ? wasm.generate_circles_as_polygons
        : shape === "square"
          ? wasm.generate_squares_as_polygons
          : shape === "rectangle"
            ? wasm.generate_rectangles_as_polygons
            : shape === "rotatedRectangle"
              ? wasm.generate_rotated_rectangles_as_polygons
              : wasm.generate_ellipses_as_polygons;
    const result = fn(
      specs,
      inputType,
      nVerts,
      seedArg,
      optimizerArg,
      lossArg,
      tolArg,
      restartsArg,
      complement,
    );
    try {
      const metrics = metricsFromPolygonResult(result);
      const container = containerFrom(result.container);
      const containerField = container ? { container } : {};

      if (output === "polygons") {
        const polysArr = result.polygons;
        const polygons = polysArr.map(polygonFrom);
        freeAll(polysArr);
        if (shape === "circle") {
          const arr = result.circles;
          const circles = arr.map(circleFrom);
          freeAll(arr);
          freeAll(result.ellipses);
          freeAll(result.squares);
          freeAll(result.rectangles);
          freeAll(result.rotated_rectangles);
          return {
            mode: "polygons",
            shape: "circle",
            polygons,
            circles,
            metrics,
            ...containerField,
          };
        }
        if (shape === "ellipse") {
          const arr = result.ellipses;
          const ellipses = arr.map(ellipseFrom);
          freeAll(arr);
          freeAll(result.circles);
          freeAll(result.squares);
          freeAll(result.rectangles);
          freeAll(result.rotated_rectangles);
          return {
            mode: "polygons",
            shape: "ellipse",
            polygons,
            ellipses,
            metrics,
            ...containerField,
          };
        }
        if (shape === "rectangle") {
          const arr = result.rectangles;
          const rectangles = arr.map(rectangleFrom);
          freeAll(arr);
          freeAll(result.circles);
          freeAll(result.ellipses);
          freeAll(result.squares);
          freeAll(result.rotated_rectangles);
          return {
            mode: "polygons",
            shape: "rectangle",
            polygons,
            rectangles,
            metrics,
            ...containerField,
          };
        }
        if (shape === "rotatedRectangle") {
          const arr = result.rotated_rectangles;
          const rotatedRectangles = arr.map(rotatedRectangleFrom);
          freeAll(arr);
          freeAll(result.circles);
          freeAll(result.ellipses);
          freeAll(result.squares);
          freeAll(result.rectangles);
          return {
            mode: "polygons",
            shape: "rotatedRectangle",
            polygons,
            rotatedRectangles,
            metrics,
            ...containerField,
          };
        }
        const arr = result.squares;
        const squares = arr.map(squareFrom);
        freeAll(arr);
        freeAll(result.circles);
        freeAll(result.ellipses);
        freeAll(result.rectangles);
        freeAll(result.rotated_rectangles);
        return {
          mode: "polygons",
          shape: "square",
          polygons,
          squares,
          metrics,
          ...containerField,
        };
      }

      // "shapes"
      if (shape === "circle") {
        const arr = result.circles;
        const circles = arr.map(circleFrom);
        freeAll(arr);
        freeAll(result.ellipses);
        freeAll(result.squares);
        freeAll(result.rectangles);
        freeAll(result.rotated_rectangles);
        freeAll(result.polygons);
        return {
          mode: "shapes",
          shape: "circle",
          circles,
          metrics,
          ...containerField,
        };
      }
      if (shape === "ellipse") {
        const arr = result.ellipses;
        const ellipses = arr.map(ellipseFrom);
        freeAll(arr);
        freeAll(result.circles);
        freeAll(result.squares);
        freeAll(result.rectangles);
        freeAll(result.rotated_rectangles);
        freeAll(result.polygons);
        return {
          mode: "shapes",
          shape: "ellipse",
          ellipses,
          metrics,
          ...containerField,
        };
      }
      if (shape === "rectangle") {
        const arr = result.rectangles;
        const rectangles = arr.map(rectangleFrom);
        freeAll(arr);
        freeAll(result.circles);
        freeAll(result.ellipses);
        freeAll(result.squares);
        freeAll(result.rotated_rectangles);
        freeAll(result.polygons);
        return {
          mode: "shapes",
          shape: "rectangle",
          rectangles,
          metrics,
          ...containerField,
        };
      }
      if (shape === "rotatedRectangle") {
        const arr = result.rotated_rectangles;
        const rotatedRectangles = arr.map(rotatedRectangleFrom);
        freeAll(arr);
        freeAll(result.circles);
        freeAll(result.ellipses);
        freeAll(result.squares);
        freeAll(result.rectangles);
        freeAll(result.polygons);
        return {
          mode: "shapes",
          shape: "rotatedRectangle",
          rotatedRectangles,
          metrics,
          ...containerField,
        };
      }
      const arr = result.squares;
      const squares = arr.map(squareFrom);
      freeAll(arr);
      freeAll(result.circles);
      freeAll(result.ellipses);
      freeAll(result.rectangles);
      freeAll(result.rotated_rectangles);
      freeAll(result.polygons);
      return {
        mode: "shapes",
        shape: "square",
        squares,
        metrics,
        ...containerField,
      };
    } finally {
      result.free();
    }
  } finally {
    freeAll(specs);
  }
}

/**
 * Build a canonical n-set Venn diagram and return its outlines
 * (`output: "polygons"`, default) or its exclusive-region decomposition
 * (`output: "regions"`).
 *
 * The shape defaults to `"ellipse"` (the only shape covering all of
 * `n ∈ 1..=5`). `"circle"` gives the classic one-, two-, and three-circle
 * diagrams; `"square"` and `"rectangle"` are axis-aligned. Every non-ellipse
 * shape caps at `n = 3` and throws a `RangeError` above that.
 *
 * No fitting is performed — the layout is hardcoded. Loss-style metrics in
 * the returned `Layout` are computed against a synthetic spec where every
 * region is requested at area 1.0; treat them as informational only.
 */
export function venn(options: VennOptions): Layout {
  const {
    n,
    shape = "ellipse",
    output = "polygons",
    polygonVertices = 256,
    complement,
  } = options;
  const maxN = shape === "ellipse" ? 5 : 3;
  if (!Number.isInteger(n) || n < 1 || n > maxN) {
    throw new RangeError(
      `venn: \`n\` must be an integer in 1..=${maxN} for shape "${shape}"`,
    );
  }
  const nVerts = Math.max(3, Math.floor(polygonVertices));

  if (output === "regions") {
    const fn =
      shape === "circle"
        ? wasm.generate_venn_regions_circles
        : shape === "square"
          ? wasm.generate_venn_regions_squares
          : shape === "rectangle"
            ? wasm.generate_venn_regions_rectangles
            : shape === "rotatedRectangle"
              ? wasm.generate_venn_regions_rotated_rectangles
              : wasm.generate_venn_regions_ellipses;
    const result = fn(n, nVerts, complement);
    try {
      const regionsArr = result.regions;
      const regions = regionsArr.map(regionFrom);
      freeAll(regionsArr);
      const container = containerFrom(result.container);
      return {
        mode: "regions",
        shape,
        regions,
        setAnchors: parseAnchors(result.set_anchors_json),
        setAnchorRegions: parseStringRecord(result.set_anchor_regions_json),
        metrics: metricsFromPolygonResult(result),
        ...(container ? { container } : {}),
      };
    } finally {
      result.free();
    }
  }

  const fn =
    shape === "circle"
      ? wasm.generate_venn_polygons_circles
      : shape === "square"
        ? wasm.generate_venn_polygons_squares
        : shape === "rectangle"
          ? wasm.generate_venn_polygons_rectangles
          : shape === "rotatedRectangle"
            ? wasm.generate_venn_polygons_rotated_rectangles
            : wasm.generate_venn_polygons_ellipses;
  const result = fn(n, nVerts, complement);
  try {
    const polysArr = result.polygons;
    const polygons = polysArr.map(polygonFrom);
    freeAll(polysArr);
    const metrics = metricsFromPolygonResult(result);
    const container = containerFrom(result.container);
    const containerField = container ? { container } : {};

    if (shape === "circle") {
      const arr = result.circles;
      const circles = arr.map(circleFrom);
      freeAll(arr);
      freeAll(result.ellipses);
      freeAll(result.squares);
      freeAll(result.rectangles);
      freeAll(result.rotated_rectangles);
      return {
        mode: "polygons",
        shape: "circle",
        polygons,
        circles,
        metrics,
        ...containerField,
      };
    }
    if (shape === "square") {
      const arr = result.squares;
      const squares = arr.map(squareFrom);
      freeAll(arr);
      freeAll(result.circles);
      freeAll(result.ellipses);
      freeAll(result.rectangles);
      freeAll(result.rotated_rectangles);
      return {
        mode: "polygons",
        shape: "square",
        polygons,
        squares,
        metrics,
        ...containerField,
      };
    }
    if (shape === "rectangle") {
      const arr = result.rectangles;
      const rectangles = arr.map(rectangleFrom);
      freeAll(arr);
      freeAll(result.circles);
      freeAll(result.ellipses);
      freeAll(result.squares);
      freeAll(result.rotated_rectangles);
      return {
        mode: "polygons",
        shape: "rectangle",
        polygons,
        rectangles,
        metrics,
        ...containerField,
      };
    }
    if (shape === "rotatedRectangle") {
      const arr = result.rotated_rectangles;
      const rotatedRectangles = arr.map(rotatedRectangleFrom);
      freeAll(arr);
      freeAll(result.circles);
      freeAll(result.ellipses);
      freeAll(result.squares);
      freeAll(result.rectangles);
      return {
        mode: "polygons",
        shape: "rotatedRectangle",
        polygons,
        rotatedRectangles,
        metrics,
        ...containerField,
      };
    }
    const arr = result.ellipses;
    const ellipses = arr.map(ellipseFrom);
    freeAll(arr);
    freeAll(result.circles);
    freeAll(result.squares);
    freeAll(result.rectangles);
    freeAll(result.rotated_rectangles);
    return {
      mode: "polygons",
      shape: "ellipse",
      polygons,
      ellipses,
      metrics,
      ...containerField,
    };
  } finally {
    result.free();
  }
}

const EXTERIOR_POLICY_MAP: Record<
  ExteriorPolicyName,
  "Raycast" | "ForceDirected" | "Matched"
> = {
  raycast: "Raycast",
  forceDirected: "ForceDirected",
  matched: "Matched",
};

const TETHER_SOURCE_MAP: Record<TetherSource, "Poi" | "Boundary"> = {
  poi: "Poi",
  boundary: "Boundary",
};

const PLACEMENT_KIND_MAP: Record<
  | "Interior"
  | "ExteriorRaycast"
  | "ExteriorForceDirected"
  | "ExteriorElbow"
  | "ExteriorMatched",
  PlacementKind
> = {
  Interior: "interior",
  ExteriorRaycast: "exteriorRaycast",
  ExteriorForceDirected: "exteriorForceDirected",
  ExteriorElbow: "exteriorElbow",
  ExteriorMatched: "exteriorMatched",
};

/**
 * Place a label per region.
 *
 * Returns a position for **every** requested region. The returned
 * [`LabelPlacement.kind`] tells the renderer whether the anchor is inside
 * the region (`"interior"`) or outside it (`"exteriorRaycast"` /
 * `"exteriorForceDirected"`), in which case `tether` points back into the
 * region so callers can draw a leader line.
 *
 * The default strategy uses the raycast exterior solver — anchor at the
 * POI when the label fits inside the region, otherwise raycast from the
 * diagram centroid through the POI to land outside the diagram bbox (or
 * container, when complement is set), padded by a per-label proportional
 * margin. Switch to `strategy.exterior = "forceDirected"` for crowded
 * diagrams where the raycast solver lands labels on top of unrelated
 * regions.
 *
 * @example
 * ```ts
 * import { placeLabelsForRegions } from "@jolars/eunoia";
 *
 * const placements = placeLabelsForRegions({
 *   regions: layout.regions,           // from euler({ output: "regions" })
 *   container: layout.container,       // optional; when complement was set
 *   sizes: { A: { w: 0.4, h: 0.2 }, "A&B": { w: 0.3, h: 0.15 } },
 * });
 * for (const [combo, p] of Object.entries(placements)) {
 *   if (p.kind === "interior") {
 *     drawLabelInside(combo, p.anchor);
 *   } else {
 *     drawLabelOutsideWithLeaderLine(combo, p.anchor, p.tether!);
 *   }
 * }
 * ```
 */
export function placeLabelsForRegions(
  options: PlaceLabelsForRegionsOptions,
): Record<string, LabelPlacement> {
  const { regions, container, sizes, strategy } = options;
  if (!regions || !Array.isArray(regions)) {
    throw new TypeError("placeLabelsForRegions: `regions` must be an array");
  }
  if (!sizes || typeof sizes !== "object") {
    throw new TypeError(
      "placeLabelsForRegions: `sizes` must be an object of region → { w, h }",
    );
  }

  const polygonsPayload: Record<
    string,
    { outer: [number, number][]; holes: [number, number][][] }[]
  > = {};
  for (const r of regions) {
    polygonsPayload[r.combination] = r.pieces.map(
      (p: RegionInput["pieces"][number]) => ({
        outer: p.outer.vertices.map((v: Point): [number, number] => [v.x, v.y]),
        holes: p.holes.map((h: { vertices: ReadonlyArray<Point> }) =>
          h.vertices.map((v: Point): [number, number] => [v.x, v.y]),
        ),
      }),
    );
  }

  const sizesPayload: Record<string, [number, number]> = {};
  for (const [k, v] of Object.entries(sizes)) {
    if (
      v &&
      Number.isFinite(v.w) &&
      Number.isFinite(v.h) &&
      v.w > 0 &&
      v.h > 0
    ) {
      sizesPayload[k] = [v.w, v.h];
    }
  }

  const containerJson = container
    ? JSON.stringify({
        x: container.x,
        y: container.y,
        width: container.width,
        height: container.height,
      })
    : undefined;

  let strategyJson: string | undefined;
  if (strategy) {
    type LeaderPayload =
      | {
          type: "straight";
          placement?: "Raycast" | "ForceDirected" | "Matched";
          margin?: number;
          iterations?: number;
        }
      | { type: "elbow"; margin?: number; minGap?: number };
    const payload: {
      leader?: LeaderPayload;
      precision?: number;
      tether?: "Poi" | "Boundary";
      leaderGap?: number;
    } = {};
    if (strategy.leader !== undefined) {
      const leader = strategy.leader;
      if (leader.type === "elbow") {
        const leaderPayload: LeaderPayload = { type: "elbow" };
        if (leader.margin !== undefined) leaderPayload.margin = leader.margin;
        if (leader.minGap !== undefined) leaderPayload.minGap = leader.minGap;
        payload.leader = leaderPayload;
      } else {
        const leaderPayload: LeaderPayload = { type: "straight" };
        if (leader.placement !== undefined) {
          const mapped = EXTERIOR_POLICY_MAP[leader.placement];
          if (mapped === undefined) {
            throw new RangeError(
              `placeLabelsForRegions: unknown leader placement "${leader.placement}"`,
            );
          }
          leaderPayload.placement = mapped;
        }
        if (leader.margin !== undefined) leaderPayload.margin = leader.margin;
        if (leader.iterations !== undefined)
          leaderPayload.iterations = leader.iterations;
        payload.leader = leaderPayload;
      }
    }
    if (strategy.precision !== undefined)
      payload.precision = strategy.precision;
    if (strategy.tether !== undefined) {
      const mapped = TETHER_SOURCE_MAP[strategy.tether];
      if (mapped === undefined) {
        throw new RangeError(
          `placeLabelsForRegions: unknown tether source "${strategy.tether}"`,
        );
      }
      payload.tether = mapped;
    }
    if (strategy.leaderGap !== undefined)
      payload.leaderGap = strategy.leaderGap;
    strategyJson = JSON.stringify(payload);
  }

  const json = wasm.place_region_labels(
    JSON.stringify(polygonsPayload),
    containerJson,
    JSON.stringify(sizesPayload),
    strategyJson,
  );
  type RawPlacement = {
    anchor: [number, number];
    kind: keyof typeof PLACEMENT_KIND_MAP;
    tether?: [number, number];
    leaderEnd?: [number, number];
    leaderWaypoints?: [number, number][];
  };
  const raw = JSON.parse(json) as Record<string, RawPlacement>;
  const out: Record<string, LabelPlacement> = {};
  for (const [k, v] of Object.entries(raw)) {
    const placement: LabelPlacement = {
      anchor: { x: v.anchor[0], y: v.anchor[1] },
      kind: PLACEMENT_KIND_MAP[v.kind],
      leaderWaypoints: (v.leaderWaypoints ?? []).map((p) => ({
        x: p[0],
        y: p[1],
      })),
    };
    if (v.tether) placement.tether = { x: v.tether[0], y: v.tether[1] };
    if (v.leaderEnd)
      placement.leaderEnd = { x: v.leaderEnd[0], y: v.leaderEnd[1] };
    out[k] = placement;
  }
  return out;
}

export interface PlacementsBboxOptions {
  /**
   * Placements as returned from [`placeLabelsForRegions`] (or any other
   * source that produces the same `Record<string, LabelPlacement>` shape).
   */
  placements: Record<string, LabelPlacement>;
  /**
   * Label dimensions per region, keyed by canonical combination. Entries
   * with no matching placement, or with non-finite or non-positive
   * dimensions, are skipped.
   */
  sizes: Record<string, LabelSize>;
}

/**
 * Bounding box of every placed label box.
 *
 * Returns the union AABB of every `(anchor.x ± w/2, anchor.y ± h/2)`,
 * shaped like the existing [`Container`] type (centre + extents). Returns
 * `undefined` when no placement contributed.
 *
 * Useful in resize loops: the canvas extent of a diagram is
 * `union(region_bbox, container?, placementsBbox(...))`. Pair it with
 * [`placeLabelsForRegions`] to drive the size → place → measure → re-place
 * fixed point in your own loop (font size in physical units, label size
 * in user coords = `font_pt / scale`).
 *
 * @example
 * ```ts
 * import { placeLabelsForRegions, placementsBbox } from "@jolars/eunoia";
 *
 * const placements = placeLabelsForRegions({ regions, sizes });
 * const labelBbox = placementsBbox({ placements, sizes });
 * if (labelBbox) {
 *   // Extend the canvas viewBox to cover exterior labels.
 *   extendViewport(labelBbox);
 * }
 * ```
 */
export function placementsBbox(
  options: PlacementsBboxOptions,
): Container | undefined {
  const { placements, sizes } = options;
  if (!placements || typeof placements !== "object") {
    throw new TypeError(
      "placementsBbox: `placements` must be a record of region → LabelPlacement",
    );
  }
  if (!sizes || typeof sizes !== "object") {
    throw new TypeError(
      "placementsBbox: `sizes` must be a record of region → { w, h }",
    );
  }

  const placementsPayload: Record<
    string,
    { anchor: [number, number]; kind: string }
  > = {};
  for (const [k, v] of Object.entries(placements)) {
    if (!v || !v.anchor) continue;
    placementsPayload[k] = {
      anchor: [v.anchor.x, v.anchor.y],
      // The Rust helper only reads `anchor`; we still ship `kind` so the
      // JSON shape matches `place_region_labels`'s output and the WASM
      // deserializer doesn't reject it.
      kind: "Interior",
    };
  }

  const sizesPayload: Record<string, [number, number]> = {};
  for (const [k, v] of Object.entries(sizes)) {
    if (
      v &&
      Number.isFinite(v.w) &&
      Number.isFinite(v.h) &&
      v.w > 0 &&
      v.h > 0
    ) {
      sizesPayload[k] = [v.w, v.h];
    }
  }

  const json = wasm.placements_bbox(
    JSON.stringify(placementsPayload),
    JSON.stringify(sizesPayload),
  );
  if (!json) return undefined;
  const raw = JSON.parse(json) as {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  return { x: raw.x, y: raw.y, width: raw.width, height: raw.height };
}

export interface LabelObstaclesOptions {
  /**
   * Placements as returned from [`placeLabelsForRegions`] (or any other
   * source producing the same `Record<string, LabelPlacement>` shape).
   */
  placements: Record<string, LabelPlacement>;
  /**
   * Label dimensions per region, keyed by canonical combination. Entries
   * with no matching placement, or with non-finite or non-positive
   * dimensions, are skipped.
   */
  sizes: Record<string, LabelSize>;
  /**
   * Extra margin added on every side of each box, in region coordinates.
   * Default `0`.
   */
  padding?: number;
}

/**
 * Turn placed labels into keep-out boxes for {@link placeGlyphsForRegions}.
 *
 * Labels are painted over glyphs, so the boxes measured for
 * [`placeLabelsForRegions`] are exactly the areas glyphs should avoid. Every
 * placement is included, interior and exterior alike — an exterior box is
 * the one most likely to land on a region that is not its own. Pure JS: no
 * wasm call.
 *
 * @example
 * ```ts
 * const placements = placeLabelsForRegions({ regions, sizes });
 * const glyphs = placeGlyphsForRegions({
 *   regions,
 *   counts,
 *   options: { obstacles: labelObstacles({ placements, sizes }) },
 * });
 * ```
 */
export function labelObstacles(options: LabelObstaclesOptions): Rect[] {
  const { placements, sizes, padding = 0 } = options;
  if (!placements || typeof placements !== "object") {
    throw new TypeError(
      "labelObstacles: `placements` must be a record of region → LabelPlacement",
    );
  }
  if (!sizes || typeof sizes !== "object") {
    throw new TypeError(
      "labelObstacles: `sizes` must be a record of region → { w, h }",
    );
  }
  const pad = Number.isFinite(padding) && padding > 0 ? padding : 0;

  const out: Rect[] = [];
  for (const key of Object.keys(placements).sort()) {
    const placement = placements[key];
    const size = sizes[key];
    if (!placement?.anchor || !size) continue;
    const { x, y } = placement.anchor;
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(size.w) ||
      !Number.isFinite(size.h) ||
      size.w <= 0 ||
      size.h <= 0
    ) {
      continue;
    }
    out.push({ x, y, width: size.w + 2 * pad, height: size.h + 2 * pad });
  }
  return out;
}

const GLYPH_ARRANGEMENT_MAP: Record<GlyphArrangement, "Uniform" | "Random"> = {
  uniform: "Uniform",
  random: "Random",
};

/**
 * Pack equally-sized circular glyphs — one mark per data unit,
 * eulerGlyphs-style — inside each region.
 *
 * All glyphs share a single radius so counts stay visually comparable
 * across regions. Omit `options.radius` to have the largest feasible
 * radius chosen automatically; pass one to take control, in which case
 * regions that cannot hold their count report the shortfall in
 * [`GlyphPlacements.unplaced`].
 *
 * @example
 * ```ts
 * import { placeGlyphsForRegions } from "@jolars/eunoia";
 *
 * const glyphs = placeGlyphsForRegions({
 *   regions: layout.regions,           // from euler({ output: "regions" })
 *   counts: { A: 20, B: 12, "A&B": 5 },
 * });
 * for (const [combo, points] of Object.entries(glyphs.positions)) {
 *   for (const p of points) {
 *     drawCircle(p.x, p.y, glyphs.radius);
 *   }
 * }
 * ```
 */
export function placeGlyphsForRegions(
  options: PlaceGlyphsForRegionsOptions,
): GlyphPlacements {
  const { regions, counts, options: glyphOptions } = options;
  if (!regions || !Array.isArray(regions)) {
    throw new TypeError("placeGlyphsForRegions: `regions` must be an array");
  }
  if (!counts || typeof counts !== "object") {
    throw new TypeError(
      "placeGlyphsForRegions: `counts` must be an object of region → count",
    );
  }

  const polygonsPayload: Record<
    string,
    { outer: [number, number][]; holes: [number, number][][] }[]
  > = {};
  for (const r of regions) {
    polygonsPayload[r.combination] = r.pieces.map(
      (p: RegionInput["pieces"][number]) => ({
        outer: p.outer.vertices.map((v: Point): [number, number] => [v.x, v.y]),
        holes: p.holes.map((h: { vertices: ReadonlyArray<Point> }) =>
          h.vertices.map((v: Point): [number, number] => [v.x, v.y]),
        ),
      }),
    );
  }

  const countsPayload: Record<string, number> = {};
  for (const [k, v] of Object.entries(counts)) {
    if (Number.isFinite(v) && v >= 0) {
      countsPayload[k] = Math.round(v);
    }
  }

  let optionsJson: string | undefined;
  if (glyphOptions) {
    const payload: {
      arrangement?: "Uniform" | "Random";
      radius?: number;
      gap?: number;
      seed?: number;
      precision?: number;
      maxAttempts?: number;
      obstacles?: Rect[];
    } = {};
    if (glyphOptions.arrangement !== undefined) {
      const mapped = GLYPH_ARRANGEMENT_MAP[glyphOptions.arrangement];
      if (mapped === undefined) {
        throw new RangeError(
          `placeGlyphsForRegions: unknown arrangement "${glyphOptions.arrangement}"`,
        );
      }
      payload.arrangement = mapped;
    }
    if (glyphOptions.radius !== undefined) payload.radius = glyphOptions.radius;
    if (glyphOptions.gap !== undefined) payload.gap = glyphOptions.gap;
    if (glyphOptions.seed !== undefined) payload.seed = glyphOptions.seed;
    if (glyphOptions.precision !== undefined)
      payload.precision = glyphOptions.precision;
    if (glyphOptions.maxAttempts !== undefined)
      payload.maxAttempts = glyphOptions.maxAttempts;
    if (glyphOptions.obstacles !== undefined) {
      const obstacles: Rect[] = [];
      for (const r of glyphOptions.obstacles) {
        if (
          r &&
          Number.isFinite(r.x) &&
          Number.isFinite(r.y) &&
          Number.isFinite(r.width) &&
          Number.isFinite(r.height) &&
          r.width > 0 &&
          r.height > 0
        ) {
          obstacles.push({ x: r.x, y: r.y, width: r.width, height: r.height });
        }
      }
      if (obstacles.length > 0) payload.obstacles = obstacles;
    }
    optionsJson = JSON.stringify(payload);
  }

  const json = wasm.place_region_glyphs(
    JSON.stringify(polygonsPayload),
    JSON.stringify(countsPayload),
    optionsJson,
  );
  const raw = JSON.parse(json) as {
    radius: number;
    positions: Record<string, [number, number][]>;
    unplaced?: Record<string, number>;
  };
  const positions: Record<string, Point[]> = {};
  for (const [k, pts] of Object.entries(raw.positions)) {
    positions[k] = pts.map((p) => ({ x: p[0], y: p[1] }));
  }
  const out: GlyphPlacements = { radius: raw.radius, positions };
  if (raw.unplaced && Object.keys(raw.unplaced).length > 0) {
    out.unplaced = raw.unplaced;
  }
  return out;
}

/**
 * Pack per-item text boxes — member names, typically — inside each region.
 *
 * The rectangular-footprint sibling of {@link placeGlyphsForRegions}: hand
 * it the `w × h` box you measured for every item and it returns one rect per
 * item plus the single diagram-wide `scale` those boxes were packed at.
 * Re-render your text at `fontSize * scale`.
 *
 * Auto-scale only ever *shrinks* (the bracket is `[minScale, 1]`), since you
 * own the reference font size — deliberately asymmetric with the disc
 * packer, which grows to fill.
 *
 * ## Dropped items
 *
 * Text boxes are typically five to ten times wider than tall, so a region
 * with ample *area* can still fail to seat a row of them. The packer takes
 * each region's items in order and stops at the first that fits nowhere, so
 * `boxes[key]` is a prefix of `sizes[key]`: **which** items get dropped is
 * decided by the order you supply them in. Sort meaningfully, check
 * `unplaced`, and consider a "+n more" affordance — or measure wrapped,
 * multi-line boxes, which the packer handles with no special support.
 *
 * @example
 * ```ts
 * import { placeGlyphBoxesForRegions } from "@jolars/eunoia";
 *
 * const members = { A: ["Ada", "Grace"], "A&B": ["Katherine"] };
 * const sizes = Object.fromEntries(
 *   Object.entries(members).map(([k, names]) => [k, names.map(measure)]),
 * );
 *
 * const placed = placeGlyphBoxesForRegions({
 *   regions: layout.regions,          // from euler({ output: "regions" })
 *   sizes,
 * });
 * for (const [combo, boxes] of Object.entries(placed.boxes)) {
 *   boxes.forEach((box, i) => {
 *     drawText(members[combo][i], box.x, box.y, fontSize * placed.scale);
 *   });
 * }
 * ```
 */
export function placeGlyphBoxesForRegions(
  options: PlaceGlyphBoxesForRegionsOptions,
): GlyphBoxPlacements {
  const { regions, sizes, options: boxOptions } = options;
  if (!regions || !Array.isArray(regions)) {
    throw new TypeError(
      "placeGlyphBoxesForRegions: `regions` must be an array",
    );
  }
  if (!sizes || typeof sizes !== "object") {
    throw new TypeError(
      "placeGlyphBoxesForRegions: `sizes` must be an object of region → measured boxes",
    );
  }

  const polygonsPayload: Record<
    string,
    { outer: [number, number][]; holes: [number, number][][] }[]
  > = {};
  for (const r of regions) {
    polygonsPayload[r.combination] = r.pieces.map(
      (p: RegionInput["pieces"][number]) => ({
        outer: p.outer.vertices.map((v: Point): [number, number] => [v.x, v.y]),
        holes: p.holes.map((h: { vertices: ReadonlyArray<Point> }) =>
          h.vertices.map((v: Point): [number, number] => [v.x, v.y]),
        ),
      }),
    );
  }

  // Degenerate extents are clamped rather than dropped: an item must keep
  // its index, since the result is matched back positionally.
  const sizesPayload: Record<string, [number, number][]> = {};
  for (const [k, items] of Object.entries(sizes)) {
    if (!Array.isArray(items)) continue;
    sizesPayload[k] = items.map((s): [number, number] => [
      s && Number.isFinite(s.w) && s.w > 0 ? s.w : 0,
      s && Number.isFinite(s.h) && s.h > 0 ? s.h : 0,
    ]);
  }

  let optionsJson: string | undefined;
  if (boxOptions) {
    const payload: {
      arrangement?: "Uniform" | "Random";
      scale?: number;
      minScale?: number;
      gap?: number;
      seed?: number;
      precision?: number;
      maxAttempts?: number;
      obstacles?: Rect[];
    } = {};
    if (boxOptions.arrangement !== undefined) {
      const mapped = GLYPH_ARRANGEMENT_MAP[boxOptions.arrangement];
      if (mapped === undefined) {
        throw new RangeError(
          `placeGlyphBoxesForRegions: unknown arrangement "${boxOptions.arrangement}"`,
        );
      }
      payload.arrangement = mapped;
    }
    if (boxOptions.scale !== undefined) payload.scale = boxOptions.scale;
    if (boxOptions.minScale !== undefined)
      payload.minScale = boxOptions.minScale;
    if (boxOptions.gap !== undefined) payload.gap = boxOptions.gap;
    if (boxOptions.seed !== undefined) payload.seed = boxOptions.seed;
    if (boxOptions.precision !== undefined)
      payload.precision = boxOptions.precision;
    if (boxOptions.maxAttempts !== undefined)
      payload.maxAttempts = boxOptions.maxAttempts;
    if (boxOptions.obstacles !== undefined) {
      const obstacles: Rect[] = [];
      for (const r of boxOptions.obstacles) {
        if (
          r &&
          Number.isFinite(r.x) &&
          Number.isFinite(r.y) &&
          Number.isFinite(r.width) &&
          Number.isFinite(r.height) &&
          r.width > 0 &&
          r.height > 0
        ) {
          obstacles.push({ x: r.x, y: r.y, width: r.width, height: r.height });
        }
      }
      if (obstacles.length > 0) payload.obstacles = obstacles;
    }
    optionsJson = JSON.stringify(payload);
  }

  const json = wasm.place_region_glyph_boxes(
    JSON.stringify(polygonsPayload),
    JSON.stringify(sizesPayload),
    optionsJson,
  );
  const raw = JSON.parse(json) as {
    scale: number;
    boxes: Record<string, [number, number, number, number][]>;
    unplaced?: Record<string, number>;
  };
  const boxes: Record<string, Rect[]> = {};
  for (const [k, quads] of Object.entries(raw.boxes)) {
    boxes[k] = quads.map((q) => ({
      x: q[0],
      y: q[1],
      width: q[2],
      height: q[3],
    }));
  }
  const out: GlyphBoxPlacements = { scale: raw.scale, boxes };
  if (raw.unplaced && Object.keys(raw.unplaced).length > 0) {
    out.unplaced = raw.unplaced;
  }
  return out;
}
