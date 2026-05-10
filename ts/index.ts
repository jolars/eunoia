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

export type ShapeType = "circle" | "ellipse" | "square" | "rectangle";
export type InputType = "exclusive" | "inclusive";
export type OutputMode = "shapes" | "polygons" | "regions";
export type Optimizer =
  | "cmaEsLm"
  | "levenbergMarquardt"
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
      mode: "regions";
      shape: ShapeType;
      regions: Region[];
      setAnchors: Record<string, Point>;
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
  /** Final-stage optimizer. Default `"cmaEsLm"`. */
  optimizer?: Optimizer;
  /** Loss function. Defaults to the optimizer's preferred loss. */
  loss?: LossType;
  /** Optimizer convergence tolerance. */
  tolerance?: number;
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

export interface PlaceLabelsOptions extends EulerOptions {
  /**
   * Label dimensions per region, keyed by canonical combination
   * (e.g. `"A"`, `"A&B"`, `""` for the complement region).
   *
   * Measure your label text once with the renderer (e.g. SVG `getBBox()`,
   * canvas `measureText`, R `grid::grobWidth`/`grobHeight`) before calling.
   */
  sizes: Record<string, LabelSize>;
  /**
   * Polylabel-style search precision in diagram coordinates. Smaller values
   * yield more accurate anchors at higher cost. Default `0.01`.
   */
  precision?: number;
}

/**
 * Minimal region shape accepted by [`placeRegionLabelsForRegions`]. Both
 * the wrapper's [`Region`] type and the web app's `RegionPolygon` satisfy
 * this — only `combination` and `pieces` (with outer + holes vertex lists)
 * are read.
 */
export interface RegionInput {
  combination: string;
  pieces: ReadonlyArray<{
    outer: { vertices: ReadonlyArray<Point> };
    holes: ReadonlyArray<{ vertices: ReadonlyArray<Point> }>;
  }>;
}

export interface PlaceLabelsForRegionsOptions {
  /**
   * Already-decomposed regions in any coordinate space — the fit-check is
   * scale-invariant as long as `sizes` are in the same units as the
   * polygon vertices.
   */
  regions: ReadonlyArray<RegionInput>;
  /**
   * Label dimensions per region, keyed by canonical combination
   * (e.g. `"A"`, `"A&B"`, `""` for the complement region).
   */
  sizes: Record<string, LabelSize>;
  /** Polylabel-style search precision. Default `0.01`. */
  precision?: number;
}

/**
 * What to do when a label box would (or would not) fit inside its region.
 *
 * - `"strict"` — anchor at the POI only when the box fits; otherwise fall
 *   through to the exterior fallback.
 * - `"loose"` — always anchor at the POI, even when the box overflows the
 *   polygon. **Not implemented yet.**
 */
export type InteriorPolicy = "strict" | "loose";

/**
 * What to do for regions where the strict interior check says "doesn't fit".
 *
 * - `"raycast"` — deterministic ray from the diagram centroid through the
 *   region's POI; anchor lands outside the diagram bbox (or container, when
 *   complement is set), padded by `margin`.
 * - `"none"` — omit the region from the result. **Not implemented yet** —
 *   callers wanting that behaviour should use [`placeRegionLabelsForRegions`]
 *   directly.
 * - `"forceDirected"` — iterative spring + repulsion solve. Initial
 *   positions come from the raycast, then each label is pulled toward
 *   that "home" by a soft spring while being repelled from other labels
 *   *and* from foreign region polygons (the eunoia-specific bit: ggrepel
 *   can only see labels). Use this when raycast labels visually overlap
 *   unrelated regions or pile up at similar angles.
 */
export type ExteriorPolicyName = "raycast" | "none" | "forceDirected";

export interface PlacementStrategy {
  /** Default `"strict"`. */
  interior?: InteriorPolicy;
  /** Default `"raycast"`. */
  exterior?: ExteriorPolicyName;
  /**
   * Margin around the diagram bbox/container, applied to both
   * `"raycast"` and `"forceDirected"` exteriors. Omit to use a per-region
   * proportional default of `0.5 * max(label_w, label_h)`.
   */
  margin?: number;
  /**
   * Iteration cap for the `"forceDirected"` solver. Ignored otherwise.
   * Defaults to 200; raise for crowded diagrams that haven't converged.
   */
  iterations?: number;
  /** Polylabel-style search precision. Default `0.01`. */
  precision?: number;
}

/**
 * Discriminator on [`LabelPlacement`] — tells the renderer whether the
 * anchor is inside or outside the region.
 */
export type PlacementKind =
  | "interior"
  | "interiorOverflow"
  | "exteriorRaycast"
  | "exteriorForceDirected";

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
}

export interface PlaceLabelsForRegionsStrategicOptions {
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

export interface VennOptions {
  /** Number of sets in the Venn diagram (1 ≤ n ≤ 5). */
  n: number;
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
  cmaEsLm: wasm.WasmOptimizer.CmaEsLm,
  levenbergMarquardt: wasm.WasmOptimizer.LevenbergMarquardt,
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
    optimizer = "cmaEsLm",
    loss,
    tolerance,
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
  const optimizerArg = OPTIMIZER_MAP[optimizer];
  if (optimizerArg === undefined) {
    throw new RangeError(`euler: unknown optimizer "${optimizer}"`);
  }
  const lossArg = loss !== undefined ? LOSS_MAP[loss] : undefined;
  if (loss !== undefined && lossArg === undefined) {
    throw new RangeError(`euler: unknown loss "${loss}"`);
  }
  const tolArg = tolerance && tolerance > 0 ? tolerance : undefined;
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
              : wasm.generate_region_polygons_ellipses;
      const result = fn(
        specs,
        inputType,
        nVerts,
        seedArg,
        optimizerArg,
        lossArg,
        tolArg,
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
            : wasm.generate_ellipses_as_polygons;
    const result = fn(
      specs,
      inputType,
      nVerts,
      seedArg,
      optimizerArg,
      lossArg,
      tolArg,
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
          return {
            mode: "polygons",
            shape: "rectangle",
            polygons,
            rectangles,
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
        freeAll(result.polygons);
        return {
          mode: "shapes",
          shape: "rectangle",
          rectangles,
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
 * Build a canonical n-set Venn diagram (1 ≤ n ≤ 5) and return its outlines
 * (`output: "polygons"`, default) or its exclusive-region decomposition
 * (`output: "regions"`).
 *
 * No fitting is performed — the layout is hardcoded. Loss-style metrics in
 * the returned `Layout` are computed against a synthetic spec where every
 * region is requested at area 1.0; treat them as informational only.
 */
export function venn(options: VennOptions): Layout {
  const { n, output = "polygons", polygonVertices = 256, complement } = options;
  if (!Number.isInteger(n) || n < 1 || n > 5) {
    throw new RangeError("venn: `n` must be an integer in 1..=5");
  }
  const nVerts = Math.max(3, Math.floor(polygonVertices));

  if (output === "regions") {
    const result = wasm.generate_venn_regions(n, nVerts, complement);
    try {
      const regionsArr = result.regions;
      const regions = regionsArr.map(regionFrom);
      freeAll(regionsArr);
      const container = containerFrom(result.container);
      return {
        mode: "regions",
        shape: "ellipse",
        regions,
        setAnchors: parseAnchors(result.set_anchors_json),
        metrics: metricsFromPolygonResult(result),
        ...(container ? { container } : {}),
      };
    } finally {
      result.free();
    }
  }

  const result = wasm.generate_venn_polygons(n, nVerts, complement);
  try {
    const polysArr = result.polygons;
    const polygons = polysArr.map(polygonFrom);
    freeAll(polysArr);
    const ellipsesArr = result.ellipses;
    const ellipses = ellipsesArr.map(ellipseFrom);
    freeAll(ellipsesArr);
    freeAll(result.circles);
    freeAll(result.squares);
    freeAll(result.rectangles);
    const container = containerFrom(result.container);
    return {
      mode: "polygons",
      shape: "ellipse",
      polygons,
      ellipses,
      metrics: metricsFromPolygonResult(result),
      ...(container ? { container } : {}),
    };
  } finally {
    result.free();
  }
}

/**
 * Per-region label-fit predicate.
 *
 * Re-fits the diagram and, for each entry in `sizes`, asks whether an
 * axis-aligned rectangle of the given `w × h` fits inside the corresponding
 * exclusive region (hole-aware). On success, returns the rectangle's centre
 * as the label anchor; on failure, the region is omitted from the result.
 *
 * The fit-check is shape-agnostic — the `shape` field of `options` only
 * affects how the diagram is fitted before its regions are decomposed.
 *
 * Caveat: the underlying inscribed-rectangle bound is radial-conservative,
 * so `None` for very wide-and-short or tall-and-narrow regions can be a
 * false negative. A directional-clearance solver is a planned follow-up.
 *
 * @example
 * ```ts
 * import { placeRegionLabels } from "@jolars/eunoia";
 *
 * const placements = placeRegionLabels({
 *   sets: { A: 5, B: 3, "A&B": 1 },
 *   sizes: {
 *     A: { w: 0.4, h: 0.2 },
 *     B: { w: 0.4, h: 0.2 },
 *     "A&B": { w: 0.2, h: 0.1 },
 *   },
 * });
 * // placements: { A: { x, y }, B: { x, y }, "A&B": { x, y } } — any
 * // region whose label didn't fit is absent.
 * ```
 */
export function placeRegionLabels(
  options: PlaceLabelsOptions,
): Record<string, Point> {
  const {
    sets,
    sizes,
    inputType = "exclusive",
    shape = "circle",
    seed,
    optimizer = "cmaEsLm",
    loss,
    tolerance,
    polygonVertices = 256,
    complement,
    precision,
  } = options;

  if (!sets || typeof sets !== "object") {
    throw new TypeError(
      "placeRegionLabels: `sets` must be an object of name → size",
    );
  }
  if (!sizes || typeof sizes !== "object") {
    throw new TypeError(
      "placeRegionLabels: `sizes` must be an object of region → { w, h }",
    );
  }

  const specs = buildSpecs(sets);
  if (specs.length === 0) {
    throw new Error(
      "placeRegionLabels: `sets` must contain at least one entry with size > 0",
    );
  }

  const seedArg = toSeed(seed);
  const optimizerArg = OPTIMIZER_MAP[optimizer];
  if (optimizerArg === undefined) {
    throw new RangeError(`placeRegionLabels: unknown optimizer "${optimizer}"`);
  }
  const lossArg = loss !== undefined ? LOSS_MAP[loss] : undefined;
  if (loss !== undefined && lossArg === undefined) {
    throw new RangeError(`placeRegionLabels: unknown loss "${loss}"`);
  }
  const tolArg = tolerance && tolerance > 0 ? tolerance : undefined;
  const nVerts = Math.max(3, Math.floor(polygonVertices));

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

  try {
    const json = wasm.compute_region_label_placements(
      specs,
      inputType,
      shape,
      nVerts,
      JSON.stringify(sizesPayload),
      precision,
      seedArg,
      optimizerArg,
      lossArg,
      tolArg,
      complement,
    );
    const raw = JSON.parse(json) as Record<string, [number, number]>;
    const out: Record<string, Point> = {};
    for (const [k, v] of Object.entries(raw)) out[k] = { x: v[0], y: v[1] };
    return out;
  } finally {
    freeAll(specs);
  }
}

/**
 * Per-region label-fit predicate operating on already-decomposed regions
 * (no re-fit). Use this when you already have a fitted layout (e.g. the
 * `regions` from `euler({ output: "regions" })` or the web app's
 * `result.regions`) and want to ask "does my label of size `(w, h)` fit?"
 * cheaply, without paying for a full diagram re-fit on every label-size
 * change.
 *
 * Returns a map of fitting region anchor by canonical combination string.
 * Regions whose label does not fit are absent from the result.
 *
 * The fit-check is scale-invariant: pass `regions` and `sizes` in whatever
 * coordinate space you have (original fit units, normalised SVG units,
 * pixels — as long as both sides agree).
 */
export function placeRegionLabelsForRegions(
  options: PlaceLabelsForRegionsOptions,
): Record<string, Point> {
  const { regions, sizes, precision } = options;
  if (!regions || !Array.isArray(regions)) {
    throw new TypeError(
      "placeRegionLabelsForRegions: `regions` must be an array",
    );
  }
  if (!sizes || typeof sizes !== "object") {
    throw new TypeError(
      "placeRegionLabelsForRegions: `sizes` must be an object of region → { w, h }",
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

  const json = wasm.fit_labels_for_polygons(
    JSON.stringify(polygonsPayload),
    JSON.stringify(sizesPayload),
    precision,
  );
  const raw = JSON.parse(json) as Record<string, [number, number]>;
  const out: Record<string, Point> = {};
  for (const [k, v] of Object.entries(raw)) out[k] = { x: v[0], y: v[1] };
  return out;
}

const INTERIOR_POLICY_MAP: Record<InteriorPolicy, "Strict" | "Loose"> = {
  strict: "Strict",
  loose: "Loose",
};

const EXTERIOR_POLICY_MAP: Record<
  ExteriorPolicyName,
  "Raycast" | "None" | "ForceDirected"
> = {
  raycast: "Raycast",
  none: "None",
  forceDirected: "ForceDirected",
};

const PLACEMENT_KIND_MAP: Record<
  "Interior" | "InteriorOverflow" | "ExteriorRaycast" | "ExteriorForceDirected",
  PlacementKind
> = {
  Interior: "interior",
  InteriorOverflow: "interiorOverflow",
  ExteriorRaycast: "exteriorRaycast",
  ExteriorForceDirected: "exteriorForceDirected",
};

/**
 * Strategy-driven label placement on already-decomposed regions (no re-fit).
 *
 * Unlike [`placeRegionLabelsForRegions`] (a predicate that omits regions
 * where the label doesn't fit), this function returns a position for **every**
 * requested region. The returned [`LabelPlacement.kind`] tells the renderer
 * whether the anchor is inside the region (`"interior"`) or outside it
 * (`"exteriorRaycast"`, with a `tether` pointing back at the region's POI).
 *
 * The default strategy is `Strict + Raycast` — anchor at the POI when the
 * label fits inside the region, otherwise raycast from the diagram centroid
 * through the POI to land outside the diagram bbox (or container, when
 * complement is set), padded by a per-label proportional margin.
 *
 * Selecting unimplemented strategy variants (`interior: "loose"`,
 * `exterior: "none"`) throws — pattern-match on the error and fall back
 * to a different strategy or to the predicate
 * [`placeRegionLabelsForRegions`].
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
  options: PlaceLabelsForRegionsStrategicOptions,
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
    const payload: {
      interior?: "Strict" | "Loose";
      exterior?: "Raycast" | "None" | "ForceDirected";
      margin?: number;
      iterations?: number;
      precision?: number;
    } = {};
    if (strategy.interior !== undefined) {
      const mapped = INTERIOR_POLICY_MAP[strategy.interior];
      if (mapped === undefined) {
        throw new RangeError(
          `placeLabelsForRegions: unknown interior policy "${strategy.interior}"`,
        );
      }
      payload.interior = mapped;
    }
    if (strategy.exterior !== undefined) {
      const mapped = EXTERIOR_POLICY_MAP[strategy.exterior];
      if (mapped === undefined) {
        throw new RangeError(
          `placeLabelsForRegions: unknown exterior policy "${strategy.exterior}"`,
        );
      }
      payload.exterior = mapped;
    }
    if (strategy.margin !== undefined) payload.margin = strategy.margin;
    if (strategy.iterations !== undefined)
      payload.iterations = strategy.iterations;
    if (strategy.precision !== undefined)
      payload.precision = strategy.precision;
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
  };
  const raw = JSON.parse(json) as Record<string, RawPlacement>;
  const out: Record<string, LabelPlacement> = {};
  for (const [k, v] of Object.entries(raw)) {
    const placement: LabelPlacement = {
      anchor: { x: v.anchor[0], y: v.anchor[1] },
      kind: PLACEMENT_KIND_MAP[v.kind],
    };
    if (v.tether) placement.tether = { x: v.tether[0], y: v.tether[1] };
    out[k] = placement;
  }
  return out;
}
