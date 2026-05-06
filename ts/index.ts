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

export type Layout =
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
    };

export interface FitOptions {
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
}

export interface VennOptions {
  /** Number of sets in the Venn diagram (1 ≤ n ≤ 5). */
  n: number;
  /** Output mode: polygon outlines per set, or exclusive regions. Default `"polygons"`. */
  output?: "polygons" | "regions";
  /** Number of vertices per polygon outline. Default 256. */
  polygonVertices?: number;
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
export function fit(options: FitOptions): Layout {
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
  } = options;

  if (!sets || typeof sets !== "object") {
    throw new TypeError("fit: `sets` must be an object of name → size");
  }

  const specs = buildSpecs(sets);
  if (specs.length === 0) {
    throw new Error(
      "fit: `sets` must contain at least one entry with size > 0",
    );
  }

  const seedArg = toSeed(seed);
  const optimizerArg = OPTIMIZER_MAP[optimizer];
  if (optimizerArg === undefined) {
    throw new RangeError(`fit: unknown optimizer "${optimizer}"`);
  }
  const lossArg = loss !== undefined ? LOSS_MAP[loss] : undefined;
  if (loss !== undefined && lossArg === undefined) {
    throw new RangeError(`fit: unknown loss "${loss}"`);
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
      );
      try {
        const regionsArr = result.regions;
        const regions = regionsArr.map(regionFrom);
        freeAll(regionsArr);
        return {
          mode: "regions",
          shape,
          regions,
          setAnchors: parseAnchors(result.set_anchors_json),
          metrics: metricsFromPolygonResult(result),
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
    );
    try {
      const metrics = metricsFromPolygonResult(result);

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
        return { mode: "shapes", shape: "circle", circles, metrics };
      }
      if (shape === "ellipse") {
        const arr = result.ellipses;
        const ellipses = arr.map(ellipseFrom);
        freeAll(arr);
        freeAll(result.circles);
        freeAll(result.squares);
        freeAll(result.rectangles);
        freeAll(result.polygons);
        return { mode: "shapes", shape: "ellipse", ellipses, metrics };
      }
      if (shape === "rectangle") {
        const arr = result.rectangles;
        const rectangles = arr.map(rectangleFrom);
        freeAll(arr);
        freeAll(result.circles);
        freeAll(result.ellipses);
        freeAll(result.squares);
        freeAll(result.polygons);
        return { mode: "shapes", shape: "rectangle", rectangles, metrics };
      }
      const arr = result.squares;
      const squares = arr.map(squareFrom);
      freeAll(arr);
      freeAll(result.circles);
      freeAll(result.ellipses);
      freeAll(result.rectangles);
      freeAll(result.polygons);
      return { mode: "shapes", shape: "square", squares, metrics };
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
  const { n, output = "polygons", polygonVertices = 256 } = options;
  if (!Number.isInteger(n) || n < 1 || n > 5) {
    throw new RangeError("venn: `n` must be an integer in 1..=5");
  }
  const nVerts = Math.max(3, Math.floor(polygonVertices));

  if (output === "regions") {
    const result = wasm.generate_venn_regions(n, nVerts);
    try {
      const regionsArr = result.regions;
      const regions = regionsArr.map(regionFrom);
      freeAll(regionsArr);
      return {
        mode: "regions",
        shape: "ellipse",
        regions,
        setAnchors: parseAnchors(result.set_anchors_json),
        metrics: metricsFromPolygonResult(result),
      };
    } finally {
      result.free();
    }
  }

  const result = wasm.generate_venn_polygons(n, nVerts);
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
    return {
      mode: "polygons",
      shape: "ellipse",
      polygons,
      ellipses,
      metrics: metricsFromPolygonResult(result),
    };
  } finally {
    result.free();
  }
}
