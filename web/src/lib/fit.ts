import {
  type Layout,
  type LossType,
  type Optimizer,
  type Point,
  euler as runFitWrapper,
  venn,
} from "@jolars/eunoia";
import type {
  AdvancedOptions,
  DiagramType,
  FitResult,
  InputType,
  LossName,
  OptimizerName,
  Row,
  ShapeType,
  VennSetCount,
} from "./types/diagram";

const POLYGON_VERTICES = 256;

const OPTIMIZER_MAP: Record<OptimizerName, Optimizer> = {
  CmaEsLm: "cmaEsLm",
  LevenbergMarquardt: "levenbergMarquardt",
  Lbfgs: "lbfgs",
  NelderMead: "nelderMead",
};

const LOSS_MAP: Record<LossName, LossType> = {
  SumSquared: "sumSquared",
  SumAbsolute: "sumAbsolute",
  SumAbsoluteRegionError: "sumAbsoluteRegionError",
  SumSquaredRegionError: "sumSquaredRegionError",
  MaxAbsolute: "maxAbsolute",
  MaxSquared: "maxSquared",
  RootMeanSquared: "rootMeanSquared",
  Stress: "stress",
  DiagError: "diagError",
};

function buildSets(rows: Row[]): Record<string, number> {
  const sets: Record<string, number> = {};
  for (const r of rows) {
    const input = r.input.trim();
    if (input === "" || r.size <= 0) continue;
    if (input.endsWith("&") || input.endsWith("|")) continue;
    sets[input] = r.size;
  }
  return sets;
}

interface NormalizationContext {
  scale: number;
  minX: number;
  minY: number;
  precision: number;
}

function normalizeBounds(boundsItems: Point[][]): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const item of boundsItems) {
    for (const v of item) {
      if (v.x < minX) minX = v.x;
      if (v.y < minY) minY = v.y;
      if (v.x > maxX) maxX = v.x;
      if (v.y > maxY) maxY = v.y;
    }
  }
  if (!Number.isFinite(minX)) {
    minX = 0;
    minY = 0;
    maxX = 1;
    maxY = 1;
  }
  return { minX, minY, maxX, maxY };
}

function buildContext(boundsItems: Point[][]): NormalizationContext {
  const { minX, minY, maxX, maxY } = normalizeBounds(boundsItems);
  const maxDim = Math.max(maxX - minX, maxY - minY) || 1;
  const scale = 100 / maxDim;
  return { scale, minX, minY, precision: maxDim * 0.001 };
}

function normPoint(p: Point, ctx: NormalizationContext): Point {
  return { x: (p.x - ctx.minX) * ctx.scale, y: (p.y - ctx.minY) * ctx.scale };
}

type RegionPieces = Extract<
  Layout,
  { mode: "regions" }
>["regions"][number]["pieces"];

function pieceVerts(piece: RegionPieces[number]): Point[] {
  const out: Point[] = [];
  out.push(...piece.outer.vertices);
  for (const h of piece.holes) out.push(...h.vertices);
  return out;
}

function normSetAnchors(
  anchors: Record<string, Point>,
  ctx: NormalizationContext,
): Record<string, Point> {
  const out: Record<string, Point> = {};
  for (const [k, v] of Object.entries(anchors)) out[k] = normPoint(v, ctx);
  return out;
}

export interface FitInputs {
  rows: Row[];
  inputType: InputType;
  shapeType: ShapeType;
  diagramType: DiagramType;
  vennN: VennSetCount;
  advanced: AdvancedOptions;
}

export type {
  AdvancedOptions,
  DiagramType,
  InputType,
  Row,
  ShapeType,
  VennSetCount,
};

function containerBoundsFor(
  c: { x: number; y: number; width: number; height: number } | undefined,
): Point[] {
  if (!c) return [];
  const hw = c.width / 2;
  const hh = c.height / 2;
  return [
    { x: c.x - hw, y: c.y - hh },
    { x: c.x + hw, y: c.y + hh },
  ];
}

function normContainer(
  c: { x: number; y: number; width: number; height: number } | undefined,
  ctx: NormalizationContext,
): { x: number; y: number; width: number; height: number } | undefined {
  if (!c) return undefined;
  const center = normPoint({ x: c.x, y: c.y }, ctx);
  return {
    x: center.x,
    y: center.y,
    width: c.width * ctx.scale,
    height: c.height * ctx.scale,
  };
}

// Scale every geometric coordinate in a `Layout` into the ~100-unit canvas the
// app's style defaults are calibrated for, returning the *same* `Layout` shape
// (so it feeds straight into `@jolars/eunoia/svg`). Non-geometric fields
// (areas, totals, metrics, labels) pass through untouched.
function scaleLayout(layout: Layout, ctx: NormalizationContext): Layout {
  const s = ctx.scale;
  const pt = (p: Point): Point => normPoint(p, ctx);
  const poly = <T extends { vertices: Point[] }>(p: T): T => ({
    ...p,
    vertices: p.vertices.map(pt),
  });
  const container = normContainer(layout.container, ctx);
  const tail = container
    ? { metrics: layout.metrics, container }
    : { metrics: layout.metrics };

  if (layout.mode === "regions") {
    return {
      mode: "regions",
      shape: layout.shape,
      regions: layout.regions.map((r) => ({
        combination: r.combination,
        totalArea: r.totalArea,
        pieces: r.pieces.map((pc) => ({
          ...pc,
          outer: poly(pc.outer),
          holes: pc.holes.map(poly),
        })),
        labelAnchor: pt(r.labelAnchor),
      })),
      setAnchors: normSetAnchors(layout.setAnchors, ctx),
      ...tail,
    };
  }

  if (layout.mode !== "polygons") {
    throw new Error(`unexpected layout mode: ${layout.mode}`);
  }

  const polygons = layout.polygons.map(poly);
  switch (layout.shape) {
    case "circle":
      return {
        mode: "polygons",
        shape: "circle",
        polygons,
        circles: layout.circles.map((c) => ({
          label: c.label,
          x: pt(c).x,
          y: pt(c).y,
          radius: c.radius * s,
          labelAnchor: pt(c.labelAnchor),
        })),
        ...tail,
      };
    case "ellipse":
      return {
        mode: "polygons",
        shape: "ellipse",
        polygons,
        ellipses: layout.ellipses.map((e) => ({
          label: e.label,
          x: pt(e).x,
          y: pt(e).y,
          semiMajor: e.semiMajor * s,
          semiMinor: e.semiMinor * s,
          rotation: e.rotation,
          labelAnchor: pt(e.labelAnchor),
        })),
        ...tail,
      };
    case "square":
      return {
        mode: "polygons",
        shape: "square",
        polygons,
        squares: layout.squares.map((sq) => ({
          label: sq.label,
          x: pt(sq).x,
          y: pt(sq).y,
          side: sq.side * s,
          labelAnchor: pt(sq.labelAnchor),
        })),
        ...tail,
      };
    case "rectangle":
      return {
        mode: "polygons",
        shape: "rectangle",
        polygons,
        rectangles: layout.rectangles.map((r) => ({
          label: r.label,
          x: pt(r).x,
          y: pt(r).y,
          width: r.width * s,
          height: r.height * s,
          labelAnchor: pt(r.labelAnchor),
        })),
        ...tail,
      };
  }
}

function layoutToFitResult(
  layout: Layout,
  shapeType: ShapeType,
  complement?: number,
): FitResult {
  const m = layout.metrics;
  const metrics = {
    loss: m.loss,
    stress: m.stress,
    diagError: m.diagError,
    iterations: m.iterations,
    target: m.targetAreas,
    fitted: m.fittedAreas,
    regionError: m.regionError,
    residuals: m.residuals,
  };

  let boundsItems: Point[][];
  if (layout.mode === "regions") {
    boundsItems = [
      layout.regions.flatMap((r) => r.pieces.flatMap(pieceVerts)),
      containerBoundsFor(layout.container),
    ];
  } else if (layout.mode === "polygons") {
    boundsItems = [
      ...layout.polygons.map((p) => p.vertices),
      containerBoundsFor(layout.container),
    ];
  } else {
    throw new Error(`unexpected layout mode: ${layout.mode}`);
  }
  const ctx = buildContext(boundsItems);

  return {
    layout: scaleLayout(layout, ctx),
    shapeType,
    complement,
    metrics,
  };
}

export function runFit(inputs: FitInputs): FitResult | null {
  const c = inputs.advanced.complement;
  const complement =
    inputs.advanced.useComplement &&
    typeof c === "number" &&
    Number.isFinite(c) &&
    c >= 0
      ? c
      : undefined;

  if (inputs.diagramType === "venn") {
    const layout = venn({
      n: inputs.vennN,
      output: inputs.advanced.showRegions ? "regions" : "polygons",
      polygonVertices: POLYGON_VERTICES,
      complement,
    });
    return layoutToFitResult(layout, "ellipse", complement);
  }

  const sets = buildSets(inputs.rows);
  if (Object.keys(sets).length === 0) return null;

  const tolerance =
    Number.isFinite(inputs.advanced.tolerance) && inputs.advanced.tolerance > 0
      ? inputs.advanced.tolerance
      : undefined;

  const layout = runFitWrapper({
    sets,
    inputType: inputs.inputType,
    shape: inputs.shapeType,
    output: inputs.advanced.showRegions ? "regions" : "polygons",
    seed:
      inputs.advanced.useSeed && inputs.advanced.seed !== undefined
        ? inputs.advanced.seed
        : undefined,
    optimizer: OPTIMIZER_MAP[inputs.advanced.optimizer],
    loss: LOSS_MAP[inputs.advanced.lossType],
    tolerance,
    polygonVertices: POLYGON_VERTICES,
    complement,
  });

  return layoutToFitResult(layout, inputs.shapeType, complement);
}
