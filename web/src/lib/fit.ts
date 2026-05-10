import {
  type Layout,
  type LossType,
  type Optimizer,
  type Point,
  euler as runFitWrapper,
  venn,
  type Polygon as WrapperPolygon,
  type Region as WrapperRegion,
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
  if (!isFinite(minX)) {
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

function normPolygon(p: WrapperPolygon, ctx: NormalizationContext) {
  return {
    label: p.label,
    vertices: p.vertices.map((v) => normPoint(v, ctx)),
  };
}

function pieceVerts(piece: WrapperRegion["pieces"][number]): Point[] {
  const out: Point[] = [];
  out.push(...piece.outer.vertices);
  for (const h of piece.holes) out.push(...h.vertices);
  return out;
}

function normRegions(regions: WrapperRegion[], ctx: NormalizationContext) {
  return regions.map((r) => ({
    combination: r.combination,
    totalArea: r.totalArea,
    pieces: r.pieces.map((piece) => ({
      outer: normPolygon(piece.outer, ctx),
      holes: piece.holes.map((h) => normPolygon(h, ctx)),
    })),
    labelX: (r.labelAnchor.x - ctx.minX) * ctx.scale,
    labelY: (r.labelAnchor.y - ctx.minY) * ctx.scale,
  }));
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

  if (layout.mode === "regions") {
    const allVerts = layout.regions.flatMap((r) =>
      r.pieces.flatMap((p) => pieceVerts(p)),
    );
    const ctx = buildContext([allVerts, containerBoundsFor(layout.container)]);
    return {
      shapeMode: "region",
      shapeType,
      polygons: [],
      circles: [],
      ellipses: [],
      squares: [],
      rectangles: [],
      regions: normRegions(layout.regions, ctx),
      setAnchors: normSetAnchors(layout.setAnchors, ctx),
      container: normContainer(layout.container, ctx),
      complement,
      metrics,
    };
  }

  if (layout.mode !== "polygons") {
    throw new Error(`unexpected layout mode: ${layout.mode}`);
  }

  const polyVerts = layout.polygons.map((p) => p.vertices);
  const ctx = buildContext([
    ...polyVerts,
    containerBoundsFor(layout.container),
  ]);

  const normPolygons = layout.polygons.map((p) => ({
    label: p.label,
    vertices: p.vertices.map((v) => normPoint(v, ctx)),
  }));

  const result: FitResult = {
    shapeMode: "outline",
    shapeType,
    polygons: normPolygons,
    circles: [],
    ellipses: [],
    squares: [],
    rectangles: [],
    regions: [],
    setAnchors: {},
    container: normContainer(layout.container, ctx),
    complement,
    metrics,
  };

  if (layout.shape === "circle") {
    result.circles = layout.circles.map((c) => ({
      label: c.label,
      x: (c.x - ctx.minX) * ctx.scale,
      y: (c.y - ctx.minY) * ctx.scale,
      radius: c.radius * ctx.scale,
      labelX: (c.labelAnchor.x - ctx.minX) * ctx.scale,
      labelY: (c.labelAnchor.y - ctx.minY) * ctx.scale,
    }));
  } else if (layout.shape === "ellipse") {
    result.ellipses = layout.ellipses.map((e) => ({
      label: e.label,
      x: (e.x - ctx.minX) * ctx.scale,
      y: (e.y - ctx.minY) * ctx.scale,
      semi_major: e.semiMajor * ctx.scale,
      semi_minor: e.semiMinor * ctx.scale,
      rotation: e.rotation,
      labelX: (e.labelAnchor.x - ctx.minX) * ctx.scale,
      labelY: (e.labelAnchor.y - ctx.minY) * ctx.scale,
    }));
  } else if (layout.shape === "square") {
    result.squares = layout.squares.map((s) => ({
      label: s.label,
      x: (s.x - ctx.minX) * ctx.scale,
      y: (s.y - ctx.minY) * ctx.scale,
      side: s.side * ctx.scale,
      labelX: (s.labelAnchor.x - ctx.minX) * ctx.scale,
      labelY: (s.labelAnchor.y - ctx.minY) * ctx.scale,
    }));
  } else if (layout.shape === "rectangle") {
    result.rectangles = layout.rectangles.map((r) => ({
      label: r.label,
      x: (r.x - ctx.minX) * ctx.scale,
      y: (r.y - ctx.minY) * ctx.scale,
      width: r.width * ctx.scale,
      height: r.height * ctx.scale,
      labelX: (r.labelAnchor.x - ctx.minX) * ctx.scale,
      labelY: (r.labelAnchor.y - ctx.minY) * ctx.scale,
    }));
  }

  return result;
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
