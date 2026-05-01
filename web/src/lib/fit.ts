import type {
  AdvancedOptions,
  DiagramType,
  FitResult,
  InputType,
  Row,
  ShapeType,
  VennSetCount,
} from "../types/diagram";

const POLYGON_VERTICES = 256;

function buildSpecs(wasm: any, rows: Row[]): any[] {
  return rows
    .filter((r) => {
      const input = r.input.trim();
      if (input === "" || r.size <= 0) return false;
      if (input.endsWith("&") || input.endsWith("|")) return false;
      return true;
    })
    .map((r) => new wasm.DiagramSpec(r.input.trim(), r.size));
}

function optimizerEnum(wasm: any, name: AdvancedOptions["optimizer"]): any {
  return wasm.WasmOptimizer[name];
}

function lossEnum(wasm: any, name: AdvancedOptions["lossType"]): any {
  return wasm.WasmLossType[name];
}

function parseRecord(json: string): Record<string, number> {
  if (!json) return {};
  try {
    return JSON.parse(json) as Record<string, number>;
  } catch {
    return {};
  }
}

function normalizeBounds(boundsItems: { x: number; y: number }[][]): {
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

interface NormalizationContext {
  scale: number;
  minX: number;
  minY: number;
  precision: number;
}

function buildContext(boundsItems: { x: number; y: number }[][]): NormalizationContext {
  const { minX, minY, maxX, maxY } = normalizeBounds(boundsItems);
  const maxDim = Math.max(maxX - minX, maxY - minY) || 1;
  const scale = 100 / maxDim;
  return { scale, minX, minY, precision: maxDim * 0.001 };
}

function normPoint(p: { x: number; y: number }, ctx: NormalizationContext) {
  return { x: (p.x - ctx.minX) * ctx.scale, y: (p.y - ctx.minY) * ctx.scale };
}

export interface FitInputs {
  rows: Row[];
  inputType: InputType;
  shapeType: ShapeType;
  diagramType: DiagramType;
  vennN: VennSetCount;
  advanced: AdvancedOptions;
}

// Re-export so the worker can import the type from a single place.
export type {
  Row,
  InputType,
  ShapeType,
  AdvancedOptions,
  DiagramType,
  VennSetCount,
};

function runVennFit(wasm: any, inputs: FitInputs): FitResult {
  const showRegions = inputs.advanced.showRegions;

  if (showRegions) {
    const result = wasm.generate_venn_regions(inputs.vennN, POLYGON_VERTICES);
    const rawRegions = Array.from(result.regions) as any[];
    const allVerts = rawRegions.flatMap((r) =>
      Array.from(r.polygons).flatMap((p: any) =>
        Array.from(p.vertices) as { x: number; y: number }[],
      ),
    );
    const ctx = buildContext([allVerts]);
    const regions = rawRegions.map((r: any) => {
      const polys = Array.from(r.polygons).map((p: any) => {
        const pole = p.pole_of_inaccessibility(ctx.precision);
        return {
          label: r.combination as string,
          vertices: (Array.from(p.vertices) as { x: number; y: number }[]).map(
            (v) => normPoint(v, ctx),
          ),
          labelPosition: normPoint(pole, ctx),
        };
      });
      return {
        combination: r.combination as string,
        totalArea: r.total_area as number,
        polygons: polys,
      };
    });
    return {
      shapeMode: "region",
      shapeType: "ellipse",
      polygons: [],
      circles: [],
      ellipses: [],
      squares: [],
      regions,
      metrics: {
        loss: result.loss,
        stress: result.stress,
        diagError: result.diag_error,
        iterations: result.iterations,
        target: parseRecord(result.target_areas_json),
        fitted: parseRecord(result.fitted_areas_json),
        regionError: parseRecord(result.region_error_json),
        residuals: parseRecord(result.residuals_json),
      },
    };
  }

  const result = wasm.generate_venn_polygons(inputs.vennN, POLYGON_VERTICES);
  const polygons = Array.from(result.polygons) as any[];
  const ellipses = Array.from(result.ellipses) as any[];

  const polyVerts = polygons.map(
    (p) => Array.from(p.vertices) as { x: number; y: number }[],
  );
  const ctx = buildContext(polyVerts);

  const normPolygons = polygons.map((p: any) => {
    const pole = p.pole_of_inaccessibility(ctx.precision);
    return {
      label: p.label as string,
      vertices: (Array.from(p.vertices) as { x: number; y: number }[]).map(
        (v) => normPoint(v, ctx),
      ),
      labelPosition: normPoint(pole, ctx),
    };
  });
  const normEllipses = ellipses.map((e: any) => ({
    label: e.label as string,
    x: (e.x - ctx.minX) * ctx.scale,
    y: (e.y - ctx.minY) * ctx.scale,
    semi_major: e.semi_major * ctx.scale,
    semi_minor: e.semi_minor * ctx.scale,
    rotation: e.rotation,
  }));

  return {
    shapeMode: "outline",
    shapeType: "ellipse",
    polygons: normPolygons,
    circles: [],
    ellipses: normEllipses,
    squares: [],
    regions: [],
    metrics: {
      loss: result.loss,
      stress: result.stress,
      diagError: result.diag_error,
      iterations: result.iterations,
      target: parseRecord(result.target_areas_json),
      fitted: parseRecord(result.fitted_areas_json),
      regionError: parseRecord(result.region_error_json),
      residuals: parseRecord(result.residuals_json),
    },
  };
}

export function runFit(wasm: any, inputs: FitInputs): FitResult | null {
  if (inputs.diagramType === "venn") {
    return runVennFit(wasm, inputs);
  }
  const specs = buildSpecs(wasm, inputs.rows);
  if (specs.length === 0) return null;
  const seed =
    inputs.advanced.useSeed && inputs.advanced.seed !== undefined
      ? BigInt(inputs.advanced.seed)
      : undefined;
  const optimizer = optimizerEnum(wasm, inputs.advanced.optimizer);
  const lossType = lossEnum(wasm, inputs.advanced.lossType);
  const tolerance =
    Number.isFinite(inputs.advanced.tolerance) && inputs.advanced.tolerance > 0
      ? inputs.advanced.tolerance
      : undefined;
  const showRegions = inputs.advanced.showRegions;

  if (showRegions) {
    const fn =
      inputs.shapeType === "circle"
        ? wasm.generate_region_polygons_circles
        : inputs.shapeType === "square"
          ? wasm.generate_region_polygons_squares
          : wasm.generate_region_polygons_ellipses;
    const result = fn(
      specs,
      inputs.inputType,
      POLYGON_VERTICES,
      seed,
      optimizer,
      lossType,
      tolerance,
    );
    const rawRegions = Array.from(result.regions) as any[];
    const allVerts = rawRegions.flatMap((r) =>
      Array.from(r.polygons).flatMap((p: any) =>
        Array.from(p.vertices) as { x: number; y: number }[],
      ),
    );
    const ctx = buildContext([allVerts]);
    const regions = rawRegions.map((r: any) => {
      const polys = Array.from(r.polygons).map((p: any) => {
        const pole = p.pole_of_inaccessibility(ctx.precision);
        return {
          label: r.combination as string,
          vertices: (Array.from(p.vertices) as { x: number; y: number }[]).map(
            (v) => normPoint(v, ctx),
          ),
          labelPosition: normPoint(pole, ctx),
        };
      });
      return {
        combination: r.combination as string,
        totalArea: r.total_area as number,
        polygons: polys,
      };
    });
    return {
      shapeMode: "region",
      shapeType: inputs.shapeType,
      polygons: [],
      circles: [],
      ellipses: [],
      squares: [],
      regions,
      metrics: {
        loss: result.loss,
        stress: result.stress,
        diagError: result.diag_error,
        iterations: result.iterations,
        target: parseRecord(result.target_areas_json),
        fitted: parseRecord(result.fitted_areas_json),
        regionError: parseRecord(result.region_error_json),
        residuals: parseRecord(result.residuals_json),
      },
    };
  }

  const fn =
    inputs.shapeType === "circle"
      ? wasm.generate_circles_as_polygons
      : inputs.shapeType === "square"
        ? wasm.generate_squares_as_polygons
        : wasm.generate_ellipses_as_polygons;
  const result = fn(
    specs,
    inputs.inputType,
    POLYGON_VERTICES,
    seed,
    optimizer,
    lossType,
    tolerance,
  );
  const polygons = Array.from(result.polygons) as any[];
  const circles = Array.from(result.circles) as any[];
  const ellipses = Array.from(result.ellipses) as any[];
  const squares = Array.from(result.squares) as any[];

  const polyVerts = polygons.map(
    (p) => Array.from(p.vertices) as { x: number; y: number }[],
  );
  const ctx = buildContext(polyVerts);

  const normPolygons = polygons.map((p: any) => {
    const pole = p.pole_of_inaccessibility(ctx.precision);
    return {
      label: p.label as string,
      vertices: (Array.from(p.vertices) as { x: number; y: number }[]).map(
        (v) => normPoint(v, ctx),
      ),
      labelPosition: normPoint(pole, ctx),
    };
  });
  const normCircles = circles.map((c: any) => ({
    label: c.label as string,
    x: (c.x - ctx.minX) * ctx.scale,
    y: (c.y - ctx.minY) * ctx.scale,
    radius: c.radius * ctx.scale,
  }));
  const normEllipses = ellipses.map((e: any) => ({
    label: e.label as string,
    x: (e.x - ctx.minX) * ctx.scale,
    y: (e.y - ctx.minY) * ctx.scale,
    semi_major: e.semi_major * ctx.scale,
    semi_minor: e.semi_minor * ctx.scale,
    rotation: e.rotation,
  }));
  const normSquares = squares.map((s: any) => ({
    label: s.label as string,
    x: (s.x - ctx.minX) * ctx.scale,
    y: (s.y - ctx.minY) * ctx.scale,
    side: s.side * ctx.scale,
  }));

  return {
    shapeMode: "outline",
    shapeType: inputs.shapeType,
    polygons: normPolygons,
    circles: normCircles,
    ellipses: normEllipses,
    squares: normSquares,
    regions: [],
    metrics: {
      loss: result.loss,
      stress: result.stress,
      diagError: result.diag_error,
      iterations: result.iterations,
      target: parseRecord(result.target_areas_json),
      fitted: parseRecord(result.fitted_areas_json),
      regionError: parseRecord(result.region_error_json),
      residuals: parseRecord(result.residuals_json),
    },
  };
}
