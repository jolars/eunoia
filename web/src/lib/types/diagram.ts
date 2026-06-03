// The fitted geometry now travels as the high-level package `Layout`
// (`@jolars/eunoia`), so the app no longer redefines per-shape/region structs
// here — see `FitResult.layout`.
import type { Layout } from "@jolars/eunoia";

export interface FitMetrics {
  loss: number;
  stress: number;
  diagError: number;
  iterations: number;
  target: Record<string, number>;
  fitted: Record<string, number>;
  regionError: Record<string, number>;
  residuals: Record<string, number>;
}

export interface FitResult {
  /**
   * The fitted layout in the high-level package shape (`@jolars/eunoia`),
   * already normalized to the ~100-unit canvas so the style knobs
   * (`labelSize`, `strokeWidth`, …) stay calibrated. This is exactly what the
   * `@jolars/eunoia/svg` serializer (`toSvg`/`svgBody`/`viewBox`) consumes, so
   * the app renders through the same surface external consumers use.
   */
  layout: Layout;
  shapeType: "circle" | "ellipse" | "square" | "rectangle";
  /** Complement value carried alongside the spec (items outside every named set). */
  complement?: number;
  /**
   * Fit metrics, kept on the result (rather than read off `layout.metrics`) in
   * the app's renamed `FitMetrics` shape so the metrics/table panels consume it
   * directly.
   */
  metrics: FitMetrics;
}

export type Row = { input: string; size: number };
export type InputType = "exclusive" | "inclusive";
export type ShapeType = "circle" | "ellipse" | "square" | "rectangle";
export type DiagramType = "euler" | "venn";
/** Set count for canonical Venn diagrams. Limited to the range supported by `VennDiagram::new` (1..=5). */
export type VennSetCount = 1 | 2 | 3 | 4 | 5;
export type LegendPosition = "right" | "left" | "top" | "bottom";
export type OptimizerName =
  | "CmaEsLm"
  | "LevenbergMarquardt"
  | "Lbfgs"
  | "NelderMead";
export type LossName =
  | "SumSquared"
  | "SumAbsolute"
  | "SumAbsoluteRegionError"
  | "SumSquaredRegionError"
  | "MaxAbsolute"
  | "MaxSquared"
  | "RootMeanSquared"
  | "Stress"
  | "DiagError";
export type ExportFormat = "svg" | "png" | "pdf" | "json";

export type LabelPlacementMode = "raycast" | "forceDirected" | "elbow";

export interface DiagramStyle {
  /** Base fill palette id (see `lib/colors.ts`). Per-set `colors` override it. */
  palette: string;
  /** Per-set fill colors keyed by set name. Missing sets fall back to the palette. */
  colors: Record<string, string>;
  alpha: number;
  showLegend: boolean;
  legendPosition: LegendPosition;
  fontBold: boolean;
  fontItalic: boolean;
  /**
   * CSS font-family stack for all diagram labels. Set explicitly on the SVG
   * root so the chosen font travels with SVG/PNG/PDF exports — a serialized
   * SVG carries no page CSS to inherit, so without this exports fall back to
   * the SVG default serif. See `lib/fonts.ts` for the available stacks.
   */
  fontFamily: string;
  /** SVG stroke width in user units. 0 hides the border. */
  strokeWidth: number;
  /** Label font size in user units. */
  labelSize: number;
  showCounts: boolean;
  /**
   * Exterior-fallback solver used when a label doesn't fit inside its
   * region:
   *
   * - `"raycast"` (default) — deterministic ray from the diagram centroid
   *   through the region's POI.
   * - `"forceDirected"` — iterative spring + polygon-aware repulsion;
   *   slower than raycast but better for crowded layouts where raycast
   *   labels pile up or land on top of unrelated regions.
   */
  labelPlacement: LabelPlacementMode;
  /**
   * Where the exterior-leader tether attaches on the source region:
   *
   * - `"poi"` (default) — region's pole of inaccessibility (deep inside the
   *   polygon). Safe for any rendering style, including stroke-less fills.
   * - `"boundary"` — point where the outgoing ray exits the polygon's outer
   *   ring. Rendered leader starts on the polygon edge; recommended when
   *   the renderer draws shape strokes.
   */
  labelTether: "poi" | "boundary";
}

export interface RasterSize {
  /** Pixels. */
  width: number;
  height: number;
}

export interface VectorSize {
  /** Inches. */
  width: number;
  height: number;
}

export interface AdvancedOptions {
  optimizer: OptimizerName;
  lossType: LossName;
  showRegions: boolean;
  seed: number | undefined;
  useSeed: boolean;
  /** Final-stage cost-change tolerance. Wired to `Fitter::tolerance`. */
  tolerance: number;
  /**
   * When true, fit a universe-bounding container around the diagram and pass
   * `complement` (items outside every named set) into the fitter. The
   * container is rendered as a separate axis-aligned rectangle.
   */
  useComplement: boolean;
  /**
   * Number of items outside every named set. Only used when `useComplement` is
   * true. `null` when the user has ticked Complement but not yet entered a
   * value — the fit then skips the universe container until a positive number
   * is supplied.
   */
  complement: number | null;
}

export interface ExportSettings {
  format: ExportFormat;
  raster: RasterSize;
  vector: VectorSize;
}

export interface PersistedState {
  rows: Row[];
  inputType: InputType;
  shapeType: ShapeType;
  diagramType?: DiagramType;
  vennN?: VennSetCount;
  style: DiagramStyle;
  advanced: AdvancedOptions;
  exportSettings: ExportSettings;
}
