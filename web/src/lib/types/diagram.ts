export interface Point {
  x: number;
  y: number;
}

export interface Polygon {
  vertices: Point[];
  label: string;
}

/**
 * One connected component of a region: a CCW outer ring plus zero or more
 * CW hole rings (other regions cutting through this piece). Orientations
 * are normalised by the core library so renderers can use `fill-rule: nonzero`.
 */
export interface RegionPiece {
  outer: Polygon;
  holes: Polygon[];
}

export interface CircleShape {
  x: number;
  y: number;
  radius: number;
  label: string;
  /** Label anchor — pole of inaccessibility of `shape \ ⋃ others`. Defaults to (x, y). */
  labelX: number;
  labelY: number;
}

export interface EllipseShape {
  x: number;
  y: number;
  semi_major: number;
  semi_minor: number;
  rotation: number;
  label: string;
  labelX: number;
  labelY: number;
}

export interface SquareShape {
  x: number;
  y: number;
  side: number;
  label: string;
  labelX: number;
  labelY: number;
}

export interface RectangleShape {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  labelX: number;
  labelY: number;
}

export interface RegionPolygon {
  combination: string;
  pieces: RegionPiece[];
  totalArea: number;
  /** Hole-aware label anchor for this region — one point per region, even when fragmented across multiple pieces. */
  labelX: number;
  labelY: number;
}

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
  shapeMode: "outline" | "region";
  shapeType: "circle" | "ellipse" | "square" | "rectangle";
  polygons: Polygon[];
  circles: CircleShape[];
  ellipses: EllipseShape[];
  squares: SquareShape[];
  rectangles: RectangleShape[];
  regions: RegionPolygon[];
  /** Per-set label anchors keyed by set name. Populated in region mode from `PlotData::set_anchors`; empty in outline mode (use shape `labelX/labelY` instead). */
  setAnchors: Record<string, { x: number; y: number }>;
  /**
   * Universe / container rectangle (x, y are the center; width and height
   * are full extents) when the spec carried a complement. Same coordinate
   * frame as the rest of the layout — already normalized to the ~100-unit
   * canvas. `undefined` when no complement was set.
   */
  container?: { x: number; y: number; width: number; height: number };
  /** Complement value carried alongside the spec (number of items outside every named set). Useful for displaying as a label. */
  complement?: number;
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

export type LabelPlacementMode = "raycast" | "forceDirected";

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
