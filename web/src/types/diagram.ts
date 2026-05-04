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
  shapeType: "circle" | "ellipse" | "square";
  polygons: Polygon[];
  circles: CircleShape[];
  ellipses: EllipseShape[];
  squares: SquareShape[];
  regions: RegionPolygon[];
  /** Per-set label anchors keyed by set name. Populated in region mode from `PlotData::set_anchors`; empty in outline mode (use shape `labelX/labelY` instead). */
  setAnchors: Record<string, { x: number; y: number }>;
  metrics: FitMetrics;
}

export type Row = { input: string; size: number };
export type InputType = "exclusive" | "inclusive";
export type ShapeType = "circle" | "ellipse" | "square";
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
export type TabKey = "app" | "about" | "cite";

export interface DiagramStyle {
  /** Per-set fill colors keyed by set name. Missing sets fall back to the default palette. */
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
