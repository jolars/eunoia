// Headless SVG serializer for eunoia layouts.
//
// This module is the renderer-agnostic glue that turns a `Layout` (from
// `euler()` / `venn()`) into an SVG *string*. It is deliberately:
//
// - **wasm-free** — it imports only *types* from the main entry, so importing
//   `@jolars/eunoia/svg` never instantiates the WebAssembly module. Geometry-
//   only and server-side (SSR / static export / resvg / satori) consumers pay
//   no wasm cost.
// - **dependency-free & DOM-free** — pure string assembly. No `document`, no
//   canvas, no measurement. Anything that needs the browser (measuring label
//   text, running `placeLabelsForRegions`) is supplied by the caller.
//
// Coordinates are emitted in the layout's own coordinate space; the `viewBox`
// adapts to the geometry, so absolute scale is irrelevant for display. Style
// sizes (`labelSize`, `strokeWidth`, `padding`) are therefore in **layout
// units** — if you need fixed-pixel text, normalise the layout to a known
// canvas size first (the web app scales to a ~100-unit canvas before calling
// in). When omitted, sizes default to fractions of the bounding-box diagonal
// so a zero-config `toSvg(layout)` still looks reasonable.

import type {
  Circle,
  Container,
  Ellipse,
  LabelPlacement,
  LabelSize,
  Layout,
  Point,
  Polygon,
  Rect,
  Rectangle,
  Region,
  Square,
} from "./index.js";

// ============================================================================
// Palettes (mirrors the web app's `$lib/colors`; the single source of truth now
// lives here so the app and external consumers can't drift).
// ============================================================================

/** A named, ordered list of fill colors (hex) for set diagrams. */
export interface Palette {
  id: string;
  name: string;
  colors: string[];
}

/**
 * Selectable fill palettes. All qualitative/categorical (the right kind for
 * nominal sets) and chosen to read well under the diagrams' alpha blending.
 * Slot 0 is the default/brand palette (eulerr's default 12-color set).
 */
export const PALETTES: Palette[] = [
  {
    id: "default",
    name: "Default",
    colors: [
      "#ffffff",
      "#d9d9d9",
      "#add8e6",
      "#f08080",
      "#fffacd",
      "#cd96cd",
      "#76eec6",
      "#8c8c8c",
      "#5cacee",
      "#ffa07a",
      "#ffb6c1",
      "#eedd82",
    ],
  },
  {
    id: "okabe-ito",
    name: "Okabe–Ito",
    colors: [
      "#999999",
      "#e69f00",
      "#56b4e9",
      "#009e73",
      "#f0e442",
      "#0072b2",
      "#d55e00",
      "#cc79a7",
    ],
  },
  {
    id: "set2",
    name: "Set2",
    colors: [
      "#66c2a5",
      "#fc8d62",
      "#8da0cb",
      "#e78ac3",
      "#a6d854",
      "#ffd92f",
      "#e5c494",
      "#b3b3b3",
    ],
  },
  {
    id: "dark2",
    name: "Dark2",
    colors: [
      "#1b9e77",
      "#d95f02",
      "#7570b3",
      "#e7298a",
      "#66a61e",
      "#e6ab02",
      "#a6761d",
      "#666666",
    ],
  },
  {
    id: "pastel1",
    name: "Pastel1",
    colors: [
      "#fbb4ae",
      "#b3cde3",
      "#ccebc5",
      "#decbe4",
      "#fed9a6",
      "#ffffcc",
      "#e5d8bd",
      "#fddaec",
      "#f2f2f2",
    ],
  },
  {
    id: "tableau10",
    name: "Tableau 10",
    colors: [
      "#4e79a7",
      "#f28e2b",
      "#e15759",
      "#76b7b2",
      "#59a14f",
      "#edc948",
      "#b07aa1",
      "#ff9da7",
      "#9c755f",
      "#bab0ac",
    ],
  },
];

/** Id of the default palette (slot 0). */
export const DEFAULT_PALETTE_ID = PALETTES[0].id;

const PALETTE_MAP = new Map(PALETTES.map((p) => [p.id, p]));

/** The default palette's colors. */
export const DEFAULT_PALETTE = PALETTES[0].colors;

/** Colors for a palette id, falling back to the default palette. */
export function paletteColors(paletteId: string): string[] {
  return (PALETTE_MAP.get(paletteId) ?? PALETTES[0]).colors;
}

/** Color for set index `i` from the given palette (wraps on overflow). */
export function defaultColorFor(
  i: number,
  paletteId: string = DEFAULT_PALETTE_ID,
): string {
  const colors = paletteColors(paletteId);
  return colors[i % colors.length];
}

/** Resolve a set's fill: an explicit override wins, else the palette color. */
export function colorForSet(
  setName: string,
  index: number,
  overrides: Record<string, string>,
  paletteId: string = DEFAULT_PALETTE_ID,
): string {
  return overrides[setName] || defaultColorFor(index, paletteId);
}

/**
 * Average a list of CSS colors in sRGB and return an `rgb(...)` string.
 *
 * Pure (no canvas) reimplementation of the web app's intersection-color
 * blend. Parses `#rgb`, `#rrggbb`, and `rgb()/rgba()` forms (palettes are
 * hex). Unparseable inputs are skipped; if nothing parses, the first color is
 * returned unchanged.
 */
export function mixColors(colors: string[]): string {
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  for (const c of colors) {
    const rgb = parseColor(c);
    if (rgb) {
      r += rgb[0];
      g += rgb[1];
      b += rgb[2];
      n++;
    }
  }
  if (n === 0) return colors[0];
  return `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`;
}

/**
 * Blend a color toward `target` by `amount` (`0` = unchanged, `1` = the
 * target). Unparseable colors are returned as-is.
 */
function blendToward(color: string, target: number, amount: number): string {
  const rgb = parseColor(color);
  if (!rgb) return color;
  const t = Math.min(Math.max(amount, 0), 1);
  const [r, g, b] = rgb.map((v) => Math.round(v + (target - v) * t));
  return `rgb(${r},${g},${b})`;
}

/** Lighten by a positive `amount`, darken by a negative one. */
function tintColor(color: string, amount: number): string {
  return amount < 0
    ? blendToward(color, 0, -amount)
    : blendToward(color, 255, amount);
}

/** Blend a color toward black by `amount`. */
function darken(color: string, amount: number): string {
  return blendToward(color, 0, amount);
}

/**
 * Glyphs in the complement region belong to no set, so they have no region
 * color to borrow; this neutral matches the container frame's family.
 */
const COMPLEMENT_GLYPH_COLOR = "#6b7280";

/** How far a glyph's default edge is darkened from its region's color. */
const GLYPH_EDGE_SHADE = 0.4;

function parseColor(c: string): [number, number, number] | null {
  const s = c.trim();
  const rgbMatch = s.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (rgbMatch) {
    return [+rgbMatch[1], +rgbMatch[2], +rgbMatch[3]];
  }
  if (s.startsWith("#")) {
    const hex = s.slice(1);
    const full =
      hex.length === 3
        ? hex
            .split("")
            .map((ch) => ch + ch)
            .join("")
        : hex;
    if (full.length >= 6) {
      const r = parseInt(full.slice(0, 2), 16);
      const g = parseInt(full.slice(2, 4), 16);
      const b = parseInt(full.slice(4, 6), 16);
      if (!Number.isNaN(r) && !Number.isNaN(g) && !Number.isNaN(b)) {
        return [r, g, b];
      }
    }
  }
  return null;
}

// ============================================================================
// Path builders
// ============================================================================

/** SVG `d` string for a single closed polygon ring. */
export function polygonPath(poly: { vertices: ReadonlyArray<Point> }): string {
  const v = poly.vertices;
  if (v.length === 0) return "";
  let d = `M ${v[0].x},${v[0].y}`;
  for (let i = 1; i < v.length; i++) {
    d += ` L ${v[i].x},${v[i].y}`;
  }
  return `${d} Z`;
}

/**
 * One SVG `d` string for a region piece — its outer ring plus any hole rings
 * concatenated. The core normalises orientations (CCW outer, CW holes), so the
 * SVG default `fill-rule: nonzero` fills only the donut/cookie shape correctly.
 */
export function regionPath(piece: {
  outer: { vertices: ReadonlyArray<Point> };
  holes: ReadonlyArray<{ vertices: ReadonlyArray<Point> }>;
}): string {
  let d = polygonPath(piece.outer);
  for (const h of piece.holes) {
    const hp = polygonPath(h);
    if (hp) d += ` ${hp}`;
  }
  return d;
}

/**
 * SVG `d` string for an exterior-label leader: a polyline
 * `tether → waypoints… → end`. Straight leaders pass no waypoints and yield
 * `M tether L end`.
 */
export function leaderPath(
  tether: Point,
  end: Point,
  waypoints: ReadonlyArray<Point> = [],
): string {
  let d = `M ${tether.x},${tether.y}`;
  for (const w of waypoints) {
    d += ` L ${w.x},${w.y}`;
  }
  return `${d} L ${end.x},${end.y}`;
}

// ============================================================================
// Options
// ============================================================================

export type LegendPosition = "right" | "left" | "top" | "bottom";

export interface LegendOptions {
  /** Draw a legend. Default `false`. */
  show?: boolean;
  /** Where to place it relative to the diagram. Default `"right"`. */
  position?: LegendPosition;
}

/**
 * Descriptor passed to the interactivity hooks ({@link ToSvgOptions.tooltip},
 * {@link ToSvgOptions.regionAttrs}) for one drawn region or set shape.
 */
export interface RegionInfo {
  /**
   * Canonical combination string, e.g. `"A"`, `"A&B"`, or `""` for the
   * complement region. For a shape-mode fill this is the set's own name.
   */
  combination: string;
  /**
   * The individual sets in the combination — a single-element list for a
   * shape-mode fill, empty for the complement.
   */
  sets: string[];
  /**
   * Net region area (`region.totalArea`) or, in shape mode, the set's fitted
   * area (`0` when no fit metrics are attached).
   */
  area: number;
}

export interface ToSvgOptions {
  /** Base fill palette id (see {@link PALETTES}). Default `"default"`. */
  palette?: string;
  /** Per-set fill overrides keyed by set name. Missing sets use the palette. */
  colors?: Record<string, string>;
  /**
   * Stable set order used for palette indexing, so a set keeps its color
   * across reseeds. Defaults to the order sets first appear in the layout.
   */
  setOrder?: string[];
  /** Fill opacity for shapes/regions. Default `1`. */
  alpha?: number;
  /** Stroke width in layout units. `0` hides strokes. Default proportional. */
  strokeWidth?: number;
  /** Stroke color. Default `"black"`. */
  strokeColor?: string;
  /** Label font size in layout units. Default proportional to the bbox. */
  labelSize?: number;
  /** CSS font-family applied to the root `<svg>` (only by {@link toSvg}). */
  fontFamily?: string;
  /** Font weight for labels. Default `400`. */
  fontWeight?: number | string;
  /** Font style for labels (e.g. `"italic"`). Default `"normal"`. */
  fontStyle?: string;
  /** Color for set/region title labels. Default `"black"`. */
  labelColor?: string;
  /** Draw set/region labels at all. Default `true`. */
  showLabels?: boolean;
  /** Draw region/set quantity counts. Default `false`. */
  showCounts?: boolean;
  /**
   * Quantities to draw for {@link ToSvgOptions.showCounts}, keyed by canonical
   * combination (`"A"`, `"A&B"`). Omit to use the layout's own numbers — the
   * region area in `"regions"` mode, the fitted area in `"polygons"` mode.
   *
   * Supplying this makes it *authoritative*: a region with no entry draws no
   * count. That is the point for a Venn layout, whose geometry is topological
   * rather than area-proportional, so its areas are not quantities at all and
   * only the regions you have numbers for should show one.
   */
  counts?: Record<string, number>;
  /** Count color. Default `"#374151"`. */
  countColor?: string;
  /** Formatter for count values. Default a fixed-precision helper. */
  formatCount?: (value: number) => string;
  /** Legend configuration. Off by default. */
  legend?: LegendOptions;
  /** Padding around the diagram in layout units. Default proportional. */
  padding?: number;
  /**
   * Pre-computed label placements keyed by canonical combination (from
   * `placeLabelsForRegions`). When present, region labels use these anchors and
   * exterior placements draw leader lines. Omit for interior-only labels at the
   * layout's built-in anchors.
   */
  placements?: Record<string, LabelPlacement>;
  /**
   * Measured label sizes keyed by combination, used to expand the `viewBox`
   * around exterior labels. Pair with {@link ToSvgOptions.placements}.
   */
  labelSizes?: Record<string, LabelSize>;
  /**
   * Pre-computed **set**-label placements keyed by set name (from
   * `placeSetLabels`): each set's name drawn just outside its own shape, with
   * no leader line.
   *
   * Supplying this changes where set names live. Normally a set name is drawn
   * inside the diagram — at its shape's anchor in shape/polygon mode, or as a
   * title line in its region's label box in region mode. With these
   * placements, names are drawn at the given anchors instead and the interior
   * copies are suppressed, so nothing is duplicated. Region counts stay where
   * they were, moving up to fill the space the title line vacated.
   */
  setLabelPlacements?: Record<string, LabelPlacement>;
  /**
   * Measured set-label sizes keyed by set name, used to expand the `viewBox`
   * around the exterior set labels. Pair with
   * {@link ToSvgOptions.setLabelPlacements}.
   */
  setLabelSizes?: Record<string, LabelSize>;
  /**
   * Pre-computed glyph placements (from `placeGlyphsForRegions`): one
   * equally-sized circle per data unit, drawn inside its region — above the
   * region fills and strokes, below labels. Only rendered for region-mode
   * layouts (`euler({ output: "regions" })` / `venn(...)`), which is also
   * where the positions come from. Glyphs are strictly interior, so the
   * `viewBox` needs no expansion.
   */
  glyphs?: {
    /** Glyph radius in layout units (`GlyphPlacements.radius`). */
    radius: number;
    /** Glyph centers per combination (`GlyphPlacements.positions`). */
    positions: Record<string, Point[]>;
    /**
     * Fill color for every glyph. Default: the region's own color shifted
     * by `tint`, so glyphs read as belonging to the region they sit in.
     */
    fill?: string;
    /**
     * How far the derived fill is shifted from the region color: positive
     * lightens toward white, negative darkens toward black, `0` keeps the
     * region color. Ignored when `fill` is set. Default `0.45`. Palettes of
     * light colors (the default palette opens on white) give airy marks
     * read mostly by their edge; dial in a negative value for dots that sit
     * darker than the region instead.
     */
    tint?: number;
    /**
     * Outline color. Default: the region's own color darkened, which stays
     * visible even where the region color is very light (the default
     * palette opens on white). Set to `"none"` for flat, edgeless dots.
     */
    stroke?: string;
    /**
     * Outline width in layout units. Default: half
     * {@link ToSvgOptions.strokeWidth}, so glyph edges stay visibly finer
     * than the set outlines.
     */
    strokeWidth?: number;
    /** Fill opacity. Default `1`. */
    opacity?: number;
    /** `class` attribute on each circle. Default `"eunoia-glyph"`. */
    className?: string;
  };
  /**
   * Pre-computed member-label boxes (from `placeGlyphBoxesForRegions`): one
   * text box per item, drawn inside its region in the same slot as
   * {@link ToSvgOptions.glyphs} — above the region fills and strokes, below
   * region labels. Both may be set at once; discs are drawn first.
   *
   * The placer deliberately returns no text, so pass the strings separately
   * via `labels`. Boxes are strictly interior, so the `viewBox` needs no
   * expansion.
   */
  glyphBoxes?: {
    /** Diagram-wide box scale (`GlyphBoxPlacements.scale`). */
    scale: number;
    /** Placed boxes per combination (`GlyphBoxPlacements.boxes`). */
    boxes: Record<string, Rect[]>;
    /**
     * Member strings per combination, **index-aligned with `boxes`** — the
     * placer returns a prefix of what you measured, so `labels[combo][i]`
     * belongs to `boxes[combo][i]`. Omit to emit background chips only.
     */
    labels?: Record<string, ReadonlyArray<string>>;
    /**
     * The reference font size the boxes were measured at; the rendered size
     * is `fontSize * scale`. Default {@link ToSvgOptions.labelSize}.
     */
    fontSize?: number;
    /** Text color. Default {@link ToSvgOptions.labelColor}. */
    color?: string;
    /** Text weight. Default {@link ToSvgOptions.fontWeight}. */
    fontWeight?: string | number;
    /** Text family. Default: inherited from the root `<svg>`. */
    fontFamily?: string;
    /**
     * Background chip behind each label. Off by default — member text over
     * the (alpha-blended, light) default palette reads fine, and chips add
     * clutter. Turn it on for saturated palettes. `true` takes the
     * defaults: the region's own color shifted by `tint` (as glyph discs
     * are), corners rounded at a quarter of the box's short side.
     */
    background?:
      | boolean
      | { fill?: string; tint?: number; rx?: number; opacity?: number };
    /** `class` attribute on each `<text>`. Default `"eunoia-glyph-label"`. */
    className?: string;
  };
  /** Leader-line color. Default `"#6b7280"`. */
  leaderColor?: string;
  /** Container (universe) frame stroke color. Default `"#9ca3af"`. */
  containerStroke?: string;
  /**
   * Complement value (items outside every set). When set together with
   * `showCounts`, it's drawn in the container's corner. The container frame is
   * drawn whenever `layout.container` is present regardless of this.
   */
  complement?: number;
  /** Complement count color. Default `"#6b7280"`. */
  complementColor?: string;
  /** Legend label for the complement/container region. Default `"Complement"`. */
  complementLabel?: string;
  /**
   * Emit stable identifying attributes — `data-combination` and `data-area` —
   * on every region/shape fill element. Default `false`. Turn on to give a host
   * (the web app, a notebook, an htmlwidget) a delegation hook for hover/click
   * without baking any payload into the SVG. Combines with
   * {@link ToSvgOptions.regionAttrs} (explicit attrs win on key collision).
   */
  interactive?: boolean;
  /**
   * Per-region/shape tooltip text. When it returns a non-empty string, a
   * `<title>` child is added to the fill element, giving native (zero-JS)
   * browser hover tooltips. Return `null`, `undefined`, or `""` to skip.
   *
   * Keep large payloads (e.g. long member lists) out of here — bake only what
   * you want in the file. For big lists, prefer {@link ToSvgOptions.interactive}
   * plus a host-side lookup keyed on `data-combination`.
   */
  tooltip?: (info: RegionInfo) => string | null | undefined;
  /**
   * Extra attributes for each region/shape fill element (typically `data-*`).
   * Values that are `null`/`undefined` are skipped; others are stringified and
   * XML-escaped. Runs on top of {@link ToSvgOptions.interactive}'s defaults and
   * can override them.
   */
  regionAttrs?: (
    info: RegionInfo,
  ) => Record<string, string | number | null | undefined>;
}

export interface BoundsOptions {
  placements?: Record<string, LabelPlacement>;
  labelSizes?: Record<string, LabelSize>;
  /** Set-label placements from `placeSetLabels`, keyed by set name. */
  setLabelPlacements?: Record<string, LabelPlacement>;
  /** Measured set-label sizes, keyed by set name. */
  setLabelSizes?: Record<string, LabelSize>;
}

// ============================================================================
// Layout introspection helpers
// ============================================================================

interface ShapeArrays {
  polygons: Polygon[];
  circles: Circle[];
  ellipses: Ellipse[];
  squares: Square[];
  rectangles: Rectangle[];
}

function shapeArrays(layout: Layout): ShapeArrays {
  const l = layout as unknown as Partial<ShapeArrays>;
  return {
    polygons: l.polygons ?? [],
    circles: l.circles ?? [],
    ellipses: l.ellipses ?? [],
    squares: l.squares ?? [],
    rectangles: l.rectangles ?? [],
  };
}

function isRegions(
  layout: Layout,
): layout is Extract<Layout, { mode: "regions" }> {
  return layout.mode === "regions";
}

function setsOf(combination: string): string[] {
  return combination
    .split("&")
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Sets that appear anywhere in the layout, in first-seen order. */
function setLabelsOf(layout: Layout): string[] {
  if (isRegions(layout)) {
    const seen = new Set<string>();
    for (const r of layout.regions) {
      for (const s of setsOf(r.combination)) seen.add(s);
    }
    return Array.from(seen);
  }
  const a = shapeArrays(layout);
  if (a.circles.length) return a.circles.map((c) => c.label);
  if (a.ellipses.length) return a.ellipses.map((e) => e.label);
  if (a.squares.length) return a.squares.map((s) => s.label);
  if (a.rectangles.length) return a.rectangles.map((r) => r.label);
  return a.polygons.map((p) => p.label);
}

/**
 * Sets with no exclusive single-set region get their name folded into the
 * largest region that contains them (mirroring the core's nested-set anchor
 * fallback). Returns combination → ordered list of nested set names.
 */
export function nestedSets(layout: Layout): Record<string, string[]> {
  const map: Record<string, string[]> = {};
  if (!isRegions(layout)) return map;

  // Prefer the core's authoritative mapping (`PlotData::set_anchor_regions`):
  // it already records which region each set label was anchored to, so we fold
  // a set into a *multi-set* region exactly when the core did — no need to
  // re-derive the "largest containing region" fallback by re-scanning areas.
  // A set anchored to its own single-set region is titled directly by
  // `regionTitleLines`, so those are skipped here.
  const fromCore = layout.setAnchorRegions;
  if (fromCore && Object.keys(fromCore).length > 0) {
    for (const name of setLabelsOf(layout)) {
      const combo = fromCore[name];
      if (!combo?.includes("&")) continue;
      if (!map[combo]) map[combo] = [];
      map[combo].push(name);
    }
    return map;
  }

  // Fallback for layouts assembled without `setAnchorRegions` (e.g. hand-built
  // region inputs): re-derive the largest containing region from piece areas.
  const hasExclusive = new Set<string>();
  for (const r of layout.regions) {
    if (!r.combination.includes("&")) hasExclusive.add(r.combination.trim());
  }
  for (const name of setLabelsOf(layout)) {
    if (hasExclusive.has(name)) continue;
    let best: { combo: string; area: number } | null = null;
    for (const r of layout.regions) {
      if (!setsOf(r.combination).includes(name)) continue;
      if (!best || r.totalArea > best.area) {
        best = { combo: r.combination, area: r.totalArea };
      }
    }
    if (best) {
      if (!map[best.combo]) map[best.combo] = [];
      map[best.combo].push(name);
    }
  }
  return map;
}

/**
 * Title lines (set names) shown inside a region's label box: the set's own
 * name for an exclusive single-set region, else any sets nested into it.
 */
export function regionTitleLines(
  combination: string,
  nested: Record<string, string[]>,
): string[] {
  if (!combination.includes("&")) return [combination];
  return nested[combination] ?? [];
}

function defaultFormatCount(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// ============================================================================
// Bounds / viewBox
// ============================================================================

interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function ellipseExtent(e: Ellipse): { dx: number; dy: number } {
  const cos = Math.cos(e.rotation);
  const sin = Math.sin(e.rotation);
  const dx = Math.sqrt(
    e.semiMajor * e.semiMajor * cos * cos +
      e.semiMinor * e.semiMinor * sin * sin,
  );
  const dy = Math.sqrt(
    e.semiMajor * e.semiMajor * sin * sin +
      e.semiMinor * e.semiMinor * cos * cos,
  );
  return { dx, dy };
}

/**
 * Geometry bounding box of a layout — union of all shape/region extents and
 * the container, optionally extended to cover exterior label boxes when
 * `placements` + `labelSizes` are supplied.
 */
export function boundingBox(layout: Layout, opts: BoundsOptions = {}): Bounds {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const consume = (x: number, y: number) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };

  if (isRegions(layout)) {
    for (const r of layout.regions) {
      for (const piece of r.pieces) {
        for (const v of piece.outer.vertices) consume(v.x, v.y);
        for (const h of piece.holes) {
          for (const v of h.vertices) consume(v.x, v.y);
        }
      }
    }
  } else {
    const a = shapeArrays(layout);
    for (const p of a.polygons) for (const v of p.vertices) consume(v.x, v.y);
    for (const c of a.circles) {
      consume(c.x - c.radius, c.y - c.radius);
      consume(c.x + c.radius, c.y + c.radius);
    }
    for (const e of a.ellipses) {
      const { dx, dy } = ellipseExtent(e);
      consume(e.x - dx, e.y - dy);
      consume(e.x + dx, e.y + dy);
    }
    for (const s of a.squares) {
      const h = s.side / 2;
      consume(s.x - h, s.y - h);
      consume(s.x + h, s.y + h);
    }
    for (const r of a.rectangles) {
      const hw = r.width / 2;
      const hh = r.height / 2;
      consume(r.x - hw, r.y - hh);
      consume(r.x + hw, r.y + hh);
    }
  }

  const container = layout.container as Container | undefined;
  if (container) {
    consume(
      container.x - container.width / 2,
      container.y - container.height / 2,
    );
    consume(
      container.x + container.width / 2,
      container.y + container.height / 2,
    );
  }

  const labelBox = placementsBounds(opts.placements, opts.labelSizes);
  if (labelBox) {
    consume(labelBox.minX, labelBox.minY);
    consume(labelBox.maxX, labelBox.maxY);
  }

  const setLabelBox = placementsBounds(
    opts.setLabelPlacements,
    opts.setLabelSizes,
  );
  if (setLabelBox) {
    consume(setLabelBox.minX, setLabelBox.minY);
    consume(setLabelBox.maxX, setLabelBox.maxY);
  }

  if (!Number.isFinite(minX)) {
    return { minX: 0, minY: 0, maxX: 100, maxY: 100 };
  }
  return { minX, minY, maxX, maxY };
}

/** Union AABB of every placed label box; `null` when nothing contributes. */
function placementsBounds(
  placements: Record<string, LabelPlacement> | undefined,
  sizes: Record<string, LabelSize> | undefined,
): Bounds | null {
  if (!placements || !sizes) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [combo, p] of Object.entries(placements)) {
    const size = sizes[combo];
    if (!p?.anchor || !size) continue;
    const hw = size.w / 2;
    const hh = size.h / 2;
    if (p.anchor.x - hw < minX) minX = p.anchor.x - hw;
    if (p.anchor.y - hh < minY) minY = p.anchor.y - hh;
    if (p.anchor.x + hw > maxX) maxX = p.anchor.x + hw;
    if (p.anchor.y + hh > maxY) maxY = p.anchor.y + hh;
  }
  if (!Number.isFinite(minX)) return null;
  return { minX, minY, maxX, maxY };
}

interface ViewBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * The `viewBox` for a layout: its bounding box, padded, with room reserved for
 * the legend when enabled. Use this to set the attribute on your own `<svg>`
 * when composing with {@link svgBody}.
 */
export function viewBox(layout: Layout, opts: ToSvgOptions = {}): ViewBox {
  const o = resolve(layout, opts);
  const b = o.bounds;
  let lx = b.minX - o.padding;
  let ly = b.minY - o.padding;
  let lw = b.maxX - b.minX + 2 * o.padding;
  let lh = b.maxY - b.minY + 2 * o.padding;

  if (o.legendShow && o.legendLabels.length > 0) {
    const legendW = Math.max(20, o.labelSize * 2);
    const legendH = Math.max(8, o.labelSize * 1.4) * o.legendLabels.length + 8;
    switch (o.legendPosition) {
      case "right":
        lw += legendW;
        break;
      case "left":
        lw += legendW;
        lx -= legendW;
        break;
      case "top":
        lh += legendH;
        ly -= legendH;
        break;
      case "bottom":
        lh += legendH;
        break;
    }
  }
  return { x: lx, y: ly, w: lw, h: lh };
}

// ============================================================================
// Resolved options (defaults + derived state shared by viewBox/svgBody)
// ============================================================================

interface Resolved {
  bounds: Bounds;
  diag: number;
  padding: number;
  labelSize: number;
  strokeWidth: number;
  strokeColor: string;
  showStroke: boolean;
  alpha: number;
  palette: string;
  colors: Record<string, string>;
  fontWeight: number | string;
  fontStyle: string;
  labelColor: string;
  countColor: string;
  leaderColor: string;
  containerStroke: string;
  complementColor: string;
  complementLabel: string;
  hasContainer: boolean;
  showLabels: boolean;
  showCounts: boolean;
  counts?: Record<string, number>;
  formatCount: (v: number) => string;
  legendShow: boolean;
  legendPosition: LegendPosition;
  legendLabels: string[];
  setColor: Map<string, string>;
  nested: Record<string, string[]>;
  placements: Record<string, LabelPlacement>;
  setLabelPlacements?: Record<string, LabelPlacement>;
  glyphs?: ToSvgOptions["glyphs"];
  glyphBoxes?: ToSvgOptions["glyphBoxes"];
  interactive: boolean;
  tooltip?: (info: RegionInfo) => string | null | undefined;
  regionAttrs?: (
    info: RegionInfo,
  ) => Record<string, string | number | null | undefined>;
}

function resolve(layout: Layout, opts: ToSvgOptions): Resolved {
  const bounds = boundingBox(layout, {
    placements: opts.placements,
    labelSizes: opts.labelSizes,
    setLabelPlacements: opts.setLabelPlacements,
    setLabelSizes: opts.setLabelSizes,
  });
  const diag =
    Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) || 100;

  const palette = opts.palette ?? DEFAULT_PALETTE_ID;
  const colors = opts.colors ?? {};
  const setLabels = setLabelsOf(layout);
  const order = opts.setOrder ?? setLabels;
  const indexOf = new Map<string, number>();
  order.forEach((name, i) => {
    indexOf.set(name, i);
  });
  const setColor = new Map<string, string>();
  setLabels.forEach((label, i) => {
    const idx = indexOf.get(label) ?? i;
    setColor.set(label, colors[label] ?? defaultColorFor(idx, palette));
  });

  const present = new Set(setLabels);
  const legendLabels = order.filter((n) => present.has(n));

  const strokeWidth = opts.strokeWidth ?? diag * 0.003;

  return {
    bounds,
    diag,
    padding: opts.padding ?? diag * 0.08,
    labelSize: opts.labelSize ?? diag * 0.045,
    strokeWidth,
    strokeColor: opts.strokeColor ?? "black",
    showStroke: strokeWidth > 0,
    alpha: opts.alpha ?? 1,
    palette,
    colors,
    fontWeight: opts.fontWeight ?? 400,
    fontStyle: opts.fontStyle ?? "normal",
    labelColor: opts.labelColor ?? "black",
    countColor: opts.countColor ?? "#374151",
    leaderColor: opts.leaderColor ?? "#6b7280",
    containerStroke: opts.containerStroke ?? "#9ca3af",
    complementColor: opts.complementColor ?? "#6b7280",
    complementLabel: opts.complementLabel ?? "Complement",
    hasContainer: !!layout.container,
    showLabels: opts.showLabels ?? true,
    showCounts: opts.showCounts ?? false,
    counts: opts.counts,
    formatCount: opts.formatCount ?? defaultFormatCount,
    legendShow: opts.legend?.show ?? false,
    legendPosition: opts.legend?.position ?? "right",
    legendLabels,
    setColor,
    nested: nestedSets(layout),
    placements: opts.placements ?? {},
    setLabelPlacements: opts.setLabelPlacements,
    glyphs: opts.glyphs,
    glyphBoxes: opts.glyphBoxes,
    interactive: opts.interactive ?? false,
    tooltip: opts.tooltip,
    regionAttrs: opts.regionAttrs,
  };
}

/**
 * Build the extra attribute string and optional `<title>` child for a region /
 * shape fill element from the interactivity options. Returns empty strings when
 * no interactivity is configured, so callers keep emitting self-closing tags.
 */
function interactiveMarkup(
  info: RegionInfo,
  o: Resolved,
): { attrs: string; title: string } {
  const map: Record<string, string | number | null | undefined> = {};
  if (o.interactive) {
    map["data-combination"] = info.combination;
    map["data-area"] = info.area;
  }
  if (o.regionAttrs) Object.assign(map, o.regionAttrs(info));
  let attrs = "";
  for (const [k, v] of Object.entries(map)) {
    if (v === undefined || v === null) continue;
    attrs += ` ${k}="${escAttr(String(v))}"`;
  }
  const text = o.tooltip?.(info);
  return { attrs, title: text ? `<title>${escText(text)}</title>` : "" };
}

function regionFill(combination: string, o: Resolved): string {
  const sets = setsOf(combination);
  const fallback = defaultColorFor(0, o.palette);
  if (sets.length === 1) return o.setColor.get(sets[0]) || fallback;
  const cols = sets
    .map((s) => o.setColor.get(s))
    .filter((c): c is string => !!c);
  if (cols.length === 0) return fallback;
  return mixColors(cols);
}

// ============================================================================
// XML escaping
// ============================================================================

function escAttr(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escText(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ============================================================================
// Body assembly
// ============================================================================

/**
 * The inner SVG markup for a layout — everything that goes *inside* a `<svg>`
 * element, with no wrapper. Use this with your own `<svg>` (e.g. to keep a
 * framework `bind:this`/ref) and set its `viewBox` from {@link viewBox}.
 */
export function svgBody(layout: Layout, opts: ToSvgOptions = {}): string {
  const o = resolve(layout, opts);
  const parts: string[] = [];

  const container = layout.container as Container | undefined;
  if (container) renderContainer(parts, container, o, opts.complement);

  if (isRegions(layout)) {
    renderRegions(parts, layout, o);
  } else {
    renderShapes(parts, layout, o);
  }

  // Exterior set names, drawn last so they sit above every fill. Both render
  // paths above suppress their own interior copies when this is set.
  renderSetLabels(parts, o);

  if (o.legendShow && (o.legendLabels.length > 0 || o.hasContainer))
    renderLegend(parts, o);

  return parts.join("\n");
}

/**
 * A complete standalone `<svg>` document string for a layout. Suitable for
 * writing to a file, server-side rendering, or piping to resvg/satori. For
 * embedding in a framework where you want to own the `<svg>` element, use
 * {@link viewBox} + {@link svgBody} instead.
 */
export function toSvg(layout: Layout, opts: ToSvgOptions = {}): string {
  const vb = viewBox(layout, opts);
  const font = opts.fontFamily
    ? ` font-family="${escAttr(opts.fontFamily)}"`
    : "";
  const body = svgBody(layout, opts);
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb.x} ${vb.y} ${vb.w} ${vb.h}"` +
    ` preserveAspectRatio="xMidYMid meet"${font}>\n${body}\n</svg>`
  );
}

function renderContainer(
  parts: string[],
  c: Container,
  o: Resolved,
  complement: number | undefined,
): void {
  const sw = Math.max(o.strokeWidth, 0.5);
  parts.push(
    `<rect x="${c.x - c.width / 2}" y="${c.y - c.height / 2}" width="${c.width}" height="${c.height}" fill="none" stroke="${o.containerStroke}" stroke-width="${sw}" stroke-dasharray="2 2" />`,
  );
  if (complement !== undefined && o.showCounts) {
    const fs = o.labelSize * 0.75;
    parts.push(
      `<text x="${c.x + c.width / 2 - 1.5}" y="${c.y - c.height / 2 + 2}" text-anchor="end" dominant-baseline="hanging" font-size="${fs}" fill="${o.complementColor}">${escText(o.formatCount(complement))}</text>`,
    );
  }
}

function renderRegions(
  parts: string[],
  layout: Extract<Layout, { mode: "regions" }>,
  o: Resolved,
): void {
  // Fills.
  for (const r of layout.regions) {
    const fill = regionFill(r.combination, o);
    const { attrs, title } = interactiveMarkup(
      {
        combination: r.combination,
        sets: setsOf(r.combination),
        area: r.totalArea,
      },
      o,
    );
    for (const piece of r.pieces) {
      const open = `<path d="${regionPath(piece)}" fill="${fill}" fill-opacity="${o.alpha}" stroke="none"`;
      parts.push(
        attrs || title ? `${open}${attrs}>${title}</path>` : `${open} />`,
      );
    }
  }
  // Strokes.
  if (o.showStroke) {
    for (const r of layout.regions) {
      for (const piece of r.pieces) {
        parts.push(
          `<path d="${regionPath(piece)}" fill="none" stroke="${o.strokeColor}" stroke-width="${o.strokeWidth}" />`,
        );
      }
    }
  }
  // Glyphs — above fills and strokes, below labels. Discs first, so a
  // "dot behind, name in front" hybrid composes.
  if (o.glyphs) renderGlyphs(parts, o.glyphs, o);
  if (o.glyphBoxes) renderGlyphBoxes(parts, o.glyphBoxes, o);
  // Labels + leaders.
  for (const r of layout.regions) {
    renderRegionLabel(parts, r, o);
  }
}

function renderGlyphs(
  parts: string[],
  glyphs: NonNullable<ToSvgOptions["glyphs"]>,
  o: Resolved,
): void {
  const opacity = glyphs.opacity ?? 1;
  const cls = glyphs.className ?? "eunoia-glyph";
  const tint = glyphs.tint ?? 0.45;
  const strokeWidth = glyphs.strokeWidth ?? o.strokeWidth / 2;
  const opacityAttr = opacity !== 1 ? ` fill-opacity="${opacity}"` : "";
  for (const [combination, points] of Object.entries(glyphs.positions)) {
    if (points.length === 0) continue;
    // Glyphs take the color of the region they sit in: a tinted fill plus a
    // darker edge, so the marks read as discrete units even where they
    // cluster, without fighting the region for attention.
    const base = combination
      ? regionFill(combination, o)
      : COMPLEMENT_GLYPH_COLOR;
    const fill = glyphs.fill ?? tintColor(base, tint);
    // Darkened, not the region color itself: the region color *is* the
    // background here, and the default palette opens on `#ffffff`, so an
    // undarkened edge would be invisible.
    const stroke = glyphs.stroke ?? darken(base, GLYPH_EDGE_SHADE);
    const strokeAttr =
      stroke === "none" || !(strokeWidth > 0)
        ? ""
        : ` stroke="${stroke}" stroke-width="${strokeWidth}"`;
    parts.push(
      `<g data-glyphs="${escAttr(combination)}" fill="${fill}"${strokeAttr}${opacityAttr}>`,
    );
    for (const p of points) {
      parts.push(
        `<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${glyphs.radius}" />`,
      );
    }
    parts.push("</g>");
  }
}

function renderGlyphBoxes(
  parts: string[],
  glyphBoxes: NonNullable<ToSvgOptions["glyphBoxes"]>,
  o: Resolved,
): void {
  const cls = glyphBoxes.className ?? "eunoia-glyph-label";
  // The boxes were measured at this size and packed at `scale`, so this is
  // exactly the size the text has to render at for the layout to hold.
  const fontSize = (glyphBoxes.fontSize ?? o.labelSize) * glyphBoxes.scale;
  if (!(fontSize > 0)) return;
  const bg =
    glyphBoxes.background === true ? {} : glyphBoxes.background || undefined;
  const family = glyphBoxes.fontFamily
    ? ` font-family="${escAttr(glyphBoxes.fontFamily)}"`
    : "";

  for (const [combination, boxes] of Object.entries(glyphBoxes.boxes)) {
    if (boxes.length === 0) continue;
    const labels = glyphBoxes.labels?.[combination];
    // Chips borrow the region's color exactly as glyph discs do.
    const base = combination
      ? regionFill(combination, o)
      : COMPLEMENT_GLYPH_COLOR;
    // `font-size` rides on the group: one attribute instead of N, and it
    // makes the `scale` contract legible in the output.
    parts.push(
      `<g data-glyph-boxes="${escAttr(combination)}" font-size="${fontSize}" fill="${glyphBoxes.color ?? o.labelColor}" font-weight="${glyphBoxes.fontWeight ?? o.fontWeight}"${family} text-anchor="middle" dominant-baseline="central">`,
    );
    boxes.forEach((box, i) => {
      if (bg) {
        const rx = bg.rx ?? Math.min(box.width, box.height) * 0.25;
        const opacity =
          bg.opacity !== undefined ? ` fill-opacity="${bg.opacity}"` : "";
        parts.push(
          `<rect class="${cls}-bg" x="${box.x - box.width / 2}" y="${box.y - box.height / 2}" width="${box.width}" height="${box.height}" rx="${rx}" fill="${bg.fill ?? tintColor(base, bg.tint ?? 0.45)}"${opacity} />`,
        );
      }
      const text = labels?.[i];
      if (text === undefined) return;
      parts.push(
        `<text class="${cls}" x="${box.x}" y="${box.y}">${escText(text)}</text>`,
      );
    });
    parts.push("</g>");
  }
}

/**
 * Quantity to draw for a combination: the caller's number when `counts` was
 * supplied (and `undefined`, meaning "draw nothing", when that map has no entry
 * for it), otherwise the layout's own area.
 */
function countFor(
  o: Resolved,
  combination: string,
  fallback: number,
): number | undefined {
  if (!o.counts) return fallback;
  return o.counts[combination];
}

function renderRegionLabel(parts: string[], r: Region, o: Resolved): void {
  const placement = o.placements[r.combination];
  const anchor = placement?.anchor ?? r.labelAnchor;
  // Set names live outside the diagram in set-label mode, so the region box
  // keeps only its count — and the count moves up into the space the title
  // line would have taken.
  const titleLines = o.setLabelPlacements
    ? []
    : regionTitleLines(r.combination, o.nested);
  const isExterior =
    placement?.kind !== undefined && placement.kind !== "interior";

  if (isExterior && placement?.tether) {
    const lw = Math.max(o.strokeWidth * 0.5, 0.3);
    parts.push(
      `<path d="${leaderPath(placement.tether, placement.leaderEnd ?? anchor, placement.leaderWaypoints)}" fill="none" stroke="${o.leaderColor}" stroke-width="${lw}" stroke-opacity="0.6" />`,
    );
  }

  if (o.showLabels) {
    titleLines.forEach((title, i) => {
      parts.push(
        `<text x="${anchor.x}" y="${anchor.y + i * o.labelSize}" text-anchor="middle" dominant-baseline="central" font-size="${o.labelSize}" font-weight="${o.fontWeight}" font-style="${o.fontStyle}" fill="${o.labelColor}">${escText(title)}</text>`,
      );
    });
  }
  if (o.showCounts) {
    const count = countFor(o, r.combination, r.totalArea);
    if (count === undefined) return;
    const fs = o.labelSize * 0.75;
    const y = anchor.y + titleLines.length * o.labelSize;
    parts.push(
      `<text x="${anchor.x}" y="${y}" text-anchor="middle" dominant-baseline="central" font-size="${fs}" fill="${o.countColor}">${escText(o.formatCount(count))}</text>`,
    );
  }
}

/**
 * Set names at their exterior placements ({@link ToSvgOptions.setLabelPlacements}).
 * No-op when the option is absent, and no leader lines are drawn — the whole
 * point of the mode is that the label sits close enough to its shape not to
 * need one. Keys are sorted so the emitted markup is stable.
 */
function renderSetLabels(parts: string[], o: Resolved): void {
  if (!o.setLabelPlacements || !o.showLabels) return;
  for (const name of Object.keys(o.setLabelPlacements).sort()) {
    const anchor = o.setLabelPlacements[name]?.anchor;
    if (!anchor) continue;
    parts.push(
      `<text x="${anchor.x}" y="${anchor.y}" text-anchor="middle" dominant-baseline="central" font-size="${o.labelSize}" font-weight="${o.fontWeight}" font-style="${o.fontStyle}" fill="${o.labelColor}">${escText(name)}</text>`,
    );
  }
}

function renderShapes(parts: string[], layout: Layout, o: Resolved): void {
  const a = shapeArrays(layout);

  const fillFor = (label: string, i: number) =>
    o.setColor.get(label) || defaultColorFor(i, o.palette);

  // A single-set shape fill is a region of one set; attach the same
  // tooltip/data-* hooks keyed on the set's own name and fitted area.
  const fittedAreas = layout.metrics?.fittedAreas ?? {};
  const emitFill = (open: string, tag: string, label: string) => {
    const { attrs, title } = interactiveMarkup(
      { combination: label, sets: [label], area: fittedAreas[label] ?? 0 },
      o,
    );
    parts.push(
      attrs || title ? `${open}${attrs}>${title}</${tag}>` : `${open} />`,
    );
  };

  // Fills.
  a.circles.forEach((c, i) => {
    emitFill(
      `<circle cx="${c.x}" cy="${c.y}" r="${c.radius}" fill="${fillFor(c.label, i)}" fill-opacity="${o.alpha}" stroke="none"`,
      "circle",
      c.label,
    );
  });
  a.ellipses.forEach((e, i) => {
    emitFill(
      `<ellipse cx="${e.x}" cy="${e.y}" rx="${e.semiMajor}" ry="${e.semiMinor}" transform="${ellipseTransform(e)}" fill="${fillFor(e.label, i)}" fill-opacity="${o.alpha}" stroke="none"`,
      "ellipse",
      e.label,
    );
  });
  a.squares.forEach((s, i) => {
    emitFill(
      `<rect x="${s.x - s.side / 2}" y="${s.y - s.side / 2}" width="${s.side}" height="${s.side}" fill="${fillFor(s.label, i)}" fill-opacity="${o.alpha}" stroke="none"`,
      "rect",
      s.label,
    );
  });
  a.rectangles.forEach((r, i) => {
    emitFill(
      `<rect x="${r.x - r.width / 2}" y="${r.y - r.height / 2}" width="${r.width}" height="${r.height}" fill="${fillFor(r.label, i)}" fill-opacity="${o.alpha}" stroke="none"`,
      "rect",
      r.label,
    );
  });

  // Strokes.
  if (o.showStroke) {
    for (const c of a.circles) {
      parts.push(
        `<circle cx="${c.x}" cy="${c.y}" r="${c.radius}" fill="none" stroke="${o.strokeColor}" stroke-width="${o.strokeWidth}" />`,
      );
    }
    for (const e of a.ellipses) {
      parts.push(
        `<ellipse cx="${e.x}" cy="${e.y}" rx="${e.semiMajor}" ry="${e.semiMinor}" transform="${ellipseTransform(e)}" fill="none" stroke="${o.strokeColor}" stroke-width="${o.strokeWidth}" />`,
      );
    }
    for (const s of a.squares) {
      parts.push(
        `<rect x="${s.x - s.side / 2}" y="${s.y - s.side / 2}" width="${s.side}" height="${s.side}" fill="none" stroke="${o.strokeColor}" stroke-width="${o.strokeWidth}" />`,
      );
    }
    for (const r of a.rectangles) {
      parts.push(
        `<rect x="${r.x - r.width / 2}" y="${r.y - r.height / 2}" width="${r.width}" height="${r.height}" fill="none" stroke="${o.strokeColor}" stroke-width="${o.strokeWidth}" />`,
      );
    }
  }

  // Set labels. Skipped entirely in set-label mode — `renderSetLabels` draws
  // them at their exterior anchors instead, so drawing them here too would
  // double them up.
  if (o.showLabels && !o.setLabelPlacements) {
    const label = (lx: number, ly: number, text: string) =>
      parts.push(
        `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="central" font-size="${o.labelSize}" font-weight="${o.fontWeight}" font-style="${o.fontStyle}" fill="${o.labelColor}">${escText(text)}</text>`,
      );
    for (const c of a.circles) label(c.labelAnchor.x, c.labelAnchor.y, c.label);
    for (const e of a.ellipses)
      label(e.labelAnchor.x, e.labelAnchor.y, e.label);
    for (const s of a.squares) label(s.labelAnchor.x, s.labelAnchor.y, s.label);
    for (const r of a.rectangles)
      label(r.labelAnchor.x, r.labelAnchor.y, r.label);
  }

  // Per-set counts (single-set combinations only).
  if (o.showCounts) {
    const fitted = layout.metrics?.fittedAreas ?? {};
    const fs = o.labelSize * 0.75;
    const findAnchor = (label: string): Point | null => {
      const c = a.circles.find((s) => s.label === label);
      if (c) return c.labelAnchor;
      const e = a.ellipses.find((s) => s.label === label);
      if (e) return e.labelAnchor;
      const sq = a.squares.find((s) => s.label === label);
      if (sq) return sq.labelAnchor;
      const rc = a.rectangles.find((s) => s.label === label);
      if (rc) return rc.labelAnchor;
      return null;
    };
    // Normally the count sits one line below the set name; in set-label mode
    // the name has moved outside, so the count takes the anchor itself.
    const dy = o.setLabelPlacements ? 0 : o.labelSize;
    // `counts`, when supplied, replaces the fitted areas outright — including
    // which sets get a number at all.
    for (const [combo, value] of Object.entries(o.counts ?? fitted)) {
      if (combo.includes("&")) continue;
      const anchor = findAnchor(combo);
      if (!anchor) continue;
      parts.push(
        `<text x="${anchor.x}" y="${anchor.y + dy}" text-anchor="middle" dominant-baseline="central" font-size="${fs}" fill="${o.countColor}">${escText(o.formatCount(value))}</text>`,
      );
    }
  }
}

function ellipseTransform(e: Ellipse): string {
  return `rotate(${(e.rotation * 180) / Math.PI} ${e.x} ${e.y})`;
}

function renderLegend(parts: string[], o: Resolved): void {
  const b = o.bounds;
  const swatch = o.labelSize * 0.9;
  const gap = swatch * 0.4;
  const lineH = swatch + gap;
  const rows = o.legendLabels.length + (o.hasContainer ? 1 : 0);
  const totalH = lineH * rows;
  const pad = o.padding * 0.5;
  let x = 0;
  let y = 0;
  switch (o.legendPosition) {
    case "right":
      x = b.maxX + pad;
      y = (b.minY + b.maxY) / 2 - totalH / 2;
      break;
    case "left":
      x = b.minX - pad - swatch * 6;
      y = (b.minY + b.maxY) / 2 - totalH / 2;
      break;
    case "top":
      x = (b.minX + b.maxX) / 2 - swatch * 3;
      y = b.minY - pad - totalH;
      break;
    case "bottom":
      x = (b.minX + b.maxX) / 2 - swatch * 3;
      y = b.maxY + pad;
      break;
  }

  const swStroke = Math.max(0.5, o.strokeWidth * 0.75);
  const items: string[] = ["<g>"];
  o.legendLabels.forEach((label, i) => {
    const yi = y + i * lineH;
    const color = o.setColor.get(label) || defaultColorFor(i, o.palette);
    items.push(
      `<rect x="${x}" y="${yi}" width="${swatch}" height="${swatch}" fill="${color}" fill-opacity="${o.alpha}" stroke="${o.showStroke ? o.strokeColor : "none"}" stroke-width="${swStroke}" />`,
    );
    items.push(
      `<text x="${x + swatch + gap}" y="${yi + swatch / 2}" dominant-baseline="central" font-size="${o.labelSize}" font-weight="${o.fontWeight}" font-style="${o.fontStyle}" fill="${o.labelColor}">${escText(label)}</text>`,
    );
  });
  if (o.hasContainer) {
    const yi = y + o.legendLabels.length * lineH;
    items.push(
      `<rect x="${x}" y="${yi}" width="${swatch}" height="${swatch}" fill="none" stroke="${o.containerStroke}" stroke-width="${swStroke}" stroke-dasharray="2 2" />`,
    );
    items.push(
      `<text x="${x + swatch + gap}" y="${yi + swatch / 2}" dominant-baseline="central" font-size="${o.labelSize}" font-weight="${o.fontWeight}" font-style="${o.fontStyle}" fill="${o.labelColor}">${escText(o.complementLabel)}</text>`,
    );
  }
  items.push("</g>");
  parts.push(items.join("\n"));
}

// Re-export the geometric types most consumers will touch, so a caller can
// stay within `@jolars/eunoia/svg` for rendering.
export type {
  Container,
  LabelPlacement,
  LabelSize,
  Layout,
  Point,
  Polygon,
  Region,
  RegionPiece,
  SetOutline,
} from "./index.js";
