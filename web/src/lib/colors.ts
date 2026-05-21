/** A named, ordered list of fill colors (hex) for set diagrams. */
export interface Palette {
  id: string;
  name: string;
  colors: string[];
}

/**
 * Selectable fill palettes. All qualitative/categorical (the right kind for
 * nominal sets) and chosen to read well under the diagrams' alpha blending.
 * Hex form so the per-set color pickers can preselect them. Slot 0 is the
 * default/brand palette.
 */
export const PALETTES: Palette[] = [
  {
    id: "default",
    name: "Default",
    // eulerr's default 12-color palette.
    colors: [
      "#ffffff", // 255,255,255
      "#d9d9d9", // 217,217,217
      "#add8e6", // 173,216,230
      "#f08080", // 240,128,128
      "#fffacd", // 255,250,205
      "#cd96cd", // 205,150,205
      "#76eec6", // 118,238,198
      "#8c8c8c", // 140,140,140
      "#5cacee", // 92,172,238
      "#ffa07a", // 255,160,122
      "#ffb6c1", // 255,182,193
      "#eedd82", // 238,221,130
    ],
  },
  {
    id: "okabe-ito",
    name: "Okabe–Ito",
    // Colorblind-safe qualitative palette (Okabe & Ito), grey-first variant —
    // grey (#999999) substitutes for canonical black, which reads poorly as a
    // fill. Remaining colors keep the canonical order.
    colors: [
      "#999999", // grey
      "#e69f00", // orange
      "#56b4e9", // sky blue
      "#009e73", // bluish green
      "#f0e442", // yellow
      "#0072b2", // blue
      "#d55e00", // vermillion
      "#cc79a7", // reddish purple
    ],
  },
  {
    id: "set2",
    name: "Set2",
    // ColorBrewer Set2 — soft, colorblind-friendly.
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
    // ColorBrewer Dark2 — muted-saturated, good under transparency.
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
    // ColorBrewer Pastel1 — pastels, good for heavy overlap.
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
    // Tableau 10 — familiar modern categorical.
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

/**
 * The default palette's colors. Kept as a named export for callers that don't
 * track a palette selection (e.g. docs examples).
 */
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
