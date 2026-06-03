// Fill palettes for set diagrams.
//
// The palette data and color helpers now live in the published package
// (`@jolars/eunoia/svg`) so the app, external consumers, and the SVG serializer
// share one source of truth. This module just re-exports them under the
// familiar `$lib/colors` path.

export {
  colorForSet,
  DEFAULT_PALETTE,
  DEFAULT_PALETTE_ID,
  defaultColorFor,
  PALETTES,
  type Palette,
  paletteColors,
} from "@jolars/eunoia/svg";
