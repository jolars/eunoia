/**
 * Member names: the "which ones" companion to the unit-glyph "how many".
 *
 * The spec rows carry an optional free-text roster per combination; the
 * `"members"` glyph mode measures those strings and hands them to
 * `placeGlyphBoxesForRegions`. This module owns the two bits of glue that
 * neither the core nor the TS wrapper provides: parsing the roster text, and
 * matching a row's *typed* combination to the *canonical* one the core hands
 * back on each region.
 */
import type { Region } from "@jolars/eunoia";

import type { Row } from "./types/diagram";

/** Split a roster field into names. Commas and newlines both separate. */
export function parseMembers(raw: string | undefined): string[] {
  if (!raw) return [];
  return raw
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Order-insensitive key for a combination: `"B & A"` and `"A&B"` both fold to
 * `"A&B"`. Used only to *match* a typed row against a region — never as an
 * output key when geometry exists, since the core picks the canonical
 * spelling. Missing regions retain this folded key for their callout.
 */
export function foldKey(combination: string): string {
  return combination
    .split("&")
    .map((s) => s.trim())
    .filter(Boolean)
    .sort()
    .join("&");
}

/**
 * Member names per region, keyed by the region's own canonical combination.
 *
 * Rows are matched to regions by set membership rather than by their raw text,
 * because `buildSets` passes the typed string through verbatim and the core
 * canonicalizes it. Rows with no names, or with a `|` (union inputs, which name
 * no single region), are skipped — as is the complement region `""`, which has
 * no roster to draw from. Names for missing regions are retained under their
 * folded key so the renderer can report them without inventing geometry.
 */
export function memberListsForRegions(
  rows: readonly Row[],
  regions: readonly Region[],
): Record<string, string[]> {
  const byFold = new Map<string, string>();
  for (const r of regions) byFold.set(foldKey(r.combination), r.combination);

  const out: Record<string, string[]> = {};
  for (const row of rows) {
    const input = row.input.trim();
    if (!input || input.includes("|")) continue;
    const names = parseMembers(row.members);
    if (names.length === 0) continue;
    const combination = byFold.get(foldKey(input)) ?? foldKey(input);
    // Two rows folding to the same region (e.g. `A&B` and `B&A`) concatenate
    // rather than clobber, matching how a user would read them.
    out[combination] = [...(out[combination] ?? []), ...names];
  }
  return out;
}
