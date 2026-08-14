/**
 * Venn-side companion to the Euler spec rows.
 *
 * A Venn diagram is topological: its geometry follows from the set count alone,
 * so there is no size to fit and the app's `rows` editor has nothing to do. The
 * data a Venn *can* carry is per-region annotation — a quantity to print and a
 * roster of member names to pack inside the region — which is what
 * {@link VennRegion} holds and this module keys, orders, and adapts.
 *
 * Keys are the canonical combination strings the core hands back on each region
 * (`"A"`, `"A&B"`, …): set names are the first `n` uppercase letters and
 * `Combination`'s `Display` joins them with `&` in index order, so generating
 * them here matches without a lookup.
 */
import { foldKey, parseMembers } from "./members.svelte";
import type { VennRegion, VennSetCount } from "./types/diagram";

/**
 * Every set count the Venn UI offers, in order. It starts at 2 because a
 * one-set Venn is just a circle, and stops at 5 because no Venn diagram of 6+
 * ellipses exists.
 */
export const VENN_SET_COUNTS: readonly VennSetCount[] = [2, 3, 4, 5];

/** Largest `n` the Venn UI offers; the key space below covers all of `1..=MAX`. */
export const MAX_VENN_SETS = 5;

/** Guard for set counts arriving from storage or a pasted debug blob. */
export function isVennSetCount(n: unknown): n is VennSetCount {
  return VENN_SET_COUNTS.includes(n as VennSetCount);
}

/** Set names for an `n`-set Venn: `["A", "B", …]`, matching `VennDiagram`. */
export function vennSetNames(n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(String.fromCharCode(65 + i));
  return out;
}

/**
 * The `2ⁿ − 1` region keys of an `n`-set Venn, ordered by how many sets they
 * intersect and then by set order — the order the editor lists them in, and the
 * order someone filling in a table works through.
 */
export function vennCombinations(n: number): string[] {
  const names = vennSetNames(n);
  const masks: number[] = [];
  for (let mask = 1; mask < 1 << n; mask++) masks.push(mask);
  masks.sort((a, b) => popcount(a) - popcount(b) || a - b);
  return masks.map((mask) => names.filter((_, i) => mask & (1 << i)).join("&"));
}

function popcount(mask: number): number {
  let n = 0;
  for (let m = mask; m !== 0; m >>= 1) n += m & 1;
  return n;
}

/** An empty annotation record covering every region of every supported `n`. */
export function emptyVennRegions(): Record<string, VennRegion> {
  const out: Record<string, VennRegion> = {};
  for (const combo of vennCombinations(MAX_VENN_SETS)) {
    out[combo] = { size: null, members: "" };
  }
  return out;
}

/**
 * Split the annotations across the combinations actually being drawn, keyed the
 * way the drawing side keys them.
 *
 * `combinations` is the region list of the current layout (or, when the layout
 * carries set outlines rather than regions, the current `n`'s keys). Restricting
 * to it is what keeps a smaller `n` from drawing the four- and five-set entries
 * still sitting in the record. Matching goes through `foldKey` for the same
 * reason the Euler rows do: the core owns the canonical spelling, and this
 * module should not assume it reproduced it.
 *
 * Quantities that were never entered — or that are negative or non-finite — are
 * left out entirely rather than coerced to zero, so their region draws no count
 * and no dots.
 */
export function vennAnnotationsForRegions(
  entries: Record<string, VennRegion>,
  combinations: readonly string[],
): { counts: Record<string, number>; members: Record<string, string[]> } {
  const byFold = new Map<string, string>();
  for (const c of combinations) byFold.set(foldKey(c), c);

  const counts: Record<string, number> = {};
  const members: Record<string, string[]> = {};
  for (const [combo, entry] of Object.entries(entries)) {
    const combination = byFold.get(foldKey(combo));
    if (combination === undefined) continue;
    const size = entry.size;
    if (size !== null && Number.isFinite(size) && size >= 0) {
      counts[combination] = size;
    }
    const names = parseMembers(entry.members);
    if (names.length > 0) members[combination] = names;
  }
  return { counts, members };
}
