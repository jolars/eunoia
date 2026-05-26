/**
 * SVG path assembly for exterior label leaders.
 *
 * The leader geometry lives in the core: `placeLabelsForRegions` returns
 * `LabelPlacement.tether`, `leaderEnd`, and `leaderWaypoints` (the
 * intermediate polyline vertices, in draw order). This is just the thin
 * renderer-side formatter — a polyline `tether → waypoints… → leaderEnd`.
 * Straight leaders carry no waypoints, so this reduces to a single segment.
 */

interface Pt {
  x: number;
  y: number;
}

/**
 * SVG `d` string for a leader from `tether` to `end`, passing through
 * `waypoints` in draw order. Straight leaders pass no waypoints and yield
 * `M tether L end`.
 */
export function leaderPath(
  tether: Pt,
  end: Pt,
  waypoints: ReadonlyArray<Pt> = [],
): string {
  let d = `M ${tether.x},${tether.y}`;
  for (const w of waypoints) {
    d += ` L ${w.x},${w.y}`;
  }
  return `${d} L ${end.x},${end.y}`;
}
