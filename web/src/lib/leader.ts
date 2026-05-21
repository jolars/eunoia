/**
 * SVG path assembly for exterior label leaders.
 *
 * The leader-curve geometry lives in the core: `placeLabelsForRegions`
 * returns `LabelPlacement.leaderControl1` / `leaderControl2` (computed in the
 * Rust crate from `strategy.leaderCurvature`). This is just the thin
 * renderer-side formatter — a cubic bezier when the core supplied control
 * points (curved leaders), a straight segment otherwise (interior
 * placements, or `leaderCurvature === 0`).
 */

interface Pt {
  x: number;
  y: number;
}

/**
 * SVG `d` string for a leader from `tether` to `end`. Draws the cubic
 * `M tether C control1 control2 end` when both control points are present,
 * otherwise a straight `M tether L end`.
 */
export function leaderPath(
  tether: Pt,
  end: Pt,
  control1?: Pt,
  control2?: Pt,
): string {
  const move = `M ${tether.x},${tether.y}`;
  if (control1 && control2) {
    return `${move} C ${control1.x},${control1.y} ${control2.x},${control2.y} ${end.x},${end.y}`;
  }
  return `${move} L ${end.x},${end.y}`;
}
