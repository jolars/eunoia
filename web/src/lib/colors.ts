/**
 * Default fill palette used when no per-set color has been chosen.
 * Hex form so the color pickers can preselect them.
 */
export const DEFAULT_PALETTE = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#22c55e", // green
  "#eab308", // yellow
  "#a855f7", // purple
  "#14b8a6", // teal
  "#ec4899", // pink
];

export function defaultColorFor(i: number): string {
  return DEFAULT_PALETTE[i % DEFAULT_PALETTE.length];
}

export function colorForSet(
  setName: string,
  index: number,
  overrides: Record<string, string>,
): string {
  return overrides[setName] || defaultColorFor(index);
}
