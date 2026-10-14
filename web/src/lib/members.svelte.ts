export { foldKey, memberListsForRegions, parseMembers } from "./members";

/**
 * Member callouts and skipped rendering, surfaced from `DiagramSvg` (which
 * runs the packer) to `GlyphControls` (which shows the note). A module-level
 * rune store, like `appState`, so it needs no prop threading through
 * `routes/app/+page.svelte`.
 */
class GlyphStatus {
  /** Per-region overflow count, from `GlyphBoxPlacements.unplaced`. */
  unplaced: Record<string, number> = $state({});
  /** Set when the whole pass was skipped (budget) rather than partly dropped. */
  skipped = $state("");
  /** Regions whose complete roster is shown as a callout. */
  callouts: string[] = $state([]);

  get totalUnplaced(): number {
    let n = 0;
    for (const v of Object.values(this.unplaced)) n += v;
    return n;
  }

  /** Regions that dropped at least one name, for the inline note. */
  get overflowedRegions(): string[] {
    return Object.entries(this.unplaced)
      .filter(([, n]) => n > 0)
      .map(([combo]) => combo || "outside");
  }

  clear() {
    this.unplaced = {};
    this.skipped = "";
    this.callouts = [];
  }
}

export const glyphStatus = new GlyphStatus();
