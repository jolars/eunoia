// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Glyphs",
    description:
      "How Eunoia packs equally-sized unit glyphs (eulerGlyphs-style dots) inside diagram regions, with uniform or random arrangements and an auto-sized shared radius.",
  };
}
