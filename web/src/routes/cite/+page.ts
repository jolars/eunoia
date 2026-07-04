// Page metadata consumed by the root +layout.svelte through `page.data`.
// A unique title/description differentiates this page for search engines,
// which otherwise saw the shared site-wide defaults.
export function load() {
  return {
    title: "Citation",
    description:
      "How to cite Eunoia and eulerr: ready-to-copy BibTeX, BibLaTeX, CSL, APA, Vancouver, and Chicago entries for the area-proportional Euler diagram paper by Larsson and Gustafsson (2018).",
  };
}
