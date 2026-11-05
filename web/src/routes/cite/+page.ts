// Page metadata consumed by the root +layout.svelte through `page.data`.
// A unique title/description differentiates this page for search engines,
// which otherwise saw the shared site-wide defaults.
export function load() {
  return {
    title: "Citation",
    description:
      "How to cite Eunoia and its language bindings: ready-to-copy BibTeX, BibLaTeX, CSL, APA, Vancouver, and Chicago entries for the Eunoia preprint by Larsson and Lee (2026).",
  };
}
