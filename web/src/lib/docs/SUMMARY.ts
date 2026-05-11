// Source-of-truth nav for the /docs/ tree. Sidebar reads from this.
// Adding a chapter = drop a +page.svx file at the matching slug, then
// add an entry here.

export interface DocChapter {
  title: string;
  /** URL slug, including leading slash. Empty section "" = the docs index. */
  slug: string;
}

export interface DocSection {
  /** Section heading shown in the sidebar. Use null for chapters that
   * sit at the docs root (no group). */
  title: string | null;
  chapters: DocChapter[];
}

export const SUMMARY: DocSection[] = [
  {
    title: null,
    chapters: [{ title: "Introduction", slug: "/docs/" }],
  },
  {
    title: "Quickstart",
    chapters: [
      { title: "Rust", slug: "/docs/quickstart/rust/" },
      { title: "JavaScript", slug: "/docs/quickstart/javascript/" },
    ],
  },
  {
    title: "Concepts",
    chapters: [
      { title: "Fitter pipeline", slug: "/docs/concepts/fitter-pipeline/" },
      { title: "Shapes", slug: "/docs/concepts/shapes/" },
      { title: "Label placement", slug: "/docs/concepts/label-placement/" },
      { title: "Complement", slug: "/docs/concepts/complement/" },
    ],
  },
  {
    title: "Bindings",
    chapters: [
      { title: "WASM contract", slug: "/docs/bindings/wasm-contract/" },
      { title: "Resize loops", slug: "/docs/bindings/resize-loops/" },
    ],
  },
  {
    title: null,
    chapters: [{ title: "Reference", slug: "/docs/reference/" }],
  },
];
