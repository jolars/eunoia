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
      { title: "R", slug: "/docs/quickstart/r/" },
      { title: "Python", slug: "/docs/quickstart/python/" },
      { title: "Julia", slug: "/docs/quickstart/julia/" },
    ],
  },
  {
    title: "Concepts",
    chapters: [
      { title: "Fitter pipeline", slug: "/docs/concepts/fitter-pipeline/" },
      { title: "Goodness of fit", slug: "/docs/concepts/goodness-of-fit/" },
      { title: "Shapes", slug: "/docs/concepts/shapes/" },
      { title: "Label placement", slug: "/docs/concepts/label-placement/" },
      { title: "Complement", slug: "/docs/concepts/complement/" },
    ],
  },
  {
    title: "Bindings",
    chapters: [{ title: "C ABI contract", slug: "/docs/bindings/c-abi/" }],
  },
  {
    title: "Reference",
    chapters: [
      { title: "Overview", slug: "/docs/reference/" },
      { title: "JavaScript API", slug: "/docs/reference/javascript/" },
    ],
  },
];
