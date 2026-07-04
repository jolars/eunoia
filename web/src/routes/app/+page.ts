// The diagram tool runs entirely in the browser (WASM worker, large
// transitive WASM bundle from @jolars/eunoia via DiagramSvg). Prerender
// the empty HTML shell, but don't try to SSR the page module.
export const ssr = false;

// Page metadata consumed by the root +layout.svelte through `page.data`.
// Because `ssr = false`, this only sets the head after hydration (the
// prerendered shell has no title) — acceptable for an interactive tool.
export function load() {
  return {
    title: "Diagram builder",
    description:
      "Interactive area-proportional Euler and Venn diagram builder: enter set sizes and intersections, tune the fit, and export SVG. Runs entirely in your browser.",
  };
}
