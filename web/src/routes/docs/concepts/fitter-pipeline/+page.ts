// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Fitter pipeline",
    description:
      "Inside Eunoia's spec-to-fit pipeline: MDS initialization, loss-driven refinement, and how ellipses handle lopsided overlaps that circles cannot.",
  };
}
