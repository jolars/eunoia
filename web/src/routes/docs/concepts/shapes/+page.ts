// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Shapes",
    description:
      "The five fittable shapes in Eunoia (circles, ellipses, squares, rectangles, and rotated rectangles), their Venn limits, and when to use each.",
  };
}
