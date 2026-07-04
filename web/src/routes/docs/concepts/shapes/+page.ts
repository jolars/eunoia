// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Shapes",
    description:
      "The fittable shapes in Eunoia (circles, ellipses, squares, and rectangles) and when each best reproduces a set of region sizes.",
  };
}
