// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Documentation",
    description:
      "Introduction to Eunoia: a Rust engine for area-proportional Euler and Venn diagrams, with bindings for R, Python, Julia, and JavaScript.",
  };
}
