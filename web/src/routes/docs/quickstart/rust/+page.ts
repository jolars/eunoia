// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Quickstart: Rust",
    description:
      "Generate area-proportional Euler and Venn diagrams in Rust with the eunoia crate, the pure-Rust core behind every binding.",
  };
}
