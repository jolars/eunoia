// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Quickstart: Julia",
    description:
      "Generate area-proportional Euler and Venn diagrams in Julia with the Eunoia.jl binding.",
  };
}
