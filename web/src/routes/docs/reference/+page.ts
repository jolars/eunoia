// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Reference",
    description:
      "API reference index for Eunoia, linking the Rust, R, Python, Julia, and JavaScript binding documentation.",
  };
}
