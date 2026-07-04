// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "JavaScript API reference",
    description:
      "The public API of @jolars/eunoia: exported functions, their options, and return types for the JavaScript and WebAssembly build.",
  };
}
