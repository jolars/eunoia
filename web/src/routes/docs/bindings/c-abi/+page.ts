// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "C ABI contract",
    description:
      'The eunoia-capi C ABI seam (JSON in and out over extern "C") backing the Julia binding and other C-interop hosts.',
  };
}
