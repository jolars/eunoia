// Page metadata consumed by the root +layout.svelte through `page.data`.
export function load() {
  return {
    title: "Goodness of fit",
    description:
      "The scalar metrics Eunoia reports (stress, diagError, and residuals) that summarize how faithfully the fitted geometry reproduces the requested region sizes.",
  };
}
