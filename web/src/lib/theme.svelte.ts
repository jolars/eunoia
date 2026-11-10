class Theme {
  resolved = $state<"light" | "dark">("light");
}

export const theme = new Theme();

/** Keep a widget's local styles and palette synchronized with Diplodocus. */
export function observeTheme(element: HTMLElement): () => void {
  const system = matchMedia("(prefers-color-scheme: dark)");
  const update = () => {
    const preference = document.documentElement.dataset.theme;
    const dark =
      preference === "dark" || (preference !== "light" && system.matches);
    theme.resolved = dark ? "dark" : "light";
    element.classList.toggle("dark", dark);
  };
  const observer = new MutationObserver(update);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  system.addEventListener("change", update);
  update();
  return () => {
    observer.disconnect();
    system.removeEventListener("change", update);
  };
}
