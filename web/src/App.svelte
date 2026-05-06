<script lang="ts">
import { onMount } from "svelte";
import { hydrateFromStorage } from "./lib/state.svelte";
import TopNav from "./lib/components/TopNav.svelte";
import AppPage from "./lib/pages/AppPage.svelte";
import AboutPage from "./lib/pages/AboutPage.svelte";
import CitePage from "./lib/pages/CitePage.svelte";
import LandingPage from "./lib/pages/LandingPage.svelte";

// Single SPA, four routes, one shared <TopNav>:
//   /        → LandingPage
//   /app/    → AppPage  (the wasm-backed diagram tool)
//   /about/  → AboutPage
//   /cite/   → CitePage
// Internal `<a>` clicks are intercepted and turned into history.pushState
// so navigation between routes is soft — no bundle re-parse, no worker
// teardown unless we genuinely leave /app, no scroll/state reset.
//
// Direct loads / refreshes / shared deep links still work because
// vite.config.ts emits `dist/404.html` as a copy of `dist/index.html`,
// which GitHub Pages serves for any path without a backing file.
let currentPath = $state(
  typeof window !== "undefined" ? window.location.pathname : "/",
);

type RouteKey = "landing" | "app" | "about" | "cite";
const route: () => RouteKey = () => {
  if (currentPath.startsWith("/app")) return "app";
  if (currentPath.startsWith("/about")) return "about";
  if (currentPath.startsWith("/cite")) return "cite";
  return "landing";
};
let active = $derived(route());

function onAnchorClick(e: MouseEvent) {
  if (e.defaultPrevented) return;
  if (e.button !== 0) return;
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  const a = (e.target as HTMLElement | null)?.closest("a");
  if (!a) return;
  if (a.target && a.target !== "_self") return;
  if (a.hasAttribute("download")) return;
  const href = a.getAttribute("href");
  if (!href) return;
  // Skip protocol URLs (https:, mailto:, ...) and protocol-relative ones.
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("//")) return;
  // Bare hash links use the browser's native hashchange. Pages that care
  // (none currently) can listen for it themselves.
  if (href.startsWith("#")) return;
  const url = new URL(href, window.location.href);
  if (url.origin !== window.location.origin) return;
  e.preventDefault();
  const target = url.pathname + url.search + url.hash;
  if (target !== window.location.pathname + window.location.search + window.location.hash) {
    history.pushState(null, "", target);
    currentPath = url.pathname;
  }
  // Reset scroll on real navigation — but not for in-page hash links.
  if (!url.hash) window.scrollTo({ top: 0 });
}

onMount(() => {
  hydrateFromStorage();
  const onPop = () => {
    currentPath = window.location.pathname;
  };
  window.addEventListener("popstate", onPop);
  document.addEventListener("click", onAnchorClick);
  return () => {
    window.removeEventListener("popstate", onPop);
    document.removeEventListener("click", onAnchorClick);
  };
});
</script>

<div class="min-h-screen bg-gray-50">
  <TopNav {currentPath} />

  {#if active === "app"}
    <AppPage />
  {:else if active === "about"}
    <AboutPage />
  {:else if active === "cite"}
    <CitePage />
  {:else}
    <LandingPage />
  {/if}
</div>
