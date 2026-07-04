<script lang="ts">
  import { onMount, type Snippet } from "svelte";
  import { page } from "$app/state";
  import "../app.css";
  import TopNav from "$lib/components/TopNav.svelte";
  import { hydrateFromStorage } from "$lib/state.svelte";
  import { initTheme } from "$lib/theme.svelte";

  interface Props {
    children: Snippet;
  }
  let { children }: Props = $props();

  const canonical = $derived(`https://eunoia.bz${page.url.pathname}`);

  // Site-wide fallbacks. Individual routes override these by returning
  // `{ title, description }` from a `load` function (see e.g. cite/+page.ts);
  // the values surface here via `page.data`. Kept in one place so every page
  // emits exactly one `<title>`/`<meta name="description">` — hence they were
  // removed from app.html, which can't be deduped against `<svelte:head>`.
  const DEFAULT_DESCRIPTION =
    "Eunoia is an open-source library for generating area-proportional Euler and Venn diagrams, with bindings for R, Python, Julia, and JavaScript and a full-featured web app.";
  const title = $derived(
    page.data?.title ? `${page.data.title} — Eunoia` : "Eunoia",
  );
  const description = $derived(page.data?.description ?? DEFAULT_DESCRIPTION);

  onMount(() => {
    hydrateFromStorage();
    // The inline guard in app.html already set the class for first paint; this
    // keeps `system` tracking the OS live and returns the listener cleanup.
    return initTheme();
  });
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
  <link rel="canonical" href={canonical} />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={description} />
  <meta property="og:url" content={canonical} />
  <meta name="twitter:title" content={title} />
  <meta name="twitter:description" content={description} />
</svelte:head>

<div class="min-h-screen bg-canvas">
  <TopNav />
  {@render children()}
</div>
