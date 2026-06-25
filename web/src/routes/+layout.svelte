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

  onMount(() => {
    hydrateFromStorage();
    // The inline guard in app.html already set the class for first paint; this
    // keeps `system` tracking the OS live and returns the listener cleanup.
    return initTheme();
  });
</script>

<svelte:head>
  <link rel="canonical" href={canonical} />
  <meta property="og:url" content={canonical} />
</svelte:head>

<div class="min-h-screen bg-canvas">
  <TopNav />
  {@render children()}
</div>
