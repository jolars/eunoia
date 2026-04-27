<script lang="ts">
  import { onMount } from "svelte";
  import { appState, hydrateFromStorage, syncTabToHash } from "./lib/state.svelte";
  import Tabs from "./lib/components/Tabs.svelte";
  import AppPage from "./lib/pages/AppPage.svelte";
  import AboutPage from "./lib/pages/AboutPage.svelte";
  import CitePage from "./lib/pages/CitePage.svelte";

  onMount(() => {
    hydrateFromStorage();
    syncTabToHash();
    const handler = () => syncTabToHash();
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  });
</script>

<div class="min-h-screen bg-gray-50">
  <Tabs />

  {#if appState.tab === "about"}
    <AboutPage />
  {:else if appState.tab === "cite"}
    <CitePage />
  {:else}
    <AppPage />
  {/if}
</div>
