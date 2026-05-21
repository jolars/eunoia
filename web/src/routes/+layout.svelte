<script lang="ts">
import { onMount, type Snippet } from "svelte";
import "../app.css";
import { hydrateFromStorage } from "$lib/state.svelte";
import { initTheme } from "$lib/theme.svelte";
import TopNav from "$lib/components/TopNav.svelte";

interface Props {
  children: Snippet;
}
let { children }: Props = $props();

onMount(() => {
  hydrateFromStorage();
  // The inline guard in app.html already set the class for first paint; this
  // keeps `system` tracking the OS live and returns the listener cleanup.
  return initTheme();
});
</script>

<div class="min-h-screen bg-canvas">
  <TopNav />
  {@render children()}
</div>
