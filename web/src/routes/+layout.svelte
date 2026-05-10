<script lang="ts">
import { onMount, type Snippet } from "svelte";
import "../app.css";
import { hydrateFromStorage } from "$lib/state.svelte";
import TopNav from "$lib/components/TopNav.svelte";

interface Props {
  children: Snippet;
}
let { children }: Props = $props();

onMount(() => {
  hydrateFromStorage();
});
</script>

<div class="min-h-screen bg-gray-50">
  <TopNav />
  {@render children()}
</div>
