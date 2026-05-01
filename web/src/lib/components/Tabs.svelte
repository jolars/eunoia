<script lang="ts">
  import { appState } from "../state.svelte";
  import type { TabKey } from "../../types/diagram";

  const tabs: { key: TabKey; label: string; hash: string }[] = [
    { key: "app", label: "App", hash: "" },
    { key: "about", label: "About", hash: "#about" },
    { key: "cite", label: "Cite", hash: "#cite" },
  ];

  function select(t: TabKey, hash: string) {
    appState.tab = t;
    if (typeof window !== "undefined") {
      if (hash === "") {
        history.replaceState(
          null,
          "",
          window.location.pathname + window.location.search,
        );
      } else {
        history.replaceState(
          null,
          "",
          window.location.pathname + window.location.search + hash,
        );
      }
    }
  }
</script>

<nav class="border-b border-gray-200 bg-white">
  <div class="max-w-7xl mx-auto px-6 flex items-center justify-between">
    <h1 class="text-2xl font-bold text-gray-900 py-4">Eunoia</h1>
    <ul class="flex">
      {#each tabs as t}
        <li>
          <button
            type="button"
            class="px-4 py-4 text-sm font-medium border-b-2 transition-colors"
            class:border-blue-500={appState.tab === t.key}
            class:text-blue-600={appState.tab === t.key}
            class:border-transparent={appState.tab !== t.key}
            class:text-gray-500={appState.tab !== t.key}
            class:hover:text-gray-900={appState.tab !== t.key}
            onclick={() => select(t.key, t.hash)}
          >
            {t.label}
          </button>
        </li>
      {/each}
    </ul>
  </div>
</nav>
