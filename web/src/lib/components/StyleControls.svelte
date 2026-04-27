<script lang="ts">
  import { appState } from "../state.svelte";
  import { defaultColorFor } from "../colors";

  let setNames = $derived.by(() => {
    const r = appState.result;
    if (r) {
      if (r.circles.length > 0) return r.circles.map((c) => c.label);
      if (r.ellipses.length > 0) return r.ellipses.map((e) => e.label);
      if (r.polygons.length > 0) return r.polygons.map((p) => p.label);
      const seen = new Set<string>();
      for (const region of r.regions) {
        for (const ch of region.combination.split("&")) {
          const t = ch.trim();
          if (t) seen.add(t);
        }
      }
      return Array.from(seen);
    }
    const seen = new Set<string>();
    for (const row of appState.rows) {
      for (const ch of row.input.split("&")) {
        const t = ch.trim();
        if (t) seen.add(t);
      }
    }
    return Array.from(seen);
  });

  function colorFor(name: string, idx: number): string {
    return appState.style.colors[name] ?? defaultColorFor(idx);
  }

  function setColor(name: string, value: string) {
    appState.style.colors = { ...appState.style.colors, [name]: value };
  }

  function resetColor(name: string) {
    const next = { ...appState.style.colors };
    delete next[name];
    appState.style.colors = next;
  }

  function resetAllColors() {
    appState.style.colors = {};
  }
</script>

<div class="space-y-4">
  <div>
    <div class="flex items-center justify-between mb-1.5">
      <span class="block text-xs font-medium text-gray-600">Set colors</span>
      {#if Object.keys(appState.style.colors).length > 0}
        <button
          type="button"
          onclick={resetAllColors}
          class="text-xs text-blue-600 hover:underline"
        >Reset</button>
      {/if}
    </div>
    {#if setNames.length === 0}
      <p class="text-xs text-gray-500">Add a set to assign colors.</p>
    {:else}
      <div class="flex flex-wrap gap-2">
        {#each setNames as name, i}
          <label class="flex items-center gap-1.5 cursor-pointer text-sm">
            <input
              type="color"
              aria-label={`Color for set ${name}`}
              value={colorFor(name, i)}
              oninput={(e) => setColor(name, (e.target as HTMLInputElement).value)}
              class="h-6 w-7 cursor-pointer rounded border border-gray-300 p-0"
            />
            <span class="font-mono">{name}</span>
            {#if appState.style.colors[name]}
              <button
                type="button"
                onclick={() => resetColor(name)}
                class="text-xs text-gray-400 hover:text-gray-700"
                aria-label={`Reset ${name} to default color`}
                title="Use default"
              >×</button>
            {/if}
          </label>
        {/each}
      </div>
    {/if}
  </div>

  <div>
    <label for="alpha" class="block text-xs font-medium text-gray-600 mb-1">
      Opacity <span class="font-mono text-gray-400">{appState.style.alpha.toFixed(2)}</span>
    </label>
    <input
      id="alpha"
      type="range"
      min="0"
      max="1"
      step="0.01"
      bind:value={appState.style.alpha}
      class="w-full"
    />
  </div>

  <div>
    <label for="strokeWidth" class="block text-xs font-medium text-gray-600 mb-1">
      Border width <span class="font-mono text-gray-400">{appState.style.strokeWidth.toFixed(1)}</span>
    </label>
    <input
      id="strokeWidth"
      type="range"
      min="0"
      max="4"
      step="0.1"
      bind:value={appState.style.strokeWidth}
      class="w-full"
    />
  </div>

  <div>
    <label for="labelSize" class="block text-xs font-medium text-gray-600 mb-1">
      Label size <span class="font-mono text-gray-400">{appState.style.labelSize}</span>
    </label>
    <input
      id="labelSize"
      type="range"
      min="6"
      max="40"
      step="1"
      bind:value={appState.style.labelSize}
      class="w-full"
    />
    <div class="flex gap-3 mt-1.5">
      <label class="flex items-center cursor-pointer text-sm">
        <input type="checkbox" bind:checked={appState.style.fontBold} class="mr-1.5" />
        Bold
      </label>
      <label class="flex items-center cursor-pointer text-sm">
        <input type="checkbox" bind:checked={appState.style.fontItalic} class="mr-1.5" />
        Italic
      </label>
    </div>
  </div>

  <div class="flex flex-wrap gap-x-4 gap-y-2">
    <label class="flex items-center cursor-pointer text-sm">
      <input type="checkbox" bind:checked={appState.style.showCounts} class="mr-1.5" />
      Region counts
    </label>
    <label class="flex items-center cursor-pointer text-sm">
      <input type="checkbox" bind:checked={appState.style.showLegend} class="mr-1.5" />
      Legend
    </label>
  </div>

  {#if appState.style.showLegend}
    <div>
      <label for="legendPos" class="block text-xs font-medium text-gray-600 mb-1">
        Legend position
      </label>
      <select
        id="legendPos"
        bind:value={appState.style.legendPosition}
        class="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
      >
        <option value="right">Right</option>
        <option value="left">Left</option>
        <option value="top">Top</option>
        <option value="bottom">Bottom</option>
      </select>
    </div>
  {/if}
</div>
