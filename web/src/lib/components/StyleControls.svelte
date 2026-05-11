<script lang="ts">
  import { appState } from "../state.svelte";
  import { defaultColorFor } from "../colors";

  let setNames = $derived(appState.setNames);

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
      min="2"
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

  <div>
    <label for="labelPlacement" class="block text-xs font-medium text-gray-600 mb-1">
      Exterior label solver
    </label>
    <select
      id="labelPlacement"
      bind:value={appState.style.labelPlacement}
      class="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
    >
      <option value="raycast">Raycast</option>
      <option value="forceDirected">Force-directed</option>
    </select>
    <p class="text-xs text-gray-500 mt-1">
      Labels that fit inside their region anchor at the POI. For labels that
      don't fit, raycast emits a deterministic ray from the diagram centroid
      through the POI; force-directed adds polygon-aware repulsion so labels
      avoid both other labels and unrelated regions.
    </p>
  </div>

  <div>
    <label for="labelTether" class="block text-xs font-medium text-gray-600 mb-1">
      Leader tether
    </label>
    <select
      id="labelTether"
      bind:value={appState.style.labelTether}
      class="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
    >
      <option value="poi">Region POI (deep inside)</option>
      <option value="boundary">Polygon edge</option>
    </select>
    <p class="text-xs text-gray-500 mt-1">
      Where exterior leader lines anchor on the source region. POI is safe
      with any rendering style; polygon edge looks cleaner when the shape
      has a visible border (raise the border width above).
    </p>
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
