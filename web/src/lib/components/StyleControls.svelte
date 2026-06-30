<script lang="ts">
  import { defaultColorFor, PALETTES } from "../colors";
  import { FONT_FAMILIES } from "../fonts";
  import { appState } from "../state.svelte";

  let setNames = $derived(appState.setNames);

  function colorFor(name: string, idx: number): string {
    return (
      appState.style.colors[name] ??
      defaultColorFor(idx, appState.style.palette)
    );
  }

  // Switching palettes clears per-set overrides so the new scheme applies
  // cleanly; the per-set pickers below let you re-tweak afterwards.
  function selectPalette(id: string) {
    appState.style.palette = id;
    appState.style.colors = {};
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
    <span class="block text-xs font-medium text-muted mb-1.5">Palette</span>
    <div class="space-y-1">
      {#each PALETTES as p}
        {@const selected = appState.style.palette === p.id}
        <button
          type="button"
          onclick={() => selectPalette(p.id)}
          class="w-full flex items-center gap-2 rounded border px-1.5 py-1 transition-colors"
          class:border-accent={selected}
          class:bg-accent-soft={selected}
          class:border-transparent={!selected}
          class:hover:bg-inset={!selected}
          aria-pressed={selected}
        >
          <span class="flex h-4 flex-1 overflow-hidden rounded-sm">
            {#each p.colors as c}
              <span class="flex-1" style="background-color: {c}"></span>
            {/each}
          </span>
          <span class="w-16 shrink-0 text-left text-xs">{p.name}</span>
        </button>
      {/each}
    </div>
  </div>

  <div>
    <div class="flex items-center justify-between mb-1.5">
      <span class="block text-xs font-medium text-muted">Set colors</span>
      {#if Object.keys(appState.style.colors).length > 0}
        <button
          type="button"
          onclick={resetAllColors}
          class="text-xs text-accent hover:underline">Reset</button
        >
      {/if}
    </div>
    {#if setNames.length === 0}
      <p class="text-xs text-muted">Add a set to assign colors.</p>
    {:else}
      <div class="flex flex-wrap gap-2">
        {#each setNames as name, i}
          <label class="flex items-center gap-1.5 cursor-pointer text-sm">
            <input
              type="color"
              aria-label={`Color for set ${name}`}
              value={colorFor(name, i)}
              oninput={(e) =>
                setColor(name, (e.target as HTMLInputElement).value)}
              class="h-6 w-7 cursor-pointer rounded border border-line p-0"
            />
            <span class="font-mono">{name}</span>
            {#if appState.style.colors[name]}
              <button
                type="button"
                onclick={() => resetColor(name)}
                class="text-xs text-faint hover:text-ink"
                aria-label={`Reset ${name} to default color`}
                title="Use default">×</button
              >
            {/if}
          </label>
        {/each}
      </div>
    {/if}
  </div>

  <div>
    <label for="alpha" class="block text-xs font-medium text-muted mb-1">
      Opacity <span class="font-mono text-faint"
        >{appState.style.alpha.toFixed(2)}</span
      >
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
    <label for="strokeWidth" class="block text-xs font-medium text-muted mb-1">
      Border width <span class="font-mono text-faint"
        >{appState.style.strokeWidth.toFixed(1)}</span
      >
    </label>
    <input
      id="strokeWidth"
      type="range"
      min="0"
      max="2"
      step="0.1"
      bind:value={appState.style.strokeWidth}
      class="w-full"
    />
  </div>

  <div>
    <label for="labelSize" class="block text-xs font-medium text-muted mb-1">
      Label size <span class="font-mono text-faint"
        >{appState.style.labelSize}</span
      >
    </label>
    <input
      id="labelSize"
      type="range"
      min="2"
      max="20"
      step="1"
      bind:value={appState.style.labelSize}
      class="w-full"
    />
    <div class="flex gap-3 mt-1.5">
      <label class="flex items-center cursor-pointer text-sm">
        <input
          type="checkbox"
          bind:checked={appState.style.fontBold}
          class="mr-1.5"
        />
        Bold
      </label>
      <label class="flex items-center cursor-pointer text-sm">
        <input
          type="checkbox"
          bind:checked={appState.style.fontItalic}
          class="mr-1.5"
        />
        Italic
      </label>
    </div>
  </div>

  <div>
    <label for="fontFamily" class="block text-xs font-medium text-muted mb-1">
      Font
    </label>
    <select
      id="fontFamily"
      bind:value={appState.style.fontFamily}
      class="w-full px-2 py-1.5 text-sm border border-line rounded"
      style="font-family: {appState.style.fontFamily};"
    >
      {#each FONT_FAMILIES as font}
        <option value={font.value} style="font-family: {font.value};">
          {font.label}
        </option>
      {/each}
    </select>
  </div>

  <div class="flex flex-wrap gap-x-4 gap-y-2">
    <label class="flex items-center cursor-pointer text-sm">
      <input
        type="checkbox"
        bind:checked={appState.style.showCounts}
        class="mr-1.5"
      />
      Region counts
    </label>
    <label class="flex items-center cursor-pointer text-sm">
      <input
        type="checkbox"
        bind:checked={appState.style.showLegend}
        class="mr-1.5"
      />
      Legend
    </label>
  </div>

  <div>
    <label
      for="labelPlacement"
      class="block text-xs font-medium text-muted mb-1"
    >
      Exterior label solver
    </label>
    <select
      id="labelPlacement"
      bind:value={appState.style.labelPlacement}
      class="w-full px-2 py-1.5 text-sm border border-line rounded"
    >
      <option value="raycast">Raycast</option>
      <option value="forceDirected">Force-directed</option>
      <option value="matched">Matched (boundary labeling)</option>
      <option value="elbow">Elbow (columns)</option>
    </select>
    <p class="text-xs text-muted mt-1">
      Labels that fit inside their region anchor at the POI. For labels that
      don't fit, raycast emits a deterministic ray from the diagram centroid
      through the POI; force-directed adds polygon-aware repulsion so labels
      avoid both other labels and unrelated regions; matched places labels on a
      ring that hugs the diagram, spread by their width so they don't overlap,
      and uncrosses the leaders so they never cross while staying close; elbow
      stacks labels in left/right columns connected by orthogonal (d3-pie style)
      leaders.
    </p>
  </div>

  <div>
    <label for="labelTether" class="block text-xs font-medium text-muted mb-1">
      Leader tether
    </label>
    <select
      id="labelTether"
      bind:value={appState.style.labelTether}
      class="w-full px-2 py-1.5 text-sm border border-line rounded"
    >
      <option value="poi">Region POI (deep inside)</option>
      <option value="boundary">Polygon edge</option>
    </select>
    <p class="text-xs text-muted mt-1">
      Where exterior leader lines anchor on the source region. POI is safe with
      any rendering style; polygon edge looks cleaner when the shape has a
      visible border (raise the border width above).
    </p>
  </div>

  {#if appState.style.showLegend}
    <div>
      <label for="legendPos" class="block text-xs font-medium text-muted mb-1">
        Legend position
      </label>
      <select
        id="legendPos"
        bind:value={appState.style.legendPosition}
        class="w-full px-2 py-1.5 text-sm border border-line rounded"
      >
        <option value="right">Right</option>
        <option value="left">Left</option>
        <option value="top">Top</option>
        <option value="bottom">Bottom</option>
      </select>
    </div>
  {/if}
</div>
