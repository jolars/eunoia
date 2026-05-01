<script lang="ts">
  import { appState } from "../state.svelte";
  import type { VennSetCount } from "../../types/diagram";
</script>

<div class="space-y-4">
  <div>
    <div class="block text-xs font-medium text-gray-600 mb-1.5">Diagram type</div>
    <div class="flex gap-3">
      <label class="flex items-center cursor-pointer">
        <input
          type="radio"
          bind:group={appState.diagramType}
          value="euler"
          class="mr-1.5"
        />
        <span class="text-sm">Euler</span>
      </label>
      <label class="flex items-center cursor-pointer">
        <input
          type="radio"
          bind:group={appState.diagramType}
          value="venn"
          class="mr-1.5"
        />
        <span class="text-sm">Venn</span>
      </label>
    </div>
  </div>

  {#if appState.diagramType === "venn"}
    <div>
      <div class="block text-xs font-medium text-gray-600 mb-1.5">Number of sets</div>
      <div class="flex gap-2">
        {#each [1, 2, 3, 4, 5] as n}
          <label class="flex items-center cursor-pointer">
            <input
              type="radio"
              bind:group={appState.vennN}
              value={n as VennSetCount}
              class="mr-1"
            />
            <span class="text-sm">{n}</span>
          </label>
        {/each}
      </div>
      <p class="mt-1.5 text-xs text-gray-500">
        Canonical ellipse arrangement. Limited to n ≤ 5 (no Venn diagram of 6+
        ellipses exists).
      </p>
    </div>
  {:else}
    <div>
      <div class="block text-xs font-medium text-gray-600 mb-1.5">Input type</div>
      <div class="flex gap-3">
        <label class="flex items-center cursor-pointer">
          <input
            type="radio"
            bind:group={appState.inputType}
            value="exclusive"
            class="mr-1.5"
          />
          <span class="text-sm">Exclusive</span>
        </label>
        <label class="flex items-center cursor-pointer">
          <input
            type="radio"
            bind:group={appState.inputType}
            value="inclusive"
            class="mr-1.5"
          />
          <span class="text-sm">Inclusive</span>
        </label>
      </div>
    </div>

    <div>
      <div class="block text-xs font-medium text-gray-600 mb-1.5">Shape</div>
      <div class="flex gap-3">
        <label class="flex items-center cursor-pointer">
          <input
            type="radio"
            bind:group={appState.shapeType}
            value="circle"
            class="mr-1.5"
          />
          <span class="text-sm">Circle</span>
        </label>
        <label class="flex items-center cursor-pointer">
          <input
            type="radio"
            bind:group={appState.shapeType}
            value="ellipse"
            class="mr-1.5"
          />
          <span class="text-sm">Ellipse</span>
        </label>
        <label class="flex items-center cursor-pointer">
          <input
            type="radio"
            bind:group={appState.shapeType}
            value="square"
            class="mr-1.5"
          />
          <span class="text-sm">Square</span>
        </label>
      </div>
    </div>

    <div>
      <div class="block text-xs font-medium text-gray-600 mb-1.5">Combinations</div>
      <div class="space-y-1.5">
        {#each appState.rows as row, i (i)}
          <div class="flex gap-1.5">
            <input
              type="text"
              bind:value={appState.rows[i].input}
              placeholder="A or A&B"
              class="flex-1 min-w-0 px-2 py-1.5 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <input
              type="number"
              bind:value={appState.rows[i].size}
              min="0"
              step="0.1"
              class="w-20 px-2 py-1.5 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="button"
              onclick={() => appState.removeRow(i)}
              class="px-2 py-1.5 bg-red-50 text-red-600 rounded hover:bg-red-100 text-sm"
              aria-label="Remove row"
              title="Remove"
            >×</button>
          </div>
        {/each}

        <button
          type="button"
          onclick={() => appState.addRow()}
          class="w-full px-3 py-1.5 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 text-sm"
        >+ Add row</button>
      </div>
    </div>
  {/if}
</div>
