<script lang="ts">
  import { glyphStatus, parseMembers } from "../members.svelte";
  import { appState } from "../state.svelte";

  const colorControls = [
    { key: "glyphFill", label: "Fill color", fallback: "#ffffff" },
    { key: "glyphStroke", label: "Border color", fallback: "#000000" },
  ] as const;

  // Only over the *current* n's regions: entries left behind by a larger n are
  // not drawn, so they must not silence the "nothing to draw" notes either.
  const vennEntries = $derived(
    appState.vennCombos.map((c) => appState.vennRegions[c]),
  );
  const hasVennQuantities = $derived(
    vennEntries.some((e) => e?.size !== null && e?.size !== undefined),
  );
  const hasVennMembers = $derived(
    vennEntries.some((e) => parseMembers(e?.members).length > 0),
  );
</script>

<div class="space-y-4">
  <div>
    <label for="glyphMode" class="block text-xs font-medium text-muted mb-1">
      Draw
    </label>
    <select
      id="glyphMode"
      bind:value={appState.style.glyphMode}
      class="w-full px-2 py-1.5 text-sm border border-line rounded"
    >
      <option value="none">Nothing</option>
      <option value="dots">Dots (unit glyphs)</option>
      <option value="members">Member names</option>
    </select>
    {#if appState.style.glyphMode === "dots"}
      <p class="text-xs text-muted mt-1">
        One equally-sized dot per data unit, packed inside its region
        (eulerGlyphs-style). Counts are the region quantities rounded to
        integers; the dot radius is auto-sized so every region fits its count.
      </p>
      {#if appState.diagramType === "venn" && !hasVennQuantities}
        <p class="text-xs text-amber-600 dark:text-amber-400 mt-1">
          No Venn region quantities entered, so there is nothing to count out in
          dots. Fill in <span class="font-medium">Diagram → Regions</span>.
        </p>
      {/if}
    {:else if appState.style.glyphMode === "members"}
      <p class="text-xs text-muted mt-1">
        The member names typed on the combination rows — or, for a Venn, on its
        region rows — measured and packed inside their region. If the full list
        does not fit, it moves together into a labeled callout. Names shrink to
        fit but never grow past the size below.
      </p>
      {#if appState.diagramType === "venn" && !hasVennMembers}
        <p class="text-xs text-amber-600 dark:text-amber-400 mt-1">
          No Venn region names entered. Fill in <span class="font-medium"
            >Diagram → Regions</span
          >.
        </p>
      {:else if glyphStatus.skipped}
        <p class="text-xs text-amber-600 dark:text-amber-400 mt-1">
          Skipped: {glyphStatus.skipped}
        </p>
      {:else if glyphStatus.totalUnplaced > 0}
        <p class="text-xs text-amber-600 dark:text-amber-400 mt-1">
          {glyphStatus.totalUnplaced} name{glyphStatus.totalUnplaced === 1
            ? ""
            : "s"} did not fit ({glyphStatus.overflowedRegions.join(", ")}).
          Lower the name size or shorten the names.
        </p>
      {:else if glyphStatus.callouts.length > 0}
        <p class="text-xs text-muted mt-1">
          Complete lists shown in callouts: {glyphStatus.callouts.join(", ")}. A
          list marked “no geometry” has no drawable region to point to.
        </p>
      {/if}
    {/if}
  </div>

  {#if appState.style.glyphMode !== "none"}
    <div>
      <label
        for="glyphArrangement"
        class="block text-xs font-medium text-muted mb-1"
      >
        Arrangement
      </label>
      <select
        id="glyphArrangement"
        bind:value={appState.style.glyphArrangement}
        class="w-full px-2 py-1.5 text-sm border border-line rounded"
      >
        <option value="uniform">
          {appState.style.glyphMode === "members"
            ? "Uniform (rows)"
            : "Uniform (hex lattice)"}
        </option>
        <option value="rectangular">Rectangular grid</option>
        <option value="lowDiscrepancy">Low discrepancy (Halton)</option>
        <option value="random">Random (seeded scatter)</option>
      </select>
    </div>

    {#if appState.style.glyphMode === "members"}
      <div>
        <label
          for="memberLabelSize"
          class="block text-xs font-medium text-muted mb-1"
        >
          Name size <span class="font-mono text-faint"
            >{appState.style.memberLabelSize.toFixed(1)}</span
          >
        </label>
        <input
          id="memberLabelSize"
          type="range"
          min="2"
          max="12"
          step="0.5"
          bind:value={appState.style.memberLabelSize}
          class="w-full"
        />
      </div>
    {/if}

    <div>
      <label for="glyphGap" class="block text-xs font-medium text-muted mb-1">
        {appState.style.glyphMode === "dots" ? "Dot spacing" : "Spacing"}
        <span class="font-mono text-faint"
          >{appState.style.glyphGap.toFixed(2)}</span
        >
      </label>
      <input
        id="glyphGap"
        type="range"
        min="0"
        max="1"
        step="0.05"
        bind:value={appState.style.glyphGap}
        class="w-full"
      />
      <p class="text-xs text-muted mt-1">
        {#if appState.style.glyphMode === "dots"}
          Minimum gap between dots, in dot diameters. The grid may spread dots
          farther apart.
        {:else}
          Breathing room between names, as a fraction of the text row height.
          Half of it is also kept to region boundaries, holes, and labels.
        {/if}
      </p>
    </div>

    {#if appState.style.glyphMode === "dots"}
      {#each colorControls as control}
        <div class="flex items-center justify-between gap-2">
          <label class="flex items-center gap-2 text-xs font-medium text-muted">
            <input
              type="color"
              aria-label={`Glyph ${control.label.toLowerCase()}`}
              value={appState.style[control.key] ?? control.fallback}
              oninput={(e) => {
                appState.style[control.key] = e.currentTarget.value;
              }}
              class="h-6 w-7 cursor-pointer rounded border border-line p-0"
            />
            {control.label}
          </label>
          {#if appState.style[control.key] !== null}
            <button
              type="button"
              onclick={() => {
                appState.style[control.key] = null;
              }}
              class="text-xs text-accent hover:underline"
              aria-label={`Reset glyph ${control.label.toLowerCase()} to automatic`}
              >Reset</button
            >
          {:else}
            <span class="text-xs text-faint">Automatic</span>
          {/if}
        </div>
      {/each}
      <div>
        <div class="flex items-center justify-between mb-1">
          <label for="glyphStrokeWidth" class="text-xs font-medium text-muted">
            Border width <span class="font-mono text-faint"
              >{(
                appState.style.glyphStrokeWidth ??
                appState.style.strokeWidth / 2
              ).toFixed(2)}</span
            >
          </label>
          {#if appState.style.glyphStrokeWidth !== null}
            <button
              type="button"
              onclick={() => {
                appState.style.glyphStrokeWidth = null;
              }}
              class="text-xs text-accent hover:underline"
              aria-label="Reset glyph border width to automatic">Reset</button
            >
          {:else}
            <span class="text-xs text-faint">Automatic</span>
          {/if}
        </div>
        <input
          id="glyphStrokeWidth"
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={appState.style.glyphStrokeWidth ??
            appState.style.strokeWidth / 2}
          oninput={(e) => {
            appState.style.glyphStrokeWidth = Number(e.currentTarget.value);
          }}
          class="w-full"
        />
        <p class="text-xs text-muted mt-1">
          Set to zero to hide borders. Automatic uses half the set border width.
        </p>
      </div>

      <div>
        <label
          for="glyphBoundaryGap"
          class="block text-xs font-medium text-muted mb-1"
        >
          Boundary gap <span class="font-mono text-faint"
            >{appState.style.glyphBoundaryGap.toFixed(2)}</span
          >
        </label>
        <input
          id="glyphBoundaryGap"
          type="range"
          min="0"
          max="3"
          step="0.05"
          bind:value={appState.style.glyphBoundaryGap}
          class="w-full"
        />
        <p class="text-xs text-muted mt-1">
          Minimum gap to region edges, holes, and labels, in dot radii.
        </p>
      </div>
    {/if}

    {#if appState.style.glyphArrangement === "random"}
      <div>
        <label
          for="glyphSeed"
          class="block text-xs font-medium text-muted mb-1"
        >
          Seed
        </label>
        <input
          id="glyphSeed"
          type="number"
          min="0"
          step="1"
          bind:value={appState.style.glyphSeed}
          class="w-full px-2 py-1.5 text-sm border border-line rounded"
        />
      </div>
    {/if}
  {/if}
</div>
