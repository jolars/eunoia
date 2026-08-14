<script lang="ts">
  import { glyphStatus, parseMembers } from "../members.svelte";
  import { appState } from "../state.svelte";

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
        region rows — measured and packed inside their region. A single
        diagram-wide scale is found by shrinking the name size until every
        region fits, never by growing it, so the size below is a ceiling.
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
        Spacing <span class="font-mono text-faint"
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
        Breathing room around each mark, as a fraction of the dot radius or of
        the text row height. Half of it is also kept to region boundaries,
        holes, and the region labels.
      </p>
    </div>

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
