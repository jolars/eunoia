<script lang="ts">
  import { appState } from "../state.svelte";
  import type { AdvancedOptions, DiagramType, InputType, Row, ShapeType, VennSetCount } from "../../types/diagram";

  interface Repro {
    rows: Row[];
    inputType: InputType;
    shapeType: ShapeType;
    diagramType: DiagramType;
    vennN: VennSetCount;
    advanced: AdvancedOptions;
  }

  let editor = $state("");
  let status = $state("");
  let statusKind: "ok" | "err" | "" = $state("");

  const current: () => Repro = () => ({
    rows: $state.snapshot(appState.rows),
    inputType: appState.inputType,
    shapeType: appState.shapeType,
    diagramType: appState.diagramType,
    vennN: appState.vennN,
    advanced: $state.snapshot(appState.advanced),
  });

  let snapshot = $derived(JSON.stringify(current(), null, 2));

  $effect(() => {
    // Keep textarea synced with state until the user starts editing.
    editor = snapshot;
  });

  function flash(msg: string, kind: "ok" | "err") {
    status = msg;
    statusKind = kind;
    setTimeout(() => {
      if (status === msg) {
        status = "";
        statusKind = "";
      }
    }, 2000);
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(snapshot);
      flash("Copied to clipboard", "ok");
    } catch (e) {
      flash(`Copy failed: ${e instanceof Error ? e.message : String(e)}`, "err");
    }
  }

  function apply() {
    try {
      const parsed = JSON.parse(editor) as Partial<Repro>;
      if (parsed.rows && Array.isArray(parsed.rows)) appState.rows = parsed.rows;
      if (parsed.inputType) appState.inputType = parsed.inputType;
      if (parsed.shapeType) appState.shapeType = parsed.shapeType;
      if (parsed.diagramType) appState.diagramType = parsed.diagramType;
      if (parsed.vennN && parsed.vennN >= 1 && parsed.vennN <= 5) {
        appState.vennN = parsed.vennN;
      }
      if (parsed.advanced) {
        appState.advanced = { ...appState.advanced, ...parsed.advanced };
      }
      flash("Applied", "ok");
    } catch (e) {
      flash(`Parse error: ${e instanceof Error ? e.message : String(e)}`, "err");
    }
  }

  function refresh() {
    editor = snapshot;
    flash("Synced from state", "ok");
  }
</script>

<div class="space-y-2">
  <p class="text-xs text-gray-500">
    Reproducer for the current diagram (spec + advanced options). Copy to share,
    or paste a snapshot and click Apply.
  </p>
  <textarea
    bind:value={editor}
    spellcheck="false"
    class="w-full h-48 px-2 py-1.5 text-xs font-mono border border-gray-300 rounded resize-y"
  ></textarea>
  <div class="flex flex-wrap gap-2">
    <button
      type="button"
      onclick={copy}
      class="px-2.5 py-1 text-xs font-medium bg-blue-600 text-white rounded hover:bg-blue-700"
    >Copy</button>
    <button
      type="button"
      onclick={apply}
      class="px-2.5 py-1 text-xs font-medium bg-gray-700 text-white rounded hover:bg-gray-800"
    >Apply</button>
    <button
      type="button"
      onclick={refresh}
      class="px-2.5 py-1 text-xs font-medium bg-gray-100 text-gray-700 border border-gray-300 rounded hover:bg-gray-200"
    >Sync from state</button>
    {#if status}
      <span
        class="text-xs self-center"
        class:text-green-700={statusKind === "ok"}
        class:text-red-700={statusKind === "err"}
      >{status}</span>
    {/if}
  </div>
</div>
