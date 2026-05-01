<script lang="ts">
  import { onMount } from "svelte";
  import { appState, saveToStorage } from "../state.svelte";
  import type { FitInputs } from "../fit";
  import type { FitResult } from "../../types/diagram";
  import Section from "../components/Section.svelte";
  import SpecEditor from "../components/SpecEditor.svelte";
  import StyleControls from "../components/StyleControls.svelte";
  import AdvancedControls from "../components/AdvancedControls.svelte";
  import DebugPanel from "../components/DebugPanel.svelte";
  import ExportToolbar from "../components/ExportToolbar.svelte";
  import MetricsPanel from "../components/MetricsPanel.svelte";
  import FitTable from "../components/FitTable.svelte";
  import DiagramSvg from "../components/DiagramSvg.svelte";

  type WorkerResponse =
    | { id: number; ready: true }
    | { id: number; result: FitResult | null }
    | { id: number; error: string };

  let svgEl: SVGSVGElement | null = $state(null);

  let worker: Worker | null = null;
  let nextId = 0;
  let pendingId = -1;
  let debounce: ReturnType<typeof setTimeout> | null = null;

  function postFit(inputs: FitInputs) {
    if (!worker) return;
    pendingId = ++nextId;
    appState.fitting = true;
    worker.postMessage({ id: pendingId, type: "fit", inputs });
  }

  onMount(() => {
    const w = new Worker(
      new URL("../fit.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker = w;
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if ("ready" in msg) {
        appState.loading = false;
        return;
      }
      if (msg.id !== pendingId) return;
      appState.fitting = false;
      if ("error" in msg) {
        appState.error = msg.error;
        appState.result = null;
      } else {
        appState.error = "";
        appState.result = msg.result;
      }
    };
    w.onerror = (e) => {
      appState.loading = false;
      appState.fitting = false;
      appState.error = `Worker error: ${e.message}`;
    };
    w.postMessage({ id: ++nextId, type: "init" });

    return () => {
      w.terminate();
      worker = null;
    };
  });

  $effect(() => {
    if (appState.loading) return;
    const inputs: FitInputs = {
      rows: $state.snapshot(appState.rows),
      inputType: appState.inputType,
      shapeType: appState.shapeType,
      diagramType: appState.diagramType,
      vennN: appState.vennN,
      advanced: $state.snapshot(appState.advanced),
    };
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => postFit(inputs), 150);
  });

  let persistTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    void appState.rows;
    void appState.inputType;
    void appState.shapeType;
    void appState.diagramType;
    void appState.vennN;
    void appState.style;
    void appState.advanced;
    void appState.exportSettings;
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => saveToStorage(), 300);
  });
</script>

<div class="max-w-7xl mx-auto p-6">
  {#if appState.loading}
    <div class="bg-white rounded-lg shadow p-8 text-center">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
      <p class="mt-4 text-gray-600">Loading WASM module…</p>
    </div>
  {:else}
    {#if appState.error}
      <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <p class="text-sm text-red-800">{appState.error}</p>
      </div>
    {/if}

    <div class="grid grid-cols-1 lg:grid-cols-[18rem_minmax(0,1fr)] gap-6">
      <aside class="space-y-3">
        <Section title="Diagram">
          <SpecEditor />
        </Section>
        <Section title="Style">
          <StyleControls />
        </Section>
        <Section title="Advanced" open={false}>
          <AdvancedControls />
        </Section>
        <Section title="Debug" open={false}>
          <DebugPanel />
        </Section>
      </aside>

      <main class="space-y-4">
        <div class="bg-white rounded-lg shadow p-6">
          <div class="flex items-center justify-between gap-4 mb-4">
            <div class="flex items-center gap-3">
              <h2 class="text-lg font-semibold">Diagram</h2>
              {#if appState.fitting}
                <span class="flex items-center text-xs text-gray-500">
                  <span class="inline-block h-3 w-3 mr-1.5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></span>
                  Fitting…
                </span>
              {/if}
            </div>
            <ExportToolbar {svgEl} />
          </div>
          <DiagramSvg
            result={appState.result}
            style={appState.style}
            bind={(s) => (svgEl = s)}
          />
        </div>
        <MetricsPanel metrics={appState.result?.metrics ?? null} />
        <FitTable metrics={appState.result?.metrics ?? null} />
      </main>
    </div>
  {/if}
</div>
