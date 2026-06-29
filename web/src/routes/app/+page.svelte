<script lang="ts">
  import { onMount } from "svelte";
  import AdvancedControls from "$lib/components/AdvancedControls.svelte";
  import DebugPanel from "$lib/components/DebugPanel.svelte";
  import DiagramSvg from "$lib/components/DiagramSvg.svelte";
  import ExportToolbar from "$lib/components/ExportToolbar.svelte";
  import FitTable from "$lib/components/FitTable.svelte";
  import MetricsPanel from "$lib/components/MetricsPanel.svelte";
  import Section from "$lib/components/Section.svelte";
  import SpecEditor from "$lib/components/SpecEditor.svelte";
  import StyleControls from "$lib/components/StyleControls.svelte";
  import type { FitInputs } from "$lib/fit";
  import { appState, saveToStorage } from "$lib/state.svelte";
  import type { FitResult } from "$lib/types/diagram";
  import { RELEASE_URL, VERSION } from "$lib/version";

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
    const w = new Worker(new URL("../../lib/fit.worker.ts", import.meta.url), {
      type: "module",
    });
    worker = w;
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if ("ready" in msg) {
        appState.loading = false;
        return;
      }
      // Errors are surfaced regardless of `pendingId` — boot failures (init
      // throws before any fit is dispatched) would otherwise be filtered out
      // and present as an infinite spinner.
      if ("error" in msg) {
        appState.loading = false;
        appState.fitting = false;
        appState.error = msg.error;
        appState.result = null;
        return;
      }
      if (msg.id !== pendingId) return;
      appState.fitting = false;
      appState.error = "";
      appState.result = msg.result;
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
  <h1 class="sr-only">Euler and Venn diagram builder</h1>
  {#if appState.loading}
    <div class="bg-surface rounded-lg shadow p-8 text-center">
      <div
        class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"
      ></div>
      <p class="mt-4 text-muted">Loading WASM module…</p>
    </div>
  {:else}
    {#if appState.error}
      <div
        class="bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 rounded-lg p-4 mb-6"
      >
        <p class="text-sm text-red-800 dark:text-red-300">{appState.error}</p>
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
        <!-- `paper`: the diagram artboard stays light in dark mode so the
             renderer's hardcoded-black strokes/labels stay readable and match
             the (always-white) PNG/SVG exports. -->
        <div class="paper bg-surface rounded-lg shadow p-6">
          <div class="flex items-center justify-between gap-4 mb-4">
            <div class="flex items-center gap-3">
              <h2 class="text-lg font-semibold">Diagram</h2>
              {#if appState.fitting}
                <span class="flex items-center text-xs text-muted">
                  <span
                    class="inline-block h-3 w-3 mr-1.5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"
                  ></span>
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

  <footer class="mt-12 border-t border-line pt-6 text-xs text-muted">
    <span>
      Eunoia
      <a href={RELEASE_URL} class="text-accent hover:underline">v{VERSION}</a>
    </span>
  </footer>
</div>
