<script lang="ts">
  import { exportJson, exportPdf, exportPng, exportSvg } from "../export";
  import { embeddableFontFaceCss, pdfFontFor } from "../fonts";
  import { appState } from "../state.svelte";

  let { svgEl }: { svgEl: SVGSVGElement | null } = $props();

  let busy = $state(false);
  let popoverOpen = $state(false);
  let errMsg = $state("");

  async function save() {
    if (!svgEl && appState.exportSettings.format !== "json") {
      errMsg = "No diagram to export.";
      return;
    }
    busy = true;
    errMsg = "";
    try {
      const { fontFamily, fontBold, fontItalic } = appState.style;
      switch (appState.exportSettings.format) {
        case "svg":
          if (svgEl) {
            const css = await embeddableFontFaceCss(fontFamily, {
              bold: fontBold,
              italic: fontItalic,
            });
            exportSvg(svgEl, css);
          }
          break;
        case "png":
          if (svgEl) {
            const { width, height } = appState.exportSettings.raster;
            const css = await embeddableFontFaceCss(fontFamily, {
              bold: fontBold,
              italic: fontItalic,
            });
            await exportPng(svgEl, width, height, css);
          }
          break;
        case "pdf":
          if (svgEl) {
            const { width, height } = appState.exportSettings.vector;
            await exportPdf(svgEl, width, height, pdfFontFor(fontFamily));
          }
          break;
        case "json":
          exportJson(appState.toPersisted());
          break;
      }
      popoverOpen = false;
    } catch (e) {
      errMsg = `Export failed: ${e instanceof Error ? e.message : String(e)}`;
    } finally {
      busy = false;
    }
  }
</script>

<div class="relative">
  <div class="flex items-center gap-1">
    <select
      bind:value={appState.exportSettings.format}
      class="px-2 py-1.5 text-sm border border-line rounded bg-surface"
      aria-label="Export format"
    >
      <option value="svg">SVG</option>
      <option value="png">PNG</option>
      <option value="pdf">PDF</option>
      <option value="json">JSON spec</option>
    </select>
    {#if appState.exportSettings.format === "png" || appState.exportSettings.format === "pdf"}
      <button
        type="button"
        onclick={() => (popoverOpen = !popoverOpen)}
        class="px-2 py-1.5 text-xs text-muted border border-line rounded bg-surface hover:bg-inset"
        aria-label="Export size"
        title="Size">Size</button
      >
    {/if}
    <button
      type="button"
      onclick={save}
      disabled={busy}
      class="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
      >{busy ? "Saving…" : "Download"}</button
    >
  </div>

  {#if popoverOpen}
    <div
      class="absolute right-0 top-full mt-1 w-56 bg-surface border border-line rounded-lg shadow-lg p-3 z-10"
    >
      {#if appState.exportSettings.format === "png"}
        <div class="grid grid-cols-2 gap-2">
          <label class="text-xs">
            <span class="text-muted">Width (px)</span>
            <input
              type="number"
              min="64"
              step="64"
              bind:value={appState.exportSettings.raster.width}
              class="mt-1 w-full px-2 py-1 border border-line rounded text-sm"
            />
          </label>
          <label class="text-xs">
            <span class="text-muted">Height (px)</span>
            <input
              type="number"
              min="64"
              step="64"
              bind:value={appState.exportSettings.raster.height}
              class="mt-1 w-full px-2 py-1 border border-line rounded text-sm"
            />
          </label>
        </div>
      {:else if appState.exportSettings.format === "pdf"}
        <div class="grid grid-cols-2 gap-2">
          <label class="text-xs">
            <span class="text-muted">Width (in)</span>
            <input
              type="number"
              min="1"
              step="0.5"
              bind:value={appState.exportSettings.vector.width}
              class="mt-1 w-full px-2 py-1 border border-line rounded text-sm"
            />
          </label>
          <label class="text-xs">
            <span class="text-muted">Height (in)</span>
            <input
              type="number"
              min="1"
              step="0.5"
              bind:value={appState.exportSettings.vector.height}
              class="mt-1 w-full px-2 py-1 border border-line rounded text-sm"
            />
          </label>
        </div>
      {/if}
    </div>
  {/if}

  {#if errMsg}
    <p class="absolute right-0 top-full mt-2 text-xs text-red-600">{errMsg}</p>
  {/if}
</div>
