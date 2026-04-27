<script lang="ts">
  import type { FitMetrics } from "../../types/diagram";

  let { metrics }: { metrics: FitMetrics | null } = $props();

  let rows = $derived.by(() => {
    if (!metrics) return [] as { combo: string; target: number; fitted: number; regionError: number }[];
    const keys = new Set([
      ...Object.keys(metrics.target),
      ...Object.keys(metrics.fitted),
      ...Object.keys(metrics.regionError),
    ]);
    return Array.from(keys)
      .sort()
      .map((combo) => ({
        combo,
        target: metrics.target[combo] ?? 0,
        fitted: metrics.fitted[combo] ?? 0,
        regionError: metrics.regionError[combo] ?? 0,
      }));
  });

  function fmt(v: number): string {
    return v.toFixed(3);
  }
</script>

<div class="bg-white rounded-lg shadow p-6">
  <h2 class="text-lg font-semibold mb-4">Fit table</h2>
  {#if rows.length === 0}
    <p class="text-sm text-gray-500">No data.</p>
  {:else}
    <table class="w-full text-sm">
      <thead>
        <tr class="border-b border-gray-200 text-left text-gray-600 text-xs uppercase tracking-wide">
          <th class="py-1.5 pr-2">Combination</th>
          <th class="py-1.5 pr-2 text-right">Input</th>
          <th class="py-1.5 pr-2 text-right">Fit</th>
          <th class="py-1.5 text-right">regionError</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row}
          <tr class="border-b border-gray-100 last:border-0">
            <td class="py-1 pr-2 font-mono">{row.combo}</td>
            <td class="py-1 pr-2 text-right font-mono">{fmt(row.target)}</td>
            <td class="py-1 pr-2 text-right font-mono">{fmt(row.fitted)}</td>
            <td class="py-1 text-right font-mono">{fmt(row.regionError)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>
