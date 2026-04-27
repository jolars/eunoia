<script lang="ts">
  import type { FitMetrics } from "../../types/diagram";

  let { metrics }: { metrics: FitMetrics | null } = $props();

  function fmt(v: number, d = 4): string {
    if (!isFinite(v)) return "—";
    return v.toFixed(d);
  }
</script>

<div class="bg-white rounded-lg shadow p-6">
  <h2 class="text-lg font-semibold mb-4">Goodness of fit</h2>
  {#if metrics}
    <dl class="grid grid-cols-3 gap-4 text-sm">
      <div>
        <dt class="text-gray-500 uppercase text-xs tracking-wide">stress</dt>
        <dd class="font-mono text-base">{fmt(metrics.stress)}</dd>
      </div>
      <div>
        <dt class="text-gray-500 uppercase text-xs tracking-wide">diagError</dt>
        <dd class="font-mono text-base">{fmt(metrics.diagError)}</dd>
      </div>
      <div>
        <dt class="text-gray-500 uppercase text-xs tracking-wide">loss</dt>
        <dd class="font-mono text-base">{fmt(metrics.loss)}</dd>
      </div>
    </dl>
    <p class="mt-3 text-xs text-gray-500">
      stress: venneuler-style. diagError: max region error. iterations: {metrics.iterations}.
    </p>
  {:else}
    <p class="text-sm text-gray-500">No diagram fitted yet.</p>
  {/if}
</div>
