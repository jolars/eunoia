<script lang="ts">
  import { appState } from "../state.svelte";
</script>

<div class="space-y-4">
  <div>
    <label
      for="optimizer"
      class="block text-xs font-medium text-gray-600 mb-1"
    >Optimizer</label>
    <select
      id="optimizer"
      bind:value={appState.advanced.optimizer}
      class="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
    >
      <option value="Lbfgs">L-BFGS</option>
      <option value="NelderMead">Nelder-Mead</option>
      <option value="ConjugateGradient">Conjugate Gradient</option>
      <option value="TrustRegion">Trust Region</option>
    </select>
  </div>

  <div>
    <label
      for="loss"
      class="block text-xs font-medium text-gray-600 mb-1"
    >Loss function</label>
    <select
      id="loss"
      bind:value={appState.advanced.lossType}
      class="w-full px-2 py-1.5 text-sm border border-gray-300 rounded"
    >
      <option value="SumSquared">Sum of squared errors</option>
      <option value="RootMeanSquared">Root mean squared</option>
      <option value="SumAbsolute">Sum of absolute errors</option>
      <option value="MaxAbsolute">Max absolute error</option>
      <option value="MaxSquared">Max squared error</option>
      <option value="SumSquaredRegionError">Sum sq. region error</option>
      <option value="SumAbsoluteRegionError">Sum abs. region error</option>
      <option value="Stress">Stress (venneuler)</option>
      <option value="DiagError">DiagError (EulerAPE)</option>
    </select>
  </div>

  <label class="flex items-center cursor-pointer text-sm">
    <input
      type="checkbox"
      bind:checked={appState.advanced.showRegions}
      class="mr-2"
    />
    Render filled regions
  </label>

  <div>
    <label class="flex items-center cursor-pointer text-sm">
      <input
        type="checkbox"
        bind:checked={appState.advanced.useSeed}
        class="mr-2"
      />
      Random seed
    </label>
    {#if appState.advanced.useSeed}
      <input
        type="number"
        bind:value={appState.advanced.seed}
        min="0"
        step="1"
        placeholder="42"
        class="mt-1.5 w-full px-2 py-1 text-sm border border-gray-300 rounded"
      />
    {/if}
  </div>
</div>
