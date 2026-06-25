<script lang="ts">
  import { appState } from "../state.svelte";

  // Tolerance spans orders of magnitude, so the slider works in log10 space:
  // each step is one power of ten, from 1e-1 (loose, fast) to 1e-6 (tight,
  // slow). The stored value stays a plain number (1e-1 … 1e-6).
  const TOL_MIN_EXP = -6;
  const TOL_MAX_EXP = -1;

  const tolExp = $derived(Math.log10(appState.advanced.tolerance));

  function setTolExp(exp: number) {
    appState.advanced.tolerance = 10 ** Math.round(exp);
  }
</script>

<div class="space-y-4">
  <div>
    <label for="optimizer" class="block text-xs font-medium text-muted mb-1"
      >Optimizer</label
    >
    <select
      id="optimizer"
      bind:value={appState.advanced.optimizer}
      class="w-full px-2 py-1.5 text-sm border border-line rounded"
    >
      <option value="CmaEsLm">CMA-ES → LM (default)</option>
      <option value="LevenbergMarquardt">Levenberg-Marquardt</option>
      <option value="Lbfgs">L-BFGS</option>
      <option value="NelderMead">Nelder-Mead</option>
    </select>
  </div>

  <div>
    <label for="tolerance" class="block text-xs font-medium text-muted mb-1">
      Tolerance <span class="font-mono text-faint"
        >{appState.advanced.tolerance.toExponential(0)}</span
      >
    </label>
    <input
      id="tolerance"
      type="range"
      min={TOL_MIN_EXP}
      max={TOL_MAX_EXP}
      step="1"
      value={tolExp}
      oninput={(e) =>
        setTolExp(parseFloat((e.target as HTMLInputElement).value))}
      class="w-full"
    />
    <p class="mt-1 text-xs text-muted">
      Final-stage cost-change exit. Left = tighter fit, slower.
    </p>
  </div>

  <div>
    <label for="loss" class="block text-xs font-medium text-muted mb-1"
      >Loss function</label
    >
    <select
      id="loss"
      bind:value={appState.advanced.lossType}
      class="w-full px-2 py-1.5 text-sm border border-line rounded"
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
        class="mt-1.5 w-full px-2 py-1 text-sm border border-line rounded"
      />
    {/if}
  </div>
</div>
