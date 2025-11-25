<script lang="ts">
  import { onMount } from 'svelte';
  
  interface Circle {
    x: number;
    y: number;
    radius: number;
    label: string;
  }
  
  interface DiagramRow {
    input: string;
    size: number;
  }
  
  let circles = $state<Circle[]>([]);
  let wasmModule = $state<any>(null);
  let loading = $state(true);
  let error = $state('');
  let loss = $state<number>(0);
  let targetAreas = $state<Record<string, number>>({});
  let fittedAreas = $state<Record<string, number>>({});
  
  // Diagram specification
  let diagramRows = $state<DiagramRow[]>([
    { input: 'A', size: 3 },
    { input: 'B', size: 5 },
    { input: 'A&B', size: 1 }
  ]);
  
  let inputType = $state<'disjoint' | 'union'>('disjoint');
  
  const colors = [
    'rgba(59, 130, 246, 0.3)',   // blue
    'rgba(239, 68, 68, 0.3)',    // red
    'rgba(34, 197, 94, 0.3)',    // green
    'rgba(234, 179, 8, 0.3)',    // yellow
    'rgba(168, 85, 247, 0.3)',   // purple
  ];

  onMount(async () => {
    try {
      // Import and initialize the WASM module
      const wasm = await import('../../pkg/eunoia.js');
      await wasm.default(); // Initialize WASM
      wasmModule = wasm;
      
      loading = false;
      // The reactive statement will generate the diagram from spec
    } catch (e) {
      error = `Failed to load WASM: ${e}`;
      loading = false;
      console.error(e);
    }
  });
  
  function addRow() {
    diagramRows = [...diagramRows, { input: '', size: 0 }];
  }
  
  function removeRow(index: number) {
    diagramRows = diagramRows.filter((_, i) => i !== index);
  }
  
  function generateFromSpec() {
    if (!wasmModule || diagramRows.length === 0) return;
    
    try {
      // Convert diagramRows to DiagramSpec objects
      const specs = diagramRows
        .filter(row => row.input.trim() !== '' && row.size >= 0)
        .map(row => new wasmModule.DiagramSpec(row.input, row.size));
      
      if (specs.length === 0) {
        circles = [];
        error = '';
        loss = 0;
        targetAreas = {};
        fittedAreas = {};
        return;
      }
      
      // Generate diagram from specification
      const result = wasmModule.generate_from_spec(specs, inputType);
      circles = Array.from(result);
      error = '';
      
      // Get debug info separately using simple arrays
      try {
        const inputs = diagramRows
          .filter(row => row.input.trim() !== '' && row.size >= 0)
          .map(row => row.input);
        const sizes = diagramRows
          .filter(row => row.input.trim() !== '' && row.size >= 0)
          .map(row => row.size);
        
        const debugJson = wasmModule.get_debug_info_simple(inputs, sizes, inputType);
        const debugData = JSON.parse(debugJson);
        loss = debugData.loss;
        targetAreas = debugData.target_areas;
        fittedAreas = debugData.fitted_areas;
      } catch (debugError) {
        console.error('Debug info error:', debugError);
        loss = 0;
        targetAreas = {};
        fittedAreas = {};
      }
    } catch (e) {
      error = `Failed to generate diagram: ${e}`;
      circles = []; // Clear circles on error
      loss = 0;
      targetAreas = {};
      fittedAreas = {};
      console.error(e);
    }
  }
  
  // Auto-generate diagram when specification changes
  // Only trigger when size values or input type change, not when input text is being edited
  $effect(() => {
    if (wasmModule && diagramRows.length > 0) {
      // Track only the size values and input type to avoid premature updates while typing
      const sizeSignature = diagramRows.map(row => row.size).join(',');
      console.log('Generating diagram from spec (sizes/type changed):', sizeSignature, inputType);
      generateFromSpec();
    }
  });
  
  // Calculate SVG viewBox to fit all circles with proper padding
  let viewBox = $derived.by(() => {
    if (circles.length === 0) return '0 0 400 400';
    
    const xs = circles.map(c => c.x);
    const ys = circles.map(c => c.y);
    const rs = circles.map(c => c.radius);
    
    const minX = Math.min(...xs.map((x, i) => x - rs[i]));
    const maxX = Math.max(...xs.map((x, i) => x + rs[i]));
    const minY = Math.min(...ys.map((y, i) => y - rs[i]));
    const maxY = Math.max(...ys.map((y, i) => y + rs[i]));
    
    const width = maxX - minX;
    const height = maxY - minY;
    
    // Add padding as a percentage of the size (10% on each side)
    const paddingPercent = 0.1;
    const paddingX = width * paddingPercent;
    const paddingY = height * paddingPercent;
    
    return `${minX - paddingX} ${minY - paddingY} ${width + 2 * paddingX} ${height + 2 * paddingY}`;
  });
  
  // Calculate appropriate stroke width and font size based on viewBox dimensions
  let avgRadius = $derived(circles.length > 0 
    ? circles.reduce((sum, c) => sum + c.radius, 0) / circles.length 
    : 1);
  let strokeWidth = $derived(avgRadius * 0.02); // 2% of average radius
  
  // Font size based on the viewBox dimensions, not individual circles
  let viewBoxDimension = $derived.by(() => {
    if (circles.length === 0) return 400;
    const xs = circles.map(c => c.x);
    const ys = circles.map(c => c.y);
    const rs = circles.map(c => c.radius);
    const minX = Math.min(...xs.map((x, i) => x - rs[i]));
    const maxX = Math.max(...xs.map((x, i) => x + rs[i]));
    const minY = Math.min(...ys.map((y, i) => y - rs[i]));
    const maxY = Math.max(...ys.map((y, i) => y + rs[i]));
    return Math.max(maxX - minX, maxY - minY);
  });
  let fontSize = $derived(viewBoxDimension * 0.05); // 5% of viewBox dimension
</script>

<div class="min-h-screen bg-gray-50 p-8">
  <div class="max-w-6xl mx-auto">
    <header class="mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-2">Eunoia Debug Viewer</h1>
      <p class="text-gray-600">Visualize Euler diagram layouts</p>
    </header>
    
    {#if loading}
      <div class="bg-white rounded-lg shadow p-8 text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
        <p class="mt-4 text-gray-600">Loading WASM module...</p>
      </div>
    {:else}
      {#if error}
        <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p class="text-red-800">{error}</p>
        </div>
      {/if}
      
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Controls -->
        <div class="lg:col-span-1 space-y-6">
          <!-- Diagram Specification -->
          <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-xl font-semibold mb-4">Diagram Specification</h2>
            
            <!-- Input Type Selection -->
            <div class="mb-4">
              <div class="block text-sm font-medium text-gray-700 mb-2">Input Type</div>
              <div class="flex gap-4">
                <label class="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    bind:group={inputType}
                    value="disjoint"
                    class="mr-2"
                  />
                  <span class="text-sm">Disjoint</span>
                </label>
                <label class="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    bind:group={inputType}
                    value="union"
                    class="mr-2"
                  />
                  <span class="text-sm">Union</span>
                </label>
              </div>
              <p class="mt-1 text-xs text-gray-500">
                {inputType === 'disjoint' 
                  ? 'Values are disjoint regions (A=5, B=2, A&B=1 → total A=6, B=3)' 
                  : 'Values are total set sizes (A=6, B=3, A&B=1)'}
              </p>
            </div>
            
            <div class="space-y-3">
              <div class="grid grid-cols-12 gap-2 text-sm font-medium text-gray-700">
                <div class="col-span-6">Input</div>
                <div class="col-span-4">Size</div>
                <div class="col-span-2"></div>
              </div>
              
              {#each diagramRows as row, i}
                <div class="grid grid-cols-12 gap-2">
                  <input
                    type="text"
                    bind:value={row.input}
                    placeholder="e.g., A or A&B"
                    class="col-span-6 px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <input
                    type="number"
                    bind:value={row.size}
                    min="0"
                    step="0.1"
                    class="col-span-4 px-3 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button
                    onclick={() => removeRow(i)}
                    class="col-span-2 px-2 py-2 bg-red-100 text-red-600 rounded hover:bg-red-200"
                    title="Remove row"
                  >
                    ×
                  </button>
                </div>
              {/each}
              
              <button
                onclick={() => addRow()}
                class="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                + Add Row
              </button>
            </div>
          </div>
          
          <!-- Debug Information -->
          {#if circles.length > 0}
            <div class="bg-white rounded-lg shadow p-6 mt-6">
              <h2 class="text-xl font-semibold mb-4">Debug Information</h2>
              
              <div class="mb-4">
                <div class="text-sm font-medium text-gray-700 mb-1">Loss: <span class="font-mono">{loss.toFixed(4)}</span></div>
                <div class="text-xs text-gray-500 mt-2">
                  Target keys: {Object.keys(targetAreas).length} | 
                  Fitted keys: {Object.keys(fittedAreas).length}
                </div>
              </div>
              
              <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <h3 class="font-semibold text-gray-700 mb-2">Target (Disjoint)</h3>
                  <div class="space-y-1 font-mono text-xs">
                    {#if Object.keys(targetAreas).length === 0}
                      <div class="text-gray-400">No data</div>
                    {:else}
                      {#each Object.entries(targetAreas).sort() as [combo, area]}
                        <div class="flex justify-between">
                          <span>{combo}:</span>
                          <span>{area.toFixed(3)}</span>
                        </div>
                      {/each}
                    {/if}
                  </div>
                </div>
                
                <div>
                  <h3 class="font-semibold text-gray-700 mb-2">Fitted (Disjoint)</h3>
                  <div class="space-y-1 font-mono text-xs">
                    {#if Object.keys(fittedAreas).length === 0}
                      <div class="text-gray-400">No data</div>
                    {:else}
                      {#each Object.entries(fittedAreas).sort() as [combo, area]}
                        <div class="flex justify-between">
                          <span>{combo}:</span>
                          <span class:text-red-600={Math.abs(area - (targetAreas[combo] || 0)) > 0.1}>
                            {area.toFixed(3)}
                          </span>
                        </div>
                      {/each}
                    {/if}
                  </div>
                </div>
              </div>
            </div>
          {/if}
        </div>
        
        <!-- Visualization -->
        <div class="lg:col-span-2">
          <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-xl font-semibold mb-4">Diagram</h2>
            
            <svg
              {viewBox}
              class="w-full h-auto border border-gray-200 rounded"
              preserveAspectRatio="xMidYMid meet"
            >
              <!-- Circles -->
              {#each circles as circle, i}
                <circle
                  cx={circle.x}
                  cy={circle.y}
                  r={circle.radius}
                  fill={colors[i % colors.length]}
                  stroke={colors[i % colors.length].replace('0.3', '1')}
                  stroke-width={strokeWidth}
                />
                <text
                  x={circle.x}
                  y={circle.y}
                  text-anchor="middle"
                  dominant-baseline="middle"
                  font-size={fontSize}
                  class="font-semibold"
                >
                  {circle.label}
                </text>
              {/each}
            </svg>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>
